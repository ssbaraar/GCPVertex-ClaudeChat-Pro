import streamlit as st
from anthropic import AnthropicVertex
import os
import PyPDF2
from docx import Document
import pyperclip
import json
import time
import sqlite3
from sqlite3 import Connection
from threading import Lock
import queue
from datetime import datetime, timedelta
import hashlib
import atexit
import secrets

# Constants
COMMON_PASSWORD = "claude2023"  # Change this to your desired common password
DB_NAME = 'chatbot.db'
POOL_SIZE = 5

st.set_page_config(page_title="VertexClaude Pro", page_icon="blockchain_laboratories_logo.jpg", layout="wide")

# SQLite datetime handling
def adapt_datetime(dt):
    return dt.isoformat()

def convert_datetime(s):
    return datetime.fromisoformat(s)

sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("datetime", convert_datetime)

# Database connection pool
class ConnectionPool:
    def __init__(self, database, max_connections):
        self.database = database
        self.max_connections = max_connections
        self.connections = queue.Queue(maxsize=max_connections)
        self.connection_count = 0
        self.lock = Lock()

    def get_connection(self) -> Connection:
        if not self.connections.empty():
            return self.connections.get()

        with self.lock:
            if self.connection_count < self.max_connections:
                connection = sqlite3.connect(self.database, 
                                             detect_types=sqlite3.PARSE_DECLTYPES,
                                             check_same_thread=False)
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('PRAGMA synchronous=NORMAL')
                self.connection_count += 1
                return connection

        return self.connections.get(block=True, timeout=30)

    def return_connection(self, connection: Connection):
        self.connections.put(connection)

    def close_all(self):
        while not self.connections.empty():
            connection = self.connections.get()
            connection.close()

# Initialize the connection pool
db_pool = ConnectionPool(DB_NAME, POOL_SIZE)

# Database operations
def safe_db_operation(operation):
    conn = db_pool.get_connection()
    try:
        with conn:
            operation(conn.cursor())
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        conn.rollback()
    finally:
        db_pool.return_connection(conn)

def execute_query(query, params=None):
    conn = db_pool.get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
    finally:
        db_pool.return_connection(conn)

def execute_insert(query, params):
    conn = db_pool.get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid
    finally:
        db_pool.return_connection(conn)

# Database initialization
def init_db():
    conn = db_pool.get_connection()
    try:
        c = conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY, username TEXT UNIQUE)''')
        c.execute('''CREATE TABLE IF NOT EXISTS conversations
                     (id INTEGER PRIMARY KEY, user_id INTEGER, conversation TEXT, timestamp TEXT, context TEXT, title TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS chat_state
                     (id INTEGER PRIMARY KEY, user_id INTEGER, conversation TEXT, document_content TEXT, context TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS sessions
                     (id TEXT PRIMARY KEY, user_id INTEGER, expiry TIMESTAMP)''')

        conn.commit()
    finally:
        db_pool.return_connection(conn)

# Call init_db at the start of your application
init_db()

# Load API credentials
def load_credentials():
    try:
        with open('credentials.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Credentials file not found. Please ensure 'credentials.json' exists.")
        return None

credentials = load_credentials()
if credentials:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'
    LOCATION = credentials.get('location', "europe-west1")
    PROJECT_ID = credentials.get('project_id')
    MODEL = credentials.get('model', "claude-3-5-sonnet@20240620")
else:
    st.error("Failed to load credentials. Chat functionality will be limited.")

# Initialize Anthropic client
@st.cache_resource
def init_client():
    try:
        return AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Anthropic client: {e}")
        return None

client = init_client()

# User authentication functions
def create_session(user_id):
    session_id = secrets.token_urlsafe(32)
    expiry = datetime.now() + timedelta(hours=24)  # 24-hour session
    execute_insert("INSERT INTO sessions (id, user_id, expiry) VALUES (?, ?, ?)",
                   (session_id, user_id, expiry))
    return session_id

def validate_session(session_id):
    result = execute_query("SELECT user_id, expiry FROM sessions WHERE id = ?", (session_id,))
    if result:
        user_id, expiry = result[0]
        if datetime.now() < expiry:
            return user_id
    return None

def login_user(username, password):
    if password == COMMON_PASSWORD:
        result = execute_query("SELECT id FROM users WHERE username = ?", (username,))
        if result:
            user_id = result[0][0]
        else:
            user_id = execute_insert("INSERT INTO users (username) VALUES (?)", (username,))

        session_id = create_session(user_id)
        st.session_state.session_id = session_id
        st.session_state.logged_in = True
        st.session_state.user_id = user_id
        st.session_state.username = username

        # Reset chat state for new session
        st.session_state.conversation = []
        st.session_state.document_content = ""
        st.session_state.context = "When a user asks you a <query>{$QUERY}</query>, your goal is to assist them with computer science concepts. First, analyze the query to determine if it relates to computer science. If it is related, think through the relevant concepts, theories, and examples that could help answer the query. Organize your thoughts logically and provide a detailed, accurate response within <answer> tags, explaining the concepts using examples and analogies where appropriate, while tailoring your response to the user's level of understanding. If the query is not related to computer science, politely inform the user that it is outside the scope of your knowledge and suggest they rephrase their query or ask a different question related to computer science, also within <answer> tags. Your goal is to be a helpful and knowledgeable assistant, and if unsure about a concept, acknowledge it and provide a partial response or suggest additional resources."
        save_chat_state()

        return True
    else:
        st.error("Incorrect password")
        return False

def logout_user():
    if 'session_id' in st.session_state:
        safe_db_operation(lambda cur: cur.execute("DELETE FROM sessions WHERE id = ?", (st.session_state.session_id,)))

    # Save current chat state before logging out
    save_chat_state()
    save_conversation()

    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.session_state.logged_in = False

# Chat state management functions
def save_chat_state():
    safe_db_operation(lambda cur: cur.execute(
        "DELETE FROM chat_state WHERE user_id = ?", (st.session_state.user_id,)
    ))
    safe_db_operation(lambda cur: cur.execute(
        "INSERT INTO chat_state (user_id, conversation, document_content, context) VALUES (?, ?, ?, ?)",
        (st.session_state.user_id, json.dumps(st.session_state.conversation),
         st.session_state.document_content, st.session_state.context)
    ))

def load_chat_state():
    result = execute_query("SELECT * FROM chat_state WHERE user_id = ?", (st.session_state.user_id,))
    if result:
        st.session_state.conversation = json.loads(result[0][2])
        st.session_state.document_content = result[0][3]
        st.session_state.context = result[0][4]
    else:
        st.session_state.conversation = []
        st.session_state.document_content = ""
        st.session_state.context = "When a user asks you a <query>{$QUERY}</query>, your goal is to assist them with computer science concepts. First, analyze the query to determine if it relates to computer science. If it is related, think through the relevant concepts, theories, and examples that could help answer the query. Organize your thoughts logically and provide a detailed, accurate response within <answer> tags, explaining the concepts using examples and analogies where appropriate, while tailoring your response to the user's level of understanding. If the query is not related to computer science, politely inform the user that it is outside the scope of your knowledge and suggest they rephrase their query or ask a different question related to computer science, also within <answer> tags. Your goal is to be a helpful and knowledgeable assistant, and if unsure about a concept, acknowledge it and provide a partial response or suggest additional resources."

def generate_conversation_title(conversation):
    sample = json.loads(conversation)[:3]
    sample_text = "\n".join([f"{msg['role']}: {msg['content'][:50]}..." for msg in sample])

    prompt = f"Generate a short, 1-2 word title for this conversation:\n\n{sample_text}\n\nTitle:"

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
            system="You are a helpful assistant that generates short, concise titles."
        )
        title = response.content[0].text.strip()
        return title
    except Exception as e:
        st.error(f"Error generating title: {e}")
        return "Untitled Chat"

def save_conversation():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_json = json.dumps(st.session_state.conversation)
    title = generate_conversation_title(conversation_json)
    execute_insert(
        "INSERT INTO conversations (user_id, conversation, timestamp, context, title) VALUES (?, ?, ?, ?, ?)",
        (st.session_state.user_id, conversation_json, timestamp, st.session_state.context, title)
    )

# Main chat function
def chat(user_input):
    if not user_input or not client:
        return

    st.session_state.generating = True

    # Add the user's message to the conversation
    st.session_state.conversation.append({"role": "user", "content": user_input})

    # Display the user's message
    with st.chat_message("user"):
        st.markdown(user_input)
        display_message_stats(user_input)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        try:
            combined_context = create_combined_context()

            with st.spinner("Generating response..."):
                # Send the entire conversation history
                messages_to_send = truncate_conversation(st.session_state.conversation)

                for event in client.messages.create(
                        max_tokens=4096,
                        system=combined_context,
                        messages=messages_to_send,
                        model=MODEL,
                        stream=True,
                ):
                    if st.session_state.generating:
                        if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                            full_response += event.delta.text
                            response_container.markdown(f"{full_response}▌")
                        time.sleep(0.01)
                    else:
                        break

            response_container.markdown(full_response)
            display_message_stats(full_response)

            # Add the assistant's response to the conversation history
            st.session_state.conversation.append({"role": "assistant", "content": full_response})

            # Save the updated chat state
            save_chat_state()

        except Exception as e:
            response_container.error(f"An error occurred: {e}")
            st.error(f"Error during API call: {e}")
        finally:
            st.session_state.generating = False

# Ensure chat state is loaded when the user logs in
if st.session_state.logged_in:
    load_chat_state()
# Helper functions
def create_combined_context():
    combined_context = f"{st.session_state.context}\n\n"

    # Include a summary of the last 10 chats
    conversation_summary = ""
    chat_count = 0
    total_tokens = 0
    max_tokens = 50000  # Increased token limit

    for msg in reversed(st.session_state.conversation):
        if chat_count >= 10:
            break

        role = msg['role']
        content = msg['content']

        # Simple compression: truncate long messages and remove extra whitespace
        if len(content) > 500:
            content = content[:497] + "..."
        content = ' '.join(content.split())  # Remove extra whitespace

        summary = f"{role.capitalize()}: {content}\n\n"
        summary_tokens = len(summary.encode('utf-8'))

        if total_tokens + summary_tokens > max_tokens:
            break

        conversation_summary = summary + conversation_summary
        total_tokens += summary_tokens
        chat_count += 1

    combined_context += f"Recent conversation history:\n{conversation_summary}"

    if st.session_state.document_content:
        # Compress document content if needed
        doc_content = st.session_state.document_content
        if len(doc_content) > 1000:
            doc_content = doc_content[:997] + "..."
        doc_content = ' '.join(doc_content.split())  # Remove extra whitespace
        combined_context += f"\n\nDocument Content:\n{doc_content}"

    # Ensure we don't exceed the max token limit
    if len(combined_context.encode('utf-8')) > max_tokens:
        combined_context = combined_context[:max_tokens].rsplit(' ', 1)[0] + "..."

    return combined_context

def display_message_stats(text):
    word_count = len(text.split())
    token_count = len(text.encode('utf-8'))
    st.caption(f"Word count: {word_count} | Token count: {token_count}")

def parse_document(file):
    try:
        if file.type == "application/pdf":
            return parse_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return parse_docx(file)
        elif file.type == "text/plain":
            return file.getvalue().decode("utf-8")
        else:
            return "Unsupported file type"
    except Exception as e:
        return f"Error parsing document: {str(e)}"

def parse_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() for page in reader.pages)

def parse_docx(file):
    doc = Document(file)
    return "\n".join(para.text for para in doc.paragraphs)

def truncate_conversation(conversation, max_messages=50):
    if len(conversation) > max_messages:
        truncated = conversation[-max_messages:]
        truncated[0]["content"] = f"[Earlier conversation truncated] ... {truncated[0]['content']}"
        return truncated
    return conversation

# Function to close all database connections
def close_db_connections():
    db_pool.close_all()

# Main execution
if __name__ == "__main__":
    # Initialize session state variables
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'document_content' not in st.session_state:
        st.session_state.document_content = ""
    if 'context' not in st.session_state:
        st.session_state.context = "When a user asks you a <query>{$QUERY}</query>, your goal is to assist them with computer science concepts. First, analyze the query to determine if it relates to computer science. If it is related, think through the relevant concepts, theories, and examples that could help answer the query. Organize your thoughts logically and provide a detailed, accurate response within <answer> tags, explaining the concepts using examples and analogies where appropriate, while tailoring your response to the user's level of understanding. If the query is not related to computer science, politely inform the user that it is outside the scope of your knowledge and suggest they rephrase their query or ask a different question related to computer science, also within <answer> tags. Your goal is to be a helpful and knowledgeable assistant, and if unsure about a concept, acknowledge it and provide a partial response or suggest additional resources."
    if 'generating' not in st.session_state:
        st.session_state.generating = False

    # Check if this is a new session
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = True
        # Register the function to be called when the Streamlit script stops
        atexit.register(close_db_connections)

    if not st.session_state.logged_in:
        st.title("🤖 VertexClade Pro - Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            if submit_button:
                if login_user(username, password):
                    st.success("Logged in successfully!")
                    st.rerun()
    else:
        st.sidebar.title("Chatbot Configuration")
        st.sidebar.write(f"Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.rerun()

        # Load chat state for the logged-in user
        if "conversation" not in st.session_state:
            load_chat_state()

        st.sidebar.subheader("🎭 Define Chatbot Role")
        new_context = st.sidebar.text_area("Enter the role and purpose:", st.session_state.context, height=100)
        if new_context != st.session_state.context:
            st.session_state.context = new_context
            save_chat_state()
            st.sidebar.success("Context updated successfully!")
        
        st.sidebar.subheader("🧠 Current Context")
        current_context = st.sidebar.text_area("Current context:", st.session_state.context, height=80)
        if current_context != st.session_state.context:
            st.session_state.context = current_context
            save_chat_state()
            st.sidebar.success("Context updated successfully!")

        # Add a button to view the full context including conversation history
        if st.sidebar.button("View Full Context"):
            full_context = create_combined_context()
            st.sidebar.text_area("Full Context (including conversation history):", full_context, height=300)

        st.sidebar.subheader("💬 Conversation Management")
        col1, col2 = st.sidebar.columns(2)
        if col1.button("🗑️Clear Chat", key="clear_chat"):
            st.session_state.conversation = []
            st.session_state.document_content = ""
            save_chat_state()
            st.sidebar.success("Chat cleared!")
            st.rerun()

        if col2.button("🆕New Chat", key="new_chat"):
            if st.session_state.conversation:
                save_conversation()
            st.session_state.conversation = []
            st.session_state.document_content = ""
            st.session_state.context = "When a user asks you a <query>{$QUERY}</query>, your goal is to assist them with computer science concepts. First, analyze the query to determine if it relates to computer science. If it is related, think through the relevant concepts, theories, and examples that could help answer the query. Organize your thoughts logically and provide a detailed, accurate response within <answer> tags, explaining the concepts using examples and analogies where appropriate, while tailoring your response to the user's level of understanding. If the query is not related to computer science, politely inform the user that it is outside the scope of your knowledge and suggest they rephrase their query or ask a different question related to computer science, also within <answer> tags. Your goal is to be a helpful and knowledgeable assistant, and if unsure about a concept, acknowledge it and provide a partial response or suggest additional resources."
            save_chat_state()
            st.sidebar.success("New chat started!")
            st.rerun()

        st.sidebar.subheader("📜 Conversation History")
        conversation_history = execute_query(
            "SELECT id, conversation, timestamp, context, title FROM conversations WHERE user_id = ? ORDER BY timestamp DESC",
            (st.session_state.user_id,)
        )

        for idx, (conv_id, conv, timestamp, context, title) in enumerate(conversation_history):
            col1, col2, col3 = st.sidebar.columns([0.7, 0.35, 0.2])
            col1.write(f"{title} - {timestamp}")
            if col2.button("Load", key=f"load_conv_{idx}", use_container_width=True):
                st.session_state.conversation = json.loads(conv)
                st.session_state.context = context
                save_chat_state()
                st.rerun()
            if col3.button("🗑️", key=f"delete_conv_{idx}", use_container_width=True):
                safe_db_operation(lambda cur: cur.execute("DELETE FROM conversations WHERE id = ?", (conv_id,)))
                st.sidebar.success(f"{title} deleted!")
                st.rerun()

        if conversation_history:
            if st.sidebar.button("Delete All Saved Conversations"):
                safe_db_operation(lambda cur: cur.execute("DELETE FROM conversations WHERE user_id = ?", (st.session_state.user_id,)))
                st.sidebar.success("All saved conversations deleted!")
                st.rerun()

        st.sidebar.subheader("📄 Upload Document")
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
        if uploaded_file:
            with st.spinner("Processing document..."):
                st.session_state.document_content = parse_document(uploaded_file)
                save_chat_state()
                st.sidebar.success("✅ Document processed successfully!")

        if st.session_state.document_content:
            st.sidebar.text_area("Document Content:", st.session_state.document_content, height=150)
            if st.sidebar.button("❌ Delete Document Content"):
                st.session_state.document_content = ""
                save_chat_state()
                st.sidebar.success("Document content deleted!")
                st.rerun()

        st.title("🤖 Claude AI Chatbot")
        st.markdown("Welcome to your AI assistant! How can I help you today?")

        for idx, message in enumerate(st.session_state.conversation):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                display_message_stats(message["content"])
                col1, col2 = st.columns([0.13, 0.9])
                if col1.button("📋 Copy", key=f"copy_{message['role']}_{idx}", help="Copy this message"):
                    pyperclip.copy(message["content"])
                    st.success("Copied to clipboard!", icon="✅")

                if idx == len(st.session_state.conversation) - 1:
                    if col2.button("📋 Copy entire conversation", key="copy_entire_conv", help="Copy the entire conversation to clipboard"):
                        conversation_text = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation])
                        pyperclip.copy(conversation_text)
                        st.success("Entire conversation copied to clipboard!", icon="✅")

        # User input
        user_input = st.chat_input("Ask me anything or share your thoughts...", key="user_input")

        if user_input:
            chat(user_input)

# Register the function to be called when the Streamlit script stops
atexit.register(close_db_connections)