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
from datetime import datetime
import hashlib
import atexit

# Constants
COMMON_PASSWORD = "claude2023"  # Change this to your desired common password
DB_NAME = 'chatbot.db'
POOL_SIZE = 5

st.set_page_config(page_title="VertexClaude Pro", page_icon="blockchain_laboratories_logo.jpg", layout="wide")

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
                connection = sqlite3.connect(self.database, check_same_thread=False)
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

        # Create tables if they don't exist
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY, username TEXT UNIQUE)''')
        c.execute('''CREATE TABLE IF NOT EXISTS conversations
                     (id INTEGER PRIMARY KEY, user_id INTEGER, conversation TEXT, timestamp TEXT, context TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS chat_state
                     (id INTEGER PRIMARY KEY, user_id INTEGER, conversation TEXT, document_content TEXT, context TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS prompt_cache
                     (key TEXT PRIMARY KEY, response TEXT, timestamp TEXT)''')

        # Check if user_id column exists in conversations table
        c.execute("PRAGMA table_info(conversations)")
        columns = [column[1] for column in c.fetchall()]
        if 'user_id' not in columns:
            # Add user_id column to conversations table
            c.execute("ALTER TABLE conversations ADD COLUMN user_id INTEGER")

        # Check if user_id column exists in chat_state table
        c.execute("PRAGMA table_info(chat_state)")
        columns = [column[1] for column in c.fetchall()]
        if 'user_id' not in columns:
            # Add user_id column to chat_state table
            c.execute("ALTER TABLE chat_state ADD COLUMN user_id INTEGER")

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
def login_user(username, password):
    if password == COMMON_PASSWORD:
        result = execute_query("SELECT id FROM users WHERE username = ?", (username,))
        if result:
            st.session_state.user_id = result[0][0]
            st.session_state.username = username
            st.session_state.logged_in = True
        else:
            user_id = execute_insert("INSERT INTO users (username) VALUES (?)", (username,))
            st.session_state.user_id = user_id
            st.session_state.username = username
            st.session_state.logged_in = True

        # Reset chat state for new session
        st.session_state.conversation = []
        st.session_state.document_content = ""
        st.session_state.context = "You are a helpful assistant."
        save_chat_state()

        return True
    else:
        st.error("Incorrect password")
        return False

def logout_user():
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
        st.session_state.context = "You are a helpful assistant."

def save_conversation():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    execute_insert(
        "INSERT INTO conversations (user_id, conversation, timestamp, context) VALUES (?, ?, ?, ?)",
        (st.session_state.user_id, json.dumps(st.session_state.conversation), timestamp, st.session_state.context)
    )

# Caching functions
def generate_cache_key(user_input, context, document_content):
    combined = f"{user_input}|{context}|{document_content}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_cached_response(cache_key):
    result = execute_query("SELECT response, timestamp FROM prompt_cache WHERE key = ?", (cache_key,))
    if result:
        response, timestamp = result[0]
        if (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds() < 86400:
            return response
    return None

def set_cached_response(cache_key, response):
    timestamp = datetime.now().isoformat()
    execute_insert(
        "INSERT OR REPLACE INTO prompt_cache (key, response, timestamp) VALUES (?, ?, ?)",
        (cache_key, response, timestamp)
    )

# Main chat function
def chat(user_input):
    if not user_input or not client:
        return

    st.session_state.generating = True
    st.session_state.conversation.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)
        display_message_stats(user_input)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        try:
            combined_context = create_combined_context()
            cache_key = generate_cache_key(user_input, combined_context, st.session_state.document_content)
            cached_response = get_cached_response(cache_key)

            if cached_response:
                full_response = cached_response
                response_container.markdown(full_response)
                display_message_stats(full_response)
            else:
                for event in client.messages.create(
                        max_tokens=4096,
                        system=combined_context,
                        messages=[msg for msg in st.session_state.conversation if msg['role'] in ['user', 'assistant']],
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

                if st.session_state.generating:
                    response_container.markdown(full_response)
                    display_message_stats(full_response)
                    set_cached_response(cache_key, full_response)

            st.session_state.conversation.append({"role": "assistant", "content": full_response})
            save_chat_state()

        except Exception as e:
            response_container.error(f"An error occurred: {e}")
            st.error(f"Error during API call: {e}")
        finally:
            st.session_state.generating = False

# Helper functions
def create_combined_context():
    combined_context = st.session_state.context
    if st.session_state.document_content:
        combined_context += "\nDocument Content:\n" + st.session_state.document_content
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

def clear_cache():
    safe_db_operation(lambda cur: cur.execute("DELETE FROM prompt_cache"))
    st.sidebar.success("Cache cleared successfully!")

# Function to close all database connections
def close_db_connections():
    db_pool.close_all()

# Main execution
if __name__ == "__main__":
    # Check if this is a new session
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = True
        # Register the function to be called when the Streamlit script stops
        atexit.register(close_db_connections)

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("🤖 VertexClade Pro - Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
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

        if "generating" not in st.session_state:
            st.session_state.generating = False

        st.sidebar.subheader("🎭 Define Chatbot Role")
        new_context = st.sidebar.text_area("Enter the role and purpose:", st.session_state.context, height=100)
        if new_context != st.session_state.context:
            st.session_state.context = new_context
            save_chat_state()
            st.sidebar.success("Context updated successfully!")

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
            save_chat_state()
            st.sidebar.success("New chat started!")
            st.rerun()

        st.sidebar.subheader("📜 Conversation History")
        conversation_history = execute_query(
            "SELECT id, conversation, timestamp, context FROM conversations WHERE user_id = ? ORDER BY timestamp DESC",
            (st.session_state.user_id,)
        )

        def generate_conversation_title(conversation):
            # Extract the first few exchanges or a representative portion of the conversation
            sample = json.loads(conversation)[:3]  # Take the first 3 messages
            sample_text = "\n".join([f"{msg['role']}: {msg['content'][:50]}..." for msg in sample])

            prompt = f"Generate a short, 1-3 word title for this conversation:\n\n{sample_text}\n\nTitle:"

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

        for idx, (conv_id, conv, timestamp, context) in enumerate(conversation_history):
            title = generate_conversation_title(conv)

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

        st.markdown("""
            <style>
            .stButton>button {
                font-size: 0.8em;
            }
            </style>
            """, unsafe_allow_html=True)

        if conversation_history:
            if st.sidebar.button("Delete All Saved Conversations"):
                safe_db_operation(lambda cur: cur.execute("DELETE FROM conversations WHERE user_id = ?", (st.session_state.user_id,)))
                st.sidebar.success("All saved conversations deleted!")
                st.rerun()

        st.sidebar.subheader("📄 Upload Document")
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
        if uploaded_file:
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

        st.sidebar.subheader("🗑️ Clear Cache")
        if st.sidebar.button("Clear Prompt Cache"):
            clear_cache()

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

# This is the end of the main execution block

# Function to close all database connections
def close_db_connections():
    db_pool.close_all()

# Register the function to be called when the Streamlit script stops
atexit.register(close_db_connections)