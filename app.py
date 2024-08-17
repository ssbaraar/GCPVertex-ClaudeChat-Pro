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
import base64

# Local storage component
import streamlit.components.v1 as components

def init_local_storage():
    components.html(
        """
        <script>
        const sendMessageToStreamlit = (type, message) => {
            window.parent.postMessage({type, message}, "*");
        };

        const getFromLocalStorage = (key) => {
            sendMessageToStreamlit("GET_LOCAL_STORAGE", localStorage.getItem(key));
        };

        const setToLocalStorage = (key, value) => {
            localStorage.setItem(key, value);
            sendMessageToStreamlit("SET_LOCAL_STORAGE", "OK");
        };

        const removeFromLocalStorage = (key) => {
            localStorage.removeItem(key, value);
            sendMessageToStreamlit("REMOVE_LOCAL_STORAGE", "OK");
        };

        window.addEventListener("message", (event) => {
            if (event.data.type === "GET_LOCAL_STORAGE") {
                getFromLocalStorage(event.data.key);
            } else if (event.data.type === "SET_LOCAL_STORAGE") {
                setToLocalStorage(event.data.key, event.data.value);
            } else if (event.data.type === "REMOVE_LOCAL_STORAGE") {
                removeFromLocalStorage(event.data.key);
            }
        });
        </script>
        """,
        height=0,
        width=0,
    )

def get_local_storage(key):
    result = st.session_state.get("local_storage_result", None)
    components.html(
        f"""
        <script>
        window.parent.postMessage({{type: "GET_LOCAL_STORAGE", key: "{key}"}}, "*");
        </script>
        """,
        height=0,
        width=0,
    )
    return result

def set_local_storage(key, value):
    components.html(
        f"""
        <script>
        window.parent.postMessage({{type: "SET_LOCAL_STORAGE", key: "{key}", value: "{value}"}}, "*");
        </script>
        """,
        height=0,
        width=0,
    )

def remove_local_storage(key):
    components.html(
        f"""
        <script>
        window.parent.postMessage({{type: "REMOVE_LOCAL_STORAGE", key: "{key}"}}, "*");
        </script>
        """,
        height=0,
        width=0,
    )

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
        c.execute('''CREATE TABLE IF NOT EXISTS prompt_cache
                     (key TEXT PRIMARY KEY, response TEXT, timestamp TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS sessions
                     (id TEXT PRIMARY KEY, user_id INTEGER, expiry TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS user_preferences
                     (user_id INTEGER PRIMARY KEY, tone TEXT, language TEXT, interests TEXT)''')

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
def set_login_expiration():
    st.session_state.login_expiration = time.time() + 3600  # 1 hour expiration

def check_login_expiration():
    return 'login_expiration' in st.session_state and time.time() < st.session_state.login_expiration

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

def load_user_preferences(user_id):
    result = execute_query("SELECT tone, language, interests FROM user_preferences WHERE user_id = ?", (user_id,))
    if result:
        return result[0]
    return None

def save_user_preferences(user_id, tone, language, interests):
    safe_db_operation(lambda cur: cur.execute(
        "INSERT OR REPLACE INTO user_preferences (user_id, tone, language, interests) VALUES (?, ?, ?, ?)",
        (user_id, tone, language, interests)
    ))

def login_user(username, password):
    if password == COMMON_PASSWORD:
        result = execute_query("SELECT id FROM users WHERE username = ?", (username,))
        if result:
            user_id = result[0][0]
        else:
            user_id = execute_insert("INSERT INTO users (username) VALUES (?)", (username,))

        # Load user preferences and profile
        preferences = load_user_preferences(user_id)
        if preferences:
            st.session_state.tone, st.session_state.language, st.session_state.interests = preferences

        session_id = create_session(user_id)
        st.session_state.session_id = session_id
        st.session_state.logged_in = True
        st.session_state.user_id = user_id
        st.session_state.username = username  # Set the username in session state

        # Set session in local storage
        set_local_storage('session_id', session_id)

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
    session_id = get_local_storage('session_id')
    if session_id:
        safe_db_operation(lambda cur: cur.execute("DELETE FROM sessions WHERE id = ?", (session_id,)))
    # Clear session from local storage
    remove_local_storage('session_id')

    # Save current chat state before logging out
    save_chat_state()
    save_conversation()

    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.session_state.logged_in = False

#basic task breakdown feature.
def handle_complex_task(task):
    progress_placeholder = st.empty()
    progress_placeholder.info("Generating step-by-step guide...")

    prompt = f"""
    The user has requested help with the following task:
    {task}

    Please break this task down into a series of steps. For each step, provide:
    1. A brief description of the step
    2. Any additional details or explanations needed

    Format your response as a numbered list.
    """

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            system="You are a helpful assistant that breaks down complex tasks into manageable steps."
        )

        steps = response.content[0].text.strip()
        progress_placeholder.success("Step-by-step guide generated successfully!")
        return steps
    except Exception as e:
        progress_placeholder.error(f"Error during task breakdown: {e}")
        return "I apologize, I encountered an error while trying to break down the task."

def guide_through_steps(steps):
    st.write("I've broken down the task into steps for you. Let's go through them one by one:")
    step_list = steps.split('\n')
    for i, step in enumerate(step_list):
        if step.strip():  # Check if the step is not empty
            st.write(step)
            if i < len(step_list) - 1:  # If it's not the last step
                proceed = st.button(f"I've completed this step", key=f"step_{i}")
                if not proceed:
                    break  # Stop if the user hasn't completed the current step

    st.write("Great job! You've completed all the steps. Is there anything else you need help with?")

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

def generate_conversation_title(conversation):
    sample = json.loads(conversation)[:3]
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

def save_conversation():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_json = json.dumps(st.session_state.conversation)
    title = generate_conversation_title(conversation_json)
    execute_insert(
        "INSERT INTO conversations (user_id, conversation, timestamp, context, title) VALUES (?, ?, ?, ?, ?)",
        (st.session_state.user_id, conversation_json, timestamp, st.session_state.context, title)
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

    # Ensure the conversation starts with a user message
    if not st.session_state.conversation:
        st.session_state.conversation = [{"role": "user", "content": user_input}]
    else:
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

            # Check if the user is asking for help with a complex task
            if "help me" in user_input.lower() and "step by step" in user_input.lower():
                steps = handle_complex_task(user_input)
                full_response = "Certainly! I'd be happy to help you with that task. Here's a breakdown of the steps:\n\n" + steps
                guide_through_steps(steps)
            else:
                # Existing code for normal chat responses
                cache_key = generate_cache_key(user_input, combined_context, st.session_state.document_content)
                cached_response = get_cached_response(cache_key)

                if cached_response:
                    full_response = cached_response
                else:
                    with st.spinner("Generating response..."):
                        # Ensure we're sending at least one user message
                        messages_to_send = st.session_state.conversation[-5:]
                        if not any(msg["role"] == "user" for msg in messages_to_send):
                            messages_to_send = [{"role": "user", "content": user_input}] + messages_to_send

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

                    if st.session_state.generating:
                        set_cached_response(cache_key, full_response)

            response_container.markdown(full_response)
            display_message_stats(full_response)

            # Add the assistant's response to the conversation history
            st.session_state.conversation.append({"role": "assistant", "content": full_response})
            save_chat_state()

        except Exception as e:
            response_container.error(f"An error occurred: {e}")
            st.error(f"Error during API call: {e}")
        finally:
            st.session_state.generating = False

# Now, define the create_combined_context function
def create_combined_context(window_size=5):
    combined_context = f"You are a helpful assistant with a {st.session_state.tone} tone, speaking in {st.session_state.language}. You are knowledgeable in {st.session_state.interests}.\n"

    # Use a sliding window of the last N messages
    recent_messages = st.session_state.conversation[-window_size:]
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

    combined_context += f"\nRecent conversation history:\n{conversation_text}"

    if st.session_state.document_content:
        combined_context += "\n\nDocument Content:\n" + st.session_state.document_content

    return combined_context


# Helper functions
def create_combined_context(window_size=5):
    combined_context = f"You are a helpful assistant with a {st.session_state.tone} tone, speaking in {st.session_state.language}. You are knowledgeable in {st.session_state.interests}.\n"

    # Use a sliding window of the last N messages
    recent_messages = st.session_state.conversation[-window_size:]
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

    combined_context += f"\nRecent conversation history:\n{conversation_text}"

    if st.session_state.document_content:
        combined_context += "\n\nDocument Content:\n" + st.session_state.document_content

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

def translate_text(text, target_language):
    prompt = f"Translate the following text to {target_language}:\n{text}"
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=200,  # Adjust as necessary
            messages=[{"role": "user", "content": prompt}],
            system="You are a language translation assistant."
        )
        translation = response.content[0].text.strip()
        return translation
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return "Translation failed."

def summarize_content(content_type):
    progress_placeholder = st.empty()
    progress_placeholder.info(f"Generating {content_type} report...")

    if content_type == "conversation":
        # Use the entire conversation history
        text_to_summarize = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation])
    elif content_type == "document":
        text_to_summarize = st.session_state.document_content
    else:
        progress_placeholder.error("Invalid content type.")
        return "Invalid content type."

    # Truncate the text if it's too long
    max_chars = 12000  # Adjust this value based on your model's limitations
    if len(text_to_summarize) > max_chars:
        text_to_summarize = text_to_summarize[:max_chars] + "... (truncated)"

    prompt = f"""
    Please provide a comprehensive report based on the following {content_type}. Structure the report as follows:
    1. Introduction: Briefly introduce the main focus of the conversation or document.
    2. Key Topics Discussed: Summarize the main points and topics that were discussed.
    3. Decisions Made: Highlight any decisions or conclusions reached during the conversation.
    4. Action Items: List any actions that need to be taken based on the conversation.
    5. Conclusion: Summarize the overall outcome of the conversation or document.
    
    Here is the content to be summarized:
    {text_to_summarize}
    """

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,  # Adjust as necessary for a more detailed report
            messages=[{"role": "user", "content": prompt}],
            system="You are a helpful assistant that generates detailed reports."
        )
        report = response.content[0].text.strip()
        progress_placeholder.success(f"{content_type.capitalize()} report generated successfully!")
        return report
    except Exception as e:
        progress_placeholder.error(f"Error during report generation: {e}")
        return f"Report generation failed: {str(e)}"

def display_report_modal(report, content_type):
    modal = st.empty()
    with modal.container():
        st.subheader(f"{content_type.capitalize()} Report")
        st.write(report)
        if st.button("Close"):
            modal.empty()
        if st.button("Copy Report"):
            pyperclip.copy(report)
            st.success("Report copied to clipboard!", icon="✅")


# Add this function to create a modal for displaying the summary
def display_summary_modal(summary, content_type):
    modal = st.empty()
    with modal.container():
        st.subheader(f"{content_type.capitalize()} Summary")
        st.write(summary)
        if st.button("Close"):
            modal.empty()
        if st.button("Copy Summary"):
            pyperclip.copy(summary)
            st.success("Summary copied to clipboard!", icon="✅")

# Function to close all database connections
def close_db_connections():
    db_pool.close_all()

# Ensure that the attributes are initialized in session state
def initialize_session_state():
    if 'tone' not in st.session_state:
        st.session_state.tone = 'Formal'  # Default value, you can change it
    if 'language' not in st.session_state:
        st.session_state.language = 'English'  # Default value, you can change it
    if 'interests' not in st.session_state:
        st.session_state.interests = ''  # Default value, you can change it

# Main execution
if __name__ == "__main__":
    initialize_session_state()  # Initialize session state attributes

    init_local_storage()

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
        st.session_state.context = "You are a helpful assistant."
    if 'generating' not in st.session_state:
        st.session_state.generating = False

    # Check if this is a new session
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = True
        # Register the function to be called when the Streamlit script stops
        atexit.register(close_db_connections)

    if not st.session_state.logged_in:
        session_id = get_local_storage('session_id')
        if session_id:
            user_id = validate_session(session_id)
            if user_id:
                st.session_state.user_id = user_id
                st.session_state.logged_in = True
                load_chat_state()
                st.rerun()

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
        set_login_expiration()  # Refresh the login expiration on each interaction
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

        st.sidebar.subheader("🗑️ Clear Cache")
        if st.sidebar.button("Clear Prompt Cache"):
            clear_cache()

        # Add summarization buttons
        st.sidebar.subheader("📊 Summarization")
        col1, col2 = st.sidebar.columns(2)

        if col1.button("Summarize Conversation"):
            with st.spinner("Generating conversation summary..."):
                summary = summarize_content("conversation")
            display_summary_modal(summary, "conversation")

        if col2.button("Summarize Document"):
            if st.session_state.document_content:
                with st.spinner("Generating document summary..."):
                    summary = summarize_content("document")
                display_summary_modal(summary, "document")
            else:
                st.sidebar.warning("No document uploaded to summarize.")

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
