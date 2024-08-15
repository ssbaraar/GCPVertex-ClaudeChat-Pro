import streamlit as st
from anthropic import AnthropicVertex
import os
import PyPDF2
from docx import Document
import pyperclip
import json
import time
import sqlite3
from datetime import datetime

st.set_page_config(page_title="Claude AI Chatbot", page_icon="🤖", layout="wide")

def init_db():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id INTEGER PRIMARY KEY, conversation TEXT, timestamp TEXT, context TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_state
                 (id INTEGER PRIMARY KEY, conversation TEXT, document_content TEXT, context TEXT)''')
    conn.commit()
    return conn

conn = init_db()

def safe_db_operation(operation):
    try:
        operation()
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        conn.rollback()

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

@st.cache_resource
def init_client():
    try:
        return AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Anthropic client: {e}")
        return None

client = init_client()

def save_chat_state():
    safe_db_operation(lambda: conn.cursor().execute(
        "DELETE FROM chat_state"
    ))
    safe_db_operation(lambda: conn.cursor().execute(
        "INSERT INTO chat_state (conversation, document_content, context) VALUES (?, ?, ?)",
        (json.dumps(st.session_state.conversation),
         st.session_state.document_content,
         st.session_state.context)
    ))

def load_chat_state():
    c = conn.cursor()
    c.execute("SELECT * FROM chat_state")
    result = c.fetchone()
    if result:
        st.session_state.conversation = json.loads(result[1])
        st.session_state.document_content = result[2]
        st.session_state.context = result[3]
    else:
        st.session_state.conversation = []
        st.session_state.document_content = ""
        st.session_state.context = "You are a helpful assistant."

def save_conversation():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_db_operation(lambda: conn.cursor().execute(
        "INSERT INTO conversations (conversation, timestamp, context) VALUES (?, ?, ?)",
        (json.dumps(st.session_state.conversation), timestamp, st.session_state.context)
    ))

if "conversation" not in st.session_state:
    load_chat_state()

if "generating" not in st.session_state:
    st.session_state.generating = False

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
                st.session_state.conversation.append({"role": "assistant", "content": full_response})
                save_chat_state()

        except Exception as e:
            response_container.error(f"An error occurred: {e}")
            st.error(f"Error during API call: {e}")
        finally:
            st.session_state.generating = False

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

st.sidebar.title("🛠️ Chatbot Configuration")

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
c = conn.cursor()
c.execute("SELECT id, conversation, timestamp, context FROM conversations ORDER BY timestamp DESC")
conversation_history = c.fetchall()

for idx, (conv_id, conv, timestamp, context) in enumerate(conversation_history):
    col1, col2, col3 = st.sidebar.columns([0.7, 0.35, 0.2])
    col1.write(f"Conversation {idx + 1} - {timestamp}")
    if col2.button("Load", key=f"load_conv_{idx}", use_container_width=True):
        st.session_state.conversation = json.loads(conv)
        st.session_state.context = context
        save_chat_state()
        st.rerun()
    if col3.button("🗑️", key=f"delete_conv_{idx}", use_container_width=True):
        safe_db_operation(lambda: c.execute("DELETE FROM conversations WHERE id = ?", (conv_id,)))
        st.sidebar.success(f"Conversation {idx + 1} deleted!")
        st.rerun()

    # Custom CSS to reduce font size of the "Load" button
    st.markdown("""
        <style>
        .stButton>button {
            font-size: 0.8em;
        }
        </style>
        """, unsafe_allow_html=True)

if conversation_history:
    if st.sidebar.button("Delete All Saved Conversations"):
        safe_db_operation(lambda: c.execute("DELETE FROM conversations"))
        st.sidebar.success("All saved conversations deleted!")
        st.rerun()

st.sidebar.subheader("🎭 Define Chatbot Role")
new_context = st.sidebar.text_area("Enter the role and purpose:", st.session_state.context, height=100)
if new_context != st.session_state.context:
    st.session_state.context = new_context
    save_chat_state()
    st.sidebar.success("Context updated successfully!")

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

        # Only show the "Copy entire conversation" button next to the last message
        if idx == len(st.session_state.conversation) - 1:
            if col2.button("📋 Copy entire conversation", key="copy_entire_conv", help="Copy the entire conversation to clipboard"):
                conversation_text = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation])
                pyperclip.copy(conversation_text)
                st.success("Entire conversation copied to clipboard!", icon="✅")

user_input = st.chat_input("Ask me anything or share your thoughts...", key="user_input")

if user_input:
    chat(user_input)


conn.close()