import streamlit as st
from anthropic import AnthropicVertex
import os
import PyPDF2
from docx import Document
import pyperclip
import json
import time


# Set page configuration
st.set_page_config(page_title="Claude AI Chatbot", page_icon="🤖", layout="wide")

# Function to stop generation
def stop_generation():
    st.session_state.generating = False

# Secure handling of credentials
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

# Initialize the client with error handling
@st.cache_resource
def init_client():
    try:
        return AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Anthropic client: {e}")
        return None

client = init_client()

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "document_content" not in st.session_state:
    st.session_state.document_content = ""
if "context" not in st.session_state:
    st.session_state.context = "You are a helpful assistant."
if "tags" not in st.session_state:
    st.session_state.tags = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "generating" not in st.session_state:
    st.session_state.generating = False
if "run_chat" not in st.session_state:
    st.session_state.run_chat = False

# Improved chat function with error handling and rate limiting
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

# Improved document parsing with additional file type support
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

# Sidebar enhancements
st.sidebar.title("🛠️ Chatbot Configuration")

# Conversation Management
st.sidebar.subheader("💬 Conversation Management")
col1, col2 = st.sidebar.columns(2)
if col1.button("🗑️ Clear Chat"):
    st.session_state.conversation = []
    st.sidebar.success("Chat cleared!")
if col2.button("🆕 New Chat"):
    if st.session_state.conversation:
        if len(st.session_state.conversation_history) >= 3:
            st.session_state.conversation_history.pop(0)  # Remove oldest conversation
        st.session_state.conversation_history.append(st.session_state.conversation)
    st.session_state.conversation = []
    st.session_state.tags = []
    st.sidebar.success("New chat started!")

# Display conversation history
st.sidebar.subheader("📜 Conversation History")
for idx, hist in enumerate(st.session_state.conversation_history):
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    if col1.button(f"Conversation {idx + 1}", key=f"load_conv_{idx}"):
        st.session_state.conversation = hist
    if col2.button("x", key=f"delete_conv_{idx}"):
        st.session_state.conversation_history.pop(idx)
        st.sidebar.success(f"Conversation {idx + 1} deleted!")
        st.rerun()

# Option to delete all saved conversations
if st.session_state.conversation_history:
    if st.sidebar.button("Delete All Saved Conversations"):
        st.session_state.conversation_history = []
        st.sidebar.success("All saved conversations deleted!")
        st.rerun()

# Context Input
st.sidebar.subheader("🎭 Define Chatbot Role")
st.session_state.context = st.sidebar.text_area("Enter the role and purpose:", st.session_state.context, height=100)

# Document Upload
st.sidebar.subheader("📄 Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
if uploaded_file:
    st.session_state.document_content = parse_document(uploaded_file)
    st.sidebar.success("✅ Document processed successfully!")

# Display document content and delete option
if st.session_state.document_content:
    st.sidebar.text_area("Document Content:", st.session_state.document_content, height=150)
    if st.sidebar.button("❌ Delete Document Content"):
        st.session_state.document_content = ""
        st.sidebar.success("Document content deleted!")

def display_input_and_stop_button():
    input_container = st.empty()
    stop_button_container = st.empty()

    if not st.session_state.generating:
        user_input = input_container.chat_input("Ask me anything or share your thoughts...", key="user_input")
        if user_input:
            st.session_state.run_chat = True
            return user_input
    else:
        input_container.text_input("AI is generating a response...", disabled=True, key="disabled_input")
        if stop_button_container.button("🛑 Stop Generation", key="stop_button"):
            stop_generation()
    return None

# Main chat interface
st.title("🤖 Claude AI Chatbot")
st.markdown("Welcome to your AI assistant! How can I help you today?")

# Display conversation history
for idx, message in enumerate(st.session_state.conversation):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        col1, col2 = st.columns([0.85, 0.15])
        display_message_stats(message["content"])
        if col2.button("📋 Copy", key=f"copy_{message['role']}_{idx}"):
            pyperclip.copy(message["content"])
            st.success("Copied to clipboard!", icon="✅")

# Display input and stop button
user_input = display_input_and_stop_button()

if user_input:
    chat(user_input)

# Copy entire conversation
if st.session_state.conversation:
    if st.button("📋 Copy entire conversation"):
        conversation_text = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation])
        pyperclip.copy(conversation_text)
        st.success("Entire conversation copied to clipboard!", icon="✅")