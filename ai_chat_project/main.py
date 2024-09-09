import streamlit as st

# Set page config at the very beginning
st.set_page_config(page_title="VertexClaude Pro", page_icon="blockchain_laboratories_logo.jpg", layout="wide")

# Import other necessary modules
from streamlit_option_menu import option_menu
from config import init_client, MODEL
from database.connection import init_db, close_db_connections
from auth.user_management import login_user, logout_user
from chat.chat_logic import chat, load_chat_state, save_chat_state, save_conversation
from chat.message_handling import display_conversation_history
from utils.helpers import parse_document
from tools import summarization, content_generation, data_extraction, qa, translation, text_analysis, code_assistant
import atexit

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
    st.session_state.context = "You are a helpful assistant with tool calling capabilities. The user has access to the tool's outputs that you as a model cannot see. This could include text, images and more."
if 'generating' not in st.session_state:
    st.session_state.generating = False
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7
if 'top_p' not in st.session_state:
    st.session_state.top_p = 0.9
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 4096

# Initialize client
client = init_client()

# Initialize database
init_db()

# Main execution
def main():
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
        st.title("🤖 VertexClade Pro")

        # Sidebar
        with st.sidebar:
            st.sidebar.title("Chatbot Configuration")

            # User info and logout button
            col1, col2 = st.sidebar.columns([2, 1])
            col1.write(f"Logged in as: {st.session_state.username}")
            if col2.button("Log out", help="Click here to log out"):
                logout_user()
                st.rerun()

            # Navigation dropdown
            selected = st.selectbox(
                "Select a tool:",
                options=[
                    "Chat", "Text Summarization", "Content Generation",
                    "Data Extraction", "Q&A", "Translation",
                    "Text Analysis", "Code Assistant"
                ],
                format_func=lambda x: {
                    "Chat": "💬 Chat",
                    "Text Summarization": "📝 Text Summarization",
                    "Content Generation": "✍️ Content Generation",
                    "Data Extraction": "🔍 Data Extraction",
                    "Q&A": "❓ Q&A",
                    "Translation": "🌐 Translation",
                    "Text Analysis": "📊 Text Analysis",
                    "Code Assistant": "💻 Code Assistant"
                }[x]
            )

            # Conversation Management section
            st.sidebar.subheader("💬 Conversation Management")
            col1, col2 = st.sidebar.columns(2)
            if col1.button("Clear Chat", key="clear_chat"):
                st.session_state.conversation = []
                st.session_state.document_content = ""
                save_chat_state()
                st.success("Chat cleared!")
                st.rerun()

            if col2.button("New Chat", key="new_chat"):
                if st.session_state.conversation:
                    save_conversation()
                st.session_state.conversation = []
                st.session_state.document_content = ""
                st.session_state.context = "You are a helpful assistant with tool calling capabilities. The user has access to the tool's outputs that you as a model cannot see. This could include text, images and more."
                save_chat_state()
                st.success("New chat started!")
                st.rerun()

            # Model Settings
            with st.expander("⚙️ Model Settings", expanded=False):
                st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.01)
                st.session_state.top_p = st.slider("Top-p (Nucleus Sampling)", 0.0, 1.0, st.session_state.top_p, 0.01)
                st.session_state.max_tokens = st.slider("Max Tokens", 256, 4096, st.session_state.max_tokens, 64)

            # Chatbot Role Definition
            with st.expander("🎭 Define Chatbot Role", expanded=False):
                new_context = st.text_area("Enter the role and purpose:", st.session_state.context, height=80)
                if new_context != st.session_state.context:
                    st.session_state.context = new_context
                    save_chat_state()
                    st.success("Context updated successfully!")

            # Document Upload
            with st.expander("📄 Upload Document", expanded=False):
                uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
                if uploaded_file:
                    with st.spinner("Processing document..."):
                        st.session_state.document_content = parse_document(uploaded_file)
                        save_chat_state()
                        st.success("✅ Document processed successfully!")

                if st.session_state.document_content:
                    st.text_area("Document Content:", st.session_state.document_content, height=100)
                    if st.button("❌ Delete Document Content"):
                        st.session_state.document_content = ""
                        save_chat_state()
                        st.success("Document content deleted!")
                        st.rerun()

        # Main content area
        if selected == "Chat":
            st.subheader("💬 Chat")
            st.markdown("<small>Welcome to your AI assistant! How can I help you today?</small>", unsafe_allow_html=True)

            # Display conversation history
            display_conversation_history()

            # User input for Chat
            user_input = st.chat_input("Ask me anything or share your thoughts...", key="chat_input")

        else:
            with st.expander("🛠️ AI Tools", expanded=True):
                if selected == "Text Summarization":
                    summarization.render()
                elif selected == "Content Generation":
                    content_generation.render()
                elif selected == "Data Extraction":
                    data_extraction.render()
                elif selected == "Q&A":
                    qa.render()
                elif selected == "Translation":
                    translation.render()
                elif selected == "Text Analysis":
                    text_analysis.render()
                elif selected == "Code Assistant":
                    code_assistant.render()

            # Display conversation history for non-Chat tools
            st.subheader("Conversation History")
            display_conversation_history()

            # User input for non-Chat tools
            user_input = st.chat_input("Ask a follow-up question or provide additional input...", key="tool_input")

        # Process user input
        if user_input:
            chat(user_input)

if __name__ == "__main__":
    main()
    # Register the function to be called when the script exits
    atexit.register(close_db_connections)