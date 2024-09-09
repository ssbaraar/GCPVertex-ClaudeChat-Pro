import streamlit as st
import pyperclip

def display_message_stats(text):
    if isinstance(text, list):
        text = ' '.join(map(str, text))
    word_count = len(text.split())
    token_count = len(text.encode('utf-8'))
    st.caption(f"Word count: {word_count} | Token count: {token_count}")

def display_conversation_history():
    for idx, message in enumerate(st.session_state.conversation):
        with st.chat_message(message["role"]):
            st.markdown(f"<small>{message['content']}</small>", unsafe_allow_html=True)
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