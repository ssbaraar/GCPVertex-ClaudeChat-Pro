import streamlit as st
from chat.chat_logic import chat

def render():
    st.subheader("📝 Text Summarization")
    text_to_summarize = st.text_area("Enter the text you want to summarize:", height=200)
    if st.button("Summarize"):
        if text_to_summarize:
            with st.spinner("Generating summary..."):
                summary = chat(f"Please summarize the following text:\n\n{text_to_summarize}")
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")