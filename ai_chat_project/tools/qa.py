import streamlit as st
from chat.chat_logic import chat

def render():
    st.subheader("❓ Question Answering")
    context = st.text_area("Enter the context or background information:", height=150)
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if context and question:
            with st.spinner("Finding the answer..."):
                answer = chat(f"Context: {context}\n\nQuestion: {question}\n\nAnswer:")
            st.subheader("Answer:")
            st.write(answer)
        else:
            st.warning("Please provide both context and a question.")