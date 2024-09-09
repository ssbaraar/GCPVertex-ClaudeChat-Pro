import streamlit as st
from chat.chat_logic import chat

def render():
    st.subheader("✍️ Content Generation")
    content_type = st.selectbox("Select content type:", ["Blog Post", "Email", "Marketing Slogan", "Product Description"])
    topic = st.text_input("Enter the topic or product:")
    if st.button("Generate Content"):
        if topic:
            with st.spinner("Generating content..."):
                generated_content = chat(f"Generate a {content_type} about {topic}")
            st.subheader("Generated Content:")
            st.write(generated_content)
        else:
            st.warning("Please enter a topic or product.")