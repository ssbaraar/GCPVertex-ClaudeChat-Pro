import streamlit as st
from chat.chat_logic import chat

def render():
    st.subheader("🌐 Text Translation")
    source_lang = st.selectbox("Source Language:", ["English", "Spanish", "French", "German", "Chinese"])
    target_lang = st.selectbox("Target Language:", ["Spanish", "English", "French", "German", "Chinese"])
    text_to_translate = st.text_area("Enter text to translate:", height=150)
    if st.button("Translate"):
        if text_to_translate:
            with st.spinner("Translating..."):
                translated_text = chat(f"Translate the following {source_lang} text to {target_lang}:\n\n{text_to_translate}")
            st.subheader("Translated Text:")
            st.write(translated_text)
        else:
            st.warning("Please enter some text to translate.")