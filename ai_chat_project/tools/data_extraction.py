import streamlit as st
from chat.chat_logic import chat

def render():
    st.subheader("🔍 Data / Entity Extraction")
    text_for_extraction = st.text_area("Enter the text for entity extraction:", height=200)
    if st.button("Extract Entities"):
        if text_for_extraction:
            with st.spinner("Extracting entities..."):
                entities = chat(f"Extract key entities (like names, organizations, locations, dates) from this text:\n\n{text_for_extraction}")
            st.subheader("Extracted Entities:")
            st.write(entities)
        else:
            st.warning("Please enter some text for entity extraction.")