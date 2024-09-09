import streamlit as st
from chat.chat_logic import chat

def render():
    st.subheader("📊 Text Analysis & Recommendations")
    analysis_text = st.text_area("Enter the text for analysis:", height=200)
    analysis_type = st.multiselect("Select analysis types:", ["Sentiment Analysis", "Keyword Extraction", "Topic Classification"])
    if st.button("Analyze"):
        if analysis_text and analysis_type:
            with st.spinner("Analyzing text..."):
                analysis_results = chat(f"Perform the following analyses on this text: {', '.join(analysis_type)}.\n\nText: {analysis_text}")
            st.subheader("Analysis Results:")
            st.write(analysis_results)
        else:
            st.warning("Please enter text and select at least one analysis type.")