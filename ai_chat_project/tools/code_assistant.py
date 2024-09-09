import streamlit as st
from chat.chat_logic import chat

def render():
    st.subheader("💻 Code Explanation & Generation")
    code_action = st.radio("Select action:", ["Explain Code", "Generate Code", "Review Code"])
    if code_action == "Explain Code":
        code_to_explain = st.text_area("Enter the code you want explained:", height=200)
        if st.button("Explain"):
            if code_to_explain:
                with st.spinner("Generating explanation..."):
                    explanation = chat(f"Explain this code:\n\n```\n{code_to_explain}\n```")
                st.subheader("Explanation:")
                st.write(explanation)
            else:
                st.warning("Please enter some code to explain.")
    elif code_action == "Generate Code":
        code_description = st.text_area("Describe the code you want generated:", height=150)
        programming_language = st.selectbox("Select programming language:", ["Python", "JavaScript", "Java", "C++", "Ruby"])
        if st.button("Generate"):
            if code_description:
                with st.spinner("Generating code..."):
                    generated_code = chat(f"Generate {programming_language} code for the following description:\n\n{code_description}")
                st.subheader("Generated Code:")
                st.code(generated_code, language=programming_language.lower() if programming_language else None)
            else:
                st.warning("Please describe the code you want generated.")
    elif code_action == "Review Code":
        code_to_review = st.text_area("Enter the code you want reviewed:", height=200)
        if st.button("Review"):
            if code_to_review:
                with st.spinner("Reviewing code..."):
                    review = chat(f"Review this code and provide suggestions for improvement:\n\n```\n{code_to_review}\n```")
                st.subheader("Code Review:")
                st.write(review)
            else:
                st.warning("Please enter some code to review.")