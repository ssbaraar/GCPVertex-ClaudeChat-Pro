# Claude AI Chatbot
![image](https://github.com/user-attachments/assets/c7e78e55-b05c-48c0-9bab-f8a04dbcd3dc)


## 🤖 Overview

This project implements an interactive AI chatbot using Anthropic's Claude model via the AnthropicVertex API. Built with Streamlit, it provides a user-friendly interface for engaging in conversations with the AI, uploading documents for context, and managing chat history.

## 🌟 Features

- **Interactive Chat Interface**: Engage in real-time conversations with Claude AI.
- **Document Upload**: Support for PDF, DOCX, and TXT files to provide context to the AI.
- **Conversation Management**: Save, load, and delete conversation histories.
- **Customizable AI Context**: Define the AI's role and purpose for each conversation.
- **Message Statistics**: View word and token counts for each message.
- **Copy Functionality**: Easily copy individual messages or entire conversations.
- **Stream Responses**: AI responses are streamed in real-time for a dynamic experience.
- **Stop Generation**: Ability to stop the AI's response generation mid-stream.

## 🛠️ Technologies Used

- Python
- Streamlit
- Anthropic API (via AnthropicVertex)
- Google Cloud AI Platform
- PyPDF2 (for PDF parsing)
- python-docx (for DOCX parsing)
- pyperclip (for copy functionality)

## 📋 Prerequisites

- Python 3.7+
- Google Cloud Project with AnthropicVertex API enabled
- `credentials.json` file with your Google Cloud credentials

## 🚀 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/claude-ai-chatbot.git
   cd claude-ai-chatbot
   
2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use venv\\Scripts\\activate
    
3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt

4. **Set up your credentials:**

   Place your credentials.json file in the project root directory. This file should contain your Google Cloud credentials with access to the AnthropicVertex API.

   Example credentials.json file:
    ```bash
    {
   "location": "europe-west1",
   "project_id": "your-project-id",
   "model": "claude-3-5-sonnet@20240620"
    }
Replace "your-project-id" with your actual Google Cloud project ID.

## 🔧 Configuration
Ensure your credentials.json file is correctly configured.
Modify the LOCATION, PROJECT_ID, and MODEL variables in the script if necessary.

## 🚀 Running the Application
Run the Streamlit app:

    streamlit run app.py
    Navigate to the URL provided by Streamlit (usually http://localhost:8501).

## 🖥️ Usage
1. -Start a Conversation: Type your message in the chat input at the bottom of the page.
2. -Upload a Document: Use the sidebar to upload a PDF, DOCX, or TXT file for context.
3. -Manage Conversations: Clear the current chat, start a new one, or load previous conversations from the sidebar.
4. -Customize AI Context: Enter a specific role or instructions for the AI in the sidebar.
5. -Copy Messages: Use the copy buttons to copy individual messages or the entire conversation.
6. -View Message Stats: See word and token counts below each message.
7. -Stop Generation: Click the "Stop Generation" button to halt the AI's response mid-stream.
## 🔒 Security Notes
1. [x] Keep your credentials.json file secure and never commit it to version control.
2. [x] The app uses environment variables and secure handling for API credentials.
## 🔍 Code Structure
* app.py: Main application file containing the Streamlit interface and core logic.
* requirements.txt: List of Python package dependencies.
* .gitignore: Specifies intentionally untracked files to ignore.
