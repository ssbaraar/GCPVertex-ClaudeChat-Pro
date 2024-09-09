import json
import os
from anthropic import AnthropicVertex
import streamlit as st

def load_credentials():
    try:
        with open('credentials.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Credentials file not found. Please ensure 'credentials.json' exists.")
        return None

@st.cache_resource
def init_client():
    credentials = load_credentials()
    if credentials:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'
        LOCATION = credentials.get('location', "europe-west1")
        PROJECT_ID = credentials.get('project_id')
        MODEL = credentials.get('model', "claude-3-5-sonnet@20240620")
        try:
            return AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
        except Exception as e:
            st.sidebar.error(f"Failed to initialize Anthropic client: {e}")
            return None
    else:
        st.error("Failed to load credentials. Chat functionality will be limited.")
        return None

# Add this line to define MODEL
MODEL = "claude-3-5-sonnet@20240620"