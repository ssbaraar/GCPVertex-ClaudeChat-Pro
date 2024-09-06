import os
import time
import re
import json
from datetime import datetime
from typing import List, Dict, Type
import tiktoken
import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, create_model
import html2text
import tiktoken
from anthropic import AnthropicVertex
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from anthropic import AnthropicVertex
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from pydantic import BaseModel, Field, create_model, ValidationError
from typing import List, Dict, Type, Any
from pydantic import BaseModel, Field, create_model
load_dotenv()

def load_credentials():
    try:
        with open('credentials.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Credentials file not found. Please ensure 'credentials.json' exists.")
        return None

credentials = load_credentials()
if not credentials:
    raise ValueError("Failed to load credentials")

LOCATION = credentials.get('location', "europe-west1")
PROJECT_ID = credentials.get('project_id')
MODEL = credentials.get('model', "claude-3-5-sonnet@20240620")

# Create credentials object
gcp_credentials = service_account.Credentials.from_service_account_info(
    credentials,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Refresh the credentials
gcp_credentials.refresh(Request())

# Create the AnthropicVertex client
client = AnthropicVertex(
    region=LOCATION,
    project_id=PROJECT_ID,
    credentials=gcp_credentials
)
# Set up the Chrome WebDriver options
def setup_selenium():
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    service = Service(r"D:/Browser_Downloads/Downloads/#apps/chromedriver-win64/chromedriver-win64/chromedriver.exe")  
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def fetch_html_selenium(url):
    driver = setup_selenium()
    try:
        driver.get(url)
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        html = driver.page_source
        return html
    finally:
        driver.quit()

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for element in soup.find_all(['header', 'footer']):
        element.decompose()
    return str(soup)

def html_to_markdown_with_readability(html_content):
    cleaned_html = clean_html(html_content)  
    markdown_converter = html2text.HTML2Text()
    markdown_converter.ignore_links = False
    markdown_content = markdown_converter.handle(cleaned_html)
    return markdown_content

pricing = {
    "claude-3-5-sonnet@20240620": {
        "input": 15 / 1_000_000,  # $15 per 1M input tokens
        "output": 75 / 1_000_000, # $75 per 1M output tokens
    },
}

model_used = MODEL

def save_raw_data(raw_data, timestamp, output_folder='output'):
    os.makedirs(output_folder, exist_ok=True)
    raw_output_path = os.path.join(output_folder, f'rawData_{timestamp}.md')
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        f.write(raw_data)
    print(f"Raw data saved to {raw_output_path}")
    return raw_output_path

def remove_urls_from_file(file_path):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_cleaned{ext}"
    with open(file_path, 'r', encoding='utf-8') as file:
        markdown_content = file.read()
    cleaned_content = re.sub(url_pattern, '', markdown_content)
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)
    print(f"Cleaned file saved as: {new_file_path}")
    return cleaned_content

def create_dynamic_listing_model(field_names: List[str]) -> Type[BaseModel]:
    field_definitions = {field: (Any, ...) for field in field_names}
    return create_model('DynamicListingModel', **field_definitions)

def create_listings_container_model(listing_model: Type[BaseModel]) -> Type[BaseModel]:
    return create_model('DynamicListingsContainer', newsItems=(List[listing_model], ...))

def trim_to_token_limit(text, model, max_tokens=200000):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        trimmed_text = encoder.decode(tokens[:max_tokens])
        return trimmed_text
    return text

def format_data(data: str, DynamicListingsContainer: Type[BaseModel]):
    system_message = """You are an intelligent text extraction and conversion assistant. Your task is to extract structured information 
                        from the given text and convert it into a pure JSON format. The JSON should contain only the structured data extracted from the text, 
                        with no additional commentary, explanations, or extraneous information. 
                        Please process the following text and provide the output in pure JSON format with no words before or after the JSON.
                        The output should be a JSON object with a single key 'newsItems' containing an array of news item objects.
                        Each news item object should contain the fields specified in the user's request.
                        If you cannot find data for a specific field, use null or an empty string as appropriate.
                        Do not include empty objects in the newsItems array.
                        Ensure that you extract as many items as possible from the given text."""

    field_names = list(DynamicListingsContainer.__annotations__['newsItems'].__args__[0].__annotations__.keys())

    user_message = f"""Extract the following information from the provided text:
                       Fields to extract for each news item: {', '.join(field_names)}

                       Page content:

                       {data}"""

    print("Sending request to AI model...")
    message = client.messages.create(
        max_tokens=4096,
        system=system_message,
        messages=[
            {"role": "user", "content": user_message},
        ],
        model=MODEL,
    )

    json_content = message.content[0].text
    print("Raw response from AI (first 500 characters):")
    print(json_content[:500])

    try:
        parsed_data = json.loads(json_content)
        print("Successfully parsed JSON. Number of items:", len(parsed_data.get('newsItems', [])))

        # Type inference and conversion
        for item in parsed_data.get('newsItems', []):
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    item[key] = str(value)  # Convert numbers to strings

        result = DynamicListingsContainer.parse_obj(parsed_data)
        print("Successfully created DynamicListingsContainer")
        return result
    except ValidationError as e:
        print(f"Validation error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print("Failed to parse JSON. Content:")
        print(json_content)
        return None

    
def save_formatted_data(formatted_data, timestamp, output_folder='output'):
    if formatted_data is None:
        print("Formatted data is None, cannot save.")
        return None

    os.makedirs(output_folder, exist_ok=True)
    formatted_data_dict = formatted_data.dict() if hasattr(formatted_data, 'dict') else formatted_data

    print("Formatted data dictionary:")
    print(json.dumps(formatted_data_dict, indent=2))

    json_output_path = os.path.join(output_folder, f'sorted_data_{timestamp}.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data_dict, f, indent=4)
    print(f"Formatted data saved to JSON at {json_output_path}")

    if isinstance(formatted_data_dict, dict) and 'newsItems' in formatted_data_dict:
        data_for_df = formatted_data_dict['newsItems']
    elif isinstance(formatted_data_dict, list):
        data_for_df = formatted_data_dict
    else:
        print(f"Unexpected data type: {type(formatted_data_dict)}")
        return None

    try:
        df = pd.DataFrame(data_for_df)
        print("DataFrame created successfully.")
        print("DataFrame shape:", df.shape)
        print("DataFrame columns:", df.columns)
        print("First few rows of the DataFrame:")
        print(df.head())
        excel_output_path = os.path.join(output_folder, f'sorted_data_{timestamp}.xlsx')
        df.to_excel(excel_output_path, index=False)
        print(f"Formatted data saved to Excel at {excel_output_path}")
        return df
    except Exception as e:
        print(f"Error creating DataFrame or saving Excel: {str(e)}")
        return None
    
def calculate_price(input_text, output_text, model="claude-3-5-sonnet@20240620"):
    # Use cl100k_base tokenizer for Claude models
    encoder = tiktoken.get_encoding("cl100k_base")

    input_token_count = len(encoder.encode(input_text))
    output_token_count = len(encoder.encode(output_text))

    # Pricing for Claude models (you may need to adjust these values)
    input_price_per_1k = 0.015  # $0.015 per 1K tokens for input
    output_price_per_1k = 0.075  # $0.075 per 1K tokens for output

    input_cost = (input_token_count / 1000) * input_price_per_1k
    output_cost = (output_token_count / 1000) * output_price_per_1k
    total_cost = input_cost + output_cost

    return input_token_count, output_token_count, total_cost    
def test_claude_connection():
    try:
        message = client.messages.create(
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Hello, Claude. Are you working?"},
            ],
            model=MODEL,
        )
        return message.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    url = 'https://store.steampowered.com/app/2358720/Black_Myth_Wukong/'
    fields = ['title', 'points', 'creator', 'timePosted', 'commentCount']

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_html = fetch_html_selenium(url)
        markdown = html_to_markdown_with_readability(raw_html)
        save_raw_data(markdown, timestamp)

        DynamicListingModel = create_dynamic_listing_model(fields)
        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)

        formatted_data = format_data(markdown, DynamicListingsContainer)
        if formatted_data is None:
            print("Failed to format data")
        else:
            df = save_formatted_data(formatted_data, timestamp)
            if df is not None:
                print(f"Successfully saved data with {len(df)} rows")
            else:
                print("Failed to save formatted data")

            formatted_data_text = json.dumps(formatted_data.dict()) 
            input_tokens, output_tokens, total_cost = calculate_price(markdown, formatted_data_text, model=MODEL)
            print(f"Input token count: {input_tokens}")
            print(f"Output token count: {output_tokens}")
            print(f"Estimated total cost: ${total_cost:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
