import streamlit as st
from streamlit_tags import st_tags
import pandas as pd
import json
from datetime import datetime
import io
import traceback
from config import init_client, MODEL
from bs4 import BeautifulSoup
import html2text
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from pydantic import BaseModel, Field, create_model
from typing import List, Type
import time
import re
import os
import random
from dotenv import load_dotenv
from assets import USER_AGENTS, PRICING, HEADLESS_OPTIONS, SYSTEM_MESSAGE, USER_MESSAGE

load_dotenv()

client = init_client()

def setup_selenium():
    options = Options()
    options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    for option in HEADLESS_OPTIONS:
        options.add_argument(option)
    service = Service(r"./chromedriver-win64/chromedriver.exe")  
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

def create_dynamic_listing_model(field_names: List[str]) -> Type[BaseModel]:
    field_definitions = {field: (str, Field(default='')) for field in field_names}
    return create_model('DynamicListingModel', **field_definitions, __base__=BaseModel)

def create_listings_container_model(listing_model: Type[BaseModel]) -> Type[BaseModel]:
    return create_model('DynamicListingsContainer', listings=(List[listing_model], ...))

def format_data(data: str, DynamicListingsContainer: Type[BaseModel], DynamicListingModel: Type[BaseModel]):
    prompt = SYSTEM_MESSAGE + "\n" + USER_MESSAGE + data
    
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_MESSAGE,
            messages=[
                {"role": "user", "content": USER_MESSAGE + data}
            ]
        )
        
        # The content is directly available in the response
        response_text = response.content[0].text
        st.text("Raw API Response:")
        st.text(response_text)  # Display the raw response for debugging
        
        try:
            parsed_response = json.loads(response_text)
        except json.JSONDecodeError:
            st.error("Failed to parse JSON response. Displaying raw response:")
            st.text(response_text)
            return None, {"input_tokens": 0, "output_tokens": 0}

        token_counts = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
        return parsed_response, token_counts
    except Exception as e:
        st.error(f"Error in format_data: {str(e)}")
        return None, {"input_tokens": 0, "output_tokens": 0}

def calculate_price(token_counts, model=MODEL):
    input_token_count = token_counts.get("input_tokens", 0)
    output_token_count = token_counts.get("output_tokens", 0)
    
    input_cost = input_token_count * PRICING[model]["input"]
    output_cost = output_token_count * PRICING[model]["output"]
    total_cost = input_cost + output_cost
    
    return input_token_count, output_token_count, total_cost

def render():
    st.subheader("🕷️ Web Scraper")
    
    url_input = st.text_input("Enter URL")
    
    # Predefined tags
    predefined_tags = ["title", "description", "price", "date", "author", "category", "image_url", "rating", "review_count"]
    
    # Use st_tags for multiple field input, including predefined and custom tags
    selected_fields = st_tags(
        label='Select or Enter Fields to Extract:',
        text='You can select predefined fields or enter custom ones',
        value=predefined_tags[:3],  # Default to the first 3 predefined tags
        suggestions=predefined_tags,
        maxtags=-1,
        key='fields_input'
    )

    if st.button("Scrape"):
        if not url_input or not selected_fields:
            st.warning("Please enter both URL and select at least one field to extract.")
            return

        with st.spinner("Scraping in progress..."):
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                raw_html = fetch_html_selenium(url_input)
                st.success("HTML fetched successfully")

                markdown = html_to_markdown_with_readability(raw_html)
                st.success("Converted to Markdown successfully")

                # Display markdown content
                st.subheader("Markdown Content")
                st.markdown(markdown)

                DynamicListingModel = create_dynamic_listing_model(selected_fields)
                DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
                formatted_data, token_counts = format_data(markdown, DynamicListingsContainer, DynamicListingModel)
                
                if formatted_data is not None:
                    st.success("Data formatted successfully")

                    # Display raw formatted data for debugging
                    st.subheader("Raw Formatted Data")
                    st.json(formatted_data)

                    try:
                        df = pd.DataFrame(formatted_data.get('listings', []))
                        if not df.empty:
                            st.success("DataFrame created successfully")
                            st.subheader("Scraped Data")
                            st.dataframe(df)

                            st.subheader("Download Options")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.download_button(
                                    label="Download JSON",
                                    data=json.dumps(formatted_data, indent=2),
                                    file_name=f"scraped_data_{timestamp}.json",
                                    mime="application/json"
                                )

                            with col2:
                                excel_buffer = io.BytesIO()
                                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                    df.to_excel(writer, index=False, sheet_name='Sheet1')
                                excel_data = excel_buffer.getvalue()
                                st.download_button(
                                    label="Download Excel",
                                    data=excel_data,
                                    file_name=f"scraped_data_{timestamp}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                            with col3:
                                st.download_button(
                                    label="Download Markdown",
                                    data=markdown,
                                    file_name=f"scraped_data_{timestamp}.md",
                                    mime="text/markdown"
                                )

                            input_tokens, output_tokens, total_cost = calculate_price(token_counts)
                            st.markdown("## Token Usage")
                            st.markdown(f"**Input Tokens:** {input_tokens}")
                            st.markdown(f"**Output Tokens:** {output_tokens}")
                            st.markdown(f"**Total Cost:** ${total_cost:.4f}")
                        else:
                            st.warning("No data was extracted. Please check the selected fields and try again.")
                    except Exception as e:
                        st.error(f"Error creating DataFrame: {str(e)}")
                        st.code(traceback.format_exc())

                else:
                    st.error("Failed to format data")

            except Exception as e:
                st.error(f"An error occurred during scraping: {str(e)}")
                st.code(traceback.format_exc())

    if 'dataframe' not in st.session_state:
        st.write("Enter a URL and select fields to extract, then click 'Scrape' to begin.")