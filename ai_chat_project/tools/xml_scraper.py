import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import requests
from io import BytesIO
from datetime import datetime, timezone
import json
from config import init_client, MODEL
from typing import List, Dict, Any, Optional
from assets import SYSTEM_MESSAGE, USER_MESSAGE

client = init_client()

class BlogPost:
    def __init__(self, title: str, description: str, link: str, pub_date: datetime, creator: str):
        self.title = title
        self.description = description
        self.link = link
        self.pub_date = pub_date
        self.creator = creator

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "link": self.link,
            "publicationDate": self.pub_date.isoformat() if self.pub_date else None,
            "creator": self.creator
        }

class BlogScraper:
    def __init__(self, url: str):
        self.url = url
        self.posts: List[BlogPost] = []

    def fetch_xml(self) -> str:
        response = requests.get(self.url)
        response.raise_for_status()
        return response.text

    def parse_xml(self, xml_content: str):
        root = ET.fromstring(xml_content)
        channel = root.find('channel') or root  # Some RSS feeds don't have a 'channel' element
        items = channel.findall('.//item') or channel.findall('.//entry')  # Support both RSS and Atom feeds
        
        for item in items:
            title = self.get_element_text(item, 'title')
            description = self.get_element_text(item, 'description') or self.get_element_text(item, 'content')
            link = self.get_element_text(item, 'link')
            pub_date_str = self.get_element_text(item, 'pubDate') or self.get_element_text(item, 'published')
            creator = self.get_element_text(item, '{http://purl.org/dc/elements/1.1/}creator') or self.get_element_text(item, 'author')
            
            pub_date = self.parse_date(pub_date_str)
            
            self.posts.append(BlogPost(title, description, link, pub_date, creator))

    def get_element_text(self, item: ET.Element, tag: str) -> str:
        element = item.find(tag)
        if element is not None:
            return element.text.strip() if element.text else ''
        return ''

    def parse_date(self, date_str: str) -> Optional[datetime]:
        date_formats = [
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d"
        ]
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def filter_by_date(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[BlogPost]:
        filtered_posts = self.posts
        if start_date:
            filtered_posts = [post for post in filtered_posts if post.pub_date and post.pub_date >= start_date]
        if end_date:
            filtered_posts = [post for post in filtered_posts if post.pub_date and post.pub_date <= end_date]
        return filtered_posts

    def scrape(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[BlogPost]:
        xml_content = self.fetch_xml()
        self.parse_xml(xml_content)
        return self.filter_by_date(start_date, end_date)

def format_data(posts: List[BlogPost]):
    data = json.dumps([post.to_dict() for post in posts], indent=2)
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

def render():
    st.subheader("🕷️ XML Blog Scraper")
    
    url_input = st.text_input("Enter XML URL")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=None)
    with col2:
        end_date = st.date_input("End Date", value=None)
    
    if st.button("Scrape"):
        if not url_input:
            st.warning("Please enter an XML URL to scrape.")
            return

        with st.spinner("Scraping in progress..."):
            try:
                scraper = BlogScraper(url_input)
                
                start_datetime = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc) if start_date else None
                end_datetime = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc) if end_date else None

                filtered_posts = scraper.scrape(start_datetime, end_datetime)
                
                if not filtered_posts:
                    st.warning("No items found within the specified date range.")
                    return

                st.success(f"Found {len(filtered_posts)} posts within the specified date range.")

                # Display raw XML data for debugging
                st.subheader("Raw XML Data (First 1000 characters)")
                st.text(scraper.fetch_xml()[:1000])

                formatted_data, token_counts = format_data(filtered_posts)
                
                if formatted_data is not None:
                    st.success("Data formatted successfully")

                    st.subheader("Formatted Data")
                    st.json(formatted_data)

                    try:
                        df = pd.DataFrame([post.to_dict() for post in filtered_posts])
                        if not df.empty:
                            st.success("DataFrame created successfully")
                            st.subheader("Scraped Data")
                            st.dataframe(df)

                            # Download options
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            st.download_button(
                                label="Download JSON",
                                data=json.dumps(formatted_data, indent=2),
                                file_name=f"scraped_data_{timestamp}.json",
                                mime="application/json"
                            )

                            # Excel download
                            try:
                                excel_buffer = BytesIO()
                                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                    df.to_excel(writer, index=False, sheet_name='Sheet1')
                                excel_data = excel_buffer.getvalue()
                                st.download_button(
                                    label="Download Excel",
                                    data=excel_data,
                                    file_name=f"scraped_data_{timestamp}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            except Exception as excel_error:
                                st.error(f"Error creating Excel file: {str(excel_error)}")

                        else:
                            st.warning("No data was extracted. Please check the XML structure.")
                    except Exception as e:
                        st.error(f"Error creating DataFrame: {str(e)}")

                else:
                    st.error("Failed to format data")

            except Exception as e:
                st.error(f"An error occurred during scraping: {str(e)}")

    st.write("Enter an XML URL and click 'Scrape' to begin.")