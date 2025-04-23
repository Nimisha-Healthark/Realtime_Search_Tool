from datetime import datetime
import gspread
import streamlit as st
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import re
from pinecone import Pinecone
import openai
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import json


pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
#Openai key
openai.api_key = st.secrets["openai_key"]

# --- Utility Functions ---
def connect_to_google_sheet(sheet_name):
    """
    Connect to Google Sheets using credentials stored in Streamlit secrets.
    """
    try:
        # Load credentials from Streamlit secrets
        credentials_dict = json.loads(st.secrets["google_sheets"]["credentials"])
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(
            credentials_dict,
            ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        )
        client = gspread.authorize(credentials)
        return client.open(sheet_name).sheet1
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        raise

# Log tokens to the Google Sheet
def log_tokens_to_sheet(query, tokens_used,response,answer_type):
    try:
        # Connect to the sheet
        sheet_name = "GPT_log"
        sheet = connect_to_google_sheet(sheet_name)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, query, response, tokens_used,answer_type])
        # st.write("Logging:", [timestamp, query, response, tokens_used, answer_type])
    except Exception as e:
        st.error(f"Failed to log tokens: {e}")
        print(f"Error: {e}")


def fetch_google_urls(query, num_results=5):
    try:
        return list(search(query, num_results=num_results))
    except Exception as e:
        st.error(f"Error fetching URLs: {e}")
        return []

def extract_and_filter_content(urls, start_year, end_year):
    try:
        scraped_texts = []
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        for url in urls:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Check for request errors
            soup = BeautifulSoup(response.content, 'html.parser')

            # Collect multiple tag types
            tag_texts = []
            for tag in ['p', 'li', 'h1', 'h2', 'h3', 'div']:
                tag_texts.extend([el.get_text(strip=True) for el in soup.find_all(tag)])

            full_text = "\n".join(tag_texts)

            # Year filtering
            year_range = list(range(start_year, end_year + 1))
            years_in_text = re.findall(r"20\d{2}", full_text)
            years_found = [int(y) for y in years_in_text if int(y) in year_range]

            if years_found:
                scraped_texts.append(full_text[:3000] + ("..." if len(full_text) > 3000 else ""))
            else:
                scraped_texts.append(None)

        return scraped_texts
    except Exception as e:
        return f"Error scraping URLs: {e}"

def get_gpt4_combine_response(retrieved_texts, query, max_tokens, temperature,response_type_instruction="",start_year="", end_year=""):
    #logger.info("Combined Function for Response")
    """
    Combines user question with search results and Pinecone context to generate a GPT-based response.
    """
     # Fetch Google Search Results
    google_results = fetch_google_urls(query)
    # Scrape text from the fetched Google search result URLs
    scraped_texts = extract_and_filter_content(google_results,start_year,end_year)
    print(scraped_texts)
    
    # Combine URLs with their respective scraped content
    google_context = [
        f"Source {i+1}: {google_results[i]}\nContent:\n{scraped_texts[i]}" 
        for i in range(len(google_results))
    ]
    
    prompt = f"""
    User's Question: {query}

    Context: Below are relevant details related to the user's question:
    
    1. Google Search Results with URLs and Content:
    {chr(10).join(f"{i+1}. {context}" for i, context in enumerate(google_context))}
    
    2. Database Context (referenced as "from database"):
    {chr(10).join(f"{i+1}. {context}" for i, context in enumerate(retrieved_texts))}

    Task:
    f"- {response_type_instruction}\n"
    You are required to generate a **comprehensive and balanced response** to the user's query using the provided Context from both Google Search Results and Pinecone Context.
    - Ensure the following instructions are strictly followed:
    1. **Numerical Data Priority**:
        - Always prioritize numerical data from Pinecone Context when available.
        - If numerical data is not available in Pinecone Context, use Google Search Results. Clearly indicate the source of numerical data in parentheses (e.g., '(from database)' or '(from Google)').
    2. **Content Integration**:
        - Ensure at least **50% of the response** is based on Pinecone Context if it contains relevant information.
        - Combine insights from both sources to create a cohesive and comprehensive response.
    3. **Clarity and User-Friendliness**:
        - Present the response in a clear, easy-to-read format.
        - Avoid repetition and ensure the response aligns with the query's intent.
        - Use **bullet points** to address multiple aspects or points clearly.
    4. **Reference Inclusion**:
        - Always include the source URL(only url) in parentheses after any information taken from Google Search Results (e.g., '(<URL>)').
    Now generate the response.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # or gpt-4
            messages=[
                {"role": "system", "content": "You are an intelligent assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature)
        response_content = response['choices'][0]['message']['content']
         # Get the tokens used
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        return response_content, prompt_tokens, completion_tokens
    except Exception as e:
        user_msg = "⚠️No Answer generated, please check Openai key"
        #logger.error(f"OpenAI InvalidRequestError: {e}")
        return user_msg, 0, 0
   

def search_pinecone(index, embedding,namespace=""):
    """
    Perform a search in Pinecone using the query embedding.
    """
    try:
        # Perform the search
        response = index.query(
            namespace=namespace,      # Search within the default namespace
            vector=embedding,  # Convert embedding to a list format
            top_k=5,                  # Retrieve the top 5 matches
            include_metadata=True     # Include metadata for matched vectors
        )

        # Extract all matches
        all_matches = response
        # Filter matches with a similarity score >= 65%
        filtered_matches = [match for match in response['matches'] if match['score'] >= 0]

        return all_matches, filtered_matches
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return {"matches": []}, []

st.title("🧠 Smart Search with Year Filter")

# --- Input Section ---
query = st.text_input("Enter your search query:", "global AI market forecast")

# Radio button to select answer type
answer_type = st.radio("Select answer length", ("Short", "Detailed"))

col1, col2 = st.columns(2)
with col1:
    start_year = st.number_input("Start Year", min_value=2000, max_value=2100, value=2022)
with col2:
    end_year = st.number_input("End Year", min_value=2000, max_value=2100, value=2025)
    
submit = st.button("Submit")

# --- Main Execution ---
if submit:
    with st.spinner("Fetching and processing results..."):
        if query:
                    # Initialize Pinecone 
                    index=pc.Index("intelligent-search-tool")
                    # index = pc.Index("test-larg-openai")
                    print("Setting up model configurations...")  # Debug print
                    # Set up OpenAI API key
                    openai.api_key = st.secrets["openai_key"]
                    # Embed the query for vector search
                    embedding1 = openai.Embedding.create(
                        input=query,
                        model="text-embedding-3-large"
                    )
                    
                    pinecone_res, filtered_matches = search_pinecone(index, embedding1['data'][0]['embedding'])
                    #logger.info(f"PINECONE CONTEXT : {pinecone_res} ")
                    # Handle empty Pinecone results
                    if not pinecone_res.get('matches'):
                        print("No results found in Pinecone.")
                        context = []
                        pdf_title, pdf_page, pdf_link = [], [], []
                    else:
                        context = [match['metadata'].get('text', '') for match in pinecone_res['matches']]
                        pdf_title = [match['metadata'].get('title', 'N/A') for match in filtered_matches]
                        pdf_page = [match['metadata'].get('page_number', 'N/A') for match in filtered_matches]
                        pdf_link = [match['metadata'].get('link', '#') for match in filtered_matches]
                        
                    # Set response length based on answer type
                    max_tokens = 500 if answer_type == "Short" else 4096
                    response_type_instruction = "Provide a short answer." if answer_type == "Short" else "Provide a detailed answer."
                    answer, prompt_tokens, completion_tokens = get_gpt4_combine_response(context, query, max_tokens, 0.2, response_type_instruction,start_year, end_year)
                    log_tokens_to_sheet(query, prompt_tokens+completion_tokens,answer,answer_type)
                     # Display GPT response
                    st.markdown("### GPT Response")
                    st.write(answer)
                    
                    # Display Pinecone data in DataFrame
                    pinecone_df = pd.DataFrame({
                        'Title': pdf_title,
                        'Page': pdf_page,
                        'Link': pdf_link,
                    })
                    st.dataframe(pinecone_df)
                    
        else:
                    st.info("Enter query.")
