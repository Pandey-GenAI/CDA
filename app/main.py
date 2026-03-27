import streamlit as st
import requests
import sqlite3
import os
from qdrant_client import QdrantClient
st.set_page_config(page_title="Opportunity Hunt", page_icon="🚀", layout="centered")

QDRANT_CLIENT_URL = os.environ["QDRANT_CLIENT"]
QWEN_API_URL = "http://wiphack30qx5aw.cloudloka.com:8000/v1/chat/completions"
QWEN_MODEL = "qwen/Qwen3-8B"

client_qdrant = QdrantClient(QDRANT_CLIENT_URL)

@st.cache_resource
def get_wines_names():
    conn = sqlite3.connect('wine_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT wine FROM wine_links')
    result = cursor.fetchall()
    wine_names = [wine[0] for wine in result]
    conn.close()
    return wine_names

# Mock function to simulate fetching wine URLs from an e-commerce API
def fetch_wine_url(wine_name):
    conn = sqlite3.connect('wine_database.db')
    cursor = conn.cursor()
    cursor.execute(f'SELECT link FROM wine_links WHERE wine="{wine_name}"')
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result[0]
    else:
        return f"https://www.wine.com/product/search?query={wine_name.replace(' ', '+')}"

# Updated add_wine_links function
def add_wine_links(response_text):
    wine_names = get_wines_names()  # List of wine names in response 
    for wine_name in wine_names:
        if wine_name in response_text:
            wine_url = fetch_wine_url(wine_name)
            response_text = response_text.replace(wine_name, f"[{wine_name}]({wine_url})")
    return response_text

def show():
    st.title("🚀 Opportunity Hunt")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", 
                                         "content": "Enter the dish or cuisine (e.g., 'Bolognese lasagna'):"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    PROMPT_TEMPLATE_1 = '''
    FROM mistral
    SYSTEM You are a sommelier with extensive knowledge of wines from around the world.
    Your goal is to provide the best possible information about wine taste paired with the food.
    Just in a single sentence, no other informations
    USER INPUT: {query}
    '''

    PROMPT_TEMPLATE_2 = '''
    FROM mistral
    SYSTEM You are a sommelier with extensive knowledge of wines from around the world.
    Your goal is to use the provided bullet point list of wines to suggest how to pair them with the food. 
    Add additional, educative informantion about the wines. Just five sentences at max.
    USER INPUT: Pair {food} with {search_result}
    '''

    def qwen_chat(prompt_text):
        candidate_models = [
            os.environ.get("QWEN_MODEL", QWEN_MODEL),
            "qwen/Qwen3-8B",
            "Qwen/Qwen3-8B",
            "qwen3-8b",
            "Qwen3-8B"
        ]

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

        last_error = None
        for model_name in candidate_models:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": 256,
                "temperature": 0.2,
                "stream": False
            }

            r = requests.post(QWEN_API_URL, json=payload, headers=headers, timeout=120)
            if r.status_code == 404 and "model" in r.text:
                last_error = f"model not found ({model_name}): {r.text}"
                continue

            try:
                r.raise_for_status()
            except requests.exceptions.HTTPError as e:
                raise RuntimeError(f"Qwen API HTTP {r.status_code}: {r.text}") from e

            data = r.json()
            # Assumes standard OpenAI-like response format
            return data.get("choices", [])[0].get("message", {}).get("content", "").strip()

        # If all candidates failed with model not found, fetch model list for debug.
        try:
            model_list_resp = requests.get("http://wiphack30qx5aw.cloudloka.com:8000/v1/models", headers=headers, timeout=30)
            model_list = model_list_resp.text
        except Exception as e:
            model_list = f"unavailable: {e}"

        raise RuntimeError(
            f"Qwen model not found on any candidates. last_error={last_error}. available models: {model_list}"
        )

    def get_recommendation(query_text):
        prompt = PROMPT_TEMPLATE_1.format(query=query_text)
        return qwen_chat(prompt)

    def get_wines(query_text, llm_response):
        print(llm_response)
        try:
            search_result = client_qdrant.query(
                collection_name="wines5",
                query_text=llm_response
            )
        except ImportError as e:
            st.error("fastembed is required for Qdrant text search. Add fastembed to requirements.txt and rebuild.")
            return "Unable to fetch from vector DB: fastembed not installed."
        except Exception as e:
            st.error(f"Qdrant query failed: {e}")
            return f"Unable to fetch from vector DB: {e}"

        # As text for LLM prompt
        prompt = PROMPT_TEMPLATE_2.format(food=query_text,
                                          search_result=search_result)
        return qwen_chat(prompt)

    if prompt := st.chat_input():

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        pairing = get_recommendation(prompt)
        resp_wines = get_wines(prompt, pairing)

        msg = add_wine_links(resp_wines)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

if __name__ == '__main__':
    show()