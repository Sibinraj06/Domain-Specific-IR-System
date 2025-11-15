import streamlit as st
import requests
import os

# --- Page Configuration ---
st.set_page_config(page_title="Domain-Specific Search", page_icon="🚀", layout="wide")

# --- Backend API URL ---
API_BASE_URL = "http://127.0.0.1:8000"
API_SEARCH_URL = f"{API_BASE_URL}/search"
API_SUGGEST_URL = f"{API_BASE_URL}/suggest"

# --- Functions ---
def check_vector_db():
    # Check for both index types
    faiss_ready = os.path.exists("faiss_index/index.faiss")
    bm25_ready = os.path.exists("bm25_index/cleaned_texts.pkl")
    return faiss_ready and bm25_ready

@st.cache_data(ttl=30)
def check_backend_api():
    try:
        response = requests.get(API_BASE_URL, timeout=1)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

@st.cache_data(ttl=60)
def get_suggestions_from_db():
    try:
        response = requests.get(API_SUGGEST_URL, params={"q": ""}, timeout=2)
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.RequestException:
        return []

# --- Sidebar ---
with st.sidebar:
    st.header(" About This System")
    st.markdown(
        "This IR system uses a hybrid retrieval pipeline:"
    )
    st.markdown(
        "1.  **Query Transformation:** Your query is rewritten by a T5 model.\n"
        "2.  **Hybrid Retrieval:** A `EnsembleRetriever` fuses results (50/50) from:\n" # Added (50/50)
        "    - **BM25** (Keyword Search)\n"
        "    - **FAISS** (Vector Search)\n"
        "3.  **Re-ranking:** A `CrossEncoder` re-ranks the hybrid results for final accuracy."
    )
    st.divider()
    
    st.subheader("System Status")
    db_ready = check_vector_db()
    api_ready = check_backend_api()
    
    if db_ready:
        st.success(" FAISS & BM25 Indexes are ready.")
    else:
        st.error(" Indexes not found! Please run: `python 2_ingest.py`")

    if api_ready:
        st.success(" Backend API is running.")
    else:
        st.error(" Backend API not running! Please run: `uvicorn 3_backend_api:app --reload`")
    
    st.divider()
    st.caption("Built with Streamlit, FastAPI, & LangChain.")

# --- Main Page ---
col1, col2 = st.columns(2)

# --- Column 1: Search Controls ---
with col1:
    st.title("Domain-Specific Search")
    st.caption("Powered by Hybrid Retrieval (BM25+FAISS) and Re-ranking")

    if 'input_query' not in st.session_state:
        st.session_state.input_query = ""

    st.subheader("Select from your search history:")
    def set_query_from_button(question):
        st.session_state.input_query = question

    suggestions = get_suggestions_from_db()
    if not suggestions:
        suggestions = ["What are greenhouse gases?", "What is the Paris Agreement?", "Tell me about renewable energy"]

    cols_buttons = st.columns(3)
    for i, question in enumerate(suggestions[:3]):
        cols_buttons[i].button(
            question, on_click=set_query_from_button, args=(question,), use_container_width=True
        )
    
    st.divider()
    # --- Search Interface ---
    user_query = st.text_input(
        "Ask a new question:", # <-- Changed label to be more direct
        key="input_query",
        placeholder="e.g., What are the causes of climate change?"
    )

# --- Column 2: Results Pane ---
with col2:
    st.subheader("Top 5 Results:")
    
    if user_query:
        if len(user_query.strip()) < 15:
            st.warning("Please enter a more specific query (at least 15 characters).")
            st.stop()
        
        with st.spinner("Searching, retrieving, and re-ranking..."):
            try:
                payload = {"query": user_query}
                response = requests.post(API_SEARCH_URL, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    st.info(f"**Original Query:** `{data['original_query']}`\n"
                            f"**Transformed Query:** `{data['transformed_query']}`")
                    st.divider()
                    
                    if data["results"]:
                        for i, result in enumerate(data["results"]):
                            with st.container(border=True):
                                col_res_1, col_res_2 = st.columns([4, 1])
                                with col_res_1:
                                    st.markdown(f"**Source:** `{result['source']}`")
                                with col_res_2:
                                    st.markdown(f"**Score:** `{result['relevance_score']:.4f}`")

                                with st.expander("Show Retrieved Text", expanded=i < 2):
                                    st.text(result['text'])
                            st.markdown("")
                    else:
                        st.warning("No relevant results found.")
                else:
                    st.error(f"Error from backend: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the backend API: {e}")