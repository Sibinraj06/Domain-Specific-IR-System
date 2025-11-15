from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import warnings
import sqlite3
import pickle
import os

# --- Model Imports ---
from sentence_transformers import CrossEncoder # For re-ranking
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain.docstore.document import Document # Import Document

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- CONFIGURATION ---
FAISS_INDEX_PATH = "faiss_index"
BM25_INDEX_PATH = "bm25_index"
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
QUERY_TRANSFORM_MODEL = 'google/flan-t5-base'
DEVICE = 'cpu'
LOG_DB_PATH = "query_log.db"
# ---------------------

models = {}

def create_log_db():
    conn = sqlite3.connect(LOG_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL UNIQUE,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models and connecting to DBs...")
    
    # Load OpenAI Embeddings
    models['embeddings'] = OpenAIEmbeddings()
    
    # Load FAISS Index
    models['faiss_index'] = FAISS.load_local(
        FAISS_INDEX_PATH, 
        models['embeddings'],
        allow_dangerous_deserialization=True
    )
    models['faiss_retriever'] = models['faiss_index'].as_retriever(search_kwargs={"k": 10})
    print("FAISS retriever loaded.")

    # Load BM25 Index
    with open(os.path.join(BM25_INDEX_PATH, "cleaned_texts.pkl"), 'rb') as f:
        cleaned_texts = pickle.load(f)
    # with open(os.path.join(BM25_INDEX_PATH, "bm25_model.pkl"), 'rb') as f:
    #     bm25_model = pickle.load(f)
    
    models['bm25_retriever'] = BM25Retriever.from_documents(
        # bm25_model=bm25_model,
        documents=cleaned_texts
    )
    models['bm25_retriever'].k = 10
    print("BM25 retriever loaded.")

    # Create Ensemble Retriever
    models['ensemble_retriever'] = EnsembleRetriever(
        retrievers=[models['bm25_retriever'], models['faiss_retriever']],
        weights=[0.5, 0.5]  # Default weights, can be changed per query
    )
    print("Ensemble retriever created.")

    # Load Re-ranker (CrossEncoder)
    models['reranker'] = CrossEncoder(RERANKER_MODEL, device=DEVICE)
    print("CrossEncoder re-ranker loaded.")

    # Load Query Transformer
    models['query_tokenizer'] = T5Tokenizer.from_pretrained(QUERY_TRANSFORM_MODEL)
    models['query_transformer'] = T5ForConditionalGeneration.from_pretrained(QUERY_TRANSFORM_MODEL)
    print("Query transformer loaded.")
    
    # Connect to SQLite Log DB
    models['log_db_conn'] = create_log_db()
    print("All models and DB connections loaded.")
    
    yield
    
    # Clean up
    print("Cleaning up models and closing DB connections...")
    if 'log_db_conn' in models:
        models['log_db_conn'].close()
    models.clear()
    print("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

class SearchQuery(BaseModel):
    query: str
    top_n: int = 5
    alpha: float = 0.5 # Weight for BM25 vs FAISS

class SearchResult(BaseModel):
    text: str
    source: str
    relevance_score: float

class SearchResponse(BaseModel):
    original_query: str
    transformed_query: str
    results: list[SearchResult]

# --- 2. PROMPT UPGRADE ---
def transform_query(user_query: str) -> str:
    """Uses a T5 model to rewrite the user's query into a better search query."""
    
    # New, more explicit prompt
    prompt = (
        "You are an expert search query rewriter. "
        "Convert the following user question into a concise and effective keyword-based search query "
        "for a technical document database.\n\n"
        f"USER QUESTION: '{user_query}'\n\n"
        "SEARCH QUERY:"
    )
    
    inputs = models['query_tokenizer'](prompt, return_tensors="pt")
    outputs = models['query_transformer'].generate(**inputs, max_length=100)
    transformed_query = models['query_tokenizer'].decode(outputs[0], skip_special_tokens=True)
    
    # Clean up output if model adds extra text
    if transformed_query.startswith("SEARCH QUERY:"):
        transformed_query = transformed_query[len("SEARCH QUERY:"):].strip()
        
    print(f"Original Query: {user_query}")
    print(f"Transformed Query: {transformed_query}")
    return transformed_query

def log_query_to_db(query: str):
    try:
        conn = models['log_db_conn']
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO search_log (query) VALUES (?)", (query,))
        conn.commit()
    except Exception as e:
        print(f"Error logging query: {e}")

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchQuery):
    
    # 1. Query Intelligence: Transform the query
    transformed_query = transform_query(request.query)
    
    # 2. Stage 1: Hybrid Retrieval (FAISS + BM25)
    # Set the weights for the ensemble based on user's 'alpha'
    # Alpha = 0.5 -> 50% BM25, 50% FAISS
    # Alpha = 1.0 -> 100% BM25, 0% FAISS (pure keyword)
    # Alpha = 0.0 -> 0% BM25, 100% FAISS (pure vector)
    bm25_weight = request.alpha
    faiss_weight = 1.0 - request.alpha
    models['ensemble_retriever'].weights = [bm25_weight, faiss_weight]
    
    # Get hybrid results (these are not ranked yet)
    retrieved_docs = models['ensemble_retriever'].get_relevant_documents(transformed_query)
    
    if not retrieved_docs:
        return SearchResponse(original_query=request.query, transformed_query=transformed_query, results=[])

    # 3. Stage 2: Re-ranking
    reranker_inputs = [[request.query, doc.page_content] for doc in retrieved_docs]
    reranker_scores = models['reranker'].predict(reranker_inputs)
    
    final_results = []
    for doc, score in zip(retrieved_docs, reranker_scores):
        final_results.append({
            "text": doc.page_content,
            "source": doc.metadata.get('source', 'Unknown'),
            "relevance_score": float(score)
        })
    
    final_results.sort(key=lambda x: x['relevance_score'], reverse=True)
    top_n_results = final_results[:request.top_n]
    
    if top_n_results:
        log_query_to_db(request.query)
    
    return SearchResponse(
        original_query=request.query,
        transformed_query=transformed_query,
        results=top_n_results
    )

@app.get("/suggest")
async def get_suggestions(q: str):
    conn = models['log_db_conn']
    cursor = conn.cursor()
    if not q:
        cursor.execute("SELECT query FROM search_log ORDER BY timestamp DESC LIMIT 5")
    else:
        cursor.execute("SELECT query FROM search_log WHERE query LIKE ? ORDER BY timestamp DESC LIMIT 5", (q + '%',))
    results = [row[0] for row in cursor.fetchall()]
    return results

@app.get("/")
def read_root():
    return {"message": "Domain IR System API is running."}