import os
import pickle
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- CONFIGURATION ---
DATA_PATH = "data"
FAISS_INDEX_PATH = "faiss_index"
BM25_INDEX_PATH = "bm25_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# ---------------------

def replace_t_with_space(list_of_documents):
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')
    return list_of_documents
 
def ingest_data():
    """
    Loads PDF, splits it, and creates/saves BOTH a FAISS index
    (for vector search) and a BM25 index (for keyword search).
    """
    print(f"Starting ingestion from '{DATA_PATH}'...")
    
    # 1. Load Documents
    loader = PyPDFLoader(os.path.join(DATA_PATH, "Understanding_Climate_Change.pdf"))
    documents = loader.load()

    if not documents:
        print("No PDF document found. Exiting.")
        return

    print(f"Loaded {len(documents)} pages from the PDF.")

    # 2. Split Documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)
    print(f"Split document into {len(cleaned_texts)} chunks.")

    # --- 3. Create and Save FAISS (Vector) Index ---
    print("Initializing OpenAI Embeddings...")
    embeddings = OpenAIEmbeddings()
    
    print("Creating FAISS index... This may take a moment.")
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved to '{FAISS_INDEX_PATH}'")

    # --- 4. Create and Save BM25 (Keyword) Index ---
    print("Creating BM25 index...")
    # Get the text content of all chunks
    tokenized_docs = [doc.page_content.split() for doc in cleaned_texts]
    bm25 = BM25Okapi(tokenized_docs)
    
    # Save the BM25 model and the document chunks
    # with open(os.path.join(BM25_INDEX_PATH, "bm25_model.pkl"), 'wb') as f:
    #     pickle.dump(bm25, f)

    print("Saving cleaned texts for BM25...")
        
    with open(os.path.join(BM25_INDEX_PATH, "cleaned_texts.pkl"), 'wb') as f:
        pickle.dump(cleaned_texts, f)

    print(f"BM25 index and texts saved to '{BM25_INDEX_PATH}'")
    print("\nIngestion complete.")

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory '{DATA_PATH}' not found.")
    
    # Create index directories if they don't exist
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    os.makedirs(BM25_INDEX_PATH, exist_ok=True)
    
    ingest_data()