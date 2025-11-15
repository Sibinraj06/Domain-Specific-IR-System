# Domain-Specific Information Retrieval System

## 🚀 Key Features

  * **Hybrid Retrieval:** Implements an `EnsembleRetriever` that combines **BM25 (keyword search)** and **FAISS (vector search)** to find the most relevant document chunks.
  * **Two-Stage Ranking:** Uses a fast hybrid retrieval (Stage 1) followed by a high-accuracy **Cross-Encoder Re-ranker** (Stage 2) to provide the most precise answers and meaningful relevance scores.
  * **Query Intelligence:** Employs a `google/flan-t5-base` model to automatically transform user queries into more effective search terms, enhancing retrieval accuracy.
  * **Modular API Architecture:** The system is decoupled into a **FastAPI backend** (for all AI logic) and a **Streamlit frontend** (for the UI), which is a scalable and professional design.
  * **Interactive UI:** The Streamlit app features a two-column layout, a system status sidebar, and clickable search history buttons.
  * **Controllable Retrieval:** Includes an interactive **slider** that allows the user to adjust the balance (`alpha`) between keyword and vector search in real-time.
  * **Search History:** Logs all successful queries to an SQLite database (`query_log.db`) and displays them as clickable suggestions.

##  System Architecture

The data flow is a multi-stage process designed for maximum accuracy.

1.  **Offline (Ingestion):**

      * The `ingest.py` script loads the domain PDF from the `data/` folder.
      * It splits the text into manageable chunks.
      * It creates and saves two indexes: a **FAISS vector index** (using OpenAI embeddings) and the **cleaned texts** for the BM25 index.

2.  **Online (Querying):**

      * The user enters a query in the **Streamlit** frontend.
      * The UI validates the query (must be \> 15 characters).
      * The query and `alpha` setting are sent to the **FastAPI** `/search` endpoint.
      * **(Step 1: Transform)** The T5 model rewrites the query for better search.
      * **(Step 2: Retrieve)** The `EnsembleRetriever` (BM25 + FAISS) uses the `alpha` weight to fetch the top-K hybrid candidates.
      * **(Step 3: Re-rank)** The `CrossEncoder` re-scores all candidates against the *original* user query to find the true best answer.
      * **(Step 4: Log)** The original query is logged to `query_log.db`.
      * **(Step 5: Respond)** The backend returns the Top-N sorted, re-ranked results to the Streamlit UI, which displays them in the results pane.

## Setup & Installation

Follow these steps to run the project locally.

### 1\. Prerequisites

  * Python 3.12
  * An **OpenAI API Key** (for creating the embeddings)

### 2\. Installation

1.  **Clone the repository (or create the file structure):**

    ```bash
    git clone https://github.com/Sibinraj06/Domain-Specific-IR-System.git
    cd Domain-Specific-IR-System
    ```

2.  **Create a Virtual Environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Set Your API Key:**
    You *must* set your OpenAI API key as an environment variable.

    ```bash
    export OPENAI_API_KEY="sk-YourSecretKeyHere"
    ```

4.  **Install Dependencies:**
    Install all required packages from the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

### 3\. Running the System

1.  **Add Your Data:**
    Place your domain-specific PDF (e.g., `Understanding_Climate_Change.pdf`) inside the `data/` folder.

2.  **Run the Ingestion Script:**
    This script *must* be run once to process your PDF and create the indexes.

    ```bash
    ingest.py
    ```

    This will create the `faiss_index/` and `bm25_index/` folders.

3.  **Run the Backend API:**
    In your first terminal, start the FastAPI server.

    ```bash
    uvicorn backend_api:app --reload
    ```

    Wait for it to display: `Application startup complete.`

4.  **Run the Frontend UI:**
    In a *second* terminal, run the Streamlit app.

    ```bash
    streamlit run frontend_app.py
    ```

    Your default web browser will automatically open to the app.

## File Structure

```
domain-specific-ir-system/
│
├── requirements.txt        # All Python dependencies
├── ingest.py               # Script to run once: Ingests PDF, creates indexes
├── backend_api.py          # FastAPI server: Handles all AI/ML logic
├── rontend_app.py         # Streamlit UI: The web app you interact with
│
├── data/
│   └── Understanding_Climate_Change.pdf  # Your source document(s)
│
├── faiss_index/                # Stores the FAISS vector index (created by 2_ingest.py)
│   ├── index.faiss
│   └── index.pkl
│
├── bm25_index/                 # Stores the BM25 data (created by 2_ingest.py)
│   └── cleaned_texts.pkl
│
└── query_log.db                # SQLite DB for search history (created by 3_backend_api.py)
```