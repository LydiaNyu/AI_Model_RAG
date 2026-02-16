# Personal Research Portal - RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for academic research papers on AI-assisted programming, built with LangChain, ChromaDB, and OpenAI.

## 📁 Project Structure

```
AI_Model_RAG/
├── .env                    # Environment variables (OPENAI_API_KEY)
├── requirements.txt        # Python dependencies
├── data/
│   ├── raw/                # 15 PDF research papers
│   ├── data_manifest.csv   # Metadata for all papers
│   └── chroma_db/          # Pre-built vector store (already generated)
├── src/
│   ├── ingest.py           # Data ingestion pipeline (already run)
│   ├── rag_engine.py       # RAG query engine (main entry point)
│   └── run_eval.py         # Evaluation script (20 research questions)
├── logs/
│   ├── query_log.json      # Detailed query audit logs
│   └── evaluation_results.json  # Evaluation results
└── report/
    └── [Project Report]    # Research report
```

---

## 🚀 Quick Start for TA

### Step 1: Create and Activate Virtual Environment

```bash
# Navigate to project directory
cd AI_Model_RAG

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set OpenAI API Key

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=sk-your-api-key-here

# Optional: Specify model (defaults to gpt-4o)
OPENAI_MODEL=gpt-4o

# Optional: Embedding model (defaults to text-embedding-3-small)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### Step 4: (Optional) Run Data Ingestion

> ⚠️ **Note**: The data ingestion has **already been run**. The vector store is pre-built and stored in `data/chroma_db/`. **Skip this step unless you need to re-embed the documents.**

```bash
cd src
python ingest.py
```

### Step 5: Run the RAG Engine

Run the RAG engine directly:

```bash
cd src
python rag_engine.py
```

### Step 6: Test with a Question

When prompted, enter a test question:

```
📚 Question: Does using AI assistants improve developer productivity, and are there any security risks?
```

**Expected Output:**
- The terminal will display the **answer** directly
- For **detailed retrieved chunks** and full audit information, check the **last entry** in `logs/query_log.json`


## 📊 Running Full Evaluation (Optional)

To run all 20 research questions:

```bash
cd src
python run_eval.py
```

Results saved to `logs/evaluation_results.json`.

---

## 📝 Log Format (Auditability)

Each query is logged in `logs/query_log.json` with:

```json
{
  "timestamp": "2026-02-15T...",
  "query": "What does the corpus say about...",
  "retrieved_source_ids": ["Weber2024", "Peng2023"],
  "retrieved_chunks": [
    {
      "source_id": "Weber2024",
      "title": "Significant Productivity Gains...",
      "year": 2024,
      "page": 5,
      "authors": "Weber, T. et al.",
      "content_snippet": "First 300 characters of retrieved text..."
    }
  ],
  "full_response": "Complete LLM response with citations...",
  "model_name": "gpt-4o",
  "retrieval_method": "MMR + CrossEncoder"
}
```

---

## � Key Features

| Feature | Description |
|---------|-------------|
| **MMR Retrieval** | Maximal Marginal Relevance for diverse results |
| **Cross-Encoder Reranking** | Neural reranking with `ms-marco-MiniLM-L-6-v2` |
| **Strict Grounding** | LLM answers only from provided context |
| **Citation Enforcement** | Every claim cites source (e.g., `(Peng2023)`) |
| **Audit Logging** | Full chunk details + content snippets logged |

---

## �️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `OPENAI_API_KEY not found` | Create `.env` file with your API key |
| `Vector store not found` | Run `python src/ingest.py` (normally not needed) |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Cross-encoder warnings | Install: `pip install sentence-transformers` |