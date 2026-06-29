# Vectorless RAG

> FAQ / Document Q&A using **BM25 keyword retrieval** + **Ollama local LLM**.  
> **No embeddings. No vector DB. Zero cost.**

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────┐
│  BM25Retriever  │  ← rank_bm25 (Okapi BM25, pure keyword)
│  (no embeddings)│
└────────┬────────┘
         │  Top-K chunks
         ▼
┌─────────────────┐
│  Ollama LLM     │  ← llama3.2 running locally
│  (local, free)  │
└────────┬────────┘
         │
         ▼
      Answer + Sources
```

---

## Quick Start

### 1. Install Ollama & pull a model
```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2
```

### 2. Clone & install
```bash
git clone <repo>
cd vectorless-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 4. Run
```bash
uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Ollama status + index stats |
| `POST` | `/api/v1/ingest/upload` | Upload & index a document |
| `GET`  | `/api/v1/ingest/documents` | List indexed documents |
| `DELETE` | `/api/v1/ingest/documents/{doc_id}` | Remove a document |
| `POST` | `/api/v1/query` | Ask a question (JSON response) |
| `POST` | `/api/v1/query/stream` | Ask a question (SSE streaming) |
| `POST` | `/api/v1/query/retrieve-only` | BM25 retrieval only (debug) |

---

## Example Usage

```bash
# 1. Upload a document
curl -X POST http://localhost:8000/api/v1/ingest/upload \
  -F "file=@company_faq.pdf"

# 2. Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "top_k": 5}'
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Upgrade Path

| Step | What to add | Why |
|------|------------|-----|
| V2 | Ollama embeddings + ChromaDB | Semantic search for synonyms |
| V2 | Hybrid BM25 + vector (RRF) | Best of both worlds |
| V3 | Redis cache for repeated queries | Speed + cost |
| V3 | LangGraph re-ranking agent | Smarter context selection |

---

## Stack
- **FastAPI** — API framework
- **rank-bm25** — Okapi BM25 retrieval (no vector DB)
- **LangChain + langchain-ollama** — LLM orchestration
- **Ollama** — Local LLM runner (llama3.2)
- **PyMuPDF** — PDF parsing
- **Loguru** — Structured logging
