# ClutchQuery 🏀
 
An AI-powered NBA stats chatbot that answers both descriptive and aggregation questions about the 2025-26 NBA season using a hybrid RAG (Retrieval-Augmented Generation) architecture.

## Features
- Ask descriptive questions about specific games: *"How did Wembanyama play vs the Kings on February 21st?"*
- Ask aggregation questions across multiple games: *"How many points is SGA averaging this season?"*

## Architecture
 
```
User Question
      ↓
   LLM Router (Groq - Llama 3.3 70B)
      ↓              ↓
  Pinecone         PostgreSQL
(descriptive)    (aggregation)
      ↓              ↓
   LLM generates final answer
```

### Data Pipeline
1. **`ingest.py`** — Fetches boxscores from the NBA API, uploads CSV to AWS S3
2. **`load_db.py`** — Loads CSV into Supabase PostgreSQL
3. **`embed.py`** — Embeds text chunks into Pinecone using Pinecone's multilingual-e5-large model

### Backend
- **FastAPI** — REST API with a single `/ask` endpoint
- **Groq (Llama 3.3 70B)** — LLM for routing, SQL generation, and answer generation
- **Pinecone** — Vector database for semantic search on individual game descriptions
- **PostgreSQL** — Relational database for aggregation queries

## Tech Stack
 
| Layer | Technology |
|---|---|
| LLM | Groq (Llama 3.3 70B) |
| Vector DB | Pinecone |
| Relational DB | PostgreSQL |
| Embeddings | Pinecone multilingual-e5-large |
| Backend | FastAPI |
| Frontend | React + TypeScript |
| Data Source | NBA API |
| Storage | AWS S3 |
| Backend Hosting | Railway |
| Frontend Hosting | Vercel |
