# Redis Context Engine

A production-grade context assembly pipeline for AI agents, built on Redis Cloud. Retrieves relevant documentation using Redis Vector Sets, maintains per-user session memory via Redis Hashes, and caches semantically similar queries to reduce token cost and latency.

Inspired by the architecture of [Redis Context Engine](https://redis.io/contexts/) — and built on the same primitives.

---

## Why This Exists

LLMs are stateless and context-limited. When an agent needs to answer questions about a large knowledge base, it cannot load everything into the prompt. The real problem is:

> Given a user's question, how do you find the right information, assemble it into a focused context window, and deliver it to the LLM — fast, accurately, and without burning tokens on repeated queries?

This project solves that problem using Redis as the single infrastructure layer for vector search, session memory, and semantic caching — the same architecture Redis's Context Engine provides as managed infrastructure.

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────┐
│  Semantic Cache  │ ──── Cache hit? Return instantly (no LLM call)
└────────┬────────┘
         │ Cache miss
         ▼
┌─────────────────┐
│  Vector Search   │ ──── Redis Vector Sets → Top-K relevant doc chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Session Memory  │ ──── Redis Hash → Last N messages for this user
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Context Assembler │ ──── Retrieved chunks + history + question → prompt
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Claude API     │ ──── Generate answer with assembled context
└────────┬────────┘
         │
         ▼
  Store in cache + Update session memory → Return answer
```

### Components

| Component | Technology | Responsibility |
|-----------|-----------|----------------|
| Vector Store | Redis Vector Sets | Store and search embedded doc chunks |
| Session Memory | Redis Hash (TTL 30min) | Per-user conversation history |
| Semantic Cache | Redis Vector Sets | Deduplicate similar queries |
| Context Assembler | Python | Rank, trim, and assemble prompt |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | 384-dim local embeddings — free, no API key needed |
| LLM | Anthropic / OpenAI / Groq (configurable) | Generation via `LLM_PROVIDER` env var |
| API | FastAPI | HTTP interface |

---

## Key Design Decisions

**Chunking strategy (500 tokens, 50 token overlap)**
Too small and chunks lack context. Too large and irrelevant content pollutes the prompt. 50-token overlap ensures ideas don't get cut off at chunk boundaries.

**Similarity threshold (0.75)**
Mathematically similar chunks aren't always contextually relevant. Anything below 0.75 cosine similarity is dropped even if it's in the top-K results.

**Semantic caching vs. exact caching**
Exact caching misses variations of the same question. Semantic caching embeds the query and checks similarity against cached queries — if similarity > 0.90, return the cached response.

**Session isolation**
Each user gets a unique session ID. Memory is stored under `session:{session_id}` in Redis with a 30-minute TTL. Sessions never bleed across users.

**Token budget**
Context assembly enforces a hard cap before sending to Claude: top-3 retrieved chunks + last 3 conversation turns + current question. Prioritizes recency in history and relevance in retrieved chunks.

---

## Project Structure

```
redis-context-engine/
├── src/
│   ├── ingestion/
│   │   ├── scraper.py          # Scrape Redis docs from redis.io/docs
│   │   ├── chunker.py          # Split docs into overlapping chunks
│   │   └── indexer.py          # Embed chunks + store in Redis Vector Sets
│   ├── retrieval/
│   │   └── vector_search.py    # Semantic search over indexed docs
│   ├── memory/
│   │   └── session.py          # Per-user session memory (Redis Hash)
│   ├── cache/
│   │   └── semantic_cache.py   # Semantic response cache (Redis)
│   ├── assembly/
│   │   └── context_assembler.py # Assemble context window for LLM
│   ├── generation/
│   │   └── llm_client.py       # Multi-provider LLM routing (anthropic/openai/groq)
│   └── api/
│       └── main.py             # FastAPI application
├── config/
│   └── settings.py             # Centralized config from environment
├── scripts/
│   └── ingest.py               # One-time ingestion script
├── docs/
│   └── ARCHITECTURE.md         # Deep-dive architecture documentation
├── architecture/               # Architecture diagrams
├── .env.example                # Environment variable template
└── requirements.txt
```

---

## Setup

### 1. Redis Cloud

Sign up for a free Redis Cloud account at [redis.io/try-free](https://redis.io/try-free).

Create a database and note your:
- Host
- Port
- Password

### 2. LLM Provider (pick one)

The engine supports three providers — set `LLM_PROVIDER` in your `.env`:

| Provider | Key needed | Notes |
|----------|-----------|-------|
| `groq` | `GROQ_API_KEY` | Free tier, fast — recommended for getting started |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude models |
| `openai` | `OPENAI_API_KEY` | GPT models, or any OpenAI-compatible endpoint |

Embeddings are handled locally by `sentence-transformers` — no API key needed.

### 3. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env — set Redis credentials and choose your LLM_PROVIDER
```

### 5. Ingest Redis Documentation

This is a one-time step. Scrapes Redis docs, chunks them, and indexes into Redis Vector Sets.

```bash
python scripts/ingest.py
```

### 6. Start the API

```bash
uvicorn src.api.main:app --reload
```

---

## Usage

### Ask a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123",
    "question": "How does Redis Vector Search work?"
  }'
```

Response:
```json
{
  "answer": "Redis Vector Search allows you to...",
  "sources": ["https://redis.io/docs/interact/search-and-query/advanced-concepts/vectors/"],
  "cache_hit": false,
  "chunks_used": 3,
  "session_turns": 0
}
```

### Follow-up question (session memory in action)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123",
    "question": "What index type should I use for that?"
  }'
```

Claude knows what "that" refers to — session history is injected automatically.

### Ask the same question again (cache in action)

The second request returns instantly with `"cache_hit": true` — no LLM call made.

### View session history

```bash
curl http://localhost:8000/session/user-123
```

---

## Demo Highlights

| Feature | What it shows |
|---------|--------------|
| Vector retrieval | Relevant doc chunks surface without keyword matching |
| Session memory | Follow-up questions maintain context across turns |
| Semantic cache | Repeated/similar queries return instantly, zero token cost |
| Source attribution | Every answer cites the Redis doc page it came from |
| Session isolation | Different session IDs get completely independent memory |

---

## Pain Points Solved

**Problem:** LLMs can't hold entire documentation in context.
**Solution:** Redis Vector Sets index all doc chunks. Only the relevant 3 are injected per query.

**Problem:** Every question burns tokens, even repeated ones.
**Solution:** Semantic cache checks for similar past queries (similarity > 0.90). Cache hits return in <50ms.

**Problem:** Follow-up questions lose context ("what about its TTL?").
**Solution:** Session memory injects the last 3 conversation turns for every query.

**Problem:** Multiple users need isolated context.
**Solution:** Each session ID maps to a separate Redis Hash with independent TTL.

---

## Connection to Redis Context Engine

This project is a self-built implementation of the same primitives that [Redis Context Engine](https://redis.io/contexts/) provides as managed infrastructure:

| This project | Redis Context Engine |
|---|---|
| Redis Vector Sets for retrieval | Managed vector storage and search |
| Redis Hash for session memory | Managed agent memory |
| Semantic cache on Redis | Managed response deduplication |
| Manual context assembly | Managed context window optimization |

The difference is scale, reliability, and the managed layer. This project demonstrates understanding of the underlying architecture.

---

## Author

**Puneet Gupta** — [LinkedIn](https://linkedin.com/in/puneet-gupta-6bb335ab) · [GitHub](https://github.com/puneetGupts)

Built this to understand Redis's context infrastructure primitives from the ground up — after building a similar production system at Paycom for multi-agent RCA workflows.
