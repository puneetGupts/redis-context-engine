# Architecture Deep Dive

## Problem Statement

Large language models have two fundamental limitations for documentation QA:

1. **Context window limits** — You cannot fit an entire documentation corpus into a single prompt. Redis has hundreds of doc pages; Claude's context window has a token cap.
2. **Statelessness** — Every API call starts from scratch. Follow-up questions like "what about its TTL?" break without conversational memory.

Additionally, in any real system:
- The same questions get asked repeatedly → every call burns tokens and adds latency
- Multiple users need isolated memory → contexts must not bleed across sessions

This architecture solves all four problems using Redis as the single infrastructure layer.

---

## System Components

### 1. Ingestion Pipeline (offline, run once)

```
redis.io/docs
     │
     ▼
[Scraper]          beautifulsoup4 — extracts clean text, skips nav/footer
     │
     ▼
[Chunker]          500-token chunks with 50-token overlap
     │                - Splits at paragraph boundaries first
     │                - Falls back to sentence splits for long paragraphs
     │                - Overlap preserves ideas at chunk boundaries
     ▼
[Indexer]          sentence-transformers/all-MiniLM-L6-v2 (384 dims)
     │                - Free, local — no API key needed
     │                - Redis Pipeline for batch writes
     │                - HNSW index (cosine distance)
     ▼
Redis Vector Sets  doc:{hash} → {text, url, title, section, embedding}
```

**Why HNSW over FLAT index?**

| | HNSW | FLAT |
|---|---|---|
| Search complexity | O(log n) | O(n) |
| Accuracy | ~99% recall | 100% |
| Memory | Higher | Lower |
| Production use | ✅ | Small datasets only |

For a documentation QA system where 99% recall is acceptable, HNSW is the correct choice. FLAT becomes a bottleneck at thousands of chunks.

---

### 2. Semantic Cache

```
User question
     │
     ▼
[Embed query]       all-MiniLM-L6-v2 (384-dim, same model as index time)
     │
     ▼
[KNN search]        Top-1 match against cached query embeddings
     │
     ▼
similarity ≥ 0.90?
     │
  YES │                              NO │
     ▼                                ▼
Return cached answer           Proceed to retrieval
(increment hit counter)
```

**Threshold choice: 0.90 (vs. 0.75 for retrieval)**

The cache threshold is higher than the retrieval threshold deliberately:
- Retrieval: cast a wide net, partial matches are useful context
- Cache: reusing an answer — near-miss is worse than a fresh LLM call

At 0.90, "How does Redis Vector Search work?" and "Explain Redis vector search to me" will hit the cache. "How does Redis Vector Search compare to Elasticsearch?" will not.

---

### 3. Vector Search (Retrieval)

```
User question
     │
     ▼
[Embed query]       all-MiniLM-L6-v2 — same model used at index time (critical)
     │
     ▼
[KNN search]        Top-5 against redis_docs_idx
     │
     ▼
[Threshold filter]  Drop chunks with similarity < 0.75
     │
     ▼
Ranked chunks       Sorted by similarity, highest first
```

**Why same embedding model at index and query time?**

If you index with model A and query with model B, the vector spaces are
incompatible — similarity scores become meaningless. Model consistency is
non-negotiable in production vector systems.

---

### 4. Session Memory

```
Redis Hash: session:{session_id}
  messages:    JSON array of {role, content}
  created_at:  Unix timestamp
  last_active: Unix timestamp (refreshed on every read)
  TTL:         1800s (reset on every access)
```

**Design decisions:**

- **TTL reset on access** — Active sessions stay alive indefinitely. Inactive sessions expire automatically. No cleanup job needed.
- **Max N turns** — We keep only the last `MAX_HISTORY_TURNS` (default 3) conversation turns. Recency matters more than completeness — older history is less relevant and wastes tokens.
- **Per-session isolation** — Each `session_id` is a separate Redis Hash. There is no shared state between users.

---

### 5. Context Assembly

```
retrieved_chunks[:MAX_CONTEXT_CHUNKS]   (top 3 by relevance)
         +
history[-MAX_HISTORY_TURNS*2:]          (last 3 turns, most recent)
         +
current question
         │
         ▼
┌─────────────────────────────────────────┐
│ System: You are a Redis doc assistant…  │
│                                         │
│ <context>                               │
│   [Source 1] title (section)            │
│   URL: …                                │
│   Relevance: 0.92                       │
│   {chunk text}                          │
│                                         │
│   [Source 2] …                          │
│ </context>                              │
│                                         │
│ [history messages]                      │
│                                         │
│ <question>                              │
│   {user question}                       │
│ </question>                             │
└─────────────────────────────────────────┘
         │
         ▼
    LLM API (anthropic / openai / groq)
```

**Token budget:**
- System prompt: ~150 tokens
- 3 chunks at 500 tokens each: ~1500 tokens
- 3 turns of history at ~200 tokens each: ~600 tokens
- Current question: ~50 tokens
- **Total estimate: ~2300 tokens** — well within Claude's limits

---

## Data Flow: Full Request Lifecycle

```
POST /chat {session_id, question}
         │
         ├─1─► [Semantic Cache lookup]
         │          │
         │       cache hit ──────────────────────► return answer (< 50ms)
         │          │ miss
         │          ▼
         ├─2─► [Vector Search]
         │       embed question → KNN → filter → ranked chunks
         │          │
         │          ▼
         ├─3─► [Session Memory read]
         │       GET session:{id} → last N turns
         │          │
         │          ▼
         ├─4─► [Context Assembly]
         │       chunks + history + question → structured prompt
         │          │
         │          ▼
         ├─5─► [LLM API (anthropic/openai/groq)]
         │       messages.create → answer text
         │          │
         │          ▼
         ├─6─► [Cache store]
         │       embed question → store in cache index
         │          │
         │          ▼
         ├─7─► [Session Memory write]
         │       append turn → trim → set TTL
         │          │
         │          ▼
         └─────► return ChatResponse
```

---

## Redis Key Space

```
doc:{12-char-md5}          # indexed document chunks (HNSW vector index)
cache:{12-char-md5}        # semantic cache entries (HNSW vector index)
session:{session_id}       # user session memory (Hash, TTL 1800s)
```

**Three data structures, one Redis instance.** No separate vector DB, no separate cache service, no separate session store. This consolidation is the central architectural argument for Redis as context infrastructure.

---

## Comparison: This Project vs. Production Context Engine

| Aspect | This Project | Production Context Engine |
|--------|-------------|--------------------------|
| Vector storage | Redis Cloud free tier | Managed, multi-tenant |
| Embedding | sentence-transformers (all-MiniLM-L6-v2, 384-dim) | Configurable |
| Sources | Redis docs (scraped) | Any connector |
| Session memory | Redis Hash, 1 server | Distributed, replicated |
| Cache | Single-node Redis | Clustered, shared across instances |
| Context freshness | Manual re-ingest | TTL + change detection |
| Multi-tenancy | session_id isolation | Full tenant isolation |
| Scale | Single instance | Horizontally scalable |

The architecture is identical. The difference is operationalization.

---

## Open Problems (Discussion Points)

**Context freshness:** When Redis docs update, how do you invalidate the right vectors without re-indexing everything? Options: TTL on doc vectors, URL-based invalidation, webhook-triggered re-indexing.

**Multi-tenant context isolation:** How do you ensure one tenant's context cannot leak into another's retrieval results? Options: namespace prefixes, per-tenant indexes, row-level security on vector search.

**Embedding latency at scale:** Local embeddings (sentence-transformers) are free but CPU-bound — each query embedding takes ~200ms on a single core. At high RPS this becomes the bottleneck. Options: GPU inference, batching, or switching to a hosted embedding API for sub-10ms latency.

**Chunking quality:** Current chunking is syntactic (token-based). Semantic chunking (split at topic boundaries) would improve retrieval quality but requires an LLM pass at index time.
