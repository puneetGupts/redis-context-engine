"""
FastAPI application — ties all layers together.

Endpoints:
  POST /chat            Main Q&A endpoint
  GET  /session/{id}    View session history and metadata
  DELETE /session/{id}  Clear a session
  GET  /health          Health check
"""

import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieval import vector_search
from src.memory import session as session_store
from src.cache import semantic_cache
from src.assembly import context_assembler
from src.generation import llm_client
from config.settings import settings

app = FastAPI(
    title="Redis Context Engine",
    description=(
        "Context assembly pipeline for AI agents — built on Redis Vector Sets, "
        "session memory, and semantic caching."
    ),
    version="1.0.0",
)


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique identifier for the user session")
    question: str = Field(..., description="User's question", min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    cache_hit: bool
    cache_similarity: float | None = None
    chunks_used: int
    session_turns: int
    latency_ms: float


# ── Main chat endpoint ────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer a question using Redis-powered context assembly.

    Flow:
      1. Check semantic cache — return instantly if similar query was seen before
      2. Retrieve relevant doc chunks from Redis Vector Sets
      3. Load session memory from Redis Hash
      4. Assemble context window (chunks + history + question)
      5. Call Claude with assembled context
      6. Store response in cache and update session memory
      7. Return answer with metadata
    """
    start = time.time()

    # ── Step 1: Semantic cache lookup ─────────────────────────────────────────
    cache_result = semantic_cache.lookup(request.question)
    if cache_result.hit:
        history = session_store.get_history(request.session_id)
        session_store.append_turn(
            request.session_id,
            request.question,
            cache_result.answer,
        )
        return ChatResponse(
            answer=cache_result.answer,
            sources=[],
            cache_hit=True,
            cache_similarity=cache_result.similarity,
            chunks_used=0,
            session_turns=len(history) // 2,
            latency_ms=round((time.time() - start) * 1000, 2),
        )

    # ── Step 2: Vector search ─────────────────────────────────────────────────
    retrieved_chunks = vector_search.search(request.question)

    if not retrieved_chunks:
        raise HTTPException(
            status_code=404,
            detail=(
                "No relevant documentation found for this question. "
                "Try rephrasing or check that the docs have been ingested."
            ),
        )

    # ── Step 3: Session memory ────────────────────────────────────────────────
    history = session_store.get_history(request.session_id)

    # ── Step 4: Context assembly ──────────────────────────────────────────────
    assembled = context_assembler.assemble(
        question=request.question,
        retrieved_chunks=retrieved_chunks,
        history=history,
    )

    # ── Step 5: LLM generation ────────────────────────────────────────────────
    # Provider routed via LLM_PROVIDER in .env (anthropic|openai).
    # Retrieval / cache / memory layers are unaware of which model answers.
    answer = llm_client.generate(
        system=context_assembler.SYSTEM_PROMPT,
        messages=assembled.prompt_messages,
        max_tokens=1024,
    )

    # ── Step 6: Store in cache + update session memory ────────────────────────
    semantic_cache.store(request.question, answer)
    session_store.append_turn(request.session_id, request.question, answer)

    # ── Step 7: Return ────────────────────────────────────────────────────────
    return ChatResponse(
        answer=answer,
        sources=assembled.sources,
        cache_hit=False,
        cache_similarity=None,
        chunks_used=assembled.chunks_used,
        session_turns=assembled.history_turns_used,
        latency_ms=round((time.time() - start) * 1000, 2),
    )


# ── Session endpoints ─────────────────────────────────────────────────────────

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """View session history and metadata for a given session ID."""
    info = session_store.get_session_info(session_id)
    if not info["exists"]:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return info


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clear all memory for a session."""
    session_store.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared"}


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Check that Redis is reachable and the vector index exists."""
    import redis as redis_lib
    r = redis_lib.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=True,
    )
    try:
        r.ping()
        redis_status = "ok"
    except Exception as e:
        redis_status = f"error: {e}"

    try:
        index_info = r.ft(settings.DOCS_INDEX_NAME).info()
        doc_count = index_info.get("num_docs", "unknown")
        index_status = "ok"
    except Exception:
        doc_count = 0
        index_status = "not found — run scripts/ingest.py first"

    return {
        "status": "ok" if redis_status == "ok" else "degraded",
        "redis": redis_status,
        "docs_index": index_status,
        "indexed_chunks": doc_count,
    }
