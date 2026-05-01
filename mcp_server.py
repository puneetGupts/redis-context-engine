#!/usr/bin/env python3
"""
Redis Context Engine — MCP Server

Exposes the context engine as MCP tools so any MCP client
(Claude Desktop, Claude Code, etc.) can use Redis-powered
context assembly transparently.

Tools exposed:
  ask_redis        — Full pipeline Q&A (cache → retrieval → memory → LLM)
  search_redis_docs — Pure semantic search, returns raw chunks
  clear_session    — Wipe conversation memory for a session

Usage (Claude Desktop):
  Add to ~/Library/Application Support/Claude/claude_desktop_config.json
  See README for exact config snippet.
"""

import sys
import os
import time

# Ensure project root is on the path so src.* imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

from src.retrieval import vector_search
from src.memory import session as session_store
from src.cache import semantic_cache
from src.assembly import context_assembler
from src.generation import llm_client

# ── MCP Server definition ─────────────────────────────────────────────────────

mcp = FastMCP(
    name="Redis Context Engine",
    instructions=(
        "Use these tools to answer questions about Redis. "
        "ask_redis runs the full pipeline: semantic cache check, "
        "vector retrieval from Redis docs, session memory injection, "
        "and LLM generation. Use the same session_id across turns "
        "to maintain conversation context."
    ),
)


# ── Tool 1: Full pipeline Q&A ─────────────────────────────────────────────────

@mcp.tool()
def ask_redis(question: str, session_id: str = "default") -> str:
    """
    Ask a question about Redis using the full context engine pipeline.

    Runs three Redis layers before touching the LLM:
      1. Semantic cache  — returns instantly if a similar question was seen before
      2. Vector retrieval — finds the most relevant Redis doc chunks
      3. Session memory  — injects semantically relevant past turns

    Args:
        question:   Your question about Redis
        session_id: Session identifier — use the same value across turns to
                    maintain conversation context (default: "default")
    """
    start = time.time()

    # ── Layer 1: Semantic cache ───────────────────────────────────────────────
    cache_result = semantic_cache.lookup(question)
    if cache_result.hit:
        session_store.append_turn(session_id, question, cache_result.answer)
        latency = round((time.time() - start) * 1000, 1)
        return (
            f"{cache_result.answer}\n\n"
            f"---\n"
            f"⚡ Cache hit (similarity: {cache_result.similarity:.2f}) · {latency}ms · No LLM call made"
        )

    # ── Layer 2: Vector retrieval ─────────────────────────────────────────────
    chunks = vector_search.search(question)
    if not chunks:
        return (
            "No relevant Redis documentation found for that question. "
            "Try rephrasing, or run `python scripts/ingest.py` to expand the knowledge base."
        )

    # ── Layer 3: Session memory ───────────────────────────────────────────────
    history = session_store.get_relevant_history(session_id, question)

    # ── Context assembly ──────────────────────────────────────────────────────
    assembled = context_assembler.assemble(
        question=question,
        retrieved_chunks=chunks,
        history=history,
    )

    # ── LLM generation ────────────────────────────────────────────────────────
    answer = llm_client.generate(
        system=context_assembler.SYSTEM_PROMPT,
        messages=assembled.prompt_messages,
        max_tokens=1024,
    )

    # ── Store in cache + session ──────────────────────────────────────────────
    semantic_cache.store(question, answer)
    session_store.append_turn(session_id, question, answer)

    latency = round((time.time() - start) * 1000, 1)
    sources_text = (
        "\n".join(f"  - {s}" for s in assembled.sources)
        if assembled.sources else "  (no sources)"
    )

    return (
        f"{answer}\n\n"
        f"---\n"
        f"📚 Sources:\n{sources_text}\n"
        f"🔍 Chunks used: {assembled.chunks_used}  "
        f"💬 Session turns: {assembled.history_turns_used}  "
        f"⏱ {latency}ms"
    )


# ── Tool 2: Pure semantic search ──────────────────────────────────────────────

@mcp.tool()
def search_redis_docs(query: str) -> str:
    """
    Search Redis documentation and return matching chunks without generating an answer.

    Useful for exploring what's in the knowledge base or verifying that
    specific topics are indexed before asking questions about them.

    Args:
        query: What to search for (natural language)
    """
    chunks = vector_search.search(query)
    if not chunks:
        return "No relevant documentation found for that query."

    results = []
    for i, chunk in enumerate(chunks, 1):
        preview = chunk.text[:250].replace("\n", " ").strip()
        results.append(
            f"[{i}] {chunk.title}  ({chunk.section})\n"
            f"     URL:        {chunk.url}\n"
            f"     Similarity: {chunk.similarity_score:.3f}\n"
            f"     Preview:    {preview}..."
        )

    return f"Found {len(chunks)} relevant chunks:\n\n" + "\n\n".join(results)


# ── Tool 3: Clear session memory ──────────────────────────────────────────────

@mcp.tool()
def clear_session(session_id: str = "default") -> str:
    """
    Clear all conversation memory for a session.

    Call this to start a fresh conversation without history from previous turns.

    Args:
        session_id: The session to clear (default: "default")
    """
    session_store.clear_session(session_id)
    return f"Session '{session_id}' cleared. Starting fresh."


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
