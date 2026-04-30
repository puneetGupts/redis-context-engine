"""
Per-user session memory backed by redisvl SemanticSessionManager.

Upgrade from the raw Redis Hash implementation:
  The previous version stored the last N conversation turns by recency and
  injected all of them into every prompt. This works fine for short sessions
  but degrades as history grows — the context fills with irrelevant old turns.

  redisvl's SemanticSessionManager stores each message as a vector. When a
  new question arrives, we retrieve the most *semantically relevant* past
  turns rather than just the most recent ones. If a user asked about HNSW
  10 turns ago and now asks "what about that index type again?", semantic
  retrieval finds it. Last-N-turns retrieval doesn't.

Why redisvl here, but raw primitives for the semantic cache?
  The session memory upgrade is a genuine functional improvement — semantic
  search over history is meaningfully better than recency-only. The cache,
  by contrast, is the core primitive we want to demonstrate at the raw Redis
  level. Using redisvl for session memory shows we know when to reach for
  the managed layer; keeping the raw cache shows we understand what it
  abstracts.

Redis data model (managed by redisvl):
  Each message is stored as a Hash with a vector embedding under a
  prefixed key namespace. redisvl creates and manages the HNSW index.
  TTL is not natively supported by SemanticSessionManager — we handle
  expiry by clearing on explicit delete or via the /session DELETE endpoint.
"""

import redis as redis_lib
from dataclasses import dataclass
from typing import Optional

from redisvl.extensions.session_manager import SemanticSessionManager
from redisvl.utils.vectorize import HFTextVectorizer

from config.settings import settings


@dataclass
class Message:
    role: str      # "user" or "assistant"
    content: str


# Shared vectorizer — same model used everywhere in the pipeline so all
# vector spaces are compatible.
_vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")

# Redis URL built from settings
_REDIS_URL = (
    f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"
)


def _get_manager(session_id: str) -> SemanticSessionManager:
    """
    Create a SemanticSessionManager scoped to this session.

    Each session_id gets its own tag so messages never bleed across sessions.
    The underlying index (redisvl_session_idx) is shared — redisvl filters
    by session_tag at query time.
    """
    return SemanticSessionManager(
        name="rce_sessions",          # index name prefix in Redis
        session_tag=session_id,       # per-user isolation via tag filter
        vectorizer=_vectorizer,
        distance_threshold=0.7,       # relevance threshold for semantic retrieval
        redis_url=_REDIS_URL,
    )


def get_relevant_history(session_id: str, question: str) -> list[Message]:
    """
    Retrieve the most semantically relevant past turns for the current question.

    This is the key upgrade over last-N-turns retrieval: instead of always
    injecting the most recent messages, we find the past turns that are most
    relevant to what the user is asking right now. A question about HNSW will
    surface the HNSW conversation even if it happened 10 turns ago.

    Falls back to recent history if no semantically similar turns are found.

    Args:
        session_id: Unique identifier for the user's session.
        question: The current user question (used for semantic search).

    Returns:
        List of Message objects relevant to the question, oldest first.
    """
    manager = _get_manager(session_id)

    try:
        # Try semantic retrieval first — find turns most relevant to this question
        raw = manager.get_relevant(
            prompt=question,
            top_k=settings.MAX_HISTORY_TURNS * 2,
            as_text=False,
        )
        if raw:
            return [
                Message(role=m["role"], content=m["content"])
                for m in raw
                if "role" in m and "content" in m
            ]

        # Fall back to most recent turns if nothing is semantically similar
        raw = manager.get_recent(
            top_k=settings.MAX_HISTORY_TURNS * 2,
            as_text=False,
        )
        return [
            Message(role=m["role"], content=m["content"])
            for m in raw
            if "role" in m and "content" in m
        ]
    except Exception:
        # New session or index not yet created — return empty history
        return []


def get_history(session_id: str) -> list[Message]:
    """
    Retrieve the most recent turns (recency-based, no question context).
    Used in the cache-hit path where we don't have a question to search with.
    """
    manager = _get_manager(session_id)
    try:
        raw = manager.get_recent(
            top_k=settings.MAX_HISTORY_TURNS * 2,
            as_text=False,
        )
        return [
            Message(role=m["role"], content=m["content"])
            for m in raw
            if "role" in m and "content" in m
        ]
    except Exception:
        return []


def append_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    """
    Append a user/assistant turn to the session's vector store.

    redisvl embeds each message individually, so both the user question and
    the assistant answer are searchable by future semantic queries.

    Args:
        session_id: Unique identifier for the user's session.
        user_message: The user's question.
        assistant_message: The assistant's response.
    """
    manager = _get_manager(session_id)
    manager.add_messages([
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ])


def clear_session(session_id: str) -> None:
    """Delete all memory for a session."""
    manager = _get_manager(session_id)
    try:
        manager.clear()
    except Exception:
        pass


def get_session_info(session_id: str) -> dict:
    """Return metadata about a session (for the /session endpoint)."""
    manager = _get_manager(session_id)
    try:
        messages = manager.get_recent(as_text=False)
        if not messages:
            return {"session_id": session_id, "exists": False}
        return {
            "session_id": session_id,
            "exists": True,
            "turn_count": len(messages) // 2,
            "messages": messages,
        }
    except Exception:
        return {"session_id": session_id, "exists": False}
