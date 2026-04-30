"""
Per-user session memory backed by Redis Hashes.

Pain point this solves:
  LLMs are stateless — every API call starts from scratch. Without session
  memory, follow-up questions break entirely. If a user asks "How does Redis
  Vector Search work?" and follows up with "What index type should I use for
  that?", Claude has no idea what "that" refers to without the prior turn.

  This module stores conversation history in Redis under session:{session_id}
  with a TTL. Each user's memory is completely isolated. When the session
  expires (30 min of inactivity), memory is automatically cleared by Redis —
  no manual cleanup needed.

Redis data model:
  Key:    session:{session_id}
  Type:   Hash
  Fields:
    - messages: JSON-encoded list of {role, content} dicts
    - created_at: Unix timestamp
    - last_active: Unix timestamp (refreshed on every access)
  TTL:    SESSION_TTL_SECONDS (default 1800s = 30 min)
"""

import json
import time
import redis
from dataclasses import dataclass

from config.settings import settings


@dataclass
class Message:
    role: str    # "user" or "assistant"
    content: str


def _get_redis_client() -> redis.Redis:
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=True,
    )


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def get_history(session_id: str) -> list[Message]:
    """
    Retrieve conversation history for a session.

    Also refreshes the TTL — any activity resets the 30-minute expiry clock.

    Args:
        session_id: Unique identifier for the user's session.

    Returns:
        List of Message objects, oldest first. Empty list if no history.
    """
    r = _get_redis_client()
    key = _session_key(session_id)

    raw = r.hget(key, "messages")
    if not raw:
        return []

    # Refresh TTL on every read — session stays alive as long as user is active
    r.expire(key, settings.SESSION_TTL_SECONDS)
    r.hset(key, "last_active", str(time.time()))

    messages_data = json.loads(raw)
    return [Message(role=m["role"], content=m["content"]) for m in messages_data]


def append_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    """
    Append a user/assistant turn to the session history.

    Keeps only the last MAX_HISTORY_TURNS turns to stay within token budget.
    Older history is dropped — recency matters more than completeness.

    Args:
        session_id: Unique identifier for the user's session.
        user_message: The user's question.
        assistant_message: Claude's response.
    """
    r = _get_redis_client()
    key = _session_key(session_id)

    now = time.time()
    existing = get_history(session_id)

    # Append new turn
    existing.append(Message(role="user", content=user_message))
    existing.append(Message(role="assistant", content=assistant_message))

    # Trim to last MAX_HISTORY_TURNS turns (each turn = 2 messages)
    max_messages = settings.MAX_HISTORY_TURNS * 2
    if len(existing) > max_messages:
        existing = existing[-max_messages:]

    messages_json = json.dumps([{"role": m.role, "content": m.content} for m in existing])

    r.hset(
        key,
        mapping={
            "messages": messages_json,
            "last_active": str(now),
        },
    )

    # Set created_at only if it doesn't exist
    if not r.hexists(key, "created_at"):
        r.hset(key, "created_at", str(now))

    r.expire(key, settings.SESSION_TTL_SECONDS)


def clear_session(session_id: str) -> None:
    """Delete all memory for a session."""
    r = _get_redis_client()
    r.delete(_session_key(session_id))


def get_session_info(session_id: str) -> dict:
    """Return metadata about a session (for the /session endpoint)."""
    r = _get_redis_client()
    key = _session_key(session_id)

    if not r.exists(key):
        return {"session_id": session_id, "exists": False}

    data = r.hgetall(key)
    messages_data = json.loads(data.get("messages", "[]"))

    return {
        "session_id": session_id,
        "exists": True,
        "turn_count": len(messages_data) // 2,
        "created_at": data.get("created_at"),
        "last_active": data.get("last_active"),
        "ttl_seconds": r.ttl(key),
        "messages": messages_data,
    }
