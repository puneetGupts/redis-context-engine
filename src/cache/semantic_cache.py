"""
Semantic response cache built directly on Redis Vector Sets.

Pain point this solves:
  Exact caching misses variations of the same question. "How does vector search
  work?" and "Explain Redis vector search" are the same question phrased
  differently — an exact cache returns nothing, burning tokens and adding
  latency for both. A semantic cache embeds the query and checks similarity
  against previously cached queries. If similarity exceeds 0.90, we return the
  cached answer instantly — no LLM call, no token cost, <50ms latency.

  In production documentation QA systems, a significant fraction of queries are
  variations of a small set of common questions. Semantic caching captures this.

Design decision — why 0.90 threshold (vs. 0.75 for retrieval)?
  Retrieval needs lower threshold because we want to cast a wide net for
  relevant context — partial matches are useful. Caching needs higher threshold
  because we're reusing an exact answer — a near-miss answer is worse than a
  fresh one. 0.90 = "these questions are essentially the same."

Redis data model:
  Each cached entry is a Redis Hash under cache:{hash}:
    - query:     original question text
    - answer:    Claude's response
    - embedding: binary float32 vector of the query
    - hits:      number of times this cache entry was used
    - created_at: timestamp

  A separate vector index (cache_idx) enables KNN search over cached queries.
"""

import hashlib
import json
import struct
import time
import redis
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from redis.commands.search.field import TextField, VectorField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from config.settings import settings

VECTOR_DIM = 384

_MODEL = None

def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


@dataclass
class CacheResult:
    hit: bool
    answer: str | None = None
    original_query: str | None = None
    similarity: float | None = None


def _get_redis_client() -> redis.Redis:
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=False,
    )


def _embed(text: str) -> list[float]:
    model = _get_model()
    return model.encode(text, convert_to_numpy=True).tolist()


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _cache_key(query: str) -> str:
    return f"cache:{hashlib.md5(query.encode()).hexdigest()[:12]}"


def _ensure_cache_index(r: redis.Redis) -> None:
    """Create the cache vector index if it doesn't exist."""
    try:
        r.ft(settings.CACHE_INDEX_NAME).info()
        return
    except Exception:
        pass

    schema = (
        TextField("query"),
        TextField("answer"),
        NumericField("hits"),
        VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIM,
                "DISTANCE_METRIC": "COSINE",
                "M": 16,
                "EF_CONSTRUCTION": 100,
            },
        ),
    )
    definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
    r.ft(settings.CACHE_INDEX_NAME).create_index(schema, definition=definition)


def lookup(query: str) -> CacheResult:
    """
    Check if a semantically similar query has been answered before.

    Args:
        query: The user's current question.

    Returns:
        CacheResult with hit=True and the cached answer if found,
        or hit=False if no similar query exists.
    """
    r = _get_redis_client()
    _ensure_cache_index(r)

    query_vector = _embed(query)
    query_bytes = _pack_vector(query_vector)

    redis_query = (
        Query("*=>[KNN 1 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("query", "answer", "score")
        .dialect(2)
    )

    try:
        results = r.ft(settings.CACHE_INDEX_NAME).search(
            redis_query, query_params={"vec": query_bytes}
        )
    except Exception:
        return CacheResult(hit=False)

    if not results.docs:
        return CacheResult(hit=False)

    top = results.docs[0]
    distance = float(top.score)
    similarity = 1.0 - (distance / 2.0)

    if similarity < settings.CACHE_SIMILARITY_THRESHOLD:
        return CacheResult(hit=False)

    # Increment hit counter
    r.hincrby(top.id, "hits", 1)

    answer = top.answer
    if isinstance(answer, bytes):
        answer = answer.decode("utf-8")

    original_query = top.query
    if isinstance(original_query, bytes):
        original_query = original_query.decode("utf-8")

    return CacheResult(
        hit=True,
        answer=answer,
        original_query=original_query,
        similarity=round(similarity, 4),
    )


def store(query: str, answer: str) -> None:
    """
    Cache a query/answer pair for future semantic lookup.

    Args:
        query: The user's question.
        answer: Claude's response to store.
    """
    r = _get_redis_client()
    _ensure_cache_index(r)

    embedding = _embed(query)
    key = _cache_key(query)

    r.hset(
        key,
        mapping={
            "query": query.encode("utf-8"),
            "answer": answer.encode("utf-8"),
            "embedding": _pack_vector(embedding),
            "hits": 0,
            "created_at": str(time.time()),
        },
    )
