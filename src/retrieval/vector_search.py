"""
Semantic search over indexed Redis documentation chunks.

Pain point this solves:
  Keyword search fails for natural language questions. "How do I store vectors?"
  won't match a page that talks about "embedding storage" and "ANN indexing."
  Vector search converts the question into an embedding and finds the most
  semantically similar chunks regardless of exact wording.

  The similarity threshold (default 0.75) is a critical tuning parameter:
  - Too low: irrelevant chunks pollute the context window
  - Too high: relevant chunks get filtered out
  0.75 was chosen after testing — it balances precision vs. recall for
  technical documentation queries.
"""

import struct
import redis
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
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
class RetrievedChunk:
    chunk_key: str
    text: str
    url: str
    title: str
    section: str
    similarity_score: float


def _get_redis_client() -> redis.Redis:
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=False,
    )


def _embed_query(query: str) -> list[float]:
    """Generate an embedding for a user query."""
    model = _get_model()
    return model.encode(query, convert_to_numpy=True).tolist()


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """
    Search for the most relevant document chunks for a given query.

    Process:
      1. Embed the query into a vector
      2. Run KNN search against the Redis vector index
      3. Filter results below the similarity threshold
      4. Return ranked chunks with metadata

    Args:
        query: Natural language question from the user.
        top_k: Number of results to retrieve before threshold filtering.
               Defaults to settings.TOP_K_RESULTS.

    Returns:
        List of RetrievedChunk sorted by similarity (highest first),
        filtered to >= SIMILARITY_THRESHOLD.
    """
    k = top_k or settings.TOP_K_RESULTS
    r = _get_redis_client()

    query_vector = _embed_query(query)
    query_bytes = _pack_vector(query_vector)

    # Redis vector search query using KNN
    # RETURN fields: text, url, title, section + the distance score
    # Distance metric is COSINE, so score = 1 - cosine_similarity
    # We convert: similarity = 1 - distance
    redis_query = (
        Query(f"*=>[KNN {k} @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("text", "url", "title", "section", "score")
        .dialect(2)
    )

    results = r.ft(settings.DOCS_INDEX_NAME).search(
        redis_query, query_params={"vec": query_bytes}
    )

    chunks = []
    for doc in results.docs:
        # Redis returns cosine distance (0=identical, 2=opposite)
        # Convert to similarity score (1=identical, -1=opposite), then normalize 0-1
        distance = float(doc.score)
        similarity = 1.0 - (distance / 2.0)

        if similarity < settings.SIMILARITY_THRESHOLD:
            continue  # Drop low-relevance chunks

        chunks.append(
            RetrievedChunk(
                chunk_key=doc.id,
                text=doc.text.decode("utf-8") if isinstance(doc.text, bytes) else doc.text,
                url=doc.url.decode("utf-8") if isinstance(doc.url, bytes) else doc.url,
                title=doc.title.decode("utf-8") if isinstance(doc.title, bytes) else doc.title,
                section=doc.section.decode("utf-8") if isinstance(doc.section, bytes) else doc.section,
                similarity_score=round(similarity, 4),
            )
        )

    return chunks  # Already sorted by score (ascending distance = descending similarity)
