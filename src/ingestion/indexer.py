"""
Embeds document chunks and stores them in Redis Vector Sets.

Pain point this solves:
  Embeddings are expensive to generate. You only want to do this once — at
  ingestion time — and then query the stored vectors at runtime. This indexer
  creates a Redis vector index (HNSW) and stores each chunk as a Redis Hash
  with its embedding, text, and metadata. Query time becomes a fast ANN search
  with no embedding cost for the documents.

Redis data model:
  Each chunk is stored as a Redis Hash:
    Key:    doc:{chunk_id_hash}
    Fields:
      - text:        raw chunk text
      - url:         source page URL
      - title:       page title
      - section:     doc section label
      - chunk_index: position within the source doc
      - embedding:   binary-packed float32 vector (1536 dims)

  A vector index (redis_docs_idx) is created over the embedding field
  using HNSW for approximate nearest-neighbor search.
"""

import hashlib
import struct
import time
import redis
from sentence_transformers import SentenceTransformer
from redis.commands.search.field import TextField, VectorField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from src.ingestion.chunker import DocumentChunk
from config.settings import settings

VECTOR_DIM = 384   # all-MiniLM-L6-v2 embedding dimension
BATCH_SIZE = 64    # sentence-transformers handles larger batches efficiently

# Load model once at module level — expensive to reload on every call
_MODEL = None

def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print("  Loading embedding model (first time only)...")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _get_redis_client() -> redis.Redis:
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=False,  # binary for vector storage
    )


def _chunk_id_to_key(chunk_id: str) -> str:
    """Convert a chunk_id to a safe Redis key."""
    hash_suffix = hashlib.md5(chunk_id.encode()).hexdigest()[:12]
    return f"doc:{hash_suffix}"


def _pack_vector(vector: list[float]) -> bytes:
    """Pack a list of floats into binary format for Redis."""
    return struct.pack(f"{len(vector)}f", *vector)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts using sentence-transformers."""
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def create_vector_index(r: redis.Redis) -> None:
    """
    Create the Redis vector index for document chunks.

    Uses HNSW (Hierarchical Navigable Small World) — an approximate nearest
    neighbor algorithm that trades a small accuracy loss for dramatically faster
    search at scale. For a documentation QA system, this tradeoff is acceptable.

    FLAT index (exact search) would be more accurate but O(n) at query time.
    HNSW is O(log n) — the right choice for production use.
    """
    # If an index already exists, validate that its vector dimension matches.
    # A mismatched dim (e.g. left over from a different embedding model) silently
    # fails every hash insert under hash_indexing_failures and leaves num_docs=0.
    try:
        existing = r.ft(settings.DOCS_INDEX_NAME).info()
        existing_dim = None
        for attr in existing.get("attributes", []) or []:
            flat = {}
            for i in range(0, len(attr), 2):
                k = attr[i].decode() if isinstance(attr[i], bytes) else attr[i]
                v = attr[i + 1]
                if isinstance(v, bytes):
                    v = v.decode()
                flat[k] = v
            if flat.get("type") == "VECTOR":
                try:
                    existing_dim = int(flat.get("dim"))
                except (TypeError, ValueError):
                    existing_dim = None
                break

        if existing_dim == VECTOR_DIM:
            print(f"  Index '{settings.DOCS_INDEX_NAME}' already exists (dim={existing_dim}) — skipping creation")
            return

        print(
            f"  Index '{settings.DOCS_INDEX_NAME}' has dim={existing_dim}, "
            f"but embeddings are dim={VECTOR_DIM}. Dropping and recreating."
        )
        r.execute_command("FT.DROPINDEX", settings.DOCS_INDEX_NAME)
    except Exception:
        pass  # Index doesn't exist yet

    schema = (
        TextField("text"),
        TextField("url"),
        TextField("title"),
        TextField("section"),
        NumericField("chunk_index"),
        VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIM,
                "DISTANCE_METRIC": "COSINE",
                "M": 16,            # number of connections per layer
                "EF_CONSTRUCTION": 200,  # accuracy vs. speed at index time
            },
        ),
    )

    definition = IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
    r.ft(settings.DOCS_INDEX_NAME).create_index(schema, definition=definition)
    print(f"  Created vector index '{settings.DOCS_INDEX_NAME}' (HNSW, COSINE, {VECTOR_DIM}d)")


def index_chunks(chunks: list[DocumentChunk], force_reindex: bool = False) -> int:
    """
    Embed and store a list of document chunks in Redis.

    Args:
        chunks: List of DocumentChunk objects to index.
        force_reindex: If True, delete existing docs before indexing.

    Returns:
        Number of chunks successfully indexed.
    """
    r = _get_redis_client()

    if force_reindex:
        print("  Force reindex: flushing existing doc keys...")
        existing = r.keys("doc:*")
        if existing:
            r.delete(*existing)

    create_vector_index(r)

    total = len(chunks)
    indexed = 0
    pipeline = r.pipeline(transaction=False)

    print(f"  Indexing {total} chunks in batches of {BATCH_SIZE}...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [chunk.text for chunk in batch]

        try:
            embeddings = _embed_texts(texts)
        except Exception as e:
            print(f"  Batch {batch_start}-{batch_start + BATCH_SIZE} failed: {e}")
            continue

        for chunk, embedding in zip(batch, embeddings):
            key = _chunk_id_to_key(chunk.chunk_id)
            pipeline.hset(
                key,
                mapping={
                    "text": chunk.text.encode("utf-8"),
                    "url": chunk.url.encode("utf-8"),
                    "title": chunk.title.encode("utf-8"),
                    "section": chunk.section.encode("utf-8"),
                    "chunk_index": chunk.chunk_index,
                    "embedding": _pack_vector(embedding) if isinstance(embedding, list) else _pack_vector(list(embedding)),
                },
            )
            indexed += 1

        pipeline.execute()
        print(f"  Indexed {min(batch_start + BATCH_SIZE, total)}/{total}")
        time.sleep(0.1)  # brief pause between batches

    print(f"  Done — {indexed} chunks stored in Redis")
    return indexed
