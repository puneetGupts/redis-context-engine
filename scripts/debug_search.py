"""
Diagnostic: figure out why retrieval returns zero matches.

Checks (in order):
  1. How many doc:* keys exist in Redis?
  2. Does the index 'redis_docs_idx' exist and how many docs are in it?
  3. Run a raw KNN query and print actual distance scores (no threshold filter).
  4. Show what similarity_score the retriever would compute for the top-5.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import struct
import redis
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query

from config.settings import settings

QUERY = "How does Redis Vector Search work?"
TOP_K = 5

r = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    decode_responses=False,
)

print(f"Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
print(f"PING: {r.ping()}")
print()

# 1. How many doc keys?
doc_keys = r.keys("doc:*")
print(f"doc:* keys in Redis:  {len(doc_keys)}")
if doc_keys:
    sample_key = doc_keys[0]
    fields = r.hkeys(sample_key)
    print(f"  Sample key:        {sample_key.decode()}")
    print(f"  Sample fields:     {[f.decode() for f in fields]}")
    title = r.hget(sample_key, "title")
    url = r.hget(sample_key, "url")
    text = r.hget(sample_key, "text")
    print(f"  Sample title:      {title.decode()[:80] if title else '(none)'}")
    print(f"  Sample url:        {url.decode() if url else '(none)'}")
    print(f"  Sample text[:120]: {text.decode()[:120] if text else '(none)'}")
print()

# 2. Index info
try:
    info = r.ft(settings.DOCS_INDEX_NAME).info()
    # info is a list of [key, value, key, value, ...]
    info_dict = {}
    for i in range(0, len(info), 2):
        k = info[i].decode() if isinstance(info[i], bytes) else info[i]
        v = info[i + 1]
        if isinstance(v, bytes):
            v = v.decode()
        info_dict[k] = v
    print(f"Index '{settings.DOCS_INDEX_NAME}' exists")
    print(f"  num_docs:          {info_dict.get('num_docs')}")
    print(f"  num_records:       {info_dict.get('num_records')}")
    print(f"  hash_indexing_failures: {info_dict.get('hash_indexing_failures')}")
except Exception as e:
    print(f"Index '{settings.DOCS_INDEX_NAME}' error: {e}")
    sys.exit(1)
print()

# 3. Raw KNN query — no threshold filter
print(f"Query: {QUERY!r}")
model = SentenceTransformer("all-MiniLM-L6-v2")
qvec = model.encode(QUERY, convert_to_numpy=True).tolist()
qbytes = struct.pack(f"{len(qvec)}f", *qvec)

redis_query = (
    Query(f"*=>[KNN {TOP_K} @embedding $vec AS score]")
    .sort_by("score")
    .return_fields("text", "url", "title", "score")
    .dialect(2)
)

results = r.ft(settings.DOCS_INDEX_NAME).search(
    redis_query, query_params={"vec": qbytes}
)

print(f"Raw results count: {len(results.docs)}")
print()
print(f"{'rank':<5} {'distance':<12} {'sim_formula':<12} {'title':<60}")
print("-" * 100)
for i, doc in enumerate(results.docs):
    distance = float(doc.score)
    similarity = 1.0 - (distance / 2.0)
    title = doc.title.decode("utf-8") if isinstance(doc.title, bytes) else doc.title
    pass_threshold = "PASS" if similarity >= settings.SIMILARITY_THRESHOLD else "FAIL"
    print(f"{i:<5} {distance:<12.6f} {similarity:<12.6f} {title[:60]:<60} [{pass_threshold} @ {settings.SIMILARITY_THRESHOLD}]")
print()
print(f"SIMILARITY_THRESHOLD = {settings.SIMILARITY_THRESHOLD}")
