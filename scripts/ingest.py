"""
One-time ingestion script — scrape Redis docs, chunk, embed, and index into Redis.

Run this once before starting the API:
  python scripts/ingest.py

Options:
  --max-pages N     Number of doc pages to scrape (default: 50)
  --force           Re-index even if docs already exist
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.scraper import scrape_docs
from src.ingestion.chunker import chunk_document
from src.ingestion.indexer import index_chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest Redis docs into Redis Vector Sets")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages to scrape")
    parser.add_argument("--force", action="store_true", help="Force re-index")
    args = parser.parse_args()

    print("=" * 60)
    print("Redis Context Engine — Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Scrape
    print("\n[1/3] Scraping Redis documentation...")
    docs = list(scrape_docs(max_pages=args.max_pages))
    print(f"  Collected {len(docs)} pages")

    if not docs:
        print("  No docs scraped — check your internet connection")
        sys.exit(1)

    # Step 2: Chunk
    print(f"\n[2/3] Chunking documents (500 tokens, 50 overlap)...")
    all_chunks = []
    for doc in docs:
        chunks = list(chunk_document(doc))
        all_chunks.extend(chunks)
    print(f"  Produced {len(all_chunks)} chunks from {len(docs)} pages")

    # Step 3: Index
    print(f"\n[3/3] Embedding and indexing into Redis Vector Sets...")
    count = index_chunks(all_chunks, force_reindex=args.force)

    print("\n" + "=" * 60)
    print(f"Ingestion complete — {count} chunks indexed")
    print("You can now start the API: uvicorn src.api.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
