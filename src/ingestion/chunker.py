"""
Splits scraped documents into overlapping chunks for indexing.

Pain point this solves:
  You cannot embed an entire documentation page as one vector — it's too long
  and the embedding loses specificity. But splitting naively at fixed character
  counts cuts sentences mid-thought. This chunker splits by tokens (not chars),
  uses tiktoken to count accurately, and adds overlap between chunks so ideas
  at chunk boundaries are preserved in both neighboring chunks.

Design decisions:
  - 500 token chunks: large enough to carry context, small enough to be specific.
  - 50 token overlap: prevents ideas from being cut off at boundaries.
  - Paragraph-first splitting: tries to split at natural paragraph breaks before
    falling back to sentence splits, preserving semantic coherence.
"""

import re
import tiktoken
from dataclasses import dataclass
from typing import Generator

from src.ingestion.scraper import ScrapedDocument
from config.settings import settings

# Tokenizer for Claude / GPT-compatible models
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Cap chunks per document. Redis docs pages render every command example in 8+
# programming-language tabs (Python, Ruby, JS, Go, C#, etc.), so a single page
# can balloon to 500+ chunks. The first N chunks always carry the conceptual
# overview and the most-used languages; capping here keeps the index focused
# and prevents OOM on free-tier Redis Cloud.
MAX_CHUNKS_PER_DOC = 30


@dataclass
class DocumentChunk:
    chunk_id: str        # "{url}::{index}"
    url: str
    title: str
    section: str
    text: str
    token_count: int
    chunk_index: int


def _token_count(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text at double newlines (paragraph breaks)."""
    paragraphs = re.split(r"\n{2,}", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_paragraph_into_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences as a fallback for long paragraphs."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(doc: ScrapedDocument) -> Generator[DocumentChunk, None, None]:
    """
    Chunk a scraped document into overlapping token-bounded segments.

    Strategy:
      1. Split into paragraphs first (natural semantic boundaries).
      2. Accumulate paragraphs into a chunk until we approach the token limit.
      3. When the chunk is full, yield it and start the next chunk with
         the last `overlap` tokens of the current chunk (the overlap window).
      4. If a single paragraph exceeds the chunk size, split it at sentences.

    Args:
        doc: A ScrapedDocument to chunk.

    Yields:
        DocumentChunk for each chunk produced.
    """
    chunk_size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP

    paragraphs = _split_into_paragraphs(doc.content)

    # Expand any paragraph that's too long into sentences
    units: list[str] = []
    for para in paragraphs:
        if _token_count(para) > chunk_size:
            units.extend(_split_paragraph_into_sentences(para))
        else:
            units.append(para)

    current_tokens: list[str] = []  # token-level buffer (encoded)
    current_text_parts: list[str] = []
    chunk_index = 0

    for unit in units:
        if chunk_index >= MAX_CHUNKS_PER_DOC:
            return
        unit_tokens = _TOKENIZER.encode(unit)

        # If this single unit is larger than chunk_size, force-split it
        if len(unit_tokens) > chunk_size:
            # Flush current buffer first
            if current_text_parts:
                text = "\n\n".join(current_text_parts)
                yield DocumentChunk(
                    chunk_id=f"{doc.url}::{chunk_index}",
                    url=doc.url,
                    title=doc.title,
                    section=doc.section,
                    text=text,
                    token_count=len(current_tokens),
                    chunk_index=chunk_index,
                )
                chunk_index += 1
                current_tokens = []
                current_text_parts = []
                if chunk_index >= MAX_CHUNKS_PER_DOC:
                    return

            # Force-split the oversized unit by tokens
            for start in range(0, len(unit_tokens), chunk_size - overlap):
                if chunk_index >= MAX_CHUNKS_PER_DOC:
                    return
                segment_tokens = unit_tokens[start: start + chunk_size]
                segment_text = _TOKENIZER.decode(segment_tokens)
                yield DocumentChunk(
                    chunk_id=f"{doc.url}::{chunk_index}",
                    url=doc.url,
                    title=doc.title,
                    section=doc.section,
                    text=segment_text,
                    token_count=len(segment_tokens),
                    chunk_index=chunk_index,
                )
                chunk_index += 1
            continue

        # Would adding this unit exceed the chunk size?
        if len(current_tokens) + len(unit_tokens) > chunk_size and current_text_parts:
            # Yield current chunk
            text = "\n\n".join(current_text_parts)
            yield DocumentChunk(
                chunk_id=f"{doc.url}::{chunk_index}",
                url=doc.url,
                title=doc.title,
                section=doc.section,
                text=text,
                token_count=len(current_tokens),
                chunk_index=chunk_index,
            )
            chunk_index += 1
            if chunk_index >= MAX_CHUNKS_PER_DOC:
                return

            # Start next chunk with overlap: keep the tail of the current token buffer
            overlap_tokens = current_tokens[-overlap:] if len(current_tokens) > overlap else current_tokens
            overlap_text = _TOKENIZER.decode(overlap_tokens)
            current_tokens = list(overlap_tokens)
            current_text_parts = [overlap_text]

        current_tokens.extend(unit_tokens)
        current_text_parts.append(unit)

    # Yield any remaining content
    if current_text_parts and chunk_index < MAX_CHUNKS_PER_DOC:
        text = "\n\n".join(current_text_parts)
        yield DocumentChunk(
            chunk_id=f"{doc.url}::{chunk_index}",
            url=doc.url,
            title=doc.title,
            section=doc.section,
            text=text,
            token_count=len(current_tokens),
            chunk_index=chunk_index,
        )


def chunk_documents(docs: list[ScrapedDocument]) -> Generator[DocumentChunk, None, None]:
    """Chunk a list of documents."""
    for doc in docs:
        yield from chunk_document(doc)
