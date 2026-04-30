"""
Assembles the final context window sent to Claude.

Pain point this solves:
  You have retrieved chunks and session history — but you cannot blindly dump
  all of it into the prompt. Claude has a context window limit, and more
  content doesn't mean better answers. Irrelevant chunks add noise. Stale
  history wastes tokens. This module enforces a token budget: top N chunks
  by relevance, most recent M turns of history, and a structured prompt that
  tells Claude exactly what it's working with and how to use it.

  The structured format (System → Context → History → Question) follows
  the same pattern used in production RAG systems — it gives Claude clear
  signal about which parts are retrieved knowledge vs. conversation history
  vs. the user's actual question.
"""

import tiktoken
from dataclasses import dataclass

from src.retrieval.vector_search import RetrievedChunk
from src.memory.session import Message
from config.settings import settings

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

SYSTEM_PROMPT = """You are a Redis documentation assistant. You answer questions about Redis
using only the provided documentation context.

Rules:
- Base your answer on the provided context. If the context doesn't contain enough
  information to answer confidently, say so clearly.
- Always cite the source URL when referencing specific documentation.
- Be precise and technical — your audience is software engineers.
- If the user's question refers to something from the conversation history
  (e.g., "what about that?", "how does it scale?"), use the history to resolve
  the reference before answering."""


@dataclass
class AssembledContext:
    prompt_messages: list[dict]   # formatted for Anthropic messages API
    chunks_used: int
    history_turns_used: int
    total_tokens_estimate: int
    sources: list[str]


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def assemble(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    history: list[Message],
) -> AssembledContext:
    """
    Assemble a focused, token-bounded context window for Claude.

    Strategy:
      1. Take top MAX_CONTEXT_CHUNKS chunks (already sorted by relevance).
      2. Take last MAX_HISTORY_TURNS turns of history (recency > completeness).
      3. Build the prompt in structured sections.
      4. Estimate token count and warn if approaching limits.

    Args:
        question: Current user question.
        retrieved_chunks: Ranked chunks from vector search.
        history: Full session history (we'll trim to last N turns).

    Returns:
        AssembledContext ready to pass to the Claude API.
    """
    # Select top N chunks
    selected_chunks = retrieved_chunks[: settings.MAX_CONTEXT_CHUNKS]

    # Select last N turns of history (each turn = 2 messages: user + assistant)
    max_history_messages = settings.MAX_HISTORY_TURNS * 2
    recent_history = history[-max_history_messages:] if history else []

    # Build context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(selected_chunks, 1):
        context_parts.append(
            f"[Source {i}] {chunk.title} ({chunk.section})\n"
            f"URL: {chunk.url}\n"
            f"Relevance: {chunk.similarity_score:.2f}\n\n"
            f"{chunk.text}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    if not context_block:
        context_block = "No relevant documentation found for this query."

    # Build the user turn: context + question
    user_content = f"""<context>
{context_block}
</context>

<question>
{question}
</question>"""

    # Build messages array for Anthropic API
    # Format: [history messages...] + [current user message with context]
    messages = []

    for msg in recent_history:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": user_content})

    # Estimate token usage
    all_text = SYSTEM_PROMPT + " ".join(
        m["content"] for m in messages
    )
    token_estimate = _count_tokens(all_text)

    sources = list(dict.fromkeys(chunk.url for chunk in selected_chunks))

    return AssembledContext(
        prompt_messages=messages,
        chunks_used=len(selected_chunks),
        history_turns_used=len(recent_history) // 2,
        total_tokens_estimate=token_estimate,
        sources=sources,
    )
