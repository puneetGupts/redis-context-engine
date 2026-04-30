"""
LLM client abstraction — routes generation calls to a configured provider.

Pain point this solves:
  Tying the engine to a single LLM vendor is a single point of failure for the
  demo: a billing issue, an outage, or a model deprecation can take the whole
  system down. The retrieval, caching, and memory layers (the actual Redis-
  powered parts of this engine) are completely independent of which model
  generates the final answer. This module isolates the generation step behind
  a small interface so any provider that accepts {role, content} messages can
  be plugged in by changing one env var.

Architecture:
  Set LLM_PROVIDER in .env to one of: anthropic | openai
  The rest of the API code calls generate(system, messages, max_tokens) and
  doesn't know or care which provider answered.

Why not also Ollama / local models:
  Tested but excluded from the demo path — local 3B-class models give
  noticeably worse answers on technical Redis questions than hosted Haiku
  or gpt-4o-mini, and adding a local-model dependency increases demo risk
  (cold-start latency, model-not-loaded errors). Easy to add later if needed:
  Ollama exposes an OpenAI-compatible endpoint, so the OpenAI client below
  can talk to it by overriding base_url=http://localhost:11434/v1.
"""

from typing import Protocol

from config.settings import settings


class LLMClient(Protocol):
    """Minimal interface every provider implementation must satisfy."""

    def generate(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
    ) -> str:
        """
        Args:
            system: System prompt (provider-specific placement handled internally).
            messages: List of {"role": "user"|"assistant", "content": str}.
            max_tokens: Cap on response length.

        Returns:
            The generated answer text.
        """
        ...


class AnthropicClient:
    """Anthropic Claude client. system prompt is a top-level field."""

    def __init__(self, api_key: str, model: str):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return message.content[0].text


class OpenAIClient:
    """
    OpenAI client. system is sent as the first message with role='system'.

    Note: the messages list shape ({role, content}) is identical between
    providers, so no transformation needed beyond prepending the system msg.
    """

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self._model = model

    def generate(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
        full_messages = [{"role": "system", "content": system}] + messages
        completion = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=full_messages,
        )
        return completion.choices[0].message.content


def get_llm_client() -> LLMClient:
    """Construct the configured provider client. Called once at module import."""
    provider = settings.LLM_PROVIDER.lower().strip()

    if provider == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty")
        return AnthropicClient(
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.CLAUDE_MODEL,
        )

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty")
        return OpenAIClient(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            base_url=settings.OPENAI_BASE_URL or None,
        )

    if provider == "groq":
        # Groq is OpenAI-compatible; we reuse the same client with a different base_url.
        # Free tier is generous and Llama 3.3 70B is excellent for technical Q&A.
        if not settings.GROQ_API_KEY:
            raise RuntimeError("LLM_PROVIDER=groq but GROQ_API_KEY is empty")
        return OpenAIClient(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            base_url="https://api.groq.com/openai/v1",
        )

    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Set to 'anthropic', 'openai', or 'groq' in .env."
    )


# Module-level singleton — instantiated once when first imported.
# This is cheaper than per-request construction (avoids repeated TLS handshakes
# in the underlying http client).
_client: LLMClient | None = None


def generate(system: str, messages: list[dict], max_tokens: int = 1024) -> str:
    """Public entrypoint — lazy-initializes the configured client on first call."""
    global _client
    if _client is None:
        _client = get_llm_client()
    return _client.generate(system=system, messages=messages, max_tokens=max_tokens)
