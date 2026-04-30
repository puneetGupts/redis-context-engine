import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Redis Cloud
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")

    # LLM provider — anthropic | openai | groq
    # The engine's retrieval/cache/memory layers are provider-agnostic; this
    # only governs the final generation step.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")

    # Anthropic
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # OpenAI (fallback / alternate provider)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Optional base_url override — lets the OpenAI client point at any
    # OpenAI-compatible endpoint (Ollama at localhost:11434/v1, vLLM, etc.)
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")

    # Groq (free tier, OpenAI-compatible, fast Llama models)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Retrieval
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

    # Semantic cache
    CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.90"))

    # Session memory
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "1800"))

    # Context assembly
    MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "3"))
    MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "3"))

    # Redis index names
    DOCS_INDEX_NAME: str = "redis_docs_idx"
    CACHE_INDEX_NAME: str = "cache_idx"

    # Claude model for generation
    CLAUDE_MODEL: str = "claude-haiku-4-5"


settings = Settings()
