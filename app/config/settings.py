"""Central configuration, loaded from environment variables / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # LLM backend: "transformers" (default, zero extra setup) or "ollama"
    # (better quality, requires Ollama installed + running locally).
    LLM_BACKEND = os.getenv("LLM_BACKEND", "transformers").strip().lower()
    _default_model = "llama3.2" if LLM_BACKEND == "ollama" else "google/flan-t5-base"
    LLM_MODEL = os.getenv("LLM_MODEL", _default_model)
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "220"))       # words per chunk
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))  # overlapping words between chunks

    # Retrieval
    TOP_K = int(os.getenv("TOP_K", "5"))

    # Storage
    UPLOAD_DIR = BASE_DIR / "uploads"
    INDEX_CACHE_DIR = UPLOAD_DIR / ".index_cache"
    LOG_DIR = BASE_DIR / "logs"

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
