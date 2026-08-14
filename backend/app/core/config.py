"""Centralized configuration for Chatbot RAG backend.

Pakai pydantic-settings untuk validasi tipe otomatis + baca dari .env.
Modul lain (ingestion/, retrieval/, generation/) TETAP punya default
masing-masing untuk pemakaian standalone/CLI -- config ini dipakai khusus
oleh layer FastAPI (main.py) untuk override default tersebut secara
terpusat saat deployment/serving API.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- API Keys ---
    gemini_api_key: str 

    # --- Embedding ---
    embed_dim: int | None = None  # None = native 1024 (Qwen3-Embedding-0.6B)

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 100

    # --- Retrieval ---
    top_k: int = 8
    score_threshold: float | None = None

    # --- Generation ---
    gemini_model: str = "gemini-3.6-flash"
    max_output_tokens: int = 4096
    history_turns: int = 5

    # --- App ---
    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    """Cached singleton -- .env cuma dibaca sekali per proses, bukan tiap request.

    Dipakai sebagai FastAPI dependency: `settings: Settings = Depends(get_settings)`.
    """
    return Settings()