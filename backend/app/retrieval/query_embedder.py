"""Query embedding utilities for Chatbot RAG retrieval pipeline.

Counterpart to ingestion/embedder.py, tapi untuk QUERY bukan dokumen.
Qwen3-Embedding-0.6B instruction-tuned -- retrieval quality terbaik kalau
query di-encode dengan instruction prefix bawaan model (prompt_name="query"),
sedangkan dokumen/chunk (lihat ingestion/embedder.py) TIDAK pakai instruction.
Model sudah punya prompt template "query" built-in di config HuggingFace-nya
-- tidak perlu menulis instruction text manual.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from app.ingestion.embedder import MODEL_NAME, NATIVE_EMBED_DIM

logger = logging.getLogger(__name__)

_QUERY_PROMPT_NAME = "query"

# Cache module-level supaya model tidak di-load ulang tiap panggilan (mahal).
# FastAPI endpoint (Fase 4) sebaiknya trigger get_query_embedder() sekali
# saat startup, bukan per-request.
_embedder_instance: SentenceTransformer | None = None


def get_query_embedder(embed_dim: int | None = None) -> SentenceTransformer:
    """Instantiate (atau reuse cached) Qwen3-Embedding-0.6B untuk query.

    Args:
        embed_dim: Truncate dimension (Matryoshka). HARUS SAMA dengan yang
            dipakai saat indexing dokumen (ingestion/embedder.py) -- kalau
            beda, vector space tidak cocok dan similarity search jadi invalid.

    Returns:
        SentenceTransformer instance (cached setelah panggilan pertama).
    """
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance

    kwargs: dict = {"device": "cpu", "trust_remote_code": True}
    if embed_dim is not None:
        kwargs["truncate_dim"] = embed_dim

    logger.info(
        "Loading query embedding model '%s' (device=cpu, embed_dim=%s)",
        MODEL_NAME, embed_dim or f"native({NATIVE_EMBED_DIM})",
    )
    _embedder_instance = SentenceTransformer(MODEL_NAME, **kwargs)
    return _embedder_instance


def embed_query(query: str, embed_dim: int | None = None) -> list[float]:
    """Embed satu query pakai instruction prefix 'query' (asimetri vs dokumen).

    Args:
        query: Teks pertanyaan user.
        embed_dim: Lihat `get_query_embedder()` -- wajib sama dengan embed_dim
            yang dipakai saat ingest dokumen ke Qdrant.

    Returns:
        Vector embedding (list[float]), sudah dinormalisasi (cocok Cosine
        distance di Qdrant).

    Raises:
        ValueError: kalau query kosong/cuma whitespace.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query kosong -- tidak bisa di-embed.")

    model = get_query_embedder(embed_dim=embed_dim)

    vector = model.encode(
        query,
        prompt_name=_QUERY_PROMPT_NAME,
        normalize_embeddings=True,  # match embedder.py -- Qdrant Cosine distance
        convert_to_numpy=True,
    )

    preview = query[:60] + ("..." if len(query) > 60 else "")
    logger.info("Query embedded (dim=%d): '%s'", len(vector), preview)

    return vector.tolist()