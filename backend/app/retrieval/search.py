"""Vector search utilities for Chatbot RAG retrieval pipeline.

Top-k similarity search ke Qdrant, pakai query_points() (API modern
qdrant-client >=1.10, menggantikan search() yang deprecated).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlmodel import Session, select
 
from app.core.db import Chunk, Document, engine
from app.retrieval.query_embedder import embed_query

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 9  # hipotesis DECISIONS.md Fase 3 -- belum divalidasi


@dataclass
class RetrievedChunk:
    """Satu hasil retrieval -- chunk + skor kemiripan + metadata sitasi."""

    text: str
    score: float
    file_name: str | None
    page_label: str | None


def search(
    query: str,
    document_id: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float | None = None,
    embed_dim: int | None = None,
) -> list[RetrievedChunk]:
    """Cari top-k chunk paling relevan dengan query, DIBATASI ke 1 dokumen.
 
    Args:
        query: Pertanyaan user.
        document_id: ID dokumen yang jadi konteks pencarian -- WAJIB, karena
            database sekarang bisa berisi banyak dokumen sekaligus.
        top_k: Jumlah chunk yang di-retrieve. Default = Keputusan Final
            DECISIONS.md (k=5).
        score_threshold: Kalau di-set, buang hasil dengan similarity di
            bawah nilai ini. Default None (nonaktif).
        embed_dim: Lihat query_embedder.embed_query() -- wajib sama dengan
            embed_dim saat indexing dokumen.
 
    Returns:
        List RetrievedChunk, urut dari paling relevan (score tertinggi).
        List KOSONG kalau dokumen tidak punya chunk -- caller (generation/)
        menangani ini sebagai sinyal fallback "tidak ditemukan".
 
    Raises:
        ValueError: kalau query kosong (diteruskan dari embed_query()).
    """
    query_vector = embed_query(query, embed_dim=embed_dim)

    with Session(engine) as session:
        # cosine_distance: 0 = identik, 2 = berlawanan (vector ternormalisasi).
        # similarity = 1 - distance, supaya semantik "makin tinggi makin
        # mirip" tetap konsisten dengan versi Qdrant lama.
        distance_expr = Chunk.embedding.cosine_distance(query_vector)
 
        statement = (
            select(Chunk, Document.file_name, distance_expr.label("distance"))
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.document_id == document_id)
            .order_by(distance_expr)
            .limit(top_k)
        )
 
        results = session.exec(statement).all()
 
    chunks: list[RetrievedChunk] = []
    for chunk, file_name, distance in results:
        score = 1.0 - float(distance)
        if score_threshold is not None and score < score_threshold:
            continue
        chunks.append(
            RetrievedChunk(
                text=chunk.text,
                score=score,
                file_name=file_name,
                page_label=chunk.page_label,
            )
        )
 
    preview = query[:50] + ("..." if len(query) > 50 else "")
    logger.info(
        "Search '%s' (document_id=%s) -> %d chunk ditemukan (top_k=%d)",
        preview, document_id, len(chunks), top_k,
    )
 
    return chunks

# test -> python -c "from app.retrieval.search import search; r = search('berapa akurasi model MobileNetV2?', document_id='fc19bb36-13ec-4427-98c5-83fac81fde2d'); [print(f'{c.score:.3f} | hal.{c.page_label} | {c.text[:60]}') for c in r]"