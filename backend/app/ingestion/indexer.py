from __future__ import annotations

"""Chunk + vector storage for Chatbot RAG ingestion pipeline (Supabase/pgvector).
"""

import logging
from sqlmodel import Session

from app.core.db import Chunk, Document, engine

logger = logging.getLogger(__name__)


def create_document(file_name: str, pages_loaded: int, chunks_created: int) -> Document:
    document = Document(
        file_name=file_name,
        pages_loaded=pages_loaded,
        chunks_created=chunks_created,
    )
    with Session(engine) as session:
        session.add(document)
        session.commit()
        session.refresh(document)

    logger.info("Document baru dibuat: id=%s, file_name='%s'", document.id, file_name)
    return document


def upsert_nodes(document_id: str, embedded_chunks: list[dict]) -> int:
    """Simpan nodes (chunk + embedding + metadata) sebagai baris Chunk.

    Args:
        document_id: ID Document yang barusan dibuat (dari create_document()).
        nodes: Nodes hasil embed_nodes() -- wajib sudah punya `.embedding`.

    Returns:
        Jumlah chunk yang berhasil disimpan.

    Raises:
        ValueError: kalau nodes kosong, atau ada node tanpa embedding.
    """
    """Simpan chunk + embedding ke database."""
    if not embedded_chunks:
        raise ValueError("nodes kosong -- tidak ada yang bisa disimpan.")

    chunks: list[Chunk] = []
    for item in embedded_chunks:
        chunks.append(
            Chunk(
                document_id=document_id,
                text=item["text"],
                page_label=item["metadata"].get("page_label", "1"),
                embedding=item["embedding"],
            )
        )

    with Session(engine) as session:
        session.add_all(chunks)
        session.commit()

    logger.info("Upserted %d chunk untuk document_id=%s", len(chunks), document_id)
    return len(chunks)