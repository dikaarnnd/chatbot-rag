from __future__ import annotations

"""Chunk + vector storage for Chatbot RAG ingestion pipeline (Supabase/pgvector).
"""


import logging

from llama_index.core.schema import BaseNode
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


def upsert_nodes(document_id: str, nodes: list[BaseNode]) -> int:
    """Simpan nodes (chunk + embedding + metadata) sebagai baris Chunk.

    Args:
        document_id: ID Document yang barusan dibuat (dari create_document()).
        nodes: Nodes hasil embed_nodes() -- wajib sudah punya `.embedding`.

    Returns:
        Jumlah chunk yang berhasil disimpan.

    Raises:
        ValueError: kalau nodes kosong, atau ada node tanpa embedding.
    """
    if not nodes:
        raise ValueError("nodes kosong -- tidak ada yang bisa disimpan.")

    chunks: list[Chunk] = []
    for node in nodes:
        if node.embedding is None:
            raise ValueError(
                f"Node {node.node_id} belum punya embedding -- jalankan "
                "embed_nodes() dulu sebelum upsert_nodes()."
            )
        chunks.append(
            Chunk(
                document_id=document_id,
                text=node.get_content(),
                page_label=node.metadata.get("page_label"),
                embedding=node.embedding,
            )
        )

    with Session(engine) as session:
        session.add_all(chunks)
        session.commit()

    logger.info("Upserted %d chunk untuk document_id=%s", len(chunks), document_id)
    return len(chunks)