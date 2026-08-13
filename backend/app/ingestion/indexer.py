from __future__ import annotations

"""Qdrant indexing utilities for Chatbot RAG ingestion pipeline.

Uses qdrant-client directly (bukan llama-index-vector-stores-qdrant wrapper)
-- konsisten dengan pola project ini: embedding sudah dikelola manual
(embedder.py), jadi reuse LlamaIndex vector store wrapper cuma nambah lapisan
re-wrapping tanpa manfaat. Qdrant collection dikonfigurasi Cosine distance,
match dengan normalize_embeddings=True di embedder.py.

MVP scope = 1 PDF per sesi (lihat PRD.md) -> setiap index run me-reset
collection dari awal, bukan menambah/accumulate ke collection lama. Ini
mencegah jawaban ter-grounded ke dokumen sesi sebelumnya yang sudah tidak
relevan (out-of-scope: multi-dokumen corpus-level QA).
"""

"""Chunk + vector storage for Chatbot RAG ingestion pipeline (Supabase/pgvector).

Menggantikan indexer.py versi Qdrant. Beda filosofi mendasar: TIDAK reset
collection tiap upload (versi Qdrant lama begitu, sesuai MVP "1 dokumen aktif
per sesi"). Sekarang SEMUA dokumen yang pernah di-upload tetap tersimpan
permanen -- retrieval difilter per document_id (lihat retrieval/search.py
revisi berikutnya). Ini yang memungkinkan riwayat chat lama tetap bisa
dibuka & ditanya ulang setelah restart.
"""


import logging

from llama_index.core.schema import BaseNode
from sqlmodel import Session

from app.core.db import Chunk, Document, engine

logger = logging.getLogger(__name__)


def create_document(file_name: str, pages_loaded: int, chunks_created: int) -> Document:
    """Buat 1 baris Document baru -- dipanggil sekali per upload PDF.

    Args:
        file_name: Nama file PDF asli.
        pages_loaded: Jumlah halaman berhasil di-load (dari loader.py).
        chunks_created: Jumlah chunk hasil splitting (dari chunker.py).

    Returns:
        Document yang baru dibuat (sudah punya `.id` ter-generate) --
        `.id` ini yang dipakai untuk mengaitkan ChatSession nanti.
    """
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