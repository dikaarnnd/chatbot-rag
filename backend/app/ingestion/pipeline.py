from __future__ import annotations

"""End-to-end ingestion pipeline orchestrator for Chatbot RAG.

Menggabungkan loader -> chunker -> embedder -> indexer jadi satu alur untuk
1 PDF. Direvisi untuk Supabase: sekarang bikin Document baru dulu (dapat
document_id), baru simpan Chunk-nya -- BUKAN reset collection global seperti
versi Qdrant lama (lihat DECISIONS.md soal riwayat chat persisten).
"""


import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_documents
from .embedder import embed_nodes
from .indexer import create_document, upsert_nodes
from .loader import PDFLoadError, load_pdf

logger = logging.getLogger(__name__)

@dataclass
class IngestionResult:
    """Ringkasan hasil satu run ingestion -- basis logging MLflow (Fase 6)."""

    document_id: str
    file_name: str
    pages_loaded: int
    chunks_created: int
    points_upserted: int
    embed_dim: int
    chunk_size: int
    chunk_overlap: int
    duration_seconds: float


def ingest_pdf(
    file_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embed_dim: int | None = None,
) -> IngestionResult:
    """Jalankan ingestion end-to-end untuk 1 PDF: load -> chunk -> embed -> simpan.

    Args:
        file_path: Path ke PDF sumber.
        chunk_size: Lihat `chunker.chunk_documents()`.
        chunk_overlap: Lihat `chunker.chunk_documents()`.
        embed_dim: Lihat `embedder.embed_nodes()`.

    Returns:
        IngestionResult berisi ringkasan metrik run ini, termasuk
        `document_id` untuk dikaitkan ke ChatSession (API layer).

    Raises:
        PDFLoadError: gagal di tahap load (lihat loader.py).
        ValueError: gagal di tahap chunk/embed/index (input kosong/invalid).
    """
    start = time.perf_counter()
    file_name = Path(file_path).name
    logger.info("=== Ingestion start: %s ===", file_name)

    try:
        documents = load_pdf(file_path)
    except PDFLoadError:
        logger.exception("Ingestion gagal di tahap load_pdf untuk '%s'", file_name)
        raise

    try:
        nodes = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except ValueError:
        logger.exception("Ingestion gagal di tahap chunk_documents untuk '%s'", file_name)
        raise

    try:
        nodes = embed_nodes(nodes, embed_dim=embed_dim)
    except ValueError:
        logger.exception("Ingestion gagal di tahap embed_nodes untuk '%s'", file_name)
        raise

    try:
        vector_size = len(nodes[0].embedding)
        document = create_document(
            file_name=file_name,
            pages_loaded=len(documents),
            chunks_created=len(nodes),
        )
        points_count = upsert_nodes(document.id, nodes)
    except Exception:
        logger.exception("Ingestion gagal di tahap penyimpanan untuk '%s'", file_name)
        raise

    duration = round(time.perf_counter() - start, 2)

    result = IngestionResult(
        document_id=document.id,
        file_name=file_name,
        pages_loaded=len(documents),
        chunks_created=len(nodes),
        points_upserted=points_count,
        embed_dim=vector_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        duration_seconds=duration,
    )

    logger.info("=== Ingestion done in %.2fs: %s ===", duration, result)
    return result


def _main() -> None:
    """CLI cepat untuk test manual: python -m app.ingestion.pipeline <path_pdf>."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Test ingestion pipeline Chatbot RAG")
    parser.add_argument("pdf_path", type=str, help="Path ke file PDF")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    result = ingest_pdf(
        args.pdf_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(result)


if __name__ == "__main__":
    _main()

# run --> python -m app.ingestion.pipeline "..\data\raw\SINTECH Journal 2023 - Main2_ID.pdf"