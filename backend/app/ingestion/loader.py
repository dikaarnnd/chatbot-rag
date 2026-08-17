"""PDF loading utilities for Chatbot RAG ingestion pipeline.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Halaman dengan teks di bawah threshold ini dicurigai hasil scan/gambar.
MIN_CHARS_PER_PAGE = 20


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or has no usable extractable text."""


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF and return Document per page.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        List of Document objects, one per page, dengan metadata:
        - file_name: nama file asal
        - page_label: nomor halaman 1-indexed (string)
        - total_pages: total jumlah halaman di PDF sumber
    """
    path = Path(file_path)
    if not path.exists():
        raise PDFLoadError(f"File tidak ditemukan: {path}")
    if path.suffix.lower() != ".pdf":
        raise PDFLoadError(f"File bukan PDF: {path}")

    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        raise PDFLoadError(f"Gagal membaca PDF '{path.name}': {exc}") from exc

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise PDFLoadError(f"PDF '{path.name}' tidak punya halaman.")

    documents: list[Document] = []
    low_text_pages: list[int] = []

    for i, page in enumerate(reader.pages, start=1):
        raw_text = (page.extract_text() or "").strip()
        # pypdf mempertahankan line-wrap asli PDF sebagai '\n' literal, bahkan
        # di TENGAH kalimat (PDF tidak punya konsep paragraf, cuma posisi
        # baris visual). Kolaps semua whitespace/newline jadi 1 spasi supaya
        # tidak mengganggu sentence-boundary detection (chunker.py) dan
        # fuzzy string matching (eval harness).
        text = re.sub(r"\s+", " ", raw_text).strip()

        if len(text) < MIN_CHARS_PER_PAGE:
            low_text_pages.append(i)
            continue  # skip halaman kosong/near-empty

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "file_name": path.name,
                    "page_label": str(i),
                    "total_pages": total_pages,
                },
            )
        )

    if len(low_text_pages) > total_pages * 0.5:
        raise PDFLoadError(
            f"PDF '{path.name}' tampak berupa hasil scan/gambar "
            f"({len(low_text_pages)}/{total_pages} halaman tanpa teks). "
            "OCR belum didukung di MVP ini."
        )

    if not documents:
        raise PDFLoadError(f"Tidak ada teks yang bisa diekstrak dari '{path.name}'.")

    logger.info(
        "Loaded %d/%d halaman dari '%s' (%d halaman kosong dilewati)",
        len(documents), total_pages, path.name, len(low_text_pages),
    )

    return documents