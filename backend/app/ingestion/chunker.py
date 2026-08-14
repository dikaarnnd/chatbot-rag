from __future__ import annotations

import logging

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100
_PAGE_SEPARATOR = " "

def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[BaseNode]:
    """Split per-page Documents into sentence-aware chunks (nodes).

    Semua halaman digabung dulu jadi 1 teks kontinu sebelum di-chunk, supaya
    chunk_overlap benar-benar menjembatani apa yang dulunya batas halaman.
    page_label tiap chunk ditentukan lewat pemetaan posisi karakter awal
    chunk terhadap peta offset->halaman

    Args:
        documents: Output dari `load_pdf()` -- satu Document per halaman.
        chunk_size: Target ukuran chunk dalam token.
        chunk_overlap: Overlap antar chunk dalam token.

    Returns:
        List node hasil chunking. Tiap node membawa metadata dari Document
        asalnya (file_name, page_label, total_pages).

    Raises:
        ValueError: kalau `documents` kosong.
    """
    if not documents:
        raise ValueError("documents kosong -- tidak ada yang bisa di-chunk.")

    file_name = documents[0].metadata.get("file_name")
    total_pages = documents[0].metadata.get("total_pages")
 
    # Gabungkan semua halaman jadi 1 teks, simpan peta (offset_awal, page_label)
    # supaya nanti bisa dipetakan balik per chunk.
    text_parts: list[str] = []
    page_boundaries: list[tuple[int, str]] = [] 
    offset = 0
 
    for doc in documents:
        page_label = doc.metadata.get("page_label", "?")
        marker = f"\n[Halaman {page_label}] "
        page_boundaries.append((offset, page_label))
        text_parts.append(marker)
        text_parts.append(doc.text)
        offset += len(marker) + len(doc.text) + len(_PAGE_SEPARATOR)
        text_parts.append(_PAGE_SEPARATOR)
 
    full_text = "".join(text_parts)
    combined_doc = Document(
        text=full_text,
        metadata={"file_name": file_name, "total_pages": total_pages},
    )

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([combined_doc], show_progress=False)

    # Petakan tiap node balik ke page_label berdasarkan posisi karakter awalnya.
    for node in nodes:
        start = node.start_char_idx if node.start_char_idx is not None else 0
        page_label = page_boundaries[0][1]
        for boundary_offset, label in page_boundaries:
            if start >= boundary_offset:
                page_label = label
            else:
                break
        node.metadata["page_label"] = page_label
        node.metadata["file_name"] = file_name
        node.metadata["total_pages"] = total_pages

    logger.info(
        "Chunked %d halaman -> %d chunks (chunk_size=%d, chunk_overlap=%d)",
        len(documents), len(nodes), chunk_size, chunk_overlap,
    )

    return nodes