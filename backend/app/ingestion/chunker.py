from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 2048   # ~512 token
DEFAULT_CHUNK_OVERLAP = 400 # ~100 token
_PAGE_SEPARATOR = " "

def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
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
        text_parts.append(doc.page_content)
        offset += len(marker) + len(doc.page_content) + len(_PAGE_SEPARATOR)
        text_parts.append(_PAGE_SEPARATOR)
 
    full_text = "".join(text_parts)
    combined_doc = Document(
        page_content=full_text,
        metadata={"file_name": file_name, "total_pages": total_pages},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["(?<=[.?!]) ", " ", ""],
        is_separator_regex=True
    )
    
    # Return List of Document
    chunks = splitter.split_documents([combined_doc])
    
    for chunk in chunks:
        # Menyalin metadata dasar
        chunk.metadata["file_name"] = file_name
        chunk.metadata["total_pages"] = total_pages
        # Ekstrak manual label halaman dari marker
        import re
        match = re.search(r"\[Halaman (.*?)\]", chunk.page_content)
        if match:
             chunk.metadata["page_label"] = match.group(1)
        else:
             chunk.metadata["page_label"] = "1"

    logger.info(
        "Chunked %d halaman -> %d chunks (chunk_size=%d, chunk_overlap=%d)",
        len(documents), len(chunks), chunk_size, chunk_overlap,
    )

    return chunks