"""End-to-end query pipeline for Chatbot RAG: retrieval -> generation.

Menggabungkan retrieval/search.py -> generation/prompt.py ->
generation/gemini_client.py jadi satu alur untuk 1 pertanyaan.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import Iterator

from app.generation.gemini_client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    GenerationError,
    generate_answer,
    generate_answer_stream,
)
from app.generation.prompt import NO_CONTEXT_MESSAGE, build_user_message
from app.retrieval.search import FINAL_TOP_K, search

logger = logging.getLogger(__name__)


@dataclass
class SourceCitation:
    """Sitasi sumber -- diambil langsung dari metadata retrieval, BUKAN
    di-parse dari teks jawaban (lihat catatan desain di DECISIONS.md)."""

    file_name: str | None
    page_label: str | None
    score: float


@dataclass
class QueryResult:
    """Hasil akhir 1 query end-to-end."""

    question: str
    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
    used_fallback: bool = False
    duration_seconds: float = 0.0


def answer_question(
    question: str,
    document_id: str,
    top_k: int = FINAL_TOP_K,
    score_threshold: float | None = None,
    model: str = DEFAULT_MODEL,
    embed_dim: int | None = None,
) -> QueryResult:
    """Jawab 1 pertanyaan end-to-end: retrieval -> prompt assembly -> generation.

    Args:
        question: Pertanyaan user.
        document_id: ID dokumen yang jadi konteks -- lihat retrieval/search.py.
        top_k: Lihat retrieval/search.py.
        score_threshold: Lihat retrieval/search.py.
        model: Model Gemini yang dipakai, lihat generation/gemini_client.py.
        embed_dim: Wajib sama dengan embed_dim saat indexing dokumen.

    Returns:
        QueryResult -- jawaban + sitasi sumber (sources kosong kalau fallback).

    Raises:
        ValueError: kalau question kosong (diteruskan dari embed_query()).
        GenerationError: kalau panggilan Gemini API gagal.
    """
    start = time.perf_counter()
    preview = question[:80]
    logger.info("=== Query start: '%s' (document_id=%s) ===", preview, document_id)

    try:
        chunks = search(
            question,
            document_id=document_id,
            top_k=top_k,
            score_threshold=score_threshold,
            embed_dim=embed_dim,
        )
    except ValueError:
        logger.exception("Query gagal di tahap search untuk: '%s'", preview)
        raise

    if not chunks:
        duration = round(time.perf_counter() - start, 2)
        logger.info("Tidak ada chunk relevan -- fallback tanpa panggil Gemini API.")
        return QueryResult(
            question=question,
            answer=NO_CONTEXT_MESSAGE,
            sources=[],
            used_fallback=True,
            duration_seconds=duration,
        )

    try:
        user_message = build_user_message(question, chunks)
    except ValueError:
        logger.exception("Query gagal di tahap prompt assembly untuk: '%s'", preview)
        raise

    try:
        answer = generate_answer(user_message, model=model)
    except GenerationError:
        logger.exception("Query gagal di tahap generation untuk: '%s'", preview)
        raise

    sources = [
        SourceCitation(file_name=c.file_name, page_label=c.page_label, score=c.score)
        for c in chunks
    ]

    duration = round(time.perf_counter() - start, 2)

    result = QueryResult(
        question=question,
        answer=answer,
        sources=sources,
        used_fallback=False,
        duration_seconds=duration,
    )

    logger.info(
        "=== Query done in %.2fs: %d sumber, fallback=%s ===",
        duration, len(sources), result.used_fallback,
    )

    return result


def answer_question_stream(
    question: str,
    document_id: str,
    history: list[dict] | None = None,
    top_k: int = FINAL_TOP_K,
    score_threshold: float | None = None,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    embed_dim: int | None = None,
) -> Iterator[dict]:
    """Versi streaming dari answer_question() -- yield event dict bertahap.

    Retrieval dilakukan dulu secara penuh (cepat, tidak perlu di-stream), baru
    generation di-stream token demi token. Sources dikirim SEBELUM teks
    jawaban mulai, supaya frontend bisa render sitasi sambil jawaban "mengetik".

    Args:
        question: Pertanyaan user.
        document_id: ID dokumen yang jadi konteks -- lihat retrieval/search.py.
        (parameter lain sama seperti answer_question())

    Yields:
        dict dengan salah satu bentuk:
        - {"type": "sources", "sources": [{"file_name", "page_label", "score"}]}
          -- sekali di awal
        - {"type": "delta", "text": "..."} -- berkali-kali, potongan jawaban
        - {"type": "done", "used_fallback": bool, "duration_seconds": float}
          -- sekali di akhir

    Raises:
        ValueError: kalau question kosong (diteruskan dari embed_query()).
        GenerationError: kalau panggilan Gemini API gagal -- BISA terjadi di
            TENGAH streaming (setelah event "sources" terkirim). Caller (FastAPI
            endpoint) wajib tangkap ini dan kirim sebagai event "error" dalam
            stream, BUKAN ubah HTTP status code (header sudah terkirim duluan).
    """
    start = time.perf_counter()
    preview = question[:80]
    logger.info("=== Query (stream) start: '%s' (document_id=%s) ===", preview, document_id)

    yield {"type": "stage", "stage": "retrieval", "status": "start"}
    retrieval_start = time.perf_counter()
    chunks = search(
        question,
        document_id=document_id,
        top_k=top_k,
        score_threshold=score_threshold,
        embed_dim=embed_dim,
    )
    retrieval_duration = round(time.perf_counter() - retrieval_start, 3)
    yield {
        "type": "stage",
        "stage": "retrieval",
        "status": "done",
        "duration_seconds": retrieval_duration,
        "chunks_found": len(chunks),
    }

    if not chunks:
        duration = round(time.perf_counter() - start, 2)
        logger.info("Tidak ada chunk relevan -- fallback tanpa panggil Gemini API.")
        yield {"type": "sources", "sources": []}
        yield {"type": "delta", "text": NO_CONTEXT_MESSAGE}
        yield {"type": "done", "used_fallback": True, "duration_seconds": duration}
        return

    user_message = build_user_message(question, chunks, history=history)

    sources = [
        {"file_name": c.file_name, "page_label": c.page_label, "score": c.score}
        for c in chunks
    ]
    yield {"type": "sources", "sources": sources}

    yield {"type": "stage", "stage": "generation", "status": "start"}
    generation_start = time.perf_counter()
    
    for text_chunk in generate_answer_stream(
        user_message, model=model, max_output_tokens=max_output_tokens
    ):
        yield {"type": "delta", "text": text_chunk}

    generation_duration = round(time.perf_counter() - generation_start, 3)
    yield {
        "type": "stage",
        "stage": "generation",
        "status": "done",
        "duration_seconds": generation_duration,
    }
    
    duration = round(time.perf_counter() - start, 2)
    yield {"type": "done", "used_fallback": False, "duration_seconds": duration}
    logger.info("=== Query (stream) done in %.2fs ===", duration)


def _main() -> None:
    """CLI cepat: python -m app.generation.pipeline <document_id> "pertanyaan"."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Test query pipeline Chatbot RAG")
    parser.add_argument("document_id", type=str, help="ID dokumen (dari tabel Document)")
    parser.add_argument("question", type=str, help="Pertanyaan untuk diajukan")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    args = parser.parse_args()

    result = answer_question(args.question, document_id=args.document_id, top_k=args.top_k)

    print(f"\nPertanyaan : {result.question}")
    print(f"Jawaban    : {result.answer}")
    print(f"Fallback   : {result.used_fallback}")
    print(f"Durasi     : {result.duration_seconds}s")
    if result.sources:
        print("Sumber:")
        for s in result.sources:
            print(f"  - {s.file_name} (hal. {s.page_label}, score={s.score:.3f})")


if __name__ == "__main__":
    _main()