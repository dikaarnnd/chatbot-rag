"""Vector search utilities for Chatbot RAG retrieval pipeline.

Impelentasi Hybrid Search (Dense + Sparse BM25) menggunakan
Reciprocal Rank Fusion (RRF), dilanjutkan dengan Cross-Encoder Reranking
untuk mencegah fenomena lost-in-the-middle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlmodel import Session, select
from rank_bm25 import BM25Okapi
 
from app.core.db import Chunk, Document, engine
from app.retrieval.query_embedder import embed_query
from app.retrieval.reranker import rerank_chunks

logger = logging.getLogger(__name__)

FINAL_TOP_K = 3
CANDIDATE_K = 15


@dataclass
class RetrievedChunk:
    text: str
    score: float
    file_name: str | None
    page_label: str | None

def _tokenize(text: str) -> list[str]:
    """Tokenisasi sederhana untuk BM25 (dapat ditingkatkan dengan Sastrawi)."""
    return text.lower().split()

def search(
    query: str,
    document_id: str,
    top_k: int = FINAL_TOP_K,
    score_threshold: float | None = None,
    embed_dim: int | None = None,
    use_hybrid: bool = True,
) -> list[RetrievedChunk]:
    """Cari chunk menggunakan Hybrid Search (RRF) -> Reranking.
 
    Args:
        query: Pertanyaan user.
        document_id: ID dokumen yang jadi konteks pencarian -- WAJIB, karena
            database sekarang bisa berisi banyak dokumen sekaligus.
        top_k: Jumlah chunk yang di-retrieve..
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

    # 1. TARIK SELURUH CHUNK DARI DOKUMEN AKTIF BESERTA JARAK VEKTORNYA
    with Session(engine) as session:
        distance_expr = Chunk.embedding.cosine_distance(query_vector)
        statement = (
            select(Chunk, Document.file_name, distance_expr.label("distance"))
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.document_id == document_id)
        )
        results = session.exec(statement).all()
 
    if not results:
        return []

    # Format data untuk pemrosesan memori
    candidates: list[dict[str, Any]] = []
    for chunk, file_name, distance in results:
        candidates.append({
            "id": chunk.id,
            "text": chunk.text,
            "page_label": chunk.page_label,
            "file_name": file_name,
            "dense_distance": float(distance),
        })

    if not use_hybrid:
        candidates.sort(key=lambda x: x["dense_distance"])
        top_dense = candidates[:top_k]
        output: list[RetrievedChunk] = []
        for c in top_dense:
            output.append(
                RetrievedChunk(
                    text=c["text"],
                    score=c["dense_distance"],
                    file_name=c["file_name"],
                    page_label=c["page_label"],
                )
            )
        return output
 
    # 2. HITUNG DENSE RANK
    # Urutkan dari distance terkecil (paling mirip)
    candidates.sort(key=lambda x: x["dense_distance"])
    dense_rank = {c["id"]: rank for rank, c in enumerate(candidates, start=1)}

    # 3. HITUNG SPARSE RANK (BM25)
    tokenized_corpus = [_tokenize(c["text"]) for c in candidates]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = _tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    for i, c in enumerate(candidates):
        c["bm25_score"] = float(bm25_scores[i])
        
    # Urutkan dari skor BM25 tertinggi
    candidates.sort(key=lambda x: x["bm25_score"], reverse=True)
    sparse_rank = {c["id"]: rank for rank, c in enumerate(candidates, start=1)}

    # 4. RECIPROCAL RANK FUSION (RRF)
    # RRF Score = 1 / (k + dense_rank) + 1 / (k + sparse_rank) -- standar k=60
    k_constant = 60
    for c in candidates:
        r_dense = dense_rank[c["id"]]
        r_sparse = sparse_rank[c["id"]]
        c["rrf_score"] = (1.0 / (k_constant + r_dense)) + (1.0 / (k_constant + r_sparse))

    # Ambil Top-CANDIDATE_K berdasarkan skor hibrida
    candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
    hybrid_top_candidates = candidates[:CANDIDATE_K]

    # 5. CROSS-ENCODER RERANKING
    # Evaluasi kandidat hibrida menggunakan cross-attention yang presisi
    final_candidates = rerank_chunks(query, hybrid_top_candidates, top_k)

    # 6. PEMBENTUKAN OUTPUT AKHIR
    output: list[RetrievedChunk] = []
    for c in final_candidates:
        # Catatan: Logit CrossEncoder bisa bernilai negatif, butuh konversi sigmoid jika ingin skor 0-1.
        output.append(
            RetrievedChunk(
                text=c["text"],
                score=c["rrf_score"],
                file_name=c["file_name"],
                page_label=c["page_label"],
            )
        )

    logger.info(
        "Hybrid+Rerank '%s' -> dari %d chunk, difilter jadi %d, Rerank Top-%d",
        query[:50], len(results), len(hybrid_top_candidates), len(output)
    )

    print("\n" + "="*50)
    print("HASIL HYBRID SEARCH + RERANKING")
    print("="*50)
    for i, c in enumerate(final_candidates, start=1):
        print(f"Peringkat Akhir [{i}]:")
        # print(f"  - Teks      : {c['text'][:60]}...")
        print(f"  - Jarak Vektor : {c['dense_distance']:.4f} (Makin kecil makin mirip)")
        print(f"  - Skor BM25 : {c['bm25_score']:.4f} (Makin besar makin cocok keyword)")
        print(f"  - Skor RRF  : {c['rrf_score']:.4f} (Skor gabungan hybrid)")
        print(f"  - Rerank Logit : {c['rerank_score']:.4f} (Skor mutlak Cross-Encoder)")
        print("-" * 50)
 
    return output