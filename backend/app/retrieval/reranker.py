"""Cross-Encoder Reranking utilities for Chatbot RAG.

Menggunakan model BAAI/bge-reranker-v2-m3 untuk membandingkan secara
langsung (cross-attention) antara kueri dan kandidat dokumen.
"""

from __future__ import annotations

import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_reranker_instance: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Instantiate (atau reuse cached) model CrossEncoder.
    
    Model ini memproses [Query + Chunk] sekaligus, berbeda dengan
    Bi-Encoder yang memisahkannya.
    """
    global _reranker_instance
    if _reranker_instance is not None:
        return _reranker_instance

    logger.info("Loading Cross-Encoder model '%s' (device=cpu)", RERANKER_MODEL_NAME)
    # Gunakan CPU untuk keamanan memori. Jika VRAM memadai, ubah ke 'cuda'
    _reranker_instance = CrossEncoder(RERANKER_MODEL_NAME, device="cpu", trust_remote_code=True)
    return _reranker_instance


def rerank_chunks(query: str, chunks: list, top_k: int) -> list:
    """Urutkan ulang kandidat chunk menggunakan Cross-Encoder.
    
    Args:
        query: Pertanyaan pengguna.
        chunks: List berisi kandidat awal (misal dictionary berisi id, text, dll).
        top_k: Jumlah hasil akhir yang ingin diambil setelah reranking.
        
    Returns:
        List chunk yang sudah diurutkan ulang berdasarkan cross-attention logit.
    """
    if not chunks:
        return []

    model = get_reranker()
    
    # Siapkan pasangan teks (Query, Chunk)
    sentence_pairs = [[query, chunk["text"]] for chunk in chunks]
    
    # Prediksi skor logit (semakin tinggi semakin relevan)
    scores = model.predict(sentence_pairs)
    
    # Gabungkan skor kembali ke data chunk
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
        
    # Urutkan berdasarkan rerank_score tertinggi
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    return reranked[:top_k]