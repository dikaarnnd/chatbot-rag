"""Hybrid dense and BM25 retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlmodel import Session, select

from app.core.db import Chunk, Document, engine
from app.retrieval.query_embedder import embed_query

logger = logging.getLogger(__name__)

FINAL_TOP_K = 3
RRF_K = 60
MIN_CANDIDATES = 20
CANDIDATE_MULTIPLIER = 4


@dataclass
class RetrievedChunk:
    text: str
    score: float
    file_name: str | None
    page_label: str | None


def _tokenize(text: str) -> list[str]:
    """Split Indonesian/English text into lowercase word and number tokens."""
    return re.findall(r"\w+", text.lower())


def _reciprocal_rank_fusion(dense_ids: list[str], sparse_ids: list[str]) -> list[str]:
    """Return unique chunk IDs ordered by their combined dense and BM25 ranks."""
    dense_ranks = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, start=1)}
    sparse_ranks = {chunk_id: rank for rank, chunk_id in enumerate(sparse_ids, start=1)}
    candidate_ids = list(dict.fromkeys(dense_ids + sparse_ids))

    def rank_key(chunk_id: str) -> tuple[float, int]:
        score = (
            (1 / (RRF_K + dense_ranks[chunk_id]) if chunk_id in dense_ranks else 0)
            + (1 / (RRF_K + sparse_ranks[chunk_id]) if chunk_id in sparse_ranks else 0)
        )
        best_rank = min(
            dense_ranks.get(chunk_id, len(candidate_ids) + 1),
            sparse_ranks.get(chunk_id, len(candidate_ids) + 1),
        )
        return -score, best_rank

    return sorted(candidate_ids, key=rank_key)


def search(
    query: str,
    document_id: str,
    top_k: int = FINAL_TOP_K,
    score_threshold: float | None = None,
    embed_dim: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse cosine-similarity and BM25 rankings using Reciprocal Rank Fusion.

    The final ordering comes from RRF; no cross-encoder reranking is performed.
    `score` remains the cosine distance to preserve the existing API contract.
    """
    if top_k <= 0:
        return []

    query_vector = embed_query(query, embed_dim=embed_dim)
    candidate_k = max(MIN_CANDIDATES, top_k * CANDIDATE_MULTIPLIER)

    with Session(engine) as session:
        distance_expr = Chunk.embedding.cosine_distance(query_vector)
        statement = (
            select(Chunk, Document.file_name, distance_expr.label("distance"))
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.document_id == document_id)
            .order_by(distance_expr)
        )
        rows = session.exec(statement).all()

    if not rows:
        return []

    candidates = [
        (chunk, file_name, float(distance))
        for chunk, file_name, distance in rows
        if score_threshold is None or float(distance) <= score_threshold
    ]
    if not candidates:
        return []

    tokenized_corpus = [_tokenize(chunk.text) for chunk, _, _ in candidates]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))

    dense_indices = list(range(min(candidate_k, len(candidates))))
    sparse_indices = sorted(
        (
            index
            for index, score in enumerate(bm25_scores)
            if float(score) != 0.0
        ),
        key=lambda index: float(bm25_scores[index]),
        reverse=True,
    )[:candidate_k]

    dense_ids = [candidates[index][0].id for index in dense_indices]
    sparse_ids = [candidates[index][0].id for index in sparse_indices]
    candidates_by_id = {
        chunk.id: (chunk, file_name, distance)
        for chunk, file_name, distance in candidates
    }
    ranked_ids = _reciprocal_rank_fusion(dense_ids, sparse_ids)

    output = [
        RetrievedChunk(
            text=candidates_by_id[chunk_id][0].text,
            score=candidates_by_id[chunk_id][2],
            file_name=candidates_by_id[chunk_id][1],
            page_label=candidates_by_id[chunk_id][0].page_label,
        )
        for chunk_id in ranked_ids[:top_k]
    ]

    logger.info(
        "Hybrid retrieval '%s' -> dense=%d, BM25=%d, fused=%d (top_k=%d)",
        query[:50],
        len(dense_indices),
        len(sparse_indices),
        len(output),
        top_k,
    )
    return output
