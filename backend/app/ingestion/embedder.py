from __future__ import annotations

import logging

from llama_index.core.schema import BaseNode
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
NATIVE_EMBED_DIM = 1024  # dimensi native model, dipakai indexer.py kalau embed_dim=None
DEFAULT_BATCH_SIZE = 16  # kecil karena CPU-only


def get_embedder(embed_dim: int | None = None) -> SentenceTransformer:
    """Instantiate Qwen3-Embedding-0.6B untuk embedding dokumen.

    Args:
        embed_dim: Kalau di-set, truncate output embedding ke dimensi ini
            (Matryoshka truncation, model native = 1024). Default None =
            pakai dimensi native penuh

    Returns:
        SentenceTransformer instance, siap dipakai untuk embed dokumen.
    """
    kwargs: dict = {"device": "cpu", "trust_remote_code": True}
    if embed_dim is not None:
        kwargs["truncate_dim"] = embed_dim

    logger.info(
        "Loading embedding model '%s' (device=cpu, embed_dim=%s)",
        MODEL_NAME, embed_dim or f"native({NATIVE_EMBED_DIM})",
    )
    return SentenceTransformer(MODEL_NAME, **kwargs)


def embed_nodes(
    nodes: list[BaseNode],
    embed_dim: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[BaseNode]:
    """Embed nodes (chunks) sebagai dokumen -- TANPA instruction prefix.

    Args:
        nodes: Output dari `chunk_documents()`.
        embed_dim: Lihat `get_embedder()`.
        batch_size: Ukuran batch encoding. Kecilkan kalau OOM di CPU.

    Returns:
        Nodes yang sama, dengan `.embedding` (list[float]) terisi.

    Raises:
        ValueError: kalau nodes kosong.
    """
    if not nodes:
        raise ValueError("nodes kosong -- tidak ada yang bisa di-embed.")

    model = get_embedder(embed_dim=embed_dim)
    texts = [node.get_content() for node in nodes]

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    for node, vector in zip(nodes, vectors):
        node.embedding = vector.tolist()

    dim = len(vectors[0]) if len(vectors) else 0
    logger.info("Embedded %d nodes (dim=%d)", len(nodes), dim)

    return nodes