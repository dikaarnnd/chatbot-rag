"""Debug script: inspect real token counts (Qwen3 tokenizer vs tiktoken) dan
overlap aktual antar chunk bersebelahan yang dihasilkan chunker.py.

SentenceSplitter LlamaIndex menghitung chunk_size/chunk_overlap pakai
tokenizer tiktoken (cl100k_base) secara default -- BUKAN tokenizer asli
Qwen3-Embedding-0.6B yang dipakai saat embedding. Script ini menunjukkan
selisihnya secara konkret, plus teks overlap aktual antar chunk.

Usage:
    cd backend
    .\\venv\\Scripts\\Activate.ps1
    pip install tiktoken   # belum ada di requirements.txt, tambahkan manual
    python scripts\\debug_chunking.py "..\\data\\raw\\nama_file.pdf"
    python scripts\\debug_chunking.py "..\\data\\raw\\nama_file.pdf" --chunk-size 256 --chunk-overlap 50
"""

from __future__ import annotations

import argparse
import difflib
import logging
import sys
from pathlib import Path

# Supaya `import app...` bisa jalan tanpa install package -- sisipkan
# folder backend/ (parent dari scripts/) ke sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from app.ingestion.chunker import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_documents,
)
from app.ingestion.embedder import MODEL_NAME as QWEN_MODEL_NAME  # noqa: E402
from app.ingestion.loader import load_pdf  # noqa: E402

logging.basicConfig(level=logging.WARNING)  # redam log INFO module lain

TIKTOKEN_ENCODING = "cl100k_base"  # dipakai SentenceSplitter secara default
SEPARATOR = "=" * 78


def count_tokens_tiktoken(text: str, encoding) -> int:
    return len(encoding.encode(text))


def count_tokens_qwen(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def find_overlap(text_a: str, text_b: str) -> str:
    """Cari substring terpanjang yang sama antara akhir text_a dan awal text_b."""
    matcher = difflib.SequenceMatcher(None, text_a, text_b, autojunk=False)
    block = matcher.find_longest_match(0, len(text_a), 0, len(text_b))
    if block.size == 0:
        return ""
    return text_a[block.a : block.a + block.size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug tokenisasi & overlap chunking Chatbot RAG")
    parser.add_argument("pdf_path", type=str)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--max-chunks", type=int, default=5,
        help="Batasi jumlah chunk yang ditampilkan detail (default 5)",
    )
    args = parser.parse_args()

    print(SEPARATOR)
    print(f"DEBUG TOKENIZATION -- {Path(args.pdf_path).name}")
    print(f"chunk_size (target, tiktoken cl100k_base)    = {args.chunk_size}")
    print(f"chunk_overlap (target, tiktoken cl100k_base) = {args.chunk_overlap}")
    print(SEPARATOR)

    print("\n[1/3] Loading PDF...")
    documents = load_pdf(args.pdf_path)
    print(f"    -> {len(documents)} halaman berhasil di-load")

    print("\n[2/3] Chunking (SentenceSplitter, tokenizer internal = tiktoken cl100k_base)...")
    nodes = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"    -> {len(nodes)} chunk dihasilkan")

    print("\n[3/3] Loading tokenizer untuk perbandingan...")
    tiktoken_enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    print(f"    -> tiktoken '{TIKTOKEN_ENCODING}' siap (dipakai SentenceSplitter secara internal)")
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    print(f"    -> Qwen3 tokenizer '{QWEN_MODEL_NAME}' siap (tokenizer asli saat embedding)")

    n_show = min(args.max_chunks, len(nodes))
    print(f"\n{SEPARATOR}")
    print(f"DETAIL {n_show} CHUNK PERTAMA (dari total {len(nodes)})")
    print(SEPARATOR)

    for i, node in enumerate(nodes[:n_show], start=1):
        text = node.get_content()
        n_chars = len(text)
        n_tik = count_tokens_tiktoken(text, tiktoken_enc)
        n_qwen = count_tokens_qwen(text, qwen_tokenizer)
        page = node.metadata.get("page_label", "?")

        selisih = n_qwen - n_tik
        pct = (selisih / n_tik * 100) if n_tik else 0.0
        preview = text[:120].replace("\n", " ")

        print(f"\n--- Chunk {i} (halaman {page}) ---")
        print(f"  Karakter        : {n_chars}")
        print(f"  Token (tiktoken): {n_tik}   <- basis chunk_size={args.chunk_size} di SentenceSplitter")
        print(f"  Token (Qwen3)   : {n_qwen}   <- token real saat masuk embedding model")
        print(f"  Selisih         : {selisih:+d} token ({pct:+.1f}%)")
        print(f'  Preview awal    : "{preview}..."')

    print(f"\n{SEPARATOR}")
    print("OVERLAP ANTAR CHUNK BERSEBELAHAN")
    print(SEPARATOR)

    n_pairs = max(min(args.max_chunks, len(nodes)) - 1, 0)
    for i in range(n_pairs):
        text_a = nodes[i].get_content()
        text_b = nodes[i + 1].get_content()
        overlap_text = find_overlap(text_a, text_b)

        page_a = nodes[i].metadata.get("page_label", "?")
        page_b = nodes[i + 1].metadata.get("page_label", "?")

        print(f"\n--- Chunk {i + 1} (hal. {page_a}) <-> Chunk {i + 2} (hal. {page_b}) ---")
        if overlap_text:
            n_overlap_tik = count_tokens_tiktoken(overlap_text, tiktoken_enc)
            n_overlap_qwen = count_tokens_qwen(overlap_text, qwen_tokenizer)
            preview = overlap_text[:150].replace("\n", " ")
            print(f"  Overlap ditemukan       : {len(overlap_text)} karakter")
            print(f"  Token overlap (tiktoken): {n_overlap_tik}   (target overlap={args.chunk_overlap})")
            print(f"  Token overlap (Qwen3)   : {n_overlap_qwen}")
            print(f'  Teks overlap: "{preview}..."')
        else:
            print(
                "  Tidak ada overlap terdeteksi (kemungkinan chunk beda halaman -- "
                "chunking per-halaman, overlap tidak menjembatani lintas halaman)"
            )

    print(f"\n{SEPARATOR}")
    print("RINGKASAN AGREGAT (seluruh chunk)")
    print(SEPARATOR)

    all_tik = [count_tokens_tiktoken(n.get_content(), tiktoken_enc) for n in nodes]
    all_qwen = [count_tokens_qwen(n.get_content(), qwen_tokenizer) for n in nodes]

    def stats(values: list[int]) -> tuple[int, float, int]:
        return min(values), sum(values) / len(values), max(values)

    tmin, tavg, tmax = stats(all_tik)
    qmin, qavg, qmax = stats(all_qwen)
    avg_ratio = (qavg / tavg) if tavg else 0.0
    over_limit = sum(1 for t in all_qwen if t > args.chunk_size)

    print(f"  Total chunk                    : {len(nodes)}")
    print(f"  Token tiktoken (min/avg/max)   : {tmin} / {tavg:.1f} / {tmax}")
    print(f"  Token Qwen3    (min/avg/max)   : {qmin} / {qavg:.1f} / {qmax}")
    print(f"  Rasio rata-rata Qwen3/tiktoken : {avg_ratio:.2f}x")
    print(f"  Chunk yang > chunk_size={args.chunk_size} token (dihitung Qwen3): {over_limit}/{len(nodes)}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()

# run --> python scripts\debug_chunking.py "..\data\raw\nama_file.pdf"