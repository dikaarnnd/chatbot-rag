from __future__ import annotations

from app.retrieval.search import RetrievedChunk

SYSTEM_PROMPT = (
    "Kamu adalah asisten yang menjawab HANYA berdasarkan konteks yang diberikan. "
    "Konteks ini berisi teks dari dokumen asli beserta penanda halamannya. "
    "PERHATIAN: Baca konteks secara saksama kata demi kata. Jangan lewatkan detail kecil, "
    "catatan kaki, atau kalimat pendek yang tersembunyi di dalam teks.\n\n"
    "Jika jawaban tidak ditemukan dalam konteks, katakan dengan jelas bahwa "
    "informasi tidak tersedia dalam dokumen -- jangan mengarang atau menggunakan "
    "pengetahuan di luar konteks. Sertakan referensi halaman jika tersedia di "
    "metadata konteks.\n\n"
    "Jawab langsung ke inti pertanyaan, seperti menjelaskan ke rekan kerja. "
    "JANGAN awali jawaban dengan frasa seperti 'berdasarkan konteks yang diberikan', "
    "'dari dokumen tersebut', atau variasi sejenisnya -- langsung masuk ke jawabannya. "
    "Tulis dalam teks polos (plain text): JANGAN gunakan markdown seperti tanda "
    "bintang untuk bold/italic, heading dengan tanda pagar, atau bullet point "
    "bertanda '-'/'*'. Kalau perlu daftar, tulis dalam kalimat naratif atau pakai "
    "penomoran biasa (1., 2., dst)."
    "Kalau ada riwayat percakapan sebelumnya, gunakan itu HANYA untuk memahami "
    "maksud pertanyaan lanjutan (misal kata ganti 'itu'/'nya', atau pertanyaan "
    "susulan yang merujuk topik sebelumnya). TAPI tetap HANYA jawab dari konteks "
    "dokumen yang diberikan saat ini -- JANGAN menjawab dari ingatan percakapan "
    "sebelumnya kalau topik pertanyaan baru ternyata di luar cakupan konteks yang "
    "diberikan sekarang. Kalau itu terjadi, sampaikan dengan sopan dan jujur bahwa "
    "informasi tersebut tidak ditemukan dalam dokumen ini -- boleh singgung bahwa "
    "pertanyaan ini tampaknya beda topik dari percakapan sebelumnya, tapi jangan "
    "menebak atau mengarang jawaban hanya supaya terlihat membantu."
)

def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks jadi blok teks dengan sitasi halaman per potongan.

    Args:
        chunks: Hasil dari retrieval/search.py, urut dari paling relevan.

    Returns:
        String konteks siap masuk prompt. Tiap chunk diberi label sumber +
        halaman.
    """
    blocks = [
        # f"[Sumber {i} - Halaman {chunk.page_label or '?'}]\n{chunk.text}"
        f"[Potongan Dokumen {i}]\n{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)

def format_history(history: list[dict] | None) -> str:
    """Format riwayat percakapan (N pesan terakhir) jadi blok teks untuk prompt.
 
    Args:
        history: List {"role": "user"|"assistant", "content": str}, urut
            kronologis (lama ke baru). None/kosong = tidak ada riwayat.
 
    Returns:
        String riwayat siap disisipkan ke prompt, string kosong kalau
        history kosong/None (supaya tidak nambah bagian prompt yang tidak
        perlu untuk pertanyaan pertama di sebuah sesi).
    """
    if not history:
        return ""
 
    lines = [
        f"{'User' if msg['role'] == 'user' else 'Asisten'}: {msg['content']}"
        for msg in history
    ]
    return "Riwayat percakapan sebelumnya (untuk konteks pertanyaan lanjutan):\n" + "\n".join(lines)

def build_user_message(
    question: str, 
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> str:
    """Bangun isi user message: konteks (dengan sitasi) + pertanyaan.

    Args:
        question: Pertanyaan user.
        chunks: Hasil retrieval (list RetrievedChunk). Asumsikan sudah
            dipastikan tidak kosong oleh caller (lihat NO_CONTEXT_MESSAGE).

    Returns:
        String yang siap dikirim ke LLM.

    Raises:
        ValueError: kalau question kosong/whitespace saja.
    """
    question = question.strip()
    if not question:
        raise ValueError("Pertanyaan kosong.")

    context = format_context(chunks)
    history_block = format_history(history)

    parts = [p for p in (history_block, f"Konteks:\n{context}", f"Pertanyaan: {question}") if p]
    return "\n\n".join(parts)