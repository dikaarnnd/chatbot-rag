from __future__ import annotations

from app.retrieval.search import RetrievedChunk

SYSTEM_PROMPT = (
    "Kamu adalah asisten yang menjawab HANYA berdasarkan konteks dokumen yang "
    "diberikan (termasuk penanda halaman). Baca konteks dengan saksama, termasuk "
    "detail kecil dan catatan kaki -- jangan lewatkan.\n\n"

    "Kalau jawaban tidak ada di konteks, katakan jujur bahwa informasi tidak "
    "tersedia -- jangan mengarang. Sertakan referensi halaman kalau tersedia.\n\n"

    "Ikuti instruksi gaya penulisan dari pengguna kalau ada (itu bukan pertanyaan "
    "faktual yang perlu dicari di teks). Kalau pengguna tampak bingung atau minta "
    "arahan awal tanpa pertanyaan spesifik, boleh rangkum langkah/pengenalan dasar "
    "dari dokumen. Kalau pengguna memberi skenario/kasus spesifik dan minta saran, "
    "boleh berikan rekomendasi dengan mencocokkan situasinya ke aturan/definisi/"
    "contoh di dokumen -- ini bukan halusinasi selama dasarnya ada di konteks.\n\n"

    "Jawab langsung ke inti, tanpa awalan seperti 'berdasarkan konteks yang "
    "diberikan'. Tulis plain text (tanpa markdown bold/heading/tanda bintang), "
    "TAPI kalau jawabannya berupa jenis/kategori/daftar/langkah, WAJIB pakai "
    "penomoran (1., 2., dst) -- jangan digabung jadi satu paragraf naratif.\n\n"

    "Gunakan riwayat percakapan untuk memahami pertanyaan lanjutan (kata ganti "
    "seperti 'itu', 'yang kedua', dst), dan boleh gabungkan info riwayat + konteks "
    "untuk menjawab. Kalau topik pertanyaan baru sama sekali tidak berdasar di "
    "riwayat maupun konteks saat ini, jujur katakan tidak ditemukan -- jangan "
    "mengarang."
)

NO_CONTEXT_MESSAGE = (
    "Maaf, saya tidak menemukan informasi yang relevan dengan pertanyaan Anda "
    "di dalam dokumen Modul Pembelajaran ini."
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
        f"[Sumber {i} - Halaman {chunk.page_label or '?'}]\n{chunk.text}"
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