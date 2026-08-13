"""FastAPI application entrypoint for Chatbot RAG backend."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlmodel import Session, select

from app.core.config import get_settings
from app.core.db import Chunk, ChatSession, Document, Message, engine, get_db_session, init_db
from app.core.schemas import (
    ChatRequest,
    IngestResponse,
    MessageSchema,
    SessionSummary,
    SourceCitationSchema,
    StatsResponse,
)
from app.generation.gemini_client import GenerationError
from app.generation.pipeline import answer_question_stream
from app.ingestion.loader import PDFLoadError
from app.ingestion.pipeline import ingest_pdf

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title="Chatbot RAG API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Pastikan tabel & extension pgvector siap sebelum menerima request."""
    init_db()


@app.post("/documents", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db_session),
) -> IngestResponse:
    """Upload 1 PDF, jalankan ingestion, dan otomatis buat 1 ChatSession baru.

    Beda dari versi sebelumnya: dokumen TIDAK menggantikan dokumen lama --
    semua dokumen yang pernah di-upload tetap tersimpan permanen di Supabase,
    masing-masing dengan sesi chat sendiri (lihat DECISIONS.md).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File harus berformat .pdf")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = ingest_pdf(
            tmp_path,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            embed_dim=settings.embed_dim,
        )
    except PDFLoadError as exc:
        logger.warning("Upload ditolak untuk '%s': %s", file.filename, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion gagal tak terduga untuk file '%s'", file.filename)
        raise HTTPException(status_code=500, detail="Gagal memproses dokumen.") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    # Judul chat = nama file PDF (proxy untuk judul jurnal/skripsi -- kita
    # tidak parse judul asli dari isi PDF, di luar scope saat ini).
    session = ChatSession(document_id=result.document_id, title=file.filename)
    db.add(session)
    db.commit()
    db.refresh(session)

    return IngestResponse(
        document_id=result.document_id,
        session_id=session.id,
        file_name=file.filename,
        pages_loaded=result.pages_loaded,
        chunks_created=result.chunks_created,
        points_upserted=result.points_upserted,
        embed_dim=result.embed_dim,
        chunk_size=result.chunk_size,
        chunk_overlap=result.chunk_overlap,
        duration_seconds=result.duration_seconds,
    )


@app.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db_session)) -> StreamingResponse:
    """Jawab 1 pertanyaan secara streaming (SSE), simpan user+assistant message
    ke database supaya riwayat persisten lintas sesi/restart.

    Event types: sources, delta, done, error -- lihat generation/pipeline.py.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong")

    session = db.get(ChatSession, request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session tidak ditemukan")

    # PENTING: ekstrak nilai yang dibutuhkan generator SEKARANG, selagi `db`
    # masih hidup. event_generator() baru benar-benar jalan SETELAH endpoint
    # ini return -- di titik itu FastAPI sudah menutup `db` (StreamingResponse
    # gotcha), jadi objek ORM `session` tidak boleh diakses lagi di dalamnya
    # (akan raise DetachedInstanceError kalau dipaksa).
    session_id = session.id
    document_id = session.document_id

    history_rows = db.exec(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(settings.history_turns * 2)
    ).all()
    history: list[dict] = [
        {"role": m.role, "content": m.content} for m in reversed(history_rows)
    ]

    # Simpan pesan user SEBELUM generation mulai -- kalau generation gagal
    # di tengah jalan, riwayat pertanyaan user tetap tercatat.
    user_message = Message(session_id=session_id, role="user", content=request.question)
    db.add(user_message)
    db.commit()

    top_k = request.top_k if request.top_k is not None else settings.top_k

    def event_generator():
        accumulated_answer = ""
        accumulated_sources: list[dict] = []
        used_fallback = False

        try:
            for event in answer_question_stream(
                request.question,
                document_id=document_id,
                history=history,
                top_k=top_k,
                score_threshold=settings.score_threshold,
                model=settings.gemini_model,
                max_output_tokens=settings.max_output_tokens,
                embed_dim=settings.embed_dim,
            ):
                if event["type"] == "sources":
                    accumulated_sources = event["sources"]
                elif event["type"] == "delta":
                    accumulated_answer += event["text"]
                elif event["type"] == "done":
                    used_fallback = event["used_fallback"]

                yield f"data: {json.dumps(event)}\n\n"

            # Stream selesai tanpa error -- simpan pesan assistant lengkap.
            try:
                assistant_message = Message(
                    session_id=session_id,
                    role="assistant",
                    content=accumulated_answer,
                    used_fallback=used_fallback,
                )
                assistant_message.set_sources(accumulated_sources)
                with Session(engine) as persist_session:
                    persist_session.add(assistant_message)
                    persist_session.commit()
            except Exception:
                logger.exception(
                    "Gagal simpan pesan assistant ke DB untuk session_id=%s", session.id
                )
                yield f"data: {json.dumps({'type': 'error', 'detail': 'Jawaban berhasil dibuat tapi gagal disimpan ke riwayat.'})}\n\n"

        except GenerationError as exc:
            logger.exception("Generation gagal di tengah stream untuk: '%s'", request.question[:80])
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
        except Exception as exc:
            # Tangkap SEMUA exception
            logger.exception(
                "Error tak terduga di /chat event_generator untuk: '%s'", request.question[:80]
            )
            yield f"data: {json.dumps({'type': 'error', 'detail': f'{type(exc).__name__}: {exc}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/sessions", response_model=list[SessionSummary])
def list_sessions(db: Session = Depends(get_db_session)) -> list[SessionSummary]:
    """Daftar semua chat session untuk sidebar berdasarkan terbaru"""
    sessions = db.exec(select(ChatSession).order_by(ChatSession.created_at.desc())).all()
    return [
        SessionSummary(id=s.id, title=s.title, created_at=s.created_at) for s in sessions
    ]


@app.get("/sessions/{session_id}/messages", response_model=list[MessageSchema])
def get_session_messages(session_id: str, db: Session = Depends(get_db_session)) -> list[MessageSchema]:
    """Riwayat pesan lengkap 1 sesi, urut kronologis -- dipanggil saat user
    klik sesi lama di sidebar untuk melanjutkan percakapan."""
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session tidak ditemukan")

    messages = db.exec(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    ).all()

    return [
        MessageSchema(
            id=m.id,
            role=m.role,
            content=m.content,
            sources=[SourceCitationSchema(**s) for s in m.get_sources()],
            used_fallback=m.used_fallback,
            created_at=m.created_at,
        )
        for m in messages
    ]

@app.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db_session)) -> StatsResponse:
    """Statistik agregat sistem -- dipakai halaman /monitor untuk kartu angka."""
    document_count = db.exec(select(func.count()).select_from(Document)).one()
    chunk_count = db.exec(select(func.count()).select_from(Chunk)).one()
    session_count = db.exec(select(func.count()).select_from(ChatSession)).one()
    message_count = db.exec(select(func.count()).select_from(Message)).one()
 
    assistant_messages = db.exec(select(Message).where(Message.role == "assistant")).all()
    fallback_count = sum(1 for m in assistant_messages if m.used_fallback)
    fallback_rate = (
        round(fallback_count / len(assistant_messages), 3) if assistant_messages else 0.0
    )
 
    return StatsResponse(
        document_count=document_count,
        chunk_count=chunk_count,
        session_count=session_count,
        message_count=message_count,
        fallback_rate=fallback_rate,
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}