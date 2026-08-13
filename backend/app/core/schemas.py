from __future__ import annotations

"""Pydantic request/response schemas for Chatbot RAG API endpoints (Fase 4).

Terpisah dari dataclass core (ingestion/pipeline.py IngestionResult,
generation/pipeline.py QueryResult/SourceCitation) supaya modul core tetap
framework-agnostic (bisa dipakai standalone/CLI tanpa dependency Pydantic) --
schema di sini murni untuk serialisasi API layer.
"""

"""Pydantic request/response schemas for Chatbot RAG API endpoints.

Terpisah dari dataclass core dan dari SQLModel database models (core/db.py)
-- schema di sini murni untuk serialisasi API layer.
"""

from datetime import datetime

from pydantic import BaseModel, Field


# --- Upload / Ingestion ---

class IngestResponse(BaseModel):
    """Response setelah upload & ingest 1 PDF."""

    session_id: str = Field(..., description="ID ChatSession baru yang dibuat untuk dokumen ini")
    document_id: str = Field(..., description="ID Document -- dipakai internal untuk filter retrieval")
    file_name: str = Field(..., description="Nama file PDF yang di-upload")
    pages_loaded: int = Field(..., description="Jumlah halaman berhasil di-load")
    chunks_created: int = Field(..., description="Jumlah chunk hasil splitting")
    points_upserted: int = Field(..., description="Jumlah baris Chunk tersimpan di database")
    embed_dim: int = Field(..., description="Dimensi embedding yang dipakai")
    chunk_size: int
    chunk_overlap: int
    duration_seconds: float = Field(..., description="Waktu proses ingestion end-to-end")


# --- Chat / Query ---

class ChatRequest(BaseModel):
    """Request untuk endpoint tanya-jawab."""

    session_id: str = Field(..., description="ID ChatSession -- menentukan dokumen mana yang ditanya")
    question: str = Field(..., min_length=1, description="Pertanyaan user, tidak boleh kosong")
    top_k: int | None = Field(
        default=None, ge=3, le=20,
        description="Override jumlah chunk yang di-retrieve. Default: Settings (top_k=5).",
    )


class SourceCitationSchema(BaseModel):
    """Sitasi sumber -- mirror dari SourceCitation (generation/pipeline.py)."""

    file_name: str | None
    page_label: str | None
    score: float


class ChatResponse(BaseModel):
    """(Tidak dipakai langsung -- /chat streaming SSE. Disimpan untuk referensi
    bentuk data per-event, dan dipakai ulang di MessageSchema di bawah.)"""

    question: str
    answer: str
    sources: list[SourceCitationSchema] = Field(default_factory=list)
    used_fallback: bool
    duration_seconds: float


# --- Riwayat chat (sidebar) ---

class SessionSummary(BaseModel):
    """1 entri di sidebar riwayat chat."""

    id: str
    title: str = Field(..., description="Judul chat -- diisi nama file dokumen saat sesi dibuat")
    created_at: datetime


class MessageSchema(BaseModel):
    """1 pesan dalam riwayat percakapan (dipakai GET /sessions/{id}/messages)."""

    id: str
    role: str  # "user" | "assistant"
    content: str
    sources: list[SourceCitationSchema] = Field(default_factory=list)
    used_fallback: bool
    created_at: datetime

class StatsResponse(BaseModel):
    """Statistik agregat sistem -- dipakai halaman /monitor."""
 
    document_count: int
    chunk_count: int
    session_count: int
    message_count: int
    fallback_rate: float = Field(..., description="Proporsi jawaban assistant yang fallback (0.0-1.0)")