"""Database models for Chatbot RAG -- Supabase (Postgres + pgvector).

Menggantikan db.py versi SQLite sebelumnya. Satu database Supabase menyimpan
SEMUANYA: dokumen, chunk+vector (menggantikan Qdrant), riwayat sesi chat,
dan pesan -- satu ORM (SQLAlchemy via SQLModel), bukan 2 sistem terpisah.

Model embedding TETAP Qwen3-Embedding-0.6B (self-hosted) -- tidak jadi pivot
ke Gemini Embedding API, karena alasan RAM-constraint deployment sudah tidak
berlaku (backend tetap Docker lokal, tidak di-deploy ke platform manapun).
"""

import json
import os
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlmodel import Field, Relationship, Session, SQLModel, create_engine

# Dimensi native Qwen3-Embedding-0.6B -- HARUS sama persis dengan output
# embedder.py/query_embedder.py. Kalau model embedding ganti, ini yang
# pertama harus diupdate (dan tabel Chunk perlu di-recreate/migrate).
EMBEDDING_DIM = 1024


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(SQLModel, table=True):
    """1 baris = 1 PDF yang pernah di-upload & di-index."""

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    file_name: str
    pages_loaded: int
    chunks_created: int
    uploaded_at: datetime = Field(default_factory=_utcnow)

    sessions: list["ChatSession"] = Relationship(back_populates="document")
    chunks: list["Chunk"] = Relationship(back_populates="document")


class Chunk(SQLModel, table=True):
    """1 baris = 1 potongan teks + embedding-nya -- menggantikan point Qdrant.

    Kolom `embedding` bertipe pgvector -- similarity search (cosine distance)
    dilakukan langsung lewat SQL operator `<=>`, bukan panggil vector DB
    terpisah. `document_id` di-index karena retrieval SELALU difilter per
    dokumen (lihat retrieval/search.py revisi berikutnya).
    """

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    document_id: str = Field(foreign_key="document.id", index=True)
    text: str
    page_label: str | None = None
    embedding: list[float] = Field(sa_type=Vector(EMBEDDING_DIM))

    document: Document = Relationship(back_populates="chunks")


class ChatSession(SQLModel, table=True):
    """1 baris = 1 thread percakapan, terkait ke 1 dokumen.

    `title` didenormalisasi dari Document.file_name saat sesi dibuat --
    query listing sidebar jadi cepat tanpa join, judul tetap stabil historis.
    """

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    document_id: str = Field(foreign_key="document.id")
    title: str
    created_at: datetime = Field(default_factory=_utcnow)

    document: Document = Relationship(back_populates="sessions")
    messages: list["Message"] = Relationship(back_populates="session")


class Message(SQLModel, table=True):
    """1 baris = 1 pesan (user atau assistant) dalam 1 ChatSession."""

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    session_id: str = Field(foreign_key="chatsession.id", index=True)
    role: str  # "user" | "assistant"
    content: str
    sources_json: str = Field(default="[]")  # list[SourceCitation], serialize manual
    used_fallback: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)

    session: ChatSession = Relationship(back_populates="messages")

    def get_sources(self) -> list[dict]:
        """Deserialize sources_json -- dipakai saat kirim history ke frontend."""
        return json.loads(self.sources_json)

    def set_sources(self, sources: list[dict]) -> None:
        """Serialize list sources jadi JSON string sebelum disimpan."""
        self.sources_json = json.dumps(sources)


# --- Engine & session management ---

# DATABASE_URL dibaca dari env var -- isi connection string Supabase
# (Database > Connection string > mode "Session pooler", BUKAN "Transaction
# pooler" -- mode itu punya batasan prepared statement yang bentrok
# dengan SQLAlchemy) di backend/.env.
load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# pool_pre_ping=True: cek koneksi masih hidup sebelum dipakai -- penting
# untuk database cloud (Supabase) yang bisa drop idle connection sepihak.
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)


def init_db() -> None:
    """Aktifkan extension pgvector + buat semua tabel kalau belum ada.

    Dipanggil sekali saat app startup (lihat main.py).
    """
    with engine.connect() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    SQLModel.metadata.create_all(engine)


def get_db_session():
    """FastAPI dependency -- 1 DB session per request, auto-close setelahnya."""
    with Session(engine) as session:
        yield session

# test koneksi -> python -c "from app.core.db import init_db; init_db(); print('OK')"