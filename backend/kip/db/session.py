"""Database session management.

Supports both SQLite (default, zero-setup) and PostgreSQL (production).
The schema is defined in this module and created by ``init_db``.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, relationship

from kip.config import get_settings

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="owner", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    safe_name = Column(String(255), nullable=False)
    content_type = Column(String(64), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    storage_path = Column(String(512), nullable=False)
    title = Column(String(512))
    author = Column(String(255))
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    extractor = Column(String(64))
    warnings = Column(Text)
    status = Column(String(32), default="pending", nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    owner = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_documents_owner_created", "owner_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.filename})>"


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String(128), primary_key=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    body = Column(Text, nullable=False)
    embed_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    page_start = Column(Integer)
    page_end = Column(Integer)
    section_key = Column(String(255))
    section_path = Column(Text)
    heading = Column(String(512))
    block_orders = Column(Text)
    kinds = Column(Text)
    has_overlap = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_doc_index"),
        Index("ix_chunks_document_section", "document_id", "section_key"),
    )

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, doc={self.document_id}, idx={self.chunk_index})>"


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(512))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    owner = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title={self.title})>"


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    citations = Column(Text)
    retrieval_diagnostics = Column(Text)
    groundedness = Column(String(32))
    refused = Column(Boolean, default=False)
    refusal_reason = Column(String(64))
    model = Column(String(64))
    provider = Column(String(32))
    usage_prompt_tokens = Column(Integer)
    usage_completion_tokens = Column(Integer)
    total_ms = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role={self.role})>"


def _make_async_url(url: str) -> str:
    """Convert a sync SQLAlchemy URL to async."""
    if url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://")
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://")
    return url


def _make_sync_url(url: str) -> str:
    """Ensure a sync URL for migrations."""
    if url.startswith("sqlite+aiosqlite://"):
        return url.replace("sqlite+aiosqlite://", "sqlite://")
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://")
    return url


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_db() -> None:
    """Initialise the database engine and session factory."""
    global _engine, _session_factory
    settings = get_settings()
    sync_url = _make_sync_url(settings.database_url)
    async_url = _make_async_url(settings.database_url)

    if sync_url.startswith("sqlite"):
        path = Path(sync_url.replace("sqlite:///", "").replace("sqlite://", ""))
        path.parent.mkdir(parents=True, exist_ok=True)

    # Use NullPool for SQLite to ensure each connection is closed after use.
    # This works with file-based SQLite databases (not in-memory).
    from sqlalchemy.pool import NullPool
    pool_class = NullPool if sync_url.startswith("sqlite") else None
    pool_kwargs = {"poolclass": pool_class} if pool_class else {}

    _engine = create_async_engine(async_url, echo=settings.is_production is False, future=True, **pool_kwargs)
    _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session. Commits on success, rolls back on exception."""
    if _session_factory is None:
        init_db()
    assert _session_factory is not None
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """Create all tables. Safe to call multiple times."""
    if _engine is None:
        init_db()
    assert _engine is not None
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables() -> None:
    """Drop all tables. Used by tests."""
    if _engine is None:
        init_db()
    assert _engine is not None
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def close_db() -> None:
    """Close the database engine."""
    global _engine, _session_factory
    if _engine is not None:
        # Dispose the engine which closes all connections in the pool
        await _engine.dispose()
        # Give the event loop a chance to process any pending connection close callbacks
        import asyncio
        await asyncio.sleep(0)
        _engine = None
        _session_factory = None
