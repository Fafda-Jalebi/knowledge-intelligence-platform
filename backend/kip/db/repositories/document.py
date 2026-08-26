"""Document repository."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from kip.db.session import Document, Chunk


@dataclass(slots=True)
class DocumentSummary:
    """Lightweight document info for lists."""
    id: str
    filename: str
    title: str | None
    page_count: int
    chunk_count: int
    status: str
    created_at: str
    size_bytes: int


class DocumentRepository:
    """Document data access."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        doc_id: str,
        owner_id: int,
        filename: str,
        safe_name: str,
        content_type: str,
        size_bytes: int,
        storage_path: str,
    ) -> Document:
        """Create a new document record."""
        doc = Document(
            id=doc_id,
            owner_id=owner_id,
            filename=filename,
            safe_name=safe_name,
            content_type=content_type,
            size_bytes=size_bytes,
            storage_path=storage_path,
            status="pending",
        )
        self._session.add(doc)
        await self._session.flush()
        await self._session.refresh(doc)
        return doc

    async def get_by_id(self, doc_id: str, owner_id: int | None = None) -> Optional[Document]:
        """Get document by ID, optionally scoped to owner."""
        stmt = select(Document).where(Document.id == doc_id)
        if owner_id is not None:
            stmt = stmt.where(Document.owner_id == owner_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_with_chunks(self, doc_id: str, owner_id: int | None = None) -> Optional[Document]:
        """Get document with chunks loaded."""
        from sqlalchemy.orm import selectinload
        stmt = select(Document).options(selectinload(Document.chunks)).where(Document.id == doc_id)
        if owner_id is not None:
            stmt = stmt.where(Document.owner_id == owner_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_for_user(
        self,
        owner_id: int,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> Sequence[DocumentSummary]:
        """List documents for a user."""
        stmt = select(Document).where(Document.owner_id == owner_id)
        if status:
            stmt = stmt.where(Document.status == status)
        stmt = stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        docs = result.scalars().all()
        return [
            DocumentSummary(
                id=d.id,
                filename=d.filename,
                title=d.title,
                page_count=d.page_count,
                chunk_count=d.chunk_count,
                status=d.status,
                created_at=d.created_at.isoformat() if d.created_at else "",
                size_bytes=d.size_bytes,
            )
            for d in docs
        ]

    async def count_for_user(self, owner_id: int, status: str | None = None) -> int:
        """Count documents for a user."""
        stmt = select(func.count(Document.id)).where(Document.owner_id == owner_id)
        if status:
            stmt = stmt.where(Document.status == status)
        result = await self._session.execute(stmt)
        return result.scalar_one()

    async def update_metadata(
        self,
        doc_id: str,
        owner_id: int,
        *,
        title: str | None = None,
        author: str | None = None,
        page_count: int | None = None,
        chunk_count: int | None = None,
        extractor: str | None = None,
        warnings: str | None = None,
        status: str | None = None,
    ) -> bool:
        """Update document metadata after ingestion."""
        doc = await self.get_by_id(doc_id, owner_id)
        if not doc:
            return False
        if title is not None:
            doc.title = title
        if author is not None:
            doc.author = author
        if page_count is not None:
            doc.page_count = page_count
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        if extractor is not None:
            doc.extractor = extractor
        if warnings is not None:
            doc.warnings = warnings
        if status is not None:
            doc.status = status
        await self._session.flush()
        return True

    async def delete(self, doc_id: str, owner_id: int) -> bool:
        """Delete a document and its chunks."""
        doc = await self.get_by_id(doc_id, owner_id)
        if not doc:
            return False
        await self._session.delete(doc)
        await self._session.flush()
        return True

    async def get_chunks(
        self,
        doc_id: str,
        owner_id: int | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Chunk]:
        """Get chunks for a document."""
        stmt = select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.chunk_index)
        if owner_id is not None:
            stmt = stmt.join(Document).where(Document.owner_id == owner_id)
        if limit:
            stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def count_chunks(self, doc_id: str, owner_id: int | None = None) -> int:
        """Count chunks for a document."""
        stmt = select(func.count(Chunk.id)).where(Chunk.document_id == doc_id)
        if owner_id is not None:
            stmt = stmt.join(Document).where(Document.owner_id == owner_id)
        result = await self._session.execute(stmt)
        return result.scalar_one()

    async def bulk_insert_chunks(self, chunks: list[Chunk]) -> int:
        """Insert multiple chunks at once."""
        self._session.add_all(chunks)
        await self._session.flush()
        return len(chunks)

    async def delete_chunks_for_document(self, doc_id: str) -> int:
        """Delete all chunks for a document."""
        stmt = delete(Chunk).where(Chunk.document_id == doc_id)
        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount or 0
