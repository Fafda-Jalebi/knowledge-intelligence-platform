"""Chunk repository."""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from kip.db.session import Chunk


class ChunkRepository:
    """Chunk data access."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, chunk_id: str, owner_id: int | None = None) -> Optional[Chunk]:
        """Get a chunk by ID."""
        stmt = select(Chunk).where(Chunk.id == chunk_id)
        if owner_id is not None:
            stmt = stmt.join(Chunk.document).where(Chunk.document.has(id=chunk_id.split(":")[0]))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_ids(self, chunk_ids: list[str], owner_id: int | None = None) -> list[Chunk]:
        """Get multiple chunks by ID."""
        if not chunk_ids:
            return []
        stmt = select(Chunk).where(Chunk.id.in_(chunk_ids))
        if owner_id is not None:
            doc_ids = list({cid.split(":")[0] for cid in chunk_ids})
            stmt = stmt.join(Chunk.document).where(Chunk.document.has(id=doc_ids[0]))
            for doc_id in doc_ids[1:]:
                stmt = stmt.where(Chunk.document.has(id=doc_id))
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_text_by_ids(self, chunk_ids: list[str]) -> dict[str, str]:
        """Get chunk text keyed by chunk ID."""
        chunks = await self.get_by_ids(chunk_ids)
        return {c.id: c.body for c in chunks}

    async def get_titles_for_documents(self, doc_ids: list[str]) -> dict[str, str]:
        """Get document titles for a list of document IDs."""
        from kip.db.session import Document
        if not doc_ids:
            return {}
        stmt = select(Document.id, Document.title).where(Document.id.in_(doc_ids))
        result = await self._session.execute(stmt)
        return {row.id: row.title or row.id for row in result.all()}

    async def search_by_document(
        self,
        document_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Chunk]:
        """Get chunks for a document in order."""
        stmt = (
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def count_by_document(self, document_id: str) -> int:
        """Count chunks for a document."""
        stmt = select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
        result = await self._session.execute(stmt)
        return result.scalar_one()

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        stmt = delete(Chunk).where(Chunk.document_id == document_id)
        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount or 0

    async def bulk_insert_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Bulk insert multiple chunks."""
        if not chunks:
            return 0
        self._session.add_all(chunks)
        await self._session.flush()
        return len(chunks)
