"""Document ingestion service."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from kip.config import get_settings
from kip.core import chunk_document, extract_document
from kip.core.embeddings import get_embedder
from kip.core.retrieval import get_keyword_index
from kip.core.vectorstore import get_vector_store, records_from_chunks
from kip.db.repositories import ChunkRepository, DocumentRepository
from kip.db import session as session_module
from kip.db.session import Chunk, Document
from kip.security.files import sanitise_filename, storage_key, validate_upload, PayloadTooLargeError, UnsupportedMediaTypeError


@dataclass(slots=True)
class IngestionResult:
    document_id: str
    filename: str
    title: str | None
    page_count: int
    chunk_count: int
    status: str
    created_at: str
    warnings: list[str]


class DocumentService:
    """Document ingestion and management."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._embedder = get_embedder(self._settings)
        self._vector_store = get_vector_store(self._settings)
        self._keyword_index = get_keyword_index(self._settings)
        self._vector_store.ensure_collection(self._embedder.spec)
        self._storage_root = self._settings.storage_path
        self._storage_root.mkdir(parents=True, exist_ok=True)

    async def ingest(
        self,
        owner_id: int,
        filename: str,
        content: bytes,
        content_type: str | None = None,
    ) -> IngestionResult:
        """Ingest a document: store, extract, chunk, embed, index."""
        # Validate
        safe_name, ext = validate_upload(
            filename,
            content,
            max_bytes=self._settings.max_upload_bytes,
            allowed_extensions=self._settings.allowed_extension_set,
        )

        # Generate IDs and paths
        doc_id = uuid.uuid4().hex[:12]
        storage_path = self._storage_root / storage_key(ext)
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        storage_path.write_bytes(content)

        # Extract
        extracted = extract_document(filename, content)
        warnings = list(extracted.warnings)

        # Chunk
        chunking_result = chunk_document(extracted)
        chunks = list(chunking_result.chunks)

        # Embed
        embed_texts = [c.embed_text for c in chunks]
        vectors = self._embedder.embed_documents(embed_texts)

        # Keyword index
        if self._keyword_index:
            from kip.core.retrieval.keyword import KeywordDocument
            kw_docs = [
                KeywordDocument(
                    id=f"{doc_id}:{c.index}",
                    text=c.body,
                    payload={"document_id": doc_id, "chunk_index": c.index},
                )
                for c in chunks
            ]
            self._keyword_index.add(kw_docs)

        # Persist to relational DB + vector store
        async with session_module.get_session() as session:
            doc_repo = DocumentRepository(session)
            chunk_repo = ChunkRepository(session)

            # Create document record
            doc = await doc_repo.create(
                doc_id=doc_id,
                owner_id=owner_id,
                filename=filename,
                safe_name=safe_name,
                content_type=content_type or f"application/{ext}",
                size_bytes=len(content),
                storage_path=str(storage_path),
            )

            # Build chunk records
            chunk_records = []
            vector_records = []
            for chunk, vector in zip(chunks, vectors):
                chunk_record = Chunk(
                    id=f"{doc_id}:{chunk.index}",
                    document_id=doc_id,
                    chunk_index=chunk.index,
                    body=chunk.body,
                    embed_text=chunk.embed_text,
                    token_count=chunk.token_count,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_key=chunk.section_key,
                    section_path=" > ".join(chunk.section_path) if chunk.section_path else None,
                    heading=chunk.heading,
                    block_orders=",".join(str(o) for o in chunk.block_orders),
                    kinds=",".join(chunk.kinds),
                    has_overlap=chunk.has_overlap,
                )
                chunk_records.append(chunk_record)

            vrecords = records_from_chunks(
                chunks, vectors, document_id=doc_id, user_id=owner_id
            )
            self._vector_store.upsert(vrecords)

            # Bulk insert chunks
            await chunk_repo.bulk_insert_chunks(chunk_records)

            # Update document metadata
            await doc_repo.update_metadata(
                doc_id,
                owner_id,
                title=extracted.metadata.title,
                author=extracted.metadata.author,
                page_count=extracted.page_count,
                chunk_count=len(chunks),
                extractor=extracted.extractor,
                warnings="; ".join(warnings) if warnings else None,
                status="ready",
            )
            created_at = doc.created_at.isoformat() if doc.created_at else ""

        return IngestionResult(
            document_id=doc_id,
            filename=filename,
            title=extracted.metadata.title,
            page_count=extracted.page_count,
            chunk_count=len(chunks),
            status="ready",
            created_at=created_at,
            warnings=warnings,
        )

    async def list_documents(
        self,
        owner_id: int,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> Sequence[Document]:
        async with session_module.get_session() as session:
            repo = DocumentRepository(session)
            return await repo.list_for_user(owner_id, limit=limit, offset=offset, status=status)

    async def count_for_user(
        self,
        owner_id: int,
        status: str | None = None,
    ) -> int:
        async with session_module.get_session() as session:
            repo = DocumentRepository(session)
            return await repo.count_for_user(owner_id, status)

    async def get_document(self, owner_id: int, doc_id: str) -> Document | None:
        async with session_module.get_session() as session:
            repo = DocumentRepository(session)
            return await repo.get_by_id(doc_id, owner_id)

    async def get_document_with_chunks(self, owner_id: int, doc_id: str) -> Document | None:
        async with session_module.get_session() as session:
            repo = DocumentRepository(session)
            return await repo.get_with_chunks(doc_id, owner_id)

    async def get_chunks(
        self,
        owner_id: int,
        doc_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Chunk]:
        async with session_module.get_session() as session:
            repo = DocumentRepository(session)
            return await repo.get_chunks(doc_id, owner_id, limit=limit, offset=offset)

    async def count_chunks(self, owner_id: int, doc_id: str) -> int:
        async with session_module.get_session() as session:
            repo = DocumentRepository(session)
            return await repo.count_chunks(doc_id, owner_id)

    async def delete_document(self, owner_id: int, doc_id: str) -> bool:
        """Delete a document and all its data from all stores."""
        async with session_module.get_session() as session:
            doc_repo = DocumentRepository(session)
            chunk_repo = ChunkRepository(session)

            doc = await doc_repo.get_by_id(doc_id, owner_id)
            if not doc:
                return False

            # Delete from vector store
            self._vector_store.delete(filters={"document_id": [doc_id]})

            # Delete from keyword index
            if self._keyword_index:
                self._keyword_index.delete(filters={"document_id": [doc_id]})

            # Delete chunks from relational DB
            await chunk_repo.delete_by_document(doc_id)

            # Delete document record
            await doc_repo.delete(doc_id, owner_id)

            # Delete file from storage
            try:
                Path(doc.storage_path).unlink(missing_ok=True)
            except OSError:
                pass

            return True

    async def get_chunk_texts(self, chunk_ids: list[str]) -> dict[str, str]:
        async with session_module.get_session() as session:
            repo = ChunkRepository(session)
            return await repo.get_text_by_ids(chunk_ids)

    async def get_titles_for_documents(self, doc_ids: list[str]) -> dict[str, str]:
        async with session_module.get_session() as session:
            repo = ChunkRepository(session)
            return await repo.get_titles_for_documents(doc_ids)
