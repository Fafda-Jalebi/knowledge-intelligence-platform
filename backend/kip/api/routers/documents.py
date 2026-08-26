"""Document management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from kip.db import session as session_module
from kip.db.repositories import DocumentRepository
from kip.errors import UnsupportedMediaTypeError, PayloadTooLargeError
from kip.services.auth import AuthService
from kip.services.documents import DocumentService

router = APIRouter()
doc_service = DocumentService()
auth_service = AuthService()
security = HTTPBearer(auto_error=False)


class DocumentResponse(BaseModel):
    id: str
    filename: str
    title: str | None
    page_count: int
    chunk_count: int
    status: str
    created_at: str
    size_bytes: int
    warnings: list[str] = []


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class DeleteResponse(BaseModel):
    message: str


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> int:
    """Extract and validate user ID from bearer token."""
    from kip.api.routers.auth import get_current_user_id as _get_current_user_id
    return await _get_current_user_id(credentials)


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
) -> DocumentResponse:
    """Upload and ingest a document."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    try:
        result = await doc_service.ingest(user_id, file.filename or "unknown", content, file.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except UnsupportedMediaTypeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except PayloadTooLargeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ingestion failed: {exc}") from exc

    return DocumentResponse(
        id=result.document_id,
        filename=result.filename,
        title=result.title,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        status=result.status,
        created_at="",
        size_bytes=len(content),
        warnings=result.warnings,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    status_filter: str | None = None,
    user_id: int = Depends(get_current_user_id),
) -> DocumentListResponse:
    """List user's documents."""
    docs = await doc_service.list_documents(user_id, limit=limit, offset=offset, status=status_filter)
    total = await doc_service.count_for_user(user_id, status_filter)
    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=d.id,
                filename=d.filename,
                title=d.title,
                page_count=d.page_count,
                chunk_count=d.chunk_count,
                status=d.status,
                created_at=d.created_at,
                size_bytes=d.size_bytes,
                warnings=d.warnings.split("; ") if d.warnings else [],
            )
            for d in docs
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    user_id: int = Depends(get_current_user_id),
) -> DocumentResponse:
    """Get document details."""
    doc = await doc_service.get_document(user_id, doc_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        title=doc.title,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        status=doc.status,
        created_at=doc.created_at.isoformat() if doc.created_at else "",
        size_bytes=doc.size_bytes,
        warnings=doc.warnings.split("; ") if doc.warnings else [],
    )


@router.get("/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    limit: int = 50,
    offset: int = 0,
    user_id: int = Depends(get_current_user_id),
) -> dict:
    """Get chunks for a document."""
    doc = await doc_service.get_document(user_id, doc_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    chunks = await doc_service.get_chunks(doc_id, user_id, limit=limit, offset=offset)
    total = await doc_service.count_chunks(doc_id, user_id)

    return {
        "chunks": [
            {
                "id": c.id,
                "index": c.chunk_index,
                "body": c.body[:500] + "..." if len(c.body) > 500 else c.body,
                "token_count": c.token_count,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section_key": c.section_key,
                "heading": c.heading,
            }
            for c in chunks
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(
    doc_id: str,
    user_id: int = Depends(get_current_user_id),
) -> DeleteResponse:
    """Delete a document and all its data."""
    success = await doc_service.delete_document(user_id, doc_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DeleteResponse(message="Document deleted successfully")


async def count_for_user(self, owner_id: int, status: str | None = None) -> int:
        async with session_module.get_session() as session:
            repo = DocumentRepository(session)
            return await repo.count_for_user(owner_id, status)


DocumentService.count_for_user = count_for_user


async def get_chunks(self, doc_id: str, owner_id: int | None = None, *, limit: int | None = None, offset: int = 0):
    async with session_module.get_session() as session:
        repo = ChunkRepository(session)
        return await repo.search_by_document(doc_id, limit=limit or 50, offset=offset)


DocumentService.get_chunks = get_chunks


async def count_chunks(self, doc_id: str, owner_id: int | None = None) -> int:
    async with session_module.get_session() as session:
        repo = ChunkRepository(session)
        return await repo.count_by_document(doc_id)


DocumentService.count_chunks = count_chunks
