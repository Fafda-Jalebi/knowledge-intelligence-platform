"""Chat/RAG endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional

from kip.api.routers.auth import get_current_user_id
from kip.services.chat import ChatService

router = APIRouter()
chat_service = ChatService()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: Optional[int] = None
    document_ids: list[str] = []
    mode: Optional[str] = None
    min_score: Optional[float] = None


class CitationResponse(BaseModel):
    marker: int
    chunk_id: str
    document_id: str
    document_label: str
    label: str
    text: str
    page_start: int | None
    page_end: int | None
    section_path: list[str]
    score: float
    truncated: bool
    count: int


class ContextResponse(BaseModel):
    passages: list[dict]
    token_count: int
    budget: int
    document_ids: list[str]


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str
    explanation: str
    citations: list[CitationResponse]
    context: ContextResponse
    groundedness: float
    usage: dict
    stages: list[dict]
    total_ms: float
    conversation_id: int | None = None


class ConversationResponse(BaseModel):
    id: int
    title: str | None
    message_count: int
    created_at: str
    updated_at: str


class ConversationDetailResponse(BaseModel):
    id: int
    title: str | None
    created_at: str
    updated_at: str
    messages: list[dict]


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    user_id: int = Depends(get_current_user_id),
) -> AskResponse:
    """Ask a question and get a grounded answer."""
    try:
        result = await chat_service.ask(
            user_id,
            request.question,
            conversation_id=request.conversation_id,
            document_ids=request.document_ids or None,
            mode=request.mode,
            min_score=request.min_score,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Chat failed: {exc}") from exc

    return AskResponse(
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        explanation=result.explanation,
        citations=[CitationResponse(**c) for c in result.citations],
        context=ContextResponse(**result.context),
        groundedness=result.groundedness,
        usage=result.usage,
        stages=result.stages,
        total_ms=result.total_ms,
        conversation_id=result.conversation_id,
    )


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    user_id: int = Depends(get_current_user_id),
) -> list[ConversationResponse]:
    """List user's conversations."""
    convs = await chat_service.list_conversations(user_id, limit=limit, offset=offset)
    return [ConversationResponse(**c) for c in convs]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: int,
    user_id: int = Depends(get_current_user_id),
) -> ConversationDetailResponse:
    """Get a conversation with its messages."""
    conv = await chat_service.get_conversation(user_id, conversation_id)
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return ConversationDetailResponse(**conv)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    user_id: int = Depends(get_current_user_id),
) -> dict:
    """Delete a conversation."""
    success = await chat_service.delete_conversation(user_id, conversation_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return {"message": "Conversation deleted"}


@router.patch("/conversations/{conversation_id}")
async def update_conversation_title(
    conversation_id: int,
    title: str,
    user_id: int = Depends(get_current_user_id),
) -> dict:
    """Update conversation title."""
    success = await chat_service.update_conversation_title(user_id, conversation_id, title)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return {"message": "Title updated"}


@router.get("/pipeline/config")
async def get_pipeline_config() -> dict:
    """Get the active pipeline configuration."""
    return chat_service.describe_pipeline()
