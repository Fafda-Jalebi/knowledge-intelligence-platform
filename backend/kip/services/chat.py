"""Chat/RAG service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from kip.config import get_settings
from kip.core import RagPipeline, RagPipeline
from kip.core.embeddings import get_embedder
from kip.core.llm import get_llm_client
from kip.core.rerank import get_reranker
from kip.core.retrieval import HybridRetriever, get_keyword_index
from kip.core.vectorstore import get_vector_store
from kip.db.repositories import ChunkRepository, ConversationRepository
from kip.db.session import get_session
from kip.services.documents import DocumentService


@dataclass(slots=True)
class ChatAnswer:
    answer: str
    refused: bool
    refusal_reason: str
    explanation: str
    citations: list[dict]
    context: dict
    groundedness: float
    usage: dict
    stages: list[dict]
    total_ms: float
    conversation_id: int | None = None


class ChatService:
    """Chat/RAG orchestration."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._embedder = get_embedder(self._settings)
        self._vector_store = get_vector_store(self._settings)
        self._keyword_index = get_keyword_index(self._settings)
        self._llm = get_llm_client(self._settings)
        self._reranker = get_reranker(self._settings)
        self._doc_service = DocumentService()

        self._vector_store.ensure_collection(self._embedder.spec)

        self._pipeline = RagPipeline.from_settings(
            self._settings,
            retriever=HybridRetriever.from_settings(
                self._settings,
                embedder=self._embedder,
                vector_store=self._vector_store,
                keyword_index=self._keyword_index,
            ),
            llm=self._llm,
            hydrate=lambda ids: self._hydrate(ids),
            titles=self._titles,
            reranker=self._reranker,
        )

    async def _hydrate(self, chunk_ids: Sequence[str]) -> dict[str, str]:
        """Hydrate chunk IDs to text (async)."""
        return await self._doc_service.get_chunk_texts(list(chunk_ids))

    async def _titles(self, doc_ids: Sequence[str]) -> dict[str, str]:
        return await self._doc_service.get_titles_for_documents(list(doc_ids))

    async def ask(
        self,
        user_id: int,
        question: str,
        *,
        conversation_id: int | None = None,
        document_ids: Sequence[str] | None = None,
        mode: str | None = None,
        min_score: float | None = None,
    ) -> ChatAnswer:
        """Answer a question, optionally within a conversation."""
        # Get history if conversation specified
        history = []
        conv_id = conversation_id
        if conversation_id:
            async with get_session() as session:
                repo = ConversationRepository(session)
                conv = await repo.get_with_messages(conversation_id, user_id, limit=self._settings.rerank_top_n * 4)
                if not conv:
                    raise ValueError("Conversation not found or access denied.")
                history = [
                    {"role": m.role, "content": m.content}
                    for m in conv.messages
                ]
        else:
            # Create a new conversation
            async with get_session() as session:
                repo = ConversationRepository(session)
                conv = await repo.create(user_id, title=question[:80])
                conv_id = conv.id

        # Run pipeline
        from kip.core.llm.base import Message
        messages = [Message(**m) for m in history]

        answer = await self._pipeline.answer(
            question,
            filters={"user_id": [user_id]},
            document_ids=document_ids,
            history=messages,
            mode=mode,
            min_score=min_score,
        )

        # Persist messages
        async with get_session() as session:
            repo = ConversationRepository(session)
            await repo.add_message(
                conv_id,
                "user",
                question,
            )
            await repo.add_message(
                conv_id,
                "assistant",
                answer.text,
                citations=str([c.to_dict() for c in answer.citations]),
                retrieval_diagnostics=str(answer.retrieval),
                groundedness=str(answer.groundedness),
                refused=answer.refused,
                refusal_reason=answer.refusal_reason,
                model=answer.model,
                provider=answer.provider,
                usage_prompt_tokens=answer.usage.prompt_tokens,
                usage_completion_tokens=answer.usage.completion_tokens,
                total_ms=int(answer.total_ms),
            )

        return ChatAnswer(
            answer=answer.text,
            refused=answer.refused,
            refusal_reason=answer.refusal_reason,
            explanation=answer.explanation,
            citations=[c.to_dict() for c in answer.citations],
            context=answer.context.to_dict(),
            groundedness=answer.groundedness,
            usage=answer.usage.to_dict(),
            stages=[s.to_dict() for s in answer.stages],
            total_ms=answer.total_ms,
            conversation_id=conv_id,
        )

    async def list_conversations(
        self,
        user_id: int,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[dict]:
        async with get_session() as session:
            repo = ConversationRepository(session)
            convs = await repo.list_for_user(user_id, limit=limit, offset=offset)
            return [
                {
                    "id": c.id,
                    "title": c.title,
                    "message_count": c.message_count,
                    "created_at": c.created_at,
                    "updated_at": c.updated_at,
                }
                for c in convs
            ]

    async def get_conversation(
        self,
        user_id: int,
        conversation_id: int,
    ) -> dict | None:
        async with get_session() as session:
            repo = ConversationRepository(session)
            conv = await repo.get_with_messages(conversation_id, user_id)
            if not conv:
                return None
            return {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat() if conv.created_at else "",
                "updated_at": conv.updated_at.isoformat() if conv.updated_at else "",
                "messages": [
                    {
                        "id": m.id,
                        "role": m.role,
                        "content": m.content,
                        "citations": m.citations,
                        "refused": m.refused,
                        "refusal_reason": m.refusal_reason,
                        "model": m.model,
                        "provider": m.provider,
                        "usage_prompt_tokens": m.usage_prompt_tokens,
                        "usage_completion_tokens": m.usage_completion_tokens,
                        "total_ms": m.total_ms,
                        "created_at": m.created_at.isoformat() if m.created_at else "",
                    }
                    for m in conv.messages
                ],
            }

    async def delete_conversation(self, user_id: int, conversation_id: int) -> bool:
        async with get_session() as session:
            repo = ConversationRepository(session)
            return await repo.delete(conversation_id, user_id)

    async def update_conversation_title(self, user_id: int, conversation_id: int, title: str) -> bool:
        async with get_session() as session:
            repo = ConversationRepository(session)
            return await repo.update_title(conversation_id, user_id, title)

    def describe_pipeline(self) -> dict:
        """Return pipeline configuration for the Settings screen."""
        return self._pipeline.describe()
