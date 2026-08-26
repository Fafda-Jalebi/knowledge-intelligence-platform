"""Conversation and message repository."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from kip.db.session import Conversation, Message


@dataclass(slots=True)
class ConversationSummary:
    """Lightweight conversation info for lists."""
    id: int
    title: str | None
    message_count: int
    created_at: str
    updated_at: str


class ConversationRepository:
    """Conversation and message data access."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- Conversations ---------------------------------------------------------

    async def create(self, owner_id: int, title: str | None = None) -> Conversation:
        """Create a new conversation."""
        conv = Conversation(owner_id=owner_id, title=title)
        self._session.add(conv)
        await self._session.flush()
        await self._session.refresh(conv)
        return conv

    async def get_by_id(self, conv_id: int, owner_id: int | None = None) -> Optional[Conversation]:
        """Get conversation by ID."""
        stmt = select(Conversation).where(Conversation.id == conv_id)
        if owner_id is not None:
            stmt = stmt.where(Conversation.owner_id == owner_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_with_messages(
        self,
        conv_id: int,
        owner_id: int | None = None,
        *,
        limit: int | None = None,
    ) -> Optional[Conversation]:
        """Get conversation with messages loaded."""
        stmt = select(Conversation).options(selectinload(Conversation.messages)).where(Conversation.id == conv_id)
        if owner_id is not None:
            stmt = stmt.where(Conversation.owner_id == owner_id)
        result = await self._session.execute(stmt)
        conv = result.scalar_one_or_none()
        if conv and limit and len(conv.messages) > limit:
            conv.messages = conv.messages[-limit:]
        return conv

    async def list_for_user(
        self,
        owner_id: int,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[ConversationSummary]:
        """List conversations for a user."""
        stmt = (
            select(
                Conversation.id,
                Conversation.title,
                func.count(Message.id).label("message_count"),
                Conversation.created_at,
                Conversation.updated_at,
            )
            .outerjoin(Message, Message.conversation_id == Conversation.id)
            .where(Conversation.owner_id == owner_id)
            .group_by(Conversation.id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return [
            ConversationSummary(
                id=row.id,
                title=row.title,
                message_count=row.message_count,
                created_at=row.created_at.isoformat() if row.created_at else "",
                updated_at=row.updated_at.isoformat() if row.updated_at else "",
            )
            for row in result.all()
        ]

    async def count_for_user(self, owner_id: int) -> int:
        """Count conversations for a user."""
        stmt = select(func.count(Conversation.id)).where(Conversation.owner_id == owner_id)
        result = await self._session.execute(stmt)
        return result.scalar_one()

    async def update_title(self, conv_id: int, owner_id: int, title: str) -> bool:
        """Update conversation title."""
        conv = await self.get_by_id(conv_id, owner_id)
        if not conv:
            return False
        conv.title = title
        await self._session.flush()
        return True

    async def delete(self, conv_id: int, owner_id: int) -> bool:
        """Delete a conversation and its messages."""
        conv = await self.get_by_id(conv_id, owner_id)
        if not conv:
            return False
        await self._session.delete(conv)
        await self._session.flush()
        return True

    # -- Messages --------------------------------------------------------------

    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        *,
        citations: str | None = None,
        retrieval_diagnostics: str | None = None,
        groundedness: str | None = None,
        refused: bool = False,
        refusal_reason: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        usage_prompt_tokens: int | None = None,
        usage_completion_tokens: int | None = None,
        total_ms: int | None = None,
    ) -> Message:
        """Add a message to a conversation."""
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citations,
            retrieval_diagnostics=retrieval_diagnostics,
            groundedness=groundedness,
            refused=refused,
            refusal_reason=refusal_reason,
            model=model,
            provider=provider,
            usage_prompt_tokens=usage_prompt_tokens,
            usage_completion_tokens=usage_completion_tokens,
            total_ms=total_ms,
        )
        self._session.add(msg)
        await self._session.flush()
        await self._session.refresh(msg)
        return msg

    async def get_messages(
        self,
        conversation_id: int,
        owner_id: int | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Message]:
        """Get messages for a conversation."""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        if owner_id is not None:
            stmt = stmt.join(Conversation).where(Conversation.owner_id == owner_id)
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def get_recent_messages(
        self,
        conversation_id: int,
        owner_id: int | None = None,
        *,
        limit: int = 20,
    ) -> Sequence[Message]:
        """Get the most recent messages for context."""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        if owner_id is not None:
            stmt = stmt.join(Conversation).where(Conversation.owner_id == owner_id)
        result = await self._session.execute(stmt)
        messages = list(result.scalars().all())
        messages.reverse()
        return messages

    async def delete_messages_before(self, conversation_id: int, before_id: int) -> int:
        """Delete messages older than a given ID."""
        stmt = delete(Message).where(
            Message.conversation_id == conversation_id,
            Message.id < before_id,
        )
        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount or 0
