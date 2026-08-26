"""User repository."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kip.db.session import User
from kip.security.passwords import hash_password, verify_password


class UserRepository:
    """User data access."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, email: str, password: str) -> User:
        """Create a new user."""
        user = User(
            email=email.lower().strip(),
            password_hash=hash_password(password),
        )
        self._session.add(user)
        await self._session.flush()
        await self._session.refresh(user)
        return user

    async def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        result = await self._session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self._session.execute(select(User).where(User.email == email.lower().strip()))
        return result.scalar_one_or_none()

    async def authenticate(self, email: str, password: str) -> Optional[User]:
        """Verify credentials and return user if valid."""
        user = await self.get_by_email(email)
        if user and verify_password(password, user.password_hash):
            return user
        return None

    async def update_password(self, user_id: int, new_password: str) -> bool:
        """Update user's password."""
        user = await self.get_by_id(user_id)
        if not user:
            return False
        user.password_hash = hash_password(new_password)
        await self._session.flush()
        return True

    async def deactivate(self, user_id: int) -> bool:
        """Deactivate a user."""
        user = await self.get_by_id(user_id)
        if not user:
            return False
        user.is_active = False
        await self._session.flush()
        return True
