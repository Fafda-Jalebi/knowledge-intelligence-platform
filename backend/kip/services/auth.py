"""Authentication service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from kip.config import get_settings
from kip.db.repositories import UserRepository
from kip.db.session import get_session
from kip.security.tokens import TokenError, decode, encode
from kip.security.passwords import validate_password_strength, ValidationError


@dataclass(slots=True)
class TokenPair:
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 0


class AuthService:
    """Authentication and token management."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def register(self, email: str, password: str) -> TokenPair:
        """Register a new user and return tokens."""
        validate_password_strength(password)
        async with get_session() as session:
            repo = UserRepository(session)
            existing = await repo.get_by_email(email)
            if existing:
                raise ValidationError("An account with this email already exists.")
            user = await repo.create(email, password)
            return self._create_tokens(user.id)

    async def login(self, email: str, password: str) -> TokenPair:
        """Authenticate user and return tokens."""
        async with get_session() as session:
            repo = UserRepository(session)
            user = await repo.authenticate(email, password)
            if not user:
                raise ValidationError("Invalid email or password.")
            if not user.is_active:
                raise ValidationError("This account has been deactivated.")
            return self._create_tokens(user.id)

    async def get_current_user(self, token: str) -> int:
        """Validate token and return user ID."""
        try:
            claims = decode(token, self._settings.jwt_secret, issuer="kip")
        except TokenError as exc:
            raise ValidationError(str(exc)) from exc
        sub = claims.get("sub")
        if not sub:
            raise ValidationError("Invalid token: missing subject.")
        return int(sub)

    async def change_password(self, user_id: int, current_password: str, new_password: str) -> bool:
        """Change user's password after verifying current one."""
        validate_password_strength(new_password)
        async with get_session() as session:
            repo = UserRepository(session)
            user = await repo.get_by_id(user_id)
            if not user:
                raise ValidationError("User not found.")
            from kip.security.passwords import verify_password
            if not verify_password(current_password, user.password_hash):
                raise ValidationError("Current password is incorrect.")
            return await repo.update_password(user_id, new_password)

    def _create_tokens(self, user_id: int) -> TokenPair:
        expires_minutes = self._settings.jwt_expire_minutes
        access = encode(
            {"sub": str(user_id), "role": "user"},
            self._settings.jwt_secret,
            expires_in=expires_minutes * 60,
            issuer="kip",
        )
        return TokenPair(access_token=access, expires_in=expires_minutes * 60)
