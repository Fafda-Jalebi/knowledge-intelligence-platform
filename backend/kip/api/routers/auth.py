"""Authentication endpoints."""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field

from kip.config import get_settings
from kip.errors import ValidationError
from kip.security.tokens import TokenError, decode
from kip.services.auth import AuthService

router = APIRouter()
security = HTTPBearer(auto_error=False)
auth_service = AuthService()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: int
    email: str
    is_active: bool
    is_superuser: bool


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest) -> TokenResponse:
    """Register a new user."""
    try:
        tokens = await auth_service.register(request.email, request.password)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return TokenResponse(**dataclasses.asdict(tokens))


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    """Authenticate and return access token."""
    try:
        tokens = await auth_service.login(request.email, request.password)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return TokenResponse(**dataclasses.asdict(tokens))


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> int:
    """Extract and validate user ID from bearer token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    settings = get_settings()
    try:
        claims = decode(credentials.credentials, settings.jwt_secret, issuer="kip")
    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    sub = claims.get("sub")
    if not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return int(sub)


@router.get("/me", response_model=UserResponse)
async def get_me(user_id: int = Depends(get_current_user_id)) -> UserResponse:
    """Get current user profile."""
    from kip.db.repositories import UserRepository
    from kip.db.session import get_session

    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        return UserResponse(
            id=user.id,
            email=user.email,
            is_active=user.is_active,
            is_superuser=user.is_superuser,
        )


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=10, max_length=128)


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    user_id: int = Depends(get_current_user_id),
) -> dict:
    """Change password."""
    try:
        await auth_service.change_password(user_id, request.current_password, request.new_password)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"message": "Password changed successfully"}
