"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from kip.config import get_settings
from kip.db.session import get_session as db_session

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Basic health check."""
    return {"status": "ok", "service": "kip"}


@router.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(db_session)) -> dict:
    """Readiness check including database connectivity."""
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    settings = get_settings()
    return {
        "status": "ready" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "environment": settings.app_env,
        "vector_store": settings.vector_store,
        "embedding_provider": settings.embedding_provider,
        "llm_provider": settings.llm_provider,
    }


@router.get("/health/live")
async def liveness_check() -> dict:
    """Liveness check for container orchestration."""
    return {"status": "alive"}
