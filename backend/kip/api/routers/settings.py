"""Settings endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from kip.api.routers.auth import get_current_user_id
from kip.config import Settings
from kip.config import get_settings as _get_settings

router = APIRouter()


async def get_settings_dep() -> Settings:
    return _get_settings()


@router.get("")
async def get_settings(
    user_id: int = Depends(get_current_user_id),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    """Get current settings (non-sensitive)."""
    return settings.redacted()


@router.get("/embedding-providers")
async def get_embedding_providers() -> list[dict]:
    """Get available embedding providers."""
    from kip.core.embeddings import describe_providers
    return describe_providers()


@router.get("/llm-providers")
async def get_llm_providers() -> list[dict]:
    """Get available LLM providers."""
    from kip.core.llm import describe_llm_providers
    return describe_llm_providers()


@router.get("/rerankers")
async def get_rerankers() -> list[dict]:
    """Get available rerankers."""
    from kip.core.rerank import describe_rerankers
    return describe_rerankers()


@router.get("/vector-stores")
async def get_vector_stores() -> list[dict]:
    """Get available vector stores."""
    from kip.core.vectorstore import describe_backends
    return describe_backends()


@router.get("/keyword-indexes")
async def get_keyword_indexes() -> list[dict]:
    """Get available keyword indexes."""
    from kip.core.retrieval import describe_backends
    return describe_backends()


@router.get("/grounding")
async def get_grounding_config(settings: Settings = Depends(get_settings_dep)) -> dict:
    """Get grounding thresholds."""
    from kip.core.rag.grounding import describe_thresholds
    return describe_thresholds(settings)
