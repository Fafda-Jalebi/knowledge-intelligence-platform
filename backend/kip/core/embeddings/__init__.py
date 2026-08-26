"""Embedding provider registry.

``EMBEDDING_PROVIDER`` selects a backend; nothing else in the platform knows
which one is active. Registry entries are *factories*, not instances, so the
heavy backends are never constructed (and never imported) unless selected.

Add a provider by writing a class that subclasses
:class:`~kip.core.embeddings.base.Embedder` and adding one line to
:data:`PROVIDERS`.
"""

from __future__ import annotations

from typing import Any, Callable

from kip.core.embeddings.base import (
    Embedder,
    EmbeddingError,
    EmbeddingSpec,
    VECTOR_DTYPE,
    band,
    cosine_similarity,
    from_bytes,
    l2_normalise,
    similarity_to_confidence,
    sublinear_tf,
    to_bytes,
)
from kip.core.embeddings.hashing import HashingEmbedder

__all__ = [
    "Embedder",
    "EmbeddingError",
    "EmbeddingSpec",
    "HashingEmbedder",
    "VECTOR_DTYPE",
    "band",
    "cosine_similarity",
    "describe_providers",
    "from_bytes",
    "get_embedder",
    "l2_normalise",
    "similarity_to_confidence",
    "sublinear_tf",
    "to_bytes",
    "PROVIDERS",
]


def _hashing(**kwargs: Any) -> Embedder:
    from kip.core.embeddings.hashing import DEFAULT_DIM

    return HashingEmbedder(dim=int(kwargs.get("dim") or DEFAULT_DIM))


def _sentence_transformers(**kwargs: Any) -> Embedder:
    from kip.core.embeddings.providers import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(
        model=str(kwargs.get("model") or "sentence-transformers/all-MiniLM-L6-v2"),
        dim=kwargs.get("dim"),
        batch_size=int(kwargs.get("batch_size") or 32),
        device=kwargs.get("device"),
    )


def _openai(**kwargs: Any) -> Embedder:
    from kip.core.embeddings.providers import OpenAIEmbedder

    return OpenAIEmbedder(
        model=str(kwargs.get("model") or "text-embedding-3-small"),
        api_key=str(kwargs.get("api_key") or ""),
        base_url=str(kwargs.get("base_url") or "https://api.openai.com/v1"),
        dim=kwargs.get("dim"),
        batch_size=int(kwargs.get("batch_size") or 64),
        timeout=float(kwargs.get("timeout") or 60.0),
    )


def _ollama(**kwargs: Any) -> Embedder:
    from kip.core.embeddings.providers import OllamaEmbedder

    return OllamaEmbedder(
        model=str(kwargs.get("model") or "nomic-embed-text"),
        base_url=str(kwargs.get("base_url") or "http://localhost:11434"),
        dim=kwargs.get("dim"),
        batch_size=int(kwargs.get("batch_size") or 16),
        timeout=float(kwargs.get("timeout") or 120.0),
    )


#: Provider name -> factory. ``hashing`` is first because it is the default and
#: the only one guaranteed to work with no setup.
PROVIDERS: dict[str, Callable[..., Embedder]] = {
    "hashing": _hashing,
    "sentence-transformers": _sentence_transformers,
    "sentence_transformers": _sentence_transformers,  # tolerate the underscore
    "openai": _openai,
    "ollama": _ollama,
}

#: Human-readable notes surfaced on the Settings screen. Being explicit that the
#: default is lexical rather than semantic is a deliberate anti-overclaim
#: measure, not a disclaimer added after the fact.
PROVIDER_NOTES: dict[str, str] = {
    "hashing": (
        "Deterministic lexical embeddings. No download, no API key, fully "
        "offline and reproducible. Matches wording, not meaning."
    ),
    "sentence-transformers": (
        "Local semantic embeddings. Requires the sentence-transformers package "
        "and a one-time model download. No data leaves the machine."
    ),
    "openai": (
        "Hosted semantic embeddings via an OpenAI-compatible endpoint. Requires "
        "an API key; passage text is sent to the provider."
    ),
    "ollama": (
        "Semantic embeddings from a local Ollama server. No API key and no data "
        "egress; requires the model to be pulled."
    ),
}


def get_embedder(settings: Any = None, **overrides: Any) -> Embedder:
    """Build the configured :class:`Embedder`.

    Accepts a :class:`kip.config.Settings` (duck-typed, so tests can pass a
    stub) and keyword overrides. Unknown provider names fail with the list of
    valid options rather than silently falling back -- a typo in
    ``EMBEDDING_PROVIDER`` that quietly downgrades retrieval quality is far
    worse than a startup error.
    """
    if settings is None:
        from kip.config import get_settings

        settings = get_settings()

    name = str(overrides.pop("provider", None) or getattr(settings, "embedding_provider", "hashing"))
    name = name.strip().lower()
    factory = PROVIDERS.get(name)
    if factory is None:
        raise EmbeddingError(
            f"Unknown EMBEDDING_PROVIDER {name!r}. Available providers: "
            + ", ".join(sorted(set(PROVIDERS) - {"sentence_transformers"}))
            + "."
        )

    kwargs: dict[str, Any] = {
        "model": getattr(settings, "embedding_model", None),
        # 0 / unset means "let the provider use its native width"; see the
        # EMBEDDING_DIM comment in kip.config.
        "dim": getattr(settings, "embedding_dim", None) or None,
        "batch_size": getattr(settings, "embedding_batch_size", 32),
    }
    if name == "openai":
        kwargs["api_key"] = getattr(settings, "openai_api_key", "")
        kwargs["base_url"] = getattr(settings, "openai_base_url", None)
        kwargs["timeout"] = getattr(settings, "llm_timeout_seconds", 60)
    if name == "ollama":
        kwargs["base_url"] = getattr(settings, "ollama_base_url", None)

    # The hashing provider owns its model name; carrying over an unrelated
    # EMBEDDING_MODEL would produce a misleading fingerprint.
    if name == "hashing":
        kwargs.pop("model", None)

    kwargs.update(overrides)
    return factory(**kwargs)


def describe_providers() -> list[dict[str, str]]:
    """Provider catalogue for the Settings screen."""
    return [
        {"name": name, "note": PROVIDER_NOTES.get(name, "")}
        for name in ("hashing", "sentence-transformers", "openai", "ollama")
    ]
