"""Vector store registry.

``VECTOR_STORE`` picks a backend; the retrieval layer only ever sees the
:class:`~kip.core.vectorstore.base.VectorStore` interface. As with the embedding
registry, entries are factories so ``qdrant-client`` is imported only when
``VECTOR_STORE=qdrant``.

Choosing between them:

===========  =========================  =====================================
Backend      Good for                   Trade-off
===========  =========================  =====================================
``memory``   tests, evaluation runs     nothing survives the process
``sqlite``   the default; local, demo   exact search, single file, ~100k cap
``qdrant``   production, large corpora  needs a server; approximate (HNSW)
===========  =========================  =====================================

All three are exercised by the self-checks (Qdrant only when a server is
reachable), and ``memory`` is the reference ranking the others must reproduce.
"""

from __future__ import annotations

from typing import Any, Callable

from kip.core.vectorstore.base import (
    REQUIRED_PAYLOAD_KEYS,
    Filters,
    SearchHit,
    VectorRecord,
    VectorStore,
    VectorStoreError,
    VectorStoreMismatch,
    normalise_filters,
    payload_matches,
    records_from_chunks,
)
from kip.core.vectorstore.memory import MemoryVectorStore
from kip.core.vectorstore.sqlite_store import SqliteVectorStore

__all__ = [
    "BACKEND_NOTES",
    "Filters",
    "MemoryVectorStore",
    "REQUIRED_PAYLOAD_KEYS",
    "STORES",
    "SearchHit",
    "SqliteVectorStore",
    "VectorRecord",
    "VectorStore",
    "VectorStoreError",
    "VectorStoreMismatch",
    "describe_backends",
    "get_vector_store",
    "normalise_filters",
    "payload_matches",
    "records_from_chunks",
]


def _memory(**kwargs: Any) -> VectorStore:
    return MemoryVectorStore(dim=kwargs.get("dim"))


def _sqlite(**kwargs: Any) -> VectorStore:
    return SqliteVectorStore(
        path=kwargs.get("path") or "./var/vectors.sqlite3",
        dim=kwargs.get("dim"),
    )


def _qdrant(**kwargs: Any) -> VectorStore:
    from kip.core.vectorstore.qdrant import QdrantVectorStore

    return QdrantVectorStore(
        url=str(kwargs.get("url") or "http://localhost:6333"),
        api_key=kwargs.get("api_key") or None,
        collection=str(kwargs.get("collection") or "kip_chunks"),
        dim=kwargs.get("dim"),
        timeout=float(kwargs.get("timeout") or 30.0),
    )


#: Backend name -> factory.
STORES: dict[str, Callable[..., VectorStore]] = {
    "memory": _memory,
    "sqlite": _sqlite,
    "qdrant": _qdrant,
}

#: Surfaced on the Settings screen so the operational trade-off is visible in
#: the product, not only in the README.
BACKEND_NOTES: dict[str, str] = {
    "memory": "In-process and exact. Nothing is persisted; intended for tests and evaluation.",
    "sqlite": (
        "Persistent single-file index with exact search. No extra services. "
        "Comfortable to roughly 100k chunks."
    ),
    "qdrant": (
        "Dedicated vector database with an HNSW index and server-side filtering. "
        "Approximate search; the right choice above ~100k chunks."
    ),
}


def get_vector_store(settings: Any = None, **overrides: Any) -> VectorStore:
    """Build the configured :class:`VectorStore`.

    An unknown backend name raises rather than falling back, for the same reason
    the embedding registry does: a typo that silently switches to an in-memory
    index would look like a working system that forgets every document on
    restart.
    """
    if settings is None:
        from kip.config import get_settings

        settings = get_settings()

    name = str(overrides.pop("backend", None) or getattr(settings, "vector_store", "sqlite"))
    name = name.strip().lower()
    factory = STORES.get(name)
    if factory is None:
        raise VectorStoreError(
            f"Unknown VECTOR_STORE {name!r}. Available backends: "
            + ", ".join(sorted(STORES))
            + "."
        )

    # ``dim`` is only a hint here: every store learns its authoritative width
    # from the embedding spec passed to ``ensure_collection``. 0/unset is the
    # normal case.
    kwargs: dict[str, Any] = {"dim": getattr(settings, "embedding_dim", None) or None}
    if name == "sqlite":
        configured = getattr(settings, "vector_store_path", None)
        resolver = getattr(settings, "resolve_path", None)
        kwargs["path"] = (
            resolver(configured) if callable(resolver) and configured else configured
        )
    if name == "qdrant":
        kwargs["url"] = getattr(settings, "qdrant_url", None)
        kwargs["api_key"] = getattr(settings, "qdrant_api_key", None)
        kwargs["collection"] = getattr(settings, "qdrant_collection", None)

    kwargs.update(overrides)
    return factory(**kwargs)


def describe_backends() -> list[dict[str, str]]:
    """Backend catalogue for the Settings screen."""
    return [
        {"name": name, "note": BACKEND_NOTES.get(name, "")}
        for name in ("sqlite", "qdrant", "memory")
    ]
