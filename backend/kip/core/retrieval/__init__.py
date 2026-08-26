"""Retrieval: the two axes, their fusion, and the registry that selects them.

Layout:

``keyword``
    The :class:`KeywordIndex` interface, ``KeywordHit``, and the shared BM25
    constants.
``bm25``
    Exact in-memory Okapi BM25. The reference implementation, and what the
    self-checks and the evaluation harness compare against.
``fts``
    SQLite FTS5. Persistent and shared across workers; the default.
``fusion``
    Reciprocal Rank Fusion and weighted score fusion, with per-retriever rank
    provenance preserved so a result can be explained.
``hybrid``
    :class:`HybridRetriever`, which runs both axes and reports what it did.

Reranking lives in :mod:`kip.core.rerank`, not here, because it needs passage
text that only the service layer has hydrated. See :mod:`kip.core.retrieval.hybrid`.
"""

from __future__ import annotations

from typing import Any, Callable

from kip.core.retrieval.bm25 import Bm25Index, bm25_idf, bm25_tf
from kip.core.retrieval.fusion import (
    DEFAULT_RRF_K,
    FUSION_METHODS,
    Contribution,
    FusedHit,
    dedupe_by_document,
    fuse,
    min_max,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from kip.core.retrieval.hybrid import (
    DENSE,
    KEYWORD,
    RETRIEVAL_MODES,
    HybridRetriever,
    RetrievalDiagnostics,
    RetrievalResult,
)
from kip.core.retrieval.keyword import (
    BM25_B,
    BM25_K1,
    Filters,
    KeywordDocument,
    KeywordHit,
    KeywordIndex,
    KeywordIndexError,
)

__all__ = [
    "BACKEND_NOTES",
    "BM25_B",
    "BM25_K1",
    "Bm25Index",
    "Contribution",
    "DEFAULT_RRF_K",
    "DENSE",
    "FUSION_METHODS",
    "Filters",
    "FusedHit",
    "HybridRetriever",
    "INDEXES",
    "KEYWORD",
    "KeywordDocument",
    "KeywordHit",
    "KeywordIndex",
    "KeywordIndexError",
    "RETRIEVAL_MODES",
    "RetrievalDiagnostics",
    "RetrievalResult",
    "SqliteFtsIndex",
    "bm25_idf",
    "bm25_tf",
    "dedupe_by_document",
    "describe_backends",
    "fuse",
    "get_keyword_index",
    "min_max",
    "reciprocal_rank_fusion",
    "weighted_fusion",
]


def __getattr__(name: str) -> Any:
    """Import ``SqliteFtsIndex`` lazily.

    Constructing it probes for the FTS5 module, so importing this package should
    not require an FTS5-capable SQLite build unless that backend is selected.
    """
    if name == "SqliteFtsIndex":
        from kip.core.retrieval.fts import SqliteFtsIndex

        return SqliteFtsIndex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _bm25(**kwargs: Any) -> KeywordIndex:
    return Bm25Index(
        k1=float(kwargs.get("k1") or BM25_K1),
        b=BM25_B if kwargs.get("b") is None else float(kwargs["b"]),
    )


def _fts5(**kwargs: Any) -> KeywordIndex:
    from kip.core.retrieval.fts import SqliteFtsIndex

    return SqliteFtsIndex(
        path=kwargs.get("path") or "./var/keyword.sqlite3",
        connection=kwargs.get("connection"),
    )


#: Backend name -> factory.
INDEXES: dict[str, Callable[..., KeywordIndex]] = {
    "bm25": _bm25,
    "fts5": _fts5,
}

#: Surfaced on the Settings screen. The FTS5 IDF caveat is stated in the product,
#: not buried in a docstring, because it affects ranking on small corpora.
BACKEND_NOTES: dict[str, str] = {
    "fts5": (
        "Persistent SQLite FTS5 index, shared by all workers. Uses SQLite's "
        "textbook BM25, which under-weights terms common to most documents."
    ),
    "bm25": (
        "Exact in-process BM25 with non-negative IDF. Rebuilt at startup and not "
        "shared between workers; the reference implementation for evaluation."
    ),
    "none": "Keyword retrieval disabled. Retrieval is semantic only.",
}


def get_keyword_index(settings: Any = None, **overrides: Any) -> KeywordIndex | None:
    """Build the configured keyword index, or ``None`` when disabled.

    ``KEYWORD_INDEX=none`` is a supported configuration -- it is how a deployment
    turns hybrid retrieval into pure semantic retrieval -- so this returns
    ``None`` rather than raising. Every other unrecognised value raises, because a
    typo that silently halved retrieval recall would be invisible.

    >>> class S:
    ...     keyword_index = "bm25"
    >>> get_keyword_index(S()).name
    'bm25'
    >>> class Off:
    ...     keyword_index = "none"
    >>> get_keyword_index(Off()) is None
    True
    >>> class Bad:
    ...     keyword_index = "lucene"
    >>> get_keyword_index(Bad())
    Traceback (most recent call last):
        ...
    kip.core.retrieval.keyword.KeywordIndexError: Unknown KEYWORD_INDEX 'lucene'. Available backends: bm25, fts5, none.
    """
    if settings is None:
        from kip.config import get_settings

        settings = get_settings()

    name = str(overrides.pop("backend", None) or getattr(settings, "keyword_index", "fts5"))
    name = name.strip().lower()
    if name in {"none", "off", "disabled", ""}:
        return None

    factory = INDEXES.get(name)
    if factory is None:
        raise KeywordIndexError(
            f"Unknown KEYWORD_INDEX {name!r}. Available backends: "
            + ", ".join([*sorted(INDEXES), "none"])
            + "."
        )

    kwargs: dict[str, Any] = {}
    if name == "fts5":
        configured = getattr(settings, "keyword_index_path", None)
        resolver = getattr(settings, "resolve_path", None)
        kwargs["path"] = (
            resolver(configured) if callable(resolver) and configured else configured
        )
    kwargs.update(overrides)
    return factory(**kwargs)


def describe_backends() -> list[dict[str, str]]:
    """Keyword backend catalogue for the Settings screen."""
    return [
        {"name": name, "note": BACKEND_NOTES.get(name, "")}
        for name in ("fts5", "bm25", "none")
    ]
