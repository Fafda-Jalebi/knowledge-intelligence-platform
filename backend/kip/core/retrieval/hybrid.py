"""Hybrid retriever -- runs both axes, fuses, and reports what it did.

Pipeline position:

.. code-block:: text

    query
      |-- embed  --> vector store  --> dense hits    (cosine, [-1, 1])
      |-- stem   --> keyword index --> keyword hits  (BM25, unbounded)
                            |
                            v
                          fuse (RRF by default)
                            |
                            v
                     fused candidates  ->  [service hydrates text]
                                        ->  rerank  ->  context  ->  LLM

Reranking is intentionally *not* performed here. A reranker scores
``(query, passage text)`` pairs, and passage text lives in the relational
database -- the vector payload deliberately does not duplicate it, so that a
citation can never quote text the document no longer contains. Retrieval
therefore returns chunk ids with provenance; the service layer hydrates the text
and then reranks. Putting the reranker here would have forced the vector payload
to carry passage bodies, which is the exact coupling the storage design avoids.

Two failure modes are handled differently on purpose. A **missing or empty
keyword index** degrades to dense-only and says so in
:attr:`RetrievalResult.diagnostics`, because a partial answer with a visible
caveat beats an error. An **embedding fingerprint mismatch** propagates as an
exception, because the alternative is ranking a query against vectors from a
different model and presenting the result as an answer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from kip.core.embeddings.base import Embedder
from kip.core.retrieval.fusion import (
    DEFAULT_RRF_K,
    FUSION_METHODS,
    FusedHit,
    fuse,
)
from kip.core.retrieval.keyword import Filters, KeywordIndex
from kip.core.vectorstore.base import VectorStore

#: Retriever names used as fusion keys, diagnostic labels and API field names.
DENSE = "dense"
KEYWORD = "keyword"

RETRIEVAL_MODES = ("hybrid", "dense", "keyword")


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    """Everything needed to explain, debug or evaluate one retrieval.

    Surfaced on the Search screen and recorded by the evaluation harness. The
    per-axis candidate counts and timings are what make claims like "hybrid
    improves recall" checkable instead of decorative.
    """

    mode: str
    fusion: str
    dense_candidates: int = 0
    keyword_candidates: int = 0
    fused_candidates: int = 0
    returned: int = 0
    dense_ms: float = 0.0
    keyword_ms: float = 0.0
    fusion_ms: float = 0.0
    total_ms: float = 0.0
    embedding_fingerprint: str | None = None
    keyword_backend: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "fusion": self.fusion,
            "dense_candidates": self.dense_candidates,
            "keyword_candidates": self.keyword_candidates,
            "fused_candidates": self.fused_candidates,
            "returned": self.returned,
            "dense_ms": round(self.dense_ms, 2),
            "keyword_ms": round(self.keyword_ms, 2),
            "fusion_ms": round(self.fusion_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "embedding_fingerprint": self.embedding_fingerprint,
            "keyword_backend": self.keyword_backend,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Fused candidates plus the diagnostics describing how they were found."""

    query: str
    hits: tuple[FusedHit, ...] = ()
    diagnostics: RetrievalDiagnostics = field(
        default_factory=lambda: RetrievalDiagnostics(mode="hybrid", fusion="rrf")
    )

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self):
        return iter(self.hits)

    def __bool__(self) -> bool:
        return bool(self.hits)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(hit.id for hit in self.hits)

    @property
    def document_ids(self) -> tuple[str, ...]:
        """Distinct source documents, in first-appearance order."""
        seen: dict[str, None] = {}
        for hit in self.hits:
            if hit.document_id:
                seen.setdefault(hit.document_id, None)
        return tuple(seen)

    @property
    def top_dense_score(self) -> float | None:
        """Best raw cosine similarity across the results.

        The grounding check needs an uncalibrated similarity, not a fused score:
        a fused score says "this was the best of what we found", which is true even
        when everything found was irrelevant. See ``GROUNDING_MIN_SCORE``.
        """
        scores = [
            score
            for hit in self.hits
            if (score := hit.best_score(DENSE)) is not None
        ]
        return max(scores) if scores else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "hits": [hit.to_dict() for hit in self.hits],
            "diagnostics": self.diagnostics.to_dict(),
        }


class HybridRetriever:
    """Dense + keyword retrieval with configurable fusion.

    >>> from kip.core.embeddings import HashingEmbedder
    >>> from kip.core.vectorstore import MemoryVectorStore, records_from_chunks
    >>> from kip.core.retrieval.bm25 import Bm25Index
    >>> from kip.core.retrieval.keyword import KeywordDocument
    >>> passages = {
    ...     "d1:0": "Hot air drying of mango slices at 60 C for eight hours.",
    ...     "d2:0": "Retort sterilisation of canned vegetables at 121 C.",
    ...     "d3:0": "Water activity below 0.6 inhibits microbial growth.",
    ... }
    >>> embedder = HashingEmbedder(dim=1024)
    >>> store = MemoryVectorStore()
    >>> store.ensure_collection(embedder.spec)
    >>> vectors = embedder.embed_documents(list(passages.values()))
    >>> from kip.core.vectorstore.base import VectorRecord
    >>> _ = store.upsert([
    ...     VectorRecord(key, vectors[i],
    ...                  {"document_id": key.split(":")[0], "chunk_index": 0})
    ...     for i, key in enumerate(passages)
    ... ])
    >>> keyword = Bm25Index()
    >>> _ = keyword.add([
    ...     KeywordDocument(key, text,
    ...                     {"document_id": key.split(":")[0], "chunk_index": 0})
    ...     for key, text in passages.items()
    ... ])
    >>> retriever = HybridRetriever(embedder, store, keyword)

    A query naming a literal value is carried by the keyword axis:

    >>> result = retriever.retrieve("0.6", top_k=1)
    >>> result.ids
    ('d3:0',)
    >>> result.diagnostics.mode
    'hybrid'

    Both axes contribute, and which ones found a hit is recorded:

    >>> result = retriever.retrieve("dried mango", top_k=3)
    >>> result.hits[0].id
    'd1:0'
    >>> result.hits[0].retrievers
    ('dense', 'keyword')
    >>> result.diagnostics.dense_candidates > 0
    True
    >>> result.diagnostics.keyword_candidates > 0
    True

    Single-axis modes are available for evaluation and for debugging:

    >>> retriever.retrieve("dried mango", mode="keyword").diagnostics.dense_candidates
    0
    >>> retriever.retrieve("dried mango", mode="dense").diagnostics.keyword_candidates
    0

    Selecting zero documents returns nothing rather than searching everything:

    >>> retriever.retrieve("mango", filters={"document_id": []}).ids
    ()

    A blank query is answered, not crashed on:

    >>> blank = retriever.retrieve("   ")
    >>> blank.ids, blank.diagnostics.notes
    ((), ('Empty query: nothing to retrieve.',))

    Without a keyword index the retriever degrades visibly instead of failing:

    >>> lonely = HybridRetriever(embedder, store, None)
    >>> result = lonely.retrieve("dried mango", top_k=1)
    >>> result.diagnostics.notes
    ('No keyword index configured; using semantic retrieval only.',)
    >>> len(result.hits)
    1
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex | None = None,
        *,
        mode: str = "hybrid",
        fusion: str = "rrf",
        dense_top_k: int = 24,
        keyword_top_k: int = 24,
        rrf_k: int = DEFAULT_RRF_K,
        dense_weight: float = 0.65,
        keyword_weight: float = 0.35,
        candidate_limit: int = 40,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.mode = _check_mode(mode)
        self.fusion = _check_fusion(fusion)
        self.dense_top_k = max(1, int(dense_top_k))
        self.keyword_top_k = max(1, int(keyword_top_k))
        self.rrf_k = max(1, int(rrf_k))
        self.weights = {DENSE: float(dense_weight), KEYWORD: float(keyword_weight)}
        self.candidate_limit = max(1, int(candidate_limit))

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        keyword_index: KeywordIndex | None = None,
    ) -> "HybridRetriever":
        """Build a retriever from configuration, with overrides for tests.

        Components are injectable so a test can pass in-memory backends without
        the configuration having to describe a test environment.
        """
        if embedder is None:
            from kip.core.embeddings import get_embedder

            embedder = get_embedder(settings)
        if vector_store is None:
            from kip.core.vectorstore import get_vector_store

            vector_store = get_vector_store(settings)
        if keyword_index is None:
            from kip.core.retrieval import get_keyword_index

            keyword_index = get_keyword_index(settings)

        return cls(
            embedder,
            vector_store,
            keyword_index,
            mode=getattr(settings, "retrieval_mode", "hybrid"),
            fusion=getattr(settings, "retrieval_fusion", "rrf"),
            dense_top_k=getattr(settings, "retrieval_dense_top_k", 24),
            keyword_top_k=getattr(settings, "retrieval_keyword_top_k", 24),
            rrf_k=getattr(settings, "retrieval_rrf_k", DEFAULT_RRF_K),
            dense_weight=getattr(settings, "retrieval_dense_weight", 0.65),
            keyword_weight=getattr(settings, "retrieval_keyword_weight", 0.35),
            candidate_limit=getattr(settings, "retrieval_candidate_limit", 40),
        )

    # -- retrieval ---------------------------------------------------------- #

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Filters | None = None,
        mode: str | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        active_mode = _check_mode(mode or self.mode)
        limit = self.candidate_limit if top_k is None else max(0, int(top_k))
        notes: list[str] = []

        cleaned = (query or "").strip()
        if not cleaned:
            notes.append("Empty query: nothing to retrieve.")
            return RetrievalResult(
                query=cleaned,
                diagnostics=self._diagnostics(
                    active_mode, notes=notes, total_ms=_elapsed(started)
                ),
            )

        rankings: dict[str, Sequence[Any]] = {}
        dense_ms = keyword_ms = 0.0

        if active_mode in {"hybrid", "dense"}:
            dense_started = time.perf_counter()
            rankings[DENSE] = self._dense(cleaned, filters)
            dense_ms = _elapsed(dense_started)

        if active_mode in {"hybrid", "keyword"}:
            if self.keyword_index is None:
                if active_mode == "keyword":
                    raise ValueError(
                        "RETRIEVAL_MODE=keyword requires a keyword index, but none "
                        "is configured. Set KEYWORD_INDEX, or use hybrid/dense."
                    )
                notes.append("No keyword index configured; using semantic retrieval only.")
            else:
                keyword_started = time.perf_counter()
                rankings[KEYWORD] = self.keyword_index.search(
                    cleaned, top_k=self.keyword_top_k, filters=filters
                )
                keyword_ms = _elapsed(keyword_started)
                if not rankings[KEYWORD] and active_mode == "hybrid":
                    notes.append(
                        "Keyword search matched nothing; ranking is semantic only."
                    )

        fusion_started = time.perf_counter()
        fused = fuse(
            rankings,
            method=self.fusion,
            k=self.rrf_k,
            weights=self.weights,
            limit=limit,
        )
        fusion_ms = _elapsed(fusion_started)

        diagnostics = self._diagnostics(
            active_mode,
            dense_candidates=len(rankings.get(DENSE, ())),
            keyword_candidates=len(rankings.get(KEYWORD, ())),
            fused_candidates=len(fused),
            returned=len(fused),
            dense_ms=dense_ms,
            keyword_ms=keyword_ms,
            fusion_ms=fusion_ms,
            total_ms=_elapsed(started),
            notes=notes,
        )
        return RetrievalResult(query=cleaned, hits=tuple(fused), diagnostics=diagnostics)

    # -- internals ---------------------------------------------------------- #

    def _dense(self, query: str, filters: Filters | None) -> Sequence[Any]:
        """Embed and search. Fingerprint mismatches are allowed to propagate."""
        vector: np.ndarray = self.embedder.embed_query(query)
        return self.vector_store.search(vector, top_k=self.dense_top_k, filters=filters)

    def _diagnostics(
        self,
        mode: str,
        *,
        notes: Sequence[str] = (),
        **fields: Any,
    ) -> RetrievalDiagnostics:
        return RetrievalDiagnostics(
            mode=mode,
            fusion=self.fusion,
            embedding_fingerprint=self.embedder.fingerprint,
            keyword_backend=getattr(self.keyword_index, "name", None),
            notes=tuple(notes),
            **fields,
        )

    def describe(self) -> dict[str, Any]:
        """Active retrieval configuration, for the Settings screen and health."""
        return {
            "mode": self.mode,
            "fusion": self.fusion,
            "dense_top_k": self.dense_top_k,
            "keyword_top_k": self.keyword_top_k,
            "rrf_k": self.rrf_k,
            "weights": dict(self.weights),
            "candidate_limit": self.candidate_limit,
            "embedding": self.embedder.spec.to_dict(),
            "vector_store": self.vector_store.name,
            "keyword_index": getattr(self.keyword_index, "name", None),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<HybridRetriever mode={self.mode} fusion={self.fusion} "
            f"dense={self.vector_store.name} "
            f"keyword={getattr(self.keyword_index, 'name', None)}>"
        )


def _elapsed(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _check_mode(mode: str) -> str:
    """Validate ``RETRIEVAL_MODE``.

    >>> _check_mode("HYBRID")
    'hybrid'
    >>> _check_mode("semantic")
    Traceback (most recent call last):
        ...
    ValueError: Unknown RETRIEVAL_MODE 'semantic'. Available modes: hybrid, dense, keyword.
    """
    name = str(mode or "hybrid").strip().lower()
    if name not in RETRIEVAL_MODES:
        raise ValueError(
            f"Unknown RETRIEVAL_MODE {name!r}. Available modes: "
            + ", ".join(RETRIEVAL_MODES)
            + "."
        )
    return name


def _check_fusion(method: str) -> str:
    """Validate ``RETRIEVAL_FUSION``.

    >>> _check_fusion("RRF")
    'rrf'
    >>> _check_fusion("mean")
    Traceback (most recent call last):
        ...
    ValueError: Unknown RETRIEVAL_FUSION 'mean'. Available methods: rrf, weighted.
    """
    name = str(method or "rrf").strip().lower()
    if name not in FUSION_METHODS:
        raise ValueError(
            f"Unknown RETRIEVAL_FUSION {name!r}. Available methods: "
            + ", ".join(FUSION_METHODS)
            + "."
        )
    return name


__all__ = [
    "DENSE",
    "KEYWORD",
    "RETRIEVAL_MODES",
    "HybridRetriever",
    "RetrievalDiagnostics",
    "RetrievalResult",
]
