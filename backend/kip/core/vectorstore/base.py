"""Vector store abstraction.

The retrieval layer talks to a :class:`VectorStore`, never to a specific
database. ``VECTOR_STORE`` selects the implementation: ``memory`` (tests and
evaluation), ``sqlite`` (default -- persistent, zero infrastructure) or
``qdrant`` (the production path, and what ``docker compose`` starts).

Two properties are enforced here rather than left to each backend, because
getting either wrong produces a system that looks fine and answers wrongly:

**Model fingerprint agreement.** A collection records the
``provider:model:dim`` of the embedder that populated it. Searching it with
vectors from a different model is refused with an instruction to re-index,
instead of returning confidently ranked noise.

**Filters are mandatory-capable, not optional.** Every implementation must
support equality and membership filters, because tenant isolation
(``user_id``) and multi-document selection (``document_id IN (...)``) are
enforced through them. A store that silently ignored an unsupported filter
would leak one user's documents into another user's answers, so unsupported
filter shapes raise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from kip.core.embeddings.base import VECTOR_DTYPE, EmbeddingSpec, l2_normalise

#: Payload keys the retrieval and citation layers rely on being present.
REQUIRED_PAYLOAD_KEYS: frozenset[str] = frozenset({"document_id", "chunk_index"})


class VectorStoreError(RuntimeError):
    """A vector store operation failed."""


class VectorStoreMismatch(VectorStoreError):
    """The stored vectors were produced by a different embedding model."""


@dataclass(slots=True)
class VectorRecord:
    """One indexed chunk: an id, its vector, and its provenance payload.

    ``payload`` is the *only* channel through which retrieval learns where a
    passage came from, so it must carry enough to rebuild a citation without a
    second lookup: document id, chunk index, page range and section path.
    """

    id: str
    vector: np.ndarray
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.vector = np.asarray(self.vector, dtype=VECTOR_DTYPE).reshape(-1)
        missing = REQUIRED_PAYLOAD_KEYS - set(self.payload)
        if missing:
            raise VectorStoreError(
                "Vector payload is missing provenance keys "
                f"{sorted(missing)}; citations could not be resolved from it."
            )

    @property
    def dim(self) -> int:
        return int(self.vector.shape[0])


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A scored result. ``score`` is cosine similarity in ``[-1, 1]``."""

    id: str
    score: float
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        return str(self.payload.get("document_id", ""))

    @property
    def chunk_index(self) -> int:
        try:
            return int(self.payload.get("chunk_index", -1))
        except (TypeError, ValueError):
            return -1

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "score": round(float(self.score), 6), **dict(self.payload)}


Filters = Mapping[str, Any]


def normalise_filters(filters: Filters | None) -> dict[str, tuple[Any, ...]]:
    """Normalise a filter mapping into ``{field: (allowed, values...)}``.

    A scalar becomes a one-element tuple, so every backend has exactly one shape
    to implement -- membership -- rather than branching on type.

    >>> normalise_filters({"user_id": 7, "document_id": ["a", "b"]})
    {'user_id': (7,), 'document_id': ('a', 'b')}
    >>> normalise_filters(None)
    {}
    >>> normalise_filters({"document_id": []})
    {'document_id': ()}
    """
    if not filters:
        return {}
    out: dict[str, tuple[Any, ...]] = {}
    for key, value in filters.items():
        if isinstance(value, (list, tuple, set, frozenset)):
            out[str(key)] = tuple(value)
        else:
            out[str(key)] = (value,)
    return out


def payload_matches(payload: Mapping[str, Any], filters: Mapping[str, tuple[Any, ...]]) -> bool:
    """True when ``payload`` satisfies every normalised filter.

    An empty allowed-set matches nothing. That is deliberate: "the user selected
    zero documents" must return zero results, not silently search everything.

    >>> payload_matches({"user_id": 1}, {"user_id": (1, 2)})
    True
    >>> payload_matches({"user_id": 3}, {"user_id": (1, 2)})
    False
    >>> payload_matches({"user_id": 1}, {"document_id": ()})
    False
    """
    for key, allowed in filters.items():
        if not allowed:
            return False
        value = payload.get(key)
        if value in allowed:
            continue
        # Ids cross the JSON/SQL boundary as either str or int; compare as text
        # so a filter of 7 still matches a stored "7".
        if str(value) in {str(item) for item in allowed}:
            continue
        return False
    return True


class VectorStore(ABC):
    """Minimal interface every backend implements."""

    name: str = "abstract"

    # -- lifecycle ---------------------------------------------------------- #

    @abstractmethod
    def ensure_collection(self, spec: EmbeddingSpec) -> None:
        """Create the collection if absent; verify the fingerprint if present."""

    @abstractmethod
    def stored_fingerprint(self) -> str | None:
        """Fingerprint recorded for the collection, or ``None`` if empty/new."""

    def close(self) -> None:
        """Release resources. Safe to call more than once."""

    # -- writes ------------------------------------------------------------- #

    @abstractmethod
    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Insert or replace ``records``; returns the number written."""

    @abstractmethod
    def delete(self, *, filters: Filters) -> int:
        """Delete every vector matching ``filters``; returns the count removed."""

    # -- reads -------------------------------------------------------------- #

    @abstractmethod
    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 10,
        filters: Filters | None = None,
    ) -> list[SearchHit]:
        """Return the ``top_k`` most similar vectors matching ``filters``."""

    @abstractmethod
    def count(self, *, filters: Filters | None = None) -> int:
        """Number of stored vectors matching ``filters``."""

    @abstractmethod
    def fetch(self, ids: Sequence[str]) -> list[VectorRecord]:
        """Return stored records by id, skipping ids that are absent."""

    # -- introspection ------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        """Backend summary for the dashboard and ``/api/health``."""
        return {
            "backend": self.name,
            "vectors": self.count(),
            "fingerprint": self.stored_fingerprint(),
        }

    # -- shared helpers ----------------------------------------------------- #

    def _guard_fingerprint(self, spec: EmbeddingSpec) -> None:
        stored = self.stored_fingerprint()
        if stored and stored != spec.fingerprint:
            raise VectorStoreMismatch(
                f"This index was built with '{stored}' but the active embedding "
                f"model is '{spec.fingerprint}'. Vectors from different models "
                "are not comparable. Re-index the documents, or set "
                "EMBEDDING_PROVIDER/EMBEDDING_MODEL back to the previous value."
            )

    @staticmethod
    def _prepare_query(vector: np.ndarray, dim: int | None = None) -> np.ndarray:
        query = l2_normalise(np.asarray(vector, dtype=VECTOR_DTYPE).reshape(-1))
        if dim is not None and query.shape[0] != dim:
            raise VectorStoreMismatch(
                f"Query vector has {query.shape[0]} dimensions but the index "
                f"stores {dim}. The embedding model changed; re-index required."
            )
        return query

    @staticmethod
    def _top_k(
        scores: np.ndarray,
        ids: Sequence[str],
        payloads: Sequence[Mapping[str, Any]],
        top_k: int,
    ) -> list[SearchHit]:
        """Select the ``top_k`` highest scores, breaking ties by id.

        ``argpartition`` keeps the common case O(n) instead of O(n log n), but it
        chooses arbitrarily among *equal* scores -- so sorting the selected slice
        afterwards is not enough to make the result deterministic. Whichever tied
        indices happened to land inside the partition would win, and that depends
        on array order, which differs between an in-memory store and rows read
        back from SQLite.

        So every index tied with the cut-off score is folded back in before
        ordering. Determinism matters here beyond tidiness: two backends holding
        the same vectors must rank them the same way, and a citation must not
        depend on which store served the query.
        """
        if scores.size == 0 or top_k <= 0:
            return []
        limit = min(int(top_k), scores.shape[0])
        if limit < scores.shape[0]:
            candidate = np.argpartition(-scores, limit - 1)[:limit]
            cutoff = float(scores[candidate].min())
            tied = np.flatnonzero(scores == cutoff)
            if tied.shape[0] > 1:
                candidate = np.union1d(candidate, tied)
        else:
            candidate = np.arange(scores.shape[0])
        ordered = sorted(
            candidate.tolist(), key=lambda index: (-float(scores[index]), str(ids[index]))
        )[:limit]
        return [
            SearchHit(id=str(ids[index]), score=float(scores[index]), payload=dict(payloads[index]))
            for index in ordered
        ]


def records_from_chunks(
    chunks: Iterable[Any],
    vectors: np.ndarray,
    *,
    document_id: str,
    user_id: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> list[VectorRecord]:
    """Pair chunks with their vectors into storable records.

    The id is ``<document_id>:<chunk_index>``, which makes re-indexing a
    document idempotent: the same chunk overwrites its own vector instead of
    accumulating duplicates that would then both be retrieved.

    The chunk *body* is intentionally not stored in the payload. The relational
    database owns passage text; duplicating it here would let the two drift and
    let a citation quote text that is no longer in the document.
    """
    chunk_list = list(chunks)
    matrix = np.asarray(vectors, dtype=VECTOR_DTYPE)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[0] != len(chunk_list):
        raise VectorStoreError(
            f"Got {matrix.shape[0]} vectors for {len(chunk_list)} chunks; "
            "refusing to index a misaligned batch because citations would "
            "point at the wrong passages."
        )

    out: list[VectorRecord] = []
    for position, chunk in enumerate(chunk_list):
        index = int(getattr(chunk, "index", position))
        payload: dict[str, Any] = {
            "document_id": str(document_id),
            "chunk_index": index,
            "page_start": getattr(chunk, "page_start", None),
            "page_end": getattr(chunk, "page_end", None),
            "section_key": getattr(chunk, "section_key", None),
            "section_path": list(getattr(chunk, "section_path", ()) or ()),
            "heading": getattr(chunk, "heading", None),
            "token_count": getattr(chunk, "token_count", None),
        }
        if user_id is not None:
            payload["user_id"] = user_id
        if extra:
            payload.update(extra)
        out.append(
            VectorRecord(
                id=f"{document_id}:{index}",
                vector=matrix[position],
                payload=payload,
            )
        )
    return out
