"""Keyword (lexical) retrieval abstraction -- the second axis of hybrid search.

Why have this at all when there is already a vector index? Because dense
retrieval and keyword retrieval fail in different, complementary ways. An
embedding model generalises: it will happily return a passage about "thermal
processing" for a query about "heat treatment", which is usually what you want.
But it also blurs exactly the tokens a technical user is most likely to type
verbatim -- ``a_w``, ``0.85``, ``72 C``, ``E 471``, a product code, a clause
number. BM25 does the opposite: it is literal, it rewards rare terms, and it
cannot generalise at all.

Fusing the two recovers most of the recall that either loses alone, and the
project measures the difference rather than asserting it: ``docs/EVALUATION.md``
reports Recall@K and MRR for dense-only, keyword-only and hybrid retrieval over
the same query set.

Relationship to the citation source of truth
-------------------------------------------
A keyword index necessarily stores the passage text -- an inverted index *is*
derived from text. That makes it a **derived index**, not a source: citations are
always resolved from the ``chunks`` table in the relational database, never from
here. The index can be dropped and rebuilt at any time, which is what
:meth:`KeywordIndex.rebuild` is for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# Payload filtering is identical to the vector side -- tenant isolation and
# document selection have to behave the same way on both axes or hybrid search
# would leak on one of them. Reusing the helpers is deliberate: two
# implementations of "does this payload match" is one implementation too many.
from kip.core.vectorstore.base import Filters, normalise_filters, payload_matches

#: Okapi BM25 term-frequency saturation. 1.2 is the standard default; higher
#: values make repeated terms count for more.
BM25_K1 = 1.2
#: Length normalisation. 0.75 is standard; 0 ignores document length entirely.
BM25_B = 0.75


class KeywordIndexError(RuntimeError):
    """A keyword index operation failed."""


@dataclass(frozen=True, slots=True)
class KeywordHit:
    """A scored lexical match.

    ``score`` is a BM25 score: unbounded, non-negative, and **not** comparable
    to a cosine similarity. That incomparability is the entire reason fusion
    works on ranks rather than raw scores by default.

    ``matched_terms`` carries the query terms that actually fired. The source
    viewer uses it to highlight, and it is what makes a keyword hit explainable
    to a user who is wondering why a passage was retrieved.
    """

    id: str
    score: float
    payload: Mapping[str, Any] = field(default_factory=dict)
    matched_terms: tuple[str, ...] = ()

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
        return {
            "id": self.id,
            "score": round(float(self.score), 6),
            "matched_terms": list(self.matched_terms),
            **dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class KeywordDocument:
    """One indexable unit: a chunk id, its text, and its provenance payload."""

    id: str
    text: str
    payload: Mapping[str, Any] = field(default_factory=dict)


class KeywordIndex(ABC):
    """Minimal interface every keyword backend implements."""

    name: str = "abstract"

    @abstractmethod
    def add(self, documents: Sequence[KeywordDocument]) -> int:
        """Insert or replace ``documents``; returns the number written."""

    @abstractmethod
    def delete(self, *, filters: Filters) -> int:
        """Remove every entry matching ``filters``; returns the count removed."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Filters | None = None,
    ) -> list[KeywordHit]:
        """Return the ``top_k`` best lexical matches for ``query``."""

    @abstractmethod
    def count(self, *, filters: Filters | None = None) -> int:
        """Number of indexed entries matching ``filters``."""

    def rebuild(self, documents: Iterable[KeywordDocument]) -> int:
        """Discard everything and re-index from scratch.

        The default implementation deletes and re-adds. Backends that can do
        this faster (``DROP TABLE`` and recreate) may override it.
        """
        self.clear()
        batch: list[KeywordDocument] = []
        written = 0
        for document in documents:
            batch.append(document)
            if len(batch) >= 500:
                written += self.add(batch)
                batch = []
        if batch:
            written += self.add(batch)
        return written

    @abstractmethod
    def clear(self) -> None:
        """Remove every entry. Used by :meth:`rebuild` and by tests."""

    def close(self) -> None:
        """Release resources. Safe to call more than once."""

    def stats(self) -> dict[str, Any]:
        """Backend summary for the dashboard and ``/api/health``."""
        return {"backend": self.name, "documents": self.count()}


__all__ = [
    "BM25_B",
    "BM25_K1",
    "Filters",
    "KeywordDocument",
    "KeywordHit",
    "KeywordIndex",
    "KeywordIndexError",
    "normalise_filters",
    "payload_matches",
]
