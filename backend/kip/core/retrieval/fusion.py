"""Rank fusion -- combining the dense and keyword result lists into one.

The problem fusion solves is that the two retrievers speak different languages.
Dense search returns cosine similarities bounded in ``[-1, 1]``; BM25 returns
unbounded non-negative scores whose magnitude depends on corpus size, document
length and term rarity. A cosine of 0.42 and a BM25 of 8.3 cannot be added,
averaged, or compared, and any code that does so is producing a number with no
defensible meaning.

Two strategies are offered, and the default is the one that sidesteps the problem
entirely.

**Reciprocal Rank Fusion (default).** Uses only each retriever's *ordering*:

.. code-block:: text

    score(d) = SUM over retrievers r of   weight[r] / (k + rank[r](d))

Rank is 1-based; documents a retriever did not return contribute nothing. ``k``
(default 60, from Cormack et al. 2009) flattens the curve so that the difference
between rank 1 and rank 2 does not dwarf the signal from the other retriever --
with ``k = 0`` the top hit of either list would almost always win outright,
which defeats the purpose of fusing. RRF needs no calibration, no normalisation,
and no assumption that either score scale is stable, which is why it is the
default.

**Weighted score fusion.** Min-max normalises each retriever's scores to
``[0, 1]`` and takes a weighted sum. Available because it can outperform RRF when
the score distributions genuinely are informative, but it carries a flaw worth
stating plainly: min-max always maps the best hit in a list to 1.0, so it cannot
distinguish "the top result is excellent" from "the top result is the least bad
of a uniformly bad set". That makes it a poor foundation for the
insufficient-evidence check, which is why ``GROUNDING_MIN_SCORE`` is applied to
raw similarity rather than to a fused score.

Which one this platform actually uses is settled by measurement, not preference:
``docs/EVALUATION.md`` reports Recall@K, MRR and nDCG for dense-only,
keyword-only, RRF and weighted fusion over the same query set.

What fusion deliberately does not do
------------------------------------
It deduplicates by chunk id only. Overlapping chunks share sentences by design,
so two adjacent chunks can both be retrieved and say much the same thing --
wasteful, since they compete for the same context budget. Suppressing that
requires the passage *text*, which lives in the relational database and is not
available at this layer. Near-duplicate suppression therefore belongs to the
context builder, where the text has been hydrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

#: Cormack et al. (2009). Large enough that no single retriever's top hit can
#: dominate the fused ordering on its own.
DEFAULT_RRF_K = 60

FUSION_METHODS = ("rrf", "weighted")


class Scored(Protocol):
    """Anything fusion can consume: a vector ``SearchHit`` or a ``KeywordHit``."""

    id: str
    score: float
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Contribution:
    """One retriever's view of one candidate.

    Kept per-retriever rather than collapsed into the fused score because it is
    what makes a result explainable. The UI can say "returned by keyword search at
    rank 2 and semantic search at rank 7, matching *sterilisation*", which is a
    real answer to "why am I seeing this passage". A single fused float is not.
    """

    retriever: str
    rank: int
    score: float
    normalised: float = 0.0
    matched_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "retriever": self.retriever,
            "rank": self.rank,
            "score": round(float(self.score), 6),
            "normalised": round(float(self.normalised), 6),
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True, slots=True)
class FusedHit:
    """A candidate after fusion, carrying its provenance from each retriever."""

    id: str
    score: float
    payload: Mapping[str, Any] = field(default_factory=dict)
    contributions: tuple[Contribution, ...] = ()

    @property
    def retrievers(self) -> tuple[str, ...]:
        return tuple(item.retriever for item in self.contributions)

    @property
    def matched_terms(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for item in self.contributions:
            for term in item.matched_terms:
                seen.setdefault(term, None)
        return tuple(seen)

    @property
    def document_id(self) -> str:
        return str(self.payload.get("document_id", ""))

    @property
    def chunk_index(self) -> int:
        try:
            return int(self.payload.get("chunk_index", -1))
        except (TypeError, ValueError):
            return -1

    def best_score(self, retriever: str) -> float | None:
        """Raw score from one retriever, or ``None`` if it did not return this.

        The dense score is needed unmodified for the grounding threshold, so it
        has to remain reachable after fusion has replaced it.
        """
        for item in self.contributions:
            if item.retriever == retriever:
                return float(item.score)
        return None

    def rank_in(self, retriever: str) -> int | None:
        for item in self.contributions:
            if item.retriever == retriever:
                return item.rank
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": round(float(self.score), 6),
            "retrievers": list(self.retrievers),
            "matched_terms": list(self.matched_terms),
            "contributions": [item.to_dict() for item in self.contributions],
            **dict(self.payload),
        }


def min_max(values: Sequence[float]) -> list[float]:
    """Scale ``values`` into ``[0, 1]``.

    An all-equal input maps to 1.0 rather than 0.0 or 0.5: the values are tied,
    and tied candidates should keep whatever weight their retriever carries
    instead of being zeroed out of the fusion.

    >>> min_max([1.0, 3.0, 2.0])
    [0.0, 1.0, 0.5]
    >>> min_max([5.0, 5.0])
    [1.0, 1.0]
    >>> min_max([])
    []
    """
    if not values:
        return []
    lowest = min(values)
    highest = max(values)
    span = highest - lowest
    if span <= 0.0:
        return [1.0 for _ in values]
    return [(value - lowest) / span for value in values]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Scored]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[FusedHit]:
    """Fuse ranked lists by reciprocal rank.

    >>> from kip.core.vectorstore.base import SearchHit
    >>> dense = [SearchHit("c2", 0.81, {"document_id": "d"}),
    ...          SearchHit("c1", 0.44, {"document_id": "d"})]
    >>> keyword = [SearchHit("c1", 9.2, {"document_id": "d"}),
    ...            SearchHit("c3", 4.1, {"document_id": "d"})]
    >>> fused = reciprocal_rank_fusion({"dense": dense, "keyword": keyword})
    >>> [hit.id for hit in fused]
    ['c1', 'c2', 'c3']

    ``c1`` wins because it appears in both lists: agreement between independent
    retrievers is the signal RRF is built to reward.

    >>> fused[0].retrievers
    ('dense', 'keyword')
    >>> fused[0].rank_in("keyword")
    1
    >>> round(fused[0].score, 6) == round(1 / 62 + 1 / 61, 6)
    True

    Raw per-retriever scores survive fusion:

    >>> fused[0].best_score("dense")
    0.44
    >>> fused[0].best_score("missing") is None
    True
    >>> reciprocal_rank_fusion({})
    []
    """
    weights = weights or {}
    accumulated: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    contributions: dict[str, list[Contribution]] = {}

    for retriever, hits in rankings.items():
        weight = float(weights.get(retriever, 1.0))
        for position, hit in enumerate(hits, start=1):
            identifier = str(hit.id)
            accumulated[identifier] = accumulated.get(identifier, 0.0) + weight / (
                k + position
            )
            _merge_payload(payloads, identifier, hit)
            contributions.setdefault(identifier, []).append(
                Contribution(
                    retriever=retriever,
                    rank=position,
                    score=float(hit.score),
                    matched_terms=tuple(getattr(hit, "matched_terms", ()) or ()),
                )
            )

    return _assemble(accumulated, payloads, contributions, limit)


def weighted_fusion(
    rankings: Mapping[str, Sequence[Scored]],
    *,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[FusedHit]:
    """Fuse ranked lists by weighted, min-max normalised score.

    >>> from kip.core.vectorstore.base import SearchHit
    >>> dense = [SearchHit("c2", 0.81, {}), SearchHit("c1", 0.44, {})]
    >>> keyword = [SearchHit("c1", 9.2, {}), SearchHit("c3", 4.1, {})]
    >>> fused = weighted_fusion({"dense": dense, "keyword": keyword},
    ...                         weights={"dense": 0.65, "keyword": 0.35})
    >>> [hit.id for hit in fused]
    ['c2', 'c1', 'c3']
    >>> [round(hit.score, 4) for hit in fused]
    [0.65, 0.35, 0.0]

    ``c2`` scores ``0.65 * 1.0`` for topping the dense list; ``c1`` scores
    ``0.65 * 0.0 + 0.35 * 1.0``. Note that ``c2``, the best dense hit, and ``c3``,
    the worst keyword hit, normalise to their list's extremes regardless of how
    good either actually was -- and that ``c1``, which *both* retrievers returned,
    is pushed below a hit only one of them found. That is the flaw described in the
    module docstring, made visible: compare the RRF example above, where agreement
    between retrievers wins.

    >>> weighted_fusion({"dense": []})
    []
    """
    weights = weights or {}
    accumulated: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    contributions: dict[str, list[Contribution]] = {}

    for retriever, hits in rankings.items():
        if not hits:
            continue
        weight = float(weights.get(retriever, 1.0))
        normalised = min_max([float(hit.score) for hit in hits])
        for position, (hit, unit) in enumerate(zip(hits, normalised), start=1):
            identifier = str(hit.id)
            accumulated[identifier] = accumulated.get(identifier, 0.0) + weight * unit
            _merge_payload(payloads, identifier, hit)
            contributions.setdefault(identifier, []).append(
                Contribution(
                    retriever=retriever,
                    rank=position,
                    score=float(hit.score),
                    normalised=unit,
                    matched_terms=tuple(getattr(hit, "matched_terms", ()) or ()),
                )
            )

    return _assemble(accumulated, payloads, contributions, limit)


def fuse(
    rankings: Mapping[str, Sequence[Scored]],
    *,
    method: str = "rrf",
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[FusedHit]:
    """Dispatch to the configured fusion strategy.

    An unknown method raises rather than defaulting, so a typo in
    ``RETRIEVAL_FUSION`` cannot silently change how every answer is ranked.

    >>> fuse({}, method="rrf")
    []
    >>> fuse({}, method="average")
    Traceback (most recent call last):
        ...
    ValueError: Unknown RETRIEVAL_FUSION 'average'. Available methods: rrf, weighted.
    """
    name = str(method or "rrf").strip().lower()
    if name == "rrf":
        return reciprocal_rank_fusion(rankings, k=k, weights=weights, limit=limit)
    if name == "weighted":
        return weighted_fusion(rankings, weights=weights, limit=limit)
    raise ValueError(
        f"Unknown RETRIEVAL_FUSION {name!r}. Available methods: "
        + ", ".join(FUSION_METHODS)
        + "."
    )


def _merge_payload(
    payloads: dict[str, dict[str, Any]],
    identifier: str,
    hit: Scored,
) -> None:
    """Accumulate payload keys across retrievers.

    Both axes should agree, but if one carries a key the other omits, the union is
    strictly more useful than whichever list happened to be processed last.
    Existing keys are never overwritten, so the result is deterministic.
    """
    target = payloads.setdefault(identifier, {})
    for key, value in dict(getattr(hit, "payload", {}) or {}).items():
        target.setdefault(key, value)


def _assemble(
    accumulated: Mapping[str, float],
    payloads: Mapping[str, Mapping[str, Any]],
    contributions: Mapping[str, Sequence[Contribution]],
    limit: int | None,
) -> list[FusedHit]:
    """Order the fused candidates, breaking ties deterministically by id."""
    ordered = sorted(accumulated.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None and limit >= 0:
        ordered = ordered[:limit]
    return [
        FusedHit(
            id=identifier,
            score=float(score),
            payload=dict(payloads.get(identifier, {})),
            contributions=tuple(
                sorted(contributions.get(identifier, ()), key=lambda item: item.retriever)
            ),
        )
        for identifier, score in ordered
    ]


def dedupe_by_document(hits: Iterable[FusedHit], *, max_per_document: int) -> list[FusedHit]:
    """Cap how many chunks any single document may contribute.

    Without a cap, one long, on-topic document can occupy every context slot, and
    the answer then reflects a single source while looking well-cited. Ordering is
    otherwise preserved.

    >>> hits = [FusedHit("a:0", 0.9, {"document_id": "a"}),
    ...         FusedHit("a:1", 0.8, {"document_id": "a"}),
    ...         FusedHit("a:2", 0.7, {"document_id": "a"}),
    ...         FusedHit("b:0", 0.6, {"document_id": "b"})]
    >>> [hit.id for hit in dedupe_by_document(hits, max_per_document=2)]
    ['a:0', 'a:1', 'b:0']
    >>> [hit.id for hit in dedupe_by_document(hits, max_per_document=0)]
    ['a:0', 'a:1', 'a:2', 'b:0']
    """
    if max_per_document <= 0:
        return list(hits)
    seen: dict[str, int] = {}
    kept: list[FusedHit] = []
    for hit in hits:
        key = hit.document_id or hit.id
        count = seen.get(key, 0)
        if count >= max_per_document:
            continue
        seen[key] = count + 1
        kept.append(hit)
    return kept


__all__ = [
    "Contribution",
    "DEFAULT_RRF_K",
    "FUSION_METHODS",
    "FusedHit",
    "Scored",
    "dedupe_by_document",
    "fuse",
    "min_max",
    "reciprocal_rank_fusion",
    "weighted_fusion",
]
