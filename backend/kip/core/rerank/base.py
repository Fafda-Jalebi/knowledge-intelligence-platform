"""Reranker abstraction -- the second-stage scorer between retrieval and context.

Why a second stage exists at all. Retrieval is optimised for *recall* over a
whole corpus: it must be cheap enough to score every chunk, which forces it to
use representations that discard word order and sentence structure. A dense
retriever compares one averaged vector per passage; BM25 sums independent term
weights. Neither can tell that a passage mentioning all five query concepts in a
single sentence is a better answer than one that repeats a single rare term nine
times in unrelated paragraphs.

A reranker sees far fewer candidates -- the 24-40 that survived retrieval -- so it
can afford to look at the query and the passage *together*. That is the whole
trade: quadratically more attention over a linearly smaller set.

The order matters for a reason specific to this platform's storage design.
Reranking needs passage **text**, and the vector payload deliberately does not
store text, so that a citation can never quote something the document no longer
contains. Text is hydrated from the relational database by the service layer,
which is therefore where reranking happens -- after retrieval, before the context
builder.

.. code-block:: text

    fused candidates (ids + provenance)
        -> service hydrates text from the chunks table
        -> reranker scores (query, text) pairs
        -> top N by rerank score
        -> context builder -> LLM

Every reranker is optional and swappable via ``RERANKER``, and ``none`` is a
supported setting: the evaluation harness runs with and without reranking so the
stage has to justify itself with measured numbers rather than with the argument
above.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: Passage text handed to a reranker is truncated to this many characters.
#: Relevance is decided by the opening of a chunk far more often than by its tail,
#: and for the LLM reranker this is also a cost and privacy control: it bounds how
#: much document text can leave the machine per candidate.
MAX_CANDIDATE_CHARS = 1600


class RerankError(RuntimeError):
    """A reranker backend was unavailable or failed."""


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """A retrieved passage with its text hydrated, ready to be scored."""

    id: str
    text: str
    score: float = 0.0
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def heading(self) -> str:
        """Best available human label for where this passage sits."""
        heading = self.payload.get("heading")
        if heading:
            return str(heading)
        path = self.payload.get("section_path") or ()
        return " > ".join(str(part) for part in path)

    def truncated(self, limit: int = MAX_CANDIDATE_CHARS) -> str:
        text = self.text or ""
        return text if len(text) <= limit else text[:limit]


@dataclass(frozen=True, slots=True)
class RerankResult:
    """One scored candidate, with enough detail to explain the reordering.

    ``prior_rank`` and ``rank`` are both retained so the interface can show that
    a passage moved from 9th to 2nd. A reranker that never moves anything is a
    reranker that should be switched off, and this makes that visible instead of
    hiding it behind a changed number.
    """

    id: str
    score: float
    prior_score: float
    prior_rank: int
    rank: int
    text: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    signals: Mapping[str, float] = field(default_factory=dict)

    @property
    def movement(self) -> int:
        """Positions gained: positive means promoted.

        >>> RerankResult("a", 0.9, 0.4, prior_rank=9, rank=2).movement
        7
        """
        return self.prior_rank - self.rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": round(float(self.score), 6),
            "prior_score": round(float(self.prior_score), 6),
            "prior_rank": self.prior_rank,
            "rank": self.rank,
            "movement": self.movement,
            "signals": {key: round(float(value), 4) for key, value in self.signals.items()},
        }


class Reranker(ABC):
    """Base class handling ordering, truncation and top-N selection.

    Subclasses implement :meth:`_score` only: given a query and candidates,
    return one float per candidate, higher meaning more relevant. Everything the
    rest of the platform depends on -- stable ordering, correct rank bookkeeping,
    honouring ``top_n``, never crashing on an empty candidate list -- is enforced
    here so no backend can get it subtly wrong.
    """

    name: str = "abstract"

    #: Set by backends whose scores are comparable across queries. Heuristic and
    #: cross-encoder scores are not absolute judgements, so nothing downstream is
    #: allowed to treat them as confidence.
    calibrated: bool = False

    def __init__(self, *, max_chars: int = MAX_CANDIDATE_CHARS) -> None:
        self.max_chars = max(1, int(max_chars))

    # -- subclass contract -------------------------------------------------- #

    @abstractmethod
    def _score(self, query: str, candidates: Sequence[RerankCandidate]) -> Sequence[float]:
        """Return one relevance score per candidate, higher is better."""

    def _signals(
        self, query: str, candidates: Sequence[RerankCandidate]
    ) -> Sequence[Mapping[str, float]]:
        """Optional per-candidate explanation of the score."""
        return [{} for _ in candidates]

    def _evaluate(
        self, query: str, candidates: Sequence[RerankCandidate]
    ) -> tuple[Sequence[float], Sequence[Mapping[str, float]]]:
        """Produce scores and signals together.

        Override this instead of :meth:`_score` when the two share expensive
        work. The default keeps the simple contract for backends whose signals
        are free or absent.

        A single call rather than two is deliberate: the obvious alternative --
        caching the last computation on the instance -- would break under
        concurrency, because one reranker instance serves every request and a
        second query could read the first query's cached signals.
        """
        return self._score(query, candidates), self._signals(query, candidates)

    def warm_up(self) -> None:
        """Optional hook: load weights before the first request."""

    # -- public API --------------------------------------------------------- #

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Reorder ``candidates`` and return at most ``top_n`` of them."""
        if not candidates:
            return []

        cleaned = (query or "").strip()
        if not cleaned:
            # Nothing to score against: preserve the retriever's order rather
            # than inventing one.
            return self._passthrough(candidates, top_n)

        raw_scores, raw_signals = self._evaluate(cleaned, candidates)
        scores = list(raw_scores)
        if len(scores) != len(candidates):
            raise RerankError(
                f"{type(self).__name__} returned {len(scores)} scores for "
                f"{len(candidates)} candidates."
            )
        signals = list(raw_signals)
        if len(signals) != len(candidates):
            signals = [{} for _ in candidates]

        prior_ranks = {candidate.id: position for position, candidate in enumerate(candidates, 1)}
        # Ties break by the retriever's original order, so a reranker with no
        # opinion is a no-op rather than a shuffle.
        order = sorted(
            range(len(candidates)),
            key=lambda index: (-float(scores[index]), prior_ranks[candidates[index].id]),
        )
        limit = len(order) if top_n is None or top_n <= 0 else min(int(top_n), len(order))

        results: list[RerankResult] = []
        for new_rank, index in enumerate(order[:limit], start=1):
            candidate = candidates[index]
            results.append(
                RerankResult(
                    id=candidate.id,
                    score=float(scores[index]),
                    prior_score=float(candidate.score),
                    prior_rank=prior_ranks[candidate.id],
                    rank=new_rank,
                    text=candidate.text,
                    payload=dict(candidate.payload),
                    signals=dict(signals[index]),
                )
            )
        return results

    def _passthrough(
        self, candidates: Sequence[RerankCandidate], top_n: int | None
    ) -> list[RerankResult]:
        limit = len(candidates) if top_n is None or top_n <= 0 else int(top_n)
        return [
            RerankResult(
                id=candidate.id,
                score=float(candidate.score),
                prior_score=float(candidate.score),
                prior_rank=position,
                rank=position,
                text=candidate.text,
                payload=dict(candidate.payload),
            )
            for position, candidate in enumerate(candidates[:limit], start=1)
        ]

    def describe(self) -> dict[str, Any]:
        return {"reranker": self.name, "calibrated": self.calibrated}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__}>"


class NoOpReranker(Reranker):
    """Keeps the retriever's ordering. ``RERANKER=none``.

    Not a stub: it is the control condition the evaluation harness measures every
    other reranker against, and the correct choice when retrieval is already
    good enough that a second stage only adds latency.

    >>> candidates = [RerankCandidate("a", "first", 0.9),
    ...               RerankCandidate("b", "second", 0.4)]
    >>> results = NoOpReranker().rerank("anything", candidates)
    >>> [(r.id, r.rank, r.movement) for r in results]
    [('a', 1, 0), ('b', 2, 0)]
    >>> NoOpReranker().rerank("anything", [])
    []
    """

    name = "none"

    def _score(self, query: str, candidates: Sequence[RerankCandidate]) -> Sequence[float]:
        # Descending by position preserves the incoming order through the sort
        # without depending on the incoming score scale, which differs between
        # RRF and weighted fusion.
        return [float(len(candidates) - index) for index in range(len(candidates))]


def candidates_from_hits(
    hits: Sequence[Any],
    texts: Mapping[str, str],
) -> list[RerankCandidate]:
    """Pair fused hits with hydrated passage text.

    Hits whose text is missing are dropped rather than scored as empty strings: a
    chunk with no text is a chunk that was deleted or failed to hydrate, and
    ranking it would risk a citation pointing at nothing.

    >>> from kip.core.retrieval.fusion import FusedHit
    >>> hits = [FusedHit("a:0", 0.5, {"document_id": "a"}),
    ...         FusedHit("a:1", 0.4, {"document_id": "a"})]
    >>> candidates_from_hits(hits, {"a:0": "Drying reduces water activity."})
    [RerankCandidate(id='a:0', text='Drying reduces water activity.', score=0.5, payload={'document_id': 'a'})]
    """
    out: list[RerankCandidate] = []
    for hit in hits:
        text = texts.get(str(hit.id))
        if not text or not str(text).strip():
            continue
        out.append(
            RerankCandidate(
                id=str(hit.id),
                text=str(text),
                score=float(getattr(hit, "score", 0.0)),
                payload=dict(getattr(hit, "payload", {}) or {}),
            )
        )
    return out


__all__ = [
    "MAX_CANDIDATE_CHARS",
    "NoOpReranker",
    "RerankCandidate",
    "RerankError",
    "RerankResult",
    "Reranker",
    "candidates_from_hits",
]
