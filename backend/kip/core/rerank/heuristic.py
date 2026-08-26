"""Heuristic reranker -- the zero-dependency default.

This exists because the honest default for a portfolio project should work with
no model download and no API key, and because a cross-encoder is a large
dependency to impose on someone who just wants to run ``docker compose up``.

It scores what a retriever structurally cannot, using five signals that are cheap
to compute and easy to explain:

``coverage`` (0.40)
    Fraction of the query's distinct content terms present in the passage. This
    is the signal BM25 most needs correcting: BM25 sums independent term weights,
    so a passage repeating one rare query term nine times can outrank a passage
    that addresses all five of the query's concepts once each. For a question,
    the second passage is almost always the better answer.

``proximity`` (0.20)
    How tightly the matched terms cluster, measured as the smallest token window
    containing all of them. "Drying temperature for mango" is answered by a
    passage where those terms appear in one sentence, not by one that mentions
    drying in the introduction and mango in a table caption four hundred words
    later. Order-insensitive representations cannot see this at all.

``phrase`` (0.10)
    Adjacent query pairs occurring adjacently in the passage. A weak signal on
    its own -- so it carries a small weight -- but it is the only one that
    rewards word order.

``heading`` (0.05)
    Query terms appearing in the passage's heading or section path. A section
    titled "Water activity and shelf life" is about that subject in a way its
    body text does not always restate, because technical writing does not repeat
    the heading in the paragraph beneath it.

``prior`` (0.25)
    How highly the retriever ranked this passage. Retained deliberately: the
    retriever saw the whole corpus and this stage sees forty passages, so
    discarding its opinion entirely would throw away the only corpus-level
    evidence available.

    It is computed from the candidate's **rank**, not from its score, for two
    reasons. First, after Reciprocal Rank Fusion the incoming score *is* a
    function of rank and carries no magnitude meaning, so normalising it is a
    lossy re-derivation of a number already available exactly. Second, min-max
    normalising a score across the candidate set is degenerate on small sets:
    ``min_max([0.9, 0.8])`` is ``[1.0, 0.0]``, which hands the retriever a
    full-weight veto over a gap of 0.1. :func:`rank_prior` decays smoothly and
    identically whether there are two candidates or forty.

The weights below are defaults, not tuned constants, and this reranker makes no
claim to beat a cross-encoder. ``docs/EVALUATION.md`` measures ``none``,
``heuristic`` and ``cross-encoder`` over the same query set; whatever that
measurement says is what the README reports.

What this is not
----------------
It is lexical. It cannot tell that "thermal processing" and "heat treatment" are
the same idea -- that is the dense retriever's job, and it has already run. This
stage refines an ordering produced with semantic evidence; it does not replace it.

Its scores are also not calibrated: ``calibrated = False``, and nothing
downstream may read a heuristic score as confidence. The insufficient-evidence
check uses raw dense similarity for exactly that reason.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from kip.core.rerank.base import RerankCandidate, Reranker
from kip.core.text import STOP_WORDS, stem, stemmed_content_tokens, tokenize

WEIGHT_COVERAGE = 0.40
WEIGHT_PROXIMITY = 0.20
WEIGHT_PHRASE = 0.10
WEIGHT_HEADING = 0.05
WEIGHT_PRIOR = 0.25

#: Windows wider than this contribute no proximity credit. Roughly a long
#: paragraph: beyond it, co-occurrence stops being evidence that two terms are
#: discussed together.
PROXIMITY_HORIZON = 60

#: Damping constant for :func:`rank_prior`, in the same spirit as RRF's ``k``.
#: Chosen so that one position of retriever preference is worth less than a
#: single strong lexical signal, while ten positions are worth more: adjacent
#: ranks differ by about 0.02 of the final score, ranks 1 and 11 by about 0.13.
PRIOR_RANK_DAMPING = 10


def rank_prior(rank: int, *, damping: int = PRIOR_RANK_DAMPING) -> float:
    """Damped reciprocal rank, 1.0 at the top and decaying without reaching zero.

    Absolute rather than normalised across the candidate set, so a passage's
    prior does not change when its neighbours do.

    >>> [round(rank_prior(r), 4) for r in (1, 2, 3, 11, 40)]
    [1.0, 0.9091, 0.8333, 0.5, 0.2041]

    The gap between adjacent top ranks is small enough that a lexical signal can
    overcome it, which is the entire purpose of a second stage:

    >>> gap = rank_prior(1) - rank_prior(2)
    >>> round(WEIGHT_PRIOR * gap, 4) < WEIGHT_HEADING
    True

    Ranks are 1-based; anything lower is clamped rather than dividing by zero.

    >>> rank_prior(0) == rank_prior(1)
    True
    """
    position = max(1, int(rank))
    return damping / (damping + position - 1)


def term_coverage(query_terms: Sequence[str], passage_terms: Sequence[str]) -> float:
    """Fraction of distinct query terms present in the passage.

    >>> term_coverage(["dry", "mango", "temperatur"], ["dry", "mango", "60"])
    0.6667
    >>> term_coverage(["dry"], ["dry", "dry", "dry"])
    1.0
    >>> term_coverage([], ["dry"])
    0.0
    """
    wanted = set(query_terms)
    if not wanted:
        return 0.0
    present = wanted & set(passage_terms)
    return round(len(present) / len(wanted), 4)


def proximity(query_terms: Sequence[str], passage_terms: Sequence[str]) -> float:
    """How tightly the matched query terms cluster in the passage.

    Finds the smallest token window containing every matched term, then scores it
    against the ideal window (one token per term) and the
    :data:`PROXIMITY_HORIZON`. A single matched term has no meaningful spread, so
    it returns a neutral 0.5 rather than a perfect 1.0 -- otherwise a passage
    matching one word would outscore one matching four words in two sentences.

    >>> proximity(["dry", "mango"], "dry mango slice".split())
    1.0
    >>> proximity(["dry", "mango"], "hot air dry of mango slice".split())
    0.9828
    >>> tight = proximity(["dry", "mango"], "dry mango".split())
    >>> loose = proximity(["dry", "mango"], ["dry"] + ["filler"] * 30 + ["mango"])
    >>> tight > loose
    True
    >>> proximity(["dry"], "dry mango".split())
    0.5
    >>> proximity(["extrus"], "dry mango".split())
    0.0
    """
    wanted = set(query_terms)
    if not wanted or not passage_terms:
        return 0.0

    positions = [
        (index, token) for index, token in enumerate(passage_terms) if token in wanted
    ]
    if not positions:
        return 0.0

    distinct = {token for _, token in positions}
    if len(distinct) == 1:
        return 0.5

    # Two-pointer minimum window covering every distinct matched term.
    needed = len(distinct)
    counts: dict[str, int] = {}
    satisfied = 0
    best = len(passage_terms)
    left = 0
    for index, token in positions:
        counts[token] = counts.get(token, 0) + 1
        if counts[token] == 1:
            satisfied += 1
        while satisfied == needed:
            span = index - positions[left][0] + 1
            best = min(best, span)
            drop = positions[left][1]
            counts[drop] -= 1
            if counts[drop] == 0:
                satisfied -= 1
            left += 1

    ideal = needed
    if best <= ideal:
        return 1.0
    if best >= PROXIMITY_HORIZON:
        return 0.0
    return round(1.0 - (best - ideal) / (PROXIMITY_HORIZON - ideal), 4)


def phrase_overlap(query_terms: Sequence[str], passage_terms: Sequence[str]) -> float:
    """Fraction of adjacent query pairs that appear adjacently in the passage.

    >>> phrase_overlap(["water", "activ"], "reduc water activ below".split())
    1.0
    >>> phrase_overlap(["water", "activ"], "activ of water".split())
    0.0
    >>> phrase_overlap(["water"], "water".split())
    0.0
    """
    if len(query_terms) < 2 or len(passage_terms) < 2:
        return 0.0
    query_pairs = list(zip(query_terms, query_terms[1:]))
    passage_pairs = set(zip(passage_terms, passage_terms[1:]))
    matched = sum(1 for pair in query_pairs if pair in passage_pairs)
    return round(matched / len(query_pairs), 4)


def heading_overlap(query_terms: Sequence[str], heading: str) -> float:
    """Fraction of distinct query terms present in the passage's heading.

    >>> heading_overlap(["water", "activ"], "Water Activity and Shelf Life")
    1.0
    >>> heading_overlap(["water", "activ"], "Introduction")
    0.0
    >>> heading_overlap(["water"], "")
    0.0
    """
    wanted = set(query_terms)
    if not wanted or not heading:
        return 0.0
    # Heading tokens are stemmed but stop words are kept, since a heading is short
    # enough that dropping words can empty it entirely.
    tokens = {stem(token) for token in tokenize(heading)}
    return round(len(wanted & tokens) / len(wanted), 4)


class HeuristicReranker(Reranker):
    """Lexical coverage/proximity reranker. ``RERANKER=heuristic`` (default).

    The case it is built to fix -- a passage that repeats one query term beating a
    passage that actually addresses the whole question:

    >>> candidates = [
    ...     RerankCandidate("noise", "Mango. Mango. Mango. Mango harvest notes.", 0.9),
    ...     RerankCandidate("answer", "Mango slices are dried at 60 C in a hot air drier.", 0.8),
    ... ]
    >>> results = HeuristicReranker().rerank("drying temperature for mango", top_n=1,
    ...                                      candidates=candidates)
    >>> results[0].id
    'answer'
    >>> results[0].movement
    1

    Neither passage is a perfect answer -- neither states a temperature -- and the
    signals say so: coverage is 2/3 because ``temperatur`` is absent, and the
    phrase signal is 0.0 because "drying temperature" appears nowhere. A score of
    0.69 out of a possible 1.0 is a ranking position, not a claim about
    correctness, which is why ``calibrated`` is ``False``.

    >>> results[0].signals["coverage"], results[0].signals["phrase"]
    (0.6667, 0.0)
    >>> HeuristicReranker().calibrated
    False

    Signals are reported, not just the total, so a ranking can be explained:

    >>> sorted(results[0].signals)
    ['coverage', 'heading', 'phrase', 'prior', 'proximity']
    >>> results[0].signals["coverage"] > results[0].signals["phrase"]
    True

    A heading counts toward relevance, and can break a tie between passages whose
    bodies are identical -- which a score-normalised prior could not do, because
    ``min_max`` would have pinned the two priors to 1.0 and 0.0:

    >>> pair = [
    ...     RerankCandidate("plain", "The value must be controlled during storage.", 0.5),
    ...     RerankCandidate("titled", "The value must be controlled during storage.", 0.5,
    ...                     {"heading": "Water activity limits"}),
    ... ]
    >>> [r.id for r in HeuristicReranker().rerank("water activity", pair)]
    ['titled', 'plain']

    An unanswerable query does not crash, and does not invent an ordering:

    >>> [r.id for r in HeuristicReranker().rerank("extrusion cooking", candidates)]
    ['noise', 'answer']

    Weights are constructor arguments, so the evaluation harness can sweep them
    without editing this module:

    >>> flat = HeuristicReranker(coverage_weight=0.0, proximity_weight=0.0,
    ...                          phrase_weight=0.0, heading_weight=0.0)
    >>> [r.id for r in flat.rerank("drying temperature for mango", candidates)]
    ['noise', 'answer']
    """

    name = "heuristic"
    calibrated = False

    def __init__(
        self,
        *,
        coverage_weight: float = WEIGHT_COVERAGE,
        proximity_weight: float = WEIGHT_PROXIMITY,
        phrase_weight: float = WEIGHT_PHRASE,
        heading_weight: float = WEIGHT_HEADING,
        prior_weight: float = WEIGHT_PRIOR,
        max_chars: int = 4000,
    ) -> None:
        # A larger text budget than the base default: this reranker runs locally,
        # costs nothing per character, and coverage improves with more of the
        # passage visible.
        super().__init__(max_chars=max_chars)
        self.weights = {
            "coverage": float(coverage_weight),
            "proximity": float(proximity_weight),
            "phrase": float(phrase_weight),
            "heading": float(heading_weight),
            "prior": float(prior_weight),
        }

    def _components(
        self, query: str, candidates: Sequence[RerankCandidate]
    ) -> list[dict[str, float]]:
        query_terms = stemmed_content_tokens(query)
        if not query_terms:
            # An all-stop-word query ("what is it about") has no content terms.
            # Fall back to the raw stems so the query is degraded, not empty.
            query_terms = [stem(token) for token in tokenize(query) if token in STOP_WORDS]

        rows: list[dict[str, float]] = []
        for rank, candidate in enumerate(candidates, start=1):
            passage_terms = stemmed_content_tokens(candidate.truncated(self.max_chars))
            rows.append(
                {
                    "coverage": term_coverage(query_terms, passage_terms),
                    "proximity": proximity(query_terms, passage_terms),
                    "phrase": phrase_overlap(query_terms, passage_terms),
                    "heading": heading_overlap(query_terms, candidate.heading),
                    "prior": rank_prior(rank),
                }
            )
        return rows

    def _score(self, query: str, candidates: Sequence[RerankCandidate]) -> Sequence[float]:
        return [self._total(row) for row in self._components(query, candidates)]

    def _signals(
        self, query: str, candidates: Sequence[RerankCandidate]
    ) -> Sequence[Mapping[str, float]]:
        return self._components(query, candidates)

    def _evaluate(
        self, query: str, candidates: Sequence[RerankCandidate]
    ) -> tuple[Sequence[float], Sequence[Mapping[str, float]]]:
        rows = self._components(query, candidates)
        return [self._total(row) for row in rows], rows

    def _total(self, row: Mapping[str, float]) -> float:
        return round(sum(self.weights[key] * value for key, value in row.items()), 6)

    def describe(self) -> dict[str, object]:
        return {
            "reranker": self.name,
            "calibrated": self.calibrated,
            "weights": dict(self.weights),
            "prior_rank_damping": PRIOR_RANK_DAMPING,
            "proximity_horizon": PROXIMITY_HORIZON,
        }


__all__ = [
    "HeuristicReranker",
    "PRIOR_RANK_DAMPING",
    "PROXIMITY_HORIZON",
    "WEIGHT_COVERAGE",
    "WEIGHT_HEADING",
    "WEIGHT_PHRASE",
    "WEIGHT_PRIOR",
    "WEIGHT_PROXIMITY",
    "heading_overlap",
    "phrase_overlap",
    "proximity",
    "rank_prior",
    "term_coverage",
]
