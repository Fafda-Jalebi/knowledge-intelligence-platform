"""Grounding checks -- two gates that make "I don't know" a real outcome.

A retrieval-augmented system that always answers has not solved hallucination; it
has moved it somewhere harder to see. These two checks are how the platform earns
the claim that its answers are grounded.

**Before generation: the evidence gate.** If retrieval found nothing similar
enough to the question, the model is never called. This matters for a reason that
is easy to miss: a fused ranking score always has a top value. Reciprocal rank
fusion will happily report a best hit of 0.016 for a question about tensile
strength asked of a corpus about drying, because "best of what we found" is
well-defined even when everything found was irrelevant. So the gate reads the raw
cosine similarity instead, which is a statement about the query and the passage
rather than about the ranking. See :func:`check_evidence`.

**After generation: the support check.** Each sentence of the answer is compared
against the passages it cites. The result is reported, not enforced -- and the
distinction is deliberate. The comparison is lexical (stemmed term containment),
which is a *proxy* for entailment, and a faithful paraphrase can score low while a
fluent fabrication that reuses the passage's vocabulary can score high. Treating
that proxy as a verdict would silently suppress correct answers. So the check
annotates: the interface can mark a weakly-supported sentence, the evaluation
harness can measure groundedness over a dataset, and
:attr:`GroundingReport.advises_refusal` flags only the unambiguous case where
*nothing* in the answer traces back to any supplied passage.

The thresholds are not universal constants. They were chosen by measurement on
this platform's evaluation set with its default embedding model, and both are
configurable for that reason. ``docs/EVALUATION.md`` reports the sweep behind the
defaults and the false-positive rate that comes with them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from kip.core.rag.citations import extract_markers, split_cited_sentences, strip_markers
from kip.core.rag.context import ContextBlock, ContextPassage
from kip.core.text import containment, stem_tokens, stemmed_content_tokens

#: Why the evidence gate refused. Recorded rather than collapsed into a boolean so
#: the interface can say *which* condition failed, and evaluation can separate
#: "retrieval returned nothing" from "retrieval returned something irrelevant".
REASON_NO_PASSAGES = "no_passages"
REASON_TOO_FEW_PASSAGES = "too_few_passages"
REASON_WEAK_MATCH = "weak_match"

#: A sentence whose best-matching passage beats its cited passage by this margin
#: is flagged as possibly citing the wrong number. Wide, because the lexical score
#: is noisy and a narrow margin would generate false accusations.
MISATTRIBUTION_MARGIN = 0.25

#: Connectives that survive stopword removal but assert nothing on their own. A
#: sentence built only from these ("However, therefore.") is excluded from the
#: groundedness denominator.
#:
#: This set is deliberately tiny and must stay that way. Excluding a sentence
#: *raises* the groundedness score, so every word added here is a way to make the
#: metric look better without making the answers better. Nothing that could ever
#: be an answer belongs in it -- "Sixty degrees." is a one-word sentence and a
#: complete, checkable claim. "additionally" and "alternatively" are absent for
#: exactly that reason: they stem to ``addition`` and ``altern``, colliding with
#: the content nouns "addition" and "alternative".
DISCOURSE_TERMS = frozenset(
    stem_tokens(
        [
            "also",
            "consequently",
            "conversely",
            "furthermore",
            "however",
            "meanwhile",
            "moreover",
            "nevertheless",
            "nonetheless",
            "overall",
            "similarly",
            "therefore",
            "thus",
        ]
    )
)


@dataclass(frozen=True, slots=True)
class EvidenceCheck:
    """Outcome of the pre-generation gate."""

    sufficient: bool
    reason: str = ""
    top_score: float | None = None
    passages: int = 0
    min_score: float = 0.0
    min_passages: int = 1
    #: True when retrieval produced no similarity at all (keyword-only mode), so
    #: the score threshold could not be applied.
    score_unavailable: bool = False

    def __bool__(self) -> bool:
        return self.sufficient

    @property
    def explanation(self) -> str:
        """Plain-language reason, for the API and the Chat view.

        >>> EvidenceCheck(False, REASON_NO_PASSAGES).explanation
        'No passages were retrieved for this question.'
        >>> print(EvidenceCheck(False, REASON_WEAK_MATCH, top_score=0.04,
        ...                     min_score=0.16).explanation)
        The closest passage scored 0.04 against a minimum of 0.16, which is too
        weak to answer from.
        >>> EvidenceCheck(True).explanation
        ''
        """
        if self.sufficient:
            return ""
        if self.reason == REASON_NO_PASSAGES:
            return "No passages were retrieved for this question."
        if self.reason == REASON_TOO_FEW_PASSAGES:
            return (
                f"Only {self.passages} passage(s) were retrieved, below the "
                f"minimum of {self.min_passages}."
            )
        if self.reason == REASON_WEAK_MATCH:
            return (
                f"The closest passage scored {self.top_score:.2f} against a "
                f"minimum of {self.min_score:.2f}, which is too\nweak to answer "
                "from."
            )
        return "The retrieved passages are not sufficient to answer this question."

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "reason": self.reason,
            "explanation": self.explanation,
            "top_score": None if self.top_score is None else round(self.top_score, 6),
            "passages": self.passages,
            "min_score": self.min_score,
            "min_passages": self.min_passages,
            "score_unavailable": self.score_unavailable,
        }


@dataclass(frozen=True, slots=True)
class SentenceSupport:
    """How well one answer sentence is supported by the passages it cites."""

    text: str
    supported: bool
    #: Best containment against the passages this sentence cites.
    score: float = 0.0
    #: Markers the sentence carries.
    markers: tuple[int, ...] = ()
    #: Best containment against *any* supplied passage, and which one.
    best_score: float = 0.0
    best_marker: int | None = None
    #: False for sentences that assert nothing measurable (no content words), which
    #: are excluded from the groundedness denominator rather than counted as
    #: failures.
    claim: bool = True

    @property
    def misattributed(self) -> bool:
        """True when a different passage supports this sentence much better.

        >>> SentenceSupport("x", True, score=0.3, markers=(1,), best_score=0.9,
        ...                 best_marker=2).misattributed
        True
        >>> SentenceSupport("x", True, score=0.8, markers=(1,), best_score=0.9,
        ...                 best_marker=2).misattributed
        False
        >>> SentenceSupport("x", True, score=0.0, best_score=0.9,
        ...                 best_marker=2).misattributed
        False
        """
        if not self.markers or self.best_marker is None:
            return False
        if self.best_marker in self.markers:
            return False
        return (self.best_score - self.score) >= MISATTRIBUTION_MARGIN

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "supported": self.supported,
            "score": round(self.score, 4),
            "markers": list(self.markers),
            "best_score": round(self.best_score, 4),
            "best_marker": self.best_marker,
            "claim": self.claim,
            "misattributed": self.misattributed,
        }


@dataclass(frozen=True, slots=True)
class GroundingReport:
    """Per-sentence support analysis for one answer."""

    sentences: tuple[SentenceSupport, ...] = ()
    threshold: float = 0.0
    enforce_citations: bool = True
    refused: bool = False

    def __len__(self) -> int:
        return len(self.sentences)

    @property
    def claims(self) -> tuple[SentenceSupport, ...]:
        return tuple(sentence for sentence in self.sentences if sentence.claim)

    @property
    def unsupported(self) -> tuple[SentenceSupport, ...]:
        return tuple(s for s in self.claims if not s.supported)

    @property
    def misattributed(self) -> tuple[SentenceSupport, ...]:
        return tuple(s for s in self.claims if s.misattributed)

    @property
    def groundedness(self) -> float:
        """Fraction of claim sentences supported by their cited passages.

        A refusal is 1.0: it asserts nothing, so nothing is ungrounded.

        >>> GroundingReport(refused=True).groundedness
        1.0
        >>> GroundingReport().groundedness
        0.0
        """
        if self.refused:
            return 1.0
        claims = self.claims
        if not claims:
            return 0.0
        supported = sum(1 for sentence in claims if sentence.supported)
        return round(supported / len(claims), 4)

    @property
    def advises_refusal(self) -> bool:
        """True only when no claim in the answer traces to any passage.

        This is the one unambiguous failure the lexical proxy can detect: not "the
        wording differs from the source" but "no supplied passage shares any
        vocabulary with anything the answer asserts". A paraphrase scores low; a
        fabrication scores zero everywhere.

        >>> weak = SentenceSupport("Shelf life is 12 months.", False, score=0.0,
        ...                        best_score=0.0)
        >>> GroundingReport(sentences=(weak,)).advises_refusal
        True
        >>> partial = SentenceSupport("Dried at 60 C.", True, score=0.9,
        ...                           best_score=0.9, best_marker=1)
        >>> GroundingReport(sentences=(weak, partial)).advises_refusal
        False
        >>> GroundingReport(refused=True).advises_refusal
        False
        """
        if self.refused:
            return False
        claims = self.claims
        if not claims:
            return False
        return all(sentence.best_score <= 0.0 for sentence in claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "groundedness": self.groundedness,
            "threshold": self.threshold,
            "enforce_citations": self.enforce_citations,
            "refused": self.refused,
            "claims": len(self.claims),
            "unsupported": [s.to_dict() for s in self.unsupported],
            "misattributed": [s.to_dict() for s in self.misattributed],
            "sentences": [s.to_dict() for s in self.sentences],
            "advises_refusal": self.advises_refusal,
        }


def is_claim(terms: Sequence[str]) -> bool:
    """Whether a sentence's content terms assert something checkable.

    >>> is_claim(stemmed_content_tokens("Mango is dried at 60 C."))
    True
    >>> is_claim(stemmed_content_tokens("Sixty degrees."))
    True
    >>> is_claim(stemmed_content_tokens("However, therefore."))
    False
    >>> is_claim(stemmed_content_tokens("Also, similarly:"))
    False
    >>> is_claim([])
    False

    A connective in front of a real claim does not disqualify it:

    >>> is_claim(stemmed_content_tokens("However, the temperature was 60 C."))
    True
    """
    if not terms:
        return False
    return any(term not in DISCOURSE_TERMS for term in terms)


def check_evidence(
    retrieval: Any,
    *,
    min_score: float = 0.16,
    min_passages: int = 1,
) -> EvidenceCheck:
    """Decide whether retrieval found enough to attempt an answer.

    ``retrieval`` is anything exposing ``__len__`` and ``top_dense_score`` -- a
    :class:`~kip.core.retrieval.hybrid.RetrievalResult` in the pipeline, and a
    stand-in in tests.

    >>> class R:
    ...     def __init__(self, n, score): self.n, self.top_dense_score = n, score
    ...     def __len__(self): return self.n
    >>> check_evidence(R(5, 0.61)).sufficient
    True
    >>> gate = check_evidence(R(5, 0.04))
    >>> gate.sufficient, gate.reason
    (False, 'weak_match')
    >>> check_evidence(R(0, None)).reason
    'no_passages'
    >>> check_evidence(R(1, 0.9), min_passages=3).reason
    'too_few_passages'

    Exactly at the threshold passes -- the setting reads as a minimum, and a
    boundary that excluded its own value would be a surprise:

    >>> check_evidence(R(1, 0.16), min_score=0.16).sufficient
    True

    Keyword-only retrieval produces no similarities, so the score threshold cannot
    be applied and is skipped rather than failing every question:

    >>> gate = check_evidence(R(3, None))
    >>> gate.sufficient, gate.score_unavailable
    (True, True)
    """
    count = len(retrieval) if retrieval is not None else 0
    top = getattr(retrieval, "top_dense_score", None)
    floor = float(min_score)
    minimum = max(1, int(min_passages))

    if count <= 0:
        return EvidenceCheck(
            False,
            REASON_NO_PASSAGES,
            top_score=top,
            passages=0,
            min_score=floor,
            min_passages=minimum,
            score_unavailable=top is None,
        )
    if count < minimum:
        return EvidenceCheck(
            False,
            REASON_TOO_FEW_PASSAGES,
            top_score=top,
            passages=count,
            min_score=floor,
            min_passages=minimum,
            score_unavailable=top is None,
        )
    if top is None:
        return EvidenceCheck(
            True,
            top_score=None,
            passages=count,
            min_score=floor,
            min_passages=minimum,
            score_unavailable=True,
        )
    if float(top) < floor:
        return EvidenceCheck(
            False,
            REASON_WEAK_MATCH,
            top_score=float(top),
            passages=count,
            min_score=floor,
            min_passages=minimum,
        )
    return EvidenceCheck(
        True,
        top_score=float(top),
        passages=count,
        min_score=floor,
        min_passages=minimum,
    )


def check_support(
    answer: str,
    block: ContextBlock | Sequence[ContextPassage],
    *,
    threshold: float = 0.32,
    enforce_citations: bool = True,
    refused: bool = False,
) -> GroundingReport:
    """Score each answer sentence against the passages it cites.

    >>> from kip.core.rag.context import ContextBuilder
    >>> from kip.core.rerank.base import RerankResult
    >>> ranked = [
    ...     RerankResult("d1:0", 0.9, 0.5, prior_rank=1, rank=1,
    ...                  text="Mango slices are dried at 60 C for eight hours.",
    ...                  payload={"document_id": "d1", "chunk_index": 0}),
    ...     RerankResult("d2:1", 0.8, 0.4, prior_rank=2, rank=2,
    ...                  text="Water activity below 0.6 inhibits microbial growth.",
    ...                  payload={"document_id": "d2", "chunk_index": 1}),
    ... ]
    >>> block = ContextBuilder().build(ranked)
    >>> report = check_support("Mango slices are dried at 60 C [1].", block)
    >>> report.groundedness, report.unsupported
    (1.0, ())

    A marker written after the full stop still counts for the claim it follows,
    because many models cite that way and penalising the style would understate
    groundedness:

    >>> check_support("Mango slices are dried at 60 C. [1]", block).groundedness
    1.0

    A sentence with no basis in the passage it cites is flagged:

    >>> report = check_support("Shelf life is 12 months in foil [1].", block)
    >>> report.groundedness, len(report.unsupported)
    (0.0, 1)
    >>> report.sentences[0].score < 0.32
    True

    With ``enforce_citations`` an uncited claim is unsupported however well its
    words match, because an uncheckable claim is the thing the platform promises
    not to present as grounded:

    >>> check_support("Mango slices are dried at 60 C.", block).groundedness
    0.0
    >>> check_support("Mango slices are dried at 60 C.", block,
    ...               enforce_citations=False).groundedness
    1.0

    Citing the wrong passage is detected separately from being unsupported -- the
    claim is true of the corpus, just attributed to the wrong number:

    >>> report = check_support("Water activity below 0.6 inhibits growth [1].", block)
    >>> [s.best_marker for s in report.sentences], len(report.misattributed)
    ([2], 1)

    Sentences that assert nothing measurable are excluded rather than counted as
    ungrounded:

    >>> report = check_support("However, therefore [1]. Dried at 60 C [1].", block)
    >>> len(report.sentences), len(report.claims), report.groundedness
    (2, 1, 1.0)

    A refusal is not analysed:

    >>> from kip.core.rag.prompts import INSUFFICIENT_EVIDENCE
    >>> report = check_support(INSUFFICIENT_EVIDENCE, block, refused=True)
    >>> report.refused, report.groundedness, report.sentences
    (True, 1.0, ())

    An empty answer produces an empty report:

    >>> check_support("", block).sentences
    ()
    """
    passages = block.passages if isinstance(block, ContextBlock) else tuple(block)
    text = str(answer or "").strip()
    if refused or not text or not passages:
        return GroundingReport(
            threshold=float(threshold),
            enforce_citations=bool(enforce_citations),
            refused=bool(refused),
        )

    terms_by_marker: dict[int, list[str]] = {
        passage.marker: stemmed_content_tokens(passage.text) for passage in passages
    }

    analysed: list[SentenceSupport] = []
    for sentence in split_cited_sentences(text):
        markers = tuple(m for m in extract_markers(sentence) if m in terms_by_marker)
        claim_terms = stemmed_content_tokens(strip_markers(sentence))
        if not is_claim(claim_terms):
            analysed.append(
                SentenceSupport(sentence.strip(), True, markers=markers, claim=False)
            )
            continue

        best_score = 0.0
        best_marker: int | None = None
        for marker, passage_terms in terms_by_marker.items():
            score = containment(claim_terms, passage_terms)
            if score > best_score:
                best_score, best_marker = score, marker

        cited_score = max(
            (containment(claim_terms, terms_by_marker[m]) for m in markers), default=0.0
        )
        if markers:
            supported = cited_score >= float(threshold)
        elif enforce_citations:
            supported = False
        else:
            supported = best_score >= float(threshold)

        analysed.append(
            SentenceSupport(
                text=sentence.strip(),
                supported=supported,
                score=cited_score if markers else best_score,
                markers=markers,
                best_score=best_score,
                best_marker=best_marker,
            )
        )

    return GroundingReport(
        sentences=tuple(analysed),
        threshold=float(threshold),
        enforce_citations=bool(enforce_citations),
        refused=False,
    )


def describe_thresholds(settings: Any) -> dict[str, Any]:
    """The active grounding configuration, for the Settings screen.

    Exposed because a user comparing two answers needs to know the gate they were
    produced under; a threshold that only exists in an environment variable is not
    an auditable one.
    """
    return {
        "min_score": getattr(settings, "grounding_min_score", 0.16),
        "min_passages": getattr(settings, "grounding_min_passages", 1),
        "support_threshold": getattr(settings, "grounding_support_threshold", 0.32),
        "enforce_citations": getattr(settings, "grounding_enforce_citations", True),
        "note": (
            "The support score is stemmed lexical containment, a proxy for "
            "entailment. A faithful paraphrase can score low; the score annotates "
            "an answer, it does not suppress one."
        ),
    }


__all__ = [
    "DISCOURSE_TERMS",
    "EvidenceCheck",
    "GroundingReport",
    "MISATTRIBUTION_MARGIN",
    "REASON_NO_PASSAGES",
    "REASON_TOO_FEW_PASSAGES",
    "REASON_WEAK_MATCH",
    "SentenceSupport",
    "check_evidence",
    "check_support",
    "describe_thresholds",
    "is_claim",
]
