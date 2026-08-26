"""Extractive answerer -- the default ``LLM_PROVIDER``, and a real baseline.

This backend does not generate text. It selects the sentences from the retrieved
passages that best match the question, quotes them verbatim, and attaches the
citation marker of the passage each one came from. Nothing it outputs can be
absent from the source documents.

Why this is the default rather than a hosted model:

* **The platform runs end to end with no API key and no network.** Clone, start,
  upload, ask, get a cited answer. A default that required a credential would
  make the first experience a configuration error.
* **It is a faithfulness control, not a placeholder.** Because it can only quote,
  its groundedness is 1.0 by construction. The evaluation harness reports it
  alongside the hosted providers so that a generative model's faithfulness score
  has a ceiling to be compared against. A number with no control condition is
  hard to interpret.
* **It is deterministic.** The same question over the same corpus produces the
  same answer, which is what makes retrieval changes measurable: if the answer
  moves, retrieval moved, not the sampler.

What it cannot do, stated plainly because the README must not overclaim: it
cannot paraphrase, synthesise across passages into new prose, resolve
contradictions, or answer a question whose answer is implied but never stated.
Those are the reasons to configure a hosted model. It is a strong baseline, not a
substitute for generation.
"""

from __future__ import annotations

from typing import Any, Sequence

from kip.core.llm.base import LlmClient, LlmResponse, Message, Usage
from kip.core.text import (
    containment,
    count_tokens,
    jaccard,
    split_sentences,
    stemmed_content_tokens,
)

#: A sentence must contain at least this fraction of the question's content
#: terms to be quoted. Set low enough that a partial lexical match still counts
#: (few sentences restate a whole question), high enough that a passage which
#: merely shares one common word is not offered as an answer.
MIN_SENTENCE_OVERLAP = 0.30

#: Two selected sentences this similar are treated as the same statement. Chunk
#: overlap means adjacent chunks legitimately share sentences, so without this
#: the answer would repeat itself with two different citation markers.
DUPLICATE_JACCARD = 0.80

#: Sentences carry a small bonus for coming from a better-ranked passage, so that
#: an equally-matching sentence from the top passage wins. Kept far below the
#: overlap signal: rank is a tiebreaker, not evidence.
RANK_BONUS = 0.06

DEFAULT_MAX_SENTENCES = 6
DEFAULT_MODEL = "kip-extractive-v1"

#: Signals "the passages do not answer this". The RAG layer owns the user-facing
#: wording so that every provider's version of this outcome reads identically.
INSUFFICIENT = "insufficient_evidence"


def score_sentence(query_terms: Sequence[str], sentence: str, *, rank: int) -> float:
    """Relevance of one sentence to the question.

    >>> terms = stemmed_content_tokens("drying temperature for mango")
    >>> round(score_sentence(terms, "Mango slices are dried at 60 C.", rank=1), 4)
    0.7267
    >>> round(score_sentence(terms, "The packaging line was recommissioned.", rank=1), 4)
    0.06
    >>> score_sentence(terms, "", rank=1)
    0.0

    A better-ranked passage breaks a tie but cannot manufacture relevance:

    >>> a = score_sentence(terms, "Mango is dried.", rank=1)
    >>> b = score_sentence(terms, "Mango is dried.", rank=9)
    >>> a > b, round(a - b, 4)
    (True, 0.0533)
    """
    if not sentence.strip() or not query_terms:
        return 0.0
    sentence_terms = stemmed_content_tokens(sentence)
    if not sentence_terms:
        return 0.0
    overlap = containment(query_terms, sentence_terms)
    return overlap + RANK_BONUS / max(1, int(rank))


def cite(sentence: str, marker: int) -> str:
    """Attach a citation marker inside the sentence it supports.

    The marker goes before the terminating punctuation, which is the convention
    :mod:`kip.core.rag.prompts` asks generative providers to follow. Matching it
    matters for more than tidiness: a marker written after the full stop is a
    separate fragment to a sentence splitter, so the claim would be analysed as
    uncited and this backend -- the one provider that is grounded by construction
    -- would score worst on citation coverage.

    >>> cite("Mango slices are dried at 60 C.", 1)
    'Mango slices are dried at 60 C [1].'
    >>> cite("Is the belt food-grade?", 2)
    'Is the belt food-grade [2]?'
    >>> cite("Table 4 lists the values", 3)
    'Table 4 lists the values [3]'
    """
    body = str(sentence).rstrip()
    if body and body[-1] in ".!?" and not body.endswith(".."):
        return f"{body[:-1].rstrip()} [{marker}]{body[-1]}"
    return f"{body} [{marker}]"


class ExtractiveClient(LlmClient):
    """Answers by quoting the best-matching sentences from the context.

    >>> client = ExtractiveClient()
    >>> passages = [
    ...     (1, "Hot air drying of mango slices is carried out at 60 C. "
    ...         "The tunnel was installed in 1998."),
    ...     (2, "Water activity below 0.6 inhibits microbial growth."),
    ... ]
    >>> reply = client.complete(
    ...     [Message("user", "At what temperature are mango slices dried?")],
    ...     passages=passages,
    ... )
    >>> reply.text
    'Hot air drying of mango slices is carried out at 60 C [1].'
    >>> reply.provider, reply.finish_reason
    ('extractive', 'stop')
    >>> reply.meta["sentences"], reply.meta["passages_used"]
    (1, [1])

    Irrelevant sentences in a cited passage are left out rather than padded in:

    >>> "1998" in reply.text
    False

    A question the passages do not address yields no answer and says so, instead
    of quoting the closest thing available:

    >>> miss = client.complete(
    ...     [Message("user", "What is the tensile strength of the conveyor belt?")],
    ...     passages=passages,
    ... )
    >>> miss.text, miss.finish_reason
    ('', 'insufficient_evidence')

    Every quoted sentence keeps the marker of the passage it came from:

    >>> both = client.complete(
    ...     [Message("user", "mango drying and water activity")], passages=passages)
    >>> both.meta["passages_used"]
    [1, 2]
    >>> both.text.count("[1]"), both.text.count("[2]")
    (1, 1)

    Usage is reported as an estimate, never as a metered figure:

    >>> reply.usage.estimated
    True

    With no passages there is nothing to extract:

    >>> client.complete([Message("user", "anything")]).finish_reason
    'insufficient_evidence'
    """

    name = "extractive"
    extractive = True

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_sentences: int = DEFAULT_MAX_SENTENCES,
        min_overlap: float = MIN_SENTENCE_OVERLAP,
        max_output_tokens: int = 900,
        **kwargs: Any,
    ) -> None:
        # temperature is accepted and ignored: selection is deterministic. It is
        # not rejected, because a caller switching providers should not have to
        # strip parameters that no longer apply.
        kwargs.pop("temperature", None)
        super().__init__(model=model, max_output_tokens=max_output_tokens, **kwargs)
        self.max_sentences = max(1, int(max_sentences))
        self.min_overlap = float(min_overlap)

    def _generate(
        self,
        messages: Sequence[Message],
        *,
        passages: Sequence[tuple[int, str]],
        temperature: float,
        max_output_tokens: int,
    ) -> LlmResponse:
        question = _last_question(messages)
        query_terms = stemmed_content_tokens(question)
        scored = self._select(query_terms, passages, max_output_tokens)

        if not scored:
            return LlmResponse(
                text="",
                provider=self.name,
                model=self.model,
                finish_reason=INSUFFICIENT,
                meta={"sentences": 0, "passages_used": [], "candidates": len(passages)},
            )

        # Restore document order: reading follows the passages, not the scores.
        scored.sort(key=lambda item: (item[1], item[2]))
        text = " ".join(cite(sentence, marker) for _, marker, _, sentence in scored)
        used = sorted({marker for _, marker, _, _ in scored})
        return LlmResponse(
            text=text,
            provider=self.name,
            model=self.model,
            usage=Usage(
                prompt_tokens=sum(count_tokens(str(body)) for _, body in passages),
                completion_tokens=count_tokens(text),
                estimated=True,
            ),
            finish_reason="stop",
            meta={
                "sentences": len(scored),
                "passages_used": used,
                "candidates": len(passages),
            },
        )

    def _select(
        self,
        query_terms: Sequence[str],
        passages: Sequence[tuple[int, str]],
        max_output_tokens: int,
    ) -> list[tuple[float, int, int, str]]:
        """Pick the sentences to quote, as ``(score, marker, position, text)``."""
        candidates: list[tuple[float, int, int, str]] = []
        for rank, (marker, body) in enumerate(passages, start=1):
            for position, sentence in enumerate(split_sentences(str(body))):
                score = score_sentence(query_terms, sentence, rank=rank)
                if score >= self.min_overlap:
                    candidates.append((score, int(marker), position, sentence.strip()))

        # Best first, then by passage rank and position so the choice is total and
        # does not depend on the order the corpus happened to be indexed in.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        chosen: list[tuple[float, int, int, str]] = []
        budget = max(1, int(max_output_tokens))
        spent = 0
        for candidate in candidates:
            if len(chosen) >= self.max_sentences:
                break
            terms = stemmed_content_tokens(candidate[3])
            if any(
                jaccard(terms, stemmed_content_tokens(taken[3])) >= DUPLICATE_JACCARD
                for taken in chosen
            ):
                continue
            # +4 leaves room for the " [12]" marker this sentence will carry.
            cost = count_tokens(candidate[3]) + 4
            if spent + cost > budget and chosen:
                break
            chosen.append(candidate)
            spent += cost
        return chosen

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "max_sentences": self.max_sentences,
                "min_overlap": self.min_overlap,
                "note": (
                    "Quotes matching sentences from the retrieved passages. "
                    "Grounded by construction; cannot paraphrase or synthesise."
                ),
            }
        )
        return info


def _last_question(messages: Sequence[Message]) -> str:
    """The most recent user turn -- what is actually being asked.

    Earlier turns are deliberately ignored. A hosted model can use conversation
    history to resolve "and at what humidity?"; a sentence matcher cannot, and
    folding history into the match terms would drag in words from previous
    questions and quietly degrade the selection.

    >>> _last_question([Message("user", "First?"), Message("assistant", "..."),
    ...                 Message("user", "Second?")])
    'Second?'
    >>> _last_question([Message("system", "Be exact.")])
    ''
    """
    for message in reversed(list(messages)):
        if message.role == "user" and message.content:
            return message.content
    return ""


__all__ = [
    "DEFAULT_MAX_SENTENCES",
    "DUPLICATE_JACCARD",
    "ExtractiveClient",
    "INSUFFICIENT",
    "MIN_SENTENCE_OVERLAP",
    "RANK_BONUS",
    "cite",
    "score_sentence",
]
