"""LLM-scored reranker -- the most capable and the most expensive option.

A cross-encoder was trained to rank web-search passages. An instruction-following
LLM has no such specialisation, but it can be *told* what relevance means for this
query, and it can reason about a passage that answers the question indirectly --
a table of water-activity limits answering "will this product support mould
growth" -- in a way a similarity model cannot.

The costs are real and are the reason this is not the default: one API call per
query, added latency on every question, and document text leaving the machine.

Design constraints this module is built around
----------------------------------------------
**No dependency on the LLM provider layer.** The scoring callable is injected, so
this module has no import of ``kip.core.llm`` and the two can be developed and
tested independently. Any ``Callable[[str], str]`` works, which also means the
self-checks can exercise the parser with a scripted function and no network.

**Bounded data egress.** Each passage is truncated to ``max_chars`` before it
enters the prompt, honouring the platform rule that whole documents are never
sent to a model -- only the retrieved extract. Candidates are labelled ``[1]``,
``[2]`` rather than by chunk id, which saves tokens and stops the model from
inventing plausible-looking identifiers.

**An unusable response must not silently reorder anything.** LLMs omit items,
wrap output in prose, and occasionally return nothing parseable. Missing scores
are imputed as the mean of the scores that *were* returned, because a missing
judgement is not evidence of irrelevance; and if less than
:data:`MIN_SCORED_RATIO` of the candidates came back scored, the response is
treated as a failure and the retriever's ordering is preserved untouched.

The prompt contains document text and is therefore never logged.
"""

from __future__ import annotations

import json
import re
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

from kip.core.rerank.base import RerankCandidate, Reranker, RerankError

#: Upper bound of the rating scale shown to the model. A 0-10 integer scale asks
#: for less precision than the model can actually justify, which is deliberate:
#: finer scales produce confident-looking decimals that do not survive rephrasing
#: the question.
SCORE_MAX = 10.0

#: Fraction of candidates that must come back scored for the response to be used.
MIN_SCORED_RATIO = 0.5

PROMPT_TEMPLATE = """You are ranking retrieved passages by how well each one helps answer a question.

Question: {query}

Passages:
{passages}

Rate every passage from 0 to {score_max} on how much it helps answer the question:
  0  = unrelated, or mentions the topic without addressing the question
  {score_max:g} = directly and completely answers the question

Judge only what each passage says. Do not use outside knowledge. Do not answer
the question.

Reply with JSON only, mapping each passage number to its rating:
{{"1": 0, "2": {score_max:g}}}"""

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*|\s*```\s*$")
_LINE_SCORE = re.compile(
    r"""(?:^|\n)          # start of a line
        \s*(?:[-*]\s*)?   # optional bullet
        \[?(\d{1,3})\]?   # passage label, with or without brackets
        \s*[:=.\)]?\s*    # optional separator
        (?:score\s*[:=]?\s*)?
        (\d{1,3}(?:\.\d+)?)  # the rating
    """,
    re.VERBOSE,
)


def build_prompt(
    query: str,
    candidates: Sequence[RerankCandidate],
    *,
    max_chars: int,
) -> str:
    """Render the scoring prompt.

    Separate from the reranker so the exact text sent to a provider can be
    inspected, diffed and reviewed without making a call.

    >>> candidates = [RerankCandidate("chunk-7f3ad2:0", "Water activity below 0.6 stops moulds.",
    ...                               0.5, {"heading": "Water activity"}),
    ...               RerankCandidate("chunk-7f3ad2:1", "Harvest scheduling notes.", 0.4)]
    >>> prompt = build_prompt("what stops mould growth", candidates, max_chars=200)
    >>> "[1] (Water activity) Water activity below 0.6 stops moulds." in prompt
    True
    >>> "[2] Harvest scheduling notes." in prompt
    True

    Chunk ids never reach the model -- they cost tokens and invite the model to
    invent plausible-looking identifiers:

    >>> "chunk-7f3ad2" in prompt
    False

    The passage budget is enforced here, not left to the provider:

    >>> long_candidate = [RerankCandidate("c", "x" * 5000, 0.1)]
    >>> len(build_prompt("q", long_candidate, max_chars=50)) < 800
    True
    """
    lines: list[str] = []
    for position, candidate in enumerate(candidates, start=1):
        text = " ".join(candidate.truncated(max_chars).split())
        heading = candidate.heading
        label = f"[{position}] ({heading}) " if heading else f"[{position}] "
        lines.append(label + text)
    return PROMPT_TEMPLATE.format(
        query=query.strip(),
        passages="\n\n".join(lines),
        score_max=SCORE_MAX,
    )


def parse_scores(text: str, count: int) -> dict[int, float]:
    """Extract ``{position: score}`` from a model response.

    Tries JSON first, since that is what the prompt asks for, then falls back to
    line scanning, because models wrap JSON in prose often enough that treating
    it as a hard failure would waste a paid call.

    >>> parse_scores('{"1": 8, "2": 0}', 2)
    {1: 8.0, 2: 0.0}

    Markdown fences and surrounding commentary are tolerated:

    >>> parse_scores('Sure!\\n```json\\n{"1": 9.5, "2": 2}\\n```\\n', 2)
    {1: 9.5, 2: 2.0}
    >>> parse_scores('[1] 7 - directly relevant\\n[2] 1 - off topic', 2)
    {1: 7.0, 2: 1.0}
    >>> parse_scores('- 1: 4\\n- 2: 6', 2)
    {1: 4.0, 2: 6.0}

    Labels outside the candidate range are dropped rather than shifting the rest,
    and ratings are clamped to the scale:

    >>> parse_scores('{"1": 5, "9": 10}', 2)
    {1: 5.0}
    >>> parse_scores('{"1": 42, "2": -3}', 2)
    {1: 10.0, 2: 0.0}

    Nothing parseable yields nothing, rather than a guess:

    >>> parse_scores('I cannot rate these passages.', 2)
    {}
    >>> parse_scores('', 2)
    {}
    """
    if not text or count <= 0:
        return {}

    cleaned = _FENCE.sub("", text.strip())
    scores = _from_json(cleaned, count)
    if scores:
        return scores

    found: dict[int, float] = {}
    for label, value in _LINE_SCORE.findall(cleaned):
        position = int(label)
        if 1 <= position <= count and position not in found:
            found[position] = _clamp(value)
    return found


def _from_json(text: str, count: int) -> dict[int, float]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    found: dict[int, float] = {}
    for key, value in payload.items():
        try:
            position = int(str(key).strip().strip("[]"))
            rating = _clamp(value)
        except (TypeError, ValueError):
            continue
        if 1 <= position <= count:
            found[position] = rating
    return found


def _clamp(value: Any) -> float:
    rating = float(value)
    return max(0.0, min(SCORE_MAX, rating))


class LlmReranker(Reranker):
    """Relevance scoring by an instruction-following model. ``RERANKER=llm``.

    The scoring function is injected, so this class is fully testable offline:

    >>> def fake_llm(prompt):
    ...     return '{"1": 2, "2": 9}'
    >>> candidates = [RerankCandidate("a", "Harvest scheduling notes.", 0.9),
    ...               RerankCandidate("b", "Water activity below 0.6 stops moulds.", 0.5)]
    >>> results = LlmReranker(score_fn=fake_llm).rerank("what stops mould growth", candidates)
    >>> [(r.id, r.score, r.movement) for r in results]
    [('b', 9.0, 1), ('a', 2.0, -1)]

    The rating is reported as a signal so the interface can show why a passage
    moved, and it is normalised to 0-1 there because a raw "9" invites being read
    as a percentage:

    >>> results[0].signals
    {'llm_relevance': 0.9}

    An omitted passage is imputed as the mean of the ratings that were returned,
    not as zero -- a missing judgement says nothing about the passage:

    >>> three = candidates + [RerankCandidate("c", "Retort sterilisation at 121 C.", 0.4)]
    >>> results = LlmReranker(score_fn=lambda p: '{"1": 2, "2": 8}').rerank("q", three)
    >>> [(r.id, r.score) for r in results]
    [('b', 8.0), ('c', 5.0), ('a', 2.0)]

    A response too incomplete to trust preserves the retriever's ordering instead
    of reordering on fragments:

    >>> results = LlmReranker(score_fn=lambda p: '{"1": 9}').rerank("q", three)
    >>> [r.id for r in results]
    ['a', 'b', 'c']
    >>> results[0].signals
    {}

    A provider that fails is an error, not a silent no-op, because a chat request
    that quietly stopped reranking should be visible in the logs:

    >>> def broken(prompt):
    ...     raise TimeoutError("upstream timeout")
    >>> LlmReranker(score_fn=broken).rerank("q", candidates)
    Traceback (most recent call last):
        ...
    kip.core.rerank.base.RerankError: LLM reranker call failed: upstream timeout
    """

    name = "llm"
    calibrated = False

    def __init__(
        self,
        *,
        score_fn: Callable[[str], str],
        model: str = "",
        max_chars: int | None = None,
    ) -> None:
        super().__init__(**({} if max_chars is None else {"max_chars": max_chars}))
        if not callable(score_fn):
            raise RerankError("LlmReranker needs a callable score_fn(prompt) -> text.")
        self._score_fn = score_fn
        self.model = model

    def _evaluate(
        self, query: str, candidates: Sequence[RerankCandidate]
    ) -> tuple[Sequence[float], Sequence[Mapping[str, float]]]:
        prompt = build_prompt(query, candidates, max_chars=self.max_chars)
        try:
            response = self._score_fn(prompt)
        except RerankError:
            raise
        except Exception as exc:
            raise RerankError(f"LLM reranker call failed: {exc}") from exc

        parsed = parse_scores(str(response or ""), len(candidates))
        # A float comparison, not int(): with three candidates, int(0.5 * 3) is 1,
        # which would let a single rating reorder the whole set.
        if len(parsed) < MIN_SCORED_RATIO * len(candidates):
            # Not enough of a judgement to act on. Descending by position keeps
            # the incoming order through the base class's sort.
            fallback = [float(len(candidates) - index) for index in range(len(candidates))]
            return fallback, [{} for _ in candidates]

        imputed = fmean(parsed.values())
        scores = [parsed.get(position, imputed) for position in range(1, len(candidates) + 1)]
        signals = [{"llm_relevance": round(score / SCORE_MAX, 4)} for score in scores]
        return scores, signals

    def _score(self, query: str, candidates: Sequence[RerankCandidate]) -> Sequence[float]:
        return self._evaluate(query, candidates)[0]

    def describe(self) -> dict[str, Any]:
        return {
            "reranker": self.name,
            "model": self.model,
            "calibrated": self.calibrated,
            "score_max": SCORE_MAX,
            "min_scored_ratio": MIN_SCORED_RATIO,
        }


__all__ = [
    "MIN_SCORED_RATIO",
    "PROMPT_TEMPLATE",
    "SCORE_MAX",
    "LlmReranker",
    "build_prompt",
    "parse_scores",
]
