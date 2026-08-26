"""Reranking: the second-stage scorer, and the registry that selects it.

Layout:

``base``
    :class:`Reranker`, which owns ordering, tie-breaking, truncation and top-N so
    that no backend can get those subtly wrong, plus :class:`NoOpReranker`.
``heuristic``
    Zero-dependency lexical coverage/proximity scorer. The default.
``cross_encoder``
    ``sentence-transformers`` cross-encoder. Optional dependency, lazily loaded.
``llm``
    Instruction-following model scoring, with the provider call injected so this
    package never imports the LLM layer.

Every backend is optional and ``none`` is a first-class setting, because a stage
that cannot be switched off cannot be measured. ``docs/EVALUATION.md`` reports
retrieval quality with each of them over the same query set.
"""

from __future__ import annotations

from typing import Any, Callable

from kip.core.rerank.base import (
    MAX_CANDIDATE_CHARS,
    NoOpReranker,
    RerankCandidate,
    RerankError,
    Reranker,
    RerankResult,
    candidates_from_hits,
)
from kip.core.rerank.heuristic import (
    PRIOR_RANK_DAMPING,
    PROXIMITY_HORIZON,
    WEIGHT_COVERAGE,
    WEIGHT_HEADING,
    WEIGHT_PHRASE,
    WEIGHT_PRIOR,
    WEIGHT_PROXIMITY,
    HeuristicReranker,
    rank_prior,
)

__all__ = [
    "MAX_CANDIDATE_CHARS",
    "PRIOR_RANK_DAMPING",
    "PROXIMITY_HORIZON",
    "RERANKERS",
    "RERANKER_NOTES",
    "WEIGHT_COVERAGE",
    "WEIGHT_HEADING",
    "WEIGHT_PHRASE",
    "WEIGHT_PRIOR",
    "WEIGHT_PROXIMITY",
    "CrossEncoderReranker",
    "HeuristicReranker",
    "LlmReranker",
    "NoOpReranker",
    "RerankCandidate",
    "RerankError",
    "RerankResult",
    "Reranker",
    "candidates_from_hits",
    "describe_rerankers",
    "get_reranker",
    "rank_prior",
]


def __getattr__(name: str) -> Any:
    """Import the optional backends lazily.

    ``cross_encoder`` is cheap to import but its *construction* is not, and
    ``llm`` is only usable once a provider callable exists. Keeping both out of
    the eager import path means selecting ``RERANKER=heuristic`` never pays for
    either.
    """
    if name == "CrossEncoderReranker":
        from kip.core.rerank.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker
    if name == "LlmReranker":
        from kip.core.rerank.llm import LlmReranker

        return LlmReranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _none(**kwargs: Any) -> Reranker:
    return NoOpReranker()


def _heuristic(**kwargs: Any) -> Reranker:
    return HeuristicReranker()


def _cross_encoder(**kwargs: Any) -> Reranker:
    from kip.core.rerank.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(
        model=str(kwargs.get("model") or ""),
        client=kwargs.get("client"),
    )


def _llm(**kwargs: Any) -> Reranker:
    from kip.core.rerank.llm import LlmReranker

    score_fn = kwargs.get("score_fn")
    if score_fn is None:
        raise RerankError(
            "RERANKER=llm needs a scoring callable. The service layer supplies it "
            "as get_reranker(settings, score_fn=...); the rerank package does not "
            "import the LLM providers itself. Use RERANKER=heuristic for a "
            "reranker with no such dependency."
        )
    return LlmReranker(score_fn=score_fn, model=str(kwargs.get("model") or ""))


#: Backend name -> factory. ``none`` is present deliberately: it is the control
#: condition, not a missing entry.
RERANKERS: dict[str, Callable[..., Reranker]] = {
    "none": _none,
    "heuristic": _heuristic,
    "cross-encoder": _cross_encoder,
    "llm": _llm,
}

#: Surfaced on the Settings screen, so the cost of each choice is visible where
#: the choice is made.
RERANKER_NOTES: dict[str, str] = {
    "heuristic": (
        "Lexical coverage, term proximity and phrase match, blended with the "
        "retriever's rank. No dependencies, no network, sub-millisecond."
    ),
    "cross-encoder": (
        "Transformer cross-encoder scoring each query/passage pair jointly. "
        "Strongest local option; needs sentence-transformers and ~80 MB of model."
    ),
    "llm": (
        "An instruction-following model rates each passage. Highest quality on "
        "indirect questions; adds one API call and passage text egress per query."
    ),
    "none": "Reranking disabled. The retriever's fused ordering is used as-is.",
}


def get_reranker(settings: Any = None, **overrides: Any) -> Reranker:
    """Build the configured reranker.

    Unlike :func:`kip.core.retrieval.get_keyword_index`, this never returns
    ``None``: ``none`` maps to :class:`NoOpReranker`, so the service layer always
    has an object to call and does not need a branch for the disabled case.

    >>> class S:
    ...     reranker = "heuristic"
    ...     reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    >>> get_reranker(S()).name
    'heuristic'
    >>> class Off:
    ...     reranker = "none"
    >>> get_reranker(Off()).name
    'none'

    ``RERANKER`` casing and stray whitespace are tolerated, since these arrive
    from an environment file edited by hand:

    >>> class Messy:
    ...     reranker = "  Cross-Encoder "
    ...     reranker_model = "stub-model"
    >>> get_reranker(Messy()).describe()["model"]
    'stub-model'

    An unrecognised value raises rather than silently falling back, because a
    typo that quietly disabled reranking would show up only as slightly worse
    answers:

    >>> class Bad:
    ...     reranker = "colbert"
    >>> get_reranker(Bad())
    Traceback (most recent call last):
        ...
    kip.core.rerank.base.RerankError: Unknown RERANKER 'colbert'. Available: cross-encoder, heuristic, llm, none.

    The LLM backend refuses to be built without its injected callable, instead of
    constructing an object that fails on the first question asked:

    >>> class UsesLlm:
    ...     reranker = "llm"
    ...     reranker_model = "gpt-4o-mini"
    >>> get_reranker(UsesLlm())
    Traceback (most recent call last):
        ...
    kip.core.rerank.base.RerankError: RERANKER=llm needs a scoring callable. ...
    >>> get_reranker(UsesLlm(), score_fn=lambda prompt: '{"1": 5}').name
    'llm'
    """
    if settings is None:
        from kip.config import get_settings

        settings = get_settings()

    name = str(overrides.pop("backend", None) or getattr(settings, "reranker", "heuristic"))
    name = name.strip().lower()
    if name in {"off", "disabled", ""}:
        name = "none"

    factory = RERANKERS.get(name)
    if factory is None:
        raise RerankError(
            f"Unknown RERANKER {name!r}. Available: " + ", ".join(sorted(RERANKERS)) + "."
        )

    kwargs: dict[str, Any] = {}
    if name in {"cross-encoder", "llm"}:
        kwargs["model"] = getattr(settings, "reranker_model", "")
    kwargs.update(overrides)
    return factory(**kwargs)


def describe_rerankers() -> list[dict[str, str]]:
    """Reranker catalogue for the Settings screen."""
    return [
        {"name": name, "note": RERANKER_NOTES.get(name, "")}
        for name in ("heuristic", "cross-encoder", "llm", "none")
    ]
