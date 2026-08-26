"""Cross-encoder reranker -- the highest-quality local option.

A dense retriever is a *bi*-encoder: it embeds the query and the passage
separately, then compares two fixed vectors. That separation is what makes it
fast enough to search a whole corpus, and also what limits it -- the passage
vector was computed without ever seeing the query, so it has to be a summary of
everything the passage might be asked about.

A cross-encoder concatenates query and passage into one sequence and runs full
attention across the pair, so every passage token can attend to every query
token. It cannot be pre-computed and therefore cannot be used for retrieval, but
over the 24-40 candidates retrieval already produced it is affordable, and it is
consistently the strongest reranking option in the IR literature.

The default model, ``cross-encoder/ms-marco-MiniLM-L-6-v2``, is trained on MS
MARCO passage ranking: six transformer layers, roughly 80 MB, and it runs on CPU.
It emits an unbounded logit, not a probability. Those logits are monotonic within
a single query and meaningless across queries, so :attr:`calibrated` stays
``False`` and no logit is ever shown to a user as a percentage.

Optional by design
------------------
``sentence-transformers`` pulls in ``torch``, which is a very large dependency to
require of someone cloning a portfolio project. It is therefore imported lazily,
inside :meth:`warm_up`, and its absence produces an actionable error naming the
install command and the zero-dependency alternative -- never a bare
``ImportError`` from deep inside a request.
"""

from __future__ import annotations

from typing import Any, Sequence

from kip.core.rerank.base import RerankCandidate, Reranker, RerankError

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Pairs handed to the model at once. Small because each pair is a full forward
#: pass and the reranked set is at most a few dozen items; a larger batch mostly
#: buys memory pressure.
DEFAULT_BATCH_SIZE = 16

_INSTALL_HINT = (
    "The cross-encoder reranker needs sentence-transformers:\n"
    "    pip install sentence-transformers\n"
    "Or set RERANKER=heuristic to use the zero-dependency reranker, "
    "or RERANKER=none to disable reranking."
)


class CrossEncoderReranker(Reranker):
    """Transformer cross-encoder relevance scorer. ``RERANKER=cross-encoder``.

    The model is loaded on first use rather than at construction, so building the
    provider registry never triggers a model download:

    >>> reranker = CrossEncoderReranker(model="stub-model")
    >>> reranker.name
    'cross-encoder'
    >>> reranker.loaded
    False
    >>> reranker.describe()["model"]
    'stub-model'

    Scores are not calibrated, and the class says so rather than leaving callers
    to guess:

    >>> reranker.calibrated
    False

    A pre-loaded model can be injected, which is how the self-checks exercise the
    batching and ordering logic without a 80 MB download. Any object with a
    ``predict(pairs)`` method works, since that is the whole surface used:

    >>> class KeywordCounter:
    ...     def predict(self, pairs, **kwargs):
    ...         return [float(p[1].lower().count(p[0].lower())) for p in pairs]
    >>> reranker = CrossEncoderReranker(model="stub", client=KeywordCounter())
    >>> reranker.loaded
    True
    >>> candidates = [RerankCandidate("a", "Storage notes.", 0.9),
    ...               RerankCandidate("b", "Water activity and water activity limits.", 0.5)]
    >>> results = reranker.rerank("water activity", candidates)
    >>> [(r.id, r.score, r.movement) for r in results]
    [('b', 2.0, 1), ('a', 0.0, -1)]

    Long passages are truncated before scoring, because the model has a fixed
    context window and silently dropping the overflow inside the tokenizer is
    worse than doing it visibly here:

    >>> reranker.max_chars
    1600
    """

    name = "cross-encoder"
    calibrated = False

    def __init__(
        self,
        *,
        model: str = DEFAULT_CROSS_ENCODER_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_chars: int | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(**({} if max_chars is None else {"max_chars": max_chars}))
        self.model = model or DEFAULT_CROSS_ENCODER_MODEL
        self.batch_size = max(1, int(batch_size))
        self._client = client

    @property
    def loaded(self) -> bool:
        """Whether the model is resident. Useful for a readiness probe."""
        return self._client is not None

    def warm_up(self) -> None:
        """Load the model. Called on first use, or eagerly at startup."""
        if self._client is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore import-not-found
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RerankError(_INSTALL_HINT) from exc
        try:
            self._client = CrossEncoder(self.model)
        except Exception as exc:  # pragma: no cover - network/disk dependent
            raise RerankError(
                f"Could not load cross-encoder model {self.model!r}: {exc}. "
                "Check the model name and that the machine can reach the model "
                "cache, or set RERANKER=heuristic."
            ) from exc

    def _score(self, query: str, candidates: Sequence[RerankCandidate]) -> Sequence[float]:
        self.warm_up()
        client = self._client
        if client is None:  # pragma: no cover - warm_up raises instead
            raise RerankError("Cross-encoder model is not loaded.")

        pairs = [(query, candidate.truncated(self.max_chars)) for candidate in candidates]
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            try:
                predicted = client.predict(batch)
            except Exception as exc:  # pragma: no cover - model runtime failure
                raise RerankError(f"Cross-encoder scoring failed: {exc}") from exc
            scores.extend(float(value) for value in predicted)

        if len(scores) != len(candidates):
            raise RerankError(
                f"Cross-encoder returned {len(scores)} scores for {len(candidates)} pairs."
            )
        return scores

    def _signals(
        self, query: str, candidates: Sequence[RerankCandidate]
    ) -> Sequence[dict[str, float]]:
        # The model gives one opaque logit per pair and no decomposition, so
        # reporting invented sub-signals here would be dishonest. The interface
        # shows the score and the rank movement instead.
        return [{} for _ in candidates]

    def describe(self) -> dict[str, Any]:
        return {
            "reranker": self.name,
            "model": self.model,
            "calibrated": self.calibrated,
            "batch_size": self.batch_size,
            "loaded": self.loaded,
        }


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CROSS_ENCODER_MODEL",
    "CrossEncoderReranker",
]
