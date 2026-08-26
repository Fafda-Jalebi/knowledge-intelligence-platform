"""Optional embedding backends: on-device models and hosted APIs.

Every backend here is **lazily** constructed. Importing
:mod:`kip.core.embeddings` must never require ``sentence-transformers``,
``torch`` or network access -- the import graph is what makes the core testable
offline, so the heavy import happens inside the constructor and failures are
reported as an :class:`~kip.core.embeddings.base.EmbeddingError` with an
actionable message instead of an ``ImportError`` traceback.

Dimension handling deserves a note. Each of these models has a fixed output
width, and a mismatch between the model and ``EMBEDDING_DIM`` produces an index
that ranks nonsense. Rather than trusting configuration, each backend discovers
its true dimension from the model (or from a one-token probe request) and
reports the mismatch loudly.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from kip.core.embeddings.base import (
    DEFAULT_BATCH_SIZE,
    VECTOR_DTYPE,
    Embedder,
    EmbeddingError,
    EmbeddingSpec,
)
from kip.core.http import HttpError, post_json

#: Known output widths, used only to fail fast with a helpful message before a
#: model is downloaded or an API is billed. The authoritative value always comes
#: from the model itself.
KNOWN_DIMENSIONS: dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "intfloat/e5-small-v2": 384,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
}


class SentenceTransformerEmbedder(Embedder):
    """Local semantic embeddings via ``sentence-transformers``.

    This is the recommended production default when the deployment can afford
    the model download: it needs no API key, no per-request cost and no data
    egress, which matters for a platform whose whole purpose is indexing
    documents the user may not want to send anywhere.
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        dim: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise EmbeddingError(
                "EMBEDDING_PROVIDER=sentence-transformers requires the "
                "'sentence-transformers' package. Install it with "
                "`pip install sentence-transformers`, or set "
                "EMBEDDING_PROVIDER=hashing to run without it."
            ) from exc

        try:
            self._model = SentenceTransformer(model, device=device)
        except Exception as exc:  # pragma: no cover - download/runtime failure
            raise EmbeddingError(
                f"Could not load the embedding model {model!r}: {exc}"
            ) from exc

        actual = _discover_st_dimension(self._model) or KNOWN_DIMENSIONS.get(model) or dim
        if actual is None:
            raise EmbeddingError(
                f"Could not determine the output dimension of {model!r}. "
                "Set EMBEDDING_DIM explicitly."
            )
        if dim is not None and int(dim) != int(actual):
            raise EmbeddingError(
                f"EMBEDDING_DIM is {dim} but {model} produces {actual}-dimensional "
                f"vectors. Set EMBEDDING_DIM={actual} (and re-index, because "
                "existing vectors are not comparable)."
            )
        super().__init__(
            EmbeddingSpec(provider="sentence-transformers", model=model, dim=int(actual)),
            batch_size=batch_size,
        )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        try:
            vectors = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover - runtime failure
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return np.asarray(vectors, dtype=VECTOR_DTYPE)


def _discover_st_dimension(model: Any) -> int | None:  # pragma: no cover
    for attribute in ("get_sentence_embedding_dimension",):
        getter = getattr(model, attribute, None)
        if callable(getter):
            try:
                value = int(getter())
                if value > 0:
                    return value
            except Exception:
                continue
    return None


class OpenAIEmbedder(Embedder):
    """Hosted embeddings via the OpenAI-compatible ``/embeddings`` endpoint.

    ``base_url`` is configurable, so this also covers Azure OpenAI, Together,
    Groq, vLLM and any other service that implements the same contract -- one
    integration instead of five.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        dim: int | None = None,
        batch_size: int = 64,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise EmbeddingError(
                "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY to be set."
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        #: ``text-embedding-3-*`` supports server-side truncation to a smaller
        #: width, which is a legitimate cost/quality trade-off, so an explicit
        #: EMBEDDING_DIM smaller than the native size is honoured rather than
        #: rejected.
        self._requested_dim = int(dim) if dim else None

        resolved = self._requested_dim or KNOWN_DIMENSIONS.get(model)
        if resolved is None:
            resolved = self._probe_dimension(model)
        super().__init__(
            EmbeddingSpec(provider="openai", model=model, dim=int(resolved)),
            batch_size=batch_size,
        )

    def _probe_dimension(self, model: str) -> int:
        """Ask the provider for one embedding to learn the true width."""
        try:
            data = self._call(model, ["dimension probe"])
        except HttpError as exc:
            raise EmbeddingError(str(exc)) from exc
        if not data:
            raise EmbeddingError(
                f"The provider returned no embedding for {model!r}, so its "
                "dimension could not be determined. Set EMBEDDING_DIM."
            )
        return len(data[0])

    def _call(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": model, "input": list(texts)}
        # Only send `dimensions` when it is a genuine reduction request; older
        # models reject the parameter outright.
        if self._requested_dim and KNOWN_DIMENSIONS.get(model, 0) > self._requested_dim:
            payload["dimensions"] = self._requested_dim

        response = post_json(
            f"{self._base_url}/embeddings",
            payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )
        if not isinstance(response, dict) or "data" not in response:
            raise EmbeddingError("The embeddings response was not in the expected shape.")
        rows = response.get("data") or []
        # The API documents ordered output, but it also returns an explicit
        # index; sorting by it costs nothing and removes an entire class of
        # silent misalignment between chunks and vectors.
        rows = sorted(rows, key=lambda row: int(row.get("index", 0)))
        return [list(row.get("embedding") or []) for row in rows]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        try:
            vectors = self._call(self.spec.model, texts)
        except HttpError as exc:
            raise EmbeddingError(str(exc)) from exc
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Requested {len(texts)} embeddings but received {len(vectors)}."
            )
        return np.asarray(vectors, dtype=VECTOR_DTYPE)


class OllamaEmbedder(Embedder):
    """Local embeddings through an Ollama server.

    Useful middle ground: semantic quality without a Python ML stack in the API
    container, and without sending document text off the machine.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        base_url: str = "http://localhost:11434",
        dim: int | None = None,
        batch_size: int = 16,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        resolved = int(dim) if dim else KNOWN_DIMENSIONS.get(model, 0)
        if not resolved:
            resolved = len(self._embed_one(model, "dimension probe"))
        if not resolved:
            raise EmbeddingError(
                f"Ollama returned no embedding for {model!r}. Is the model pulled "
                f"(`ollama pull {model}`)?"
            )
        super().__init__(
            EmbeddingSpec(provider="ollama", model=model, dim=int(resolved)),
            batch_size=batch_size,
        )

    def _embed_one(self, model: str, text: str) -> list[float]:
        try:
            response = post_json(
                f"{self._base_url}/api/embeddings",
                {"model": model, "prompt": text},
                timeout=self._timeout,
            )
        except HttpError as exc:
            raise EmbeddingError(str(exc)) from exc
        if not isinstance(response, dict):
            raise EmbeddingError("Ollama returned an unexpected response shape.")
        return list(response.get("embedding") or [])

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        # Ollama's embeddings endpoint is single-input, so the batch loop lives
        # here. Kept explicit rather than hidden behind concurrency: ordering
        # must match the input exactly or citations point at the wrong chunk.
        rows = [self._embed_one(self.spec.model, text) for text in texts]
        widths = {len(row) for row in rows}
        if len(widths) > 1:
            raise EmbeddingError(f"Ollama returned inconsistent widths: {sorted(widths)}.")
        return np.asarray(rows, dtype=VECTOR_DTYPE)
