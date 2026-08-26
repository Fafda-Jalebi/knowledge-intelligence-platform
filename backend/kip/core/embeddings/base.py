"""Embedding provider abstraction.

The platform never talks to an embedding model directly. It talks to an
:class:`Embedder`, which is resolved from configuration
(``EMBEDDING_PROVIDER`` / ``EMBEDDING_MODEL``) by
:func:`kip.core.embeddings.get_embedder`. Swapping a local hashing embedder for
``sentence-transformers`` or a hosted API is a one-line environment change and
touches no retrieval code.

Two conventions make the rest of the system simpler:

**Vectors are L2-normalised.** Cosine similarity therefore reduces to a dot
product, so every vector store can rank with a single matrix multiply and no
store needs its own notion of a distance metric.

**Every embedder carries a fingerprint.** ``provider:model:dim`` is recorded
alongside the vectors it produced. Vectors from different models are not
comparable -- searching a MiniLM index with an OpenAI query vector returns
confident nonsense -- so the vector store refuses the mismatch instead of
silently degrading answer quality. This is the single most common way a RAG
deployment starts lying after a model upgrade.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

#: Storage dtype for persisted vectors. Explicit little-endian float32 keeps the
#: on-disk representation portable between architectures and halves the size of
#: a float64 index at no measurable cost to ranking quality.
VECTOR_DTYPE = np.dtype("<f4")

DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    """Identity of the model that produced a set of vectors."""

    provider: str
    model: str
    dim: int
    normalised: bool = True

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("Embedding dimension must be positive.")

    @property
    def fingerprint(self) -> str:
        """Stable identity string recorded next to stored vectors.

        >>> EmbeddingSpec("hashing", "kip-hashing-v1", 384).fingerprint
        'hashing:kip-hashing-v1:384'
        """
        return f"{self.provider}:{self.model}:{self.dim}"

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dim": self.dim,
            "normalised": self.normalised,
            "fingerprint": self.fingerprint,
        }


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend is unavailable or fails."""


class Embedder(ABC):
    """Base class handling batching, validation and shape guarantees.

    Subclasses implement :meth:`_encode` only. Everything the rest of the
    platform relies on -- 2-D ``float32`` output, exact row count, unit norms,
    graceful handling of blank input -- is enforced here rather than trusted to
    each backend.
    """

    def __init__(self, spec: EmbeddingSpec, *, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.spec = spec
        self.batch_size = max(1, int(batch_size))

    # -- Subclass contract -------------------------------------------------- #

    @abstractmethod
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a ``(len(texts), dim)`` array. May be un-normalised."""

    def warm_up(self) -> None:
        """Optional hook: load weights / open a session before first use."""

    # -- Public API --------------------------------------------------------- #

    @property
    def dim(self) -> int:
        return self.spec.dim

    @property
    def fingerprint(self) -> str:
        return self.spec.fingerprint

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of passages into a ``(n, dim)`` float32 matrix."""
        cleaned = [_prepare(text) for text in texts]
        if not cleaned:
            return np.zeros((0, self.dim), dtype=VECTOR_DTYPE)

        chunks: list[np.ndarray] = []
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            chunks.append(self._encode_checked(batch))
        matrix = np.vstack(chunks)
        return l2_normalise(matrix) if self.spec.normalised else matrix

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query into a ``(dim,)`` float32 vector."""
        return self.embed_documents([text])[0]

    def embed_iter(self, texts: Iterable[str]) -> Iterable[np.ndarray]:
        """Stream embeddings batch by batch, for large ingests."""
        buffer: list[str] = []
        for text in texts:
            buffer.append(text)
            if len(buffer) >= self.batch_size:
                yield self.embed_documents(buffer)
                buffer = []
        if buffer:
            yield self.embed_documents(buffer)

    # -- Internals ---------------------------------------------------------- #

    def _encode_checked(self, batch: Sequence[str]) -> np.ndarray:
        raw = self._encode(batch)
        matrix = np.asarray(raw, dtype=VECTOR_DTYPE)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[0] != len(batch):
            raise EmbeddingError(
                f"{type(self).__name__} returned {matrix.shape[0]} vectors for "
                f"{len(batch)} inputs."
            )
        if matrix.shape[1] != self.dim:
            raise EmbeddingError(
                f"{type(self).__name__} returned dimension {matrix.shape[1]}, "
                f"but {self.spec.model} is configured as {self.dim}. "
                "Set EMBEDDING_DIM to match the model."
            )
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.fingerprint}>"


def _prepare(text: object) -> str:
    """Coerce input to a non-empty string.

    A blank passage still needs a row in the output matrix, otherwise chunk
    indices and vector indices drift apart and every citation after the blank
    points at the wrong chunk. A single space keeps the row count honest.
    """
    value = "" if text is None else str(text)
    value = value.strip()
    return value or " "


# --------------------------------------------------------------------------- #
# Vector maths
# --------------------------------------------------------------------------- #


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows untouched.

    Rows are compared with a tolerance rather than by repr: float32 cannot hold
    0.6 exactly, so ``round(3)`` still prints 0.6000000238418579.

    >>> bool(np.allclose(l2_normalise(np.array([[3.0, 4.0]], dtype="<f4")), [[0.6, 0.8]]))
    True
    >>> l2_normalise(np.zeros((1, 3), dtype="<f4")).tolist()
    [[0.0, 0.0, 0.0]]
    """
    array = np.asarray(matrix, dtype=VECTOR_DTYPE)
    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        return array if norm == 0.0 else (array / norm).astype(VECTOR_DTYPE)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (array / norms).astype(VECTOR_DTYPE)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of ``query`` against every row of ``matrix``.

    Normalises defensively, so this is correct even if the caller passes raw
    vectors from a backend that ignores our normalisation convention.

    >>> import numpy as np
    >>> q = np.array([1.0, 0.0], dtype="<f4")
    >>> m = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype="<f4")
    >>> cosine_similarity(q, m).round(3).tolist()
    [1.0, 0.0, -1.0]
    """
    if matrix.size == 0:
        return np.zeros((0,), dtype=VECTOR_DTYPE)
    q = l2_normalise(np.asarray(query, dtype=VECTOR_DTYPE).reshape(-1))
    m = l2_normalise(np.asarray(matrix, dtype=VECTOR_DTYPE))
    return np.clip(m @ q, -1.0, 1.0).astype(VECTOR_DTYPE)


def to_bytes(vector: np.ndarray) -> bytes:
    """Serialise a vector for storage as a compact BLOB."""
    return np.asarray(vector, dtype=VECTOR_DTYPE).reshape(-1).tobytes()


def from_bytes(blob: bytes, dim: int) -> np.ndarray:
    """Inverse of :func:`to_bytes`.

    >>> import numpy as np
    >>> v = np.array([0.5, -0.25, 0.125], dtype="<f4")
    >>> from_bytes(to_bytes(v), 3).tolist()
    [0.5, -0.25, 0.125]
    """
    array = np.frombuffer(blob, dtype=VECTOR_DTYPE)
    if array.size != dim:
        raise EmbeddingError(
            f"Stored vector has {array.size} dimensions but the active model "
            f"expects {dim}. The index was built with a different embedding "
            "model and must be rebuilt."
        )
    return array.copy()


def similarity_to_confidence(score: float) -> float:
    """Map a cosine score to a bounded 0-1 *relative* strength value.

    This exists so the API never hands a raw cosine number to a user interface
    as if it meant something. A cosine of 0.42 is not "42% correct"; its
    meaning depends entirely on the embedding model. What survives model changes
    is the *ordering* and a coarse banding, so we expose a monotonic transform
    and label it as relative strength in the UI, never as certainty.

    >>> similarity_to_confidence(-1.0)
    0.0
    >>> similarity_to_confidence(1.0)
    1.0
    >>> round(similarity_to_confidence(0.0), 3)
    0.5
    """
    clamped = max(-1.0, min(1.0, float(score)))
    return round((clamped + 1.0) / 2.0, 6)


def band(score: float) -> str:
    """Coarse, model-agnostic label for a similarity score.

    Deliberately three wide buckets rather than a percentage: the bucket
    boundaries are the only part of a similarity score that stays meaningful
    across embedding models.

    >>> band(0.72), band(0.4), band(0.1)
    ('strong', 'moderate', 'weak')
    """
    if score >= 0.55:
        return "strong"
    if score >= 0.28:
        return "moderate"
    return "weak"


def sublinear_tf(count: int) -> float:
    """Damped term-frequency weight, ``1 + ln(count)``.

    >>> sublinear_tf(0)
    0.0
    >>> sublinear_tf(1)
    1.0
    >>> round(sublinear_tf(4), 4)
    2.3863
    """
    if count <= 0:
        return 0.0
    return 1.0 + math.log(count)
