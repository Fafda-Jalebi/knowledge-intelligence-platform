"""In-process vector store backed by a numpy matrix.

Used by the evaluation harness, the unit tests, and anyone who wants to run the
platform without touching disk. It is also the reference implementation: the
SQLite and Qdrant backends are expected to return the same ranking for the same
inputs, and the self-checks assert that.

Scaling honestly: search is an exact brute-force dot product over every vector
that passes the filter. That is O(n·d) per query -- roughly 10 ms for 50k
384-dimensional vectors on a laptop -- and it is *exact*, with no ANN recall
loss. It stops being the right choice somewhere in the low hundreds of
thousands of chunks, which is where Qdrant's HNSW index earns its
infrastructure cost. The README states this trade-off rather than implying the
default scales forever.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Sequence

import numpy as np

from kip.core.embeddings.base import VECTOR_DTYPE, EmbeddingSpec, l2_normalise
from kip.core.vectorstore.base import (
    Filters,
    SearchHit,
    VectorRecord,
    VectorStore,
    VectorStoreError,
    normalise_filters,
    payload_matches,
)

#: Grow the backing matrix geometrically so a large ingest does not reallocate
#: on every batch.
_GROWTH_FACTOR = 2
_INITIAL_CAPACITY = 256


class MemoryVectorStore(VectorStore):
    """Exact nearest-neighbour search over an in-memory matrix."""

    name = "memory"

    def __init__(self, dim: int | None = None) -> None:
        self._lock = threading.RLock()
        self._dim = int(dim) if dim else None
        self._fingerprint: str | None = None
        self._ids: list[str] = []
        self._payloads: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}
        self._matrix = np.zeros((0, 0), dtype=VECTOR_DTYPE)
        self._used = 0

    # -- lifecycle ---------------------------------------------------------- #

    def ensure_collection(self, spec: EmbeddingSpec) -> None:
        with self._lock:
            self._guard_fingerprint(spec)
            if self._dim is None:
                self._dim = spec.dim
            elif self._dim != spec.dim:
                raise VectorStoreError(
                    f"Store was created for dimension {self._dim} but the model "
                    f"produces {spec.dim}."
                )
            self._fingerprint = spec.fingerprint
            if self._matrix.size == 0:
                self._matrix = np.zeros((_INITIAL_CAPACITY, self._dim), dtype=VECTOR_DTYPE)

    def stored_fingerprint(self) -> str | None:
        return self._fingerprint

    def close(self) -> None:
        with self._lock:
            self._ids.clear()
            self._payloads.clear()
            self._index.clear()
            self._matrix = np.zeros((0, 0), dtype=VECTOR_DTYPE)
            self._used = 0

    # -- writes ------------------------------------------------------------- #

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        with self._lock:
            if self._dim is None:
                self._dim = records[0].dim
            if self._matrix.size == 0:
                self._matrix = np.zeros((_INITIAL_CAPACITY, self._dim), dtype=VECTOR_DTYPE)

            for record in records:
                if record.dim != self._dim:
                    raise VectorStoreError(
                        f"Vector {record.id!r} has dimension {record.dim}, "
                        f"expected {self._dim}."
                    )
                vector = l2_normalise(record.vector)
                existing = self._index.get(record.id)
                if existing is not None:
                    self._matrix[existing] = vector
                    self._payloads[existing] = dict(record.payload)
                    continue
                self._ensure_capacity(self._used + 1)
                self._matrix[self._used] = vector
                self._ids.append(record.id)
                self._payloads.append(dict(record.payload))
                self._index[record.id] = self._used
                self._used += 1
            return len(records)

    def _ensure_capacity(self, needed: int) -> None:
        capacity = self._matrix.shape[0]
        if needed <= capacity:
            return
        target = max(needed, capacity * _GROWTH_FACTOR, _INITIAL_CAPACITY)
        grown = np.zeros((target, self._dim or 1), dtype=VECTOR_DTYPE)
        grown[: self._used] = self._matrix[: self._used]
        self._matrix = grown

    def delete(self, *, filters: Filters) -> int:
        normalised = normalise_filters(filters)
        if not normalised:
            # ValueError, not VectorStoreError: nothing about the backend failed,
            # the caller passed no filter. The keyword indexes raise the same type
            # for the same mistake, so a service deleting a document across both
            # axes catches one exception, not two.
            raise ValueError(
                "Refusing to delete with no filter. Pass a document_id/user_id "
                "filter, or call clear() if wiping the collection is the intent."
            )
        with self._lock:
            keep = [
                position
                for position in range(self._used)
                if not payload_matches(self._payloads[position], normalised)
            ]
            removed = self._used - len(keep)
            if not removed:
                return 0
            # Compact in place rather than masking, so `count()` stays honest and
            # memory is actually reclaimed after a document is deleted.
            self._matrix[: len(keep)] = self._matrix[keep]
            self._ids = [self._ids[position] for position in keep]
            self._payloads = [self._payloads[position] for position in keep]
            self._used = len(keep)
            self._index = {identifier: index for index, identifier in enumerate(self._ids)}
            return removed

    # -- reads -------------------------------------------------------------- #

    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 10,
        filters: Filters | None = None,
    ) -> list[SearchHit]:
        normalised = normalise_filters(filters)
        with self._lock:
            if self._used == 0:
                return []
            query = self._prepare_query(vector, self._dim)
            if normalised:
                positions = [
                    position
                    for position in range(self._used)
                    if payload_matches(self._payloads[position], normalised)
                ]
                if not positions:
                    return []
                subset = self._matrix[positions]
                scores = np.clip(subset @ query, -1.0, 1.0)
                ids = [self._ids[position] for position in positions]
                payloads: list[Mapping[str, Any]] = [
                    self._payloads[position] for position in positions
                ]
            else:
                scores = np.clip(self._matrix[: self._used] @ query, -1.0, 1.0)
                ids = list(self._ids)
                payloads = list(self._payloads)
            return self._top_k(scores, ids, payloads, top_k)

    def count(self, *, filters: Filters | None = None) -> int:
        normalised = normalise_filters(filters)
        with self._lock:
            if not normalised:
                return self._used
            return sum(
                1
                for position in range(self._used)
                if payload_matches(self._payloads[position], normalised)
            )

    def fetch(self, ids: Sequence[str]) -> list[VectorRecord]:
        with self._lock:
            out: list[VectorRecord] = []
            for identifier in ids:
                position = self._index.get(str(identifier))
                if position is None:
                    continue
                out.append(
                    VectorRecord(
                        id=str(identifier),
                        vector=self._matrix[position].copy(),
                        payload=dict(self._payloads[position]),
                    )
                )
            return out

    # -- introspection ------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": self.name,
                "vectors": self._used,
                "dim": self._dim,
                "fingerprint": self._fingerprint,
                "capacity": int(self._matrix.shape[0]),
                "bytes": int(self._used * (self._dim or 0) * VECTOR_DTYPE.itemsize),
            }
