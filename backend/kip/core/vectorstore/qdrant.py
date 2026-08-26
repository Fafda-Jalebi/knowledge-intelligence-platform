"""Qdrant backend -- the production path.

Selected with ``VECTOR_STORE=qdrant``, and what ``docker compose up`` starts.
Qdrant earns its container once the corpus outgrows exact search: HNSW gives
sub-linear approximate nearest-neighbour lookup, server-side payload filtering,
and an index that survives restarts without being reloaded into the API
process's memory.

The import is lazy and the failure message is actionable, so the platform still
boots -- and still tells you exactly what to do -- when ``qdrant-client`` is
absent or the server is unreachable. That matters because the default
configuration deliberately does not require either.

Two details are load-bearing:

* **Point ids.** Qdrant accepts unsigned integers or UUIDs, not arbitrary
  strings, but the platform's ids are ``"<document_id>:<chunk_index>"``. Each id
  is mapped to a deterministic UUIDv5, so re-indexing the same chunk overwrites
  the same point (idempotent) while the original id travels in the payload and
  is what gets handed back to the caller.
* **Fingerprint storage.** Qdrant has no collection-level key/value store, so the
  ``provider:model:dim`` fingerprint is written as a single reserved point that
  is excluded from every search by an explicit filter. Losing the fingerprint
  would mean losing the ability to detect a model change, which is the one
  failure this layer exists to prevent.
"""

from __future__ import annotations

import uuid
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
)

#: Namespace for deriving stable point UUIDs from platform ids.
ID_NAMESPACE = uuid.UUID("6f1c9d2e-6b5a-4c3d-8e7f-0a1b2c3d4e5f")

#: Payload key holding the platform's own id, since the point id is a UUID.
KIP_ID_KEY = "kip_id"

#: Payload marker on the metadata point, filtered out of every search.
META_KEY = "kip_meta"

DEFAULT_COLLECTION = "kip_chunks"


def point_uuid(identifier: str) -> str:
    """Deterministic UUIDv5 for a platform id.

    >>> point_uuid("doc-1:0") == point_uuid("doc-1:0")
    True
    >>> point_uuid("doc-1:0") != point_uuid("doc-1:1")
    True
    """
    return str(uuid.uuid5(ID_NAMESPACE, str(identifier)))


class QdrantVectorStore(VectorStore):
    """Approximate nearest-neighbour search backed by a Qdrant server."""

    name = "qdrant"

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection: str = DEFAULT_COLLECTION,
        dim: int | None = None,
        timeout: float = 30.0,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 128,
    ) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.http import models as qmodels  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise VectorStoreError(
                "VECTOR_STORE=qdrant requires the 'qdrant-client' package. "
                "Install it with `pip install qdrant-client`, or set "
                "VECTOR_STORE=sqlite to run without a vector database server."
            ) from exc

        self._models = qmodels
        self._collection = collection
        self._dim = int(dim) if dim else None
        self._hnsw = (int(hnsw_m), int(hnsw_ef_construct))
        try:
            self._client = QdrantClient(url=url, api_key=api_key or None, timeout=timeout)
        except Exception as exc:  # pragma: no cover - connection failure
            raise VectorStoreError(
                f"Could not connect to Qdrant at {url}: {exc}. Is the service "
                "running? `docker compose up qdrant` starts it."
            ) from exc

    # -- lifecycle ---------------------------------------------------------- #

    def ensure_collection(self, spec: EmbeddingSpec) -> None:
        models = self._models
        self._guard_fingerprint(spec)
        if not self._collection_exists():
            try:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=models.VectorParams(
                        size=spec.dim,
                        # Vectors are L2-normalised before storage, so cosine and
                        # dot product rank identically; COSINE is declared anyway
                        # so a hand-written query against the same collection
                        # behaves the same way.
                        distance=models.Distance.COSINE,
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=self._hnsw[0], ef_construct=self._hnsw[1]
                    ),
                )
            except Exception as exc:  # pragma: no cover
                raise VectorStoreError(f"Could not create collection: {exc}") from exc
            # Indexing the two hot filter keys server-side: without these Qdrant
            # falls back to a full payload scan for tenant isolation.
            for field in ("document_id", "user_id"):
                try:
                    self._client.create_payload_index(
                        collection_name=self._collection,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:  # pragma: no cover - index may already exist
                    pass
        self._dim = spec.dim
        self._write_fingerprint(spec)

    def stored_fingerprint(self) -> str | None:
        if not self._collection_exists():
            return None
        try:
            points = self._client.retrieve(
                collection_name=self._collection,
                ids=[point_uuid("__kip_meta__")],
                with_payload=True,
                with_vectors=False,
            )
        except Exception:  # pragma: no cover - server error
            return None
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            fingerprint = payload.get("fingerprint")
            if fingerprint:
                return str(fingerprint)
        return None

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover
            pass

    def _collection_exists(self) -> bool:
        try:
            return bool(self._client.collection_exists(self._collection))
        except AttributeError:  # pragma: no cover - older client
            try:
                self._client.get_collection(self._collection)
                return True
            except Exception:
                return False
        except Exception:  # pragma: no cover
            return False

    def _write_fingerprint(self, spec: EmbeddingSpec) -> None:
        models = self._models
        marker = np.zeros(spec.dim, dtype=VECTOR_DTYPE)
        # A zero vector scores 0.0 against everything under cosine, so even if
        # the exclusion filter were somehow bypassed the marker cannot outrank a
        # real passage.
        self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=point_uuid("__kip_meta__"),
                    vector=marker.tolist(),
                    payload={
                        META_KEY: True,
                        "fingerprint": spec.fingerprint,
                        **spec.to_dict(),
                    },
                )
            ],
        )

    # -- writes ------------------------------------------------------------- #

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        models = self._models
        points = []
        for record in records:
            if self._dim is not None and record.dim != self._dim:
                raise VectorStoreError(
                    f"Vector {record.id!r} has dimension {record.dim}, expected {self._dim}."
                )
            payload = dict(record.payload)
            payload[KIP_ID_KEY] = record.id
            payload[META_KEY] = False
            # Ids arrive as str or int depending on the caller; Qdrant's keyword
            # index only matches strings, so normalise here.
            for key in ("document_id", "user_id"):
                if payload.get(key) is not None:
                    payload[key] = str(payload[key])
            points.append(
                models.PointStruct(
                    id=point_uuid(record.id),
                    vector=l2_normalise(record.vector).tolist(),
                    payload=payload,
                )
            )
        try:
            self._client.upsert(collection_name=self._collection, points=points, wait=True)
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc
        return len(points)

    def delete(self, *, filters: Filters) -> int:
        normalised = normalise_filters(filters)
        if not normalised:
            # See MemoryVectorStore.delete: a missing filter is a caller error,
            # not a backend failure. Especially here, where an unfiltered delete
            # would clear a shared server-side collection.
            raise ValueError(
                "Refusing to delete with no filter. Pass a document_id/user_id "
                "filter, or call clear() if wiping the collection is the intent."
            )
        condition = self._build_filter(normalised)
        if condition is None:
            return 0
        before = self.count(filters=filters)
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=self._models.FilterSelector(filter=condition),
                wait=True,
            )
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc
        return before

    # -- reads -------------------------------------------------------------- #

    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 10,
        filters: Filters | None = None,
    ) -> list[SearchHit]:
        normalised = normalise_filters(filters)
        if any(not allowed for allowed in normalised.values()):
            # An empty allowed-set means "nothing selected"; matching the
            # in-process reference implementation, that returns no results
            # rather than searching everything.
            return []
        query = self._prepare_query(vector, self._dim)
        condition = self._build_filter(normalised)
        try:
            results = self._client.search(
                collection_name=self._collection,
                query_vector=query.tolist(),
                query_filter=condition,
                limit=max(int(top_k), 1),
                with_payload=True,
            )
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

        hits: list[SearchHit] = []
        for result in results:
            payload = dict(getattr(result, "payload", None) or {})
            if payload.pop(META_KEY, False):
                continue
            identifier = str(payload.pop(KIP_ID_KEY, getattr(result, "id", "")))
            hits.append(
                SearchHit(id=identifier, score=float(getattr(result, "score", 0.0)), payload=payload)
            )
        return hits

    def count(self, *, filters: Filters | None = None) -> int:
        normalised = normalise_filters(filters)
        if any(not allowed for allowed in normalised.values()):
            return 0
        condition = self._build_filter(normalised)
        try:
            result = self._client.count(
                collection_name=self._collection, count_filter=condition, exact=True
            )
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(f"Qdrant count failed: {exc}") from exc
        return int(getattr(result, "count", 0))

    def fetch(self, ids: Sequence[str]) -> list[VectorRecord]:
        wanted = [str(identifier) for identifier in ids]
        if not wanted:
            return []
        try:
            points = self._client.retrieve(
                collection_name=self._collection,
                ids=[point_uuid(identifier) for identifier in wanted],
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(f"Qdrant retrieve failed: {exc}") from exc

        by_id: dict[str, VectorRecord] = {}
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            if payload.pop(META_KEY, False):
                continue
            identifier = str(payload.pop(KIP_ID_KEY, getattr(point, "id", "")))
            raw = getattr(point, "vector", None) or []
            by_id[identifier] = VectorRecord(
                id=identifier,
                vector=np.asarray(raw, dtype=VECTOR_DTYPE),
                payload=payload,
            )
        return [by_id[identifier] for identifier in wanted if identifier in by_id]

    # -- introspection ------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "backend": self.name,
            "collection": self._collection,
            "dim": self._dim,
            "fingerprint": self.stored_fingerprint(),
        }
        try:
            info["vectors"] = self.count()
        except VectorStoreError:  # pragma: no cover
            info["vectors"] = None
        return info

    # -- internals ---------------------------------------------------------- #

    def _build_filter(self, filters: Mapping[str, tuple[Any, ...]]) -> Any:
        """Translate normalised filters into a Qdrant ``Filter``.

        The metadata point is excluded here rather than in each caller, so there
        is exactly one place that can forget to do it.
        """
        models = self._models
        must: list[Any] = [
            models.FieldCondition(key=META_KEY, match=models.MatchValue(value=False))
        ]
        for key, allowed in filters.items():
            values = [str(value) for value in allowed]
            if len(values) == 1:
                must.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=values[0]))
                )
            else:
                must.append(models.FieldCondition(key=key, match=models.MatchAny(any=values)))
        return models.Filter(must=must)
