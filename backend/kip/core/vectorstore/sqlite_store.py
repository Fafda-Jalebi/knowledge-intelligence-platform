"""Persistent vector store on SQLite (the zero-infrastructure default).

Why SQLite for vectors at all? Because ``docker compose up`` should not be a
prerequisite for trying the platform, and because a single-file index that
survives a restart is a genuine improvement over an in-memory one. Ranking is
exact brute force, identical to :class:`~kip.core.vectorstore.memory.MemoryVectorStore`
-- SQLite is used for durability and for filter push-down, not for the maths.

Design decisions worth knowing:

* **``document_id`` and ``user_id`` are real columns**, not JSON payload keys.
  Tenant isolation and multi-document selection are the two filters on the hot
  path, so they push down into indexed SQL instead of being applied in Python
  after loading every vector in the database.
* **Vectors are raw little-endian float32 BLOBs.** ``np.frombuffer`` over the
  concatenation of the selected rows reconstructs the candidate matrix in one
  allocation, with no per-row Python object.
* **A generation counter invalidates a cached matrix.** Repeated queries against
  an unchanged index (which is what a chat session does) skip the decode
  entirely; any write bumps the counter and the next query reloads. Cheap, and
  provably consistent because every write path goes through ``_bump``.
* **WAL journaling** so a long ingest does not block reads.

Scaling is stated plainly in the README: exact search over ~100k chunks is
comfortable; beyond that, switch ``VECTOR_STORE=qdrant``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
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

SCHEMA = """
CREATE TABLE IF NOT EXISTS collection_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vectors (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    user_id     TEXT,
    vector      BLOB NOT NULL,
    payload     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vectors_document ON vectors(document_id);
CREATE INDEX IF NOT EXISTS idx_vectors_user     ON vectors(user_id);
"""

#: Columns promoted out of the payload so filters become indexed SQL.
PUSHDOWN_COLUMNS = ("document_id", "user_id")


class SqliteVectorStore(VectorStore):
    """Durable exact-search vector store in a single SQLite file."""

    name = "sqlite"

    def __init__(self, path: str | Path, *, dim: int | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._dim = int(dim) if dim else None
        self._generation = 0
        self._cache_generation = -1
        self._cache: tuple[list[str], np.ndarray, list[dict[str, Any]]] | None = None

        # check_same_thread=False plus an explicit RLock: the connection is
        # shared, but every entry point below holds the lock, so SQLite never
        # sees concurrent use from two threads.
        self._connection = sqlite3.connect(str(self._path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.executescript(SCHEMA)
            self._connection.commit()
            stored_dim = self._meta("dim")
            if stored_dim and self._dim is None:
                self._dim = int(stored_dim)

    # -- lifecycle ---------------------------------------------------------- #

    def ensure_collection(self, spec: EmbeddingSpec) -> None:
        with self._lock:
            self._guard_fingerprint(spec)
            if self._dim is not None and self._dim != spec.dim and self._row_count() > 0:
                raise VectorStoreError(
                    f"Index stores {self._dim}-dimensional vectors but the model "
                    f"produces {spec.dim}. Re-index required."
                )
            self._dim = spec.dim
            self._set_meta("dim", str(spec.dim))
            self._set_meta("fingerprint", spec.fingerprint)
            self._connection.commit()

    def stored_fingerprint(self) -> str | None:
        with self._lock:
            # An empty index has no meaningful fingerprint: reporting one would
            # make the first ingest after a model change look like a mismatch.
            if self._row_count() == 0:
                return None
            return self._meta("fingerprint")

    def close(self) -> None:
        with self._lock:
            self._cache = None
            try:
                self._connection.close()
            except sqlite3.Error:  # pragma: no cover
                pass

    # -- writes ------------------------------------------------------------- #

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        with self._lock:
            if self._dim is None:
                self._dim = records[0].dim
                self._set_meta("dim", str(self._dim))
            rows = []
            for record in records:
                if record.dim != self._dim:
                    raise VectorStoreError(
                        f"Vector {record.id!r} has dimension {record.dim}, "
                        f"expected {self._dim}."
                    )
                payload = dict(record.payload)
                rows.append(
                    (
                        record.id,
                        str(payload.get("document_id", "")),
                        None if payload.get("user_id") is None else str(payload["user_id"]),
                        l2_normalise(record.vector).tobytes(),
                        json.dumps(payload, ensure_ascii=False, default=str),
                    )
                )
            self._connection.executemany(
                "INSERT INTO vectors (id, document_id, user_id, vector, payload) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "document_id=excluded.document_id, user_id=excluded.user_id, "
                "vector=excluded.vector, payload=excluded.payload",
                rows,
            )
            self._connection.commit()
            self._bump()
            return len(rows)

    def delete(self, *, filters: Filters) -> int:
        normalised = normalise_filters(filters)
        if not normalised:
            # See MemoryVectorStore.delete: a missing filter is a caller error,
            # not a backend failure, so it raises ValueError like every other
            # index in the platform.
            raise ValueError(
                "Refusing to delete with no filter. Pass a document_id/user_id "
                "filter, or call clear() if wiping the collection is the intent."
            )
        with self._lock:
            where, params, residual = self._build_where(normalised)
            if residual:
                # A filter on a payload-only key cannot be expressed in SQL, so
                # resolve the matching ids in Python first and delete by id.
                ids = [
                    row["id"]
                    for row in self._connection.execute(
                        f"SELECT id, payload FROM vectors {where}", params
                    )
                    if payload_matches(json.loads(row["payload"]), residual)
                ]
                if not ids:
                    return 0
                removed = 0
                for start in range(0, len(ids), 400):
                    batch = ids[start : start + 400]
                    placeholders = ",".join("?" * len(batch))
                    cursor = self._connection.execute(
                        f"DELETE FROM vectors WHERE id IN ({placeholders})", batch
                    )
                    removed += cursor.rowcount or 0
            else:
                cursor = self._connection.execute(f"DELETE FROM vectors {where}", params)
                removed = cursor.rowcount or 0
            self._connection.commit()
            if removed:
                self._bump()
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
            if self._row_count() == 0:
                return []
            query = self._prepare_query(vector, self._dim)
            ids, matrix, payloads = self._candidates(normalised)
            if matrix.shape[0] == 0:
                return []
            scores = np.clip(matrix @ query, -1.0, 1.0)
            return self._top_k(scores, ids, payloads, top_k)

    def count(self, *, filters: Filters | None = None) -> int:
        normalised = normalise_filters(filters)
        with self._lock:
            if not normalised:
                return self._row_count()
            where, params, residual = self._build_where(normalised)
            if not residual:
                row = self._connection.execute(
                    f"SELECT COUNT(*) AS n FROM vectors {where}", params
                ).fetchone()
                return int(row["n"])
            return sum(
                1
                for row in self._connection.execute(
                    f"SELECT payload FROM vectors {where}", params
                )
                if payload_matches(json.loads(row["payload"]), residual)
            )

    def fetch(self, ids: Sequence[str]) -> list[VectorRecord]:
        wanted = [str(identifier) for identifier in ids]
        if not wanted:
            return []
        with self._lock:
            out: list[VectorRecord] = []
            for start in range(0, len(wanted), 400):
                batch = wanted[start : start + 400]
                placeholders = ",".join("?" * len(batch))
                for row in self._connection.execute(
                    f"SELECT id, vector, payload FROM vectors WHERE id IN ({placeholders})",
                    batch,
                ):
                    out.append(
                        VectorRecord(
                            id=row["id"],
                            vector=np.frombuffer(row["vector"], dtype=VECTOR_DTYPE).copy(),
                            payload=json.loads(row["payload"]),
                        )
                    )
            order = {identifier: position for position, identifier in enumerate(wanted)}
            out.sort(key=lambda record: order.get(record.id, len(order)))
            return out

    # -- introspection ------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        with self._lock:
            documents = self._connection.execute(
                "SELECT COUNT(DISTINCT document_id) AS n FROM vectors"
            ).fetchone()["n"]
            size = self._path.stat().st_size if self._path.exists() else 0
            return {
                "backend": self.name,
                "vectors": self._row_count(),
                "documents": int(documents),
                "dim": self._dim,
                "fingerprint": self.stored_fingerprint(),
                "path": str(self._path),
                "file_bytes": int(size),
                "cached": self._cache_generation == self._generation,
            }

    # -- internals ---------------------------------------------------------- #

    def _bump(self) -> None:
        self._generation += 1
        self._cache = None

    def _row_count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) AS n FROM vectors").fetchone()["n"])

    def _meta(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM collection_meta WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row["value"])

    def _set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO collection_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _build_where(
        self, filters: Mapping[str, tuple[Any, ...]]
    ) -> tuple[str, list[Any], dict[str, tuple[Any, ...]]]:
        """Split filters into an indexed SQL predicate and a Python residual.

        Every clause is parameterised -- no filter value is ever interpolated
        into SQL text.
        """
        clauses: list[str] = []
        params: list[Any] = []
        residual: dict[str, tuple[Any, ...]] = {}
        for key, allowed in filters.items():
            if key not in PUSHDOWN_COLUMNS:
                residual[key] = allowed
                continue
            if not allowed:
                # "match nothing" -- expressed in SQL so the row scan is skipped.
                clauses.append("1 = 0")
                continue
            placeholders = ",".join("?" * len(allowed))
            clauses.append(f"{key} IN ({placeholders})")
            params.extend(str(value) for value in allowed)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params, residual

    def _candidates(
        self, filters: Mapping[str, tuple[Any, ...]]
    ) -> tuple[list[str], np.ndarray, list[Mapping[str, Any]]]:
        if not filters and self._cache is not None and self._cache_generation == self._generation:
            ids, matrix, payloads = self._cache
            return ids, matrix, list(payloads)

        where, params, residual = self._build_where(filters)
        ids: list[str] = []
        blobs: list[bytes] = []
        payloads: list[dict[str, Any]] = []
        for row in self._connection.execute(
            f"SELECT id, vector, payload FROM vectors {where} ORDER BY id", params
        ):
            payload = json.loads(row["payload"])
            if residual and not payload_matches(payload, residual):
                continue
            ids.append(row["id"])
            blobs.append(row["vector"])
            payloads.append(payload)

        dim = self._dim or 0
        if not blobs or dim <= 0:
            return [], np.zeros((0, max(dim, 1)), dtype=VECTOR_DTYPE), []
        matrix = np.frombuffer(b"".join(blobs), dtype=VECTOR_DTYPE).reshape(len(blobs), dim)

        if not filters:
            self._cache = (ids, matrix, payloads)
            self._cache_generation = self._generation
        return ids, matrix, list(payloads)
