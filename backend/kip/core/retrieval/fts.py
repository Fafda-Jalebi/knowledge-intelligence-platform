"""SQLite FTS5 keyword index -- the persistent, shared-state keyword backend.

FTS5 is SQLite's built-in inverted index. It gives the platform a real keyword
search engine with no extra service, no extra process, and no extra dependency,
and because the index lives in a file rather than in a Python process it has the
one property :class:`~kip.core.retrieval.bm25.Bm25Index` cannot have: **every
worker sees the same index**. Under ``uvicorn --workers 4``, an in-memory index
would mean a document uploaded through worker A is invisible to a keyword query
served by worker B. That is a silent, intermittent correctness bug, and avoiding
it is why this is the default backend rather than the in-memory one.

One deliberate departure from the obvious approach
-------------------------------------------------
FTS5 ships a ``porter`` tokenizer, and using it would be the one-line solution.
This class does not use it. Instead the text is stemmed in Python with
:func:`kip.core.text.stemmed_content_tokens` before insertion, and the FTS
tokenizer is plain ``unicode61``.

The reason is that SQLite's Porter implementation and this project's are not
byte-identical, and a platform with two disagreeing tokenizers has a failure mode
nobody can debug from the outside: a query matches the dense axis and misses the
keyword axis, or matches this backend and misses the in-memory one, for reasons
invisible in the logs. Pre-stemming means **one** tokenizer defines what a term
is, everywhere -- dense features, both keyword backends, and evaluation.

``tokenchars '_.'`` is needed as a consequence: after stemming, terms such as
``a_w`` and ``0.6`` are single tokens, and ``unicode61`` would otherwise split
them on the underscore and the decimal point.

Known ranking difference from the reference implementation
---------------------------------------------------------
``bm25()`` uses the textbook IDF, ``log((N - n + 0.5) / (n + 0.5))``, which goes
to zero and then negative for terms appearing in more than half the documents.
:func:`~kip.core.retrieval.bm25.bm25_idf` uses Lucene's non-negative form. On a
large corpus the two agree closely; on a small one this backend under-weights
common terms, and a term present in every document contributes nothing at all.
The effect is measured rather than asserted -- ``docs/EVALUATION.md`` reports both
keyword backends over the same query set -- and it is the reason the in-memory
implementation, not this one, is what the self-checks treat as the reference.

``bm25()`` also returns *lower is better*, so scores are negated on the way out to
match the "higher is better" convention every other retriever in the platform
uses.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from kip.core.retrieval.keyword import (
    Filters,
    KeywordDocument,
    KeywordHit,
    KeywordIndex,
    KeywordIndexError,
    normalise_filters,
    payload_matches,
)
from kip.core.text import stemmed_content_tokens

#: Columns promoted out of the JSON payload so the database can filter on them.
#: Everything else is filtered in Python after the fact.
PUSHDOWN_COLUMNS = ("document_id", "user_id")

#: Batch size for ``id IN (...)`` deletes, well under SQLite's 999-parameter cap.
DELETE_BATCH = 400

SCHEMA = """
CREATE TABLE IF NOT EXISTS kw_chunks (
    ordinal     INTEGER PRIMARY KEY,
    chunk_id    TEXT    NOT NULL UNIQUE,
    document_id TEXT    NOT NULL,
    user_id     TEXT,
    payload     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS kw_chunks_document ON kw_chunks(document_id);
CREATE INDEX IF NOT EXISTS kw_chunks_user     ON kw_chunks(user_id);

CREATE VIRTUAL TABLE IF NOT EXISTS kw_fts USING fts5(
    body,
    tokenize = "unicode61 tokenchars '_.'"
);
"""

#: Terms are quoted as FTS5 string literals, so the only character that needs
#: handling is the double quote. Anything else -- ``*``, ``:``, ``NEAR``, ``(`` --
#: loses its operator meaning inside a quoted string, which is what makes user
#: input safe to embed. Belt and braces: stemmed tokens cannot contain a quote in
#: the first place, because the tokenizer only emits [a-z0-9._].
_UNSAFE_TERM = re.compile(r"[^a-z0-9._]")


def fts5_available(connection: sqlite3.Connection | None = None) -> bool:
    """True when this SQLite build has FTS5 compiled in.

    Checked explicitly at construction so the failure is a clear startup error
    naming the alternative backend, rather than an opaque "no such module" the
    first time somebody searches.

    >>> fts5_available()
    True
    """
    own = connection is None
    conn = connection or sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _kip_fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS _kip_fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        if own:
            conn.close()


def build_match_query(terms: Sequence[str]) -> str:
    """Compose a safe FTS5 ``MATCH`` expression from stemmed terms.

    Terms are OR-ed, which reproduces BM25's disjunctive scoring: a passage
    matching three of five query terms should rank above one matching two, not be
    excluded for missing the other two.

    >>> build_match_query(["dry", "mango"])
    '"dry" OR "mango"'
    >>> build_match_query(["0.6", "a_w"])
    '"0.6" OR "a_w"'
    >>> build_match_query([])
    ''

    Anything that is not a stemmer-producible character is dropped rather than
    escaped, so no FTS5 operator can survive into the expression:

    >>> build_match_query(['dry" OR body:*', "steril"])
    '"dryorbody" OR "steril"'
    """
    cleaned: list[str] = []
    for term in terms:
        safe = _UNSAFE_TERM.sub("", str(term).lower())
        if safe and safe not in cleaned:
            cleaned.append(safe)
    return " OR ".join(f'"{term}"' for term in cleaned)


class SqliteFtsIndex(KeywordIndex):
    """Persistent keyword index backed by SQLite FTS5.

    >>> index = SqliteFtsIndex(":memory:")
    >>> _ = index.add([
    ...     KeywordDocument("a", "Hot air drying of mango slices at 60 C.",
    ...                     {"document_id": "d1", "chunk_index": 0, "user_id": 1}),
    ...     KeywordDocument("b", "Retort sterilisation of canned vegetables at 121 C.",
    ...                     {"document_id": "d2", "chunk_index": 0, "user_id": 1}),
    ...     KeywordDocument("c", "Water activity below 0.6 inhibits microbial growth.",
    ...                     {"document_id": "d3", "chunk_index": 0, "user_id": 2}),
    ... ])
    >>> index.count()
    3

    Stemming happens before insertion, so inflections match:

    >>> [hit.id for hit in index.search("dried mangoes", top_k=1)]
    ['a']

    A decimal survives as one token -- the case a dense embedder handles worst:

    >>> [hit.id for hit in index.search("0.6", top_k=1)]
    ['c']

    Scores are negated so higher is better, matching every other retriever:

    >>> index.search("sterilisation", top_k=1)[0].score > 0
    True

    Tenant isolation and document selection are enforced, and selecting nothing
    returns nothing rather than searching everything:

    >>> [hit.id for hit in index.search("water", filters={"user_id": 1})]
    []
    >>> [hit.id for hit in index.search("water", filters={"user_id": 2})]
    ['c']
    >>> index.search("water", filters={"document_id": []})
    []

    Re-adding the same chunk id replaces it instead of duplicating it:

    >>> _ = index.add([KeywordDocument("a", "Freeze drying preserves structure.",
    ...                               {"document_id": "d1", "chunk_index": 0})])
    >>> index.count()
    3
    >>> [hit.id for hit in index.search("mango")]
    []
    >>> index.delete(filters={"document_id": "d1"})
    1
    >>> index.count()
    2
    >>> index.close()
    """

    name = "fts5"

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._owns_connection = connection is None
        self.path = str(path)

        if connection is not None:
            self._conn: sqlite3.Connection | None = connection
        else:
            if self.path not in {":memory:", ""}:
                Path(self.path).expanduser().resolve().parent.mkdir(
                    parents=True, exist_ok=True
                )
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.row_factory = sqlite3.Row
        if not fts5_available(self._conn):
            raise KeywordIndexError(
                "This SQLite build has no FTS5 module, so the fts5 keyword index "
                "cannot be created. Set KEYWORD_INDEX=bm25 to use the in-memory "
                "index instead, or install a SQLite build with FTS5 enabled."
            )
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- internals ---------------------------------------------------------- #

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise KeywordIndexError("This keyword index has been closed.")
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None and self._owns_connection:
                self._conn.close()
            # A borrowed connection stays open -- its owner closes it -- but this
            # index stops using it either way.
            self._conn = None

    @staticmethod
    def _body(text: str) -> str:
        """The indexable form of a passage: space-joined stems."""
        return " ".join(stemmed_content_tokens(text))

    # -- writes ------------------------------------------------------------- #

    def add(self, documents: Sequence[KeywordDocument]) -> int:
        if not documents:
            return 0
        written = 0
        with self._lock:
            conn = self.connection
            for document in documents:
                identifier = str(document.id)
                payload = dict(document.payload)
                body = self._body(document.text)
                row = conn.execute(
                    "SELECT ordinal FROM kw_chunks WHERE chunk_id = ?", (identifier,)
                ).fetchone()
                values = (
                    identifier,
                    str(payload.get("document_id", "")),
                    None if payload.get("user_id") is None else str(payload["user_id"]),
                    json.dumps(payload, ensure_ascii=False, default=str),
                )
                if row is None:
                    cursor = conn.execute(
                        "INSERT INTO kw_chunks(chunk_id, document_id, user_id, payload) "
                        "VALUES (?, ?, ?, ?)",
                        values,
                    )
                    ordinal = int(cursor.lastrowid or 0)
                    conn.execute(
                        "INSERT INTO kw_fts(rowid, body) VALUES (?, ?)", (ordinal, body)
                    )
                else:
                    ordinal = int(row["ordinal"])
                    conn.execute(
                        "UPDATE kw_chunks SET document_id = ?, user_id = ?, payload = ? "
                        "WHERE ordinal = ?",
                        (*values[1:], ordinal),
                    )
                    conn.execute("UPDATE kw_fts SET body = ? WHERE rowid = ?", (body, ordinal))
                written += 1
            conn.commit()
        return written

    def delete(self, *, filters: Filters) -> int:
        normalised = normalise_filters(filters)
        if not normalised:
            raise ValueError(
                "Refusing to delete with no filter. Pass a document_id/user_id "
                "filter, or call clear() if wiping the index is the intent."
            )
        with self._lock:
            conn = self.connection
            where, params, residual = _build_where(normalised)
            rows = conn.execute(
                f"SELECT ordinal, payload FROM kw_chunks WHERE {where}", params
            ).fetchall()
            targets = [
                int(row["ordinal"])
                for row in rows
                if not residual or payload_matches(_load(row["payload"]), residual)
            ]
            for start in range(0, len(targets), DELETE_BATCH):
                batch = targets[start : start + DELETE_BATCH]
                marks = ", ".join("?" * len(batch))
                conn.execute(f"DELETE FROM kw_fts WHERE rowid IN ({marks})", batch)
                conn.execute(f"DELETE FROM kw_chunks WHERE ordinal IN ({marks})", batch)
            conn.commit()
            return len(targets)

    def clear(self) -> None:
        with self._lock:
            conn = self.connection
            conn.execute("DELETE FROM kw_fts")
            conn.execute("DELETE FROM kw_chunks")
            conn.commit()

    def rebuild(self, documents: Iterable[KeywordDocument]) -> int:
        with self._lock:
            self.clear()
            written = 0
            batch: list[KeywordDocument] = []
            for document in documents:
                batch.append(document)
                if len(batch) >= 500:
                    written += self.add(batch)
                    batch = []
            if batch:
                written += self.add(batch)
            return written

    # -- reads -------------------------------------------------------------- #

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Filters | None = None,
    ) -> list[KeywordHit]:
        normalised = normalise_filters(filters)
        if any(not allowed for allowed in normalised.values()):
            return []
        if top_k <= 0:
            return []

        terms = stemmed_content_tokens(query)
        match = build_match_query(terms)
        if not match:
            return []
        wanted = set(terms)

        with self._lock:
            conn = self.connection
            where, params, residual = _build_where(normalised, alias="c")
            # Over-fetch when a residual (Python-side) filter is present, so the
            # limit is applied after filtering rather than before it.
            limit = top_k if not residual else max(top_k * 8, 200)
            sql = (
                "SELECT c.chunk_id AS chunk_id, c.payload AS payload, "
                "       f.body AS body, -bm25(kw_fts) AS score "
                "FROM kw_fts AS f "
                "JOIN kw_chunks AS c ON c.ordinal = f.rowid "
                f"WHERE kw_fts MATCH ? AND {where} "
                "ORDER BY score DESC, c.chunk_id ASC "
                "LIMIT ?"
            )
            rows = conn.execute(sql, (match, *params, limit)).fetchall()

        hits: list[KeywordHit] = []
        for row in rows:
            payload = _load(row["payload"])
            if residual and not payload_matches(payload, residual):
                continue
            body_terms = set(str(row["body"]).split())
            hits.append(
                KeywordHit(
                    id=str(row["chunk_id"]),
                    score=float(row["score"]),
                    payload=payload,
                    matched_terms=tuple(t for t in terms if t in wanted & body_terms),
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def count(self, *, filters: Filters | None = None) -> int:
        normalised = normalise_filters(filters)
        if any(not allowed for allowed in normalised.values()):
            return 0
        with self._lock:
            conn = self.connection
            where, params, residual = _build_where(normalised)
            if not residual:
                row = conn.execute(
                    f"SELECT COUNT(*) AS n FROM kw_chunks WHERE {where}", params
                ).fetchone()
                return int(row["n"])
            rows = conn.execute(
                f"SELECT payload FROM kw_chunks WHERE {where}", params
            ).fetchall()
            return sum(1 for row in rows if payload_matches(_load(row["payload"]), residual))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            conn = self.connection
            documents = int(conn.execute("SELECT COUNT(*) AS n FROM kw_chunks").fetchone()["n"])
        return {
            "backend": self.name,
            "documents": documents,
            "path": self.path,
            "shared": True,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SqliteFtsIndex path={self.path!r}>"


def _load(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _build_where(
    filters: dict[str, tuple[Any, ...]],
    *,
    alias: str = "",
) -> tuple[str, list[Any], dict[str, tuple[Any, ...]]]:
    """Split filters into a parameterised SQL clause and a Python residual.

    Promoted columns are pushed into SQL; anything else is returned as a residual
    for :func:`payload_matches`. Values are always parameters, never interpolated.

    >>> _build_where({"document_id": ("a", "b")})
    ('document_id IN (?, ?)', ['a', 'b'], {})
    >>> _build_where({"user_id": (7,)}, alias="c")
    ('c.user_id IN (?)', ['7'], {})
    >>> _build_where({})
    ('1 = 1', [], {})
    >>> _build_where({"section_key": ("intro",)})
    ('1 = 1', [], {'section_key': ('intro',)})
    >>> _build_where({"document_id": ()})
    ('1 = 0', [], {})
    """
    prefix = f"{alias}." if alias else ""
    clauses: list[str] = []
    params: list[Any] = []
    residual: dict[str, tuple[Any, ...]] = {}

    for key, allowed in filters.items():
        if key not in PUSHDOWN_COLUMNS:
            residual[key] = allowed
            continue
        if not allowed:
            # An empty allowed-set matches nothing, and saying so in SQL keeps
            # the semantics identical to the in-memory backend.
            return "1 = 0", [], {}
        marks = ", ".join("?" * len(allowed))
        clauses.append(f"{prefix}{key} IN ({marks})")
        # Ids are stored as TEXT so a filter of 7 still matches a stored "7".
        params.extend(str(value) for value in allowed)

    return (" AND ".join(clauses) if clauses else "1 = 1"), params, residual


__all__ = [
    "DELETE_BATCH",
    "PUSHDOWN_COLUMNS",
    "SCHEMA",
    "SqliteFtsIndex",
    "build_match_query",
    "fts5_available",
]
