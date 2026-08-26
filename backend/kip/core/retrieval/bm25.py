"""In-memory Okapi BM25 -- the reference keyword index.

BM25 is used rather than plain TF-IDF because of two corrections that matter on
real passages. **Term-frequency saturation**: a chunk that says "sterilisation"
nine times is more relevant than one that says it once, but not nine times more,
and ``k1`` bounds that growth. **Length normalisation**: without ``b``, a long
chunk wins simply by containing more words, which in a RAG system means the
context window fills up with the longest passages rather than the best ones.

The formula, which is the standard one and not a variation:

.. code-block:: text

    score(q, d) = SUM over t in q of  idf(t) * (tf * (k1 + 1))
                                      -----------------------------------
                                      tf + k1 * (1 - b + b * dl / avgdl)

    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

The ``ln(1 + ...)`` form of IDF is Lucene's: the textbook version goes negative
for terms appearing in more than half the corpus, which on a small collection can
make a passage score *lower* for containing a query term. That is indefensible in
a citation-bearing system, so the non-negative form is used.

This class is the reference implementation. ``SqliteFtsIndex`` is faster and
persistent, but this one is exact, dependency-free, has no tokenizer of its own
to disagree with, and is what the self-checks use to verify the SQLite backend
retrieves the same passages.

Why an inverted index and not a scan
------------------------------------
Only passages containing at least one query term can score above zero, so
scoring visits the union of the query's postings lists rather than the whole
corpus. On the demo corpus this is the difference between scoring a handful of
chunks and scoring all of them; the shape is what matters, since it is the
property that keeps keyword search viable when the corpus outgrows the dense
index's ability to hold every vector in RAM.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from kip.core.retrieval.keyword import (
    BM25_B,
    BM25_K1,
    Filters,
    KeywordDocument,
    KeywordHit,
    KeywordIndex,
    normalise_filters,
    payload_matches,
)
from kip.core.text import stem, stemmed_content_tokens, tokenize

#: Compact once more than this fraction of the postings refers to deleted rows.
#: Deletion is a tombstone (cheap) plus a deferred ordinal remap (not cheap), so
#: the threshold trades a little wasted memory for far fewer remaps during the
#: delete-heavy phase of a re-index.
COMPACT_RATIO = 0.3
COMPACT_MIN = 64


def bm25_idf(document_count: int, document_frequency: int) -> float:
    """Non-negative inverse document frequency.

    A term in every document contributes almost nothing; a term in one document
    out of many contributes a lot. Never negative -- see the module docstring.

    >>> round(bm25_idf(100, 1), 4)
    4.2097
    >>> round(bm25_idf(100, 50), 4)
    0.6931
    >>> bm25_idf(100, 100) > 0
    True
    >>> bm25_idf(0, 0)
    0.0
    """
    if document_count <= 0 or document_frequency <= 0:
        return 0.0
    numerator = document_count - document_frequency + 0.5
    return math.log(1.0 + numerator / (document_frequency + 0.5))


def bm25_tf(
    term_frequency: int,
    length: int,
    average_length: float,
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    """Saturating, length-normalised term-frequency weight.

    Saturation: the tenth occurrence adds far less than the second.

    >>> avg = 100.0
    >>> first = bm25_tf(1, 100, avg)
    >>> tenth = bm25_tf(10, 100, avg)
    >>> round(first, 4), round(tenth, 4)
    (1.0, 1.9643)
    >>> tenth < 10 * first
    True

    Length normalisation: the same term count counts for more in a short passage.

    >>> bm25_tf(3, 40, avg) > bm25_tf(3, 400, avg)
    True
    >>> bm25_tf(0, 100, avg)
    0.0
    """
    if term_frequency <= 0:
        return 0.0
    average = average_length if average_length > 0 else 1.0
    denominator = term_frequency + k1 * (1.0 - b + b * (length / average))
    if denominator <= 0:
        return 0.0
    return (term_frequency * (k1 + 1.0)) / denominator


class Bm25Index(KeywordIndex):
    """Exact in-memory BM25 over stemmed content tokens.

    Tokenisation is :func:`kip.core.text.stemmed_content_tokens`, the same
    function the dense embedder uses for its unigram features. That sharing is
    the point: if this index stored ``drying`` while the embedder hashed
    ``dry``, a query would hit one axis and miss the other for no reason a user
    could understand.

    >>> index = Bm25Index()
    >>> _ = index.add([
    ...     KeywordDocument("a", "Hot air drying of mango slices at 60 C.",
    ...                     {"document_id": "d1", "chunk_index": 0}),
    ...     KeywordDocument("b", "Retort sterilisation of canned vegetables at 121 C.",
    ...                     {"document_id": "d2", "chunk_index": 0}),
    ...     KeywordDocument("c", "Water activity below 0.6 inhibits microbial growth.",
    ...                     {"document_id": "d3", "chunk_index": 0}),
    ... ])
    >>> index.count()
    3

    Inflections match, because both sides are stemmed:

    >>> [hit.id for hit in index.search("dried mangoes", top_k=1)]
    ['a']

    Rare literal tokens are exactly where keyword search beats an embedder:

    >>> [hit.id for hit in index.search("0.6", top_k=1)]
    ['c']

    A term nothing contains scores nothing, rather than returning a weak guess:

    >>> index.search("extrusion")
    []

    Filters are enforced the same way as on the vector side:

    >>> [hit.id for hit in index.search("drying", filters={"document_id": ["d2"]})]
    []
    >>> index.search("drying", filters={"document_id": []})
    []
    """

    name = "bm25"

    def __init__(self, *, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._lock = threading.RLock()
        # Postings: term -> {ordinal: term frequency}
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        # Parallel per-ordinal arrays. Ordinals are append-only; deletion
        # tombstones a slot and compaction reclaims it.
        self._ids: list[str] = []
        self._payloads: list[dict[str, Any]] = []
        self._lengths: list[int] = []
        self._terms: list[tuple[str, ...]] = []
        self._live: list[bool] = []
        self._id_to_ordinal: dict[str, int] = {}
        self._live_count = 0
        self._total_length = 0
        self._dead_count = 0

    # -- writes ------------------------------------------------------------- #

    def add(self, documents: Sequence[KeywordDocument]) -> int:
        written = 0
        with self._lock:
            for document in documents:
                identifier = str(document.id)
                existing = self._id_to_ordinal.get(identifier)
                if existing is not None:
                    self._tombstone(existing)
                self._append(identifier, document)
                written += 1
        return written

    def _append(self, identifier: str, document: KeywordDocument) -> None:
        tokens = stemmed_content_tokens(document.text)
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        ordinal = len(self._ids)
        self._ids.append(identifier)
        self._payloads.append(dict(document.payload))
        self._lengths.append(len(tokens))
        self._terms.append(tuple(counts))
        self._live.append(True)
        self._id_to_ordinal[identifier] = ordinal
        self._live_count += 1
        self._total_length += len(tokens)

        for term, count in counts.items():
            self._postings[term][ordinal] = count

    def _tombstone(self, ordinal: int) -> None:
        if not self._live[ordinal]:
            return
        self._live[ordinal] = False
        self._live_count -= 1
        self._total_length -= self._lengths[ordinal]
        self._dead_count += 1
        for term in self._terms[ordinal]:
            postings = self._postings.get(term)
            if postings is not None:
                postings.pop(ordinal, None)
                if not postings:
                    del self._postings[term]
        self._id_to_ordinal.pop(self._ids[ordinal], None)
        # Drop the payload immediately: a tombstoned row must not be able to
        # leak provenance, and holding it would keep the whole payload dict
        # alive until the next compaction.
        self._payloads[ordinal] = {}
        self._terms[ordinal] = ()

    def delete(self, *, filters: Filters) -> int:
        normalised = normalise_filters(filters)
        if not normalised:
            raise ValueError(
                "Refusing to delete with no filter. Pass a document_id/user_id "
                "filter, or call clear() if wiping the index is the intent."
            )
        with self._lock:
            targets = [
                ordinal
                for ordinal in range(len(self._ids))
                if self._live[ordinal]
                and payload_matches(self._payloads[ordinal], normalised)
            ]
            for ordinal in targets:
                self._tombstone(ordinal)
            self._maybe_compact()
            return len(targets)

    def clear(self) -> None:
        with self._lock:
            self._postings = defaultdict(dict)
            self._ids = []
            self._payloads = []
            self._lengths = []
            self._terms = []
            self._live = []
            self._id_to_ordinal = {}
            self._live_count = 0
            self._total_length = 0
            self._dead_count = 0

    def _maybe_compact(self) -> None:
        total = len(self._ids)
        if total == 0 or self._dead_count < COMPACT_MIN:
            return
        if self._dead_count < COMPACT_RATIO * total:
            return

        remap: dict[int, int] = {}
        ids: list[str] = []
        payloads: list[dict[str, Any]] = []
        lengths: list[int] = []
        terms: list[tuple[str, ...]] = []
        for ordinal in range(total):
            if not self._live[ordinal]:
                continue
            remap[ordinal] = len(ids)
            ids.append(self._ids[ordinal])
            payloads.append(self._payloads[ordinal])
            lengths.append(self._lengths[ordinal])
            terms.append(self._terms[ordinal])

        postings: dict[str, dict[int, int]] = defaultdict(dict)
        for term, entries in self._postings.items():
            moved = {remap[old]: count for old, count in entries.items() if old in remap}
            if moved:
                postings[term] = moved

        self._postings = postings
        self._ids = ids
        self._payloads = payloads
        self._lengths = lengths
        self._terms = terms
        self._live = [True] * len(ids)
        self._id_to_ordinal = {identifier: index for index, identifier in enumerate(ids)}
        self._live_count = len(ids)
        self._total_length = sum(lengths)
        self._dead_count = 0

    # -- reads -------------------------------------------------------------- #

    @property
    def average_length(self) -> float:
        """Mean token count over live documents; 1.0 when the index is empty."""
        if self._live_count <= 0:
            return 1.0
        return self._total_length / self._live_count

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Filters | None = None,
    ) -> list[KeywordHit]:
        normalised = normalise_filters(filters)
        if any(not allowed for allowed in normalised.values()):
            # An empty allowed-set means "the user selected zero documents".
            return []
        if top_k <= 0:
            return []

        with self._lock:
            surface = _surface_forms(query)
            if not surface or self._live_count == 0:
                return []

            average = self.average_length
            total = self._live_count
            scores: dict[int, float] = defaultdict(float)
            fired: dict[int, list[str]] = defaultdict(list)

            for term, word in surface.items():
                postings = self._postings.get(term)
                if not postings:
                    continue
                idf = bm25_idf(total, len(postings))
                if idf <= 0.0:
                    continue
                for ordinal, frequency in postings.items():
                    weight = bm25_tf(
                        frequency,
                        self._lengths[ordinal],
                        average,
                        k1=self.k1,
                        b=self.b,
                    )
                    if weight <= 0.0:
                        continue
                    scores[ordinal] += idf * weight
                    fired[ordinal].append(word)

            candidates = [
                (score, self._ids[ordinal], ordinal)
                for ordinal, score in scores.items()
                if score > 0.0 and payload_matches(self._payloads[ordinal], normalised)
            ]
            if not candidates:
                return []

            # Sorted rather than heap-selected: ties must break deterministically
            # by id so that fusion and the self-checks are reproducible, and the
            # candidate set is the union of the query's postings, not the corpus.
            candidates.sort(key=lambda item: (-item[0], item[1]))
            return [
                KeywordHit(
                    id=identifier,
                    score=float(score),
                    payload=dict(self._payloads[ordinal]),
                    matched_terms=tuple(dict.fromkeys(fired[ordinal])),
                )
                for score, identifier, ordinal in candidates[:top_k]
            ]

    def count(self, *, filters: Filters | None = None) -> int:
        normalised = normalise_filters(filters)
        with self._lock:
            if not normalised:
                return self._live_count
            if any(not allowed for allowed in normalised.values()):
                return 0
            return sum(
                1
                for ordinal in range(len(self._ids))
                if self._live[ordinal]
                and payload_matches(self._payloads[ordinal], normalised)
            )

    def rebuild(self, documents: Iterable[KeywordDocument]) -> int:
        with self._lock:
            self.clear()
            return sum(self.add([document]) for document in documents)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": self.name,
                "documents": self._live_count,
                "terms": len(self._postings),
                "average_length": round(self.average_length, 2),
                "k1": self.k1,
                "b": self.b,
            }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Bm25Index docs={self._live_count} terms={len(self._postings)}>"


def _surface_forms(query: str) -> dict[str, str]:
    """Map each query stem to a representative original word.

    The stem is what the index is keyed on; the original word is what a user
    recognises when the source viewer highlights it. Reporting ``dried`` rather
    than ``dri`` back to the interface costs one dict and removes a whole class
    of "why is it highlighting nonsense" confusion.

    >>> _surface_forms("Drying dried mangoes")
    {'dry': 'Drying', 'mango': 'mangoes'}
    >>> _surface_forms("the and of")
    {}
    """
    # ``stemmed_content_tokens`` drops stop words, so align by stem rather than
    # by position: look the original up from the first word that produced it.
    # ``stem`` expects the lowercase form that ``tokenize`` normally produces, so
    # lowercase for the lookup while keeping the user's casing for display.
    indexable = set(stemmed_content_tokens(query))
    surface: dict[str, str] = {}
    for word in tokenize(query, lowercase=False):
        stemmed = stem(word.lower())
        if stemmed in indexable and stemmed not in surface:
            surface[stemmed] = word
    return surface


__all__ = ["Bm25Index", "COMPACT_MIN", "COMPACT_RATIO", "bm25_idf", "bm25_tf"]
