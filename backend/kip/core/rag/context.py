"""Context builder -- turns ranked passages into the numbered block the model reads.

This is the stage where retrieval quality becomes answer quality, and where the
platform's honesty guarantees are physically enforced. Three of them:

**Only relevant context is sent.** Whole documents never reach the model. The
budget is a token count, not a passage count, so a chunk that would overflow the
window is left out rather than silently truncating whatever came after it.

**Every passage carries the identity of the chunk it came from.** A
:class:`ContextPassage` holds the chunk id, document id and chunk index that
produced it, so :mod:`kip.core.rag.citations` can resolve marker ``[3]`` back to
a real row instead of guessing. Markers are assigned here and nowhere else.

**Nothing is invented.** Passage text is the text the retriever hydrated. If a
passage has to be shortened to fit, the shortening is recorded on the passage, so
a citation can never quote words the model was not actually shown.

Selection is greedy by rank, subject to a per-document cap. Greedy rather than
interleaved because the reranker's ordering is the best available estimate of
relevance and reordering it to look balanced would discard that information --
when one document genuinely holds the answer, taking its top four passages is the
right outcome. The cap exists for the opposite case: without it, a long document
that merely repeats a phrase can fill the entire window and crowd out the one
short document that actually answers the question.

Passages are taken whole or not at all, and shortening is a second pass that runs
only when the first pass selected nothing. A passage that overflows the remaining
budget is skipped so a smaller one further down can still be included -- spending
the budget on complete passages beats spending it on one truncated passage whose
missing tail may have been the answer. Only when *every* candidate overflowed does
the builder go back and shorten the best of them, because a corpus whose chunks
are all larger than the window must still produce an answerable context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from kip.core.text import count_tokens, truncate_to_tokens

#: Reasons a passage was left out, recorded per dropped chunk so the Search and
#: Chat screens can explain a short context instead of appearing to lose results.
DROP_BUDGET = "token_budget"
DROP_DOCUMENT_CAP = "per_document_cap"
DROP_PASSAGE_CAP = "passage_cap"
DROP_EMPTY = "empty_text"


@dataclass(frozen=True, slots=True)
class ContextPassage:
    """One numbered passage as the model will see it.

    >>> passage = ContextPassage(marker=1, id="d1:4", text="Dried at 60 C.",
    ...                          document_id="d1", chunk_index=4,
    ...                          document_label="Drying Study",
    ...                          page_start=3, page_end=3,
    ...                          section_path=("Methods", "Air drying"))
    >>> passage.label
    'Drying Study, p. 3 - Methods > Air drying'
    >>> print(passage.render())
    [1] Drying Study, p. 3 - Methods > Air drying
    Dried at 60 C.
    """

    marker: int
    id: str
    text: str
    document_id: str
    chunk_index: int
    document_label: str = ""
    page_start: int | None = None
    page_end: int | None = None
    section_path: tuple[str, ...] = ()
    #: Score from the reranker (or the retriever when reranking is off). Kept for
    #: display and evaluation only -- never shown to a user as a certainty.
    score: float = 0.0
    #: True when the body was shortened to fit the window.
    truncated: bool = False

    @property
    def label(self) -> str:
        """Human-readable provenance line."""
        parts: list[str] = []
        if self.document_label:
            parts.append(str(self.document_label))
        pages = _format_pages(self.page_start, self.page_end)
        if pages:
            parts.append(pages)
        head = ", ".join(parts)
        section = " > ".join(str(part) for part in self.section_path if part)
        if head and section:
            return f"{head} - {section}"
        return head or section

    def render(self) -> str:
        """The passage as it appears in the prompt."""
        label = self.label
        header = f"[{self.marker}] {label}" if label else f"[{self.marker}]"
        return f"{header}\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "document_label": self.document_label,
            "label": self.label,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_path": list(self.section_path),
            "score": round(float(self.score), 6),
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """The assembled context plus an account of what was left out."""

    passages: tuple[ContextPassage, ...] = ()
    text: str = ""
    token_count: int = 0
    budget: int = 0
    dropped: tuple[tuple[str, str], ...] = ()

    def __len__(self) -> int:
        return len(self.passages)

    def __bool__(self) -> bool:
        return bool(self.passages)

    @property
    def document_ids(self) -> tuple[str, ...]:
        """Distinct documents represented, in marker order."""
        seen: dict[str, None] = {}
        for passage in self.passages:
            if passage.document_id:
                seen.setdefault(passage.document_id, None)
        return tuple(seen)

    def by_marker(self) -> dict[int, ContextPassage]:
        """Lookup used by citation resolution."""
        return {passage.marker: passage for passage in self.passages}

    def as_pairs(self) -> tuple[tuple[int, str], ...]:
        """``(marker, text)`` pairs for the structured LLM channel."""
        return tuple((passage.marker, passage.text) for passage in self.passages)

    @property
    def utilisation(self) -> float:
        """Fraction of the token budget used, for tuning the budget honestly.

        >>> ContextBlock(token_count=1300, budget=2600).utilisation
        0.5
        >>> ContextBlock().utilisation
        0.0
        """
        return round(self.token_count / self.budget, 4) if self.budget else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passages": [passage.to_dict() for passage in self.passages],
            "token_count": self.token_count,
            "budget": self.budget,
            "utilisation": self.utilisation,
            "document_ids": list(self.document_ids),
            "documents": len(self.document_ids),
            "dropped": [
                {"id": chunk_id, "reason": reason} for chunk_id, reason in self.dropped
            ],
        }


class ContextBuilder:
    """Packs ranked passages into a token-budgeted numbered block.

    >>> from kip.core.rerank.base import RerankResult
    >>> ranked = [
    ...     RerankResult("d1:0", 0.9, 0.5, prior_rank=1, rank=1,
    ...                  text="Mango slices are dried at 60 C.",
    ...                  payload={"document_id": "d1", "chunk_index": 0,
    ...                           "page_start": 2, "page_end": 2}),
    ...     RerankResult("d2:7", 0.7, 0.4, prior_rank=2, rank=2,
    ...                  text="Water activity below 0.6 inhibits growth.",
    ...                  payload={"document_id": "d2", "chunk_index": 7}),
    ... ]
    >>> builder = ContextBuilder(token_budget=200)
    >>> block = builder.build(ranked, titles={"d1": "Drying Study"})
    >>> len(block), block.document_ids
    (2, ('d1', 'd2'))
    >>> print(block.text)
    [1] Drying Study, p. 2
    Mango slices are dried at 60 C.
    <BLANKLINE>
    [2] d2
    Water activity below 0.6 inhibits growth.

    Markers are 1-based, contiguous, and match the passage order:

    >>> [p.marker for p in block.passages]
    [1, 2]

    The per-document cap stops one document filling the window:

    >>> many = [
    ...     RerankResult(f"d1:{i}", 1.0 - i / 100, 0.5, prior_rank=i + 1,
    ...                  rank=i + 1, text=f"Sentence {i}.",
    ...                  payload={"document_id": "d1", "chunk_index": i})
    ...     for i in range(5)
    ... ]
    >>> block = ContextBuilder(max_per_document=2).build(many)
    >>> [p.id for p in block.passages]
    ['d1:0', 'd1:1']
    >>> block.dropped
    (('d1:2', 'per_document_cap'), ('d1:3', 'per_document_cap'), ('d1:4', 'per_document_cap'))

    A passage that does not fit is skipped, and a later one that does still gets
    in -- the budget is spent, not abandoned at the first overflow:

    >>> mixed = [
    ...     RerankResult("a:0", 0.9, 0.5, prior_rank=1, rank=1,
    ...                  text="word " * 400,
    ...                  payload={"document_id": "a", "chunk_index": 0}),
    ...     RerankResult("b:0", 0.8, 0.4, prior_rank=2, rank=2, text="Short answer.",
    ...                  payload={"document_id": "b", "chunk_index": 0}),
    ... ]
    >>> block = ContextBuilder(token_budget=60).build(mixed)
    >>> [p.id for p in block.passages], block.dropped
    (['b:0'], (('a:0', 'token_budget'),))

    Only when nothing at all fits is the best passage shortened -- a corpus of
    chunks larger than the window still yields an answerable context, and the
    shortening is recorded so a citation cannot quote unseen text:

    >>> huge = [RerankResult("a:0", 0.9, 0.5, prior_rank=1, rank=1,
    ...                      text="alpha beta gamma delta " * 60,
    ...                      payload={"document_id": "a", "chunk_index": 0})]
    >>> block = ContextBuilder(token_budget=40).build(huge)
    >>> len(block), block.passages[0].truncated, block.dropped
    (1, True, ())
    >>> block.token_count <= 40
    True

    Passages with no hydrated text are dropped rather than numbered, so a marker
    never points at an empty quote:

    >>> blank = [RerankResult("c:0", 0.9, 0.5, prior_rank=1, rank=1, text="   ",
    ...                       payload={"document_id": "c", "chunk_index": 0})]
    >>> ContextBuilder().build(blank).dropped
    (('c:0', 'empty_text'),)

    An empty ranking produces an empty block, not an error:

    >>> empty = ContextBuilder().build([])
    >>> bool(empty), empty.text, empty.token_count
    (False, '', 0)
    """

    def __init__(
        self,
        *,
        token_budget: int = 2600,
        max_passages: int = 8,
        max_per_document: int = 4,
    ) -> None:
        self.token_budget = max(1, int(token_budget))
        self.max_passages = max(1, int(max_passages))
        self.max_per_document = max(1, int(max_per_document))

    @classmethod
    def from_settings(cls, settings: Any) -> "ContextBuilder":
        return cls(
            token_budget=getattr(settings, "context_token_budget", 2600),
            max_passages=getattr(settings, "context_max_passages", 8),
            max_per_document=getattr(settings, "context_max_per_document", 4),
        )

    def build(
        self,
        ranked: Iterable[Any],
        *,
        titles: Mapping[str, str] | None = None,
    ) -> ContextBlock:
        """Assemble a :class:`ContextBlock` from reranked results.

        ``ranked`` items need ``id``, ``text``, ``score`` and ``payload`` -- both
        :class:`~kip.core.rerank.base.RerankResult` and a fused hit with
        hydrated text satisfy that, so reranking can be switched off without
        changing this call.
        """
        labels = dict(titles or {})
        chosen: list[ContextPassage] = []
        dropped: list[tuple[str, str]] = []
        overflowed: list[ContextPassage] = []
        per_document: dict[str, int] = {}
        spent = 0

        for item in ranked:
            chunk_id = str(getattr(item, "id", ""))
            body = str(getattr(item, "text", "") or "").strip()
            if not body:
                dropped.append((chunk_id, DROP_EMPTY))
                continue
            if len(chosen) >= self.max_passages:
                dropped.append((chunk_id, DROP_PASSAGE_CAP))
                continue

            payload: Mapping[str, Any] = dict(getattr(item, "payload", {}) or {})
            document_id = str(payload.get("document_id") or _document_of(chunk_id))
            if per_document.get(document_id, 0) >= self.max_per_document:
                dropped.append((chunk_id, DROP_DOCUMENT_CAP))
                continue

            candidate = ContextPassage(
                marker=len(chosen) + 1,
                id=chunk_id,
                text=body,
                document_id=document_id,
                chunk_index=_as_int(payload.get("chunk_index"), -1),
                document_label=str(labels.get(document_id) or document_id),
                page_start=_as_int(payload.get("page_start"), None),
                page_end=_as_int(payload.get("page_end"), None),
                section_path=tuple(
                    str(part) for part in (payload.get("section_path") or ()) if part
                ),
                score=float(getattr(item, "score", 0.0) or 0.0),
            )

            cost = count_tokens(candidate.render())
            if cost > self.token_budget - spent:
                overflowed.append(candidate)
                dropped.append((chunk_id, DROP_BUDGET))
                continue

            chosen.append(candidate)
            per_document[document_id] = per_document.get(document_id, 0) + 1
            spent += cost

        if not chosen and overflowed:
            # Every candidate was larger than the window. Shorten the best-ranked
            # one rather than return an empty context, and mark it truncated so a
            # citation can never quote text the model was not shown.
            fitted = _shrink(overflowed[0], self.token_budget)
            if fitted is not None:
                chosen.append(fitted)
                dropped.remove((fitted.id, DROP_BUDGET))

        text = "\n\n".join(passage.render() for passage in chosen)
        return ContextBlock(
            passages=tuple(chosen),
            text=text,
            token_count=count_tokens(text) if text else 0,
            budget=self.token_budget,
            dropped=tuple(dropped),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "token_budget": self.token_budget,
            "max_passages": self.max_passages,
            "max_per_document": self.max_per_document,
        }


def _shrink(passage: ContextPassage, budget: int) -> ContextPassage | None:
    """Shorten a passage's body to fit ``budget`` tokens, header included."""
    header_cost = count_tokens(passage.render()) - count_tokens(passage.text)
    room = budget - header_cost
    if room <= 0:
        return None
    shortened = truncate_to_tokens(passage.text, room).strip()
    if not shortened:
        return None
    return ContextPassage(
        marker=passage.marker,
        id=passage.id,
        text=shortened,
        document_id=passage.document_id,
        chunk_index=passage.chunk_index,
        document_label=passage.document_label,
        page_start=passage.page_start,
        page_end=passage.page_end,
        section_path=passage.section_path,
        score=passage.score,
        truncated=True,
    )


def _format_pages(start: int | None, end: int | None) -> str:
    """Render a page range compactly.

    >>> _format_pages(3, 3), _format_pages(3, 5), _format_pages(None, None)
    ('p. 3', 'pp. 3-5', '')
    >>> _format_pages(4, None), _format_pages(None, 9)
    ('p. 4', 'p. 9')
    """
    if start is None and end is None:
        return ""
    if start is None:
        return f"p. {end}"
    if end is None or int(end) == int(start):
        return f"p. {start}"
    return f"pp. {start}-{end}"


def _document_of(chunk_id: str) -> str:
    """Recover the document id from a ``<document>:<index>`` chunk id.

    A fallback for callers that pass hits without a payload; the payload is the
    authority when present.

    >>> _document_of("doc-7:12"), _document_of("plain"), _document_of("")
    ('doc-7', 'plain', '')
    """
    return str(chunk_id).rsplit(":", 1)[0] if ":" in str(chunk_id) else str(chunk_id)


def _as_int(value: Any, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "ContextBlock",
    "ContextBuilder",
    "ContextPassage",
    "DROP_BUDGET",
    "DROP_DOCUMENT_CAP",
    "DROP_EMPTY",
    "DROP_PASSAGE_CAP",
]
