"""Citation resolution -- turns ``[n]`` markers into verifiable source references.

A citation in this platform is a claim that can be checked, not a footnote that
looks reassuring. That distinction is the whole point of the module, and it is
enforced two ways.

**Every marker resolves to a real chunk or is removed.** Markers are numbered by
:mod:`kip.core.rag.context`, which knows the chunk id, document id and chunk index
behind each one. A marker outside that range refers to a passage the model was
never given, so it is stripped from the answer and recorded in
``invalid_markers``. It is not silently left in the text to be read as evidence,
and it is not quietly dropped without record either -- the evaluation harness
counts it as a citation error.

**The quoted text is the text the model was shown.** A :class:`Citation` carries
the passage body verbatim from the context block, so the Source Viewer displays
what the answer was actually derived from rather than a re-fetch that may have
changed, and never a paraphrase.

Markers are never renumbered. Renumbering ``[2], [4]`` to ``[1], [2]`` would read
more tidily, but the numbers in the answer text are how the interface links a
sentence to a source panel, and rewriting them creates a class of bug where the
answer cites one passage and the panel shows another. Gaps in the numbering are
the honest representation of "the model used two of the eight passages".

The parser tolerates ``[1, 3]`` and ``[1; 3]`` as well as the ``[1][3]`` the
prompt asks for, and normalises them. Models drift toward comma lists, and
discarding a real citation over punctuation would understate citation coverage --
a measurement error dressed up as a model failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from kip.core.rag.context import ContextBlock, ContextPassage
from kip.core.rag.prompts import is_insufficient
from kip.core.text import snippet, split_sentences

#: Matches a bracketed marker or a bracketed list of them. Bounded to three
#: digits so a bracketed year or measurement in the source text cannot be read as
#: a citation to passage 1998.
MARKER_PATTERN = re.compile(r"\[\s*(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\s*\]")

#: One or more markers at the very start of a fragment. Sentence splitting puts a
#: marker written *after* a full stop into the next fragment, where it would be
#: read as citing the following claim instead of the preceding one.
LEADING_MARKERS = re.compile(r"^((?:\[\s*\d{1,3}(?:\s*[,;]\s*\d{1,3})*\s*\]\s*)+)")


@dataclass(frozen=True, slots=True)
class Citation:
    """One resolved source reference, ready for the Source Viewer.

    >>> passage = ContextPassage(marker=2, id="d1:5", text="Dried at 60 C.",
    ...                          document_id="d1", chunk_index=5,
    ...                          document_label="Drying Study", page_start=4,
    ...                          page_end=4)
    >>> citation = Citation.from_passage(passage, count=3)
    >>> citation.marker, citation.id, citation.count
    (2, 'd1:5', 3)
    >>> citation.label
    'Drying Study, p. 4'
    """

    marker: int
    id: str
    document_id: str
    chunk_index: int
    document_label: str
    label: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: tuple[str, ...] = ()
    score: float = 0.0
    #: True when the passage was shortened to fit the context window. Shown in the
    #: Source Viewer so a reader knows the model saw less than the full chunk.
    truncated: bool = False
    #: How many times the answer cited this passage.
    count: int = 1

    @classmethod
    def from_passage(cls, passage: ContextPassage, *, count: int = 1) -> "Citation":
        return cls(
            marker=passage.marker,
            id=passage.id,
            document_id=passage.document_id,
            chunk_index=passage.chunk_index,
            document_label=passage.document_label,
            label=passage.label,
            text=passage.text,
            page_start=passage.page_start,
            page_end=passage.page_end,
            section_path=passage.section_path,
            score=passage.score,
            truncated=passage.truncated,
            count=max(1, int(count)),
        )

    def to_dict(self, *, preview: int = 0) -> dict[str, Any]:
        """Serialise for the API. ``preview`` truncates ``text`` for list views."""
        return {
            "marker": self.marker,
            "chunk_id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "document_label": self.document_label,
            "label": self.label,
            "text": snippet(self.text, preview) if preview else self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_path": list(self.section_path),
            "score": round(float(self.score), 6),
            "truncated": self.truncated,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class CitationReport:
    """The answer after marker validation, plus what validation found."""

    answer: str = ""
    citations: tuple[Citation, ...] = ()
    #: Markers the model produced that no supplied passage carries.
    invalid_markers: tuple[int, ...] = ()
    #: Answer sentences carrying no marker at all.
    uncited_sentences: tuple[str, ...] = ()
    #: Sentences examined -- the denominator of :attr:`coverage`.
    sentence_count: int = 0
    #: True when the answer is the insufficient-evidence response, for which
    #: having no citations is correct rather than a coverage failure.
    refused: bool = False

    def __len__(self) -> int:
        return len(self.citations)

    @property
    def markers(self) -> tuple[int, ...]:
        return tuple(citation.marker for citation in self.citations)

    @property
    def coverage(self) -> float:
        """Fraction of answer sentences carrying at least one marker.

        A refusal scores 1.0: it makes no claim, so there is nothing to cite.

        >>> CitationReport(sentence_count=4, uncited_sentences=("a.",)).coverage
        0.75
        >>> CitationReport(refused=True).coverage
        1.0
        >>> CitationReport().coverage
        0.0
        """
        if self.refused:
            return 1.0
        if not self.sentence_count:
            return 0.0
        cited = self.sentence_count - len(self.uncited_sentences)
        return round(max(0, cited) / self.sentence_count, 4)

    @property
    def documents(self) -> tuple[str, ...]:
        """Distinct documents the answer draws on, in citation order."""
        seen: dict[str, None] = {}
        for citation in self.citations:
            if citation.document_id:
                seen.setdefault(citation.document_id, None)
        return tuple(seen)

    def to_dict(self, *, preview: int = 0) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict(preview=preview) for c in self.citations],
            "invalid_markers": list(self.invalid_markers),
            "uncited_sentences": list(self.uncited_sentences),
            "sentence_count": self.sentence_count,
            "coverage": self.coverage,
            "refused": self.refused,
            "documents": len(self.documents),
        }


def extract_markers(text: str) -> tuple[int, ...]:
    """Marker numbers in order of first appearance, de-duplicated.

    >>> extract_markers("Dried at 60 C [1]. Water activity matters [2][1].")
    (1, 2)
    >>> extract_markers("Both agree [1, 3] on this.")
    (1, 3)
    >>> extract_markers("Also written [2; 4].")
    (2, 4)
    >>> extract_markers("No markers here.")
    ()

    A bracketed number that is not a citation-sized integer is left alone:

    >>> extract_markers("The array was [1998] wide and [12.5] deep.")
    ()
    """
    seen: dict[int, None] = {}
    for match in MARKER_PATTERN.finditer(str(text or "")):
        for part in re.split(r"[,;]", match.group(1)):
            part = part.strip()
            if part.isdigit():
                seen.setdefault(int(part), None)
    return tuple(seen)


def marker_counts(text: str) -> dict[int, int]:
    """How many times each marker appears.

    >>> marker_counts("A [1]. B [1][2]. C [2].") == {1: 2, 2: 2}
    True
    """
    counts: dict[int, int] = {}
    for match in MARKER_PATTERN.finditer(str(text or "")):
        for part in re.split(r"[,;]", match.group(1)):
            part = part.strip()
            if part.isdigit():
                counts[int(part)] = counts.get(int(part), 0) + 1
    return counts


def strip_markers(text: str) -> str:
    """Remove every marker, leaving readable prose.

    Used by the grounding check, which compares the *claims* in an answer against
    the passages and must not be thrown off by bracket punctuation.

    >>> strip_markers("Mango is dried at 60 C [1]. Water activity is low [2][3].")
    'Mango is dried at 60 C. Water activity is low.'
    >>> strip_markers("")
    ''
    """
    return _tidy(MARKER_PATTERN.sub("", str(text or "")))


def normalise_markers(text: str, keep: Iterable[int] | None = None) -> str:
    """Rewrite marker groups as individual ``[n]`` markers, dropping unknown ones.

    ``keep`` is the set of markers that exist; ``None`` keeps all of them.

    >>> normalise_markers("Both sources agree [1, 3].")
    'Both sources agree [1][3].'
    >>> normalise_markers("Cited [1] and [9].", keep={1, 2})
    'Cited [1] and.'
    >>> normalise_markers("Mixed group [1, 9].", keep={1})
    'Mixed group [1].'

    An entire group of unknown markers leaves the sentence intact, minus the
    bracket -- the claim is preserved and reported as uncited rather than deleted,
    because deleting text a model produced would hide the failure:

    >>> normalise_markers("Unsupported claim [7].", keep={1})
    'Unsupported claim.'
    """
    allowed = None if keep is None else {int(value) for value in keep}

    def _replace(match: re.Match[str]) -> str:
        numbers: list[int] = []
        for part in re.split(r"[,;]", match.group(1)):
            part = part.strip()
            if not part.isdigit():
                continue
            value = int(part)
            if allowed is not None and value not in allowed:
                continue
            if value not in numbers:
                numbers.append(value)
        return "".join(f"[{value}]" for value in numbers)

    return _tidy(MARKER_PATTERN.sub(_replace, str(text or "")))


def split_cited_sentences(text: str) -> list[str]:
    """Split into sentences, keeping each marker with the claim it supports.

    Sentence splitting alone gets citations wrong whenever a model writes the
    marker after the full stop, which many do. ``"A claim. [1]"`` splits into two
    fragments, leaving the claim looking uncited and the marker looking like a
    sentence; ``"First. [1] Second. [2]"`` is worse, because ``[1]`` lands at the
    head of the second fragment and would be read as citing the wrong claim.

    Markers found at the start of a fragment therefore move back to the previous
    sentence, inserted before its terminator so the result reads naturally, and a
    fragment left with nothing but markers disappears.

    >>> split_cited_sentences("Dried at 60 C [1]. Water activity is low [2].")
    ['Dried at 60 C [1].', 'Water activity is low [2].']
    >>> split_cited_sentences("Dried at 60 C. [1]")
    ['Dried at 60 C [1].']
    >>> split_cited_sentences("First claim. [1] Second claim. [2]")
    ['First claim [1].', 'Second claim [2].']
    >>> split_cited_sentences("No markers at all.")
    ['No markers at all.']
    >>> split_cited_sentences("[1]")
    ['[1]']
    >>> split_cited_sentences("")
    []
    """
    out: list[str] = []
    for raw in split_sentences(str(text or "")):
        sentence = raw.strip()
        if not sentence:
            continue
        lead = LEADING_MARKERS.match(sentence)
        if lead and out:
            out[-1] = _attach_markers(out[-1], lead.group(1))
            sentence = sentence[lead.end() :].strip()
            if not sentence:
                continue
        out.append(sentence)
    return out


def _attach_markers(sentence: str, markers: str) -> str:
    """Insert ``markers`` before a sentence's terminating punctuation.

    >>> _attach_markers("Dried at 60 C.", "[1]")
    'Dried at 60 C [1].'
    >>> _attach_markers("Is it dried?", "[2] ")
    'Is it dried [2]?'
    >>> _attach_markers("No terminator", "[3]")
    'No terminator [3]'
    """
    tidy_markers = _tidy(markers)
    body = sentence.rstrip()
    if body and body[-1] in ".!?" and not body.endswith(".."):
        return f"{body[:-1].rstrip()} {tidy_markers}{body[-1]}"
    return f"{body} {tidy_markers}"


def resolve(answer: str, block: ContextBlock | Sequence[ContextPassage]) -> CitationReport:
    """Validate an answer's markers against the passages that were supplied.

    >>> from kip.core.rag.context import ContextBuilder
    >>> from kip.core.rerank.base import RerankResult
    >>> ranked = [
    ...     RerankResult("d1:0", 0.9, 0.5, prior_rank=1, rank=1,
    ...                  text="Mango slices are dried at 60 C.",
    ...                  payload={"document_id": "d1", "chunk_index": 0}),
    ...     RerankResult("d2:3", 0.8, 0.4, prior_rank=2, rank=2,
    ...                  text="Water activity below 0.6 inhibits growth.",
    ...                  payload={"document_id": "d2", "chunk_index": 3}),
    ... ]
    >>> block = ContextBuilder().build(ranked)
    >>> report = resolve("Mango is dried at 60 C [1].", block)
    >>> report.answer
    'Mango is dried at 60 C [1].'
    >>> [(c.marker, c.id, c.document_id) for c in report.citations]
    [(1, 'd1:0', 'd1')]
    >>> report.coverage, report.invalid_markers
    (1.0, ())

    The cited text is the passage the model was shown, so the Source Viewer can
    display it without a second lookup:

    >>> report.citations[0].text
    'Mango slices are dried at 60 C.'

    An invented marker is removed from the answer and recorded:

    >>> report = resolve("Shelf life is 12 months [5].", block)
    >>> report.answer, report.invalid_markers
    ('Shelf life is 12 months.', (5,))
    >>> report.citations, report.uncited_sentences
    ((), ('Shelf life is 12 months.',))

    Citations are listed in passage order, not the order the answer happened to
    mention them, so the source panel matches the numbering the reader sees:

    >>> report = resolve("Growth is inhibited [2]. Drying is at 60 C [1].", block)
    >>> report.markers
    (1, 2)

    Repeated use of one passage is counted, not duplicated:

    >>> report = resolve("Dried at 60 C [1], and only at 60 C [1].", block)
    >>> len(report.citations), report.citations[0].count
    (1, 2)

    A refusal is recognised, so it is not penalised for having no citations:

    >>> from kip.core.rag.prompts import INSUFFICIENT_EVIDENCE
    >>> report = resolve(INSUFFICIENT_EVIDENCE, block)
    >>> report.refused, report.citations, report.coverage
    (True, (), 1.0)

    An empty answer resolves to an empty report rather than raising:

    >>> resolve("", block).to_dict()["sentence_count"]
    0
    """
    passages = block.passages if isinstance(block, ContextBlock) else tuple(block)
    by_marker: dict[int, ContextPassage] = {p.marker: p for p in passages}

    raw = str(answer or "").strip()
    if not raw:
        return CitationReport()

    produced = extract_markers(raw)
    invalid = tuple(marker for marker in produced if marker not in by_marker)
    cleaned = normalise_markers(raw, keep=set(by_marker)) if produced else raw

    counts = marker_counts(cleaned)
    citations = tuple(
        Citation.from_passage(by_marker[marker], count=counts.get(marker, 1))
        for marker in sorted(counts)
        if marker in by_marker
    )

    refused = is_insufficient(cleaned)
    sentences = [] if refused else split_cited_sentences(cleaned)
    uncited = tuple(
        sentence for sentence in sentences if not MARKER_PATTERN.search(sentence)
    )

    return CitationReport(
        answer=cleaned,
        citations=citations,
        invalid_markers=invalid,
        uncited_sentences=uncited,
        sentence_count=len(sentences),
        refused=refused,
    )


def _tidy(text: str) -> str:
    """Repair the spacing removing a marker leaves behind.

    >>> _tidy("Dried at 60 C .")
    'Dried at 60 C.'
    >>> _tidy("First  claim ,  second .")
    'First claim, second.'
    >>> _tidy("Line one \\n\\n Line two")
    'Line one\\n\\nLine two'
    """
    out = str(text or "")
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+([.,;:!?)\]])", r"\1", out)
    out = re.sub(r"([(\[])[ \t]+", r"\1", out)
    out = "\n".join(line.strip() for line in out.split("\n"))
    return out.strip()


__all__ = [
    "Citation",
    "CitationReport",
    "LEADING_MARKERS",
    "MARKER_PATTERN",
    "extract_markers",
    "marker_counts",
    "normalise_markers",
    "resolve",
    "split_cited_sentences",
    "strip_markers",
]
