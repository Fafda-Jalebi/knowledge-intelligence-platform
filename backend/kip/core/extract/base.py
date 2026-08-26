"""Extraction data model.

Every extractor returns the same shape -- an :class:`ExtractedDocument` made of
:class:`Page` objects made of :class:`Block` objects -- so the rest of the
pipeline (structure detection, chunking, citation) is completely independent of
the input file format. Adding a new format means adding one extractor; nothing
downstream changes.

Provenance is preserved at block level (page number, ordinal, block kind,
heading level) because that is what makes a citation verifiable later: a chunk
can always be traced back to the page and section it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator


class BlockKind(str, Enum):
    """Semantic role of a block of text within a document."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CAPTION = "caption"
    CODE = "code"
    QUOTE = "quote"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value


@dataclass(slots=True)
class Block:
    """A contiguous run of text with a known role and position."""

    text: str
    kind: BlockKind = BlockKind.PARAGRAPH
    page: int = 1
    order: int = 0
    #: Heading depth (1 = top level). ``None`` for non-headings.
    level: int | None = None

    @property
    def is_heading(self) -> bool:
        return self.kind is BlockKind.HEADING

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind.value,
            "page": self.page,
            "order": self.order,
            "level": self.level,
        }


@dataclass(slots=True)
class Page:
    """One page (PDF) or one logical segment (DOCX/TXT/MD)."""

    number: int
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text)

    @property
    def is_empty(self) -> bool:
        return not any(block.text.strip() for block in self.blocks)


@dataclass(slots=True)
class DocumentMetadata:
    """Metadata recovered from the file itself (never invented)."""

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    creator: str | None = None
    created_at: str | None = None
    modified_at: str | None = None
    language: str | None = None
    #: Raw extra keys, kept for the document-details view.
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "creator": self.creator,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "language": self.language,
        }
        cleaned = {k: v for k, v in payload.items() if v}
        if self.extra:
            cleaned["extra"] = dict(self.extra)
        return cleaned


@dataclass(slots=True)
class ExtractedDocument:
    """Format-independent representation of an ingested file."""

    pages: list[Page] = field(default_factory=list)
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    #: Which extractor produced this, e.g. ``"pdf:pypdf"`` or ``"pdf:builtin"``.
    extractor: str = "unknown"
    #: Non-fatal problems worth surfacing in the UI (e.g. "3 pages had no text").
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def blocks(self) -> list[Block]:
        return [block for page in self.pages for block in page.blocks]

    @property
    def text(self) -> str:
        """Full plain text, pages separated by a blank line."""
        return "\n\n".join(page.text for page in self.pages if not page.is_empty)

    @property
    def char_count(self) -> int:
        return sum(block.char_count for block in self.blocks)

    @property
    def word_count(self) -> int:
        return sum(len(block.text.split()) for block in self.blocks)

    @property
    def has_text(self) -> bool:
        return self.char_count > 0

    @property
    def empty_page_numbers(self) -> list[int]:
        return [page.number for page in self.pages if page.is_empty]

    def iter_blocks(self) -> Iterator[Block]:
        for page in self.pages:
            yield from page.blocks

    def headings(self) -> list[Block]:
        return [block for block in self.iter_blocks() if block.is_heading]

    def add_warning(self, message: str) -> None:
        if message and message not in self.warnings:
            self.warnings.append(message)

    def renumber(self) -> "ExtractedDocument":
        """Assign a stable, gap-free ``order`` across the whole document."""
        counter = 0
        for page in self.pages:
            for block in page.blocks:
                block.order = counter
                counter += 1
        return self

    def stats(self) -> dict[str, Any]:
        """Summary used for the ingestion log and the document-details view."""
        return {
            "extractor": self.extractor,
            "pages": self.page_count,
            "blocks": len(self.blocks),
            "headings": len(self.headings()),
            "characters": self.char_count,
            "words": self.word_count,
            "empty_pages": len(self.empty_page_numbers),
            "warnings": len(self.warnings),
        }


def build_page(number: int, items: list[tuple[str, BlockKind, int | None]]) -> Page:
    """Convenience constructor used by extractors and tests."""
    page = Page(number=number)
    for index, (text, kind, level) in enumerate(items):
        if text and text.strip():
            page.blocks.append(
                Block(text=text.strip(), kind=kind, page=number, order=index, level=level)
            )
    return page
