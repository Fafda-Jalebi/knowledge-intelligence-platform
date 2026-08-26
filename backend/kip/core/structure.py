"""Document outline recovery: turn a flat block list into a section tree.

Why this exists
---------------
Chunk-level provenance is what makes a citation checkable. A page number alone
is weak ("page 7 of 24"); a *section path* -- "Drying and Dehydration >
Thin-layer Drying Models" -- tells the reader where the claim came from and lets
the source viewer scroll to it. The outline is also used by the chunker, which
never merges text across a heading boundary and prefixes each chunk with its
section path so a retrieved passage remains interpretable on its own.

The module is format-agnostic: it consumes :class:`~kip.core.extract.base.Block`
objects, so PDF, DOCX, Markdown and plain text all produce the same structure.

Heading levels are *normalised* before the tree is built. Real documents are
inconsistent -- a PDF heuristic may emit only level 1 and level 4, Word files
skip from Heading 1 to Heading 3 -- so observed levels are ranked into a dense
1..N sequence. Without this, a tree built from raw levels ends up with spurious
empty intermediate nodes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from kip.core.extract.base import Block, ExtractedDocument

MAX_PATH_DEPTH = 4
MAX_TITLE_CHARS = 160

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")
_LEADING_NUMBER_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+")


@dataclass(slots=True)
class Section:
    """One node in the document outline."""

    title: str
    level: int
    #: Slug path, unique within the document, e.g. ``"2-drying/2-1-kinetics"``.
    key: str
    #: Page the heading itself appeared on.
    page: int
    #: ``order`` of the heading block (``-1`` for the synthetic root section).
    heading_order: int
    #: Blocks belonging directly to this section, in document order.
    blocks: list[Block] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)
    parent_key: str | None = None
    #: Human-readable ancestry, outermost first, including this section's title.
    path: tuple[str, ...] = ()

    @property
    def is_synthetic(self) -> bool:
        """True for the implicit section holding pre-heading front matter."""
        return self.heading_order < 0

    @property
    def path_label(self) -> str:
        return " > ".join(self.path)

    @property
    def char_count(self) -> int:
        own = sum(block.char_count for block in self.blocks)
        return own + sum(child.char_count for child in self.children)

    @property
    def page_range(self) -> tuple[int, int]:
        pages = [self.page] + [b.page for b in self.blocks]
        for child in self.children:
            low, high = child.page_range
            pages += [low, high]
        return (min(pages), max(pages))

    def walk(self) -> Iterator["Section"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def to_dict(self, *, include_children: bool = True) -> dict[str, Any]:
        low, high = self.page_range
        payload: dict[str, Any] = {
            "key": self.key,
            "title": self.title,
            "level": self.level,
            "path": list(self.path),
            "page": self.page,
            "page_start": low,
            "page_end": high,
            "characters": self.char_count,
            "block_count": len(self.blocks),
        }
        if include_children:
            payload["children"] = [child.to_dict() for child in self.children]
        return payload


@dataclass(slots=True)
class Outline:
    """The recovered outline of a single document."""

    roots: list[Section] = field(default_factory=list)
    #: ``block.order`` -> owning section.
    by_block: dict[int, Section] = field(default_factory=dict)
    #: ``section.key`` -> section.
    by_key: dict[str, Section] = field(default_factory=dict)
    #: True when no headings were found and everything sits in one section.
    flat: bool = False

    def sections(self) -> list[Section]:
        return [section for root in self.roots for section in root.walk()]

    def section_for(self, block: Block) -> Section | None:
        return self.by_block.get(block.order)

    def path_for(self, block: Block) -> tuple[str, ...]:
        section = self.by_block.get(block.order)
        return section.path if section else ()

    def headings(self) -> list[str]:
        return [section.path_label for section in self.sections() if not section.is_synthetic]

    def to_dict(self) -> list[dict[str, Any]]:
        return [root.to_dict() for root in self.roots]

    def stats(self) -> dict[str, Any]:
        sections = [s for s in self.sections() if not s.is_synthetic]
        depths = [s.level for s in sections] or [0]
        return {
            "sections": len(sections),
            "max_depth": max(depths),
            "flat": self.flat,
        }


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def build_outline(document: ExtractedDocument) -> Outline:
    """Recover the section tree for an extracted document."""
    blocks = list(document.iter_blocks())
    fallback_title = (document.metadata.title or "Document").strip() or "Document"
    return outline_from_blocks(blocks, fallback_title=fallback_title)


def outline_from_blocks(
    blocks: Sequence[Block], *, fallback_title: str = "Document"
) -> Outline:
    """Build an :class:`Outline` from an ordered block sequence."""
    outline = Outline()
    headings = [block for block in blocks if block.is_heading]
    ranks = _normalise_levels(headings)

    used_keys: set[str] = set()
    stack: list[Section] = []

    # Everything before the first heading (title page, abstract, preamble) needs
    # a home; a synthetic root keeps that text addressable instead of dropping it.
    preamble: Section | None = None
    if blocks and not blocks[0].is_heading:
        preamble = Section(
            title=_shorten(fallback_title),
            level=1,
            key=_unique_key(used_keys, "", _slug(fallback_title) or "document"),
            page=blocks[0].page,
            heading_order=-1,
        )
        preamble.path = (preamble.title,)
        outline.roots.append(preamble)
        outline.by_key[preamble.key] = preamble
        stack = [preamble]

    for block in blocks:
        if block.is_heading:
            level = ranks.get(id(block), 1)
            while stack and (stack[-1].is_synthetic or stack[-1].level >= level):
                # A synthetic preamble never parents a real heading.
                if stack[-1].is_synthetic and level > 1 and len(stack) == 1:
                    break
                stack.pop()
            parent = stack[-1] if stack else None
            title = _shorten(block.text)
            prefix = parent.key if parent else ""
            section = Section(
                title=title,
                level=level,
                key=_unique_key(used_keys, prefix, _slug(title) or f"section-{block.order}"),
                page=block.page,
                heading_order=block.order,
                parent_key=parent.key if parent else None,
            )
            base_path = parent.path if parent else ()
            section.path = _trim_path(base_path + (title,))
            if parent is None:
                outline.roots.append(section)
            else:
                parent.children.append(section)
            outline.by_key[section.key] = section
            outline.by_block[block.order] = section
            stack.append(section)
            continue

        if not stack:
            preamble = Section(
                title=_shorten(fallback_title),
                level=1,
                key=_unique_key(used_keys, "", _slug(fallback_title) or "document"),
                page=block.page,
                heading_order=-1,
            )
            preamble.path = (preamble.title,)
            outline.roots.append(preamble)
            outline.by_key[preamble.key] = preamble
            stack = [preamble]

        section = stack[-1]
        section.blocks.append(block)
        outline.by_block[block.order] = section

    if not outline.roots:
        root = Section(
            title=_shorten(fallback_title),
            level=1,
            key=_slug(fallback_title) or "document",
            page=1,
            heading_order=-1,
        )
        root.path = (root.title,)
        outline.roots.append(root)
        outline.by_key[root.key] = root

    outline.flat = not headings
    return outline


def _normalise_levels(headings: Sequence[Block]) -> dict[int, int]:
    """Map raw heading levels onto a dense 1..N ranking.

    ``{1, 3, 4}`` becomes ``{1: 1, 3: 2, 4: 3}`` so the tree has no empty
    intermediate nodes. Headings with no level are treated as the shallowest.
    """
    observed = sorted({block.level for block in headings if block.level is not None})
    ranking = {raw: index + 1 for index, raw in enumerate(observed)}
    default = 1
    return {
        id(block): ranking.get(block.level, default) if block.level is not None else default
        for block in headings
    }


def _trim_path(path: tuple[str, ...]) -> tuple[str, ...]:
    """Keep the outermost and innermost context when nesting gets deep."""
    if len(path) <= MAX_PATH_DEPTH:
        return path
    return (path[0],) + path[-(MAX_PATH_DEPTH - 1) :]


def _shorten(title: str) -> str:
    collapsed = " ".join(title.split())
    if len(collapsed) <= MAX_TITLE_CHARS:
        return collapsed
    return collapsed[: MAX_TITLE_CHARS - 1].rstrip() + "…"


def _slug(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_STRIP_RE.sub("-", ascii_only).strip("-")
    return slug[:60]


def _unique_key(used: set[str], prefix: str, slug: str) -> str:
    base = f"{prefix}/{slug}" if prefix else slug
    candidate = base
    counter = 2
    while candidate in used:
        candidate = f"{base}-{counter}"
        counter += 1
    used.add(candidate)
    return candidate


# --------------------------------------------------------------------------- #
# Helpers used by the chunker and the UI
# --------------------------------------------------------------------------- #


def section_number(title: str) -> str | None:
    """Return the leading section number of a heading, if it has one.

    >>> section_number("3.2 Drying Kinetics")
    '3.2'
    >>> section_number("Introduction") is None
    True
    """
    match = _LEADING_NUMBER_RE.match(title)
    return match.group(1) if match else None


def format_path(path: Sequence[str], *, separator: str = " > ") -> str:
    """Render a section path for display or for a chunk header.

    >>> format_path(["Drying", "Thin-layer Models"])
    'Drying > Thin-layer Models'
    >>> format_path([])
    ''
    """
    return separator.join(part for part in path if part)
