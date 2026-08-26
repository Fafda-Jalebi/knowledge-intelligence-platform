"""Structure-aware chunking.

Chunking is the single highest-leverage decision in a RAG system: it fixes the
granularity of everything downstream. Split too small and a passage loses the
context needed to answer; split too large and the embedding averages several
topics together, the retriever gets vaguer, and the context window fills with
text that is not about the question.

Strategy
--------
1. **Never cross a section boundary.** Blocks are grouped by the recovered
   outline (:mod:`kip.core.structure`), so a chunk is always about one thing.
2. **Split on sentences, not characters.** A chunk that ends mid-sentence
   damages both the embedding and the quotation shown to the user.
3. **Keep tables and code atomic.** Half a table is worse than no table; these
   blocks are only split when a single one exceeds ``max_tokens``, and then on
   row boundaries with the header row repeated.
4. **Overlap by whole sentences.** ``overlap_tokens`` of trailing sentences are
   repeated at the start of the next chunk, so a fact spanning a boundary is
   still retrievable. Overlapping by characters would reintroduce problem 2.
5. **Carry a section header.** The text sent to the embedding model and to the
   LLM is prefixed with the section path, which keeps short chunks
   ("...it should not exceed 0.6") interpretable. :attr:`Chunk.body` keeps the
   verbatim document text, so a citation quotes the source exactly and never
   shows text the platform added.
6. **Absorb runts.** A chunk below ``min_tokens`` is merged backwards when that
   stays under ``max_tokens``; otherwise it is kept, because a genuinely short
   final section is real content, not noise.

Every threshold is configuration, not a constant: see ``CHUNK_*`` in
``.env.example``.

This module is domain-agnostic. Nothing here knows or cares what the documents
are about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from kip.core.extract.base import Block, BlockKind, ExtractedDocument
from kip.core.structure import Outline, Section, build_outline, format_path
from kip.core.text import clean_text, count_tokens, split_sentences

#: Block kinds that lose their meaning when split.
ATOMIC_KINDS = frozenset({BlockKind.TABLE, BlockKind.CODE, BlockKind.CAPTION})

DEFAULT_TARGET_TOKENS = 320
DEFAULT_OVERLAP_TOKENS = 60
DEFAULT_MIN_TOKENS = 40
DEFAULT_MAX_TOKENS = 520

SECTION_HEADER_PREFIX = "Section: "


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ChunkConfig:
    """Tunable chunking parameters.

    Values are clamped in :meth:`__post_init__` so a misconfigured ``.env``
    degrades the output instead of hanging the ingestion worker -- an overlap at
    or above the target size, for instance, would make forward progress
    impossible.

    >>> ChunkConfig(target_tokens=200, overlap_tokens=500).overlap_tokens
    100
    >>> ChunkConfig(target_tokens=200, max_tokens=50).max_tokens
    200
    """

    target_tokens: int = DEFAULT_TARGET_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    min_tokens: int = DEFAULT_MIN_TOKENS
    max_tokens: int = DEFAULT_MAX_TOKENS
    #: Prefix the embedded/context text with the section path.
    include_section_header: bool = True
    #: Keep chunks inside section boundaries. Disable only for ablation studies.
    respect_sections: bool = True

    def __post_init__(self) -> None:
        self.target_tokens = max(48, int(self.target_tokens))
        self.max_tokens = max(self.target_tokens, int(self.max_tokens))
        self.min_tokens = max(1, min(int(self.min_tokens), max(1, self.target_tokens // 2)))
        self.overlap_tokens = max(0, min(int(self.overlap_tokens), self.target_tokens // 2))

    @classmethod
    def from_settings(cls, settings: Any) -> "ChunkConfig":
        """Build from a :class:`kip.config.Settings`-shaped object."""
        return cls(
            target_tokens=getattr(settings, "chunk_target_tokens", DEFAULT_TARGET_TOKENS),
            overlap_tokens=getattr(settings, "chunk_overlap_tokens", DEFAULT_OVERLAP_TOKENS),
            min_tokens=getattr(settings, "chunk_min_tokens", DEFAULT_MIN_TOKENS),
            max_tokens=getattr(settings, "chunk_max_tokens", DEFAULT_MAX_TOKENS),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_tokens": self.target_tokens,
            "overlap_tokens": self.overlap_tokens,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "include_section_header": self.include_section_header,
            "respect_sections": self.respect_sections,
        }


# --------------------------------------------------------------------------- #
# Output model
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Chunk:
    """One indexable unit of a document, with provenance intact."""

    #: Position within the document, assigned after all sections are packed.
    index: int
    #: Verbatim document text. This is what a citation quotes.
    body: str
    #: Text actually embedded and placed in the LLM context (may carry a header).
    embed_text: str
    token_count: int
    page_start: int
    page_end: int
    section_key: str | None = None
    section_path: tuple[str, ...] = ()
    heading: str | None = None
    #: ``Block.order`` values this chunk was built from - exact provenance.
    block_orders: tuple[int, ...] = ()
    kinds: tuple[str, ...] = ()
    #: True when the chunk repeats trailing sentences from the previous chunk.
    has_overlap: bool = False

    @property
    def char_count(self) -> int:
        return len(self.body)

    @property
    def section_label(self) -> str:
        return format_path(self.section_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.body,
            "embed_text": self.embed_text,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_key": self.section_key,
            "section_path": list(self.section_path),
            "section_label": self.section_label,
            "heading": self.heading,
            "block_orders": list(self.block_orders),
            "kinds": list(self.kinds),
            "has_overlap": self.has_overlap,
        }


@dataclass(slots=True)
class ChunkingResult:
    """Chunks plus the measurements shown in the document-details view."""

    chunks: list[Chunk] = field(default_factory=list)
    config: ChunkConfig = field(default_factory=ChunkConfig)
    outline: Outline | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    def __getitem__(self, item: int) -> Chunk:
        return self.chunks[item]

    @property
    def token_counts(self) -> list[int]:
        return [chunk.token_count for chunk in self.chunks]

    def stats(self) -> dict[str, Any]:
        counts = self.token_counts
        if not counts:
            return {"chunks": 0, "config": self.config.to_dict()}
        ordered = sorted(counts)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        return {
            "chunks": len(counts),
            "tokens_total": sum(counts),
            "tokens_min": ordered[0],
            "tokens_max": ordered[-1],
            "tokens_mean": round(sum(counts) / len(counts), 1),
            "tokens_median": median,
            "sections_covered": len({c.section_key for c in self.chunks if c.section_key}),
            "with_overlap": sum(1 for c in self.chunks if c.has_overlap),
            "pages_covered": len({p for c in self.chunks for p in (c.page_start, c.page_end)}),
            "config": self.config.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def chunk_document(
    document: ExtractedDocument, config: ChunkConfig | None = None
) -> ChunkingResult:
    """Chunk an extracted document, preserving section and page provenance."""
    cfg = config or ChunkConfig()
    outline = build_outline(document)
    result = ChunkingResult(config=cfg, outline=outline)

    if cfg.respect_sections:
        groups: list[tuple[Section | None, list[Block]]] = [
            (section, list(section.blocks))
            for section in outline.sections()
            if section.blocks
        ]
    else:
        groups = [(None, list(document.iter_blocks()))]

    for section, blocks in groups:
        result.chunks.extend(_chunk_section(section, blocks, cfg))

    for position, chunk in enumerate(result.chunks):
        chunk.index = position
    return result


def chunk_text(
    text: str,
    config: ChunkConfig | None = None,
    *,
    section_path: Sequence[str] = (),
    page: int = 1,
) -> list[Chunk]:
    """Chunk a bare string. Used by tests and by the evaluation harness."""
    cfg = config or ChunkConfig()
    blocks = [
        Block(text=clean_text(paragraph), kind=BlockKind.PARAGRAPH, page=page, order=order)
        for order, paragraph in enumerate(p for p in text.split("\n\n") if p.strip())
    ]
    section: Section | None = None
    if section_path:
        section = Section(
            title=section_path[-1],
            level=len(section_path),
            key="/".join(section_path),
            page=page,
            heading_order=0,
        )
        section.path = tuple(section_path)
    chunks = _chunk_section(section, blocks, cfg)
    for position, chunk in enumerate(chunks):
        chunk.index = position
    return chunks


# --------------------------------------------------------------------------- #
# Internals: units
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Unit:
    """The smallest thing the packer moves around: a sentence or atomic block."""

    text: str
    tokens: int
    page: int
    order: int
    kind: BlockKind
    atomic: bool = False


@dataclass(slots=True)
class _Group:
    """A packed set of units that will become one chunk."""

    units: list[_Unit] = field(default_factory=list)
    #: How many leading units were carried over from the previous group.
    carried: int = 0

    @property
    def tokens(self) -> int:
        return sum(unit.tokens for unit in self.units)


def _units_for(blocks: Iterable[Block], cfg: ChunkConfig) -> list[_Unit]:
    """Flatten blocks into sentence-or-atomic units."""
    units: list[_Unit] = []
    for block in blocks:
        if not block.text.strip():
            continue

        if block.kind in ATOMIC_KINDS:
            tokens = count_tokens(block.text)
            if tokens <= cfg.max_tokens:
                units.append(
                    _Unit(block.text, tokens, block.page, block.order, block.kind, atomic=True)
                )
            else:
                for piece in _split_lines_to_budget(block.text, cfg.max_tokens):
                    units.append(
                        _Unit(
                            piece, count_tokens(piece), block.page, block.order, block.kind, True
                        )
                    )
            continue

        for sentence in split_sentences(block.text) or [block.text]:
            tokens = count_tokens(sentence)
            if tokens <= cfg.max_tokens:
                units.append(_Unit(sentence, tokens, block.page, block.order, block.kind))
                continue
            # A single sentence longer than max_tokens is pathological (missing
            # punctuation, OCR run-on). Split on words as a last resort.
            for piece in _split_words_to_budget(sentence, cfg.max_tokens):
                units.append(
                    _Unit(piece, count_tokens(piece), block.page, block.order, block.kind)
                )
    return units


# --------------------------------------------------------------------------- #
# Internals: packing
# --------------------------------------------------------------------------- #


def _pack(units: Sequence[_Unit], cfg: ChunkConfig) -> list[_Group]:
    """Greedily pack units into groups of about ``target_tokens``."""
    groups: list[_Group] = []
    current = _Group()

    for unit in units:
        if not current.units:
            current.units.append(unit)
            continue

        projected = current.tokens + unit.tokens
        if projected <= cfg.target_tokens:
            current.units.append(unit)
            continue

        # Slight overflow is preferable to emitting a fragment: pull the unit in
        # when it is too small to stand alone and the hard cap still holds.
        if unit.tokens < cfg.min_tokens and projected <= cfg.max_tokens:
            current.units.append(unit)
            continue

        groups.append(current)
        overlap = _overlap_units(current.units, cfg.overlap_tokens)
        current = _Group(units=list(overlap) + [unit], carried=len(overlap))

    if current.units:
        groups.append(current)
    return groups


def _absorb_runt(groups: list[_Group], cfg: ChunkConfig) -> list[_Group]:
    """Merge a too-small final group into its predecessor when it fits."""
    if len(groups) < 2:
        return groups
    tail, previous = groups[-1], groups[-2]
    novel = tail.units[tail.carried :]
    if not novel:
        return groups[:-1]
    novel_tokens = sum(unit.tokens for unit in novel)
    if novel_tokens >= cfg.min_tokens:
        return groups
    if previous.tokens + novel_tokens > cfg.max_tokens:
        return groups
    merged = _Group(units=previous.units + novel, carried=previous.carried)
    return groups[:-2] + [merged]


def _overlap_units(units: Sequence[_Unit], budget: int) -> list[_Unit]:
    """Take trailing units up to ``budget`` tokens, never the whole group.

    Atomic units (tables, code) stop the overlap: repeating a whole table in the
    next chunk burns context budget without adding retrievable meaning.
    """
    if budget <= 0 or len(units) < 2:
        return []
    picked: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        if unit.atomic or total + unit.tokens > budget:
            break
        picked.append(unit)
        total += unit.tokens
        if len(picked) >= len(units) - 1:
            break
    picked.reverse()
    return picked


def _chunk_section(
    section: Section | None, blocks: list[Block], cfg: ChunkConfig
) -> list[Chunk]:
    units = _units_for(blocks, cfg)
    if not units:
        return []

    groups = _absorb_runt(_pack(units, cfg), cfg)

    section_path = section.path if section else ()
    section_key = section.key if section else None
    heading = section.title if section and not section.is_synthetic else None

    chunks: list[Chunk] = []
    for group in groups:
        body = _join_units(group.units)
        if not body.strip():
            continue
        pages = [unit.page for unit in group.units] or [1]
        chunks.append(
            Chunk(
                index=len(chunks),
                body=body,
                embed_text=_augment(body, section_path, cfg),
                token_count=group.tokens,
                page_start=min(pages),
                page_end=max(pages),
                section_key=section_key,
                section_path=section_path,
                heading=heading,
                block_orders=tuple(sorted({unit.order for unit in group.units})),
                kinds=tuple(sorted({unit.kind.value for unit in group.units})),
                has_overlap=group.carried > 0,
            )
        )
    return chunks


# --------------------------------------------------------------------------- #
# Internals: text assembly
# --------------------------------------------------------------------------- #


def _join_units(units: Sequence[_Unit]) -> str:
    """Reassemble units, keeping atomic blocks on their own lines.

    >>> from kip.core.extract.base import BlockKind as K
    >>> a = _Unit("Dry the slices.", 3, 1, 0, K.PARAGRAPH)
    >>> b = _Unit("Then cool them.", 3, 1, 0, K.PARAGRAPH)
    >>> _join_units([a, b])
    'Dry the slices. Then cool them.'
    >>> t = _Unit("a | b", 2, 1, 1, K.TABLE, atomic=True)
    >>> _join_units([a, t, b])
    'Dry the slices.\\na | b\\nThen cool them.'
    """
    if not units:
        return ""
    parts = [units[0].text]
    for index in range(1, len(units)):
        unit = units[index]
        separator = "\n" if unit.atomic or units[index - 1].atomic else " "
        parts.append(separator + unit.text)
    return "".join(parts).strip()


def _augment(body: str, section_path: tuple[str, ...], cfg: ChunkConfig) -> str:
    """Prefix the section path for embedding/context, leaving ``body`` verbatim."""
    if not cfg.include_section_header or not section_path:
        return body
    return f"{SECTION_HEADER_PREFIX}{format_path(section_path)}\n{body}"


def _split_words_to_budget(text: str, max_tokens: int) -> list[str]:
    """Last-resort split of a run-on sentence on word boundaries."""
    words = text.split()
    if not words:
        return []
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if count_tokens(" ".join(current)) >= max_tokens:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_lines_to_budget(text: str, max_tokens: int) -> list[str]:
    """Split an oversized table or code block on line boundaries.

    The first line is repeated at the top of each piece when the block looks
    like it has a header row, so a split table stays readable on its own.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return []
    header = lines[0] if len(lines) > 2 and "|" in lines[0] else ""
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for line in lines:
        tokens = count_tokens(line)
        if tokens > max_tokens:
            if current:
                pieces.append("\n".join(current))
                current, current_tokens = [], 0
            pieces.extend(_split_words_to_budget(line, max_tokens))
            continue
        if current and current_tokens + tokens > max_tokens:
            pieces.append("\n".join(current))
            if header and line != header:
                current = [header]
                current_tokens = count_tokens(header)
            else:
                current, current_tokens = [], 0
        current.append(line)
        current_tokens += tokens

    if current:
        pieces.append("\n".join(current))
    return [piece for piece in pieces if piece.strip()]
