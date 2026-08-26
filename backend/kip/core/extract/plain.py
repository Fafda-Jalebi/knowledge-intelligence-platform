"""Plain-text and Markdown extraction (standard library only).

Pagination
----------
``.txt``/``.md`` files have no intrinsic pages. Rather than reporting page 1
for an entire 40-page document -- which would make citations useless -- we
paginate on form-feed characters when present, and otherwise synthesise pages
of roughly ``chars_per_page`` characters, never splitting a block. The
extractor records ``pagination=synthetic`` in the metadata so the UI can label
the page number honestly as a *position* rather than a printed page.
"""

from __future__ import annotations

import re

from kip.core.extract.base import (
    Block,
    BlockKind,
    DocumentMetadata,
    ExtractedDocument,
    Page,
)
from kip.core.text import clean_text, normalise_unicode

DEFAULT_CHARS_PER_PAGE = 3200

# --- Markdown / plain-text structure patterns ------------------------------- #

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_H1_RE = re.compile(r"^=+\s*$")
_SETEXT_H2_RE = re.compile(r"^-{2,}\s*$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_BULLET_RE = re.compile(r"^\s{0,6}([-*+•]|\d{1,3}[.)])\s+\S")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$")
_QUOTE_RE = re.compile(r"^\s{0,3}>\s?")
_CAPTION_RE = re.compile(
    r"^\s*(table|figure|fig|chart|exhibit|appendix)\s*[\d.]*\s*[:.\-–]", re.IGNORECASE
)
_FRONTMATTER_FENCE = re.compile(r"^---\s*$")

# A plain-text heading: short, no terminal period, and either numbered
# ("3.2 Drying Kinetics"), ALL CAPS, or Title Case.
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+(\S.*)$")


def extract_plain(
    data: bytes | str,
    *,
    is_markdown: bool = False,
    chars_per_page: int = DEFAULT_CHARS_PER_PAGE,
) -> ExtractedDocument:
    """Extract text/Markdown bytes into an :class:`ExtractedDocument`."""
    raw = _decode(data) if isinstance(data, (bytes, bytearray)) else str(data)
    document = ExtractedDocument(extractor="markdown:builtin" if is_markdown else "text:builtin")

    raw = normalise_unicode(raw).replace("\r\n", "\n").replace("\r", "\n")
    front_matter, raw = _split_front_matter(raw)
    if front_matter:
        document.metadata = _metadata_from_front_matter(front_matter)

    segments = raw.split("\f")
    hard_pages = len(segments) > 1

    blocks: list[Block] = []
    for segment in segments:
        blocks.extend(_blocks_from_text(segment, is_markdown=is_markdown))
        if hard_pages:
            blocks.append(_PAGE_BREAK)

    if hard_pages:
        document.pages = _paginate_on_breaks(blocks)
        document.metadata.extra.setdefault("pagination", "form-feed")
    else:
        document.pages = _paginate_by_size(blocks, chars_per_page)
        document.metadata.extra.setdefault("pagination", "synthetic")
        document.metadata.extra.setdefault("chars_per_page", str(chars_per_page))

    if not document.metadata.title:
        document.metadata.title = _infer_title(document)

    if not document.has_text:
        document.add_warning("The file contained no readable text.")

    return document.renumber()


#: Sentinel block used to mark a hard page boundary during assembly.
_PAGE_BREAK = Block(text="\x00PAGEBREAK", kind=BlockKind.PARAGRAPH, page=0)


def _decode(data: bytes | bytearray) -> str:
    """Decode bytes, trying the encodings real-world text files actually use.

    UTF-16 is only attempted when a BOM is present or the byte stream is dense
    with NULs. Trying it unconditionally is actively harmful: almost any
    even-length single-byte text decodes "successfully" into CJK mojibake, so a
    cp1252 file such as ``Caf\\xe9 r\\xe9sum\\xe9`` would be silently destroyed
    rather than falling through to the correct codec.
    """
    payload = bytes(data)
    for bom, encoding in (
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    ):
        if payload.startswith(bom):
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                break

    candidates = ["utf-8"]
    if _looks_like_utf16(payload):
        candidates.append("utf-16")
    candidates += ["cp1252", "latin-1"]

    for encoding in candidates:
        try:
            return payload.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return payload.decode("utf-8", errors="replace")


def _looks_like_utf16(payload: bytes) -> bool:
    """Heuristic for BOM-less UTF-16: many NUL bytes, consistently aligned."""
    sample = payload[:4096]
    if len(sample) < 4 or len(sample) % 2:
        return False
    nulls = sample.count(0)
    if nulls < len(sample) // 4:
        return False
    even = sum(1 for index in range(0, len(sample), 2) if sample[index] == 0)
    odd = sum(1 for index in range(1, len(sample), 2) if sample[index] == 0)
    return max(even, odd) > 4 * min(even, odd) + 1


def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Peel a leading YAML-ish front-matter block off a Markdown file."""
    lines = raw.split("\n")
    if not lines or not _FRONTMATTER_FENCE.match(lines[0]):
        return {}, raw
    for index in range(1, min(len(lines), 60)):
        if _FRONTMATTER_FENCE.match(lines[index]):
            body = "\n".join(lines[index + 1 :])
            parsed: dict[str, str] = {}
            for line in lines[1:index]:
                key, sep, value = line.partition(":")
                if sep and key.strip():
                    parsed[key.strip().lower()] = value.strip().strip("\"'")
            return parsed, body
    return {}, raw


def _metadata_from_front_matter(data: dict[str, str]) -> DocumentMetadata:
    known = {"title", "author", "subject", "date", "language", "lang"}
    return DocumentMetadata(
        title=data.get("title") or None,
        author=data.get("author") or None,
        subject=data.get("subject") or data.get("description") or None,
        created_at=data.get("date") or None,
        language=data.get("language") or data.get("lang") or None,
        extra={k: v for k, v in data.items() if k not in known and v},
    )


def _blocks_from_text(segment: str, *, is_markdown: bool) -> list[Block]:
    """Group raw lines into typed blocks."""
    lines = segment.split("\n")
    blocks: list[Block] = []
    buffer: list[str] = []
    buffer_kind = BlockKind.PARAGRAPH
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        nonlocal buffer, buffer_kind
        if buffer:
            text = clean_text("\n".join(buffer)) if buffer_kind is not BlockKind.CODE else "\n".join(buffer).strip("\n")
            if text.strip():
                blocks.append(Block(text=text, kind=buffer_kind))
        buffer = []
        buffer_kind = BlockKind.PARAGRAPH

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        # ---- fenced code -------------------------------------------------- #
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                flush()
                in_fence, fence_marker = True, marker[:3]
                buffer_kind = BlockKind.CODE
            elif marker.startswith(fence_marker):
                flush()
                in_fence, fence_marker = False, ""
            index += 1
            continue
        if in_fence:
            buffer.append(line)
            index += 1
            continue

        # ---- blank line = block separator --------------------------------- #
        if not stripped:
            flush()
            index += 1
            continue

        # ---- ATX heading -------------------------------------------------- #
        atx = _ATX_HEADING_RE.match(line)
        if atx:
            flush()
            blocks.append(
                Block(
                    text=clean_text(atx.group(2)),
                    kind=BlockKind.HEADING,
                    level=len(atx.group(1)),
                )
            )
            index += 1
            continue

        # ---- Setext heading (underlined) ---------------------------------- #
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if stripped and not buffer:
            if _SETEXT_H1_RE.match(following):
                flush()
                blocks.append(Block(text=clean_text(stripped), kind=BlockKind.HEADING, level=1))
                index += 2
                continue
            if _SETEXT_H2_RE.match(following) and not _BULLET_RE.match(line):
                flush()
                blocks.append(Block(text=clean_text(stripped), kind=BlockKind.HEADING, level=2))
                index += 2
                continue

        # ---- Markdown table ----------------------------------------------- #
        if _TABLE_ROW_RE.match(line):
            flush()
            table_lines: list[str] = []
            while index < len(lines) and _TABLE_ROW_RE.match(lines[index]):
                candidate = lines[index].strip()
                index += 1
                # Drop the alignment row (``|---|:--:|``). It carries no
                # information but would otherwise be embedded, retrieved and
                # shown to the user as part of a cited passage.
                if _TABLE_SEP_RE.match(candidate):
                    continue
                table_lines.append(candidate)
            if table_lines:
                blocks.append(Block(text="\n".join(table_lines), kind=BlockKind.TABLE))
            continue

        # ---- Block quote --------------------------------------------------- #
        if _QUOTE_RE.match(line):
            flush()
            quote_lines: list[str] = []
            while index < len(lines) and _QUOTE_RE.match(lines[index]):
                quote_lines.append(_QUOTE_RE.sub("", lines[index]).strip())
                index += 1
            blocks.append(Block(text=clean_text(" ".join(quote_lines)), kind=BlockKind.QUOTE))
            continue

        # ---- List item ----------------------------------------------------- #
        if _BULLET_RE.match(line):
            flush()
            item_lines = [stripped]
            index += 1
            # Absorb continuation lines that are indented and not new items.
            while (
                index < len(lines)
                and lines[index].strip()
                and not _BULLET_RE.match(lines[index])
                and lines[index].startswith((" ", "\t"))
            ):
                item_lines.append(lines[index].strip())
                index += 1
            blocks.append(Block(text=clean_text(" ".join(item_lines)), kind=BlockKind.LIST_ITEM))
            continue

        # ---- Caption ------------------------------------------------------- #
        if _CAPTION_RE.match(stripped) and len(stripped) < 220:
            flush()
            blocks.append(Block(text=clean_text(stripped), kind=BlockKind.CAPTION))
            index += 1
            continue

        # ---- Plain-text heading (only for non-Markdown, or numbered) ------- #
        if _looks_like_plain_heading(stripped, following, is_markdown=is_markdown):
            flush()
            level, title = _heading_level_and_text(stripped)
            blocks.append(Block(text=clean_text(title), kind=BlockKind.HEADING, level=level))
            index += 1
            continue

        buffer.append(stripped)
        index += 1

    flush()
    return blocks


def _looks_like_plain_heading(line: str, following: str, *, is_markdown: bool) -> bool:
    """Heuristic heading detector for documents without explicit markup."""
    if not line or len(line) > 110:
        return False
    if line.endswith((".", ",", ";", ":")) and not _NUMBERED_HEADING_RE.match(line):
        return False
    words = line.split()
    if not 1 <= len(words) <= 14:
        return False

    numbered = _NUMBERED_HEADING_RE.match(line)
    if numbered and len(numbered.group(2).split()) <= 12:
        # "3.2 Drying Kinetics" - but not "1. and then we did the following"
        rest = numbered.group(2)
        if rest[:1].isupper() or rest.isupper():
            return True

    if is_markdown:
        # In Markdown, trust '#' rather than guessing; avoids treating emphatic
        # short sentences as headings.
        return False

    letters = [c for c in line if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and len(words) <= 10:
        return True

    # Title Case with no sentence punctuation, followed by a blank line or body.
    capitalised = sum(1 for w in words if w[:1].isupper())
    if capitalised >= max(2, int(0.7 * len(words))) and not line.endswith("."):
        if not following.strip() or following.strip()[:1].isupper() or True:
            return True
    return False


def _heading_level_and_text(line: str) -> tuple[int, str]:
    """Derive a heading depth from a leading section number."""
    match = _NUMBERED_HEADING_RE.match(line)
    if match:
        depth = match.group(1).count(".") + 1
        return min(depth, 6), f"{match.group(1)} {match.group(2)}".strip()
    letters = [c for c in line if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return 1, line
    return 2, line


def _paginate_on_breaks(blocks: list[Block]) -> list[Page]:
    pages: list[Page] = []
    current = Page(number=1)
    for block in blocks:
        if block.text == _PAGE_BREAK.text:
            if current.blocks:
                pages.append(current)
            current = Page(number=len(pages) + 1)
            continue
        block.page = current.number
        current.blocks.append(block)
    if current.blocks:
        pages.append(current)
    return pages or [Page(number=1)]


def _paginate_by_size(blocks: list[Block], chars_per_page: int) -> list[Page]:
    """Group blocks into synthetic pages without splitting any block."""
    limit = max(500, int(chars_per_page))
    pages: list[Page] = []
    current = Page(number=1)
    used = 0
    for block in blocks:
        if block.text == _PAGE_BREAK.text:
            continue
        size = block.char_count + 2
        # Start a new page when full, but never leave a page empty and never
        # break immediately before a heading's own body.
        if used and used + size > limit and not (block.is_heading and used > limit * 0.9):
            pages.append(current)
            current = Page(number=len(pages) + 1)
            used = 0
        block.page = current.number
        current.blocks.append(block)
        used += size
    if current.blocks or not pages:
        pages.append(current)
    return pages


def _infer_title(document: ExtractedDocument) -> str | None:
    """Use the first heading, or the first short line, as a title."""
    for block in document.iter_blocks():
        if block.is_heading and 3 <= len(block.text) <= 160:
            return block.text
    for block in document.iter_blocks():
        first_line = block.text.split("\n", 1)[0].strip()
        if 3 <= len(first_line) <= 120:
            return first_line
    return None
