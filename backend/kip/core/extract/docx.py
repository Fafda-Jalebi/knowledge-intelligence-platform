"""DOCX extraction using only ``zipfile`` + ``xml.etree`` from the stdlib.

A ``.docx`` is an OPC (ZIP) package containing WordprocessingML. Everything we
need is available without ``python-docx``:

* ``word/document.xml``   - body content, in document order
* ``docProps/core.xml``   - Dublin Core metadata (title, author, dates)
* ``docProps/app.xml``    - rendered page/word counts written by Word

Structure recovery
------------------
* ``w:pStyle`` values matching ``Heading N``/``Title``/``Subtitle`` become
  headings with the correct level; ``w:outlineLvl`` is used as a fallback so
  documents using custom style names still yield structure.
* ``w:numPr`` marks list items.
* ``w:tbl`` is flattened to a pipe-delimited table block, which keeps row and
  column relationships readable for both retrieval and the source viewer.
* ``w:br w:type="page"`` and ``w:lastRenderedPageBreak`` drive pagination, so
  citations can carry a real page number when Word recorded one.

Security
--------
XML is parsed with entity resolution disabled and a hard limit on both
compressed and uncompressed size, which blocks billion-laughs and ZIP-bomb
style payloads. Only the specific parts listed above are read from the archive,
so a hostile package cannot make us open arbitrary members.
"""

from __future__ import annotations

import io
import re
import zipfile
from xml.etree import ElementTree

from kip.core.extract.base import (
    Block,
    BlockKind,
    DocumentMetadata,
    ExtractedDocument,
    Page,
)
from kip.core.text import clean_text
from kip.errors import ExtractionError

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DC = "{http://purl.org/dc/elements/1.1/}"
DCTERMS = "{http://purl.org/dc/terms/}"
CP = "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}"
EP = "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}"

DOCUMENT_PART = "word/document.xml"
CORE_PART = "docProps/core.xml"
APP_PART = "docProps/app.xml"

#: Refuse to inflate more than this from a single part (ZIP-bomb guard).
MAX_PART_BYTES = 64 * 1024 * 1024
#: Refuse archives claiming an implausible inflate ratio.
MAX_COMPRESSION_RATIO = 200

_HEADING_STYLE_RE = re.compile(r"^heading\s*([1-9])$", re.IGNORECASE)
_LIST_STYLE_RE = re.compile(r"list(paragraph|bullet|number)", re.IGNORECASE)
_CAPTION_STYLE_RE = re.compile(r"caption", re.IGNORECASE)
_QUOTE_STYLE_RE = re.compile(r"quote", re.IGNORECASE)
_CODE_STYLE_RE = re.compile(r"(code|sourcecode|preformatted|htmlpre)", re.IGNORECASE)


def extract_docx(data: bytes) -> ExtractedDocument:
    """Extract a ``.docx`` byte payload into an :class:`ExtractedDocument`."""
    document = ExtractedDocument(extractor="docx:builtin")

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ExtractionError(
            "This file is not a readable Word document. It may be corrupt."
        ) from exc

    with archive:
        _guard_archive(archive)
        try:
            body_xml = _read_part(archive, DOCUMENT_PART)
        except KeyError as exc:
            raise ExtractionError(
                "The Word document is missing its main content part "
                "(word/document.xml) and cannot be read."
            ) from exc

        document.metadata = _read_metadata(archive)
        root = _parse_xml(body_xml)
        body = root.find(f"{W}body")
        if body is None:
            raise ExtractionError("The Word document contains no document body.")

        blocks, page_breaks = _walk_body(body)

    if not blocks:
        document.add_warning("The Word document contained no readable text.")
        document.pages = [Page(number=1)]
        return document

    document.pages = _assign_pages(blocks, page_breaks)
    if not document.metadata.title:
        document.metadata.title = _infer_title(blocks)
    if len(document.pages) == 1 and len(blocks) > 40:
        document.metadata.extra.setdefault("pagination", "single-page")
        document.add_warning(
            "Word did not record page boundaries in this file, so all passages "
            "are reported on page 1. Section names are still available."
        )
    return document.renumber()


# --------------------------------------------------------------------------- #
# Archive / XML safety
# --------------------------------------------------------------------------- #


def _guard_archive(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        if info.file_size > MAX_PART_BYTES:
            raise ExtractionError("The Word document contains an implausibly large part.")
        if info.compress_size > 0:
            ratio = info.file_size / max(1, info.compress_size)
            if ratio > MAX_COMPRESSION_RATIO and info.file_size > 4 * 1024 * 1024:
                raise ExtractionError(
                    "The Word document appears to be a compression bomb and was rejected."
                )


def _read_part(archive: zipfile.ZipFile, name: str) -> bytes:
    with archive.open(name) as handle:
        payload = handle.read(MAX_PART_BYTES + 1)
    if len(payload) > MAX_PART_BYTES:
        raise ExtractionError("The Word document part is too large to process.")
    return payload


def _parse_xml(payload: bytes) -> ElementTree.Element:
    """Parse XML with external entity resolution disabled."""
    parser = ElementTree.XMLParser()
    # Defuse entity expansion: ElementTree's C parser does not expand external
    # entities, and we additionally refuse any DOCTYPE declaration.
    head = payload[:2048].lower()
    if b"<!doctype" in head or b"<!entity" in head:
        raise ExtractionError(
            "The document declares an XML DOCTYPE, which is not accepted."
        )
    try:
        return ElementTree.fromstring(payload, parser=parser)
    except ElementTree.ParseError as exc:
        raise ExtractionError(
            "The Word document contains malformed XML and cannot be read."
        ) from exc


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def _read_metadata(archive: zipfile.ZipFile) -> DocumentMetadata:
    metadata = DocumentMetadata()
    names = set(archive.namelist())

    if CORE_PART in names:
        try:
            core = _parse_xml(_read_part(archive, CORE_PART))
            metadata.title = _text_of(core.find(f"{DC}title"))
            metadata.author = _text_of(core.find(f"{DC}creator"))
            metadata.subject = _text_of(core.find(f"{DC}subject")) or _text_of(
                core.find(f"{DC}description")
            )
            metadata.created_at = _text_of(core.find(f"{DCTERMS}created"))
            metadata.modified_at = _text_of(core.find(f"{DCTERMS}modified"))
            metadata.language = _text_of(core.find(f"{DC}language"))
            keywords = _text_of(core.find(f"{CP}keywords"))
            if keywords:
                metadata.extra["keywords"] = keywords
        except ExtractionError:
            pass  # metadata is best-effort; never fail ingestion over it

    if APP_PART in names:
        try:
            app = _parse_xml(_read_part(archive, APP_PART))
            for tag, key in (("Pages", "reported_pages"), ("Words", "reported_words"),
                             ("Application", "creator")):
                value = _text_of(app.find(f"{EP}{tag}"))
                if value:
                    if key == "creator":
                        metadata.creator = value
                    else:
                        metadata.extra[key] = value
        except ExtractionError:
            pass

    return metadata


def _text_of(node: ElementTree.Element | None) -> str | None:
    if node is None:
        return None
    value = "".join(node.itertext()).strip()
    return value or None


# --------------------------------------------------------------------------- #
# Body traversal
# --------------------------------------------------------------------------- #


def _walk_body(body: ElementTree.Element) -> tuple[list[Block], set[int]]:
    """Return document-ordered blocks plus the indices that start a new page."""
    blocks: list[Block] = []
    page_breaks: set[int] = set()

    for element in body:
        tag = element.tag
        if tag == f"{W}p":
            text, kind, level, breaks_before = _read_paragraph(element)
            if breaks_before:
                page_breaks.add(len(blocks))
            if text:
                blocks.append(Block(text=text, kind=kind, level=level))
        elif tag == f"{W}tbl":
            rendered = _read_table(element)
            if rendered:
                blocks.append(Block(text=rendered, kind=BlockKind.TABLE))
        elif tag == f"{W}sectPr":
            continue
        else:
            # Content controls (w:sdt) and similar wrappers hold paragraphs.
            for nested in element.iter(f"{W}p"):
                text, kind, level, _ = _read_paragraph(nested)
                if text:
                    blocks.append(Block(text=text, kind=kind, level=level))

    return blocks, page_breaks


def _read_paragraph(
    paragraph: ElementTree.Element,
) -> tuple[str, BlockKind, int | None, bool]:
    """Return ``(text, kind, heading_level, starts_new_page)``."""
    properties = paragraph.find(f"{W}pPr")
    style = ""
    outline: int | None = None
    is_list = False

    if properties is not None:
        style_node = properties.find(f"{W}pStyle")
        if style_node is not None:
            style = (style_node.get(f"{W}val") or "").strip()
        outline_node = properties.find(f"{W}outlineLvl")
        if outline_node is not None:
            try:
                outline = int(outline_node.get(f"{W}val") or "")
            except ValueError:
                outline = None
        is_list = properties.find(f"{W}numPr") is not None

    starts_new_page = False
    pieces: list[str] = []

    for node in paragraph.iter():
        tag = node.tag
        if tag == f"{W}t":
            pieces.append(node.text or "")
        elif tag == f"{W}tab":
            pieces.append("\t")
        elif tag == f"{W}br":
            if (node.get(f"{W}type") or "") == "page":
                starts_new_page = True
            else:
                pieces.append("\n")
        elif tag == f"{W}lastRenderedPageBreak":
            starts_new_page = True
        elif tag == f"{W}cr":
            pieces.append("\n")
        elif tag == f"{W}noBreakHyphen":
            pieces.append("-")

    text = clean_text("".join(pieces))
    if not text:
        return "", BlockKind.PARAGRAPH, None, starts_new_page

    kind, level = _classify(style, outline, is_list, text)
    return text, kind, level, starts_new_page


def _classify(
    style: str, outline: int | None, is_list: bool, text: str
) -> tuple[BlockKind, int | None]:
    heading = _HEADING_STYLE_RE.match(style)
    if heading:
        return BlockKind.HEADING, int(heading.group(1))
    lowered = style.lower()
    if lowered == "title":
        return BlockKind.HEADING, 1
    if lowered == "subtitle":
        return BlockKind.HEADING, 2
    if _CAPTION_STYLE_RE.search(style):
        return BlockKind.CAPTION, None
    if _CODE_STYLE_RE.search(style):
        return BlockKind.CODE, None
    if _QUOTE_STYLE_RE.search(style):
        return BlockKind.QUOTE, None
    if outline is not None and 0 <= outline <= 8:
        return BlockKind.HEADING, outline + 1
    # ``w:numPr`` is the authoritative list marker, but Word frequently defines
    # numbering on the *style* instead, leaving ``pPr`` without it. Treating
    # ListParagraph/ListBullet/ListNumber as a list too keeps bulleted content
    # from being flattened into prose, which matters because chunking groups
    # list items differently from paragraphs.
    if is_list or _LIST_STYLE_RE.search(style):
        return BlockKind.LIST_ITEM, None
    return BlockKind.PARAGRAPH, None


def _read_table(table: ElementTree.Element) -> str:
    """Flatten a table to pipe-delimited rows."""
    rows: list[str] = []
    for row in table.findall(f"{W}tr"):
        cells: list[str] = []
        for cell in row.findall(f"{W}tc"):
            pieces = [node.text or "" for node in cell.iter(f"{W}t")]
            cells.append(" ".join("".join(pieces).split()))
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _assign_pages(blocks: list[Block], page_breaks: set[int]) -> list[Page]:
    pages: list[Page] = []
    current = Page(number=1)
    for index, block in enumerate(blocks):
        if index in page_breaks and current.blocks:
            pages.append(current)
            current = Page(number=len(pages) + 1)
        block.page = current.number
        current.blocks.append(block)
    if current.blocks or not pages:
        pages.append(current)
    return pages


def _infer_title(blocks: list[Block]) -> str | None:
    for block in blocks[:12]:
        if block.is_heading and 3 <= len(block.text) <= 160:
            return block.text
    for block in blocks[:4]:
        if 3 <= len(block.text) <= 140:
            return block.text
    return None
