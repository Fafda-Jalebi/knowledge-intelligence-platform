"""Minimal PDF and DOCX writers built on the standard library.

Purpose
-------
The demo knowledge base and the extractor test fixtures need real ``.pdf`` and
``.docx`` files. Generating them here rather than committing binaries has three
benefits:

* the demo corpus is reproducible from plain text and reviewable in a diff;
* the extractors are tested against files whose exact expected text is known,
  which is what makes the round-trip assertions meaningful;
* no third-party writer (``reportlab``, ``python-docx``) is required to seed a
  fresh checkout.

Scope is deliberately small: single-column text, Helvetica regular/bold, and
the WordprocessingML subset the DOCX extractor understands. This is a fixture
generator, not a typesetting engine.
"""

from __future__ import annotations

import zipfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Shared document model
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DocBlock:
    """One renderable block: a heading, a paragraph, a bullet or a table row set."""

    text: str
    kind: str = "paragraph"  # paragraph | heading | bullet | table
    level: int = 1


@dataclass(slots=True)
class DocSpec:
    """A document to render, independent of the output format."""

    title: str
    author: str = "Knowledge Intelligence Platform (synthetic demo corpus)"
    subject: str = ""
    blocks: list[DocBlock] = field(default_factory=list)

    def heading(self, text: str, level: int = 1) -> "DocSpec":
        self.blocks.append(DocBlock(text, "heading", level))
        return self

    def para(self, text: str) -> "DocSpec":
        self.blocks.append(DocBlock(" ".join(text.split()), "paragraph"))
        return self

    def bullet(self, text: str) -> "DocSpec":
        self.blocks.append(DocBlock(" ".join(text.split()), "bullet"))
        return self

    def table(self, rows: Sequence[Sequence[str]]) -> "DocSpec":
        rendered = "\n".join(" | ".join(str(cell) for cell in row) for row in rows)
        self.blocks.append(DocBlock(rendered, "table"))
        return self


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #

PAGE_WIDTH = 595  # A4 at 72 dpi
PAGE_HEIGHT = 842
MARGIN_X = 64
MARGIN_TOP = 782
MARGIN_BOTTOM = 72

FONT_REGULAR = "F1"
FONT_BOLD = "F2"

_BODY_SIZE = 10.5
_BODY_LEADING = 15.0
_HEADING_SIZES = {1: 16.0, 2: 13.0, 3: 11.5}

#: Approximate Helvetica advance widths, in units of 1/1000 em, for the ASCII
#: range. Only used for line wrapping, so a coarse table is sufficient.
_NARROW = set("ijltI.,;:'`|!()[]{}/\\ ")
_WIDE = set("mwMW@%")


def _text_width(text: str, size: float) -> float:
    total = 0.0
    for ch in text:
        if ch in _NARROW:
            total += 0.30
        elif ch in _WIDE:
            total += 0.83
        elif ch.isupper():
            total += 0.68
        else:
            total += 0.53
    return total * size


def _wrap(text: str, size: float, max_width: float, *, prefix: str = "") -> list[str]:
    """Greedy word wrap using the approximate width table."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = prefix
    for word in words:
        candidate = f"{current}{word}" if not current or current.endswith(" ") else f"{current} {word}"
        if current.strip() and _text_width(candidate, size) > max_width:
            lines.append(current.rstrip())
            current = " " * len(prefix) + word
        else:
            current = candidate
    if current.strip():
        lines.append(current.rstrip())
    return lines


def _escape_pdf_text(text: str) -> str:
    out = []
    for ch in text:
        if ch in "()\\":
            out.append("\\" + ch)
        elif ord(ch) < 32:
            out.append(" ")
        elif ord(ch) > 255:
            out.append("?")
        else:
            out.append(ch)
    return "".join(out)


@dataclass(slots=True)
class _Line:
    text: str
    font: str
    size: float
    leading: float


def _layout_pdf(spec: DocSpec) -> list[list[_Line]]:
    """Break the spec into pages of positioned lines."""
    usable = PAGE_WIDTH - 2 * MARGIN_X
    pages: list[list[_Line]] = []
    current: list[_Line] = []
    y = MARGIN_TOP

    def newpage() -> None:
        nonlocal current, y
        if current:
            pages.append(current)
        current = []
        y = MARGIN_TOP

    # Title block.
    for line in _wrap(spec.title, 18.0, usable):
        current.append(_Line(line, FONT_BOLD, 18.0, 24.0))
        y -= 24.0
    current.append(_Line("", FONT_REGULAR, _BODY_SIZE, 10.0))
    y -= 10.0

    for block in spec.blocks:
        if block.kind == "heading":
            size = _HEADING_SIZES.get(block.level, 11.0)
            leading = size + 6.0
            lines = _wrap(block.text, size, usable)
            needed = leading * (len(lines) + 1) + 8
            if y - needed < MARGIN_BOTTOM:
                newpage()
            current.append(_Line("", FONT_REGULAR, _BODY_SIZE, 8.0))
            y -= 8.0
            for line in lines:
                current.append(_Line(line, FONT_BOLD, size, leading))
                y -= leading
        elif block.kind == "bullet":
            lines = _wrap(block.text, _BODY_SIZE, usable - 14, prefix="- ")
            for line in lines:
                if y - _BODY_LEADING < MARGIN_BOTTOM:
                    newpage()
                current.append(_Line(line, FONT_REGULAR, _BODY_SIZE, _BODY_LEADING))
                y -= _BODY_LEADING
        elif block.kind == "table":
            for row in block.text.split("\n"):
                for line in _wrap(row, _BODY_SIZE, usable):
                    if y - _BODY_LEADING < MARGIN_BOTTOM:
                        newpage()
                    current.append(_Line(line, FONT_REGULAR, _BODY_SIZE, _BODY_LEADING))
                    y -= _BODY_LEADING
        else:
            lines = _wrap(block.text, _BODY_SIZE, usable)
            for line in lines:
                if y - _BODY_LEADING < MARGIN_BOTTOM:
                    newpage()
                current.append(_Line(line, FONT_REGULAR, _BODY_SIZE, _BODY_LEADING))
                y -= _BODY_LEADING
        # Paragraph spacing.
        if y - 7.0 >= MARGIN_BOTTOM:
            current.append(_Line("", FONT_REGULAR, _BODY_SIZE, 7.0))
            y -= 7.0

    if current:
        pages.append(current)
    return pages or [[]]


def _content_stream(lines: Iterable[_Line]) -> bytes:
    parts: list[str] = ["BT"]
    y = MARGIN_TOP
    started = False
    current_font = ""
    current_size = 0.0
    for line in lines:
        if not started:
            parts.append(f"1 0 0 1 {MARGIN_X} {y:.1f} Tm")
            started = True
        else:
            parts.append(f"0 -{line.leading:.1f} Td")
        y -= line.leading
        if line.font != current_font or abs(line.size - current_size) > 0.01:
            parts.append(f"/{line.font} {line.size:.1f} Tf")
            parts.append(f"{line.leading:.1f} TL")
            current_font, current_size = line.font, line.size
        if line.text:
            parts.append(f"({_escape_pdf_text(line.text)}) Tj")
    parts.append("ET")
    return ("\n".join(parts) + "\n").encode("latin-1", errors="replace")


def render_pdf(spec: DocSpec, *, compress: bool = True) -> bytes:
    """Render ``spec`` to a valid single-column PDF."""
    pages = _layout_pdf(spec)
    n_pages = len(pages)

    # Object numbering: 1 Catalog, 2 Pages, 3 Font, 4 Font-Bold, 5 Info,
    # then page/content pairs from 6.
    first_page_obj = 6
    page_objs = [first_page_obj + 2 * i for i in range(n_pages)]
    content_objs = [first_page_obj + 2 * i + 1 for i in range(n_pages)]
    total_objects = first_page_obj + 2 * n_pages - 1

    kids = " ".join(f"{num} 0 R" for num in page_objs)
    now = datetime.now(timezone.utc).strftime("D:%Y%m%d%H%M%SZ")

    bodies: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode("latin-1"),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
           b"/Encoding /WinAnsiEncoding >>",
        4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
           b"/Encoding /WinAnsiEncoding >>",
        5: (
            "<< /Title ({title}) /Author ({author}) /Subject ({subject}) "
            "/Producer (Knowledge Intelligence Platform docgen) "
            "/CreationDate ({now}) /ModDate ({now}) >>"
        ).format(
            title=_escape_pdf_text(spec.title),
            author=_escape_pdf_text(spec.author),
            subject=_escape_pdf_text(spec.subject or spec.title),
            now=now,
        ).encode("latin-1", errors="replace"),
    }

    for index, page_obj in enumerate(page_objs):
        bodies[page_obj] = (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /{FONT_REGULAR} 3 0 R /{FONT_BOLD} 4 0 R >> >> "
            f"/Contents {content_objs[index]} 0 R >>"
        ).encode("latin-1")

        raw = _content_stream(pages[index])
        if compress:
            payload = zlib.compress(raw, 6)
            header = f"<< /Length {len(payload)} /Filter /FlateDecode >>".encode("latin-1")
        else:
            payload = raw
            header = f"<< /Length {len(payload)} >>".encode("latin-1")
        bodies[content_objs[index]] = header + b"\nstream\n" + payload + b"\nendstream"

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for number in range(1, total_objects + 1):
        body = bodies.get(number)
        if body is None:
            body = b"<< >>"
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode("latin-1")
        out += body
        out += b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {total_objects + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for number in range(1, total_objects + 1):
        out += f"{offsets[number]:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {total_objects + 1} /Root 1 0 R /Info 5 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:styleId="Normal" w:default="1"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:pPr><w:outlineLvl w:val="2"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/></w:style>
</w:styles>"""


def _docx_paragraph(text: str, *, style: str | None = None, numbered: bool = False) -> str:
    properties = ""
    if style or numbered:
        inner = ""
        if style:
            inner += f'<w:pStyle w:val="{style}"/>'
        if numbered:
            inner += '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
        properties = f"<w:pPr>{inner}</w:pPr>"
    runs = ""
    for index, segment in enumerate(text.split("\n")):
        if index:
            runs += "<w:r><w:br/></w:r>"
        runs += f'<w:r><w:t xml:space="preserve">{_xml_escape(segment)}</w:t></w:r>'
    return f"<w:p>{properties}{runs}</w:p>"


def _docx_table(rendered: str) -> str:
    rows = [row.split(" | ") for row in rendered.split("\n") if row.strip()]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    grid = "".join('<w:gridCol w:w="2400"/>' for _ in range(width))
    body = ""
    for row in rows:
        cells = ""
        padded = list(row) + [""] * (width - len(row))
        for cell in padded:
            cells += (
                "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
                f'<w:p><w:r><w:t xml:space="preserve">{_xml_escape(cell.strip())}</w:t></w:r></w:p></w:tc>'
            )
        body += f"<w:tr>{cells}</w:tr>"
    return f"<w:tbl><w:tblGrid>{grid}</w:tblGrid>{body}</w:tbl>"


def render_docx(spec: DocSpec, *, page_break_every: int = 0) -> bytes:
    """Render ``spec`` to a valid ``.docx`` package.

    ``page_break_every`` inserts an explicit page break after that many
    body blocks, so the extractor's pagination logic can be exercised.
    """
    import io

    parts: list[str] = [_docx_paragraph(spec.title, style="Title")]
    body_blocks = 0
    for block in spec.blocks:
        if block.kind == "heading":
            level = min(max(block.level, 1), 3)
            parts.append(_docx_paragraph(block.text, style=f"Heading{level}"))
        elif block.kind == "bullet":
            parts.append(_docx_paragraph(block.text, style="ListParagraph", numbered=True))
        elif block.kind == "table":
            parts.append(_docx_table(block.text))
        else:
            parts.append(_docx_paragraph(block.text))
        body_blocks += 1
        if page_break_every and body_blocks % page_break_every == 0:
            parts.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>" + "".join(parts) +
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr>'
        "</w:body></w:document>"
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{_xml_escape(spec.title)}</dc:title>"
        f"<dc:creator>{_xml_escape(spec.author)}</dc:creator>"
        f"<dc:subject>{_xml_escape(spec.subject or spec.title)}</dc:subject>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "<dc:language>en</dc:language>"
        "</cp:coreProperties>"
    )

    word_count = sum(len(block.text.split()) for block in spec.blocks)
    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
        "<Application>Knowledge Intelligence Platform docgen</Application>"
        f"<Words>{word_count}</Words>"
        "</Properties>"
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
        archive.writestr("_rels/.rels", _ROOT_RELS)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/_rels/document.xml.rels", _DOC_RELS)
        archive.writestr("word/styles.xml", _STYLES)
        archive.writestr("docProps/core.xml", core_xml)
        archive.writestr("docProps/app.xml", app_xml)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Markdown / plain text
# --------------------------------------------------------------------------- #


def render_markdown(spec: DocSpec) -> str:
    """Render ``spec`` to Markdown with YAML front matter."""
    lines = [
        "---",
        f"title: {spec.title}",
        f"author: {spec.author}",
    ]
    if spec.subject:
        lines.append(f"subject: {spec.subject}")
    lines += ["language: en", "---", "", f"# {spec.title}", ""]

    for block in spec.blocks:
        if block.kind == "heading":
            lines += ["#" * min(max(block.level + 1, 2), 6) + f" {block.text}", ""]
        elif block.kind == "bullet":
            lines += [f"- {block.text}", ""]
        elif block.kind == "table":
            rows = [row.split(" | ") for row in block.text.split("\n") if row.strip()]
            if rows:
                header, *rest = rows
                lines.append("| " + " | ".join(header) + " |")
                lines.append("|" + "|".join(" --- " for _ in header) + "|")
                for row in rest:
                    padded = list(row) + [""] * (len(header) - len(row))
                    lines.append("| " + " | ".join(padded[: len(header)]) + " |")
                lines.append("")
        else:
            lines += [block.text, ""]
    return "\n".join(lines).rstrip() + "\n"


def render_text(spec: DocSpec) -> str:
    """Render ``spec`` to plain text with underlined headings."""
    lines = [spec.title, "=" * len(spec.title), "", f"Author: {spec.author}", ""]
    for block in spec.blocks:
        if block.kind == "heading":
            lines += ["", block.text, "-" * len(block.text), ""]
        elif block.kind == "bullet":
            lines.append(f"  - {block.text}")
        elif block.kind == "table":
            lines += ["", block.text, ""]
        else:
            lines += [block.text, ""]
    return "\n".join(lines).rstrip() + "\n"
