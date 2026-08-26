"""Self-checks for extraction, outline recovery and chunking.

The interesting property here is the *round trip*: :mod:`kip.core.docgen` writes
real PDF and DOCX files, the extractors read them back, and the checks assert on
text we know went in. That is what makes these assertions meaningful rather than
tautological -- a bug in the PDF stream parser or the WordprocessingML walker
shows up as missing or mangled text.

The citation-integrity check is the most important one: every chunk body must be
reconstructible from the blocks it claims to come from. If that ever fails, the
platform would be capable of showing a user a "source passage" that is not
actually in their document.
"""

from __future__ import annotations

import re
from typing import Any

from kip.core import docgen
from kip.core.chunking import ChunkConfig, chunk_document, chunk_text
from kip.core.extract import (
    ExtractedDocument,
    available_backends,
    extract_docx,
    extract_document,
    extract_pdf,
    extract_plain,
    looks_like_scanned,
)
from kip.core.extract.base import Block, BlockKind
from kip.core.structure import build_outline, outline_from_blocks, section_number
from kip.errors import ExtractionError, UnsupportedMediaTypeError
from selfcheck.harness import Harness

# --------------------------------------------------------------------------- #
# Fixture document
# --------------------------------------------------------------------------- #

TITLE = "Thin-Layer Drying of Fruit Slices"
SENTINEL_TABLE = ("Temperature (C)", "Moisture ratio", "Time (min)")


def build_spec() -> docgen.DocSpec:
    """A document with every structural feature the extractors must recover."""
    spec = docgen.DocSpec(
        title=TITLE,
        author="Synthetic Demo Corpus",
        subject="Drying kinetics reference sheet",
    )
    spec.heading("1 Introduction", 1)
    spec.para(
        "Thin-layer drying describes moisture removal from a single layer of "
        "material exposed to a controlled airflow. The moisture ratio falls "
        "rapidly during the initial falling-rate period and then approaches "
        "equilibrium asymptotically."
    )
    spec.para(
        "Water activity governs microbial stability. For most intermediate "
        "moisture products the target water activity is at or below 0.60, which "
        "suppresses growth of osmophilic yeasts."
    )
    spec.heading("2 Method", 1)
    spec.heading("2.1 Sample preparation", 2)
    spec.para(
        "Slices were cut to a uniform thickness of four millimetres using a "
        "calibrated mandoline. Each batch was blanched for ninety seconds to "
        "inactivate polyphenol oxidase before loading onto the trays."
    )
    spec.bullet("Blanch for ninety seconds at ninety-five degrees Celsius.")
    spec.bullet("Drain for two minutes before loading the trays.")
    spec.bullet("Load a single layer with no slice overlap.")
    spec.heading("2.2 Drying schedule", 2)
    spec.table(
        [
            list(SENTINEL_TABLE),
            ["50", "0.42", "240"],
            ["60", "0.28", "180"],
            ["70", "0.19", "120"],
        ]
    )
    spec.para(
        "Table 1 shows the measured moisture ratio at three air temperatures. "
        "Higher air temperature shortened drying time but increased surface "
        "case hardening, which slowed later moisture diffusion."
    )
    spec.heading("3 Results and discussion", 1)
    spec.para(
        "The Page model fitted the measured curves more closely than the "
        "Newton model across all three temperatures. Residuals were largest in "
        "the first thirty minutes, where evaporative cooling of the surface is "
        "not captured by either model."
    )
    spec.para(
        "Rehydration capacity declined as drying temperature rose. This is "
        "consistent with irreversible collapse of the cell wall structure at "
        "higher surface temperatures."
    )
    spec.heading("4 Conclusion", 1)
    spec.para("Drying at sixty degrees Celsius gave the best balance of throughput and quality.")
    return spec


def build_long_spec() -> docgen.DocSpec:
    """A document long enough to span several PDF pages."""
    spec = docgen.DocSpec(title="Batch Drying Log", author="Synthetic Demo Corpus")
    for batch in range(1, 25):
        spec.heading(f"Batch {batch}", 2)
        spec.para(
            f"Batch {batch} was held at sixty degrees Celsius with an air velocity of "
            f"one point two metres per second. The final moisture content was "
            f"{6 + (batch % 5)} percent on a wet basis, and the tray load was "
            f"{2 + (batch % 3)} kilograms per square metre. No case hardening was "
            f"observed on visual inspection of the finished slices."
        )
    return spec


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _flat(text: str) -> str:
    """Collapse all whitespace so comparisons ignore line wrapping."""
    return re.sub(r"\s+", " ", text).strip()


def _check_common(h: Harness, document: ExtractedDocument, label: str) -> None:
    """Assertions that must hold for every extractor."""
    flat = _flat(document.text)
    h.ok(document.has_text, f"{label}: extracted some text")
    h.ok(document.page_count >= 1, f"{label}: has at least one page")
    h.contains(flat, "Thin-layer drying describes moisture removal", f"{label}: body text intact")
    h.contains(flat, "target water activity is at or below 0.60", f"{label}: numbers preserved")
    h.contains(flat, "polyphenol oxidase", f"{label}: technical term preserved")
    h.contains(flat, "Page model fitted the measured curves", f"{label}: later section intact")
    h.ok(len(document.headings()) >= 4, f"{label}: recovered >= 4 headings")
    h.ok(
        all(block.page >= 1 for block in document.iter_blocks()),
        f"{label}: every block has a page number",
    )
    orders = [block.order for block in document.iter_blocks()]
    h.equal(orders, list(range(len(orders))), f"{label}: block order is dense and gap-free")


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def check_pdf(h: Harness) -> ExtractedDocument:
    h.group("PDF round trip")
    spec = build_spec()

    compressed = docgen.render_pdf(spec, compress=True)
    h.ok(compressed.startswith(b"%PDF-1.4"), "PDF: header written")
    h.contains(compressed, b"/Filter /FlateDecode", "PDF: content stream is compressed")
    h.contains(compressed, b"%%EOF", "PDF: trailer written")
    h.ok(len(compressed) > 2000, "PDF: non-trivial size")

    document = h.no_raise(lambda: extract_pdf(compressed), "PDF: FlateDecode stream parsed")
    if document is None:
        return ExtractedDocument()
    _check_common(h, document, "PDF")
    h.contains(document.extractor, "pdf:", "PDF: extractor identified itself")
    h.equal(document.metadata.title, TITLE, "PDF: /Info title recovered")
    h.equal(document.metadata.author, "Synthetic Demo Corpus", "PDF: /Info author recovered")
    h.contains(_flat(document.text), "Temperature (C)", "PDF: table header text preserved")
    h.ok(not looks_like_scanned(document), "PDF: text layer is legible")

    # Both backends must be exercised. pypdf is preferred at runtime when
    # installed, which would otherwise leave the stdlib parser untested -- and
    # the stdlib parser is what makes the platform work without any wheel.
    backends = available_backends()
    h.contains(backends, "builtin", "PDF: builtin backend always advertised")
    builtin = h.no_raise(
        lambda: extract_pdf(compressed, prefer_pypdf=False),
        "PDF: builtin (zlib) parser extracted the document",
    )
    if builtin is not None:
        h.equal(builtin.extractor, "pdf:builtin", "PDF: builtin backend labelled correctly")
        _check_common(h, builtin, "PDF/builtin")
        if "pypdf" in backends:
            h.equal(
                _flat(builtin.text),
                _flat(document.text),
                "PDF: builtin parser agrees with pypdf character for character",
            )
            h.equal(
                builtin.metadata.title,
                document.metadata.title,
                "PDF: both backends recover the same title",
            )

    uncompressed = docgen.render_pdf(spec, compress=False)
    h.ok(b"/Filter" not in uncompressed, "PDF: uncompressed variant has no filter")
    plain_doc = h.no_raise(
        lambda: extract_pdf(uncompressed, prefer_pypdf=False),
        "PDF: raw (unfiltered) stream parsed by builtin backend",
    )
    if plain_doc is not None:
        h.equal(
            _flat(plain_doc.text),
            _flat(document.text),
            "PDF: compressed and uncompressed extractions agree",
        )

    # Pagination: a document longer than one page must yield several pages, and
    # every page must be reachable, or citations would point at the wrong place.
    long_doc = h.no_raise(
        lambda: extract_pdf(docgen.render_pdf(build_long_spec())),
        "PDF: multi-page document parsed",
    )
    if long_doc is not None:
        h.ok(long_doc.page_count >= 3, "PDF: long document spans three or more pages")
        h.equal(
            [page.number for page in long_doc.pages],
            list(range(1, long_doc.page_count + 1)),
            "PDF: page numbers are consecutive from 1",
        )
        h.equal(long_doc.empty_page_numbers, [], "PDF: no page came back empty")
        h.contains(_flat(long_doc.text), "Batch 24 was held", "PDF: text on the last page recovered")
        h.ok(
            len({block.page for block in long_doc.iter_blocks()}) >= 3,
            "PDF: blocks are distributed across pages",
        )
        builtin_long = h.no_raise(
            lambda: extract_pdf(docgen.render_pdf(build_long_spec()), prefer_pypdf=False),
            "PDF: multi-page document parsed by builtin backend",
        )
        if builtin_long is not None:
            h.equal(
                builtin_long.page_count,
                long_doc.page_count,
                "PDF: both backends agree on page count",
            )

    # Failure modes.
    h.raises(ExtractionError, lambda: extract_pdf(b"not a pdf at all"), "PDF: garbage rejected")
    h.raises(
        ExtractionError,
        lambda: extract_pdf(b"%PDF-1.4\n" + b"\x00" * 400),
        "PDF: header-only garbage rejected",
    )
    truncated = compressed[: len(compressed) // 3]
    h.raises(
        ExtractionError,
        lambda: extract_pdf(truncated),
        "PDF: truncated file rejected rather than silently empty",
    )
    h.raises(
        ExtractionError,
        lambda: extract_pdf(truncated, prefer_pypdf=False),
        "PDF: builtin backend also rejects a truncated file",
    )
    return document


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #


def check_docx(h: Harness) -> ExtractedDocument:
    h.group("DOCX round trip")
    spec = build_spec()
    payload = docgen.render_docx(spec, page_break_every=6)
    h.ok(payload.startswith(b"PK"), "DOCX: is a ZIP package")

    document = h.no_raise(lambda: extract_docx(payload), "DOCX: package parsed")
    if document is None:
        return ExtractedDocument()
    _check_common(h, document, "DOCX")
    h.equal(document.extractor, "docx:builtin", "DOCX: extractor label")
    h.equal(document.metadata.title, TITLE, "DOCX: core.xml title recovered")
    h.equal(document.metadata.author, "Synthetic Demo Corpus", "DOCX: core.xml author recovered")
    h.equal(document.metadata.language, "en", "DOCX: core.xml language recovered")
    h.ok(document.page_count >= 2, "DOCX: explicit page breaks produced pages")

    tables = [b for b in document.iter_blocks() if b.kind is BlockKind.TABLE]
    h.equal(len(tables), 1, "DOCX: exactly one table block")
    if tables:
        h.contains(tables[0].text, "Temperature (C) | Moisture ratio", "DOCX: table row flattened")
        h.equal(len(tables[0].text.split("\n")), 4, "DOCX: all four table rows kept")

    lists = [b for b in document.iter_blocks() if b.kind is BlockKind.LIST_ITEM]
    h.equal(len(lists), 3, "DOCX: three list items classified")

    levels = {b.text: b.level for b in document.headings()}
    h.equal(levels.get("1 Introduction"), 1, "DOCX: Heading1 level")
    h.equal(levels.get("2.1 Sample preparation"), 2, "DOCX: Heading2 level")

    # Failure modes.
    h.raises(ExtractionError, lambda: extract_docx(b"PK\x03\x04 truncated"), "DOCX: bad ZIP rejected")
    h.raises(
        ExtractionError,
        lambda: extract_docx(_zip_without_document()),
        "DOCX: missing word/document.xml rejected",
    )
    h.raises(
        ExtractionError,
        lambda: extract_docx(_zip_with_doctype()),
        "DOCX: XML DOCTYPE declaration rejected (XXE guard)",
    )
    return document


def _zip_without_document() -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/settings.xml", "<settings/>")
    return buffer.getvalue()


def _zip_with_doctype() -> bytes:
    import io
    import zipfile

    hostile = (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE root [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>&xxe;</w:t></w:r></w:p></w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", hostile)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Markdown and plain text
# --------------------------------------------------------------------------- #


def check_markdown(h: Harness) -> ExtractedDocument:
    h.group("Markdown extraction")
    spec = build_spec()
    source = docgen.render_markdown(spec)
    h.ok(source.startswith("---"), "MD: front matter written")

    document = h.no_raise(
        lambda: extract_plain(source.encode("utf-8"), is_markdown=True), "MD: parsed"
    )
    if document is None:
        return ExtractedDocument()
    _check_common(h, document, "MD")
    h.equal(document.extractor, "markdown:builtin", "MD: extractor label")
    h.equal(document.metadata.title, TITLE, "MD: front-matter title used")
    h.equal(document.metadata.author, "Synthetic Demo Corpus", "MD: front-matter author used")
    h.equal(document.metadata.extra.get("pagination"), "synthetic", "MD: pagination labelled honestly")

    tables = [b for b in document.iter_blocks() if b.kind is BlockKind.TABLE]
    h.equal(len(tables), 1, "MD: pipe table detected as a table block")
    lists = [b for b in document.iter_blocks() if b.kind is BlockKind.LIST_ITEM]
    h.equal(len(lists), 3, "MD: three bullets detected")
    h.ok(
        all(not b.text.startswith("#") for b in document.headings()),
        "MD: heading markers stripped",
    )

    # Form-feed pagination is honoured when present.
    ff = extract_plain("Alpha section\n\nBody one.\n\fBeta section\n\nBody two.\n")
    h.equal(ff.page_count, 2, "TXT: form feed produced two pages")
    h.equal(ff.metadata.extra.get("pagination"), "form-feed", "TXT: form-feed pagination labelled")

    # Encoding resilience.
    utf16 = "Título del documento\n\nCuerpo del texto.\n".encode("utf-16")
    decoded = h.no_raise(lambda: extract_plain(utf16), "TXT: UTF-16 with BOM decoded")
    if decoded is not None:
        h.contains(_flat(decoded.text), "Cuerpo del texto", "TXT: UTF-16 content intact")
    latin = "Café résumé naïve\n\nBody line.\n".encode("cp1252")
    decoded2 = h.no_raise(lambda: extract_plain(latin), "TXT: cp1252 fallback decoded")
    if decoded2 is not None:
        h.contains(_flat(decoded2.text), "Body line", "TXT: cp1252 content intact")

    empty = extract_plain(b"")
    h.ok(not empty.has_text, "TXT: empty input yields no text")
    h.ok(bool(empty.warnings), "TXT: empty input records a warning")
    return document


def check_plain_headings(h: Harness) -> None:
    h.group("Plain-text structure heuristics")
    source = docgen.render_text(build_spec())
    document = extract_plain(source.encode("utf-8"))
    headings = [b.text for b in document.headings()]
    h.contains(headings, "1 Introduction", "TXT: numbered heading detected")
    h.contains(headings, "2.1 Sample preparation", "TXT: nested numbered heading detected")
    levels = {b.text: b.level for b in document.headings()}
    h.equal(levels.get("2.1 Sample preparation"), 2, "TXT: depth inferred from section number")

    # A sentence must not be promoted to a heading.
    prose = extract_plain(
        b"The moisture ratio fell steadily throughout the run, and the trays were rotated.\n"
    )
    h.equal(len(prose.headings()), 0, "TXT: long sentence not treated as a heading")


# --------------------------------------------------------------------------- #
# Registry dispatch
# --------------------------------------------------------------------------- #


def check_registry(h: Harness) -> None:
    h.group("Registry dispatch")
    spec = build_spec()
    pdf_bytes = docgen.render_pdf(spec)
    docx_bytes = docgen.render_docx(spec)
    md_bytes = docgen.render_markdown(spec).encode("utf-8")

    doc = h.no_raise(lambda: extract_document("paper.pdf", pdf_bytes), "registry: .pdf routed")
    if doc is not None:
        h.contains(doc.extractor, "pdf", "registry: PDF extractor selected")
    doc = h.no_raise(lambda: extract_document("notes.docx", docx_bytes), "registry: .docx routed")
    if doc is not None:
        h.equal(doc.extractor, "docx:builtin", "registry: DOCX extractor selected")
    doc = h.no_raise(lambda: extract_document("notes.md", md_bytes), "registry: .md routed")
    if doc is not None:
        h.equal(doc.extractor, "markdown:builtin", "registry: Markdown extractor selected")
    doc = h.no_raise(lambda: extract_document("notes.txt", b"Plain body text.\n"), "registry: .txt routed")
    if doc is not None:
        h.equal(doc.extractor, "text:builtin", "registry: text extractor selected")

    # Content beats the declared extension.
    doc = h.no_raise(
        lambda: extract_document("mislabelled.txt", pdf_bytes),
        "registry: PDF bytes named .txt still parsed as PDF",
    )
    if doc is not None:
        h.contains(doc.extractor, "pdf", "registry: content sniffing overrode extension")

    h.raises(
        ExtractionError,
        lambda: extract_document("fake.pdf", b"I am plain text pretending to be a PDF."),
        "registry: text named .pdf rejected",
    )
    h.raises(
        ExtractionError,
        lambda: extract_document("fake.docx", b"I am plain text pretending to be a docx."),
        "registry: text named .docx rejected",
    )
    h.raises(
        UnsupportedMediaTypeError,
        lambda: extract_document("payload.exe", b"MZ\x90\x00binary"),
        "registry: unsupported extension refused",
    )
    h.raises(
        ExtractionError,
        lambda: extract_document("empty.txt", b""),
        "registry: empty payload refused",
    )
    h.raises(
        UnsupportedMediaTypeError,
        lambda: extract_document("notes.md", md_bytes, allowed=["pdf"]),
        "registry: allow-list honoured",
    )


# --------------------------------------------------------------------------- #
# Outline
# --------------------------------------------------------------------------- #


def check_structure(h: Harness, document: ExtractedDocument) -> None:
    h.group("Outline recovery")
    outline = build_outline(document)
    labels = outline.headings()
    h.ok(len(labels) >= 4, "outline: recovered the section tree")
    h.contains(labels, "2 Method > 2.1 Sample preparation", "outline: nested path built")
    h.contains(labels, "2 Method > 2.2 Drying schedule", "outline: sibling path built")
    h.ok(not outline.flat, "outline: not reported as flat")

    method = next((s for s in outline.sections() if s.title == "2 Method"), None)
    h.ok(method is not None, "outline: '2 Method' section present")
    if method is not None:
        h.equal(len(method.children), 2, "outline: two subsections under Method")
        h.ok(method.char_count > 0, "outline: section aggregates child characters")
        low, high = method.page_range
        h.ok(low <= high, "outline: page range is ordered")

    keys = [section.key for section in outline.sections()]
    h.equal(len(keys), len(set(keys)), "outline: section keys are unique")
    h.ok(
        all(outline.section_for(block) is not None for block in document.iter_blocks()),
        "outline: every block is owned by a section",
    )

    # Sparse heading levels must be densified, not left with holes.
    sparse = [
        Block(text="Top", kind=BlockKind.HEADING, level=1, order=0),
        Block(text="Body A", order=1),
        Block(text="Deep", kind=BlockKind.HEADING, level=3, order=2),
        Block(text="Body B", order=3),
        Block(text="Deeper", kind=BlockKind.HEADING, level=4, order=4),
        Block(text="Body C", order=5),
    ]
    sparse_outline = outline_from_blocks(sparse)
    h.equal(
        sparse_outline.headings(),
        ["Top", "Top > Deep", "Top > Deep > Deeper"],
        "outline: sparse levels {1,3,4} densified to a 3-deep tree",
    )

    # Text before the first heading must remain addressable.
    preamble = outline_from_blocks(
        [
            Block(text="Abstract paragraph.", order=0),
            Block(text="Chapter One", kind=BlockKind.HEADING, level=1, order=1),
            Block(text="Chapter body.", order=2),
        ],
        fallback_title="My Report",
    )
    h.equal(len(preamble.roots), 2, "outline: preamble kept as its own root")
    h.equal(preamble.roots[0].title, "My Report", "outline: preamble titled from metadata")
    h.ok(preamble.roots[0].is_synthetic, "outline: preamble flagged synthetic")

    # A document with no headings is flat but still valid.
    flat = outline_from_blocks([Block(text="Only body.", order=0)], fallback_title="Untitled")
    h.ok(flat.flat, "outline: heading-free document reported flat")
    h.equal(len(flat.sections()), 1, "outline: heading-free document has one section")
    h.equal(len(outline_from_blocks([]).roots), 1, "outline: empty document still has a root")

    h.equal(section_number("3.2 Drying Kinetics"), "3.2", "outline: section number parsed")
    h.equal(section_number("Introduction"), None, "outline: prose has no section number")


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def check_chunking(h: Harness, document: ExtractedDocument) -> None:
    h.group("Chunking")
    config = ChunkConfig(target_tokens=55, overlap_tokens=18, min_tokens=12, max_tokens=110)
    result = chunk_document(document, config)

    h.ok(len(result) >= 4, "chunking: produced multiple chunks")
    h.ok(
        all(chunk.token_count <= config.max_tokens for chunk in result),
        "chunking: no chunk exceeds max_tokens",
    )
    h.equal(
        [chunk.index for chunk in result],
        list(range(len(result))),
        "chunking: indices are dense and ordered",
    )
    h.ok(all(chunk.body.strip() for chunk in result), "chunking: no empty chunk bodies")
    h.ok(
        all(chunk.page_start <= chunk.page_end for chunk in result),
        "chunking: page ranges are ordered",
    )
    h.ok(
        all(chunk.block_orders for chunk in result),
        "chunking: every chunk records its source blocks",
    )

    # Section confinement: a chunk must never mix two sections.
    h.equal(
        len({chunk.section_key for chunk in result if chunk.section_key is None}),
        0,
        "chunking: every chunk is attributed to a section",
    )
    by_section: dict[str | None, list[int]] = {}
    for chunk in result:
        by_section.setdefault(chunk.section_key, []).append(chunk.index)
    h.ok(
        all(
            indices == list(range(min(indices), max(indices) + 1))
            for indices in by_section.values()
        ),
        "chunking: each section's chunks are contiguous",
    )

    # The embedded text carries section context; the body stays verbatim.
    headed = [c for c in result if c.section_path]
    h.ok(bool(headed), "chunking: section paths propagated to chunks")
    if headed:
        sample = headed[0]
        h.ok(sample.embed_text.startswith("Section: "), "chunking: embed text carries section header")
        h.ok(not sample.body.startswith("Section: "), "chunking: body has no injected header")
        h.contains(sample.embed_text, sample.body, "chunking: embed text wraps the verbatim body")

    # Citation integrity: a chunk body must be reconstructible from its blocks.
    blocks = {block.order: block for block in document.iter_blocks()}
    violations: list[str] = []
    for chunk in result:
        source = _flat(" ".join(blocks[order].text for order in chunk.block_orders))
        if _flat(chunk.body) not in source:
            violations.append(f"chunk {chunk.index} (section {chunk.section_key})")
    h.equal(violations, [], "chunking: every chunk body comes verbatim from its source blocks")

    # Overlap must actually happen, and must repeat real text.
    overlapped = [c for c in result if c.has_overlap]
    h.ok(bool(overlapped), "chunking: overlap applied where a section spanned chunks")
    for chunk in overlapped[:3]:
        previous = result.chunks[chunk.index - 1]
        opening = _flat(chunk.body)[:40]
        h.contains(_flat(previous.body), opening, f"chunking: chunk {chunk.index} overlaps predecessor")

    # Tables stay whole.
    table_chunks = [c for c in result if "table" in c.kinds]
    h.ok(bool(table_chunks), "chunking: table reached a chunk")
    if table_chunks:
        merged = " ".join(_flat(c.body) for c in table_chunks)
        h.contains(merged, "Temperature (C) | Moisture ratio", "chunking: table header row intact")
        h.contains(merged, "70 | 0.19 | 120", "chunking: final table row intact")

    # Determinism.
    again = chunk_document(document, config)
    h.equal(
        [c.body for c in again],
        [c.body for c in result],
        "chunking: identical input yields identical chunks",
    )

    stats = result.stats()
    h.equal(stats["chunks"], len(result), "chunking: stats chunk count agrees")
    h.ok(stats["tokens_max"] <= config.max_tokens, "chunking: stats respect the cap")
    h.between(stats["tokens_mean"], 1, config.max_tokens, "chunking: mean token count is sane")

    h.group("Chunking edge cases")
    h.equal(chunk_text(""), [], "chunking: empty text yields no chunks")
    h.equal(len(chunk_text("Short body.")), 1, "chunking: tiny document yields one chunk")

    # Config clamping prevents a non-terminating overlap.
    clamped = ChunkConfig(target_tokens=100, overlap_tokens=500, min_tokens=999, max_tokens=10)
    h.equal(clamped.overlap_tokens, 50, "chunking: overlap clamped below target")
    h.equal(clamped.max_tokens, 100, "chunking: max raised to at least target")
    h.ok(clamped.min_tokens <= clamped.target_tokens, "chunking: min clamped below target")

    # A run-on with no sentence punctuation must still be split.
    runon = " ".join(f"word{index}" for index in range(900))
    pieces = chunk_text(runon, ChunkConfig(target_tokens=60, overlap_tokens=10, max_tokens=80))
    h.ok(len(pieces) > 5, "chunking: run-on text split into many chunks")
    h.ok(
        all(piece.token_count <= 80 for piece in pieces),
        "chunking: run-on split respects the hard cap",
    )
    h.contains(_flat(" ".join(p.body for p in pieces)), "word899", "chunking: no trailing text lost")

    # Every token of the source must survive chunking (modulo overlap).
    words = set(re.findall(r"word\d+", " ".join(p.body for p in pieces)))
    h.equal(len(words), 900, "chunking: all 900 tokens retained across chunks")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(verbose: bool = False) -> Harness:
    h = Harness(name="extraction", verbose=verbose)
    pdf_doc = check_pdf(h)
    docx_doc = check_docx(h)
    check_markdown(h)
    check_plain_headings(h)
    check_registry(h)
    reference = docx_doc if docx_doc.has_text else pdf_doc
    check_structure(h, reference)
    check_chunking(h, reference)
    return h


def main(argv: list[str] | None = None) -> int:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    harness = run(verbose="-v" in args or "--verbose" in args)
    print(harness.report())
    return 0 if harness.succeeded else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
