"""PDF text extraction.

Strategy
--------
Two implementations, tried in order:

1. **pypdf** (``pdf:pypdf``) - used whenever the optional dependency is
   installed. It is the right tool: it handles cross-reference streams, object
   streams, encryption, and the long tail of font encodings.
2. **Built-in** (``pdf:builtin``) - a compact, dependency-free reader used when
   pypdf is absent. It walks the object graph, inflates ``FlateDecode`` content
   streams with ``zlib``, and interprets the text-showing operators (``Tj``,
   ``TJ``, ``'``, ``"``) plus the positioning operators needed to reconstruct
   line breaks. ``/ToUnicode`` CMaps are honoured so subset-encoded fonts
   still decode.

The fallback exists so the platform ingests PDFs out of the box with zero
installs. It is deliberately *not* presented as equivalent to pypdf: when the
decoded text fails a legibility check the extractor records a warning
recommending pypdf, rather than silently indexing mojibake. Encrypted PDFs are
detected and reported instead of producing empty chunks.
"""

from __future__ import annotations

import logging
import re
import zlib
from contextlib import contextmanager
from typing import Iterable, Iterator

from kip.core.extract.base import (
    Block,
    BlockKind,
    DocumentMetadata,
    ExtractedDocument,
    Page,
)
from kip.core.text import clean_text, collapse_repeated_lines, strip_page_furniture
from kip.errors import ExtractionError

MAX_STREAM_BYTES = 64 * 1024 * 1024
#: Below this ratio of letters/digits/spaces we assume the decode failed.
LEGIBILITY_THRESHOLD = 0.72


def extract_pdf(data: bytes, *, prefer_pypdf: bool = True) -> ExtractedDocument:
    """Extract a PDF byte payload into an :class:`ExtractedDocument`."""
    if not data[:1024].lstrip().startswith(b"%PDF"):
        raise ExtractionError("This file is not a valid PDF.")

    if prefer_pypdf:
        result = _extract_with_pypdf(data)
        if result is not None:
            return result
    return _extract_builtin(data)


# --------------------------------------------------------------------------- #
# Preferred path: pypdf
# --------------------------------------------------------------------------- #


def _extract_with_pypdf(data: bytes) -> ExtractedDocument | None:  # pragma: no cover
    """Return an extraction via pypdf, or ``None`` if pypdf is unavailable."""
    try:
        import io

        from pypdf import PdfReader  # type: ignore
        from pypdf.errors import PdfReadError  # type: ignore
    except Exception:
        return None

    document = ExtractedDocument(extractor="pdf:pypdf")
    # pypdf writes recovery notices ("EOF marker not found", "Ignoring wrong
    # pointing object") straight to its own logger. Those are diagnostics about
    # the uploaded file, not about the platform, and they bypass our structured
    # logging entirely. Silence them here and let the caller's ExtractionError
    # message be the single explanation the user sees.
    with _quiet_logger("pypdf"):
        try:
            reader = PdfReader(io.BytesIO(data), strict=False)
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception:
                    raise ExtractionError(
                        "This PDF is password protected. Remove the password and "
                        "upload it again."
                    )
            info = getattr(reader, "metadata", None) or {}
            document.metadata = DocumentMetadata(
                title=_clean_meta(info.get("/Title")),
                author=_clean_meta(info.get("/Author")),
                subject=_clean_meta(info.get("/Subject")),
                creator=_clean_meta(info.get("/Producer")) or _clean_meta(info.get("/Creator")),
                created_at=_clean_meta(info.get("/CreationDate")),
                modified_at=_clean_meta(info.get("/ModDate")),
            )
            raw_pages: list[list[str]] = []
            for page in reader.pages:
                try:
                    raw_pages.append((page.extract_text() or "").split("\n"))
                except Exception:
                    raw_pages.append([])
        except ExtractionError:
            raise
        except PdfReadError as exc:
            raise ExtractionError(
                "This PDF could not be parsed. The file may be corrupt or truncated."
            ) from exc
        except Exception:
            return None

    _finalise(document, raw_pages)
    return document


@contextmanager
def _quiet_logger(name: str) -> Iterator[None]:
    """Temporarily silence a third-party logger without touching global config."""
    logger = logging.getLogger(name)
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(logging.CRITICAL + 1)
    logger.propagate = False
    try:
        yield
    finally:
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def _clean_meta(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


# --------------------------------------------------------------------------- #
# Fallback path: built-in reader
# --------------------------------------------------------------------------- #

_OBJ_RE = re.compile(rb"(\d+)\s+(\d+)\s+obj\b(.*?)\bendobj", re.DOTALL)
_STREAM_RE = re.compile(rb"stream\r?\n?(.*?)[\r\n]{0,2}endstream", re.DOTALL)
_REF_RE = re.compile(rb"(\d+)\s+\d+\s+R")
_ENCRYPT_RE = re.compile(rb"/Encrypt\b")


def _extract_builtin(data: bytes) -> ExtractedDocument:
    document = ExtractedDocument(extractor="pdf:builtin")

    objects = _parse_objects(data)
    if not objects:
        raise ExtractionError(
            "This PDF has no readable objects. It may be corrupt, or it may use "
            "compressed object streams; install pypdf for full PDF support "
            "(pip install pypdf)."
        )
    if _ENCRYPT_RE.search(data) and _is_truly_encrypted(objects):
        raise ExtractionError(
            "This PDF is encrypted. Remove the password protection and upload it again."
        )

    document.metadata = _builtin_metadata(objects)
    page_numbers = _page_order(objects)

    raw_pages: list[list[str]] = []
    for page_num in page_numbers:
        body = objects.get(page_num, b"")
        fonts = _page_fonts(objects, body)
        content = _page_content(objects, body)
        raw_pages.append(_text_from_content(content, fonts))

    if not raw_pages:
        raise ExtractionError(
            "No pages could be located in this PDF. Install pypdf for broader "
            "PDF support (pip install pypdf)."
        )

    _finalise(document, raw_pages)

    joined = document.text
    if joined and _legibility(joined) < LEGIBILITY_THRESHOLD:
        document.add_warning(
            "Some text in this PDF used an embedded font encoding the built-in "
            "reader could not fully decode. Install pypdf for higher-fidelity "
            "PDF extraction (pip install pypdf)."
        )
    if not document.has_text:
        document.add_warning(
            "No selectable text was found. This PDF is probably a scanned image; "
            "OCR is required before it can be indexed."
        )
    return document


def _parse_objects(data: bytes) -> dict[int, bytes]:
    objects: dict[int, bytes] = {}
    for match in _OBJ_RE.finditer(data):
        number = int(match.group(1))
        # Later definitions (incremental updates) win.
        objects[number] = match.group(3)
    return objects


def _is_truly_encrypted(objects: dict[int, bytes]) -> bool:
    """True when an /Encrypt dictionary with a real filter exists."""
    for body in objects.values():
        if b"/Filter" in body and (b"/Standard" in body and b"/CF" in body or b"/O <" in body or b"/O(" in body):
            if b"/V" in body and b"/R" in body:
                return True
    return False


def _resolve(objects: dict[int, bytes], token: bytes) -> bytes:
    """Follow an indirect reference to its object body."""
    match = _REF_RE.match(token.strip())
    if match:
        return objects.get(int(match.group(1)), b"")
    return token


def _page_order(objects: dict[int, bytes]) -> list[int]:
    """Return page object numbers in reading order."""
    catalog_pages: int | None = None
    for body in objects.values():
        if b"/Type" in body and b"/Catalog" in body:
            match = re.search(rb"/Pages\s+(\d+)\s+\d+\s+R", body)
            if match:
                catalog_pages = int(match.group(1))
                break

    ordered: list[int] = []
    if catalog_pages is not None:
        seen: set[int] = set()
        _walk_page_tree(objects, catalog_pages, ordered, seen, depth=0)

    if not ordered:
        # Fallback: every /Type /Page object, in object-number order.
        ordered = sorted(
            number
            for number, body in objects.items()
            if re.search(rb"/Type\s*/Page\b", body) and not re.search(rb"/Type\s*/Pages\b", body)
        )
    return ordered


def _walk_page_tree(
    objects: dict[int, bytes],
    node_number: int,
    ordered: list[int],
    seen: set[int],
    *,
    depth: int,
) -> None:
    if depth > 64 or node_number in seen:
        return
    seen.add(node_number)
    body = objects.get(node_number, b"")
    if re.search(rb"/Type\s*/Page\b", body) and not re.search(rb"/Type\s*/Pages\b", body):
        ordered.append(node_number)
        return
    kids_match = re.search(rb"/Kids\s*\[(.*?)\]", body, re.DOTALL)
    if not kids_match:
        return
    for ref in _REF_RE.finditer(kids_match.group(1)):
        _walk_page_tree(objects, int(ref.group(1)), ordered, seen, depth=depth + 1)


def _page_content(objects: dict[int, bytes], page_body: bytes) -> bytes:
    """Return the concatenated, decoded content stream(s) for a page."""
    match = re.search(rb"/Contents\s*(\[.*?\]|\d+\s+\d+\s+R)", page_body, re.DOTALL)
    if not match:
        return b""
    token = match.group(1).strip()
    numbers: list[int] = [int(ref.group(1)) for ref in _REF_RE.finditer(token)]

    chunks: list[bytes] = []
    for number in numbers:
        body = objects.get(number, b"")
        stream = _STREAM_RE.search(body)
        if not stream:
            continue
        chunks.append(_decode_stream(body[: stream.start()], stream.group(1)))
    return b"\n".join(chunks)


def _decode_stream(header: bytes, payload: bytes) -> bytes:
    """Apply the stream's filter chain (Flate / AHx / A85 supported)."""
    if len(payload) > MAX_STREAM_BYTES:
        return b""
    filters = [f.decode("ascii", "ignore") for f in re.findall(rb"/(\w+)Decode", header)]
    data = payload
    for name in filters or []:
        if name == "Flate":
            data = _inflate(data)
        elif name == "ASCIIHex":
            data = _ascii_hex_decode(data)
        elif name == "ASCII85":
            data = _ascii85_decode(data)
        elif name in {"DCT", "JPX", "JBIG2", "CCITTFax"}:
            return b""  # image data, no text
        elif name == "LZW":
            return b""  # rare in modern PDFs; pypdf handles it
    return data


def _inflate(payload: bytes) -> bytes:
    for attempt in (payload, payload.lstrip(b"\r\n "), payload[1:]):
        try:
            return zlib.decompress(attempt)
        except zlib.error:
            pass
    # Truncated stream: decompress as much as possible.
    try:
        decompressor = zlib.decompressobj()
        return decompressor.decompress(payload)
    except zlib.error:
        try:
            return zlib.decompressobj(-15).decompress(payload)
        except zlib.error:
            return b""


def _ascii_hex_decode(payload: bytes) -> bytes:
    hexdigits = re.sub(rb"[^0-9A-Fa-f]", b"", payload.split(b">")[0])
    if len(hexdigits) % 2:
        hexdigits += b"0"
    try:
        return bytes.fromhex(hexdigits.decode("ascii"))
    except ValueError:
        return b""


def _ascii85_decode(payload: bytes) -> bytes:
    import base64

    body = payload.strip()
    if body.startswith(b"<~"):
        body = body[2:]
    body = body.split(b"~>")[0]
    try:
        return base64.a85decode(body, adobe=False)
    except Exception:
        return b""


# --------------------------------------------------------------------------- #
# Font decoding (/ToUnicode CMaps)
# --------------------------------------------------------------------------- #

_BFCHAR_RE = re.compile(rb"beginbfchar(.*?)endbfchar", re.DOTALL)
_BFRANGE_RE = re.compile(rb"beginbfrange(.*?)endbfrange", re.DOTALL)
_HEXTOK_RE = re.compile(rb"<([0-9A-Fa-f]+)>")


def _page_fonts(objects: dict[int, bytes], page_body: bytes) -> dict[str, dict[int, str]]:
    """Map font resource names ("F1") to a code->unicode table, if available."""
    resources = _find_resources(objects, page_body)
    if not resources:
        return {}
    font_dict = re.search(rb"/Font\s*(<<.*?>>|\d+\s+\d+\s+R)", resources, re.DOTALL)
    if not font_dict:
        return {}
    font_body = _resolve(objects, font_dict.group(1))
    mapping: dict[str, dict[int, str]] = {}
    for match in re.finditer(rb"/([A-Za-z0-9#_.\-]+)\s+(\d+)\s+\d+\s+R", font_body):
        name = match.group(1).decode("ascii", "ignore")
        font_obj = objects.get(int(match.group(2)), b"")
        tounicode = re.search(rb"/ToUnicode\s+(\d+)\s+\d+\s+R", font_obj)
        if not tounicode:
            continue
        cmap_body = objects.get(int(tounicode.group(1)), b"")
        stream = _STREAM_RE.search(cmap_body)
        if not stream:
            continue
        decoded = _decode_stream(cmap_body[: stream.start()], stream.group(1))
        table = _parse_cmap(decoded)
        if table:
            mapping[name] = table
    return mapping


def _find_resources(objects: dict[int, bytes], page_body: bytes) -> bytes:
    match = re.search(rb"/Resources\s*(<<.*?>>|\d+\s+\d+\s+R)", page_body, re.DOTALL)
    if match:
        return _resolve(objects, match.group(1))
    # Inherited from the parent Pages node.
    parent = re.search(rb"/Parent\s+(\d+)\s+\d+\s+R", page_body)
    if parent:
        parent_body = objects.get(int(parent.group(1)), b"")
        inner = re.search(rb"/Resources\s*(<<.*?>>|\d+\s+\d+\s+R)", parent_body, re.DOTALL)
        if inner:
            return _resolve(objects, inner.group(1))
    return b""


def _parse_cmap(payload: bytes) -> dict[int, str]:
    table: dict[int, str] = {}
    for block in _BFCHAR_RE.findall(payload):
        tokens = _HEXTOK_RE.findall(block)
        for index in range(0, len(tokens) - 1, 2):
            code = _hex_int(tokens[index])
            table[code] = _hex_utf16(tokens[index + 1])
    for block in _BFRANGE_RE.findall(payload):
        for line in block.split(b"\n"):
            tokens = _HEXTOK_RE.findall(line)
            if len(tokens) >= 3:
                low, high, start = _hex_int(tokens[0]), _hex_int(tokens[1]), tokens[2]
                if high - low > 65535:
                    continue
                base = _hex_int(start)
                for offset in range(high - low + 1):
                    table[low + offset] = _codepoint_to_str(base + offset)
    return table


def _hex_int(token: bytes) -> int:
    try:
        return int(token, 16)
    except ValueError:
        return 0


def _hex_utf16(token: bytes) -> str:
    try:
        raw = bytes.fromhex(token.decode("ascii"))
    except ValueError:
        return ""
    if len(raw) >= 2:
        try:
            return raw.decode("utf-16-be", errors="ignore")
        except Exception:
            return ""
    return _codepoint_to_str(raw[0] if raw else 0)


def _codepoint_to_str(value: int) -> str:
    try:
        return chr(value) if 0 < value < 0x110000 else ""
    except (ValueError, OverflowError):
        return ""


# --------------------------------------------------------------------------- #
# Content-stream text extraction
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(
    rb"""
      \((?:\\.|[^\\()]|\((?:\\.|[^\\()])*\))*\)   # literal string
    | <[0-9A-Fa-f\s]*>                            # hex string
    | \[|\]                                       # array delimiters
    | /[^\s/\[\]<>(){}]+                          # name
    | -?\d*\.?\d+                                 # number
    | [A-Za-z'"*]+                                # operator
    """,
    re.VERBOSE | re.DOTALL,
)

#: A y-shift larger than this fraction of the leading starts a new line.
_LINE_BREAK_FACTOR = 0.55


def _text_from_content(content: bytes, fonts: dict[str, dict[int, str]]) -> list[str]:
    """Interpret text operators and reconstruct visual lines."""
    if not content:
        return []

    lines: list[str] = []
    current: list[str] = []
    operands: list[bytes] = []
    array_depth = 0
    array_items: list[bytes] = []

    leading = 12.0
    last_y: float | None = None
    active_font: dict[int, str] | None = None

    def flush_line() -> None:
        if current:
            lines.append("".join(current).strip())
            current.clear()

    for match in _TOKEN_RE.finditer(content):
        token = match.group(0)

        if token == b"[":
            array_depth += 1
            array_items = []
            continue
        if token == b"]":
            array_depth = 0
            operands = [b"[" + b" ".join(array_items) + b"]"]
            continue
        if array_depth:
            array_items.append(token)
            continue

        first = token[:1]
        if first in b"(<" or first == b"/" or first.isdigit() or first in b"-.":
            operands.append(token)
            if len(operands) > 12:
                del operands[:-12]
            continue

        operator = token

        if operator == b"Tf":
            if len(operands) >= 2 and operands[-2].startswith(b"/"):
                name = operands[-2][1:].decode("ascii", "ignore")
                active_font = fonts.get(name)
            operands.clear()
        elif operator == b"TL":
            leading = _as_float(operands[-1] if operands else b"12") or 12.0
            operands.clear()
        elif operator in (b"Td", b"TD"):
            dy = _as_float(operands[-1]) if operands else 0.0
            if operator == b"TD":
                leading = abs(dy) or leading
            if abs(dy) > max(1.0, leading * _LINE_BREAK_FACTOR):
                flush_line()
            elif dy != 0.0:
                _append_space(current)
            operands.clear()
        elif operator == b"Tm":
            y = _as_float(operands[-1]) if operands else None
            if y is not None and last_y is not None:
                if abs(y - last_y) > max(1.0, leading * _LINE_BREAK_FACTOR):
                    flush_line()
            last_y = y
            operands.clear()
        elif operator == b"T*":
            flush_line()
            operands.clear()
        elif operator == b"Tj":
            if operands:
                current.append(_decode_pdf_string(operands[-1], active_font))
            operands.clear()
        elif operator in (b"'", b'"'):
            flush_line()
            if operands:
                current.append(_decode_pdf_string(operands[-1], active_font))
            operands.clear()
        elif operator == b"TJ":
            if operands:
                current.append(_decode_pdf_array(operands[-1], active_font))
            operands.clear()
        elif operator in (b"ET", b"BT"):
            flush_line()
            operands.clear()
        else:
            operands.clear()

    flush_line()
    return [line for line in lines if line]


def _append_space(current: list[str]) -> None:
    if current and not current[-1].endswith(" "):
        current.append(" ")


def _as_float(token: bytes) -> float:
    try:
        return float(token)
    except (TypeError, ValueError):
        return 0.0


_ESCAPES = {
    ord("n"): "\n", ord("r"): "\r", ord("t"): "\t", ord("b"): "\b",
    ord("f"): "\f", ord("("): "(", ord(")"): ")", ord("\\"): "\\",
}


def _decode_pdf_string(token: bytes, font: dict[int, str] | None) -> str:
    if token.startswith(b"<"):
        return _decode_hex_string(token, font)
    if not token.startswith(b"("):
        return ""
    raw = token[1:-1]
    out: list[int] = []
    index = 0
    while index < len(raw):
        byte = raw[index]
        if byte == 0x5C:  # backslash
            index += 1
            if index >= len(raw):
                break
            nxt = raw[index]
            if nxt in _ESCAPES:
                out.extend(_ESCAPES[nxt].encode("latin-1"))
                index += 1
            elif nxt in b"\n":
                index += 1  # line continuation
            elif nxt in b"\r":
                index += 1
                if index < len(raw) and raw[index] == 0x0A:
                    index += 1
            elif 0x30 <= nxt <= 0x37:  # octal
                digits = raw[index : index + 3]
                literal = bytes(d for d in digits if 0x30 <= d <= 0x37)
                out.append(int(literal.decode("ascii"), 8) & 0xFF)
                index += len(literal)
            else:
                out.append(nxt)
                index += 1
        else:
            out.append(byte)
            index += 1
    return _bytes_to_text(bytes(out), font)


def _decode_hex_string(token: bytes, font: dict[int, str] | None) -> str:
    digits = re.sub(rb"[^0-9A-Fa-f]", b"", token)
    if len(digits) % 2:
        digits += b"0"
    try:
        raw = bytes.fromhex(digits.decode("ascii"))
    except ValueError:
        return ""
    return _bytes_to_text(raw, font)


def _bytes_to_text(raw: bytes, font: dict[int, str] | None) -> str:
    if font:
        # Two-byte codes are typical for CID-keyed fonts; try those first.
        if len(raw) % 2 == 0 and any((raw[i] << 8 | raw[i + 1]) in font for i in range(0, len(raw), 2)):
            return "".join(font.get(raw[i] << 8 | raw[i + 1], "") for i in range(0, len(raw), 2))
        if any(byte in font for byte in raw):
            return "".join(font.get(byte, chr(byte)) for byte in raw)
    if raw[:2] in (b"\xfe\xff", b"\xff\xfe"):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass
    return raw.decode("cp1252", errors="replace")


def _decode_pdf_array(token: bytes, font: dict[int, str] | None) -> str:
    """Decode a ``TJ`` array, turning large negative kerns into spaces."""
    inner = token[1:-1] if token.startswith(b"[") else token
    pieces: list[str] = []
    for match in _TOKEN_RE.finditer(inner):
        item = match.group(0)
        if item[:1] in b"(<":
            pieces.append(_decode_pdf_string(item, font))
        elif item[:1].isdigit() or item[:1] in b"-.":
            # A kern below -150 thousandths of an em usually means a space.
            if _as_float(item) < -150 and pieces and not pieces[-1].endswith(" "):
                pieces.append(" ")
    return "".join(pieces)


# --------------------------------------------------------------------------- #
# Shared post-processing
# --------------------------------------------------------------------------- #

_META_KEYS = {
    b"/Title": "title",
    b"/Author": "author",
    b"/Subject": "subject",
    b"/Producer": "creator",
    b"/CreationDate": "created_at",
    b"/ModDate": "modified_at",
}


def _builtin_metadata(objects: dict[int, bytes]) -> DocumentMetadata:
    metadata = DocumentMetadata()
    for body in objects.values():
        if b"/Producer" not in body and b"/Title" not in body and b"/Author" not in body:
            continue
        if b"/Type" in body and b"/Page" in body:
            continue
        for key, attribute in _META_KEYS.items():
            match = re.search(
                re.escape(key) + rb"\s*(\((?:\\.|[^\\()])*\)|<[0-9A-Fa-f\s]*>)", body, re.DOTALL
            )
            if match and not getattr(metadata, attribute, None):
                value = _decode_pdf_string(match.group(1), None).strip()
                value = re.sub(r"^D:", "", value)
                if value:
                    setattr(metadata, attribute, value)
    return metadata


def _finalise(document: ExtractedDocument, raw_pages: list[list[str]]) -> None:
    """Clean lines, drop running headers/footers, and build typed blocks."""
    from kip.core.extract.plain import _blocks_from_text  # local import: same layer

    cleaned = [strip_page_furniture(lines) for lines in raw_pages]
    furniture = collapse_repeated_lines(cleaned)

    pages: list[Page] = []
    empty = 0
    for index, lines in enumerate(cleaned, start=1):
        kept = [line for line in lines if line.strip() and line.strip() not in furniture]
        page = Page(number=index)
        if kept:
            text = clean_text("\n".join(kept))
            for block in _blocks_from_text(text, is_markdown=False):
                block.page = index
                page.blocks.append(block)
        if page.is_empty:
            empty += 1
        pages.append(page)

    document.pages = pages
    if furniture:
        document.metadata.extra["removed_running_lines"] = str(len(furniture))
    if empty and empty < len(pages):
        document.add_warning(
            f"{empty} of {len(pages)} pages contained no extractable text "
            "(they may be images or blank)."
        )
    if not document.metadata.title:
        for block in document.iter_blocks():
            if block.is_heading and 3 <= len(block.text) <= 160:
                document.metadata.title = block.text
                break
    document.renumber()


def _legibility(text: str) -> float:
    """Fraction of characters that are plausible prose characters."""
    sample = text[:8000]
    if not sample:
        return 1.0
    good = sum(1 for ch in sample if ch.isalnum() or ch.isspace() or ch in ".,;:%()-/'\"")
    return good / len(sample)


def looks_like_scanned(document: ExtractedDocument) -> bool:
    """True when a PDF yielded almost no text (i.e. it needs OCR)."""
    if document.page_count == 0:
        return True
    return document.char_count / max(1, document.page_count) < 60


def available_backends() -> list[str]:
    """Report which PDF backends this deployment can use."""
    backends = ["builtin"]
    try:  # pragma: no cover
        import pypdf  # type: ignore  # noqa: F401

        backends.insert(0, "pypdf")
    except Exception:
        pass
    return backends


def iter_page_texts(document: ExtractedDocument) -> Iterable[tuple[int, str]]:
    for page in document.pages:
        yield page.number, page.text
