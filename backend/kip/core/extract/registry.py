"""Format dispatch for document extraction.

The registry is the *only* place that maps a file to an extractor. Services call
:func:`extract_document` and receive an :class:`ExtractedDocument`; they never
import a format-specific module. Supporting a new format therefore means adding
one entry here plus one extractor module -- chunking, embedding, retrieval and
citation are untouched.

Dispatch uses the extension *and* the leading bytes. A ``.pdf`` that is really a
ZIP, or a ``.docx`` that is really plain text, is routed by content rather than
by its (possibly wrong, possibly hostile) name.
"""

from __future__ import annotations

from typing import Callable, Iterable

from kip.core.extract.base import ExtractedDocument
from kip.core.extract.docx import extract_docx
from kip.core.extract.pdf import extract_pdf
from kip.core.extract.plain import extract_plain
from kip.errors import ExtractionError, UnsupportedMediaTypeError
from kip.security.files import extension_of, sniff_kind

#: Extensions the platform accepts, in the order shown to users.
SUPPORTED_EXTENSIONS: tuple[str, ...] = ("pdf", "docx", "md", "txt")

#: Human labels for the UI.
FORMAT_LABELS: dict[str, str] = {
    "pdf": "PDF document",
    "docx": "Word document",
    "md": "Markdown",
    "txt": "Plain text",
}


def supported_extensions() -> tuple[str, ...]:
    return SUPPORTED_EXTENSIONS


def is_supported(filename: str) -> bool:
    return extension_of(filename) in SUPPORTED_EXTENSIONS


def describe(extension: str) -> str:
    return FORMAT_LABELS.get(extension.lower().lstrip("."), extension.upper())


def extract_document(
    filename: str,
    payload: bytes,
    *,
    allowed: Iterable[str] | None = None,
) -> ExtractedDocument:
    """Extract ``payload`` using the extractor implied by name and content.

    Raises :class:`UnsupportedMediaTypeError` when the format is not accepted and
    :class:`ExtractionError` when the file is accepted but unreadable.
    """
    permitted = {e.lower().lstrip(".") for e in (allowed or SUPPORTED_EXTENSIONS)}
    extension = extension_of(filename)
    if extension not in permitted:
        raise UnsupportedMediaTypeError(
            f"'{extension or 'unknown'}' files are not supported. "
            f"Accepted formats: {', '.join(sorted(permitted))}.",
            details={"extension": extension, "supported": sorted(permitted)},
        )
    if not payload:
        raise ExtractionError("The file is empty, so there is nothing to index.")

    kind = sniff_kind(payload[:4096])
    extractor = _resolve(extension, kind)
    document = extractor(payload)

    if not document.has_text:
        document.add_warning(
            "No text could be extracted. If this is a scanned document it needs "
            "OCR before it can be indexed."
        )
    return document


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

Extractor = Callable[[bytes], ExtractedDocument]


def _resolve(extension: str, kind: str) -> Extractor:
    """Pick an extractor from the declared extension and the sniffed content."""
    # Content wins when the two disagree in a way we can safely honour.
    if kind == "pdf":
        return extract_pdf
    if kind == "zip":
        # Only DOCX is an accepted ZIP-based format; anything else is refused by
        # the upload validator long before this point.
        return extract_docx

    if extension == "pdf":
        raise ExtractionError(
            "This file is named .pdf but does not start with a PDF header, so it "
            "cannot be read as a PDF."
        )
    if extension == "docx":
        raise ExtractionError(
            "This file is named .docx but is not a valid Word package "
            "(a .docx must be a ZIP archive containing word/document.xml)."
        )
    if extension == "md":
        return lambda data: extract_plain(data, is_markdown=True)
    if extension == "txt":
        return extract_plain

    raise UnsupportedMediaTypeError(f"No extractor is registered for '{extension}'.")
