"""Public extraction API.

Import from here rather than from the format-specific modules::

    from kip.core.extract import extract_document, ExtractedDocument
"""

from kip.core.extract.base import (
    Block,
    BlockKind,
    DocumentMetadata,
    ExtractedDocument,
    Page,
    build_page,
)
from kip.core.extract.docx import extract_docx
from kip.core.extract.pdf import (
    available_backends,
    extract_pdf,
    iter_page_texts,
    looks_like_scanned,
)
from kip.core.extract.plain import extract_plain
from kip.core.extract.registry import (
    FORMAT_LABELS,
    SUPPORTED_EXTENSIONS,
    describe,
    extract_document,
    is_supported,
    supported_extensions,
)

__all__ = [
    "Block",
    "BlockKind",
    "DocumentMetadata",
    "ExtractedDocument",
    "Page",
    "build_page",
    "extract_docx",
    "extract_pdf",
    "extract_plain",
    "extract_document",
    "available_backends",
    "iter_page_texts",
    "looks_like_scanned",
    "describe",
    "is_supported",
    "supported_extensions",
    "FORMAT_LABELS",
    "SUPPORTED_EXTENSIONS",
]
