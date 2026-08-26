"""RAG engine.

This package is intentionally free of third-party dependencies apart from
NumPy. It can be imported, unit-tested and benchmarked without installing a
web framework, a database driver, a vector database client or a model runtime.
Optional providers (sentence-transformers, Qdrant, OpenAI, ...) are imported
lazily inside their own modules and only when selected by configuration.

Rationale: ``docs/adr/0001-zero-dependency-core.md``.
"""

from kip.core.chunking import chunk_document
from kip.core.extract import extract_document
from kip.core.rag import RagPipeline

__all__ = ["chunk_document", "extract_document", "RagPipeline"]
