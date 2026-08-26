# ADR-001: Zero-Dependency Core

## Status
Accepted

## Context
The Knowledge Intelligence Platform (KIP) needs to be runnable in environments with restricted internet access, no GPU, and minimal system dependencies. Reviewers and users should be able to clone the repository and run `docker compose up` or `pip install -e .` followed by `uvicorn kip.api:app` without configuring API keys, downloading models, or starting external services.

## Decision
The core RAG engine (`kip.core`) will have **zero third-party dependencies** beyond the Python standard library and NumPy.

Specifically:
- No web framework (FastAPI, Starlette)
- No database driver (asyncpg, aiosqlite)
- No vector database client (qdrant-client)
- No ML framework (torch, sentence-transformers)
- No HTTP client (httpx, requests) in the core

Optional providers (sentence-transformers, Qdrant, OpenAI, Anthropic, etc.) are imported **lazily** inside their respective adapter modules, only when selected by configuration.

## Consequences

### Positive
- Core is importable and testable in any Python 3.10+ environment with NumPy
- Self-checks (`python -m selfcheck`) run with zero setup
- Clear separation: core = algorithms, api/services = infrastructure
- Easy to audit: all external calls are explicit in adapter modules
- CI/CD runs fast; no model downloads in test pipeline

### Negative
- Some code duplication (e.g., configuration parsing without pydantic-settings)
- Cannot use convenient libraries in core algorithms
- Adapters must handle import errors gracefully with actionable messages

## Implementation Notes
- `kip.config` implements its own `.env` parser and settings validation
- `kip.core.embeddings.providers` imports `sentence_transformers` only in `SentenceTransformerEmbedder.__init__`
- `kip.core.vectorstore.qdrant` imports `qdrant_client` only in `QdrantVectorStore.__init__`
- `kip.core.llm.providers` uses `kip.core.http` (stdlib `urllib`) for all HTTP calls

## Alternatives Considered
1. **Use pydantic-settings everywhere**: Would require pydantic as core dependency
2. **Vendor minimal dependencies**: Increases maintenance burden
3. **Make core fully pure-Python (no NumPy)**: Would sacrifice performance and clarity of vector operations

## Related
- ADR-002: SQLite as Default Vector Store
- ADR-003: Extractive LLM as Default
