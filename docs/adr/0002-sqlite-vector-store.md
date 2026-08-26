# ADR-002: SQLite as Default Vector Store

## Status
Accepted

## Context
The platform needs a persistent vector store that works out of the box without requiring a separate database server. The default configuration should allow users to ingest documents, restart the application, and still search their data.

## Decision
Use **SQLite with a custom vector store implementation** as the default `VECTOR_STORE=sqlite`.

The implementation (`kip.core.vectorstore.sqlite_store.SqliteVectorStore`):
- Stores vectors as raw little-endian float32 BLOBs
- Promotes `document_id` and `user_id` to indexed columns for filter push-down
- Uses WAL journaling for concurrent read/write
- Caches the full vector matrix in memory, invalidated by a generation counter on writes
- Records the embedding model fingerprint (`provider:model:dim`) in a metadata table
- Refuses to search if the active embedding model differs from the stored fingerprint

## Consequences

### Positive
- Zero infrastructure: single file, no server process
- Exact brute-force search (cosine = dot product on normalized vectors)
- Filter push-down for tenant isolation and document selection
- Fingerprint checking prevents silent model-mismatch bugs
- Comfortable performance up to ~100k chunks on modern hardware

### Negative
- O(N) search complexity; not suitable for large corpora
- Single-writer; concurrent writes from multiple processes require coordination
- Memory usage grows with corpus size (cached matrix)
- No built-in replication or HA

## Scaling Guidance
| Corpus Size | Recommended Backend |
|-------------|---------------------|
| < 10k chunks | SQLite (default) |
| 10k - 100k | SQLite (still fine) |
| > 100k | Qdrant (`VECTOR_STORE=qdrant`) |

## Implementation Details
- Vectors are L2-normalized on insert and query
- Cosine similarity = dot product (since vectors are normalized)
- `argpartition` + tie-breaking for deterministic top-K
- Generation counter ensures cache consistency without locks

## Alternatives Considered
1. **FAISS**: Excellent performance but adds C++ dependency, not pure Python
2. **ChromaDB**: Requires separate server or heavy client
3. **LanceDB**: Promising but less mature
4. **pgvector**: Requires PostgreSQL, not zero-setup

## Related
- ADR-001: Zero-Dependency Core
- ADR-004: Hybrid Retrieval with RRF
