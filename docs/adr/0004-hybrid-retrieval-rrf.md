# ADR-004: Hybrid Retrieval with Reciprocal Rank Fusion

## Status
Accepted

## Context
Semantic (dense) retrieval and keyword (sparse) retrieval have complementary strengths. Dense retrieval excels at conceptual similarity but misses exact terminology. Keyword retrieval excels at exact matches but misses semantic equivalence. A production RAG system should combine both.

## Decision
Use **hybrid retrieval** as the default (`RETRIEVAL_MODE=hybrid`) with **Reciprocal Rank Fusion (RRF)** as the default fusion method (`RETRIEVAL_FUSION=rrf`).

### Pipeline
```
query
  |-- embed  --> vector store  --> dense hits    (cosine, [-1, 1])
  |-- stem   --> keyword index --> keyword hits  (BM25, unbounded)
                             |
                             v
                          fuse (RRF)
                             |
                             v
                      fused candidates  ->  rerank  ->  context  ->  LLM
```

### RRF Formula
```
score(doc) = Σ 1 / (k + rank_i(doc))
```
where `k = 60` (default, `RETRIEVAL_RRF_K`) and the sum is over retrievers that returned the document.

### Retrieval Modes
- `hybrid` (default): Run both axes, fuse with RRF
- `dense`: Semantic only
- `keyword`: Keyword only

### Fusion Methods
- `rrf` (default): Reciprocal Rank Fusion, parameter `k`
- `weighted`: Linear combination of min-max normalized scores

## Consequences

### Positive
- **Better recall**: Queries with exact terminology (model numbers, chemical formulas) are caught by keyword axis
- **Robustness**: If one axis fails (e.g., embedding model mismatch), the other still works
- **Explainability**: Each fused hit records which retrievers found it (`retrievers` field)
- **No score calibration needed**: RRF operates on ranks, not raw scores
- **Deterministic**: Tie-breaking by document ID ensures reproducible rankings

### Negative
- **Two indexes to maintain**: Vector store + keyword index
- **Keyword index rebuild**: BM25 index rebuilt at startup; FTS5 is persistent
- **Parameter tuning**: `k`, `dense_top_k`, `keyword_top_k` affect results
- **Latency**: Two retrieval calls per query (mitigated by parallel execution in future)

## Keyword Index Backends
- `fts5` (default): SQLite FTS5, persistent, shared across workers
- `bm25`: In-memory Okapi BM25, exact, reference for evaluation
- `none`: Disable keyword axis (reduces to dense-only)

**Note**: FTS5 uses SQLite's textbook BM25 (IDF reaches zero for terms in >50% of docs). BM25 backend uses Lucene's non-negative IDF. Rankings differ on small corpora; both are correct.

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `RETRIEVAL_MODE` | `hybrid` | `hybrid`, `dense`, `keyword` |
| `RETRIEVAL_FUSION` | `rrf` | `rrf`, `weighted` |
| `RETRIEVAL_DENSE_TOP_K` | 24 | Dense candidates per query |
| `RETRIEVAL_KEYWORD_TOP_K` | 24 | Keyword candidates per query |
| `RETRIEVAL_RRF_K` | 60 | RRF damping parameter |
| `RETRIEVAL_CANDIDATE_LIMIT` | 40 | Max fused candidates before rerank |

## Alternatives Considered
1. **Weighted score fusion**: Requires score normalization; sensitive to score distribution shifts
2. **Concat embeddings**: Early fusion loses the complementary nature of the two signals
3. **Cascade (dense then keyword filter)**: Loses keyword-only hits
4. **Learned fusion**: Requires training data; overkill for general-purpose retrieval

## Related
- ADR-002: SQLite as Default Vector Store
- ADR-006: Reranking
