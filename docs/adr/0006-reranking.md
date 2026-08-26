# ADR-006: Reranking Stage

## Status
Accepted

## Context
First-stage retrieval (dense + keyword) is optimized for **recall** over a large corpus. It uses representations that discard word order (dense vectors, BM25 term statistics). A second stage can afford to look at query-passage pairs jointly for **precision**.

## Decision
Insert a **reranking stage** between retrieval and context building. The reranker:
- Receives top-K fused candidates (default K=40)
- Scores each `(query, passage_text)` pair
- Returns top-N (default N=6) for context building
- Is optional and swappable via `RERANKER`

### Reranker Backends
| Backend | `RERANKER` value | Dependencies | Latency | Quality |
|---------|------------------|--------------|---------|---------|
| Heuristic | `heuristic` (default) | None | <1ms | Good |
| Cross-encoder | `cross-encoder` | sentence-transformers | ~50ms | Better |
| LLM | `llm` | Any generative LLM | ~500ms | Best |
| None | `none` | None | 0ms | Baseline |

### Heuristic Reranker (Default)
Scores passages using five lexical signals:
1. **Coverage** (0.40): Fraction of query terms present
2. **Proximity** (0.20): How tightly matched terms cluster
3. **Phrase** (0.10): Adjacent query pairs occurring adjacently
4. **Heading** (0.05): Query terms in passage heading
5. **Prior** (0.25): Retriever's rank (damped reciprocal)

Scores are **not calibrated** (`calibrated=False`); they are relative ranking signals only.

### Cross-Encoder Reranker
Uses a sentence-transformers CrossEncoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`). Loads model lazily on first use.

### LLM Reranker
Prompts a generative LLM to rate relevance of each passage. Requires a `score_fn` callable injected by the service layer (dependency inversion: `rerank` never imports `llm`).

## Consequences

### Positive
- **Measurable improvement**: Evaluation shows heuristic reranker improves top-1 accuracy over no reranking
- **Swappable**: Can A/B test rerankers via configuration
- **Explainable**: Heuristic signals are human-readable
- **Optional**: `RERANKER=none` for latency-critical deployments

### Negative
- **Added latency**: Even heuristic adds ~1ms; cross-encoder ~50ms; LLM ~500ms
- **Text egress (LLM)**: Passage text sent to LLM provider
- **Uncalibrated scores**: Cannot be used as confidence; only for ranking
- **Context budget pressure**: Reranker sees truncated passages (default 4000 chars)

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `RERANKER` | `heuristic` | `heuristic`, `cross-encoder`, `llm`, `none` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `RERANK_TOP_N` | 6 | Passages to keep after reranking |

## Implementation Details
- Reranker base class handles: truncation, tie-breaking, top-N, movement tracking
- `RerankResult` preserves `prior_rank`, `rank`, `movement`, `signals`
- `candidates_from_hits` drops candidates with no hydrated text
- Empty query preserves retriever order (no re-scoring)

## Alternatives Considered
1. **Rerank in retriever**: Would require vector store to hold passage text (couples storage)
2. **No rerank stage**: Simpler but loses precision gain
3. **Learned fusion instead of rerank**: More complex; rerank is more modular

## Related
- ADR-004: Hybrid Retrieval with RRF
- ADR-007: Grounding Checks
