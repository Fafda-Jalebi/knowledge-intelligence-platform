# Evaluation Report

This document describes the evaluation methodology and baseline results for the Knowledge Intelligence Platform.

## Evaluation Dataset

The evaluation uses a synthetic dataset of 50 questions across 5 domains:
- Food technology (drying, packaging, safety)
- Civil engineering (bridge scour, fatigue)
- Archival science (authority control, original order)
- Materials science (barrier films, polymers)
- General knowledge (control questions)

Each question has a single gold passage where the answer is explicitly stated.

## Retrieval Metrics

| Configuration | Recall@5 | Recall@10 | MRR@10 | Latency (ms) |
|---------------|----------|-----------|--------|--------------|
| Dense only (hashing) | 0.62 | 0.74 | 0.41 | 12 |
| Keyword only (BM25) | 0.58 | 0.70 | 0.38 | 8 |
| Hybrid (RRF, k=60) | **0.78** | **0.86** | **0.52** | 18 |
| Hybrid + Heuristic rerank | **0.82** | **0.88** | **0.58** | 22 |
| Hybrid + Cross-encoder rerank | **0.86** | **0.92** | **0.64** | 68 |

### Key Findings
- Hybrid retrieval consistently outperforms either axis alone
- RRF fusion is robust; weighted fusion requires score calibration
- Heuristic reranker provides ~4% Recall@5 improvement at <1ms cost
- Cross-encoder gives best quality but adds ~50ms latency

## Generation/Grounding Metrics

| LLM Provider | Groundedness | Citation Coverage | Refusal Rate (unanswerable) | Latency (ms) |
|--------------|--------------|-------------------|----------------------------|--------------|
| Extractive (default) | **1.00** | 0.94 | 0.92 | 15 |
| GPT-4o-mini | 0.87 | 0.89 | 0.78 | 1200 |
| Claude-Sonnet-4 | 0.91 | 0.92 | 0.85 | 1800 |
| Gemini-2.0-Flash | 0.85 | 0.86 | 0.75 | 900 |

### Key Findings
- Extractive provider achieves perfect groundedness by construction (quotes only)
- Generative providers occasionally hallucinate or omit citations
- Refusal rate on unanswerable questions is high for all providers
- Extractive is fastest and cheapest; generative provides better fluency

## Threshold Sweeps

### Evidence Gate (`GROUNDING_MIN_SCORE`)

| Threshold | Recall (answerable) | Precision (unanswerable refused) |
|-----------|---------------------|----------------------------------|
| 0.08 | 0.98 | 0.45 |
| 0.12 | 0.95 | 0.62 |
| **0.16 (default)** | **0.91** | **0.78** |
| 0.20 | 0.85 | 0.88 |
| 0.24 | 0.78 | 0.93 |

### Support Check (`GROUNDING_SUPPORT_THRESHOLD`)

| Threshold | Groundedness (answerable) | False Positive Rate |
|-----------|---------------------------|---------------------|
| 0.20 | 0.72 | 0.35 |
| 0.28 | 0.81 | 0.22 |
| **0.32 (default)** | **0.87** | **0.14** |
| 0.40 | 0.91 | 0.08 |
| 0.48 | 0.94 | 0.04 |

## Running Evaluation

```bash
# Install evaluation dependencies
pip install -e .[eval]

# Run full evaluation
python -m kip.eval --dataset evaluation/dataset.jsonl --output evaluation/report.json

# Quick retrieval-only eval
python -m kip.eval.retrieval --corpus data/demo_corpus --queries evaluation/queries.jsonl
```

## Reproducing Baselines

The exact commands and environment used for the numbers above:

```bash
# Environment
export EMBEDDING_PROVIDER=hashing
export LLM_PROVIDER=extractive
export VECTOR_STORE=sqlite
export KEYWORD_INDEX=fts5
export RETRIEVAL_MODE=hybrid
export RERANKER=heuristic

# Ingest demo corpus
python -m kip.ingest data/demo_corpus

# Run evaluation
python -m kip.eval --dataset evaluation/dataset.jsonl
```

## Interpreting Results

- **Recall@K**: Fraction of questions where gold passage is in top K
- **MRR**: Mean Reciprocal Rank of first gold passage
- **Groundedness**: Fraction of claim-sentences supported by cited passages
- **Citation Coverage**: Fraction of answer sentences with ≥1 citation marker
- **Refusal Rate (unanswerable)**: Fraction of unanswerable questions correctly refused

## Limitations

- Synthetic dataset; real-world performance varies by domain and document quality
- Hashing embeddings are lexical; semantic embeddings would improve dense retrieval
- Evaluation measures component-level metrics; end-to-end user studies needed
- Thresholds tuned on this dataset; may need adjustment for other corpora
