# ADR-007: Grounding Checks

## Status
Accepted

## Context
A RAG system that always answers has not solved hallucination; it has moved it somewhere harder to see. The platform needs mechanisms to refuse answering when the evidence is insufficient, and to verify that generated answers are actually supported by the retrieved passages.

## Decision
Implement **two grounding checks** at different pipeline stages:

### 1. Pre-Generation Evidence Gate (`check_evidence`)
Runs **before** calling the LLM. If retrieval found nothing sufficiently relevant, the model is never called.

- Reads **raw cosine similarity** from dense retrieval (not fused score)
- Fused scores always have a "top" value; cosine similarity does not
- Three refusal reasons:
  - `no_passages`: Zero candidates retrieved
  - `too_few_passages`: Below `GROUNDING_MIN_PASSAGES` (default 1)
  - `weak_match`: Top cosine < `GROUNDING_MIN_SCORE` (default 0.16)
- Keyword-only mode: Cosine unavailable → skips score threshold, logs `score_unavailable=true`

### 2. Post-Generation Support Check (`check_support`)
Runs **after** generation. Scores each answer sentence against the passages it cites.

- Lexical proxy: stemmed term containment (sentence terms ⊆ passage terms)
- Threshold: `GROUNDING_SUPPORT_THRESHOLD` (default 0.32)
- **Enforces citations** by default (`GROUNDING_ENFORCE_CITATIONS=true`):
  - Uncited sentence = unsupported, even if lexically matching
- Three outcomes per sentence:
  - `supported`: Cited passage meets threshold
  - `unsupported`: Below threshold or no citation
  - `misattributed`: Different passage supports it much better (≥0.25 margin)
- **Groundedness** = fraction of claim-sentences supported
- **Refusal trigger** (`advises_refusal`): *No* claim traces to *any* passage

### Refusal Handling
Seven distinct refusal causes, all producing the **same user-facing sentence**:
> "The available documents do not contain enough information to answer this question."

Causes:
1. `no_passages` (evidence gate)
2. `weak_match` (evidence gate)
3. `too_few_passages` (evidence gate)
4. `no_passage_text` (hydration failed)
5. `empty_context` (context budget too small)
6. `model_refused` (LLM said insufficient evidence)
7. `unsupported_answer` (post-generation suppression)

**Key principle**: User cannot infer internal cause from refusal wording. Evaluation dashboard counts refusals by cause.

## Consequences

### Positive
- **Cost savings**: Evidence gate avoids LLM calls for unanswerable questions
- **Safety net**: Post-generation check catches hallucinations evidence gate missed
- **Auditability**: Every refusal has a machine-readable cause
- **No silent hallucination**: Unsupported generations are suppressed, not shown
- **Configurable thresholds**: Operators can tune conservatism

### Negative
- **False negatives**: Conservative thresholds may refuse answerable questions
- **Lexical proxy limitations**: Faithful paraphrase can score low; fluent fabrication can score high
- **Two-stage complexity**: More code paths to test
- **Threshold tuning**: Requires evaluation data to set well

## Threshold Selection
Defaults chosen by measurement on evaluation set with hashing embeddings:
- `GROUNDING_MIN_SCORE=0.16`: Above noise floor (unrelated passages ~0.03)
- `GROUNDING_SUPPORT_THRESHOLD=0.32`: Balances false positive/negative on eval set
- `GROUNDING_MIN_PASSAGES=1`: At least one passage required

See `docs/EVALUATION.md` for sweep results.

## Implementation Details
- `EvidenceCheck` dataclass: `sufficient`, `reason`, `top_score`, `passages`, `explanation`
- `GroundingReport`: per-sentence `SentenceSupport` with `supported`, `score`, `misattributed`
- `advises_refusal` only true when **all** claims have `best_score == 0`
- `DISCOURSE_TERMS` excluded from groundedness denominator (e.g., "however", "therefore")

## Alternatives Considered
1. **Single post-generation check**: Misses cost savings of pre-generation gate
2. **Entailment model (NLI)**: Adds dependency; still not perfect
3. **No refusal (always answer)**: Moves hallucination to harder-to-detect place
4. **Hard threshold on support score**: Would suppress correct paraphrases

## Related
- ADR-003: Extractive LLM as Default
- ADR-005: Citation Integrity Design
- ADR-006: Reranking Stage
