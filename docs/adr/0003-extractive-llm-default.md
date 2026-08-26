# ADR-003: Extractive LLM as Default

## Status
Accepted

## Context
The platform needs a default LLM provider that works without API keys, model downloads, or GPU. The default should produce grounded answers with citations so that the platform is immediately useful for evaluation and demonstration.

## Decision
Use an **extractive answerer** (`kip.core.llm.extractive.ExtractiveClient`) as the default `LLM_PROVIDER=extractive`.

The extractive client:
- Does not generate text; it selects and quotes sentences from retrieved passages
- Scores sentences by stemmed term overlap with the question + rank bonus
- Attaches citation markers `[n]` to each quoted sentence
- Returns `finish_reason="insufficient_evidence"` when no sentence meets the minimum overlap threshold
- Reports usage as estimated (not metered)

## Consequences

### Positive
- **Zero setup**: No API key, no model download, no network, no GPU
- **Grounded by construction**: Every output token comes from the retrieved passages
- **Deterministic**: Same question + same corpus = same answer (enables regression testing)
- **Faithfulness control**: Evaluation harness uses it as the 1.0 groundedness ceiling
- **Fast**: Sub-millisecond on CPU
- **Privacy**: Document text never leaves the machine

### Negative
- **Cannot paraphrase or synthesize**: Answers are verbatim quotes only
- **Cannot resolve contradictions**: If passages disagree, both may be quoted
- **Cannot answer implied questions**: If the answer isn't explicitly stated, it refuses
- **Verbose**: Quotes full sentences rather than concise summaries

## When to Switch
Configure a generative provider (`openai`, `anthropic`, `gemini`, `ollama`) when you need:
- Paraphrasing and synthesis across passages
- Concise, natural-language answers
- Reasoning over implicit information
- Multilingual output

## Implementation Details
- Minimum sentence overlap threshold: 0.30 (configurable via `MIN_SENTENCE_OVERLAP`)
- Duplicate detection via Jaccard ≥ 0.80 (`DUPLICATE_JACCARD`)
- Rank bonus: `0.06 / rank` (`RANK_BONUS`)
- Max sentences per answer: 6 (`DEFAULT_MAX_SENTENCES`)
- Citation format: `Sentence text [n].` (marker before period)

## Alternatives Considered
1. **Small local model (e.g., TinyLlama)**: Still requires model download and GPU/CPU inference time
2. **Template-based responses**: Too rigid, poor coverage
3. **No default (require configuration)**: Creates "configuration error" as first experience

## Related
- ADR-001: Zero-Dependency Core
- ADR-005: Citation Integrity Design
