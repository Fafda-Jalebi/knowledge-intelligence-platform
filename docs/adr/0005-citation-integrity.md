# ADR-005: Citation Integrity Design

## Status
Accepted

## Context
A RAG system's credibility depends on whether users can verify that answers actually come from the cited sources. Many systems produce citations that are decorative: the marker numbers don't correspond to actual passages, or the quoted text differs from what the model saw.

## Decision
Enforce **citation integrity** at three levels:

### 1. Marker Resolution (Pre-generation)
- Context builder assigns **contiguous 1-based markers** to passages
- Markers are never renumbered or compacted (gaps preserved)
- Passages carry: chunk ID, document ID, page range, section path, score

### 2. Marker Validation (Post-generation)
- Every `[n]` in the answer must resolve to a passage that was in the context
- Invented markers are **stripped from the answer text** and recorded in `invalid_markers`
- Answer sentences with no markers are recorded in `uncited_sentences`
- Citation coverage = fraction of sentences with ≥1 valid marker

### 3. Source Viewer Fidelity
- Citation object carries the **exact passage text** the model was shown
- Truncation flag indicates if passage was shortened for context budget
- Frontend Source Viewer displays this text verbatim (no re-fetch)

## Consequences

### Positive
- **Verifiable**: Users can check every citation against the source text
- **Honest**: Invented citations are visible in the UI and evaluation metrics
- **No silent drift**: If a document is re-ingested, old citations still show original text
- **Evaluation-grade**: Citation coverage and invalid marker count are first-class metrics

### Negative
- **Gaps in numbering**: `[1][3]` instead of `[1][2]` may look odd to users
- **Context budget limits**: Long passages may be truncated; truncation flag signals this
- **Model compliance needed**: Prompt must instruct models to use `[n]` format

## Implementation Details

### Marker Pattern
```regex
\[\s*(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\s*\]
```
- Matches `[1]`, `[1][2]`, `[1, 3]`, `[1; 3]`
- Bounded to 3 digits to avoid matching years like `[1998]`

### Leading Marker Reattachment
Models often write: `"Claim. [1] Next claim. [2]"`
Sentence splitter would put `[1]` at start of next fragment.
Reattacher moves leading markers to previous sentence before period.

### Invalid Marker Handling
```python
# Strip invalid, keep text
"Fake claim [99]." -> "Fake claim."
# Record for evaluation
invalid_markers = (99,)
```

### Citation Object
```python
@dataclass
class Citation:
    marker: int
    id: str              # chunk_id
    document_id: str
    chunk_index: int
    document_label: str
    label: str           # "Doc Title, p. 5"
    text: str            # verbatim passage from context
    page_start: int | None
    page_end: int | None
    section_path: tuple[str, ...]
    score: float
    truncated: bool
    count: int           # how many times marker appears
```

## Alternatives Considered
1. **Renumber markers**: Cleaner UI but breaks source viewer linkage
2. **Silently drop invalid**: Hides hallucination; evaluation can't detect it
3. **Re-fetch on display**: Source text may have changed; breaks verifiability
4. **No markers (footnotes only)**: Loses sentence-level traceability

## Related
- ADR-003: Extractive LLM as Default
- ADR-007: Grounding Checks
