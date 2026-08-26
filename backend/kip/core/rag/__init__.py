"""Retrieval-augmented generation: context assembly, prompting and verification.

This package is the answer path. Everything below it -- extraction, chunking,
embeddings, vector stores, retrieval, reranking -- exists to put candidate
passages in front of :class:`~kip.core.rag.pipeline.RagPipeline`, and everything
above it is transport.

The design commitment, in one line: **an answer is only produced when it can be
traced back to a passage, and the trace is returned with it.** The two mechanisms
that make that more than a slogan are the pre-generation evidence gate and the
post-generation citation and support checks, both in
:mod:`kip.core.rag.grounding`.

Module map:

* :mod:`kip.core.rag.context` -- token-budgeted, numbered context blocks
* :mod:`kip.core.rag.prompts` -- every instruction the model receives
* :mod:`kip.core.rag.citations` -- ``[n]`` markers resolved to real chunks
* :mod:`kip.core.rag.grounding` -- the evidence gate and the support check
* :mod:`kip.core.rag.pipeline` -- the orchestration and its timings

Nothing here imports a database, an HTTP framework or a provider SDK. Text
hydration arrives as a callable and the LLM as an injected client, which is what
lets the evaluation harness run the identical pipeline over a fixture corpus and
produce numbers comparable to the running application's.
"""

from __future__ import annotations

from kip.core.rag.citations import (
    Citation,
    CitationReport,
    extract_markers,
    resolve,
    split_cited_sentences,
    strip_markers,
)
from kip.core.rag.context import ContextBlock, ContextBuilder, ContextPassage
from kip.core.rag.grounding import (
    EvidenceCheck,
    GroundingReport,
    SentenceSupport,
    check_evidence,
    check_support,
    describe_thresholds,
)
from kip.core.rag.pipeline import (
    REFUSAL_EMPTY_CONTEXT,
    REFUSAL_MODEL,
    REFUSAL_NO_TEXT,
    REFUSAL_UNSUPPORTED,
    Answer,
    RagPipeline,
    Stage,
)
from kip.core.rag.prompts import (
    INSUFFICIENT_EVIDENCE,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_messages,
    is_insufficient,
)

__all__ = [
    "Answer",
    "Citation",
    "CitationReport",
    "ContextBlock",
    "ContextBuilder",
    "ContextPassage",
    "EvidenceCheck",
    "GroundingReport",
    "INSUFFICIENT_EVIDENCE",
    "PROMPT_VERSION",
    "REFUSAL_EMPTY_CONTEXT",
    "REFUSAL_MODEL",
    "REFUSAL_NO_TEXT",
    "REFUSAL_UNSUPPORTED",
    "RagPipeline",
    "SYSTEM_PROMPT",
    "SentenceSupport",
    "Stage",
    "build_messages",
    "check_evidence",
    "check_support",
    "describe_thresholds",
    "extract_markers",
    "is_insufficient",
    "resolve",
    "split_cited_sentences",
    "strip_markers",
]
