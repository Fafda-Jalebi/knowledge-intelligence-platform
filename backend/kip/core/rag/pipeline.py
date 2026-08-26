"""The RAG pipeline -- one question in, one grounded and cited answer out.

The stages, and what each one is responsible for:

1. **Retrieve** (:mod:`kip.core.retrieval.hybrid`) -- dense and keyword axes,
   fused. Recall-oriented: it is cheaper to over-fetch here than to miss the only
   passage that held the answer.
2. **Gate** (:func:`kip.core.rag.grounding.check_evidence`) -- if nothing similar
   enough came back, refuse *before* calling the model. No generation happens on
   an empty evidence base, which is the cheapest possible hallucination defence
   and the only one that is airtight.
3. **Hydrate** -- passage text is fetched by an injected callable. This package
   never imports a database; see the note on layering below.
4. **Rerank** (:mod:`kip.core.rerank`) -- precision-oriented reordering of the
   candidates that survived.
5. **Build context** (:mod:`kip.core.rag.context`) -- token-budgeted, numbered,
   capped per document. Whole documents never reach the model.
6. **Generate** (:mod:`kip.core.llm`) -- the only stage that may call out to a
   third party, and the only stage whose output cannot be trusted on its own.
7. **Verify** (:mod:`kip.core.rag.citations`, :mod:`kip.core.rag.grounding`) --
   markers resolved to real chunks, invented markers stripped and counted, every
   claim scored against the passage it cites.

**Layering.** Hydration and document titles arrive as callables rather than a
repository object, so ``kip.core`` stays free of any persistence import and the
whole pipeline is testable with a dictionary. It also means the evaluation harness
can run the identical pipeline over a fixture corpus with no database at all,
which is what makes its numbers comparable to the ones the running application
produces.

**Refusal is an outcome, not an exception.** Every path that cannot produce a
grounded answer returns an :class:`Answer` with ``refused=True`` and a machine-
readable ``refusal_reason``, so the interface can explain *why* and the evaluation
dashboard can count refusals by cause rather than lumping them together.

**What is measured is recorded.** Every stage's duration is on the answer. The
performance numbers in the README come from this field, on a stated machine, over
a stated corpus -- not from an estimate.

**Async hydration support.** The ``hydrate`` callable passed to :class:`RagPipeline`
may be either a regular function or an async function. If it returns a coroutine,
it will be awaited automatically. This allows the hydration step to perform async
I/O (e.g. database lookups) while keeping the pipeline itself synchronous for
the common case.
"""

from __future__ import annotations

import asyncio

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from kip.core.llm.base import LlmClient, LlmResponse, Message, Usage
from kip.core.rag import prompts
from kip.core.rag.citations import Citation, CitationReport, resolve
from kip.core.rag.context import ContextBlock, ContextBuilder
from kip.core.rag.grounding import (
    EvidenceCheck,
    GroundingReport,
    check_evidence,
    check_support,
)
from kip.core.rerank.base import Reranker, RerankResult, candidates_from_hits

#: Machine-readable refusal causes. ``weak_match``, ``no_passages`` and
#: ``too_few_passages`` are re-used from the evidence gate; these cover the
#: later stages.
REFUSAL_NO_TEXT = "no_passage_text"
REFUSAL_EMPTY_CONTEXT = "empty_context"
REFUSAL_MODEL = "model_refused"
REFUSAL_UNSUPPORTED = "unsupported_answer"

#: Hydration: chunk ids -> chunk text. Missing ids are allowed; the pipeline
#: treats a chunk it cannot hydrate as absent rather than as empty.
Hydrator = Callable[[Sequence[str]], Mapping[str, str]]
#: Document ids -> display title. Optional; ids are shown when it is absent.
TitleResolver = Callable[[Sequence[str]], Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class Stage:
    """One pipeline stage's cost and outcome.

    >>> Stage("retrieve", 12.345, {"candidates": 8}).to_dict()
    {'stage': 'retrieve', 'ms': 12.35, 'detail': {'candidates': 8}}
    """

    name: str
    ms: float
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.name, "ms": round(self.ms, 2), "detail": dict(self.detail)}


@dataclass(frozen=True, slots=True)
class Answer:
    """The complete result of one question, with everything needed to audit it."""

    question: str
    text: str = ""
    refused: bool = False
    refusal_reason: str = ""
    citations: tuple[Citation, ...] = ()
    context: ContextBlock = field(default_factory=ContextBlock)
    evidence: EvidenceCheck = field(default_factory=lambda: EvidenceCheck(True))
    grounding: GroundingReport = field(default_factory=GroundingReport)
    citation_report: CitationReport = field(default_factory=CitationReport)
    provider: str = ""
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    stages: tuple[Stage, ...] = ()
    prompt_version: str = prompts.PROMPT_VERSION
    retrieval: Mapping[str, Any] = field(default_factory=dict)
    #: An answer withdrawn by the verification stage, kept rather than discarded so
    #: the evaluation dashboard can count and inspect suppressed generations. Never
    #: shown as an answer.
    suppressed_text: str = ""

    @property
    def total_ms(self) -> float:
        return round(sum(stage.ms for stage in self.stages), 2)

    @property
    def document_ids(self) -> tuple[str, ...]:
        return self.citation_report.documents

    @property
    def groundedness(self) -> float:
        return self.grounding.groundedness

    @property
    def explanation(self) -> str:
        """Why this answer was refused, in plain language.

        >>> from kip.core.rag.grounding import REASON_WEAK_MATCH
        >>> gate = EvidenceCheck(False, REASON_WEAK_MATCH, top_score=0.03,
        ...                      min_score=0.16)
        >>> Answer("q", refused=True, refusal_reason=REASON_WEAK_MATCH,
        ...        evidence=gate).explanation.startswith("The closest passage")
        True
        >>> Answer("q", refused=True, refusal_reason=REFUSAL_UNSUPPORTED).explanation
        'The generated answer could not be traced to any retrieved passage, so it was withheld.'
        >>> Answer("q", text="Dried at 60 C [1].").explanation
        ''
        """
        if not self.refused:
            return ""
        if self.refusal_reason == REFUSAL_UNSUPPORTED:
            return (
                "The generated answer could not be traced to any retrieved "
                "passage, so it was withheld."
            )
        if self.refusal_reason == REFUSAL_NO_TEXT:
            return (
                "The retrieved passages could not be loaded from storage, so "
                "there was nothing to answer from."
            )
        if self.refusal_reason == REFUSAL_EMPTY_CONTEXT:
            return "No passage fitted the context window for this question."
        if self.refusal_reason == REFUSAL_MODEL:
            return "The model reported that the passages do not answer this question."
        return self.evidence.explanation or (
            "The retrieved passages are not sufficient to answer this question."
        )

    def to_dict(self, *, preview: int = 0) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "explanation": self.explanation,
            "citations": [c.to_dict(preview=preview) for c in self.citations],
            "context": self.context.to_dict(),
            "evidence": self.evidence.to_dict(),
            "grounding": self.grounding.to_dict(),
            "citation_coverage": self.citation_report.coverage,
            "invalid_markers": list(self.citation_report.invalid_markers),
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "prompt_version": self.prompt_version,
            "stages": [stage.to_dict() for stage in self.stages],
            "total_ms": self.total_ms,
            "retrieval": dict(self.retrieval),
        }


class RagPipeline:
    """
    Orchestrates retrieval, generation and verification for one question.

    The example below is the whole platform in miniature: an in-memory store, a
    hashing embedder, a BM25 index, the default extractive generator, and a
    dictionary standing in for the chunks table.

    >>> from kip.core.embeddings import HashingEmbedder
    >>> from kip.core.llm.extractive import ExtractiveClient
    >>> from kip.core.retrieval.bm25 import Bm25Index
    >>> from kip.core.retrieval.hybrid import HybridRetriever
    >>> from kip.core.retrieval.keyword import KeywordDocument
    >>> from kip.core.vectorstore import MemoryVectorStore
    >>> from kip.core.vectorstore.base import VectorRecord
    >>> passages = {
    ...     "d1:0": "Hot air drying of mango slices is carried out at 60 C for "
    ...             "eight hours.",
    ...     "d2:0": "Retort sterilisation of canned vegetables is performed at "
    ...             "121 C.",
    ...     "d3:0": "Water activity below 0.6 inhibits microbial growth.",
    ... }
    >>> embedder = HashingEmbedder(dim=1024)
    >>> store = MemoryVectorStore()
    >>> store.ensure_collection(embedder.spec)
    >>> vectors = embedder.embed_documents(list(passages.values()))
    >>> _ = store.upsert([
    ...     VectorRecord(key, vectors[i],
    ...                  {"document_id": key.split(":")[0], "chunk_index": 0})
    ...     for i, key in enumerate(passages)
    ... ])
    >>> keyword = Bm25Index()
    >>> _ = keyword.add([
    ...     KeywordDocument(key, text,
    ...                     {"document_id": key.split(":")[0], "chunk_index": 0})
    ...     for key, text in passages.items()
    ... ])
    >>> pipeline = RagPipeline(
    ...     retriever=HybridRetriever(embedder, store, keyword),
    ...     llm=ExtractiveClient(),
    ...     hydrate=lambda ids: {i: passages[i] for i in ids if i in passages},
    ...     titles=lambda ids: {"d1": "Mango Drying Study"},
    ... )

    A question the corpus answers produces a cited answer, and the citation
    resolves to the chunk the text actually came from:

    >>> answer = asyncio.run(pipeline.answer("At what temperature are mango slices dried?"))
    >>> answer.refused
    False
    >>> answer.text
    'Hot air drying of mango slices is carried out at 60 C for eight hours [1].'
    >>> [(c.marker, c.id, c.document_label) for c in answer.citations]
    [(1, 'd1:0', 'Mango Drying Study')]
    >>> answer.citations[0].text in passages.values()
    True
    >>> answer.groundedness, answer.citation_report.coverage
    (1.0, 1.0)

    A question the corpus does not answer is refused rather than answered from
    the nearest available passage:

    >>> miss = asyncio.run(pipeline.answer("What is the tensile strength of the conveyor belt?"))
    >>> miss.refused, miss.text
    (True, 'The available documents do not contain enough information to answer this question.')
    >>> miss.refusal_reason in {'weak_match', 'model_refused'}
    True

    Every stage is timed, and the refusal path stops early -- there is no generate
    stage when the evidence gate refuses:

    >>> [s.name for s in answer.stages]
    ['retrieve', 'gate', 'hydrate', 'rerank', 'context', 'generate', 'verify']
    >>> gate = asyncio.run(pipeline.answer("xylophone tessellation", min_score=0.99))
    >>> gate.refused, gate.refusal_reason, [s.name for s in gate.stages]
    (True, 'weak_match', ['retrieve', 'gate'])
    >>> gate.usage.prompt_tokens
    0

    Restricting to a document restricts the evidence base, and a question the
    selected document does not cover is refused rather than answered from a
    document the user excluded:

    >>> scoped = asyncio.run(pipeline.answer("retort sterilisation temperature",
    ...                          document_ids=["d2"]))
    >>> scoped.refused, scoped.context.document_ids
    (False, ('d2',))
    >>> scoped.text
    'Retort sterilisation of canned vegetables is performed at 121 C [1].'
    >>> asyncio.run(pipeline.answer("retort sterilisation temperature",
    ...                 document_ids=["d1"])).refusal_reason
    'weak_match'

    Selecting no documents retrieves nothing rather than silently searching
    everything:

    >>> asyncio.run(pipeline.answer("mango", document_ids=[])).refusal_reason
    'no_passages'

    The context holds every passage the model was shown; the citations hold only
    the ones it used. Both are reported, because conflating them would hide how
    much evidence was considered:

    >>> answer.context.document_ids, answer.document_ids
    (('d1', 'd2', 'd3'), ('d1',))

    An empty question refuses at the gate instead of prompting the model with
    nothing:

    >>> asyncio.run(pipeline.answer("   ")).refusal_reason
    'no_passages'

    A chunk that cannot be hydrated is dropped, and if that leaves nothing the
    refusal says so instead of citing a passage with no text:

    >>> blind = RagPipeline(
    ...     retriever=HybridRetriever(embedder, store, keyword),
    ...     llm=ExtractiveClient(), hydrate=lambda ids: {})
    >>> asyncio.run(blind.answer("dried mango")).refusal_reason
    'no_passage_text'
    >>> asyncio.run(pipeline.answer("retort sterilisation temperature",
    ...                 document_ids=["d1"])).refusal_reason
    'weak_match'
    """

    def __init__(
        self,
        *,
        retriever: Any,
        llm: LlmClient,
        hydrate: Hydrator,
        reranker: Reranker | None = None,
        builder: ContextBuilder | None = None,
        titles: TitleResolver | None = None,
        candidate_limit: int | None = None,
        rerank_top_n: int = 6,
        min_score: float = 0.16,
        min_passages: int = 1,
        support_threshold: float = 0.32,
        enforce_citations: bool = True,
        max_history_turns: int = prompts.MAX_HISTORY_TURNS,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.hydrate = hydrate
        self.reranker = reranker
        self.builder = builder or ContextBuilder()
        self.titles = titles
        self.candidate_limit = candidate_limit
        self.rerank_top_n = max(1, int(rerank_top_n))
        self.min_score = float(min_score)
        self.min_passages = max(1, int(min_passages))
        self.support_threshold = float(support_threshold)
        self.enforce_citations = bool(enforce_citations)
        self.max_history_turns = max(0, int(max_history_turns))

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        retriever: Any,
        llm: LlmClient,
        hydrate: Hydrator,
        reranker: Reranker | None = None,
        titles: TitleResolver | None = None,
    ) -> "RagPipeline":
        """Build a pipeline from configuration, with collaborators injected.

        The collaborators are arguments rather than constructed here because two
        of them (the vector store and the chunks table) are process-wide resources
        the application owns, and a pipeline that opened its own would make the
        request path responsible for connection lifetimes.
        """
        return cls(
            retriever=retriever,
            llm=llm,
            hydrate=hydrate,
            reranker=reranker,
            builder=ContextBuilder.from_settings(settings),
            titles=titles,
            candidate_limit=getattr(settings, "retrieval_candidate_limit", None),
            rerank_top_n=getattr(settings, "rerank_top_n", 6),
            min_score=getattr(settings, "grounding_min_score", 0.16),
            min_passages=getattr(settings, "grounding_min_passages", 1),
            support_threshold=getattr(settings, "grounding_support_threshold", 0.32),
            enforce_citations=getattr(settings, "grounding_enforce_citations", True),
        )

    # -- the pipeline ------------------------------------------------------- #

    async def answer(
        self,
        question: str,
        *,
        filters: Mapping[str, Any] | None = None,
        document_ids: Iterable[str] | None = None,
        history: Sequence[Message] = (),
        mode: str | None = None,
        min_score: float | None = None,
    ) -> Answer:
        """Answer one question, or refuse with a reason."""
        text = str(question or "").strip()
        stages: list[Stage] = []
        active_filters = self._filters(filters, document_ids)

        # 1. Retrieve.
        started = time.perf_counter()
        retrieval = self.retriever.retrieve(
            text, top_k=self.candidate_limit, filters=active_filters, mode=mode
        )
        stages.append(
            Stage("retrieve", _ms(started), {"candidates": len(retrieval)})
        )
        diagnostics = (
            retrieval.diagnostics.to_dict()
            if hasattr(retrieval, "diagnostics")
            else {}
        )

        # 2. Gate -- before any generation.
        started = time.perf_counter()
        evidence = check_evidence(
            retrieval,
            min_score=self.min_score if min_score is None else float(min_score),
            min_passages=self.min_passages,
        )
        stages.append(Stage("gate", _ms(started), evidence.to_dict()))
        if not evidence.sufficient:
            return self._refusal(
                text, evidence.reason, evidence, stages, retrieval=diagnostics
            )

        # 3. Hydrate passage text.
        started = time.perf_counter()
        hydrate_result = self.hydrate(list(retrieval.ids))
        if asyncio.iscoroutine(hydrate_result):
            hydrate_result = await hydrate_result
        texts = dict(hydrate_result or {})
        candidates = candidates_from_hits(list(retrieval.hits), texts)
        stages.append(
            Stage(
                "hydrate",
                _ms(started),
                {"requested": len(retrieval), "hydrated": len(candidates)},
            )
        )
        if not candidates:
            return self._refusal(
                text, REFUSAL_NO_TEXT, evidence, stages, retrieval=diagnostics
            )

        # 4. Rerank.
        started = time.perf_counter()
        if self.reranker is not None:
            ranked: Sequence[Any] = self.reranker.rerank(
                text, candidates, top_n=self.rerank_top_n
            )
            detail = {
                "reranker": type(self.reranker).__name__,
                "in": len(candidates),
                "out": len(ranked),
                "moved": sum(
                    1 for r in ranked if isinstance(r, RerankResult) and r.movement
                ),
            }
        else:
            ranked = candidates[: self.rerank_top_n]
            detail = {"reranker": "none", "in": len(candidates), "out": len(ranked)}
        stages.append(Stage("rerank", _ms(started), detail))

        # 5. Build the context block.
        started = time.perf_counter()
        block = self.builder.build(ranked, titles=await self._titles(ranked))
        stages.append(Stage("context", _ms(started), block.to_dict()))
        if not block:
            return self._refusal(
                text,
                REFUSAL_EMPTY_CONTEXT,
                evidence,
                stages,
                context=block,
                retrieval=diagnostics,
            )

        # 6. Generate.
        started = time.perf_counter()
        response = self.llm.complete(
            prompts.build_messages(
                text,
                block.text,
                history=history,
                max_history_turns=self.max_history_turns,
            ),
            passages=block.as_pairs(),
        )
        stages.append(
            Stage(
                "generate",
                _ms(started),
                {
                    "provider": response.provider,
                    "model": response.model,
                    "finish_reason": response.finish_reason,
                    "truncated": response.truncated,
                    # Prompt and completion text are deliberately absent: this
                    # dict is logged and returned over the API.
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            )
        )

        # 7. Verify.
        return self._verify(text, response, block, evidence, stages, diagnostics)

    # -- stages ------------------------------------------------------------- #

    def _verify(
        self,
        question: str,
        response: LlmResponse,
        block: ContextBlock,
        evidence: EvidenceCheck,
        stages: list[Stage],
        diagnostics: Mapping[str, Any],
    ) -> Answer:
        """Resolve citations, score support, and decide whether to stand behind it."""
        started = time.perf_counter()
        raw = str(response.text or "").strip()

        # The extractive backend signals "no matching sentence" structurally; a
        # generative one says it in words. Both mean the same thing.
        if not raw or response.finish_reason == "insufficient_evidence":
            stages.append(Stage("verify", _ms(started), {"outcome": "model_refused"}))
            return self._refusal(
                question,
                REFUSAL_MODEL,
                evidence,
                stages,
                context=block,
                response=response,
                retrieval=diagnostics,
            )

        report = resolve(raw, block)
        grounding = check_support(
            report.answer,
            block,
            threshold=self.support_threshold,
            enforce_citations=self.enforce_citations,
            refused=report.refused,
        )

        if report.refused:
            stages.append(Stage("verify", _ms(started), {"outcome": "model_refused"}))
            return self._refusal(
                question,
                REFUSAL_MODEL,
                evidence,
                stages,
                context=block,
                response=response,
                retrieval=diagnostics,
            )

        if grounding.advises_refusal:
            stages.append(
                Stage(
                    "verify",
                    _ms(started),
                    {"outcome": "suppressed", "groundedness": grounding.groundedness},
                )
            )
            refusal = self._refusal(
                question,
                REFUSAL_UNSUPPORTED,
                evidence,
                stages,
                context=block,
                response=response,
                retrieval=diagnostics,
            )
            # Keep the withdrawn text for the evaluation dashboard: a suppressed
            # hallucination that leaves no trace cannot be counted.
            return Answer(
                **{
                    **_as_kwargs(refusal),
                    "grounding": grounding,
                    "citation_report": report,
                    "suppressed_text": report.answer,
                }
            )

        stages.append(
            Stage(
                "verify",
                _ms(started),
                {
                    "outcome": "answered",
                    "citations": len(report.citations),
                    "invalid_markers": list(report.invalid_markers),
                    "groundedness": grounding.groundedness,
                    "coverage": report.coverage,
                },
            )
        )
        return Answer(
            question=question,
            text=report.answer,
            citations=report.citations,
            context=block,
            evidence=evidence,
            grounding=grounding,
            citation_report=report,
            provider=response.provider,
            model=response.model,
            usage=response.usage,
            stages=tuple(stages),
            retrieval=dict(diagnostics),
        )

    def _refusal(
        self,
        question: str,
        reason: str,
        evidence: EvidenceCheck,
        stages: Sequence[Stage],
        *,
        context: ContextBlock | None = None,
        response: LlmResponse | None = None,
        retrieval: Mapping[str, Any] | None = None,
    ) -> Answer:
        """A refusal carrying the same standard sentence whatever caused it."""
        return Answer(
            question=question,
            text=prompts.INSUFFICIENT_EVIDENCE,
            refused=True,
            refusal_reason=reason,
            context=context or ContextBlock(),
            evidence=evidence,
            grounding=GroundingReport(
                threshold=self.support_threshold,
                enforce_citations=self.enforce_citations,
                refused=True,
            ),
            citation_report=CitationReport(
                answer=prompts.INSUFFICIENT_EVIDENCE, refused=True
            ),
            provider=response.provider if response else self.llm.name,
            model=response.model if response else getattr(self.llm, "model", ""),
            usage=response.usage if response else Usage(),
            stages=tuple(stages),
            retrieval=dict(retrieval or {}),
        )

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _filters(
        filters: Mapping[str, Any] | None, document_ids: Iterable[str] | None
    ) -> dict[str, Any]:
        """Merge the document selection into the caller's filters.

        ``document_ids`` is a convenience for the multi-document selector; other
        filters (notably ``user_id``, which is how tenant isolation is enforced)
        are the caller's responsibility and are preserved.

        >>> RagPipeline._filters({"user_id": 7}, ["a", "b"])
        {'user_id': 7, 'document_id': ['a', 'b']}
        >>> RagPipeline._filters(None, None)
        {}

        An empty selection means "no documents", not "all documents":

        >>> RagPipeline._filters(None, [])
        {'document_id': []}
        """
        merged = dict(filters or {})
        if document_ids is not None:
            merged["document_id"] = list(document_ids)
        return merged

    async def _titles(self, ranked: Sequence[Any]) -> Mapping[str, str]:
        """Resolve display titles for the documents in ``ranked``."""
        if self.titles is None:
            return {}
        ids: dict[str, None] = {}
        for item in ranked:
            payload = getattr(item, "payload", {}) or {}
            document_id = str(payload.get("document_id") or "")
            if document_id:
                ids.setdefault(document_id, None)
        titles_result = self.titles(list(ids))
        if asyncio.iscoroutine(titles_result):
            titles_result = await titles_result
        return dict(titles_result or {})

    def describe(self) -> dict[str, Any]:
        """The active configuration, for the Settings screen and evaluation runs."""
        return {
            "llm": self.llm.describe(),
            "reranker": (
                type(self.reranker).__name__ if self.reranker else "none"
            ),
            "rerank_top_n": self.rerank_top_n,
            "context": self.builder.describe(),
            "grounding": {
                "min_score": self.min_score,
                "min_passages": self.min_passages,
                "support_threshold": self.support_threshold,
                "enforce_citations": self.enforce_citations,
            },
            "prompt_version": prompts.PROMPT_VERSION,
        }


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _as_kwargs(answer: Answer) -> dict[str, Any]:
    """Field values of an :class:`Answer`, for building a modified copy.

    ``dataclasses.asdict`` would recurse into the nested reports and turn them
    into dictionaries, which is not what a replacement constructor needs.
    """
    return {
        "question": answer.question,
        "text": answer.text,
        "refused": answer.refused,
        "refusal_reason": answer.refusal_reason,
        "citations": answer.citations,
        "context": answer.context,
        "evidence": answer.evidence,
        "grounding": answer.grounding,
        "citation_report": answer.citation_report,
        "provider": answer.provider,
        "model": answer.model,
        "usage": answer.usage,
        "stages": answer.stages,
        "prompt_version": answer.prompt_version,
        "retrieval": answer.retrieval,
        "suppressed_text": answer.suppressed_text,
    }


__all__ = [
    "Answer",
    "Hydrator",
    "REFUSAL_EMPTY_CONTEXT",
    "REFUSAL_MODEL",
    "REFUSAL_NO_TEXT",
    "REFUSAL_UNSUPPORTED",
    "RagPipeline",
    "Stage",
    "TitleResolver",
]
