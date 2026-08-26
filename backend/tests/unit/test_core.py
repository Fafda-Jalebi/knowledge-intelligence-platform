"""Unit tests for core retrieval and RAG components."""

import pytest
import numpy as np
from kip.core.embeddings.hashing import HashingEmbedder, features
from kip.core.vectorstore.memory import MemoryVectorStore
from kip.core.vectorstore.base import VectorRecord
from kip.core.retrieval.bm25 import Bm25Index
from kip.core.retrieval.keyword import KeywordDocument
from kip.core.retrieval.hybrid import HybridRetriever
from kip.core.retrieval.fusion import fuse, reciprocal_rank_fusion
from kip.core.rerank.heuristic import HeuristicReranker
from kip.core.rerank.base import RerankCandidate, NoOpReranker
from kip.core.rag.citations import resolve, CitationReport
from kip.core.rag.context import ContextBuilder
from kip.core.rag.grounding import check_evidence, check_support, EvidenceCheck
from kip.core.rag.pipeline import RagPipeline
from kip.core.llm.extractive import ExtractiveClient
from kip.core.llm.base import Message


class TestHashingEmbedder:
    def test_deterministic(self):
        embedder = HashingEmbedder(dim=1024)
        texts = ["hello world", "test document"]
        v1 = embedder.embed_documents(texts)
        v2 = embedder.embed_documents(texts)
        assert np.allclose(v1, v2)

    def test_normalized(self):
        embedder = HashingEmbedder(dim=1024)
        vecs = embedder.embed_documents(["test"])
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0)

    def test_fingerprint(self):
        embedder = HashingEmbedder(dim=1024)
        assert embedder.fingerprint == "hashing:kip-hashing-v1:1024"

    def test_features(self):
        f = features("Water activity")
        assert "w:water" in f
        assert "w:activ" in f
        assert "b:water activ" in f


class TestMemoryVectorStore:
    def test_upsert_and_search(self):
        embedder = HashingEmbedder(dim=1024)
        store = MemoryVectorStore(dim=1024)
        store.ensure_collection(embedder.spec)

        texts = ["doc one about apples", "doc two about oranges"]
        vecs = embedder.embed_documents(texts)
        records = [
            VectorRecord(id=f"d1:{i}", vector=vecs[i], payload={"document_id": "d1", "chunk_index": i})
            for i in range(2)
        ]
        store.upsert(records)

        query_vec = embedder.embed_query("apples")
        hits = store.search(query_vec, top_k=1)
        assert len(hits) == 1
        assert hits[0].id == "d1:0"

    def test_filter_by_document(self):
        embedder = HashingEmbedder(dim=1024)
        store = MemoryVectorStore(dim=1024)
        store.ensure_collection(embedder.spec)

        texts = ["apples", "oranges", "bananas"]
        vecs = embedder.embed_documents(texts)
        records = [
            VectorRecord(id=f"d1:{i}", vector=vecs[i], payload={"document_id": "d1", "chunk_index": i})
            for i in range(3)
        ]
        store.upsert(records)

        query_vec = embedder.embed_query("fruit")
        hits = store.search(query_vec, top_k=10, filters={"document_id": ["d1"]})
        assert all(h.payload["document_id"] == "d1" for h in hits)


class TestBM25:
    def test_basic_search(self):
        idx = Bm25Index()
        docs = [
            KeywordDocument("d1:0", "apples are red", {"document_id": "d1"}),
            KeywordDocument("d2:0", "oranges are orange", {"document_id": "d2"}),
        ]
        idx.add(docs)

        hits = idx.search("apples", top_k=1)
        assert hits[0].id == "d1:0"

    def test_stemming(self):
        idx = Bm25Index()
        docs = [KeywordDocument("d1:0", "drying mangoes", {"document_id": "d1"})]
        idx.add(docs)

        # Query with different form
        hits = idx.search("dried mango", top_k=1)
        assert len(hits) > 0

    def test_filter(self):
        idx = Bm25Index()
        docs = [
            KeywordDocument("d1:0", "apples", {"document_id": "d1"}),
            KeywordDocument("d2:0", "oranges", {"document_id": "d2"}),
        ]
        idx.add(docs)

        hits = idx.search("fruit", top_k=10, filters={"document_id": ["d1"]})
        assert all(h.id.startswith("d1:") for h in hits)


class TestFusion:
    def test_rrf_fusion(self):
        class Hit:
            def __init__(self, id, score, doc):
                self.id = id
                self.score = score
                self.payload = {"document_id": doc}

        dense = [Hit("both", 0.9, "d1"), Hit("dense_only", 0.8, "d2")]
        keyword = [Hit("keyword_only", 12.0, "d3"), Hit("both", 4.0, "d1")]

        fused = fuse({"dense": dense, "keyword": keyword}, method="rrf")
        assert fused[0].id == "both"
        assert set(fused[0].retrievers) == {"dense", "keyword"}

    def test_weighted_fusion(self):
        class Hit:
            def __init__(self, id, score, doc):
                self.id = id
                self.score = score
                self.payload = {"document_id": doc}

        dense = [Hit("a", 0.9, "d1"), Hit("b", 0.5, "d2")]
        keyword = [Hit("b", 10.0, "d2"), Hit("a", 5.0, "d1")]

        fused = fuse({"dense": dense, "keyword": keyword}, method="weighted", weights={"dense": 0.5, "keyword": 0.5})
        assert len(fused) == 2


class TestHeuristicReranker:
    def test_rerank_basic(self):
        reranker = HeuristicReranker()
        candidates = [
            RerankCandidate("noise", "Mango. Mango. Mango.", 0.9),
            RerankCandidate("answer", "Mango slices are dried at 60 C.", 0.8),
        ]
        results = reranker.rerank("drying temperature for mango", candidates, top_n=1)
        assert results[0].id == "answer"

    def test_noop_reranker(self):
        reranker = NoOpReranker()
        candidates = [
            RerankCandidate("a", "first", 0.9),
            RerankCandidate("b", "second", 0.4),
        ]
        results = reranker.rerank("anything", candidates)
        assert results[0].id == "a"
        assert results[0].movement == 0
        assert results[1].movement == 0


class TestCitations:
    def test_resolve_valid(self):
        from kip.core.rag.context import ContextBlock, ContextPassage

        block = ContextBlock(
            passages=(
                ContextPassage(marker=1, id="d1:0", text="Dried at 60 C.", document_id="d1", chunk_index=0, document_label="Doc 1", page_start=1, page_end=1, section_path=(), score=0.9, truncated=False),
                ContextPassage(marker=2, id="d2:0", text="Water activity low.", document_id="d2", chunk_index=0, document_label="Doc 2", page_start=1, page_end=1, section_path=(), score=0.8, truncated=False),
            ),
            token_count=100,
            budget=2000,
        )

        report = resolve("Dried at 60 C [1]. Water activity is low [2].", block)
        assert not report.refused
        assert len(report.citations) == 2
        assert report.citations[0].marker == 1
        assert report.citations[1].marker == 2

    def test_invented_marker_removed(self):
        from kip.core.rag.context import ContextBlock, ContextPassage

        block = ContextBlock(
            passages=(
                ContextPassage(marker=1, id="d1:0", text="Dried at 60 C.", document_id="d1", chunk_index=0, document_label="Doc 1", page_start=1, page_end=1, section_path=(), score=0.9, truncated=False),
            ),
            token_count=50,
            budget=2000,
        )

        report = resolve("Dried at 60 C [1]. Fake claim [99].", block)
        assert not report.refused
        assert len(report.citations) == 1
        assert report.citations[0].marker == 1
        assert 99 in report.invalid_markers
        assert "[99]" not in report.answer


class TestContextBuilder:
    def test_build_context(self):
        from kip.core.rerank.base import RerankResult

        ranked = [
            RerankResult("d1:0", 0.9, 0.5, prior_rank=1, rank=1, text="Passage one.", payload={"document_id": "d1", "chunk_index": 0}),
            RerankResult("d2:0", 0.8, 0.4, prior_rank=2, rank=2, text="Passage two.", payload={"document_id": "d2", "chunk_index": 0}),
        ]

        builder = ContextBuilder(token_budget=1000, max_passages=8, max_per_document=4)
        block = builder.build(ranked, titles={"d1": "Doc 1", "d2": "Doc 2"})

        assert len(block) == 2
        assert block.passages[0].marker == 1
        assert block.passages[1].marker == 2
        assert block.token_count > 0


class TestGrounding:
    def test_evidence_gate_sufficient(self):
        class Retrieval:
            def __len__(self): return 5
            @property
            def top_dense_score(self): return 0.5

        gate = check_evidence(Retrieval(), min_score=0.16, min_passages=1)
        assert gate.sufficient

    def test_evidence_gate_weak_match(self):
        class Retrieval:
            def __len__(self): return 5
            @property
            def top_dense_score(self): return 0.05

        gate = check_evidence(Retrieval(), min_score=0.16, min_passages=1)
        assert not gate.sufficient
        assert gate.reason == "weak_match"

    def test_evidence_gate_no_passages(self):
        class Retrieval:
            def __len__(self): return 0
            @property
            def top_dense_score(self): return None

        gate = check_evidence(Retrieval(), min_score=0.16, min_passages=1)
        assert not gate.sufficient
        assert gate.reason == "no_passages"

    def test_support_check(self):
        from kip.core.rag.context import ContextBlock, ContextPassage

        block = ContextBlock(
            passages=(
                ContextPassage(marker=1, id="d1:0", text="Mango slices are dried at 60 C.", document_id="d1", chunk_index=0, document_label="Doc 1", page_start=1, page_end=1, section_path=(), score=0.9, truncated=False),
            ),
            token_count=50,
            budget=2000,
        )

        report = check_support("Mango slices are dried at 60 C [1].", block)
        assert report.groundedness == 1.0
        assert len(report.unsupported) == 0

    def test_support_check_unsupported(self):
        from kip.core.rag.context import ContextBlock, ContextPassage

        block = ContextBlock(
            passages=(
                ContextPassage(marker=1, id="d1:0", text="Mango slices are dried at 60 C.", document_id="d1", chunk_index=0, document_label="Doc 1", page_start=1, page_end=1, section_path=(), score=0.9, truncated=False),
            ),
            token_count=50,
            budget=2000,
        )

        report = check_support("Shelf life is 12 months [1].", block)
        assert report.groundedness == 0.0
        assert len(report.unsupported) == 1


class TestRagPipeline:
    async def test_pipeline_end_to_end(self):
        from kip.core.vectorstore.memory import MemoryVectorStore
        from kip.core.vectorstore.base import VectorRecord
        from kip.core.retrieval import Bm25Index, HybridRetriever
        from kip.core.retrieval.keyword import KeywordDocument

        passages = {
            "d1:0": "Hot air drying of mango slices is carried out at 60 C for eight hours.",
            "d2:0": "Water activity below 0.6 inhibits microbial growth.",
        }

        embedder = HashingEmbedder(dim=1024)
        store = MemoryVectorStore(dim=1024)
        store.ensure_collection(embedder.spec)
        vecs = embedder.embed_documents(list(passages.values()))
        store.upsert([
            VectorRecord(k, vecs[i], {"document_id": k.split(":")[0], "chunk_index": 0})
            for i, k in enumerate(passages)
        ])

        keyword = Bm25Index()
        keyword.add([
            KeywordDocument(k, v, {"document_id": k.split(":")[0], "chunk_index": 0})
            for k, v in passages.items()
        ])

        retriever = HybridRetriever(embedder, store, keyword)
        llm = ExtractiveClient()

        pipeline = RagPipeline(
            retriever=retriever,
            llm=llm,
            hydrate=lambda ids: {i: passages[i] for i in ids if i in passages},
            titles=lambda ids: {"d1": "Mango Drying Study", "d2": "Water Activity Note"},
        )

        answer = await pipeline.answer("At what temperature are mango slices dried?")
        assert not answer.refused
        assert "60 C" in answer.text
        # At least one citation should be from the relevant document
        assert any(c.id == "d1:0" for c in answer.citations)
        assert answer.citations[0].id == "d1:0"

        # Unanswerable question
        refused = await pipeline.answer("What is the tensile strength of steel?")
        assert refused.refused
        assert refused.refusal_reason in {"weak_match", "model_refused"}
