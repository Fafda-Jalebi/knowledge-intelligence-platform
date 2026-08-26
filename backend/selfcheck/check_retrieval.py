"""Cross-implementation retrieval checks.

Doctests verify each module against its own docstring. These checks verify things
a doctest structurally cannot: that two *independent implementations* of the same
contract agree, that a pipeline of six components composes correctly, and that
the engine is domain-agnostic rather than merely claimed to be.

What is checked here and nowhere else
-------------------------------------
**Backend parity.** ``MemoryVectorStore`` and ``SqliteVectorStore`` must return
identical rankings -- same algorithm, different storage, so any divergence is a
bug. ``Bm25Index`` and ``SqliteFtsIndex`` must retrieve the same gold passages but
are *not* required to rank identically, because their IDF formulas genuinely
differ (see ``kip/core/retrieval/fts.py``). Encoding that distinction as two
different assertions is the point: a check that demanded identical BM25 rankings
would be wrong, and a check that only demanded overlap would miss a real vector
bug.

**Domain agnosticism.** The probe corpus deliberately mixes food technology with
bridge inspection and archival cataloguing. A query about scour countermeasures
must retrieve the civil engineering passage, not the nearest food passage. If any
domain assumption ever leaks into the retrieval engine, this is where it surfaces.

**Deletion and isolation.** Deleting a document must remove it from every index,
and a per-document filter must be enforced on both axes -- otherwise a citation
could quote a passage the user no longer has, or never had.

The probe corpus below is deliberately tiny and separate from the evaluation
dataset used by ``kip.eval``. This is a smoke floor with hand-written gold
answers; measured retrieval quality is reported by the evaluation harness, and
the two must not be confused. All passages are synthetic, written for this test,
and paraphrase no source.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from selfcheck.harness import Harness

# --------------------------------------------------------------------------- #
# Probe corpus. SYNTHETIC -- written for this test, not extracted from any source.
# Three domains, so that a domain assumption in the engine cannot pass unnoticed.
# --------------------------------------------------------------------------- #

PASSAGES: list[tuple[str, str, str, str]] = [
    # (chunk_id, document_id, heading, text)
    (
        "food:0",
        "food",
        "Hot air drying",
        "Mango slices are dried in a hot air drier at 60 C until the moisture "
        "content falls below 15 percent on a wet basis. Slice thickness governs "
        "drying time far more strongly than air velocity does.",
    ),
    (
        "food:1",
        "food",
        "Water activity and shelf life",
        "Reducing water activity below 0.6 arrests the growth of moulds and "
        "osmophilic yeasts. Water activity, not total moisture content, is the "
        "controlling variable for microbial stability during storage.",
    ),
    (
        "food:2",
        "food",
        "Retort sterilisation",
        "Low acid canned vegetables are retorted at 121 C. The process is "
        "designed around a target lethality expressed in minutes at that "
        "reference temperature.",
    ),
    (
        "food:3",
        "food",
        "Acidified products",
        "Products acidified to pH 4.6 or below are treated as high acid foods "
        "and do not require a full botulinum cook.",
    ),
    (
        "pack:0",
        "pack",
        "Barrier films",
        "Multilayer films combine a moisture barrier with an oxygen barrier "
        "because no single economical polymer provides both. Oxygen transmission "
        "rate is quoted per square metre per day.",
    ),
    (
        "pack:1",
        "pack",
        "Modified atmosphere",
        "Replacing headspace air with a nitrogen and carbon dioxide mixture slows "
        "oxidative rancidity in fatty snack products.",
    ),
    (
        "bridge:0",
        "bridge",
        "Scour countermeasures",
        "Riprap aprons and articulated concrete blocks are the usual scour "
        "countermeasures at bridge piers. Inspection intervals shorten after any "
        "flood exceeding the design discharge.",
    ),
    (
        "bridge:1",
        "bridge",
        "Fatigue in welded details",
        "Fatigue cracking initiates at welded attachment details subject to "
        "distortion induced stress. Crack growth is monitored against the "
        "detail category assigned during design.",
    ),
    (
        "arch:0",
        "arch",
        "Authority control",
        "Authority control reconciles variant forms of a name to a single "
        "preferred heading so that a catalogue collocates all works by one "
        "author regardless of how the name was transcribed.",
    ),
    (
        "arch:1",
        "arch",
        "Original order",
        "The principle of original order requires that records be kept in the "
        "arrangement imposed by their creator, because that arrangement is "
        "itself evidence of how the creator worked.",
    ),
]

#: (query, expected chunk id). Hand-written; each answer is stated explicitly in
#: exactly one passage, so a miss is a retrieval failure and not an ambiguity.
QUERIES: list[tuple[str, str]] = [
    ("what stops mould growth in dried fruit", "food:1"),
    ("drying temperature for mango slices", "food:0"),
    ("retort temperature for canned vegetables", "food:2"),
    ("pH 4.6", "food:3"),
    ("why are multilayer films used", "pack:0"),
    ("scour countermeasures at bridge piers", "bridge:0"),
    ("welded detail fatigue cracking", "bridge:1"),
    ("reconciling variant forms of an author name", "arch:0"),
]

#: Queries whose answer lies outside food technology. Asserted separately so a
#: regression here reads as "the engine acquired a domain bias", not as a
#: general recall dip.
CROSS_DOMAIN = ("scour countermeasures at bridge piers", "reconciling variant forms of an author name")


def _payload(chunk_id: str, document_id: str, heading: str) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "chunk_index": int(chunk_id.rsplit(":", 1)[1]),
        "heading": heading,
        "user_id": "u1",
    }


def _texts() -> dict[str, str]:
    return {chunk_id: text for chunk_id, _doc, _heading, text in PASSAGES}


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_embeddings(h: Harness) -> Any:
    """The dense axis: determinism, normalisation, and ordering sanity."""
    from kip.core.embeddings import cosine_similarity
    from kip.core.embeddings.hashing import DEFAULT_DIM, HashingEmbedder

    h.group("Embeddings")
    embedder = HashingEmbedder()
    h.equal(embedder.dim, DEFAULT_DIM, "embeddings: hashing embedder reports its width")
    h.equal(
        embedder.fingerprint,
        f"hashing:kip-hashing-v1:{DEFAULT_DIM}",
        "embeddings: fingerprint identifies provider, model and width",
    )

    texts = [text for _id, _doc, _heading, text in PASSAGES]
    matrix = embedder.embed_documents(texts)
    h.equal(matrix.shape, (len(texts), DEFAULT_DIM), "embeddings: one row per passage")

    again = embedder.embed_documents(texts)
    h.ok(
        bool((matrix == again).all()),
        "embeddings: encoding is deterministic across calls",
    )

    norms = (matrix**2).sum(axis=1) ** 0.5
    h.between(
        float(norms.min()), 0.999, 1.001, "embeddings: every vector is L2 normalised"
    )

    # Cosine equals dot product only because the vectors are unit length; if
    # normalisation ever regresses, similarity silently stops being bounded.
    query = embedder.embed_query("water activity and mould growth")
    scores = cosine_similarity(query, matrix)
    h.between(float(scores.max()), 0.0, 1.0001, "embeddings: similarity stays within [0, 1]")
    best = int(scores.argmax())
    h.equal(
        PASSAGES[best][0],
        "food:1",
        "embeddings: nearest neighbour is the water activity passage",
    )
    self_score = float(cosine_similarity(embedder.embed_query(texts[1]), matrix)[1])
    h.between(self_score, 0.999, 1.001, "embeddings: a passage is maximally similar to itself")
    return embedder


def check_vector_store_parity(h: Harness, embedder: Any, tmp: Path) -> None:
    """Two storage backends, one algorithm: rankings must be identical."""
    from kip.core.vectorstore.base import VectorRecord
    from kip.core.vectorstore.memory import MemoryVectorStore
    from kip.core.vectorstore.sqlite_store import SqliteVectorStore

    h.group("Vector store parity")
    vectors = embedder.embed_documents([text for _i, _d, _hd, text in PASSAGES])
    records = [
        VectorRecord(id=chunk_id, vector=vectors[index], payload=_payload(chunk_id, doc, heading))
        for index, (chunk_id, doc, heading, _text) in enumerate(PASSAGES)
    ]

    memory = MemoryVectorStore(dim=embedder.dim)
    sqlite_store = SqliteVectorStore(tmp / "vectors.sqlite3", dim=embedder.dim)
    try:
        for store in (memory, sqlite_store):
            store.ensure_collection(embedder.spec)
            h.equal(
                store.upsert(records), len(records), f"vectors[{store.name}]: upsert wrote every record"
            )
            h.equal(store.count(), len(records), f"vectors[{store.name}]: count matches")
            h.equal(
                store.stored_fingerprint(),
                embedder.fingerprint,
                f"vectors[{store.name}]: fingerprint recorded on the collection",
            )

        agreed = 0
        for query, _gold in QUERIES:
            vector = embedder.embed_query(query)
            left = [(hit.id, round(hit.score, 6)) for hit in memory.search(vector, top_k=5)]
            right = [(hit.id, round(hit.score, 6)) for hit in sqlite_store.search(vector, top_k=5)]
            if h.equal(right, left, f"vectors: memory and sqlite agree for {query!r}"):
                agreed += 1
        h.equal(agreed, len(QUERIES), "vectors: every probe query agreed across backends")

        # Filters must be enforced identically too, since document selection in
        # the UI depends on them and a leak would expose another user's passage.
        for store in (memory, sqlite_store):
            vector = embedder.embed_query("water activity")
            scoped = store.search(vector, top_k=10, filters={"document_id": ["bridge"]})
            h.ok(
                scoped and all(hit.payload["document_id"] == "bridge" for hit in scoped),
                f"vectors[{store.name}]: document filter restricts results",
            )
            h.equal(
                store.search(vector, top_k=10, filters={"document_id": []}),
                [],
                f"vectors[{store.name}]: an empty selection returns nothing",
            )
            h.equal(
                store.delete(filters={"document_id": ["arch"]}),
                2,
                f"vectors[{store.name}]: delete removes the whole document",
            )
            h.equal(
                store.count(filters={"document_id": ["arch"]}),
                0,
                f"vectors[{store.name}]: deleted passages are gone",
            )
            h.raises(
                ValueError,
                lambda store=store: store.delete(filters={}),
                f"vectors[{store.name}]: refuses an unfiltered delete",
            )
    finally:
        memory.close()
        sqlite_store.close()


def check_keyword_agreement(h: Harness, tmp: Path) -> None:
    """Two BM25 implementations: same gold passages, rankings allowed to differ."""
    from kip.core.retrieval import Bm25Index
    from kip.core.retrieval.keyword import KeywordDocument

    h.group("Keyword backend agreement")
    documents = [
        KeywordDocument(id=chunk_id, text=f"{heading}\n{text}", payload=_payload(chunk_id, doc, heading))
        for chunk_id, doc, heading, text in PASSAGES
    ]

    bm25 = Bm25Index()
    bm25.add(documents)
    h.equal(bm25.count(), len(PASSAGES), "keyword[bm25]: indexed every passage")

    fts: Any = None
    try:
        from kip.core.retrieval.fts import SqliteFtsIndex, fts5_available

        if fts5_available():
            fts = SqliteFtsIndex(path=str(tmp / "keyword.sqlite3"))
            fts.add(documents)
            h.equal(fts.count(), len(PASSAGES), "keyword[fts5]: indexed every passage")
    except Exception as exc:  # pragma: no cover - depends on the sqlite build
        h.ok(False, f"keyword[fts5]: backend unavailable ({exc})")

    backends = [("bm25", bm25)] + ([("fts5", fts)] if fts is not None else [])
    for label, index in backends:
        found = 0
        for query, gold in QUERIES:
            hits = index.search(query, top_k=5)
            if gold in {hit.id for hit in hits}:
                found += 1
        # A floor, not a measurement: the reported numbers come from kip.eval.
        h.ok(
            found >= 6,
            f"keyword[{label}]: gold passage in top 5 for {found}/{len(QUERIES)} probe queries",
        )

        h.equal(index.search("photosynthesis chlorophyll", top_k=5), [], f"keyword[{label}]: an absent term returns nothing")
        scoped = index.search("water activity", top_k=5, filters={"document_id": ["bridge"]})
        h.equal(scoped, [], f"keyword[{label}]: document filter excludes non-matching documents")
        h.equal(
            index.search("water", top_k=5, filters={"document_id": []}),
            [],
            f"keyword[{label}]: an empty selection returns nothing",
        )

        literal = index.search("0.6", top_k=3)
        h.contains(
            {hit.id for hit in literal},
            "food:1",
            f"keyword[{label}]: a decimal literal survives tokenisation and matches",
        )

        stemmed = index.search("dried mangoes", top_k=3)
        h.contains(
            {hit.id for hit in stemmed},
            "food:0",
            f"keyword[{label}]: inflected query matches the stemmed index",
        )

        h.equal(
            index.delete(filters={"document_id": ["arch"]}),
            2,
            f"keyword[{label}]: delete removes the whole document",
        )
        # Asserting an empty result for a topical query would be wrong: "authority
        # control" stems to include "control", which the food passages also use.
        # What must hold is that no passage of the deleted document is reachable.
        after = index.search("authority control preferred heading original order", top_k=10)
        h.equal(
            [hit.id for hit in after if hit.id.startswith("arch:")],
            [],
            f"keyword[{label}]: no passage of the deleted document is reachable",
        )
        h.equal(
            index.count(filters={"document_id": ["arch"]}),
            0,
            f"keyword[{label}]: the deleted document holds no passages",
        )

    if fts is not None:
        # Both backends must agree on *which* passages are relevant even where
        # they disagree on order, because ranking differences are expected and
        # membership differences are not.
        overlap_ok = True
        for query, _gold in QUERIES:
            left = {hit.id for hit in bm25.search(query, top_k=6)}
            right = {hit.id for hit in fts.search(query, top_k=6)}
            if left and right and not (left & right):
                overlap_ok = False
                h.ok(False, f"keyword: backends found disjoint passages for {query!r}")
        if overlap_ok:
            h.ok(True, "keyword: bm25 and fts5 agree on relevant passages for every probe query")
        fts.close()


def check_fusion(h: Harness) -> None:
    """Fusion must reward agreement between the two axes, and be deterministic."""
    from kip.core.retrieval.fusion import DEFAULT_RRF_K, fuse

    h.group("Fusion")

    class Hit:
        def __init__(self, identifier: str, score: float, document: str) -> None:
            self.id = identifier
            self.score = score
            self.payload = {"document_id": document}

    dense = [Hit("both", 0.9, "d1"), Hit("dense_only", 0.88, "d2")]
    keyword = [Hit("keyword_only", 12.0, "d3"), Hit("both", 4.0, "d1")]

    fused = fuse({"dense": dense, "keyword": keyword}, method="rrf", k=DEFAULT_RRF_K)
    h.equal(
        fused[0].id,
        "both",
        "fusion: a passage both axes returned outranks one only a single axis found",
    )
    h.equal(
        fused[0].retrievers,
        ("dense", "keyword"),
        "fusion: retriever provenance is preserved for the interface",
    )
    h.equal(
        [hit.id for hit in fuse({"dense": dense, "keyword": keyword}, method="rrf")],
        [hit.id for hit in fused],
        "fusion: identical inputs produce identical output",
    )
    h.equal(
        len({hit.id for hit in fused}),
        len(fused),
        "fusion: each passage appears exactly once after merging",
    )
    h.between(
        fused[0].best_score("dense"),
        0.9,
        0.9,
        "fusion: the raw dense score survives fusion for the grounding check",
    )


def check_hybrid(h: Harness, embedder: Any, tmp: Path) -> None:
    """The composed pipeline: three modes, diagnostics, and degraded inputs."""
    from kip.core.retrieval import Bm25Index, HybridRetriever
    from kip.core.retrieval.keyword import KeywordDocument
    from kip.core.vectorstore.base import VectorRecord
    from kip.core.vectorstore.memory import MemoryVectorStore

    h.group("Hybrid retrieval")
    vectors = embedder.embed_documents([text for _i, _d, _hd, text in PASSAGES])
    store = MemoryVectorStore(dim=embedder.dim)
    store.ensure_collection(embedder.spec)
    store.upsert(
        [
            VectorRecord(id=chunk_id, vector=vectors[index], payload=_payload(chunk_id, doc, heading))
            for index, (chunk_id, doc, heading, _text) in enumerate(PASSAGES)
        ]
    )
    keyword = Bm25Index()
    keyword.add(
        [
            KeywordDocument(id=chunk_id, text=f"{heading}\n{text}", payload=_payload(chunk_id, doc, heading))
            for chunk_id, doc, heading, text in PASSAGES
        ]
    )

    try:
        recalled: dict[str, int] = {}
        for mode in ("hybrid", "dense", "keyword"):
            retriever = HybridRetriever(embedder, store, keyword, mode=mode, candidate_limit=8)
            hits = 0
            for query, gold in QUERIES:
                result = retriever.retrieve(query, top_k=5)
                if gold in result.ids:
                    hits += 1
                h.equal(
                    result.diagnostics.mode,
                    mode,
                    f"hybrid[{mode}]: diagnostics report the mode actually used",
                )
            recalled[mode] = hits
            h.ok(
                hits >= 6,
                f"hybrid[{mode}]: gold passage retrieved for {hits}/{len(QUERIES)} probe queries",
            )

        h.ok(
            recalled["hybrid"] >= max(recalled["dense"], recalled["keyword"]),
            f"hybrid: fusion is at least as good as either axis alone "
            f"(hybrid {recalled['hybrid']}, dense {recalled['dense']}, keyword {recalled['keyword']})",
        )

        retriever = HybridRetriever(embedder, store, keyword, candidate_limit=8)

        # Domain agnosticism: the engine must not prefer the demonstration domain.
        for query in CROSS_DOMAIN:
            gold = dict(QUERIES)[query]
            result = retriever.retrieve(query, top_k=3)
            h.contains(
                result.ids,
                gold,
                f"hybrid: non-food query {query!r} retrieves its own domain",
            )
            h.ok(
                result.ids[0].split(":")[0] != "food",
                f"hybrid: top hit for {query!r} is not a food technology passage",
            )

        diagnostics = retriever.retrieve("water activity", top_k=3).diagnostics
        h.ok(diagnostics.dense_candidates > 0, "hybrid: dense candidate count is reported")
        h.ok(diagnostics.keyword_candidates > 0, "hybrid: keyword candidate count is reported")
        h.equal(
            diagnostics.embedding_fingerprint,
            embedder.fingerprint,
            "hybrid: diagnostics record which embedding model produced the ranking",
        )
        h.ok(diagnostics.total_ms >= 0.0, "hybrid: elapsed time is measured, not estimated")

        blank = retriever.retrieve("   ", top_k=5)
        h.equal(len(blank), 0, "hybrid: a blank query retrieves nothing")
        h.ok(blank.diagnostics.notes, "hybrid: a blank query explains itself in the diagnostics")

        dense_only = HybridRetriever(embedder, store, None, candidate_limit=8)
        result = dense_only.retrieve("water activity", top_k=3)
        h.ok(result.ids, "hybrid: retrieval still works with no keyword index configured")
        h.ok(
            any("keyword" in note.lower() for note in result.diagnostics.notes),
            "hybrid: the missing keyword index is disclosed, not hidden",
        )

        scoped = retriever.retrieve("water activity", top_k=5, filters={"document_id": ["pack"]})
        h.ok(
            scoped.ids and set(scoped.document_ids) == {"pack"},
            "hybrid: document selection is enforced across both axes",
        )
    finally:
        store.close()


def check_rerank(h: Harness, embedder: Any, tmp: Path) -> None:
    """The second stage: it must reorder for a reason, and account for itself."""
    from kip.core.rerank import (
        HeuristicReranker,
        NoOpReranker,
        RerankError,
        candidates_from_hits,
        get_reranker,
    )
    from kip.core.retrieval import Bm25Index, HybridRetriever
    from kip.core.retrieval.keyword import KeywordDocument
    from kip.core.vectorstore.base import VectorRecord
    from kip.core.vectorstore.memory import MemoryVectorStore

    h.group("Reranking")
    texts = _texts()
    vectors = embedder.embed_documents([text for _i, _d, _hd, text in PASSAGES])
    store = MemoryVectorStore(dim=embedder.dim)
    store.ensure_collection(embedder.spec)
    store.upsert(
        [
            VectorRecord(id=chunk_id, vector=vectors[index], payload=_payload(chunk_id, doc, heading))
            for index, (chunk_id, doc, heading, _text) in enumerate(PASSAGES)
        ]
    )
    keyword = Bm25Index()
    keyword.add(
        [
            KeywordDocument(id=chunk_id, text=f"{heading}\n{text}", payload=_payload(chunk_id, doc, heading))
            for chunk_id, doc, heading, text in PASSAGES
        ]
    )

    try:
        retriever = HybridRetriever(embedder, store, keyword, candidate_limit=10)
        results_by_backend: dict[str, int] = {}

        for label, reranker in (("none", NoOpReranker()), ("heuristic", HeuristicReranker())):
            top1 = 0
            for query, gold in QUERIES:
                retrieved = retriever.retrieve(query, top_k=10)
                candidates = candidates_from_hits(retrieved.hits, texts)
                ranked = reranker.rerank(query, candidates, top_n=5)

                h.ok(len(ranked) <= 5, f"rerank[{label}]: top_n is honoured for {query!r}")
                h.equal(
                    [item.rank for item in ranked],
                    list(range(1, len(ranked) + 1)),
                    f"rerank[{label}]: ranks are contiguous and 1-based for {query!r}",
                )
                h.ok(
                    all(item.movement == item.prior_rank - item.rank for item in ranked),
                    f"rerank[{label}]: reported movement matches the rank change for {query!r}",
                )
                h.ok(
                    all(item.text for item in ranked),
                    f"rerank[{label}]: every ranked passage carries its text for {query!r}",
                )
                if ranked and ranked[0].id == gold:
                    top1 += 1
            results_by_backend[label] = top1
            h.ok(
                top1 >= 5,
                f"rerank[{label}]: gold passage ranked first for {top1}/{len(QUERIES)} probe queries",
            )

        h.ok(
            results_by_backend["heuristic"] >= results_by_backend["none"],
            f"rerank: the heuristic stage does not degrade top-1 "
            f"(heuristic {results_by_backend['heuristic']}, none {results_by_backend['none']})",
        )

        # A no-op reranker must be exactly a no-op, or evaluation comparisons
        # against it are meaningless.
        retrieved = retriever.retrieve("water activity", top_k=6)
        candidates = candidates_from_hits(retrieved.hits, texts)
        passthrough = NoOpReranker().rerank("water activity", candidates)
        h.equal(
            [item.id for item in passthrough],
            [candidate.id for candidate in candidates],
            "rerank[none]: ordering is preserved exactly",
        )
        h.ok(
            all(item.movement == 0 for item in passthrough),
            "rerank[none]: nothing moves",
        )

        h.equal(
            HeuristicReranker().rerank("water activity", []),
            [],
            "rerank: an empty candidate list is not an error",
        )
        h.equal(
            [item.id for item in HeuristicReranker().rerank("   ", candidates)],
            [candidate.id for candidate in candidates],
            "rerank: a blank query preserves the retriever's order",
        )

        # A chunk whose text failed to hydrate must be dropped, never ranked --
        # a citation pointing at nothing is worse than a missing citation.
        partial = candidates_from_hits(retrieved.hits, {retrieved.ids[0]: texts[retrieved.ids[0]]})
        h.equal(len(partial), 1, "rerank: candidates with no hydrated text are dropped")

        h.group("Reranker registry")
        h.equal(get_reranker(_Settings("heuristic")).name, "heuristic", "registry: heuristic builds")
        h.equal(get_reranker(_Settings("none")).name, "none", "registry: none builds a no-op")
        h.equal(get_reranker(_Settings(" HEURISTIC ")).name, "heuristic", "registry: value is normalised")
        h.raises(
            RerankError,
            lambda: get_reranker(_Settings("colbert")),
            "registry: an unknown reranker raises instead of falling back",
        )
        h.raises(
            RerankError,
            lambda: get_reranker(_Settings("llm")),
            "registry: the llm backend refuses to build without its scoring callable",
        )
        h.equal(
            get_reranker(_Settings("llm"), score_fn=lambda prompt: '{"1": 9, "2": 1}').name,
            "llm",
            "registry: the llm backend builds when the callable is injected",
        )

        h.group("LLM reranker contract")
        from kip.core.rerank.llm import LlmReranker, build_prompt, parse_scores

        prompt = build_prompt("water activity", candidates[:2], max_chars=200)
        for candidate in candidates[:2]:
            h.ok(
                candidate.id not in prompt,
                f"llm rerank: chunk id {candidate.id!r} is not sent to the model",
            )
        h.ok(len(prompt) < 4000, "llm rerank: the prompt is bounded, not a whole document")

        h.equal(parse_scores("garbage", 3), {}, "llm rerank: unparseable output yields no scores")
        fragmentary = LlmReranker(score_fn=lambda p: '{"1": 9}').rerank("q", candidates[:4])
        h.equal(
            [item.id for item in fragmentary],
            [candidate.id for candidate in candidates[:4]],
            "llm rerank: a response too incomplete to trust preserves retrieval order",
        )
        h.raises(
            RerankError,
            lambda: LlmReranker(score_fn=_boom).rerank("q", candidates[:2]),
            "llm rerank: a provider failure surfaces as an error, not a silent no-op",
        )

        h.group("Cross-encoder reranker contract")
        from kip.core.rerank.cross_encoder import CrossEncoderReranker

        lazy = CrossEncoderReranker(model="stub")
        h.ok(not lazy.loaded, "cross-encoder: no model is loaded at construction")
        injected = CrossEncoderReranker(model="stub", client=_Counter())
        ranked = injected.rerank("water activity", candidates[:3])
        h.equal(len(ranked), 3, "cross-encoder: every candidate is scored")
        h.ok(
            all(
                left.score >= right.score
                for left, right in zip(ranked, ranked[1:])
            ),
            "cross-encoder: output is sorted by descending score",
        )
        h.ok(not injected.calibrated, "cross-encoder: scores are declared uncalibrated")
    finally:
        store.close()


class _Settings:
    """Minimal settings stand-in for the registry checks."""

    def __init__(self, reranker: str) -> None:
        self.reranker = reranker
        self.reranker_model = "stub-model"


class _Counter:
    """Cross-encoder stand-in: counts query occurrences. No model, no download."""

    def predict(self, pairs: Any, **_: Any) -> list[float]:
        return [float(passage.lower().count(query.lower())) for query, passage in pairs]


def _boom(_prompt: str) -> str:
    raise TimeoutError("upstream timeout")


def check_sqlite_capabilities(h: Harness) -> None:
    """Record what this SQLite build can do, so a failure elsewhere is explicable."""
    h.group("SQLite capabilities")
    h.ok(sqlite3.sqlite_version_info >= (3, 9), f"sqlite: version {sqlite3.sqlite_version} supports FTS5 era features")
    try:
        from kip.core.retrieval.fts import fts5_available

        available = fts5_available()
    except Exception:  # pragma: no cover - defensive
        available = False
    # Not an assertion that FTS5 exists: KEYWORD_INDEX=bm25 is a supported
    # configuration precisely because some builds lack it. This records the fact.
    h.ok(True, f"sqlite: FTS5 {'available' if available else 'unavailable (KEYWORD_INDEX=bm25 required)'}")


def run(verbose: bool = False) -> Harness:
    h = Harness(name="retrieval", verbose=verbose)
    with tempfile.TemporaryDirectory(prefix="kip-retrieval-check-") as raw:
        tmp = Path(raw)
        check_sqlite_capabilities(h)
        embedder = check_embeddings(h)
        check_vector_store_parity(h, embedder, tmp)
        check_keyword_agreement(h, tmp)
        check_fusion(h)
        check_hybrid(h, embedder, tmp)
        check_rerank(h, embedder, tmp)
    return h


def main(argv: list[str] | None = None) -> int:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    harness = run(verbose="-v" in args or "--verbose" in args)
    print(harness.report())
    return 0 if harness.succeeded else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
