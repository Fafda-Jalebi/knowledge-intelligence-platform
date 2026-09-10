# Architectural Decisions & ADR Log — Agentic Multimodal Research Platform

This document logs the key architectural and design decisions established across all phases of the platform.

---

## Architecture Decision Records Summary

| ADR # | Title | Phase | Status | Primary Rationale & Outcome |
|---|---|---|---|---|
| **0001** | Zero-Dependency Fallback Core | Phase 1 | **Accepted** | Provide out-of-the-box offline operation with hashing embeddings and extractive generator so setup never blocks on external API keys. |
| **0002** | Multi-Agent Orchestration over Single LLM | Phase 2 & 4 | **Accepted** | Research requires distinct subtasks (`Plan → Investigate → Critic → Synthesize`). Dedicated agents with discrete prompts outperform single monolithic calls. |
| **0003** | Multimodal Document Extraction Pipeline | Phase 3 | **Accepted** | Standardize PDF (tables via `pdfplumber`), DOCX, Markdown, and Images into normalized block structures with structural metadata. |
| **0004** | Hybrid Retrieval with Reciprocal Rank Fusion | Phase 5 | **Accepted** | Combine dense vector proximity with BM25 keyword matching using RRF ($k=60$) to eliminate terminology mismatch errors. |
| **0005** | Strict Citation Integrity & Grounding Gating | Phase 5 | **Accepted** | Every statement in the research output must cite source evidence `[1]`, `[2]`. Invented citations are stripped; unsupported conclusions trigger critic review. |
| **0006** | Official Gemini Provider Integration Cleanup | Phase 7 | **Accepted** | Replaced fragile Web2API scrapers with direct official Google Gemini SDK integration via standard Model Gateway interfaces. |
| **0007** | Intelligent Capability-Based Model Routing | Phase 8A | **Accepted** | Decoupled agents from specific LLM providers. `ModelRouter` inspects task requirements (fast, reasoning, vision, large-context) and routes dynamically. |
| **0008** | Persistent Usage Tracking & Concurrency Row Locking | Phase 8B | **Accepted** | User attribution carried through pipeline (`JWT → AgentContext → ModelGateway → UsageRepository`). Row-level locking on `UserQuota` prevents race-condition oversubscription. |

---

## Architectural Principles

1. **Agent Specialization**: Separation of planning, gathering, critiquing, and writing into independent roles with distinct model profiles.
2. **SSRF & Network Safety**: All external tool interactions strictly validate URLs, resolving and rejecting loopback, link-local, private RFC 1918, and multicast addresses.
3. **Traceability**: Every claim maps to exact evidence chunks, document page numbers, or web URLs with full confidence scoring.
