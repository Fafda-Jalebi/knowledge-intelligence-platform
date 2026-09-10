# AGENTS.md — Agent & Developer Operating Instructions

This document is the authoritative guide for AI coding assistants and developers contributing to the **Agentic Multimodal Research Platform** (AI Research OS).

---

## 1. Project Identity & Philosophy

- **Platform Name**: Agentic Multimodal Research Platform (AI Research OS)
- **Core Premise**: This is **NOT** a simple LLM wrapper or generic chatbot. It is a multi-agent autonomous research engine executing:
  $$\text{Question} \longrightarrow \text{Planner} \longrightarrow \text{Research Strategy} \longrightarrow \begin{bmatrix} \text{Web Research} \\ \text{Knowledge / RAG} \end{bmatrix} \longrightarrow \text{Evidence} \longrightarrow \text{Critic} \longrightarrow \text{Synthesis} \longrightarrow \text{Report}$$
- **Current Git Branch**: `develop/v1.1`
- **Current State**: Completed Phase 1 through Phase 8B (`a603114 feat(ai): add persistent usage and quota tracking`). The immediate active focus is **Phase 9: Intelligent Knowledge Automation**.

---

## 2. Completed Phases (Do Not Break or Re-Implement)

1. **Phase 1 (Foundation)**: FastAPI async app, React/Vite/TS shell, DB session, shared modules.
2. **Phase 2 (Research MVP)**: DAG task runner, Planner/Synthesis agent pipeline, WebSocket streaming.
3. **Phase 3 (Multimodal Ingestion)**: Ingestion for PDF (with tables), DOCX, TXT, MD, and image vision.
4. **Phase 4 (Agentic System)**: Tool registry, SSRF-hardened web fetch, search tools, Critic agent, execution tracing.
5. **Phase 5 (RAG / Knowledge Layer)**: Provider-agnostic embedders, ChromaDB / SQLite vector stores, BM25 sparse index, Reciprocal Rank Fusion (RRF).
6. **Phase 6 (Production & Security)**: JWT auth, RBAC (Admin, Researcher, Viewer), security filters, Prometheus metrics.
7. **Phase 7 (Application Maturity)**: Dashboard research job mapping, persistent users, official Gemini provider alignment.
8. **Phase 8A (Intelligent Model Routing)**: `ModelRegistry` → `ModelRouter` → `ModelGateway` → `Providers`.
9. **Phase 8B (Usage & Quotas)**: Persistent `UsageRecord`, `UserQuota`, concurrency row locking, user attribution carrying `user_id` through the full pipeline (`JWT → API → Pipeline → Orchestrator → Gateway → UsageRepository`).

---

## 3. Architecture & Subsystem Invariants

### 3.1 Model Routing & Gateway
- All LLM calls MUST route through `ModelGateway` / `ModelRouter` — never invoke provider APIs directly from business logic.
- Model selection dynamically adheres to task suitability:
  - Quick classification / extraction $\rightarrow$ Fast/cheap model
  - Deep reasoning & planning $\rightarrow$ High-reasoning model
  - Image analysis $\rightarrow$ Vision-capable model
  - Synthesis & reporting $\rightarrow$ Large-context model

### 3.2 Usage, Quotas & Concurrency
- All model calls must record input/output tokens and cost estimates in `UsageRecord`.
- Quota checks MUST utilize transactional row locking (`SELECT ... FOR UPDATE` in PostgreSQL / lock equivalents in SQLite) to prevent race-condition oversubscription under concurrent worker loads.

### 3.3 Evidence Gating & Citations
- Claims must be grounded with numeric citations (`[1]`, `[2]`).
- The `Critic` agent and grounding gates verify source text support. Fabricated citations must be removed.

---

## 4. Coding Conventions

- **Python**: Strict type hints (PEP 484/585), async/await everywhere for I/O operations, SQLAlchemy 2.0 async select statements, Pydantic v2 schemas.
- **Frontend**: React 18, TypeScript, TailwindCSS + CSS custom properties, Lucide icons, responsive layout.
- **Testing**: Pytest with `pytest-asyncio` for backend; ensure 100% passing test status before submitting changes.

---

## 5. Standard Commands

```bash
# Backend test execution
pytest backend/tests/ -v

# Backend server run
uvicorn kip.api:app --reload --host 0.0.0.0 --port 8000

# Frontend development server
cd frontend && npm run dev
```
