# Changelog — Agentic Multimodal Research Platform

All major version milestones and feature completions are documented in this file.

---

## [Phase 8B] - `a603114` - Persistent Usage & Quotas (Branch: `develop/v1.1`)
### Added
- Persistent `UsageRecord` database model tracking user ID, provider, model, task type, token usage, estimated cost, and timestamps.
- `UserQuota` model supporting configurable lifetime token and cost limits.
- Transactional row-level locking (`SELECT ... FOR UPDATE`) in quota deduction to prevent oversubscription under high concurrency.
- Pipeline-wide user attribution propagating authenticated user ID from JWT through `ResearchPipeline`, `AgentOrchestrator`, `AgentContext`, and `ModelGateway`.
- Quota-aware fallback routing.

---

## [Phase 8A] - `88ac57d` - Intelligent Model Routing
### Added
- `ModelRegistry`: Centralized metadata registry tracking model capabilities, token limits, vision support, streaming support, and costs.
- `ModelRouter`: Dynamic task-to-model dispatcher matching fast/cheap models, deep reasoning models, vision models, and large-context synthesis models.
- `ModelGateway`: Unified execution gateway with automatic retry, provider fallback chains, and telemetry recording.

---

## [Phase 7] - Application Maturity & Real-Time WebSockets
### Added
- Authenticated real-time WebSocket progress streaming for multi-agent DAG task executions.
- Persistent user authentication lifecycle with JWT access and refresh tokens.
- Official Gemini provider integration cleanup replacing legacy Web2API layer.
- Frontend research dashboard job mapping and status synchronization.

---

## [Phase 6] - Production Readiness & Security
### Added
- Role-Based Access Control (RBAC with Admin, Researcher, Viewer roles).
- Security sanitizers: prompt injection defenses and safe file upload name validation.
- Prometheus `/metrics` exposition endpoint.
- Containerization and Kubernetes deployment manifests.

---

## [Phase 5] - Multimodal RAG & Knowledge Layer
### Added
- Hybrid retriever combining dense vector search and BM25 sparse lexical search via Reciprocal Rank Fusion (RRF).
- Vector store adapters for ChromaDB, SQLite Vector, and In-Memory.
- Citation integrity and evidence grounding gatekeeper.

---

## [Phase 4] - Agentic Core & Tool Framework
### Added
- Tool registry with SSRF-hardened `WebFetchTool`, `WebSearchTool`, and `DocumentReadTool`.
- Critic agent for adversarial evidence evaluation and contradiction detection.
- Agent execution tracing and short/long-term memory in `agent_runs` and `model_calls`.

---

## [Phase 1–3] - Foundation, MVP & Multimodal Ingestion
### Added
- Core FastAPI and React 18 / TypeScript SPA shell.
- Extractor engines for PDF (with table extraction), DOCX, Plain Text, and Markdown.
- Vision model gateway for image understanding and analysis.
