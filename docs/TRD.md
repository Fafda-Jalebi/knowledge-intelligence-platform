# Technical Requirements Document (TRD) — Agentic Multimodal Research Platform

## 1. System Engineering Architecture

The platform is designed around a multi-agent orchestrated research pipeline operating over a layered AI infrastructure and hybrid knowledge subsystem.

```
┌─────────────────┐       REST / WebSocket       ┌─────────────────────────────────────┐
│  React 18 SPA   │ ◄──────────────────────────► │    FastAPI Application Gateway      │
└─────────────────┘                              └──────────────────┬──────────────────┘
                                                                    │
          ┌───────────────────────────────────┬─────────────────────┴─────────────────────┐
          ▼                                   ▼                                           ▼
┌──────────────────┐               ┌───────────────────────┐                  ┌───────────────────────┐
│ Database Layer   │               │ Research Orchestrator │                  │ AI Model Gateway      │
│ (PostgreSQL/SQLi)│               │ (DAG Execution & WS)  │                  │ (Routing & Telemetry) │
├──────────────────┤               ├───────────────────────┤                  ├───────────────────────┤
│• Users & Quotas  │               │• Planner Agent        │                  │• ModelRegistry        │
│• UsageRecords    │               │• Web Research Agent   │                  │• ModelRouter          │
│• Documents/Chunks│               │• Doc Research Agent   │                  │• ModelGateway         │
│• Research Jobs   │               │• Critic Agent         │                  │• Gemini / Ollama /    │
│• Reports & Traces│               │• Synthesis Agent      │                  │  OpenAI / Anthropic   │
└──────────────────┘               └───────────────────────┘                  └───────────────────────┘
```

---

## 2. Technical Stack & Invariants

| Layer | Technologies | Architectural Invariants |
|---|---|---|
| **API Server** | FastAPI (Python 3.10+), Uvicorn, WebSockets | Async execution, authenticated WebSocket endpoints for real-time research streaming. |
| **Relational DB** | PostgreSQL (Prod) / SQLite (Dev), SQLAlchemy 2.0 Async | Strict foreign-key cascades, transactional row-locking (`SELECT FOR UPDATE`) for quota checks. |
| **Vector Index** | ChromaDB / SQLite Vector / Qdrant | Dense cosine similarity vector search alongside BM25 sparse index. |
| **Orchestration** | Python Async DAG / Task Graph | Deterministic agent execution tracing, memory persistence in `agent_runs` and `model_calls`. |
| **AI Routing** | `ModelRegistry`, `ModelRouter`, `ModelGateway` | Model-agnostic routing dynamically matching tasks to capabilities (Fast, Reasoning, Vision, Large-Context). |
| **Security** | JWT (Access + Refresh), RBAC, SSRF Protection | Private IP/multicast rejection in web tools, safe filename sanitization, prompt injection defenses. |

---

## 3. Subsystem Technical Specifications

### 3.1 Model Routing & Gateway Architecture (Phase 8A)
- **`ModelRegistry`**: Maintains registry of available models with metadata: provider, capabilities, context window, output limit, vision support, streaming support, task suitability, priority, cost per token.
- **`ModelRouter`**: Maps incoming task types (`planning`, `web_search`, `image_analysis`, `critique`, `synthesis`) to the optimal model candidate based on rules, quota availability, and latency budgets.
- **`ModelGateway`**: Central execution gateway handling model invocation, telemetry recording, automatic retry/fallback, and error normalization.

### 3.2 Concurrency-Safe Usage & Quota Engine (Phase 8B)
- **Data Models**:
  - `UsageRecord`: `id`, `user_id`, `provider`, `model`, `input_tokens`, `output_tokens`, `total_tokens`, `estimated_cost`, `timestamp`.
  - `UserQuota`: `user_id`, `lifetime_token_limit`, `lifetime_cost_limit`, `tokens_used`, `cost_used`.
- **Transactional Row Locking**:
  ```python
  # Row-level lock ensures no oversubscription under concurrent worker queries
  stmt = select(UserQuota).where(UserQuota.user_id == user_id).with_for_update()
  quota = await session.execute(stmt).scalar_one_or_none()
  if quota and quota.is_exceeded(requested_tokens):
      raise QuotaExceededError("User quota exhausted.")
  ```

### 3.3 Hybrid Retrieval & RRF Math (Phase 5)
Candidates from dense vector search and sparse BM25 are fused using Reciprocal Rank Fusion:
$$RRF\_Score(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{k + \text{rank}_m(d)}, \quad k = 60$$

### 3.4 Multi-Agent State Machine & WebSocket Streaming (Phase 2 & 7)
During execution, the agent orchestrator streams state updates over `/api/v1/research/{job_id}/ws`:
`research_started` $\rightarrow$ `planning` $\rightarrow$ `searching` $\rightarrow$ `analyzing` $\rightarrow$ `critiquing` $\rightarrow$ `synthesizing` $\rightarrow$ `completed`.
