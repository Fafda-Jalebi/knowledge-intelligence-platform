# System Architecture — Agentic Multimodal Research Platform

This document details the architectural topology, subsystem boundaries, data contracts, and long-term design of the **Agentic Multimodal Research Platform** (AI Research OS).

---

## 1. Complete Architecture Diagram

```mermaid
flowchart TB
    subgraph Client["Frontend Client (React 18 + TS + Vite)"]
        UI["Web Platform (Dashboard, Research Workspace, Documents, Settings)"]
        WSClient["WebSocket Client (Live Progress Streaming)"]
    end

    subgraph APILayer["FastAPI Gateway"]
        AuthRouter["/auth (JWT & RBAC)"]
        ResearchRouter["/research (Jobs, Execution & Streaming)"]
        DocumentsRouter["/documents (Multimodal Ingestion)"]
        KnowledgeRouter["/knowledge (Hybrid Search)"]
        MetricsRouter["/metrics (Prometheus)"]
    end

    subgraph CorePlatform["Platform & Security Core"]
        AuthService["Auth & Token Lifecycle"]
        UsageQuotas["Usage Tracking & Row-Locked Quotas"]
        DB[(PostgreSQL / SQLite Database)]
    end

    subgraph AgenticEngine["Agentic Research Engine"]
        Orchestrator["Agent Orchestrator (DAG Pipeline)"]
        Planner["Planner Agent (Subquestion Decomposition)"]
        WebAgent["Web Research Agent (SSRF Protected)"]
        DocAgent["Document Research Agent"]
        Critic["Critic Agent (Evidence & Contradiction Check)"]
        Synthesis["Synthesis & Report Agent"]
    end

    subgraph KnowledgeLayer["Multimodal Knowledge & RAG Layer"]
        Ingestion["Parsers: PDF (Tables), DOCX, TXT, MD, Images"]
        Chunker["Structure-Aware Chunking"]
        Retriever["Hybrid Retriever (Dense + BM25 Lexical)"]
        RRF["Reciprocal Rank Fusion (RRF)"]
        VectorDB[(ChromaDB / SQLite Vector)]
    end

    subgraph ModelInfrastructure["AI Infrastructure Layer (Phase 8A)"]
        ModelRegistry["ModelRegistry (Metadata & Capabilities)"]
        ModelRouter["ModelRouter (Task-to-Model Matching)"]
        ModelGateway["ModelGateway (Fallback, Retry & Telemetry)"]
        Providers["Providers (Gemini, Ollama, OpenAI, Anthropic)"]
    end

    UI --> APILayer
    WSClient <-->|Live Stream| ResearchRouter

    APILayer --> CorePlatform
    APILayer --> AgenticEngine
    APILayer --> KnowledgeLayer

    AgenticEngine --> Orchestrator
    Orchestrator --> Planner
    Planner --> WebAgent & DocAgent
    DocAgent --> KnowledgeLayer
    WebAgent & DocAgent --> Critic
    Critic --> Synthesis

    AgenticEngine --> ModelGateway
    ModelGateway --> ModelRouter
    ModelRouter --> ModelRegistry
    ModelGateway --> Providers
    ModelGateway --> UsageQuotas
```

---

## 2. Subsystems & Component Responsibilities

### 2.1 Agentic Research Pipeline
1. **Planner Agent**: Decomposes broad questions into manageable sub-investigations, scopes required evidence sources, and constructs DAG tasks.
2. **Web Research Agent**: Queries external web search APIs, safely fetches live pages (with comprehensive SSRF defenses blocking private/loopback/multicast IPv4/IPv6 addresses), and summarizes relevant text.
3. **Document Research Agent**: Interfaces directly with the knowledge retrieval layer to pull internal files and user evidence.
4. **Critic Agent**: Reviews incoming evidence from web and document agents. Identifies knowledge gaps, ungrounded claims, and contradictions. Re-triggers research tasks if confidence is below threshold.
5. **Synthesis & Report Agent**: Aggregates verified evidence into structured executive summaries, findings tables, methodology notes, and citation maps.

### 2.2 Intelligent Model Infrastructure (Phase 8A & 8B)
- **Model Registry**: Centralized repository of model configurations, rate limits, token costs, context capacities, and vision support.
- **Model Router**: Intelligently matches incoming tasks to the best-suited model.
- **Model Gateway**: Handles provider communication, error translation, fallback chains, and usage telemetry.
- **Usage & Quotas**: Real-time token consumption ledger with concurrency-safe row locking.

---

## 3. Long-Term Architectural Target (AI Research OS)

```
                            ┌───────────────────────────────────┐
                            │          AI RESEARCH OS           │
                            └─────────────────┬─────────────────┘
                                              │
        ┌─────────────────────────────────────┼─────────────────────────────────────┐
        ▼                                     ▼                                     ▼
┌────────────────┐                   ┌────────────────┐                    ┌────────────────┐
│Research Engine │                   │Knowledge Engine│                    │ Data Analysis  │
└───────┬────────┘                   └───────┬────────┘                    └───────┬────────┘
        │                                    │                                     │
        └────────────────────────────────────┼─────────────────────────────────────┘
                                             ▼
                                     ┌────────────────┐
                                     │  Agent System  │
                                     │ (Plan/Web/Doc/ │
                                     │ Critic/Synth)  │
                                     └───────┬────────┘
                                             ▼
                                     ┌────────────────┐
                                     │Evidence-Backed │
                                     │ Structured Rpt │
                                     └────────────────┘
```
