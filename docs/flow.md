# System & Data Flows — Agentic Multimodal Research Platform

This document illustrates the execution lifecycles, agentic loops, and model routing pipelines.

---

## 1. End-to-End Autonomous Research Flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Web as Web Client / React
    participant API as FastAPI Gateway
    participant Orch as Agent Orchestrator
    participant Planner as Planner Agent
    participant Router as Model Router & Gateway
    participant WebAgent as Web Research Agent
    participant DocAgent as Document Research Agent
    participant Critic as Critic Agent
    participant Synth as Synthesis Agent
    participant DB as Postgres / Usage Ledger

    User->>Web: Submit Research Question
    Web->>API: POST /api/v1/research (query)
    API->>DB: Check & Row-Lock UserQuota
    API->>DB: Create ResearchJob (status="planning")
    API-->>Web: 202 Accepted (job_id)
    Web->>API: Connect WebSocket (/api/v1/research/{job_id}/ws)

    API->>Orch: Spawn ResearchPipeline
    Orch-->>Web: WS Event: "Planning research..."

    Orch->>Planner: Formulate research plan & subquestions
    Planner->>Router: Request Reasoning Model
    Router-->>Planner: Decomposed plan + strategy
    Planner-->>Orch: Task Graph (Web + Doc tasks)

    par Parallel Investigation
        Orch->>WebAgent: Execute Web Searches
        WebAgent->>Router: Request Fast/Cheap Model
        WebAgent-->>Orch: External Web Evidence
    and
        Orch->>DocAgent: Query Knowledge Layer (Hybrid RRF)
        DocAgent-->>Orch: Retrieved Internal Document Chunks
    end

    Orch-->>Web: WS Event: "Cross-checking & critiquing evidence..."
    Orch->>Critic: Evaluate Evidence & Contradictions
    Critic->>Router: Request Reasoning Model
    
    alt Insufficient Evidence or Contradictions
        Critic-->>Orch: Trigger iterative loop with refined queries
        Orch->>WebAgent: Additional focused search
    else Evidence Confirmed
        Critic-->>Orch: Verified Evidence Matrix
    end

    Orch-->>Web: WS Event: "Synthesizing research report..."
    Orch->>Synth: Synthesize Final Report with Citations
    Synth->>Router: Request Large-Context Synthesis Model
    Router-->>Synth: Final Multi-Section Report Markdown
    Synth->>DB: Persist Report & Record Token Usage
    Orch-->>Web: WS Event: "Completed" (Final Report Data)
```

---

## 2. Model Routing & Quota Enforcement Flow (Phase 8A & 8B)

```mermaid
flowchart TD
    TaskIn["Incoming Agent Task (e.g. Synthesis)"] --> Gateway["ModelGateway.call()"]
    Gateway --> Attrib["Attach Authenticated user_id from AgentContext"]
    Attrib --> LockQuota["Row-Lock UserQuota (SELECT ... FOR UPDATE)"]
    
    LockQuota --> CheckQuota{"Is Quota Available?"}
    CheckQuota -- No --> FallbackQuota["Route to Local/Free Provider or Refuse"]
    CheckQuota -- Yes --> MatchModel["ModelRouter: Match Task Requirements"]
    
    MatchModel --> Registry["ModelRegistry Lookup (Capabilities, Context, Cost)"]
    Registry --> Exec["Invoke Model Provider API (Gemini / Ollama / OpenAI)"]
    
    Exec --> Telemetry["Record UsageRecord (tokens, cost, latency)"]
    Telemetry --> Deduct["Increment tokens_used & cost_used in DB"]
    Deduct --> ReturnResult["Return Output to Calling Agent"]
```

---

## 3. Knowledge Automation Ingestion Lifecycle (Phase 9 Target)

```mermaid
flowchart LR
    Upload["1. Upload Doc / PDF / Image"] --> Validate["2. Validate & Sanitize"]
    Validate --> Extract["3. Extract Hierarchy & Tables"]
    Extract --> Normalize["4. Normalize Text & Blocks"]
    Normalize --> Chunk["5. Structure-Aware Chunking"]
    Chunk --> Embed["6. Compute Embeddings"]
    Embed --> Index["7. Upsert to Vector Store & BM25"]
    Index --> Ready["8. Knowledge Ready for Agents"]
```
