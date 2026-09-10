# Backend Database & API Schema — Agentic Multimodal Research Platform

This document details the database models, vector schemas, and API contracts, including recent enhancements from **Phase 8A (Model Routing)** and **Phase 8B (Usage & Quotas)**.

---

## 1. Database Schema

### 1.1 `users` & `user_quotas` (Phase 8B)
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(32) DEFAULT 'researcher' NOT NULL, -- admin, researcher, viewer
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE user_quotas (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE UNIQUE NOT NULL,
    lifetime_token_limit BIGINT NULL, -- NULL indicates unlimited
    lifetime_cost_limit NUMERIC(10, 4) NULL,
    tokens_used BIGINT DEFAULT 0 NOT NULL,
    cost_used NUMERIC(10, 4) DEFAULT 0 NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 1.2 `usage_records` (Phase 8B)
```sql
CREATE TABLE usage_records (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL,
    provider VARCHAR(64) NOT NULL, -- gemini, ollama, openai, anthropic
    model VARCHAR(128) NOT NULL,
    task_type VARCHAR(64) NOT NULL, -- planning, web_research, doc_research, critique, synthesis
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    estimated_cost NUMERIC(10, 6) DEFAULT 0 NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);
CREATE INDEX ix_usage_records_user_timestamp ON usage_records(user_id, timestamp);
```

### 1.3 `research_jobs`, `tasks`, `evidence` & `reports`
- **`research_jobs`**: `id` (UUID PK), `user_id` (FK), `query`, `status` (`pending`, `planning`, `in_progress`, `completed`, `failed`), `progress_pct`, `created_at`, `completed_at`.
- **`research_tasks`**: `id` (UUID PK), `job_id` (FK), `title`, `agent_type` (`planner`, `web`, `document`, `critic`, `synthesis`), `status`, `dependencies` (JSON), `output` (JSON).
- **`research_evidence`**: `id` (UUID PK), `job_id` (FK), `source_type` (`document`, `web`), `source_url` / `document_id`, `chunk_id`, `page_number`, `claim_text`, `snippet`, `confidence_score`.
- **`research_reports`**: `id` (UUID PK), `job_id` (FK), `executive_summary`, `key_findings` (JSON), `evidence_matrix` (JSON), `contradictions` (JSON), `analysis` (TEXT), `full_markdown` (TEXT), `confidence_pct`, `created_at`.

### 1.4 `documents` & `chunks`
- **`documents`**: `id` (PK), `owner_id` (FK), `filename`, `safe_name`, `content_type`, `size_bytes`, `page_count`, `chunk_count`, `status`.
- **`chunks`**: `id` (PK), `document_id` (FK), `chunk_index`, `body`, `embed_text`, `token_count`, `page_start`, `page_end`, `section_key`, `heading`, `kinds`.

---

## 2. Key API Endpoints & Contracts

### 2.1 Research Endpoints (`/api/v1/research`)
- `POST /`: Submit research job. Payload: `{"query": "...", "depth": "deep", "sources": ["web", "knowledge"]}`. Returns `{"job_id": "...", "status": "pending"}`.
- `GET /{job_id}`: Retrieve job status, task progress, and generated report.
- `GET /{job_id}/ws`: Authenticated WebSocket stream emitting live DAG task transitions and token streaming.

### 2.2 Model Gateway & Usage Endpoints (`/api/v1/ai`)
- `GET /models`: Returns list of registered models from `ModelRegistry` with capabilities.
- `GET /usage`: Returns aggregated token consumption and quota remaining for the current authenticated user.
