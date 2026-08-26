# Knowledge Intelligence Platform (KIP)

A production-ready, open-source Knowledge Intelligence Platform for document ingestion, semantic search, and grounded question-answering with citations.

## Features

- **Document Ingestion**: PDF, DOCX, TXT, Markdown support with structure-aware chunking
- **Hybrid Retrieval**: Dense (semantic) + sparse (keyword) search with Reciprocal Rank Fusion
- **Reranking**: Heuristic, cross-encoder, and LLM-based rerankers
- **Grounded Generation**: Extractive (default) and generative LLM backends with citation enforcement
- **Evidence Gates**: Pre-generation similarity threshold and post-generation support checking
- **Citation Integrity**: Every answer traces to source passages; invented markers are removed
- **Multi-document Reasoning**: Answers can synthesize across multiple documents
- **Conversation History**: Persistent chat sessions with context
- **Authentication**: JWT-based auth with registration, login, password change
- **Zero-dependency Default**: Runs entirely offline with hashing embeddings and extractive generation
- **Production Ready**: PostgreSQL, Qdrant, Docker Compose, comprehensive test suite

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Or Python 3.10+ and Node.js 20+ for local development

### With Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd knowledge-intelligence-platform

# Copy environment file and customize
cp .env.example .env
# Edit .env - at minimum set JWT_SECRET (generate: python -c "import secrets;print(secrets.token_urlsafe(48))")

# Start all services
docker compose up -d

# Access the application
# Frontend: http://localhost:5173
# API Docs: http://localhost:8000/docs
```

### Local Development

#### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .[dev,embeddings]

# Set environment variables
cp ../.env.example .env
# Edit .env

# Database tables are created at application startup from the SQLAlchemy models.
# This project does not currently ship Alembic migration revisions.

# Start the API server
uvicorn kip.api:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React + Vite)                  │
│  Dashboard │ Documents │ Chat │ Settings                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API (JWT Auth)
┌──────────────────────────▼──────────────────────────────────────┐
│                      FastAPI Backend                            │
│  Auth │ Documents │ Chat │ Settings │ Health                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Service Layer
┌──────────────────────────▼──────────────────────────────────────┐
│                     RAG Pipeline (kip.core)                     │
│  Retrieve → Gate → Hydrate → Rerank → Context → Generate → Verify│
└──────────────────────────┬──────────────────────────────────────┘
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌────────────────┐ ┌───────────────┐ ┌────────────────┐
│ Vector Store   │ │ Keyword Index │ │ Relational DB  │
│ (Qdrant/SQLite)│ │ (FTS5/BM25)   │ │ (PostgreSQL/   │
│                │ │               │ │  SQLite)       │
└────────────────┘ └───────────────┘ └────────────────┘
```

## Configuration

All configuration is via environment variables (`.env` file). Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | `development` or `production` |
| `DATABASE_URL` | `sqlite:///./var/kip.sqlite3` | SQLite or PostgreSQL |
| `VECTOR_STORE` | `sqlite` | `memory`, `sqlite`, `qdrant` |
| `EMBEDDING_PROVIDER` | `hashing` | `hashing`, `sentence-transformers`, `openai`, `ollama` |
| `LLM_PROVIDER` | `extractive` | `extractive`, `openai`, `anthropic`, `gemini`, `ollama` |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid`, `dense`, `keyword` |
| `RERANKER` | `heuristic` | `heuristic`, `cross-encoder`, `llm`, `none` |
| `JWT_SECRET` | (auto-generated in dev) | **Required in production** |

See `.env.example` for all options.

## Default Providers (Zero-Setup)

| Component | Default | Description |
|-----------|---------|-------------|
| Embeddings | Hashing | Deterministic lexical embeddings, no download, offline |
| LLM | Extractive | Quotes matching sentences, grounded by construction |
| Vector Store | SQLite | Persistent single-file index, exact search |
| Keyword Index | SQLite FTS5 | Persistent, shared across workers |
| Reranker | Heuristic | Lexical coverage/proximity, no dependencies |

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login, returns JWT
- `GET /api/auth/me` - Get current user
- `POST /api/auth/change-password` - Change password

### Documents
- `POST /api/documents/upload` - Upload and ingest document
- `GET /api/documents` - List documents (paginated)
- `GET /api/documents/{id}` - Get document details
- `GET /api/documents/{id}/chunks` - List document chunks
- `DELETE /api/documents/{id}` - Delete document

### Chat
- `POST /api/chat/ask` - Ask a question
- `GET /api/chat/conversations` - List conversations
- `GET /api/chat/conversations/{id}` - Get conversation with messages
- `DELETE /api/chat/conversations/{id}` - Delete conversation
- `PATCH /api/chat/conversations/{id}` - Update conversation title

### Settings
- `GET /api/settings` - Get current settings
- `GET /api/settings/embedding-providers` - Available embedding providers
- `GET /api/settings/llm-providers` - Available LLM providers
- `GET /api/settings/rerankers` - Available rerankers
- `GET /api/settings/vector-stores` - Available vector stores
- `GET /api/settings/keyword-indexes` - Available keyword indexes
- `GET /api/settings/grounding` - Grounding thresholds

## Running Tests

```bash
# Backend self-checks (zero-dependency)
cd backend
python -m selfcheck

# Backend pytest tests
pytest tests/ -v

# Frontend tests
cd frontend
npm test
```

## Evaluation

The platform includes a comprehensive evaluation harness that measures retrieval and generation quality on a labeled dataset.

```bash
# Run evaluation on the built-in demo corpus
cd backend
python -m kip.eval
```

The evaluation uses the demo corpus (`data/demo_corpus/`) with 20 questions (17 answerable, 3 unanswerable) covering FoodTech, Civil Engineering, Packaging, and Archival domains.

### Actual Results (Zero-Dependency Stack: HashingEmbedder + ExtractiveLLM)

| Metric | Value |
|--------|-------|
| **Retrieval** | |
| Recall@1 | 0.221 |
| Recall@3 | 0.438 |
| Recall@5 | 0.577 |
| Recall@10 | 0.749 |
| MRR | 0.922 |
| **Generation** (when system answers) | |
| Groundedness | 1.000 |
| Citation Coverage | 1.000 |
| Citation Correctness | 1.000 |
| **Refusal** | |
| Refusal Accuracy (unanswerable) | 1.000 |
| Answer Rate (answerable) | 0.294 |
| **Latency** | ~55 ms total |

### Methodology & Limitations

**Retrieval Evaluation**: Uses the exact production `HybridRetriever` (RRF fusion of dense + BM25). Ground truth is document-level: a retrieved chunk is "relevant" if it comes from the document that contains the answer.

**Generation Evaluation**: Uses the full production `RagPipeline` with `ExtractiveClient`. Metrics are computed on questions the system *chooses to answer* (non-refused):
- **Groundedness**: Fraction of answer claims supported by cited passages (extractive LLM quotes verbatim → 1.0 when it answers)
- **Citation Coverage**: Fraction of answer sentences with citations (extractive LLM always cites → 1.0 when it answers)
- **Citation Correctness**: Whether citations reference real retrieved passages (1.0)
- **Refusal Accuracy**: Fraction of unanswerable questions correctly refused (1.0)
- **Answer Rate**: Fraction of answerable questions the system actually answers (0.294)

**Known Limitations of Zero-Dependency Stack**:
- **HashingEmbedder** provides no semantic understanding - dense retrieval is essentially random. The strong Recall@10 (0.749) comes primarily from BM25 keyword matching.
- **ExtractiveClient** can only quote verbatim from retrieved passages. It cannot synthesize, paraphrase, or infer. Questions requiring inference (e.g., "What temperature gives best balance?") fail even when the answer is in the text.
- **Answer Rate (0.294)** reflects this: only 5 of 17 answerable questions are successfully answered.
- For production use, configure `EMBEDDING_PROVIDER=sentence-transformers` and `LLM_PROVIDER=openai` (or similar) for dramatically better retrieval and generation.

**Evaluation Dataset**: `data/eval_dataset.jsonl` - 20 QA pairs derived from the demo corpus. Extend this file for domain-specific evaluation.

## Project Structure

```
knowledge-intelligence-platform/
├── backend/
│   ├── kip/
│   │   ├── api/              # FastAPI routers
│   │   ├── config.py         # Configuration
│   │   ├── db/               # SQLAlchemy models & repositories
│   │   ├── security/         # Passwords, JWT, file validation
│   │   ├── services/         # Business logic
│   │   └── core/             # RAG engine (zero-dep)
│   │       ├── embeddings/   # Embedding providers
│   │       ├── vectorstore/  # Vector store backends
│   │       ├── retrieval/    # Hybrid retrieval
│   │       ├── rerank/       # Rerankers
│   │       ├── rag/          # Pipeline, grounding, citations
│   │       └── llm/          # LLM providers
│   ├── selfcheck/            # Zero-dependency verification
│   └── tests/                # Pytest tests
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Page components
│   │   ├── lib/              # Auth, API client
│   │   └── styles/           # CSS
│   └── ...
├── docker-compose.yml
└── docs/
    └── adr/                  # Architecture Decision Records
```

## Deployment

### Production Checklist

1. Set `APP_ENV=production`
2. Generate strong `JWT_SECRET` (32+ chars)
3. Use PostgreSQL: `DATABASE_URL=postgresql://user:pass@host:5432/db`
4. Use Qdrant: `VECTOR_STORE=qdrant`, `QDRANT_URL=http://qdrant:6333`
5. Configure `CORS_ORIGINS` for your domain
6. Set `ALLOW_REGISTRATION=false` if needed
7. Configure LLM provider API keys
8. Enable HTTPS (reverse proxy with TLS termination)

### Scaling

- **API**: Run multiple backend replicas behind a load balancer
- **Database**: PostgreSQL with connection pooling (PgBouncer)
- **Vector Store**: Qdrant cluster for >100k chunks
- **Keyword Index**: SQLite FTS5 is single-writer; for high write throughput use Qdrant's payload filtering

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `python -m selfcheck` and `pytest`
5. Submit a PR

## License

MIT License - see [LICENSE](LICENSE)

## Architecture Decision Records

See [docs/adr/](docs/adr/) for key architectural decisions:

- ADR-001: Zero-dependency core
- ADR-002: SQLite as default vector store
- ADR-003: Extractive LLM as default
- ADR-004: Hybrid retrieval with RRF
- ADR-005: Citation integrity design
