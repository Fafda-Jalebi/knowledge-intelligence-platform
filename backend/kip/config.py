"""Application configuration.

Settings are resolved from, in order of increasing precedence:

1. The defaults declared in :class:`Settings`.
2. A ``.env`` file discovered by walking up from this file to the repo root.
3. Real process environment variables.

Why not ``pydantic-settings``?
-----------------------------
``kip.core`` is deliberately dependency-free (stdlib + numpy) so the retrieval
and evaluation engine can be imported, unit-tested and benchmarked without
installing a web framework. Configuration sits below the API layer and is
imported by ``kip.core`` consumers, so it follows the same rule: no third-party
imports. See ``docs/adr/0001-zero-dependency-core.md``.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# .env loading
# --------------------------------------------------------------------------- #

_TRUE = {"1", "true", "yes", "on", "y", "t"}
_FALSE = {"0", "false", "no", "off", "n", "f"}


def repo_root() -> Path:
    """Return the repository root (the directory that contains ``backend/``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "backend").is_dir() and (parent / ".env.example").is_file():
            return parent
    # Fallback: <repo>/backend/kip/config.py -> up two levels.
    return here.parents[2]


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``.env`` style text into a mapping.

    Supports ``KEY=value``, ``export KEY=value``, ``#`` comments, blank lines,
    single/double quoted values, and inline trailing comments on unquoted
    values. Deliberately small: no variable interpolation, no multiline values.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not _is_env_key(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # Strip an inline comment that is preceded by whitespace.
            hash_at = value.find(" #")
            if hash_at != -1:
                value = value[:hash_at].rstrip()
        out[key] = value
    return out


def _is_env_key(key: str) -> bool:
    return all(ch.isalnum() or ch == "_" for ch in key)


def load_dotenv(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``.env`` into a dict, and into ``os.environ`` when absent there."""
    path = path or (repo_root() / ".env")
    if not path.is_file():
        return {}
    try:
        values = parse_dotenv(path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


# --------------------------------------------------------------------------- #
# Typed getters
# --------------------------------------------------------------------------- #


def env_str(key: str, default: str = "") -> str:
    value = os.environ.get(key)
    return default if value is None else value.strip()


def env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return default


def env_list(key: str, default: Iterable[str] = ()) -> list[str]:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return [item for item in default]
    return [part.strip() for part in raw.split(",") if part.strip()]


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

#: Keys whose values must never appear in logs, API responses or error strings.
SECRET_KEYS: frozenset[str] = frozenset(
    {
        "jwt_secret",
        "openai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        "qdrant_api_key",
        "database_url",
    }
)


@dataclass(slots=True)
class Settings:
    """Fully resolved, immutable-by-convention application settings."""

    # Application
    app_env: str = "development"
    app_name: str = "Knowledge Intelligence Platform"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    log_format: str = "json"

    # Security
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720
    password_min_length: int = 10
    pbkdf2_iterations: int = 390_000
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    allow_registration: bool = True

    # Databases
    database_url: str = "sqlite:///./var/kip.sqlite3"
    vector_store: str = "sqlite"
    vector_store_path: str = "./var/vectors.sqlite3"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "kip_chunks"

    # Embeddings
    embedding_provider: str = "hashing"
    embedding_model: str = "kip-hashing-v1"
    #: ``0`` means "use the provider's native width". The embedder is the
    #: authority on its own dimension -- it is discovered from the loaded model
    #: or a probe request -- and the vector store learns it from the embedding
    #: spec at ``ensure_collection`` time. Pinning it here is therefore optional,
    #: and leaving it at 0 removes the commonest configuration mistake: switching
    #: EMBEDDING_PROVIDER and forgetting that the new model is a different width.
    embedding_dim: int = 0
    embedding_batch_size: int = 32

    # LLM
    llm_provider: str = "extractive"
    llm_model: str = "kip-extractive-v1"
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 900
    llm_timeout_seconds: int = 60
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Chunking
    chunk_target_tokens: int = 320
    chunk_overlap_tokens: int = 60
    chunk_min_tokens: int = 40
    chunk_max_tokens: int = 520
    chunk_respect_sections: bool = True
    chunk_respect_pages: bool = True

    # Retrieval
    retrieval_mode: str = "hybrid"
    retrieval_dense_top_k: int = 24
    retrieval_keyword_top_k: int = 24
    retrieval_fusion: str = "rrf"
    retrieval_rrf_k: int = 60
    retrieval_dense_weight: float = 0.65
    retrieval_keyword_weight: float = 0.35
    retrieval_candidate_limit: int = 40

    #: Keyword (lexical) backend for the second retrieval axis.
    #: ``fts5`` is the default because the index lives in a file and is therefore
    #: shared by every worker process; an in-process index would make a document
    #: uploaded via one worker invisible to a keyword query served by another.
    #: ``bm25`` is exact and is the reference used by evaluation. ``none``
    #: disables keyword retrieval, reducing the platform to semantic-only.
    keyword_index: str = "fts5"
    keyword_index_path: str = "./var/keyword.sqlite3"

    # Reranking
    reranker: str = "heuristic"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 6

    # Context & grounding
    context_token_budget: int = 2600
    context_max_passages: int = 8
    context_max_per_document: int = 4
    grounding_min_score: float = 0.16
    grounding_min_passages: int = 1
    grounding_support_threshold: float = 0.32
    grounding_enforce_citations: bool = True

    # Uploads
    storage_dir: str = "./var/storage"
    max_upload_mb: int = 25
    allowed_extensions: list[str] = field(
        default_factory=lambda: ["pdf", "docx", "txt", "md"]
    )

    # ------------------------------------------------------------------ #
    # Normalisation
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        """Coerce list-valued fields that are commonly passed as strings.

        ``Settings(allowed_extensions="pdf,md")`` is an easy mistake to make --
        from a test, a script, or a future config source -- and without this a
        plain string would be iterated character by character, silently turning
        the allow-list into ``{'p','d','f',',','m'}``. Failing loudly would be
        worse than accepting the obvious intent, so we split on commas.
        """
        self.cors_origins = _as_list(self.cors_origins)
        self.allowed_extensions = [
            ext.lower().lstrip(".") for ext in _as_list(self.allowed_extensions) if ext
        ]

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"}

    @property
    def max_upload_bytes(self) -> int:
        return max(1, self.max_upload_mb) * 1024 * 1024

    @property
    def allowed_extension_set(self) -> frozenset[str]:
        return frozenset(ext.lower().lstrip(".") for ext in self.allowed_extensions)

    def resolve_path(self, value: str) -> Path:
        """Resolve a possibly-relative configured path against the repo root."""
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = repo_root() / path
        return path

    @property
    def storage_path(self) -> Path:
        return self.resolve_path(self.storage_dir)

    def redacted(self) -> dict[str, Any]:
        """Return settings as a dict with every secret value masked.

        Used by the ``/api/settings`` endpoint and by startup logging so that
        configuration is observable without ever leaking credentials.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in SECRET_KEYS:
                out[f.name] = _mask(value)
            else:
                out[f.name] = value
        return out

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems."""
        problems: list[str] = []
        if self.is_production and not self.jwt_secret:
            problems.append(
                "JWT_SECRET must be set when APP_ENV=production "
                "(generate: python -c \"import secrets;print(secrets.token_urlsafe(48))\")"
            )
        if self.is_production and len(self.jwt_secret) < 32:
            if self.jwt_secret:
                problems.append("JWT_SECRET should be at least 32 characters.")
        if self.embedding_dim < 0:
            problems.append("EMBEDDING_DIM must be 0 (auto) or a positive integer.")
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            problems.append(
                "CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TARGET_TOKENS."
            )
        if self.chunk_max_tokens < self.chunk_target_tokens:
            problems.append("CHUNK_MAX_TOKENS must be >= CHUNK_TARGET_TOKENS.")
        if self.retrieval_mode not in {"hybrid", "dense", "keyword"}:
            problems.append("RETRIEVAL_MODE must be one of: hybrid, dense, keyword.")
        if self.retrieval_fusion not in {"rrf", "weighted"}:
            problems.append("RETRIEVAL_FUSION must be one of: rrf, weighted.")
        if self.vector_store not in {"memory", "sqlite", "qdrant"}:
            problems.append("VECTOR_STORE must be one of: memory, sqlite, qdrant.")
        if self.keyword_index not in {"fts5", "bm25", "none"}:
            problems.append("KEYWORD_INDEX must be one of: fts5, bm25, none.")
        if self.retrieval_mode == "keyword" and self.keyword_index == "none":
            problems.append(
                "RETRIEVAL_MODE=keyword needs a keyword index, but KEYWORD_INDEX=none."
            )
        if self.reranker not in {"heuristic", "cross-encoder", "llm", "none"}:
            problems.append(
                "RERANKER must be one of: heuristic, cross-encoder, llm, none."
            )
        if not 0.0 <= self.grounding_min_score <= 1.0:
            problems.append("GROUNDING_MIN_SCORE must be between 0.0 and 1.0.")
        if self.max_upload_mb <= 0:
            problems.append("MAX_UPLOAD_MB must be positive.")
        if self.is_production and "*" in self.cors_origins:
            problems.append("CORS_ORIGINS must not be '*' in production.")
        needs_key = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        key_attr = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
        }
        if self.llm_provider in needs_key and not key_attr[self.llm_provider]:
            problems.append(
                f"LLM_PROVIDER={self.llm_provider} requires "
                f"{needs_key[self.llm_provider]} to be set."
            )
        if self.embedding_provider in needs_key and not key_attr.get(
            self.embedding_provider, ""
        ):
            problems.append(
                f"EMBEDDING_PROVIDER={self.embedding_provider} requires "
                f"{needs_key[self.embedding_provider]} to be set."
            )
        return problems


def _as_list(value: Any) -> list[str]:
    """Coerce a str / iterable / None into a clean list of strings.

    >>> _as_list("pdf, md")
    ['pdf', 'md']
    >>> _as_list(["pdf", " md "])
    ['pdf', 'md']
    >>> _as_list(None)
    []
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts: Iterable[str] = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        parts = [str(item) for item in value]
    else:
        parts = [str(value)]
    return [part.strip() for part in parts if str(part).strip()]


def _mask(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= 8:
        return "***"
    return f"{text[:3]}***{text[-2:]} (len={len(text)})"


def build_settings() -> Settings:
    """Construct :class:`Settings` from the current environment."""
    jwt_secret = env_str("JWT_SECRET")
    app_env = env_str("APP_ENV", "development")
    if not jwt_secret and app_env.lower() not in {"production", "prod"}:
        # Ephemeral development key: keeps the app runnable with zero setup,
        # while guaranteeing tokens do not survive a restart.
        jwt_secret = secrets.token_urlsafe(48)

    return Settings(
        app_env=app_env,
        app_name=env_str("APP_NAME", "Knowledge Intelligence Platform"),
        api_host=env_str("API_HOST", "0.0.0.0"),
        api_port=env_int("API_PORT", 8000),
        log_level=env_str("LOG_LEVEL", "INFO").upper(),
        log_format=env_str("LOG_FORMAT", "json").lower(),
        jwt_secret=jwt_secret,
        jwt_algorithm=env_str("JWT_ALGORITHM", "HS256").upper(),
        jwt_expire_minutes=env_int("JWT_EXPIRE_MINUTES", 720),
        password_min_length=env_int("PASSWORD_MIN_LENGTH", 10),
        pbkdf2_iterations=env_int("PBKDF2_ITERATIONS", 390_000),
        cors_origins=env_list(
            "CORS_ORIGINS", ["http://localhost:5173", "http://127.0.0.1:5173"]
        ),
        allow_registration=env_bool("ALLOW_REGISTRATION", True),
        database_url=env_str("DATABASE_URL", "sqlite:///./var/kip.sqlite3"),
        vector_store=env_str("VECTOR_STORE", "sqlite").lower(),
        vector_store_path=env_str("VECTOR_STORE_PATH", "./var/vectors.sqlite3"),
        qdrant_url=env_str("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=env_str("QDRANT_API_KEY"),
        qdrant_collection=env_str("QDRANT_COLLECTION", "kip_chunks"),
        embedding_provider=env_str("EMBEDDING_PROVIDER", "hashing").lower(),
        embedding_model=env_str("EMBEDDING_MODEL", "kip-hashing-v1"),
        embedding_dim=env_int("EMBEDDING_DIM", 0),
        embedding_batch_size=env_int("EMBEDDING_BATCH_SIZE", 32),
        llm_provider=env_str("LLM_PROVIDER", "extractive").lower(),
        llm_model=env_str("LLM_MODEL", "kip-extractive-v1"),
        llm_temperature=env_float("LLM_TEMPERATURE", 0.1),
        llm_max_output_tokens=env_int("LLM_MAX_OUTPUT_TOKENS", 900),
        llm_timeout_seconds=env_int("LLM_TIMEOUT_SECONDS", 60),
        openai_api_key=env_str("OPENAI_API_KEY"),
        openai_base_url=env_str("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        anthropic_api_key=env_str("ANTHROPIC_API_KEY"),
        gemini_api_key=env_str("GEMINI_API_KEY"),
        ollama_base_url=env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        chunk_target_tokens=env_int("CHUNK_TARGET_TOKENS", 320),
        chunk_overlap_tokens=env_int("CHUNK_OVERLAP_TOKENS", 60),
        chunk_min_tokens=env_int("CHUNK_MIN_TOKENS", 40),
        chunk_max_tokens=env_int("CHUNK_MAX_TOKENS", 520),
        chunk_respect_sections=env_bool("CHUNK_RESPECT_SECTIONS", True),
        chunk_respect_pages=env_bool("CHUNK_RESPECT_PAGES", True),
        retrieval_mode=env_str("RETRIEVAL_MODE", "hybrid").lower(),
        retrieval_dense_top_k=env_int("RETRIEVAL_DENSE_TOP_K", 24),
        retrieval_keyword_top_k=env_int("RETRIEVAL_KEYWORD_TOP_K", 24),
        retrieval_fusion=env_str("RETRIEVAL_FUSION", "rrf").lower(),
        retrieval_rrf_k=env_int("RETRIEVAL_RRF_K", 60),
        retrieval_dense_weight=env_float("RETRIEVAL_DENSE_WEIGHT", 0.65),
        retrieval_keyword_weight=env_float("RETRIEVAL_KEYWORD_WEIGHT", 0.35),
        retrieval_candidate_limit=env_int("RETRIEVAL_CANDIDATE_LIMIT", 40),
        keyword_index=env_str("KEYWORD_INDEX", "fts5").lower(),
        keyword_index_path=env_str("KEYWORD_INDEX_PATH", "./var/keyword.sqlite3"),
        reranker=env_str("RERANKER", "heuristic").lower(),
        reranker_model=env_str(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        rerank_top_n=env_int("RERANK_TOP_N", 6),
        context_token_budget=env_int("CONTEXT_TOKEN_BUDGET", 2600),
        context_max_passages=env_int("CONTEXT_MAX_PASSAGES", 8),
        context_max_per_document=env_int("CONTEXT_MAX_PER_DOCUMENT", 4),
        grounding_min_score=env_float("GROUNDING_MIN_SCORE", 0.16),
        grounding_min_passages=env_int("GROUNDING_MIN_PASSAGES", 1),
        grounding_support_threshold=env_float("GROUNDING_SUPPORT_THRESHOLD", 0.32),
        grounding_enforce_citations=env_bool("GROUNDING_ENFORCE_CITATIONS", True),
        storage_dir=env_str("STORAGE_DIR", "./var/storage"),
        max_upload_mb=env_int("MAX_UPLOAD_MB", 25),
        allowed_extensions=env_list("ALLOWED_EXTENSIONS", ["pdf", "docx", "txt", "md"]),
    )


_settings: Settings | None = None


def get_settings(*, refresh: bool = False) -> Settings:
    """Return the process-wide settings singleton."""
    global _settings
    if _settings is None or refresh:
        load_dotenv()
        _settings = build_settings()
    return _settings


def reset_settings() -> None:
    """Clear the settings cache. Used by tests."""
    global _settings
    _settings = None
