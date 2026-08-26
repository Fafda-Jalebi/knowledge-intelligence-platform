"""Unit tests for configuration."""

import os
from kip.config import Settings, build_settings, parse_dotenv


def test_settings_defaults():
    """Test default settings values."""
    s = Settings()
    assert s.app_env == "development"
    assert s.api_port == 8000
    assert s.vector_store == "sqlite"
    assert s.embedding_provider == "hashing"
    assert s.llm_provider == "extractive"
    assert s.retrieval_mode == "hybrid"
    assert s.reranker == "heuristic"


def test_settings_validation():
    """Test settings validation."""
    # Production without JWT_SECRET should fail
    prod = Settings(app_env="production", jwt_secret="")
    problems = prod.validate()
    assert any("JWT_SECRET" in p for p in problems)

    # Development with defaults should pass
    dev = Settings(app_env="development")
    assert dev.validate() == []


def test_settings_validate_embedding_dim():
    """Test embedding dimension validation."""
    s = Settings(embedding_dim=-1)
    problems = s.validate()
    assert any("EMBEDDING_DIM" in p for p in problems)


def test_settings_validate_chunking():
    """Test chunking parameter validation."""
    s = Settings(chunk_overlap_tokens=400, chunk_target_tokens=300)
    problems = s.validate()
    assert any("CHUNK_OVERLAP_TOKENS" in p for p in problems)

    s = Settings(chunk_max_tokens=100, chunk_target_tokens=300)
    problems = s.validate()
    assert any("CHUNK_MAX_TOKENS" in p for p in problems)


def test_parse_dotenv():
    """Test .env parsing."""
    content = """
# Comment
PLAIN=value
QUOTED="spaced value"
SINGLE='single quoted'
EMPTY=
EXPORTED=yes
WITH_EQUALS=a=b=c
  INDENTED = trimmed
NOT A LINE
"""
    parsed = parse_dotenv(content)
    assert parsed["PLAIN"] == "value"
    assert parsed["QUOTED"] == "spaced value"
    assert parsed["SINGLE"] == "single quoted"
    assert parsed["EMPTY"] == ""
    assert parsed["EXPORTED"] == "yes"
    assert parsed["WITH_EQUALS"] == "a=b=c"
    assert parsed["INDENTED"] == "trimmed"
    assert "NOT A LINE" not in parsed


def test_build_settings_from_env(monkeypatch):
    """Test building settings from environment variables."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET", "a" * 48)
    monkeypatch.setenv("VECTOR_STORE", "qdrant")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    settings = build_settings()
    assert settings.app_env == "production"
    assert settings.vector_store == "qdrant"
    assert settings.embedding_provider == "openai"
    assert settings.openai_api_key == "sk-test"
