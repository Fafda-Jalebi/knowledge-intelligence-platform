"""Pytest configuration."""

import pytest
import tempfile
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from kip.db.session import Base, get_session, init_db, create_tables, drop_tables, close_db
# Import models to register them with Base metadata
from kip.db.session import User, Document, Chunk, Conversation, Message
from kip.config import Settings, reset_settings, get_settings


@pytest.fixture(scope="function")
async def test_db():
    """Create a test database for each test function."""
    import tempfile
    import os
    import kip.db.session as session_module
    import kip.api.routers.chat as chat_router
    import kip.api.routers.documents as doc_router
    import kip.api.routers.auth as auth_router
    from kip.services.chat import ChatService
    from kip.services.documents import DocumentService
    from kip.services.auth import AuthService

    # Create temporary files for database, vector store and keyword index
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    with tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False) as tmp:
        vector_path = tmp.name
    with tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False) as tmp:
        keyword_path = tmp.name

    # Save original settings
    original_engine = session_module._engine
    original_session_factory = session_module._session_factory

    try:
        # Dispose the original engine if it exists (e.g., from app lifespan)
        if original_engine is not None:
            await original_engine.dispose()
            import asyncio
            await asyncio.sleep(0)

        # Reset global state
        session_module._engine = None
        session_module._session_factory = None

        # Set test database URL and vector/keyword store paths
        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
        os.environ["VECTOR_STORE_PATH"] = vector_path
        os.environ["KEYWORD_INDEX_PATH"] = keyword_path

        # Use a fixed JWT secret for tests so tokens remain valid across requests
        os.environ["JWT_SECRET"] = "test-secret-key-for-testing-only-32chars!!"

        # Reset settings cache so new paths are picked up
        reset_settings()

        # Initialize database with test URL
        init_db()

        # Drop and recreate tables explicitly to ensure clean state
        await drop_tables()
        await create_tables()

        # Verify tables exist
        async with session_module._engine.begin() as conn:
            result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = result.fetchall()
            print(f"Created tables: {[t[0] for t in tables]}")

        # Re-instantiate service singletons with fresh settings
        chat_router.chat_service = ChatService()
        doc_router.doc_service = DocumentService()
        auth_router.auth_service = AuthService()

        # Override the session dependency
        from kip.api import app
        from kip.db.session import get_session as original_get_session

        async def override_get_session():
            async with session_module._session_factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        app.dependency_overrides[original_get_session] = override_get_session

        yield
    finally:
        from kip.api import app
        from kip.db.session import get_session as original_get_session, close_db
        app.dependency_overrides.clear()
        # Dispose the test engine before restoring the original
        await close_db()
        # Restore original state
        session_module._engine = original_engine
        session_module._session_factory = original_session_factory
        # Clean up temp files
        for path in (db_path, vector_path, keyword_path):
            try:
                os.unlink(path)
            except OSError:
                pass
