"""Pytest configuration."""

import pytest
import asyncio
import tempfile
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from kip.db.session import Base, get_session, init_db, create_tables, drop_tables, close_db
# Import models to register them with Base metadata
from kip.db.session import User, Document, Chunk, Conversation, Message
from kip.config import Settings


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_db():
    """Create a test database for each test function."""
    import tempfile
    import os
    import kip.db.session as session_module

    # Create a temporary file for the database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    # Save original settings
    original_engine = session_module._engine
    original_session_factory = session_module._session_factory

    try:
        # Reset global state
        session_module._engine = None
        session_module._session_factory = None

        # Set test database URL
        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"

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

        # Override the session dependency
        from kip.api import app
        from kip.db.session import get_session as original_get_session

        async def override_get_session():
            async with session_module._session_factory() as session:
                yield session

        app.dependency_overrides[original_get_session] = override_get_session

        yield
    finally:
        from kip.api import app
        from kip.db.session import get_session as original_get_session
        app.dependency_overrides.clear()
        # Restore original state
        session_module._engine = original_engine
        session_module._session_factory = original_session_factory
        # Clean up
        try:
            os.unlink(db_path)
        except OSError:
            pass
