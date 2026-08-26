"""Database layer: schema and repositories.

This package owns all SQL. Upper layers (services, API) import repository
classes and call their methods; they never write raw SQL.
"""

from __future__ import annotations

from kip.db.repositories import (
    ChunkRepository,
    ConversationRepository,
    DocumentRepository,
    UserRepository,
)

__all__ = [
    "ChunkRepository",
    "ConversationRepository",
    "DocumentRepository",
    "UserRepository",
    "create_tables",
    "drop_tables",
    "get_session",
    "init_db",
]

# Re-export the session factory and init functions
from kip.db.session import (
    create_tables,
    drop_tables,
    get_session,
    init_db,
)
