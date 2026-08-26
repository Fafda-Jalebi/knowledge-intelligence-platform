"""Service layer: business logic orchestrating repositories and core engine.

Services are the single place where:
- Transactions are managed (via the session from the API layer)
- Multiple repositories are coordinated
- The core RAG engine (kip.core) is driven with real storage
"""

from __future__ import annotations

from kip.services.auth import AuthService
from kip.services.documents import DocumentService
from kip.services.chat import ChatService

__all__ = ["AuthService", "DocumentService", "ChatService"]
