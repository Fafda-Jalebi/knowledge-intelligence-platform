"""Knowledge Intelligence Platform - backend package.

Layering (imports only ever point downwards):

    kip.api        FastAPI routers, request/response schemas  (thin)
    kip.services   business logic, orchestration              (testable)
    kip.db         SQL schema + repositories                   (stdlib sqlite3 / psycopg)
    kip.core       RAG engine: extract, chunk, embed, retrieve,
                   rerank, ground, cite, evaluate              (stdlib + numpy only)
    kip.security   passwords, JWT, upload validation           (stdlib only)
    kip.config     settings                                    (stdlib only)

``kip.core`` never imports ``kip.api``, ``kip.services`` or ``kip.db``.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
