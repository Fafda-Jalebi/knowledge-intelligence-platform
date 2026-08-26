"""Self-check package.

Run everything::

    cd backend && python -m selfcheck

Run one group::

    python -m selfcheck security
    python -m selfcheck extraction -v

Exit status is non-zero when any check fails, so this is usable directly in CI
as a smoke gate before the full pytest suite.
"""

from __future__ import annotations

__all__ = ["GROUPS"]

#: Group name -> module path. Kept as strings so importing the package stays
#: cheap and a broken group cannot prevent the others from running.
GROUPS: dict[str, str] = {
    "doctests": "selfcheck.check_doctests",
    "security": "selfcheck.check_security",
    "extraction": "selfcheck.check_extraction",
    "retrieval": "selfcheck.check_retrieval",
}
