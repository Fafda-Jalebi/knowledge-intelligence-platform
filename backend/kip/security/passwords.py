"""Password hashing and password policy.

Algorithm
---------
PBKDF2-HMAC-SHA256 with a 16-byte random salt and a configurable iteration
count (default 390 000, matching the OWASP 2023 recommendation for
PBKDF2-HMAC-SHA256). Implemented on ``hashlib.pbkdf2_hmac`` from the standard
library, so there is no build-time dependency on ``bcrypt``/``argon2-cffi``.

If ``argon2-cffi`` is installed, :func:`hash_password` will prefer Argon2id --
verification transparently supports both formats, so an existing database
keeps working after the optional dependency is added.

Storage format
--------------
A single self-describing string, so no schema change is needed to rotate
parameters::

    pbkdf2_sha256$<iterations>$<base64(salt)>$<base64(dk)>
    argon2$<argon2-encoded-hash>

Verification is constant-time (``hmac.compare_digest``).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import unicodedata

from kip.errors import ValidationError

PBKDF2_PREFIX = "pbkdf2_sha256"
ARGON2_PREFIX = "argon2"
DEFAULT_ITERATIONS = 390_000
SALT_BYTES = 16
DK_BYTES = 32
#: Guard against DoS via multi-megabyte password payloads.
MAX_PASSWORD_BYTES = 1024

_COMMON_PASSWORDS = frozenset(
    {
        "password", "password1", "password123", "passw0rd", "12345678",
        "123456789", "1234567890", "qwertyuiop", "qwerty123", "letmein123",
        "iloveyou1", "admin123", "administrator", "welcome123", "abc12345",
        "changeme", "changeme123", "secret123", "monkey123", "dragon123",
        "football1", "baseball1", "sunshine1", "princess1", "trustno1",
        "knowledge123", "knowledgeplatform", "ragplatform123",
    }
)


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def normalise_password(password: str) -> str:
    """Apply Unicode NFKC normalisation (so visually identical inputs match)."""
    if not isinstance(password, str):
        raise ValidationError("Password must be a string.")
    return unicodedata.normalize("NFKC", password)


def validate_password_strength(password: str, *, min_length: int = 10) -> None:
    """Raise :class:`ValidationError` if ``password`` fails the policy.

    Policy: length >= ``min_length``, at least one letter, at least one digit
    or symbol, not in a small common-password deny list, and not composed of a
    single repeated character.
    """
    if not isinstance(password, str) or not password:
        raise ValidationError("Password is required.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValidationError("Password is too long (maximum 1024 bytes).")

    candidate = normalise_password(password)
    problems: list[str] = []
    if len(candidate) < min_length:
        problems.append(f"be at least {min_length} characters long")
    if not re.search(r"[A-Za-z]", candidate):
        problems.append("contain at least one letter")
    if not re.search(r"[0-9]|[^A-Za-z0-9]", candidate):
        problems.append("contain at least one number or symbol")
    if len(set(candidate)) <= 2:
        problems.append("use more than two distinct characters")
    if candidate.lower() in _COMMON_PASSWORDS:
        problems.append("not be a commonly used password")

    if problems:
        raise ValidationError(
            "Password must " + "; ".join(problems) + ".",
            details={"requirements": problems},
        )


def hash_password(password: str, *, iterations: int = DEFAULT_ITERATIONS) -> str:
    """Return a self-describing hash string for ``password``."""
    if not isinstance(password, str) or not password:
        raise ValidationError("Password is required.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValidationError("Password is too long (maximum 1024 bytes).")

    candidate = normalise_password(password)

    argon2 = _try_import_argon2()
    if argon2 is not None:
        try:
            return f"{ARGON2_PREFIX}${argon2.PasswordHasher().hash(candidate)}"
        except Exception:  # pragma: no cover - fall back to stdlib
            pass

    iterations = max(1_000, int(iterations))
    salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", candidate.encode("utf-8"), salt, iterations, DK_BYTES)
    return f"{PBKDF2_PREFIX}${iterations}${_b64e(salt)}${_b64e(dk)}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verification of ``password`` against ``stored``.

    Returns ``False`` (never raises) for malformed or unknown hash formats, so
    a corrupt row cannot turn into a 500 on the login path.
    """
    if not isinstance(password, str) or not isinstance(stored, str) or not stored:
        return False
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return False

    candidate = normalise_password(password)

    if stored.startswith(ARGON2_PREFIX + "$"):
        argon2 = _try_import_argon2()
        if argon2 is None:
            return False
        try:
            return bool(
                argon2.PasswordHasher().verify(stored[len(ARGON2_PREFIX) + 1 :], candidate)
            )
        except Exception:
            return False

    parts = stored.split("$")
    if len(parts) != 4 or parts[0] != PBKDF2_PREFIX:
        return False
    try:
        iterations = int(parts[1])
        salt = _b64d(parts[2])
        expected = _b64d(parts[3])
    except (ValueError, TypeError, base64.binascii.Error):  # type: ignore[attr-defined]
        return False
    if iterations <= 0 or not salt or not expected:
        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256", candidate.encode("utf-8"), salt, iterations, len(expected)
    )
    return hmac.compare_digest(actual, expected)


def needs_rehash(stored: str, *, iterations: int = DEFAULT_ITERATIONS) -> bool:
    """True when ``stored`` should be replaced by a freshly computed hash.

    Callers re-hash opportunistically on a successful login, which is the only
    moment the plaintext is available. Three cases:

    * Argon2id -- already the strongest format we produce, so ``False``.
    * PBKDF2 with fewer iterations than current policy -- ``True``.
    * Anything else, including unparseable rows -- ``True``. An unrecognised
      format is exactly the case that most needs upgrading, and returning
      ``False`` there would silently pin a legacy hash forever.

    >>> needs_rehash("")
    False
    >>> needs_rehash("garbage")
    True
    >>> needs_rehash("argon2$$argon2id$v=19$m=65536,t=3,p=4$abc$def")
    False
    >>> needs_rehash("pbkdf2_sha256$1000$c2FsdA==$ZGs=")
    True
    """
    if not stored:
        # No stored hash at all is not a rehash decision; it is an auth failure.
        return False
    if stored.startswith(ARGON2_PREFIX + "$"):
        return False
    if not stored.startswith(PBKDF2_PREFIX + "$"):
        return True
    parts = stored.split("$")
    if len(parts) != 4:
        return True
    try:
        return int(parts[1]) < int(iterations)
    except ValueError:
        return True


def _try_import_argon2():  # pragma: no cover - optional dependency
    try:
        import argon2  # type: ignore

        return argon2
    except Exception:
        return None
