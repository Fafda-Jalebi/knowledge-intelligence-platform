"""JWT (HS256) encode/decode implemented on the standard library.

Scope
-----
A deliberately small, audited subset of RFC 7519 sufficient for first-party
session tokens:

* HS256 only. ``alg`` is validated against the expected algorithm, so the
  ``alg: none`` and RS256->HS256 confusion attacks are structurally
  impossible.
* Signature verified with ``hmac.compare_digest`` before any claim is trusted.
* ``exp``, ``nbf`` and ``iat`` validated with a small configurable leeway.
* ``iss``/``aud`` validated when expected values are supplied.

Why not PyJWT? ``kip.core`` and everything below the API layer is
dependency-free (see ``docs/adr/0001-zero-dependency-core.md``), and this lets
the auth path be unit-tested without any install step. The implementation is
intentionally ~200 lines so it can be reviewed in full.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Iterable

from kip.errors import AuthenticationError

ALGORITHM = "HS256"
#: Reject absurdly large tokens before doing any work.
MAX_TOKEN_BYTES = 8192
DEFAULT_LEEWAY_SECONDS = 30


class TokenError(AuthenticationError):
    """Raised for any malformed, tampered or expired token."""

    code = "invalid_token"


class TokenExpiredError(TokenError):
    code = "token_expired"


def b64url_encode(raw: bytes) -> str:
    """Base64url without padding (RFC 7515 §2)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    """Base64url decode, restoring stripped padding."""
    if not isinstance(text, str):
        raise TokenError("Malformed token segment.")
    padding = "=" * (-len(text) % 4)
    try:
        return base64.urlsafe_b64decode(text + padding)
    except Exception as exc:  # noqa: BLE001
        raise TokenError("Malformed token segment.") from exc


def _sign(signing_input: bytes, secret: str) -> bytes:
    if not secret:
        raise TokenError("Server signing key is not configured.")
    return hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()


def encode(
    payload: dict[str, Any],
    secret: str,
    *,
    expires_in: int | None = None,
    issuer: str | None = None,
    audience: str | None = None,
    now: int | None = None,
) -> str:
    """Encode ``payload`` into a compact HS256 JWT.

    ``iat`` and ``exp`` are added automatically unless already present.
    """
    if not isinstance(payload, dict):
        raise TokenError("JWT payload must be a mapping.")

    issued_at = int(now if now is not None else time.time())
    claims: dict[str, Any] = dict(payload)
    claims.setdefault("iat", issued_at)
    if expires_in is not None and "exp" not in claims:
        claims["exp"] = issued_at + int(expires_in)
    if issuer is not None:
        claims.setdefault("iss", issuer)
    if audience is not None:
        claims.setdefault("aud", audience)

    header = {"alg": ALGORITHM, "typ": "JWT"}
    header_b64 = b64url_encode(_compact_json(header))
    payload_b64 = b64url_encode(_compact_json(claims))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature_b64 = b64url_encode(_sign(signing_input, secret))
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode(
    token: str,
    secret: str,
    *,
    algorithm: str = ALGORITHM,
    leeway: int = DEFAULT_LEEWAY_SECONDS,
    issuer: str | None = None,
    audience: str | None = None,
    required_claims: Iterable[str] = ("sub", "exp"),
    now: int | None = None,
) -> dict[str, Any]:
    """Verify ``token`` and return its claims.

    Raises :class:`TokenExpiredError` when the token is well-formed but past
    ``exp``, and :class:`TokenError` for every other failure.
    """
    if not isinstance(token, str) or not token:
        raise TokenError("Authentication token is missing.")
    if len(token.encode("utf-8")) > MAX_TOKEN_BYTES:
        raise TokenError("Authentication token is too large.")

    parts = token.split(".")
    if len(parts) != 3:
        raise TokenError("Authentication token is malformed.")
    header_b64, payload_b64, signature_b64 = parts

    header = _decode_json_segment(header_b64, "header")
    if header.get("typ") not in (None, "JWT", "jwt"):
        raise TokenError("Unsupported token type.")
    if header.get("alg") != algorithm:
        # Blocks 'alg: none' and algorithm-substitution attacks.
        raise TokenError("Unsupported token algorithm.")

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected = _sign(signing_input, secret)
    provided = b64url_decode(signature_b64)
    if not hmac.compare_digest(expected, provided):
        raise TokenError("Token signature verification failed.")

    claims = _decode_json_segment(payload_b64, "payload")

    current = int(now if now is not None else time.time())
    leeway = max(0, int(leeway))

    for claim in required_claims or ():
        if claim not in claims:
            raise TokenError(f"Token is missing the required '{claim}' claim.")

    if "exp" in claims:
        exp = _as_int(claims["exp"], "exp")
        if current > exp + leeway:
            raise TokenExpiredError("Session has expired. Please sign in again.")
    if "nbf" in claims:
        nbf = _as_int(claims["nbf"], "nbf")
        if current + leeway < nbf:
            raise TokenError("Token is not valid yet.")
    if "iat" in claims:
        iat = _as_int(claims["iat"], "iat")
        if current + max(leeway, 60) < iat:
            raise TokenError("Token was issued in the future.")

    if issuer is not None and claims.get("iss") != issuer:
        raise TokenError("Token issuer is not recognised.")
    if audience is not None:
        aud = claims.get("aud")
        allowed = aud if isinstance(aud, list) else [aud]
        if audience not in allowed:
            raise TokenError("Token audience is not recognised.")

    return claims


def peek(token: str) -> dict[str, Any]:
    """Decode claims **without** verifying the signature.

    Only for diagnostics and logging. Never use the result for an
    authorisation decision.
    """
    parts = str(token or "").split(".")
    if len(parts) != 3:
        return {}
    try:
        return _decode_json_segment(parts[1], "payload")
    except TokenError:
        return {}


def _compact_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _decode_json_segment(segment: str, what: str) -> dict[str, Any]:
    try:
        decoded = json.loads(b64url_decode(segment).decode("utf-8"))
    except TokenError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TokenError(f"Authentication token {what} is malformed.") from exc
    if not isinstance(decoded, dict):
        raise TokenError(f"Authentication token {what} is malformed.")
    return decoded


def _as_int(value: Any, claim: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TokenError(f"Token claim '{claim}' is not a valid timestamp.") from exc
