"""Minimal JSON-over-HTTP client for provider integrations.

Built on ``urllib.request`` rather than ``httpx``/``requests`` so that
:mod:`kip.core` keeps its no-third-party-imports rule (see
``docs/adr/0001-zero-dependency-core.md``). Every hosted embedding and LLM
provider this platform supports speaks JSON over HTTPS, so a small, careful
client covers all of them.

What "careful" means here, concretely:

* **Timeouts are mandatory.** A hosted model that stops responding must not pin
  a request worker forever; the caller always passes a timeout.
* **Retries only where retrying is correct.** 429 and 5xx are retried with
  exponential backoff and jitter; 400/401/403/404 are not, because retrying a
  malformed request or a bad key just burns time and quota.
* **Errors never carry credentials.** Headers are redacted before they can
  reach a log line or an exception message, and the ``Authorization`` value is
  never interpolated into an error string.
* **Response size is capped.** A provider returning an unexpectedly huge body
  should raise, not exhaust memory.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping

from kip.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 2
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
#: Status codes worth trying again: rate limiting and transient server faults.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
USER_AGENT = "knowledge-intelligence-platform/1.0"


class HttpError(RuntimeError):
    """An HTTP call failed in a way the caller should surface to the user."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass(slots=True)
class HttpResponse:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HttpError(
                "The provider returned a response that is not valid JSON."
            ) from exc


def post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Any:
    """POST ``payload`` as JSON and return the decoded JSON response."""
    body = json.dumps(payload).encode("utf-8")
    merged = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
        **(headers or {}),
    }
    return _request(url, body=body, headers=merged, timeout=timeout, retries=retries).json()


def get_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Any:
    """GET a JSON document."""
    merged = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
        **(headers or {}),
    }
    return _request(url, body=None, headers=merged, timeout=timeout, retries=retries).json()


def _request(
    url: str,
    *,
    body: bytes | None,
    headers: Mapping[str, str],
    timeout: float,
    retries: int,
) -> HttpResponse:
    if not str(url).lower().startswith(("http://", "https://")):
        raise HttpError(f"Refusing to call a non-HTTP(S) URL: {url!r}")

    attempts = max(1, int(retries) + 1)
    last: HttpError | None = None

    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method="POST" if body is not None else "GET",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise HttpError("The provider response exceeded the size limit.")
                logger.debug(
                    "http.call",
                    extra={
                        "url": _safe_url(url),
                        "status": response.status,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                        "attempt": attempt + 1,
                    },
                )
                return HttpResponse(
                    status=int(response.status),
                    body=raw,
                    headers={k.lower(): v for k, v in response.headers.items()},
                )
        except urllib.error.HTTPError as exc:
            detail = _describe_http_error(exc)
            last = HttpError(
                detail, status=exc.code, retryable=exc.code in RETRYABLE_STATUS
            )
        except urllib.error.URLError as exc:
            # DNS failure, refused connection, TLS problem, timeout.
            last = HttpError(
                f"Could not reach the provider at {_safe_url(url)}: {exc.reason}",
                retryable=True,
            )
        except TimeoutError:
            last = HttpError(
                f"The provider at {_safe_url(url)} did not respond within "
                f"{timeout:.0f}s.",
                retryable=True,
            )
        except HttpError:
            raise
        except OSError as exc:  # pragma: no cover - socket-level oddities
            last = HttpError(f"Network error calling the provider: {exc}", retryable=True)

        if last is not None and not last.retryable:
            break
        if attempt < attempts - 1:
            delay = _backoff(attempt)
            logger.warning(
                "http.retry",
                extra={
                    "url": _safe_url(url),
                    "attempt": attempt + 1,
                    "sleep_s": round(delay, 3),
                    "status": getattr(last, "status", None),
                },
            )
            time.sleep(delay)

    raise last or HttpError("The provider call failed for an unknown reason.")


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter, capped at 8 seconds.

    Jitter matters: without it, a burst of ingestion workers hitting a 429 all
    retry in lockstep and reproduce the overload they are backing off from.
    """
    return min(8.0, (2.0**attempt) * 0.5) * (0.5 + random.random() / 2.0)


def _describe_http_error(exc: urllib.error.HTTPError) -> str:
    """Turn a provider error into something a user can act on."""
    detail = ""
    try:
        raw = exc.read(8192).decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or "")
            elif isinstance(error, str):
                detail = error
            detail = detail or str(parsed.get("message") or "")
        detail = detail or raw.strip()
    except Exception:  # pragma: no cover - error bodies are unreliable
        detail = ""

    guidance = {
        401: "The provider rejected the API key. Check the key configured for this provider.",
        403: "The provider denied access. The key may lack permission for this model.",
        404: "The provider does not recognise that model or endpoint.",
        429: "The provider rate-limited the request.",
    }.get(exc.code, "")

    parts = [f"Provider returned HTTP {exc.code}."]
    if guidance:
        parts.append(guidance)
    if detail:
        parts.append(f"Detail: {detail[:400]}")
    return " ".join(parts)


def _safe_url(url: str) -> str:
    """Strip any query string, which is where API keys sometimes hide.

    >>> _safe_url("https://api.example.com/v1/embed?key=sk-secret")
    'https://api.example.com/v1/embed'
    """
    return str(url).split("?", 1)[0]
