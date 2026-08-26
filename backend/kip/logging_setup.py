"""Structured logging.

Emits one JSON object per line (``LOG_FORMAT=json``) or a compact human
readable line (``LOG_FORMAT=text``). JSON logs are what you want in a
container: they are trivially parseable by Loki/CloudWatch/Datadog.

Security
--------
Two protections are built in and covered by tests:

1. ``SensitiveFilter`` redacts anything that looks like a credential
   (``api_key``, ``password``, ``token``, ``authorization``, bearer values)
   from both the message and the structured extras.
2. Document *content* is never logged at INFO. Passage text appears only
   under DEBUG and is truncated, because indexed documents may be
   confidential.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Iterable

_request_id: ContextVar[str] = ContextVar("kip_request_id", default="-")
_user_id: ContextVar[str] = ContextVar("kip_user_id", default="-")

#: Attribute names on a LogRecord that are *not* user-supplied extras.
_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "stacklevel",
        "thread",
        "threadName",
        "taskName",
    }
)

_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|secret|password|passwd|token|authorization|credential|cookie)",
    re.IGNORECASE,
)
#: HTTP authentication schemes whose credential follows the scheme name.
_AUTH_SCHEMES = ("Bearer", "Basic", "Digest", "Token", "ApiKey")
_BEARER_RE = re.compile(
    r"((?:" + "|".join(_AUTH_SCHEMES) + r")\s+)[A-Za-z0-9+/._\-=]{8,}",
    re.IGNORECASE,
)
#: Matches ``password=x``, ``password: x`` and the JSON form ``"password": "x"``.
#: The optional quote after the key name is what makes serialised payloads --
#: which is how most credentials actually reach a log line -- get redacted.
_ASSIGN_RE = re.compile(
    r"((?:api[_-]?key|secret|password|passwd|token|authorization|credential)"
    r"[\"']?\s*[=:]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^\s,;)}\]]+)",
    re.IGNORECASE,
)
_LONG_KEY_RE = re.compile(r"\b(sk-|xoxb-|ghp_|gho_|AIza)[A-Za-z0-9_\-]{10,}")

REDACTED = "***redacted***"

#: Values that :func:`_mask_assignment` must leave alone. ``REDACTED`` keeps
#: redaction idempotent; the scheme names keep ``Authorization: Bearer <token>``
#: readable as an auth header after :data:`_BEARER_RE` has already masked the
#: credential itself.
_ALREADY_SAFE = frozenset(
    {REDACTED.lower(), *(scheme.lower() for scheme in _AUTH_SCHEMES)}
)


def redact_text(text: str) -> str:
    """Redact credential-looking substrings from free text.

    >>> redact_text('Authorization: Bearer abcdefghijkl')
    'Authorization: Bearer ***redacted***'
    >>> redact_text('{"token": "abcdefghijklmnop"}')
    '{"token": "***redacted***"}'
    >>> redact_text('nothing sensitive here')
    'nothing sensitive here'

    Redaction is idempotent, so a message that passes through the filter twice
    is not progressively mangled:

    >>> once = redact_text('password=hunter2 api_key=sk-abcdefghijklmno')
    >>> once == redact_text(once)
    True
    """
    if not text:
        return text
    text = _BEARER_RE.sub(r"\1" + REDACTED, text)
    text = _ASSIGN_RE.sub(_mask_assignment, text)
    text = _LONG_KEY_RE.sub(REDACTED, text)
    return text


def _mask_assignment(match: "re.Match[str]") -> str:
    """Replace an assigned value, preserving the quoting style around it."""
    value = match.group(2)
    quote = value[0] if value[:1] in "\"'" else ""
    inner = value[1:-1] if quote and len(value) >= 2 else value
    if inner.lower() in _ALREADY_SAFE:
        return match.group(0)
    return f"{match.group(1)}{quote}{REDACTED}{quote}"


def redact_value(key: str, value: Any) -> Any:
    """Redact ``value`` when ``key`` names a credential."""
    if _SENSITIVE_KEY_RE.search(key or ""):
        return REDACTED if value not in (None, "") else value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(key, v) for v in value]
    return value


class SensitiveFilter(logging.Filter):
    """Scrub credentials from the message and from structured extras."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            if isinstance(record.msg, str):
                record.msg = redact_text(record.msg)
            for key, value in list(record.__dict__.items()):
                if key in _RESERVED or key.startswith("_"):
                    continue
                record.__dict__[key] = redact_value(key, value)
        except Exception:  # pragma: no cover - logging must never explode
            pass
        return True


class ContextFilter(logging.Filter):
    """Attach the current request/user correlation ids to every record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not hasattr(record, "request_id"):
            record.request_id = _request_id.get()
        if not hasattr(record, "user_id"):
            record.user_id = _user_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = _jsonable(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):  # pragma: no cover
            return json.dumps({"level": record.levelname, "msg": str(record.msg)})


class TextFormatter(logging.Formatter):
    """Compact, colour-free line format for local development."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{time.strftime('%H:%M:%S', time.localtime(record.created))} "
            f"{record.levelname:<7} {record.name:<28} {record.getMessage()}"
        )
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED
            and not k.startswith("_")
            and k not in {"request_id", "user_id"}
        }
        if extras:
            rendered = " ".join(f"{k}={_jsonable(v)}" for k, v in extras.items())
            base = f"{base}  [{rendered}]"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


_configured = False


def configure_logging(
    level: str | int | None = None,
    fmt: str | None = None,
    *,
    force: bool = False,
) -> None:
    """Install the root handler. Idempotent unless ``force=True``."""
    global _configured
    if _configured and not force:
        return

    level = level or os.environ.get("LOG_LEVEL", "INFO")
    fmt = (fmt or os.environ.get("LOG_FORMAT", "json")).lower()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    handler.addFilter(ContextFilter())
    handler.addFilter(SensitiveFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level if isinstance(level, int) else str(level).upper())

    # Uvicorn duplicates access logs through its own handlers; let ours win.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(noisy)
        logger.handlers = []
        logger.propagate = True

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Correlation ids
# --------------------------------------------------------------------------- #


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def set_request_context(request_id: str | None = None, user_id: str | None = None) -> str:
    rid = request_id or new_request_id()
    _request_id.set(rid)
    if user_id is not None:
        _user_id.set(str(user_id))
    return rid


def get_request_id() -> str:
    return _request_id.get()


def clear_request_context() -> None:
    _request_id.set("-")
    _user_id.set("-")


class Timer:
    """Context manager that measures wall-clock duration in milliseconds.

    Used throughout the ingestion and retrieval paths so latency is measured
    rather than estimated.

    >>> with Timer() as t:
    ...     pass
    >>> t.ms >= 0
    True
    """

    __slots__ = ("_start", "ms")

    def __init__(self) -> None:
        self._start = 0.0
        self.ms = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.ms = round((time.perf_counter() - self._start) * 1000.0, 3)


def log_stage(logger: logging.Logger, stage: str, ms: float, **extra: Any) -> None:
    """Emit a uniform timing record for a named pipeline stage."""
    logger.info("stage.%s", stage, extra={"stage": stage, "duration_ms": ms, **extra})


def summarise(items: Iterable[Any], limit: int = 3) -> str:
    """Render a short, log-safe preview of an iterable."""
    values = [str(v) for v in list(items)[:limit]]
    return ", ".join(values)
