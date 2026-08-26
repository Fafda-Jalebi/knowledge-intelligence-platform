"""Application error taxonomy.

The API layer maps every :class:`AppError` onto an HTTP status code and a
stable machine-readable ``code``. Anything that is *not* an ``AppError`` is
treated as an unexpected fault: it is logged with a traceback and reported to
the client as a generic 500 so internal details never leak.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all expected, client-reportable failures."""

    status_code: int = 400
    code: str = "app_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return {"error": payload}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"


class AuthenticationError(AppError):
    status_code = 401
    code = "authentication_error"


class AuthorizationError(AppError):
    status_code = 403
    code = "authorization_error"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class UnsupportedMediaTypeError(AppError):
    status_code = 415
    code = "unsupported_media_type"


class RateLimitError(AppError):
    status_code = 429
    code = "rate_limited"


class ConfigurationError(AppError):
    """A provider or service is misconfigured (e.g. missing API key)."""

    status_code = 500
    code = "configuration_error"


class ProviderError(AppError):
    """An upstream model/vector provider failed."""

    status_code = 502
    code = "provider_error"


class ExtractionError(AppError):
    """A document could not be parsed into text."""

    status_code = 422
    code = "extraction_error"


class IngestionError(AppError):
    status_code = 500
    code = "ingestion_error"


class DependencyMissingError(ConfigurationError):
    """An optional Python package required by a selected provider is absent."""

    code = "dependency_missing"

    def __init__(self, package: str, *, provider: str, install: str | None = None) -> None:
        install = install or package
        super().__init__(
            f"Provider '{provider}' requires the optional package '{package}'. "
            f"Install it with: pip install {install}",
            details={"package": package, "provider": provider},
        )
