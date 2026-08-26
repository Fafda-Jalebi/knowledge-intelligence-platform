"""Zero-dependency check runner.

The full test suite uses pytest, but pytest is not always installed (and in a
restricted/offline environment it may not be installable). These self-checks
exercise the same behaviour using nothing but the standard library, so anyone who
clones the repository can verify the core immediately::

    python -m selfcheck            # run every check
    python -m selfcheck extraction # run one group

They are not a replacement for ``pytest``; they are a floor.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable


class CheckFailure(AssertionError):
    """Raised by the assertion helpers below."""


@dataclass
class Harness:
    """Tiny assertion recorder with a human-readable summary."""

    name: str
    passed: int = 0
    failures: list[str] = field(default_factory=list)
    verbose: bool = False
    _group: str = ""

    # -- structure ---------------------------------------------------------- #

    def group(self, title: str) -> None:
        self._group = title
        if self.verbose:
            print(f"\n  {title}")

    # -- assertions --------------------------------------------------------- #

    def ok(self, condition: Any, label: str) -> bool:
        if condition:
            self._pass(label)
            return True
        self._fail(label, "expected a truthy value")
        return False

    def equal(self, actual: Any, expected: Any, label: str) -> bool:
        if actual == expected:
            self._pass(label)
            return True
        self._fail(label, f"expected {expected!r}, got {actual!r}")
        return False

    def contains(self, haystack: Any, needle: Any, label: str) -> bool:
        try:
            hit = needle in haystack
        except TypeError as exc:  # pragma: no cover - defensive
            self._fail(label, f"not containable: {exc}")
            return False
        if hit:
            self._pass(label)
            return True
        preview = str(haystack)
        if len(preview) > 220:
            preview = preview[:220] + "..."
        self._fail(label, f"{needle!r} not found in {preview!r}")
        return False

    def between(self, value: float, low: float, high: float, label: str) -> bool:
        if low <= value <= high:
            self._pass(label)
            return True
        self._fail(label, f"expected {low} <= {value!r} <= {high}")
        return False

    def raises(self, exception: type[BaseException], call: Callable[[], Any], label: str) -> bool:
        try:
            call()
        except exception:
            self._pass(label)
            return True
        except Exception as exc:  # noqa: BLE001 - we want the actual type reported
            self._fail(label, f"expected {exception.__name__}, got {type(exc).__name__}: {exc}")
            return False
        self._fail(label, f"expected {exception.__name__}, nothing was raised")
        return False

    def no_raise(self, call: Callable[[], Any], label: str) -> Any:
        try:
            value = call()
        except Exception as exc:  # noqa: BLE001
            self._fail(label, f"raised {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            return None
        self._pass(label)
        return value

    # -- reporting ---------------------------------------------------------- #

    def _pass(self, label: str) -> None:
        self.passed += 1
        if self.verbose:
            print(f"    ok   {label}")

    def _fail(self, label: str, detail: str) -> None:
        prefix = f"[{self._group}] " if self._group else ""
        self.failures.append(f"{prefix}{label}: {detail}")
        if self.verbose:
            print(f"    FAIL {label}: {detail}")

    @property
    def total(self) -> int:
        return self.passed + len(self.failures)

    @property
    def succeeded(self) -> bool:
        return not self.failures

    def report(self) -> str:
        lines = [f"{self.name}: {self.passed}/{self.total} checks passed"]
        for failure in self.failures:
            lines.append(f"  FAIL {failure}")
        return "\n".join(lines)
