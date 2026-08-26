"""``python -m selfcheck`` entry point."""

from __future__ import annotations

import importlib
import sys
import time

from selfcheck import GROUPS


def main(argv: list[str]) -> int:
    verbose = any(flag in argv for flag in ("-v", "--verbose"))
    requested = [arg for arg in argv if not arg.startswith("-")] or list(GROUPS)

    unknown = [name for name in requested if name not in GROUPS]
    if unknown:
        print(f"unknown group(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(GROUPS)}", file=sys.stderr)
        return 2

    total = passed = 0
    failures: list[str] = []
    started = time.perf_counter()

    for name in requested:
        module = importlib.import_module(GROUPS[name])
        harness = module.run(verbose=verbose)
        print(harness.report())
        total += harness.total
        passed += harness.passed
        failures.extend(harness.failures)

    elapsed = (time.perf_counter() - started) * 1000
    print("-" * 60)
    print(f"total: {passed}/{total} checks passed in {elapsed:.0f} ms")
    if failures:
        print(f"{len(failures)} failure(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
