"""Run every doctest in ``kip`` as part of the self-check suite.

Doctests are the cheapest possible regression net for the small pure functions
that the rest of the platform is built on (tokenisation, path handling, size
formatting, redaction). They are only useful if they actually run, so they are
wired into ``python -m selfcheck`` rather than left to a separate command that
nobody remembers to invoke.

Discovery walks the ``kip`` package on disk rather than hard-coding a module
list, so a new module with doctests is covered the moment it is written.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil
from pathlib import Path

from selfcheck.harness import Harness

#: Modules that must not be imported during discovery. Anything requiring an
#: optional third-party package belongs here; there is nothing yet, and the
#: check below asserts the list stays honest.
SKIP: frozenset[str] = frozenset()


def iter_modules() -> list[str]:
    """Return every importable module name under ``kip``, in stable order."""
    import kip

    package_root = Path(kip.__file__).parent
    names = ["kip"]
    for info in pkgutil.walk_packages([str(package_root)], prefix="kip."):
        if info.name in SKIP:
            continue
        names.append(info.name)
    return sorted(names)


def run(verbose: bool = False) -> Harness:
    h = Harness(name="doctests", verbose=verbose)
    h.group("Doctest discovery")

    names = iter_modules()
    h.ok(len(names) >= 10, f"doctests: discovered {len(names)} modules under kip/")

    h.group("Doctest execution")
    attempted = 0
    covered = 0
    for name in names:
        module = h.no_raise(
            lambda name=name: importlib.import_module(name),
            f"doctests: {name} imports cleanly",
        )
        if module is None:
            continue
        finder = doctest.DocTestFinder()
        tests = [test for test in finder.find(module) if test.examples]
        if not tests:
            continue
        covered += 1
        runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS, verbose=False)
        for test in tests:
            runner.run(test, out=lambda _text: None)
        result = runner.summarize(verbose=False)
        attempted += result.attempted
        h.equal(
            result.failed,
            0,
            f"doctests: {name} ({result.attempted} example"
            f"{'' if result.attempted == 1 else 's'})",
        )

    h.ok(attempted >= 40, f"doctests: {attempted} examples executed across {covered} modules")
    return h


def main(argv: list[str] | None = None) -> int:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    harness = run(verbose="-v" in args or "--verbose" in args)
    print(harness.report())
    return 0 if harness.succeeded else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
