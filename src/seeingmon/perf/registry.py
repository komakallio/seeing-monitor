"""The case registry, and what a case receives when it runs.

A case is a function that takes a `CaseContext` and returns its `Measurement` figures. Register it
with the `case` decorator of `REGISTRY`:

    @REGISTRY.case("kernel", summary="Time per frame of the fast-path kernel")
    def kernel(ctx: CaseContext) -> list[Measurement]:
        from seeingmon.fastpath.kernel import measure_frame  # import the code under test here
        ...

**Import the code under test inside the case.** The harness then works at every commit: a case
whose component is not on `main` yet raises `SkipCase("the core process is not on main")`, or its
`ModuleNotFoundError` becomes that skip, and the other cases run on.

**Sizes.** `ctx.smoke` is true for `--smoke`, which shrinks every case to tens of milliseconds
of work. Take sizes with `ctx.pick(full, smoke)` and timers with `ctx.timer(...)`, which already
shrink in smoke mode.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from seeingmon.perf.memory import current_rss_bytes
from seeingmon.perf.report import Measurement
from seeingmon.perf.timing import Timer

T = TypeVar("T")

_NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


class SkipCase(Exception):  # noqa: N818 (a signal, not an error)
    """Raise it inside a case to skip the case. `reason` appears in the report."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class UnknownCaseError(KeyError):
    """The registry has no case with this name."""


class CaseContext:
    """What a case receives: the smoke flag, shrinking helpers, and a place for notes.

    `mark_baseline` records the resident size of the process at that moment. Call it after the
    imports and the setup, before the work, so that the report can tell the cost of the code from
    the cost of the workload.
    """

    def __init__(self, smoke: bool = False) -> None:
        self.smoke = smoke
        self.baseline_rss_bytes: int | None = None
        self.notes: list[str] = []

    def pick(self, full: T, smoke: T) -> T:
        """`full` for a normal run, and `smoke` for `--smoke`."""
        return smoke if self.smoke else full

    def timer(self, repeats: int, *, number: int = 1, warmup: int = 3) -> Timer:
        """A timer with these settings, or a tiny one in smoke mode (2 batches of at most 4)."""
        if self.smoke:
            return Timer(repeats=2, number=min(number, 4), warmup=1)
        return Timer(repeats=repeats, number=number, warmup=warmup)

    def mark_baseline(self) -> None:
        """Record the resident size of the process now as the baseline of the case."""
        self.baseline_rss_bytes = current_rss_bytes()

    def note(self, text: str) -> None:
        """Add a sentence to the report that explains how the case measured."""
        self.notes.append(text)


CaseFunction = Callable[[CaseContext], list[Measurement]]


@dataclass(frozen=True, slots=True)
class Case:
    """A registered case: its name, a one-line summary, and the function that runs it."""

    name: str
    summary: str
    run: CaseFunction


class Registry:
    """An ordered collection of cases. Cases run in the order of their registration."""

    def __init__(self) -> None:
        self._cases: dict[str, Case] = {}

    def case(self, name: str, *, summary: str) -> Callable[[CaseFunction], CaseFunction]:
        """Decorator that registers a function as the case `name`.

        Raises `ValueError` for a name that is not lowercase words joined by hyphens, and for a
        name that the registry already holds.
        """
        if not _NAME.fullmatch(name):
            raise ValueError(f"the case name {name!r} must be lowercase words joined by hyphens")

        def register(function: CaseFunction) -> CaseFunction:
            if name in self._cases:
                raise ValueError(f"the case {name!r} is registered twice")
            self._cases[name] = Case(name, summary, function)
            return function

        return register

    def get(self, name: str) -> Case:
        """The case with this name. Raises `UnknownCaseError` with the list of known names."""
        try:
            return self._cases[name]
        except KeyError:
            known = ", ".join(self._cases) or "none"
            raise UnknownCaseError(f"there is no case {name!r}; the cases are: {known}") from None

    def names(self) -> list[str]:
        """The names of all cases, in registration order."""
        return list(self._cases)

    def cases(self) -> list[Case]:
        return list(self._cases.values())

    def __contains__(self, name: object) -> bool:
        return name in self._cases

    def __len__(self) -> int:
        return len(self._cases)


REGISTRY = Registry()
"""The registry of the harness. `seeingmon.perf.cases` fills it when you import it."""


def load_registry(module: str | None = None) -> Registry:
    """The registry that holds the cases to run.

    With no argument, the function imports `seeingmon.perf.cases`, which registers the cases of the
    harness, and returns `REGISTRY`. Pass the name of a module to use its `REGISTRY` attribute
    instead, which lets a test run its own cases in a child process.
    """
    if module is None:
        importlib.import_module("seeingmon.perf.cases")
        return REGISTRY
    registry = getattr(importlib.import_module(module), "REGISTRY", None)
    if not isinstance(registry, Registry):
        raise TypeError(f"the module {module!r} has no REGISTRY that is a Registry")
    return registry
