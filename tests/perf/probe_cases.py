"""Cases for the tests of the runner. A child process loads them with `--registry`.

The registry is private to this module, so the cases never appear in the harness's own list.
"""

from __future__ import annotations

import importlib
import os

from seeingmon.perf.registry import CaseContext, Registry, SkipCase
from seeingmon.perf.report import Measurement

REGISTRY = Registry()

ALLOCATED_MB = 150
MESSAGE = "no such file: C:\\Users\\someone\\data\\frames.ser"  # repo-check: allow


@REGISTRY.case("probe-small", summary="Allocates nothing")
def small(ctx: CaseContext) -> list[Measurement]:
    ctx.mark_baseline()
    return [Measurement("answer", "count", 42.0)]


@REGISTRY.case("probe-alloc", summary="Holds a block of memory")
def allocate(ctx: CaseContext) -> list[Measurement]:
    ctx.mark_baseline()
    block = bytearray(b"\x01") * (ALLOCATED_MB * 1024 * 1024)  # touches every page
    return [Measurement("block", "bytes", float(len(block)))]


@REGISTRY.case("probe-threads", summary="Reports the thread variables of the child")
def threads(ctx: CaseContext) -> list[Measurement]:
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    return [
        Measurement(name.lower(), "threads", float(os.environ.get(name, "0"))) for name in names
    ]


@REGISTRY.case("probe-skip", summary="Skips with a reason")
def skip(ctx: CaseContext) -> list[Measurement]:
    raise SkipCase("the component is not on main")


@REGISTRY.case("probe-missing", summary="Needs a module that does not exist")
def missing(ctx: CaseContext) -> list[Measurement]:
    importlib.import_module("seeingmon.not_on_main_yet")
    return []


@REGISTRY.case("probe-fail", summary="Raises with a path in the message")
def fail(ctx: CaseContext) -> list[Measurement]:
    raise FileNotFoundError(MESSAGE)


@REGISTRY.case("probe-crash", summary="Ends the process without a result")
def crash(ctx: CaseContext) -> list[Measurement]:
    os._exit(3)


@REGISTRY.case("probe-noisy", summary="Prints before the result")
def noisy(ctx: CaseContext) -> list[Measurement]:
    print("some library prints a banner")
    print("SEEINGMON_PERF_RESULT this line is not a result")
    return [Measurement("value", "count", 1.0)]
