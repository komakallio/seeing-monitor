"""The `core-sim` case: the `core` process against `acquire` with the simulator, a placeholder.

The entry function of the `core` process, `run_core` in `seeingmon.services.core.main` (the entry
of `seeingmon core`), is on `main`, and the case skips with the reason "the core process is on
main, and measure_core is not written yet". Before the entry function existed, the case skipped
with "the core process is not on main". The lead enables the case by writing one function,
`measure_core`, in this file:

1. Start `acquire` with the simulator (`seeingmon.perf.cases.ipc.AcquireProcess` shows how to run
   a process and read its output), and start `core` against it with a temporary data folder
   (delete the folder at the end).
2. Let both run for `ctx.pick(60.0, 2.0)` seconds, and read the CPU time and the peak memory of
   the `core` process. The `ipc` case reads the CPU time of a child process through a line
   protocol on its standard streams (`seeingmon.perf._acquire`), which works on Windows and
   Linux. Do the same for `core`, or have `core` print its own `process_cpu_ns()` on a signal.
3. Return these figures (the names and units are what the budgets read):

   - `cpu_share`, in `percent`, scale `interpreter`: the CPU time of `core` divided by the wall
     time of the run, in percent of one core. It adds to the 25% of the fast path, so the page
     `docs/performance.md` compares it with the sum of the fast-path shares of the `fastpath`
     case.
   - `peak_rss`, in `bytes`, scale `memory`: the peak memory of the `core` process. When this
     case runs, the memory budgets use it in place of the stand-in that the `fastpath` and
     `store` cases give.

Nothing else changes: the case is registered, the budgets look for the figures, and the test in
`tests/perf/test_cases.py` accepts a result or a skip.
"""

from __future__ import annotations

import importlib
import importlib.util

from seeingmon.perf.registry import REGISTRY, CaseContext, SkipCase
from seeingmon.perf.report import Measurement

CORE_MODULE = "seeingmon.services.core.main"
CORE_ENTRY = "run_core"


def core_process_exists() -> bool:
    """Whether the entry function of the `core` process is on `main`.

    The function imports the module of the entry only when the module exists, so the check costs
    nothing before the process lands.
    """
    try:
        found = importlib.util.find_spec(CORE_MODULE)
    except ModuleNotFoundError:  # a parent package is missing
        return False
    if found is None:
        return False
    return hasattr(importlib.import_module(CORE_MODULE), CORE_ENTRY)


def measure_core(ctx: CaseContext) -> list[Measurement]:
    """Measure the `core` process. Replace this body with the measurement (see the module text)."""
    raise SkipCase("the core process is on main, and measure_core is not written yet")


@REGISTRY.case("core-sim", summary="The core process against acquire with the simulator")
def core_sim(ctx: CaseContext) -> list[Measurement]:
    if not core_process_exists():
        raise SkipCase("the core process is not on main")
    return measure_core(ctx)
