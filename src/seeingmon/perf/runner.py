"""Run cases, each in a fresh child process, and assemble the report.

**Isolation.** `run_case_in_child` starts `python -m seeingmon.perf._child <case>`, so the peak
memory of one case never includes the imports or the allocations of another. The child prints one
result line on its standard output, and the parent reads the last line that carries the marker, so
a case may print what it likes.

**One core.** The child gets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` set to
1 (and two more libraries that read their own variables), so NumPy and SciPy use one thread, and
every figure is per core. That is how the budgets read: a share of one core.

**Failures.** A case that raises, and a child that dies, give a `failed` result with a reason, and
the run goes on. A case that raises `SkipCase`, or whose module is missing, gives `skipped`.
The reasons contain no absolute path, because a report never records where the code lives.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from seeingmon.clock import Clock, SystemClock, utc_ns_to_iso
from seeingmon.perf.environment import collect_environment
from seeingmon.perf.load import system_busy_percent
from seeingmon.perf.memory import current_rss_bytes, peak_rss_bytes, process_cpu_ns
from seeingmon.perf.registry import Case, CaseContext, Registry, SkipCase, load_registry
from seeingmon.perf.report import CaseResult, Measurement, Report, Status

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
RESULT_MARKER = "SEEINGMON_PERF_RESULT "
DEFAULT_TIMEOUT_S = 1800.0
SMOKE_TIMEOUT_S = 300.0
_TRACEBACK_LINES = 8

Progress = Callable[[str], None]


def child_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment of a child: this one, with one thread for each math library.

    `extra` overrides any variable, for a test that needs `PYTHONPATH`.
    """
    env = dict(os.environ)
    for name in THREAD_VARIABLES:
        env[name] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra or {})
    return env


def _missing_reason(error: ModuleNotFoundError) -> str:
    name = error.name or "a module"
    if name.split(".")[0] == "seeingmon":
        return f"the module {name} is not on main yet"
    return f"the package {name} is not installed (install the extra that provides it)"


_PATH = re.compile(
    r"[A-Za-z]:[\\/][^\s'\"<>|]+"  # a Windows path with a drive letter
    r"|(?<![\w.])/(?:home|Users|mnt|tmp|var|run|root|opt)(?:/[^\s'\"<>|]*)?"  # a POSIX path
)


def scrub_paths(text: str) -> str:
    """Replace each absolute path in `text` with `<path>`, so that a report names no folder."""
    return _PATH.sub("<path>", text)


def _failure_reason(error: BaseException) -> tuple[str, tuple[str, ...]]:
    """A one-line reason, and the last frames of the traceback with file names and no paths."""
    reason = scrub_paths(f"{type(error).__name__}: {error}".strip())
    frames = traceback.extract_tb(error.__traceback__)[-_TRACEBACK_LINES:]
    lines = tuple(f"{Path(frame.filename).name}:{frame.lineno} in {frame.name}" for frame in frames)
    return reason, lines


def execute_case(case: Case, *, smoke: bool, busy_sample_s: float | None = None) -> CaseResult:
    """Run one case in this process and describe the outcome. It never raises.

    The result carries the peak memory of this process, so call it in a fresh process (see
    `run_case_in_child`) when the peak must belong to this case alone.
    """
    context = CaseContext(smoke)
    sample_s = busy_sample_s if busy_sample_s is not None else (0.05 if smoke else 0.25)
    busy = system_busy_percent(sample_s)
    start_rss = current_rss_bytes()
    wall_start = time.perf_counter()
    cpu_start = process_cpu_ns()
    status: Status = "ok"
    reason: str | None = None
    measurements: list[Measurement] = []
    notes_after: tuple[str, ...] = ()
    try:
        measurements = case.run(context)
    except SkipCase as skip:
        status, reason = "skipped", skip.reason
    except ModuleNotFoundError as error:
        status, reason = "skipped", _missing_reason(error)
    except Exception as error:  # a case that fails must not stop the run
        status = "failed"
        reason, notes_after = _failure_reason(error)
    return CaseResult(
        name=case.name,
        status=status,
        reason=reason or None,
        duration_s=time.perf_counter() - wall_start,
        cpu_s=(process_cpu_ns() - cpu_start) / 1e9,
        baseline_rss_bytes=context.baseline_rss_bytes or start_rss,
        peak_rss_bytes=peak_rss_bytes(),
        system_busy_percent=busy,
        measurements=tuple(measurements) if status == "ok" else (),
        notes=tuple(scrub_paths(note) for note in context.notes) + notes_after,
    )


def parse_child_output(stdout: str) -> CaseResult | None:
    """The result that a child printed, or `None` when it printed none that can be read."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                return CaseResult.from_dict(json.loads(line[len(RESULT_MARKER) :]))
            except ValueError:
                return None
    return None


def run_case_in_child(
    name: str,
    *,
    smoke: bool = False,
    timeout_s: float | None = None,
    registry_module: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    log: Progress | None = None,
) -> CaseResult:
    """Run the case `name` in a fresh child process and return its result.

    `registry_module` names a module whose `REGISTRY` holds the cases (the default is the
    harness's own). A child that exceeds `timeout_s`, exits without a result, or prints a result
    that cannot be read gives a `failed` result. `log` receives the last lines of the standard
    error of such a child.
    """
    limit = (
        timeout_s if timeout_s is not None else (SMOKE_TIMEOUT_S if smoke else DEFAULT_TIMEOUT_S)
    )
    command = [sys.executable, "-m", "seeingmon.perf._child", name]
    if smoke:
        command.append("--smoke")
    if registry_module is not None:
        command.extend(["--registry", registry_module])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_environment(extra_env),
            stdin=subprocess.DEVNULL,
            timeout=limit,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CaseResult(name, "failed", f"the child process did not finish in {limit:.0f} s")
    result = parse_child_output(completed.stdout)
    if result is not None:
        return result
    if log is not None:
        for line in completed.stderr.strip().splitlines()[-_TRACEBACK_LINES:]:
            log(f"  {name}: {line}")
    return CaseResult(
        name, "failed", f"the child process exited with code {completed.returncode} and no result"
    )


def run_cases(
    names: Sequence[str] | None = None,
    *,
    smoke: bool = False,
    label: str = "",
    isolate: bool = True,
    registry: Registry | None = None,
    registry_module: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
    progress: Progress | None = None,
    clock: Clock | None = None,
) -> Report:
    """Run the cases `names` (all of them by default) and return the report.

    With `isolate`, each case runs in a child process. Without it, the cases run here, which suits
    a test, and their peak memory is shared. `extra_env` adds variables to the environment of each
    child. Raises `UnknownCaseError` before it runs anything when a name is not registered.
    `progress` receives one line before and after each case.
    """
    if registry is None:
        registry = load_registry(registry_module)
    selected = list(names) if names else registry.names()
    cases = [registry.get(name) for name in selected]  # fail early for an unknown name
    say = progress or (lambda _line: None)
    results: list[CaseResult] = []
    for case in cases:
        say(f"{case.name}: running")
        if isolate:
            result = run_case_in_child(
                case.name,
                smoke=smoke,
                timeout_s=timeout_s,
                registry_module=registry_module,
                extra_env=extra_env,
                log=say,
            )
        else:
            result = execute_case(case, smoke=smoke)
        suffix = f" ({result.reason})" if result.reason else ""
        say(f"{case.name}: {result.status} in {result.duration_s:.1f} s{suffix}")
        results.append(result)
    stamp = utc_ns_to_iso((clock or SystemClock()).utc_ns(), digits=0)
    return Report(
        label=label,
        smoke=smoke,
        created_utc=stamp,
        environment=collect_environment(),
        cases=tuple(results),
    )
