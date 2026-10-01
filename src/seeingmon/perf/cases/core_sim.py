"""The `core-sim` case: the whole system on the simulated sky, measured from outside.

The case starts `acquire` (with the simulator), `core`, and `web` from the plan of `seeingmon dev`,
reads them from outside for a few minutes, and stops them. `seeingmon.perf.sysrun` has the method.
The figures replace the stand-ins of the memory budget, and they give the fast path and the receive
a measured share of a core.

**The system.** The sensor is the reference sensor (`full`), so the survey frames have the real
bin2 size (4144 x 2822 pixels) and the survey worker peaks where it peaks on a real night. The
clock runs at the speed of real time. The fast stream takes the fast mode of the architecture (an
exposure of 2 ms, which the readout of the sensor stretches to about 88 frames per second) and the
real Polaris. The windows are those of the dev launcher (20 s). The run waits for two survey
steps (a short and a long exposure each), which takes the survey cadence of 3 minutes, and it
pauses the scheduler at the end to read the load that does not belong to the frames. In `--smoke`
mode the run uses the `small` sensor and takes a few seconds of sampling after the start of the
processes.

**What the figures include.**

- The simulator renders each frame inside `acquire`, and its cost is several times the cost of
  `acquire` itself. The CPU time of `acquire` here and its peak memory include the simulator, so
  the budgets keep reading `acquire` from the `ipc` case, which uses prebuilt frames.
- The peak of `core` and of the survey worker are the real ones: neither process runs the simulator.
- `all.peak_rss_sum` adds the peak of each process. The peaks do not fall at the same moment, so the
  sum is an upper bound of the resident memory at any moment, and it includes the simulator.
- One client polls `web` every 5 s, as an open page would. `web` serves nobody else.
- The load of `core` that does not belong to the frames is the scheduler, the store, the health
  timer, and the answers to `web`. It is the CPU time of `core` with the scheduler paused (the
  figure `core.idle_share`), and the cost of a frame is the difference of the two shares divided by
  the frame rate.

**The machine.** Other work changes the figures, so the case records the load of the machine during
the run (`run.machine_busy`) and the share that the system itself used. On Linux in a virtual
machine, the load covers the virtual machine only.
"""

from __future__ import annotations

import importlib
import importlib.util

from seeingmon.perf.registry import REGISTRY, CaseContext, SkipCase
from seeingmon.perf.report import Measurement
from seeingmon.perf.sysrun import (
    FULL_PLAN,
    SMOKE_PLAN,
    SystemRun,
    cost_per_frame_us,
    run_system,
)

CORE_MODULE = "seeingmon.services.core.main"
CORE_ENTRY = "run_core"
BUDGET_FPS = 98.0  # the rate of the budgets: a share is the cost of a frame times this rate
_FLOOR = 1e-6


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


def _threads_text(threads: list[tuple[int, float]]) -> str:
    return ", ".join(f"{tid}:{share:.1f}%" for tid, share in threads)


def measurements_from(run: SystemRun) -> list[Measurement]:
    """The figures of a run, in the names that the budgets read."""
    fast, idle = run.fast, run.idle
    out: list[Measurement] = []

    def add(
        name: str, unit: str, value: float, scale: str, detail: dict[str, float | int | str]
    ) -> None:
        out.append(Measurement(name, unit, max(value, _FLOOR), None, scale, detail))

    for role in ("core", "survey_worker", "web", "acquire"):
        peak = run.peaks.get(role)
        if peak:
            including = {"includes": "the simulator"} if role == "acquire" else {}
            add(f"{role}.peak_rss", "bytes", float(peak), "memory", {"process": role, **including})
    add(
        "other.peak_rss",
        "bytes",
        float(run.peaks.get("other", 0)),
        "memory",
        {"process": "the other children of core, such as the resource tracker"},
    )
    if run.peaks:
        parts = {role: f"{peak / 1e6:.0f} MB" for role, peak in sorted(run.peaks.items())}
        add(
            "all.peak_rss_sum",
            "bytes",
            float(sum(run.peaks.values())),
            "memory",
            {**parts, "includes": "the simulator in acquire"},
        )

    for role in ("core", "acquire", "web"):
        busy, quiet = fast.share(role), idle.share(role)
        if busy is None or quiet is None:
            continue
        including = {"includes": "the simulator"} if role == "acquire" else {}
        detail: dict[str, float | int | str] = {
            "fast_seconds": round(fast.seconds, 1),
            "idle_seconds": round(idle.seconds, 1),
            **including,
        }
        threads_fast = fast.top_threads(role)
        if threads_fast:
            detail["threads_fast"] = _threads_text(threads_fast)
            detail["threads_idle"] = _threads_text(idle.top_threads(role))
        add(f"{role}.fast_share", "percent", busy, "interpreter", detail)
        add(f"{role}.idle_share", "percent", quiet, "interpreter", dict(detail))

    cost = cost_per_frame_us(fast, idle, "core")
    if cost is not None:
        facts: dict[str, float | int | str] = {
            "frames": fast.frames,
            "fps": round(fast.fps, 1),
            "fast_seconds": round(fast.seconds, 1),
        }
        add("core.frame_cost", "us/frame", cost, "interpreter", facts)
        add(
            "core.fastpath_receive_share",
            "percent",
            cost * BUDGET_FPS / 1e4,
            "interpreter",
            {**facts, "share_at_hz": BUDGET_FPS},
        )

    if run.worker_cpu_ns:
        add(
            "survey_worker.cpu",
            "s",
            run.worker_cpu_ns / 1e9,
            "interpreter",
            {"survey_steps": run.survey_steps, "survey_results": run.survey_results},
        )
    add("run.frame_rate", "frames/s", fast.fps, "none", {"frames": fast.frames})
    stream = {key: str(value) for key, value in run.stream.items()}
    add(
        "run.length",
        "s",
        run.sampling_s,
        "none",
        {
            "startup_s": round(run.startup_s, 1),
            "sample_interval_s": run.plan.sample_interval_s,
            "fast_s": round(fast.seconds, 1),
            "idle_s": round(idle.seconds, 1),
            "survey_steps": run.survey_steps,
            "survey_results": run.survey_results,
            "sensor": run.plan.sensor,
            "speed": run.plan.speed,
            "samples": len(run.snapshots),
            **stream,
        },
    )
    if run.machine_busy_percent is not None:
        add(
            "run.machine_busy",
            "percent",
            run.machine_busy_percent,
            "none",
            {
                "system_share_percent": round(run.own_busy_percent or 0.0, 2),
                "logical_cpus": run.logical_cpus,
            },
        )
    return out


def measure_core(ctx: CaseContext) -> list[Measurement]:
    """Run the system, read it from outside, and return the figures. See the module text."""
    plan = SMOKE_PLAN if ctx.smoke else FULL_PLAN
    ctx.mark_baseline()
    run = run_system(plan)
    fast, idle = run.fast, run.idle
    sentences = [
        f"The system ran from the plan of seeingmon dev with the {plan.sensor} sensor at speed "
        f"{plan.speed:g}.",
        f"The processes needed {run.startup_s:.0f} s to start, and the sampling lasted "
        f"{run.sampling_s:.0f} s with one sample every {plan.sample_interval_s:g} s.",
        f"The fast phase has {fast.seconds:.0f} s ({fast.frames} frames, {fast.fps:.1f} frames "
        f"per second), and the idle phase has {idle.seconds:.0f} s with the scheduler paused.",
        f"{run.survey_steps} survey steps ran, and the worker analyzed {run.survey_results} "
        "survey frames.",
        "The simulator renders inside acquire, so the CPU time and the peak memory of acquire "
        "include it, and the budgets read acquire from the ipc case.",
        f"One client polled web every {plan.poll_interval_s:g} s.",
    ]
    if run.machine_busy_percent is not None:
        sentences.append(
            f"The machine was {run.machine_busy_percent:.0f}% busy during the run, and the system "
            f"itself used {run.own_busy_percent or 0:.1f}% of it."
        )
    ctx.note(" ".join(sentences))
    for note in run.notes:
        ctx.note(note)
    return measurements_from(run)


@REGISTRY.case("core-sim", summary="The whole system on the simulated sky, measured from outside")
def core_sim(ctx: CaseContext) -> list[Measurement]:
    if not core_process_exists():
        raise SkipCase("the core process is not on main")
    return measure_core(ctx)
