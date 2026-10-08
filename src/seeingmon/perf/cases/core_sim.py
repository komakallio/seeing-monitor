"""The `core-sim` case: the whole system on the simulated sky, measured from outside.

The case starts `acquire` (with the simulator), `core`, and `web` from the plan of `seeingmon dev`,
reads them from outside for a few minutes, and stops them. `seeingmon.perf.sysrun` has the method.
The figures replace the stand-ins of the memory budget, and they give the fast path and the receive
a measured share of a core.

**The system.** The sensor is the reference sensor (`full`), so the survey frames have the real
bin2 size (4144 x 2822 pixels) and the survey worker peaks where it peaks on a real night. The
clock runs at the speed of real time. The fast stream takes the fast mode of the architecture (an
exposure of 2 ms, which the readout of the sensor stretches to about 82 frames per second) and the
real Polaris. The windows are those of the dev launcher (20 s). The run waits for two survey
steps (a short and a long exposure each), which takes two cycles of a few minutes, and it
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
from collections.abc import Mapping, Sequence

from seeingmon.perf.registry import REGISTRY, CaseContext, SkipCase
from seeingmon.perf.report import DetailValue, Measurement
from seeingmon.perf.sysrun import (
    FULL_PLAN,
    SMOKE_PLAN,
    Phase,
    Snapshot,
    SystemRun,
    cost_per_frame_us,
    run_share,
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


class Figures:
    """The figures of a run in the order that they come, each at least a tiny positive floor."""

    def __init__(self) -> None:
        self.items: list[Measurement] = []

    def add(
        self, name: str, unit: str, value: float, scale: str, detail: dict[str, DetailValue]
    ) -> None:
        self.items.append(Measurement(name, unit, max(value, _FLOOR), None, scale, detail))


def _including(role: str) -> dict[str, DetailValue]:
    return {"includes": "the simulator"} if role == "acquire" else {}


def add_peaks(figures: Figures, run: SystemRun) -> None:
    """The peak memory of each process, and their sum."""
    for role in ("core", "survey_worker", "web", "acquire"):
        peak = run.peaks.get(role)
        if peak:
            figures.add(
                f"{role}.peak_rss",
                "bytes",
                float(peak),
                "memory",
                {"process": role, **_including(role)},
            )
    figures.add(
        "other.peak_rss",
        "bytes",
        float(run.peaks.get("other", 0)),
        "memory",
        {"process": "the other children of core, such as the resource tracker"},
    )
    if run.peaks:
        parts: dict[str, DetailValue] = {
            role: f"{peak / 1e6:.0f} MB" for role, peak in sorted(run.peaks.items())
        }
        figures.add(
            "all.peak_rss_sum",
            "bytes",
            float(sum(run.peaks.values())),
            "memory",
            {**parts, "includes": "the simulator in acquire"},
        )


def add_phase_shares(figures: Figures, busy: Phase, idle: Phase, phase: str) -> None:
    """The share of a core of each process in a busy phase and when idle, and its resident size.

    `phase` names the busy phase (`fast` or `search`) in the names of the figures.
    """
    for role in ("core", "acquire", "web"):
        share, quiet = busy.share(role), idle.share(role)
        if share is None or quiet is None:
            continue
        detail: dict[str, DetailValue] = {
            f"{phase}_seconds": round(busy.seconds, 1),
            "idle_seconds": round(idle.seconds, 1),
            **_including(role),
        }
        threads_busy = busy.top_threads(role)
        if threads_busy:
            detail[f"threads_{phase}"] = _threads_text(threads_busy)
            detail["threads_idle"] = _threads_text(idle.top_threads(role))
        figures.add(f"{role}.{phase}_share", "percent", share, "interpreter", detail)
        figures.add(f"{role}.idle_share", "percent", quiet, "interpreter", dict(detail))
        typical = busy.resident_median(role)
        if typical is not None:
            largest = busy.resident_max(role) or typical
            figures.add(
                f"{role}.rss_{phase}",
                "bytes",
                float(typical),
                "memory",
                {f"largest_in_{phase}_phase": f"{largest / 1e6:.0f} MB", **_including(role)},
            )


RUN_COVERS = "the whole cycle: fast stream, survey steps, and gaps"


def add_run_shares(
    figures: Figures,
    snapshots: Sequence[Snapshot],
    covers: str = RUN_COVERS,
) -> None:
    """The share of a core of each process over the samples up to the pause.

    `covers` says in words what the samples cover, such as the whole cycle of the scheduler.
    """
    for role in ("core", "acquire", "web"):
        average = run_share(snapshots, role)
        if average is not None:
            figures.add(
                f"{role}.run_share",
                "percent",
                average,
                "interpreter",
                {"covers": covers, **_including(role)},
            )


def add_frame_cost(
    figures: Figures, busy: Phase, quiet: Phase, names: tuple[str, str], phase: str = "fast"
) -> None:
    """What a frame costs `core`, and that cost as a share of a core at 98 frames per second.

    `names` names the two figures. The cost is the share of `core` in the busy phase beyond its
    share in the quiet one, divided by the frame rate of the busy phase, which `phase` names.
    """
    name, share = names
    cost = cost_per_frame_us(busy, quiet, "core")
    if cost is None:
        return
    facts: dict[str, DetailValue] = {
        "frames": busy.frames,
        "fps": round(busy.fps, 1),
        f"{phase}_seconds": round(busy.seconds, 1),
    }
    figures.add(name, "us/frame", cost, "interpreter", facts)
    figures.add(
        share,
        "percent",
        cost * BUDGET_FPS / 1e4,
        "interpreter",
        {**facts, "share_at_hz": BUDGET_FPS},
    )


def add_facts(
    figures: Figures,
    run: SystemRun,
    *,
    busy: Phase | None = None,
    phase: str = "fast",
    extra: Mapping[str, DetailValue] | None = None,
) -> None:
    """The survey worker, the length of the run with the facts of its method, and the load.

    `busy` is the phase whose frame rate the run reports, the fast phase unless you pass another,
    and `phase` names it. `extra` adds facts to the length of the run.
    """
    fast = run.fast if busy is None else busy
    idle = run.idle
    if run.worker_cpu_ns:
        figures.add(
            "survey_worker.cpu",
            "s",
            run.worker_cpu_ns / 1e9,
            "interpreter",
            {"survey_steps": run.survey_steps, "survey_results": run.survey_results},
        )
    figures.add("run.frame_rate", "frames/s", fast.fps, "none", {"frames": fast.frames})
    stream = {key: str(value) for key, value in run.stream.items()}
    figures.add(
        "run.length",
        "s",
        run.sampling_s,
        "none",
        {
            "startup_s": round(run.startup_s, 1),
            "sample_interval_s": run.plan.sample_interval_s,
            f"{phase}_s": round(fast.seconds, 1),
            "idle_s": round(idle.seconds, 1),
            "survey_steps": run.survey_steps,
            "survey_results": run.survey_results,
            "sensor": run.plan.sensor,
            "speed": run.plan.speed,
            "samples": len(run.snapshots),
            **stream,
            **(extra or {}),
        },
    )
    if run.machine_busy_percent is not None:
        figures.add(
            "run.machine_busy",
            "percent",
            run.machine_busy_percent,
            "none",
            {
                "system_share_percent": round(run.own_busy_percent or 0.0, 2),
                "logical_cpus": run.logical_cpus,
            },
        )


def measurements_from(run: SystemRun) -> list[Measurement]:
    """The figures of a run, in the names that the budgets read."""
    figures = Figures()
    add_peaks(figures, run)
    add_phase_shares(figures, run.fast, run.idle, "fast")
    add_run_shares(figures, run.snapshots)
    add_frame_cost(figures, run.fast, run.idle, ("core.frame_cost", "core.fastpath_receive_share"))
    add_facts(figures, run)
    return figures.items


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
        f"The camera reads wait {plan.read_timeout_margin_s:g} s beyond the frame period, because "
        "the simulator renders a survey frame inside the read.",
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
