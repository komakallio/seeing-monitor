"""The `day-sim` and `cloudy-sim` cases: what measuring all day and searching under clouds cost.

The system measures seeing whenever Polaris is visible, at any Sun elevation, and it searches for
Polaris whenever it does not see it (see `docs/visibility.md`). Two runs of the whole system,
read from outside as in the `core-sim` case (`seeingmon.perf.sysrun` has the method), measure what
that costs:

- **`day-sim`, a day of measuring.** The simulated clock starts at noon of midsummer at the
  synthetic site of `seeingmon dev`, with the Sun 58 degrees up. The scheduler searches, finds
  Polaris in two bursts, and measures with the fast exposure that the bright sky allows (about
  1.2 ms, against 2 ms in the dark). Each survey step takes its 1 ms frame and skips its long
  exposure, because the frames show that even the shortest long exposure would pass its target.
- **`cloudy-sim`, a cloudy night of searching.** The clock starts on a winter night, and an opaque
  overcast hides every star for the whole run. The scheduler searches: a burst of 50 frames every
  15 s, each frame through the three matched filters, and a survey step every 100 s, because the
  survey frames show clouds. The long exposure of the survey grows from 1 s by 4 times a step.

Both runs take the cycle of production: fast periods of two analysis windows of 60 s, and a
survey step every 180 s (every 100 s under clouds). A day at midsummer lasts 16 to 19 hours, and a
run cannot take that long, so each run measures a stretch of a few minutes that repeats through
the day or the night. Read the share of a core over the cycle and the peak memory of each
process. The share over the cycle covers whole cycles, from the end of one survey step to the end
of a later one (`whole_cycles`), so it weighs the periods, the steps, and the gaps as production
does. The figures carry the names of the `core-sim` case where they mean the same, so the budgets
read them the same way.

**The figures of a search.** The bursts take less than a second, so the samples, one a second,
split the search phase into intervals with the frames of a burst and gaps without frames. A
search frame costs `core` what the intervals with frames use beyond the load of the gaps,
divided by the frames (`core.search_frame_cost`): the receive, the matched filters, the median
of the frame for the next exposure, and a fiftieth of the start and the end of the stream of its
burst. `core.search_burst_share` is that cost at 98 frames per second, the share of a core that a
burst takes while it runs, which the budget compares with the 25% of the fast path.
`core.search_share` is the mean load of the search phase, bursts and gaps.

**What the runs cannot show.** The simulator's sensor reads the ambient temperature plus a
constant rise (the options `ambient_c` and `sensor_rise_c` of the simulator), so it cannot warm in
the sun: a sunlit camera runs warmer, and its dark current with it. The runs say nothing about the
sensor temperature in daylight. As in `core-sim`, the simulator renders the frames
inside `acquire`, so the CPU time and the peak memory of `acquire` include it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from seeingmon.perf.cases.core_sim import (
    Figures,
    add_facts,
    add_frame_cost,
    add_peaks,
    add_phase_shares,
    add_run_shares,
    core_process_exists,
)
from seeingmon.perf.registry import REGISTRY, CaseContext, SkipCase
from seeingmon.perf.report import DetailValue, Measurement
from seeingmon.perf.sysrun import (
    SMOKE_PLAN,
    RunPlan,
    Snapshot,
    SystemRun,
    run_system,
    split_search,
)

# Noon of midsummer at the synthetic site (55 degrees north): the Sun stands 58 degrees up.
DAY_START = "2026-06-21T12:00:00Z"
# A winter evening at the same site: the Sun is 27 degrees below the horizon at the start.
NIGHT_START = "2026-01-01T19:00:00Z"
OPAQUE = 0.0  # the transmission of the overcast: no star shows
# The cycle of production: `[fastpath] window_s` and `[scheduler.fast] window_s` of 120 s.
PRODUCTION_WINDOW_S = 60.0
PRODUCTION_FAST_WINDOWS = 2

DAY_PLAN = RunPlan(
    start=DAY_START,
    window_s=PRODUCTION_WINDOW_S,
    fast_windows=PRODUCTION_FAST_WINDOWS,
    fast_seconds=60.0,
    survey_steps=2,
)
CLOUDY_PLAN = RunPlan(
    start=NIGHT_START,
    overcast=OPAQUE,
    window_s=PRODUCTION_WINDOW_S,
    fast_windows=PRODUCTION_FAST_WINDOWS,
    fast_seconds=0.0,
    search_seconds=120.0,
    survey_steps=3,
)
# The smoke runs keep the windows of the launcher, and the search counts its first burst, so that
# a run ends as soon as one burst has a sample.
DAY_SMOKE_PLAN = replace(SMOKE_PLAN, start=DAY_START)
CLOUDY_SMOKE_PLAN = replace(
    SMOKE_PLAN,
    start=NIGHT_START,
    overcast=OPAQUE,
    fast_seconds=0.0,
    search_seconds=3.0,
    warmup_bursts=0,
)

# The survey step whose end starts the first whole cycle. A day's first cycle holds the search at
# the start of the run, and ends with step 1. On a cloudy night step 1 turns on the cycle for
# clouds, and the period after it starts at once, without its gap, so the first whole cycle starts
# where step 2 ends.
DAY_FIRST_STEP = 1
CLOUDY_FIRST_STEP = 2

STEADY_DAY = (
    "whole cycles of a day, from the end of the first survey step to the end of the last: gaps, "
    "fast periods, and survey steps"
)
STEADY_DAY_PART = (
    "the run from the start of measure, which holds more measure than a whole cycle, because the "
    "run ended before two survey steps"
)
STEADY_CLOUDY = (
    "whole cycles of a cloudy night, from the end of the second survey step to the end of the "
    "last: gaps, search periods, and survey steps"
)
STEADY_CLOUDY_PART = (
    "the run from the first survey frame that showed clouds, which holds more search than a whole "
    "cycle, because the run ended before three survey steps"
)


def steady_from(
    snapshots: Sequence[Snapshot], starts: Callable[[Snapshot], bool]
) -> tuple[Snapshot, ...]:
    """The samples from the first one for which `starts` holds, or none."""
    for index, snapshot in enumerate(snapshots):
        if starts(snapshot):
            return tuple(snapshots[index:])
    return ()


def whole_cycles(snapshots: Sequence[Snapshot], first_step: int) -> tuple[Snapshot, ...]:
    """The samples from the end of survey step `first_step` to the end of the last step.

    A cycle of the scheduler runs from the end of one survey step to the end of the next: the gap
    until the next slot, the fast or search period, and the step. Between the ends of two steps
    the samples therefore hold whole cycles, once the periods start at their slots. The samples
    end before the pause. Returns none when no step ended after step `first_step`.
    """
    active: list[Snapshot] = []
    for snapshot in snapshots:
        if snapshot.state == "paused":
            break
        active.append(snapshot)
    if not active or active[-1].survey_steps <= first_step:
        return ()
    last = active[-1].survey_steps
    start = next(i for i, s in enumerate(active) if s.survey_steps >= first_step)
    end = next(i for i, s in enumerate(active) if s.survey_steps >= last)
    return tuple(active[start : end + 1])


def _span(values: Sequence[float | int | None], digits: int = 1) -> str:
    """The range of the values that are known, as text, or `unknown`."""
    known = [float(value) for value in values if value is not None]
    if not known:
        return "unknown"
    low, high = min(known), max(known)
    if round(low, digits) == round(high, digits):
        return f"{low:.{digits}f}"
    return f"{low:.{digits}f} to {high:.{digits}f}"


def conditions(run: SystemRun) -> dict[str, DetailValue]:
    """The sky, the sensor, and what the scheduler did, as facts of the length of the run."""
    active = [snapshot for snapshot in run.snapshots if snapshot.state != "paused"]
    exposures = {
        purpose: [s.exposure_us for s in active if s.purpose == purpose]
        for purpose in ("fast", "search")
    }
    counters = run.counters
    facts: dict[str, DetailValue] = {
        "start_utc": run.plan.start or "the default of seeingmon dev",
        "sun_elevation_deg": _span([s.sun_elevation_deg for s in active]),
        "sensor_temperature_c": _span([s.sensor_temperature_c for s in active]),
        "fast_exposure_us": _span(exposures["fast"], 0),
        "search_exposure_us": _span(exposures["search"], 0),
        "window_s": run.plan.window_s or "the default of seeingmon dev",
        "fast_windows": run.plan.fast_windows or "the default of seeingmon dev",
        "cloud_cycle": "yes" if any(s.cloud for s in active) else "no",
    }
    if run.plan.overcast is not None:
        facts["overcast_transmission"] = run.plan.overcast
    for name in (
        "measure_starts",
        "search_bursts",
        "search_frames",
        "detections",
        "survey_long_skips",
        "survey_unsolved",
        "survey_skipped",
        "cadence_overruns",
        "dropped",
    ):
        facts[name] = counters.get(name, 0)
    return facts


def day_measurements(run: SystemRun) -> list[Measurement]:
    """The figures of a day of measuring, in the names of the `core-sim` case.

    The shares over the cycle cover whole cycles after the first survey step, so the search at the
    start of the run, which a day does once, stays out of them. A run with fewer steps, such as a
    smoke run, gives the shares from the start of measure, and the figures say so.
    """
    figures = Figures()
    add_peaks(figures, run)
    add_phase_shares(figures, run.fast, run.idle, "fast")
    steady, covers = whole_cycles(run.snapshots, DAY_FIRST_STEP), STEADY_DAY
    if not steady:
        steady = steady_from(run.snapshots, lambda s: s.phase == "fast")
        covers = STEADY_DAY_PART
    add_run_shares(figures, steady, covers)
    add_frame_cost(figures, run.fast, run.idle, ("core.frame_cost", "core.fastpath_receive_share"))
    add_facts(figures, run, extra=conditions(run))
    return figures.items


def cloudy_measurements(run: SystemRun) -> list[Measurement]:
    """The figures of a cloudy night of searching. See the module text for the search figures.

    The shares over the cycle cover whole cycles for clouds, from the end of the second survey
    step: the first cycle of the run, before any survey frame showed clouds, has the period of a
    clear night, and the second starts late. A run with fewer steps gives the shares from the
    first sample of the cycle for clouds, and the figures say so. Without a gap in the search
    phase, the cost of a search frame counts from the paused scheduler.
    """
    figures = Figures()
    add_peaks(figures, run)
    add_phase_shares(figures, run.search, run.idle, "search")
    steady, covers = whole_cycles(run.snapshots, CLOUDY_FIRST_STEP), STEADY_CLOUDY
    if not steady:
        steady = steady_from(run.snapshots, lambda s: s.cloud)
        covers = STEADY_CLOUDY_PART
    add_run_shares(figures, steady, covers)
    bursts, gaps = split_search(run.snapshots, warmup_bursts=run.plan.warmup_bursts)
    # A run as short as a smoke run can end without a gap, and the paused scheduler then gives the
    # load beyond which a burst costs.
    quiet = gaps if gaps.seconds > 0 else run.idle
    add_frame_cost(
        figures, bursts, quiet, ("core.search_frame_cost", "core.search_burst_share"), "burst"
    )
    add_facts(
        figures,
        run,
        busy=run.search,
        phase="search",
        extra={"gap_s": round(gaps.seconds, 1), **conditions(run)},
    )
    return figures.items


def _notes(ctx: CaseContext, run: SystemRun, busy: str, summary: str) -> None:
    plan = run.plan
    facts = conditions(run)
    sentences = [
        summary,
        f"The system ran from the plan of seeingmon dev with the {plan.sensor} sensor at speed "
        f"{plan.speed:g}, from {facts['start_utc']}.",
        f"The processes needed {run.startup_s:.0f} s to start, and the sampling lasted "
        f"{run.sampling_s:.0f} s with one sample every {plan.sample_interval_s:g} s.",
        busy,
        f"{run.survey_steps} survey steps ran, {facts['survey_long_skips']} of them without their "
        f"long exposure, and the worker analyzed {run.survey_results} survey frames.",
        f"The Sun stood at {facts['sun_elevation_deg']} degrees, and the sensor read "
        f"{facts['sensor_temperature_c']} C: the simulator's sensor is the ambient temperature "
        "plus a constant rise, so it cannot warm in the sun.",
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


def measure_day(ctx: CaseContext) -> list[Measurement]:
    """Run a day of measuring, read it from outside, and return the figures."""
    plan = DAY_SMOKE_PLAN if ctx.smoke else DAY_PLAN
    ctx.mark_baseline()
    run = run_system(plan)
    fast = run.fast
    _notes(
        ctx,
        run,
        f"The fast phase has {fast.seconds:.0f} s ({fast.frames} frames, {fast.fps:.1f} frames "
        f"per second) at an exposure of {conditions(run)['fast_exposure_us']} us, and the idle "
        f"phase has {run.idle.seconds:.0f} s with the scheduler paused.",
        "A day of measuring: the Sun is up, Polaris is visible, and the scheduler measures.",
    )
    return day_measurements(run)


def measure_cloudy(ctx: CaseContext) -> list[Measurement]:
    """Run a cloudy night of searching, read it from outside, and return the figures."""
    plan = CLOUDY_SMOKE_PLAN if ctx.smoke else CLOUDY_PLAN
    ctx.mark_baseline()
    run = run_system(plan)
    search = run.search
    bursts, _ = split_search(run.snapshots, warmup_bursts=plan.warmup_bursts)
    _notes(
        ctx,
        run,
        f"The search phase has {search.seconds:.0f} s ({search.frames} frames of "
        f"{run.counters.get('search_bursts', 0)} bursts in the run, {bursts.seconds:.0f} s of "
        f"intervals with frames), and the idle phase has {run.idle.seconds:.0f} s with the "
        "scheduler paused.",
        "A cloudy night of searching: an opaque overcast hides Polaris, and the scheduler "
        "searches.",
    )
    return cloudy_measurements(run)


@REGISTRY.case("day-sim", summary="A simulated day of measuring, the whole system from outside")
def day_sim(ctx: CaseContext) -> list[Measurement]:
    if not core_process_exists():
        raise SkipCase("the core process is not on main")
    return measure_day(ctx)


@REGISTRY.case(
    "cloudy-sim", summary="A simulated cloudy night of searching, the whole system from outside"
)
def cloudy_sim(ctx: CaseContext) -> list[Measurement]:
    if not core_process_exists():
        raise SkipCase("the core process is not on main")
    return measure_cloudy(ctx)
