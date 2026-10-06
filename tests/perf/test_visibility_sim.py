"""The `day-sim` and `cloudy-sim` cases: the plans, the figures of a run, and the budgets.

The tests that start the system are the smoke test in `test_cases.py`, which runs every case, and
the slow tests at the end of this file. The others feed the cases with made-up runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from seeingmon.clock import iso_to_utc_ns
from seeingmon.perf.budgets import evaluate
from seeingmon.perf.cases import core_sim, visibility_sim
from seeingmon.perf.cases.visibility_sim import (
    CLOUDY_PLAN,
    CLOUDY_SMOKE_PLAN,
    DAY_PLAN,
    DAY_SMOKE_PLAN,
    DAY_START,
    NIGHT_START,
    cloudy_measurements,
    conditions,
    day_measurements,
    steady_from,
    whole_cycles,
)
from seeingmon.perf.registry import REGISTRY
from seeingmon.perf.report import Measurement, Report
from seeingmon.perf.runner import execute_case
from seeingmon.perf.sysrun import (
    SMOKE_PLAN,
    RunPlan,
    SystemRun,
    run_system,
    search_phase,
    split_phases,
)
from seeingmon.scheduler.ephemeris import sun_elevation_deg
from seeingmon.services.dev import SIM_LATITUDE_DEG, SIM_LONGITUDE_DEG

from .helpers import MB, Timeline, case, fixture_report, with_case


def made_up_run(
    clock: Timeline, plan: RunPlan, counters: dict[str, int], *, warmup_windows: int = 0
) -> SystemRun:
    """A run from samples, with round peaks and facts."""
    samples = clock.samples
    fast, idle = split_phases(samples, warmup_windows=warmup_windows, min_fps=1.0)
    return SystemRun(
        plan=plan,
        fast=fast,
        idle=idle,
        peaks={"acquire": 140 * MB, "core": 150 * MB, "web": 85 * MB, "survey_worker": 300 * MB},
        snapshots=tuple(samples),
        startup_s=9.0,
        sampling_s=samples[-1].t - samples[0].t,
        survey_steps=counters.get("survey_steps", 0),
        survey_results=counters.get("survey_results", 0),
        stream={"mode": "bin1_128x128_u16", "exposure_us": 1200, "gain": 0},
        worker_cpu_ns=2_000_000_000,
        logical_cpus=8,
        machine_busy_percent=10.0,
        own_busy_percent=1.0,
        search=search_phase(samples, warmup_bursts=plan.warmup_bursts),
        counters=counters,
    )


def day_run() -> SystemRun:
    """A day: two bursts, then 4 s of measure at 80 fps, a survey step, and a pause.

    In measure `core` uses 40 ms a second (4% of a core), and paused 10 ms (1%), so a frame costs
    375 us. The search at the start uses 20 ms a second, and the shares over the cycle leave it out.
    """
    sky: dict[str, Any] = {"sun_deg": 58.4, "temperature_c": 19.0}
    clock = Timeline()
    clock.repeat(2, purpose="search", phase="search", bursts=2, exposure_us=1185, core_ms=20, **sky)
    measure: dict[str, Any] = {"purpose": "fast", "phase": "fast", "bursts": 2, **sky}
    clock.repeat(4, frames=80, core_ms=40, exposure_us=1196, rss_mb={"core": 140}, **measure)
    clock.tick(purpose="watch", phase="survey_short", steps=1, exposure_us=32, core_ms=10, **sky)
    clock.repeat(3, state="paused", purpose=None, phase="paused", steps=1, core_ms=10, **sky)
    counters = {"survey_steps": 1, "survey_results": 1, "survey_long_skips": 1, "measure_starts": 1}
    return made_up_run(clock, DAY_SMOKE_PLAN, counters)


def two_step_day() -> SystemRun:
    """A day with two survey steps: a whole cycle lies between their ends.

    The search, 2 s of measure, and step 1 come first. The cycle then holds a gap of 2 s at 5 ms
    of `core` a second, 4 s of measure at 40 ms, and step 2 at 10 ms. The wait for the result of
    step 2 follows, and the pause.
    """
    sky: dict[str, Any] = {"sun_deg": 58.4, "temperature_c": 19.0}
    clock = Timeline()
    clock.repeat(2, purpose="search", phase="search", bursts=2, exposure_us=1185, core_ms=20, **sky)
    measure: dict[str, Any] = {"purpose": "fast", "phase": "fast", "bursts": 2, **sky}
    clock.repeat(2, frames=80, core_ms=40, exposure_us=1196, **measure)
    step: dict[str, Any] = {"purpose": "watch", "phase": "survey_short", "core_ms": 10, **sky}
    clock.tick(steps=1, pending=1, **step)
    clock.repeat(2, purpose="watch", phase="gap", steps=1, core_ms=5, **sky)
    clock.repeat(4, frames=80, core_ms=40, steps=1, exposure_us=1196, **measure)
    clock.tick(steps=2, pending=1, **step)
    clock.tick(purpose="watch", phase="gap", steps=2, core_ms=5, **sky)
    clock.repeat(3, state="paused", purpose=None, phase="paused", steps=2, core_ms=5, **sky)
    counters = {"survey_steps": 2, "survey_results": 2, "survey_long_skips": 2, "measure_starts": 1}
    return made_up_run(clock, DAY_PLAN, counters)


def cloudy_run(*, with_gaps: bool = True) -> SystemRun:
    """A cloudy night: bursts and gaps, a survey step that turns on the cycle for clouds, a pause.

    A gap uses 8 ms of `core` a second and an interval with a burst of 50 frames 48 ms, so a search
    frame costs 800 us. Paused, `core` uses 5 ms a second.
    """
    sky: dict[str, Any] = {"sun_deg": -26.6, "temperature_c": 19.0, "exposure_us": 2000}
    search: dict[str, Any] = {"purpose": "search", "phase": "search", **sky}
    clock = Timeline()
    clock.tick(core_ms=8, **search)
    clock.tick(frames=50, bursts=1, core_ms=48, **search)
    if with_gaps:
        clock.repeat(3, bursts=1, core_ms=8, **search)
    clock.tick(purpose="survey", phase="survey_long", bursts=1, pending=1, core_ms=8, **sky)
    clock.tick(steps=1, bursts=2, frames=50, core_ms=48, cloud=True, **search)
    if with_gaps:
        clock.repeat(3, steps=1, bursts=2, core_ms=8, cloud=True, **search)
    clock.tick(steps=1, bursts=3, frames=50, core_ms=48, cloud=True, **search)
    clock.repeat(
        3, state="paused", purpose=None, phase="paused", steps=1, bursts=3, core_ms=5, cloud=True
    )
    counters = {"survey_steps": 1, "survey_results": 2, "search_bursts": 3, "search_frames": 150}
    return made_up_run(clock, CLOUDY_SMOKE_PLAN, counters)


def figures(found: list[Measurement]) -> dict[str, Measurement]:
    return {item.name: item for item in found}


class TestThePlans:
    def test_the_day_starts_with_the_sun_high_and_the_night_in_the_dark(self) -> None:
        day = sun_elevation_deg(iso_to_utc_ns(DAY_START), SIM_LATITUDE_DEG, SIM_LONGITUDE_DEG)
        night = sun_elevation_deg(iso_to_utc_ns(NIGHT_START), SIM_LATITUDE_DEG, SIM_LONGITUDE_DEG)
        assert day > 55.0  # noon of midsummer at 55 degrees north: 58.4 degrees
        assert night < -18.0  # and it stays below for hours: the evening is long past dusk

    def test_both_runs_take_the_cycle_of_production(self) -> None:
        for plan in (DAY_PLAN, CLOUDY_PLAN):
            assert (plan.window_s, plan.fast_windows) == (60.0, 2)
            assert plan.sensor == "full"
            assert plan.speed == 1.0

    def test_the_day_measures_and_the_cloudy_night_searches_under_an_opaque_overcast(self) -> None:
        assert DAY_PLAN.overcast is None
        assert (DAY_PLAN.fast_seconds, DAY_PLAN.search_seconds) == (60.0, 0.0)
        assert CLOUDY_PLAN.overcast == 0.0
        assert CLOUDY_PLAN.fast_seconds == 0.0
        assert CLOUDY_PLAN.search_seconds >= 60.0
        assert CLOUDY_PLAN.survey_steps >= 2  # past the first step, which turns on the cycle

    def test_the_smoke_plans_are_the_smoke_plan_of_core_sim_with_the_sky_of_each_run(self) -> None:
        assert DAY_SMOKE_PLAN.start == DAY_START
        assert (DAY_SMOKE_PLAN.sensor, DAY_SMOKE_PLAN.survey_steps) == ("small", 0)
        assert DAY_SMOKE_PLAN.fast_seconds == SMOKE_PLAN.fast_seconds
        assert (CLOUDY_SMOKE_PLAN.start, CLOUDY_SMOKE_PLAN.overcast) == (NIGHT_START, 0.0)
        assert (CLOUDY_SMOKE_PLAN.fast_seconds, CLOUDY_SMOKE_PLAN.warmup_bursts) == (0.0, 0)
        assert 0 < CLOUDY_SMOKE_PLAN.search_seconds <= 5.0
        assert CLOUDY_SMOKE_PLAN.max_run_s == SMOKE_PLAN.max_run_s


class TestTheSteadyCycle:
    def test_it_starts_at_the_first_sample_that_fits(self) -> None:
        run = day_run()
        steady = steady_from(run.snapshots, lambda s: s.phase == "fast")
        assert steady[0].phase == "fast"
        assert len(steady) == len(run.snapshots) - 3  # the first sample and the two of the search

    def test_no_sample_that_fits_gives_no_samples(self) -> None:
        assert steady_from(day_run().snapshots, lambda s: s.cloud) == ()

    def test_whole_cycles_run_from_the_end_of_a_step_to_the_end_of_the_last(self) -> None:
        run = two_step_day()
        cycles = whole_cycles(run.snapshots, 1)
        assert [s.survey_steps for s in (cycles[0], cycles[-1])] == [1, 2]
        assert cycles[0].phase == "survey_short"  # the sample where step 1 ended
        assert cycles[-1].phase == "survey_short"  # and step 2: the wait for its result is out
        assert cycles[-1].t - cycles[0].t == pytest.approx(7.0)  # a gap, a period, and a step

    def test_without_a_later_step_there_is_no_whole_cycle(self) -> None:
        assert whole_cycles(day_run().snapshots, 1) == ()
        assert whole_cycles(two_step_day().snapshots, 2) == ()


class TestDayFigures:
    def test_a_day_gives_the_figures_of_core_sim_that_the_budgets_read(self) -> None:
        names = set(figures(day_measurements(day_run())))
        assert {
            "core.peak_rss",
            "survey_worker.peak_rss",
            "web.peak_rss",
            "other.peak_rss",
            "all.peak_rss_sum",
            "core.fast_share",
            "core.idle_share",
            "core.run_share",
            "core.frame_cost",
            "core.fastpath_receive_share",
            "survey_worker.cpu",
            "run.frame_rate",
            "run.length",
            "run.machine_busy",
        } <= names

    def test_a_frame_in_daylight_costs_the_difference_of_the_shares_over_the_frame_rate(
        self,
    ) -> None:
        found = figures(day_measurements(day_run()))
        assert found["core.fast_share"].value == pytest.approx(4.0)
        assert found["core.idle_share"].value == pytest.approx(1.0)
        assert found["core.frame_cost"].value == pytest.approx(375.0)  # 3% of a second, 80 frames
        assert found["core.fastpath_receive_share"].value == pytest.approx(375.0 * 98 / 1e4)

    def test_the_share_over_the_cycle_covers_whole_cycles_after_the_first_step(self) -> None:
        found = figures(day_measurements(two_step_day()))
        # A gap of 2 s at 5 ms, 4 s of measure at 40 ms, and step 2 at 10 ms: 180 ms in 7 s.
        assert found["core.run_share"].value == pytest.approx(100 * 0.180 / 7)
        assert str(found["core.run_share"].detail["covers"]).startswith("whole cycles of a day")

    def test_before_two_steps_the_share_over_the_cycle_starts_where_measure_starts(self) -> None:
        found = figures(day_measurements(day_run()))
        # From the first sample of measure: three seconds at 40 ms, and the survey step at 10 ms.
        assert found["core.run_share"].value == pytest.approx(100 * 0.130 / 4)
        covers = str(found["core.run_share"].detail["covers"])
        assert "from the start of measure" in covers
        assert "more measure than a whole cycle" in covers

    def test_the_length_of_the_run_holds_the_sky_the_sensor_and_the_scheduler(self) -> None:
        detail = figures(day_measurements(day_run()))["run.length"].detail
        assert detail["start_utc"] == DAY_START
        assert detail["sun_elevation_deg"] == "58.4"
        assert detail["sensor_temperature_c"] == "19.0"
        assert detail["fast_exposure_us"] == "1196"
        assert detail["search_exposure_us"] == "1185"
        assert detail["survey_long_skips"] == 1
        assert detail["measure_starts"] == 1
        assert detail["cloud_cycle"] == "no"
        assert "overcast_transmission" not in detail

    def test_a_range_shows_the_lowest_and_the_highest_value(self) -> None:
        run = day_run()
        warmer = [
            replace(s, sensor_temperature_c=19.0 + 0.5 * i) for i, s in enumerate(run.snapshots)
        ]
        facts = conditions(replace(run, snapshots=tuple(warmer)))
        assert facts["sensor_temperature_c"] == "19.0 to 22.5"  # the paused samples do not count


class TestCloudyFigures:
    def test_a_cloudy_night_gives_the_search_figures_and_the_peaks(self) -> None:
        names = set(figures(cloudy_measurements(cloudy_run())))
        assert {
            "core.peak_rss",
            "survey_worker.peak_rss",
            "core.search_share",
            "core.idle_share",
            "core.search_frame_cost",
            "core.search_burst_share",
            "core.run_share",
            "run.length",
        } <= names
        assert "core.fast_share" not in names
        assert "core.frame_cost" not in names

    def test_a_search_frame_costs_what_the_intervals_with_frames_use_beyond_the_gaps(self) -> None:
        found = figures(cloudy_measurements(cloudy_run()))
        # 48 ms against 8 ms a second: 40 ms for 50 frames.
        assert found["core.search_frame_cost"].value == pytest.approx(800.0)
        assert found["core.search_frame_cost"].detail["burst_seconds"] == 2.0
        assert found["core.search_burst_share"].value == pytest.approx(800.0 * 98 / 1e4)
        assert found["core.search_burst_share"].detail["share_at_hz"] == 98.0

    def test_without_a_gap_the_paused_scheduler_is_the_load_beyond_which_a_burst_costs(
        self,
    ) -> None:
        found = figures(cloudy_measurements(cloudy_run(with_gaps=False)))
        # 48 ms against 5 ms a second: 43 ms for 50 frames.
        assert found["core.search_frame_cost"].value == pytest.approx(860.0)

    def test_the_search_share_is_the_mean_load_of_the_bursts_and_the_gaps(self) -> None:
        found = figures(cloudy_measurements(cloudy_run()))
        # After the first burst: 3 gaps, and later 2 bursts and 3 gaps (the survey step is out).
        assert found["core.search_share"].value == pytest.approx(100 * (2 * 0.048 + 6 * 0.008) / 8)
        assert found["core.idle_share"].value == pytest.approx(0.5)
        assert found["run.frame_rate"].value == pytest.approx(100 / 8)

    def test_the_share_over_the_cycle_covers_whole_cycles_from_the_second_step(self) -> None:
        search: dict[str, Any] = {"purpose": "search", "phase": "search", "cloud": True}
        survey: dict[str, Any] = {"purpose": "survey", "phase": "survey_long", "cloud": True}
        clock = Timeline()
        clock.tick(frames=50, bursts=1, core_ms=48, purpose="search", phase="search")
        clock.tick(steps=1, bursts=1, core_ms=8, **survey)  # step 1 showed the clouds
        clock.tick(steps=1, bursts=2, frames=50, core_ms=48, **search)  # the period starts at once
        clock.tick(steps=2, bursts=2, core_ms=8, **survey)  # step 2 ends: the first whole cycle
        clock.repeat(2, steps=2, bursts=2, core_ms=4, purpose="survey", phase="gap", cloud=True)
        clock.tick(steps=2, bursts=3, frames=50, core_ms=48, **search)
        clock.tick(steps=2, bursts=3, core_ms=8, **search)
        clock.tick(steps=3, bursts=3, core_ms=8, **survey)  # step 3 ends
        clock.repeat(3, state="paused", purpose=None, phase="paused", steps=3, bursts=3, core_ms=5)
        run = made_up_run(clock, CLOUDY_PLAN, {"survey_steps": 3, "search_bursts": 3})
        found = figures(cloudy_measurements(run))
        # Two gaps at 4 ms, a burst at 48 ms, a gap at 8 ms, and step 3 at 8 ms: 72 ms in 5 s.
        assert found["core.run_share"].value == pytest.approx(100 * 0.072 / 5)
        covers = str(found["core.run_share"].detail["covers"])
        assert covers.startswith("whole cycles of a cloudy night")

    def test_before_three_steps_the_share_over_the_cycle_starts_with_the_clouds(self) -> None:
        found = figures(cloudy_measurements(cloudy_run()))
        # From the first sample under clouds to the pause: 3 gaps and a burst in 4 s.
        assert found["core.run_share"].value == pytest.approx(100 * (0.048 + 3 * 0.008) / 4)
        assert "first survey frame that showed clouds" in str(
            found["core.run_share"].detail["covers"]
        )

    def test_the_length_of_the_run_holds_the_overcast_and_the_bursts(self) -> None:
        detail = figures(cloudy_measurements(cloudy_run()))["run.length"].detail
        assert detail["overcast_transmission"] == 0.0
        assert detail["cloud_cycle"] == "yes"
        assert detail["search_bursts"] == 3
        assert detail["search_frames"] == 150
        assert detail["search_s"] == 8.0
        assert detail["gap_s"] == 6.0
        assert detail["sun_elevation_deg"] == "-26.6"
        assert detail["fast_exposure_us"] == "unknown"


class TestBudgetsReadTheFigures:
    """The names of the cases and the names that the budgets read must stay the same."""

    @staticmethod
    def report() -> Report:
        day = case("day-sim", *day_measurements(day_run()))
        cloudy = case("cloudy-sim", *cloudy_measurements(cloudy_run()))
        return with_case(with_case(fixture_report(), day), cloudy)

    def test_no_row_of_the_runs_misses_a_figure(self) -> None:
        found = {item.budget.key: item for item in evaluate(self.report())}
        for key in (
            "day-core",
            "day-memory-1.4",
            "day-memory-1.6",
            "search-bin1",
            "cloudy-memory-1.4",
            "cloudy-memory-1.6",
        ):
            assert found[key].verdict != "n/a", key
            assert found[key].missing == ()
        assert found["day-core"].value == pytest.approx(375.0 * 98 / 1e4)
        assert found["search-bin1"].value == pytest.approx(800.0 * 98 / 1e4)
        assert found["search-bin1"].budget.title.endswith("measured in core")
        # acquire of the ipc case, then core, the worker, web, and no other child
        assert found["day-memory-1.4"].value == pytest.approx(60 + 150 + 300 + 85)


class TestTheCases:
    @pytest.mark.parametrize("name", ["day-sim", "cloudy-sim"])
    def test_a_missing_core_process_gives_a_skip_with_the_reason(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(core_sim, "CORE_MODULE", "seeingmon.perf.not_there")
        result = execute_case(REGISTRY.get(name), smoke=True)
        assert (result.status, result.reason) == ("skipped", "the core process is not on main")

    @pytest.mark.parametrize(
        ("name", "smoke_plan", "full_plan", "made_up"),
        [
            ("day-sim", DAY_SMOKE_PLAN, DAY_PLAN, day_run),
            ("cloudy-sim", CLOUDY_SMOKE_PLAN, CLOUDY_PLAN, cloudy_run),
        ],
    )
    def test_the_mode_picks_the_plan_and_the_case_returns_the_figures_of_the_run(
        self,
        name: str,
        smoke_plan: RunPlan,
        full_plan: RunPlan,
        made_up: Callable[[], SystemRun],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plans: list[RunPlan] = []

        def fake(plan: RunPlan, log: Callable[[str], None] | None = None) -> SystemRun:
            plans.append(plan)
            return made_up()

        monkeypatch.setattr(visibility_sim, "run_system", fake)
        smoke = execute_case(REGISTRY.get(name), smoke=True)
        full = execute_case(REGISTRY.get(name), smoke=False)
        assert plans == [smoke_plan, full_plan]
        for result in (smoke, full):
            assert result.status == "ok", result.reason
            assert result.measurements
        text = "\n".join(smoke.notes)
        assert "cannot warm in the sun" in text
        assert "the sensor read 19.0 C" in text
        assert "10% busy" in text

    def test_a_system_that_fails_fails_the_case_with_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(plan: RunPlan, log: Callable[[str], None] | None = None) -> SystemRun:
            raise RuntimeError("the search never ran a burst after its warm-up")

        monkeypatch.setattr(visibility_sim, "run_system", broken)
        result = execute_case(REGISTRY.get("cloudy-sim"), smoke=True)
        assert result.status == "failed"
        assert "never ran a burst" in (result.reason or "")


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_the_full_day_measures_in_daylight_and_skips_the_long_exposure() -> None:
    """The day of the page: noon of midsummer, the `full` sensor, speed 1."""
    run = run_system(DAY_PLAN)
    assert run.fast.seconds >= DAY_PLAN.fast_seconds
    assert run.survey_steps >= DAY_PLAN.survey_steps
    assert run.counters.get("measure_starts", 0) >= 1
    assert run.counters.get("survey_long_skips", 0) >= 1  # the daylight skip
    exposures = [s.exposure_us for s in run.snapshots if s.purpose == "fast"]
    assert exposures
    assert max(e for e in exposures if e is not None) < DAY_PLAN.fast_exposure_us  # adapted
    names = set(figures(day_measurements(run)))
    assert {"core.fastpath_receive_share", "core.run_share", "survey_worker.peak_rss"} <= names


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_the_full_cloudy_night_searches_and_never_detects() -> None:
    """The cloudy night of the page: an opaque overcast, the `full` sensor, speed 1."""
    run = run_system(CLOUDY_PLAN)
    assert run.search.seconds >= CLOUDY_PLAN.search_seconds
    assert run.survey_steps >= CLOUDY_PLAN.survey_steps
    assert run.counters.get("detections", 0) == 0
    assert run.counters.get("measure_starts", 0) == 0
    assert any(s.cloud for s in run.snapshots)  # the survey frames showed the clouds
    names = set(figures(cloudy_measurements(run)))
    assert {"core.search_burst_share", "core.run_share", "survey_worker.peak_rss"} <= names
