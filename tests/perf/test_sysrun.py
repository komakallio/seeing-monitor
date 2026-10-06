"""The run of the whole system: the phases, the shares, the sampler, and the helpers.

None of these tests starts a process of the system. They feed the logic with made-up samples and
a made-up process table, so that they check the arithmetic and never the speed of a machine.
"""

from __future__ import annotations

import http.server
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from seeingmon.clock import DEFAULT_START_UTC_NS, NS_PER_S, iso_to_utc_ns
from seeingmon.perf import procs, sysrun
from seeingmon.perf.sysrun import (
    FULL_PLAN,
    MIN_FAST_FPS,
    OVERCAST_LEAD_S,
    OVERCAST_SPAN_S,
    ROLES,
    SMOKE_PLAN,
    UNNAMED_GRACE_S,
    Phase,
    RunPlan,
    Sampler,
    Snapshot,
    WebPoller,
    clean_environment,
    cost_per_frame_us,
    dev_options,
    free_port,
    is_clean_fast,
    is_clean_search,
    is_idle,
    run_share,
    search_phase,
    split_phases,
    split_search,
)
from seeingmon.services.dev import DEV_WINDOW_S, default_start_utc_ns

from .helpers import MB, Timeline


def pair(**second: object) -> tuple[Snapshot, Snapshot]:
    """Two samples one second apart. The first is a steady fast stream, and `second` overrides."""
    first = Timeline().samples[0]
    follower = Timeline().tick(frames=100).samples[-1]
    return first, replace(follower, **second)  # type: ignore[arg-type]


class TestFastInterval:
    def test_a_steady_fast_stream_counts(self) -> None:
        first, second = pair()
        assert is_clean_fast(first, second, warmup_windows=1)

    @pytest.mark.parametrize(
        "change",
        [
            {"state": "paused"},
            {"purpose": "survey"},
            {"purpose": None},
            {"stream_id": 2},
            {"survey_pending": 1},
            {"survey_steps": 1},
            {"frames": 20},  # under 30 frames per second
        ],
    )
    def test_a_change_in_the_second_sample_ends_the_phase(self, change: dict[str, object]) -> None:
        first, second = pair(**change)
        assert not is_clean_fast(first, second, warmup_windows=1)

    @pytest.mark.parametrize(
        "change", [{"state": "paused"}, {"purpose": "survey"}, {"survey_pending": 2}]
    )
    def test_a_change_in_the_first_sample_ends_the_phase(self, change: dict[str, object]) -> None:
        first, second = pair()
        assert not is_clean_fast(replace(first, **change), second, warmup_windows=1)  # type: ignore[arg-type]

    def test_the_first_windows_are_the_warm_up(self) -> None:
        first, second = pair()
        early = replace(first, windows=0)
        assert not is_clean_fast(early, second, warmup_windows=1)
        assert is_clean_fast(early, second, warmup_windows=0)

    def test_the_least_frame_rate_comes_from_the_plan(self) -> None:
        first, second = pair(frames=20)
        assert not is_clean_fast(first, second, warmup_windows=0)
        assert is_clean_fast(first, second, warmup_windows=0, min_fps=10.0)

    def test_the_smoke_plan_accepts_a_slow_stream_and_the_full_plan_asks_for_the_default_rate(
        self,
    ) -> None:
        assert SMOKE_PLAN.min_fast_fps < MIN_FAST_FPS
        assert FULL_PLAN.min_fast_fps == MIN_FAST_FPS


class TestIdleInterval:
    def test_a_paused_scheduler_without_frames_is_idle(self) -> None:
        clock = Timeline().tick(state="paused", purpose=None).tick(state="paused", purpose=None)
        assert is_idle(clock.samples[1], clock.samples[2])

    def test_a_frame_in_the_interval_ends_the_idle_phase(self) -> None:
        clock = Timeline().tick(state="paused").tick(state="paused", frames=3)
        assert not is_idle(clock.samples[1], clock.samples[2])

    def test_an_interval_that_ends_or_starts_running_is_not_idle(self) -> None:
        clock = Timeline().tick(state="paused").tick(state="auto")
        assert not is_idle(clock.samples[1], clock.samples[2])
        assert not is_idle(clock.samples[0], clock.samples[1])


class TestPhases:
    @staticmethod
    def night() -> list[Snapshot]:
        """A warm-up, a fast stream, a survey step, a second fast stream, and a pause."""
        clock = Timeline(windows=0)
        clock.tick(frames=90, windows=0, core_ms=900)  # the start: imports and allocations
        clock.tick(frames=90, windows=1, core_ms=600)  # the first window closes
        clock.repeat(2, frames=90, windows=1, core_ms=45)  # counts: 2 s
        clock.tick(purpose="survey", stream_id=2, steps=1, pending=1, frames=0, core_ms=5)
        clock.tick(purpose="survey", stream_id=2, steps=1, pending=1, frames=0, core_ms=5)
        clock.tick(purpose="fast", stream_id=3, steps=1, frames=90, core_ms=300)  # the switch
        clock.repeat(3, purpose="fast", stream_id=3, steps=1, frames=90, core_ms=45)  # counts: 3 s
        clock.tick(state="paused", purpose=None, stream_id=None, steps=1, frames=0, core_ms=10)
        clock.repeat(2, state="paused", purpose=None, stream_id=None, steps=1, core_ms=10)
        return clock.samples

    def test_only_the_steady_fast_intervals_and_the_pause_count(self) -> None:
        fast, idle = split_phases(self.night(), warmup_windows=1)
        assert fast.seconds == pytest.approx(5.0)
        assert fast.frames == 450
        assert fast.fps == pytest.approx(90.0)
        assert idle.seconds == pytest.approx(2.0)
        assert idle.frames == 0
        assert idle.fps == 0.0

    def test_the_cpu_time_of_a_phase_is_the_sum_of_its_intervals(self) -> None:
        fast, idle = split_phases(self.night(), warmup_windows=1)
        assert fast.cpu_ns["core"] == pytest.approx(5 * 45e6)
        assert fast.share("core") == pytest.approx(4.5)  # 225 ms in 5 s
        assert idle.share("core") == pytest.approx(1.0)  # 20 ms in 2 s

    def test_a_longer_warm_up_leaves_out_more(self) -> None:
        fast, _ = split_phases(self.night(), warmup_windows=2)
        assert fast.seconds == 0.0

    def test_a_role_that_was_not_read_has_no_share(self) -> None:
        fast, _ = split_phases(self.night(), warmup_windows=1)
        assert fast.share("survey_worker") is None
        assert Phase().share("core") is None

    def test_a_phase_without_time_has_no_rate_and_no_threads(self) -> None:
        empty = Phase()
        assert empty.fps == 0.0
        assert empty.top_threads("core") == []

    def test_a_counter_that_falls_adds_nothing(self) -> None:
        # A restarted process counts from zero again. The interval must not give a negative time.
        first = Timeline().tick(frames=100, core_ms=500).samples[-1]
        second = replace(first, t=first.t + 1.0, frames=10, cpu_ns={"core": 1_000_000})
        phase = Phase()
        phase.add(first, second)
        assert phase.frames == 0
        assert phase.cpu_ns["core"] == 0

    def test_the_resident_size_of_a_phase_is_read_at_the_end_of_each_interval(self) -> None:
        clock = Timeline()
        for size in (100, 140, 120, 180):
            clock.tick(frames=90, rss_mb={"core": size})
        fast, _ = split_phases(clock.samples, warmup_windows=0)
        assert fast.resident_median("core") == round(130 * MB)  # between 120 and 140
        assert fast.resident_max("core") == round(180 * MB)

    def test_a_role_without_a_resident_size_has_no_median_and_no_largest(self) -> None:
        clock = Timeline().repeat(2, frames=90, rss_mb={"core": 100})
        fast, _ = split_phases(clock.samples, warmup_windows=0)
        assert fast.resident_median("web") is None
        assert fast.resident_max("web") is None
        assert Phase().resident_median("core") is None

    def test_the_busiest_threads_come_first_with_their_share(self) -> None:
        clock = Timeline()
        clock.tick(frames=90, core_threads_ms={11: 2.0, 12: 30.0, 13: 5.0})
        clock.tick(frames=90, core_threads_ms={11: 2.0, 12: 30.0, 13: 5.0, 14: 1.0})
        clock.tick(frames=90, core_threads_ms={11: 2.0, 12: 30.0, 13: 5.0, 14: 1.0})
        fast, _ = split_phases(clock.samples, warmup_windows=0)
        top = fast.top_threads("core", count=3)
        assert [tid for tid, _ in top] == [12, 13, 11]
        assert top[0][1] == pytest.approx(3.0)  # 90 ms in 3 s
        assert [tid for tid, _ in fast.top_threads("core", count=10)] == [12, 13, 11, 14]


class TestRunShare:
    def test_it_covers_the_cycle_up_to_the_first_paused_sample(self) -> None:
        clock = Timeline()
        clock.tick(frames=90, core_ms=50)  # fast
        clock.tick(purpose="survey", stream_id=2, core_ms=10)  # a survey step
        clock.tick(purpose="survey", stream_id=2, core_ms=10)  # a gap
        clock.tick(state="paused", purpose=None, stream_id=None, core_ms=900)  # not counted
        clock.tick(state="paused", purpose=None, stream_id=None, core_ms=900)
        # 70 ms of core in 3 s
        assert run_share(clock.samples, "core") == pytest.approx(100 * 0.07 / 3)

    def test_a_role_without_a_reading_gives_none(self) -> None:
        clock = Timeline().repeat(3, frames=90, core_ms=50)
        assert run_share(clock.samples, "survey_worker") is None

    def test_a_run_that_pauses_at_once_has_no_share(self) -> None:
        clock = Timeline().tick(frames=90, core_ms=5).tick(state="paused", purpose=None, core_ms=5)
        assert run_share(clock.samples, "core") == pytest.approx(0.5)  # 5 ms in 1 s
        paused_first = [replace(clock.samples[0], state="paused"), *clock.samples[1:]]
        assert run_share(paused_first, "core") is None
        assert run_share(clock.samples[:1], "core") is None  # one sample spans no time
        assert run_share([], "core") is None

    def test_a_cpu_time_that_falls_counts_as_zero(self) -> None:
        clock = Timeline().tick(frames=90, core_ms=50)
        low = replace(clock.samples[1], cpu_ns={"core": 0})
        assert run_share([clock.samples[0], low], "core") == 0.0


class TestCostPerFrame:
    def run(
        self, *, fast_core_ms: float = 50, idle_core_ms: float = 10, frames: int = 100
    ) -> tuple[Phase, Phase]:
        clock = Timeline()
        clock.repeat(4, frames=frames, core_ms=fast_core_ms)
        clock.repeat(2, state="paused", purpose=None, stream_id=None, core_ms=idle_core_ms)
        return split_phases(clock.samples, warmup_windows=0)

    def test_it_is_the_difference_of_the_shares_over_the_frame_rate(self) -> None:
        fast, idle = self.run()
        # 5% of a core against 1% when idle: 4% of a second for 100 frames is 400 us per frame.
        assert cost_per_frame_us(fast, idle, "core") == pytest.approx(400.0)

    def test_a_role_that_uses_less_when_busy_costs_nothing_and_never_less(self) -> None:
        fast, idle = self.run(fast_core_ms=5, idle_core_ms=10)
        assert cost_per_frame_us(fast, idle, "core") == 0.0

    def test_without_frames_there_is_no_cost(self) -> None:
        fast, idle = self.run(frames=0)
        assert cost_per_frame_us(fast, idle, "core") is None

    def test_a_missing_role_gives_none(self) -> None:
        fast, idle = self.run()
        assert cost_per_frame_us(fast, idle, "survey_worker") is None
        assert cost_per_frame_us(fast, Phase(), "core") is None


def searching(**second: object) -> tuple[Snapshot, Snapshot]:
    """Two samples one second apart in a search period, after the first burst.

    The second sample holds the frames of a burst, and `second` overrides its fields.
    """
    clock = Timeline()
    first = clock.tick(purpose="search", phase="search", bursts=1).samples[-1]
    follower = clock.tick(frames=50, purpose="search", phase="search", bursts=2).samples[-1]
    return first, replace(follower, **second)  # type: ignore[arg-type]


class TestSearchInterval:
    def test_a_search_period_counts_with_the_frames_of_a_burst_and_without(self) -> None:
        first, second = searching()
        assert is_clean_search(first, second, warmup_bursts=1)
        gap = replace(second, frames=first.frames, bursts=first.bursts)
        assert is_clean_search(first, gap, warmup_bursts=1)

    @pytest.mark.parametrize(
        "change",
        [
            {"state": "paused"},
            {"phase": "survey_short"},
            {"phase": "fast"},
            {"phase": None},
            {"survey_pending": 1},
            {"survey_steps": 1},
        ],
    )
    def test_a_change_in_the_second_sample_ends_the_phase(self, change: dict[str, object]) -> None:
        first, second = searching(**change)
        assert not is_clean_search(first, second, warmup_bursts=1)

    @pytest.mark.parametrize(
        "change", [{"state": "safe"}, {"phase": "idle"}, {"survey_pending": 2}]
    )
    def test_a_change_in_the_first_sample_ends_the_phase(self, change: dict[str, object]) -> None:
        first, second = searching()
        assert not is_clean_search(replace(first, **change), second, warmup_bursts=1)  # type: ignore[arg-type]

    def test_the_first_bursts_are_the_warm_up(self) -> None:
        first, second = searching()
        assert not is_clean_search(first, second, warmup_bursts=2)
        assert is_clean_search(replace(first, bursts=0), second, warmup_bursts=0)

    def test_the_rest_of_a_warm_up_burst_after_a_sample_stays_out(self) -> None:
        # A sample falls inside the first burst: 5 of its frames before it, 45 after.
        clock = Timeline()
        search: dict[str, Any] = {"purpose": "search", "phase": "search"}
        clock.tick(frames=5, bursts=1, core_ms=60, **search)
        clock.tick(frames=45, bursts=1, core_ms=240, **search)
        clock.repeat(2, bursts=1, core_ms=8, **search)
        samples = clock.samples
        assert not is_clean_search(samples[1], samples[2], warmup_bursts=1)
        bursts, gaps = split_search(samples, warmup_bursts=1)
        assert bursts.frames == 0
        assert gaps.seconds == pytest.approx(2.0)  # the gaps after the warm-up count
        # Without a warm-up, the same interval holds frames of a burst that counts.
        assert is_clean_search(samples[1], samples[2], warmup_bursts=0)
        assert split_search(samples, warmup_bursts=0)[0].frames == 45  # before: no search yet

    def test_the_rest_of_a_burst_after_the_warm_up_counts(self) -> None:
        first, second = searching()
        inside = replace(first, bursts=2)  # the sample falls inside the second burst
        assert is_clean_search(inside, replace(second, bursts=2), warmup_bursts=1)

    def test_the_search_and_the_fast_phase_never_share_an_interval(self) -> None:
        first, second = searching()
        assert not is_clean_fast(first, second, warmup_windows=0, min_fps=1.0)
        fast_first, fast_second = pair()
        assert not is_clean_search(fast_first, fast_second, warmup_bursts=0)


class TestSearchPhases:
    @staticmethod
    def night() -> list[Snapshot]:
        """A cloudy night: bursts with gaps, a survey step, more bursts, and a pause."""
        clock = Timeline()
        search: dict[str, Any] = {"purpose": "search", "phase": "search"}
        clock.tick(core_ms=8, **search)  # the search begins
        clock.tick(frames=50, bursts=1, core_ms=300, **search)  # the first burst: the warm-up
        clock.repeat(4, bursts=1, core_ms=8, **search)  # gaps: 4 s
        clock.tick(frames=50, bursts=2, core_ms=48, **search)  # a burst: 1 s
        clock.repeat(4, bursts=2, core_ms=8, **search)  # gaps: 4 s
        survey: dict[str, Any] = {"purpose": "survey", "phase": "survey_long", "bursts": 2}
        clock.tick(steps=0, pending=1, core_ms=8, **survey)
        clock.tick(steps=1, pending=1, core_ms=8, **survey)
        clock.tick(frames=50, steps=1, bursts=3, core_ms=48, **search)  # the switch: not counted
        clock.repeat(2, steps=1, bursts=3, core_ms=8, **search)  # gaps: 2 s
        clock.tick(frames=50, steps=1, bursts=4, core_ms=48, **search)  # a burst: 1 s
        paused: dict[str, Any] = {"state": "paused", "purpose": None, "phase": "paused"}
        clock.repeat(3, steps=1, bursts=4, core_ms=5, **paused)
        return clock.samples

    def test_the_search_phase_holds_the_bursts_and_the_gaps_after_the_warm_up(self) -> None:
        search = search_phase(self.night(), warmup_bursts=1)
        assert search.seconds == pytest.approx(12.0)
        assert search.frames == 100
        assert search.share("core") == pytest.approx(100 * (2 * 0.048 + 10 * 0.008) / 12)

    def test_the_split_gives_the_intervals_with_frames_and_the_gaps(self) -> None:
        bursts, gaps = split_search(self.night(), warmup_bursts=1)
        assert bursts.seconds == pytest.approx(2.0)
        assert bursts.frames == 100
        assert gaps.seconds == pytest.approx(10.0)
        assert gaps.frames == 0
        assert gaps.share("core") == pytest.approx(0.8)  # 8 ms a second
        # A burst uses 48 ms where a gap uses 8: 40 ms for 50 frames is 800 us a frame.
        assert cost_per_frame_us(bursts, gaps, "core") == pytest.approx(800.0)

    def test_without_a_warm_up_the_first_burst_counts(self) -> None:
        bursts, _ = split_search(self.night(), warmup_bursts=0)
        assert bursts.frames == 150
        assert search_phase(self.night(), warmup_bursts=0).seconds == pytest.approx(13.0)

    def test_a_run_of_search_has_no_fast_phase(self) -> None:
        fast, idle = split_phases(self.night(), warmup_windows=0, min_fps=1.0)
        assert fast.seconds == 0.0
        assert idle.seconds == pytest.approx(2.0)


class TestThePlan:
    def test_the_default_plans_measure_the_fast_phase_of_a_night(self) -> None:
        for plan in (FULL_PLAN, SMOKE_PLAN):
            assert (plan.start, plan.overcast) == (None, None)
            assert (plan.window_s, plan.fast_windows) == (None, None)
            assert plan.search_seconds == 0.0
            assert plan.fast_seconds > 0

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"overcast": -0.1}, "transmission"),
            ({"overcast": 1.5}, "transmission"),
            ({"fast_seconds": 0.0}, "collects"),
            ({"fast_windows": 0}, "at least 1"),
        ],
    )
    def test_a_plan_that_cannot_run_is_refused(self, change: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            RunPlan(**change)

    def test_a_plan_of_the_search_alone_needs_no_fast_phase(self) -> None:
        assert RunPlan(fast_seconds=0.0, search_seconds=10.0).search_seconds == 10.0

    def test_the_default_plan_keeps_the_sky_and_the_cycle_of_the_launcher(self) -> None:
        options = dev_options(SMOKE_PLAN, 8123)
        assert options.start is None
        assert options.window_s == DEV_WINDOW_S
        assert dict(options.extra_sim) == {}
        assert options.core_overrides == {
            "scheduler": {"loop": {"read_timeout_margin_s": SMOKE_PLAN.read_timeout_margin_s}}
        }
        assert options.acquire_overrides == {
            "services": {"acquire": {"read_timeout_margin_s": SMOKE_PLAN.read_timeout_margin_s}}
        }
        assert (options.port, options.sensor, options.speed) == (8123, "small", 1.0)

    def test_a_plan_sets_the_start_the_windows_and_the_fast_period(self) -> None:
        plan = RunPlan(start="2026-06-21T12:00:00Z", window_s=60.0, fast_windows=2)
        options = dev_options(plan, 8123)
        assert options.start == "2026-06-21T12:00:00Z"
        assert options.window_s == 60.0
        assert options.core_overrides["scheduler"]["fast"] == {"window_s": 120.0}

    def test_the_overcast_counts_from_the_epoch_of_the_simulator_and_covers_the_run(self) -> None:
        start = "2026-01-01T19:00:00Z"
        options = dev_options(RunPlan(start=start, overcast=0.0), 8123)
        (cloud,) = options.extra_sim["clouds"]
        offset_s = (iso_to_utc_ns(start) - DEFAULT_START_UTC_NS) / NS_PER_S
        assert cloud["start_s"] == pytest.approx(offset_s - OVERCAST_LEAD_S)
        assert cloud["duration_s"] == OVERCAST_SPAN_S
        assert cloud["transmission"] == 0.0
        assert OVERCAST_SPAN_S - OVERCAST_LEAD_S >= 86_400.0  # a day after the start at least

    def test_the_overcast_of_the_default_start_covers_that_start(self) -> None:
        options = dev_options(RunPlan(overcast=0.25), 8123)
        (cloud,) = options.extra_sim["clouds"]
        start_s = (default_start_utc_ns() - DEFAULT_START_UTC_NS) / NS_PER_S
        assert cloud["start_s"] < start_s < cloud["start_s"] + cloud["duration_s"]


def test_the_children_of_the_launcher_read_the_sky_and_the_cycle_of_a_plan(
    tmp_path: Path,
) -> None:
    """The overcast reaches the simulator in `acquire`, and the fast period reaches `core`."""
    pytest.importorskip("sep", reason="the simulated sky of the launcher needs the survey extra")
    from seeingmon.config import load_config
    from seeingmon.drivers.sim import SimOptions
    from seeingmon.scheduler import SchedulerConfig
    from seeingmon.services.config import ServicesConfig
    from seeingmon.services.dev import build_plan

    start = "2026-01-01T19:00:00Z"
    plan = RunPlan(
        sensor="small",
        start=start,
        overcast=0.0,
        window_s=60.0,
        fast_windows=2,
        fast_seconds=0.0,
        search_seconds=1.0,
    )
    absent = tmp_path / "absent.toml"
    built = build_plan(
        dev_options(plan, 8123), directory=tmp_path / "run", local_file=absent, env={}
    )
    specs = {spec.name: spec for spec in built.children}

    def config_of(name: str) -> Any:
        env = {k: v for k, v in specs[name].env.items() if k.startswith("SEEINGMON_")}
        return load_config(local_file=absent, env=env)

    services = config_of("acquire").section("services", ServicesConfig)
    sky = SimOptions.from_mapping({"clouds": services.acquire.driver_options["clouds"]})
    start_ns = iso_to_utc_ns(start)
    hour_ns = 3600 * NS_PER_S
    assert sky.clouds.transparency(start_ns) == pytest.approx(0.0)
    assert sky.clouds.transparency(start_ns + 6 * hour_ns) == pytest.approx(0.0)
    assert sky.clouds.transparency(start_ns - 2 * hour_ns) == pytest.approx(1.0)
    scheduler = config_of("core").section("scheduler", SchedulerConfig)
    assert (scheduler.fast.window_s, scheduler.fast.analysis_window_s) == (120.0, 60.0)
    assert built.start_utc_ns == start_ns


class FakeTable:
    """A made-up process table that stands in for the readers of `procs`."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.parents: dict[int, int] = {}
        self.lines: dict[int, str] = {}
        self.names: dict[int, str] = {}
        self.images: dict[int, str] = {}
        self.readings: dict[int, procs.ProcessReading] = {}
        self.threads: dict[int, dict[int, int]] = {}
        self.launchers: set[int] = set()
        monkeypatch.setattr(procs, "parent_map", lambda: dict(self.parents))
        monkeypatch.setattr(procs, "command_line", lambda pid: self.lines.get(pid))
        monkeypatch.setattr(procs, "image_path", lambda pid: self.images.get(pid))
        monkeypatch.setattr(procs, "process_name", lambda pid: self.names.get(pid))
        monkeypatch.setattr(procs, "read_process", lambda pid: self.readings.get(pid))
        monkeypatch.setattr(procs, "thread_cpu_ns", lambda pid: dict(self.threads.get(pid, {})))
        monkeypatch.setattr(procs, "is_venv_launcher", lambda pid: pid in self.launchers)

    def process(self, pid: int, parent: int, *, cpu_ms: float = 0.0, peak_mb: float = 10.0) -> None:
        self.parents[pid] = parent
        self.readings[pid] = procs.ProcessReading(
            pid, round(cpu_ms * 1e6), round(peak_mb * MB), round(peak_mb * MB * 0.9)
        )

    def gone(self, pid: int) -> None:
        del self.parents[pid]
        del self.readings[pid]


@pytest.fixture
def table(monkeypatch: pytest.MonkeyPatch) -> FakeTable:
    found = FakeTable(monkeypatch)
    found.process(100, 1, cpu_ms=500, peak_mb=138)  # acquire
    found.process(200, 1, cpu_ms=100, peak_mb=130)  # core
    found.process(300, 1, cpu_ms=20, peak_mb=85)  # web
    return found


ROOTS = {"acquire": 100, "core": 200, "web": 300}
SPAWNED = "python -c from multiprocessing.spawn import spawn_main; spawn_main()"


class FakeClock:
    """A clock that a test sets, in seconds."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestSampler:
    def test_it_finds_the_three_roots(self, table: FakeTable) -> None:
        assert Sampler(ROOTS).processes() == {100: "acquire", 200: "core", 300: "web"}

    def test_it_reads_the_cpu_time_of_each_role(self, table: FakeTable) -> None:
        cpu, threads = Sampler(ROOTS).read()
        assert cpu == {"acquire": 500_000_000, "core": 100_000_000, "web": 20_000_000}
        assert threads == {}

    def test_it_keeps_the_resident_size_of_each_role_at_the_last_read(
        self, table: FakeTable
    ) -> None:
        table.process(210, 200, peak_mb=300)  # a child of core, which counts as `other`
        table.lines[210] = "python -c from multiprocessing.resource_tracker import main;main(7)"
        sampler = Sampler(ROOTS)
        sampler.read()
        # The table gives 90% of the peak as the resident size.
        assert sampler.resident == {
            "acquire": round(138 * MB * 0.9),
            "core": round(130 * MB * 0.9),
            "web": round(85 * MB * 0.9),
            "other": round(300 * MB * 0.9),
        }
        table.readings[200] = procs.ProcessReading(200, 0, round(130 * MB), round(50 * MB))
        sampler.read()
        assert sampler.resident["core"] == round(50 * MB)  # the size now, and not the peak

    def test_it_reads_the_threads_of_the_roles_that_have_them(self, table: FakeTable) -> None:
        table.threads[200] = {201: 70_000_000, 202: 30_000_000}
        _, threads = Sampler(ROOTS).read()
        assert threads == {"core": {201: 70_000_000, 202: 30_000_000}}

    def test_a_spawned_child_of_core_is_the_survey_worker(self, table: FakeTable) -> None:
        table.process(210, 200, cpu_ms=4000, peak_mb=460)
        table.lines[210] = "python -c from multiprocessing.spawn import spawn_main; spawn_main()"
        sampler = Sampler(ROOTS)
        assert sampler.processes()[210] == "survey_worker"
        cpu, _ = sampler.read()
        assert "survey_worker" in cpu
        assert sampler.peaks()["survey_worker"] == round(460 * MB)
        assert sampler.worker_cpu_ns() == 4_000_000_000

    def test_the_roles_name_both_workers(self) -> None:
        assert {"survey_worker", "alignment_worker"} <= set(ROLES)

    def test_a_worker_that_named_itself_has_the_role_of_its_name(self, table: FakeTable) -> None:
        table.process(210, 200, cpu_ms=4000, peak_mb=460)
        table.lines[210] = SPAWNED
        table.names[210] = "smon-survey"
        table.process(220, 200, cpu_ms=900, peak_mb=390)
        table.lines[220] = SPAWNED
        table.names[220] = "smon-align"
        sampler = Sampler(ROOTS)
        found = sampler.processes()
        assert (found[210], found[220]) == ("survey_worker", "alignment_worker")
        cpu, _ = sampler.read()
        assert cpu["survey_worker"] == 4_000_000_000
        assert cpu["alignment_worker"] == 900_000_000
        assert sampler.peaks()["survey_worker"] == round(460 * MB)
        assert sampler.peaks()["alignment_worker"] == round(390 * MB)
        assert "other" not in sampler.peaks()
        assert sampler.worker_cpu_ns() == 4_000_000_000  # the alignment does not count as survey
        assert sampler.worker_cpu_ns("alignment_worker") == 900_000_000

    def test_an_alignment_worker_alone_is_no_survey_worker(self, table: FakeTable) -> None:
        # By day, the alignment can run before any survey frame starts the survey worker.
        table.process(220, 200, cpu_ms=900, peak_mb=390)
        table.lines[220] = SPAWNED
        table.names[220] = "smon-align"
        sampler = Sampler(ROOTS)
        sampler.read()
        assert "survey_worker" not in sampler.peaks()
        assert sampler.worker_cpu_ns() == 0
        assert sampler.worker_cpu_ns("alignment_worker") == 900_000_000

    def test_a_worker_waits_for_its_name_while_it_still_has_the_name_of_the_interpreter(
        self, table: FakeTable
    ) -> None:
        clock = FakeClock()
        table.process(220, 200, cpu_ms=20, peak_mb=40)
        table.lines[220] = SPAWNED
        table.names[220] = "python3"  # the worker imports its modules before it names itself
        sampler = Sampler(ROOTS, clock=clock)
        assert sampler.processes()[220] == "other"  # not decided yet, and not cached
        clock.now += 2.0
        table.names[220] = "smon-align"
        assert sampler.processes()[220] == "alignment_worker"
        table.names[220] = "python3"
        assert sampler.processes()[220] == "alignment_worker"  # a decided child keeps its role

    def test_a_worker_that_never_names_itself_is_the_survey_worker_after_the_grace(
        self, table: FakeTable
    ) -> None:
        clock = FakeClock()
        table.process(210, 200, cpu_ms=20, peak_mb=40)
        table.lines[210] = SPAWNED
        table.names[210] = "python3"
        sampler = Sampler(ROOTS, clock=clock)
        assert sampler.processes()[210] == "other"
        clock.now += UNNAMED_GRACE_S - 1.0
        assert sampler.processes()[210] == "other"
        clock.now += 1.0
        assert sampler.processes()[210] == "survey_worker"  # as every worker was before the names

    def test_where_the_system_gives_no_name_every_worker_is_the_survey_worker(
        self, table: FakeTable
    ) -> None:
        # Windows has no such name, so the two workers look alike there.
        for pid in (210, 220):
            table.process(pid, 200, peak_mb=100)
            table.lines[pid] = SPAWNED
        found = Sampler(ROOTS).processes()
        assert (found[210], found[220]) == ("survey_worker", "survey_worker")

    def test_the_resource_tracker_keeps_its_role_whatever_its_name(self, table: FakeTable) -> None:
        table.process(211, 200, peak_mb=12)
        table.lines[211] = "python -c from multiprocessing.resource_tracker import main;main(7)"
        table.names[211] = "python3"
        assert Sampler(ROOTS).processes()[211] == "other"

    def test_the_resource_tracker_is_another_process_and_adds_to_the_other_peak(
        self, table: FakeTable
    ) -> None:
        table.process(211, 200, peak_mb=12)
        table.lines[211] = "python -c from multiprocessing.resource_tracker import main;main(7)"
        table.process(212, 200, peak_mb=3)
        table.lines[212] = "some helper"
        table.images[212] = "/usr/bin/helper"
        sampler = Sampler(ROOTS)
        assert sampler.processes()[211] == "other"
        assert sampler.processes()[212] == "other"
        sampler.read()
        assert sampler.peaks()["other"] == round(15 * MB)
        assert "survey_worker" not in sampler.peaks()

    def test_a_child_with_an_empty_command_line_is_decided_at_a_later_sample(
        self, table: FakeTable
    ) -> None:
        # Linux shows an empty command line for a moment after a process starts.
        table.process(210, 200, cpu_ms=10, peak_mb=30)
        table.lines[210] = ""
        sampler = Sampler(ROOTS)
        assert sampler.processes()[210] == "other"
        sampler.read()
        table.lines[210] = "python -c from multiprocessing.spawn import spawn_main"
        assert sampler.processes()[210] == "survey_worker"
        sampler.read()
        assert sampler.peaks()["survey_worker"] == round(30 * MB)
        assert "other" not in sampler.peaks()
        assert sampler.worker_cpu_ns() == 10_000_000

    def test_a_decided_child_keeps_its_role(self, table: FakeTable) -> None:
        table.process(211, 200, peak_mb=12)
        table.lines[211] = "python -c from multiprocessing.resource_tracker import main;main(7)"
        sampler = Sampler(ROOTS)
        assert sampler.processes()[211] == "other"
        table.lines[211] = "python -c from multiprocessing.spawn import spawn_main"
        assert sampler.processes()[211] == "other"  # the process ID gave its answer once

    def test_a_python_image_marks_the_worker_where_the_command_line_is_unknown(
        self, table: FakeTable
    ) -> None:
        # Windows gives no command line, so the program tells a worker from a console host.
        table.process(210, 200, peak_mb=400)
        table.images[210] = "interpreters\\python.exe"
        table.process(211, 200, peak_mb=6)
        table.images[211] = "system\\conhost.exe"
        found = Sampler(ROOTS).processes()
        assert (found[210], found[211]) == ("survey_worker", "other")

    def test_the_children_of_acquire_and_web_are_not_read(self, table: FakeTable) -> None:
        table.process(110, 100, peak_mb=50)
        table.process(310, 300, peak_mb=50)
        assert set(Sampler(ROOTS).processes()) == {100, 200, 300}

    def test_it_follows_a_launcher_to_the_interpreter_that_does_the_work(
        self, table: FakeTable
    ) -> None:
        # The launcher of a virtual environment on Windows has the interpreter as its only child.
        table.launchers.update({100, 200})
        table.process(101, 100, cpu_ms=480, peak_mb=140)
        table.process(201, 200, cpu_ms=90, peak_mb=131)
        found = Sampler(ROOTS).processes()
        assert found == {101: "acquire", 201: "core", 300: "web"}

    def test_the_peak_of_a_role_is_the_largest_peak_that_any_read_saw(
        self, table: FakeTable
    ) -> None:
        sampler = Sampler(ROOTS)
        sampler.read()
        table.readings[200] = procs.ProcessReading(200, 2 * 10**8, round(180 * MB), round(120 * MB))
        sampler.read()
        table.readings[200] = procs.ProcessReading(200, 3 * 10**8, round(150 * MB), round(100 * MB))
        sampler.read()
        assert sampler.peaks()["core"] == round(180 * MB)  # the largest value that any read saw
        assert sampler.peaks()["acquire"] == round(138 * MB)

    def test_a_worker_that_ends_keeps_its_peak_and_its_cpu_time(self, table: FakeTable) -> None:
        table.process(210, 200, cpu_ms=3000, peak_mb=470)
        table.lines[210] = "python -c from multiprocessing.spawn import spawn_main"
        sampler = Sampler(ROOTS)
        sampler.read()
        table.gone(210)
        cpu, _ = sampler.read()
        assert "survey_worker" not in cpu  # it is not running now
        assert sampler.peaks()["survey_worker"] == round(470 * MB)
        assert sampler.worker_cpu_ns() == 3_000_000_000

    def test_a_process_that_the_system_will_not_read_is_left_out(self, table: FakeTable) -> None:
        sampler = Sampler(ROOTS)
        sampler.read()
        del table.readings[300]  # still in the table, but the system refuses the reading
        cpu, _ = sampler.read()
        assert "web" not in cpu
        assert sampler.peaks()["web"] == round(85 * MB)

    def test_the_cpu_time_of_a_reading_without_one_stays_at_the_last_value(
        self, table: FakeTable
    ) -> None:
        sampler = Sampler(ROOTS)
        sampler.read()
        table.readings[100] = procs.ProcessReading(100, None, round(139 * MB), round(100 * MB))
        cpu, _ = sampler.read()
        assert cpu["acquire"] == 500_000_000
        assert sampler.peaks()["acquire"] == round(139 * MB)


class TestEnvironmentAndPort:
    def test_the_environment_of_a_run_leaves_out_the_settings_of_the_person(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEEINGMON_WEB__TOKEN", "x")
        monkeypatch.setenv("SEEINGMON_ANYTHING", "y")
        monkeypatch.setenv("SEEING_KEPT", "z")
        found = clean_environment()
        assert not [key for key in found if key.startswith("SEEINGMON_")]
        assert found["SEEING_KEPT"] == "z"

    def test_a_free_port_is_a_port_number_that_binds(self) -> None:
        import socket

        port = free_port()
        assert 1024 <= port <= 65535
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))


class TestWebPoller:
    @staticmethod
    def serve(status: int = 200) -> tuple[http.server.ThreadingHTTPServer, list[str]]:
        seen: list[str] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                seen.append(self.path)
                self.send_response(status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format: str, *args: object) -> None:
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, seen

    def test_it_asks_for_the_same_pages_again_and_again_until_it_stops(self) -> None:
        server, seen = self.serve()
        try:
            poller = WebPoller(int(server.socket.getsockname()[1]), 0.02)
            poller.start()
            deadline = time.monotonic() + 10.0
            while poller.requests < 6 and time.monotonic() < deadline:
                time.sleep(0.02)
            poller.stop()
            asked = poller.requests
            time.sleep(0.1)
            assert poller.requests == asked  # it has stopped
        finally:
            server.shutdown()
            server.server_close()
        assert asked >= 6
        assert set(seen) == {f"/api/v1/{path}" for path in sysrun.WEB_PATHS}

    def test_an_error_answer_does_not_stop_it_and_does_not_count(self) -> None:
        server, seen = self.serve(status=500)
        try:
            poller = WebPoller(int(server.socket.getsockname()[1]), 0.02)
            poller.start()
            deadline = time.monotonic() + 10.0
            while len(seen) < 6 and time.monotonic() < deadline:
                time.sleep(0.02)
            poller.stop()
        finally:
            server.shutdown()
            server.server_close()
        assert len(seen) >= 6
        assert poller.requests == 0
