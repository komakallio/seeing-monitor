"""Following Polaris: ROI placement, a star that goes missing, a drift to the edge, no solution.

The star sits near the middle of the bin1 sensor and drifts 0.087 pixels a second. The pointing
provider predicts that drift, so the ROI that the scheduler places before each fast period holds
the star at its center. A jolt moves the true star and leaves the prediction behind.

In the night scenarios a fast period starts at 1080.0 seconds and ends at 1200.0, because the
cycle starts at 0, and every cycle is 180 seconds.
"""

from __future__ import annotations

import itertools

import pytest

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.scheduler.config import FastConfig, LoopConfig, SchedulerConfig
from tests.scheduler.scenario import World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")


def survey_starts(world: World) -> list[tuple[float, int]]:
    """The survey exposures as (time, exposure in microseconds)."""
    return [
        (world.seconds(call.t_utc_ns), call.config.exposure_us)
        for call in world.configures(mode="bin2", video=False)
        if call.config.roi is None
    ]


def fast_starts(world: World) -> list[float]:
    return [world.seconds(call.t_utc_ns) for call in world.configures(mode="bin1", video=True)]


@pytest.fixture(scope="module")
def missing_world() -> World:
    """The star vanishes at 1100, in the first window of the period that began at 1080."""
    world = World(start_utc_ns=NIGHT)
    world.hide_star(1100, 1160)
    world.run_until(1800)
    world.close()
    return world


@pytest.fixture(scope="module")
def edge_world() -> World:
    """The mount is bumped 13 pixels at 1100, in the middle of the first window.

    The star lands 3 pixels from the edge of the 32 pixel ROI.
    """
    world = World(start_utc_ns=NIGHT)
    world.jolt(1100, 13)
    world.run_until(1500)
    world.close()
    return world


class TestMissingStar:
    def test_a_missing_star_triggers_a_survey_solve_off_schedule(
        self, missing_world: World
    ) -> None:
        (event,) = missing_world.events("scheduler.solve_requested")
        assert (event.detail or {})["reason"] == "star_missing"
        # The frames from 1100 on lack the star. The tenth of them arrives at 1118.
        assert missing_world.seconds(event.t_utc_ns) == pytest.approx(1118.0, abs=2.5)
        # The survey step starts at once, well before the regular one at 1200.
        steps = [t for t, e in survey_starts(missing_world) if 1100 < t < 1190 and e == 1000]
        assert len(steps) == 1
        assert steps[0] == pytest.approx(missing_world.seconds(event.t_utc_ns), abs=2.5)

    def test_the_period_ends_early_and_its_window_is_flushed(self, missing_world: World) -> None:
        counters = missing_world.scheduler.status().counters
        assert counters.early_window_ends == 1
        assert counters.solves_requested == 1
        affected = [
            w for w in missing_world.windows() if 1075 < missing_world.seconds(w.t_utc_ns) < 1130
        ]
        assert affected
        # No window of that period reaches past the moment that the stream stopped.
        assert max(missing_world.seconds(w.t_utc_ns) + w.duration_s for w in affected) < 1135

    def test_the_fast_stream_restarts_right_after_the_survey_step(
        self, missing_world: World
    ) -> None:
        long_frame = next(
            t for t, e in survey_starts(missing_world) if e == 30_000_000 and 1100 < t < 1190
        )
        following = next(t for t in fast_starts(missing_world) if t > long_frame)
        assert following == pytest.approx(long_frame + 30.2, abs=1.5)  # no wait for a slot

    def test_the_cadence_returns_to_the_regular_grid_afterwards(self, missing_world: World) -> None:
        after = [t for t in fast_starts(missing_world) if t > 1190]
        gaps = [later - earlier for earlier, later in itertools.pairwise(after)]
        assert gaps
        assert all(gap == pytest.approx(180.0, abs=0.05) for gap in gaps)

    def test_nothing_else_went_wrong(self, missing_world: World) -> None:
        assert missing_world.states_visited() == ["safe", "auto"]
        assert missing_world.scheduler.status().counters.faults == 0

    def test_a_star_that_stays_missing_gets_a_solve_only_once_per_interval(self) -> None:
        """Under thick cloud the star never returns, and the scheduler must not spin."""
        world = World(start_utc_ns=NIGHT)
        world.hide_star(1000, 2800)
        world.run_until(3000)
        solves = world.events("scheduler.solve_requested")
        # A solve, then 60 seconds before the next one can start, plus the survey step itself.
        assert 10 <= len(solves) <= 25
        times = [world.seconds(e.t_utc_ns) for e in solves]
        gaps = [later - earlier for earlier, later in itertools.pairwise(times)]
        assert all(gap >= 60.0 for gap in gaps)  # at least the resolve interval apart
        assert world.scheduler.state.value == "auto"
        world.close()

    def test_a_short_loss_below_the_frame_threshold_does_nothing(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.hide_star(1100, 1116)  # eight frames, and the threshold is ten
        world.run_until(1300)
        assert world.events("scheduler.solve_requested") == []
        assert world.scheduler.status().counters.early_window_ends == 0
        world.close()


class TestEdgeDrift:
    def test_a_star_near_the_edge_ends_the_window_early_with_the_partial_flag(
        self, edge_world: World
    ) -> None:
        early = [
            w
            for w in edge_world.windows()
            if 1080 <= edge_world.seconds(w.t_utc_ns) < 1140 and w.n_frames < 30
        ]
        assert len(early) == 1
        window = early[0]
        assert "partial" in window.flags
        assert 1100 <= edge_world.seconds(window.t_utc_ns) + window.duration_s <= 1108
        assert 8 <= window.n_frames <= 12  # about 20 seconds of 2 second frames

    def test_the_roi_moves_onto_the_star_without_a_new_stream(self, edge_world: World) -> None:
        moves = edge_world.camera.calls_named("move_roi")
        assert len(moves) == 1
        arguments = moves[0][1]
        assert isinstance(arguments, tuple)
        x, y = arguments
        star_x, star_y = edge_world.star_position(edge_world.t(1100))
        assert (x + 16, y + 16) == pytest.approx((star_x, star_y), abs=2.0)  # centered on it
        # The ROI move keeps the stream, so the windows before and after share a stream ID.
        period = [w for w in edge_world.windows() if 1080 <= edge_world.seconds(w.t_utc_ns) < 1200]
        assert len({w.stream_id for w in period}) == 1
        assert len(period) >= 3  # the early window, and the rest of the period in more windows

    def test_the_event_records_the_old_and_the_new_roi(self, edge_world: World) -> None:
        (event,) = edge_world.events("scheduler.roi_recentered")
        detail = event.detail or {}
        assert detail["edge_distance_px"] < detail["margin_px"] == 4.0
        old, new = detail["from"], detail["to"]
        # The ROI followed the jolt of 13 pixels and the drift of 1.7 pixels since the start.
        assert new["x"] - old["x"] == pytest.approx(14.7, abs=2)
        assert (new["width"], new["height"]) == (old["width"], old["height"])  # the size holds

    def test_the_star_ends_up_centered_and_stays_so(self, edge_world: World) -> None:
        counters = edge_world.scheduler.status().counters
        assert counters.roi_recenters == 1
        assert counters.early_window_ends == 1
        # The next survey step solved, so later periods start with a centered ROI.
        later = [
            c
            for c in edge_world.configures(mode="bin1", video=True)
            if c.t_utc_ns > edge_world.t(1200)
        ]
        assert later
        for call in later:
            roi = call.config.roi
            assert roi is not None
            x, y = edge_world.star_position(call.t_utc_ns)
            assert roi.distance_to_edge(x, y) >= 14

    def test_the_period_still_ends_on_time(self, edge_world: World) -> None:
        assert 1080.0 in [round(t, 1) for t in fast_starts(edge_world)]
        # The survey step that follows still starts at 1200, after the full 120 seconds.
        shorts = [t for t, e in survey_starts(edge_world) if 1190 < t < 1215 and e == 1000]
        assert shorts == [pytest.approx(1200.0, abs=2.0)]

    def test_a_second_bump_inside_the_cooldown_waits_for_it(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.jolt(1100, 13)
        world.jolt(1102, 13)  # 2 seconds later, which is inside the 5 second cooldown
        world.run_until(1190)
        times = [world.seconds(e.t_utc_ns) for e in world.events("scheduler.roi_recentered")]
        assert len(times) == 2
        assert times[1] - times[0] >= 5.0
        world.close()

    def test_a_star_that_the_roi_cannot_center_does_not_cut_a_window(self) -> None:
        """At the sensor edge the ROI clamps. The scheduler says so once and keeps its window."""
        world = World(start_utc_ns=NIGHT)
        # The bump moves the star far outside the ROI. The scheduler finds it missing at 1118,
        # solves, and starts a fast period at about 1149. The jolt puts the star 2 pixels from the
        # left edge of the sensor at that time, and the drift of 0.087 pixels a second moves it in.
        world.jolt(1100, 2.0 - 4144.0 - 0.087 * 1149)
        world.run_until(1700)
        assert world.scheduler.state.value == "auto"
        assert world.events("scheduler.solve_requested")
        (warning,) = world.events("scheduler.roi_at_limit")
        assert 1140 < world.seconds(warning.t_utc_ns) < 1160
        assert (warning.detail or {})["roi"]["x"] == 0  # the ROI sits against the sensor edge
        assert world.scheduler.status().counters.roi_recenters == 0
        period = [w for w in world.windows() if 1150 < world.seconds(w.t_utc_ns) < 1260]
        assert period
        assert all("partial" not in w.flags for w in period)
        world.close()

    def test_a_wide_margin_from_the_configuration_trips_at_the_first_drift(self) -> None:
        config = SchedulerConfig(
            fast=FastConfig(
                exposure_us=2_000_000,
                roi_arcmin=1.0,
                roi_edge_margin_px=15.0,
                missing_star_frames=10,
            ),
            loop=LoopConfig(max_sleep_s=5.0),
        )
        world = World(start_utc_ns=NIGHT, config=config)
        world.run_until(700)
        # The star drifts 10 pixels in a period. A margin of 15 pixels leaves 1 pixel of room.
        assert world.scheduler.status().counters.roi_recenters >= 2
        world.close()


class TestNoPointingSolution:
    def test_without_a_solution_the_first_action_is_a_survey_step_that_solves(self) -> None:
        world = World(start_utc_ns=NIGHT, solved_at_start=False)
        world.run_until(400)
        calls = world.configures()
        # A brightness frame, a short and a long survey exposure, and only then the fast stream.
        assert [(c.config.mode, c.config.exposure_us) for c in calls[:4]] == [
            ("bin2", 1000),
            ("bin2", 1000),
            ("bin2", 30_000_000),
            ("bin1", 2_000_000),
        ]
        (event,) = world.events("scheduler.solve_requested")
        assert (event.detail or {})["reason"] == "no_solution"
        # The fast stream begins as soon as the result arrives, which is right after the exposure.
        survey_end = world.seconds(calls[2].t_utc_ns) + 30.0
        assert world.seconds(calls[3].t_utc_ns) == pytest.approx(survey_end, abs=1.5)
        roi = calls[3].config.roi
        assert roi is not None
        x, y = world.star_position(calls[3].t_utc_ns)
        assert roi.distance_to_edge(x, y) >= 14
        world.close()

    def test_a_slow_analysis_is_awaited_without_a_fast_stream_in_between(self) -> None:
        world = World(start_utc_ns=NIGHT, solved_at_start=False, survey_polls=3)
        world.run_until(400)
        calls = world.configures()
        long_at = world.seconds(calls[2].t_utc_ns)
        first_fast = next(world.seconds(c.t_utc_ns) for c in calls if c.config.mode == "bin1")
        # The result arrives on the fourth poll, and the loop polls once per 5 second step here.
        assert long_at + 30 < first_fast < long_at + 30 + 30
        world.close()

    def test_a_sky_that_never_solves_is_retried_at_a_steady_pace_without_a_fast_stream(
        self,
    ) -> None:
        world = World(start_utc_ns=NIGHT, solved_at_start=False)
        world.no_solution(0, 4000)
        world.run_until(3600)
        assert world.configures(mode="bin1") == []
        shorts = [t for t, e in survey_starts(world) if e == 1000]
        gaps = [later - earlier for earlier, later in itertools.pairwise(shorts)]
        # Each attempt takes about 30 seconds, and the scheduler waits 60 seconds before the next.
        assert all(gap == pytest.approx(90.0, abs=3.0) for gap in gaps)
        assert 35 <= len(shorts) <= 42
        assert world.scheduler.state.value == "auto"
        assert world.scheduler.status().counters.survey_unsolved >= 2 * (len(shorts) - 1)
        world.close()

    def test_the_scheduler_recovers_when_a_solution_finally_arrives(self) -> None:
        world = World(start_utc_ns=NIGHT, solved_at_start=False)
        world.no_solution(0, 1000)
        world.run_until(1500)
        starts = fast_starts(world)
        assert starts
        assert 1000 <= starts[0] <= 1000 + 90 + 31
        world.close()

    def test_a_lost_solution_is_noticed_at_the_next_cycle_boundary(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.at(500, lambda w: w.pointing.clear())
        world.no_solution(400, 700)
        world.run_until(1500)
        events = world.events("scheduler.solve_requested")
        assert events
        assert all((e.detail or {})["reason"] == "no_solution" for e in events)
        assert world.scheduler.state.value == "auto"
        # The boundary is the end of the cycle that began at 360, and the next one starts at 540.
        assert world.seconds(events[0].t_utc_ns) == pytest.approx(540.0, abs=2.0)
        world.close()


def test_the_unit_of_the_virtual_clock_is_the_nanosecond() -> None:
    """A guard for the helpers above: `World.t` and `World.seconds` convert both ways."""
    world = World(start_utc_ns=NIGHT)
    assert world.t(1.5) - world.t(0) == 1_500_000_000 == round(1.5 * NS_PER_S)
    assert world.seconds(world.t(42.0)) == 42.0
