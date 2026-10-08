"""Following Polaris: ROI placement, a star that goes missing, a drift to the edge, no solution.

A star that goes missing ends the fast period early, ends measure with `polaris.hidden`, and
starts no solve: the pointing solution has no age limit, and a hidden star says nothing about the
mount. The next period searches, and two detecting bursts in a row (at the start of the period and
15 s later, 6 s each) start measure again, 21 s into the period.

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
from tests.scheduler.scenario import TEST_CONFIG, World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")


def survey_starts(world: World) -> list[tuple[float, int]]:
    """The survey exposures as (time, exposure in microseconds)."""
    return [
        (world.seconds(call.t_utc_ns), call.config.exposure_us)
        for call in world.configures(mode="bin2", video=False)
        if call.config.roi is None
    ]


def fast_starts(world: World) -> list[float]:
    """When each period of the cycle began: its first burst, or its fast stream."""
    return world.period_starts()


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
    """A missing star ends the fast period early and starts no solve: the mount did not move."""

    def test_a_missing_star_ends_the_period_early_and_requests_no_solve(
        self, missing_world: World
    ) -> None:
        assert missing_world.events("scheduler.solve_requested") == []
        counters = missing_world.scheduler.status().counters
        assert counters.solves_requested == 0
        assert counters.early_window_ends == 1

    def test_measure_ends_with_hidden_and_the_next_period_searches_until_visible(
        self, missing_world: World
    ) -> None:
        # The close at the end of the run ends the second measure.
        hidden, closed = missing_world.events("polaris.hidden")
        assert (closed.detail or {})["reason"] == "shutdown"
        assert missing_world.seconds(hidden.t_utc_ns) == pytest.approx(1118.0, abs=2.5)
        detail = hidden.detail or {}
        assert (detail["reason"], detail["frames"]) == ("star_missing", 10)
        assert detail["sun_elevation_deg"] < -18  # the night
        # The first measure began in the first period, and the next one in the period at 1260.
        visible = missing_world.visible_times()
        assert visible == [pytest.approx(21.0, abs=0.1), pytest.approx(1281.0, abs=0.1)]
        bursts = [t for t in missing_world.burst_starts() if t > 1100]
        assert bursts == [pytest.approx(1260.0, abs=0.1), pytest.approx(1275.0, abs=0.1)]

    def test_the_period_ends_early_and_its_window_is_flushed(self, missing_world: World) -> None:
        affected = [
            w for w in missing_world.windows() if 1075 < missing_world.seconds(w.t_utc_ns) < 1130
        ]
        assert affected
        # The frames from 1100 on lack the star, and the tenth of them arrives at 1118. No window
        # of that period reaches past the moment that the stream stopped.
        assert max(missing_world.seconds(w.t_utc_ns) + w.duration_s for w in affected) < 1135

    def test_the_survey_step_of_the_cycle_follows_and_the_next_period_keeps_its_slot(
        self, missing_world: World
    ) -> None:
        # The survey step that ends every period follows at once, within a frame of 2 seconds.
        (short,) = [t for t, e in survey_starts(missing_world) if 1100 < t < 1190 and e == 1000]
        assert short == pytest.approx(1118.0, abs=2.5)
        # The next period waits for its slot at 1260, and not only for the long frame.
        following = next(t for t in fast_starts(missing_world) if t > 1100)
        assert following == pytest.approx(1260.0, abs=0.05)
        measure = next(t for t in missing_world.fast_starts() if t > 1100)
        assert measure == pytest.approx(1281.0, abs=0.05)

    def test_the_cadence_stays_on_the_grid_with_one_survey_step_for_each_cycle(
        self, missing_world: World
    ) -> None:
        starts = [t for t in fast_starts(missing_world) if t > 300]
        gaps = [later - earlier for earlier, later in itertools.pairwise(starts)]
        assert gaps
        assert all(gap == pytest.approx(180.0, abs=0.05) for gap in gaps)
        # One short and one long exposure for each cycle, and no extra step for the missing star.
        steps = [(t, e) for t, e in survey_starts(missing_world) if starts[0] < t < 1800]
        shorts = [t for t, e in steps if e == 1000]
        longs = [t for t, e in steps if e == 30_000_000]
        assert len(shorts) == len(longs) == len(starts)

    def test_nothing_else_went_wrong(self, missing_world: World) -> None:
        assert missing_world.states_visited() == ["safe", "auto"]
        assert missing_world.scheduler.status().counters.faults == 0

    def test_a_star_that_stays_missing_is_searched_for_and_never_requests_a_solve(
        self,
    ) -> None:
        """Under thick cloud the star never returns, and the scheduler must not spin.

        The period at 900 loses the star 20 seconds before its end, which ends measure. Every
        period from 1080 to 2700 searches: 8 bursts of 6 s, one every 15 s, and the camera idles
        between them. The star returns at 2800, and the last burst of the period at 2700 (at 2805)
        finds it. The first burst of the period at 2880 is the second detection in a row, because
        the survey step between them breaks no row.
        """
        world = World(start_utc_ns=NIGHT)
        world.hide_star(1000, 2800)
        world.run_until(3000)
        assert world.events("scheduler.solve_requested") == []
        counters = world.scheduler.status().counters
        assert counters.solves_requested == 0
        assert counters.early_window_ends == 1
        (hidden,) = world.hidden_times()
        assert hidden == pytest.approx(1020.0, abs=2.5)
        assert world.visible_times()[-1] == pytest.approx(2886.0, abs=0.1)
        starts = [t for t in fast_starts(world) if t > 300]
        gaps = [later - earlier for earlier, later in itertools.pairwise(starts)]
        assert all(gap == pytest.approx(180.0, abs=0.05) for gap in gaps)
        searching = [t for t in world.burst_starts() if 1080 <= t < 2880]
        assert len(searching) == 10 * 8
        assert world.fast.frames_measured == 3 * counters.search_bursts
        # One survey step for each cycle, as on a clear night. The run ends before the step of the
        # period at 2880.
        shorts = [t for t, e in survey_starts(world) if e == 1000 and starts[0] < t < 3000]
        assert len(shorts) == len(starts) - 1
        assert world.scheduler.state.value == "auto"
        world.close()

    def test_a_short_loss_below_the_frame_threshold_does_nothing(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.hide_star(1100, 1116)  # eight frames, and the threshold is ten
        world.run_until(1300)
        assert world.events("scheduler.solve_requested") == []
        assert world.scheduler.status().counters.early_window_ends == 0
        world.close()

    def test_450_missing_frames_of_a_fast_stream_start_no_solve(self) -> None:
        """The default threshold, on frames of 100 ms: 45 seconds without the star."""
        config = SchedulerConfig(
            fast=FastConfig(
                exposure_us=100_000,  # the fastest frames that the scenario renders as they are
                roi_arcmin=1.0,
                roi_edge_margin_px=4.0,
                missing_star_frames=450,
                target_background_fraction=0.0,  # every frame takes 100 ms
                min_slack_fast_s=0.0,
            ),
            loop=LoopConfig(max_sleep_s=5.0),
        )
        world = World(start_utc_ns=NIGHT, config=config)
        # Two bursts of 50 frames of 100 ms, at 0 s and 15 s, find the star, and measure starts at
        # 20 s. The star goes at 25 s and returns at 200 s, after the bursts of the next period.
        world.hide_star(25, 200)
        world.run_until(310)
        assert world.events("scheduler.solve_requested") == []
        counters = world.scheduler.status().counters
        assert counters.solves_requested == 0
        assert counters.early_window_ends == 1
        assert world.fast_starts()[0] == pytest.approx(20.0, abs=0.2)
        first = [w for w in world.windows() if world.seconds(w.t_utc_ns) < 120]
        assert first
        # 450 frames of 100 ms after 25 s: the stream stopped at 70 s, to within 2 frames.
        end = max(world.seconds(w.t_utc_ns) + w.duration_s for w in first)
        assert end == pytest.approx(70.0, abs=0.2)
        assert world.hidden_times() == [pytest.approx(70.0, abs=0.2)]
        shorts = [t for t, e in survey_starts(world) if e == 1000 and t < 170]
        assert shorts == [pytest.approx(70.0, abs=0.5)]  # the step of the cycle, at once
        starts = fast_starts(world)
        assert starts[1] == pytest.approx(starts[0] + 180.0, abs=0.05)  # the slot holds
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
        # The bump moves the star far outside the ROI. The scheduler finds it missing at 1118 and
        # ends the period. The survey step that follows solves, and the next period searches on
        # its slot at 1260. Its bursts find the star at the edge of the sensor, and measure starts
        # at 1281. The jolt puts the star 2 pixels from the left edge of the sensor at the first
        # frame, at 1283, and the drift of 0.087 pixels a second moves it in.
        world.jolt(1100, 2.0 - 4144.0 - 0.087 * 1283)
        world.run_until(1700)
        assert world.scheduler.state.value == "auto"
        assert world.events("scheduler.solve_requested") == []  # a missing star requests none
        assert next(t for t in world.fast_starts() if t > 1200) == pytest.approx(1281.0, abs=0.1)
        (warning,) = world.events("scheduler.roi_at_limit")
        assert 1280 < world.seconds(warning.t_utc_ns) < 1290
        assert (warning.detail or {})["roi"]["x"] == 0  # the ROI sits against the sensor edge
        assert world.scheduler.status().counters.roi_recenters == 0
        period = [w for w in world.windows() if 1260 <= world.seconds(w.t_utc_ns) < 1380]
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
                target_background_fraction=0.0,  # the slow stream of the scenario
                min_slack_fast_s=0.0,
            ),
            search=TEST_CONFIG.search,
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
        # An attempt leaves one unsolved result, its 30 s frame. The 1 ms frame is no failed solve.
        unsolved = world.scheduler.status().counters.survey_unsolved
        assert len(shorts) - 1 <= unsolved <= len(shorts)
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
