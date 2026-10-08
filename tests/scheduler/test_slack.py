"""The slack of the cycle: the camera works in the time that the cadence leaves, and nothing slips.

A cycle is 180 s: a fast period of 120 s, the survey step, and the rest, the slack. The scenarios
here use a long survey frame of 4 s, so the step ends about 125 s into the cycle and 55 s remain.
With `[scheduler.fast] min_slack_fast_s` at 30, the stream that the cycle runs (the fast stream
while Polaris is measured, the search bursts otherwise) continues to the next slot, and its last
window is partial. No second survey step follows it, and the next period starts on its slot.

The fast stream runs one frame in 2 seconds here, so the slack of 55 s holds about 27 frames.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import pytest

from seeingmon.clock import iso_to_utc_ns
from seeingmon.records import SeeingWindowRecord
from seeingmon.scheduler import Command, Pause, Resume
from seeingmon.scheduler import activity as words
from seeingmon.scheduler.config import SchedulerConfig, SurveyConfig
from seeingmon.survey.config import TwilightConfig
from tests.scheduler.scenario import TEST_CONFIG, World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")
CYCLE_S = 180.0
STEP_ENDS_S = 126.0  # the survey step of 4 s starts at 120 s and ends before this
TWILIGHT = TwilightConfig(min_exposure_s=4.0)  # the long frame stays at 4 s


def config(min_slack_s: float) -> SchedulerConfig:
    return TEST_CONFIG.model_copy(
        update={
            "fast": TEST_CONFIG.fast.model_copy(update={"min_slack_fast_s": min_slack_s}),
            "survey": SurveyConfig(long_exposure_s=4.0),
        }
    )


def night(min_slack_s: float) -> World:
    return World(start_utc_ns=NIGHT, config=config(min_slack_s), twilight=TWILIGHT)


def send(command: Command) -> Callable[[World], None]:
    def action(world: World) -> None:
        world.scheduler.submit(command)

    return action


def phase_of(start: float) -> float:
    """Where a time falls in the cycle of 180 s."""
    return start % CYCLE_S


def slot_starts(world: World) -> list[float]:
    """The fast streams that began on their slot, which is a multiple of the cadence."""
    return [
        t
        for t in world.fast_starts()
        if t >= CYCLE_S - 1 and min(phase_of(t), CYCLE_S - phase_of(t)) < 1.0
    ]


def slack_starts(world: World) -> list[float]:
    """The fast streams that began after the survey step of their cycle."""
    return [t for t in world.fast_starts() if STEP_ENDS_S - 3 < phase_of(t) < STEP_ENDS_S + 3]


def after_the_first_cycle(world: World) -> list[SeeingWindowRecord]:
    """The windows from the second cycle on, where the pattern is steady."""
    return [w for w in world.windows() if world.seconds(w.t_utc_ns) > CYCLE_S]


def shorts(world: World) -> list[float]:
    """When each 1 ms survey frame was taken."""
    return [
        world.seconds(call.t_utc_ns)
        for call in world.configures(mode="bin2", video=False)
        if call.config.roi is None and call.config.exposure_us == 1000
    ]


@pytest.fixture(scope="module")
def filled() -> World:
    world = night(30.0)
    world.run_until(3600.0)
    return world


class TestTheCameraWorksInTheSlack:
    def test_a_second_fast_stream_starts_after_the_survey_step_of_each_cycle(
        self, filled: World
    ) -> None:
        starts = slack_starts(filled)
        assert len(starts) >= 18
        for cycle, start in enumerate(starts, start=0):
            assert start == pytest.approx(CYCLE_S * cycle + 124.5, abs=3.0)

    def test_it_runs_to_the_next_slot_and_the_next_period_starts_on_it(self, filled: World) -> None:
        slots = slot_starts(filled)
        assert len(slots) >= 18
        gaps = [later - earlier for earlier, later in itertools.pairwise(slots)]
        assert all(gap == pytest.approx(CYCLE_S, abs=0.01) for gap in gaps)

    def test_its_last_window_is_shorter_and_holds_the_rest_of_the_slack(
        self, filled: World
    ) -> None:
        late = [
            w for w in after_the_first_cycle(filled) if phase_of(filled.seconds(w.t_utc_ns)) > 100.0
        ]
        assert len(late) >= 18
        for window in late[:15]:
            assert 20 <= window.n_frames <= 29
            assert 40.0 < window.duration_s < 59.0

    def test_the_period_on_the_slot_still_makes_two_full_windows(self, filled: World) -> None:
        main = [
            w for w in after_the_first_cycle(filled) if phase_of(filled.seconds(w.t_utc_ns)) < 100.0
        ]
        assert len(main) >= 2 * 17
        assert all(w.n_frames == 30 and w.duration_s == pytest.approx(60.0) for w in main[:30])

    def test_windows_now_cover_most_of_the_time(self, filled: World) -> None:
        covered = sum(
            w.duration_s for w in filled.windows() if CYCLE_S <= filled.seconds(w.t_utc_ns) < 3420.0
        )
        assert covered / (3420.0 - CYCLE_S) > 0.9  # it was 120 of 180 s, two thirds

    def test_the_survey_step_still_comes_once_a_cycle_on_the_cadence(self, filled: World) -> None:
        times = shorts(filled)
        assert len(times) >= 19
        gaps = [later - earlier for earlier, later in itertools.pairwise(times)]
        assert all(gap == pytest.approx(CYCLE_S, abs=2.0) for gap in gaps)

    def test_the_counters_tell_the_two_kinds_of_period_apart(self, filled: World) -> None:
        counters = filled.scheduler.status().counters
        assert counters.slack_periods >= 18
        assert counters.slack_periods <= counters.fast_periods + 1
        assert counters.cadence_overruns == 0

    def test_the_activity_of_the_slack_says_what_it_is_and_what_follows(self) -> None:
        world = night(30.0)
        try:
            seen = None
            while world.seconds(world.clock.utc_ns()) < CYCLE_S + 150.0:
                world.scheduler.step()
                now = world.seconds(world.clock.utc_ns())
                activity = world.scheduler.status().activity
                if (
                    now > CYCLE_S + STEP_ENDS_S
                    and activity is not None
                    and activity.phase == "fast"
                ):
                    seen = activity
                    break
            assert seen is not None
            assert seen.label == words.FAST_LABEL
            assert seen.next_label == words.FAST_LABEL  # the next period, not a survey step
            assert seen.detail is not None
            assert "partial" in seen.detail
            assert "1 of 1" not in seen.detail
        finally:
            world.close()


class TestTheCameraStaysIdleWhenTheSlackIsShort:
    @pytest.mark.parametrize("minimum", [0.0, 60.0])
    def test_a_slack_under_the_minimum_stays_idle(self, minimum: float) -> None:
        world = night(minimum)
        world.run_until(1500.0)
        assert slack_starts(world) == []
        assert world.scheduler.status().counters.slack_periods == 0
        windows = [w for w in world.windows() if world.seconds(w.t_utc_ns) > CYCLE_S]
        assert windows
        assert all(w.n_frames == 30 for w in windows)
        world.close()

    def test_the_activity_of_a_short_slack_still_says_the_camera_rests(self) -> None:
        world = night(60.0)
        try:
            seen = None
            while world.seconds(world.clock.utc_ns()) < CYCLE_S + 170.0:
                world.scheduler.step()
                now = world.seconds(world.clock.utc_ns())
                activity = world.scheduler.status().activity
                if now > CYCLE_S + STEP_ENDS_S and activity is not None:
                    seen = activity
                    break
            assert seen is not None
            assert seen.phase == "idle"
            assert seen.label == words.IDLE_LABEL
        finally:
            world.close()


class TestTheSearchFillsTheSlackToo:
    def test_the_bursts_go_on_after_the_survey_step(self) -> None:
        with_slack = night(30.0)
        with_slack.hide_star(0.0, 4000.0)
        with_slack.run_until(500.0)
        idle = night(0.0)
        idle.hide_star(0.0, 4000.0)
        idle.run_until(500.0)
        # The survey step of the first cycle ends at 126 s: a burst falls between it and the slot.
        assert any(STEP_ENDS_S < b < CYCLE_S - 3 for b in with_slack.burst_starts())
        assert not any(STEP_ENDS_S < b < CYCLE_S - 3 for b in idle.burst_starts())

        def longest_gap(world: World) -> float:
            starts = [b for b in world.burst_starts() if b < 480.0]
            return max(later - earlier for earlier, later in itertools.pairwise(starts))

        # The search keeps its interval of 15 s but for the survey step, and it used to rest for
        # the whole slack.
        assert longest_gap(with_slack) < 25.0
        assert longest_gap(idle) > 50.0
        with_slack.close()
        idle.close()


class TestTheCycleStaysOnItsGrid:
    def test_a_star_that_goes_missing_in_the_slack_adds_no_survey_step(self) -> None:
        world = night(30.0)
        world.hide_star(CYCLE_S + 140.0, 4000.0)  # inside the slack of the second cycle
        world.run_until(4 * CYCLE_S)
        assert world.events("polaris.hidden")
        times = shorts(world)
        gaps = [later - earlier for earlier, later in itertools.pairwise(times)]
        assert all(gap == pytest.approx(CYCLE_S, abs=2.0) for gap in gaps)
        world.close()

    def test_a_pause_in_the_slack_ends_the_stream_and_the_next_cycle_starts_clean(self) -> None:
        world = night(30.0)
        world.at(CYCLE_S + 140.0, send(Pause()))
        world.at(CYCLE_S + 200.0, send(Resume()))
        world.run_until(6 * CYCLE_S)
        assert "paused" in world.states_visited()
        assert world.scheduler.status().state == "auto"
        late = [w for w in world.windows() if world.seconds(w.t_utc_ns) > 3 * CYCLE_S]
        assert late
        assert world.scheduler.status().counters.survey_steps >= 4
        world.close()

    def test_a_cadence_that_is_not_met_never_starts_a_slack_stream(self) -> None:
        """A slack of 5 s remains: the minimum of 30 s is far off, and the cycle idles."""
        world = World(
            start_utc_ns=NIGHT,
            config=TEST_CONFIG.model_copy(
                update={"fast": TEST_CONFIG.fast.model_copy(update={"min_slack_fast_s": 30.0})}
            ),
        )
        world.run_until(1000.0)  # the survey frame of 30 s leaves a slack of 27 s
        assert slack_starts(world) == []
        world.close()
