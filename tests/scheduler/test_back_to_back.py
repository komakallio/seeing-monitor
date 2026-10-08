"""Cycles that run back to back: the next one starts as soon as the survey step ends.

With `[scheduler.survey] cadence_s` at 0, the camera never waits for a slot. A cycle is the fast
period of 120 s and the survey step, which takes about 4 s here (a long frame of 4 s), so a cycle
lasts about 125 s, where a cadence of 180 s left 55 s of idle time. The fast stream runs one frame
in 2 seconds in these scenarios, so a period can end up to one frame late.
"""

from __future__ import annotations

import itertools

import pytest

from seeingmon.clock import iso_to_utc_ns
from seeingmon.scheduler import activity as words
from seeingmon.scheduler.config import SchedulerConfig, SurveyConfig
from seeingmon.survey.config import TwilightConfig
from tests.scheduler.scenario import TEST_CONFIG, World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")
PERIOD_S = 120.0
STEP_S = 4.2  # the long frame of 4 s and the overhead of the two survey exposures
TWILIGHT = TwilightConfig(min_exposure_s=4.0)  # the long frame stays at 4 s


def config(cadence_s: float) -> SchedulerConfig:
    return TEST_CONFIG.model_copy(
        update={"survey": SurveyConfig(long_exposure_s=4.0, cadence_s=cadence_s)}
    )


@pytest.fixture(scope="module")
def back_to_back() -> World:
    world = World(start_utc_ns=NIGHT, config=config(0.0), twilight=TWILIGHT)
    world.run_until(3600.0)
    return world


def survey_shorts(world: World) -> list[float]:
    return [
        world.seconds(call.t_utc_ns)
        for call in world.configures(mode="bin2", video=False)
        if call.config.roi is None and call.config.exposure_us == 1000
    ]


class TestTheCyclesFollowEachOther:
    def test_a_period_starts_right_after_the_survey_step_of_the_cycle_before(
        self, back_to_back: World
    ) -> None:
        starts = [t for t in back_to_back.fast_starts() if t > 150.0]
        gaps = [later - earlier for earlier, later in itertools.pairwise(starts)]
        assert len(gaps) >= 25
        # The period (up to one frame late) and the survey step: a little more than 124 s.
        assert all(PERIOD_S + STEP_S - 0.5 <= gap <= PERIOD_S + STEP_S + 2.5 for gap in gaps)

    def test_the_survey_step_comes_once_a_cycle(self, back_to_back: World) -> None:
        shorts = survey_shorts(back_to_back)
        gaps = [later - earlier for earlier, later in itertools.pairwise(shorts)]
        assert len(gaps) >= 25
        assert all(PERIOD_S < gap < PERIOD_S + 8.0 for gap in gaps)

    def test_the_seeing_windows_cover_nearly_all_the_time(self, back_to_back: World) -> None:
        covered = sum(
            w.duration_s
            for w in back_to_back.windows()
            if 150.0 <= back_to_back.seconds(w.t_utc_ns) < 3400.0
        )
        assert covered / (3400.0 - 150.0) > 0.9

    def test_every_window_is_whole(self, back_to_back: World) -> None:
        later = [w for w in back_to_back.windows() if back_to_back.seconds(w.t_utc_ns) > 150.0]
        assert later
        assert all(w.n_frames == 30 and w.duration_s == pytest.approx(60.0) for w in later[:40])

    def test_no_cycle_counts_as_late(self, back_to_back: World) -> None:
        assert back_to_back.scheduler.status().counters.cadence_overruns == 0


class TestTheCameraNeverRests:
    def test_the_activity_never_says_idle(self) -> None:
        world = World(start_utc_ns=NIGHT, config=config(0.0), twilight=TWILIGHT)
        phases: set[str] = set()
        try:
            while world.seconds(world.clock.utc_ns()) < 900.0:
                world.scheduler.step()
                activity = world.scheduler.status().activity
                if activity is not None:
                    phases.add(activity.phase)
            assert "idle" not in phases
            assert {"fast", "survey_short", "survey_long"} <= phases
        finally:
            world.close()

    def test_the_activity_has_no_cadence_to_name(self) -> None:
        world = World(start_utc_ns=NIGHT, config=config(0.0), twilight=TWILIGHT)
        try:
            world.run_until(60.0)
            activity = world.scheduler.status().activity
            assert activity is not None
            assert activity.cadence_s is None
        finally:
            world.close()


class TestACadenceStillSetsTheLeastCycle:
    def test_a_cadence_longer_than_the_cycle_makes_the_camera_rest(self) -> None:
        world = World(start_utc_ns=NIGHT, config=config(180.0), twilight=TWILIGHT)
        try:
            seen: set[str] = set()
            while world.seconds(world.clock.utc_ns()) < 500.0:
                world.scheduler.step()
                activity = world.scheduler.status().activity
                if activity is not None:
                    seen.add(activity.phase)
                    if activity.phase == "idle":
                        assert activity.label == words.IDLE_LABEL
                        assert activity.cadence_s == 180.0
            assert "idle" in seen
        finally:
            world.close()
