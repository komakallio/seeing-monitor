"""The context from the core, the survey backlog, and the loop's handling of time."""

from __future__ import annotations

import itertools

import pytest

from seeingmon.analysis import FastContext
from seeingmon.clock import NS_PER_S, VirtualClock, iso_to_utc_ns
from seeingmon.scheduler import StepKind
from tests.scheduler.scenario import World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")


class TestTheContextFromTheCore:
    def test_the_core_adds_its_flags_and_the_heater_duty_to_every_window(self) -> None:
        calls: list[int] = []

        def provider(t_utc_ns: int) -> FastContext:
            calls.append(t_utc_ns)
            return FastContext(flags=frozenset({"heater_on"}), heater_duty=0.35)

        world = World(start_utc_ns=NIGHT, context_provider=provider)
        world.run_until(600)
        windows = world.windows()
        assert len(windows) >= 4
        for window in windows:
            assert window.flags == ["heater_on"]
            assert window.heater_duty == pytest.approx(0.35)
            assert window.zenith_angle_deg is not None  # the scheduler still supplies the zenith
        assert calls  # the scheduler asks the provider for the context
        world.close()

    def test_the_scheduler_merges_its_own_flags_with_the_ones_from_the_core(self) -> None:
        world = World(
            start_utc_ns=NIGHT,
            context_provider=lambda t: FastContext(flags=frozenset({"heater_on"})),
        )
        world.cloud(1000, 2500, 0.8)
        world.run_until(2400)
        flags = {flag for w in world.windows() for flag in w.flags}
        assert flags == {"heater_on", "cloud"}
        world.close()

    def test_the_zenith_angle_from_the_core_wins(self) -> None:
        world = World(
            start_utc_ns=NIGHT, context_provider=lambda t: FastContext(zenith_angle_deg=12.5)
        )
        world.run_until(300)
        assert {w.zenith_angle_deg for w in world.windows()} == {12.5}
        world.close()


class TestTheSurveyBacklog:
    def test_a_survey_analysis_that_never_answers_makes_the_scheduler_skip_the_step(self) -> None:
        """Each step queues two frames. With four waiting, the next step is skipped."""
        world = World(start_utc_ns=NIGHT, survey_polls=10**9)
        world.run_until(1500)
        counters = world.scheduler.status().counters
        assert counters.survey_frames == 4  # two steps of two frames, and then the limit
        assert counters.survey_skipped >= 5
        assert world.scheduler.status().survey_pending == 4
        skipped = world.events("scheduler.survey_skipped")
        assert len(skipped) == counters.survey_skipped
        assert (skipped[0].detail or {}) == {"pending": 4, "max_pending": 4}
        assert skipped[0].level == "warning"
        world.close()

    def test_the_fast_cycle_keeps_its_cadence_while_the_survey_is_skipped(self) -> None:
        world = World(start_utc_ns=NIGHT, survey_polls=10**9)
        world.run_until(2000)
        starts = world.period_starts()
        gaps = [later - earlier for earlier, later in itertools.pairwise(starts)]
        assert len(starts) >= 10
        assert all(gap == pytest.approx(180.0, abs=0.05) for gap in gaps[2:])
        assert world.scheduler.state.value == "auto"
        world.close()

    def test_a_survey_that_catches_up_resumes_the_steps(self) -> None:
        world = World(start_utc_ns=NIGHT, survey_polls=10**9)
        world.run_until(900)
        world.survey.polls_until_ready = 0
        for item in world.survey._waiting:  # the analysis answers all at once
            item.polls_left = 0
        world.run_until(2000)
        # Two steps ran before the limit, and the cycles from 900 on run steps again.
        assert world.scheduler.status().counters.survey_steps >= 6
        assert world.scheduler.status().survey_pending <= 2
        world.close()


class TestTime:
    def test_a_sleep_never_overshoots_the_deadline_of_run_until(self) -> None:
        world = World()  # daylight: the scheduler sleeps between brightness frames
        for target in (1.0, 31.4, 59.0):
            world.run_until(target)
            # The sleep ends at the deadline. The shortest sleep is a millisecond.
            assert 0 <= world.clock.utc_ns() - world.t(target) <= 1_000_000
        world.run_until(70.0)  # the brightness frame at 60 seconds comes and goes
        assert 0 <= world.clock.utc_ns() - world.t(70.0) < NS_PER_S // 10
        world.close()

    def test_every_step_moves_the_clock_or_changes_something(self) -> None:
        """No step spins: a step that does not move the clock must have changed the state."""
        world = World(start_utc_ns=NIGHT)
        same_time_steps = 0
        for _ in range(3000):
            before = world.clock.utc_ns()
            state_before = world.scheduler.status().counters.transitions
            kind = world.scheduler.step()
            moved = world.clock.utc_ns() != before
            changed = world.scheduler.status().counters.transitions != state_before
            if not moved and not changed:
                same_time_steps += 1
                assert kind in {StepKind.TRANSITION, StepKind.WORK}, kind
        # Transitions between phases of the cycle take no time, but they are rare.
        assert same_time_steps < 300
        world.close()

    def test_the_first_steps_are_a_brightness_frame_and_the_move_to_auto(self) -> None:
        world = World(start_utc_ns=NIGHT)
        kinds = [world.scheduler.step() for _ in range(3)]
        assert kinds[0] is StepKind.WORK  # the brightness frame, which also starts auto
        assert world.scheduler.state.value == "auto"
        assert StepKind.FRAME in {world.scheduler.step() for _ in range(5)}
        world.close()

    def test_the_monotonic_clock_governs_the_cycle_when_the_wall_clock_steps(self) -> None:
        """An NTP step moves UTC, and the cycle does not notice."""
        world = World(start_utc_ns=NIGHT)
        world.run_until(200)
        starts_before = len(world.period_starts())
        assert isinstance(world.clock, VirtualClock)
        world.clock.step_utc_ns(-3 * 3600 * NS_PER_S)
        world.run_for(400)
        starts = world.period_starts()
        # Two more periods began at the usual spacing, whatever UTC did.
        later = starts[starts_before:]
        assert len(later) >= 2
        assert later[1] - later[0] == pytest.approx(180.0, abs=0.05)
        world.close()
