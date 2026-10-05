"""The alignment helper: a command preempts everything, and the stream ends in `safe`.

`align` streams the survey readout mode with a short exposure, and it hands every frame to the
consumer that the core registers. It ends on `StopAlignment`, on a pause, or after 30 minutes
without activity (a command or a call to `touch_alignment`).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from seeingmon.clock import iso_to_utc_ns
from seeingmon.frames import StreamKind
from seeingmon.scheduler import (
    Command,
    CommandResult,
    Pause,
    QueueSweep,
    RejectReason,
    StartAlignment,
    StopAlignment,
)
from tests.scheduler.scenario import World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")


def submit_at(world: World, seconds: float, command: Command) -> list[CommandResult]:
    """Submit a command at a time, and return a list that will hold the result."""
    holder: list[CommandResult] = []
    world.at(seconds, lambda w: holder.append(w.scheduler.submit(command)))
    return holder


def align_configures(world: World) -> list[tuple[float, int, int]]:
    """The configure calls of the alignment stream as (time, exposure in microseconds, gain)."""
    return [
        (world.seconds(call.t_utc_ns), call.config.exposure_us, call.config.gain)
        for call in world.configures(mode="bin2", video=True)
    ]


@dataclass(frozen=True)
class Run:
    world: World
    started: list[CommandResult]
    stopped: list[CommandResult]


@pytest.fixture(scope="module")
def run() -> Run:
    """The command comes in the middle of a fast period (360 to 480), and the stop at 700."""
    world = World(start_utc_ns=NIGHT)
    started = submit_at(world, 400, StartAlignment())
    stopped = submit_at(world, 700, StopAlignment())
    world.run_until(1000)
    world.close()
    return Run(world, started, stopped)


class TestStartAndStop:
    def test_the_command_is_accepted_and_the_state_changes_at_once(self, run: Run) -> None:
        (result,) = run.started
        assert result.accepted
        assert result.state == "align"
        assert result.message == "alignment started"
        changes = [(round(t), to) for t, _, to in run.world.state_changes()]
        assert changes == [(0, "auto"), (400, "align"), (700, "safe"), (700, "auto")]

    def test_the_fast_period_ends_with_its_window_flushed_and_nothing_is_lost(
        self, run: Run
    ) -> None:
        world = run.world
        before_alignment = [w for w in world.windows() if 360 < world.seconds(w.t_utc_ns) < 400]
        assert before_alignment
        last = before_alignment[-1]
        assert world.seconds(last.t_utc_ns) + last.duration_s == pytest.approx(402.0, abs=2.5)
        assert sum(w.n_frames for w in world.windows()) == world.fast.frames_pushed

    def test_the_camera_streams_the_survey_mode_with_the_configured_exposure_and_gain(
        self, run: Run
    ) -> None:
        world = run.world
        ((started_at, exposure_us, gain),) = align_configures(world)
        assert exposure_us == 500_000  # 0.5 s
        assert gain == 120
        assert 400 <= started_at <= 402.5  # after the frame that was in progress
        config = next(c.config for c in world.configures(mode="bin2", video=True))
        assert (config.kind, config.roi) == (StreamKind.VIDEO, None)  # the full frame

    def test_every_alignment_frame_goes_to_the_consumer(self, run: Run) -> None:
        world = run.world
        frames = world.align_frames
        assert 590 <= len(frames) <= 602  # 300 seconds of 0.5 second frames
        assert {f.exposure_us for f in frames} == {500_000}
        assert {f.mode for f in frames} == {"bin2"}
        assert len({f.stream_id for f in frames}) == 1
        assert world.scheduler.status().counters.frames >= len(frames)

    def test_the_fast_analysis_sees_no_alignment_frame(self, run: Run) -> None:
        world = run.world
        windows_during = [w for w in world.windows() if 403 < world.seconds(w.t_utc_ns) < 698]
        assert windows_during == []

    def test_stop_goes_to_safe_and_the_sky_check_sends_it_on_to_auto(self, run: Run) -> None:
        (result,) = run.stopped
        assert (result.accepted, result.state) == (True, "safe")
        assert run.world.states_visited() == ["safe", "auto", "align", "safe", "auto"]

    def test_the_cycle_starts_again_from_the_end_of_the_alignment(self, run: Run) -> None:
        world = run.world
        after = [t for t in world.period_starts() if t > 690]
        # The brightness frame comes at once, and the first period (a search) follows it.
        assert after[0] == pytest.approx(700.0, abs=1.0)
        assert after[1] - after[0] == pytest.approx(180.0, abs=0.05)

    def test_every_command_wrote_an_event(self, run: Run) -> None:
        events = run.world.events("scheduler.command")
        assert [(e.detail or {})["command"] for e in events] == ["StartAlignment", "StopAlignment"]
        assert all((e.detail or {})["accepted"] is True for e in events)
        assert all(e.level == "info" for e in events)


class TestIdleTimeout:
    def test_alignment_ends_after_thirty_minutes_without_activity(self) -> None:
        world = World(start_utc_ns=NIGHT)
        submit_at(world, 400, StartAlignment(exposure_s=10.0))  # slow frames keep the run short
        world.run_until(400 + 1800 - 5)
        assert world.scheduler.state.value == "align"
        world.run_until(400 + 1800 + 10)
        assert world.states_visited() == ["safe", "auto", "align", "safe", "auto"]
        ended_at = world.state_changes()[2][0]
        assert ended_at == pytest.approx(400 + 1800, abs=1.0)
        event = world.events("scheduler.state_change")[2]
        assert (event.detail or {})["reason"] == "alignment idle timeout"
        world.close()

    def test_touching_the_alignment_restarts_the_timer(self) -> None:
        world = World(start_utc_ns=NIGHT)
        submit_at(world, 400, StartAlignment(exposure_s=10.0))
        world.at(1400, lambda w: w.scheduler.touch_alignment())
        world.at(2400, lambda w: w.scheduler.touch_alignment())
        world.run_until(400 + 1800 + 100)  # past the timeout of the start, but not of the touch
        assert world.scheduler.state.value == "align"
        status = world.scheduler.status()
        assert status.alignment_idle_s is not None
        assert status.alignment_idle_s < 1800
        world.run_until(2400 + 1800 + 10)
        assert world.states_visited()[-2:] == ["safe", "auto"]
        ended_at = world.state_changes()[2][0]
        assert ended_at == pytest.approx(2400 + 1800, abs=1.0)
        world.close()

    def test_sending_start_again_keeps_it_alive_and_can_change_the_settings(self) -> None:
        world = World(start_utc_ns=NIGHT)
        submit_at(world, 400, StartAlignment(exposure_s=10.0))
        again = submit_at(world, 1400, StartAlignment(exposure_s=5.0, gain=60))
        world.run_until(2300)
        (result,) = again
        assert result.accepted
        assert "restarted" in result.message
        assert world.scheduler.state.value == "align"  # 400 + 1800 has passed
        configures = align_configures(world)
        assert [(round(t), e, g) for t, e, g in configures] == [
            (400, 10_000_000, 120),
            (1400, 5_000_000, 60),
        ]
        later = [f for f in world.align_frames if f.t_utc_ns > world.t(1415)]
        assert {f.exposure_us for f in later} == {5_000_000}
        assert {f.gain for f in later} == {60}
        world.close()

    def test_a_start_that_arrives_just_after_the_session_ended_starts_it_again(self) -> None:
        """A `StartAlignment` can arrive between the end of a session and the state change.

        The loop ends the session and changes the state in one hold of the lock now, so the
        window is gone. The scheduler also copes with the state alone, which this test builds.
        """
        world = World(start_utc_ns=NIGHT)
        submit_at(world, 100, StartAlignment(exposure_s=10.0))
        world.run_until(200)
        assert world.scheduler.state.value == "align"
        world.scheduler._align = None  # the stale case: the state says align, and no session
        result = world.scheduler.submit(StartAlignment(exposure_s=10.0))
        assert result.accepted
        assert world.scheduler.status().alignment_idle_s is not None
        world.run_until(260)
        assert world.scheduler.state.value == "align"
        world.close()

    def test_stopping_the_alignment_clears_the_session_with_the_state(self) -> None:
        world = World(start_utc_ns=NIGHT)
        submit_at(world, 100, StartAlignment(exposure_s=10.0))
        submit_at(world, 200, StopAlignment())
        world.run_until(300)
        status = world.scheduler.status()
        assert status.alignment_idle_s is None
        assert world.scheduler.state.value == "auto"
        world.close()

    def test_sending_start_again_with_the_same_settings_does_not_reconfigure(self) -> None:
        world = World(start_utc_ns=NIGHT)
        submit_at(world, 400, StartAlignment(exposure_s=10.0))
        submit_at(world, 800, StartAlignment(exposure_s=10.0))
        world.run_until(1200)
        assert len(align_configures(world)) == 1
        world.close()


class TestAlignmentPreempts:
    def test_alignment_runs_in_daylight_too(self) -> None:
        world = World()  # the Sun is up, and the scheduler is in safe
        started = submit_at(world, 100, StartAlignment())
        world.run_until(400)
        assert started[0].accepted
        assert world.states_visited() == ["safe", "align"]
        assert len(world.align_frames) > 500
        world.close()

    def test_alignment_starts_after_the_survey_exposure_in_progress(self) -> None:
        """A step takes a whole exposure, so a command that comes during one waits for its end.

        The test harness can inject a command only between steps, so the command arrives at 151,
        when the long exposure of the first cycle (121 to 151) ends. In a real run it arrives during
        the exposure, the state changes at once, and the camera follows after the exposure.
        """
        world = World(start_utc_ns=NIGHT)
        holder = submit_at(world, 130, StartAlignment())
        world.run_until(300)
        assert holder[0].accepted
        long_frames = [f for f in world.survey.submitted if f.exposure_us == 30_000_000]
        assert len(long_frames) == 1  # the exposure in progress finished and went to analysis
        start = align_configures(world)[0][0]
        assert 150.0 <= start <= 152.0
        world.close()

    def test_a_pause_ends_the_alignment(self) -> None:
        world = World(start_utc_ns=NIGHT)
        submit_at(world, 400, StartAlignment())
        submit_at(world, 500, Pause())
        world.run_until(600)
        assert world.states_visited() == ["safe", "auto", "align", "paused"]
        frames_at_pause = len(world.align_frames)
        world.run_until(900)
        assert len(world.align_frames) == frames_at_pause  # the stream stopped
        assert world.scheduler.status().alignment_idle_s is None
        world.close()

    def test_a_queued_task_waits_until_the_alignment_ends(self) -> None:
        world = World(start_utc_ns=NIGHT)
        submit_at(world, 400, StartAlignment())
        queued = submit_at(world, 450, QueueSweep(exposure_us=(2000,), gain=(0,), window_s=2.0))
        submit_at(world, 700, StopAlignment())
        world.run_until(1000)
        assert queued[0].accepted
        # The task ran in `safe` right after the alignment, and then `safe` checked the sky.
        assert world.states_visited() == [
            "safe",
            "auto",
            "align",
            "safe",
            "commission",
            "safe",
            "auto",
        ]
        (result,) = world.events("scheduler.sweep_result")
        assert world.seconds(result.t_utc_ns) > 700  # it ran after the alignment
        world.close()

    def test_a_fault_in_the_alignment_stream_runs_the_recovery_and_continues(self) -> None:
        world = World(start_utc_ns=NIGHT)
        submit_at(world, 400, StartAlignment())
        world.camera_fault(500, 504)
        world.run_until(700)
        assert world.events("scheduler.fault")
        assert world.scheduler.state.value == "align"
        assert len(align_configures(world)) >= 2  # the stream restarted after the recovery step
        world.close()


class TestRejections:
    def test_the_alignment_cannot_start_while_paused(self) -> None:
        world = World(start_utc_ns=NIGHT)
        submit_at(world, 100, Pause())
        refused = submit_at(world, 200, StartAlignment())
        world.run_until(300)
        (result,) = refused
        assert (result.accepted, result.reason) == (False, RejectReason.PAUSED)
        assert world.scheduler.state.value == "paused"
        event = world.events("scheduler.command")[-1]
        assert event.level == "warning"
        assert (event.detail or {})["reason"] == "paused"
        world.close()

    def test_stop_without_an_alignment_is_refused(self) -> None:
        world = World(start_utc_ns=NIGHT)
        refused = submit_at(world, 100, StopAlignment())
        world.run_until(200)
        assert refused[0].reason is RejectReason.NOT_ALIGNING
        assert world.scheduler.state.value == "auto"
        world.close()

    @pytest.mark.parametrize(
        "command",
        [
            StartAlignment(exposure_s=0.0),
            StartAlignment(exposure_s=-1.0),
            StartAlignment(exposure_s=float("nan")),
            StartAlignment(exposure_s=float("inf")),
            StartAlignment(exposure_s=1e-9),  # shorter than the camera's shortest exposure
            StartAlignment(exposure_s=3000.0),  # longer than its longest
            StartAlignment(gain=600),
            StartAlignment(gain=-1),
        ],
        ids=repr,
    )
    def test_settings_outside_the_profile_are_refused(self, command: StartAlignment) -> None:
        world = World(start_utc_ns=NIGHT)
        refused = submit_at(world, 100, command)
        world.run_until(200)
        assert refused[0].reason is RejectReason.INVALID
        assert world.scheduler.state.value == "auto"
        world.close()

    def test_the_alignment_cannot_start_while_the_camera_is_degraded(self) -> None:
        from tests.scheduler.test_fault_response import quick_config

        world = World(start_utc_ns=NIGHT, config=quick_config())
        world.camera_fault(1000, None)
        refused = submit_at(world, 1100, StartAlignment())
        world.run_until(1200)
        assert world.scheduler.status().degraded
        assert refused[0].reason is RejectReason.DEGRADED
        world.close()
