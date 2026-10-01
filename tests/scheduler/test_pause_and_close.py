"""Pause, resume, shutdown, and commands from other threads."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

import pytest

from seeingmon.clock import iso_to_utc_ns
from seeingmon.scheduler import (
    Command,
    CommandResult,
    Pause,
    QueueBurst,
    QueueReplay,
    QueueSweep,
    RejectReason,
    Resume,
    StartAlignment,
    StopAlignment,
)
from tests.scheduler.scenario import World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")

# What a call to the camera does when the scheduler runs something. A paused scheduler makes none
# of them. The calls to `stop` and `close` only release the camera, so they are fine.
RUNNING_CALLS = {"configure", "start", "read_frame", "move_roi", "recover", "open"}


def frames_accounted_for(world: World) -> int:
    """The frames in the written windows plus the frames that sweeps measured in their cells."""
    in_windows = sum(w.n_frames for w in world.windows())
    in_sweeps = sum(
        int(cell["n_frames"]) for result in world.results for cell in result.data.get("cells", [])
    )
    return in_windows + in_sweeps


def submit_at(world: World, seconds: float, command: Command) -> list[CommandResult]:
    holder: list[CommandResult] = []
    world.at(seconds, lambda w: holder.append(w.scheduler.submit(command)))
    return holder


@dataclass(frozen=True)
class PausedRun:
    world: World
    paused: list[CommandResult]
    resumed: list[CommandResult]
    calls_during_pause: list[str]


@pytest.fixture(scope="module")
def paused_run() -> PausedRun:
    """Pause at 400, in the middle of a fast period. Resume at 1200."""
    world = World(start_utc_ns=NIGHT)
    paused = submit_at(world, 400, Pause())
    resumed = submit_at(world, 1200, Resume())
    world.run_until(402.5)  # the teardown takes one step after the command
    calls_after_teardown = len(world.camera.calls)
    world.run_until(1199)
    calls = [name for name, _ in world.camera.calls[calls_after_teardown:]]
    world.run_until(1500)
    world.close()
    return PausedRun(world, paused, resumed, calls)


class TestPause:
    def test_the_command_is_accepted(self, paused_run: PausedRun) -> None:
        (result,) = paused_run.paused
        assert (result.accepted, result.state) == (True, "paused")

    def test_a_paused_scheduler_makes_no_call_that_runs_the_camera(
        self, paused_run: PausedRun
    ) -> None:
        assert paused_run.calls_during_pause == []  # not even a brightness frame

    def test_the_window_in_progress_is_flushed_when_the_pause_begins(
        self, paused_run: PausedRun
    ) -> None:
        world = paused_run.world
        last = [w for w in world.windows() if world.seconds(w.t_utc_ns) < 1000][-1]
        assert world.seconds(last.t_utc_ns) + last.duration_s == pytest.approx(402.0, abs=2.5)
        assert sum(w.n_frames for w in world.windows()) == world.fast.frames_pushed

    def test_no_record_comes_from_the_pause(self, paused_run: PausedRun) -> None:
        world = paused_run.world
        during = [w for w in world.windows() if 405 < world.seconds(w.t_utc_ns) < 1199]
        assert during == []
        surveys = [
            r for r in world.records("survey_frame") if 405 < world.seconds(r.t_utc_ns) < 1199
        ]
        assert surveys == []

    def test_the_state_stays_paused_with_its_reason(self, paused_run: PausedRun) -> None:
        world = paused_run.world
        assert world.states_visited() == ["safe", "auto", "paused", "safe", "auto"]
        reasons = [(e.detail or {})["reason"] for e in world.events("scheduler.state_change")]
        assert reasons == [
            "the sky is dark enough",
            "pause command",
            "resume command",
            "the sky is dark enough",
        ]

    def test_resume_goes_through_safe_and_the_brightness_frame_comes_at_once(
        self, paused_run: PausedRun
    ) -> None:
        (result,) = paused_run.resumed
        assert (result.accepted, result.state) == (True, "safe")
        world = paused_run.world
        starts = [world.seconds(c.t_utc_ns) for c in world.configures(mode="bin1", video=True)]
        after = [t for t in starts if t > 1190]
        assert after[0] == pytest.approx(1200.0, abs=1.0)
        assert after[1] - after[0] == pytest.approx(180.0, abs=0.05)

    def test_the_stream_stopped_for_the_pause(self, paused_run: PausedRun) -> None:
        stops = [i for i, (name, _) in enumerate(paused_run.world.camera.calls) if name == "stop"]
        assert stops  # the camera was told to stop capturing


class TestRejections:
    def test_pausing_twice_and_resuming_without_a_pause_are_refused(self) -> None:
        world = World(start_utc_ns=NIGHT)
        results = [
            submit_at(world, 100, Resume()),
            submit_at(world, 110, Pause()),
            submit_at(world, 120, Pause()),
            submit_at(world, 130, Resume()),
            submit_at(world, 140, Resume()),
        ]
        world.run_until(200)
        reasons = [r[0].reason for r in results]
        assert reasons == [
            RejectReason.NOT_PAUSED,
            None,
            RejectReason.ALREADY_PAUSED,
            None,
            RejectReason.NOT_PAUSED,
        ]
        world.close()

    def test_a_task_queued_while_paused_waits_for_the_resume(self) -> None:
        world = World(start_utc_ns=NIGHT)
        submit_at(world, 100, Pause())
        queued = submit_at(world, 150, QueueSweep(exposure_us=(2000,), gain=(0,), window_s=2.0))
        world.run_until(600)
        assert queued[0].accepted  # accepted, and held
        assert world.events("scheduler.sweep_result") == []
        assert world.scheduler.status().queued_tasks == 1
        submit_at(world, 700, Resume())
        world.run_until(900)
        assert len(world.events("scheduler.sweep_result")) == 1
        assert world.scheduler.status().queued_tasks == 0
        world.close()


class TestClose:
    def test_close_flushes_the_open_window_and_closes_the_camera(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.run_until(1100)  # in the middle of a period: 20 seconds into it
        windows_before = len(world.windows())
        world.close()
        assert len(world.windows()) == windows_before + 1  # the open window
        assert sum(w.n_frames for w in world.windows()) == world.fast.frames_pushed
        assert world.camera.calls_named("close")
        assert world.scheduler.status().stream is not None

    def test_close_twice_is_harmless(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.run_until(200)
        world.close()
        count = len(world.writer.records)
        world.close()
        assert len(world.writer.records) == count
        assert len(world.camera.calls_named("close")) == 1

    def test_a_closed_scheduler_rejects_commands_and_refuses_to_step(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.run_until(100)
        world.close()
        result = world.scheduler.submit(Pause())
        assert (result.accepted, result.reason) == (False, RejectReason.CLOSED)
        with pytest.raises(RuntimeError, match="closed"):
            world.scheduler.step()
        assert world.scheduler.status().counters.commands_rejected == 1

    def test_the_last_events_are_written_by_close(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.run_until(100)
        world.scheduler.submit(Pause())  # an event waits for the next step
        world.close()
        assert world.events("scheduler.command")  # close flushed it


class TestStopEvent:
    def test_run_returns_and_closes_the_camera_when_the_stop_event_is_set(self) -> None:
        world = World(start_utc_ns=NIGHT)
        stop = threading.Event()
        errors: list[BaseException] = []

        def loop() -> None:
            try:
                world.scheduler.run(stop)
            except BaseException as error:  # the thread must report anything that escapes
                errors.append(error)

        thread = threading.Thread(target=loop)
        thread.start()
        time.sleep(0.3)  # the virtual clock runs as fast as the CPU allows
        stop.set()
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert errors == []
        assert world.camera.calls_named("close")
        assert sum(w.n_frames for w in world.windows()) == world.fast.frames_pushed
        assert world.windows()
        # No event is lost, and no key repeats.
        keys = [event.record_key for event in world.events()]
        assert len(keys) == len(set(keys))


class TestCommandsFromAnotherThread:
    def test_random_commands_and_status_calls_race_with_the_loop_without_harm(self) -> None:
        world = World(start_utc_ns=NIGHT)
        stop = threading.Event()
        errors: list[BaseException] = []

        def loop() -> None:
            try:
                world.scheduler.run(stop)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=loop)
        thread.start()
        rng = random.Random(7)
        commands: list[Command] = [
            StartAlignment(exposure_s=0.5),
            StopAlignment(),
            Pause(),
            Resume(),
            QueueSweep(exposure_us=(2000,), gain=(0,), window_s=1.0),
            QueueBurst(),  # no handler is registered, so it is refused
            QueueReplay(source="x"),
        ]
        results: list[CommandResult] = []
        try:
            for _ in range(400):
                results.append(world.scheduler.submit(rng.choice(commands)))
                status = world.scheduler.status()
                assert status.state in {"safe", "auto", "align", "commission", "paused"}
                time.sleep(0.0005)
        finally:
            stop.set()
            thread.join(timeout=60)
        assert not thread.is_alive()
        assert errors == []
        # Every command got an answer, and a refusal names its reason.
        assert len(results) == 400
        assert all(r.accepted or r.reason is not None for r in results)
        refused_burst = [r for r in results if r.reason is RejectReason.NO_HANDLER]
        assert refused_burst
        # The records stay consistent: no lost window, and no repeated event key. A sweep keeps its
        # windows in its result, so its frames count there.
        assert frames_accounted_for(world) == world.fast.frames_pushed
        keys = [event.record_key for event in world.events()]
        assert len(keys) == len(set(keys))
        # Every state change has an event.
        counters = world.scheduler.status().counters
        assert counters.transitions == len(world.events("scheduler.state_change"))
        assert counters.commands_accepted + counters.commands_rejected == 400
        assert len(world.events("scheduler.command")) == 400
