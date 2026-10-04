"""Commissioning in the scheduler: tasks run at cycle boundaries, and results are pinned.

A sweep is the scheduler's own task. The burst and replay handlers belong to the services lane, so
these tests register small stand-ins that use the same context.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from seeingmon.clock import iso_to_utc_ns
from seeingmon.frames import StreamConfig
from seeingmon.scheduler import (
    Command,
    CommandResult,
    CommissionContext,
    CommissionResult,
    CommissionTask,
    Pause,
    QueueBurst,
    QueueReplay,
    QueueSweep,
    RejectReason,
    StartAlignment,
)
from seeingmon.scheduler.config import CommissionConfig, LoopConfig, SchedulerConfig
from tests.scheduler.scenario import TEST_CONFIG, World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")
SMALL_SWEEP = QueueSweep(exposure_us=(2000, 5000), gain=(0,), roi_arcmin=(4.1,), window_s=2.0)


def submit_at(world: World, seconds: float, command: Command) -> list[CommandResult]:
    holder: list[CommandResult] = []
    world.at(seconds, lambda w: holder.append(w.scheduler.submit(command)))
    return holder


def commission_changes(world: World) -> list[tuple[float, str, str]]:
    return [
        (t, source, target)
        for t, source, target in world.state_changes()
        if "commission" in (source, target)
    ]


class RecordingHandler:
    """A handler that records its calls, reads a few frames, and returns a result."""

    def __init__(self, kind: str, frames: int = 3) -> None:
        self.kind = kind
        self.frames = frames
        self.tasks: list[CommissionTask] = []
        self.configs: list[StreamConfig] = []

    def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult:
        self.tasks.append(task)
        started = context.clock.utc_ns()
        config = context.fast_stream_config()
        assert config is not None
        self.configs.append(config)
        context.configure(config)
        context.start()
        read = [context.read_frame() for _ in range(self.frames)]
        context.stop()
        return CommissionResult(
            task_id=task.task_id,
            kind=task.kind,
            status="ok",
            summary=f"recorded {len(read)} frames",
            started_utc_ns=started,
            finished_utc_ns=context.clock.utc_ns(),
            data={"frames": len(read)},
            artifacts=(f"{self.kind}s/{task.kind}-{task.task_id}.ser",),
        )


@dataclass(frozen=True)
class SweepRun:
    world: World
    queued: list[CommandResult]


@pytest.fixture(scope="module")
def sweep_run() -> SweepRun:
    """The sweep is queued at 50, in the middle of the first fast period (0 to 120)."""
    world = World(start_utc_ns=NIGHT)
    queued = submit_at(world, 50, SMALL_SWEEP)
    world.run_until(500)
    world.close()
    return SweepRun(world, queued)


class TestASweepAtTheBoundary:
    def test_the_command_is_accepted_with_a_task_id(self, sweep_run: SweepRun) -> None:
        (result,) = sweep_run.queued
        assert result.accepted
        assert result.task_id == 1
        assert result.state == "auto"
        assert "next cycle boundary" in result.message

    def test_the_task_waits_for_the_end_of_the_cycle_and_does_not_interrupt_it(
        self, sweep_run: SweepRun
    ) -> None:
        world = sweep_run.world
        changes = commission_changes(world)
        # The cycle runs a fast period to 120 and a survey step to 150. The boundary is at 150.
        assert [(round(t), a, b) for t, a, b in changes][:1] == [(150, "auto", "commission")]
        # The fast period and the survey step of that cycle completed before it.
        starts = [w for w in world.windows() if world.seconds(w.t_utc_ns) < 150]
        assert sum(w.n_frames for w in starts) == 60
        assert [f.exposure_us for f in world.survey.submitted[:2]] == [1000, 30_000_000]

    def test_the_scheduler_returns_to_auto_and_starts_a_new_cycle_at_once(
        self, sweep_run: SweepRun
    ) -> None:
        world = sweep_run.world
        assert world.states_visited() == ["safe", "auto", "commission", "auto"]
        end = commission_changes(world)[1][0]
        starts = [world.seconds(c.t_utc_ns) for c in world.configures(mode="bin1", video=True)]
        after_sweep = [t for t in starts if t >= end - 0.01]
        assert after_sweep[0] == pytest.approx(end, abs=0.5)
        assert after_sweep[1] - after_sweep[0] == pytest.approx(180.0, abs=0.05)

    def test_each_cell_runs_a_window_on_an_roi_centered_on_the_star(
        self, sweep_run: SweepRun
    ) -> None:
        world = sweep_run.world
        cells = [
            c for c in world.configures(mode="bin1", video=True) if c.config.exposure_us < 10_000
        ]
        assert [c.config.exposure_us for c in cells] == [2000, 5000]
        for call in cells:
            roi = call.config.roi
            assert roi is not None
            assert (roi.width, roi.height) == (128, 128)  # the profile's 4.1 arcmin in bin1
            x, y = world.star_position(call.t_utc_ns)
            assert roi.distance_to_edge(x, y) >= 60

    def test_the_result_reports_each_cell(self, sweep_run: SweepRun) -> None:
        (result,) = sweep_run.world.results
        assert (result.kind, result.status, result.task_id) == ("sweep", "ok", 1)
        assert result.summary == "2 of 2 cells measured"
        cells = result.data["cells"]
        assert [(c["cell"]["exposure_us"], c["status"]) for c in cells] == [
            (2000, "ok"),
            (5000, "ok"),
        ]
        for cell in cells:
            # A frame takes 11.3 ms (6.5 ms of overhead and 128 rows of 37.6 microseconds).
            assert cell["frame_rate_hz"] == pytest.approx(88.4, rel=0.02)
            assert cell["n_frames"] == pytest.approx(177, abs=3)
            assert cell["n_dropped"] == 0
            assert cell["drop_rate"] == 0.0
            assert cell["saturated_fraction"] == 0.0
            assert cell["star_found_fraction"] == 1.0
            # The star is 5,000 counts above a background of 200, and the level is 65,520.
            assert cell["peak_fraction_mean"] == pytest.approx(5200 / 65_520, abs=0.002)
            assert cell["background_fraction"] == pytest.approx(200 / 65_520, abs=0.001)
            assert cell["snr_median"] is None  # the fake frames have no noise
            assert cell["roi_px"] == [128, 128]

    def test_the_result_is_pinned_and_reaches_the_event_the_sink_and_the_memory(
        self, sweep_run: SweepRun
    ) -> None:
        world = sweep_run.world
        (result,) = world.results
        assert result.pinned is True
        assert world.scheduler.results() == (result,)
        (event,) = world.events("scheduler.sweep_result")
        assert event.level == "info"
        detail = event.detail or {}
        assert detail["pinned"] is True
        assert detail["task_id"] == 1
        assert len(detail["data"]["cells"]) == 2
        started = world.events("scheduler.task_started")
        assert len(started) == 1
        assert world.seconds(started[0].t_utc_ns) < world.seconds(event.t_utc_ns)

    def test_the_sweep_windows_stay_out_of_the_seeing_series(self, sweep_run: SweepRun) -> None:
        world = sweep_run.world
        exposures = {w.exposure_us for w in world.windows()}
        assert exposures == {2_000_000}  # only the windows of the regular fast stream
        # The sweep's per-frame metrics still reach the writer, under their own stream IDs.
        window_streams = {w.stream_id for w in world.windows()}
        sweep_streams = {s for s, _ in world.writer.metrics} - window_streams
        assert len(sweep_streams) == 2
        rows = sum(len(batch) for s, batch in world.writer.metrics if s in sweep_streams)
        assert rows == pytest.approx(2 * 177, abs=8)

    def test_the_status_counts_the_task(self, sweep_run: SweepRun) -> None:
        status = sweep_run.world.scheduler.status()
        assert status.counters.tasks_run == 1
        assert status.queued_tasks == 0
        assert status.state == "auto"


class TestTheQueue:
    def test_tasks_run_by_priority_and_then_in_order_of_arrival(self) -> None:
        world = World(start_utc_ns=NIGHT)
        for priority in (0, 5, 5, 1):
            command = QueueSweep(exposure_us=(2000,), gain=(0,), window_s=1.0, priority=priority)
            submit_at(world, 50, command)
        world.run_until(500)
        started = [(e.detail or {})["task_id"] for e in world.events("scheduler.task_started")]
        assert started == [2, 3, 4, 1]  # the two of priority 5, then 1, then 0
        # They ran back to back in one visit to `commission`, at one boundary.
        assert [(a, b) for _, a, b in commission_changes(world)] == [
            ("auto", "commission"),
            ("commission", "auto"),
        ]
        world.close()

    def test_a_full_queue_refuses_another_task(self) -> None:
        config = SchedulerConfig(
            fast=TEST_CONFIG.fast,
            loop=LoopConfig(max_sleep_s=5.0),
            commission=CommissionConfig(max_queued=2),
        )
        world = World(start_utc_ns=NIGHT, config=config)
        holders = [submit_at(world, 50 + i, SMALL_SWEEP) for i in range(3)]
        world.run_until(100)
        outcomes = [(h[0].accepted, h[0].reason) for h in holders]
        assert outcomes == [(True, None), (True, None), (False, RejectReason.QUEUE_FULL)]
        assert world.scheduler.status().queued_tasks == 2
        world.close()

    def test_a_task_cannot_be_queued_without_a_handler(self) -> None:
        world = World(start_utc_ns=NIGHT)
        refused = [
            submit_at(world, 50, QueueBurst()),
            submit_at(world, 51, QueueReplay(source="x")),
        ]
        world.run_until(100)
        assert [r[0].reason for r in refused] == [RejectReason.NO_HANDLER, RejectReason.NO_HANDLER]
        assert world.scheduler.status().queued_tasks == 0
        world.close()

    @pytest.mark.parametrize(
        ("command", "word"),
        [
            (QueueBurst(duration_s=0.0), "duration_s"),
            (QueueBurst(duration_s=float("nan")), "duration_s"),
            (
                QueueBurst(stream=StreamConfig(mode="bin9", exposure_us=2000, gain=0)),
                "readout mode",
            ),
            (QueueBurst(stream=StreamConfig(mode="bin1", exposure_us=2000, gain=900)), "gain"),
            (QueueBurst(stream=StreamConfig(mode="bin1", exposure_us=5, gain=0)), "exposure"),
            (QueueSweep(gain=(9999,)), "gain"),
            (QueueSweep(modes=("bin9",)), "unknown readout mode"),
            (QueueSweep(window_s=-1.0), "window_s"),
            (QueueReplay(source="x", speed=-1.0), "speed"),
        ],
        ids=repr,
    )
    def test_settings_outside_the_profile_are_refused_with_the_reason(
        self, command: Command, word: str
    ) -> None:
        world = World(start_utc_ns=NIGHT)
        world.scheduler.register_handler("burst", RecordingHandler("burst"))
        world.scheduler.register_handler("replay", RecordingHandler("replay"))
        result = world.scheduler.submit(command)
        assert (result.accepted, result.reason) == (False, RejectReason.INVALID)
        assert word in result.message
        assert world.scheduler.status().queued_tasks == 0
        world.close()


class TestHandlers:
    def test_a_registered_burst_handler_runs_at_the_boundary_with_exclusive_use_of_the_camera(
        self,
    ) -> None:
        world = World(start_utc_ns=NIGHT)
        handler = RecordingHandler("burst", frames=5)
        world.scheduler.register_handler("burst", handler)
        queued = submit_at(world, 50, QueueBurst(duration_s=5.0, label="test"))
        world.run_until(400)
        assert queued[0].accepted
        (task,) = handler.tasks
        assert isinstance(task.command, QueueBurst)
        assert task.command.label == "test"
        assert (task.kind, task.task_id) == ("burst", 1)
        # The handler's stream went through the scheduler's reconfiguration, so the camera saw it.
        assert handler.configs[0] in [c.config for c in world.configures(mode="bin1", video=True)]
        (result,) = world.results
        assert result.artifacts == ("bursts/burst-1.ser",)
        assert result.pinned
        assert [e.kind for e in world.events() if e.kind.endswith("_result")] == [
            "scheduler.burst_result"
        ]
        assert commission_changes(world)[0][0] == pytest.approx(150.0, abs=1.0)
        world.close()

    def test_the_burst_frames_count_as_frames_but_not_as_fast_analysis(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.scheduler.register_handler("burst", RecordingHandler("burst", frames=5))
        submit_at(world, 50, QueueBurst())
        world.run_until(400)
        counters = world.scheduler.status().counters
        # Every frame is counted, and the fast analyzer saw only the frames of the fast stream.
        others = counters.survey_frames + counters.watch_frames + 5  # survey, watch, and burst
        assert counters.frames == world.fast.frames_pushed + others
        world.close()

    def test_a_replay_handler_receives_the_options_of_the_command(self) -> None:
        world = World(start_utc_ns=NIGHT)
        handler = RecordingHandler("replay")
        world.scheduler.register_handler("replay", handler)
        command = QueueReplay(source="recording-1", speed=0.0, options={"store": "replay"})
        submit_at(world, 50, command)
        world.run_until(300)
        (task,) = handler.tasks
        assert task.command == command
        assert world.events("scheduler.replay_result")
        world.close()

    def test_a_handler_that_raises_fails_the_task_and_the_scheduler_goes_on(self) -> None:
        class Broken:
            def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult:
                context.configure(context.fast_stream_config() or StreamConfig("bin1", 2000, 0))
                raise RuntimeError("the disk is full")

        world = World(start_utc_ns=NIGHT)
        world.scheduler.register_handler("burst", Broken())
        submit_at(world, 50, QueueBurst())
        world.run_until(700)
        (result,) = world.results
        assert result.status == "failed"
        assert "the disk is full" in result.summary
        (error,) = world.events("scheduler.task_error")
        assert error.level == "error"
        assert world.events("scheduler.burst_result")[0].level == "warning"
        assert world.scheduler.state.value == "auto"
        # The cycle goes on, and the camera was stopped after the failure.
        assert world.windows()[-1].t_utc_ns > world.t(400)
        world.close()

    def test_a_camera_error_in_a_handler_runs_the_fault_response(self) -> None:
        class Reads:
            def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult:
                config = context.fast_stream_config()
                assert config is not None
                context.configure(config)
                context.start()
                context.read_frame()  # the camera fails
                raise AssertionError("unreachable")

        world = World(start_utc_ns=NIGHT)
        world.scheduler.register_handler("burst", Reads())
        world.camera_fault(150, 154)
        submit_at(world, 50, QueueBurst())
        world.run_until(600)
        assert world.events("scheduler.fault")
        (result,) = world.results
        assert result.status == "failed"
        assert "camera error" in result.summary
        assert world.scheduler.state.value == "auto"
        world.close()

    def test_a_result_with_data_that_is_not_json_does_not_stop_the_scheduler(self) -> None:
        class Careless:
            def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult:
                return CommissionResult(
                    task.task_id, task.kind, "ok", "done", 0, 1, data={"object": object()}
                )

        world = World(start_utc_ns=NIGHT)
        world.scheduler.register_handler("burst", Careless())
        submit_at(world, 50, QueueBurst())
        world.run_until(300)
        (event,) = world.events("scheduler.burst_result")
        assert event.detail == {"detail_dropped": True}
        assert world.scheduler.state.value == "auto"
        world.close()

    def test_a_result_sink_that_raises_is_reported_and_the_result_is_kept(self) -> None:
        world = World(start_utc_ns=NIGHT)

        def failing_sink(result: CommissionResult) -> None:
            raise OSError("the store is read-only")

        world.scheduler._result_sink = failing_sink
        submit_at(world, 50, SMALL_SWEEP)
        world.run_until(300)
        assert world.events("scheduler.result_sink_failed")
        assert len(world.scheduler.results()) == 1
        world.close()

    def test_the_scheduler_keeps_a_bounded_number_of_results(self) -> None:
        config = SchedulerConfig(
            fast=TEST_CONFIG.fast,
            loop=LoopConfig(max_sleep_s=5.0),
            commission=CommissionConfig(max_results=2),
        )
        world = World(start_utc_ns=NIGHT, config=config)
        for second in (50, 51, 52):
            submit_at(world, second, QueueSweep(exposure_us=(2000,), gain=(0,), window_s=1.0))
        world.run_until(500)
        assert [r.task_id for r in world.scheduler.results()] == [2, 3]
        world.close()

    def test_a_handler_kind_must_be_a_plain_word(self) -> None:
        world = World(start_utc_ns=NIGHT)
        with pytest.raises(ValueError, match="lowercase"):
            world.scheduler.register_handler("Bad Kind!", RecordingHandler("x"))
        world.close()


class TestPreemption:
    def test_an_alignment_that_arrives_during_a_task_stops_it_early(self) -> None:
        class Waits:
            """Reads frames until it is told to stop, and asks for the alignment on its own."""

            def __init__(self, world: World) -> None:
                self.world = world
                self.frames = 0

            def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult:
                config = context.fast_stream_config()
                assert config is not None
                context.configure(config)
                context.start()
                while not context.should_stop():
                    context.read_frame()
                    self.frames += 1
                    if self.frames == 20:  # another thread would send this command
                        self.world.scheduler.submit(StartAlignment())
                context.stop()
                return CommissionResult(
                    task.task_id,
                    task.kind,
                    "aborted",
                    "stopped early",
                    0,
                    1,
                    data={"frames": self.frames},
                )

        world = World(start_utc_ns=NIGHT)
        handler = Waits(world)
        world.scheduler.register_handler("burst", handler)
        submit_at(world, 50, QueueBurst())
        submit_at(world, 51, QueueSweep(exposure_us=(2000,), gain=(0,), window_s=1.0))
        world.run_until(400)
        assert handler.frames == 20
        (result, *_) = world.results
        assert result.status == "aborted"
        assert world.states_visited()[:4] == ["safe", "auto", "commission", "align"]
        # The queued sweep is still waiting, because the alignment preempted the queue.
        assert world.scheduler.status().queued_tasks == 1
        world.close()

    def test_a_pause_that_arrives_during_a_task_stops_it_and_keeps_the_queue(self) -> None:
        class Waits:
            def __init__(self, world: World) -> None:
                self.world = world

            def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult:
                self.world.scheduler.submit(Pause())
                aborted = context.should_stop()
                return CommissionResult(
                    task.task_id, task.kind, "aborted" if aborted else "ok", "x", 0, 1
                )

        world = World(start_utc_ns=NIGHT)
        world.scheduler.register_handler("burst", Waits(world))
        submit_at(world, 50, QueueBurst())
        submit_at(world, 51, QueueBurst())
        world.run_until(300)
        assert world.scheduler.state.value == "paused"
        assert world.scheduler.status().queued_tasks == 1
        assert world.results[0].status == "aborted"
        world.close()


class TestWhenTasksRun:
    def test_a_task_runs_in_safe_too(self) -> None:
        world = World()  # daylight
        submit_at(world, 130, SMALL_SWEEP)
        world.run_until(400)
        assert [(a, b) for _, a, b in commission_changes(world)] == [
            ("safe", "commission"),
            ("commission", "safe"),
        ]
        (result,) = world.results
        short, long = result.data["cells"]  # 2 ms and 5 ms
        # The Sun is up. The sky fills 58% of the range in 2 ms, and it saturates the frame in 5 ms.
        assert short["background_fraction"] == pytest.approx(0.58, abs=0.02)
        assert short["saturated_fraction"] == 0.0
        assert short["star_found_fraction"] == 1.0
        assert long["saturated_fraction"] == 1.0
        assert long["star_found_fraction"] == 0.0  # no contrast against a saturated sky
        # The brightness watch goes on afterwards.
        assert world.scheduler.state.value == "safe"
        world.close()

    def test_a_task_waits_while_the_camera_is_degraded_and_runs_after_the_recovery(self) -> None:
        from tests.scheduler.test_fault_response import quick_config

        world = World(start_utc_ns=NIGHT, config=quick_config())
        # Only a reboot fixes the camera. The quick steps fail, so the status turns degraded at
        # about 1016, the restart of `acquire` comes at about 1033, and the reboot at about 1098.
        world.camera_fault(1000, None, fixed_by=5)
        submit_at(world, 1050, SMALL_SWEEP)
        world.run_until(1080)
        assert world.scheduler.status().degraded
        assert world.events("scheduler.sweep_result") == []
        assert world.scheduler.status().queued_tasks == 1
        world.run_until(1500)
        assert len(world.events("scheduler.sweep_result")) == 1
        world.close()

    def test_a_sweep_that_cannot_find_polaris_skips_its_cells_and_says_why(self) -> None:
        world = World(start_utc_ns=NIGHT, solved_at_start=False)
        world.no_solution(0, 100_000)
        submit_at(world, 10, SMALL_SWEEP)
        world.run_until(500)
        (result,) = world.results
        assert result.status == "ok"
        assert result.summary == "0 of 2 cells measured"
        assert {c["status"] for c in result.data["cells"]} == {"skipped"}
        assert all("pointing" in c["note"] for c in result.data["cells"])
        world.close()
