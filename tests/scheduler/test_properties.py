"""Property tests of the state machine: random commands, weather, and faults never break its rules.

A hypothesis state machine drives a scheduler on a virtual clock. Its rules advance the clock,
send any of the commands (valid or not), and inject clouds, a missing star, a bump of the mount, a
floodlight, and camera faults. After every rule, the invariants hold:

- The scheduler is in one of the five legal states.
- The camera serves one mode at a time: a running stream matches the state (a fast stream in `auto`,
  an alignment stream or a rapid focus stream in `align`), and a paused or safe scheduler has no
  stream running.
- Every frame that the fast analyzer, the survey analyzer, the alignment consumer, or the focus
  consumer receives comes from the stream that the scheduler configured for that purpose, so no
  frame crosses a reconfiguration.
- Every change of state has an event, and no event key repeats.

When the run ends, no window is lost: every frame that the fast analyzer received is in a written
window, in the result of a sweep, or in the count of discarded frames.
"""

from __future__ import annotations

import os
from typing import Any

import hypothesis.strategies as st
from hypothesis import HealthCheck, settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from seeingmon.clock import iso_to_utc_ns
from seeingmon.frames import Frame
from seeingmon.scheduler import (
    Command,
    CommissionContext,
    CommissionResult,
    CommissionTask,
    Pause,
    QueueBurst,
    QueueReplay,
    QueueSweep,
    Resume,
    StartAlignment,
    StartRapidFocus,
    State,
    StopAlignment,
    StopRapidFocus,
)
from seeingmon.scheduler.commission import TaskStatus
from tests.scheduler.scenario import World
from tests.scheduler.test_fault_response import quick_config

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")
BRIGHT = iso_to_utc_ns("2026-01-01T14:30:00Z")  # daylight: the scheduler starts in safe, and stays
LEGAL_STATES = {state.value for state in State}


class Reader:
    """A stand-in for the burst and replay handlers of the services lane."""

    def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult:
        started = context.clock.utc_ns()
        config = context.fast_stream_config()
        status: TaskStatus = "failed"
        if config is not None:
            context.configure(config)
            context.start()
            for _ in range(3):
                if context.should_stop():
                    break
                context.read_frame()
            context.stop()
            status = "ok"
        return CommissionResult(
            task.task_id,
            task.kind,
            status,
            "read three frames",
            started,
            context.clock.utc_ns(),
        )


commands: st.SearchStrategy[Command] = st.one_of(
    st.builds(
        StartAlignment,
        exposure_s=st.one_of(st.none(), st.floats(0.2, 3.0)),
        gain=st.one_of(st.none(), st.integers(0, 300)),
    ),
    st.just(StartAlignment(exposure_s=0.0)),  # invalid
    st.just(StartAlignment(gain=9999)),  # invalid
    st.just(StopAlignment()),
    st.builds(
        StartRapidFocus,
        center_x_px=st.floats(3900.0, 4400.0),
        center_y_px=st.floats(2600.0, 3000.0),
        exposure_us=st.one_of(st.none(), st.integers(500, 3000)),
        gain=st.one_of(st.none(), st.integers(0, 100)),
    ),
    st.just(StartRapidFocus(float("nan"), 2822.0)),  # invalid
    st.just(StopRapidFocus()),
    st.just(Pause()),
    st.just(Resume()),
    st.builds(
        QueueSweep,
        exposure_us=st.just((2000,)),
        gain=st.just((0,)),
        window_s=st.floats(0.5, 2.0),
        priority=st.integers(-2, 2),
    ),
    st.builds(QueueBurst, duration_s=st.floats(0.5, 5.0), priority=st.integers(-2, 2)),
    st.builds(
        QueueReplay,
        source=st.text(alphabet="abc", min_size=1, max_size=3),
        speed=st.floats(0.0, 4.0),
        priority=st.integers(-2, 2),
    ),
    st.just(QueueBurst(duration_s=-1.0)),  # invalid
)


class SchedulerMachine(RuleBasedStateMachine):
    """The scheduler under random commands and weather."""

    START_UTC_NS = NIGHT

    def __init__(self) -> None:
        super().__init__()
        self.world = World(start_utc_ns=self.START_UTC_NS, config=quick_config())
        self.scheduler = self.world.scheduler
        self.scheduler.register_handler("burst", Reader())
        self.scheduler.register_handler("replay", Reader())
        self.submitted = 0
        self._audit_streams()

    # --- Audit hooks: every frame must come from the stream that the scheduler configured ---

    def _audit_streams(self) -> None:
        world = self.world
        scheduler = self.scheduler

        def check(frame: Frame, purposes: set[str]) -> None:
            stream = scheduler.stream
            assert stream is not None
            assert frame.stream_id == stream.stream_id, "a frame from another stream"
            assert stream.purpose in purposes, (stream.purpose, purposes)

        push, submit = world.fast.push, world.survey.submit

        def audited_push(frame: Frame) -> Any:
            check(frame, {"fast", "commission"})
            return push(frame)

        def audited_submit(frame: Frame) -> None:
            check(frame, {"survey"})
            submit(frame)

        def audited_sink(frame: Frame) -> None:
            check(frame, {"align"})
            world.align_frames.append(frame)

        focus_push = world.focus.push

        def audited_focus(frame: Frame) -> Any:
            check(frame, {"rapid_focus"})
            return focus_push(frame)

        world.fast.push = audited_push  # type: ignore[method-assign]
        world.survey.submit = audited_submit  # type: ignore[method-assign]
        world.focus.push = audited_focus  # type: ignore[method-assign]
        scheduler._alignment_sink = audited_sink

    # --- Rules ---

    def _now(self) -> float:
        return self.world.seconds(self.world.clock.utc_ns())

    @rule(seconds=st.floats(0.5, 400.0))
    def advance(self, seconds: float) -> None:
        self.world.run_for(seconds)

    @rule(command=commands)
    def send(self, command: Command) -> None:
        result = self.scheduler.submit(command)
        self.submitted += 1
        assert result.accepted or result.reason is not None
        assert result.message
        self.scheduler.step()  # let the loop notice the change, as it would a moment later

    @rule()
    def touch(self) -> None:
        self.scheduler.touch_alignment()

    @rule(duration=st.floats(1.0, 90.0), fraction=st.floats(0.0, 1.0))
    def clouds(self, duration: float, fraction: float) -> None:
        self.world.cloud(self._now(), self._now() + duration, fraction)

    @rule(duration=st.floats(1.0, 90.0))
    def hide_the_star(self, duration: float) -> None:
        self.world.hide_star(self._now(), self._now() + duration)

    @rule(dx=st.floats(-14.0, 14.0))
    def bump(self, dx: float) -> None:
        self.world.jolt(self._now(), dx)

    @rule(duration=st.floats(5.0, 200.0), level=st.floats(0.4, 1.0))
    def floodlight(self, duration: float, level: float) -> None:
        self.world.light(self._now(), self._now() + duration, level)

    @rule(duration=st.floats(1.0, 60.0), fixed_by=st.one_of(st.none(), st.integers(1, 4)))
    def camera_fault(self, duration: float, fixed_by: int | None) -> None:
        self.world.camera_fault(self._now(), self._now() + duration, fixed_by=fixed_by)

    # --- Invariants ---

    @invariant()
    def the_state_is_legal(self) -> None:
        assert self.scheduler.state.value in LEGAL_STATES

    @invariant()
    def the_camera_serves_one_mode_at_a_time(self) -> None:
        state = self.scheduler.state
        stream = self.scheduler.stream
        running = self.world.camera._running
        if state in (State.PAUSED, State.SAFE):
            assert not running, f"a stream runs in {state}"
        if running:
            assert stream is not None
            assert (stream.purpose, state.value) in {
                ("fast", "auto"),
                ("align", "align"),
                ("rapid_focus", "align"),
            }, (
                stream.purpose,
                state.value,
            )

    @invariant()
    def every_change_of_state_has_an_event(self) -> None:
        events = self.world.events("scheduler.state_change")
        assert self.scheduler.status().counters.transitions == len(events)

    @invariant()
    def the_bookkeeping_adds_up(self) -> None:
        status = self.scheduler.status()
        counters = status.counters
        assert counters.commands_accepted + counters.commands_rejected == self.submitted
        assert status.queued_tasks <= 8
        assert not status.degraded or status.fault.failures >= 3  # degraded needs 3 failures

    def teardown(self) -> None:
        world = self.world
        world.close()
        events = [e.record_key for e in world.events()]
        assert len(events) == len(set(events))
        counters = world.scheduler.status().counters
        in_windows = sum(w.n_frames for w in world.windows())
        in_sweeps = sum(
            int(cell["n_frames"]) for r in world.results for cell in r.data.get("cells", [])
        )
        accounted = in_windows + in_sweeps + counters.discarded_frames
        assert accounted == world.fast.frames_pushed, "a window was lost"
        rows = sum(len(batch) for _, batch in world.writer.metrics)
        assert rows == world.fast.frames_pushed, "metric rows were lost"


# A run costs about 0.15 seconds, so the default profile keeps the search short. The thorough
# profile (`HYPOTHESIS_PROFILE=thorough`) searches deeper.
THOROUGH = os.environ.get("HYPOTHESIS_PROFILE") == "thorough"

SchedulerMachine.TestCase.settings = settings(
    max_examples=600 if THOROUGH else 25,
    stateful_step_count=30 if THOROUGH else 14,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)
TestSchedulerMachine = SchedulerMachine.TestCase


class DaylightMachine(SchedulerMachine):
    """The same rules in daylight, where the scheduler starts in safe and the sky is too bright."""

    START_UTC_NS = BRIGHT


DaylightMachine.TestCase.settings = settings(
    max_examples=300 if THOROUGH else 15,
    stateful_step_count=30 if THOROUGH else 12,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)
TestDaylightMachine = DaylightMachine.TestCase
