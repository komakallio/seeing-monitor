"""The scheduler: one owner of the camera that shares its time between the measurement modes.

The scheduler works only against interfaces: a `CameraDriver`, a `FastAnalyzer`, a
`SurveyAnalyzer`, a `PointingProvider`, a `RecordWriter`, and a `MetricsWriter`. It takes a
`Clock` for all time, and the profile and the configuration for every setting. See the
"Scheduler" section of `docs/architecture.md` for the behavior, and `seeingmon.scheduler.machine`
for the states.

**Threads.** One thread runs the loop (`run`), and it is the only thread that touches the camera,
the analyzers, and the writers. Any thread may call `submit`, `touch_alignment`, and `status`.
`submit` changes the logical state under a lock and returns at once. The loop notices the change
at its next step and moves the camera to match. A command therefore takes effect after the unit
of work in progress, which is at most one exposure.

**Steps.** `step` does one unit of work: it reads a frame, runs a transition, takes a survey
exposure, or sleeps through the clock until the next action. With a `VirtualClock` a step never
waits in real time, so `run_until` runs a night in seconds.

**Time.** Durations and deadlines (the length of a window, the survey cadence, the watch interval,
the backoff) use the monotonic clock, so a step of the wall clock cannot stretch a window or stall
the cycle. Records and the ephemeris use UTC.

**One reconfiguration function.** Every change to the camera goes through `_reconfigure`: it ends
the stream in use (flushing the fast analyzer's window and its metrics), calls
`CameraDriver.configure`, and then `FastAnalyzer.begin_stream`. A window therefore never spans two
streams, and the camera never serves two modes at once.
"""

from __future__ import annotations

import contextlib
import math
import re
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from seeingmon import __version__
from seeingmon.analysis import (
    FastAnalyzer,
    FastContext,
    MetricsWriter,
    PointingProvider,
    RecordWriter,
    StarState,
    SurveyAnalyzer,
    SurveyOutput,
)
from seeingmon.clock import NS_PER_S, Clock
from seeingmon.config import Config
from seeingmon.drivers.base import CameraDriver, CameraError, CameraStateError, RecoveryLevel
from seeingmon.frames import ActiveStream, Frame, PixelFormat, Roi, StreamConfig, StreamKind
from seeingmon.profile import Profile
from seeingmon.records import EventRecord, Record, SeeingWindowRecord, field_specs
from seeingmon.scheduler.commands import (
    TASK_KINDS,
    Command,
    CommandResult,
    Pause,
    QueueBurst,
    QueueDark,
    QueueReplay,
    QueueSweep,
    RejectReason,
    Resume,
    StartAlignment,
    StopAlignment,
)
from seeingmon.scheduler.commission import (
    CommissionHandler,
    CommissionResult,
    CommissionTask,
    FastWindowSample,
    FrameStatsAccumulator,
    SweepHandler,
    SweepPlan,
    TaskQueue,
)
from seeingmon.scheduler.config import (
    MIN_DARK_FRAMES,
    SchedulerConfig,
    SiteConfig,
    load_site,
    seconds_to_us,
)
from seeingmon.scheduler.ephemeris import polaris_zenith_angle_deg, sun_elevation_deg
from seeingmon.scheduler.faults import FaultPlan, FaultTracker
from seeingmon.scheduler.gates import (
    CloudTracker,
    DaylightGate,
    sky_background_fraction,
)
from seeingmon.scheduler.levels import DESTRUCTIVE_STEPS, EscalationLevel, step_name
from seeingmon.scheduler.machine import State, StateMachine
from seeingmon.scheduler.roi import roi_at_sensor_center, roi_centered_on
from seeingmon.scheduler.status import (
    Counters,
    FaultStatus,
    SchedulerStatus,
    StreamInfo,
)

_MIN_SLEEP_NS = 1_000_000  # a sleep is at least 1 ms, so every step moves the clock forward
_OVERRUN_TOLERANCE_NS = NS_PER_S  # a cycle that starts later than this counts as an overrun
_CLOCK_CHECK_NS = 5 * NS_PER_S  # how often the loop asks the clock whether it is synchronized
_METRICS_DRAIN_NS = NS_PER_S  # drain per-frame metrics at least once per second of frame time
_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_MAX_EVENT_REVISIONS = 64

# The reasons that go with a move into `safe`, in the `state_change` event and in the status.
SAFE_STARTUP = "startup"
SAFE_FAULT = "fault"


class StepKind(StrEnum):
    """What one call to `Scheduler.step` did."""

    FRAME = "frame"  # read and analyzed one frame
    WORK = "work"  # configured the camera, or took a watch or survey frame
    SLEEP = "sleep"  # slept through the clock until the next action
    TRANSITION = "transition"  # changed state, or moved to another phase of the cycle
    FAULT = "fault"  # a camera error ended an activity
    RECOVERY = "recovery"  # performed a recovery step after the backoff
    TASK = "task"  # ran a commissioning task


class Purpose(StrEnum):
    """What a stream serves. The status reports it."""

    FAST = "fast"
    SURVEY = "survey"
    WATCH = "watch"
    ALIGN = "align"
    COMMISSION = "commission"


class Phase(StrEnum):
    """Where `auto` stands in its cycle."""

    BEGIN = "begin"  # a cycle boundary: check the gates, run tasks, wait for the slot
    FAST = "fast"  # a fast period runs
    SURVEY = "survey"  # the survey step runs: a short exposure, then a long one
    SOLVE_WAIT = "solve_wait"  # the survey step ran, and the pointing solution has not arrived


@dataclass(slots=True)
class _Cycle:
    phase: Phase = Phase.BEGIN
    slot_start_mono: int = 0
    next_slot_mono: int = 0
    anchored: bool = False  # the next slot is `slot_start_mono` plus the cadence at that time
    cadence_ns: int = 0  # the cadence in effect when the period began, to spot an overrun
    survey_stage: int = 0  # 0 for the short exposure, 1 for the long one
    survey_forced: bool = False  # a solve is needed, so the next fast period starts at once
    solve_deadline_mono: int = 0


@dataclass(slots=True)
class _FastRun:
    stream_id: int
    started_mono: int
    window_ns: int
    read_timeout_s: float
    last_recenter_mono: int
    last_drain_frame_ns: int | None = None
    last_drain_mono: int = 0
    missing_frames: int = 0
    frames: int = 0
    edge_blocked: bool = False  # the ROI cannot center the star, so stop cutting windows for it


@dataclass(slots=True)
class _AlignSession:
    exposure_us: int
    gain: int
    last_activity_mono: int
    dirty: bool = True  # the camera needs (re)configuring


@dataclass(slots=True)
class _PendingFault:
    plan: FaultPlan
    due_mono: int
    due_utc_ns: int


class Scheduler:
    """One scheduler owns the camera. Build it with the collaborators, then call `run`.

    Args:
        driver: The camera. It is the real driver in tests and a proxy to `acquire` in production.
        fast: The fast analyzer. The scheduler calls it from its own thread.
        survey: The survey analyzer. It returns its results later through `poll`.
        pointing: Where Polaris is on the sensor.
        records: Where windows, survey records, and events go.
        metrics: Where per-frame metric rows go.
        clock: The time source. A `VirtualClock` makes a night run in seconds.
        profile: The hardware profile. The scheduler reads the readout modes, the ROI rules, the
            limits, and the saturation levels from it.
        station_id: The ID that tags every event.
        config: The `[scheduler]` table. The defaults apply when you leave it out.
        site: The observing site for the Sun's elevation. Without it the scheduler relies on the
            measured sky background and sets no `twilight` flag.
        escalate: The supervisor's callback for the ladder steps above the driver's. It runs on the
            loop thread, so return when the step is done or has failed. Without it the ladder stops
            at the last driver step.
        context_provider: Called with the UTC time, it returns the context that only the core
            knows, such as `heater_duty` and the `heater_on` flag. The scheduler adds its own.
        alignment_sink: Receives every frame of the alignment stream.
        result_sink: Receives every commissioning result, so the core can store it as pinned.
    """

    def __init__(
        self,
        *,
        driver: CameraDriver,
        fast: FastAnalyzer,
        survey: SurveyAnalyzer,
        pointing: PointingProvider,
        records: RecordWriter,
        metrics: MetricsWriter,
        clock: Clock,
        profile: Profile,
        station_id: str,
        config: SchedulerConfig | None = None,
        site: SiteConfig | None = None,
        escalate: Callable[[EscalationLevel], None] | None = None,
        context_provider: Callable[[int], FastContext] | None = None,
        alignment_sink: Callable[[Frame], None] | None = None,
        result_sink: Callable[[CommissionResult], None] | None = None,
    ) -> None:
        self._driver = driver
        self._fast = fast
        self._survey = survey
        self._pointing = pointing
        self._records = records
        self._metrics = metrics
        self._clock = clock
        self._profile = profile
        self._station_id = station_id
        self._config = config or SchedulerConfig()
        self._site = site
        self._escalate = escalate
        self._context_provider = context_provider
        self._alignment_sink = alignment_sink
        self._result_sink = result_sink

        self._fast_mode = profile.fast_mode.mode
        self._survey_mode = profile.survey_mode.mode
        self._fast_format = profile.fast_mode.pixel_format or PixelFormat.RAW16
        self._survey_format = profile.survey_mode.pixel_format or PixelFormat.RAW16
        self._validate_against_profile()

        self._gate = DaylightGate(self._config.daylight)
        self._cloud = CloudTracker(self._config.cloud)
        self._faults = FaultTracker(self._config.faults, self._config.ladder)

        now_utc = clock.utc_ns()
        now_mono = clock.monotonic_ns()
        # State shared with other threads. The lock guards these fields.
        self._lock = threading.RLock()
        self._machine = StateMachine(State.SAFE, now_utc_ns=now_utc, reason=SAFE_STARTUP)
        self._align: _AlignSession | None = None
        self._queue = TaskQueue(self._config.commission.max_queued)
        self._handlers: dict[str, CommissionHandler] = {"sweep": SweepHandler()}
        self._outbox: deque[EventRecord] = deque()
        self._event_revisions: dict[int, int] = {}
        self._next_task_id = 1
        self._running_task: CommissionTask | None = None  # popped from the queue, and not done
        self._pause_after: str | None = None  # why a commission episode ends in `paused`
        self._next_watch_mono = now_mono
        self._closed = False

        # State of the loop thread.
        self._started = False
        self._clock_ok = True  # the clock is synchronized, as far as the scheduler knows
        self._clock_check_mono = now_mono
        self._opened = False
        self._stream: StreamInfo | None = None
        self._active: ActiveStream | None = None
        self._stream_running = False
        self._activity: Purpose | None = None
        self._fast_run: _FastRun | None = None
        self._cycle = _Cycle(next_slot_mono=now_mono)
        self._return_state = State.SAFE
        self._pending_fault: _PendingFault | None = None
        self._last_error: str | None = None
        self._background_fraction: float | None = None
        self._last_temperature_c: float | None = None
        self._last_context: FastContext | None = None
        self._context_next_mono = now_mono
        self._last_survey_mono: int | None = None
        self._next_poll_mono = now_mono
        self._survey_pending = 0  # a copy for `status`, which runs on other threads
        self._deadline_mono: int | None = None
        self._results: deque[CommissionResult] = deque(maxlen=self._config.commission.max_results)
        self._counters = Counters()
        self._stop_event: threading.Event | None = None
        self._sink_error_reported = False

    # --- Public API ------------------------------------------------------------------------

    @property
    def state(self) -> State:
        """The state of the scheduler right now."""
        with self._lock:
            return self._machine.state

    @property
    def stream(self) -> StreamInfo | None:
        """The stream that the camera runs or ran last."""
        return self._stream

    @property
    def config(self) -> SchedulerConfig:
        """The configuration that the scheduler runs with."""
        return self._config

    def register_handler(self, kind: str, handler: CommissionHandler) -> None:
        """Register the handler for a kind of commissioning task: `burst`, `replay`, or your own.

        A registered handler replaces an earlier one of the same kind. The scheduler registers
        the `sweep` handler itself. Register the others before you start the loop.
        """
        if not _KIND_PATTERN.fullmatch(kind):
            raise ValueError("a task kind is lowercase letters, digits, and underscores")
        with self._lock:
            self._handlers[kind] = handler

    def results(self) -> tuple[CommissionResult, ...]:
        """The latest commissioning results, oldest first. The scheduler keeps a bounded number."""
        with self._lock:
            return tuple(self._results)

    def touch_alignment(self) -> None:
        """Tell the scheduler that someone uses the alignment helper, which restarts the idle timer.

        Call it from any thread while a person watches the live view or moves the mount.
        """
        with self._lock:
            if self._align is not None:
                self._align.last_activity_mono = self._clock.monotonic_ns()

    def submit(self, command: Command) -> CommandResult:
        """Hand a command to the scheduler. Safe to call from any thread, and it returns at once.

        The result says whether the scheduler accepted the command, and it names a reason when it
        did not. The scheduler writes an event for the command in both cases.
        """
        with self._lock:
            if self._closed:
                result = self._reject(RejectReason.CLOSED, "the scheduler has shut down")
            elif isinstance(command, StartAlignment):
                result = self._start_alignment(command)
            elif isinstance(command, StopAlignment):
                result = self._stop_alignment()
            elif isinstance(command, Pause):
                result = self._pause()
            elif isinstance(command, Resume):
                result = self._resume()
            elif isinstance(command, QueueBurst | QueueSweep | QueueReplay | QueueDark):
                result = self._queue_task(command)
            else:
                result = self._reject(RejectReason.INVALID, f"unknown command {command!r}")
            self._record_command(command, result)
            return result

    def status(self) -> SchedulerStatus:
        """A snapshot of the scheduler for `/status` and the `health` record."""
        with self._lock:
            now_mono = self._clock.monotonic_ns()
            align = self._align
            pending = self._pending_fault
            return SchedulerStatus(
                t_utc_ns=self._clock.utc_ns(),
                state=self._machine.state.value,
                state_reason=self._machine.reason,
                state_since_utc_ns=self._machine.entered_utc_ns,
                last_transition_utc_ns=self._machine.last_transition_utc_ns,
                degraded=self._faults.degraded,
                stream=self._stream,
                cloud=self._cloud.active,
                cloud_fraction=self._cloud.fraction,
                twilight=self._gate.is_twilight(self._sun_elevation()),
                sun_elevation_deg=self._sun_elevation(),
                background_fraction=self._background_fraction,
                sensor_temperature_c=self._last_temperature_c,
                counters=replace(self._counters),
                fault=FaultStatus(
                    failures=self._faults.failures,
                    good_frames=self._faults.good_frames,
                    last_error=self._last_error,
                    next_attempt_utc_ns=None if pending is None else pending.due_utc_ns,
                    next_step=None if pending is None else step_name(pending.plan.step),
                ),
                queued_tasks=len(self._queue),
                survey_pending=self._survey_pending,
                alignment_idle_s=(
                    None if align is None else (now_mono - align.last_activity_mono) / NS_PER_S
                ),
            )

    def step(self) -> StepKind:
        """Do one unit of work and return what it was.

        The unit is a frame, a transition, an exposure, a recovery step, a commissioning task, or a
        sleep through the clock. Call it from one thread only, the one that owns the scheduler.
        """
        if self._closed:
            raise RuntimeError("the scheduler is closed")
        self._flush_events()
        try:
            self._poll_survey()
            kind = self._step_once()
        finally:
            self._flush_events()
        return kind

    def run_until(self, t_utc_ns: int) -> None:
        """Step until the clock reaches `t_utc_ns`. Sleeps never overshoot it.

        A frame read or an exposure can run past it by one frame period or one exposure.
        """
        self._deadline_mono = self._clock.monotonic_ns() + (t_utc_ns - self._clock.utc_ns())
        try:
            while self._clock.utc_ns() < t_utc_ns:
                self.step()
        finally:
            self._deadline_mono = None

    def run(self, stop_event: threading.Event) -> None:
        """The thread entry point: step until `stop_event` is set, then close.

        A step that raises an unexpected error writes an `error` event and the loop goes on. Ten
        errors in a row end the loop with the last error. In both cases the scheduler closes the
        camera on the way out.
        """
        self._stop_event = stop_event
        errors = 0
        try:
            while not stop_event.is_set():
                try:
                    self.step()
                    errors = 0
                except Exception as error:
                    errors += 1
                    self._emit(
                        "error",
                        "scheduler.internal_error",
                        f"The scheduler hit an unexpected error: {type(error).__name__}.",
                        {"error": f"{type(error).__name__}: {error}", "in_a_row": errors},
                    )
                    self._flush_events()
                    if errors >= 10:
                        raise
                    self._clock.sleep(1.0)
        finally:
            self.close()

    def close(self) -> None:
        """End the stream, flush the analyzer, and close the camera. Safe to call twice.

        Call it from the thread that runs the loop (`run` does), or after the loop ended. It
        touches the camera and the analyzer, which no other thread may do while the loop runs.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._end_stream("close")
            with contextlib.suppress(CameraError):
                self._driver.close()
            self._opened = False
        finally:
            self._flush_events()

    # --- Commands (called with the lock held) ----------------------------------------------

    def _accept(self, message: str, *, task_id: int | None = None) -> CommandResult:
        return CommandResult(
            accepted=True, message=message, state=self._machine.state.value, task_id=task_id
        )

    def _reject(self, reason: RejectReason, message: str) -> CommandResult:
        return CommandResult(
            accepted=False, message=message, state=self._machine.state.value, reason=reason
        )

    def _record_command(self, command: Command, result: CommandResult) -> None:
        if result.accepted:
            self._counters.commands_accepted += 1
        else:
            self._counters.commands_rejected += 1
        name = type(command).__name__
        verb = "Accepted" if result.accepted else "Rejected"
        self._emit(
            "info" if result.accepted else "warning",
            "scheduler.command",
            f"{verb} {name}: {result.message}",
            {
                "command": name,
                "accepted": result.accepted,
                "reason": None if result.reason is None else result.reason.value,
                "task_id": result.task_id,
                "state": result.state,
            },
        )

    def _start_alignment(self, command: StartAlignment) -> CommandResult:
        state = self._machine.state
        if state is State.PAUSED:
            return self._reject(RejectReason.PAUSED, "the scheduler is paused; resume it first")
        if self._faults.degraded:
            return self._reject(RejectReason.DEGRADED, "the camera has failed repeatedly")
        align = self._config.align
        exposure_s = align.exposure_s if command.exposure_s is None else command.exposure_s
        gain = align.gain if command.gain is None else command.gain
        if not 0 < exposure_s < float("inf"):
            return self._reject(
                RejectReason.INVALID, "exposure_s must be a positive number of seconds"
            )
        exposure_us = seconds_to_us(exposure_s)
        problem = self._check_exposure_and_gain(exposure_us, gain)
        if problem is not None:
            return self._reject(RejectReason.INVALID, problem)
        now = self._clock.monotonic_ns()
        if state is State.ALIGN:
            session = self._align
            if session is None:  # the session ended a moment ago, so start it again
                self._align = _AlignSession(exposure_us, gain, now)
            else:
                if (session.exposure_us, session.gain) != (exposure_us, gain):
                    session.exposure_us, session.gain, session.dirty = exposure_us, gain, True
                session.last_activity_mono = now
            return self._accept("alignment already runs, so the idle timer restarted")
        self._align = _AlignSession(exposure_us=exposure_us, gain=gain, last_activity_mono=now)
        self._transition("alignment started", State.ALIGN)
        return self._accept("alignment started")

    def _stop_alignment(self) -> CommandResult:
        if not self._end_alignment("alignment stopped"):
            return self._reject(RejectReason.NOT_ALIGNING, "no alignment runs")
        return self._accept("alignment stopped")

    def _end_alignment(self, reason: str) -> bool:
        """Leave `align` for `safe`. The session and the state change in one hold of the lock.

        The next brightness frame follows at once. Returns `False` when no alignment runs.
        """
        with self._lock:
            if self._machine.state is not State.ALIGN:
                return False
            self._align = None
            self._next_watch_mono = self._clock.monotonic_ns()
            return self._transition(reason, State.SAFE, expect=State.ALIGN)

    def _pause(self) -> CommandResult:
        if self._machine.state is State.PAUSED:
            return self._reject(RejectReason.ALREADY_PAUSED, "the scheduler is already paused")
        self._align = None
        self._transition("pause command", State.PAUSED)
        return self._accept("the scheduler paused, and nothing runs until you resume it")

    def _resume(self) -> CommandResult:
        if self._machine.state is not State.PAUSED:
            return self._reject(RejectReason.NOT_PAUSED, "the scheduler is not paused")
        self._next_watch_mono = self._clock.monotonic_ns()
        self._transition("resume command", State.SAFE)
        return self._accept("the scheduler resumed in safe and checks the sky")

    def _queue_task(
        self, command: QueueBurst | QueueSweep | QueueReplay | QueueDark
    ) -> CommandResult:
        kind = TASK_KINDS[type(command)]
        if kind not in self._handlers:
            return self._reject(RejectReason.NO_HANDLER, f"no handler is registered for {kind}")
        problem = self._check_task(command)
        if problem is not None:
            return self._reject(RejectReason.INVALID, problem)
        if isinstance(command, QueueDark) and self._dark_task_pending():
            return self._reject(
                RejectReason.BUSY,
                "a dark session is already queued or running; wait until it ends",
            )
        task = CommissionTask(
            task_id=self._next_task_id,
            kind=kind,
            command=command,
            submitted_utc_ns=self._clock.utc_ns(),
            priority=command.priority,
        )
        if not self._queue.push(task):
            return self._reject(RejectReason.QUEUE_FULL, "the commissioning queue is full")
        self._next_task_id += 1
        return self._accept(self._queued_message(kind, command), task_id=task.task_id)

    def _queued_message(self, kind: str, command: Command) -> str:
        """Say when a task that just joined the queue starts. The caller holds the lock."""
        state = self._machine.state
        if state is State.PAUSED:
            return f"the {kind} is queued, and the scheduler is paused: it runs after you resume"
        if state is State.ALIGN:
            return f"the {kind} is queued: it runs after the alignment helper ends"
        if self._faults.degraded:
            return f"the {kind} is queued: it runs after the camera recovers"
        if isinstance(command, QueueDark) and command.immediate:
            return f"the {kind} is queued and starts at the next step"
        return f"the {kind} is queued and runs at the next cycle boundary"

    def _dark_task_pending(self) -> bool:
        """Whether a dark task waits in the queue or runs. The caller holds the lock."""
        running = self._running_task
        return (running is not None and running.kind == "dark") or any(
            task.kind == "dark" for task in self._queue.tasks()
        )

    def _check_dark(self, command: QueueDark) -> str | None:
        """Return a problem with the settings of a dark task. A `None` field takes the default."""
        limits = self._config.dark
        for name, count in (("frames", command.frames), ("bias_frames", command.bias_frames)):
            if count is not None and not MIN_DARK_FRAMES <= count <= limits.max_frames:
                return f"{name} must be between {MIN_DARK_FRAMES} and {limits.max_frames}"
        seconds = command.exposure_s
        if seconds is not None:
            if not (math.isfinite(seconds) and seconds > 0):
                return "exposure_s must be a positive number of seconds"
            low_us, high_us = self._profile.limits.exposure_us_range
            if not low_us <= seconds_to_us(seconds) <= high_us:
                return f"exposure_s must be between {low_us / 1e6:g} and {high_us / 1e6:g} seconds"
        timeout = command.wait_for_cover_timeout_s
        if timeout is not None and not (
            math.isfinite(timeout) and 0 < timeout <= limits.max_cover_wait_s
        ):
            return (
                "wait_for_cover_timeout_s must be a positive number of seconds, "
                f"at most {limits.max_cover_wait_s:g}"
            )
        if len(command.label) > limits.max_label_chars:
            return f"the label has at most {limits.max_label_chars} characters"
        return None

    def _check_task(self, command: QueueBurst | QueueSweep | QueueReplay | QueueDark) -> str | None:
        """Return a problem with the task's settings, or `None` when they are fine."""
        if isinstance(command, QueueBurst):
            if not (command.duration_s > 0 and command.duration_s < float("inf")):
                return "duration_s must be a positive number of seconds"
            stream = command.stream
            if stream is not None:
                if stream.mode not in {mode.name for mode in self._profile.readout_modes}:
                    return f"unknown readout mode {stream.mode!r}"
                return self._check_exposure_and_gain(stream.exposure_us, stream.gain)
        elif isinstance(command, QueueSweep):
            try:
                SweepPlan.resolve(command, self._config.sweep, self._profile)
            except ValueError as error:
                return str(error)
        elif isinstance(command, QueueDark):
            return self._check_dark(command)
        elif not (command.speed >= 0 and command.speed < float("inf")):
            return "speed must be zero (as fast as possible) or a positive factor"
        return None

    def _check_exposure_and_gain(self, exposure_us: int, gain: int) -> str | None:
        limits = self._profile.limits
        low_us, high_us = limits.exposure_us_range
        if not low_us <= exposure_us <= high_us:
            return f"the exposure must be between {low_us} and {high_us} microseconds"
        low_gain, high_gain = limits.gain_range
        if not low_gain <= gain <= high_gain:
            return f"the gain must be between {low_gain} and {high_gain}"
        return None

    # --- State changes ---------------------------------------------------------------------

    def _transition(self, reason: str, to_state: State, *, expect: State | None = None) -> bool:
        """Move to `to_state` and write the event. Returns `False` when `expect` no longer holds."""
        with self._lock:
            before = self._machine.state
            if expect is not None and before is not expect:
                return False
            change = self._machine.transition(to_state, reason, self._clock.utc_ns())
            self._counters.transitions += 1
            self._emit(
                "info",
                "scheduler.state_change",
                f"The scheduler moved from {before.value} to {to_state.value}: {reason}.",
                {"from": before.value, "to": to_state.value, "reason": reason},
                t_utc_ns=change.t_utc_ns,
            )
            return True

    def _enter_auto(self) -> None:
        """Start a fresh cycle. The first fast period begins at once."""
        self._fast_run = None
        self._cycle = _Cycle(next_slot_mono=self._clock.monotonic_ns())

    def _enter_safe(self, reason: str, *, expect: State) -> bool:
        """Leave `expect` for `safe`. The next brightness frame follows one interval later."""
        if not self._transition(reason, State.SAFE, expect=expect):
            return False
        self._next_watch_mono = self._clock.monotonic_ns() + round(
            self._config.watch.interval_s * NS_PER_S
        )
        return True

    # --- Events ----------------------------------------------------------------------------

    def _emit(
        self,
        level: str,
        kind: str,
        message: str,
        detail: Mapping[str, Any] | None = None,
        *,
        t_utc_ns: int | None = None,
    ) -> None:
        """Queue an event. The loop thread writes it, so no other thread touches the writer."""
        t_ns = self._clock.utc_ns() if t_utc_ns is None else t_utc_ns
        with self._lock:
            revision = self._next_event_revision(t_ns)
            fields: dict[str, Any] = {
                "station_id": self._station_id,
                "t_utc_ns": t_ns,
                "revision": revision,
                "profile_id": self._profile.id,
                "provenance": {"scheduler": __version__},
                "level": level,
                "kind": kind,
                "message": message or kind,
            }
            try:
                record = EventRecord(**fields, detail=None if detail is None else dict(detail))
            except ValueError:  # the detail is not JSON, which a handler can cause
                record = EventRecord(**fields, detail={"detail_dropped": True})
            self._outbox.append(record)

    def _next_event_revision(self, t_utc_ns: int) -> int:
        """Give events that share a time the next revision, so that every key stays unique."""
        revision = self._event_revisions.get(t_utc_ns, -1) + 1
        self._event_revisions[t_utc_ns] = revision
        if len(self._event_revisions) > _MAX_EVENT_REVISIONS:
            del self._event_revisions[next(iter(self._event_revisions))]
        return revision

    def _flush_events(self) -> None:
        while True:
            try:
                record = self._outbox.popleft()
            except IndexError:
                return
            self._records.write(record)

    # --- The loop --------------------------------------------------------------------------

    def _mono(self) -> int:
        return self._clock.monotonic_ns()

    def _step_once(self) -> StepKind:
        if not self._started:
            self._start()
        self._check_clock()
        state = self.state
        if self._reconcile(state):
            return StepKind.TRANSITION
        if self._pending_fault is not None and state is not State.PAUSED:
            return self._step_fault()
        if state is State.SAFE:
            return self._step_safe()
        if state is State.AUTO:
            return self._step_auto()
        if state is State.ALIGN:
            return self._step_align()
        if state is State.COMMISSION:
            return self._step_commission()
        return self._sleep_until(self._mono() + self._max_sleep_ns)  # paused: nothing runs

    def _start(self) -> None:
        self._started = True
        self._emit(
            "info",
            "scheduler.start",
            "The scheduler started in safe.",
            {"site_configured": self._site is not None, "profile": self._profile.id},
        )
        if self._site is None:
            self._emit(
                "warning",
                "scheduler.no_site",
                "No site is configured, so the Sun's elevation and the twilight flag are off.",
            )

    @property
    def _max_sleep_ns(self) -> int:
        return round(self._config.loop.max_sleep_s * NS_PER_S)

    def _sleep_until(self, target_mono_ns: int) -> StepKind:
        now = self._mono()
        wait_ns = min(target_mono_ns - now, self._max_sleep_ns)
        if self._deadline_mono is not None:
            wait_ns = min(wait_ns, self._deadline_mono - now)
        self._clock.sleep(max(wait_ns, _MIN_SLEEP_NS) / NS_PER_S)
        return StepKind.SLEEP

    def _reconcile(self, state: State) -> bool:
        """End a stream that does not belong to the state. Returns `True` when it did."""
        if self._activity is Purpose.FAST and state is not State.AUTO:
            self._end_stream("state_change")
            return True
        if self._activity is Purpose.ALIGN and state is not State.ALIGN:
            self._end_stream("state_change")
            return True
        if state is State.PAUSED and (self._stream_running or self._activity is not None):
            self._end_stream("pause")
            return True
        return False

    # --- The camera ------------------------------------------------------------------------

    def _timeout_s(self, active: ActiveStream) -> float:
        """The wait for one frame: a multiple of the frame period plus a margin."""
        exposure_s = active.config.exposure_us / 1e6
        period_s = max(active.frame_period_s or exposure_s, exposure_s)
        loop = self._config.loop
        return period_s * loop.read_timeout_factor + loop.read_timeout_margin_s

    def _reconfigure(self, config: StreamConfig, purpose: Purpose) -> ActiveStream:
        """Change the camera. Every reconfiguration goes through this one function.

        It ends the stream in use, which flushes the fast analyzer's window and its metrics,
        configures the camera, and tells the fast analyzer about the new stream. The caller starts
        the stream. A `CameraError` propagates to the caller, which runs the fault response.
        """
        self._end_stream("reconfigure")
        if not self._opened:
            self._driver.open()
            self._opened = True
        active = self._driver.configure(config)
        self._write_windows(self._fast.begin_stream(active))
        self._counters.reconfigurations += 1
        self._activity = purpose
        self._active = active
        self._stream = StreamInfo(
            stream_id=active.stream_id,
            purpose=purpose.value,
            mode=active.config.mode,
            exposure_us=active.config.exposure_us,
            gain=active.config.gain,
            roi=active.config.roi,
        )
        return active

    def _start_stream(self) -> None:
        self._driver.start()
        self._stream_running = True

    def _end_stream(self, reason: str) -> None:
        """End the stream in use: flush the open window, drain the metrics, and stop the camera."""
        run, self._fast_run = self._fast_run, None
        if run is not None:
            self._write_windows(self._fast.flush(reason))
            self._drain_metrics(run.stream_id)
        if self._stream_running:
            self._stream_running = False
            try:
                self._driver.stop()
            except CameraError as error:
                self._emit(
                    "warning",
                    "scheduler.stop_failed",
                    f"The camera did not stop cleanly: {error}",
                    {"error": f"{type(error).__name__}: {error}", "reason": reason},
                )
        self._activity = None

    def _write_windows(self, windows: tuple[SeeingWindowRecord, ...]) -> None:
        for window in windows:
            self._records.write(window)
            self._counters.windows += 1

    def _drain_metrics(self, stream_id: int) -> None:
        rows = self._fast.drain_metrics()
        if rows is not None and len(rows):
            self._metrics.write_metrics(stream_id, rows)

    def _drain_if_due(self, run: _FastRun, frame: Frame, now_mono: int) -> None:
        """Drain at least once per second of frame time, and once per second of monotonic time."""
        if run.last_drain_frame_ns is None:
            run.last_drain_frame_ns = frame.t_utc_ns
            run.last_drain_mono = now_mono
            return
        if (
            frame.t_utc_ns - run.last_drain_frame_ns >= _METRICS_DRAIN_NS
            or now_mono - run.last_drain_mono >= _METRICS_DRAIN_NS
        ):
            self._drain_metrics(run.stream_id)
            run.last_drain_frame_ns = frame.t_utc_ns
            run.last_drain_mono = now_mono

    def _note_frame(self, frame: Frame) -> None:
        counters = self._counters
        counters.frames += 1
        counters.dropped += frame.dropped_before
        self._last_temperature_c = frame.temperature_c
        if self._faults.success():
            self._emit(
                "info",
                "scheduler.recovered",
                "The camera works again, so the degraded status cleared.",
                {"good_frames": self._config.faults.clear_after_frames},
            )

    # --- Faults ----------------------------------------------------------------------------

    def _camera_error(self, error: Exception, where: str) -> StepKind:
        """End the activity, count the failure, and plan the recovery."""
        now = self._mono()
        self._end_stream("fault")
        self._counters.faults += 1
        plan = self._faults.failure(now, can_escalate=self._escalate is not None)
        self._last_error = f"{type(error).__name__}: {error}"
        self._pending_fault = _PendingFault(
            plan=plan,
            due_mono=now + round(plan.wait_s * NS_PER_S),
            due_utc_ns=self._clock.utc_ns() + round(plan.wait_s * NS_PER_S),
        )
        self._emit(
            "error" if plan.degraded else "warning",
            "scheduler.fault",
            f"The camera failed during {where}: {error}",
            {
                "where": where,
                "error": self._last_error,
                "failures": plan.failures,
                "wait_s": plan.wait_s,
                "next_step": step_name(plan.step),
            },
        )
        if plan.degraded_changed:
            self._emit(
                "error",
                "scheduler.degraded",
                "The camera failed repeatedly, so the status is degraded. The scheduler keeps "
                "trying at a slow pace.",
                {"failures": plan.failures},
            )
        self._cycle = _Cycle(next_slot_mono=now)
        if plan.degraded and not self._end_alignment("camera fault"):
            for state in (State.AUTO, State.COMMISSION):
                if self._transition("camera fault", State.SAFE, expect=state):
                    self._next_watch_mono = now
                    break
        return StepKind.FAULT

    def _step_fault(self) -> StepKind:
        """Wait out the backoff, then perform the recovery step."""
        pending = self._pending_fault
        assert pending is not None
        if self._mono() < pending.due_mono:
            return self._sleep_until(pending.due_mono)
        self._pending_fault = None
        self._perform_ladder_step(pending.plan)
        if self._pending_fault is None:
            self._next_watch_mono = self._mono()
        return StepKind.RECOVERY

    def _perform_ladder_step(self, plan: FaultPlan) -> None:
        step = plan.step
        name = step_name(step)
        supervisor_level = isinstance(step, EscalationLevel)
        self._counters.recovery_steps += 1
        try:
            if isinstance(step, RecoveryLevel):
                self._driver.recover(step)
            else:
                assert self._escalate is not None
                self._escalate(step)
        except Exception as error:
            # The driver raises a `CameraError`, but the callback is outside code that can raise
            # anything. Either way, the step failed, and the failure counts.
            self._emit(
                "error",
                "scheduler.recovery_step",
                f"The recovery step {name} failed: {error}",
                {"step": name, "level": int(step), "ok": False, "failures": plan.failures},
            )
            self._camera_error(error, f"the recovery step {name}")
            return
        if supervisor_level:
            self._counters.escalations += 1
            self._opened = False  # the supervisor restarted something, so open the camera again
            if step in DESTRUCTIVE_STEPS:
                self._faults.note_destructive(self._mono())
        self._emit(
            "warning" if supervisor_level else "info",
            "scheduler.recovery_step",
            f"The scheduler performed the recovery step {name}.",
            {"step": name, "level": int(step), "ok": True, "failures": plan.failures},
        )

    # --- The `safe` state ------------------------------------------------------------------

    def _watch_config(self) -> StreamConfig:
        watch = self._config.watch
        roi = roi_at_sensor_center(self._profile, self._survey_mode, watch.roi_arcmin)
        return StreamConfig(
            mode=self._survey_mode,
            exposure_us=watch.exposure_us,
            gain=watch.gain,
            pixel_format=self._survey_format,
            roi=roi,
            kind=StreamKind.SNAPSHOT,
        )

    def _check_clock(self) -> None:
        """Ask the clock whether it is synchronized, at most once every few seconds.

        Without synchronization the UTC time can be days off (a Pi 4 has no real-time clock), so
        the Sun's elevation means nothing. The scheduler then relies on the measured sky alone, sets
        no `twilight` flag, and marks windows and survey results `time_invalid`.
        """
        now = self._mono()
        if now < self._clock_check_mono:
            return
        self._clock_check_mono = now + _CLOCK_CHECK_NS
        trusted = self._clock.status().synchronized is not False
        if trusted == self._clock_ok:
            return
        self._clock_ok = trusted
        self._emit(
            "info" if trusted else "warning",
            "scheduler.clock_synchronized" if trusted else "scheduler.clock_unsynchronized",
            "The clock is synchronized again, so the Sun's elevation applies."
            if trusted
            else "The clock is not synchronized, so the Sun's elevation and the times are off.",
            {"synchronized": trusted},
        )
        self._refresh_context(force=True)

    def _sun_elevation(self) -> float | None:
        if self._site is None or not self._clock_ok:
            return None
        return sun_elevation_deg(
            self._clock.utc_ns(), self._site.latitude_deg, self._site.longitude_deg
        )

    def _step_safe(self) -> StepKind:
        if self._tasks_ready() and self._enter_commission(State.SAFE):
            return StepKind.TRANSITION
        due = self._next_watch_mono
        if self._mono() < due:
            return self._sleep_until(due)
        return self._watch_step()

    def _watch_step(self) -> StepKind:
        """Take one brightness frame, and go to `auto` when the sky and the Sun allow it."""
        started = self._mono()
        try:
            active = self._reconfigure(self._watch_config(), Purpose.WATCH)
            self._start_stream()
            frame = self._driver.read_frame(self._timeout_s(active))
        except CameraError as error:
            return self._camera_error(error, "the brightness watch")
        self._end_stream("snapshot_done")
        self._note_frame(frame)
        self._counters.watch_frames += 1
        self._background_fraction = sky_background_fraction(frame, self._profile)
        self._next_watch_mono = started + round(self._config.watch.interval_s * NS_PER_S)
        decision = self._gate.evaluate(
            sun_elevation_deg=self._sun_elevation(),
            background_fraction=self._background_fraction,
            running=False,
        )
        if decision.allowed and self._transition(
            "the sky is dark enough", State.AUTO, expect=State.SAFE
        ):
            self._enter_auto()
        return StepKind.WORK

    # --- The `auto` state ------------------------------------------------------------------

    def _step_auto(self) -> StepKind:
        phase = self._cycle.phase
        if (
            phase is not Phase.BEGIN  # at a boundary, `_auto_begin` runs the tasks after the gates
            and self._immediate_task_waits()
            and self._enter_commission(State.AUTO, "a dark session starts at once")
        ):
            # The fast stream, if one runs, ends in `_reconcile` on the next step, with its window
            # flushed. An exposure that was in progress has finished, because the step that reads
            # it does not return before. What is left of the survey step is skipped, and the cycle
            # starts again when commissioning is done.
            return StepKind.TRANSITION
        if phase is Phase.FAST:
            return self._fast_step()
        if phase is Phase.SURVEY:
            return self._survey_step()
        if phase is Phase.SOLVE_WAIT:
            return self._solve_wait_step()
        return self._auto_begin()

    def _auto_begin(self) -> StepKind:
        """A cycle boundary: check the gates, run waiting tasks, wait for the slot, and start."""
        now = self._mono()
        cycle = self._cycle
        decision = self._gate.evaluate(
            sun_elevation_deg=self._sun_elevation(),
            background_fraction=self._background_fraction,
            running=True,
        )
        if not decision.allowed:
            reason = decision.reason or "the sky is too bright"
            self._enter_safe(reason, expect=State.AUTO)
            return StepKind.TRANSITION
        if self._tasks_ready() and self._enter_commission(State.AUTO):
            return StepKind.TRANSITION
        if cycle.anchored:
            # Read the cadence now, because a cloud result can arrive while the camera idles.
            cadence_ns = round(
                self._cloud.survey_cadence_s(self._config.survey.cadence_s) * NS_PER_S
            )
            due = cycle.slot_start_mono + cadence_ns
        else:
            due = cycle.next_slot_mono
        if now < due:
            return self._sleep_until(due)
        if cycle.anchored:
            # A change of cadence in mid-cycle is not an overrun, so judge by the longer cadence.
            latest_due = cycle.slot_start_mono + max(cycle.cadence_ns, cadence_ns)
            if now - latest_due > _OVERRUN_TOLERANCE_NS:
                self._counters.cadence_overruns += 1  # the cycle took longer than its cadence
        cycle.anchored = False
        position = self._pointing.polaris_position(self._clock.utc_ns(), self._fast_mode)
        if position is None:
            self._counters.solves_requested += 1
            self._emit(
                "warning",
                "scheduler.solve_requested",
                "No pointing solution exists, so the scheduler runs a survey step to solve.",
                {"reason": "no_solution"},
            )
            self._begin_survey(forced=True)
            return StepKind.TRANSITION
        return self._start_fast(position, now)

    def _begin_survey(self, *, forced: bool) -> None:
        cycle = self._cycle
        cycle.phase = Phase.SURVEY
        cycle.survey_stage = 0
        cycle.survey_forced = forced
        if forced:
            cycle.slot_start_mono = self._mono()

    def _start_fast(self, position: tuple[float, float], now: int) -> StepKind:
        fast = self._config.fast
        cycle = self._cycle
        cycle.slot_start_mono = now
        cycle.cadence_ns = round(
            self._cloud.survey_cadence_s(self._config.survey.cadence_s) * NS_PER_S
        )
        roi = roi_centered_on(self._profile, self._fast_mode, position, fast.roi_arcmin)
        config = StreamConfig(
            mode=self._fast_mode,
            exposure_us=fast.exposure_us,
            gain=fast.gain,
            pixel_format=self._fast_format,
            roi=roi,
            kind=StreamKind.VIDEO,
            high_speed=fast.high_speed,
        )
        try:
            active = self._reconfigure(config, Purpose.FAST)
            self._start_stream()
        except CameraError as error:
            return self._camera_error(error, "starting the fast stream")
        window_s = self._cloud.fast_window_s(fast.window_s)
        cooldown_ns = round(fast.edge_cooldown_s * NS_PER_S)
        self._fast_run = _FastRun(
            stream_id=active.stream_id,
            started_mono=now,
            window_ns=round(window_s * NS_PER_S),
            read_timeout_s=self._timeout_s(active),
            last_recenter_mono=now - cooldown_ns,  # an immediate recenter is allowed
        )
        self._refresh_context(force=True)
        cycle.phase = Phase.FAST
        return StepKind.WORK

    def _fast_step(self) -> StepKind:
        run = self._fast_run
        if run is None:  # the stream ended outside the cycle, so start the cycle again
            self._cycle = _Cycle(next_slot_mono=self._mono())
            return StepKind.TRANSITION
        fast = self._config.fast
        try:
            frame = self._driver.read_frame(run.read_timeout_s)
        except CameraError as error:
            return self._camera_error(error, "reading a fast frame")
        now = self._mono()
        self._note_frame(frame)
        update = self._fast.push(frame)
        if update.windows:
            self._write_windows(update.windows)
        run.frames += 1
        self._drain_if_due(run, frame, now)
        self._refresh_context()
        star = update.star
        if star.found:
            run.missing_frames = 0
            edge = star.edge_distance_px
            if (
                edge is not None
                and edge < fast.roi_edge_margin_px
                and not run.edge_blocked
                and now - run.last_recenter_mono >= round(fast.edge_cooldown_s * NS_PER_S)
            ):
                return self._recenter(run, star, now)
        else:
            run.missing_frames += 1
            if run.missing_frames >= fast.missing_star_frames:
                self._star_lost(run, now)
                return StepKind.FRAME
        if now - run.started_mono >= run.window_ns:
            self._end_fast_period("window_end", forced=False)
        return StepKind.FRAME

    def _recenter(self, run: _FastRun, star: StarState, now: int) -> StepKind:
        """The star neared the ROI edge: end the window early and move the ROI onto the star.

        When the profile's rules and the sensor edge leave the ROI where it is, moving it cannot
        help. The scheduler then keeps the window, stops watching the edge for this period, and
        writes one event, instead of cutting a window every cooldown.
        """
        center = (
            (star.x_px, star.y_px)
            if star.x_px is not None and star.y_px is not None
            else self._pointing.polaris_position(self._clock.utc_ns(), self._fast_mode)
        )
        previous = None if self._stream is None else self._stream.roi
        roi = (
            None
            if center is None
            else roi_centered_on(
                self._profile, self._fast_mode, center, self._config.fast.roi_arcmin
            )
        )
        if roi is None or (previous is not None and (roi.x, roi.y) == (previous.x, previous.y)):
            run.edge_blocked = True
            self._emit(
                "warning",
                "scheduler.roi_at_limit",
                "The star is near the ROI edge, and the ROI cannot move closer to it, so the "
                "scheduler keeps the window.",
                {"edge_distance_px": star.edge_distance_px, "roi": _roi_dict(previous)},
            )
            return StepKind.FRAME
        self._counters.early_window_ends += 1
        self._counters.roi_recenters += 1
        self._write_windows(self._fast.flush("edge"))
        self._drain_metrics(run.stream_id)
        try:
            applied = self._driver.move_roi(roi.x, roi.y)
        except CameraError as error:
            return self._camera_error(error, "moving the ROI")
        run.last_recenter_mono = now
        if self._stream is not None:
            self._stream = replace(self._stream, roi=applied)
        self._emit(
            "info",
            "scheduler.roi_recentered",
            "The star neared the ROI edge, so the window ended early and the ROI moved onto it.",
            {
                "edge_distance_px": star.edge_distance_px,
                "margin_px": self._config.fast.roi_edge_margin_px,
                "from": _roi_dict(previous),
                "to": _roi_dict(applied),
            },
        )
        return StepKind.FRAME

    def _star_lost(self, run: _FastRun, now: int) -> None:
        """The star was missing for the configured number of frames: solve again if allowed."""
        fast = self._config.fast
        run.missing_frames = 0
        last = self._last_survey_mono
        if last is not None and now - last < round(fast.resolve_interval_s * NS_PER_S):
            return  # a solve ran a moment ago, so keep measuring and look again later
        self._counters.solves_requested += 1
        self._counters.early_window_ends += 1
        self._emit(
            "warning",
            "scheduler.solve_requested",
            f"The star was missing for {fast.missing_star_frames} frames, so the scheduler "
            "runs a survey step to solve again.",
            {"reason": "star_missing", "frames": fast.missing_star_frames},
        )
        self._end_fast_period("star_missing", forced=True)

    def _end_fast_period(self, reason: str, *, forced: bool) -> None:
        self._end_stream(reason)
        self._counters.fast_periods += 1
        self._begin_survey(forced=forced)

    def _survey_exposure(self, stage: int) -> tuple[int, int]:
        survey = self._config.survey
        if stage == 0:
            return survey.short_exposure_us, survey.short_gain
        return survey.long_exposure_us, survey.long_gain

    def _survey_step(self) -> StepKind:
        """One survey exposure: the short one first, then the long one."""
        cycle = self._cycle
        survey = self._config.survey
        if cycle.survey_stage == 0 and self._survey.pending() >= survey.max_pending:
            self._counters.survey_skipped += 1
            self._emit(
                "warning",
                "scheduler.survey_skipped",
                "The survey analysis is behind, so the scheduler skipped this survey step.",
                {"pending": self._survey.pending(), "max_pending": survey.max_pending},
            )
            self._finish_survey(counted=False)
            return StepKind.TRANSITION
        exposure_us, gain = self._survey_exposure(cycle.survey_stage)
        config = StreamConfig(
            mode=self._survey_mode,
            exposure_us=exposure_us,
            gain=gain,
            pixel_format=self._survey_format,
            roi=None,
            kind=StreamKind.SNAPSHOT,
        )
        try:
            active = self._reconfigure(config, Purpose.SURVEY)
            self._start_stream()
            frame = self._driver.read_frame(self._timeout_s(active))
        except CameraError as error:
            return self._camera_error(error, "a survey exposure")
        self._end_stream("snapshot_done")
        self._note_frame(frame)
        self._survey.submit(frame)
        self._survey_pending = self._survey.pending()
        self._counters.survey_frames += 1
        if cycle.survey_stage == 0:
            self._background_fraction = sky_background_fraction(frame, self._profile)
            decision = self._gate.evaluate(
                sun_elevation_deg=self._sun_elevation(),
                background_fraction=self._background_fraction,
                running=True,
            )
            if not decision.allowed:  # skip the long exposure, because the sky is too bright
                self._enter_safe(decision.reason or "the sky is too bright", expect=State.AUTO)
                return StepKind.WORK
            cycle.survey_stage = 1
        else:
            self._finish_survey()
        return StepKind.WORK

    def _finish_survey(self, *, counted: bool = True) -> None:
        """The survey step is over: plan the next slot, and wait for a solution if none exists.

        A skipped step (`counted=False`) ends the same way, but it is not a step that ran.
        """
        cycle = self._cycle
        now = self._mono()
        if counted:
            self._counters.survey_steps += 1
            self._last_survey_mono = now
        cycle.anchored = not cycle.survey_forced
        cycle.next_slot_mono = now
        position = self._pointing.polaris_position(self._clock.utc_ns(), self._fast_mode)
        if position is None:
            cycle.phase = Phase.SOLVE_WAIT
            cycle.solve_deadline_mono = now + round(self._config.survey.solve_wait_s * NS_PER_S)
        else:
            cycle.phase = Phase.BEGIN

    def _solve_wait_step(self) -> StepKind:
        """Wait for the survey analysis to produce a pointing solution."""
        cycle = self._cycle
        now = self._mono()
        if self._pointing.polaris_position(self._clock.utc_ns(), self._fast_mode) is not None:
            cycle.phase = Phase.BEGIN
            cycle.next_slot_mono = now
            return StepKind.TRANSITION
        if now >= cycle.solve_deadline_mono or self._survey.pending() == 0:
            # The analysis finished, or took too long, and there is still no solution.
            cycle.phase = Phase.BEGIN
            cycle.next_slot_mono = now + round(self._config.survey.solve_retry_s * NS_PER_S)
            return StepKind.TRANSITION
        return self._sleep_until(cycle.solve_deadline_mono)

    # --- Survey results and the fast context ------------------------------------------------

    def _poll_survey(self) -> None:
        now = self._mono()
        if now < self._next_poll_mono:
            return
        self._next_poll_mono = now + round(self._config.loop.poll_interval_s * NS_PER_S)
        for output in self._survey.poll():
            self._handle_survey_output(output)
        self._survey_pending = self._survey.pending()

    def _handle_survey_output(self, output: SurveyOutput) -> None:
        self._counters.survey_results += 1
        if not output.solved:
            self._counters.survey_unsolved += 1
        flags = self._survey_flags(output)
        for record in output.records:
            self._records.write(_with_flags(record, flags))
        if self._cloud.update(output.cloud_fraction):
            started = self._cloud.active
            self._emit(
                "info",
                "scheduler.cloud",
                "Clouds were detected, so fast windows are shorter and surveys more frequent."
                if started
                else "The sky cleared, so the scheduler returned to its normal cycle.",
                {"active": started, "cloud_fraction": self._cloud.fraction},
            )
            self._refresh_context(force=True)

    def _survey_flags(self, output: SurveyOutput) -> frozenset[str]:
        flags: set[str] = set()
        if not self._clock_ok:
            flags.add("time_invalid")
        elif self._site is not None:
            elevation = sun_elevation_deg(
                output.t_utc_ns, self._site.latitude_deg, self._site.longitude_deg
            )
            if self._gate.is_twilight(elevation):
                flags.add("twilight")
        if (
            output.cloud_fraction is not None
            and output.cloud_fraction >= self._config.cloud.threshold
        ):
            flags.add("cloud")
        return frozenset(flags)

    def _refresh_context(self, *, force: bool = False) -> None:
        """Give the fast analyzer the flags and values that the frames cannot tell it."""
        now = self._mono()
        if not force and now < self._context_next_mono:
            return
        self._context_next_mono = now + round(self._config.loop.context_refresh_s * NS_PER_S)
        t_ns = self._clock.utc_ns()
        flags: set[str] = set()
        if self._cloud.active:
            flags.add("cloud")
        if self._gate.is_twilight(self._sun_elevation()):
            flags.add("twilight")
        if not self._clock_ok:
            flags.add("time_invalid")
        extra = self._context_provider(t_ns) if self._context_provider is not None else None
        zenith = None if extra is None else extra.zenith_angle_deg
        if zenith is None and self._site is not None and self._clock_ok:
            zenith = polaris_zenith_angle_deg(
                t_ns, self._site.latitude_deg, self._site.longitude_deg
            )
        context = FastContext(
            flags=frozenset(flags | (set() if extra is None else set(extra.flags))),
            heater_duty=None if extra is None else extra.heater_duty,
            zenith_angle_deg=zenith,
        )
        if context != self._last_context:
            self._fast.set_context(context)
            self._last_context = context

    # --- The `align` state -----------------------------------------------------------------

    def _step_align(self) -> StepKind:
        now = self._mono()
        with self._lock:
            session = self._align
        if session is None:
            self._end_alignment("no alignment session")
            return StepKind.TRANSITION
        idle_ns = round(self._config.align.idle_timeout_s * NS_PER_S)
        if now - session.last_activity_mono >= idle_ns:
            self._end_alignment("alignment idle timeout")
            return StepKind.TRANSITION
        if self._activity is not Purpose.ALIGN or session.dirty:
            config = StreamConfig(
                mode=self._survey_mode,
                exposure_us=session.exposure_us,
                gain=session.gain,
                pixel_format=self._survey_format,
                roi=None,
                kind=StreamKind.VIDEO,
            )
            try:
                self._reconfigure(config, Purpose.ALIGN)
                self._start_stream()
            except CameraError as error:
                return self._camera_error(error, "starting the alignment stream")
            session.dirty = False
            return StepKind.WORK
        try:
            frame = self._driver.read_frame(self._align_timeout_s(session))
        except CameraError as error:
            return self._camera_error(error, "reading an alignment frame")
        self._note_frame(frame)
        if self._alignment_sink is not None:
            try:
                self._alignment_sink(frame)
            except Exception as error:  # a broken viewer must not stop the camera
                if not self._sink_error_reported:
                    self._sink_error_reported = True
                    self._emit(
                        "warning",
                        "scheduler.alignment_sink_failed",
                        f"The alignment frame consumer raised {type(error).__name__}.",
                        {"error": f"{type(error).__name__}: {error}"},
                    )
        return StepKind.FRAME

    def _align_timeout_s(self, session: _AlignSession) -> float:
        loop = self._config.loop
        return (
            session.exposure_us / 1e6 * loop.read_timeout_factor + loop.read_timeout_margin_s + 1.0
        )

    # --- The `commission` state --------------------------------------------------------------

    def _tasks_ready(self) -> bool:
        """Whether a task waits and the camera is healthy enough to run it."""
        with self._lock:
            return len(self._queue) > 0 and not self._faults.degraded

    def _immediate_task_waits(self) -> bool:
        """Whether a queued task must not wait for the cycle boundary, and the camera can run it."""
        with self._lock:
            return not self._faults.degraded and self._queue.has(_starts_at_once)

    def _enter_commission(self, from_state: State, reason: str = "a task is queued") -> bool:
        if not self._transition(reason, State.COMMISSION, expect=from_state):
            return False
        self._return_state = from_state
        with self._lock:
            self._pause_after = None  # a request of an episode that a command cut short ends here
        return True

    def _step_commission(self) -> StepKind:
        with self._lock:
            task = self._queue.pop() if not self._faults.degraded else None
            if task is not None:
                # The task counts as running from the moment that it leaves the queue, so that
                # `submit` never sees a gap between the two.
                self._running_task = task
                if isinstance(task.command, QueueDark) and task.command.pause_after:
                    self._pause_after = (
                        "the dark session is done, and the camera may still be covered"
                    )
        if task is None:
            self._finish_commission()
            return StepKind.TRANSITION
        self._run_task(task)
        return StepKind.TASK

    def _finish_commission(self) -> None:
        with self._lock:
            pause_reason, self._pause_after = self._pause_after, None
        if pause_reason is not None:
            # Nothing may record data while the camera is covered, so the episode ends in `paused`
            # and `Resume` continues.
            self._transition(pause_reason, State.PAUSED, expect=State.COMMISSION)
            return
        target = self._return_state
        if target is State.AUTO:
            if self._transition("commissioning is done", State.AUTO, expect=State.COMMISSION):
                self._enter_auto()
        else:
            self._next_watch_mono = self._mono()
            self._transition("commissioning is done", State.SAFE, expect=State.COMMISSION)

    def _run_task(self, task: CommissionTask) -> None:
        with self._lock:
            handler = self._handlers[task.kind]
        started = self._clock.utc_ns()
        self._emit(
            "info",
            "scheduler.task_started",
            f"The {task.kind} started.",
            {"task_id": task.task_id, "kind": task.kind},
        )
        try:
            result = handler.run(task, _TaskContext(self, task))
        except CameraError as error:
            self._camera_error(error, f"the {task.kind}")
            result = _failed(task, started, self._clock.utc_ns(), f"camera error: {error}")
        except Exception as error:  # a handler bug must not stop the scheduler
            self._emit(
                "error",
                "scheduler.task_error",
                f"The {task.kind} handler raised {type(error).__name__}.",
                {"task_id": task.task_id, "error": f"{type(error).__name__}: {error}"},
            )
            result = _failed(task, started, self._clock.utc_ns(), f"handler error: {error}")
        finally:
            self._end_stream("task_end")
            with self._lock:
                self._running_task = None
        self._counters.tasks_run += 1
        with self._lock:
            self._results.append(result)
        self._emit(
            "info" if result.status == "ok" else "warning",
            f"scheduler.{result.kind}_result",
            result.summary,
            result.to_detail(),
        )
        if self._result_sink is not None:
            try:
                self._result_sink(result)
            except Exception as error:  # the store may fail, but the scheduler goes on
                self._emit(
                    "error",
                    "scheduler.result_sink_failed",
                    f"Storing the {task.kind} result failed: {type(error).__name__}.",
                    {"task_id": task.task_id, "error": f"{type(error).__name__}: {error}"},
                )

    def _should_stop(self) -> bool:
        """Whether a running task must end: the state left `commission`, or the loop shuts down."""
        if self._stop_event is not None and self._stop_event.is_set():
            return True
        with self._lock:
            return self._closed or self._machine.state is not State.COMMISSION

    def _fast_stream_config(
        self,
        mode: str | None,
        exposure_us: int | None,
        gain: int | None,
        roi_arcmin: float | None,
    ) -> StreamConfig | None:
        fast = self._config.fast
        use_mode = self._fast_mode if mode is None else mode
        position = self._pointing.polaris_position(self._clock.utc_ns(), use_mode)
        if position is None:
            return None
        roi = roi_centered_on(
            self._profile, use_mode, position, fast.roi_arcmin if roi_arcmin is None else roi_arcmin
        )
        return StreamConfig(
            mode=use_mode,
            exposure_us=fast.exposure_us if exposure_us is None else exposure_us,
            gain=fast.gain if gain is None else gain,
            pixel_format=self._fast_format,
            roi=roi,
            kind=StreamKind.VIDEO,
            high_speed=fast.high_speed,
        )

    def _run_fast_window(self, config: StreamConfig, duration_s: float) -> FastWindowSample:
        """Run the fast analysis on a stream for a while. The windows go to the caller."""
        active = self._reconfigure(config, Purpose.COMMISSION)
        saturation = self._saturation_for(active.config)
        accumulator = FrameStatsAccumulator()
        windows: list[SeeingWindowRecord] = []
        timeout_s = self._timeout_s(active)
        start = self._mono()
        deadline = start + round(duration_s * NS_PER_S)
        run = _FastRun(
            stream_id=active.stream_id,
            started_mono=start,
            window_ns=0,
            read_timeout_s=timeout_s,
            last_recenter_mono=start,
        )
        n_frames = n_dropped = 0
        first_ns = last_ns = 0
        aborted = False
        finished = False
        try:
            self._start_stream()
            self._refresh_context(force=True)
            while True:
                if self._should_stop():
                    aborted = True
                    break
                frame = self._driver.read_frame(timeout_s)
                now = self._mono()
                self._note_frame(frame)
                update = self._fast.push(frame)
                windows.extend(update.windows)
                accumulator.add(frame, saturation, star_found=update.star.found)
                if n_frames == 0:
                    first_ns = frame.t_utc_ns
                last_ns = frame.t_utc_ns
                n_frames += 1
                n_dropped += frame.dropped_before
                self._drain_if_due(run, frame, now)
                if now >= deadline:
                    break
            windows.extend(self._fast.flush("sweep_cell"))
            self._drain_metrics(active.stream_id)
            finished = True
        finally:
            if not finished:  # a camera error ended the cell, so drop its half-built window
                self._counters.discarded_frames += n_frames
                self._fast.flush("sweep_cell_failed")
                self._drain_metrics(active.stream_id)
            self._end_stream("sweep_cell_end")
        interval_ns = (last_ns - first_ns) / max(n_frames - 1, 1) if n_frames > 1 else 0
        return FastWindowSample(
            config=active.config,
            stream_id=active.stream_id,
            duration_s=(last_ns - first_ns + interval_ns) / NS_PER_S,
            n_frames=n_frames,
            n_dropped=n_dropped,
            windows=tuple(windows),
            stats=accumulator.summary(),
            aborted=aborted,
        )

    def _saturation_for(self, config: StreamConfig) -> float:
        """The saturation level, in the counts that frames of this stream carry."""
        if config.pixel_format is PixelFormat.RAW8:
            return 255.0
        return self._profile.saturation(config.mode, config.gain).container_dn

    # --- Setup -----------------------------------------------------------------------------

    def _validate_against_profile(self) -> None:
        """Fail early, with a plain message, when the configuration does not fit the profile."""
        config = self._config
        limits = self._profile.limits
        low_us, high_us = limits.exposure_us_range
        low_gain, high_gain = limits.gain_range
        exposures = {
            "fast.exposure_us": config.fast.exposure_us,
            "survey.short_exposure_s": config.survey.short_exposure_us,
            "survey.long_exposure_s": config.survey.long_exposure_us,
            "watch.exposure_us": config.watch.exposure_us,
            "align.exposure_s": seconds_to_us(config.align.exposure_s),
        }
        for name, value in exposures.items():
            if not low_us <= value <= high_us:
                raise ValueError(
                    f"scheduler.{name} is {value} us, outside the profile's {low_us} to "
                    f"{high_us} us"
                )
        gains = {
            "fast.gain": config.fast.gain,
            "survey.short_gain": config.survey.short_gain,
            "survey.long_gain": config.survey.long_gain,
            "watch.gain": config.watch.gain,
            "align.gain": config.align.gain,
        }
        for name, value in gains.items():
            if not low_gain <= value <= high_gain:
                raise ValueError(
                    f"scheduler.{name} is {value}, outside the profile's {low_gain} to {high_gain}"
                )
        # The lookups raise a `ProfileError` that names the missing readout mode, or the missing
        # high-speed values of the fast mode.
        self._profile.mode(self._fast_mode, high_speed=config.fast.high_speed)
        self._profile.mode(self._survey_mode)
        self._profile.roi_size_px(self._fast_mode, config.fast.roi_arcmin)


class _TaskContext:
    """The `CommissionContext` of one task. It runs on the scheduler's thread."""

    def __init__(self, scheduler: Scheduler, task: CommissionTask) -> None:
        self._scheduler = scheduler
        self._task = task

    @property
    def clock(self) -> Clock:
        return self._scheduler._clock

    @property
    def profile(self) -> Profile:
        return self._scheduler._profile

    @property
    def config(self) -> SchedulerConfig:
        return self._scheduler._config

    def should_stop(self) -> bool:
        return self._scheduler._should_stop()

    def emit_event(
        self, level: str, kind: str, message: str, detail: Mapping[str, Any] | None = None
    ) -> None:
        self._scheduler._emit(level, kind, message, detail)

    def fast_stream_config(
        self,
        *,
        mode: str | None = None,
        exposure_us: int | None = None,
        gain: int | None = None,
        roi_arcmin: float | None = None,
    ) -> StreamConfig | None:
        return self._scheduler._fast_stream_config(mode, exposure_us, gain, roi_arcmin)

    def configure(self, config: StreamConfig) -> ActiveStream:
        return self._scheduler._reconfigure(config, Purpose.COMMISSION)

    def start(self) -> None:
        self._scheduler._start_stream()

    def read_frame(self, timeout_s: float | None = None) -> Frame:
        scheduler = self._scheduler
        if timeout_s is None:
            active = scheduler._active
            if active is None:
                raise CameraStateError("read_frame before configure")
            timeout_s = scheduler._timeout_s(active)
        frame = scheduler._driver.read_frame(timeout_s)
        scheduler._note_frame(frame)
        return frame

    def stop(self) -> None:
        self._scheduler._end_stream("task_stop")

    def run_fast_window(self, config: StreamConfig, duration_s: float) -> FastWindowSample:
        return self._scheduler._run_fast_window(config, duration_s)


def build_scheduler(
    config: Config,
    *,
    driver: CameraDriver,
    fast: FastAnalyzer,
    survey: SurveyAnalyzer,
    pointing: PointingProvider,
    records: RecordWriter,
    metrics: MetricsWriter,
    clock: Clock,
    escalate: Callable[[EscalationLevel], None] | None = None,
    context_provider: Callable[[int], FastContext] | None = None,
    alignment_sink: Callable[[Frame], None] | None = None,
    result_sink: Callable[[CommissionResult], None] | None = None,
) -> Scheduler:
    """Build a scheduler from the layered configuration.

    The function reads the `[scheduler]` table, the `[site]` table (optional), the station ID, and
    the profile from `config`. Pass the collaborators as keywords. The caller starts the loop with
    `Scheduler.run`.
    """
    return Scheduler(
        driver=driver,
        fast=fast,
        survey=survey,
        pointing=pointing,
        records=records,
        metrics=metrics,
        clock=clock,
        profile=config.profile,
        station_id=config.station_id,
        config=config.section("scheduler", SchedulerConfig),
        site=load_site(config),
        escalate=escalate,
        context_provider=context_provider,
        alignment_sink=alignment_sink,
        result_sink=result_sink,
    )


def _starts_at_once(task: CommissionTask) -> bool:
    """Whether the task asks to start at the next step (`QueueDark.immediate`)."""
    return isinstance(task.command, QueueDark) and task.command.immediate


def _failed(task: CommissionTask, started: int, finished: int, summary: str) -> CommissionResult:
    return CommissionResult(
        task_id=task.task_id,
        kind=task.kind,
        status="failed",
        summary=summary,
        started_utc_ns=started,
        finished_utc_ns=finished,
    )


def _roi_dict(roi: Roi | None) -> dict[str, int] | None:
    if roi is None:
        return None
    return {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height}


def _with_flags(record: Record, flags: frozenset[str]) -> Record:
    """Add window-style flags to a survey record that declares a `flags` field with those codes."""
    if not flags:
        return record
    for spec in field_specs(type(record)):
        if spec.name == "flags" and spec.codes is not None:
            current: list[str] = list(record.flags)  # type: ignore[attr-defined]
            added = [flag for flag in sorted(flags) if flag in spec.codes and flag not in current]
            if added:
                return record.model_copy(update={"flags": sorted([*current, *added])})
            return record
    return record
