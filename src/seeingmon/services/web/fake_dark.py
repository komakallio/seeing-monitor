"""A dark library and a scripted dark task, for `FakeCoreClient` and the demo.

The real `core` records a dark set in its `commission` state: bias frames, a wait until short test
frames are dark (which means that a person covered the camera), dark frames at the survey exposure,
and then the master dark and the library. `DarkSimulator` plays the same story on a clock, without a
camera, so that the tests of the API and the demo of the UI can show every state.

The task follows the clock of the fake core. It is `queued` for `DarkScript.queued_s`, and then it
`runs` through the phases `bias`, `cover` (only with `wait_for_cover`), `dark`, and `build`, each
one as long as the script says. The camera counts as covered during the last third of the cover
phase. A task without `wait_for_cover` fails at the first dark frame, because nobody covered the
camera. A finished task adds a set to the library at the sensor temperature. With `pause_after`,
the fake scheduler ends in `paused`, and `Resume` works as in the real one. A `Pause` ends a queued
or running task as `aborted`.

The simulator checks a command like `core` does: 3 to 60 frames, a positive exposure, a label of at
most 80 characters, and one task at a time (`RejectReason.BUSY`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from seeingmon.clock import NS_PER_S, Clock, utc_ns_to_iso
from seeingmon.scheduler.commands import QueueDark, RejectReason
from seeingmon.services.web.contract import (
    DarkLibraryView,
    DarkModelView,
    DarkSetView,
    DarkStatusView,
    DarkTaskView,
)

MIN_FRAMES = 3
MAX_FRAMES = 60
MAX_LABEL_CHARS = 80
MAX_EXPOSURE_S = 600.0
MAX_COVER_WAIT_S = 7200.0
DEFAULT_FRAMES = 9
DEFAULT_BIAS_FRAMES = 9
TOLERANCE_C = 3.0
MAX_AGE_DAYS = 183.0
DAY_NS = 86_400 * NS_PER_S
COVER_FRACTION = 2 / 3  # the part of the cover phase before the camera counts as covered
UNCOVERED_LEVEL_DN = 2412.5
COVERED_LEVEL_DN = 11.8
NOT_DARK_REASON = "the median is 2400 counts above the expected level"
# A task runs only in `safe` or `auto`. In these states of the scheduler, it waits.
HOLD_MESSAGES = {
    "paused": "The scheduler is paused. The dark session starts after you resume it.",
    "align": "The alignment helper runs. The dark session starts after it ends.",
}
WAIT_MESSAGE = "Waiting for the next step of the scheduler."


@dataclass(frozen=True, slots=True)
class DarkScript:
    """How long each part of a scripted task lasts, in seconds of the clock."""

    queued_s: float = 0.0
    bias_s: float = 2.0
    cover_s: float = 3.0
    dark_s: float = 4.0
    build_s: float = 1.0


@dataclass(slots=True)
class _Run:
    task_id: int
    exposure_s: float
    frames: int
    bias_frames: int
    wait_for_cover: bool
    pause_after: bool
    cover_timeout_s: float | None
    submitted_ns: int
    submitted_utc_ns: int
    started: bool = False
    outcome: str | None = None  # `ok`, `failed`, or `aborted`
    summary: str = ""
    set_name: str | None = None
    finished_utc_ns: int | None = None


@dataclass(frozen=True, slots=True)
class Transition:
    """A change of the scheduler that the task causes: the new state and its reason."""

    state: str
    reason: str


class DarkSimulator:
    """The dark library of a fake `core`, and the scripted task that adds a set to it."""

    def __init__(
        self,
        clock: Clock,
        *,
        script: DarkScript | None = None,
        mode: str = "bin2",
        gain: int = 120,
        exposure_s: float = 30.0,
        sensor_temperature_c: float | None = 12.3,
        sets: list[DarkSetView] | None = None,
        model: DarkModelView | None = None,
    ) -> None:
        self._clock = clock
        self.script = script or DarkScript()
        self.mode = mode
        self.gain = gain
        self.exposure_s = exposure_s
        self.sensor_temperature_c = sensor_temperature_c
        self.sets: list[DarkSetView] = list(sets or [])  # the newest first
        self.model = model
        self.status_override: DarkStatusView | None = None
        self._run: _Run | None = None
        self._held_by: str | None = None  # the state of the scheduler that holds a queued task

    # --- The library -----------------------------------------------------------------------

    def status(self) -> DarkStatusView:
        """Whether the library needs a set: it does when none lies within the tolerance."""
        if self.status_override is not None:
            return self.status_override
        if not self.sets:
            return self._status(True, "the library holds no dark set")
        newest_age = min(item.age_days for item in self.sets)
        temperature = self.sensor_temperature_c
        if temperature is None:
            return self._status(
                False,
                "the camera reports no temperature, so the coverage is unknown",
                newest_age_days=newest_age,
            )
        nearest = min(self.sets, key=lambda item: abs(item.temperature_c - temperature))
        gap = round(abs(nearest.temperature_c - temperature), 1)
        if gap > TOLERANCE_C:
            reason = (
                f"no recent set within {TOLERANCE_C:.1f} C of {temperature:.1f} C "
                f"(the nearest is {gap:.1f} C away)"
            )
            due = True
        elif newest_age > MAX_AGE_DAYS:
            reason = (
                f"the newest set is {newest_age:.0f} days old (the limit is {MAX_AGE_DAYS:.0f})"
            )
            due = True
        else:
            reason = (
                f"a set lies within {TOLERANCE_C:.1f} C of {temperature:.1f} C, "
                f"and the newest is {newest_age:.0f} days old"
            )
            due = False
        return self._status(
            due, reason, gap_c=gap, nearest_name=nearest.name, newest_age_days=newest_age
        )

    @staticmethod
    def _status(due: bool, reason: str, **known: Any) -> DarkStatusView:
        return DarkStatusView(
            due=due, reason=reason, tolerance_c=TOLERANCE_C, max_age_days=MAX_AGE_DAYS, **known
        )

    def library(self) -> DarkLibraryView:
        return DarkLibraryView(
            mode=self.mode,
            gain=self.gain,
            exposure_s=self.exposure_s,
            sensor_temperature_c=self.sensor_temperature_c,
            status=self.status(),
            model=self.model,
            sets=list(self.sets),
            task=self.task(),
        )

    # --- The task --------------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether a task is queued or running."""
        return self._run is not None and self._run.outcome is None

    @property
    def queued(self) -> bool:
        """Whether a task waits for its turn, which counts as a queued task of the scheduler."""
        return self.active and self._run is not None and not self._run.started

    def submit(
        self, command: QueueDark, task_id: int, state: str = "auto"
    ) -> tuple[bool, RejectReason | None, str]:
        """Check a command, and queue the task. Returns whether it was accepted, and why not."""
        if self.active:
            return False, RejectReason.BUSY, "a dark session is already queued or running"
        for name, value in (("frames", command.frames), ("bias_frames", command.bias_frames)):
            if value is not None and not MIN_FRAMES <= value <= MAX_FRAMES:
                return (
                    False,
                    RejectReason.INVALID,
                    f"{name} must be between {MIN_FRAMES} and {MAX_FRAMES}",
                )
        exposure = command.exposure_s
        if exposure is not None and not 0 < exposure <= MAX_EXPOSURE_S:
            return False, RejectReason.INVALID, "exposure_s must be positive and at most 600"
        timeout = command.wait_for_cover_timeout_s
        if timeout is not None and not 0 < timeout <= MAX_COVER_WAIT_S:
            return (
                False,
                RejectReason.INVALID,
                "wait_for_cover_timeout_s must be positive and at most 7200",
            )
        if len(command.label) > MAX_LABEL_CHARS:
            return (
                False,
                RejectReason.INVALID,
                f"the label has more than {MAX_LABEL_CHARS} characters",
            )
        self._run = _Run(
            task_id=task_id,
            exposure_s=self.exposure_s if exposure is None else exposure,
            frames=DEFAULT_FRAMES if command.frames is None else command.frames,
            bias_frames=DEFAULT_BIAS_FRAMES if command.bias_frames is None else command.bias_frames,
            wait_for_cover=command.wait_for_cover,
            pause_after=command.pause_after,
            cover_timeout_s=timeout,
            submitted_ns=self._clock.monotonic_ns(),
            submitted_utc_ns=self._clock.utc_ns(),
        )
        return (
            True,
            None,
            HOLD_MESSAGES.get(state, "the dark session is queued and starts at the next step"),
        )

    def abort(self) -> bool:
        """End a queued or running task as `aborted`. Returns whether there was one."""
        run = self._run
        if run is None or run.outcome is not None:
            return False
        run.outcome = "aborted"
        run.summary = "The dark session was aborted, because the scheduler paused."
        run.finished_utc_ns = self._clock.utc_ns()
        return True

    def settle(self, *, state: str = "auto") -> list[Transition]:
        """Move the task to where the clock puts it. Returns the changes of the scheduler.

        `state` is the state of the scheduler. A paused scheduler and a running alignment hold a
        task that waits for its turn, and its queue time starts again when they let go.
        """
        run = self._run
        if run is None or run.outcome is not None:
            return []
        self._held_by = state if state in HOLD_MESSAGES else None
        if self._held_by is not None and not run.started:
            run.submitted_ns = self._clock.monotonic_ns()
            return []
        changes: list[Transition] = []
        seconds = self._seconds_running(run)
        if seconds >= 0 and not run.started:
            run.started = True
            changes.append(Transition("commission", "a dark session runs"))
        outcome = self._outcome(run, seconds)
        if outcome is not None:
            self._finish(run, outcome)
            if run.pause_after:
                changes.append(
                    Transition(
                        "paused", "the dark session ended, and the camera may still be covered"
                    )
                )
            else:
                changes.append(Transition("safe", "the dark session ended"))
        return changes

    def task(self) -> DarkTaskView:
        """The progress of the latest task, as `dark_library` reports it."""
        run = self._run
        if run is None:
            return DarkTaskView()
        if run.outcome is not None:
            finished = run.finished_utc_ns
            return self._view(
                run,
                state=run.outcome,
                summary=run.summary,
                set_name=run.set_name,
                finished_utc=None if finished is None else utc_ns_to_iso(finished, digits=0),
            )
        if not run.started:
            message = HOLD_MESSAGES.get(self._held_by or "", WAIT_MESSAGE)
            return self._view(run, state="queued", message=message)
        return self._running(run, self._seconds_running(run))

    # --- Inside ----------------------------------------------------------------------------

    def _seconds_running(self, run: _Run) -> float:
        elapsed = (self._clock.monotonic_ns() - run.submitted_ns) / NS_PER_S
        return elapsed - self.script.queued_s

    def _bounds(self, run: _Run) -> tuple[float, float, float, float]:
        """The ends of the phases `bias`, `cover`, `dark`, and `build`, in seconds of running."""
        script = self.script
        bias = script.bias_s
        cover = bias + (script.cover_s if run.wait_for_cover else 0.0)
        dark = cover + script.dark_s
        return bias, cover, dark, dark + script.build_s

    def _outcome(self, run: _Run, seconds: float) -> str | None:
        bias, _, _, end = self._bounds(run)
        if not run.wait_for_cover and seconds >= bias:
            return "failed"  # the first dark frame is not dark: nobody covered the camera
        timeout = run.cover_timeout_s
        if (
            timeout is not None
            and seconds >= bias + timeout
            and timeout < self.script.cover_s * COVER_FRACTION
        ):
            return "failed"  # the camera is still uncovered when the wait runs out
        return "ok" if seconds >= end else None

    @staticmethod
    def _view(run: _Run, **fields: Any) -> DarkTaskView:
        """A view of the task with what every state shares: the numbers that the command set."""
        return DarkTaskView(
            task_id=run.task_id,
            exposure_s=run.exposure_s,
            frames=run.frames,
            bias_frames=run.bias_frames,
            wait_for_cover=run.wait_for_cover,
            pause_after=run.pause_after,
            started_utc=utc_ns_to_iso(run.submitted_utc_ns, digits=0),
            **fields,
        )

    def _running(self, run: _Run, seconds: float) -> DarkTaskView:
        bias, cover, dark, _ = self._bounds(run)
        if seconds < bias:
            step = _step(seconds, bias, run.bias_frames)
            return self._view(
                run,
                state="running",
                phase="bias",
                step=step,
                steps=run.bias_frames,
                message=f"Bias frame {step} of {run.bias_frames}.",
            )
        if seconds < cover:
            if (seconds - bias) / self.script.cover_s < COVER_FRACTION:
                return self._view(
                    run,
                    state="running",
                    phase="cover",
                    message="Cover the camera now. Waiting for a dark frame.",
                    covered=False,
                    level_dn=UNCOVERED_LEVEL_DN,
                    reason=NOT_DARK_REASON,
                )
            return self._view(
                run,
                state="running",
                phase="cover",
                message="The frame is dark. The camera is covered.",
                covered=True,
                level_dn=COVERED_LEVEL_DN,
            )
        if seconds < dark:
            step = _step(seconds - cover, self.script.dark_s, run.frames)
            return self._view(
                run,
                state="running",
                phase="dark",
                step=step,
                steps=run.frames,
                message=f"Dark frame {step} of {run.frames}.",
                covered=True,
                level_dn=COVERED_LEVEL_DN,
            )
        return self._view(
            run,
            state="running",
            phase="build",
            message="Building the master dark and the library.",
            covered=True,
            level_dn=COVERED_LEVEL_DN,
        )

    def _finish(self, run: _Run, outcome: str) -> None:
        run.outcome = outcome
        run.finished_utc_ns = self._clock.utc_ns()
        if outcome == "failed":
            if run.wait_for_cover:
                run.summary = (
                    f"The camera was not covered within {run.cover_timeout_s:g} seconds "
                    f"({NOT_DARK_REASON}). Cover the camera, and start again."
                )
                return
            run.summary = (
                "The first dark frame was not dark, so the camera is not covered "
                f"({NOT_DARK_REASON}). Cover the camera, and start again."
            )
            return
        added = self._add_set(run)
        run.set_name = added.name
        run.summary = (
            f"Added the set {added.name} at {added.temperature_c:.1f} C: "
            f"{added.n_frames} dark frames, {added.rate_e_per_s:.3f} e/s, "
            f"{added.hot_pixels} hot pixels."
        )

    def _add_set(self, run: _Run) -> DarkSetView:
        temperature = 12.3 if self.sensor_temperature_c is None else self.sensor_temperature_c
        now_ns = self._clock.utc_ns()
        stamp = utc_ns_to_iso(now_ns, digits=0).replace("-", "").replace(":", "")
        model = self.model
        rate = 0.05
        if model is not None:
            rate = model.rate_ref_e_per_s * 2 ** (
                (temperature - model.reference_c) / model.doubling_c
            )
        name = f"dark-{stamp}-{self.mode}-g{self.gain}.fits"
        taken = {item.name for item in self.sets}
        number = 1
        while name in taken:  # a clock that stands still gives the same stamp twice
            number += 1
            name = f"dark-{stamp}-{self.mode}-g{self.gain}-{number}.fits"
        added = DarkSetView(
            name=name,
            t_utc=utc_ns_to_iso(now_ns, digits=0),
            age_days=0.0,
            temperature_c=round(temperature, 2),
            temperature_spread_c=0.3,
            exposure_s=run.exposure_s,
            n_frames=run.frames,
            n_bias_frames=run.bias_frames,
            rate_e_per_s=round(rate, 4),
            hot_pixels=187,
        )
        self.sets.insert(0, added)
        if model is not None:
            self.model = model.model_copy(update={"n_sets": len(self.sets)})
        return added


def _step(seconds: float, length_s: float, steps: int) -> int:
    """The frame (counted from 1) that a phase of `length_s` is at, `seconds` after its start."""
    if length_s <= 0:
        return steps
    return min(steps, int(seconds / length_s * steps) + 1)


__all__ = ["DarkScript", "DarkSimulator", "Transition"]
