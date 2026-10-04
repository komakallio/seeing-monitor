"""Plan the response to camera errors: back off, climb the recovery ladder, and degrade.

A camera error ends the current activity. The scheduler asks `FaultTracker.failure` what to do,
waits for the time that the plan names, performs the plan's recovery step, and tries the activity
again. Each failed attempt counts as one failure, and a failed recovery step counts too.

**Cause.** `classify` sorts an error into a `FaultCause`: `timeout` (no frame arrived),
`disconnected` (the driver finds no camera), `link` (the scheduler cannot reach `acquire`), or
`error` (anything else). `reason_text` puts the cause in words, for the status, the events, and the
`health` record. The tracker keeps the best explanation of an episode: a failing recovery step does
not replace the timeout that started it, and a step that works ends the explanation of the
failures before it.

**Backoff.** The wait doubles with each failure (by default 2 s, 4 s, 8 s, and so on) up to a
maximum (a minute). While the ladder has a step from the restart of the capture to the restart of
`acquire` that has not had its attempts, the wait is that backoff, whether or not the status is
`degraded`, so a camera that needs the third step gets it within minutes. Once those steps have had
their attempts, the wait is the slow retry period instead, and a dead camera costs one attempt
every few minutes and nothing more. A reboot or a power cycle always waits the slow period.

**The ladder.** The step for failure `k` is `LADDER[(k - 1) // attempts_per_level]`, limited to
`max_level`. After the top step, the scheduler repeats it. A reboot or a power cycle interrupts the
whole Pi, so the scheduler asks for one at most once per `destructive_interval_s`, and it falls
back to the highest gentler step in between. When no supervisor callback exists, the ladder stops
at the last driver step.

**Degraded.** The status turns `degraded` after `degraded_after` failures in a row, and at once
when the cause is `disconnected`: the driver says that the camera is not connected, and more
failures would say nothing new. It clears after `clear_after_frames` good frames in a row, which
also resets the failure count and the ladder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from seeingmon.drivers.base import (
    CameraDisconnectedError,
    CameraLinkError,
    CameraTimeoutError,
    RecoveryLevel,
)
from seeingmon.scheduler.config import FaultConfig, LadderConfig
from seeingmon.scheduler.levels import (
    DESTRUCTIVE_STEPS,
    LADDER,
    LadderStep,
    parse_step,
)

NS_PER_S = 1_000_000_000
MAX_ERROR_CHARS = 200  # the longest error message that a reason repeats

# The position of the last step that the driver performs itself.
_LAST_DRIVER_INDEX = max(i for i, step in enumerate(LADDER) if isinstance(step, RecoveryLevel))
# The position of the first step that interrupts the whole Pi. Every step before it is quick.
_FIRST_DESTRUCTIVE_INDEX = min(i for i, step in enumerate(LADDER) if step in DESTRUCTIVE_STEPS)


class FaultCause(StrEnum):
    """Why a camera call failed, in the terms that the response depends on."""

    TIMEOUT = "timeout"  # no frame arrived in time
    DISCONNECTED = "disconnected"  # the driver finds no camera
    LINK = "link"  # the scheduler cannot reach the process that owns the camera
    ERROR = "error"  # any other error


# How well a cause explains an episode of failures. A new cause replaces the old one only when it
# explains at least as much, so a failing step never hides the timeout that started the episode.
_STRENGTH = {
    FaultCause.ERROR: 0,
    FaultCause.TIMEOUT: 1,
    FaultCause.LINK: 2,
    FaultCause.DISCONNECTED: 3,
}

_REASONS = {
    FaultCause.TIMEOUT: "no frame arrived; the camera may be disconnected",
    FaultCause.DISCONNECTED: "the camera is not connected",
    FaultCause.LINK: "the camera process (acquire) does not answer",
}


def classify(error: BaseException) -> FaultCause:
    """The cause of a camera error. The nearest class decides: a `CameraLinkError` is `link`."""
    if isinstance(error, CameraLinkError):
        return FaultCause.LINK
    if isinstance(error, CameraDisconnectedError):
        return FaultCause.DISCONNECTED
    if isinstance(error, CameraTimeoutError):
        return FaultCause.TIMEOUT
    return FaultCause.ERROR


def reason_text(cause: FaultCause, error: BaseException) -> str:
    """The cause in words, such as `no frame arrived; the camera may be disconnected`.

    A cause of `error` repeats the message of the error, cut to `MAX_ERROR_CHARS` characters,
    because the message is all that the scheduler knows.
    """
    fixed = _REASONS.get(cause)
    if fixed is not None:
        return fixed
    text = " ".join(str(error).split()).rstrip(".") or type(error).__name__
    if len(text) > MAX_ERROR_CHARS:
        text = text[: MAX_ERROR_CHARS - 3] + "..."
    return f"the camera reported an error: {text}"


@dataclass(frozen=True, slots=True)
class FaultPlan:
    """What to do after a failure: wait `wait_s`, then perform `step`, then try again.

    `degraded_changed` is `True` when this failure turned the status to `degraded`. `cause` is the
    best explanation of the episode so far.
    """

    failures: int
    wait_s: float
    step: LadderStep
    degraded: bool
    degraded_changed: bool
    cause: FaultCause = FaultCause.ERROR


class FaultTracker:
    """Count failures and plan the response. It reads no clock: the caller passes the time."""

    def __init__(self, faults: FaultConfig, ladder: LadderConfig) -> None:
        self._faults = faults
        self._ladder = ladder
        self._max_index = LADDER.index(parse_step(ladder.max_level))
        self._failures = 0
        self._good_frames = 0
        self._degraded = False
        self._cause: FaultCause | None = None
        self._last_destructive_mono_ns: int | None = None

    @property
    def failures(self) -> int:
        """The number of failures in a row."""
        return self._failures

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def good_frames(self) -> int:
        """The number of good frames in a row since the last failure."""
        return self._good_frames

    @property
    def cause(self) -> FaultCause | None:
        """The best explanation of the episode, or `None` without a failure."""
        return self._cause

    def note_destructive(self, now_mono_ns: int) -> None:
        """Record that the scheduler asked for a reboot or a power cycle."""
        self._last_destructive_mono_ns = now_mono_ns

    def step_succeeded(self) -> None:
        """Record that a recovery step worked, which ends the explanation of the failures before.

        The failure count stays, because the camera must still deliver good frames to clear it.
        """
        self._cause = None

    def failure(
        self,
        now_mono_ns: int,
        *,
        can_escalate: bool,
        cause: FaultCause = FaultCause.ERROR,
    ) -> FaultPlan:
        """Count a failure and plan the response.

        `can_escalate` says whether a supervisor callback exists for the steps above the driver's,
        and `cause` says why the call failed (see `classify`).
        """
        self._failures += 1
        self._good_frames = 0
        was_degraded = self._degraded
        if self._cause is None or _STRENGTH[cause] >= _STRENGTH[self._cause]:
            self._cause = cause
        if self._failures >= self._faults.degraded_after or cause is FaultCause.DISCONNECTED:
            self._degraded = True
        index = (self._failures - 1) // self._ladder.attempts_per_level
        return FaultPlan(
            failures=self._failures,
            wait_s=self._wait_s(index, can_escalate=can_escalate),
            step=self._step(index, now_mono_ns, can_escalate=can_escalate),
            degraded=self._degraded,
            degraded_changed=self._degraded and not was_degraded,
            cause=self._cause,
        )

    def success(self) -> bool:
        """Count a good frame. Returns `True` when it clears a fault state."""
        if self._failures == 0 and not self._degraded:
            return False
        self._good_frames += 1
        if self._good_frames < self._faults.clear_after_frames:
            return False
        self._failures = 0
        self._good_frames = 0
        self._degraded = False
        self._cause = None
        return True

    def _climb_top(self, *, can_escalate: bool) -> int:
        """The position of the last quick step that the ladder can reach."""
        top = self._max_index if can_escalate else min(self._max_index, _LAST_DRIVER_INDEX)
        return min(top, _FIRST_DESTRUCTIVE_INDEX - 1)

    def _wait_s(self, index: int, *, can_escalate: bool) -> float:
        if index > self._climb_top(can_escalate=can_escalate):
            return self._faults.slow_retry_s
        exponent = min(self._failures - 1, 64)  # a larger exponent only risks an overflow
        wait_s = self._faults.backoff_initial_s * self._faults.backoff_factor**exponent
        return min(wait_s, self._faults.backoff_max_s)

    def _step(self, index: int, now_mono_ns: int, *, can_escalate: bool) -> LadderStep:
        index = min(index, self._max_index)
        if not can_escalate:
            index = min(index, _LAST_DRIVER_INDEX)
        step = LADDER[index]
        if step in DESTRUCTIVE_STEPS and self._destructive_is_too_soon(now_mono_ns):
            while step in DESTRUCTIVE_STEPS:
                index -= 1
                step = LADDER[index]
        return step

    def _destructive_is_too_soon(self, now_mono_ns: int) -> bool:
        last = self._last_destructive_mono_ns
        if last is None:
            return False
        return (now_mono_ns - last) < self._ladder.destructive_interval_s * NS_PER_S
