"""Plan the response to camera errors: back off, climb the recovery ladder, and degrade.

A camera error ends the current activity. The scheduler asks `FaultTracker.failure` what to do,
waits for the time that the plan names, performs the plan's recovery step, and tries the activity
again. Each failed attempt counts as one failure, and a failed recovery step counts too.

**Backoff.** The wait doubles with each failure (by default 2 s, 4 s, 8 s, and so on) up to a
maximum. Once the status is `degraded`, the wait is the slow retry period instead, so a dead
camera costs one attempt every few minutes and nothing more.

**The ladder.** The step for failure `k` is `LADDER[(k - 1) // attempts_per_level]`, limited to
`max_level`. After the top step, the scheduler repeats it. A reboot or a power cycle interrupts the
whole Pi, so the scheduler asks for one at most once per `destructive_interval_s`, and it falls
back to the highest gentler step in between. When no supervisor callback exists, the ladder stops
at the last driver step.

**Degraded.** The status turns `degraded` after `degraded_after` failures in a row. It clears
after `clear_after_frames` good frames in a row, which also resets the failure count and the
ladder.
"""

from __future__ import annotations

from dataclasses import dataclass

from seeingmon.drivers.base import RecoveryLevel
from seeingmon.scheduler.config import FaultConfig, LadderConfig
from seeingmon.scheduler.levels import (
    DESTRUCTIVE_STEPS,
    LADDER,
    LadderStep,
    parse_step,
)

NS_PER_S = 1_000_000_000

# The position of the last step that the driver performs itself.
_LAST_DRIVER_INDEX = max(i for i, step in enumerate(LADDER) if isinstance(step, RecoveryLevel))


@dataclass(frozen=True, slots=True)
class FaultPlan:
    """What to do after a failure: wait `wait_s`, then perform `step`, then try again.

    `degraded_changed` is `True` when this failure turned the status to `degraded`.
    """

    failures: int
    wait_s: float
    step: LadderStep
    degraded: bool
    degraded_changed: bool


class FaultTracker:
    """Count failures and plan the response. It reads no clock: the caller passes the time."""

    def __init__(self, faults: FaultConfig, ladder: LadderConfig) -> None:
        self._faults = faults
        self._ladder = ladder
        self._max_index = LADDER.index(parse_step(ladder.max_level))
        self._failures = 0
        self._good_frames = 0
        self._degraded = False
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

    def note_destructive(self, now_mono_ns: int) -> None:
        """Record that the scheduler asked for a reboot or a power cycle."""
        self._last_destructive_mono_ns = now_mono_ns

    def failure(self, now_mono_ns: int, *, can_escalate: bool) -> FaultPlan:
        """Count a failure and plan the response.

        `can_escalate` says whether a supervisor callback exists for the steps above the driver's.
        """
        self._failures += 1
        self._good_frames = 0
        was_degraded = self._degraded
        if self._failures >= self._faults.degraded_after:
            self._degraded = True
        return FaultPlan(
            failures=self._failures,
            wait_s=self._wait_s(),
            step=self._step(now_mono_ns, can_escalate=can_escalate),
            degraded=self._degraded,
            degraded_changed=self._degraded and not was_degraded,
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
        return True

    def _wait_s(self) -> float:
        if self._degraded:
            return self._faults.slow_retry_s
        exponent = min(self._failures - 1, 64)  # a larger exponent only risks an overflow
        wait_s = self._faults.backoff_initial_s * self._faults.backoff_factor**exponent
        return min(wait_s, self._faults.backoff_max_s)

    def _step(self, now_mono_ns: int, *, can_escalate: bool) -> LadderStep:
        index = min((self._failures - 1) // self._ladder.attempts_per_level, self._max_index)
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
