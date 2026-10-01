"""Faults that the simulated camera can inject, so that `acquire` can practice its recovery.

`SimFaults` describes the faults. Frame numbers count the frames that the driver delivered
since `open`, starting at 0, and they keep counting across `configure` calls. A fault set up
by frame number is repeatable, and a fault set up by probability uses a seeded generator, so a
run with the same seed fails in the same places.

- **Drops.** The driver loses frames before a delivered frame, advances the time, and reports
  the loss in `dropped_before` and `dropped_frames()`.
- **Timeouts.** The driver waits for the full timeout, then raises `CameraTimeoutError`. The
  next read can succeed.
- **Slow reads.** The driver delivers the frame late. The frame keeps its true exposure time.
- **Disconnect.** The driver raises `CameraDisconnectedError` on every call until a recovery of
  at least `disconnect_clears_at`.
- **Stall.** The driver raises `CameraTimeoutError` on every read until a recovery of at least
  `stall_clears_at`.
- **Silent geometry change.** The driver applies a ROI that differs from the request, and
  `configure` returns the applied one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seeingmon.drivers.base import RecoveryLevel


@dataclass(frozen=True, slots=True)
class GeometryChange:
    """A change that the camera applies silently to the ROI that you request.

    The camera moves the ROI by `dx` and `dy` pixels and resizes it by `d_width` and
    `d_height`, after the normal rounding. `times` is the number of `configure` calls that it
    affects, counted from `open`. `None` means every call.
    """

    dx: int = 0
    dy: int = 0
    d_width: int = 0
    d_height: int = 0
    times: int | None = 1


@dataclass(frozen=True, slots=True)
class SimFaults:
    """The faults of a simulated camera. The default has none."""

    drop_probability: float = 0.0
    drop_burst: int = 1
    scripted_drops: tuple[tuple[int, int], ...] = ()  # (frame number, frames lost before it)
    timeout_probability: float = 0.0
    scripted_timeouts: frozenset[int] = frozenset()
    slow_read_s: float = 0.0
    slow_read_probability: float = 0.0
    scripted_slow_reads: frozenset[int] = frozenset()
    disconnect_at_frame: int | None = None
    disconnect_clears_at: RecoveryLevel = RecoveryLevel.USB_RESET
    stall_at_frame: int | None = None
    stall_clears_at: RecoveryLevel = RecoveryLevel.REOPEN
    geometry_change: GeometryChange | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("drop_probability", "timeout_probability", "slow_read_probability"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.drop_burst < 1 or self.slow_read_s < 0:
            raise ValueError("drop_burst must be at least 1 and slow_read_s must not be negative")
        if any(count < 1 for _, count in self.scripted_drops):
            raise ValueError("a scripted drop must lose at least one frame")


@dataclass(frozen=True, slots=True)
class ReadFault:
    """What the faults do to one read attempt."""

    disconnected: bool = False
    timeout: bool = False
    lost_frames: int = 0
    delay_s: float = 0.0


class FaultRuntime:
    """The state of the faults while a driver runs. One instance serves one driver."""

    def __init__(self, faults: SimFaults) -> None:
        self._faults = faults
        seed = faults.seed
        self._drop_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self._timeout_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        self._slow_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
        self._scripted_drops = dict(faults.scripted_drops)
        self._drops: dict[int, int] = {}
        self._timeouts_done: set[int] = set()
        self._slow_done: set[int] = set()
        self.disconnected = False
        self.stalled = False
        self._disconnect_done = False
        self._stall_done = False
        self.geometry_changes_applied = 0

    @property
    def faults(self) -> SimFaults:
        return self._faults

    def reset_connection(self) -> None:
        """Forget the state of a connection: a reopen starts from a working camera."""
        self.disconnected = False
        self.stalled = False

    def check(self, frame_number: int) -> ReadFault:
        """Decide what happens to a read attempt for the frame with this number."""
        faults = self._faults
        if self.disconnected:
            return ReadFault(disconnected=True)
        if (
            faults.disconnect_at_frame is not None
            and not self._disconnect_done
            and frame_number >= faults.disconnect_at_frame
        ):
            self.disconnected = True
            return ReadFault(disconnected=True)
        if self.stalled:
            return ReadFault(timeout=True)
        if (
            faults.stall_at_frame is not None
            and not self._stall_done
            and frame_number >= faults.stall_at_frame
        ):
            self.stalled = True
            return ReadFault(timeout=True)
        if frame_number in faults.scripted_timeouts and frame_number not in self._timeouts_done:
            self._timeouts_done.add(frame_number)
            return ReadFault(timeout=True)
        if (
            faults.timeout_probability > 0
            and frame_number not in self._timeouts_done
            and self._timeout_rng.random() < faults.timeout_probability
        ):
            self._timeouts_done.add(frame_number)
            return ReadFault(timeout=True)
        return ReadFault(
            lost_frames=self._lost_before(frame_number), delay_s=self._delay(frame_number)
        )

    def _lost_before(self, frame_number: int) -> int:
        """The frames lost before a frame. The answer stays the same on repeated calls."""
        if frame_number not in self._drops:
            lost = self._scripted_drops.get(frame_number, 0)
            if self._faults.drop_probability > 0 and (
                self._drop_rng.random() < self._faults.drop_probability
            ):
                lost += self._faults.drop_burst
            self._drops[frame_number] = lost
        return self._drops[frame_number]

    def _delay(self, frame_number: int) -> float:
        faults = self._faults
        if faults.slow_read_s <= 0 or frame_number in self._slow_done:
            return 0.0
        slow = frame_number in faults.scripted_slow_reads
        if not slow and faults.slow_read_probability > 0:
            slow = bool(self._slow_rng.random() < faults.slow_read_probability)
        if slow:
            self._slow_done.add(frame_number)
            return faults.slow_read_s
        return 0.0

    def recover(self, level: RecoveryLevel) -> None:
        """Apply a recovery step: it clears the faults that it is strong enough to clear."""
        if self.disconnected and level >= self._faults.disconnect_clears_at:
            self.disconnected = False
            self._disconnect_done = True
        if self.stalled and level >= self._faults.stall_clears_at:
            self.stalled = False
            self._stall_done = True

    def next_geometry_change(self) -> GeometryChange | None:
        """The silent change for the next `configure`, or `None` when the camera behaves."""
        change = self._faults.geometry_change
        if change is None:
            return None
        if change.times is not None and self.geometry_changes_applied >= change.times:
            return None
        self.geometry_changes_applied += 1
        return change
