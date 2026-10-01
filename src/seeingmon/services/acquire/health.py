"""The health summary of the `acquire` process.

The watchdog thread builds an `AcquireHealth` each tick. It goes to systemd as the status line,
to the log as a periodic summary, and to `core` through the `health` call, which `core` folds into
its own health record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AcquireHealth:
    """A snapshot of the process. Counters count since the process started.

    `state` is `starting`, `closed` (the driver is not open), `ready` (open, not capturing),
    `streaming`, `stalled` (capturing, but no frame for several periods), or `stopping`.
    """

    state: str
    instance: str
    driver: str
    uptime_s: float
    opened: bool
    capturing: bool
    stream_id: int | None
    frames_captured: int
    frames_sent: int
    frame_rate_hz: float
    dropped_driver: int
    dropped_gap: int
    dropped_queue: int
    queue_frames: int
    queue_bytes: int
    queue_peak_frames: int
    flow_stalls: int
    read_timeouts: int
    read_errors: int
    internal_errors: int
    events_recorded: int
    last_error: str | None
    time_resets: int
    time_outliers: int
    clock_synchronized: bool | None
    clock_error_bound_ns: int | None
    client_connected: bool
    stream_connected: bool
    threads_alive: bool
    priority: str

    def to_json(self) -> dict[str, Any]:
        """The summary as a JSON object."""
        return asdict(self)

    def summary(self) -> str:
        """One line for the log and for `systemctl status`."""
        dropped = self.dropped_driver + self.dropped_gap + self.dropped_queue
        parts = [self.state]
        if self.capturing:
            parts.append(f"{self.frame_rate_hz:.1f} fps")
        parts.append(f"{self.frames_captured} frames")
        parts.append(f"{dropped} dropped")
        if self.last_error:
            parts.append(f"last error: {self.last_error}")
        return ", ".join(parts)
