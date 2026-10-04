"""The health summary of the `acquire` process.

The watchdog thread builds an `AcquireHealth` each tick. It goes to systemd as the status line,
to the log as a periodic summary, and to `core` through the `health` call, which `core` folds into
its own health record.

The summary line tells where the lost frames came from, and how many of the frames of the last
minute were lost or late, so that a log shows at a glance whether the capture thread keeps time:

    streaming, 82.1 fps, 9413 frames, 12 dropped (driver 12, gap 0, queue 0), 0.1% lost and 0.4%
    late in the last minute, priority: highest thread priority, timer: 1 ms resolution
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

MINUTE_S = 60.0  # a span of this length or a little less reads as "the last minute"


@dataclass(frozen=True, slots=True)
class AcquireHealth:
    """A snapshot of the process. Counters count since the process started.

    `state` is `starting`, `closed` (the driver is not open), `ready` (open, not capturing),
    `streaming`, `stalled` (capturing, but no frame for several periods), or `stopping`.

    A late read is a read that came more than the gap factor times the frame period after the one
    before, and that a catch-up read cleared: the host was late, and no frame was lost.
    `priority` says what the capture thread got, and `timer` what the request for a finer system
    timer got (Windows), or it is empty. The `recent_` fields cover the last `recent_s` seconds,
    at most a minute: the frames captured, the frames lost (from all three sources), and the late
    reads.
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
    late_reads: int = 0
    timer: str = ""
    recent_s: float = 0.0
    recent_frames: int = 0
    recent_lost: int = 0
    recent_late: int = 0

    @property
    def recent_lost_percent(self) -> float | None:
        """The share of the frames of the last minute that were lost, or `None` without frames."""
        expected = self.recent_frames + self.recent_lost
        return 100.0 * self.recent_lost / expected if expected else None

    @property
    def recent_late_percent(self) -> float | None:
        """The share of the frames of the last minute that came late, or `None` without frames."""
        return 100.0 * self.recent_late / self.recent_frames if self.recent_frames else None

    def to_json(self) -> dict[str, Any]:
        """The summary as a JSON object, with the two shares of the last minute."""
        data = asdict(self)
        data["recent_lost_percent"] = self.recent_lost_percent
        data["recent_late_percent"] = self.recent_late_percent
        return data

    def summary(self) -> str:
        """One line for the log and for `systemctl status`."""
        dropped = self.dropped_driver + self.dropped_gap + self.dropped_queue
        parts = [self.state]
        if self.capturing:
            parts.append(f"{self.frame_rate_hz:.1f} fps")
        parts.append(f"{self.frames_captured} frames")
        parts.append(
            f"{dropped} dropped (driver {self.dropped_driver}, gap {self.dropped_gap}, "
            f"queue {self.dropped_queue})"
        )
        lost, late = self.recent_lost_percent, self.recent_late_percent
        if lost is not None and late is not None:
            span = "minute" if self.recent_s >= MINUTE_S - 0.5 else f"{self.recent_s:.0f} s"
            parts.append(f"{lost:.1f}% lost and {late:.1f}% late in the last {span}")
        parts.append(f"priority: {self.priority}")
        if self.timer:
            parts.append(f"timer: {self.timer}")
        if self.last_error:
            parts.append(f"last error: {self.last_error}")
        return ", ".join(parts)
