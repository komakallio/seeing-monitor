"""Count the frames that the system loses, and say where each came from.

`Frame.dropped_before` tells analysis how many frames were lost just before a frame. Three
places lose frames, and the architecture sums them onto the next frame:

1. **The driver.** The camera and the SDK count the frames that they dropped. The driver reports
   the change of that counter in `dropped_before` of the frame that follows (the replay driver
   turns a gap in a recording into a count in the same way).
2. **A gap.** The arrival times show a hole that no counter explains: the camera never produced
   the frames, or the host stalled. An interval longer than `gap_factor` frame periods (1.5 by
   default) means that `round(interval / period) - 1` frames are missing.
3. **The queue.** A full queue drops its oldest frame (see `seeingmon.services.acquire.queue`).

A frame that a counter and a gap both report is one lost frame, so the accountant takes the
larger of the two and does not add them. It adds the third source, which the other two cannot
see. A frame that analysis cannot use (a short transfer, a saturated frame) is delivered with
flags. It is not a drop.

The gap rule reads intervals on the monotonic clock, so a step of the wall clock never counts as
lost frames. Its estimate can be too high after a stall, when the camera delivers the frames it
buffered in a burst. The driver's own counter is exact when the SDK has one.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median

from seeingmon.clock import NS_PER_S

_LEARN_FRAMES = 16  # intervals that the accountant keeps to learn the period
_MIN_LEARNED = 3  # intervals needed before a learned period counts


@dataclass(frozen=True, slots=True)
class DropCounters:
    """Frames lost since the accountant started, by source."""

    driver: int = 0
    gap: int = 0
    overflow: int = 0

    @property
    def total(self) -> int:
        """The frames lost from all three sources."""
        return self.driver + self.gap + self.overflow


class DropAccountant:
    """Decide how many frames were lost before each frame, and keep the totals."""

    def __init__(self, *, gap_factor: float = 1.5) -> None:
        if gap_factor <= 1.0:
            raise ValueError("gap_factor must exceed 1")
        self._gap_factor = gap_factor
        self._period_hint_ns: float | None = None
        self._driver = 0
        self._gap = 0
        self._overflow = 0
        self._last_ns: int | None = None
        self._learned: deque[int] = deque(maxlen=_LEARN_FRAMES)

    def reset(self, period_s: float | None = None) -> None:
        """Start a new stream. `period_s` is the frame period that the driver expects, if known.

        The totals keep counting, and no gap spans the change.
        """
        self._period_hint_ns = None if period_s is None else period_s * NS_PER_S
        self._last_ns = None
        self._learned.clear()

    @property
    def counters(self) -> DropCounters:
        """The totals by source."""
        return DropCounters(self._driver, self._gap, self._overflow)

    def _period_ns(self) -> float | None:
        if self._period_hint_ns is not None:
            return self._period_hint_ns
        if len(self._learned) >= _MIN_LEARNED:
            return float(median(self._learned))
        return None

    def frame_arrived(
        self, at_ns: int, driver_dropped: int, *, period_s: float | None = None
    ) -> int:
        """Return the frames lost before this one, from the driver and from a gap.

        `at_ns` is the arrival of the frame on the monotonic clock, and `driver_dropped` is the
        `dropped_before` that the driver set. `period_s` overrides the frame period when the
        caller knows a better one, such as the period that the time fit measures.
        """
        previous, self._last_ns = self._last_ns, at_ns
        gap_lost = 0
        if previous is not None:
            interval_ns = at_ns - previous
            period_ns = period_s * NS_PER_S if period_s else self._period_ns()
            if period_ns and interval_ns > self._gap_factor * period_ns:
                gap_lost = max(1, round(interval_ns / period_ns) - 1)
            elif interval_ns > 0 and self._period_hint_ns is None:
                self._learned.append(interval_ns)
        self._driver += driver_dropped
        self._gap += max(0, gap_lost - driver_dropped)
        return max(driver_dropped, gap_lost)

    def record_overflow(self, count: int) -> None:
        """Note frames that a full queue dropped. The queue adds them to the next frame."""
        self._overflow += count
