"""The focus measure of the alignment helper: a short history, the best value, and the spikes.

A quick solve gives one number for its frame: the median FWHM of the usable stars
(`QuickSolution.focus_fwhm_px`). One number says little while the person turns the focuser, so the
helper keeps the last `HISTORY_LENGTH` values with the capture times of their frames. The page draws
them as a rolling curve, and a page that reloads finds the curve at once. The helper owns the
history, so it outlives a viewer and a restart of `web`.

**Spikes.** Touching the telescope inflates the star width for a moment. A value that exceeds
`SPIKE_FACTOR` times the median of the preceding `SPIKE_WINDOW` values has the flag `spike`, and the
page leaves it out of its scale. The preceding values include earlier spikes, so a lasting change is
a spike only until it fills half of the window: after five values at a new level, the median
follows and the values count again. A value with fewer than `SPIKE_MIN_PREVIOUS` preceding values is
never a spike.

**The best value.** The best value is the smallest value that is not a spike since the session
began, or since `reset_best`, which the person calls after a refocus. A spike never sets it. The
history stays when the best value restarts, so the curve before and after the refocus shows
together.

The module needs no NumPy and no survey code.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass

HISTORY_LENGTH = 120
SPIKE_FACTOR = 2.0
SPIKE_WINDOW = 10
SPIKE_MIN_PREVIOUS = 3


@dataclass(frozen=True, slots=True)
class FocusPoint:
    """One value of the focus measure.

    `index` counts the values of the session from 1, and it never repeats, so a reader that holds
    the points up to one index asks for the points after it. `seq` and `t_utc_ns` name the frame
    that the value was measured in, and `spike` is the flag of the module text.
    """

    index: int
    seq: int
    t_utc_ns: int
    fwhm_px: float
    n_stars: int
    spike: bool


@dataclass(frozen=True, slots=True)
class FocusSnapshot:
    """The history at one moment: `session` changes when the history restarts, `points` run from
    the oldest to the newest, and `best_px` is the best value (`None` before the first one)."""

    session: int
    points: tuple[FocusPoint, ...]
    best_px: float | None

    def point_for(self, seq: int) -> FocusPoint | None:
        """The newest point that was measured in the frame `seq`, or `None`."""
        for point in reversed(self.points):
            if point.seq == seq:
                return point
        return None


class FocusHistory:
    """The rolling history of the focus measure. Not thread-safe: the caller holds a lock."""

    def __init__(self, length: int = HISTORY_LENGTH) -> None:
        if length < 1:
            raise ValueError("the history holds at least one value")
        self._points: deque[FocusPoint] = deque(maxlen=length)
        self._best: float | None = None
        self._next_index = 1
        self._session = 0

    def add(self, seq: int, t_utc_ns: int, fwhm_px: float, n_stars: int) -> FocusPoint | None:
        """Add the value of a frame, and return its point. A value that is not a positive finite
        number is ignored, and the call returns `None`."""
        if not math.isfinite(fwhm_px) or fwhm_px <= 0.0:
            return None
        recent = [point.fwhm_px for point in list(self._points)[-SPIKE_WINDOW:]]
        spike = len(recent) >= SPIKE_MIN_PREVIOUS and fwhm_px > SPIKE_FACTOR * statistics.median(
            recent
        )
        point = FocusPoint(self._next_index, seq, t_utc_ns, fwhm_px, n_stars, spike)
        self._next_index += 1
        self._points.append(point)
        if not spike and (self._best is None or fwhm_px < self._best):
            self._best = fwhm_px
        return point

    @property
    def best_px(self) -> float | None:
        """The best value since the session began or since `reset_best`, or `None`."""
        return self._best

    def reset_best(self) -> None:
        """Forget the best value, so that the next value that is not a spike becomes the best."""
        self._best = None

    def clear(self) -> None:
        """Start a new session: no values, no best value, and a new `session` number."""
        self._points.clear()
        self._best = None
        self._next_index = 1
        self._session += 1

    def snapshot(self) -> FocusSnapshot:
        return FocusSnapshot(self._session, tuple(self._points), self._best)
