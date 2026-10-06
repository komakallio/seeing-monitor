"""Group frames into windows of frame time, and count what a window lost.

A window is a stretch of one stream that starts at the time of its first frame and lasts
`window_s` of frame time (`Frame.t_utc_ns`), never the wall clock. The window closes when a frame
arrives at or after the end of the interval, and that frame starts the next window. A window
never spans two streams, so `begin_stream`, a frame of another stream, or a time that does not
advance closes it early. `flush` closes it early on request.

**Frame slots.** The frames of a stream arrive on a regular clock, and each lost frame leaves an
empty slot. The assembler gives every frame a slot number: the number of the previous frame plus
one plus the frames lost in between. The estimator works on this uniform grid, so a drop is a gap
and never a time shift.

**Drops.** The frames lost before a frame are `max(dropped_before, gap)`, where `gap` is the
number of missing frame periods when the interval to the previous frame exceeds 1.5 periods. The
`dropped_before` field already includes the SDK counter and the queue overflow, and the gap rule
catches a producer that counts nothing. A lost frame belongs to the window that contains its time.
When a frame opens a new window, the frames lost just before it count for the window that is
closing (up to its end), and the rest fall between windows and count for none. The first frame of
a stream or of a window after a time discontinuity has no predecessor, so it loses nothing.

**Counts.** `n_frames` counts the frames that arrived. `n_usable` counts the frames that have a
usable centroid (a star was found, not near the ROI edge, and no isolated bright pixel). The
`valid_fraction` of a window is `n_usable / (n_frames + n_dropped)`. A frame that analysis cannot
use is not a drop.

The assembler keeps lists of per-frame values and turns them into arrays when the window closes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

GAP_PERIODS = 1.5  # a frame interval above this many periods means that frames were lost
_KEPT_COLUMNS = frozenset({4, 6})  # peak and background: defined for a frame without a star
_NS_PER_S = 1_000_000_000


@dataclass(frozen=True, slots=True)
class ClosedWindow:
    """The frames of one finished window, as arrays with one entry per arrived frame.

    `slot` is the slot number of each frame (the first is 0), and `period_s` is the nominal frame
    period. `usable` marks the frames with a usable centroid, and the position, width, flux,
    noise, and SNR arrays hold `NaN` for the others. The peak, the background, and the variance of
    one pixel of sky (`pixel_var_e2`, in electrons squared) hold a value for every frame.
    `closed_early` says that `begin_stream`, `flush`, a new stream, or a time discontinuity ended
    the window before its full length, and `reason` names the cause.
    """

    stream_id: int
    mode: str
    exposure_us: int
    gain: int
    t_start_ns: int
    duration_s: float
    period_s: float
    n_frames: int
    n_dropped: int
    n_usable: int
    n_saturated: int
    closed_early: bool
    reason: str
    time_invalid: bool
    temperature_c: float | None
    slot: IntArray
    usable: npt.NDArray[np.bool_]
    x: FloatArray
    y: FloatArray
    width_x: FloatArray
    width_y: FloatArray
    peak_dn: FloatArray
    flux_e: FloatArray
    bg_dn: FloatArray
    noise_var_x: FloatArray
    noise_var_y: FloatArray
    saturated: npt.NDArray[np.bool_]
    snr: FloatArray
    pixel_var_e2: FloatArray

    @property
    def n_slots(self) -> int:
        """The number of frame slots from the first frame to the last."""
        return int(self.slot[-1]) + 1 if self.n_frames else 0

    def on_slots(self, values: FloatArray) -> FloatArray:
        """Place per-frame values on the grid of slots, with `NaN` in the empty slots."""
        grid = np.full(self.n_slots, np.nan)
        grid[self.slot] = values
        return grid


@dataclass(slots=True)
class _Open:
    """The window that is filling."""

    stream_id: int
    mode: str
    exposure_us: int
    gain: int
    t_start_ns: int
    t_last_ns: int
    last_slot: int = 0
    n_dropped: int = 0
    n_usable: int = 0
    n_saturated: int = 0
    time_invalid: bool = False
    temperature_sum: float = 0.0
    temperature_count: int = 0
    t: list[int] = field(default_factory=list)
    lost: list[int] = field(default_factory=list)
    slot: list[int] = field(default_factory=list)
    columns: tuple[list[float], ...] = field(
        default_factory=lambda: tuple([] for _ in range(9))
    )  # x, y, width_x, width_y, peak, flux_e, bg, noise_x, noise_y
    usable: list[bool] = field(default_factory=list)
    saturated: list[bool] = field(default_factory=list)
    snr: list[float] = field(default_factory=list)
    pixel_var: list[float] = field(default_factory=list)


class WindowAssembler:
    """Collect the per-frame results of a stream into windows. See the module documentation."""

    def __init__(self, window_s: float = 60.0) -> None:
        if not window_s > 0.0:
            raise ValueError("window_s must be positive")
        self._window_ns = round(window_s * _NS_PER_S)
        self._window_s = window_s
        self._open: _Open | None = None
        self._stream_id: int | None = None
        self._period_ns: float | None = None  # a running estimate of the frame period
        self._last_t_ns: int | None = None

    @property
    def window_s(self) -> float:
        return self._window_s

    @property
    def stream_id(self) -> int | None:
        return self._stream_id

    @property
    def has_open_window(self) -> bool:
        return self._open is not None

    @property
    def period_s(self) -> float | None:
        """The running estimate of the frame period, in seconds."""
        return None if self._period_ns is None else self._period_ns / _NS_PER_S

    def begin_stream(
        self, stream_id: int, period_s: float | None = None, reason: str = "new stream"
    ) -> list[ClosedWindow]:
        """Start a stream, and return the window that the old stream left open.

        `period_s` is the expected frame period, if the driver knows it.
        """
        closed = self.flush(reason)
        self._stream_id = stream_id
        self._last_t_ns = None
        self._period_ns = None if not period_s or period_s <= 0.0 else period_s * _NS_PER_S
        return closed

    def flush(self, reason: str = "end") -> list[ClosedWindow]:
        """Close the open window early, and return it. Returns nothing when none is open."""
        window, self._open = self._open, None
        self._last_t_ns = None
        if window is None:
            return []
        return [self._close(window, early=True, reason=reason)]

    def add(
        self,
        stream_id: int,
        t_ns: int,
        dropped_before: int,
        mode: str,
        exposure_us: int,
        gain: int,
        temperature_c: float | None,
        time_invalid: bool,
        usable: bool,
        saturated: bool,
        values: tuple[float, ...],
        snr: float = math.nan,
        pixel_var_e2: float = math.nan,
    ) -> list[ClosedWindow]:
        """Add one frame, and return the windows that it closed (none or one).

        `values` holds `x, y, width_x, width_y, peak_dn, flux_e, bg_dn, noise_var_x, noise_var_y`
        of the frame, `snr` the signal-to-noise ratio of its star, and `pixel_var_e2` the variance
        of one pixel of sky. The kernel gives `NaN` for the values that a missing star does not
        have.
        """
        closed: list[ClosedWindow] = []
        window = self._open
        if stream_id != self._stream_id:
            if window is not None:
                closed.append(self._close(window, early=True, reason="new stream"))
                window = self._open = None
            self._stream_id = stream_id
            self._last_t_ns = None
            self._period_ns = None
        lost = 0
        previous = self._last_t_ns
        if previous is not None:
            interval = t_ns - previous
            if interval <= 0:
                if window is not None:
                    closed.append(self._close(window, early=True, reason="time discontinuity"))
                    window = self._open = None
                previous = None
            else:
                lost = self._frames_lost(interval, dropped_before)
        self._last_t_ns = t_ns
        if window is not None and t_ns - window.t_start_ns >= self._window_ns:
            window.n_dropped += _lost_inside(
                window.t_last_ns, t_ns, lost, window.t_start_ns + self._window_ns
            )
            closed.append(self._close(window, early=False, reason="full"))
            window = self._open = None
            lost = 0
        if window is None:
            window = self._open = _Open(stream_id, mode, exposure_us, gain, t_ns, t_ns)
            lost = 0
        else:
            window.n_dropped += lost
        window.last_slot += 0 if not window.t else 1 + lost
        window.t_last_ns = t_ns
        window.t.append(t_ns)
        window.lost.append(lost)
        window.slot.append(window.last_slot)
        for column, value in zip(window.columns, values, strict=True):
            column.append(value)
        window.usable.append(usable)
        window.saturated.append(saturated)
        window.snr.append(snr)
        window.pixel_var.append(pixel_var_e2)
        window.n_usable += usable
        window.n_saturated += saturated
        if time_invalid:
            window.time_invalid = True
        if temperature_c is not None:
            window.temperature_sum += temperature_c
            window.temperature_count += 1
        return closed

    def _frames_lost(self, interval_ns: int, dropped_before: int) -> int:
        """Update the period estimate and return the frames lost before this frame."""
        period = self._period_ns
        gap = 0
        if period is None:
            self._period_ns = float(interval_ns) if dropped_before == 0 else None
        elif interval_ns > GAP_PERIODS * period:
            gap = max(0, round(interval_ns / period) - 1)
        elif dropped_before == 0:
            self._period_ns = period + (interval_ns - period) / 16.0
        return max(dropped_before, gap)

    def _close(self, window: _Open, *, early: bool, reason: str) -> ClosedWindow:
        n = len(window.t)
        t = np.asarray(window.t, dtype=np.int64)
        lost = np.asarray(window.lost, dtype=np.int64)
        steps = np.diff(t)
        clean = steps[lost[1:] == 0]
        period_ns = float(np.median(clean)) if len(clean) else (self._period_ns or 0.0)
        if period_ns <= 0.0 and len(steps):
            period_ns = float(np.median(steps))
        period_s = period_ns / _NS_PER_S if period_ns > 0.0 else 0.0
        duration_s = max((window.t_last_ns - window.t_start_ns) / _NS_PER_S + period_s, 1e-6)
        columns = [np.asarray(c, dtype=np.float64) for c in window.columns]
        usable = np.asarray(window.usable, dtype=np.bool_)
        for index, column in enumerate(columns):
            if index not in _KEPT_COLUMNS:  # the peak and the background stay for every frame
                column[~usable] = np.nan
        snr = np.asarray(window.snr, dtype=np.float64)
        snr[~usable] = np.nan
        self._open = None if self._open is window else self._open
        return ClosedWindow(
            stream_id=window.stream_id,
            mode=window.mode,
            exposure_us=window.exposure_us,
            gain=window.gain,
            t_start_ns=window.t_start_ns,
            duration_s=duration_s,
            period_s=period_s,
            n_frames=n,
            n_dropped=window.n_dropped,
            n_usable=int(window.n_usable),
            n_saturated=int(window.n_saturated),
            closed_early=early,
            reason=reason,
            time_invalid=window.time_invalid,
            temperature_c=(
                window.temperature_sum / window.temperature_count
                if window.temperature_count
                else None
            ),
            slot=np.asarray(window.slot, dtype=np.int64),
            usable=usable,
            x=columns[0],
            y=columns[1],
            width_x=columns[2],
            width_y=columns[3],
            peak_dn=columns[4],
            flux_e=columns[5],
            bg_dn=columns[6],
            noise_var_x=columns[7],
            noise_var_y=columns[8],
            saturated=np.asarray(window.saturated, dtype=np.bool_),
            snr=snr,
            pixel_var_e2=np.asarray(window.pixel_var, dtype=np.float64),
        )


def _lost_inside(t_previous_ns: int, t_ns: int, lost: int, t_end_ns: int) -> int:
    """How many of `lost` frames, spread evenly between two frames, fall before `t_end_ns`."""
    if lost <= 0:
        return 0
    step = (t_ns - t_previous_ns) / (lost + 1)
    if step <= 0.0:
        return 0
    return max(0, min(lost, math.ceil((t_end_ns - t_previous_ns) / step) - 1))


def is_partial(window: ClosedWindow, window_s: float) -> bool:
    """Whether a window that closed early is shorter than the window length.

    A window that reaches its full length (within half a frame period) is complete, whatever
    closed it.
    """
    return window.closed_early and window.duration_s < window_s - 0.5 * window.period_s
