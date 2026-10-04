"""The rolling seeing value: an estimate from the newest seconds of the fast stream.

A stored window covers 60 s and closes once a minute. A person who watches the live video wants a
number that follows the seeing from second to second. `LiveEstimator` keeps the recent frames of the
stream in a ring (`LiveRing`), and every `live_every_s` seconds of frame time it estimates the
seeing from the newest `live_span_s` seconds, with the same estimator and the same settings as a
stored window (`estimate_seeing`). The result is a `LiveSeeing`. `FastPathAnalyzer.live` holds the
newest one, and another thread may read it, because the object is immutable and the analyzer
assigns it in one step.

**How it differs from a stored window.** A live value is provisional. It uses a shorter span (10 s
against 60 s), so it has fewer independent samples, and it jitters by about 10% more than the value
of a stored window (the validation in `docs/architecture.md` has the numbers). The estimator
corrects the variance that its detrending removes for the length of the span, so the value stays
unbiased. `core` does not store the value, and it computes none of the extras of a window: the
spectrum, the scintillation, the vibration lines, and the star statistics other than the width.
The value exists only while the fast stream runs, which is about three quarters of each cycle.

**The ring.** One row per frame holds the time (in seconds from the start of the stream), the slot
on the uniform grid of frame periods, the flags (usable, saturated), the centroid, the widths, the
modeled noise of the centroid, the peak, the flux, and the frames lost before the frame. The slot
and the loss follow `seeingmon.fastpath.windows`: a lost frame leaves an empty slot, and never a
time shift. A new stream, or a time that does not advance, empties the ring.

**Cost.** `LiveRing.add` writes one row, which takes about 2 microseconds, and runs for every frame
on the scheduler thread. An estimate takes about 2 ms on a development machine, once in 2 s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from seeingmon.fastpath.config import FastPathConfig
from seeingmon.fastpath.estimator import EstimatorSettings, MotionSeries, estimate_seeing

FloatArray = npt.NDArray[np.float64]

GAP_PERIODS = 1.5  # a frame interval above this many periods means that frames were lost
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
MAX_RATE_HZ = 250.0  # the ring holds the span at up to this many frames a second
NS_PER_S = 1_000_000_000

# The columns of a row of the ring.
T, SLOT, USABLE, SATURATED, X, Y, WX, WY, NVX, NVY, PEAK, FLUX, LOST = range(13)
COLUMNS = 13

LIVE_FIELDS = (
    "seeing_fwhm_arcsec",
    "seeing_fwhm_structure_arcsec",
    "r0_cm",
    "r0_structure_cm",
    "image_motion_rms_x_arcsec",
    "image_motion_rms_y_arcsec",
    "width_fwhm_arcsec",
)


@dataclass(frozen=True, slots=True)
class LiveSeeing:
    """The seeing of the newest seconds of the fast stream. A value is `None` when it is unknown.

    `t_utc_ns` is the end of the span, in the time of the frames, and `span_s` is the length of the
    span that the estimate used. `n_frames` counts the frames that arrived in the span, `n_usable`
    those with a usable centroid, and `valid_fraction` is `n_usable` over the frames that the
    camera produced (the lost frames included). The values follow the stored `seeing` records:
    `seeing_fwhm_arcsec` and `r0_cm` come from the variance of the motion, and the `_structure_`
    values from its structure function. `flags` are the window flags that apply: `degraded` (more
    than 5% of the frames were lost), `saturated`, and the flags of the context, such as `cloud`.
    `quality` says why a value is `None`. Treat the object as read-only.
    """

    t_utc_ns: int
    span_s: float
    n_frames: int
    n_usable: int
    valid_fraction: float
    seeing_fwhm_arcsec: float | None
    seeing_fwhm_structure_arcsec: float | None
    r0_cm: float | None
    r0_structure_cm: float | None
    image_motion_rms_x_arcsec: float | None
    image_motion_rms_y_arcsec: float | None
    width_fwhm_arcsec: float | None
    stream_id: int
    readout_mode: str
    exposure_us: int
    flags: tuple[str, ...] = ()
    quality: dict[str, str] = field(default_factory=dict)


class LiveRing:
    """The newest frames of one stream, as rows of numbers. See the module text.

    The rows sit in an array of twice the capacity. When the array fills, the newest half moves to
    the front, which costs one copy of half the array for each `capacity` frames. So the newest
    rows always form one contiguous block, and an estimate reads them without a copy.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 16:
            raise ValueError("the ring holds at least 16 frames")
        self._capacity = capacity
        self._rows: FloatArray = np.zeros((2 * capacity, COLUMNS), dtype=np.float64)
        self._n = 0
        self._base_ns = 0
        self._last_ns = 0
        self._last_slot = 0
        self._period_ns: float | None = None

    def __len__(self) -> int:
        return self._n

    @property
    def period_s(self) -> float | None:
        """The running estimate of the frame period, in seconds."""
        return None if self._period_ns is None else self._period_ns / NS_PER_S

    @property
    def base_ns(self) -> int:
        """The time of the first frame of the stream, which the `T` column counts from."""
        return self._base_ns

    def reset(self, period_s: float | None = None) -> None:
        """Empty the ring, and start with an expected frame period, if the driver knows it."""
        self._n = 0
        self._period_ns = None if not period_s or period_s <= 0.0 else period_s * NS_PER_S

    def add(
        self,
        t_ns: int,
        usable: bool,
        saturated: bool,
        x: float,
        y: float,
        width_x: float,
        width_y: float,
        noise_var_x: float,
        noise_var_y: float,
        peak: float,
        flux: float,
        dropped_before: int,
    ) -> None:
        """Add the frame at `t_ns`. A time that does not advance empties the ring first."""
        n = self._n
        if n and t_ns <= self._last_ns:
            n = self._n = 0  # a time discontinuity starts a new run, as it closes a window
            self._period_ns = None
        if n == 0:
            self._base_ns = t_ns
            slot = 0
            lost = 0
        else:
            interval = t_ns - self._last_ns
            lost = self._frames_lost(interval, dropped_before)
            slot = self._last_slot + 1 + lost
        if n == 2 * self._capacity:
            half = self._capacity
            self._rows[:half] = self._rows[half:]
            n = self._n = half
        self._rows[n] = (
            (t_ns - self._base_ns) / NS_PER_S,
            slot,
            usable,
            saturated,
            x,
            y,
            width_x,
            width_y,
            noise_var_x,
            noise_var_y,
            peak,
            flux,
            lost,
        )
        self._n = n + 1
        self._last_ns = t_ns
        self._last_slot = slot

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

    def recent(self, span_s: float) -> FloatArray:
        """The rows of the newest `span_s` seconds, as a view. Empty when the ring is empty."""
        n = self._n
        if n == 0:
            return self._rows[:0]
        times = self._rows[:n, T]
        newest = times[n - 1]
        first = min(int(np.searchsorted(times, newest - span_s, side="right")), n - 1)
        return self._rows[first:n]


@dataclass(frozen=True, slots=True)
class LiveStream:
    """What the estimate needs to know about the stream: the same numbers as a stored window."""

    stream_id: int
    mode: str
    exposure_us: int
    settings: EstimatorSettings
    plate_scale_arcsec_per_px: float
    known: bool


class LiveEstimator:
    """Estimate the seeing of the newest seconds of the stream. See the module text."""

    def __init__(self, config: FastPathConfig) -> None:
        self._config = config
        self._every_ns = round(config.live_every_s * NS_PER_S)
        self._span_s = config.live_span_s
        self._min_span_ns = round(config.live_min_span_s * NS_PER_S)
        self.ring = LiveRing(max(64, math.ceil(config.live_span_s * MAX_RATE_HZ)))
        # The estimates follow the frame time. The first one is due when the ring holds the
        # minimum span, and each next one `live_every_s` after the end of the span of the last.
        self._eligible_ns = 0
        self._next_ns: int | None = None

    def reset(self, period_s: float | None = None) -> None:
        """Forget the frames and the schedule. Call it when the stream changes."""
        self.ring.reset(period_s)
        self._eligible_ns = 0
        self._next_ns = None

    def add(
        self,
        t_ns: int,
        usable: bool,
        saturated: bool,
        x: float,
        y: float,
        width_x: float,
        width_y: float,
        noise_var_x: float,
        noise_var_y: float,
        peak: float,
        flux: float,
        dropped_before: int,
    ) -> bool:
        """Add a frame, and return whether an estimate is due now. Runs for every frame."""
        ring = self.ring
        ring.add(
            t_ns, usable, saturated, x, y, width_x, width_y, noise_var_x, noise_var_y, peak,
            flux, dropped_before,
        )  # fmt: skip
        if len(ring) == 1:  # the first frame of a run, which the stream or a time step began
            self._eligible_ns = t_ns + self._min_span_ns
            self._next_ns = None
        return t_ns >= self._eligible_ns and (self._next_ns is None or t_ns >= self._next_ns)

    def estimate(
        self,
        stream: LiveStream,
        context_flags: frozenset[str],
        zenith_angle_deg: float | None,
    ) -> LiveSeeing | None:
        """Estimate the seeing of the newest span, and schedule the next estimate.

        Returns `None` when the ring holds fewer than two frames. Call it when `add` says that an
        estimate is due.
        """
        ring = self.ring
        rows = ring.recent(self._span_s)
        if len(rows) < 2:
            return None
        end_ns = ring.base_ns + round(float(rows[-1, T]) * NS_PER_S)
        self._next_ns = end_ns + self._every_ns
        return self._compute(rows, end_ns, stream, context_flags, zenith_angle_deg)

    def _compute(
        self,
        rows: FloatArray,
        end_ns: int,
        stream: LiveStream,
        context_flags: frozenset[str],
        zenith_angle_deg: float | None,
    ) -> LiveSeeing:
        config = self._config
        n_frames = len(rows)
        usable = rows[:, USABLE] > 0.5
        n_usable = int(usable.sum())
        n_dropped = int(rows[1:, LOST].sum())  # the frames lost before the first row are outside
        expected = n_frames + n_dropped
        valid_fraction = min(n_usable / expected, 1.0) if expected else 0.0
        slots = rows[:, SLOT].astype(np.int64)
        slots -= slots[0]
        period_s = _period_s(rows) or (self.ring.period_s or 0.0)
        span_s = float(rows[-1, T] - rows[0, T]) + period_s
        flags = set(context_flags)
        if n_dropped > 0.05 * expected:
            flags.add("degraded")
        saturated = rows[:, SATURATED] > 0.5
        if n_frames and saturated.sum() / n_frames > config.saturated_window_fraction:
            flags.add("saturated")
        quality: dict[str, str] = {}
        values: dict[str, float | None] = dict.fromkeys(LIVE_FIELDS)
        reason = _reason_not_estimated(stream, period_s, n_usable, span_s, valid_fraction, config)
        if reason is None:
            grid = int(slots[-1]) + 1
            series = MotionSeries(
                period_s=period_s,
                x_px=_on_slots(grid, slots, rows[:, X], usable),
                y_px=_on_slots(grid, slots, rows[:, Y], usable),
                noise_var_x_px2=_on_slots(grid, slots, rows[:, NVX], usable),
                noise_var_y_px2=_on_slots(grid, slots, rows[:, NVY], usable),
            )
            settings = stream.settings
            if zenith_angle_deg is not None and config.zenith_correction:
                settings = replace(settings, zenith_angle_deg=zenith_angle_deg)
            estimate = estimate_seeing(series, settings)
            quality.update(estimate.quality)
            values["seeing_fwhm_arcsec"] = _finite(estimate.fwhm_arcsec)
            values["seeing_fwhm_structure_arcsec"] = _finite(estimate.fwhm_structure_arcsec)
            values["r0_cm"] = _scaled(estimate.r0_m, 100.0)
            values["r0_structure_cm"] = _scaled(estimate.r0_structure_m, 100.0)
            values["image_motion_rms_x_arcsec"] = _finite(estimate.rms_x_arcsec)
            values["image_motion_rms_y_arcsec"] = _finite(estimate.rms_y_arcsec)
        if usable.any() and stream.plate_scale_arcsec_per_px == stream.plate_scale_arcsec_per_px:
            sigma = 0.5 * (rows[usable, WX] + rows[usable, WY])
            values["width_fwhm_arcsec"] = _finite(
                float(np.mean(sigma)) * FWHM_PER_SIGMA * stream.plate_scale_arcsec_per_px
            )
        for name in LIVE_FIELDS:
            if values[name] is None and name not in quality:
                quality[name] = reason or _default_reason(name, n_usable)
        return LiveSeeing(
            t_utc_ns=end_ns,
            span_s=round(span_s, 3),
            n_frames=n_frames,
            n_usable=n_usable,
            valid_fraction=round(valid_fraction, 4),
            seeing_fwhm_arcsec=values["seeing_fwhm_arcsec"],
            seeing_fwhm_structure_arcsec=values["seeing_fwhm_structure_arcsec"],
            r0_cm=values["r0_cm"],
            r0_structure_cm=values["r0_structure_cm"],
            image_motion_rms_x_arcsec=values["image_motion_rms_x_arcsec"],
            image_motion_rms_y_arcsec=values["image_motion_rms_y_arcsec"],
            width_fwhm_arcsec=values["width_fwhm_arcsec"],
            stream_id=stream.stream_id,
            readout_mode=stream.mode,
            exposure_us=stream.exposure_us,
            flags=tuple(sorted(flags)),
            quality=quality,
        )


def _period_s(rows: FloatArray) -> float:
    """The frame period of the span: the median of the intervals between frames with no loss."""
    if len(rows) < 2:
        return 0.0
    steps = np.diff(rows[:, T])
    clean = steps[rows[1:, LOST] == 0]
    chosen = clean if len(clean) else steps
    return float(np.median(chosen))


def _on_slots(
    length: int, slots: npt.NDArray[np.int64], values: FloatArray, usable: npt.NDArray[np.bool_]
) -> FloatArray:
    """Place the values of the usable frames on the grid of slots, with `NaN` in the other slots."""
    grid = np.full(length, np.nan)
    grid[slots[usable]] = values[usable]
    return grid


def _reason_not_estimated(
    stream: LiveStream,
    period_s: float,
    n_usable: int,
    span_s: float,
    valid_fraction: float,
    config: FastPathConfig,
) -> str | None:
    """Why the span cannot give a seeing, with the words of the stored windows, or `None`."""
    if not stream.known:
        return "the readout mode is not in the profile"
    if period_s <= 0.0 or n_usable < config.min_samples:
        return "too few usable frames"
    if span_s < config.live_min_span_s:
        return "the span is shorter than the minimum"
    if valid_fraction < config.min_valid_fraction:
        return "too few usable frames"
    return None


def _default_reason(name: str, n_usable: int) -> str:
    if name == "width_fwhm_arcsec":
        return "no usable frame, or the plate scale of the readout mode is unknown"
    return "the estimator gave no value"


def _finite(value: float | None) -> float | None:
    """The value as a float, or `None` when it is missing or not finite."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _scaled(value: float | None, factor: float) -> float | None:
    return None if value is None else _finite(value * factor)
