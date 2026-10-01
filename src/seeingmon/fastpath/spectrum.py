"""The Welch spectrum of the image motion, the reduced bins for the record, and vibration lines.

The input is a motion series on a uniform grid of frame slots, in arcseconds, with `NaN` in the
slots that hold no usable frame. The functions never read the time, so a replay and a simulation
give the same result.

**Segments and gaps.** A short gap (up to `max_interp_gap` slots) is filled by linear
interpolation, and a long gap breaks the series. A segment must not contain a long gap, and no more
than `max_interp_fraction` of its samples may be interpolated. The estimator averages the
periodograms of the accepted segments. They overlap by `overlap` (0.5 by default), use a Hann
window, and have the mean and the linear trend removed.

**Units.** The spectrum is one-sided, in arcsec^2/Hz, and it integrates to the variance of the
series (the Hann window's equivalent noise bandwidth is divided out). `dof` is the number of
degrees of freedom of a spectrum bin, which accounts for the overlap. A bin of the spectrum is
distributed as `chi-squared(dof) / dof` times the true spectrum, so it has the relative scatter
`sqrt(2 / dof)`.

**Aliasing.** The sampling rate sets the highest frequency, `nyquist_hz` (45 Hz at 90 frames per
second). Turbulence has power above it: the corner of the tilt spectrum lies at 0.2 to 0.5 `v / D`,
which is 40 to 100 Hz for a 50 mm aperture and 10 m/s. That power folds back into the spectrum, so
the high-frequency end sits above the true spectrum, and a vibration line above the Nyquist
frequency appears at the folded frequency `|f - k fs|`. The variance of the series stays correct,
because folding keeps the total.

**Lines.** A line is a bin whose power exceeds `threshold` times the median of its neighbors
(`local_bins` on each side), above `min_line_hz`, and a local maximum. A line spreads over a few
bins in a Hann spectrum, so the code merges adjacent bins and refines the frequency of the peak by
a parabola through the logarithm of the three highest bins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class MotionSpectrum:
    """The spectrum of the motion along both axes.

    `freq_hz`, `psd_x`, and `psd_y` hold the full-resolution spectrum without the zero
    frequency. `bin_freq_hz`, `bin_psd_x`, and `bin_psd_y` hold the logarithmically spaced
    reduction for the record. `lines_hz` lists the vibration lines of both axes.
    """

    freq_hz: FloatArray
    psd_x: FloatArray
    psd_y: FloatArray
    bin_freq_hz: FloatArray
    bin_psd_x: FloatArray
    bin_psd_y: FloatArray
    segments: int
    dof: float
    nyquist_hz: float
    lines_hz: tuple[float, ...]


def fill_short_gaps(series: FloatArray, max_gap: int) -> tuple[FloatArray, BoolArray]:
    """Fill gaps of at most `max_gap` slots by linear interpolation.

    Returns the filled series and a mask of the interpolated slots. Longer gaps stay `NaN`.
    """
    valid = np.isfinite(series)
    count = len(series)
    if valid.all() or not valid.any():
        return series.copy(), np.zeros(count, dtype=np.bool_)
    index = np.arange(count)
    previous = np.maximum.accumulate(np.where(valid, index, -1))
    following = np.minimum.accumulate(np.where(valid, index, count)[::-1])[::-1]
    gap = following - previous - 1
    fillable = ~valid & (previous >= 0) & (following < count) & (gap <= max_gap)
    filled = series.copy()
    if fillable.any():
        filled[fillable] = np.interp(index[fillable], index[valid], series[valid])
    return filled, fillable


def welch_dof(segments: int, window: FloatArray, hop: int) -> float:
    """The degrees of freedom of a spectrum averaged over overlapping, windowed segments.

    The formula is `2 K / (1 + 2 sum_j (1 - j / K) rho_j)`, where `rho_j` is the squared
    correlation of the window with itself shifted by `j` hops (Welch 1967). For Hann windows with
    50% overlap it gives `36 K^2 / (19 K - 1)`.
    """
    if segments <= 1:
        return 2.0 * max(segments, 0)
    norm = float(np.dot(window, window))
    total = 0.0
    for j in range(1, segments):
        shift = j * hop
        if shift >= len(window):
            break
        correlation = float(np.dot(window[:-shift], window[shift:])) / norm
        total += (1.0 - j / segments) * correlation * correlation
    return 2.0 * segments / (1.0 + 2.0 * total)


def _linear_detrend(segments: FloatArray) -> FloatArray:
    """Remove the mean and the linear trend of every row."""
    length = segments.shape[1]
    t = np.arange(length) - 0.5 * (length - 1)
    mean = segments.mean(axis=1, keepdims=True)
    slope = ((segments - mean) @ t) / float(t @ t)
    return np.asarray(segments - mean - slope[:, None] * t, dtype=np.float64)


def _periodograms(
    series: FloatArray,
    starts: npt.NDArray[np.intp],
    length: int,
    window: FloatArray,
    period_s: float,
) -> FloatArray:
    """The mean one-sided spectrum of the segments that begin at `starts`."""
    windows = np.lib.stride_tricks.sliding_window_view(series, length)
    segments = _linear_detrend(windows[starts])
    spectrum = np.fft.rfft(segments * window, axis=1)
    power = spectrum.real**2 + spectrum.imag**2
    psd = power * (period_s / float(np.dot(window, window)))
    if length % 2 == 0:
        psd[:, 1:-1] *= 2.0  # the one-sided spectrum folds every bin except zero and Nyquist
    else:
        psd[:, 1:] *= 2.0
    return np.asarray(psd.mean(axis=0), dtype=np.float64)


def _segment_starts(
    series: FloatArray, interpolated: BoolArray, length: int, hop: int, max_fraction: float
) -> npt.NDArray[np.intp]:
    """The start of every segment that holds no long gap and few interpolated samples."""
    count = len(series)
    if count < length:
        return np.empty(0, dtype=np.intp)
    starts = np.arange(0, count - length + 1, hop)
    missing = np.concatenate([[0], np.cumsum(~np.isfinite(series))])
    filled = np.concatenate([[0], np.cumsum(interpolated)])
    clean = (missing[starts + length] - missing[starts]) == 0
    fraction = (filled[starts + length] - filled[starts]) / length
    return np.asarray(starts[clean & (fraction <= max_fraction)], dtype=np.intp)


def _local_median(psd: FloatArray, half_width: int) -> FloatArray:
    """The median of each bin's neighborhood, with the ends reflected."""
    padded = np.pad(psd, half_width, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_width + 1)
    return np.asarray(np.median(windows, axis=1), dtype=np.float64)


def find_lines(
    freq_hz: FloatArray,
    psd: FloatArray,
    *,
    threshold: float,
    local_bins: int,
    min_line_hz: float,
) -> list[tuple[float, float]]:
    """Find the lines of one spectrum. Returns `(frequency, power over local median)` pairs."""
    if len(psd) < 2 * local_bins + 3:
        return []
    ratio = psd / np.maximum(_local_median(psd, local_bins), np.finfo(np.float64).tiny)
    peak = np.zeros(len(psd), dtype=np.bool_)
    peak[1:-1] = (psd[1:-1] >= psd[:-2]) & (psd[1:-1] >= psd[2:])
    candidates = np.flatnonzero((ratio > threshold) & peak & (freq_hz >= min_line_hz))
    step = float(freq_hz[1] - freq_hz[0])
    lines: list[tuple[float, float]] = []
    for index in candidates:
        if 0 < index < len(psd) - 1 and min(psd[index - 1], psd[index], psd[index + 1]) > 0.0:
            left, middle, right = np.log(psd[index - 1 : index + 2])
            curvature = left - 2.0 * middle + right
            offset = 0.0 if curvature == 0.0 else 0.5 * (left - right) / curvature
            offset = max(-0.5, min(0.5, float(offset)))
        else:
            offset = 0.0
        lines.append((float(freq_hz[index]) + offset * step, float(ratio[index])))
    return lines


def merge_lines(lines: list[tuple[float, float]], tolerance_hz: float) -> list[float]:
    """Merge the lines of two axes. Lines closer than `tolerance_hz` count as one, and the
    stronger one gives the frequency."""
    merged: list[tuple[float, float]] = []
    for frequency, strength in sorted(lines):
        if merged and frequency - merged[-1][0] <= tolerance_hz:
            if strength > merged[-1][1]:
                merged[-1] = (frequency, strength)
        else:
            merged.append((frequency, strength))
    return [frequency for frequency, _ in merged]


def log_bins(
    freq_hz: FloatArray, psd_x: FloatArray, psd_y: FloatArray, bins: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Reduce a spectrum to at most `bins` bins that are equally spaced in the logarithm.

    Each bin holds the mean spectrum of the frequencies in it and the mean of those frequencies.
    A bin that holds no frequency is dropped, so the low end has fewer bins than asked for.
    """
    if len(freq_hz) == 0 or bins < 1:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    edges = np.asarray(
        np.geomspace(freq_hz[0] * 0.999, freq_hz[-1] * 1.001, bins + 1), dtype=np.float64
    )
    which = np.clip(np.searchsorted(edges, freq_hz, side="right") - 1, 0, bins - 1)
    counts = np.bincount(which, minlength=bins)
    used = counts > 0
    divisor = counts[used].astype(np.float64)
    return (
        np.bincount(which, weights=freq_hz, minlength=bins)[used] / divisor,
        np.bincount(which, weights=psd_x, minlength=bins)[used] / divisor,
        np.bincount(which, weights=psd_y, minlength=bins)[used] / divisor,
    )


def compute_spectrum(
    x_arcsec: FloatArray,
    y_arcsec: FloatArray,
    period_s: float,
    *,
    segment_s: float = 2.0,
    overlap: float = 0.5,
    max_interp_gap: int = 3,
    max_interp_fraction: float = 0.1,
    bins: int = 24,
    threshold: float = 5.0,
    local_bins: int = 15,
    min_line_hz: float = 1.0,
    min_segments: int = 3,
) -> MotionSpectrum | None:
    """Compute the spectrum of two motion series, or return `None` when no segment fits.

    `x_arcsec` and `y_arcsec` lie on the same grid of frame slots, `period_s` apart, with `NaN`
    for a slot without a usable frame. The series should have their trend removed.
    """
    if period_s <= 0.0 or len(x_arcsec) != len(y_arcsec):
        raise ValueError("period_s must be positive and the two series must have equal length")
    length = max(16, round(segment_s / period_s))
    hop = max(1, round(length * (1.0 - overlap)))
    window = np.asarray(np.hanning(length + 2)[1:-1], dtype=np.float64)  # no zero end points
    filled_x, interp_x = fill_short_gaps(x_arcsec, max_interp_gap)
    filled_y, interp_y = fill_short_gaps(y_arcsec, max_interp_gap)
    joint = np.where(np.isfinite(filled_x) & np.isfinite(filled_y), 0.0, np.nan)
    starts = _segment_starts(joint, interp_x | interp_y, length, hop, max_interp_fraction)
    if len(starts) < min_segments:
        return None
    psd_x = _periodograms(np.nan_to_num(filled_x), starts, length, window, period_s)
    psd_y = _periodograms(np.nan_to_num(filled_y), starts, length, window, period_s)
    freq = np.asarray(np.fft.rfftfreq(length, d=period_s), dtype=np.float64)
    psd_x, psd_y, freq = psd_x[1:], psd_y[1:], freq[1:]
    lines = find_lines(
        freq, psd_x, threshold=threshold, local_bins=local_bins, min_line_hz=min_line_hz
    ) + find_lines(freq, psd_y, threshold=threshold, local_bins=local_bins, min_line_hz=min_line_hz)
    resolution = float(freq[1] - freq[0]) if len(freq) > 1 else 0.0
    bin_freq, bin_x, bin_y = log_bins(freq, psd_x, psd_y, bins)
    return MotionSpectrum(
        freq_hz=freq,
        psd_x=psd_x,
        psd_y=psd_y,
        bin_freq_hz=bin_freq,
        bin_psd_x=bin_x,
        bin_psd_y=bin_y,
        segments=len(starts),
        dof=welch_dof(len(starts), window, hop),
        nyquist_hz=0.5 / period_s,
        lines_hz=tuple(merge_lines(lines, 1.5 * resolution)),
    )


def aliasing_expected(nyquist_hz: float, aperture_m: float, wind_ms: float) -> bool:
    """Whether the turbulence corner (at `0.5 v / D`, its upper estimate) lies above Nyquist."""
    if aperture_m <= 0.0 or wind_ms <= 0.0:
        return False
    return nyquist_hz < 0.5 * wind_ms / aperture_m and not math.isinf(nyquist_hz)
