"""The scintillation index of a window: the normalized flux variance minus the Poisson floor.

The index is `sigma_I^2 / <I>^2` for the flux `I` of the star. Three things stand between the raw
variance of the measured flux and that quantity.

**Trend.** Slow changes of the flux (the transparency, thin cloud, the aperture losing a little
light as the star moves) are not scintillation. The estimator divides each flux by a centered
moving average over `trend_s` (1 s by default), so it normalizes by the local mean and removes
every change that is slower than about 1 Hz. The turbulence of the atmosphere puts most of the
scintillation power at frequencies of tens of hertz, so the high-pass loses well under 1% of it.
A moving average that includes the frame itself removes `1 / n` of the variance of the frame, and
the estimator multiplies by `n / (n - 1)` to restore it.

**Floor.** The flux is a count of electrons with Poisson noise and pixel noise, and it subtracts
a background level that is noisy too, so the measured variance has a floor of
`(F + A n^2 + A^2 b^2) / F^2` for the mean flux `F` in electrons, the aperture area `A` in pixels,
the variance `n^2` of one pixel in electrons squared, and the variance `b^2` of the background
level. `n^2` holds the read noise, and in a bright sky the sky's photons, which the fast path
measures on the ROI border. The level is the median of a few hundred border pixels, so `b^2` is
about `(pi / 2) n^2` over their number, but the rounding of whole counts can make it larger: the
estimator measures it from the series of levels, detrended in the same way as the flux. In the
simulator's daylight, `A^2 b^2` is about 70% of `A n^2`. The estimator subtracts the floor and
clips the result at zero.

**Saturation.** A saturated star clips its flux, which lowers the variance. The caller excludes
saturated frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

MIN_FRAMES = 20


@dataclass(frozen=True, slots=True)
class Scintillation:
    """The scintillation index of a window, and how it came about.

    `raw_variance` is the normalized variance before the floor. `floor` is the Poisson floor.
    `index` is `max(raw_variance - floor, 0)`. `frames` is the number of frames that counted.
    """

    index: float
    raw_variance: float
    floor: float
    mean_flux_e: float
    frames: int


def scintillation_index(
    flux_e: FloatArray,
    period_s: float,
    *,
    trend_s: float = 1.0,
    pixel_var_e2: float = 0.0,
    area_px2: float = 0.0,
    background_e: FloatArray | None = None,
    exclude: BoolArray | None = None,
) -> Scintillation | None:
    """Compute the scintillation index of a flux series on a grid of frame slots.

    `flux_e` holds the flux in electrons, `period_s` apart, with `NaN` for a missing frame.
    `background_e` holds the background level that each flux subtracted, in electrons per pixel on
    the same slots, and its noise enters the floor. `exclude` marks the slots to skip, such as
    saturated frames. Returns `None` when fewer than 20 frames count.
    """
    valid = np.isfinite(flux_e) & (flux_e > 0.0)
    if exclude is not None:
        valid &= ~exclude
    if int(valid.sum()) < MIN_FRAMES:
        return None
    half = max(1, round(0.5 * trend_s / period_s))
    local_mean, neighbors = _moving_mean(flux_e, valid, half)
    usable = valid & (neighbors >= 3)
    if int(usable.sum()) < MIN_FRAMES:
        return None
    trend = local_mean[usable]
    n = neighbors[usable].astype(np.float64)
    relative = flux_e[usable] / trend - 1.0
    raw = float(np.mean(relative**2 * n / (n - 1.0)))
    noise_e2 = area_px2 * pixel_var_e2
    if background_e is not None:
        noise_e2 += area_px2 * area_px2 * _detrended_variance(background_e, usable, half)
    floor = float(np.mean(1.0 / trend + noise_e2 / trend**2))
    return Scintillation(
        index=max(raw - floor, 0.0),
        raw_variance=raw,
        floor=floor,
        mean_flux_e=float(np.mean(flux_e[usable])),
        frames=int(usable.sum()),
    )


def _moving_mean(
    values: FloatArray, valid: BoolArray, half: int
) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """The centered moving mean of the valid values within `half` slots of each slot, and the
    number of valid values that it holds (the mean is `NaN` where it holds none)."""
    count = len(values)
    sums = np.concatenate([[0.0], np.cumsum(np.where(valid, values, 0.0))])
    members = np.concatenate([[0], np.cumsum(valid, dtype=np.int64)])
    index = np.arange(count)
    low = np.maximum(index - half, 0)
    high = np.minimum(index + half + 1, count)
    neighbors = members[high] - members[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (sums[high] - sums[low]) / neighbors
    return mean, neighbors


def _detrended_variance(values: FloatArray, at: BoolArray, half: int) -> float:
    """The variance of `values` about their centered moving mean, over the slots `at`.

    The moving mean holds the value itself, which removes `1 / n` of its variance, so each square
    is scaled by `n / (n - 1)`. Returns 0 when fewer than 20 slots count.
    """
    valid = np.isfinite(values)
    local_mean, neighbors = _moving_mean(values, valid, half)
    keep = at & valid & (neighbors >= 3)
    if int(keep.sum()) < MIN_FRAMES:
        return 0.0
    n = neighbors[keep].astype(np.float64)
    return float(np.mean((values[keep] - local_mean[keep]) ** 2 * n / (n - 1.0)))
