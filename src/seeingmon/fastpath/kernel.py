"""The per-frame kernel: background, centroid, widths, flux, peak, and flags.

For each frame, the kernel works in four steps.

1. **Background.** The median of the ROI border (a ring of `border_px` pixels) is the local
   background. The median ignores the star, hot pixels, and a neighbor in a corner. Half the
   spread between the 15.9th and the 84.1st percentile of the same pixels is the sky noise of
   the frame, a robust standard deviation that holds the photon noise of a bright sky.
2. **Centroid.** The kernel measures the intensity-weighted centroid of the background-subtracted
   pixels inside a circular aperture, and it recenters the aperture on the result
   `recenter_iterations` times (so it measures `recenter_iterations + 1` times). The aperture edge
   is soft: a pixel at distance `r` from the center has the weight `clip(R + 0.5 - r, 0, 1)`, so
   the centroid changes smoothly with the aperture position. The kernel starts from the previous
   centroid when you give a `guess`, and otherwise from the brightest 3 x 3 patch.
3. **Moments.** The second-moment widths, the peak (the brightest pixel in the aperture box), and
   the flux (the weighted sum, in container counts) come from the last aperture.
4. **Flags.** Saturation, a star near the ROI edge, a missing star, and an isolated bright pixel.

**Coordinates.** Centroids use the pixel coordinates of the readout mode, so the kernel adds the
ROI origin to every position. The center of pixel `(0, 0)` is `(0, 0)`. A ROI move therefore
never looks like image motion.

**Counts.** A frame holds container counts. A `uint16` frame carries the ADC value in the high
bits, and a `uint8` frame carries the top 8 bits. `FrameCalibration` turns container counts into
electrons and gives the saturation level. The kernel never rescales the pixels itself.

**Speed.** One matrix product gives all the weighted sums of an aperture, because the kernel keeps
a table of aperture weights for 16 x 16 sub-pixel positions of the aperture center. The table
quantizes the aperture position to 1/16 pixel. Because the aperture follows the star, the
quantization changes the centroid by less than 0.001 pixel (the leakage of the aperture, about 2%,
times half a step). `measure_frame` takes about 35 microseconds for a 128 x 128 frame on a quiet
desktop, 12 of them for the matched filter (`seeingmon.fastpath.benchmark` measures it), and
`measure_stack` handles a 3-D stack with the same code, so the two agree exactly.

**Noise.** `Measurement.noise_var_x` and `noise_var_y` hold the modeled variance of the centroid
from the star's photon noise and the noise of the pixels, in square pixels: `sigma_x^2 / F +
v K / F^2`, where `F` is the flux in electrons above the trimmed mean of the border, `sigma_x` the
second-moment width, `v` the variance of one pixel (`pixel_variance_e2`: the sky noise measured on
the border, or the modeled read and quantization noise when that is larger or the sky noise does
not exceed one ADC step), and `K` the sum of the squared aperture weights times the squared
distance from the center along one axis (2,935 px^4 for the aperture of 16 px). The estimator
subtracts it from the motion variance. The flux of a frame is noisy, and so is the centroid's own
denominator, so the mean of `1 / F^2` over the frames matches the noise of the centroids without a
correction. In a dark sky the star's photons dominate. In daylight the sky does: the aperture adds
the noise of about 200 pixels of sky, `v K / F^2` is about 4 times the motion variance of a window
at an `r0` of 10 cm, and a model without the sky read `r0` a third of the truth. The aperture
recenters on its own noisy centroid, which the model of a fixed aperture leaves out, and in the
simulator's bright skies the true noise exceeds the model by 4 to 6% at the exposures of the
adaptive loop (`docs/research-notes.md`, "The seeing in a bright sky").

**The Gaussian-weighted centroid.** With `centroid_fwhm_px`, the position comes from a centroid
weighted by a Gaussian of that FWHM instead of the aperture: the point `x` where `sum(W (u - x)
(I - b)) = 0` for the weights `W = exp(-((u - x)^2 + (v - y)^2) / (2 s^2))`, found by Newton steps
from the matched filter's peak (or from the aperture's centroid without one), so the weights
follow the star. On a sky that dominates the noise, a weight as wide as a Gaussian image reaches
the variance `8 pi s^4 v / F^2`, the lowest that any estimator reaches for that image (the
Cramer-Rao bound), and a weight twice as wide reaches 2.4 times that. The aperture of 16 px
reaches `v K / F^2`, about 540 times the bound for the simulator's image in daylight, and a weight
of 3 Airy FWHM about 5 times. The modeled noise of the weighted centroid is the propagation of the
pixel noise through that equation: `sum(W^2 (u - x)^2 (v + F P)) / D^2`, with `D = sum(W (I - b)
(1 - (u - x)^2 / s^2))`, both from the pixels of the frame. The flux, the widths, the SNR, and the
flags still come from the aperture.

**Detection.** Two signal-to-noise ratios describe the star, and both take the sky from the
trimmed mean of the border (its central 68%), which rounds far less than the median of whole
counts: in a faint twilight sky the median can sit half an ADC step off, and over the aperture
that looks like a star. Both take `v`, the variance of one pixel in electrons squared, as the
larger of the modeled pixel noise (read noise and quantization) and the square of the measured
sky noise, because the measured noise already holds the read noise. The measured noise counts
only when it exceeds one ADC step (`pixel_variance_e2`).

- `matched_snr` is the SNR of a filter matched to the image of the star
  (`seeingmon.fastpath.matched`). `measure_frame` takes the first filter of `matched_fwhms_px`,
  at its best position within about 1.4 px of the centroid.
- `snr` is the SNR of the centroid aperture, `F / sqrt(F + A v)`, where `A` is the area of the
  aperture and `F` the aperture sum above the trimmed mean. It describes the noise of the
  centroid, and the window reports its median as `star_snr`. On a sky that dominates the noise,
  it is about a fifth of `matched_snr` for a star in focus, because the aperture adds the noise
  of about 200 pixels of sky.

A star counts as missing when both stay below `min_snr`, so a bright sky does not make a star out
of its own noise. The matched filter keeps a faint star in a bright sky, where the aperture loses
it, and the aperture keeps a bright star whose image is far wider than the filter, such as a
defocused one in the rapid focus mode. A frame whose centroid strays more than about 1.4 px from a
faint star, as the centroid of a faint star in a bright sky can, counts as missing, and its
centroid would not be usable. Without `matched_fwhms_px`, the aperture decides alone.

Both leave out the noise of the background level. For the aperture, that makes the SNR about 1.2
times too high in a sky that dominates the noise. For a matched filter, it is about 1% of the
variance with the Airy FWHM, and 14% with 4 Airy FWHM, the widest filter of the search.

**Search.** `search_frame` serves the search bursts: it finds the star with each matched filter of
`matched_fwhms_px` within a radius of a predicted position (`seeingmon.fastpath.matched.search`),
keeps the filter with the highest SNR, so that the size of the image need not be known, and
reports the position, that `matched_snr`, and the SNR of the centroid aperture at that position,
without the centroid loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from seeingmon.fastpath import matched
from seeingmon.frames import FrameData

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
IntImage = npt.NDArray[np.integer[Any]]

# Analysis flags use the bits that the `frame` record reserves above the `FrameFlag` bits.
FLAG_SATURATED = 1 << 5
FLAG_EDGE = 1 << 6
FLAG_NO_STAR = 1 << 7
FLAG_HOT_PIXEL = 1 << 8

FRAME_FLAGS: dict[str, int] = {
    "saturated": FLAG_SATURATED,
    "edge": FLAG_EDGE,
    "no_star": FLAG_NO_STAR,
    "hot_pixel": FLAG_HOT_PIXEL,
}
"""The analysis flags of a frame, by name. Bits 0 to 4 copy `seeingmon.frames.FrameFlag`."""

UNUSABLE_FLAGS = FLAG_NO_STAR | FLAG_EDGE
"""A frame with one of these flags has no usable centroid. The window statistics skip it."""

PHASES = 16  # sub-pixel positions of the aperture center along each axis
_HALF_PHASE = PHASES // 2
_NAN = math.nan
_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
_UNSET = -(10**9)
_BATCH_CHUNK = 256


@dataclass(frozen=True, slots=True)
class KernelParams:
    """The settings of the kernel that do not depend on the frame.

    `aperture_diameter_px` is the diameter at half weight. `border_px` is the width of the ring
    that gives the background, and the median uses the pixels of the ring whose row and column are
    both multiples of `border_step` (1 uses the whole ring, and 2 uses a quarter of it). A star
    closer to the ROI edge than the aperture radius plus `edge_margin_px` carries the edge flag. A
    star whose matched filter has a signal-to-noise ratio below `min_snr` counts as missing.
    `spike_ratio` enables the isolated-pixel test: when the neighbors of the brightest pixel of the
    aperture box stay below `spike_ratio` times its excess over the background, the frame carries
    the hot-pixel flag. Use `None` for a mode that undersamples the star, because a real star can
    fill one pixel there. `matched_fwhms_px` holds the FWHMs of the matched filters, in pixels.
    `measure_frame` uses the first in its missing-star test, and `search_frame` tries them all and
    keeps the best. Empty, the kernel uses no matched filter, and the SNR of the aperture decides
    alone. `centroid_fwhm_px` switches the position from the centroid of the aperture to the
    Gaussian-weighted centroid of that FWHM, in pixels. `None`, the default, keeps the aperture.
    """

    aperture_diameter_px: float = 16.0
    recenter_iterations: int = 2
    border_px: int = 4
    border_step: int = 2
    edge_margin_px: float = 1.0
    min_snr: float = 6.0
    spike_ratio: float | None = 0.03
    matched_fwhms_px: tuple[float, ...] = ()
    centroid_fwhm_px: float | None = None

    def __post_init__(self) -> None:
        if not 3.0 <= self.aperture_diameter_px <= 60.0:
            raise ValueError("aperture_diameter_px must be between 3 and 60")
        if not 0 <= self.recenter_iterations <= 20:
            raise ValueError("recenter_iterations must be between 0 and 20")
        if self.border_px < 1 or self.border_step < 1:
            raise ValueError("border_px and border_step must be at least 1")
        if self.min_snr < 0 or self.edge_margin_px < 0:
            raise ValueError("min_snr and edge_margin_px must not be negative")
        if self.spike_ratio is not None and not 0.0 < self.spike_ratio < 1.0:
            raise ValueError("spike_ratio must be between 0 and 1, or None")
        if not all(0.3 <= fwhm <= 20.0 for fwhm in self.matched_fwhms_px):
            raise ValueError("each of matched_fwhms_px must be between 0.3 and 20")
        if self.centroid_fwhm_px is not None and not 0.5 <= self.centroid_fwhm_px <= 20.0:
            raise ValueError("centroid_fwhm_px must be between 0.5 and 20, or None")

    @property
    def radius_px(self) -> float:
        """The aperture radius: the distance at which the weight falls to one half."""
        return self.aperture_diameter_px / 2.0

    @property
    def half_box_px(self) -> int:
        """Pixels from the center pixel to the edge of the aperture box."""
        return math.ceil(self.radius_px + 1.0)

    @property
    def box_px(self) -> int:
        """The side of the square box that holds the aperture, in pixels."""
        return 2 * self.half_box_px + 1

    @property
    def area_px2(self) -> float:
        """The sum of the aperture weights, which is the area of the aperture in pixels."""
        return _tables(self.radius_px, self.half_box_px).area

    @property
    def second_moment_px4(self) -> float:
        """The sum of the weights times the squared distance along one axis from the center."""
        return _tables(self.radius_px, self.half_box_px).second_moment

    @property
    def noise_moment_px4(self) -> float:
        """The sum of the squared weights times the squared distance along one axis from the
        center: the factor of the pixel noise in the variance of the aperture's centroid."""
        return _tables(self.radius_px, self.half_box_px).noise_moment


@dataclass(frozen=True, slots=True)
class _Tables:
    """Aperture weights for 16 x 16 sub-pixel centers, as the five moments of each aperture.

    `moments[ky, kx]` has the shape `(5, box * box)`: the weight, the weight times `u`, times
    `v`, times `u^2`, and times `v^2`, for the pixel offsets `(u, v)` from the center pixel of
    the box, in row-major order. `sums_list` holds the sum of each moment over the box as Python
    floats, which the background subtraction needs. `area` is the sum of the weights of a
    centered aperture, `second_moment` is the sum of the weights times `u^2`, and `noise_moment`
    the sum of the squared weights times `u^2`. The soft edge makes `noise_moment` about 9%
    smaller than `second_moment` for an aperture of 16 px.
    """

    moments: npt.NDArray[np.float64]
    sums_list: list[list[list[float]]]
    area: float
    second_moment: float
    noise_moment: float


@lru_cache(maxsize=8)
def _tables(radius_px: float, half: int) -> _Tables:
    box = 2 * half + 1
    offsets = np.arange(box, dtype=np.float64) - half
    phase = (np.arange(PHASES, dtype=np.float64) - _HALF_PHASE) / PHASES
    # The aperture center sits at (phase_x, phase_y) from the center pixel of the box.
    du = offsets[None, None, None, :] - phase[None, :, None, None]  # (1, kx, 1, u)
    dv = offsets[None, None, :, None] - phase[:, None, None, None]  # (ky, 1, v, 1)
    weight = np.clip(radius_px + 0.5 - np.sqrt(du * du + dv * dv), 0.0, 1.0)  # (ky, kx, v, u)
    u = np.broadcast_to(offsets[None, None, None, :], weight.shape)
    v = np.broadcast_to(offsets[None, None, :, None], weight.shape)
    stacked = np.stack([weight, weight * u, weight * v, weight * u * u, weight * v * v], axis=2)
    moments = np.ascontiguousarray(stacked.reshape(PHASES, PHASES, 5, box * box))
    sums = moments.sum(axis=3)
    centered = sums[_HALF_PHASE, _HALF_PHASE]
    noise_moment = float(np.sum(weight[_HALF_PHASE, _HALF_PHASE] ** 2 * u[0, 0] ** 2))
    return _Tables(moments, sums.tolist(), float(centered[0]), float(centered[3]), noise_moment)


@dataclass(frozen=True, slots=True)
class FrameCalibration:
    """What the kernel needs to know about the counts of a stream.

    `full_scale_dn` is the largest container count that the ADC can produce, which the peak
    fraction divides by. `saturation_dn` is the count at which the saturation flag sets. `e_per_dn`
    is the number of electrons for one container count, and `pixel_var_e2` is the variance of one
    pixel in electrons squared (read noise plus the coarse quantization of the container). Both are
    `NaN` when the profile does not know the mode, and then the flux and the noise model are `NaN`.
    `step_dn` is the step between two ADC values in container counts (16 for a 12-bit ADC in a
    16-bit container), which the sky noise of the border needs.
    """

    full_scale_dn: float
    saturation_dn: float
    e_per_dn: float = _NAN
    pixel_var_e2: float = _NAN
    step_dn: float = 1.0

    @classmethod
    def for_container(
        cls,
        *,
        adc_bits: int,
        container_bits: int,
        saturation_fraction: float = 0.98,
        e_per_dn: float = _NAN,
        pixel_var_e2: float = _NAN,
    ) -> FrameCalibration:
        """Build the calibration of a stream from the ADC depth and the container size.

        A 16-bit container carries the ADC value in the high bits, so the full scale is
        `(2^adc - 1) * 2^(16 - adc)`. An 8-bit container carries the top 8 bits of a deeper ADC,
        and its full scale is 255. A shallower ADC fills the container from the bottom.
        """
        if not 1 <= adc_bits <= 16 or container_bits not in (8, 16):
            raise ValueError("adc_bits must be 1 to 16 and container_bits 8 or 16")
        if not 0.0 < saturation_fraction <= 1.0:
            raise ValueError("saturation_fraction must be in (0, 1]")
        if container_bits >= adc_bits:
            full_scale = float(((1 << adc_bits) - 1) << (container_bits - adc_bits))
            step = float(1 << (container_bits - adc_bits))
        else:
            full_scale = float((1 << container_bits) - 1)
            step = 1.0
        return cls(full_scale, saturation_fraction * full_scale, e_per_dn, pixel_var_e2, step)


class Measurement(NamedTuple):
    """The result of the kernel for one frame.

    `x` and `y` are the centroid in sensor pixels, and `width_x` and `width_y` are the second-moment
    sigmas in pixels. All four are `NaN` when `found` is false. `peak_dn` is the brightest pixel
    of the aperture box (of the whole frame when no star is found), and `bg_dn` is the background,
    both in container counts. `flux_dn` is the aperture sum minus the background, in container
    counts. `noise_var_x` and `noise_var_y` are the modeled centroid noise in square pixels, and
    `flags` holds the analysis flags (`FLAG_*`). `bg_sigma_dn` is the sky noise of the border in
    container counts, for every frame. `snr` is the signal-to-noise ratio of the star in the
    centroid aperture, and `matched_snr` the one of the matched filter. Both are `NaN` when `found`
    is false or when the calibration has no electron scale, and `matched_snr` also without a
    matched filter.
    """

    found: bool
    x: float
    y: float
    width_x: float
    width_y: float
    peak_dn: float
    flux_dn: float
    bg_dn: float
    noise_var_x: float
    noise_var_y: float
    flags: int
    bg_sigma_dn: float = _NAN
    snr: float = _NAN
    matched_snr: float = _NAN


# --- helpers --------------------------------------------------------------------------------


_SIGMA_LOW = 0.158655  # the share of a normal distribution below its mean minus one sigma


@dataclass(frozen=True, slots=True)
class _Border:
    """The flat indices of the border ring, and the ranks of its median and its one-sigma points.

    `ranks` holds every rank that one partition must place, in ascending order.
    """

    indices: IntArray
    low: int
    high: int
    sigma_low: int
    sigma_high: int
    ranks: tuple[int, ...]


@lru_cache(maxsize=16)
def _border(height: int, width: int, border: int, step: int) -> _Border:
    """The border ring of a frame size, with the ranks that give its median and its sky noise."""
    border = max(1, min(border, min(height, width) // 4))
    yy, xx = np.mgrid[0:height, 0:width]
    ring = ~((yy >= border) & (yy < height - border) & (xx >= border) & (xx < width - border))
    indices = np.flatnonzero(ring & (yy % step == 0) & (xx % step == 0)).astype(np.intp)
    if len(indices) == 0:
        indices = np.flatnonzero(ring).astype(np.intp)
    count = len(indices)
    low, high = (count - 1) // 2, count // 2
    sigma_low = round(_SIGMA_LOW * (count - 1))
    sigma_high = count - 1 - sigma_low
    ranks = tuple(sorted({sigma_low, low, high, sigma_high}))
    return _Border(indices, low, high, sigma_low, sigma_high, ranks)


def _border_indices(height: int, width: int, border: int, step: int) -> tuple[IntArray, int, int]:
    """The flat indices of the border ring, and the two ranks that give the median."""
    ring = _border(height, width, border, step)
    return ring.indices, ring.low, ring.high


def _border_level(flat: FrameData, border: _Border) -> tuple[float, float, float]:
    """The median of the border ring, its robust standard deviation, and its trimmed mean.

    All three are in container counts. One partition of a copy of the ring places the median and
    the two one-sigma points, and the trimmed mean averages the values between those two points,
    the central 68% of the ring. The median of whole counts can sit up to half an ADC step off the
    sky when the noise spans a few steps, and over the area of the aperture that offset looks like
    a star. The mean of the central values rounds far less, so the detection takes the sky from
    it. A star in a corner or a hot pixel falls in the tails and changes neither. The one-sigma
    points round in the same way, so a noise of about one step reads one step or half of one
    (`pixel_variance_e2` takes the model there).
    """
    values = flat[border.indices]
    values.partition(border.ranks)
    median = 0.5 * (float(values[border.low]) + float(values[border.high]))
    sigma = 0.5 * (float(values[border.sigma_high]) - float(values[border.sigma_low]))
    level = float(values[border.sigma_low : border.sigma_high + 1].mean())
    return median, sigma, level


def _locate(data: IntImage) -> tuple[float, float]:
    """The brightest 3 x 3 patch of a frame, as a first guess `(x, y)` in ROI pixels."""
    height, width = data.shape
    if height < 3 or width < 3:
        row, column = np.unravel_index(int(np.argmax(data)), data.shape)
        return float(column), float(row)
    wide = data.astype(np.int32)
    summed = wide[:, :-2] + wide[:, 1:-1] + wide[:, 2:]
    summed = summed[:-2] + summed[1:-1] + summed[2:]
    row, column = np.unravel_index(int(np.argmax(summed)), summed.shape)
    return float(column + 1), float(row + 1)


def _noise_variance(width_sq: float, flux_e: float, pixel_var_e2: float, k_ap: float) -> float:
    """The star's photon noise plus the pixel noise of the aperture's centroid along one axis, in
    square pixels. `pixel_var_e2` holds the sky, the read noise, and the quantization."""
    if not (flux_e > 0.0 and pixel_var_e2 == pixel_var_e2):
        return _NAN
    return width_sq / flux_e + pixel_var_e2 * k_ap / (flux_e * flux_e)


_WEIGHT_REACH = 4.0  # the weighted centroid reads pixels within this many sigma of the weight
_WEIGHT_STEPS = 12  # Newton steps at most
_WEIGHT_TOLERANCE_PX = 1e-4


class _Weighted(NamedTuple):
    """The Gaussian-weighted centroid in ROI pixels, and its modeled noise in square pixels."""

    x: float
    y: float
    noise_var_x: float
    noise_var_y: float


def _weighted_centroid(
    data: FrameData,
    gx: float,
    gy: float,
    level: float,
    fwhm_px: float,
    e_per_dn: float,
    pixel_var_e2: float,
) -> _Weighted | None:
    """The Gaussian-weighted centroid near `(gx, gy)` (ROI pixels), or `None` when it fails.

    `level` is the sky in the counts of the frame. The weight `W` is a Gaussian of `fwhm_px`
    centered on the estimate, and each Newton step solves `sum(W (u - x) (I - b)) = 0` along each
    axis, with the derivative `D = sum(W (I - b) (1 - (u - x)^2 / s^2))`. A step never moves more
    than one sigma of the weight. The centroid fails when `D` is not positive (no star under the
    weight), when it moves more than three sigma from the start, or when it does not converge.
    Pixels outside the frame read as the sky. Without an electron scale, the noise is `NaN`.
    """
    sigma = fwhm_px / _FWHM_PER_SIGMA
    inverse_var = 1.0 / (sigma * sigma)
    half = math.ceil(_WEIGHT_REACH * sigma)
    box = 2 * half + 1
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    x0, y0 = gx, gy
    x, y = gx, gy
    origin = (_UNSET, _UNSET)
    excess = np.empty((box, box))
    for _ in range(_WEIGHT_STEPS):
        cx, cy = round(x), round(y)
        if (cx, cy) != origin:
            origin = (cx, cy)
            excess = _read_box(data, cx - half, cy - half, box, level).reshape(box, box) - level
        u = offsets - (x - cx)
        v = offsets - (y - cy)
        gu = np.exp(-0.5 * inverse_var * u * u)
        gv = np.exp(-0.5 * inverse_var * v * v)
        rows = excess @ gu  # the weighted sums along x, one per row
        total = float(gv @ rows)
        dx = float(gv @ (excess @ (gu * u)))
        dy = float((gv * v) @ rows)
        curve_x = total - inverse_var * float(gv @ (excess @ (gu * u * u)))
        curve_y = total - inverse_var * float((gv * v * v) @ rows)
        if not (curve_x > 0.0 and curve_y > 0.0 and total > 0.0):
            return None
        step_x = max(-sigma, min(sigma, dx / curve_x))
        step_y = max(-sigma, min(sigma, dy / curve_y))
        x += step_x
        y += step_y
        if (x - x0) ** 2 + (y - y0) ** 2 > 9.0 * sigma * sigma:
            return None
        if abs(step_x) < _WEIGHT_TOLERANCE_PX and abs(step_y) < _WEIGHT_TOLERANCE_PX:
            break
    else:
        return None
    if not (e_per_dn == e_per_dn and pixel_var_e2 == pixel_var_e2):
        return _Weighted(x, y, _NAN, _NAN)
    # The noise at the last estimate: the step that ended the loop is far below the noise.
    gu2u2 = gu * gu * u * u
    gv2v2 = gv * gv * v * v
    gu2 = gu * gu
    gv2 = gv * gv
    sky_x = pixel_var_e2 * float(gu2u2.sum()) * float(gv2.sum())
    sky_y = pixel_var_e2 * float(gv2v2.sum()) * float(gu2.sum())
    photon_x = e_per_dn * float(gv2 @ (excess @ gu2u2))
    photon_y = e_per_dn * float(gv2v2 @ (excess @ gu2))
    scale_x = e_per_dn * curve_x
    scale_y = e_per_dn * curve_y
    return _Weighted(
        x,
        y,
        (sky_x + max(photon_x, 0.0)) / (scale_x * scale_x),
        (sky_y + max(photon_y, 0.0)) / (scale_y * scale_y),
    )


def _read_box(
    data: FrameData, ix0: int, iy0: int, box: int, background: float
) -> npt.NDArray[np.float64]:
    """The box of pixels at `(ix0, iy0)` as a flat float array.

    Pixels outside the frame read as the background, so they add no signal.
    """
    height, width = data.shape
    if ix0 >= 0 and iy0 >= 0 and ix0 + box <= width and iy0 + box <= height:
        return data[iy0 : iy0 + box, ix0 : ix0 + box].astype(np.float64).reshape(-1)
    out = np.full((box, box), background, dtype=np.float64)
    x_lo, x_hi = max(ix0, 0), min(ix0 + box, width)
    y_lo, y_hi = max(iy0, 0), min(iy0 + box, height)
    if x_lo < x_hi and y_lo < y_hi:
        out[y_lo - iy0 : y_hi - iy0, x_lo - ix0 : x_hi - ix0] = data[y_lo:y_hi, x_lo:x_hi]
    return out.reshape(-1)


# --- one frame ------------------------------------------------------------------------------


def measure_frame(
    data: FrameData,
    roi_x: int,
    roi_y: int,
    params: KernelParams,
    calibration: FrameCalibration,
    guess: tuple[float, float] | None = None,
) -> Measurement:
    """Measure one frame.

    `roi_x` and `roi_y` are the sensor coordinates of the first pixel of the frame. `guess` is the
    expected star position in sensor pixels, such as the centroid of the previous frame. Without a
    guess, or when the aperture finds no star at the guess, the kernel looks for the brightest
    3 x 3 patch of the frame.
    """
    height, width = data.shape
    flat = data.reshape(-1)
    border = _border(height, width, params.border_px, params.border_step)
    background, sigma, level = _border_level(flat, border)
    sky = (background, sigma, level)
    if guess is not None:
        result = _measure_at(
            data, roi_x, roi_y, params, calibration, sky, guess[0] - roi_x, guess[1] - roi_y
        )
        if result.found:
            return result
    gx, gy = _locate(data)
    return _measure_at(data, roi_x, roi_y, params, calibration, sky, gx, gy)


def pixel_variance_e2(sigma_dn: float, calibration: FrameCalibration) -> float:
    """The variance of one pixel in electrons squared: the measured sky or the modeled noise.

    The measured sky noise holds the read noise and the quantization too, so the two do not add.
    It counts only when it exceeds one ADC step (`step_dn`), so that the one-sigma points of the
    border lie more than two steps apart. Up to that, the rounding of whole counts decides what a
    sky of one level shows: a noise of 0.8 step, as the read noise of a dark bin1 frame, reads a
    full step, a border at the edge between two counts reads half a count, and one in the middle of
    a count reads none, while the pixels of the star, spread over many levels, have the rounding
    noise of the model (`e_per_adu^2 / 12`). In the owner's 8-bit videos, whose border sits at
    such an edge, the measured noise would have put the centroid noise of a window at about half of
    the motion variance, where the model gives 4.5 to 8% and the data about 2%
    (`docs/recordings-validation.md`). The model is also the floor, because a frame without noise
    measures less.

    The rule fails in an 8-bit container in a bright sky. One step there is 64 values of the 14-bit
    bin2 ADC (82 e- at gain 100, 259 e- at gain 0), so the sky's noise exceeds a step only near
    saturation, and the function returns the model, which holds no sky. In bin2 frames of 64 x 64
    pixels with a sky at 0.1 to 0.3 of saturation, the modeled noise of the centroid was 4 to 11
    times too low, so `r0` reads low, and `noisy` does not mark the window. The median of whole
    counts that the centroid subtracts can also sit up to half a count off such a sky, and in one
    of those tests it moved the centroids by several pixels. The fast stream uses 16-bit
    containers, where a step is one ADC value (3.5 e- in bin1 and 4 e- in bin2 at gain 0). Read
    the seeing of 8-bit video in a dark sky only (`docs/research-notes.md`, "The seeing in a
    bright sky").
    """
    modeled = calibration.pixel_var_e2
    if not sigma_dn > calibration.step_dn:
        return modeled
    measured = (sigma_dn * calibration.e_per_dn) ** 2
    return measured if measured > modeled else modeled


def _measure_at(
    data: FrameData,
    roi_x: int,
    roi_y: int,
    params: KernelParams,
    calibration: FrameCalibration,
    sky: tuple[float, float, float],
    gx: float,
    gy: float,
) -> Measurement:
    """Measure the star near `(gx, gy)`, in ROI pixels.

    `sky` holds the background (the median of the border), the sky noise, and the trimmed mean of
    the border. The centroid, the widths, and the flux subtract the background. The detection
    (the SNR and the missing-star test) subtracts the trimmed mean instead.
    """
    background, sigma, level = sky
    height, width = data.shape
    half = params.half_box_px
    box = 2 * half + 1
    tables = _tables(params.radius_px, half)
    moments, sums_list = tables.moments, tables.sums_list
    pixels: npt.NDArray[np.float64] | None = None
    last_origin = (_UNSET, _UNSET)
    last_q = (_UNSET, _UNSET)
    s0 = su = sv = suu = svv = 0.0
    for _ in range(params.recenter_iterations + 1):
        qx = round(gx * PHASES)
        qy = round(gy * PHASES)
        if (qx, qy) == last_q:
            break  # the aperture sits where it did, so the sums would not change
        last_q = (qx, qy)
        px = (qx + _HALF_PHASE) // PHASES
        py = (qy + _HALF_PHASE) // PHASES
        kx = qx - px * PHASES + _HALF_PHASE
        ky = qy - py * PHASES + _HALF_PHASE
        origin = (px - half, py - half)
        if pixels is None or origin != last_origin:
            pixels = _read_box(data, origin[0], origin[1], box, background)
            last_origin = origin
        raw = (moments[ky, kx] @ pixels).tolist()
        sums = sums_list[ky][kx]
        s0 = raw[0] - background * sums[0]
        if not s0 > 0.0:
            return _not_found(data, background, sigma)
        su = raw[1] - background * sums[1]
        sv = raw[2] - background * sums[2]
        suu = raw[3] - background * sums[3]
        svv = raw[4] - background * sums[4]
        gx = px + su / s0
        gy = py + sv / s0
    assert pixels is not None
    mean_u, mean_v = su / s0, sv / s0
    width_x_sq = max(suu / s0 - mean_u * mean_u, 0.0)
    width_y_sq = max(svv / s0 - mean_v * mean_v, 0.0)
    if not (-1.0 < gx < width and -1.0 < gy < height):
        return _not_found(data, background, sigma)
    peak_index = int(pixels.argmax())
    peak = float(pixels[peak_index])
    flux_e = s0 * calibration.e_per_dn
    snr = _NAN
    matched_snr = _NAN
    noise_x = noise_y = _NAN
    pixel_var = _NAN
    start = (gx, gy)  # where the weighted centroid starts
    if flux_e == flux_e and calibration.pixel_var_e2 == calibration.pixel_var_e2:
        detected_e = (s0 + (background - level) * sums[0]) * calibration.e_per_dn
        if not detected_e > 0.0:
            return _not_found(data, background, sigma)
        pixel_var = pixel_variance_e2(sigma, calibration)
        snr = detected_e / math.sqrt(detected_e + tables.area * pixel_var)
        if params.matched_fwhms_px:
            best = matched.near(
                data,
                round(gx),
                round(gy),
                matched.matched_filter(params.matched_fwhms_px[0]),
                level,
                calibration.e_per_dn,
                pixel_var,
            )
            matched_snr = best.snr
            start = (best.x, best.y)
        if not (snr >= params.min_snr or matched_snr >= params.min_snr):
            return _not_found(data, background, sigma)
        # The flux above the trimmed mean: the median of whole counts can sit half an ADC step
        # off the sky, and over the aperture that would bias `F` by several percent in daylight.
        k_ap = tables.noise_moment
        noise_x = _noise_variance(width_x_sq, detected_e, pixel_var, k_ap)
        noise_y = _noise_variance(width_y_sq, detected_e, pixel_var, k_ap)
    elif peak - background < params.min_snr:
        return _not_found(data, background, sigma)
    if params.centroid_fwhm_px is not None:
        weighted = _weighted_centroid(
            data,
            start[0],
            start[1],
            level,
            params.centroid_fwhm_px,
            calibration.e_per_dn,
            pixel_var,
        )
        if weighted is None or not (-1.0 < weighted.x < width and -1.0 < weighted.y < height):
            return _not_found(data, background, sigma)
        gx, gy, noise_x, noise_y = weighted
    flags = 0
    if peak >= calibration.saturation_dn:
        flags |= FLAG_SATURATED
    edge = min(gx, width - 1.0 - gx, gy, height - 1.0 - gy)
    if edge < params.radius_px + params.edge_margin_px:
        flags |= FLAG_EDGE
    if params.spike_ratio is not None and _is_spike(pixels, box, peak_index, background, params):
        flags |= FLAG_HOT_PIXEL
    return Measurement(
        True,
        roi_x + gx,
        roi_y + gy,
        math.sqrt(width_x_sq),
        math.sqrt(width_y_sq),
        peak,
        s0,
        background,
        noise_x,
        noise_y,
        flags,
        sigma,
        snr,
        matched_snr,
    )


def search_frame(
    data: FrameData,
    roi_x: int,
    roi_y: int,
    params: KernelParams,
    calibration: FrameCalibration,
    at: tuple[float, float] | None = None,
    radius_px: float | None = None,
) -> Measurement:
    """Look for the star with the matched filter, for a frame of a search burst.

    `at` is where the star should be, in sensor pixels, and each filter of `matched_fwhms_px` looks
    within `radius_px` of it. Without `at` or `radius_px`, they look over the whole frame. The
    filter with the highest SNR at its best position wins, and the star is found when that SNR
    reaches `min_snr`. The result holds that position, that SNR as `matched_snr`, the brightest
    pixel and the flux of the centroid aperture placed there with the SNR of that aperture, and
    the saturation and edge flags. It has no widths and no centroid noise, because the centroid
    loop does not run.

    Without `matched_fwhms_px` or an electron scale, the result is that of `measure_frame` from
    `at`, which may lie anywhere in the frame.
    """
    if not params.matched_fwhms_px or not (
        calibration.e_per_dn == calibration.e_per_dn
        and calibration.pixel_var_e2 == calibration.pixel_var_e2
    ):
        return measure_frame(data, roi_x, roi_y, params, calibration, at)
    height, width = data.shape
    flat = data.reshape(-1)
    background, sigma, level = _border_level(
        flat, _border(height, width, params.border_px, params.border_step)
    )
    if at is None or radius_px is None:
        cx, cy = 0.5 * (width - 1), 0.5 * (height - 1)
        radius = math.hypot(width, height)  # the whole frame, from its center
    else:
        cx, cy, radius = at[0] - roi_x, at[1] - roi_y, radius_px
    pixel_var = pixel_variance_e2(sigma, calibration)
    found: matched.MatchedPeak | None = None
    for fwhm in params.matched_fwhms_px:
        candidate = matched.search(
            data,
            cx,
            cy,
            radius,
            matched.matched_filter(fwhm),
            level,
            calibration.e_per_dn,
            pixel_var,
        )
        if candidate is not None and (found is None or candidate.snr > found.snr):
            found = candidate
    if found is None or not found.snr >= params.min_snr:
        return _not_found(data, background, sigma)
    gx, gy = found.x, found.y
    half = params.half_box_px
    box = 2 * half + 1
    tables = _tables(params.radius_px, half)
    qx = round(gx * PHASES)
    qy = round(gy * PHASES)
    px = (qx + _HALF_PHASE) // PHASES
    py = (qy + _HALF_PHASE) // PHASES
    kx = qx - px * PHASES + _HALF_PHASE
    ky = qy - py * PHASES + _HALF_PHASE
    pixels = _read_box(data, px - half, py - half, box, background)
    s0 = float(tables.moments[ky, kx, 0] @ pixels) - background * tables.sums_list[ky][kx][0]
    detected_e = (s0 + (background - level) * tables.sums_list[ky][kx][0]) * calibration.e_per_dn
    snr = _NAN
    if detected_e > 0.0:
        snr = detected_e / math.sqrt(detected_e + tables.area * pixel_var)
    peak = float(pixels.max())
    flags = 0
    if peak >= calibration.saturation_dn:
        flags |= FLAG_SATURATED
    if min(gx, width - 1.0 - gx, gy, height - 1.0 - gy) < params.radius_px + params.edge_margin_px:
        flags |= FLAG_EDGE
    return Measurement(
        True,
        roi_x + gx,
        roi_y + gy,
        _NAN,
        _NAN,
        peak,
        s0,
        background,
        _NAN,
        _NAN,
        flags,
        sigma,
        snr,
        found.snr,
    )


def _is_spike(
    pixels: FloatArray, box: int, peak_index: int, background: float, params: KernelParams
) -> bool:
    """Whether the brightest pixel of the box stands alone, as a hot pixel or a cosmic ray does."""
    assert params.spike_ratio is not None
    row, column = divmod(peak_index, box)
    peak = float(pixels[peak_index])
    neighbors = background
    if column > 0:
        neighbors = max(neighbors, float(pixels[peak_index - 1]))
    if column < box - 1:
        neighbors = max(neighbors, float(pixels[peak_index + 1]))
    if row > 0:
        neighbors = max(neighbors, float(pixels[peak_index - box]))
    if row < box - 1:
        neighbors = max(neighbors, float(pixels[peak_index + box]))
    return (neighbors - background) < params.spike_ratio * (peak - background)


def _not_found(data: FrameData, background: float, sigma: float) -> Measurement:
    return Measurement(
        False,
        _NAN,
        _NAN,
        _NAN,
        _NAN,
        float(data.max()),
        _NAN,
        background,
        _NAN,
        _NAN,
        FLAG_NO_STAR,
        sigma,
        _NAN,
        _NAN,
    )


# --- a stack of frames -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StackMeasurements:
    """The results for a stack of `n` frames, one array per field of `Measurement`."""

    found: npt.NDArray[np.bool_]
    x: FloatArray
    y: FloatArray
    width_x: FloatArray
    width_y: FloatArray
    peak_dn: FloatArray
    flux_dn: FloatArray
    bg_dn: FloatArray
    noise_var_x: FloatArray
    noise_var_y: FloatArray
    flags: npt.NDArray[np.int64]
    bg_sigma_dn: FloatArray
    snr: FloatArray
    matched_snr: FloatArray

    def __len__(self) -> int:
        return int(self.found.shape[0])

    def row(self, index: int) -> Measurement:
        """The result for one frame, in the form that `measure_frame` returns."""
        return Measurement(
            bool(self.found[index]),
            float(self.x[index]),
            float(self.y[index]),
            float(self.width_x[index]),
            float(self.width_y[index]),
            float(self.peak_dn[index]),
            float(self.flux_dn[index]),
            float(self.bg_dn[index]),
            float(self.noise_var_x[index]),
            float(self.noise_var_y[index]),
            int(self.flags[index]),
            float(self.bg_sigma_dn[index]),
            float(self.snr[index]),
            float(self.matched_snr[index]),
        )


def measure_stack(
    stack: FrameData,
    roi_x: int,
    roi_y: int,
    params: KernelParams,
    calibration: FrameCalibration,
    guess: tuple[float, float] | None = None,
) -> StackMeasurements:
    """Measure a 3-D stack of frames `(frames, rows, columns)` that share one ROI.

    The loop follows the star: each frame starts from the centroid of the previous one, and a lost
    star sends the next frame back to the brightest-patch search. The result is exactly what
    `measure_frame` gives frame by frame. A vectorized form over many frames is slower than this
    loop, because it must copy a table of aperture weights for every frame, and the copy costs more
    than the call overhead that it saves.
    """
    if stack.ndim != 3:
        raise ValueError("the stack must have the shape (frames, rows, columns)")
    count = stack.shape[0]
    found = np.zeros(count, dtype=np.bool_)
    flags = np.zeros(count, dtype=np.int64)
    # x, y, width_x, width_y, peak, flux, bg, noise_x, noise_y, bg_sigma, snr, matched_snr
    columns = np.full((12, count), _NAN)
    last = guess
    for index in range(count):
        m = measure_frame(stack[index], roi_x, roi_y, params, calibration, last)
        found[index] = m.found
        flags[index] = m.flags
        columns[:, index] = (m.x, m.y, m.width_x, m.width_y, m.peak_dn, m.flux_dn, m.bg_dn,
                             m.noise_var_x, m.noise_var_y, m.bg_sigma_dn, m.snr,
                             m.matched_snr)  # fmt: skip
        last = (m.x, m.y) if m.found else None
    return StackMeasurements(
        found,
        columns[0],
        columns[1],
        columns[2],
        columns[3],
        columns[4],
        columns[5],
        columns[6],
        columns[7],
        columns[8],
        flags,
        columns[9],
        columns[10],
        columns[11],
    )
