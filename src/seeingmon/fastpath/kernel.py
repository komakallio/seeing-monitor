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
times half a step). `measure_frame` takes about 30 microseconds for a 128 x 128 frame on a quiet
desktop, 10 of them for the matched filter (`seeingmon.fastpath.benchmark` measures it), and
`measure_stack` handles a 3-D stack with the same code, so the two agree exactly.

**Noise.** `Measurement.noise_var_x` and `noise_var_y` hold the modeled variance of the centroid
from photon noise and pixel noise, in square pixels: `sigma_x^2 / F + n^2 K / F^2`, where `F` is
the flux in electrons, `n^2` the pixel noise variance in electrons squared, and `K` the sum of
the aperture weights times the squared distance from the center along one axis. The estimator
subtracts it from the motion variance. The model counts the read noise only, and not the sky.

**Detection.** Two signal-to-noise ratios describe the star, and both take the sky from the
trimmed mean of the border (its central 68%), which rounds far less than the median of whole
counts: in a faint twilight sky the median can sit half an ADC step off, and over the aperture
that looks like a star. Both take `v`, the variance of one pixel in electrons squared, as the
larger of the modeled pixel noise (read noise and quantization) and the square of the measured
sky noise, because the measured noise already holds the read noise.

- `matched_snr` decides whether the star is there. It is the SNR of a filter matched to the image
  of the star (`seeingmon.fastpath.matched`), at its best position within about 1.4 px of the
  centroid. A star below `min_snr` counts as missing, so a bright sky does not make a star out of
  its own noise. A frame whose centroid strays farther from the star, as the centroid of a faint
  star in a bright sky can, counts as missing too, and its centroid would not be usable. The
  filter needs `matched_fwhm_px`.
- `snr` is the SNR of the centroid aperture, `F / sqrt(F + A v)`, where `A` is the area of the
  aperture and `F` the aperture sum above the trimmed mean. It describes the noise of the
  centroid, and the window reports its median as `star_snr`. On a sky that dominates the noise,
  it is about a fifth of `matched_snr`, because the aperture adds the noise of about 200 pixels
  of sky. Without `matched_fwhm_px`, it decides instead.

Both leave out the noise of the background level. For the aperture, that makes the SNR about 1.2
times too high in a sky that dominates the noise. For the matched filter, it is about 1% of the
variance.

**Search.** `search_frame` serves the search bursts: it finds the star with the matched filter
within a radius of a predicted position (`seeingmon.fastpath.matched.search`), and it reports the
position, `matched_snr`, and the SNR of the centroid aperture at that position, without the
centroid loop.
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
    fill one pixel there. `matched_fwhm_px` is the FWHM of the matched filter, in pixels. With
    `None`, the kernel uses no matched filter, and the SNR of the aperture decides instead.
    """

    aperture_diameter_px: float = 16.0
    recenter_iterations: int = 2
    border_px: int = 4
    border_step: int = 2
    edge_margin_px: float = 1.0
    min_snr: float = 6.0
    spike_ratio: float | None = 0.03
    matched_fwhm_px: float | None = None

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
        if self.matched_fwhm_px is not None and not 0.3 <= self.matched_fwhm_px <= 20.0:
            raise ValueError("matched_fwhm_px must be between 0.3 and 20, or None")

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


@dataclass(frozen=True, slots=True)
class _Tables:
    """Aperture weights for 16 x 16 sub-pixel centers, as the five moments of each aperture.

    `moments[ky, kx]` has the shape `(5, box * box)`: the weight, the weight times `u`, times
    `v`, times `u^2`, and times `v^2`, for the pixel offsets `(u, v)` from the center pixel of
    the box, in row-major order. `sums_list` holds the sum of each moment over the box as Python
    floats, which the background subtraction needs. `area` is the sum of the weights of a
    centered aperture, and `second_moment` is the sum of the weights times `u^2`.
    """

    moments: npt.NDArray[np.float64]
    sums_list: list[list[list[float]]]
    area: float
    second_moment: float


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
    return _Tables(moments, sums.tolist(), float(centered[0]), float(centered[3]))


@dataclass(frozen=True, slots=True)
class FrameCalibration:
    """What the kernel needs to know about the counts of a stream.

    `full_scale_dn` is the largest container count that the ADC can produce, which the peak
    fraction divides by. `saturation_dn` is the count at which the saturation flag sets. `e_per_dn`
    is the number of electrons for one container count, and `pixel_var_e2` is the variance of one
    pixel in electrons squared (read noise plus the coarse quantization of the container). Both are
    `NaN` when the profile does not know the mode, and then the flux and the noise model are `NaN`.
    """

    full_scale_dn: float
    saturation_dn: float
    e_per_dn: float = _NAN
    pixel_var_e2: float = _NAN

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
        else:
            full_scale = float((1 << container_bits) - 1)
        return cls(full_scale, saturation_fraction * full_scale, e_per_dn, pixel_var_e2)


class Measurement(NamedTuple):
    """The result of the kernel for one frame.

    `x` and `y` are the centroid in sensor pixels, and `width_x` and `width_y` are the second-moment
    sigmas in pixels. All four are `NaN` when `found` is false. `peak_dn` is the brightest pixel
    of the aperture box (of the whole frame when no star is found), and `bg_dn` is the background,
    both in container counts. `flux_dn` is the aperture sum minus the background, in container
    counts. `noise_var_x` and `noise_var_y` are the modeled centroid noise in square pixels, and
    `flags` holds the analysis flags (`FLAG_*`). `bg_sigma_dn` is the sky noise of the border in
    container counts, for every frame. `snr` is the signal-to-noise ratio of the star in the
    centroid aperture, and `matched_snr` the one of the matched filter, which decides whether the
    star is there. Both are `NaN` when `found` is false or when the calibration has no electron
    scale, and `matched_snr` also without a matched filter.
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
    it. A star in a corner or a hot pixel falls in the tails and changes neither.
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


def _noise_variance(
    width_sq: float, flux_e: float, calibration: FrameCalibration, k_ap: float
) -> float:
    """Photon noise plus pixel noise of a centroid along one axis, in square pixels."""
    if not (flux_e > 0.0 and calibration.pixel_var_e2 == calibration.pixel_var_e2):
        return _NAN
    return width_sq / flux_e + calibration.pixel_var_e2 * k_ap / (flux_e * flux_e)


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


def _pixel_variance_e2(sigma_dn: float, calibration: FrameCalibration) -> float:
    """The variance of one pixel in electrons squared: the measured sky or the modeled noise.

    The measured sky noise holds the read noise and the quantization too, so the two do not add.
    The model is the floor, because a frame without noise, or one whose quantized border hides
    the noise, measures less than the read noise.
    """
    measured = (sigma_dn * calibration.e_per_dn) ** 2
    modeled = calibration.pixel_var_e2
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
    if flux_e == flux_e and calibration.pixel_var_e2 == calibration.pixel_var_e2:
        detected_e = (s0 + (background - level) * sums[0]) * calibration.e_per_dn
        if not detected_e > 0.0:
            return _not_found(data, background, sigma)
        pixel_var = _pixel_variance_e2(sigma, calibration)
        snr = detected_e / math.sqrt(detected_e + tables.area * pixel_var)
        if params.matched_fwhm_px is not None:
            best = matched.near(
                data,
                round(gx),
                round(gy),
                matched.matched_filter(params.matched_fwhm_px),
                level,
                calibration.e_per_dn,
                pixel_var,
            )
            matched_snr = best.snr
            if not matched_snr >= params.min_snr:
                return _not_found(data, background, sigma)
        elif snr < params.min_snr:
            return _not_found(data, background, sigma)
    elif peak - background < params.min_snr:
        return _not_found(data, background, sigma)
    flags = 0
    if peak >= calibration.saturation_dn:
        flags |= FLAG_SATURATED
    edge = min(gx, width - 1.0 - gx, gy, height - 1.0 - gy)
    if edge < params.radius_px + params.edge_margin_px:
        flags |= FLAG_EDGE
    if params.spike_ratio is not None and _is_spike(pixels, box, peak_index, background, params):
        flags |= FLAG_HOT_PIXEL
    k_ap = tables.second_moment
    return Measurement(
        True,
        roi_x + gx,
        roi_y + gy,
        math.sqrt(width_x_sq),
        math.sqrt(width_y_sq),
        peak,
        s0,
        background,
        _noise_variance(width_x_sq, flux_e, calibration, k_ap),
        _noise_variance(width_y_sq, flux_e, calibration, k_ap),
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

    `at` is where the star should be, in sensor pixels, and the filter looks within `radius_px` of
    it. Without `at` or `radius_px`, it looks over the whole frame. The star is found when the
    matched filter reaches `min_snr` at its best position. The result holds that position,
    `matched_snr`, the brightest pixel and the flux of the centroid aperture placed there with
    the SNR of that aperture, and the saturation and edge flags. It has no widths and no centroid
    noise, because the centroid loop does not run.

    Without `matched_fwhm_px` or an electron scale, the result is that of `measure_frame` from
    `at`, which may lie anywhere in the frame.
    """
    if params.matched_fwhm_px is None or not (
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
    pixel_var = _pixel_variance_e2(sigma, calibration)
    found = matched.search(
        data,
        cx,
        cy,
        radius,
        matched.matched_filter(params.matched_fwhm_px),
        level,
        calibration.e_per_dn,
        pixel_var,
    )
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
