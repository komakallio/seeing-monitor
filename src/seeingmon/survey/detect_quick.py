"""A quick star detector for the alignment view: the bright stars of a frame, fast.

`seeingmon.survey.detect` serves the survey: it searches a copy of the frame that sums 2 x 2 blocks,
subtracts the background from the whole frame, builds masks, and fits up to 3,000 stars. On a
Raspberry Pi 4 that takes more than a second for a frame of 11.7 megapixels, and most of it goes to
passes over the whole frame. The alignment view needs a few dozen good stars, so this module does
less:

1. It sums blocks of 4 x 4 pixels (one pass over the frame, on the 16-bit container).
2. SEP finds the bright sources in that small image.
3. A model fit refines the brightest stars that are not saturated, at full resolution, in stamps
   that it takes straight from the frame. The fit gets the local background and the saturation
   level as numbers, so no pass over the whole frame subtracts or masks anything.

The result is a `seeingmon.survey.detect.Detections` object, so the pointing fit, the tracker, the
solvers, and the focus value of the alignment view take it as they take the survey detections.
Stars that the model fit does not refine keep their coarse position and carry the flags `COARSE`
and `MOMENTS_ONLY`, so the pointing fit leaves them out.

The detector measures in the counts of the 16-bit container. A native count is a container count
divided by `2 ** (16 - adc_bits)`, so only the electrons per count change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import sep

from seeingmon.survey.centroid import FWHM_PER_SIGMA, fit_stars
from seeingmon.survey.detect import (
    EXTRACT_LOCK,
    PIXEL_STACK,
    DetectionError,
    Detections,
    SearchSpec,
    StarFlag,
    _blended_by_neighbors,
)

BoolArray = npt.NDArray[np.bool_]

QUICK_PIXEL_STACK = 300_000  # SEP's default: the image is small, and a big stack costs time
START_SIGMA_PX = 0.9  # the width that the fit starts from, in pixels of the full frame
STAMP_HALF = 4  # the stamp for the peak and the saturation count is 9 x 9 pixels


@dataclass(frozen=True, slots=True)
class QuickDetectOptions:
    """The settings of `detect_quick`. The defaults suit a bin2 frame of the reference camera."""

    bin_factor: int = 4
    threshold_sigma: float = 8.0  # of a sum of `bin_factor` squared pixels
    max_stars: int = 120  # the brightest sources that the detector keeps
    refine_stars: int = 80  # the brightest of them that are not saturated get the model fit
    mesh_px: int = 128  # the background mesh, in pixels of the full frame
    min_pixels: int = 2
    saturation_fraction: float = 0.98
    max_saturated_pixels: int = 6  # a star with more is not fitted
    max_peak_fraction: float = 0.5  # the fit takes stars below this share of saturation, which
    # keeps the wings of the brightest stars out of the width that the focus value uses
    edge_margin_px: float = 6.0
    spike_peak_fraction: float = 0.93  # a star with this share of its flux in one pixel is a spike
    spike_min_snr: float = 15.0
    fit_max_chi2: float = 30.0  # bright stars never match the model as well as faint ones
    fit_max_shift_px: float = 2.0

    def __post_init__(self) -> None:
        if (
            self.bin_factor < 2
            or self.threshold_sigma <= 0
            or self.max_stars < 1
            or self.refine_stars < 1
            or self.min_pixels < 1
        ):
            raise ValueError("invalid quick detector options")


def block_sum(
    data: npt.NDArray[np.uint16], factor: int, spare_bits: int = 0
) -> npt.NDArray[np.float32]:
    """Sum `factor` x `factor` blocks of a 16-bit frame into a float32 image.

    `spare_bits` is the number of low bits of the container that always read zero (the container
    holds an ADC of fewer than 16 bits, shifted up). The sum drops the two lowest of them first
    when it can, which keeps the sums of four rows inside 16 bits: it halves the memory traffic
    and loses nothing. The result is in the counts of the container either way.
    """
    height, width = data.shape
    rows, columns = height // factor, width // factor
    if factor == 4 and spare_bits >= 2:
        # Four rows of values that lost their two lowest bits still fit in 16 bits.
        source = data >> np.uint16(2)
        narrow = source[0 : rows * 4 : 4].copy()
        for offset in range(1, 4):
            narrow += source[offset : rows * 4 : 4]
        narrow = narrow[:, : columns * 4]
        total = narrow[:, 0::4].astype(np.uint32)
        for offset in range(1, 4):
            total += narrow[:, offset::4]
        return total.astype(np.float32) * np.float32(4.0)
    wide = data[0 : rows * factor : factor].astype(np.uint32)
    for offset in range(1, factor):
        wide += data[offset : rows * factor : factor]
    wide = wide[:, : columns * factor]
    exact = wide[:, 0::factor].copy()
    for offset in range(1, factor):
        exact += wide[:, offset::factor]
    return exact.astype(np.float32)


def detect_quick(
    data: npt.NDArray[np.uint16],
    *,
    adc_bits: int,
    e_per_adu: float,
    options: QuickDetectOptions | None = None,
) -> Detections:
    """Find the bright stars of a 16-bit frame. See the module text.

    `adc_bits` is the depth of the ADC (the container holds the value in its high bits), and
    `e_per_adu` converts native counts to electrons for the noise model of the fit.
    """
    opts = options or QuickDetectOptions()
    if data.ndim != 2 or data.dtype != np.uint16:
        raise DetectionError("the quick detector takes a 2-D frame of 16-bit counts")
    height, width = data.shape
    factor = opts.bin_factor
    if min(height, width) < 8 * factor:
        raise DetectionError("the frame is too small for the quick detector")
    spare = max(0, 16 - adc_bits)
    container_per_native = float(1 << spare)
    saturation = float(((1 << adc_bits) - 1) << spare) * opts.saturation_fraction
    e_per_container = e_per_adu / container_per_native

    binned = block_sum(data, factor, spare)
    if float(binned.max()) == float(binned.min()):
        raise DetectionError("the frame is constant, so it holds no signal")
    mesh = max(2, min(opts.mesh_px // factor, binned.shape[0], binned.shape[1]))
    try:
        background = sep.Background(binned, bw=mesh, bh=mesh, fw=3, fh=3)
        level_map = np.asarray(background.back(), dtype=np.float32)
        rms_map = np.asarray(background.rms(), dtype=np.float32)
        binned -= level_map
        with EXTRACT_LOCK:
            sep.set_extract_pixstack(QUICK_PIXEL_STACK)
            try:
                objects = sep.extract(
                    binned,
                    opts.threshold_sigma,
                    err=rms_map,
                    minarea=opts.min_pixels,
                    deblend_cont=1.0,
                    clean=True,
                )
            finally:
                sep.set_extract_pixstack(PIXEL_STACK)
    except Exception as error:  # SEP raises a plain Exception for its internal limits
        raise DetectionError(f"the source extraction failed: {error}") from error
    rms = float(background.globalrms) / factor
    if rms <= 0.0 or not np.isfinite(rms):
        raise DetectionError("the background noise is zero, so the frame holds no signal")

    keep = np.flatnonzero(
        (objects["flux"] > 0) & np.isfinite(objects["x"]) & np.isfinite(objects["y"])
    )
    objects = objects[keep]
    objects = objects[np.argsort(-objects["flux"], kind="stable")[: opts.max_stars]]
    n = len(objects)
    centre = 0.5 * (factor - 1)  # the center of a block, from the center of its first pixel
    x = factor * np.asarray(objects["x"], dtype=np.float64) + centre
    y = factor * np.asarray(objects["y"], dtype=np.float64) + centre
    flux = np.asarray(objects["flux"], dtype=np.float64)
    n_pixels = np.asarray(objects["npix"], dtype=np.int32) * (factor * factor)

    # The noise and the background at each star, in the counts of one pixel of the full frame.
    block_row = np.clip(np.asarray(objects["y"]).astype(np.intp), 0, rms_map.shape[0] - 1)
    block_column = np.clip(np.asarray(objects["x"]).astype(np.intp), 0, rms_map.shape[1] - 1)
    noise = rms_map[block_row, block_column].astype(np.float64) / factor
    local = level_map[block_row, block_column].astype(np.float64) / (factor * factor)

    # A small stamp of the full frame around each star gives its peak and its saturated pixels.
    steps = np.arange(-STAMP_HALF, STAMP_HALF + 1)
    columns = np.clip(np.rint(x).astype(np.intp)[:, None] + steps, 0, width - 1)
    rows = np.clip(np.rint(y).astype(np.intp)[:, None] + steps, 0, height - 1)
    stamps = data[rows[:, :, None], columns[:, None, :]].astype(np.float64)
    peak = stamps.max(axis=(1, 2)) - local if n else np.zeros(0)
    n_saturated = (stamps >= saturation).sum(axis=(1, 2)) if n else np.zeros(0, dtype=np.intp)

    flags = np.zeros(n, dtype=np.uint16)

    def mark(mask: BoolArray, flag: StarFlag) -> None:
        flags[mask] |= np.uint16(int(flag))

    margin = opts.edge_margin_px
    mark(
        (x < margin) | (x > width - 1 - margin) | (y < margin) | (y > height - 1 - margin),
        StarFlag.NEAR_EDGE,
    )
    mark(n_saturated > 0, StarFlag.SATURATED)
    mark((np.asarray(objects["flag"]) & 3) != 0, StarFlag.BLENDED)
    snr = flux / np.sqrt(
        np.maximum(flux, 0.0) / e_per_container + np.maximum(n_pixels, 9) * noise**2
    )
    fwhm = np.full(n, FWHM_PER_SIGMA * START_SIGMA_PX)
    x_error = np.full(n, 0.5)
    y_error = np.full(n, 0.5)

    fit_these = (n_saturated <= opts.max_saturated_pixels) & (
        peak <= opts.max_peak_fraction * saturation
    )
    index = np.flatnonzero(fit_these)[: opts.refine_stars]
    refined = np.zeros(n, dtype=np.bool_)
    if index.size:
        fit = fit_stars(
            data,
            x[index],
            y[index],
            flux[index],
            np.full(index.size, START_SIGMA_PX),
            np.zeros(index.size),
            np.zeros(index.size),
            noise[index],
            e_per_adu=e_per_container,
            background0=local[index],
            bad_above=saturation,
        )
        shift = np.hypot(fit.x - x[index], fit.y - y[index])
        good = (
            fit.converged
            & (fit.chi2_reduced < opts.fit_max_chi2)
            & (shift < opts.fit_max_shift_px)
            & (fit.n_pixels >= 12)
        )
        used = index[good]
        x[used] = fit.x[good]
        y[used] = fit.y[good]
        flux[used] = fit.flux[good]
        fwhm[used] = FWHM_PER_SIGMA * fit.sigma[good]
        x_error[used] = np.maximum(fit.x_error[good], 1e-3)
        y_error[used] = np.maximum(fit.y_error[good], 1e-3)
        refined[used] = True
        mark(np.isin(np.arange(n), index[~good]), StarFlag.MOMENTS_ONLY)
    mark(~fit_these, StarFlag.MOMENTS_ONLY)
    mark(~refined, StarFlag.COARSE)
    peak_fraction = peak / np.maximum(flux, 1e-9)
    mark(
        (peak_fraction > opts.spike_peak_fraction) & (snr > opts.spike_min_snr), StarFlag.HOT_PIXEL
    )
    if n:
        reach = 3.0 * np.maximum(fwhm / FWHM_PER_SIGMA, 0.5) + 1.5
        mark(_blended_by_neighbors(x, y, reach), StarFlag.BLENDED)

    order = np.argsort(-flux, kind="stable")
    detections = Detections(
        shape=(height, width),
        x=x,
        y=y,
        flux=flux,
        peak=np.asarray(peak, dtype=np.float64),
        fwhm_px=fwhm,
        x_error_px=x_error,
        y_error_px=y_error,
        elongation=np.ones(n),
        trail_length_px=np.zeros(n),
        trail_angle_rad=np.zeros(n),
        flags=flags,
        snr=snr,
        n_pixels=n_pixels,
        background_level=float(background.globalback) / (factor * factor),
        background_rms=rms,
        background=None,
        search=SearchSpec(opts.threshold_sigma, opts.min_pixels, factor, True),
        noise_map=None,
    )
    return detections.select(order)
