"""Aperture photometry of the matched stars, in electrons.

The detector measures the position and the width of each star with a model fit. The zero point
needs the light of the whole star, and it must not depend on the shape of the profile, so this
module measures the flux again with an aperture. A star that the sky turns trails along a line,
so the aperture is a stadium: a line of the trail length with a circle of radius `aperture_px`
around it. The background comes from a ring around the aperture (the same stadium shape, wider),
which the module clips of stars and hot pixels and averages. The flux in electrons is the
aperture sum above the background times the electrons per count of the gain.

**What counts as a measurement.** A star gets a measurement when

- the ring holds at least `min_annulus_px` good pixels,
- no pixel of the aperture is saturated, masked as hot, or outside the frame, and
- no other detection lies near it and holds more than `isolation_flux_ratio` of its flux.

The error of each star follows the photon noise of the star, the noise of the pixels around it
(which includes the sky, the dark current, and the read noise), and the error of the
background that it subtracted.

**Aperture loss.** A finite aperture misses the wings of the profile (the Airy rings and the
seeing halo), 2% to 3% of the light for the reference optics, which is 0.03 mag. The sky is
measured pixel by pixel, so it has no such loss, and the two would disagree by that amount. The
module therefore measures a growth curve: the brightest isolated stars get a second, wide
aperture (`growth_aperture_px`, 12 pixels by default) with its own ring, and the median ratio of
the wide flux to the narrow flux is the aperture correction. The rates that
`measure_matched_stars` returns include it, so the zero point refers to the light within 12
pixels (about 99% of the total), and the sky brightness and the zero point share one scale.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from seeingmon.survey import _scipy
from seeingmon.survey.detect import UNRELIABLE, Detections
from seeingmon.survey.geometry import FloatArray

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]

_MAX_TRAIL_PX = 24.0  # a longer trail is not a star
_CHUNK = 96
_MAG_PER_E = 2.5 / np.log(10.0)  # 1.0857: the magnitude error of a relative flux error


@dataclass(frozen=True, slots=True)
class PhotometryOptions:
    """The geometry and the limits of the aperture photometry, in pixels of the frame."""

    aperture_px: float = 5.0
    annulus_inner_px: float = 9.0
    annulus_outer_px: float = 14.0
    min_annulus_px: int = 150
    clip_sigma: float = 3.0
    clip_iterations: int = 3
    isolation_flux_ratio: float = 0.02
    isolation_margin_px: float = 3.0
    growth_aperture_px: float = 12.0  # the wide aperture of the aperture correction
    growth_stars: int = 40  # the brightest stars that measure it (0 turns the correction off)
    growth_min_stars: int = 8
    growth_min_snr: float = 50.0
    max_aperture_correction: float = 1.25

    def __post_init__(self) -> None:
        if not 0 < self.aperture_px < self.annulus_inner_px < self.annulus_outer_px:
            raise ValueError("the radii must satisfy 0 < aperture < inner < outer")
        if self.min_annulus_px < 10 or self.clip_sigma <= 0 or self.clip_iterations < 0:
            raise ValueError("invalid clipping parameters")
        if self.growth_stars < 0 or (
            self.growth_stars > 0 and self.growth_aperture_px <= self.aperture_px
        ):
            raise ValueError("the growth aperture must be wider than the aperture")


@dataclass(frozen=True, slots=True, eq=False)
class AperturePhotometry:
    """The measurements of a list of stars. Every array has one entry for each star.

    `flux_e` is the light inside the aperture above the background, in electrons.
    `background_dn` is the level of the ring in counts per pixel (the bias included), and
    `noise_e_px` is the scatter of the ring in electrons per pixel.
    `ok` says whether the star has a valid measurement. The other arrays are NaN where it has
    none.
    """

    flux_e: FloatArray
    error_e: FloatArray
    background_dn: FloatArray
    noise_e_px: FloatArray
    aperture_px2: FloatArray
    ok: BoolArray


def aperture_photometry(
    data: npt.NDArray[np.float32],
    x: FloatArray,
    y: FloatArray,
    trail_length_px: FloatArray,
    trail_angle_rad: FloatArray,
    *,
    e_per_adu: float,
    saturation_dn: float,
    options: PhotometryOptions | None = None,
    bad: BoolArray | None = None,
) -> AperturePhotometry:
    """Measure stars with a stadium aperture and a local background.

    `data` is the frame in native counts, `x` and `y` are pixel positions in the array (the
    center of the first pixel is `(0, 0)`), and the trail arrays give the length and the
    direction (from the x axis toward the y axis) of each star's trail. `saturation_dn` is the
    count at which a pixel saturates, and `bad` is an optional mask of pixels to avoid. The
    function does not check for neighbors: see `isolated`.
    """
    cfg = options or PhotometryOptions()
    count = int(x.size)
    flux = np.full(count, np.nan)
    error = np.full(count, np.nan)
    background = np.full(count, np.nan)
    noise = np.full(count, np.nan)
    area = np.full(count, np.nan)
    ok = np.zeros(count, dtype=np.bool_)
    height, width = data.shape
    half = np.minimum(np.asarray(trail_length_px, dtype=np.float64), _MAX_TRAIL_PX) / 2.0
    cos_a = np.cos(trail_angle_rad)
    sin_a = np.sin(trail_angle_rad)
    saturated_at = np.float32(saturation_dn * 0.98)
    for start in range(0, count, _CHUNK):
        part = slice(start, min(start + _CHUNK, count))
        h = half[part].astype(np.float32)
        reach = int(np.ceil(cfg.annulus_outer_px + float(h.max()) + 1.0))
        offsets = np.arange(-reach, reach + 1)
        px = x[part].astype(np.float32)
        py = y[part].astype(np.float32)
        cx = np.rint(px).astype(np.intp)
        cy = np.rint(py).astype(np.intp)
        xi = cx[:, None] + offsets[None, :]  # (stars, columns)
        yi = cy[:, None] + offsets[None, :]  # (stars, rows)
        inside = ((yi >= 0) & (yi < height))[:, :, None] & ((xi >= 0) & (xi < width))[:, None, :]
        xc = np.clip(xi, 0, width - 1)
        yc = np.clip(yi, 0, height - 1)
        values = data[yc[:, :, None], xc[:, None, :]]
        invalid = ~inside | (values >= saturated_at)
        if bad is not None:
            invalid |= bad[yc[:, :, None], xc[:, None, :]]
        dx = xi[:, None, :].astype(np.float32) - px[:, None, None]
        dy = yi[:, :, None].astype(np.float32) - py[:, None, None]
        c = cos_a[part].astype(np.float32)[:, None, None]
        s = sin_a[part].astype(np.float32)[:, None, None]
        along = np.clip(dx * c + dy * s, -h[:, None, None], h[:, None, None])
        distance = np.hypot(dx - along * c, dy - along * s)
        in_ring = (distance >= cfg.annulus_inner_px) & (distance <= cfg.annulus_outer_px) & ~invalid
        n_ring = in_ring.sum(axis=(1, 2))
        enough = n_ring >= cfg.min_annulus_px
        ring = np.where(in_ring, values, np.float32(np.nan)).reshape(in_ring.shape[0], -1)
        ring[~enough] = 0.0  # keeps the statistics below free of empty rows
        level, scatter = _clipped_ring(ring, cfg)
        weight = np.clip(cfg.aperture_px + 0.5 - distance, 0.0, 1.0).astype(np.float32)
        clean = ~np.any(invalid & (weight > 0.0), axis=(1, 2))
        light = (weight * np.where(invalid, np.float32(0.0), values - level[:, None, None])).sum(
            axis=(1, 2), dtype=np.float64
        )
        n_aperture = weight.sum(axis=(1, 2))
        flux_e = light * e_per_adu
        noise_e = scatter * e_per_adu
        variance = np.maximum(flux_e, 0.0) + n_aperture * noise_e**2 * (
            1.0 + n_aperture / np.maximum(n_ring, 1)
        )
        good = enough & clean & np.isfinite(flux_e) & (n_aperture > 0)
        flux[part] = np.where(good, flux_e, np.nan)
        error[part] = np.where(good, np.sqrt(variance), np.nan)
        background[part] = np.where(good, level, np.nan)
        noise[part] = np.where(good, noise_e, np.nan)
        area[part] = np.where(good, n_aperture, np.nan)
        ok[part] = good
    return AperturePhotometry(flux, error, background, noise, area, ok)


def _row_median(rows: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """The median of each row, ignoring NaN. A row with no values gives 0.

    NumPy sorts NaN last, so the middle of the valid entries is the median. A sort of the whole
    block is much faster than `nanmedian`, which loops over the rows in Python.
    """
    ordered = np.sort(rows, axis=1)
    count = np.sum(~np.isnan(rows), axis=1)
    index = np.arange(rows.shape[0])
    low = ordered[index, np.maximum((count - 1) // 2, 0)]
    high = ordered[index, np.maximum(count // 2, 0)]
    return np.asarray(np.where(count > 0, 0.5 * (low + high), 0.0), dtype=np.float32)


def _clipped_ring(
    ring: npt.NDArray[np.float32], cfg: PhotometryOptions
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """The mean and the standard deviation of each row after sigma clipping. NaN entries skip."""
    center = _row_median(ring)
    spread = np.float32(1.4826) * _row_median(np.abs(ring - center[:, None]))
    level = center
    sigma = spread
    for _ in range(cfg.clip_iterations):
        limit = np.float32(cfg.clip_sigma) * np.maximum(sigma, np.float32(1e-6))
        keep = np.abs(ring - level[:, None]) <= limit[:, None]  # NaN compares false
        count = np.maximum(keep.sum(axis=1), 1).astype(np.float32)
        kept = np.where(keep, ring, np.float32(0.0))
        level = kept.sum(axis=1, dtype=np.float64).astype(np.float32) / count
        deviation = np.where(keep, ring - level[:, None], np.float32(0.0))
        sigma = np.sqrt((deviation**2).sum(axis=1, dtype=np.float64) / count).astype(np.float32)
    return level, sigma


def isolated(
    x: FloatArray,
    y: FloatArray,
    flux: FloatArray,
    trail_length_px: FloatArray,
    all_x: FloatArray,
    all_y: FloatArray,
    all_flux: FloatArray,
    options: PhotometryOptions | None = None,
) -> BoolArray:
    """For each star, whether no other detection that is bright enough lies within its aperture.

    A neighbor matters when it lies within the aperture radius plus a margin and a share of the
    trail, and it holds more than `isolation_flux_ratio` of this star's flux. The star itself
    appears in `all_x`, `all_y`, and `all_flux`, and the function ignores it by its position.
    """
    cfg = options or PhotometryOptions()
    result = np.ones(x.size, dtype=np.bool_)
    if x.size == 0 or all_x.size == 0:
        return result
    reach = cfg.aperture_px + cfg.isolation_margin_px + 0.5 * float(np.max(trail_length_px))
    star, other = _scipy.pair_indices(
        np.column_stack([x, y]), np.column_stack([all_x, all_y]), reach + 0.5
    )
    separation = np.hypot(all_x[other] - x[star], all_y[other] - y[star])
    limit = cfg.aperture_px + cfg.isolation_margin_px + 0.5 * trail_length_px[star]
    bright = all_flux[other] > cfg.isolation_flux_ratio * np.maximum(flux[star], 1e-9)
    # A separation under 0.25 pixel is the star itself.
    result[star[(separation >= 0.25) & (separation <= limit) & bright]] = False
    return result


# --- The matched stars of a frame ----------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class StarPhotometry:
    """The photometry of the matched stars of one frame. Every array has one entry per star.

    `index` is the position of the star in the `Detections`, and `cat_row` its catalog row.
    `rate_e_per_s` is the flux in electrons per second, corrected for the flat field and the
    aperture loss (`aperture_correction` is the factor, from `n_growth_stars` stars). `mag_error`
    is the photometric error in magnitudes, and `snr` the signal-to-noise ratio of the flux.
    """

    index: IntArray
    cat_row: IntArray
    rate_e_per_s: FloatArray
    mag_error: FloatArray
    snr: FloatArray
    x: FloatArray
    y: FloatArray
    aperture_correction: float = 1.0
    n_growth_stars: int = 0

    def __len__(self) -> int:
        return int(self.index.size)


def measure_matched_stars(
    data: npt.NDArray[np.float32],
    detections: Detections,
    cat_row: IntArray,
    *,
    exposure_s: float,
    e_per_adu: float,
    saturation_dn: float,
    origin_px: tuple[float, float] = (0.0, 0.0),
    options: PhotometryOptions | None = None,
    bad: BoolArray | None = None,
    flat_at: FloatArray | None = None,
    min_snr: float = 0.0,
) -> StarPhotometry:
    """Photometry of the detections that matched a catalog star and are fit to measure.

    The stars that the detector flagged as saturated, at the edge, hot, blended, streaks, or
    fitted by moments stay out, and so do stars with a neighbor (see `isolated`) or a ring that
    is not clean. `origin_px` is the position of the data array in the sensor, because
    `Detections` holds sensor positions. `flat_at` gives the flat-field value at each detection
    (1 for a unit flat), and `min_snr` drops the stars with a lower signal-to-noise ratio.
    """
    cfg = options or PhotometryOptions()
    candidates = np.flatnonzero((cat_row >= 0) & detections.reliable(UNRELIABLE))
    picked = detections.select(candidates)
    keep = isolated(
        picked.x,
        picked.y,
        picked.flux,
        picked.trail_length_px,
        detections.x,
        detections.y,
        detections.flux,
        cfg,
    )
    candidates = candidates[keep]
    picked = detections.select(candidates)
    result = aperture_photometry(
        data,
        picked.x - origin_px[0],
        picked.y - origin_px[1],
        picked.trail_length_px,
        picked.trail_angle_rad,
        e_per_adu=e_per_adu,
        saturation_dn=saturation_dn,
        options=cfg,
        bad=bad,
    )
    scale = np.ones(candidates.size) if flat_at is None else np.asarray(flat_at)[candidates]
    flux = result.flux_e / scale
    error = result.error_e / scale
    good = result.ok & (flux > 0.0) & (error > 0.0)
    snr = np.where(good, flux / np.where(good, error, 1.0), 0.0)
    good &= snr >= min_snr
    correction, n_growth = 1.0, 0
    if cfg.growth_stars > 0:
        correction, n_growth = aperture_correction(
            data,
            picked,
            detections,
            flux,
            np.where(good, snr, 0.0),
            e_per_adu=e_per_adu,
            saturation_dn=saturation_dn,
            origin_px=origin_px,
            options=cfg,
            bad=bad,
            flat_scale=scale,
        )
    return StarPhotometry(
        index=candidates[good],
        cat_row=cat_row[candidates[good]],
        rate_e_per_s=flux[good] * correction / exposure_s,
        mag_error=_MAG_PER_E / snr[good],
        snr=snr[good],
        x=picked.x[good],
        y=picked.y[good],
        aperture_correction=correction,
        n_growth_stars=n_growth,
    )


def aperture_correction(
    data: npt.NDArray[np.float32],
    picked: Detections,
    detections: Detections,
    narrow_flux_e: FloatArray,
    snr: FloatArray,
    *,
    e_per_adu: float,
    saturation_dn: float,
    origin_px: tuple[float, float],
    options: PhotometryOptions,
    bad: BoolArray | None,
    flat_scale: FloatArray,
) -> tuple[float, int]:
    """The factor from the narrow aperture to the wide one, and the number of stars behind it.

    The brightest `growth_stars` stars with a signal-to-noise ratio of at least
    `growth_min_snr` get a second measurement in the wide aperture, and the median of the flux
    ratios is the factor. Fewer than `growth_min_stars` usable stars give 1.0. The factor stays
    between 1 (the wide aperture cannot hold less light) and `max_aperture_correction`.
    """
    pool = np.flatnonzero(snr >= options.growth_min_snr)
    if pool.size < options.growth_min_stars:
        return 1.0, 0
    pool = pool[np.argsort(-snr[pool], kind="stable")][: options.growth_stars]
    wide = replace(
        options,
        aperture_px=options.growth_aperture_px,
        annulus_inner_px=options.growth_aperture_px + 4.0,
        annulus_outer_px=options.growth_aperture_px + 10.0,
        min_annulus_px=2 * options.min_annulus_px,
        growth_stars=0,
    )
    stars = picked.select(pool)
    isolated_wide = isolated(
        stars.x,
        stars.y,
        stars.flux,
        stars.trail_length_px,
        detections.x,
        detections.y,
        detections.flux,
        wide,
    )
    pool = pool[isolated_wide]
    stars = picked.select(pool)
    result = aperture_photometry(
        data,
        stars.x - origin_px[0],
        stars.y - origin_px[1],
        stars.trail_length_px,
        stars.trail_angle_rad,
        e_per_adu=e_per_adu,
        saturation_dn=saturation_dn,
        options=wide,
        bad=bad,
    )
    wide_flux = result.flux_e / flat_scale[pool]
    usable = result.ok & (wide_flux > 0.0) & (narrow_flux_e[pool] > 0.0)
    if int(usable.sum()) < options.growth_min_stars:
        return 1.0, 0
    ratio = wide_flux[usable] / narrow_flux_e[pool][usable]
    factor = float(np.median(ratio))
    return min(max(factor, 1.0), options.max_aperture_correction), int(ratio.size)
