"""Star detection: background, extraction, flags, and model centroids.

`detect_stars` turns a frame into a `Detections` object in four steps:

1. **Background.** `sep.Background` estimates the sky on a mesh (64 pixels by default) with
   sigma clipping, and the detector subtracts it. Pixels in the hot-pixel mask stay out of the
   estimate and the detection.
2. **Extraction.** `sep.extract` finds the connected pixels above the threshold in the frame
   smoothed with a 3 x 3 matched filter (a PSF of one pixel). Deblending is off, because a
   trail of several pixels has several local maxima and would split into pieces. A saturated
   star is one blob.
3. **Flags.** Each star gets flags for saturation, the frame edge, the hot-pixel mask, a trail,
   a neighbor, and a streak that the trail model does not explain.
4. **Model fit.** `seeingmon.survey.centroid.fit_stars` fits each unsaturated star with a
   Gaussian smeared along its trail, which gives an unbiased center, the PSF width across the
   trail, and the flux. The trail comes from the `TrailModel` (the pole position from the
   latest solution) or, with no solution, from the star's own second moments.

**Hot pixels.** Shape cannot tell an undersampled bin2 star from a hot pixel unless the
detector is careful: a star with the sharpest PSF that the 50 mm aperture allows (FWHM 0.66
pixel) still holds at most 86% of its flux in its brightest pixel, and the typical star holds
less. So the detector does not reject narrow sources. It takes a mask of the hot pixels that
the dark frames found (`hot_pixels`) and keeps those pixels out of the detection. A bright
source with more than `DetectOptions.spike_peak_fraction` of its flux in one pixel (a hot
pixel that the dark library has not seen yet) gets the flag `HOT_PIXEL`, which the pointing fit
ignores.

**Coordinates.** Pixel coordinates follow `StarList`: the center of the first pixel is (0, 0),
x runs along the columns, and y runs along the rows.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import sep

from seeingmon.solvers.base import StarList
from seeingmon.survey import _scipy
from seeingmon.survey.centroid import FWHM_PER_SIGMA, fit_stars
from seeingmon.survey.geometry import FloatArray
from seeingmon.survey.trail import TrailModel, trail_from_moments

# SEP stops when one object holds more pixels than its stack. A saturated star with a halo can
# hold hundreds of thousands.
sep.set_extract_pixstack(3_000_000)

BoolArray = npt.NDArray[np.bool_]

_MATCHED_KERNEL = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float32)
_PIXEL_VARIANCE = 1.0 / 12.0  # the variance of a uniform distribution across one pixel

# `paint_disks` paints stars in batches that share a window of one of these half-widths. A disk
# that reaches farther holds so many pixels that a loop of its own costs no more than a batch.
_BATCH_WINDOWS = (4, 8, 16, 32)
_BATCH_ELEMENTS = 1_000_000  # the most pixels that one batch of `paint_disks` examines
_BOX_KEYS = ("xmin", "xmax", "ymin", "ymax")


class DetectionError(Exception):
    """The detector could not process the frame."""


class StarFlag(enum.IntFlag):
    """Facts about a detected star. `Detections.flags` stores them as `uint16`."""

    NONE = 0
    SATURATED = 1  # the star has saturated pixels
    NEAR_EDGE = 2  # the center lies within the edge margin of the frame
    HOT_PIXEL = 4  # next to a masked pixel, or too narrow to be a star
    TRAILED = 8  # the trail is longer than `trail_flag_px`
    BLENDED = 16  # another detection lies inside the stamp
    STREAK = 32  # more elongated than the trail model allows (a satellite, an aircraft)
    MOMENTS_ONLY = 64  # the model fit did not apply or did not converge; the position is SEP's


# The flags that make a star a poor reference for the pointing fit.
UNRELIABLE = (
    StarFlag.SATURATED
    | StarFlag.NEAR_EDGE
    | StarFlag.HOT_PIXEL
    | StarFlag.BLENDED
    | StarFlag.STREAK
    | StarFlag.MOMENTS_ONLY
)


@dataclass(frozen=True, slots=True)
class DetectOptions:
    """Tunable parameters of the detector. The defaults suit bin2 frames of the reference camera."""

    threshold_sigma: float = 5.0  # detection threshold, in units of the local background rms
    min_pixels: int = 2  # the fewest connected pixels above the threshold
    mesh_px: int = 64  # the size of the background mesh
    filter_px: int = 3  # the size of the median filter on the background mesh
    saturation_fraction: float = 0.98  # of the saturation level counts as saturated
    edge_margin_px: float = 6.0
    max_stars: int = 3000  # the brightest stars that the detector keeps
    max_saturated_pixels: int = 6  # a star with more is not fitted, only measured by moments
    trail_flag_px: float = 1.5
    streak_ratio: float = 3.0  # a measured trail this many times the model's makes a streak
    streak_min_snr: float = 30.0
    spike_peak_fraction: float = 0.93  # the sharpest star holds at most 0.86 in its peak pixel
    spike_min_snr: float = 15.0
    fit_max_chi2: float = 6.0  # a fit with a larger reduced chi-square keeps the SEP position
    fit_max_shift_px: float = 2.0  # a fit that moves farther than this keeps the SEP position
    min_trail_px: float = 1.0  # a moment trail shorter than this counts as no trail

    def __post_init__(self) -> None:
        if self.threshold_sigma <= 0 or self.min_pixels < 1 or self.max_stars < 1:
            raise ValueError("invalid detector options")


@dataclass(frozen=True, slots=True, eq=False)
class Detections:
    """The stars of one frame, sorted by flux (the brightest first).

    All arrays have one entry for each star. Positions are in pixels of the frame the detector
    saw. `flux` is the total light above the background in the units of the data. `fwhm_px` is
    the width of the PSF across the trail. `elongation` is the ratio of the major to the minor
    second-moment axis, and `trail_length_px` and `trail_angle_rad` describe the trail that the
    fit used. `background` is the `sep.Background` object, which later steps (the sky quality)
    use for the background map.
    """

    shape: tuple[int, int]
    x: FloatArray
    y: FloatArray
    flux: FloatArray
    peak: FloatArray
    fwhm_px: FloatArray
    x_error_px: FloatArray
    y_error_px: FloatArray
    elongation: FloatArray
    trail_length_px: FloatArray
    trail_angle_rad: FloatArray
    flags: npt.NDArray[np.uint16]
    snr: FloatArray
    n_pixels: npt.NDArray[np.int32]
    background_level: float
    background_rms: float
    background: Any = None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def has(self, flag: StarFlag) -> BoolArray:
        """A mask of the stars that carry `flag`."""
        return np.asarray((self.flags & np.uint16(int(flag))) != 0, dtype=np.bool_)

    def reliable(self, exclude: StarFlag = UNRELIABLE) -> BoolArray:
        """A mask of the stars that have none of the flags in `exclude`."""
        return ~self.has(exclude)

    def select(self, mask: npt.ArrayLike) -> Detections:
        """A copy with only the stars where `mask` is true (or at the given indices)."""
        index = np.asarray(mask)
        return Detections(
            shape=self.shape,
            x=self.x[index],
            y=self.y[index],
            flux=self.flux[index],
            peak=self.peak[index],
            fwhm_px=self.fwhm_px[index],
            x_error_px=self.x_error_px[index],
            y_error_px=self.y_error_px[index],
            elongation=self.elongation[index],
            trail_length_px=self.trail_length_px[index],
            trail_angle_rad=self.trail_angle_rad[index],
            flags=self.flags[index],
            snr=self.snr[index],
            n_pixels=self.n_pixels[index],
            background_level=self.background_level,
            background_rms=self.background_rms,
            background=self.background,
        )

    def shifted(self, dx: float, dy: float) -> Detections:
        """A copy with the positions moved by `(dx, dy)`, for example by an ROI offset."""
        copy = self.select(np.arange(len(self)))
        return Detections(
            shape=copy.shape,
            x=copy.x + dx,
            y=copy.y + dy,
            flux=copy.flux,
            peak=copy.peak,
            fwhm_px=copy.fwhm_px,
            x_error_px=copy.x_error_px,
            y_error_px=copy.y_error_px,
            elongation=copy.elongation,
            trail_length_px=copy.trail_length_px,
            trail_angle_rad=copy.trail_angle_rad,
            flags=copy.flags,
            snr=copy.snr,
            n_pixels=copy.n_pixels,
            background_level=copy.background_level,
            background_rms=copy.background_rms,
            background=copy.background,
        )

    def star_list(self, exclude: StarFlag = StarFlag.HOT_PIXEL) -> StarList:
        """The stars for a plate solver, brightest first, without the stars that carry `exclude`."""
        keep = ~self.has(exclude)
        return StarList(x=self.x[keep], y=self.y[keep], flux=np.maximum(self.flux[keep], 1e-3))


def paint_disks(mask: BoolArray, x: FloatArray, y: FloatArray, radius: FloatArray) -> None:
    """Set the pixels of a disk around each position, in place. `mask` is `(height, width)`.

    A pixel is in the disk of a star when its center lies at most `radius` pixels from the
    position. A disk reaches `ceil(radius)` pixels from the nearest pixel of its center, and the
    parts outside the frame stay out. The radii must be finite.

    The function paints the stars of similar size together: it builds the pixels of a square
    window around each star, tests them all at once, and sets the pixels that pass. A disk that
    reaches more than the largest window (32 pixels) gets a loop of its own.
    """
    height, width = mask.shape
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    reach = np.ceil(radius).astype(np.intp)
    center_x = np.rint(x).astype(np.intp)
    center_y = np.rint(y).astype(np.intp)
    square = radius * radius
    batch = np.searchsorted(_BATCH_WINDOWS, reach, side="left")  # the smallest window that fits
    for index in np.unique(batch):
        members = np.flatnonzero(batch == index)
        if index == len(_BATCH_WINDOWS):
            for i in members:
                _paint_disk(mask, x[i], y[i], radius[i])
            continue
        half = _BATCH_WINDOWS[index]
        steps = np.arange(-half, half + 1)
        per_chunk = max(1, _BATCH_ELEMENTS // steps.size**2)
        for start in range(0, members.size, per_chunk):
            part = members[start : start + per_chunk]
            columns = center_x[part, None] + steps  # (stars, window)
            rows = center_y[part, None] + steps
            dx = columns - x[part, None]
            dy = rows - y[part, None]
            inside = (dy[:, :, None] ** 2 + dx[:, None, :] ** 2) <= square[part, None, None]
            inside &= ((rows >= 0) & (rows < height))[:, :, None]
            inside &= ((columns >= 0) & (columns < width))[:, None, :]
            star, row, column = np.nonzero(inside)
            mask[rows[star, row], columns[star, column]] = True


def _paint_disk(mask: BoolArray, x: float, y: float, radius: float) -> None:
    """Set the pixels of one disk. This is the loop that `paint_disks` batches."""
    height, width = mask.shape
    reach = int(np.ceil(radius))
    x0, x1 = max(round(x) - reach, 0), min(round(x) + reach + 1, width)
    y0, y1 = max(round(y) - reach, 0), min(round(y) + reach + 1, height)
    if x0 >= x1 or y0 >= y1:
        return
    gy, gx = np.ogrid[y0:y1, x0:x1]
    mask[y0:y1, x0:x1] |= (gx - x) ** 2 + (gy - y) ** 2 <= radius * radius


def star_mask(
    shape: tuple[int, int],
    detections: Detections,
    *,
    radius_scale: float = 3.0,
    minimum_px: float = 3.0,
) -> BoolArray:
    """A mask of the pixels that the stars cover, for sky measurements.

    Each star masks a disk around its center. The radius is `radius_scale` times the PSF width
    plus half the trail, and at least `minimum_px`. A saturated star masks more, in proportion
    to the square root of its flux.
    """
    mask = np.zeros(shape, dtype=np.bool_)
    saturated = detections.has(StarFlag.SATURATED)
    sigma = detections.fwhm_px / FWHM_PER_SIGMA
    radius = radius_scale * sigma + detections.trail_length_px / 2.0
    radius = np.maximum(radius, minimum_px)
    radius = np.where(np.isfinite(radius), radius, minimum_px)  # a NaN size must not fail a frame
    radius = np.where(
        saturated, np.maximum(radius, 2.0 * np.sqrt(np.maximum(detections.n_pixels, 1))), radius
    )
    paint_disks(mask, detections.x, detections.y, radius)
    return mask


def _bounding_box_any(mask: BoolArray, objects: Any) -> npt.NDArray[np.intp]:
    """For each object, how many pixels of `mask` lie inside its bounding box.

    `objects` gives the integer arrays `xmin`, `xmax`, `ymin`, and `ymax` (the limits are
    inclusive), as SEP's output does. The function lists the set pixels of the mask once, in
    row-major order, and counts the pixels of each row of each box with a binary search.
    """
    height, width = mask.shape
    xmin, xmax, ymin, ymax = (np.asarray(objects[key], dtype=np.intp) for key in _BOX_KEYS)
    count = int(xmin.size)
    keys = np.flatnonzero(mask)  # y * width + x of every set pixel, in ascending order
    if count == 0 or keys.size == 0:
        return np.zeros(count, dtype=np.intp)
    xmin = np.maximum(xmin, 0)
    xmax = np.minimum(xmax, width - 1)
    ymin = np.maximum(ymin, 0)
    ymax = np.minimum(ymax, height - 1)
    heights = np.where((xmin <= xmax) & (ymin <= ymax), ymax - ymin + 1, 0)
    owner = np.repeat(np.arange(count), heights)  # the box that each row belongs to
    first = np.cumsum(heights) - heights  # where the rows of each box start in `owner`
    y = ymin[owner] + (np.arange(owner.size) - first[owner])
    low = np.searchsorted(keys, y * width + xmin[owner], side="left")
    high = np.searchsorted(keys, y * width + xmax[owner], side="right")
    found = np.bincount(owner, weights=high - low, minlength=count)
    return np.asarray(found, dtype=np.intp)


def _blended_by_neighbors(x: FloatArray, y: FloatArray, reach: FloatArray) -> BoolArray:
    """Whether another star lies closer than the sum of the two stars' reaches.

    A star pulls the fit of its neighbor when it lies inside the neighbor's stamp, so two stars
    are blended when their distance is smaller than `reach[i] + reach[j]`, and both stars of such
    a pair get the flag. Most stars have a reach of a few pixels, and a few saturated blobs have
    tens of pixels. The function finds the close pairs of the small stars with one query, and
    asks of each blob which stars lie within its own reach and the largest reach.
    """
    count = int(x.size)
    blended = np.zeros(count, dtype=np.bool_)
    if count < 2:
        return blended
    points = np.column_stack([x, y])
    cutoff = max(2.5 * float(np.median(reach)), 8.0)
    small = np.flatnonzero(reach <= cutoff)
    wide = np.flatnonzero(reach > cutoff)
    if small.size > 1:
        radius = 2.0 * float(reach[small].max())
        pairs = _scipy.close_pairs(points[small], radius)
        first, second = small[pairs[:, 0]], small[pairs[:, 1]]
        close = np.hypot(x[first] - x[second], y[first] - y[second]) < reach[first] + reach[second]
        blended[first[close]] = True
        blended[second[close]] = True
    if wide.size:
        found = _scipy.within_radii(points, points[wide], reach[wide] + float(reach.max()))
        for i, candidates in zip(wide, found, strict=True):
            others = candidates[candidates != i]
            close = np.hypot(x[i] - x[others], y[i] - y[others]) < reach[i] + reach[others]
            if close.any():
                blended[i] = True
                blended[others[close]] = True
    return blended


def detect_stars(
    data: npt.NDArray[Any],
    *,
    saturation_dn: float,
    options: DetectOptions | None = None,
    e_per_adu: float = 1.0,
    hot_pixels: BoolArray | None = None,
    trail: TrailModel | None = None,
) -> Detections:
    """Find the stars in a frame and measure them.

    `data` is a 2-D frame in ADC counts (not the 16-bit container). `saturation_dn` is the
    count at which a pixel saturates. `e_per_adu` converts counts to electrons for the noise
    model. `hot_pixels` is a mask of known hot pixels, and `trail` is the trail model of the
    frame, if a pointing solution gives one.
    """
    opts = options or DetectOptions()
    if data.ndim != 2:
        raise DetectionError("the frame must be 2-D")
    frame = np.array(data, dtype=np.float32, order="C")  # a copy that the detector may change
    height, width = frame.shape
    if hot_pixels is not None and hot_pixels.shape != frame.shape:
        raise DetectionError("the hot-pixel mask must have the shape of the frame")
    if float(frame.max()) == float(frame.min()):
        raise DetectionError("the frame is constant, so it holds no signal")
    saturated = frame >= opts.saturation_fraction * saturation_dn

    try:
        background = sep.Background(
            frame,
            mask=hot_pixels,
            bw=opts.mesh_px,
            bh=opts.mesh_px,
            fw=opts.filter_px,
            fh=opts.filter_px,
        )
        level = float(background.globalback)
        rms = float(background.globalrms)
        rms_map = np.asarray(background.rms(), dtype=np.float32)
        background.subfrom(frame)
        varied = float(rms_map.max()) > 1.3 * max(float(rms_map.min()), 1e-6)
        objects = sep.extract(
            frame,
            opts.threshold_sigma,
            err=rms_map if varied else rms,
            minarea=opts.min_pixels,
            filter_kernel=_MATCHED_KERNEL,
            deblend_cont=1.0,
            clean=True,
            mask=hot_pixels,
        )
    except Exception as error:  # SEP raises a plain Exception for its internal limits
        raise DetectionError(f"the source extraction failed: {error}") from error
    if rms <= 0.0 or not np.isfinite(rms):
        raise DetectionError("the background noise is zero, so the frame holds no signal")

    keep = np.flatnonzero(
        (objects["flux"] > 0) & np.isfinite(objects["x"]) & np.isfinite(objects["y"])
    )
    objects = objects[keep]
    order = np.argsort(-objects["flux"], kind="stable")[: opts.max_stars]
    objects = objects[order]
    n = len(objects)

    x = np.asarray(objects["x"], dtype=np.float64)
    y = np.asarray(objects["y"], dtype=np.float64)
    flux = np.asarray(objects["flux"], dtype=np.float64)
    peak = np.asarray(objects["peak"], dtype=np.float64)
    major = np.asarray(objects["a"], dtype=np.float64)
    minor = np.maximum(np.asarray(objects["b"], dtype=np.float64), 1e-3)
    theta = np.asarray(objects["theta"], dtype=np.float64)
    n_pixels = np.asarray(objects["npix"], dtype=np.int32)
    # SEP gives a NaN shape to an object whose second moments a masked pixel leaves undefined. It
    # happened at first light, next to hot pixels of the dark library, and a NaN size then failed
    # the whole frame. Such an object gets the circle of its area and no angle.
    unmeasured = ~(np.isfinite(major) & np.isfinite(minor) & np.isfinite(theta))
    if unmeasured.any():
        circle = np.sqrt(np.maximum(n_pixels, 1) / np.pi)
        major = np.where(unmeasured, circle, major)
        minor = np.where(unmeasured, circle, minor)
        theta = np.where(unmeasured, 0.0, theta)
    elongation = major / minor

    flags = np.zeros(n, dtype=np.uint16)

    def mark(mask: BoolArray, flag: StarFlag) -> None:
        flags[mask] |= np.uint16(int(flag))

    margin = opts.edge_margin_px
    mark(
        (x < margin) | (x > width - 1 - margin) | (y < margin) | (y > height - 1 - margin),
        StarFlag.NEAR_EDGE,
    )
    saturated_pixels = _bounding_box_any(saturated, objects) if n else np.zeros(0, dtype=np.intp)
    mark(saturated_pixels > 0, StarFlag.SATURATED)
    if hot_pixels is not None and n:
        mark(_bounding_box_any(hot_pixels, objects) > 0, StarFlag.HOT_PIXEL)
    mark((np.asarray(objects["flag"]) & 3) != 0, StarFlag.BLENDED)

    # A bright source that has nearly all its flux in one pixel is a hot pixel.
    peak_fraction = peak / np.maximum(flux, 1e-9)

    # The trail that the fit assumes.
    moment_length, moment_angle = trail_from_moments(major, minor, theta)
    moment_length = np.where(moment_length < opts.min_trail_px, 0.0, moment_length)
    if trail is not None:
        trail_dx, trail_dy = trail.vectors(x, y)
        length = np.hypot(trail_dx, trail_dy)
        angle = trail.angle(x, y)
    else:
        length, angle = moment_length, moment_angle
        trail_dx, trail_dy = length * np.cos(angle), length * np.sin(angle)
    mark(length > opts.trail_flag_px, StarFlag.TRAILED)

    # The model fit, for stars that are not heavily saturated.
    sigma0 = np.sqrt(np.maximum(minor**2 - _PIXEL_VARIANCE, 0.09))
    fit_these = saturated_pixels <= opts.max_saturated_pixels
    fwhm = FWHM_PER_SIGMA * sigma0
    x_error = np.full(n, 0.5)
    y_error = np.full(n, 0.5)
    noise_at_star = rms_map[
        np.clip(np.rint(y).astype(np.intp), 0, height - 1),
        np.clip(np.rint(x).astype(np.intp), 0, width - 1),
    ].astype(np.float64)
    snr = flux / np.sqrt(
        np.maximum(flux, 0.0) / e_per_adu + np.maximum(n_pixels, 9) * noise_at_star**2
    )
    index = np.flatnonzero(fit_these)
    if index.size:
        bad = saturated if hot_pixels is None else (saturated | hot_pixels)
        fit = fit_stars(
            frame,
            x[index],
            y[index],
            flux[index],
            sigma0[index],
            trail_dx[index],
            trail_dy[index],
            noise_at_star[index],
            e_per_adu=e_per_adu,
            bad=bad,
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
        mark(np.isin(np.arange(n), index[~good]), StarFlag.MOMENTS_ONLY)
    mark(~fit_these, StarFlag.MOMENTS_ONLY)
    mark(
        (peak_fraction > opts.spike_peak_fraction) & (snr > opts.spike_min_snr),
        StarFlag.HOT_PIXEL,
    )

    if n:
        # A streak is much longer than the trail model allows, and it must be bright enough
        # for the moment estimate to mean something.
        if trail is not None:
            mark(
                (moment_length > opts.streak_ratio * length + 3.0) & (snr > opts.streak_min_snr),
                StarFlag.STREAK,
            )
        # A neighbor inside the stamp pulls the fit.
        reach = 3.0 * np.maximum(fwhm / FWHM_PER_SIGMA, 0.5) + length / 2.0 + 1.5
        mark(_blended_by_neighbors(x, y, reach), StarFlag.BLENDED)

    order = np.argsort(-flux, kind="stable")
    detections = Detections(
        shape=(height, width),
        x=x,
        y=y,
        flux=flux,
        peak=peak,
        fwhm_px=fwhm,
        x_error_px=x_error,
        y_error_px=y_error,
        elongation=elongation,
        trail_length_px=length,
        trail_angle_rad=angle,
        flags=flags,
        snr=snr,
        n_pixels=n_pixels,
        background_level=level,
        background_rms=rms,
        background=background,
    )
    return detections.select(order)
