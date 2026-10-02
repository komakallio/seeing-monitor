"""What a flat field looks like: the measures that both flat builders report.

`seeingmon flat make` (a flat from panel frames) and `seeingmon flat build` (a flat from the night
sky) print the same description of the image that they make. This module measures it, so the two
commands agree on what a vignetting profile, a shadow, and a tilt are.

**Binning.** The measures work on an image that is binned by `factor` in both directions (a block
mean), because a 4144 x 2822 flat is large and the structures of interest span many pixels. A
binned pixel `i` covers the sensor pixels `i * factor` to `(i + 1) * factor - 1`, so its center sits
at the sensor coordinate `i * factor + (factor - 1) / 2`. Positions in the results are sensor
pixels, where the center of the first pixel is (0, 0), as everywhere else in the survey path.

**The three parts.** `decompose` splits an image into the parts that the report talks about:

- the *radial part*, the azimuthal mean about the optical center as a function of radius, which is
  the vignetting of the optics;
- the *rest*, a smooth image (a Gaussian of `high_pass_px` binned pixels) that holds the tilt and
  any gradient of the light source, found after the radial part is divided out;
- the *fine part*, the image over the radial part and the rest, which holds the shadows of dust
  and the pixel pattern.

The radial part is the azimuthal mean of the image itself, not of a smoothed copy, so the
smoothing cannot bias it at the frame edge, where the vignetting is steepest.

**Reading the numbers.** A frame is point-symmetric about its center, and a plane is odd, so the
azimuthal mean of a plane is zero when the optical center is the center of the frame. The radial
part and the tilt therefore do not leak into each other. A center that you move off the middle of
the frame breaks that symmetry a little at the largest radii.

**Shadows and edge artifacts.** A shadow is a dip of the fine part that is deeper than 1%. A dip
whose center lies within 20 sensor pixels of an edge is an *edge artifact*, and the report lists
it apart, because the smoothing that finds the fine part is least certain there.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from seeingmon.survey import _scipy

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# The radii where the report gives the vignetting, in degrees from the optical center.
RADII_DEG: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)
# A shadow counts when it is at least this deep, as a fraction of the flat.
SHADOW_MIN_DEPTH = 0.01
# A dip whose center lies this close to an edge is an edge artifact, in sensor pixels.
EDGE_MARGIN_PX = 20.0
# The most shadows that a report lists. The rest are counted.
MAX_LISTED_SHADOWS = 12
# The radius of the disk that gives the value at the center, in binned pixels.
_CENTER_RADIUS_BINS = 4.0
# The width of the ring that gives the value in the corners, in binned pixels.
_CORNER_RING_BINS = 2.0
# The fewest pixels that a ring needs to give a value.
_MIN_RING_PIXELS = 30
_TINY = 1e-9


# --- Binning and geometry ------------------------------------------------------------------


def block_sum(image: npt.NDArray[Any], factor: int) -> FloatArray:
    """The sum of each `factor` x `factor` block. Rows and columns that do not fill a block drop."""
    if factor < 1:
        raise ValueError("the binning factor must be at least 1")
    height, width = image.shape
    rows, columns = height // factor, width // factor
    if rows == 0 or columns == 0:
        raise ValueError("the image is smaller than one block")
    view = image[: rows * factor, : columns * factor].reshape(rows, factor, columns, factor)
    return np.asarray(view.sum(axis=(1, 3), dtype=np.float64), dtype=np.float64)


def block_mean(image: npt.NDArray[Any], factor: int) -> FloatArray:
    """The mean of each `factor` x `factor` block."""
    return block_sum(image, factor) / float(factor * factor)


def binned_position(sensor_xy: tuple[float, float], factor: int) -> tuple[float, float]:
    """The position in binned pixels of a sensor position (the center of a block is its middle)."""
    offset = (factor - 1) / 2.0
    return (sensor_xy[0] - offset) / factor, (sensor_xy[1] - offset) / factor


def sensor_position(binned_xy: tuple[float, float], factor: int) -> tuple[float, float]:
    """The sensor position of a position in binned pixels."""
    offset = (factor - 1) / 2.0
    return binned_xy[0] * factor + offset, binned_xy[1] * factor + offset


def radius_map(shape: tuple[int, int], center_xy: tuple[float, float]) -> FloatArray:
    """The distance of every pixel from `center_xy`, in the pixels of the image."""
    height, width = shape
    x = np.arange(width, dtype=np.float64) - center_xy[0]
    y = np.arange(height, dtype=np.float64) - center_xy[1]
    return np.asarray(np.hypot(y[:, None], x[None, :]), dtype=np.float64)


def upsample_bilinear(
    binned: FloatArray, factor: int, shape: tuple[int, int]
) -> npt.NDArray[np.float32]:
    """Spread a binned image over the full sensor with bilinear interpolation.

    The centers of the blocks are the sample points. A pixel beyond the first or the last center
    takes the value of that edge row or column. The result has `shape` and the type `float32`.
    """
    height, width = shape
    rows, columns = binned.shape

    def axis(size: int, count: int) -> tuple[npt.NDArray[np.intp], FloatArray]:
        position = (np.arange(size, dtype=np.float64) + 0.5) / factor - 0.5
        position = np.clip(position, 0.0, count - 1.0)
        low = np.minimum(np.floor(position).astype(np.intp), max(count - 2, 0))
        return low, np.asarray(position - low, dtype=np.float64)

    y0, wy = axis(height, rows)
    x0, wx = axis(width, columns)
    y1 = np.minimum(y0 + 1, rows - 1)
    x1 = np.minimum(x0 + 1, columns - 1)
    out = np.empty(shape, dtype=np.float32)
    for start in range(0, height, 256):  # a strip at a time keeps the temporaries small
        stop = min(start + 256, height)
        top = binned[y0[start:stop]]
        bottom = binned[y1[start:stop]]
        mixed = top + (bottom - top) * wy[start:stop, None]
        out[start:stop] = (mixed[:, x0] + (mixed[:, x1] - mixed[:, x0]) * wx[None, :]).astype(
            np.float32
        )
    return out


# --- The radial part ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RadialProfile:
    """The azimuthal mean of an image in rings one pixel wide.

    `radius` is the mean radius of the pixels of each ring, `value` their mean, and `count` their
    number. A ring without a valid pixel is not in the profile.
    """

    radius: FloatArray
    value: FloatArray
    count: FloatArray


def azimuthal_profile(
    image: FloatArray, radius: FloatArray, *, valid: BoolArray | None = None
) -> RadialProfile:
    """The mean of the valid pixels in rings of one pixel around the center of `radius`."""
    index = np.floor(radius).astype(np.intp).ravel()
    weight = np.ones(image.size) if valid is None else valid.astype(np.float64).ravel()
    flat_image = np.where(weight > 0, image.ravel(), 0.0)
    size = int(index.max()) + 1
    count = np.bincount(index, weights=weight, minlength=size)
    total = np.bincount(index, weights=flat_image, minlength=size)
    radius_total = np.bincount(index, weights=radius.ravel() * weight, minlength=size)
    keep = count > 0
    return RadialProfile(
        radius=radius_total[keep] / count[keep],
        value=total[keep] / count[keep],
        count=count[keep],
    )


def profile_map(profile: RadialProfile, radius: FloatArray) -> FloatArray:
    """The profile at the radius of every pixel, by linear interpolation."""
    return np.asarray(np.interp(radius, profile.radius, profile.value), dtype=np.float64)


# --- The decomposition ----------------------------------------------------------------------


def normalized_gaussian(values: FloatArray, valid: BoolArray | None, sigma: float) -> FloatArray:
    """A Gaussian smoothing that counts only the valid pixels, and so ignores the invalid ones."""
    if valid is None:
        return _scipy.gaussian_filter(values, sigma)
    weight = valid.astype(np.float64)
    numerator = _scipy.gaussian_filter(np.where(valid, values, 0.0), sigma)
    denominator = _scipy.gaussian_filter(weight, sigma)
    return np.asarray(numerator / np.maximum(denominator, _TINY), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Decomposition:
    """An image split into its radial part, its smooth rest, and its fine part.

    `radial_map` is the radial part at every pixel, and `rest` the smooth part of what is left.
    The three multiply back to the image: `image = radial_map * rest * fine` on the valid
    pixels. On an invalid pixel, `fine` is 1.
    """

    radial: RadialProfile
    radial_map: FloatArray
    rest: FloatArray
    fine: FloatArray


def decompose(
    image: FloatArray,
    *,
    center_xy: tuple[float, float],
    high_pass_px: float,
    valid: BoolArray | None = None,
) -> Decomposition:
    """Split an image into the radial part, the smooth rest, and the fine part.

    `center_xy` is the optical center in the pixels of `image`, and `high_pass_px` the width of
    the Gaussian (in the same pixels) that separates the smooth rest from the fine part. The
    image must be positive where it is valid.
    """
    if high_pass_px <= 0:
        raise ValueError("the high-pass width must be positive")
    radius = radius_map(image.shape, center_xy)
    profile = azimuthal_profile(image, radius, valid=valid)
    if profile.radius.size < 2:
        raise ValueError("the image has no valid pixels to measure")
    radial_map = np.maximum(profile_map(profile, radius), _TINY)
    ratio = image / radial_map
    rest = np.maximum(normalized_gaussian(ratio, valid, high_pass_px), _TINY)
    fine = ratio / rest
    if valid is not None:
        fine = np.where(valid, fine, 1.0)
    return Decomposition(profile, radial_map, rest, fine)


# --- The measures ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfilePoint:
    """The flat at one radius, as a change against the center.

    `change_percent` is `None` when the radius lies off the frame. `corner` marks the point that
    gives the corners of the frame, whose radius is the distance to the farthest corner.
    """

    radius_deg: float
    change_percent: float | None
    corner: bool = False


def vignetting_profile(
    image: FloatArray,
    *,
    factor: int,
    scale_arcsec_px: float,
    center_xy: tuple[float, float],
    radii_deg: Sequence[float] = RADII_DEG,
    valid: BoolArray | None = None,
) -> tuple[ProfilePoint, ...]:
    """The azimuthal mean at each radius and in the corners, in percent against the center.

    `image` is binned by `factor`, `scale_arcsec_px` is the plate scale of the sensor pixels, and
    `center_xy` is the optical center in binned pixels. A radius whose ring holds fewer than 30
    valid pixels (a radius beyond the corners) gives `None`. The last point is the corners: the
    pixels within 2 binned pixels of the farthest pixel center from the optical center.
    """
    radius = radius_map(image.shape, center_xy)
    weight = np.ones(image.shape) if valid is None else valid.astype(np.float64)
    inner = (radius <= _CENTER_RADIUS_BINS) & (weight > 0)
    if int(inner.sum()) < 4:
        raise ValueError("the center of the image holds no valid pixel")
    center_value = float(image[inner].mean())
    points: list[ProfilePoint] = []
    for degrees in radii_deg:
        r_bins = degrees * 3600.0 / scale_arcsec_px / factor
        half_width = max(1.5, 0.02 * r_bins)
        ring = (np.abs(radius - r_bins) <= half_width) & (weight > 0)
        points.append(_ring_point(degrees, image, ring, center_value))
    height, width = image.shape
    corner_bins = max(
        math.hypot(x - center_xy[0], y - center_xy[1])
        for x in (0.0, width - 1.0)
        for y in (0.0, height - 1.0)
    )
    corner_ring = (radius >= corner_bins - _CORNER_RING_BINS) & (weight > 0)
    corner_deg = corner_bins * factor * scale_arcsec_px / 3600.0
    corner = _ring_point(corner_deg, image, corner_ring, center_value, minimum=4)
    points.append(ProfilePoint(corner.radius_deg, corner.change_percent, corner=True))
    return tuple(points)


def _ring_point(
    degrees: float,
    image: FloatArray,
    ring: BoolArray,
    center_value: float,
    *,
    minimum: int = _MIN_RING_PIXELS,
) -> ProfilePoint:
    if int(ring.sum()) < minimum or center_value <= 0:
        return ProfilePoint(degrees, None)
    return ProfilePoint(degrees, 100.0 * (float(image[ring].mean()) / center_value - 1.0))


@dataclass(frozen=True, slots=True)
class Tilt:
    """The plane across the frame, as the change from one edge to the other, in percent.

    `width_percent` is the change from the left edge to the right edge, and `height_percent`
    the change from the top edge to the bottom edge. A positive value means that the flat rises
    toward the right (or the bottom) edge.
    """

    width_percent: float
    height_percent: float


def fit_tilt(
    image: FloatArray, *, center_xy: tuple[float, float], valid: BoolArray | None = None
) -> Tilt:
    """Fit a plane to the image by least squares and give its slope as a change across the frame."""
    height, width = image.shape
    x = (np.arange(width, dtype=np.float64) - center_xy[0])[None, :] * np.ones((height, 1))
    y = (np.arange(height, dtype=np.float64) - center_xy[1])[:, None] * np.ones((1, width))
    weight = np.ones(image.shape) if valid is None else valid.astype(np.float64)
    v = np.where(weight > 0, image, 0.0)
    w = weight
    normal = np.array(
        [
            [w.sum(), (w * x).sum(), (w * y).sum()],
            [(w * x).sum(), (w * x * x).sum(), (w * x * y).sum()],
            [(w * y).sum(), (w * x * y).sum(), (w * y * y).sum()],
        ]
    )
    rhs = np.array([v.sum(), (v * x).sum(), (v * y).sum()])
    level, slope_x, slope_y = np.linalg.solve(normal, rhs)
    if level <= 0:
        raise ValueError("the image has no positive level")
    return Tilt(
        width_percent=float(100.0 * slope_x * width / level),
        height_percent=float(100.0 * slope_y * height / level),
    )


def tilt_after_radial(
    image: FloatArray,
    parts: Decomposition,
    *,
    center_xy: tuple[float, float],
    valid: BoolArray | None = None,
) -> Tilt:
    """The plane of the image after its radial part is divided out.

    The plane then scales the vignetting at each radius, which is how a tilt of the optics and of
    the sensor acts on a frame.
    """
    return fit_tilt(image / parts.radial_map, center_xy=center_xy, valid=valid)


@dataclass(frozen=True, slots=True)
class TiltSplit:
    """The tilts of two sets split in two parts: what stays and what turns.

    Turn the light source by 180 degrees between two sets. The tilt of the optics and the sensor
    stays, and the gradient of the source flips. `stays` is half the sum of the two tilts, and
    `turns` is half the difference (first set minus second), which is the gradient of the source
    in the first set. The split holds only when you turned the source between the sets.
    """

    stays: Tilt
    turns: Tilt


def split_tilts(first: Tilt, second: Tilt) -> TiltSplit:
    """Half the sum and half the difference of the tilts of two sets."""
    return TiltSplit(
        stays=Tilt(
            (first.width_percent + second.width_percent) / 2.0,
            (first.height_percent + second.height_percent) / 2.0,
        ),
        turns=Tilt(
            (first.width_percent - second.width_percent) / 2.0,
            (first.height_percent - second.height_percent) / 2.0,
        ),
    )


@dataclass(frozen=True, slots=True)
class Shadow:
    """A dip in the fine part: where it is, how deep, and how wide.

    `x_px` and `y_px` are sensor pixels. `depth` is a fraction (0.032 is 3.2%), and `width_px` is
    the diameter of the circle that has the area of the pixels deeper than half of the depth,
    in sensor pixels. `at_edge` says that the center lies within 20 pixels of an edge.
    """

    x_px: int
    y_px: int
    depth: float
    width_px: float
    at_edge: bool = False


def find_shadows(
    fine: FloatArray,
    *,
    factor: int,
    sensor_shape: tuple[int, int],
    min_depth: float = SHADOW_MIN_DEPTH,
    edge_margin_px: float = EDGE_MARGIN_PX,
    smooth_sigma: float = 1.0,
    valid: BoolArray | None = None,
) -> tuple[Shadow, ...]:
    """The dips deeper than `min_depth` in a fine part, deepest first.

    A light Gaussian of `smooth_sigma` binned pixels takes the noise down first. A dip is a
    connected region where the fine part lies below `1 - min_depth / 2`, and it counts when its
    deepest point is below `1 - min_depth`. `sensor_shape` is the shape of the sensor in pixels,
    which tells how far a dip lies from an edge.
    """
    deficit = 1.0 - _scipy.gaussian_filter(fine, smooth_sigma)
    if valid is not None:
        deficit = np.where(valid, deficit, 0.0)
    labels, count = _scipy.label(deficit > min_depth / 2.0)
    if count == 0:
        return ()
    inside = labels > 0
    label_ids = labels[inside]
    depth_values = deficit[inside]
    peak = np.zeros(count + 1)
    np.maximum.at(peak, label_ids, depth_values)
    half = depth_values >= 0.5 * peak[label_ids]
    ys, xs = np.nonzero(inside)
    kept_labels = label_ids[half]
    kept_depth = depth_values[half]
    area = np.bincount(kept_labels, minlength=count + 1).astype(np.float64)
    weight = np.bincount(kept_labels, weights=kept_depth, minlength=count + 1)
    x_sum = np.bincount(kept_labels, weights=xs[half] * kept_depth, minlength=count + 1)
    y_sum = np.bincount(kept_labels, weights=ys[half] * kept_depth, minlength=count + 1)
    sensor_height, sensor_width = sensor_shape
    shadows: list[Shadow] = []
    for index in range(1, count + 1):
        if peak[index] < min_depth or area[index] == 0 or weight[index] <= 0:
            continue
        x_sensor, y_sensor = sensor_position(
            (float(x_sum[index] / weight[index]), float(y_sum[index] / weight[index])), factor
        )
        distance = min(
            x_sensor, sensor_width - 1 - x_sensor, y_sensor, sensor_height - 1 - y_sensor
        )
        shadows.append(
            Shadow(
                x_px=round(x_sensor),
                y_px=round(y_sensor),
                depth=float(peak[index]),
                width_px=2.0 * math.sqrt(area[index] / math.pi) * factor,
                at_edge=distance < edge_margin_px,
            )
        )
    shadows.sort(key=lambda shadow: -shadow.depth)
    return tuple(shadows)


@dataclass(frozen=True, slots=True)
class FlatSummary:
    """The description of a flat: the vignetting, the tilt, the shadows, and the edge artifacts."""

    profile: tuple[ProfilePoint, ...]
    tilt: Tilt
    shadows: tuple[Shadow, ...]
    edge_artifacts: tuple[Shadow, ...]


def summarize_flat(
    flat_binned: FloatArray,
    *,
    factor: int,
    sensor_shape: tuple[int, int],
    scale_arcsec_px: float,
    center_xy: tuple[float, float],
    high_pass_px: float,
    edge_margin_px: float = EDGE_MARGIN_PX,
    smooth_sigma: float = 1.0,
    valid: BoolArray | None = None,
) -> FlatSummary:
    """Describe a flat that is binned by `factor`. `center_xy` is in binned pixels.

    `smooth_sigma` is the width of the light smoothing that takes the noise out of the fine part
    before the shadow search. A flat with little noise can take a smaller one, which keeps more of
    the depth of a narrow shadow.
    """
    parts = decompose(flat_binned, center_xy=center_xy, high_pass_px=high_pass_px, valid=valid)
    found = find_shadows(
        parts.fine,
        factor=factor,
        sensor_shape=sensor_shape,
        edge_margin_px=edge_margin_px,
        smooth_sigma=smooth_sigma,
        valid=valid,
    )
    return FlatSummary(
        profile=vignetting_profile(
            flat_binned,
            factor=factor,
            scale_arcsec_px=scale_arcsec_px,
            center_xy=center_xy,
            valid=valid,
        ),
        tilt=tilt_after_radial(flat_binned, parts, center_xy=center_xy, valid=valid),
        shadows=tuple(shadow for shadow in found if not shadow.at_edge),
        edge_artifacts=tuple(shadow for shadow in found if shadow.at_edge),
    )


# --- Two sets of panel frames ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SetAgreement:
    """How well two flats of the same lens agree, which shows what the light source adds.

    The two flats divide, and the quotient splits into a smooth part and a fine part. `smooth_rms`
    and `fine_rms` are their rms in percent. `expected_fine_rms` is the rms that the noise of the
    two flats predicts for the fine part, or `None` when the noise is not known. `plane` is the
    tilt of the quotient: the gradient of the source, twice over when you turned it by 180 degrees
    between the sets.
    """

    smooth_rms: float
    fine_rms: float
    expected_fine_rms: float | None
    plane: Tilt


def compare_sets(
    first: FloatArray,
    second: FloatArray,
    *,
    factor: int,
    center_xy: tuple[float, float],
    high_pass_px: float,
    noise_first: float | None = None,
    noise_second: float | None = None,
) -> SetAgreement:
    """Compare two flats that are binned by `factor`: the first over the second.

    `noise_first` and `noise_second` are the noise of one sensor pixel of each flat, as fractions.
    Binning `factor` x `factor` pixels takes the noise of the quotient down by `factor`.
    """
    ratio = first / np.maximum(second, _TINY)
    ratio = ratio / float(np.median(ratio))
    smooth = normalized_gaussian(ratio, None, high_pass_px)
    fine = ratio / np.maximum(smooth, _TINY)
    expected: float | None = None
    if noise_first is not None and noise_second is not None:
        expected = 100.0 * math.hypot(noise_first, noise_second) / factor
    return SetAgreement(
        smooth_rms=float(100.0 * np.std(smooth)),
        fine_rms=float(100.0 * np.std(fine)),
        expected_fine_rms=expected,
        plane=fit_tilt(ratio, center_xy=center_xy),
    )


# --- Text -----------------------------------------------------------------------------------


def format_percent(value: float, *, digits: int = 1) -> str:
    """A signed percentage such as `-3.2%` or `+0.4%`. A tiny value does not show a sign of zero."""
    text = f"{value:+.{digits}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{digits}f}"
    return f"{text}%"


def profile_lines(
    points: Sequence[ProfilePoint],
    *,
    title: str = "Vignetting at each radius from the center, against the center:",
) -> list[str]:
    """The lines that give the vignetting at each radius and in the corners."""
    lines = [title]
    for point in points:
        value = (
            "outside the frame"
            if point.change_percent is None
            else format_percent(point.change_percent, digits=2)
        )
        name = (
            f"corners ({point.radius_deg:.2f} degrees)"
            if point.corner
            else (f"{point.radius_deg:.1f} degrees")
        )
        lines.append(f"  {name}: {value}")
    return lines


def shadow_lines(
    shadows: Sequence[Shadow],
    *,
    title: str = "Shadows deeper than 1%:",
    none: str = "none",
) -> list[str]:
    """The lines that list the shadows (position, depth, and width), and count the ones left out."""
    if not shadows:
        return [f"{title} {none}"]
    lines = [f"{title} {len(shadows)}"]
    lines.extend(
        f"  x {shadow.x_px}, y {shadow.y_px}: depth {100 * shadow.depth:.1f}%, "
        f"width {shadow.width_px:.0f} px"
        for shadow in shadows[:MAX_LISTED_SHADOWS]
    )
    if len(shadows) > MAX_LISTED_SHADOWS:
        lines.append(f"  and {len(shadows) - MAX_LISTED_SHADOWS} more")
    return lines


def edge_artifact_lines(
    shadows: Sequence[Shadow], *, margin_px: float = EDGE_MARGIN_PX
) -> list[str]:
    """The lines that list the dips whose center lies within `margin_px` pixels of an edge."""
    return shadow_lines(
        shadows,
        title=f"Edge artifacts deeper than 1% (center within {margin_px:.0f} px of an edge):",
    )


def tilt_text(tilt: Tilt) -> str:
    """A tilt as text: `+0.56% across the width, -0.34% across the height`."""
    return (
        f"{format_percent(tilt.width_percent, digits=2)} across the width, "
        f"{format_percent(tilt.height_percent, digits=2)} across the height"
    )


def tilt_line(tilt: Tilt, *, label: str = "Tilt after the radial part") -> str:
    """One line that gives the plane across the frame, and what its sign means."""
    return (
        f"{label}: {tilt_text(tilt)} (a positive value means that the flat rises toward the "
        "right edge or the bottom edge)."
    )


def agreement_lines(agreement: SetAgreement) -> list[str]:
    """The lines that say how well two sets agree, and the plane of their quotient."""
    fine = f"fine part {agreement.fine_rms:.2f}% rms"
    if agreement.expected_fine_rms is not None:
        fine += f" (the noise predicts {agreement.expected_fine_rms:.2f}% at this binning)"
    return [
        f"  smooth part {agreement.smooth_rms:.2f}% rms, {fine}",
        f"  plane of the quotient: {tilt_text(agreement.plane)}",
    ]
