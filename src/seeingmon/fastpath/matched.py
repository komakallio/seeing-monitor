"""The matched filter: the signal-to-noise ratio that decides whether the star is in a frame.

**Why a matched filter.** A star of flux `F` whose image puts the share `P_i` of its light on
pixel `i` is detected best by the sum of the pixels weighted by its own image, `S = sum(w (I -
b))` with `w = P`. On a sky that dominates the noise, with a pixel variance `v`, its SNR is
`F sqrt(sum P^2) / sqrt(v)`, the highest that any linear sum of the pixels reaches. An aperture
of area `A` that holds the share `f` of the star reaches `F f / sqrt(A v)`. The image of Polaris
in bin1 (the Airy pattern of a 50 mm aperture, 1.33 px FWHM) has `1 / sum(P^2)` of about 6
square pixels, so the filter reaches `sqrt(sum P^2) = 0.41` of the star per unit of noise, where
the fast path's centroid aperture (201 square pixels, 97% of the star) reaches
`0.97 / sqrt(201) = 0.068`. Against the sky alone, that is six times the SNR. With the star's
own photon noise, which the filter concentrates on a few pixels, it is five times: 41 against 8.3
in the daylight sky of the detection estimate (`docs/research-notes.md`, "Polaris in a bright
sky").

**The size of the image.** The filter gives the best SNR when its width matches the image, and a
filter that is too narrow loses much of it: a Gaussian filter of width `s_f` on a Gaussian image of
width `s_i` keeps `2 s_f s_i / (s_f^2 + s_i^2)` of the best SNR on a sky that dominates the noise.
The real image can be far wider than the diffraction limit: in the owner's recordings, `1 /
sum(P^2)` is 13 times that of the simulator's image. The fast path therefore keeps a bank of
widths (`[fastpath] matched_fwhm_airy_widths`), and a search frame keeps the one with the best SNR.

**The filter.** The weights are a Gaussian of the FWHM that the caller gives (the fast path uses
1, 2, and 4 Airy FWHM of the readout mode), integrated over each pixel and normalized to a total
of 1 over the plane, on a stamp of `2 h + 1` pixels with `h = ceil(3 sigma + 0.5)`. The table
holds the weights for 4 x 4 positions of the star within a pixel, a quarter of a pixel apart.

**The SNR.** For the weights `w` at one position, `S = sum(w (I - b))` in electrons, and the
variance of `S` is `sum(w^2 var_i)`, where `var_i` is the pixel variance `v` measured on the ROI
border plus the star's own photon noise `F P_i`. The pixels measure that noise themselves, because
`I_i - b` has the mean `F P_i`:

    SNR = S / sqrt(v sum(w^2) + max(sum(w^2 (I - b)), 0))

This holds for any image, also one that the filter does not match. A form that assumes the match,
`F sum(w^3)` with `F = S / sum(w^2)`, understates the photon noise of a filter wider than the
image by up to a third, and would let a wide filter of the bank report more than the best
weighting allows.

The noise of the background level `b` (the trimmed mean of a few hundred border pixels) adds
`1.1 sum(w)^2 v / n` to the variance, and the SNR leaves it out: about 1% of the variance for a
filter of the Airy FWHM, and 14% for one of 4 Airy FWHM, whose SNR therefore reads up to 7% high
in a bright sky. In a dark sky, where the star's photons dominate, a filter of the Airy FWHM gives
about `0.83 sqrt(F)` against the `sqrt(F)` of a wide aperture, which costs nothing at a detection
threshold of 10.

**Where the SNR is taken.** `near` evaluates the filter on a grid of 12 x 12 positions a quarter
of a pixel apart, the 3 x 3 pixels around a given pixel times the 16 positions within each pixel.
It takes the grid point where `S / sqrt(sum(w^2))` peaks, refines the position between the grid
points by a parabola, and returns the SNR of that grid point. `search` first finds the brightest
pixel of the image that the centered filter makes, within a radius of a point widened by the
half width of the filter, and then calls `near` there. A position beyond the radius is no star.
The kernel uses `near` with the first filter of the bank around the centroid for its missing-star
test, and a frame of a search burst runs `search` with each filter of the bank around the
predicted position and keeps the best.

**Cost.** `near` reads one box of `2 h + 3` pixels, one matrix product gives the 144 weighted
sums, and one dot product gives the photon noise of the best one. `search` filters only the
square around the circle, widened by `h`, as two matrix products with banded matrices, never the
whole frame. On the dev desktop, `near` adds about 55% to a frame of the kernel (about 12
microseconds on top of 22), and a frame of a search burst within 20 pixels takes about 0.2 ms
with the three filters, 3.5 times as long as with one (`search_us` of
`seeingmon.fastpath.benchmark`). `docs/research-notes.md` ("Polaris in a bright sky") scales them
to a Pi 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
FrameArray = npt.NDArray[np.integer[Any]] | npt.NDArray[np.floating[Any]]

PHASES = 4  # positions of the star within a pixel, along each axis
_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
_NEAR = 3  # the pixels around the center pixel that `near` evaluates, along each axis
_GRID = _NEAR * PHASES  # grid points along each axis


@dataclass(frozen=True, slots=True)
class MatchedFilter:
    """The weights of the matched filter for one star image.

    `fwhm_px` is the FWHM of the Gaussian, `half` the half width `h` of the stamp, and `row` the
    weights of a centered star along one axis, whose outer product with itself is the centered
    stamp. `sum_w2` holds `sum(w^2)` for each of the `PHASES**2` positions of the star within a
    pixel.

    The rest serves `near`. For a box of `2 h + 3` pixels around the center pixel, the weights
    of a star at the grid point `divmod(k, 12)` (row, column), a quarter of a pixel apart from
    `(-1.375, -1.375)` pixels to `(1.375, 1.375)` around the center of the center pixel, fill
    column `k` of a matrix of the shape `(box * box, 144)`. `near_score_t` holds that matrix with
    each column divided by the root of its sum of squares, so that one product gives
    `S / sqrt(sum(w^2))` at every grid point. `near_sum_w2` holds the sum of the squares of each
    column, and row `k` of `near_w2` the squares of column `k`, which weight the star's photon
    noise.
    """

    fwhm_px: float
    half: int
    row: FloatArray
    sum_w2: FloatArray
    near_score_t: FloatArray
    near_sum_w2: FloatArray
    near_w2: FloatArray

    @property
    def stamp_px(self) -> int:
        """The side of the stamp, in pixels."""
        return 2 * self.half + 1

    @property
    def box_px(self) -> int:
        """The side of the box that `near` reads, in pixels."""
        return self.stamp_px + _NEAR - 1

    @property
    def effective_area_px2(self) -> float:
        """`1 / sum(w^2)`, averaged over the positions within a pixel: the area of the aperture
        that has the same noise on a sky that dominates."""
        return 1.0 / float(np.mean(self.sum_w2))


class MatchedPeak(NamedTuple):
    """The best position of the filter: `x` and `y` in the pixels of the frame, the SNR, and the
    flux of the star that the filter measures, `S / sum(w^2)`, in electrons."""

    x: float
    y: float
    snr: float
    flux_e: float


def _pixel_masses(sigma: float, half: int, offset: float) -> FloatArray:
    """The share of a 1-D Gaussian at `offset` that falls in each pixel from `-half` to `half`."""
    scale = 1.0 / (math.sqrt(2.0) * sigma)
    edges = [0.5 * math.erf((u - 0.5 - offset) * scale) for u in range(-half, half + 2)]
    return np.diff(np.asarray(edges, dtype=np.float64))


@lru_cache(maxsize=8)
def matched_filter(fwhm_px: float) -> MatchedFilter:
    """The table of the matched filter for a Gaussian star image of `fwhm_px`."""
    if not 0.3 <= fwhm_px <= 20.0:
        raise ValueError("the FWHM of the matched filter must be between 0.3 and 20 pixels")
    sigma = fwhm_px / _FWHM_PER_SIGMA
    half = max(1, math.ceil(3.0 * sigma + 0.5))
    stamp = 2 * half + 1
    box = stamp + _NEAR - 1
    phases = (np.arange(PHASES, dtype=np.float64) + 0.5) / PHASES - 0.5
    masses = [_pixel_masses(sigma, half, float(p)) for p in phases]
    sum_w2 = [
        float(np.sum(np.outer(masses[iy], masses[ix]) ** 2))
        for iy in range(PHASES)
        for ix in range(PHASES)
    ]
    columns = np.zeros((_GRID * _GRID, box, box), dtype=np.float64)
    for gy in range(_GRID):
        cell_y, phase_y = divmod(gy, PHASES)
        for gx in range(_GRID):
            cell_x, phase_x = divmod(gx, PHASES)
            weights = np.outer(masses[phase_y], masses[phase_x])
            columns[gy * _GRID + gx, cell_y : cell_y + stamp, cell_x : cell_x + stamp] = weights
    flat = columns.reshape(_GRID * _GRID, box * box)
    near_sum_w2 = (flat**2).sum(axis=1)
    return MatchedFilter(
        fwhm_px=fwhm_px,
        half=half,
        row=_pixel_masses(sigma, half, 0.0),
        sum_w2=np.asarray(sum_w2, dtype=np.float64),
        near_score_t=np.ascontiguousarray((flat / np.sqrt(near_sum_w2)[:, None]).T),
        near_sum_w2=near_sum_w2,
        near_w2=np.ascontiguousarray(flat**2),
    )


def _excess(
    data: FrameArray, x0: int, y0: int, rows: int, columns: int, level: float
) -> FloatArray:
    """The pixels of a box at `(x0, y0)` minus `level`, as floats. Outside the frame, they are 0."""
    height, width = data.shape
    if x0 >= 0 and y0 >= 0 and x0 + columns <= width and y0 + rows <= height:
        box = data[y0 : y0 + rows, x0 : x0 + columns]
        return np.subtract(box, level, dtype=np.float64)
    out = np.zeros((rows, columns), dtype=np.float64)
    x_lo, x_hi = max(x0, 0), min(x0 + columns, width)
    y_lo, y_hi = max(y0, 0), min(y0 + rows, height)
    if x_lo < x_hi and y_lo < y_hi:
        inside = data[y_lo:y_hi, x_lo:x_hi]
        out[y_lo - y0 : y_hi - y0, x_lo - x0 : x_hi - x0] = np.subtract(
            inside, level, dtype=np.float64
        )
    return out


def near(
    data: FrameArray,
    ix: int,
    iy: int,
    matched: MatchedFilter,
    level_dn: float,
    e_per_dn: float,
    pixel_var_e2: float,
) -> MatchedPeak:
    """The best position of the filter within about 1.4 pixels of the pixel `(ix, iy)`.

    `level_dn` is the sky level that the filter subtracts, in the counts of the frame, `e_per_dn`
    the electrons of one count, and `pixel_var_e2` the variance of one pixel of sky, in electrons
    squared. Pixels outside the frame read as the sky level.

    The position is the grid point where `S / sqrt(sum(w^2))` peaks, the most likely position of
    a star of unknown flux on a sky of even noise, refined by a parabola through it and its
    neighbors. The SNR is the one of that grid point.
    """
    box = matched.box_px
    excess = _excess(data, ix - matched.half - 1, iy - matched.half - 1, box, box, level_dn)
    score = excess.reshape(-1) @ matched.near_score_t
    best = int(np.argmax(score))
    gy, gx = divmod(best, _GRID)
    top = float(score[best])
    x = ix + (gx + 0.5) / PHASES - 0.5 * _NEAR
    y = iy + (gy + 0.5) / PHASES - 0.5 * _NEAR
    if 0 < gx < _GRID - 1:
        x += _vertex(float(score[best - 1]), top, float(score[best + 1])) / PHASES
    if 0 < gy < _GRID - 1:
        y += _vertex(float(score[best - _GRID]), top, float(score[best + _GRID])) / PHASES
    sum_w2 = float(matched.near_sum_w2[best])
    signal_e = top * math.sqrt(sum_w2) * e_per_dn
    # The star's photon noise, `sum(w^2 F P)`, from the pixels themselves: `sum(w^2 (I - b))`.
    photon_e2 = float(excess.reshape(-1) @ matched.near_w2[best]) * e_per_dn
    variance = pixel_var_e2 * sum_w2 + max(photon_e2, 0.0)
    return MatchedPeak(x, y, signal_e / math.sqrt(variance), signal_e / sum_w2)


def _vertex(low: float, middle: float, high: float) -> float:
    """The offset of the top of a parabola through three points one step apart, in steps."""
    curvature = low - 2.0 * middle + high
    if not curvature < 0.0:
        return 0.0
    return max(-0.5, min(0.5, 0.5 * (low - high) / curvature))


@lru_cache(maxsize=32)
def _band(count: int, fwhm_px: float) -> FloatArray:
    """The banded matrix that filters `count + 2 h` pixels into `count` along one axis."""
    matched = matched_filter(fwhm_px)
    taps = matched.row
    band = np.zeros((count + 2 * matched.half, count), dtype=np.float64)
    for k, weight in enumerate(taps):
        band[np.arange(count) + k, np.arange(count)] = weight
    return band


def search(
    data: FrameArray,
    cx: float,
    cy: float,
    radius_px: float,
    matched: MatchedFilter,
    level_dn: float,
    e_per_dn: float,
    pixel_var_e2: float,
) -> MatchedPeak | None:
    """The star near `(cx, cy)`: the brightest place of the filtered image within `radius_px`.

    The image that the centered filter makes gives the brightest pixel, and `near` finds the best
    position around it. The filtered image covers the circle of `radius_px` around `(cx, cy)` (in
    the pixels of the frame) widened by the half width of the filter, because a star just beyond
    the circle brightens the filtered image inside it: a wide filter would otherwise report such a
    star at the edge of the circle. The result is `None` when that position lies beyond
    `radius_px`, or when no pixel of the frame lies in the widened circle. The arguments are those
    of `near`.
    """
    height, width = data.shape
    margin = matched.half  # how far the light of a star reaches into the filtered image
    reach = math.ceil(radius_px) + margin
    x_lo = max(round(cx) - reach, 0)
    x_hi = min(round(cx) + reach, width - 1)
    y_lo = max(round(cy) - reach, 0)
    y_hi = min(round(cy) + reach, height - 1)
    if x_lo > x_hi or y_lo > y_hi:
        return None
    columns = x_hi - x_lo + 1
    rows = y_hi - y_lo + 1
    half = matched.half
    excess = _excess(data, x_lo - half, y_lo - half, rows + 2 * half, columns + 2 * half, level_dn)
    filtered = _band(rows, matched.fwhm_px).T @ excess @ _band(columns, matched.fwhm_px)
    dx = np.arange(x_lo, x_hi + 1, dtype=np.float64) - cx
    dy = np.arange(y_lo, y_hi + 1, dtype=np.float64) - cy
    outer = radius_px + margin
    outside = dy[:, None] ** 2 + dx[None, :] ** 2 > outer * outer
    if outside.all():
        return None
    row, column = divmod(int(np.argmax(np.where(outside, -np.inf, filtered))), columns)
    peak = near(data, x_lo + column, y_lo + row, matched, level_dn, e_per_dn, pixel_var_e2)
    if (peak.x - cx) ** 2 + (peak.y - cy) ** 2 > radius_px * radius_px:
        return None
    return peak
