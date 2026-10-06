"""How completely a search of the detector finds a star of a given flux.

The cloud fraction counts the catalog stars that a clear sky shows and that detection missed. It
must expect only the stars that the search that ran can find, or a clear frame reads as cloudy.
`SearchModel` describes one search of `seeingmon.survey.detect` (a `SearchSpec`) for one star
image, and it gives the chance that the search finds a star of a given flux: its completeness.
`star_completeness` gives it for each star of a field, with the star's own trail and noise.

**The limit from the physics.** The noise of the sky sets how faint a star any search can find. A
fit of the star image finds a star at 5 sigma when its flux is 5 times the pixel noise times the
square root of the noise-equivalent area of the image: about 8 times the pixel noise for the
simulator's sharp image of 0.73 px FWHM in bin2, whose noise-equivalent area is 2.6 px. Summing
2 x 2 pixels keeps the light of such a star in one sum, while the noise of four pixels adds, so the
sum has half the SNR of a single pixel that holds all the light. The 2 x 2 sum happens in software,
after the readout, so it adds the read noise of four pixels as it adds their sky noise: the loss is
the same in a dark and in a bright sky.

**The method's own loss.** The detector does less than the fit. It needs `min_pixels` (2)
connected pixels of the searched frame above the threshold:

- *The full search* smooths with a 3 x 3 kernel. With one noise for the frame, SEP compares the
  smoothed image with 5 times the noise of one pixel, which is 13.3 times the noise of the smoothed
  image, so it finds 9 in 10 sharp stars at 40 times the pixel noise, 5 times the flux that the fit
  needs. With a noise map that varies, SEP compares a matched SNR with 5, and the full search finds
  9 in 10 at 17 times.
- *The binned search* counts `min_pixels` in sums. A sharp star in the middle of a 2 x 2 block puts
  almost no light into a second block, so without a trail the binned search finds half of the sharp
  stars at 60 times the pixel noise, 76% at 200 times, and 91% at 800 times. A trail spreads the
  star across blocks: with the 1.8 px trail of a 30 s frame near the pole, it finds 90 to 95% of
  them at 130 times, by the direction of the trail. A wide image fills the blocks, and binning
  loses little: at 2.5 px FWHM, the binned search finds 9 in 10 at 65 times and the full search at
  71 times.

**The model.** The star image is a Gaussian of the measured FWHM, integrated over each pixel and
smeared along the trail. For a grid of positions of the star within one pixel of the searched
frame, the model applies the search's linear step (the kernel, or the sum of each block) and takes
the `min_pixels`-th highest pixel of the result. That pixel passes the threshold with the chance
that the noise of the sky and of the star allow, and the completeness is the mean of that chance
over the positions. Against stars injected into noise anywhere within a block and found by
`detect_stars`, for FWHMs of 0.73 to 2.5 px, trails along a row, a diagonal, and at 30 degrees, and
a noise map that varies, the model states the share of stars found to within 0.04 above and 0.11
below (`tests/survey/test_completeness.py`). It understates on the steep part of the curve,
because the detector accepts any neighbor above the threshold, and the model takes the
second-highest pixel alone. The model leaves out the hot-pixel mask, the edge, and neighbors, and
it takes the `min_pixels` highest pixels as connected, which holds for a compact image.

**The light in the wings.** A real image has wings that a Gaussian lacks: the Airy rings of the
aperture, a seeing halo, or a defocus. The wings help a sharp star into a second block of the
binned search, so there the model errs low. They also hold light that lies below the threshold,
and the Gaussian of the fitted width holds only the rest. The flux that the model takes is
therefore the light in the fitted image, which the pipeline measures on the bright stars of each
frame (`seeingmon.survey.pipeline.light_in_core`): 0.91 of the light within 12 px for the
simulator's Airy image. A model that took all the light overstates the share of such stars that
the full search finds by 0.14 where it says 0.9.

**Each star with its own trail.** A star trails along the circle around the pole, by an amount that
grows with its distance from the pole, so the trails of one frame differ in length and direction.
Near the pole they are short, and the binned search loses those stars as it loses the stars of a
short frame. One model for the median trail of a frame would let the cloud fraction expect them:
with the pole in a 30 s frame of the full sensor, at the flux where that model gives 0.9, the stars
of the field give 0.79 on average, and only 28% of them reach 0.9. `star_completeness` therefore
gives each star the completeness of its own trail. The models come from a grid of trail lengths
(`TRAIL_NODES_PX`) and directions (0, 22.5, and 45 degrees from a pixel axis, because the pixels,
the blocks, and the kernel look the same after a turn of 90 degrees or a mirror about an axis or a
diagonal), and a star takes the least complete of the four models at the corners of its cell.
Against the model of the star's own trail, over trails of 0 to 7 px in every direction, the grid
states the limit of the full search 4% high in the median and at most 25% high, and that of the
binned search 5% high in the median and up to 45% high at short trails, where it changes fastest.
Where a trail longer than 5.8 px meets the blocks at about 28 degrees, the binned limit has a bump
between the corners, and the grid states it up to 6% low. A frame whose noise map varies
(`relative`) passes the noise of the map at each star, because SEP then thresholds each pixel by
its own noise. The FWHM rounds to a grid of 2% steps, so the next frames reuse the models.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.survey import _scipy
from seeingmon.survey.centroid import FWHM_PER_SIGMA
from seeingmon.survey.detect import SEARCH_KERNEL, SearchSpec
from seeingmon.survey.geometry import FloatArray

IntArray = npt.NDArray[np.intp]

# Positions of the star along each axis of one pixel of the full frame. A trail as long as a block
# of the binned search fits inside it only within about 0.1 px of the middle, so the grid must be
# finer than that.
_GRID_PER_PX = 8
_TRAIL_STEP_PX = 0.25  # the widest spacing of the points that smear the image along its trail
_MIN_FWHM_PX = 0.3  # a narrower measured image is a fit that went wrong, not a star
_LIMIT_PASSES = 3  # `limit_e` narrows the flux in passes over a grid of fluxes
_LIMIT_POINTS = 32  # the fluxes of one pass, evenly spaced in their logarithm
_MAX_FLUX_SIGMA = 1e9  # the brightest flux, in pixel noise, that `limit_e` searches

# The trail lengths of the grid of `star_completeness`, in pixels of the full frame: every 0.25 px
# up to 2 px, where the binned search changes fastest, then 20% apart. A longer trail takes the
# last node.
TRAIL_NODES_PX: FloatArray = np.concatenate([np.arange(0.0, 2.0, 0.25), 2.0 * 1.2 ** np.arange(24)])
# The directions of the grid, from a pixel axis. `fold_angle` brings every trail into this range.
ANGLE_NODES_RAD: FloatArray = np.array([0.0, math.pi / 8.0, math.pi / 4.0])
_FWHM_STEP = 1.02  # the ratio of neighboring FWHMs of the models that `star_completeness` builds


def _pixel_integrals(edges: FloatArray, centers: FloatArray, sigma: float) -> FloatArray:
    """The share of a 1-D Gaussian in each pixel. `centers` has any shape, `edges` one axis."""
    scaled = (edges - centers[..., None]) / (sigma * math.sqrt(2.0))
    cumulative = 0.5 * (1.0 + _scipy.erf(scaled))
    return np.asarray(np.diff(cumulative, axis=-1), dtype=np.float64)


def _shifted_sum(image: FloatArray, kernel: FloatArray) -> FloatArray:
    """The correlation of each image of the stack `image` with `kernel`, with zeros outside."""
    height, width = image.shape[-2:]
    reach_y, reach_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((0, 0), (reach_y, reach_y), (reach_x, reach_x)))
    total = np.zeros_like(image)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if kernel[i, j]:
                total += kernel[i, j] * padded[:, i : i + height, j : j + width]
    return total


def fold_angle(angle_rad: npt.ArrayLike) -> FloatArray:
    """The direction of a trail as an angle from 0 to 45 degrees from the nearest pixel axis.

    The pixels, the blocks of the binned search, and the kernel of the full search look the same
    after a turn of 90 degrees or a mirror about an axis or a diagonal, so a model depends on this
    angle alone. A trail has an axis, not a direction, so its sign does not matter either.
    """
    turned = np.mod(np.asarray(angle_rad, dtype=np.float64), math.pi / 2.0)
    return np.asarray(np.minimum(turned, math.pi / 2.0 - turned), dtype=np.float64)


def model_fwhm(fwhm_px: float) -> float:
    """The FWHM of the grid of 2% steps nearest to `fwhm_px`, so the models of frames repeat."""
    if not math.isfinite(fwhm_px):
        raise ValueError(f"the FWHM of the star image must be finite, not {fwhm_px}")
    steps = round(math.log(max(fwhm_px, _MIN_FWHM_PX) / _MIN_FWHM_PX) / math.log(_FWHM_STEP))
    return _MIN_FWHM_PX * _FWHM_STEP**steps


@dataclass(frozen=True, slots=True)
class SearchModel:
    """The completeness of one search for one star image. Build it with `SearchModel.build`.

    `response` holds, for each position of the star, the `min_pixels`-th highest pixel of the
    searched frame for a star of unit flux, in units of the noise of that pixel when the pixel
    noise is 1. `shot` holds the share of the variance of that pixel that the star adds per unit
    of flux, in the same units. `threshold` is the threshold in units of the noise of a searched
    pixel.
    """

    search: SearchSpec
    response: FloatArray
    shot: FloatArray
    threshold: float

    @classmethod
    def build(
        cls,
        search: SearchSpec,
        *,
        fwhm_px: float,
        trail_px: float = 0.0,
        trail_angle_rad: float = 0.0,
    ) -> SearchModel:
        """The model of `search` for a Gaussian image of `fwhm_px` with a trail of `trail_px`.

        Sizes are in pixels of the full frame, and the angle is measured from the x axis toward y.
        A width that is not a finite number raises `ValueError`.
        """
        if not math.isfinite(fwhm_px):
            raise ValueError(f"the FWHM of the star image must be finite, not {fwhm_px}")
        factor = search.factor
        sigma = max(fwhm_px, _MIN_FWHM_PX) / FWHM_PER_SIGMA
        trail = max(trail_px, 0.0) if math.isfinite(trail_px) else 0.0
        reach = math.ceil(4.0 * sigma + trail / 2.0) + 2
        size = factor * math.ceil((2 * reach + factor) / factor)  # whole blocks
        middle = size // 2
        # The positions of the star cover one pixel of the searched frame along each axis, so
        # together they cover every position of the star relative to the grid of that frame.
        grid = _GRID_PER_PX * factor
        steps = (np.arange(grid) + 0.5) * factor / grid - 0.5
        points = max(1, math.ceil(trail / min(_TRAIL_STEP_PX, 0.5 * sigma)) + 1)
        along = np.linspace(-trail / 2.0, trail / 2.0, points) if points > 1 else np.zeros(1)
        edges = np.arange(size + 1, dtype=np.float64) - 0.5
        # The image separates into a column profile and a row profile for each point of the trail,
        # so each axis needs `grid` positions, and a matrix product joins them: the image of the
        # star at the column position i and the row position j is the sum over the points of the
        # row profile times the column profile.
        share_x = _pixel_integrals(
            edges, middle + steps[:, None] + along * math.cos(trail_angle_rad), sigma
        )  # (grid, points, size)
        share_y = _pixel_integrals(
            edges, middle + steps[:, None] + along * math.sin(trail_angle_rad), sigma
        )
        rows = share_y.transpose(0, 2, 1).reshape(grid * size, points)
        columns = share_x.transpose(1, 0, 2).reshape(points, grid * size)
        image = (rows @ columns).reshape(grid, size, grid, size).transpose(0, 2, 1, 3)
        image = image.reshape(grid * grid, size, size) / points
        if search.binned:
            blocks = size // factor
            summed = image.reshape(-1, blocks, factor, blocks, factor).sum(axis=(2, 4))
            # A sum of factor^2 pixels has factor times the noise of one, and the star adds its
            # own light to the variance of the sum.
            response = summed / factor
            shot = summed / factor**2
            threshold = search.threshold_sigma
        else:
            kernel = np.asarray(SEARCH_KERNEL, dtype=np.float64)
            norm = float(np.sqrt(np.sum(kernel**2)))
            response = _shifted_sum(image, kernel) / norm
            shot = _shifted_sum(image, kernel**2) / norm**2
            gain = 1.0 if search.relative else float(np.sum(kernel)) / norm
            threshold = search.threshold_sigma * gain
        flat = response.reshape(response.shape[0], -1)
        rank = min(max(search.min_pixels, 1), flat.shape[1]) - 1
        order = np.argsort(-flat, axis=1, kind="stable")[:, rank]
        positions = np.arange(flat.shape[0])
        return cls(
            search=search,
            response=np.asarray(flat[positions, order], dtype=np.float64),
            shot=np.asarray(shot.reshape(flat.shape[0], -1)[positions, order], dtype=np.float64),
            threshold=threshold,
        )

    def completeness(self, flux_e: npt.ArrayLike, noise_e: npt.ArrayLike) -> FloatArray:
        """The chance that the search finds a star of `flux_e` electrons in each element.

        `noise_e` is the noise of one pixel of the full frame in electrons, without the star: one
        value, or one for each star. A noise that is not a positive number gives 0.
        """
        flux = np.asarray(flux_e, dtype=np.float64)
        noise = np.asarray(noise_e, dtype=np.float64)
        flux, noise = np.broadcast_arrays(flux, noise)
        valid = (noise > 0.0) & np.isfinite(noise)
        noise = np.where(valid, noise, 1.0)[..., None]
        signal = np.maximum(flux, 0.0)[..., None] / noise  # in units of the pixel noise
        spread = np.sqrt(1.0 + signal * self.shot / noise)
        score = (signal * self.response - self.threshold) / spread
        chance = 0.5 * (1.0 + _scipy.erf(score / math.sqrt(2.0)))
        return np.asarray(np.where(valid, np.mean(chance, axis=-1), 0.0), dtype=np.float64)

    def limit_e(self, completeness: float, noise_e: float) -> float:
        """The flux in electrons at which the search finds a star with the chance `completeness`.

        The completeness grows with the flux. Each pass evaluates it at 32 fluxes, evenly spaced in
        their logarithm, and keeps the step where it crosses, so three passes narrow the flux to
        0.1%, and a straight line across the last step gives it. It returns infinity when the
        search does not reach `completeness` below 10^9 times the pixel noise.
        """
        if noise_e <= 0.0 or not math.isfinite(noise_e):
            return math.inf
        low, high = 0.0, math.log(_MAX_FLUX_SIGMA)
        below_found, above_found = 0.0, 1.0
        for _ in range(_LIMIT_PASSES):
            logs = np.linspace(low, high, _LIMIT_POINTS)
            found = self.completeness(np.exp(logs) * noise_e, noise_e)
            above = np.flatnonzero(found >= completeness)
            if above.size == 0:
                return math.inf
            if above[0] == 0:
                return math.exp(logs[0]) * noise_e
            low, high = float(logs[above[0] - 1]), float(logs[above[0]])
            below_found, above_found = float(found[above[0] - 1]), float(found[above[0]])
        share = (completeness - below_found) / max(above_found - below_found, 1e-300)
        return math.exp(low + min(max(share, 0.0), 1.0) * (high - low)) * noise_e


@functools.lru_cache(maxsize=512)
def search_model(
    search: SearchSpec, fwhm_px: float, trail_px: float = 0.0, trail_angle_rad: float = 0.0
) -> SearchModel:
    """`SearchModel.build`, kept for the next frames, which use the same nodes of the grid."""
    return SearchModel.build(
        search, fwhm_px=fwhm_px, trail_px=trail_px, trail_angle_rad=trail_angle_rad
    )


def _bracket(values: FloatArray, nodes: FloatArray) -> tuple[IntArray, IntArray]:
    """The indices of the nodes below and above each value. A value on a node or beyond the last
    one gets the same node twice."""
    lower = np.clip(np.searchsorted(nodes, values, side="right") - 1, 0, nodes.size - 1)
    upper = np.where(values > nodes[lower], np.minimum(lower + 1, nodes.size - 1), lower)
    return lower.astype(np.intp), upper.astype(np.intp)


def _corners(trail_px: FloatArray, trail_angle_rad: FloatArray) -> list[tuple[IntArray, IntArray]]:
    """The node indices (length, direction) of the four corners of each star's cell."""
    length = np.where(np.isfinite(trail_px), np.maximum(trail_px, 0.0), 0.0)
    angle = fold_angle(np.where(np.isfinite(trail_angle_rad), trail_angle_rad, 0.0))
    short, long = _bracket(length, TRAIL_NODES_PX)
    near, far = _bracket(angle, ANGLE_NODES_RAD)
    return [(short, near), (short, far), (long, near), (long, far)]


def star_completeness(
    search: SearchSpec,
    flux_e: npt.ArrayLike,
    noise_e: npt.ArrayLike,
    *,
    fwhm_px: float,
    trail_px: npt.ArrayLike,
    trail_angle_rad: npt.ArrayLike,
) -> FloatArray:
    """The completeness of `search` for each star, with its own trail and noise.

    The arguments broadcast against each other. Each star takes the least complete of the models at
    the four corners of its cell of the grid of trail lengths and directions, for the FWHM of the
    grid nearest to `fwhm_px` (see the module text). A FWHM that is not finite raises `ValueError`.
    """
    flux, noise, trail, angle = np.broadcast_arrays(
        np.asarray(flux_e, dtype=np.float64),
        np.asarray(noise_e, dtype=np.float64),
        np.asarray(trail_px, dtype=np.float64),
        np.asarray(trail_angle_rad, dtype=np.float64),
    )
    shape = flux.shape
    flux, noise = flux.ravel(), noise.ravel()
    width = model_fwhm(fwhm_px)
    result = np.ones(flux.size)
    for length_index, angle_index in _corners(trail.ravel(), angle.ravel()):
        key = length_index * ANGLE_NODES_RAD.size + angle_index
        for node in np.unique(key):
            stars = key == node
            model = search_model(
                search,
                width,
                float(TRAIL_NODES_PX[node // ANGLE_NODES_RAD.size]),
                float(ANGLE_NODES_RAD[node % ANGLE_NODES_RAD.size]),
            )
            found = model.completeness(flux[stars], noise[stars])
            result[stars] = np.minimum(result[stars], found)
    return result.reshape(shape)


def star_limit_e(
    search: SearchSpec,
    completeness: float,
    noise_e: float,
    *,
    fwhm_px: float,
    trail_px: npt.ArrayLike,
    trail_angle_rad: npt.ArrayLike,
) -> FloatArray:
    """The flux in electrons at which `search` finds each star with the chance `completeness`.

    Each star takes the highest limit of the models at the corners of its cell, as
    `star_completeness` takes the lowest completeness. `noise_e` is one value for the frame.
    """
    trail, angle = np.broadcast_arrays(
        np.asarray(trail_px, dtype=np.float64), np.asarray(trail_angle_rad, dtype=np.float64)
    )
    width = model_fwhm(fwhm_px)
    limits: dict[int, float] = {}
    result = np.zeros(trail.size)
    for length_index, angle_index in _corners(trail.ravel(), angle.ravel()):
        key = length_index * ANGLE_NODES_RAD.size + angle_index
        for node in np.unique(key):
            if int(node) not in limits:
                model = search_model(
                    search,
                    width,
                    float(TRAIL_NODES_PX[node // ANGLE_NODES_RAD.size]),
                    float(ANGLE_NODES_RAD[node % ANGLE_NODES_RAD.size]),
                )
                limits[int(node)] = model.limit_e(completeness, noise_e)
            stars = key == node
            result[stars] = np.maximum(result[stars], limits[int(node)])
    return result.reshape(trail.shape)
