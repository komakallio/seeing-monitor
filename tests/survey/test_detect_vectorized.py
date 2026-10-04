"""The batched star mask, box counts, and blend test give the results of the loops they replaced.

Each test keeps the original loop as a reference and compares it with the new code on random
inputs: positions on half pixels, stars outside the frame, disks of every size class, boxes at
the frame edge, and pairs at the limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.survey import _scipy, detect
from seeingmon.survey.centroid import FWHM_PER_SIGMA

SEEDS = range(12)
BoolArray = npt.NDArray[np.bool_]


# --- The loops that the new code replaced --------------------------------------------------


def loop_disks(mask: BoolArray, xs: Any, ys: Any, radii: Any) -> None:
    height, width = mask.shape
    for x, y, r in zip(xs, ys, radii, strict=True):
        reach = int(np.ceil(r))
        x0, x1 = max(round(x) - reach, 0), min(round(x) + reach + 1, width)
        y0, y1 = max(round(y) - reach, 0), min(round(y) + reach + 1, height)
        if x0 >= x1 or y0 >= y1:
            continue
        gy, gx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (gx - x) ** 2 + (gy - y) ** 2 <= r * r


def loop_star_mask(
    shape: tuple[int, int], found: detect.Detections, radius_scale: float, minimum_px: float
) -> BoolArray:
    mask = np.zeros(shape, dtype=np.bool_)
    saturated = found.has(detect.StarFlag.SATURATED)
    sigma = found.fwhm_px / FWHM_PER_SIGMA
    radius = radius_scale * sigma + found.trail_length_px / 2.0
    radius = np.maximum(radius, minimum_px)
    radius = np.where(np.isfinite(radius), radius, minimum_px)
    radius = np.where(
        saturated, np.maximum(radius, 2.0 * np.sqrt(np.maximum(found.n_pixels, 1))), radius
    )
    loop_disks(mask, found.x, found.y, radius)
    return mask


def loop_boxes(mask: BoolArray, objects: Any) -> npt.NDArray[np.intp]:
    counts = np.zeros(len(objects), dtype=np.intp)
    xmin, xmax, ymin, ymax = (objects[key] for key in ("xmin", "xmax", "ymin", "ymax"))
    for i in range(len(objects)):
        counts[i] = int(mask[ymin[i] : ymax[i] + 1, xmin[i] : xmax[i] + 1].sum())
    return counts


def loop_blended(x: Any, y: Any, reach: Any) -> BoolArray:
    pairs = _scipy.pairs_within(
        np.column_stack([x, y]), np.column_stack([x, y]), float(2 * reach.max())
    )
    blended = np.zeros(len(x), dtype=bool)
    for i, near in enumerate(pairs):
        for j in near:
            if j != i and np.hypot(x[i] - x[j], y[i] - y[j]) < reach[i] + reach[j]:
                blended[i] = True
                break
    return blended


# --- Random inputs -------------------------------------------------------------------------


def random_detections(rng: np.random.Generator, count: int, shape: tuple[int, int]) -> Any:
    height, width = shape
    x = rng.uniform(-8.0, width + 8.0, count)
    y = rng.uniform(-8.0, height + 8.0, count)
    half = rng.random(count) < 0.15  # a center on a half pixel tests the rounding
    x[half], y[half] = np.floor(x[half]) + 0.5, np.floor(y[half]) + 0.5
    zeros = np.zeros(count)
    return detect.Detections(
        shape=shape,
        x=x,
        y=y,
        flux=np.ones(count),
        peak=zeros,
        fwhm_px=rng.uniform(0.7, 3.0, count),
        x_error_px=zeros,
        y_error_px=zeros,
        elongation=np.ones(count),
        trail_length_px=np.where(rng.random(count) < 0.3, rng.uniform(0.0, 14.0, count), 0.0),
        trail_angle_rad=zeros,
        flags=np.where(rng.random(count) < 0.1, int(detect.StarFlag.SATURATED), 0).astype(
            np.uint16
        ),
        snr=zeros,
        n_pixels=rng.integers(1, 6000, count).astype(np.int32),
        background_level=0.0,
        background_rms=1.0,
    )


# --- The disks and the star mask -----------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_batched_disks_equal_the_loop(seed: int) -> None:
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(5, 200)), int(rng.integers(5, 200)))
    count = int(rng.integers(1, 80))
    x = rng.uniform(-30.0, shape[1] + 30.0, count)
    y = rng.uniform(-30.0, shape[0] + 30.0, count)
    # Radii on both sides of every size class (4, 8, 16, and 32 pixels of reach), and zero.
    base = rng.choice([0.0, 0.4, 1.0, 3.0, 4.0, 4.5, 8.0, 9.9, 16.0, 17.0, 32.0, 33.0, 45.0], count)
    radius = base + rng.uniform(0.0, 0.5, count) * (rng.random(count) < 0.5)
    expected = np.zeros(shape, dtype=bool)
    loop_disks(expected, x, y, radius)
    found = np.zeros(shape, dtype=bool)
    detect.paint_disks(found, x, y, radius)
    np.testing.assert_array_equal(found, expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_star_mask_equals_the_loop(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    shape = (int(rng.integers(40, 300)), int(rng.integers(40, 400)))
    stars = random_detections(rng, int(rng.integers(0, 400)), shape)
    scale = float(rng.choice([1.0, 2.0, 3.0]))
    minimum = float(rng.choice([0.0, 1.0, 3.0, 5.5]))
    expected = loop_star_mask(shape, stars, scale, minimum)
    found = detect.star_mask(shape, stars, radius_scale=scale, minimum_px=minimum)
    np.testing.assert_array_equal(found, expected)


def test_a_disk_of_radius_three_holds_29_pixels() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    detect.paint_disks(mask, np.array([10.0]), np.array([10.0]), np.array([3.0]))
    assert int(mask.sum()) == 29  # the lattice points of a disk of radius 3
    assert mask[10, 13]
    assert mask[13, 10]
    assert not mask[13, 13]


def test_a_disk_beyond_the_frame_paints_nothing_and_a_disk_at_the_corner_paints_a_quarter() -> None:
    mask = np.zeros((30, 30), dtype=bool)
    detect.paint_disks(mask, np.array([-50.0]), np.array([10.0]), np.array([5.0]))
    assert not mask.any()
    detect.paint_disks(mask, np.array([0.0]), np.array([0.0]), np.array([5.0]))
    assert int(mask.sum()) == 26  # the lattice points of a disk of radius 5 with x >= 0 and y >= 0


def test_no_stars_make_an_empty_mask() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    empty = np.zeros(0)
    detect.paint_disks(mask, empty, empty, empty)
    assert not mask.any()


# --- The bounding boxes --------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_box_counts_equal_the_loop(seed: int) -> None:
    rng = np.random.default_rng(200 + seed)
    shape = (int(rng.integers(1, 300)), int(rng.integers(1, 300)))
    mask = rng.random(shape) < float(rng.choice([0.0, 0.001, 0.01, 0.2, 0.9]))
    count = int(rng.integers(0, 200))
    x0 = rng.integers(0, shape[1], count)
    y0 = rng.integers(0, shape[0], count)
    objects = np.zeros(count, dtype=[(key, np.int32) for key in ("xmin", "xmax", "ymin", "ymax")])
    objects["xmin"], objects["ymin"] = x0, y0
    objects["xmax"] = np.minimum(x0 + rng.integers(0, 40, count), shape[1] - 1)
    objects["ymax"] = np.minimum(y0 + rng.integers(0, 40, count), shape[0] - 1)
    np.testing.assert_array_equal(
        detect._bounding_box_any(mask, objects), loop_boxes(mask, objects)
    )


def test_box_counts_accept_a_mapping_of_arrays_and_ignore_the_part_outside_the_frame() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[4, 4] = mask[0, 0] = mask[2, 3] = True
    boxes = {
        "xmin": np.array([0, 3, -2, 4]),
        "xmax": np.array([4, 9, 0, 3]),  # the last box is empty: its xmax is below its xmin
        "ymin": np.array([0, 2, -1, 4]),
        "ymax": np.array([4, 7, 0, 4]),
    }
    assert detect._bounding_box_any(mask, boxes).tolist() == [3, 2, 1, 0]


def test_box_counts_of_nothing_are_empty() -> None:
    mask = np.ones((4, 4), dtype=bool)
    nothing = {key: np.zeros(0, dtype=np.int32) for key in ("xmin", "xmax", "ymin", "ymax")}
    assert detect._bounding_box_any(mask, nothing).shape == (0,)


# --- The blend test ------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_the_blend_test_equals_the_loop(seed: int) -> None:
    rng = np.random.default_rng(300 + seed)
    count = int(rng.integers(2, 700))
    extent = float(rng.choice([30.0, 100.0, 400.0, 3000.0]))
    x = rng.uniform(0.0, extent, count)
    y = rng.uniform(0.0, extent, count)
    if rng.random() < 0.3 and count >= 20:  # stars at the same place
        x[:5], y[:5] = x[5:10], y[5:10]
    reach = rng.uniform(2.0, 9.0, count)
    wide = rng.random(count) < 0.05  # a few saturated blobs
    reach[wide] = rng.uniform(15.0, 120.0, int(wide.sum()))
    np.testing.assert_array_equal(
        detect._blended_by_neighbors(x, y, reach), loop_blended(x, y, reach)
    )


def test_two_stars_are_blended_when_they_are_closer_than_their_reaches_together() -> None:
    reach = np.array([3.0, 4.0, 3.0, 3.0])
    x = np.array([0.0, 6.9, 100.0, 106.0])
    y = np.zeros(4)
    # The first pair is 6.9 apart, inside 3 + 4. The second is 6.0 apart, and the test is strict.
    assert detect._blended_by_neighbors(x, y, reach).tolist() == [True, True, False, False]


def test_a_blob_blends_the_small_stars_inside_its_reach() -> None:
    x = np.array([0.0, 60.0, 200.0])
    y = np.zeros(3)
    reach = np.array(
        [80.0, 4.0, 4.0]
    )  # the blob reaches 80 + 4 pixels, and the star at 60 is inside
    assert detect._blended_by_neighbors(x, y, reach).tolist() == [True, True, False]


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_stars_are_never_blended(count: int) -> None:
    x = np.zeros(count)
    found = detect._blended_by_neighbors(x, x, np.full(count, 3.0))
    assert found.shape == (count,)
    assert not found.any()
