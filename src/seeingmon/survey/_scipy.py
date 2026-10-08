"""Typed access to the few SciPy functions that the survey path uses.

SciPy is an optional dependency (the `fast` extra, which the `survey` extra includes), and its
modules ship no type information that `mypy --strict` accepts. This module imports SciPy once,
raises a clear error when it is missing, and gives each function a type, so the rest of the
package stays strictly typed.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


def _load(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # depends on the installation
        raise ImportError(
            "the survey path needs SciPy: install the 'survey' extra, for example "
            "`pip install 'seeingmon[survey]'`"
        ) from exc


_spatial = _load("scipy.spatial")
_special = _load("scipy.special")
_ndimage = _load("scipy.ndimage")


def nearest(
    reference: FloatArray, query: FloatArray, max_distance: float = np.inf
) -> tuple[FloatArray, IntArray]:
    """For each row of `query`, the distance to the nearest row of `reference` and its index.

    A query point with no reference point within `max_distance` gets an infinite distance and
    the index `len(reference)`.
    """
    tree = _spatial.cKDTree(reference)
    distance, index = tree.query(query, distance_upper_bound=max_distance)
    return np.asarray(distance, dtype=np.float64), np.asarray(index, dtype=np.intp)


def pairs_within(first: FloatArray, second: FloatArray, radius: float) -> list[list[int]]:
    """For each row of `first`, the indices of the rows of `second` within `radius`."""
    tree_second = _spatial.cKDTree(second)
    tree_first = _spatial.cKDTree(first)
    found = tree_first.query_ball_tree(tree_second, radius)
    return [list(map(int, row)) for row in found]


def pair_indices(first: FloatArray, second: FloatArray, radius: float) -> tuple[IntArray, IntArray]:
    """The pairs of a row of `first` and a row of `second` that lie within `radius`, as indices.

    Pair `k` joins row `i[k]` of `first` with row `j[k]` of `second`, and the function returns
    `(i, j)`. It finds the pairs that `pairs_within` finds, as two arrays and not as lists, and the
    order of the pairs is not defined.
    """
    found = _spatial.cKDTree(first).sparse_distance_matrix(
        _spatial.cKDTree(second), radius, output_type="ndarray"
    )
    return np.asarray(found["i"], dtype=np.intp), np.asarray(found["j"], dtype=np.intp)


def close_pairs(points: FloatArray, radius: float) -> IntArray:
    """The pairs of rows of `points` that lie no farther apart than `radius`.

    Returns an array of shape `(N, 2)` with the indices `(i, j)`, where `i < j`.
    """
    tree = _spatial.cKDTree(points)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    return np.asarray(pairs, dtype=np.intp).reshape(-1, 2)


def within_radii(points: FloatArray, centers: FloatArray, radii: FloatArray) -> list[IntArray]:
    """For each row of `centers`, the indices of the rows of `points` within its own radius."""
    tree = _spatial.cKDTree(points)
    found = tree.query_ball_point(centers, radii)
    return [np.asarray(row, dtype=np.intp) for row in found]


def erf(x: FloatArray) -> FloatArray:
    """The error function."""
    return np.asarray(_special.erf(x), dtype=np.float64)


def erf32(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """The error function in single precision."""
    return np.asarray(_special.erf(x), dtype=np.float32)


def uniform_filter(image: npt.NDArray[np.float32], size: int) -> npt.NDArray[np.float32]:
    """The mean of each pixel's `size` x `size` neighborhood. The edges reflect."""
    return np.asarray(_ndimage.uniform_filter(image, size=size, mode="reflect"), dtype=np.float32)


def gaussian_filter(
    image: npt.NDArray[np.float64], sigma: float, *, mode: str = "nearest"
) -> npt.NDArray[np.float64]:
    """The image smoothed with a Gaussian of `sigma` pixels. `mode` says how the edges extend."""
    return np.asarray(_ndimage.gaussian_filter(image, sigma=sigma, mode=mode), dtype=np.float64)


def label(mask: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.int32], int]:
    """Number the connected regions of a mask (a pixel joins its four neighbors).

    Returns the label image (0 for the background, 1 to `count` for the regions) and the count.
    """
    labels, count = _ndimage.label(mask)
    return np.asarray(labels, dtype=np.int32), int(count)


def kdtree(points: FloatArray) -> Any:
    """A k-d tree of the rows of `points`: `query` and `query_ball_point` work as in SciPy."""
    return _spatial.cKDTree(points)
