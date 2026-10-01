"""Typed access to the few SciPy functions that the fast path uses.

SciPy is an optional dependency (the `fast` extra), and it ships no type information. This
module imports SciPy on first use, raises a clear error when it is missing, and gives each
function a type, so the rest of the package stays strictly typed. The per-frame kernel needs
only NumPy. The seeing models need SciPy once, when they build a table of corrections.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import cache
from types import ModuleType

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@cache
def _load(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # depends on the installation
        raise ImportError(
            "the seeing estimator needs SciPy: install the 'fast' extra, for example "
            "`pip install 'seeingmon[fast]'`"
        ) from exc


def j1(x: FloatArray) -> FloatArray:
    """The Bessel function of the first kind and order 1."""
    return np.asarray(_load("scipy.special").j1(x), dtype=np.float64)


def j1_scalar(x: float) -> float:
    """The same function for one value."""
    return float(_load("scipy.special").j1(x))


def erf(x: FloatArray) -> FloatArray:
    """The error function."""
    return np.asarray(_load("scipy.special").erf(x), dtype=np.float64)


def spherical_jn(order: int, x: FloatArray) -> FloatArray:
    """The spherical Bessel function of the first kind."""
    return np.asarray(_load("scipy.special").spherical_jn(order, x), dtype=np.float64)


def quad(
    func: Callable[[float], float],
    low: float,
    high: float,
    *,
    limit: int = 100,
    epsrel: float = 1e-9,
) -> float:
    """The integral of a function over an interval, to a relative accuracy."""
    return float(
        _load("scipy.integrate").quad(func, low, high, limit=limit, epsabs=0.0, epsrel=epsrel)[0]
    )
