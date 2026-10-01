"""Typed access to the few SciPy functions that the simulator uses.

SciPy is an optional dependency (the `fast` extra), and it ships no type information. This
module imports SciPy once, raises a clear error when it is missing, and gives each function a
type, so the rest of the package stays strictly typed.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexSingle = npt.NDArray[np.complex64]


def _load(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # depends on the installation
        raise ImportError(
            "the sim driver needs SciPy: install the 'fast' extra, for example "
            "`pip install 'seeingmon[fast]'`"
        ) from exc


_fft = _load("scipy.fft")
_special = _load("scipy.special")
_integrate = _load("scipy.integrate")
_optimize = _load("scipy.optimize")
_signal = _load("scipy.signal")


def j1(x: FloatArray) -> FloatArray:
    """The Bessel function of the first kind and order 1."""
    return np.asarray(_special.j1(x), dtype=np.float64)


def j1_scalar(x: float) -> float:
    """The same function for one value."""
    return float(_special.j1(x))


def erf(x: FloatArray) -> FloatArray:
    """The error function."""
    return np.asarray(_special.erf(x), dtype=np.float64)


def quad(
    func: Callable[[float], float],
    low: float,
    high: float,
    *,
    limit: int = 100,
    epsrel: float = 1e-9,
) -> float:
    """The integral of a function over an interval, to a relative accuracy."""
    return float(_integrate.quad(func, low, high, limit=limit, epsabs=0.0, epsrel=epsrel)[0])


def nnls(matrix: FloatArray, target: FloatArray) -> FloatArray:
    """Non-negative least squares: the vector `x >= 0` that minimizes `|matrix x - target|`."""
    solution = _optimize.nnls(matrix, target)[0]
    return np.asarray(solution, dtype=np.float64)


def fftconvolve(a: FloatArray, b: FloatArray) -> FloatArray:
    """The full two-dimensional convolution of two arrays, by FFT."""
    return np.asarray(_signal.fftconvolve(a, b), dtype=np.float64)


def fft2(a: ComplexSingle) -> ComplexSingle:
    """The two-dimensional FFT in single precision. SciPy runs this faster than NumPy does."""
    return np.asarray(_fft.fft2(a), dtype=np.complex64)


def ifft2(a: ComplexSingle) -> ComplexSingle:
    """The inverse two-dimensional FFT in single precision."""
    return np.asarray(_fft.ifft2(a), dtype=np.complex64)
