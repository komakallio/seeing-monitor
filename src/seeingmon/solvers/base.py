"""The plate solver interface.

A solver finds where a star list sits on the sky. It runs in its own process (astrometry.net
is GPL code, so it stays outside this MIT code base), and the adapters in this package wrap
that. The solver returns an initial TAN solution. The survey path then fits a world
coordinate system to all matched stars with apparent places, so the solver's result is a
starting point and a pass/fail signal.

A failure to find a solution is a normal outcome and returns `SolveResult(solved=False)`. A
failure to run (a missing binary, a crash, a timeout of the process itself) raises
`SolverError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


class SolverError(Exception):
    """The solver could not run."""


@dataclass(frozen=True, slots=True, eq=False)
class StarList:
    """Detected stars in pixel coordinates. The center of the first pixel is (0, 0).

    All arrays have the same length. `flux` is background-subtracted counts, brightest
    meaning most reliable, and solvers use it to rank stars.
    """

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    flux: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if not (self.x.ndim == 1 and self.x.shape == self.y.shape == self.flux.shape):
            raise ValueError("x, y, and flux must be 1-D arrays of the same length")

    def __len__(self) -> int:
        return int(self.x.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class SolveRequest:
    """What to solve. Scale limits bound the search, and the center hint narrows it."""

    stars: StarList
    width_px: int
    height_px: int
    scale_low_arcsec_px: float
    scale_high_arcsec_px: float
    center_ra_deg: float | None = None
    center_dec_deg: float | None = None
    radius_deg: float | None = None
    timeout_s: float = 10.0


@dataclass(frozen=True, slots=True)
class SolveResult:
    """An initial TAN solution, referenced to the image center.

    `cd_matrix` is `(CD1_1, CD1_2, CD2_1, CD2_2)` in degrees per pixel, with the reference
    pixel at the image center. `matched` lists the indices into the request's star list that
    the solver matched to catalog stars.
    """

    solved: bool
    solver: str
    elapsed_s: float
    center_ra_deg: float | None = None
    center_dec_deg: float | None = None
    scale_arcsec_px: float | None = None
    cd_matrix: tuple[float, float, float, float] | None = None
    n_matched: int = 0
    rms_arcsec: float | None = None
    matched: tuple[int, ...] = field(default_factory=tuple)


@runtime_checkable
class PlateSolver(Protocol):
    @property
    def name(self) -> str:
        """The solver name, such as `astrometry.net`, `astap`, or `fake`."""
        ...

    def solve(self, request: SolveRequest) -> SolveResult:
        """Solve one star list. Raises `SolverError` when the solver cannot run."""
        ...
