"""The astrometry.net adapter: `solve-field` as a separate process.

astrometry.net is GPL code, so the adapter runs the `solve-field` program in its own process
and talks to it through files. It never imports or links the solver. For each request it:

1. Writes the star list as an astrometry.net "xylist": a FITS binary table with the columns
   `X`, `Y`, and `FLUX`, and the card `IMAGEW` and `IMAGEH` for the frame size. Pixel positions
   are one-based in FITS, so the adapter adds 1 to the pixel positions of `StarList` (whose
   first pixel center is 0).
2. Writes a backend configuration that names the custom cap index files, so the solver looks
   at nothing else.
3. Runs `solve-field` with the scale bounds, the center hint, and a CPU limit, and with the
   plots and the filters that need Python switched off.
4. Reads `field.wcs` (the solution as a FITS header) and `field.corr` (the stars that the
   solver matched).

The solution is a linear TAN WCS in the catalog frame of the index. The adapter reads the
`CD` matrix and ignores any SIP terms, which the survey fit does not need. It converts the
reference pixel to the center of the frame, so the result matches `SolveResult`.

A run that finds no solution is a normal result (`solved=False`). A missing program, a crash,
a non-zero exit code, an unreadable output, and a process that hangs raise `SolverError`.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.solvers import fitsio
from seeingmon.solvers.base import SolveRequest, SolverError, SolveResult
from seeingmon.solvers.process import (
    PROCESS_GRACE_S,
    output_tail,
    run_process,
    split_command,
    tan_center,
)
from seeingmon.survey.geometry import (
    ARCSEC_PER_RAD,
    FloatArray,
    angular_separation,
    radec_to_vector,
)

IntArray = npt.NDArray[np.intp]

_BASE = "field"
# The matched stars that the solver reports are matched back to the request by position.
_MATCH_TOLERANCE_PX = 0.5


def _number(header: fitsio.Header, key: str) -> float:
    """A numeric header card. Raises `KeyError` or `ValueError` when it is missing or no number."""
    value = header[key]
    if isinstance(value, bool | str):
        raise ValueError(f"the card {key} is not a number")
    return float(value)


class AstrometryNetSolver:
    """Plate solving with `solve-field` and a custom index.

    `index_dir` holds the index files (`*.fits`) that `seeingmon catalog build` writes.
    `command` is the program, with any leading arguments (`"solve-field"` by default).
    `max_stars` limits how many of the brightest stars go to the solver. `extra_args` go on the
    command line before the input file. `work_dir` is the parent of the temporary folders,
    which the adapter removes after each run.
    """

    def __init__(
        self,
        index_dir: str | os.PathLike[str],
        *,
        command: str | Sequence[str] = "solve-field",
        max_stars: int = 600,
        extra_args: Sequence[str] = (),
        work_dir: str | os.PathLike[str] | None = None,
        clock: Clock | None = None,
    ) -> None:
        if max_stars < 4:
            raise ValueError("max_stars must be at least 4")
        self._index_dir = Path(index_dir)
        self._command = split_command(command)
        self._max_stars = max_stars
        self._extra_args = list(extra_args)
        self._work_dir = None if work_dir is None else Path(work_dir)
        self._clock = clock or SystemClock()

    @property
    def name(self) -> str:
        return "astrometry.net"

    def solve(self, request: SolveRequest) -> SolveResult:
        started_ns = self._clock.monotonic_ns()
        stars = request.stars
        if len(stars) < 4:
            return SolveResult(solved=False, solver=self.name, elapsed_s=0.0)
        index_files = sorted(self._index_dir.glob("*.fits"))
        if not index_files:
            raise SolverError("the index folder holds no index files")
        with tempfile.TemporaryDirectory(prefix="seeingmon-solve-", dir=self._work_dir) as folder:
            work = Path(folder)
            order = np.argsort(-stars.flux, kind="stable")[: self._max_stars]
            fitsio.write_table(
                work / f"{_BASE}.xyls",
                {
                    "X": stars.x[order] + 1.0,
                    "Y": stars.y[order] + 1.0,
                    "FLUX": stars.flux[order],
                },
                header={"IMAGEW": request.width_px, "IMAGEH": request.height_px},
            )
            config = work / "backend.cfg"
            config.write_text(
                self._backend_config(index_files, request.timeout_s), encoding="ascii"
            )
            arguments = self._arguments(request, work, config)
            result = run_process(arguments, timeout_s=request.timeout_s + PROCESS_GRACE_S, cwd=work)
            if result.returncode != 0:
                raise SolverError(
                    f"solve-field exited with code {result.returncode}: {output_tail(result)}"
                )
            elapsed = (self._clock.monotonic_ns() - started_ns) / NS_PER_S
            if not (work / f"{_BASE}.solved").is_file() or not (work / f"{_BASE}.wcs").is_file():
                return SolveResult(solved=False, solver=self.name, elapsed_s=elapsed)
            return self._read_solution(
                work, request, stars.x[order], stars.y[order], order, elapsed
            )

    def _backend_config(self, index_files: list[Path], timeout_s: float) -> str:
        lines = [f"add_path {self._index_dir.resolve().as_posix()}"]
        lines += [f"index {path.resolve().as_posix()}" for path in index_files]
        lines.append(f"cpulimit {max(1, round(timeout_s))}")
        return "\n".join(lines) + "\n"

    def _arguments(self, request: SolveRequest, work: Path, config: Path) -> list[str]:
        arguments = [
            *self._command,
            "--overwrite",
            "--no-plots",
            "--no-remove-lines",
            "--uniformize",
            "0",
            "--config",
            str(config),
            "--dir",
            str(work),
            "--out",
            _BASE,
            "--width",
            str(request.width_px),
            "--height",
            str(request.height_px),
            "--x-column",
            "X",
            "--y-column",
            "Y",
            "--sort-column",
            "FLUX",
            "--crpix-center",
            "--scale-units",
            "arcsecperpix",
            "--scale-low",
            f"{request.scale_low_arcsec_px:.6g}",
            "--scale-high",
            f"{request.scale_high_arcsec_px:.6g}",
            "--cpulimit",
            str(max(1, round(request.timeout_s))),
        ]
        if request.center_ra_deg is not None and request.center_dec_deg is not None:
            arguments += [
                "--ra",
                f"{request.center_ra_deg:.6f}",
                "--dec",
                f"{request.center_dec_deg:.6f}",
                "--radius",
                f"{request.radius_deg if request.radius_deg is not None else 5.0:.4f}",
            ]
        arguments += self._extra_args
        arguments.append(str(work / f"{_BASE}.xyls"))
        return arguments

    def _read_solution(
        self,
        work: Path,
        request: SolveRequest,
        star_x: FloatArray,
        star_y: FloatArray,
        order: IntArray,
        elapsed: float,
    ) -> SolveResult:
        try:
            header = fitsio.read_header(work / f"{_BASE}.wcs")
            crval_ra = _number(header, "CRVAL1")
            crval_dec = _number(header, "CRVAL2")
            crpix = (_number(header, "CRPIX1"), _number(header, "CRPIX2"))
            cd = (
                _number(header, "CD1_1"),
                _number(header, "CD1_2"),
                _number(header, "CD2_1"),
                _number(header, "CD2_2"),
            )
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise SolverError(f"cannot read the solution of solve-field: {error}") from error
        determinant = cd[0] * cd[3] - cd[1] * cd[2]
        if determinant == 0.0 or not np.isfinite(determinant):
            raise SolverError("solve-field returned a singular CD matrix")
        # The center of the frame in FITS pixels, and its offset from the reference pixel.
        center_fits = ((request.width_px + 1) / 2.0, (request.height_px + 1) / 2.0)
        ra, dec = tan_center(
            crval_ra, crval_dec, cd, (center_fits[0] - crpix[0], center_fits[1] - crpix[1])
        )
        matched, rms = self._read_correspondences(work, star_x, star_y, order)
        return SolveResult(
            solved=True,
            solver=self.name,
            elapsed_s=elapsed,
            center_ra_deg=ra,
            center_dec_deg=dec,
            scale_arcsec_px=float(np.sqrt(abs(determinant)) * 3600.0),
            cd_matrix=cd,
            n_matched=len(matched),
            rms_arcsec=rms,
            matched=matched,
        )

    def _read_correspondences(
        self, work: Path, star_x: FloatArray, star_y: FloatArray, order: IntArray
    ) -> tuple[tuple[int, ...], float | None]:
        """The request indices of the matched stars and the RMS of their positions, if known."""
        path = work / f"{_BASE}.corr"
        if not path.is_file():
            return (), None
        try:
            table = {name.lower(): column for name, column in fitsio.read_table(path).items()}
            field_x = np.asarray(table["field_x"], dtype=np.float64) - 1.0
            field_y = np.asarray(table["field_y"], dtype=np.float64) - 1.0
        except (fitsio.FitsError, KeyError, OSError):
            return (), None
        indices: list[int] = []
        for fx, fy in zip(field_x, field_y, strict=True):
            distance = np.hypot(star_x - fx, star_y - fy)
            nearest = int(np.argmin(distance)) if distance.size else -1
            if nearest >= 0 and distance[nearest] <= _MATCH_TOLERANCE_PX:
                indices.append(int(order[nearest]))
        rms: float | None = None
        needed = ("field_ra", "field_dec", "index_ra", "index_dec")
        if all(name in table for name in needed) and len(table["field_ra"]) > 0:
            field = radec_to_vector(table["field_ra"], table["field_dec"])
            index = radec_to_vector(table["index_ra"], table["index_dec"])
            separation = angular_separation(field, index) * ARCSEC_PER_RAD
            rms = float(np.sqrt(np.mean(separation**2)))
        return tuple(indices), rms
