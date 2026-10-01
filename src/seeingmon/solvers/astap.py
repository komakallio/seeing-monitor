"""The ASTAP adapter: the fallback plate solver, as a separate process.

ASTAP (the Astrometric STAcking Program) is MPL code with its own star detector, and it reads
an image, not a star list. The adapter therefore draws the detected stars into a small FITS
image (clean Gaussians on a flat background, so ASTAP's detector finds exactly the stars that
the survey path found and never mistakes an undersampled bin2 star for a hot pixel), runs
`astap` on it, and reads the solution. The image is `render_scale` times smaller than the
frame, which keeps it near a megapixel, and the adapter converts the solution back to the pixels
of the frame.

The command is `astap -f field.fits -fov <height> -z 1 -wcs`, plus a position hint
(`-ra` in hours and `-spd`, the south polar distance, which is declination plus 90) and a
search radius (`-r`) when the request has a center hint, and the star database (`-d`, `-D`).
ASTAP exits with 0 when it solves, 1 or 2 when it finds no solution or too few stars, and
another code when it cannot run (it cannot read the image, or it finds no star database). The
adapter reads `field.ini` (the `KEY=VALUE` file) and `field.wcs` (a header). ASTAP reports no
list of matched stars, so `matched` stays empty and `n_matched` is 0.
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
from seeingmon.survey.geometry import FloatArray

_BASE = "field"
_NO_SOLUTION_CODES = frozenset({1, 2})
_STAR_SIGMA_PX = 1.5  # in the drawn image
_BACKGROUND_DN = 500.0
_NOISE_DN = 6.0


def draw_star_image(
    x: FloatArray,
    y: FloatArray,
    flux: FloatArray,
    *,
    width: int,
    height: int,
    scale: int,
) -> npt.NDArray[np.uint16]:
    """Draw stars into a 16-bit image that is `scale` times smaller than the frame.

    Every star becomes a Gaussian of 1.5 pixels. The amplitude follows the flux to the power
    0.4, from 2,500 to 60,000 counts, so the faint stars stay detectable and none saturates.
    A small, seeded noise gives the detector a noise level to work with.
    """
    columns = -(-width // scale)
    rows = -(-height // scale)
    image = np.full((rows, columns), _BACKGROUND_DN, dtype=np.float64)
    image += np.random.default_rng(0).normal(0.0, _NOISE_DN, image.shape)
    if x.size:
        amplitude = np.clip(60_000.0 * (flux / float(flux.max())) ** 0.4, 2_500.0, 60_000.0)
        half = 6
        offsets = np.arange(-half, half + 1)
        for sx, sy, a in zip(
            (x + 0.5) / scale - 0.5, (y + 0.5) / scale - 0.5, amplitude, strict=True
        ):
            cx, cy = round(float(sx)), round(float(sy))
            px, py = cx + offsets, cy + offsets
            stamp = a * np.outer(
                np.exp(-0.5 * ((py - sy) / _STAR_SIGMA_PX) ** 2),
                np.exp(-0.5 * ((px - sx) / _STAR_SIGMA_PX) ** 2),
            )
            y0, y1 = max(py[0], 0), min(py[-1] + 1, rows)
            x0, x1 = max(px[0], 0), min(px[-1] + 1, columns)
            if y0 < y1 and x0 < x1:
                image[y0:y1, x0:x1] += stamp[y0 - py[0] : y1 - py[0], x0 - px[0] : x1 - px[0]]
    return np.clip(np.rint(image), 0, 65535).astype(np.uint16)


def _parse_ini(text: str) -> dict[str, str]:
    """Read the `KEY=VALUE` lines of the ASTAP solution file."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip().upper()] = value.strip()
    return values


def _float(values: dict[str, str], key: str) -> float | None:
    try:
        return float(values[key].replace(",", "."))
    except (KeyError, ValueError):
        return None


def _required(values: dict[str, str], key: str) -> float:
    number = _float(values, key)
    if number is None:
        raise SolverError(f"astap reported a solution without {key}")
    return number


class AstapSolver:
    """Plate solving with `astap`, as the fallback when astrometry.net fails.

    `command` is the program with any leading arguments (`"astap"` by default).
    `database_dir` and `database` name the star database (`-d` and `-D`), and an empty value
    leaves the choice to ASTAP. `render_scale` shrinks the drawn image. `extra_args` go on the
    command line before `-f`.
    """

    def __init__(
        self,
        *,
        command: str | Sequence[str] = "astap",
        database_dir: str | os.PathLike[str] | None = None,
        database: str | None = None,
        render_scale: int = 4,
        max_stars: int = 600,
        extra_args: Sequence[str] = (),
        work_dir: str | os.PathLike[str] | None = None,
        clock: Clock | None = None,
    ) -> None:
        if render_scale < 1 or max_stars < 4:
            raise ValueError("render_scale must be at least 1 and max_stars at least 4")
        self._command = split_command(command)
        self._database_dir = None if database_dir is None else str(database_dir)
        self._database = database
        self._scale = render_scale
        self._max_stars = max_stars
        self._extra_args = list(extra_args)
        self._work_dir = None if work_dir is None else Path(work_dir)
        self._clock = clock or SystemClock()

    @property
    def name(self) -> str:
        return "astap"

    def solve(self, request: SolveRequest) -> SolveResult:
        started_ns = self._clock.monotonic_ns()
        stars = request.stars
        if len(stars) < 4:
            return SolveResult(solved=False, solver=self.name, elapsed_s=0.0)
        with tempfile.TemporaryDirectory(prefix="seeingmon-astap-", dir=self._work_dir) as folder:
            work = Path(folder)
            order = np.argsort(-stars.flux, kind="stable")[: self._max_stars]
            image = draw_star_image(
                stars.x[order],
                stars.y[order],
                stars.flux[order],
                width=request.width_px,
                height=request.height_px,
                scale=self._scale,
            )
            fitsio.write_image(work / f"{_BASE}.fits", image)
            result = run_process(
                self._arguments(request, work),
                timeout_s=request.timeout_s + PROCESS_GRACE_S,
                cwd=work,
            )
            elapsed = (self._clock.monotonic_ns() - started_ns) / NS_PER_S
            if result.returncode in _NO_SOLUTION_CODES:
                return SolveResult(solved=False, solver=self.name, elapsed_s=elapsed)
            if result.returncode != 0:
                raise SolverError(
                    f"astap exited with code {result.returncode}: {output_tail(result)}"
                )
            return self._read_solution(work, request, elapsed)

    def _arguments(self, request: SolveRequest, work: Path) -> list[str]:
        mean_scale = 0.5 * (request.scale_low_arcsec_px + request.scale_high_arcsec_px)
        height_deg = request.height_px * mean_scale / 3600.0
        arguments = [
            *self._command,
            "-f",
            str(work / f"{_BASE}.fits"),
            "-fov",
            f"{height_deg:.5f}",
            "-z",
            "1",
            "-wcs",
        ]
        if request.center_ra_deg is not None and request.center_dec_deg is not None:
            arguments += [
                "-ra",
                f"{request.center_ra_deg / 15.0:.6f}",
                "-spd",
                f"{request.center_dec_deg + 90.0:.5f}",
                "-r",
                f"{request.radius_deg if request.radius_deg is not None else 5.0:.3f}",
            ]
        if self._database_dir:
            arguments += ["-d", self._database_dir]
        if self._database:
            arguments += ["-D", self._database]
        return [*arguments, *self._extra_args]

    def _read_solution(self, work: Path, request: SolveRequest, elapsed: float) -> SolveResult:
        values: dict[str, str] = {}
        ini = work / f"{_BASE}.ini"
        wcs = work / f"{_BASE}.wcs"
        if ini.is_file():
            values.update(_parse_ini(ini.read_text(encoding="utf-8", errors="replace")))
        if wcs.is_file():
            for key, value in fitsio.parse_header_text(
                wcs.read_text(encoding="ascii", errors="replace")
            ).items():
                values.setdefault(key, str(value))
        if values.get("PLTSOLVD", "T").upper().startswith("F"):
            return SolveResult(solved=False, solver=self.name, elapsed_s=elapsed)
        crval_ra = _required(values, "CRVAL1")
        crval_dec = _required(values, "CRVAL2")
        crpix_x = _required(values, "CRPIX1")
        crpix_y = _required(values, "CRPIX2")
        cd = self._cd_matrix(values)
        determinant = cd[0] * cd[3] - cd[1] * cd[2]
        if determinant == 0.0 or not np.isfinite(determinant):
            raise SolverError("astap returned a singular CD matrix")
        # The drawn image is `scale` times smaller than the frame. Move the reference to the
        # center of the frame, and give the CD matrix in the pixels of the frame.
        scale = self._scale
        center_x = (request.width_px - 1) / 2.0
        center_y = (request.height_px - 1) / 2.0
        offset = (
            (center_x + 0.5) / scale + 0.5 - crpix_x,
            (center_y + 0.5) / scale + 0.5 - crpix_y,
        )
        ra, dec = tan_center(crval_ra, crval_dec, cd, offset)
        cd_frame = (cd[0] / scale, cd[1] / scale, cd[2] / scale, cd[3] / scale)
        return SolveResult(
            solved=True,
            solver=self.name,
            elapsed_s=elapsed,
            center_ra_deg=ra,
            center_dec_deg=dec,
            scale_arcsec_px=float(np.sqrt(abs(determinant)) * 3600.0 / scale),
            cd_matrix=cd_frame,
        )

    @staticmethod
    def _cd_matrix(values: dict[str, str]) -> tuple[float, float, float, float]:
        """The CD matrix in degrees per drawn pixel, from `CD` cards or `CDELT` and `CROTA`."""
        if all(_float(values, key) is not None for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
            return (
                _required(values, "CD1_1"),
                _required(values, "CD1_2"),
                _required(values, "CD2_1"),
                _required(values, "CD2_2"),
            )
        cdelt1 = _required(values, "CDELT1")
        cdelt2 = _required(values, "CDELT2")
        rho = np.radians(_float(values, "CROTA2") or 0.0)
        return (
            float(cdelt1 * np.cos(rho)),
            float(-cdelt2 * np.sin(rho)),
            float(cdelt1 * np.sin(rho)),
            float(cdelt2 * np.cos(rho)),
        )
