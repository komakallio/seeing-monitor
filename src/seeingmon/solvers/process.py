"""Helpers for the solver adapters, which run each solver as a separate process.

A solver runs outside this process on purpose: astrometry.net is GPL code and ASTAP is MPL
code, and the separate process keeps both outside the MIT code base. These helpers split a
configured command into arguments, run it with a timeout, and turn each failure to run into
`SolverError`.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from seeingmon.solvers.base import SolverError
from seeingmon.survey.geometry import FloatArray, tangent_basis, vector_to_radec

# Time that the process gets beyond its own CPU limit, to start up and write its output.
PROCESS_GRACE_S = 5.0


def split_command(command: str | Sequence[str]) -> list[str]:
    """Split a configured command into arguments.

    A string splits like a POSIX shell line, so `"astap"` and `"python /path/to/shim.py"` both
    work. A sequence passes through unchanged. An empty command raises `SolverError`.
    """
    parts = shlex.split(command) if isinstance(command, str) else [str(part) for part in command]
    if not parts:
        raise SolverError("the solver command is empty")
    return parts


def run_process(
    arguments: Sequence[str], *, timeout_s: float, cwd: Path
) -> subprocess.CompletedProcess[str]:
    """Run a command, capture its output, and raise `SolverError` if it cannot run or hangs."""
    try:
        return subprocess.run(
            list(arguments),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
            check=False,
        )
    except FileNotFoundError as error:
        raise SolverError(f"cannot run the solver: {arguments[0]!r} is not installed") from error
    except subprocess.TimeoutExpired as error:
        raise SolverError(f"the solver did not finish within {timeout_s:.0f} s") from error
    except OSError as error:
        raise SolverError(f"cannot run the solver: {error.strerror or error}") from error


def output_tail(result: subprocess.CompletedProcess[str], limit: int = 300) -> str:
    """The last characters of a process's error output, on one line, for an error message."""
    text = (result.stderr or result.stdout or "").strip()
    return " ".join(text.split())[-limit:] or "no output"


def tan_center(
    crval_ra_deg: float,
    crval_dec_deg: float,
    cd_matrix: tuple[float, float, float, float],
    offset_px: tuple[float, float],
) -> tuple[float, float]:
    """The sky position of a pixel that lies `offset_px` from the reference pixel of a TAN WCS.

    `cd_matrix` is `(CD1_1, CD1_2, CD2_1, CD2_2)` in degrees per pixel and `offset_px` is the
    pixel offset `(dx, dy)` from the reference pixel. The result is `(ra_deg, dec_deg)`.
    """
    cd = np.asarray(cd_matrix, dtype=np.float64).reshape(2, 2) * (np.pi / 180.0)
    xi, eta = cd @ np.asarray(offset_px, dtype=np.float64)
    east, north, outward = tangent_basis(crval_ra_deg, crval_dec_deg)
    direction: FloatArray = outward + xi * east + eta * north
    ra, dec = vector_to_radec(direction / np.linalg.norm(direction))
    return float(ra), float(dec)
