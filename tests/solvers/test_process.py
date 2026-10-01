"""The helpers that the solver adapters share."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from seeingmon.solvers import process
from seeingmon.solvers.base import SolverError


def test_a_command_string_splits_like_a_shell_line() -> None:
    assert process.split_command("astap") == ["astap"]
    assert process.split_command("python '/tmp/some dir/shim.py' --flag") == [
        "python",
        "/tmp/some dir/shim.py",
        "--flag",
    ]
    assert process.split_command(["solve-field", "--verbose"]) == ["solve-field", "--verbose"]
    with pytest.raises(SolverError, match="empty"):
        process.split_command("")
    with pytest.raises(SolverError, match="empty"):
        process.split_command([])


def test_run_process_captures_the_output(tmp_path: Path) -> None:
    result = process.run_process(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
        timeout_s=30.0,
        cwd=tmp_path,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"


def test_run_process_reports_a_missing_program_and_a_timeout(tmp_path: Path) -> None:
    with pytest.raises(SolverError, match="'no-such-program-seeingmon' is not installed"):
        process.run_process(["no-such-program-seeingmon"], timeout_s=5.0, cwd=tmp_path)
    with pytest.raises(SolverError, match="did not finish within 1 s"):
        process.run_process(
            [sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=1.0, cwd=tmp_path
        )


def test_the_output_tail_keeps_the_end_on_one_line() -> None:
    result = subprocess.CompletedProcess(
        ["x"], 1, stdout="", stderr="line one\nline two\n" + "z" * 400
    )
    tail = process.output_tail(result, limit=50)
    assert "\n" not in tail
    assert len(tail) == 50
    assert tail.endswith("z")
    empty = subprocess.CompletedProcess(["x"], 1, stdout="", stderr="")
    assert process.output_tail(empty) == "no output"
    only_stdout = subprocess.CompletedProcess(["x"], 1, stdout="from stdout", stderr="")
    assert process.output_tail(only_stdout) == "from stdout"


@pytest.mark.parametrize(("ra", "dec"), [(40.0, 89.2), (0.0, 90.0), (200.0, 60.0), (359.0, -30.0)])
def test_tan_center_matches_astropy(ra: float, dec: float) -> None:
    astropy_wcs = pytest.importorskip("astropy.wcs")
    scale = 3.82 / 3600.0
    cd = (-scale * 0.8, scale * 0.6, scale * 0.6, scale * 0.8)
    offsets = [(10.0, -20.0), (-1500.0, 800.0), (0.0, 0.0), (2000.5, 1400.25)]
    wcs = astropy_wcs.WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [100.0, 200.0]
    wcs.wcs.cd = [[cd[0], cd[1]], [cd[2], cd[3]]]
    wcs.wcs.lonpole = 180.0  # the convention that continues the meridian through the pole
    for dx, dy in offsets:
        expected_ra, expected_dec = wcs.all_pix2world([100.0 + dx - 1.0], [200.0 + dy - 1.0], 0)
        got_ra, got_dec = process.tan_center(ra, dec, cd, (dx, dy))
        assert got_dec == pytest.approx(float(expected_dec[0]), abs=1e-9)
        difference = ((got_ra - float(expected_ra[0]) + 180.0) % 360.0) - 180.0
        assert np.isfinite(difference)
        if got_dec < 89.999:
            assert difference == pytest.approx(0.0, abs=1e-8)
