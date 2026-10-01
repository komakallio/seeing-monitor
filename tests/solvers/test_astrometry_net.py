"""The astrometry.net adapter, tested against a script that stands in for `solve-field`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, ClockStatus
from seeingmon.solvers import fitsio
from seeingmon.solvers import process as solver_process
from seeingmon.solvers.astrometry_net import AstrometryNetSolver
from seeingmon.solvers.base import PlateSolver, SolverError
from tests.solvers.helpers import (
    HEIGHT,
    WIDTH,
    configure_shim,
    make_index_dir,
    make_request,
    read_log,
    shim_command,
    star_list,
)

CENTER_FITS = ((WIDTH + 1) / 2.0, (HEIGHT + 1) / 2.0)
# A canned solution: the frame center at (RA 40, Dec 89.2), 3.82 arcsec per pixel, rotated.
SCALE_DEG = 3.82 / 3600.0
ROTATION = np.radians(30.0)
CD = (
    -SCALE_DEG * np.cos(ROTATION),
    SCALE_DEG * np.sin(ROTATION),
    SCALE_DEG * np.sin(ROTATION),
    SCALE_DEG * np.cos(ROTATION),
)
WCS_AT_CENTER = {
    "CRVAL1": 40.0,
    "CRVAL2": 89.2,
    "CRPIX1": CENTER_FITS[0],
    "CRPIX2": CENTER_FITS[1],
    "CD1_1": CD[0],
    "CD1_2": CD[1],
    "CD2_1": CD[2],
    "CD2_2": CD[3],
}


def solver(tmp_path: Path, **options: object) -> AstrometryNetSolver:
    return AstrometryNetSolver(
        make_index_dir(tmp_path),
        command=shim_command("shim_solve_field.py"),
        work_dir=tmp_path,
        **options,  # type: ignore[arg-type]
    )


def test_the_adapter_is_a_plate_solver(tmp_path: Path) -> None:
    adapter = solver(tmp_path)
    assert isinstance(adapter, PlateSolver)
    assert adapter.name == "astrometry.net"


def test_the_command_line_carries_the_scale_the_hint_and_the_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = configure_shim(tmp_path, monkeypatch, {"wcs": WCS_AT_CENTER})
    request = make_request(
        star_list(),
        center_ra_deg=40.0,
        center_dec_deg=89.2,
        radius_deg=2.5,
        timeout_s=12.0,
        scale_low_arcsec_px=3.5,
        scale_high_arcsec_px=4.1,
    )
    solver(tmp_path).solve(request)
    seen = read_log(log)["parsed"]
    assert seen["scale_units"] == "arcsecperpix"
    assert (seen["scale_low"], seen["scale_high"]) == ("3.5", "4.1")
    assert (seen["ra"], seen["dec"], seen["radius"]) == ("40.000000", "89.200000", "2.5000")
    assert seen["cpulimit"] == "12"
    assert (seen["width"], seen["height"]) == (str(WIDTH), str(HEIGHT))
    assert (seen["x_column"], seen["y_column"], seen["sort_column"]) == ("X", "Y", "FLUX")
    assert seen["crpix_center"] is True
    assert seen["no_plots"] is True
    assert seen["overwrite"] is True
    assert seen["no_remove_lines"] is True
    assert seen["uniformize"] == "0"
    assert seen["out"] == "field"
    assert read_log(log)["extra"] == []  # no option that the shim did not expect


def test_a_request_without_a_hint_leaves_out_the_position_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = configure_shim(tmp_path, monkeypatch, {"wcs": WCS_AT_CENTER})
    solver(tmp_path).solve(make_request(star_list()))
    seen = read_log(log)["parsed"]
    assert seen["ra"] is None
    assert seen["dec"] is None
    assert seen["radius"] is None


def test_the_backend_config_names_only_the_cap_index_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = configure_shim(tmp_path, monkeypatch, {"wcs": WCS_AT_CENTER})
    adapter = solver(tmp_path)
    adapter.solve(make_request(star_list(), timeout_s=20.0))
    lines = read_log(log)["config"].splitlines()
    index_dir = (tmp_path / "index").resolve().as_posix()
    assert lines[0] == f"add_path {index_dir}"
    assert f"index {index_dir}/index-cap-08.fits" in lines
    assert f"index {index_dir}/index-cap-10.fits" in lines
    assert "cpulimit 20" in lines
    assert not any(line.startswith("autoindex") for line in lines)


def test_the_star_list_goes_in_as_a_one_based_xylist_sorted_by_flux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = configure_shim(tmp_path, monkeypatch, {"wcs": WCS_AT_CENTER})
    stars = star_list(30, seed=3)
    solver(tmp_path).solve(make_request(stars))
    seen = read_log(log)
    assert seen["n_rows"] == 30
    assert seen["x_head"] == pytest.approx(list(stars.x[:3] + 1.0))
    assert seen["y_head"] == pytest.approx(list(stars.y[:3] + 1.0))
    assert seen["flux_head"] == pytest.approx(list(stars.flux[:3]))
    assert seen["flux_sorted_descending"] is True
    assert (seen["imagew"], seen["imageh"]) == (WIDTH, HEIGHT)


def test_an_unsorted_star_list_is_sorted_and_cut_to_the_brightest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = configure_shim(tmp_path, monkeypatch, {"wcs": WCS_AT_CENTER})
    stars = star_list(50, seed=4)
    shuffled = type(stars)(
        x=stars.x[::-1].copy(), y=stars.y[::-1].copy(), flux=stars.flux[::-1].copy()
    )
    solver(tmp_path, max_stars=20).solve(make_request(shuffled))
    seen = read_log(log)
    assert seen["n_rows"] == 20
    assert seen["flux_sorted_descending"] is True
    assert seen["flux_head"][0] == pytest.approx(float(stars.flux.max()))


def test_a_solution_centered_on_the_frame_becomes_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_shim(tmp_path, monkeypatch, {"wcs": WCS_AT_CENTER, "corr_rows": 0})
    result = solver(tmp_path).solve(make_request(star_list()))
    assert result.solved
    assert result.solver == "astrometry.net"
    assert result.center_ra_deg == pytest.approx(40.0, abs=1e-9)
    assert result.center_dec_deg == pytest.approx(89.2, abs=1e-9)
    assert result.scale_arcsec_px == pytest.approx(3.82, rel=1e-9)
    assert result.cd_matrix == pytest.approx(CD)
    assert result.n_matched == 0
    assert result.matched == ()
    assert result.rms_arcsec is None


def test_a_reference_pixel_away_from_the_center_is_moved_to_the_center(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    astropy_wcs = pytest.importorskip("astropy.wcs")
    off_center = dict(WCS_AT_CENTER, CRPIX1=1500.0, CRPIX2=2300.5)
    configure_shim(tmp_path, monkeypatch, {"wcs": off_center, "corr_rows": 0})
    result = solver(tmp_path).solve(make_request(star_list()))
    assert result.solved
    wcs = astropy_wcs.WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [40.0, 89.2]
    wcs.wcs.crpix = [1500.0, 2300.5]
    wcs.wcs.cd = [[CD[0], CD[1]], [CD[2], CD[3]]]
    # Astropy counts pixels from 0, and the center of the frame is ((W - 1) / 2, (H - 1) / 2).
    ra, dec = wcs.all_pix2world([(WIDTH - 1) / 2.0], [(HEIGHT - 1) / 2.0], 0)
    assert result.center_ra_deg == pytest.approx(float(ra[0]), abs=1e-9)
    assert result.center_dec_deg == pytest.approx(float(dec[0]), abs=1e-9)


def test_the_matched_stars_come_back_as_indices_into_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_shim(
        tmp_path, monkeypatch, {"wcs": WCS_AT_CENTER, "corr_rows": 6, "corr_offset_arcsec": 0.5}
    )
    stars = star_list(40, seed=5)
    shuffled_order = np.random.default_rng(1).permutation(40)
    request = make_request(
        type(stars)(
            x=stars.x[shuffled_order], y=stars.y[shuffled_order], flux=stars.flux[shuffled_order]
        )
    )
    result = solver(tmp_path).solve(request)
    brightest = np.argsort(-stars.flux)[:6]
    # The request holds the stars in shuffled order. The solver matched the six brightest.
    expected = tuple(int(i) for i in np.flatnonzero(np.isin(shuffled_order, brightest)))
    assert sorted(result.matched) == sorted(expected)
    assert result.n_matched == 6
    assert result.rms_arcsec == pytest.approx(0.5, rel=1e-3)


def test_no_solution_is_a_normal_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure_shim(tmp_path, monkeypatch, {"solved": False})
    result = solver(tmp_path).solve(make_request(star_list()))
    assert not result.solved
    assert result.solver == "astrometry.net"
    assert result.center_ra_deg is None
    assert result.cd_matrix is None


def test_too_few_stars_return_unsolved_without_a_process(tmp_path: Path) -> None:
    adapter = AstrometryNetSolver(
        make_index_dir(tmp_path), command=["no-such-solver-seeingmon"], work_dir=tmp_path
    )
    result = adapter.solve(make_request(star_list(3)))
    assert not result.solved
    assert result.elapsed_s == 0.0


def test_a_failing_process_raises_with_its_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_shim(tmp_path, monkeypatch, {"exit_code": 2, "message": "index file corrupted"})
    with pytest.raises(SolverError, match="exited with code 2: index file corrupted"):
        solver(tmp_path).solve(make_request(star_list()))


def test_a_missing_program_raises(tmp_path: Path) -> None:
    adapter = AstrometryNetSolver(
        make_index_dir(tmp_path), command="no-such-solver-seeingmon", work_dir=tmp_path
    )
    with pytest.raises(SolverError, match="not installed"):
        adapter.solve(make_request(star_list()))


def test_a_folder_without_index_files_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    adapter = AstrometryNetSolver(empty, command=shim_command("shim_solve_field.py"))
    with pytest.raises(SolverError, match="no index files"):
        adapter.solve(make_request(star_list()))


def test_a_process_that_hangs_is_killed_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_shim(tmp_path, monkeypatch, {"sleep_s": 30.0})
    monkeypatch.setattr(solver_process, "PROCESS_GRACE_S", 0.0)
    monkeypatch.setattr("seeingmon.solvers.astrometry_net.PROCESS_GRACE_S", 0.0)
    with pytest.raises(SolverError, match="did not finish within 2 s"):
        solver(tmp_path).solve(make_request(star_list(), timeout_s=2.0))


def test_an_unreadable_solution_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure_shim(tmp_path, monkeypatch, {"garbage_wcs": True})
    with pytest.raises(SolverError, match="cannot read the solution"):
        solver(tmp_path).solve(make_request(star_list()))
    configure_shim(tmp_path, monkeypatch, {"wcs": {"CRVAL1": 1.0, "CRVAL2": 2.0}})
    with pytest.raises(SolverError, match="cannot read the solution"):
        solver(tmp_path).solve(make_request(star_list()))
    singular = dict(WCS_AT_CENTER, CD1_1=0.0, CD1_2=0.0, CD2_1=0.0, CD2_2=0.0)
    configure_shim(tmp_path, monkeypatch, {"wcs": singular})
    with pytest.raises(SolverError, match="singular"):
        solver(tmp_path).solve(make_request(star_list()))


def test_the_temporary_folder_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure_shim(tmp_path, monkeypatch, {"wcs": WCS_AT_CENTER})
    solver(tmp_path).solve(make_request(star_list()))
    configure_shim(tmp_path, monkeypatch, {"exit_code": 1})
    with pytest.raises(SolverError):
        solver(tmp_path).solve(make_request(star_list()))
    assert list(tmp_path.glob("seeingmon-solve-*")) == []


def test_the_elapsed_time_comes_from_the_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_shim(tmp_path, monkeypatch, {"wcs": WCS_AT_CENTER})

    class SteppingClock:
        """A clock that moves 1.5 s each time anyone reads the monotonic time."""

        def __init__(self) -> None:
            self._ns = 0

        def utc_ns(self) -> int:
            return 0

        def monotonic_ns(self) -> int:
            self._ns += round(1.5 * NS_PER_S)
            return self._ns

        def sleep(self, seconds: float) -> None:
            return None

        def status(self) -> ClockStatus:
            return ClockStatus(synchronized=True, error_bound_ns=0, source="test")

    adapter = AstrometryNetSolver(
        make_index_dir(tmp_path),
        command=shim_command("shim_solve_field.py"),
        work_dir=tmp_path,
        clock=SteppingClock(),
    )
    assert adapter.solve(make_request(star_list())).elapsed_s == pytest.approx(1.5)


def test_a_mirrored_solution_keeps_its_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mirrored = dict(WCS_AT_CENTER, CD1_1=-CD[0], CD1_2=-CD[1])
    configure_shim(tmp_path, monkeypatch, {"wcs": mirrored, "corr_rows": 0})
    result = solver(tmp_path).solve(make_request(star_list()))
    assert result.cd_matrix is not None
    determinant = (
        result.cd_matrix[0] * result.cd_matrix[3] - result.cd_matrix[1] * result.cd_matrix[2]
    )
    assert determinant > 0.0  # the sign of the determinant is the parity


def test_invalid_options_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_stars"):
        AstrometryNetSolver(tmp_path, max_stars=3)
    with pytest.raises(SolverError, match="empty"):
        AstrometryNetSolver(tmp_path, command="")


def test_fits_files_written_by_the_adapter_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The shim reads the xylist with the same FITS code, so check the file with astropy too.
    fits = pytest.importorskip("astropy.io.fits")
    stars = star_list(10, seed=6)
    folder = tmp_path / "xy"
    folder.mkdir()
    fitsio.write_table(
        folder / "a.xyls",
        {"X": stars.x + 1.0, "Y": stars.y + 1.0, "FLUX": stars.flux},
        header={"IMAGEW": WIDTH, "IMAGEH": HEIGHT},
    )
    with fits.open(folder / "a.xyls") as hdus:
        assert hdus[1].header["IMAGEW"] == WIDTH
        np.testing.assert_allclose(hdus[1].data["X"], stars.x + 1.0)
    del monkeypatch
