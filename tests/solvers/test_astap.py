"""The ASTAP adapter, tested against a script that stands in for `astap`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seeingmon.solvers import astap as astap_module
from seeingmon.solvers.astap import AstapSolver, draw_star_image
from seeingmon.solvers.base import PlateSolver, SolverError
from tests.solvers.helpers import (
    HEIGHT,
    WIDTH,
    configure_shim,
    make_request,
    read_log,
    shim_command,
    star_list,
)

RENDER_SCALE = 4
SCALE_DEG = 3.82 / 3600.0
# A canned solution in the pixels of the drawn image (4 times coarser than the frame).
RENDER_CD = (-SCALE_DEG * RENDER_SCALE, 0.0, 0.0, SCALE_DEG * RENDER_SCALE)
SOLUTION_INI = [
    "PLTSOLVD=T",
    "CRPIX1=518.0",
    "CRPIX2=353.5",
    "CRVAL1=40.0",
    "CRVAL2=89.2",
    f"CD1_1={RENDER_CD[0]}",
    f"CD1_2={RENDER_CD[1]}",
    f"CD2_1={RENDER_CD[2]}",
    f"CD2_2={RENDER_CD[3]}",
    "CMDLINE=astap -f field.fits",
]


def solver(tmp_path: Path, **options: object) -> AstapSolver:
    return AstapSolver(
        command=shim_command("shim_astap.py"),
        work_dir=tmp_path,
        render_scale=RENDER_SCALE,
        **options,  # type: ignore[arg-type]
    )


def test_the_adapter_is_a_plate_solver(tmp_path: Path) -> None:
    adapter = solver(tmp_path)
    assert isinstance(adapter, PlateSolver)
    assert adapter.name == "astap"


def test_the_drawn_image_puts_each_star_where_the_frame_has_it() -> None:
    stars = star_list(8, seed=1)
    image = draw_star_image(stars.x, stars.y, stars.flux, width=WIDTH, height=HEIGHT, scale=4)
    assert image.shape == (HEIGHT // 4 + 1, WIDTH // 4)  # 2822 / 4 rounds up
    assert image.dtype == np.uint16
    for x, y in zip(stars.x, stars.y, strict=True):
        cx, cy = (x + 0.5) / 4 - 0.5, (y + 0.5) / 4 - 0.5
        window = image[round(cy) - 6 : round(cy) + 7, round(cx) - 6 : round(cx) + 7].astype(float)
        rows, columns = np.mgrid[round(cy) - 6 : round(cy) + 7, round(cx) - 6 : round(cx) + 7]
        weight = window - 500.0
        assert float((weight * columns).sum() / weight.sum()) == pytest.approx(cx, abs=0.05)
        assert float((weight * rows).sum() / weight.sum()) == pytest.approx(cy, abs=0.05)
    assert int(image.max()) <= 60_000 + 100  # no star saturates
    assert float(np.median(image)) == pytest.approx(500.0, abs=2.0)


def test_the_drawn_image_is_deterministic_and_handles_no_stars() -> None:
    stars = star_list(5, seed=2)
    first = draw_star_image(stars.x, stars.y, stars.flux, width=800, height=600, scale=2)
    second = draw_star_image(stars.x, stars.y, stars.flux, width=800, height=600, scale=2)
    np.testing.assert_array_equal(first, second)
    empty = draw_star_image(np.array([]), np.array([]), np.array([]), width=80, height=60, scale=2)
    assert empty.shape == (30, 40)


def test_brighter_stars_are_drawn_brighter_and_all_stay_detectable() -> None:
    image = draw_star_image(
        np.array([100.0, 200.0]),
        np.array([100.0, 100.0]),
        np.array([1e5, 1e2]),
        width=400,
        height=200,
        scale=1,
    )
    bright, faint = int(image[100, 100]), int(image[100, 200])
    assert bright > faint > 2_000 + 500 - 100  # even the faintest star is far above the noise
    assert bright < 61_000


def test_the_command_line_carries_the_field_the_hint_and_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = configure_shim(tmp_path, monkeypatch, {"ini": SOLUTION_INI})
    request = make_request(
        star_list(),
        center_ra_deg=37.5,
        center_dec_deg=89.0,
        radius_deg=3.0,
        scale_low_arcsec_px=3.5,
        scale_high_arcsec_px=4.1,
    )
    solver(tmp_path, database_dir="/data/astap", database="d50").solve(request)
    seen = read_log(log)
    options = seen["options"]
    assert float(options["-fov"]) == pytest.approx(HEIGHT * 3.8 / 3600.0, abs=1e-4)
    assert options["-z"] == "1"
    assert float(options["-ra"]) == pytest.approx(37.5 / 15.0)
    assert float(options["-spd"]) == pytest.approx(179.0)
    assert float(options["-r"]) == pytest.approx(3.0)
    assert options["-d"] == "/data/astap"
    assert options["-D"] == "d50"
    assert "-wcs" in seen["flags"]
    assert Path(options["-f"]).name == "field.fits"
    assert seen["shape"] == [HEIGHT // RENDER_SCALE + 1, WIDTH // RENDER_SCALE]
    assert seen["dtype"] == "uint16"


def test_a_request_without_a_hint_or_a_database_leaves_those_options_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = configure_shim(tmp_path, monkeypatch, {"ini": SOLUTION_INI})
    solver(tmp_path).solve(make_request(star_list()))
    options = read_log(log)["options"]
    assert not {"-ra", "-spd", "-r", "-d", "-D"} & set(options)


def test_the_image_holds_the_stars_at_the_scaled_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = configure_shim(tmp_path, monkeypatch, {"ini": SOLUTION_INI})
    stars = star_list(12, seed=7)
    solver(tmp_path).solve(make_request(stars))
    peaks = np.array(read_log(log)["peaks"])
    expected = np.column_stack([(stars.x[:5] + 0.5) / 4 - 0.5, (stars.y[:5] + 0.5) / 4 - 0.5])
    # The five brightest peaks are the five brightest stars, to within a pixel of the drawn image.
    for peak in peaks:
        assert np.min(np.hypot(expected[:, 0] - peak[0], expected[:, 1] - peak[1])) < 1.0


def test_the_solution_is_converted_to_the_pixels_of_the_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    astropy_wcs = pytest.importorskip("astropy.wcs")
    configure_shim(tmp_path, monkeypatch, {"ini": SOLUTION_INI})
    result = solver(tmp_path).solve(make_request(star_list()))
    assert result.solved
    assert result.solver == "astap"
    assert result.scale_arcsec_px == pytest.approx(3.82, rel=1e-9)
    assert result.cd_matrix == pytest.approx((-SCALE_DEG, 0.0, 0.0, SCALE_DEG))
    assert result.n_matched == 0
    assert result.matched == ()
    wcs = astropy_wcs.WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [40.0, 89.2]
    wcs.wcs.crpix = [518.0, 353.5]
    wcs.wcs.cd = [[RENDER_CD[0], RENDER_CD[1]], [RENDER_CD[2], RENDER_CD[3]]]
    # Frame pixel (x, y) is drawn pixel ((x + 0.5) / 4 - 0.5, (y + 0.5) / 4 - 0.5).
    cx, cy = (WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0
    ra, dec = wcs.all_pix2world([(cx + 0.5) / 4 - 0.5], [(cy + 0.5) / 4 - 0.5], 0)
    assert result.center_ra_deg == pytest.approx(float(ra[0]), abs=1e-8)
    assert result.center_dec_deg == pytest.approx(float(dec[0]), abs=1e-8)


def test_the_solution_can_come_from_the_wcs_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = {
        "CRPIX1": 518.0,
        "CRPIX2": 353.5,
        "CRVAL1": 40.0,
        "CRVAL2": 89.2,
        "CD1_1": RENDER_CD[0],
        "CD1_2": RENDER_CD[1],
        "CD2_1": RENDER_CD[2],
        "CD2_2": RENDER_CD[3],
    }
    configure_shim(tmp_path, monkeypatch, {"wcs": cards})
    from_header = solver(tmp_path).solve(make_request(star_list()))
    configure_shim(tmp_path, monkeypatch, {"ini": SOLUTION_INI})
    from_ini = solver(tmp_path).solve(make_request(star_list()))
    assert from_header.solved
    assert from_header.center_ra_deg == pytest.approx(from_ini.center_ra_deg)
    assert from_header.cd_matrix == pytest.approx(from_ini.cd_matrix)


def test_a_solution_with_cdelt_and_crota_gives_the_same_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rotation = 30.0
    rho = np.radians(rotation)
    cdelt1, cdelt2 = -SCALE_DEG * RENDER_SCALE, SCALE_DEG * RENDER_SCALE
    ini = [
        "PLTSOLVD=T",
        "CRPIX1=518.0",
        "CRPIX2=353.5",
        "CRVAL1=40.0",
        "CRVAL2=89.2",
        f"CDELT1={cdelt1}",
        f"CDELT2={cdelt2}",
        f"CROTA2={rotation}",
    ]
    configure_shim(tmp_path, monkeypatch, {"ini": ini})
    result = solver(tmp_path).solve(make_request(star_list()))
    assert result.cd_matrix == pytest.approx(
        (
            cdelt1 * np.cos(rho) / RENDER_SCALE,
            -cdelt2 * np.sin(rho) / RENDER_SCALE,
            cdelt1 * np.sin(rho) / RENDER_SCALE,
            cdelt2 * np.cos(rho) / RENDER_SCALE,
        )
    )


@pytest.mark.parametrize("code", [1, 2])
def test_no_solution_and_too_few_stars_are_normal_results(
    code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_shim(tmp_path, monkeypatch, {"exit_code": code})
    result = solver(tmp_path).solve(make_request(star_list()))
    assert not result.solved
    assert result.solver == "astap"


def test_a_solution_file_that_says_unsolved_is_unsolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_shim(tmp_path, monkeypatch, {"ini": ["PLTSOLVD=F"]})
    assert not solver(tmp_path).solve(make_request(star_list())).solved


@pytest.mark.parametrize("code", [16, 32, 33])
def test_a_process_that_cannot_run_raises(
    code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_shim(tmp_path, monkeypatch, {"exit_code": code, "message": "no star database"})
    with pytest.raises(SolverError, match=f"code {code}: no star database"):
        solver(tmp_path).solve(make_request(star_list()))


def test_a_solution_without_a_reference_or_a_scale_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_shim(tmp_path, monkeypatch, {"ini": ["PLTSOLVD=T", "CRVAL1=40.0"]})
    with pytest.raises(SolverError, match="without CRVAL2"):
        solver(tmp_path).solve(make_request(star_list()))
    no_scale = [line for line in SOLUTION_INI if not line.startswith("CD")]
    configure_shim(tmp_path, monkeypatch, {"ini": no_scale})
    with pytest.raises(SolverError, match="without CDELT1"):
        solver(tmp_path).solve(make_request(star_list()))
    singular = [line for line in SOLUTION_INI if not line.startswith("CD")]
    singular += ["CD1_1=0", "CD1_2=0", "CD2_1=0", "CD2_2=0"]
    configure_shim(tmp_path, monkeypatch, {"ini": singular})
    with pytest.raises(SolverError, match="singular"):
        solver(tmp_path).solve(make_request(star_list()))


def test_a_missing_program_and_too_few_stars_and_a_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SolverError, match="not installed"):
        AstapSolver(command="no-such-astap-seeingmon", work_dir=tmp_path).solve(
            make_request(star_list())
        )
    assert (
        not AstapSolver(command="no-such-astap-seeingmon").solve(make_request(star_list(3))).solved
    )
    configure_shim(tmp_path, monkeypatch, {"sleep_s": 30.0})
    monkeypatch.setattr("seeingmon.solvers.astap.PROCESS_GRACE_S", 0.0)
    with pytest.raises(SolverError, match="did not finish within 2 s"):
        solver(tmp_path).solve(make_request(star_list(), timeout_s=2.0))
    assert list(tmp_path.glob("seeingmon-astap-*")) == []  # the temporary folder is gone


def test_invalid_options_are_refused() -> None:
    with pytest.raises(ValueError, match="render_scale"):
        AstapSolver(render_scale=0)
    with pytest.raises(ValueError, match="max_stars"):
        AstapSolver(max_stars=3)
    assert astap_module.AstapSolver(command=["astap"]).name == "astap"
