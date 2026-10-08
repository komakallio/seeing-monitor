"""The triangle solver: a blind solve of a synthetic field near the pole, for either parity."""

from __future__ import annotations

import numpy as np
import pytest

from seeingmon.solvers.base import PlateSolver, SolveRequest, SolveResult, StarList
from seeingmon.solvers.triangles import MIN_STARS, TriangleSolver
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.geometry import radec_to_vector, tangent_basis
from tests.survey.synth import make_attitude, project, synthetic_catalog

WIDTH, HEIGHT = 4144, 2822
CENTER = ((WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0)
SCALE = 3.82  # arcseconds per pixel


@pytest.fixture(scope="module")
def catalog() -> CapCatalog:
    return synthetic_catalog(cap_radius_deg=9.0, seed=11)


@pytest.fixture(scope="module")
def solver(catalog: CapCatalog) -> TriangleSolver:
    triangles = TriangleSolver(catalog)
    triangles.prepare()
    return triangles


def field(
    catalog: CapCatalog,
    *,
    polar_distance_deg: float,
    azimuth_deg: float,
    roll_deg: float,
    parity: int,
    seed: int = 0,
    keep: float = 1.0,
    false_stars: int = 0,
    noise_px: float = 0.4,
) -> tuple[StarList, np.ndarray]:
    """The stars that a camera would detect, brightest first, and the true boresight vector."""
    rotation = make_attitude(polar_distance_deg, azimuth_deg, roll_deg)
    x, y, front = project(rotation, catalog.vectors, SCALE, parity, CENTER)
    inside = front & (x > 5) & (x < WIDTH - 6) & (y > 5) & (y < HEIGHT - 6)
    rows = np.flatnonzero(inside)
    rows = rows[np.argsort(catalog.g_mag[rows])][:300]  # the brightest 300, as the detector keeps
    rng = np.random.default_rng(seed)
    rows = rows[rng.uniform(size=rows.size) < keep]
    px = x[rows] + rng.normal(0.0, noise_px, rows.size)
    py = y[rows] + rng.normal(0.0, noise_px, rows.size)
    flux = 10.0 ** (-0.4 * catalog.g_mag[rows]) * 1e6
    if false_stars:
        px = np.append(px, rng.uniform(10, WIDTH - 10, false_stars))
        py = np.append(py, rng.uniform(10, HEIGHT - 10, false_stars))
        flux = np.append(flux, np.full(false_stars, flux.max() * 0.9))  # as bright as the best
    order = np.argsort(-flux, kind="stable")
    boresight = rotation.T @ np.array([0.0, 0.0, 1.0])
    return StarList(x=px[order], y=py[order], flux=flux[order]), boresight


def request(stars: StarList, *, tolerance: float = 0.15) -> SolveRequest:
    return SolveRequest(
        stars=stars,
        width_px=WIDTH,
        height_px=HEIGHT,
        scale_low_arcsec_px=SCALE * (1.0 - tolerance),
        scale_high_arcsec_px=SCALE * (1.0 + tolerance),
        center_ra_deg=0.0,
        center_dec_deg=90.0,
        radius_deg=15.0,
    )


def pixel_of(result: SolveResult, catalog: CapCatalog, rows: np.ndarray) -> np.ndarray:
    """Where the solution puts catalog stars in the frame, as `(n, 2)` pixels."""
    assert result.cd_matrix is not None
    assert result.center_ra_deg is not None
    assert result.center_dec_deg is not None
    east, north, outward = tangent_basis(result.center_ra_deg, result.center_dec_deg)
    vectors = catalog.vectors[rows]
    depth = vectors @ outward
    plane = np.stack([(vectors @ east) / depth, (vectors @ north) / depth], axis=1)
    cd = np.asarray(result.cd_matrix).reshape(2, 2) * (np.pi / 180.0)
    return np.asarray(plane @ np.linalg.inv(cd).T + np.array(CENTER))


def test_the_solver_is_a_plate_solver(solver: TriangleSolver) -> None:
    assert isinstance(solver, PlateSolver)
    assert solver.name == "triangles"


@pytest.mark.parametrize("parity", [1, -1])
@pytest.mark.parametrize(
    ("polar_distance_deg", "azimuth_deg", "roll_deg"),
    [(0.3, 40.0, 25.0), (2.0, 200.0, -80.0), (6.5, 310.0, 170.0)],
)
def test_it_solves_a_field_with_no_hint_of_the_roll(
    catalog: CapCatalog,
    solver: TriangleSolver,
    parity: int,
    polar_distance_deg: float,
    azimuth_deg: float,
    roll_deg: float,
) -> None:
    stars, boresight = field(
        catalog,
        polar_distance_deg=polar_distance_deg,
        azimuth_deg=azimuth_deg,
        roll_deg=roll_deg,
        parity=parity,
    )
    result = solver.solve(request(stars))
    assert result.solved
    assert result.solver == "triangles"
    assert result.center_ra_deg is not None
    assert result.center_dec_deg is not None
    separation = np.degrees(
        np.arccos(
            np.clip(
                np.dot(boresight, radec_to_vector(result.center_ra_deg, result.center_dec_deg)),
                -1.0,
                1.0,
            )
        )
    )
    assert separation < 0.02
    assert result.scale_arcsec_px == pytest.approx(SCALE, rel=0.01)
    assert result.n_matched >= 30
    assert result.elapsed_s < 2.0


def test_the_solution_puts_the_catalog_stars_on_the_detections(
    catalog: CapCatalog, solver: TriangleSolver
) -> None:
    """The returned CD matrix carries the right parity and roll: stars land within a pixel."""
    for parity in (1, -1):
        rotation = make_attitude(1.0, 120.0, 33.0)
        x, y, front = project(rotation, catalog.vectors, SCALE, parity, CENTER)
        stars, _ = field(
            catalog, polar_distance_deg=1.0, azimuth_deg=120.0, roll_deg=33.0, parity=parity
        )
        result = solver.solve(request(stars))
        assert result.solved
        inside = np.flatnonzero(front & (x > 50) & (x < WIDTH - 51) & (y > 50) & (y < HEIGHT - 51))[
            :200
        ]
        predicted = pixel_of(result, catalog, inside)
        error = np.hypot(predicted[:, 0] - x[inside], predicted[:, 1] - y[inside])
        assert np.median(error) < 1.0
        assert np.percentile(error, 90) < 3.0


def test_it_ignores_false_stars_and_stars_that_the_detector_missed(
    catalog: CapCatalog, solver: TriangleSolver
) -> None:
    stars, _ = field(
        catalog,
        polar_distance_deg=3.0,
        azimuth_deg=75.0,
        roll_deg=-20.0,
        parity=-1,
        seed=5,
        keep=0.6,
        false_stars=4,
        noise_px=0.8,
    )
    result = solver.solve(request(stars))
    assert result.solved
    assert result.n_matched >= 20
    assert result.rms_arcsec is not None
    assert result.rms_arcsec < 3 * SCALE


def test_the_matched_indices_point_into_the_request(
    catalog: CapCatalog, solver: TriangleSolver
) -> None:
    stars, _ = field(
        catalog, polar_distance_deg=2.5, azimuth_deg=10.0, roll_deg=5.0, parity=1, seed=2
    )
    result = solver.solve(request(stars))
    assert result.solved
    assert len(result.matched) == result.n_matched
    assert max(result.matched) < len(stars)
    assert len(set(result.matched)) == len(result.matched)


def test_stars_that_match_nothing_give_no_solution(solver: TriangleSolver) -> None:
    rng = np.random.default_rng(3)
    n = 60
    flux = np.sort(rng.uniform(1.0, 100.0, n))[::-1]
    stars = StarList(x=rng.uniform(0, WIDTH, n), y=rng.uniform(0, HEIGHT, n), flux=flux)
    result = solver.solve(request(stars))
    assert not result.solved
    assert result.cd_matrix is None


def test_too_few_stars_give_no_solution(catalog: CapCatalog, solver: TriangleSolver) -> None:
    stars, _ = field(catalog, polar_distance_deg=1.0, azimuth_deg=0.0, roll_deg=0.0, parity=1)
    few = StarList(
        x=stars.x[: MIN_STARS - 1], y=stars.y[: MIN_STARS - 1], flux=stars.flux[: MIN_STARS - 1]
    )
    assert not solver.solve(request(few)).solved


def test_a_plate_scale_outside_the_allowed_range_gives_no_solution(
    catalog: CapCatalog, solver: TriangleSolver
) -> None:
    stars, _ = field(catalog, polar_distance_deg=1.0, azimuth_deg=30.0, roll_deg=10.0, parity=1)
    narrow = SolveRequest(
        stars=stars,
        width_px=WIDTH,
        height_px=HEIGHT,
        scale_low_arcsec_px=2.0,
        scale_high_arcsec_px=2.5,
    )
    assert not solver.solve(narrow).solved
