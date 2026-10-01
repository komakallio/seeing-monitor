"""An end-to-end run of the real `solve-field`, for a machine that has astrometry.net.

The test builds a cap index from a synthetic catalog with `build-astrometry-index`, solves a
synthetic star list with `solve-field`, and compares the center with the truth. It needs both
programs on the path and runs only with `--slow`. Everything else in the solver tests uses a
script that stands in for the programs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from seeingmon.solvers.astrometry_net import AstrometryNetSolver
from seeingmon.solvers.base import SolveRequest, StarList
from seeingmon.survey.catalog_build import build_solver_index
from seeingmon.survey.geometry import ARCSEC_PER_RAD, angular_separation, radec_to_vector
from tests.survey import synth

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("solve-field") is None or shutil.which("build-astrometry-index") is None,
        reason="astrometry.net (solve-field and build-astrometry-index) is not installed",
    ),
]


def test_solve_field_solves_a_synthetic_star_list_with_a_cap_index(tmp_path: Path) -> None:
    profile = synth.reference_profile()
    catalog = synth.synthetic_catalog(cap_radius_deg=8.0, density_scale=2.0, seed=31)
    truth, _, _ = synth.star_truth(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        exposure_s=0.001,
    )
    tool = shutil.which("build-astrometry-index")
    assert tool is not None
    build_solver_index(catalog, tmp_path / "index", command=[tool], presets=(8, 9, 10, 11, 12))

    rng = np.random.default_rng(32)
    inside = (
        (truth.x > 5) & (truth.x < truth.width - 5) & (truth.y > 5) & (truth.y < truth.height - 5)
    )
    order = np.argsort(-truth.flux_e[inside])
    stars = StarList(
        x=truth.x[inside][order] + rng.normal(0.0, 0.1, order.size),
        y=truth.y[inside][order] + rng.normal(0.0, 0.1, order.size),
        flux=truth.flux_e[inside][order],
    )
    request = SolveRequest(
        stars=stars,
        width_px=truth.width,
        height_px=truth.height,
        scale_low_arcsec_px=0.9 * truth.scale_arcsec_px,
        scale_high_arcsec_px=1.1 * truth.scale_arcsec_px,
        timeout_s=120.0,
    )
    result = AstrometryNetSolver(tmp_path / "index").solve(request)

    assert result.solved
    expected = synth.truth_solve_result(truth, catalog)
    assert expected.center_ra_deg is not None
    assert expected.center_dec_deg is not None
    assert result.center_ra_deg is not None
    assert result.center_dec_deg is not None
    separation = angular_separation(
        radec_to_vector(result.center_ra_deg, result.center_dec_deg),
        radec_to_vector(expected.center_ra_deg, expected.center_dec_deg),
    )
    assert float(separation) * ARCSEC_PER_RAD / truth.scale_arcsec_px < 3.0  # pixels
    assert result.scale_arcsec_px == pytest.approx(truth.scale_arcsec_px, rel=0.01)
    assert result.n_matched > 20
