"""The simulated sky: the small profile, the field and the catalog of one seed, and the seed fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.clock import iso_to_utc_ns
from seeingmon.drivers.sim.params import SimParams
from seeingmon.drivers.sim.stars import Pointing, SkyProjector, make_polar_field
from seeingmon.profile import load_profile
from seeingmon.services.simsky import (
    MIN_SEED_STARS,
    polaris_rows,
    read_seed,
    seed_solution,
    sim_catalog,
    sim_field,
    write_seed,
    write_small_profile,
)
from seeingmon.survey import apparent
from seeingmon.survey.catalog_build import propagate
from seeingmon.survey.geometry import vector_to_radec
from seeingmon.survey.tracker import PointingTracker

START = iso_to_utc_ns("2026-01-01T19:00:00Z")


class TestTheSmallProfile:
    def test_it_has_a_small_sensor_and_the_real_optics(self, tmp_path: Path) -> None:
        small = load_profile(str(write_small_profile(tmp_path)))
        reference = load_profile("asi294mm-gs250")
        assert small.id == "sim-small"
        assert (small.mode("bin2").width_px, small.mode("bin2").height_px) == (640, 480)
        assert (small.mode("bin1").width_px, small.mode("bin1").height_px) == (1280, 960)
        # The field is smaller, and the plate scale of each pixel is the real one.
        assert small.plate_scale_arcsec_per_px(small.mode("bin2")) == pytest.approx(
            reference.plate_scale_arcsec_per_px(reference.mode("bin2"))
        )

    def test_another_size_and_name_are_possible(self, tmp_path: Path) -> None:
        path = write_small_profile(tmp_path, "sim-tiny", (320, 240))
        assert path.name == "sim-tiny.toml"
        assert load_profile(str(path)).mode("bin1").width_px == 640


class TestTheField:
    def test_polaris_and_its_companion_are_the_last_rows_of_the_field(self) -> None:
        plain = make_polar_field(1)
        rows = polaris_rows(plain)
        assert len(rows) == 2
        assert plain.mag[rows[0]] == pytest.approx(2.02, abs=0.01)  # Polaris
        assert plain.mag[rows[1]] > 8.0  # Polaris B
        assert plain.mag.min() < plain.mag[rows[0]]  # a random star is brighter: it is not Polaris

    def test_polaris_sits_where_the_survey_predicts_it(self) -> None:
        field = sim_field(1)
        polaris = int(polaris_rows(field)[0])
        star = apparent.POLARIS
        moved = propagate(
            np.array([star.ra_deg]),
            np.array([star.dec_deg]),
            np.array([star.pm_ra_mas_yr]),
            np.array([star.pm_dec_mas_yr]),
            apparent.CATALOG_EPOCH_JYEAR - star.epoch_jyear,
        )
        ra, dec = vector_to_radec(moved)
        assert field.ra_deg[polaris] == pytest.approx(float(ra[0]), abs=1e-9)
        assert field.dec_deg[polaris] == pytest.approx(float(dec[0]), abs=1e-9)

    def test_only_polaris_and_its_companion_move(self) -> None:
        plain = make_polar_field(1)
        field = sim_field(1)
        others = np.ones(len(plain.mag), dtype=bool)
        others[polaris_rows(plain)] = False
        assert np.array_equal(plain.ra_deg[others], field.ra_deg[others])
        assert np.array_equal(plain.dec_deg[others], field.dec_deg[others])
        assert np.array_equal(plain.mag, field.mag)  # no dimming without polaris_mag
        rows = polaris_rows(plain)
        moved = np.hypot(
            field.ra_deg[rows] - plain.ra_deg[rows], field.dec_deg[rows] - plain.dec_deg[rows]
        )
        assert np.all(moved > 0.0)

    def test_the_companion_keeps_its_place_relative_to_polaris(self) -> None:
        plain = make_polar_field(1)
        field = sim_field(1)
        a, b = polaris_rows(plain)
        for name in ("ra_deg", "dec_deg"):
            before = getattr(plain, name)[b] - getattr(plain, name)[a]
            after = getattr(field, name)[b] - getattr(field, name)[a]
            assert after == pytest.approx(before, abs=1e-12)

    def test_a_dimmer_polaris_and_its_companion_keep_their_difference(self) -> None:
        plain = sim_field(1)
        dim = sim_field(1, polaris_mag=6.0)
        a, b = polaris_rows(plain)
        assert dim.mag[a] == pytest.approx(6.0)
        assert dim.mag[b] - dim.mag[a] == pytest.approx(plain.mag[b] - plain.mag[a])
        others = np.ones(len(plain.mag), dtype=bool)
        others[[a, b]] = False
        assert np.array_equal(dim.mag[others], plain.mag[others])

    def test_one_seed_gives_one_sky(self) -> None:
        assert np.array_equal(sim_field(3).ra_deg, sim_field(3).ra_deg)
        assert not np.array_equal(sim_field(3).ra_deg, sim_field(4).ra_deg)


class TestTheCatalog:
    def test_every_simulated_star_is_a_catalog_star(self) -> None:
        catalog, field = sim_catalog(1)
        assert len(catalog) == len(field.mag)
        assert catalog.cap_radius_deg == pytest.approx(15.0)

    def test_the_catalog_is_the_same_for_the_same_seed(self) -> None:
        first, _ = sim_catalog(2)
        second, _ = sim_catalog(2)
        assert first.content_id == second.content_id

    def test_only_polaris_has_a_proper_motion(self) -> None:
        catalog, _ = sim_catalog(1)
        moving = np.flatnonzero(catalog.pm_ra_mas_yr != 0.0)
        assert len(moving) == 1
        assert catalog.g_mag[moving[0]] == pytest.approx(2.02, abs=0.01)


class TestTheSeedSolution:
    def test_the_fit_reproduces_the_pixels_of_the_simulator(self, tmp_path: Path) -> None:
        profile = load_profile(str(write_small_profile(tmp_path)))
        field = sim_field(1, polaris_mag=6.0)
        fit = seed_solution(profile, field, Pointing(t_ref_utc_ns=START), START)
        assert fit.n_stars >= MIN_SEED_STARS
        assert fit.rms_px < 0.3
        assert fit.parity in (1, -1)
        assert fit.solution.mode == "bin2"

    def test_the_tracker_then_finds_the_simulated_polaris(self, tmp_path: Path) -> None:
        profile = load_profile(str(write_small_profile(tmp_path)))
        field = sim_field(1, polaris_mag=6.0)
        pointing = Pointing(t_ref_utc_ns=START)
        fit = seed_solution(profile, field, pointing, START)
        tracker = PointingTracker(profile)
        tracker.update(fit.solution)
        params = SimParams.modes_from_profile(profile)["bin2"]
        projector = SkyProjector(field, pointing)
        x, y = projector.project(START, params.pixel_rad, params.width, params.height)
        polaris = int(np.flatnonzero(projector.indices == polaris_rows(field)[0])[0])
        found = tracker.polaris_position(START, "bin2")
        assert found is not None
        assert found[0] == pytest.approx(float(x[polaris]), abs=0.5)
        assert found[1] == pytest.approx(float(y[polaris]), abs=0.5)

    def test_a_pointing_away_from_the_field_is_refused(self, tmp_path: Path) -> None:
        profile = load_profile(str(write_small_profile(tmp_path)))
        away = Pointing(dec_deg=60.0, t_ref_utc_ns=START)  # the pole and its stars are outside
        with pytest.raises(ValueError, match="fall inside the frame"):
            seed_solution(profile, sim_field(1), away, START)

    def test_the_solution_survives_the_trip_through_a_file(self, tmp_path: Path) -> None:
        profile = load_profile(str(write_small_profile(tmp_path)))
        fit = seed_solution(profile, sim_field(1), Pointing(t_ref_utc_ns=START), START)
        path = tmp_path / "seed.json"
        write_seed(path, fit.solution)
        before, after = fit.solution.to_dict(), read_seed(path).to_dict()
        assert after.keys() == before.keys()
        for name, value in before.items():
            if isinstance(value, list):
                assert after[name] == pytest.approx(value, abs=1e-12)
            else:
                assert after[name] == value
