"""Pointing solutions: the Earth-fixed attitude, offsets, references, and the record."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S
from seeingmon.records.survey import POINTING_FLAGS, PointingRecord
from seeingmon.survey import apparent
from seeingmon.survey import pointing as pt
from seeingmon.survey.geometry import (
    ARCMIN_PER_RAD,
    ARCSEC_PER_RAD,
    angular_separation,
    exp_so3,
    radec_to_vector,
)
from seeingmon.survey.wcs_fit import CameraAttitude
from tests.survey import synth

T0 = synth.NIGHT_UTC_NS
SCALE_RAD = 3.82 / ARCSEC_PER_RAD
CENTER = (2071.5, 1410.5)


def make_solution(
    polar_distance: float = 0.9, roll: float = 25.0, t_utc_ns: int = T0, parity: int = 1
) -> tuple[pt.PointingSolution, CameraAttitude]:
    epoch = apparent.epoch_from_utc_ns(t_utc_ns)
    rotation_tirs = synth.make_attitude(polar_distance, 40.0, roll)
    attitude = CameraAttitude(
        rotation_tirs @ apparent.cirs_to_earth_fixed(epoch.era_rad), SCALE_RAD, parity, CENTER
    )
    solution = pt.PointingSolution.from_attitude(
        attitude, epoch, mode="bin2", width_px=4144, height_px=2822, n_matched=300,
        rms_arcsec=0.2, solver="test",
    )  # fmt: skip
    return solution, attitude


def test_a_solution_returns_its_own_attitude_at_its_own_time() -> None:
    solution, attitude = make_solution()
    again = solution.attitude_at(T0)
    np.testing.assert_allclose(again.rotation, attitude.rotation, atol=1e-12)
    assert again.scale_rad_px == attitude.scale_rad_px
    assert solution.rotation_earth_fixed.shape == (3, 3)
    np.testing.assert_allclose(
        solution.rotation_earth_fixed @ solution.rotation_earth_fixed.T, np.eye(3), atol=1e-12
    )


def test_the_boresight_pole_and_roll_stay_fixed_while_the_sky_turns() -> None:
    solution, _ = make_solution()
    later = [solution.attitude_at(T0 + minutes * 60 * NS_PER_S) for minutes in (0, 7, 61, 600)]
    poles = [model.pole_pixel() for model in later]
    assert all(pole is not None for pole in poles)
    for pole in poles:
        assert pole == pytest.approx(poles[0], abs=1e-6)  # the pole does not move in the image
    rolls = [model.roll_deg() for model in later]
    np.testing.assert_allclose([float(r) for r in rolls if r is not None], 25.0, atol=1e-8)
    # The boresight in the Earth-fixed frame is the same for every time.
    np.testing.assert_allclose(
        solution.boresight_earth_fixed(), solution.rotation_earth_fixed[2], atol=1e-15
    )


def test_a_star_moves_at_the_sidereal_rate_times_the_sine_of_its_pole_distance() -> None:
    solution, _ = make_solution(polar_distance=2.5)
    epoch = apparent.epoch_from_utc_ns(T0)
    star = apparent.apparent_vectors(0.0, 87.0, 0.0, 0.0, 0.0, epoch)  # 3 degrees from the pole
    seconds = 20
    x0, y0, _ = solution.attitude_at(T0).project(star)
    # Keep the star fixed in the apparent frame, as a real star is, while the camera turns.
    x1, y1, _ = solution.attitude_at(T0 + seconds * NS_PER_S).project(star)
    moved_px = float(np.hypot(x1[0] - x0[0], y1[0] - y0[0]))
    polar = float(angular_separation(star, [0.0, 0.0, 1.0]))
    expected_arcsec = 15.0411 * seconds * np.sin(polar)
    assert moved_px * 3.82 == pytest.approx(expected_arcsec, rel=2e-3)


def test_polaris_follows_the_apparent_place() -> None:
    solution, attitude = make_solution()
    t = T0 + 90 * NS_PER_S
    pixel = solution.polaris_pixel(t)
    assert pixel is not None
    epoch = apparent.epoch_from_utc_ns(t)
    vector = apparent.apparent_vectors_for(apparent.POLARIS, epoch)
    x, y, _ = solution.attitude_at(t).project(vector)
    assert pixel == pytest.approx((float(x[0]), float(y[0])), abs=1e-9)
    # Polaris sits about 0.6 degree from the pole, so it lies near the pole pixel.
    pole = solution.attitude_at(t).pole_pixel()
    assert pole is not None
    assert np.hypot(pixel[0] - pole[0], pixel[1] - pole[1]) * 3.82 / 3600.0 == pytest.approx(
        0.62, abs=0.03
    )
    del attitude


def test_polaris_is_none_behind_the_camera() -> None:
    solution, _ = make_solution(polar_distance=170.0)
    assert solution.polaris_pixel(T0) is None


def test_the_offset_between_equal_solutions_is_zero() -> None:
    solution, _ = make_solution()
    offset = pt.offset_between(solution, solution)
    assert offset.boresight_arcmin == pytest.approx(0.0, abs=1e-6)
    assert offset.roll_deg == pytest.approx(0.0, abs=1e-9)


def test_the_offset_measures_a_tilt_and_a_twist() -> None:
    reference, _ = make_solution()
    # Tilt the camera by 2 arcmin about its x axis and twist it by 0.3 degree about its axis.
    tilt = exp_so3([2.0 / ARCMIN_PER_RAD, 0.0, 0.0])
    twist = exp_so3([0.0, 0.0, np.radians(0.3)])
    moved = pt.PointingSolution(
        rotation_earth_fixed=twist @ tilt @ reference.rotation_earth_fixed,
        scale_rad_px=reference.scale_rad_px,
        parity=1,
        mode="bin2",
        width_px=4144,
        height_px=2822,
        center_px=CENTER,
        t_utc_ns=T0 + 3600 * NS_PER_S,
    )
    offset = pt.offset_between(moved, reference)
    assert offset.boresight_arcmin == pytest.approx(2.0, rel=1e-6)
    assert offset.roll_deg == pytest.approx(0.3, rel=1e-3)


def test_the_offset_works_when_the_boresight_is_on_the_pole() -> None:
    reference, _ = make_solution(polar_distance=0.0)
    twist = exp_so3([0.0, 0.0, np.radians(1.0)])
    moved = pt.PointingSolution(
        twist @ reference.rotation_earth_fixed, SCALE_RAD, 1, "bin2", 4144, 2822, CENTER, T0
    )
    offset = pt.offset_between(moved, reference)
    assert offset.boresight_arcmin == pytest.approx(0.0, abs=1e-6)
    assert offset.roll_deg == pytest.approx(1.0, rel=1e-6)


def test_a_reference_survives_json_and_a_file(tmp_path: Path) -> None:
    solution, _ = make_solution(parity=-1)
    reference = pt.ReferenceSolution("commissioning-1", solution)
    path = tmp_path / "reference.json"
    pt.save_reference(path, reference)
    assert not path.with_name("reference.json.tmp").exists()
    loaded = pt.load_reference(path)
    assert loaded is not None
    assert loaded.reference_id == "commissioning-1"
    np.testing.assert_allclose(
        loaded.solution.rotation_earth_fixed, solution.rotation_earth_fixed, atol=1e-15
    )
    assert loaded.solution.parity == -1
    assert loaded.solution.t_utc_ns == T0
    assert loaded.solution.rms_arcsec == 0.2
    assert pt.load_reference(tmp_path / "missing.json") is None


def test_a_solution_with_an_unknown_format_is_refused() -> None:
    solution, _ = make_solution()
    data = solution.to_dict()
    data["format"] = 99
    with pytest.raises(ValueError, match="format"):
        pt.PointingSolution.from_dict(data)


def record(
    *,
    solved: bool = True,
    solution: pt.PointingSolution | None = None,
    attitude: CameraAttitude | None = None,
    n_matched: int = 300,
    rms_arcsec: float | None = 0.2,
    polaris_xy: tuple[float, float] | None = (2000.0, 1000.0),
    reference: pt.ReferenceSolution | None = None,
    time_invalid: bool = False,
    limits: pt.PointingLimits | None = None,
) -> PointingRecord:
    default_solution, default_attitude = make_solution()
    return pt.build_pointing_record(
        station_id="test",
        profile_id="asi294mm-gs250",
        t_utc_ns=T0,
        mode="bin2",
        solver="astrometry.net",
        provenance={"algo": pt.POINTING_ALGORITHM},
        attitude=(attitude or default_attitude) if solved else None,
        epoch=apparent.epoch_from_utc_ns(T0) if solved else None,
        solution=(solution or default_solution) if solved else None,
        n_matched=n_matched,
        rms_arcsec=rms_arcsec,
        focus_fwhm_px=1.1,
        solve_time_s=0.7,
        polaris_xy=polaris_xy,
        reference=reference,
        time_invalid=time_invalid,
        limits=limits,
    )


def test_a_solved_record_carries_the_geometry() -> None:
    rec = record()
    assert rec.flags == []
    assert rec.center_ra_deg is not None
    assert 0.0 <= rec.center_ra_deg < 360.0
    assert rec.center_dec_deg == pytest.approx(89.5, abs=1.0)
    assert rec.roll_deg == pytest.approx(25.0, abs=1e-6)
    assert rec.plate_scale_arcsec_px == pytest.approx(3.82)
    assert rec.attitude is not None
    matrix = np.array(rec.attitude).reshape(3, 3)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert (rec.n_matched, rec.focus_fwhm_px, rec.solve_rms_arcsec) == (300, 1.1, 0.2)
    assert (rec.polaris_x_px, rec.polaris_y_px) == (2000.0, 1000.0)
    assert rec.solver == "astrometry.net"
    assert rec.reference_id is None
    assert rec.quality == {"offset_arcmin": "no reference solution"}


def test_a_record_survives_the_row_round_trip() -> None:
    rec = record()
    assert PointingRecord.from_row(rec.to_row()) == rec


def test_the_record_center_is_the_icrs_direction_of_the_boresight() -> None:
    solution, attitude = make_solution()
    rec = record()
    epoch = apparent.epoch_from_utc_ns(T0)
    boresight = apparent.astrometric_from_apparent(attitude.boresight(), epoch)
    assert rec.center_ra_deg is not None
    assert rec.center_dec_deg is not None
    center = radec_to_vector(rec.center_ra_deg, rec.center_dec_deg)
    assert float(angular_separation(center, boresight)) * ARCSEC_PER_RAD < 1e-6
    del solution


def test_a_reference_gives_an_offset_and_a_moved_flag() -> None:
    solution, _ = make_solution()
    reference, _ = make_solution(t_utc_ns=T0 - 3600 * NS_PER_S)
    same = record(reference=pt.ReferenceSolution("ref-1", reference))
    assert same.offset_arcmin == pytest.approx(0.0, abs=1e-5)
    assert same.reference_id == "ref-1"
    assert same.flags == []
    assert same.quality is None
    moved_reference = pt.PointingSolution(
        exp_so3([10.0 / ARCMIN_PER_RAD, 0.0, 0.0]) @ reference.rotation_earth_fixed,
        SCALE_RAD, 1, "bin2", 4144, 2822, CENTER, T0 - 3600 * NS_PER_S,
    )  # fmt: skip
    moved = record(reference=pt.ReferenceSolution("ref-2", moved_reference))
    assert moved.offset_arcmin == pytest.approx(10.0, rel=1e-5)
    assert moved.flags == ["moved"]
    twisted_reference = pt.PointingSolution(
        exp_so3([0.0, 0.0, np.radians(1.0)]) @ solution.rotation_earth_fixed,
        SCALE_RAD, 1, "bin2", 4144, 2822, CENTER, T0 - 3600 * NS_PER_S,
    )  # fmt: skip
    assert record(reference=pt.ReferenceSolution("ref-3", twisted_reference)).flags == ["moved"]


def test_the_limits_set_the_flags() -> None:
    assert record(n_matched=11).flags == ["few_stars"]
    assert record(n_matched=12).flags == []
    assert record(n_matched=11, limits=pt.PointingLimits(few_stars=5)).flags == []
    assert record(time_invalid=True).flags == ["time_invalid"]


def test_a_boresight_on_the_pole_has_no_roll_and_says_so() -> None:
    solution, attitude = make_solution(polar_distance=0.0)
    rec = record(solution=solution, attitude=attitude)
    assert rec.roll_deg is None
    assert rec.flags == ["roll_undefined"]
    assert rec.quality is not None
    assert "pole" in rec.quality["roll_deg"]


def test_polaris_behind_the_camera_is_a_quality_note() -> None:
    rec = record(polaris_xy=None)
    assert rec.polaris_x_px is None
    assert rec.quality is not None
    assert "Polaris" in rec.quality["polaris_x_px"]


def test_an_unsolved_record_has_null_geometry_and_the_unsolved_flag() -> None:
    rec = record(solved=False, n_matched=0, rms_arcsec=None)
    assert rec.flags == ["unsolved"]
    for name in (
        "center_ra_deg",
        "center_dec_deg",
        "roll_deg",
        "plate_scale_arcsec_px",
        "attitude",
        "offset_arcmin",
        "solve_rms_arcsec",
        "polaris_x_px",
        "polaris_y_px",
    ):
        assert getattr(rec, name) is None
        assert rec.quality is not None
        assert name in rec.quality
    assert rec.focus_fwhm_px == 1.1  # the focus can exist without a solution
    assert set(rec.flags) <= set(POINTING_FLAGS)


def test_an_unsolved_record_can_carry_the_time_flag_too() -> None:
    rec = record(solved=False, time_invalid=True)
    assert rec.flags == ["unsolved", "time_invalid"]
