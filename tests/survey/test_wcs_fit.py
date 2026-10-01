"""The camera model and the pointing fit, checked against `astropy.wcs` and a known truth."""

from __future__ import annotations

import numpy as np
import pytest

from seeingmon.survey import apparent
from seeingmon.survey import wcs_fit as wf
from seeingmon.survey.geometry import (
    ARCSEC_PER_RAD,
    FloatArray,
    angular_separation,
    exp_so3,
    radec_to_vector,
    vector_to_radec,
)
from tests.survey import synth

CENTER = (2071.5, 1410.5)  # the center of a 4144 x 2822 frame
SCALE_ARCSEC = 3.8203  # the bin2 plate scale of the reference profile, about
SHAPE = (2822, 4144)


def attitude(
    polar_distance_deg: float = 0.9,
    azimuth_deg: float = 40.0,
    roll_deg: float = 25.0,
    *,
    parity: int = 1,
    scale_arcsec: float = SCALE_ARCSEC,
    era: float = 1.0,
) -> wf.CameraAttitude:
    """A camera model for the apparent frame at an Earth rotation angle `era`."""
    tirs = synth.make_attitude(polar_distance_deg, azimuth_deg, roll_deg)
    from seeingmon.survey.geometry import rot_z

    return wf.CameraAttitude(
        rotation=tirs @ rot_z(-era),
        scale_rad_px=scale_arcsec / ARCSEC_PER_RAD,
        parity=parity,
        center_px=CENTER,
    )


def random_vectors_in_field(model: wf.CameraAttitude, count: int, seed: int) -> FloatArray:
    """Random sky directions that fall inside the frame."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, SHAPE[1] - 1.0, count)
    y = rng.uniform(0.0, SHAPE[0] - 1.0, count)
    return model.unproject(x, y)


def pointing_error_px(fitted: wf.CameraAttitude, truth: wf.CameraAttitude) -> tuple[float, float]:
    """The RMS and the largest displacement in pixels between two models, over a grid."""
    gx, gy = np.meshgrid(np.linspace(0.0, SHAPE[1] - 1.0, 9), np.linspace(0.0, SHAPE[0] - 1.0, 7))
    vectors = truth.unproject(gx.ravel(), gy.ravel())
    x, y, _ = fitted.project(vectors)
    shift = np.hypot(x - gx.ravel(), y - gy.ravel())
    return float(np.sqrt(np.mean(shift**2))), float(shift.max())


# --- The model ---------------------------------------------------------------------------


@pytest.mark.parametrize("parity", [1, -1])
@pytest.mark.parametrize(
    ("polar_distance", "roll"), [(0.9, 25.0), (0.0, 0.0), (2.8, -140.0), (0.0005, 90.0)]
)
def test_project_and_unproject_are_inverses(
    parity: int, polar_distance: float, roll: float
) -> None:
    model = attitude(polar_distance, 40.0, roll, parity=parity)
    vectors = random_vectors_in_field(model, 200, seed=1)
    x, y, front = model.project(vectors)
    assert front.all()
    again = model.unproject(x, y)
    assert float(np.max(angular_separation(vectors, again))) * ARCSEC_PER_RAD < 1e-6
    # The principal point is the boresight.
    cx, cy, _ = model.project(model.boresight())
    assert (float(cx[0]), float(cy[0])) == pytest.approx(CENTER, abs=1e-9)


def test_a_star_behind_the_camera_is_not_in_front() -> None:
    model = attitude()
    _, _, front = model.project(-model.boresight())
    assert not front[0]


def test_the_parity_flips_the_y_axis_only() -> None:
    plus = attitude(parity=1)
    minus = attitude(parity=-1)
    vectors = random_vectors_in_field(plus, 50, seed=2)
    x1, y1, _ = plus.project(vectors)
    x2, y2, _ = minus.project(vectors)
    np.testing.assert_allclose(x2, x1)
    np.testing.assert_allclose(y2 - CENTER[1], -(y1 - CENTER[1]), atol=1e-9)


@pytest.mark.parametrize("parity", [1, -1])
@pytest.mark.parametrize(
    ("polar_distance", "azimuth", "roll"),
    [(0.9, 40.0, 25.0), (0.0, 0.0, 0.0), (2.8, 200.0, -140.0), (45.0, 10.0, 170.0)],
)
def test_the_tan_wcs_matches_astropy(
    parity: int, polar_distance: float, azimuth: float, roll: float
) -> None:
    astropy_wcs = pytest.importorskip("astropy.wcs")
    model = attitude(polar_distance, azimuth, roll, parity=parity)
    ra, dec, cd = wf.attitude_to_tan_wcs(model)
    wcs = astropy_wcs.WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [CENTER[0] + 1.0, CENTER[1] + 1.0]  # FITS counts pixels from 1
    # The north that continues the meridian through a boresight on the pole is LONPOLE = 180,
    # which WCSLIB uses by default for any declination below 90 degrees.
    wcs.wcs.lonpole = 180.0
    wcs.wcs.cd = [[cd[0], cd[1]], [cd[2], cd[3]]]
    vectors = random_vectors_in_field(model, 300, seed=3)
    sky_ra, sky_dec = vector_to_radec(vectors)
    truth_x, truth_y = wcs.all_world2pix(sky_ra, sky_dec, 0)
    x, y, _ = model.project(vectors)
    np.testing.assert_allclose(x, truth_x, atol=2e-6)
    np.testing.assert_allclose(y, truth_y, atol=2e-6)
    # The scale of the FITS matrix is the plate scale.
    assert float(np.sqrt(abs(np.linalg.det(np.array(cd).reshape(2, 2))))) * 3600.0 == pytest.approx(
        SCALE_ARCSEC, rel=1e-9
    )


@pytest.mark.parametrize("parity", [1, -1])
def test_a_tan_wcs_round_trips_to_the_same_attitude(parity: int) -> None:
    model = attitude(1.3, 70.0, -33.0, parity=parity)
    ra, dec, cd = wf.attitude_to_tan_wcs(model)
    back = wf.attitude_from_tan_wcs(ra, dec, cd, center_px=CENTER)
    np.testing.assert_allclose(back.rotation, model.rotation, atol=1e-12)
    assert back.scale_rad_px == pytest.approx(model.scale_rad_px, rel=1e-12)
    assert back.parity == model.parity


def test_the_tan_wcs_conversion_works_for_a_boresight_on_the_pole() -> None:
    model = attitude(0.0, 0.0, 30.0)
    ra, dec, cd = wf.attitude_to_tan_wcs(model)
    back = wf.attitude_from_tan_wcs(ra, dec, cd, center_px=CENTER)
    vectors = random_vectors_in_field(model, 100, seed=4)
    x1, y1, _ = model.project(vectors)
    x2, y2, _ = back.project(vectors)
    np.testing.assert_allclose(x1, x2, atol=1e-6)
    np.testing.assert_allclose(y1, y2, atol=1e-6)


def test_a_singular_cd_matrix_is_refused() -> None:
    with pytest.raises(ValueError, match="singular"):
        wf.attitude_from_tan_wcs(10.0, 89.0, (1e-3, 0.0, 2e-3, 0.0), center_px=CENTER)


def test_invalid_models_are_refused() -> None:
    with pytest.raises(ValueError, match="parity"):
        wf.CameraAttitude(np.eye(3), 1e-5, 2, CENTER)
    with pytest.raises(ValueError, match="scale"):
        wf.CameraAttitude(np.eye(3), 0.0, 1, CENTER)


# --- Roll, pole, and center ---------------------------------------------------------------


@pytest.mark.parametrize("roll", [0.0, 25.0, 90.0, 179.0, -90.0, -150.0])
@pytest.mark.parametrize("parity", [1, -1])
def test_the_roll_is_the_position_angle_of_the_pole(roll: float, parity: int) -> None:
    # `make_attitude` builds the camera with the pole at this position angle (for parity +1).
    model = attitude(0.9, 40.0, roll, parity=parity)
    measured = model.roll_deg()
    assert measured is not None
    expected = roll if parity == 1 else 180.0 - roll  # a mirror reverses the sense of the angle
    assert ((measured - expected + 180.0) % 360.0) - 180.0 == pytest.approx(0.0, abs=1e-8)


def test_the_roll_does_not_change_with_the_earth_rotation() -> None:
    # A camera fixed to the Earth turns about the pole, so the direction to the pole stays.
    rolls = [attitude(0.9, 40.0, 25.0, era=era).roll_deg() for era in (0.0, 1.0, 2.5, 5.0)]
    assert all(roll is not None for roll in rolls)
    np.testing.assert_allclose([float(r) for r in rolls if r is not None], 25.0, atol=1e-8)


def test_the_pole_is_where_the_geometry_puts_it() -> None:
    model = attitude(0.9, 40.0, 0.0)
    pole = model.pole_pixel()
    assert pole is not None
    distance = model.pole_distance_px()
    assert distance == pytest.approx(0.9 * 3600.0 / SCALE_ARCSEC, rel=2e-4)
    # With roll 0, the pole is straight up: the same column, a smaller row.
    assert pole[0] == pytest.approx(CENTER[0], abs=1e-6)
    assert pole[1] < CENTER[1]


def test_the_roll_is_undefined_when_the_center_is_on_the_pole() -> None:
    on_pole = attitude(0.0, 0.0, 30.0)
    assert on_pole.roll_deg() is None
    assert on_pole.pole_distance_px() == pytest.approx(0.0, abs=1e-6)
    # 0.0005 degrees is 0.47 pixel: still undefined. 0.002 degrees is 1.9 pixels: defined.
    assert attitude(0.0005, 0.0, 30.0).roll_deg() is None
    assert attitude(0.002, 0.0, 30.0).roll_deg() is not None
    # The attitude itself stays a valid rotation there.
    np.testing.assert_allclose(on_pole.rotation @ on_pole.rotation.T, np.eye(3), atol=1e-12)


def test_the_center_in_icrs_removes_precession_and_aberration() -> None:
    t = synth.NIGHT_UTC_NS
    epoch = apparent.epoch_from_utc_ns(t)
    # A boresight on the ICRS direction (RA 30, Dec 88) must come back as that direction.
    target = radec_to_vector(30.0, 88.0)
    zeros = np.zeros(1)
    cirs = apparent.apparent_vectors(30.0, 88.0, 0.0, 0.0, 0.0, epoch)
    # The camera attitude with boresight = that apparent direction.
    z = cirs / np.linalg.norm(cirs)
    x_axis = np.cross([0.0, 0.0, 1.0], z)
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.stack([x_axis, np.cross(z, x_axis), z])
    model = wf.CameraAttitude(rotation, SCALE_ARCSEC / ARCSEC_PER_RAD, 1, CENTER)
    ra, dec = model.center_icrs(epoch)
    assert float(angular_separation(radec_to_vector(ra, dec), target)) * ARCSEC_PER_RAD < 1e-4
    del zeros


# --- The fit ---------------------------------------------------------------------------


def truth_and_detections(
    count: int = 300,
    seed: int = 5,
    *,
    noise_px: float = 0.05,
    parity: int = 1,
    polar_distance: float = 0.9,
) -> tuple[wf.CameraAttitude, FloatArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    truth = attitude(polar_distance, 40.0, 25.0, parity=parity)
    vectors = random_vectors_in_field(truth, count, seed=seed)
    x, y, _ = truth.project(vectors)
    return (
        truth,
        vectors,
        x + rng.normal(0, noise_px, count),
        y + rng.normal(0, noise_px, count),
        np.full(count, noise_px),
    )


def perturbed(model: wf.CameraAttitude, shift_px: float, scale_error: float) -> wf.CameraAttitude:
    """The model moved by about `shift_px` pixels, rolled by 0.05 degree, and with a wrong scale."""
    delta = np.array([shift_px, -0.6 * shift_px, 0.0]) * model.scale_rad_px
    roll = np.array([0.0, 0.0, np.radians(0.05)])
    return model.with_rotation(
        exp_so3(delta + roll) @ model.rotation, model.scale_rad_px * (1.0 + scale_error)
    )


@pytest.mark.parametrize("parity", [1, -1])
def test_the_fit_recovers_the_pointing_from_a_perturbed_start(parity: int) -> None:
    truth, vectors, x, y, error = truth_and_detections(parity=parity)
    initial = perturbed(truth, shift_px=3.0, scale_error=1.5e-3)
    result = wf.fit_attitude(initial, vectors, x, y, error, shape=SHAPE)
    assert result is not None
    assert result.converged
    rms, worst = pointing_error_px(result.attitude, truth)
    assert worst < 0.02  # pixels, over the whole field, with 300 stars of 0.05 pixel noise
    assert result.n_matched == 300
    assert result.rms_px == pytest.approx(0.05 * np.sqrt(2.0), rel=0.15)
    assert result.rms_arcsec == pytest.approx(result.rms_px * SCALE_ARCSEC, rel=1e-3)
    assert rms < 0.01


def test_the_fit_survives_outliers_missing_stars_and_false_detections() -> None:
    truth, vectors, x, y, error = truth_and_detections(count=400, seed=6)
    rng = np.random.default_rng(7)
    # Lose 30% of the detections, move 8% of the rest by a few pixels, and add false ones.
    keep = rng.random(400) > 0.3
    x, y, error = x[keep].copy(), y[keep].copy(), error[keep].copy()
    bad = rng.random(x.size) < 0.08
    x[bad] += rng.uniform(1.5, 3.5, bad.sum())
    y[bad] -= rng.uniform(1.5, 3.5, bad.sum())
    false_x = rng.uniform(0, SHAPE[1], 150)
    false_y = rng.uniform(0, SHAPE[0], 150)
    x = np.concatenate([x, false_x])
    y = np.concatenate([y, false_y])
    error = np.concatenate([error, np.full(150, 0.05)])
    initial = perturbed(truth, shift_px=2.0, scale_error=1e-3)
    result = wf.fit_attitude(initial, vectors, x, y, error, shape=SHAPE)
    assert result is not None
    _, worst = pointing_error_px(result.attitude, truth)
    assert worst < 0.01
    # The clip dropped the moved stars, so the residual stays at the noise level.
    assert result.rms_px < 0.12
    assert 200 < result.n_matched < 280


def test_the_fit_works_for_a_boresight_on_the_pole() -> None:
    truth, vectors, x, y, error = truth_and_detections(polar_distance=0.0, seed=8)
    result = wf.fit_attitude(perturbed(truth, 2.0, 1e-3), vectors, x, y, error, shape=SHAPE)
    assert result is not None
    _, worst = pointing_error_px(result.attitude, truth)
    assert worst < 0.02
    assert result.attitude.roll_deg() is None


def test_the_fit_needs_the_minimum_number_of_stars() -> None:
    truth, vectors, x, y, error = truth_and_detections(count=6, seed=9)
    assert wf.fit_attitude(truth, vectors, x, y, error, shape=SHAPE) is not None
    assert wf.fit_attitude(truth, vectors[:3], x[:3], y[:3], error[:3], shape=SHAPE) is None
    assert wf.fit_attitude(truth, vectors, x[:0], y[:0], error[:0], shape=SHAPE) is None


def test_a_start_that_is_too_far_off_does_not_produce_a_wrong_answer() -> None:
    truth, vectors, x, y, error = truth_and_detections(seed=10)
    far = perturbed(truth, shift_px=60.0, scale_error=0.0)
    result = wf.fit_attitude(far, vectors, x, y, error, shape=SHAPE)
    assert (
        result is None or pointing_error_px(result.attitude, truth)[1] > 5.0 or result.n_matched < 6
    )


def test_a_detection_goes_to_the_closer_of_two_catalog_stars() -> None:
    truth = attitude()
    vectors = random_vectors_in_field(truth, 40, seed=11)
    x, y, _ = truth.project(vectors)
    # Add a catalog star 1.5 pixels from star 0 and a detection at star 0.
    near = truth.unproject([x[0] + 1.5], [y[0]])
    all_vectors = np.vstack([vectors, near])
    rng = np.random.default_rng(12)
    result = wf.fit_attitude(
        truth, all_vectors, x + rng.normal(0, 0.02, 40), y + rng.normal(0, 0.02, 40),
        np.full(40, 0.02), shape=SHAPE,
    )  # fmt: skip
    assert result is not None
    assert 40 not in set(result.catalog_index)  # the farther star did not claim the detection
    assert result.n_matched == 40


def test_the_catalog_usable_mask_keeps_stars_out_of_the_fit() -> None:
    truth, vectors, x, y, error = truth_and_detections(count=100, seed=13)
    usable = np.arange(100) < 50
    result = wf.fit_attitude(truth, vectors, x, y, error, shape=SHAPE, catalog_usable=usable)
    assert result is not None
    assert result.n_matched == 50
    assert np.all(result.catalog_index < 50)


# --- From a solver's solution -------------------------------------------------------------


def test_a_solver_solution_in_the_catalog_frame_becomes_the_apparent_attitude() -> None:
    profile = synth.cropped_profile(1200, 800)
    catalog = synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=2.0, seed=2)
    rotation = synth.make_attitude(0.9, 40.0, 25.0)
    truth, _, _ = synth.star_truth(catalog, profile, rotation_tirs=rotation)
    result = synth.truth_solve_result(truth, catalog)
    assert result.cd_matrix is not None
    assert result.center_ra_deg is not None
    assert result.center_dec_deg is not None
    near = truth.rows
    start = wf.attitude_from_solver_solution(
        result.center_ra_deg,
        result.center_dec_deg,
        result.cd_matrix,
        center_px=truth.center_px,
        catalog_vectors_icrs=catalog.vectors[near],
        apparent_vectors=truth.vectors_cirs[near],
    )
    truth_model = wf.CameraAttitude(
        truth.rotation_cirs, truth.scale_arcsec_px / ARCSEC_PER_RAD, truth.parity, truth.center_px
    )
    # The solver's linear fit and the differential aberration leave a fraction of a pixel.
    gx, gy = np.meshgrid(np.linspace(0, 1199, 7), np.linspace(0, 799, 5))
    vectors = truth_model.unproject(gx.ravel(), gy.ravel())
    x, y, _ = start.project(vectors)
    assert float(np.max(np.hypot(x - gx.ravel(), y - gy.ravel()))) < 1.0
    assert start.parity == 1
    assert start.scale_arcsec_px == pytest.approx(truth.scale_arcsec_px, rel=2e-3)


def test_the_fit_uses_astropy_apparent_places_as_the_truth() -> None:
    # Render the truth positions with apparent places from astropy, and fit with ours.
    catalog = synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=2.0, seed=3)
    t = synth.NIGHT_UTC_NS
    astropy_vectors = synth.astropy_apparent_vectors(catalog, t)
    epoch = apparent.epoch_from_utc_ns(t)
    mine = apparent.apparent_vectors(
        catalog.ra_deg,
        catalog.dec_deg,
        catalog.pm_ra_mas_yr,
        catalog.pm_dec_mas_yr,
        catalog.parallax_mas,
        epoch,
    )
    rotation_tirs = synth.make_attitude(0.9, 40.0, 25.0)
    rotation_cirs = rotation_tirs @ apparent.cirs_to_earth_fixed(epoch.era_rad)
    truth = wf.CameraAttitude(rotation_cirs, SCALE_ARCSEC / ARCSEC_PER_RAD, 1, CENTER)
    x, y, front = truth.project(astropy_vectors)
    inside = front & (x > 0) & (x < SHAPE[1] - 1) & (y > 0) & (y < SHAPE[0] - 1)
    assert inside.sum() > 150
    rng = np.random.default_rng(14)
    result = wf.fit_attitude(
        perturbed(truth, 2.0, 1e-3),
        mine,
        x[inside] + rng.normal(0, 0.03, inside.sum()),
        y[inside] + rng.normal(0, 0.03, inside.sum()),
        np.full(int(inside.sum()), 0.03),
        shape=SHAPE,
    )
    assert result is not None
    _, worst = pointing_error_px(result.attitude, truth)
    assert worst < 0.005
