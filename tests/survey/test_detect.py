"""Star detection on synthetic frames: completeness, centroids, trails, saturation, hot pixels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.frames import Frame
from seeingmon.profile import Profile
from seeingmon.survey import _scipy, detect
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.centroid import FWHM_PER_SIGMA
from seeingmon.survey.detect import StarFlag
from seeingmon.survey.geometry import angular_separation
from seeingmon.survey.trail import TrailModel, analytic_trail_length_px
from tests.survey import synth

PSF_SIGMA = 0.45
EXPOSURE_S = 30.0


@dataclass
class Scene:
    profile: Profile
    catalog: CapCatalog
    frame_native: npt.NDArray[np.float32]  # ADC counts
    truth: synth.SynthTruth
    saturation_dn: float
    e_per_adu: float


def native_counts(frame: Frame) -> npt.NDArray[np.float32]:
    """The frame in ADC counts: the 16-bit container divided by 2^(16 - bits)."""
    scale = np.float32(2 ** (16 - frame.adc_bits))
    return np.asarray(np.asarray(frame.data, dtype=np.float32) / scale, dtype=np.float32)


def make_scene(
    *,
    seed: int = 1,
    polar_distance_deg: float = 0.9,
    exposure_s: float = EXPOSURE_S,
    gain: int = 120,
    psf_sigma_px: float = PSF_SIGMA,
    **render_options: object,
) -> Scene:
    profile = synth.cropped_profile(1200, 800)
    catalog = synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=3.0, seed=seed)
    rotation = synth.make_attitude(polar_distance_deg, 40.0, 25.0)
    frame, truth = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=rotation,
        exposure_s=exposure_s,
        gain=gain,
        psf_sigma_px=psf_sigma_px,
        seed=seed + 2,
        **render_options,  # type: ignore[arg-type]
    )
    return Scene(
        profile=profile,
        catalog=catalog,
        frame_native=native_counts(frame),
        truth=truth,
        saturation_dn=profile.saturation("bin2", gain).native_dn,
        e_per_adu=profile.e_per_adu("bin2", gain),
    )


@pytest.fixture(scope="module")
def scene() -> Scene:
    return make_scene()


@pytest.fixture(scope="module")
def detections(scene: Scene) -> detect.Detections:
    return detect.detect_stars(
        scene.frame_native, saturation_dn=scene.saturation_dn, e_per_adu=scene.e_per_adu
    )


@pytest.fixture(scope="module")
def far_scene() -> Scene:
    """The boresight is 2.8 degrees from the pole, so a 30 s exposure trails stars 4 to 6 pixels."""
    return make_scene(seed=3, polar_distance_deg=2.8)


@pytest.fixture(scope="module")
def far_detections(far_scene: Scene) -> detect.Detections:
    return detect.detect_stars(
        far_scene.frame_native, saturation_dn=far_scene.saturation_dn, e_per_adu=far_scene.e_per_adu
    )


def match(
    det: detect.Detections, truth: synth.SynthTruth, radius: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """For each detection, whether it matches a truth star and the index of that star."""
    distance, index = _scipy.nearest(
        np.column_stack([truth.x, truth.y]), np.column_stack([det.x, det.y]), radius
    )
    ok = np.isfinite(distance)
    return ok, np.where(ok, index, 0)


def test_detected_stars_have_hundredth_pixel_centroids(
    scene: Scene, detections: detect.Detections
) -> None:
    ok, index = match(detections, scene.truth)
    reliable = ok & detections.reliable()
    assert reliable.sum() > 200
    error_x = detections.x[reliable] - scene.truth.x[index[reliable]]
    error_y = detections.y[reliable] - scene.truth.y[index[reliable]]
    # The research notes' target is 0.1 pixel for the pointing. The centroids of the reliable
    # stars (unsaturated, unblended, away from the edge) reach a few hundredths of a pixel.
    assert np.sqrt(np.mean(error_x**2)) < 0.02
    assert np.sqrt(np.mean(error_y**2)) < 0.02
    assert abs(float(np.mean(error_x))) < 0.005
    assert abs(float(np.mean(error_y))) < 0.005


def test_almost_every_bright_star_is_detected_and_almost_none_is_false(
    scene: Scene, detections: detect.Detections
) -> None:
    ok, _ = match(detections, scene.truth, radius=3.0)
    assert ok.mean() > 0.97  # nearly every detection is a rendered star
    truth = scene.truth
    # Stars with 5000 electrons or more (G of about 11.6 and brighter) that have no neighbor
    # within 15 pixels. The detector does not deblend, so a close pair becomes one source.
    points = np.column_stack([truth.x, truth.y])
    neighbors = _scipy.pairs_within(points, points, 15.0)
    isolated = np.array([len(near) == 1 for near in neighbors])
    interior = (
        (truth.x > 10)
        & (truth.x < truth.width - 10)
        & (truth.y > 10)
        & (truth.y < truth.height - 10)
    )
    expected = (truth.flux_e > 5000.0) & interior & isolated
    assert expected.sum() > 200
    found, _ = _scipy.nearest(np.column_stack([detections.x, detections.y]), points[expected], 2.0)
    assert float(np.isfinite(found).mean()) > 0.99


def test_the_psf_width_is_the_width_across_the_trail(
    scene: Scene,
    detections: detect.Detections,
    far_scene: Scene,
    far_detections: detect.Detections,
) -> None:
    true_fwhm = PSF_SIGMA * FWHM_PER_SIGMA
    # Near the pole the trails are short. Far from it they reach 6 pixels. The fit separates
    # the trail from the PSF, so the width across the trail is the same in both.
    near = detections.reliable()
    far = far_detections.reliable() & (far_detections.trail_length_px > 4.0)
    assert far.sum() > 20
    assert float(np.median(detections.fwhm_px[near])) == pytest.approx(true_fwhm, rel=0.04)
    assert float(np.median(far_detections.fwhm_px[far])) == pytest.approx(true_fwhm, rel=0.04)


def test_the_trail_model_gives_the_analytic_trail_length(far_scene: Scene) -> None:
    truth = far_scene.truth
    model = TrailModel.for_exposure(*truth.pole_px, EXPOSURE_S)
    modeled = model.length(truth.x, truth.y)
    # The exact length: 15.04 arcsec/s * exposure * sin(angular distance from the pole) / scale.
    polar = angular_separation(truth.vectors_cirs[truth.rows], [0.0, 0.0, 1.0])
    analytic = analytic_trail_length_px(polar, EXPOSURE_S, truth.scale_arcsec_px)
    assert np.max(np.abs(modeled / analytic - 1.0)) < 0.01
    # The rendered trails (the straight distance between the ends of the arc) agree as well.
    assert np.max(np.abs(truth.trail_px / analytic - 1.0)) < 0.01
    assert float(analytic.max()) > 5.0  # a 30 s exposure trails the far stars by several pixels


def test_the_trail_direction_is_perpendicular_to_the_direction_to_the_pole(scene: Scene) -> None:
    truth = scene.truth
    model = TrailModel.for_exposure(*truth.pole_px, EXPOSURE_S)
    angle = model.angle(truth.x, truth.y)
    toward_pole = np.arctan2(truth.pole_px[1] - truth.y, truth.pole_px[0] - truth.x)
    difference = np.abs(np.sin(angle - toward_pole))  # 1 when perpendicular to the pole direction
    assert difference.min() > 0.999999 - 1e-5
    assert np.all((angle > -np.pi / 2 - 1e-9) & (angle <= np.pi / 2 + 1e-9))


def test_the_trail_model_keeps_the_centroids_accurate(far_scene: Scene) -> None:
    model = TrailModel.for_exposure(*far_scene.truth.pole_px, EXPOSURE_S)
    with_model = detect.detect_stars(
        far_scene.frame_native,
        saturation_dn=far_scene.saturation_dn,
        e_per_adu=far_scene.e_per_adu,
        trail=model,
    )
    ok, index = match(with_model, far_scene.truth)
    reliable = ok & with_model.reliable()
    error = np.hypot(
        with_model.x[reliable] - far_scene.truth.x[index[reliable]],
        with_model.y[reliable] - far_scene.truth.y[index[reliable]],
    )
    assert reliable.sum() > 150
    assert np.sqrt(np.mean(error**2)) < 0.03
    # The model's trail length is the true one, so the detector reports it.
    expected = model.length(with_model.x[reliable], with_model.y[reliable])
    np.testing.assert_allclose(with_model.trail_length_px[reliable], expected, rtol=1e-3)
    assert float(np.median(with_model.fwhm_px[reliable])) == pytest.approx(
        PSF_SIGMA * FWHM_PER_SIGMA, rel=0.04
    )


def test_without_a_model_the_moments_estimate_the_trail(
    far_scene: Scene, far_detections: detect.Detections
) -> None:
    det, truth = far_detections, far_scene.truth
    ok, index = match(det, truth)
    bright = ok & det.reliable() & (det.snr > 100) & (truth.trail_px[index] > 3.0)
    assert bright.sum() > 10
    ratio = det.trail_length_px[bright] / truth.trail_px[index[bright]]
    assert 0.7 < float(np.median(ratio)) < 1.3  # the moments are biased by the threshold
    assert bool(np.all(det.has(StarFlag.TRAILED)[bright]))
    # With the trail from the moments, the centroids of the reliable stars stay accurate.
    reliable = ok & det.reliable()
    error = np.hypot(
        det.x[reliable] - truth.x[index[reliable]], det.y[reliable] - truth.y[index[reliable]]
    )
    assert np.sqrt(np.mean(error**2)) < 0.04


def test_a_trailed_far_star_is_flagged_and_a_short_trail_is_not(
    far_scene: Scene, far_detections: detect.Detections
) -> None:
    det, truth = far_detections, far_scene.truth
    ok, index = match(det, truth)
    long_trail = ok & (truth.trail_px[index] > 4.0) & det.reliable()
    short_trail = ok & (truth.trail_px[index] < 3.0) & det.reliable() & (det.snr > 50)
    assert long_trail.sum() > 20
    assert det.has(StarFlag.TRAILED)[long_trail].mean() > 0.95
    if short_trail.sum() > 5:
        assert det.has(StarFlag.TRAILED)[short_trail].mean() < 0.35


def test_saturated_stars_are_flagged_and_not_fitted_when_heavily_saturated(
    scene: Scene, detections: detect.Detections
) -> None:
    ok, index = match(detections, scene.truth)
    # A star whose electrons exceed what the pixels hold must be saturated.
    heavy = ok & (scene.truth.flux_e[index] > 4.0e5)
    assert heavy.sum() > 3
    assert detections.has(StarFlag.SATURATED)[heavy].all()
    # A star with many saturated pixels keeps the SEP centroid and is not fitted.
    assert detections.has(StarFlag.MOMENTS_ONLY)[heavy].mean() > 0.8
    # The centroid of a clipped, symmetric star is still good to a fraction of a pixel.
    error = np.hypot(
        detections.x[heavy] - scene.truth.x[index[heavy]],
        detections.y[heavy] - scene.truth.y[index[heavy]],
    )
    assert float(np.median(error)) < 0.3
    faint = ok & (scene.truth.flux_e[index] < 3.0e4) & (scene.truth.flux_e[index] > 4000)
    assert not detections.has(StarFlag.SATURATED)[faint].any()


def test_polaris_is_one_saturated_source_with_its_halo(
    scene: Scene, detections: detect.Detections
) -> None:
    # Polaris is the last row of the synthetic catalog before sorting, so find it by magnitude.
    polaris_row = int(np.argmin(scene.catalog.g_mag))
    where = np.flatnonzero(scene.truth.rows == polaris_row)
    if where.size == 0:
        pytest.skip("Polaris lies outside this frame")
    px, py = scene.truth.x[where[0]], scene.truth.y[where[0]]
    near = np.flatnonzero(np.hypot(detections.x - px, detections.y - py) < 40.0)
    assert near.size >= 1
    brightest = near[0]  # detections sort by flux
    assert detections.has(StarFlag.SATURATED)[brightest]
    assert np.hypot(detections.x[brightest] - px, detections.y[brightest] - py) < 1.5
    assert detections.n_pixels[brightest] > 1000  # the blob includes the halo


def test_a_pure_noise_frame_gives_almost_no_detections() -> None:
    scene = make_scene(seed=7, transmission=0.0)
    det = detect.detect_stars(
        scene.frame_native, saturation_dn=scene.saturation_dn, e_per_adu=scene.e_per_adu
    )
    assert len(det) <= 3  # one million pixels at five sigma


def test_a_sky_gradient_does_not_break_the_background() -> None:
    scene = make_scene(seed=8, sky_gradient=0.6)
    det = detect.detect_stars(
        scene.frame_native, saturation_dn=scene.saturation_dn, e_per_adu=scene.e_per_adu
    )
    ok, index = match(det, scene.truth)
    reliable = ok & det.reliable()
    assert reliable.sum() > 150
    error = np.hypot(
        det.x[reliable] - scene.truth.x[index[reliable]],
        det.y[reliable] - scene.truth.y[index[reliable]],
    )
    assert np.sqrt(np.mean(error**2)) < 0.03
    # The noise is Poisson, so the background rms follows the brighter half of the sky.
    assert det.background_rms > 0


def test_the_hot_pixel_mask_keeps_hot_pixels_out_of_the_list() -> None:
    scene = make_scene(seed=9, n_hot_pixels=60, hot_e_per_s=60.0)
    hot = scene.truth.hot_pixels
    unmasked = detect.detect_stars(
        scene.frame_native, saturation_dn=scene.saturation_dn, e_per_adu=scene.e_per_adu
    )
    masked = detect.detect_stars(
        scene.frame_native,
        saturation_dn=scene.saturation_dn,
        e_per_adu=scene.e_per_adu,
        hot_pixels=hot,
    )

    def on_hot_pixel(det: detect.Detections) -> int:
        ys, xs = np.nonzero(hot)
        distance, _ = _scipy.nearest(
            np.column_stack([xs, ys]).astype(float), np.column_stack([det.x, det.y]), 0.8
        )
        return int(np.isfinite(distance).sum())

    assert on_hot_pixel(unmasked) > 30  # without the mask, the hot pixels look like stars
    assert on_hot_pixel(masked) == 0
    assert len(masked) < len(unmasked)


def test_a_narrow_star_is_not_taken_for_a_hot_pixel() -> None:
    # An undersampled star at the diffraction limit of the 50 mm aperture (FWHM 0.66 pixel in bin2,
    # sigma 0.28) is not a hot pixel. The detector must keep it and must not flag it.
    scene = make_scene(seed=10, psf_sigma_px=0.28)
    det = detect.detect_stars(
        scene.frame_native, saturation_dn=scene.saturation_dn, e_per_adu=scene.e_per_adu
    )
    ok, index = match(det, scene.truth)
    bright = ok & (scene.truth.flux_e[index] > 1.0e4) & (scene.truth.flux_e[index] < 2.0e5)
    assert bright.sum() > 30
    assert not det.has(StarFlag.HOT_PIXEL)[bright].any()
    reliable = ok & det.reliable()
    error = np.hypot(
        det.x[reliable] - scene.truth.x[index[reliable]],
        det.y[reliable] - scene.truth.y[index[reliable]],
    )
    assert np.sqrt(np.mean(error**2)) < 0.03


def test_an_unmasked_single_pixel_spike_is_flagged() -> None:
    scene = make_scene(seed=11)
    frame = scene.frame_native.copy()
    frame[200, 300] += 3000.0  # a hot pixel that the dark library has not seen yet
    det = detect.detect_stars(frame, saturation_dn=scene.saturation_dn, e_per_adu=scene.e_per_adu)
    spike = np.flatnonzero(np.hypot(det.x - 300.0, det.y - 200.0) < 1.0)
    assert spike.size == 1
    assert det.has(StarFlag.HOT_PIXEL)[spike[0]]


def test_a_bright_streak_is_flagged_when_the_trail_model_knows_better(scene: Scene) -> None:
    frame = scene.frame_native.copy()
    rows = np.arange(380, 420)
    # A satellite: a line of 40 pixels crossing the frame at a place where trails are short.
    frame[rows, 600 + (rows - 380) // 4] += 400.0
    frame[rows, 601 + (rows - 380) // 4] += 400.0
    model = TrailModel.for_exposure(*scene.truth.pole_px, EXPOSURE_S)
    det = detect.detect_stars(
        frame, saturation_dn=scene.saturation_dn, e_per_adu=scene.e_per_adu, trail=model
    )
    near = np.flatnonzero(np.hypot(det.x - 605.0, det.y - 400.0) < 25.0)
    assert near.size >= 1
    assert det.has(StarFlag.STREAK)[near[np.argmax(det.flux[near])]]


def test_the_star_list_for_a_solver_is_sorted_and_leaves_out_hot_pixels(
    scene: Scene, detections: detect.Detections
) -> None:
    stars = detections.star_list()
    assert len(stars) == len(detections) - int(detections.has(StarFlag.HOT_PIXEL).sum())
    assert np.all(np.diff(stars.flux) <= 0.0)
    assert np.all(stars.flux > 0.0)
    assert stars.x.shape == stars.y.shape == stars.flux.shape


def test_select_and_shift_copy_the_detections(detections: detect.Detections) -> None:
    keep = np.arange(len(detections)) < 10
    first = detections.select(keep)
    assert len(first) == 10
    shifted = first.shifted(100.0, -50.0)
    np.testing.assert_allclose(shifted.x, first.x + 100.0)
    np.testing.assert_allclose(shifted.y, first.y - 50.0)
    np.testing.assert_array_equal(shifted.flags, first.flags)
    assert shifted.background is detections.background


def test_the_star_mask_covers_the_stars_and_a_saturated_star_masks_more(
    scene: Scene, detections: detect.Detections
) -> None:
    mask = detect.star_mask(detections.shape, detections)
    assert mask.shape == detections.shape
    rows = np.rint(detections.y).astype(int).clip(0, mask.shape[0] - 1)
    columns = np.rint(detections.x).astype(int).clip(0, mask.shape[1] - 1)
    assert mask[rows, columns].all()
    assert 0.0 < mask.mean() < 0.2
    saturated = detections.has(StarFlag.SATURATED)
    plain = ~saturated & (detections.trail_length_px < 1.0)
    one = detections.select(np.flatnonzero(saturated)[:1])
    two = detections.select(np.flatnonzero(plain)[:1])
    assert (
        detect.star_mask(detections.shape, one).sum()
        > detect.star_mask(detections.shape, two).sum()
    )


def test_options_are_checked() -> None:
    with pytest.raises(ValueError, match="invalid detector options"):
        detect.DetectOptions(threshold_sigma=0.0)


def test_a_wrong_hot_pixel_mask_shape_is_an_error(scene: Scene) -> None:
    with pytest.raises(detect.DetectionError, match="shape"):
        detect.detect_stars(
            scene.frame_native,
            saturation_dn=scene.saturation_dn,
            hot_pixels=np.zeros((3, 3), dtype=bool),
        )
    with pytest.raises(detect.DetectionError, match="2-D"):
        detect.detect_stars(np.zeros(10, dtype=np.float32), saturation_dn=1000.0)


def test_a_flat_frame_is_an_error_not_a_crash() -> None:
    with pytest.raises(detect.DetectionError, match="constant"):
        detect.detect_stars(np.full((100, 100), 500.0, dtype=np.float32), saturation_dn=16383.0)
