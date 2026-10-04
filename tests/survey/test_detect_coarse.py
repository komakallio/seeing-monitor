"""The binned search of the detector: the stars and centroids of the full search, in less time.

With `DetectOptions.coarse_bin` above 1, the detector searches a copy of the frame that sums each
block of pixels, and it fits the brightest stars at full resolution. The tests render the scenes of
`test_detect` and compare the binned search with the truth and with the full search. The binned
search needs a trail model, as the tracker gives it, so the scenes pass the true pole position.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.survey import _scipy, detect
from seeingmon.survey.centroid import FWHM_PER_SIGMA
from seeingmon.survey.config import DetectConfig
from seeingmon.survey.detect import DetectOptions, StarFlag
from seeingmon.survey.trail import TrailModel
from tests.survey.test_detect import EXPOSURE_S, PSF_SIGMA, Scene, make_scene, match

COARSE = DetectOptions(coarse_bin=2)
FloatArray = npt.NDArray[np.float64]


def trail_of(scene: Scene) -> TrailModel:
    return TrailModel.for_exposure(*scene.truth.pole_px, EXPOSURE_S)


def search(
    scene: Scene,
    options: DetectOptions = COARSE,
    *,
    trail: TrailModel | Literal["truth"] | None = "truth",
    frame: npt.NDArray[np.float32] | None = None,
    **kwargs: object,
) -> detect.Detections:
    """Detect the stars of a scene, with the true trail model unless `trail` says otherwise."""
    model = trail_of(scene) if isinstance(trail, str) else trail
    return detect.detect_stars(
        scene.frame_native if frame is None else frame,
        saturation_dn=scene.saturation_dn,
        e_per_adu=scene.e_per_adu,
        options=options,
        trail=model,
        **kwargs,  # type: ignore[arg-type]
    )


def position_error(
    found: detect.Detections, scene: Scene, mask: npt.NDArray[np.bool_] | None = None
) -> FloatArray:
    """The distance of each detection from the true position of the star it matches."""
    ok, index = match(found, scene.truth)
    keep = ok if mask is None else ok & mask
    return np.asarray(
        np.hypot(
            found.x[keep] - scene.truth.x[index[keep]],
            found.y[keep] - scene.truth.y[index[keep]],
        ),
        dtype=np.float64,
    )


@pytest.fixture(scope="module")
def scene() -> Scene:
    return make_scene()


@pytest.fixture(scope="module")
def far_scene() -> Scene:
    """The boresight is 2.8 degrees from the pole, so a 30 s exposure trails stars 4 to 6 pixels."""
    return make_scene(seed=3, polar_distance_deg=2.8)


@pytest.fixture(scope="module")
def found(scene: Scene) -> detect.Detections:
    return search(scene)


@pytest.fixture(scope="module")
def full(scene: Scene) -> detect.Detections:
    return search(scene, DetectOptions())


# --- The pieces -----------------------------------------------------------------------------


def test_binning_sums_each_block_and_drops_a_partial_block() -> None:
    frame = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    binned = detect._bin_sum(frame, 2)
    assert binned.shape == (2, 3)
    expected = np.array(
        [[frame[2 * i : 2 * i + 2, 2 * j : 2 * j + 2].sum() for j in range(3)] for i in range(2)]
    )
    np.testing.assert_array_equal(binned, expected)
    assert binned.dtype == np.float32
    assert detect._bin_sum(frame, 3).shape == (1, 2)


def test_a_binned_pixel_is_masked_when_any_pixel_of_its_block_is() -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 1] = True  # the first block
    mask[5, 5] = True  # the last block
    binned = detect._bin_any(mask, 2)
    assert binned.shape == (3, 3)
    assert binned.sum() == 2
    assert binned[0, 0]
    assert binned[2, 2]
    assert detect._bin_any(np.zeros((5, 5), dtype=bool), 2).shape == (2, 2)


def test_the_background_map_comes_off_block_by_block_and_the_edge_takes_the_nearest_block() -> None:
    frame = np.full((5, 5), 100.0, dtype=np.float32)
    levels = np.array([[4.0, 8.0], [12.0, 16.0]], dtype=np.float32)  # blocks of 2 x 2
    detect._subtract_blocks(frame, levels, 2)
    # The map has the shape of the complete blocks, and the fifth row and column repeat the last.
    assert frame.shape == (5, 5)
    np.testing.assert_array_equal(frame[:2, :2], 96.0)
    np.testing.assert_array_equal(frame[:2, 2:4], 92.0)
    np.testing.assert_array_equal(frame[2:4, :2], 88.0)
    np.testing.assert_array_equal(frame[2:4, 2:4], 84.0)
    np.testing.assert_array_equal(frame[4, :4], [88.0, 88.0, 84.0, 84.0])
    np.testing.assert_array_equal(frame[:4, 4], [92.0, 92.0, 84.0, 84.0])
    assert frame[4, 4] == 84.0


def test_the_window_peak_is_the_highest_pixel_within_reach() -> None:
    frame = np.zeros((20, 20), dtype=np.float32)
    frame[10, 12] = 50.0
    frame[0, 0] = 9.0
    peak = detect._window_peak(frame, np.array([10.4, 15.0, 0.0]), np.array([10.0, 10.0, 0.0]), 2)
    assert peak.tolist() == [50.0, 0.0, 9.0]  # the window of the second star starts at column 13
    edge = detect._window_peak(frame, np.array([-5.0]), np.array([-5.0]), 2)
    assert edge.tolist() == [9.0]  # a position outside the frame reads the nearest corner


# --- The stars -------------------------------------------------------------------------------


def test_the_binned_search_keeps_the_hundredth_pixel_centroids(
    scene: Scene, found: detect.Detections
) -> None:
    ok, index = match(found, scene.truth)
    reliable = ok & found.reliable()
    assert reliable.sum() > 200
    error_x = found.x[reliable] - scene.truth.x[index[reliable]]
    error_y = found.y[reliable] - scene.truth.y[index[reliable]]
    assert np.sqrt(np.mean(error_x**2)) < 0.02
    assert np.sqrt(np.mean(error_y**2)) < 0.02
    assert abs(float(np.mean(error_x))) < 0.005
    assert abs(float(np.mean(error_y))) < 0.005


def test_almost_every_bright_star_is_found_and_almost_none_is_false(
    scene: Scene, found: detect.Detections
) -> None:
    ok, _ = match(found, scene.truth, radius=3.0)
    assert ok.mean() > 0.97
    truth = scene.truth
    points = np.column_stack([truth.x, truth.y])
    isolated = np.array([len(near) == 1 for near in _scipy.pairs_within(points, points, 15.0)])
    interior = (
        (truth.x > 10)
        & (truth.x < truth.width - 10)
        & (truth.y > 10)
        & (truth.y < truth.height - 10)
    )
    expected = (truth.flux_e > 5000.0) & interior & isolated
    assert expected.sum() > 200
    seen, _ = _scipy.nearest(np.column_stack([found.x, found.y]), points[expected], 2.0)
    assert float(np.isfinite(seen).mean()) > 0.99


def test_the_binned_search_finds_the_stars_of_the_full_search_at_the_same_places(
    found: detect.Detections, full: detect.Detections
) -> None:
    both = np.column_stack([full.x, full.y])
    strong = full.snr > 30.0
    seen, index = _scipy.nearest(np.column_stack([found.x, found.y]), both[strong], 1.0)
    assert strong.sum() > 200
    assert float(np.isfinite(seen).mean()) > 0.98
    # The stars that both searches fit land on the same pixel position.
    ok = np.isfinite(seen)
    pairs_full = np.flatnonzero(strong)[ok]
    pairs_found = index[ok]
    fitted = full.reliable()[pairs_full] & found.reliable()[pairs_found]
    assert fitted.sum() > 150
    gap = np.hypot(
        full.x[pairs_full][fitted] - found.x[pairs_found][fitted],
        full.y[pairs_full][fitted] - found.y[pairs_found][fitted],
    )
    assert float(np.median(gap)) < 0.005
    assert float(gap.max()) < 0.05


def test_the_width_across_the_trail_is_the_width_of_the_psf(
    scene: Scene, found: detect.Detections
) -> None:
    true_fwhm = PSF_SIGMA * FWHM_PER_SIGMA
    assert float(np.median(found.fwhm_px[found.reliable()])) == pytest.approx(true_fwhm, rel=0.04)


def test_a_trailed_star_keeps_its_centroid(far_scene: Scene) -> None:
    far = search(far_scene)
    error = position_error(far, far_scene, far.reliable())
    assert error.size > 150
    assert np.sqrt(np.mean(error**2)) < 0.03
    trailed = far.reliable() & (far.trail_length_px > 4.0)
    assert trailed.sum() > 20
    assert float(np.median(far.fwhm_px[trailed])) == pytest.approx(
        PSF_SIGMA * FWHM_PER_SIGMA, rel=0.04
    )


def test_a_frame_with_a_gradient_and_hot_pixels_keeps_its_centroids() -> None:
    scene = make_scene(seed=8, sky_gradient=0.6, n_hot_pixels=40, hot_e_per_s=60.0)
    gradient = search(scene, hot_pixels=scene.truth.hot_pixels)
    error = position_error(gradient, scene, gradient.reliable())
    assert error.size > 150
    assert np.sqrt(np.mean(error**2)) < 0.03


# --- The brightest stars get the fit ---------------------------------------------------------


def test_only_the_brightest_stars_get_the_model_fit(scene: Scene) -> None:
    limited = search(scene, DetectOptions(coarse_bin=2, refine_stars=100))
    coarse = limited.has(StarFlag.COARSE)
    fitted = ~coarse
    # A fit that does not converge, or a star with many saturated pixels, leaves its star coarse.
    assert 60 <= fitted.sum() <= 100
    assert coarse.sum() > 200
    # The fitted stars come from the 100 brightest that the fit can handle (the model flux
    # puts them first in the list), and the others have a coarse position.
    assert np.flatnonzero(fitted).max() < 150
    assert not limited.reliable()[coarse].any()
    assert limited.reliable()[fitted].sum() > 20
    np.testing.assert_array_equal(limited.x_error_px[coarse], 0.5)
    assert (limited.x_error_px[fitted] < 0.1).all()


def test_a_coarse_star_has_a_position_good_to_a_fraction_of_a_pixel(scene: Scene) -> None:
    limited = search(scene, DetectOptions(coarse_bin=2, refine_stars=100))
    coarse = limited.has(StarFlag.COARSE) & ~limited.has(StarFlag.SATURATED)
    ok, index = match(limited, scene.truth)
    keep = ok & coarse
    assert keep.sum() > 200
    error_x = limited.x[keep] - scene.truth.x[index[keep]]
    error_y = limited.y[keep] - scene.truth.y[index[keep]]
    assert float(np.median(np.hypot(error_x, error_y))) < 0.2
    assert float(np.percentile(np.hypot(error_x, error_y), 95)) < 0.5
    # The binned pixels map to the full frame without a shift (a half pixel is in the formula),
    # and the faint tail of the threshold biases the mean by a few hundredths of a pixel.
    assert abs(float(np.mean(error_x))) < 0.1
    assert abs(float(np.mean(error_y))) < 0.1


def test_the_coarse_flag_is_one_of_the_flags_that_make_a_star_unreliable() -> None:
    assert StarFlag.COARSE in detect.UNRELIABLE
    assert int(StarFlag.COARSE) == 128
    assert int(StarFlag.COARSE) & (int(detect.UNRELIABLE) ^ int(StarFlag.COARSE)) == 0


def test_a_heavily_saturated_star_keeps_the_coarse_position_and_both_flags(
    scene: Scene, found: detect.Detections
) -> None:
    ok, index = match(found, scene.truth)
    heavy = ok & (scene.truth.flux_e[index] > 4.0e5)
    assert heavy.sum() > 3
    assert found.has(StarFlag.SATURATED)[heavy].all()
    held = heavy & found.has(StarFlag.MOMENTS_ONLY)
    assert held.sum() > 3
    assert found.has(StarFlag.COARSE)[held].all()
    error = np.hypot(
        found.x[heavy] - scene.truth.x[index[heavy]], found.y[heavy] - scene.truth.y[index[heavy]]
    )
    assert float(np.median(error)) < 0.5


def test_a_narrow_star_is_not_taken_for_a_hot_pixel() -> None:
    scene = make_scene(seed=10, psf_sigma_px=0.28)
    narrow = search(scene)
    ok, index = match(narrow, scene.truth)
    bright = ok & (scene.truth.flux_e[index] > 1.0e4) & (scene.truth.flux_e[index] < 2.0e5)
    assert bright.sum() > 30
    assert not narrow.has(StarFlag.HOT_PIXEL)[bright].any()
    error = position_error(narrow, scene, narrow.reliable())
    assert error.size > 100
    assert np.sqrt(np.mean(error**2)) < 0.03


# --- Hot pixels, noise, and streaks --------------------------------------------------------


def test_a_masked_hot_pixel_stays_out_and_an_unmasked_one_is_no_star() -> None:
    scene = make_scene(seed=9, n_hot_pixels=60, hot_e_per_s=60.0)
    hot = scene.truth.hot_pixels
    ys, xs = np.nonzero(hot)
    spots = np.column_stack([xs, ys]).astype(float)

    def on_hot_pixel(found: detect.Detections) -> int:
        distance, _ = _scipy.nearest(spots, np.column_stack([found.x, found.y]), 0.8)
        return int(np.isfinite(distance).sum())

    assert on_hot_pixel(search(scene, hot_pixels=hot)) == 0
    # A hot pixel that the mask misses fills one binned pixel, and `min_pixels` rejects it. The
    # full search lists more than 30 of them.
    assert on_hot_pixel(search(scene)) < 5
    assert on_hot_pixel(search(scene, DetectOptions())) > 30


def test_an_unmasked_single_pixel_spike_is_no_source_in_the_binned_search() -> None:
    scene = make_scene(seed=11)
    frame = scene.frame_native.copy()
    frame[200, 300] += 3000.0
    spiked = search(scene, frame=frame)
    assert not (np.hypot(spiked.x - 300.0, spiked.y - 200.0) < 1.0).any()


def test_a_pure_noise_frame_gives_almost_no_detections() -> None:
    scene = make_scene(seed=7, transmission=0.0)
    assert len(search(scene)) <= 3


def test_a_bright_streak_is_flagged_when_the_trail_model_knows_better(scene: Scene) -> None:
    frame = scene.frame_native.copy()
    rows = np.arange(380, 420)
    frame[rows, 600 + (rows - 380) // 4] += 400.0  # a satellite that crosses the frame
    frame[rows, 601 + (rows - 380) // 4] += 400.0
    streaked = search(scene, frame=frame)
    near = np.flatnonzero(np.hypot(streaked.x - 605.0, streaked.y - 400.0) < 25.0)
    assert near.size >= 1
    assert streaked.has(StarFlag.STREAK)[near[np.argmax(streaked.flux[near])]]


# --- Units, sizes, and the settings ----------------------------------------------------------


def test_the_background_and_the_sizes_follow_the_pixels_of_the_full_frame(
    scene: Scene, found: detect.Detections, full: detect.Detections
) -> None:
    assert found.shape == full.shape == scene.frame_native.shape
    assert found.background_level == pytest.approx(full.background_level, rel=0.005)
    assert found.background_rms == pytest.approx(full.background_rms, rel=0.05)
    assert found.background is None
    assert full.background is not None
    # A saturated blob holds about as many pixels in both searches.
    blob = np.flatnonzero(full.n_pixels > 1000)
    assert blob.size >= 1
    seen, index = _scipy.nearest(
        np.column_stack([found.x, found.y]), np.column_stack([full.x[blob], full.y[blob]]), 2.0
    )
    assert np.isfinite(seen).all()
    ratio = found.n_pixels[index].astype(float) / full.n_pixels[blob]
    assert ((ratio > 0.5) & (ratio < 1.5)).all()
    # The signal-to-noise ratio of the same star agrees.
    pair = full.snr > 100
    seen, index = _scipy.nearest(
        np.column_stack([found.x, found.y]), np.column_stack([full.x[pair], full.y[pair]]), 1.0
    )
    ok = np.isfinite(seen)
    ratio = found.snr[index[ok]] / full.snr[pair][ok]
    assert 0.8 < float(np.median(ratio)) < 1.2


def test_a_frame_with_an_odd_size_loses_only_its_last_row_and_column_in_the_search(
    scene: Scene,
) -> None:
    even = search(scene)
    odd_frame = np.ascontiguousarray(scene.frame_native[:799, :1199])
    odd = search(scene, frame=odd_frame)
    assert odd.shape == (799, 1199)
    seen, index = _scipy.nearest(
        np.column_stack([even.x, even.y]), np.column_stack([odd.x, odd.y]), 0.5
    )
    ok = np.isfinite(seen)
    assert ok.mean() > 0.98
    reliable = odd.reliable()[ok] & even.reliable()[index[ok]]
    assert reliable.sum() > 200
    assert float(np.max(seen[ok][reliable])) < 0.05  # the same fits, on nearly the same pixels


def test_a_larger_bin_finds_the_stars_too(scene: Scene) -> None:
    triple = search(scene, DetectOptions(coarse_bin=3))
    error = position_error(triple, scene, triple.reliable())
    assert error.size > 150
    assert np.sqrt(np.mean(error**2)) < 0.03
    ok, _ = match(triple, scene.truth, radius=3.0)
    assert ok.mean() > 0.9


def test_a_tiny_frame_still_runs() -> None:
    scene = make_scene()
    patch = np.ascontiguousarray(scene.frame_native[100:150, 100:160])
    tiny = search(scene, frame=patch)
    assert tiny.shape == (50, 60)


def test_without_a_trail_model_the_search_takes_the_full_frame(scene: Scene) -> None:
    plain = search(scene, DetectOptions(), trail=None)
    no_model = search(scene, COARSE, trail=None)
    assert not no_model.has(StarFlag.COARSE).any()
    assert no_model.background is not None
    for name in ("x", "y", "flux", "fwhm_px", "flags", "snr", "n_pixels"):
        np.testing.assert_array_equal(getattr(no_model, name), getattr(plain, name))


def test_the_default_search_is_the_full_search(scene: Scene) -> None:
    assert DetectOptions().coarse_bin == 1
    with_default = search(scene, DetectOptions())
    explicit = search(scene, DetectOptions(coarse_bin=1, refine_stars=10))
    assert not with_default.has(StarFlag.COARSE).any()
    np.testing.assert_array_equal(with_default.x, explicit.x)
    np.testing.assert_array_equal(with_default.flags, explicit.flags)


def test_the_star_mask_of_a_binned_search_covers_its_stars(found: detect.Detections) -> None:
    mask = detect.star_mask(found.shape, found)
    rows = np.rint(found.y).astype(int).clip(0, mask.shape[0] - 1)
    columns = np.rint(found.x).astype(int).clip(0, mask.shape[1] - 1)
    assert mask[rows, columns].all()
    assert 0.0 < mask.mean() < 0.2


def test_the_options_are_checked_and_follow_the_configuration() -> None:
    with pytest.raises(ValueError, match="invalid detector options"):
        DetectOptions(coarse_bin=0)
    with pytest.raises(ValueError, match="invalid detector options"):
        DetectOptions(refine_stars=0)
    options = DetectOptions.from_config(
        DetectConfig(coarse_bin=2, refine_stars=900, max_stars=2500, mesh_px=48)
    )
    assert (options.coarse_bin, options.refine_stars) == (2, 900)
    assert (options.max_stars, options.mesh_px) == (2500, 48)
