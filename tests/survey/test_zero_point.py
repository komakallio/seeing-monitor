"""The zero-point fit: recovery, the sampling error, clipping, and the color term."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.survey import photometry as ph
from seeingmon.survey import zero_point as zp
from seeingmon.survey._scipy import nearest
from seeingmon.survey.detect import detect_stars
from seeingmon.survey.rawdata import native_counts
from tests.survey import synth

FloatArray = npt.NDArray[np.float64]
ZP_TRUE = 19.25
C_TRUE = -0.07


def make_stars(
    n: int = 120,
    *,
    seed: int = 0,
    scatter_mag: float = 0.015,
    color_low: float = 0.0,
    color_high: float = 2.5,
    zero_point: float = ZP_TRUE,
    color_term: float = C_TRUE,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Catalog magnitudes and colors, rates that follow the model, and the errors of each star."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(8.0, 12.5, n)
    color = rng.uniform(color_low, color_high, n)
    truth = g + 0.0  # the catalog magnitude is the model's G
    noise = rng.normal(0.0, scatter_mag, n)
    rate = 10.0 ** (0.4 * (zero_point - truth + color_term * color + noise))
    mag_error = np.full(n, scatter_mag)
    catalog_error = np.full(n, 0.005)
    return g, color, rate, mag_error, catalog_error


def test_the_fit_recovers_the_zero_point_and_the_color_term() -> None:
    g, color, rate, error, cat_error = make_stars()
    fit = zp.fit_zero_point(g, color, rate, error, cat_error)
    assert fit is not None
    assert fit.color_fitted
    assert fit.n_used == fit.n_input == 120
    assert fit.zero_point_mag == pytest.approx(ZP_TRUE, abs=3.0 * fit.zero_point_error_mag)
    assert fit.color_term == pytest.approx(C_TRUE, abs=3.0 * fit.color_term_error)
    assert fit.rms_mag == pytest.approx(0.015, rel=0.2)
    # About 120 stars that scatter by 0.015 mag fix the zero point to 0.0015 mag, and the color
    # term, which the zero point at color 0 couples to it, loosens that a little.
    assert 0.001 < fit.zero_point_error_mag < 0.006


def test_the_stated_error_is_the_real_scatter_of_the_zero_point() -> None:
    """Many frames: the spread of the fitted zero points matches the sampling error."""
    estimates = []
    errors = []
    for seed in range(300):
        g, color, rate, error, cat_error = make_stars(n=80, seed=seed)
        fit = zp.fit_zero_point(g, color, rate, error, cat_error)
        assert fit is not None
        estimates.append(fit.zero_point_mag)
        errors.append(fit.zero_point_error_mag)
    assert np.mean(estimates) == pytest.approx(ZP_TRUE, abs=0.0006)
    assert np.std(estimates, ddof=1) == pytest.approx(np.mean(errors), rel=0.25)


def test_a_narrow_range_of_colors_holds_the_color_term_at_its_prior() -> None:
    g, color, rate, error, cat_error = make_stars(color_low=0.8, color_high=1.0)
    fit = zp.fit_zero_point(g, color, rate, error, cat_error)
    assert fit is not None
    assert not fit.color_fitted
    assert fit.color_term == 0.0
    assert fit.color_term_error == 0.0
    # With the true color term as the prior, the zero point comes out right.
    options = zp.ZeroPointOptions(color_term_prior=C_TRUE)
    held = zp.fit_zero_point(g, color, rate, error, cat_error, options)
    assert held is not None
    assert held.color_term == C_TRUE
    assert held.zero_point_mag == pytest.approx(ZP_TRUE, abs=3.0 * held.zero_point_error_mag)


def test_outliers_leave_the_sample_and_the_zero_point_stays_put() -> None:
    g, color, rate, error, cat_error = make_stars(n=100, seed=3)
    rate = rate.copy()
    outliers = np.array([5, 17, 40, 41, 77])
    rate[outliers] *= 10.0 ** (
        -0.4 * 0.5
    )  # five stars half a magnitude too faint: blends or clouds
    clean = zp.fit_zero_point(g, color, rate, error, cat_error)
    assert clean is not None
    assert not clean.used[outliers].any()
    assert clean.n_used == 95
    assert clean.zero_point_mag == pytest.approx(ZP_TRUE, abs=3.0 * clean.zero_point_error_mag)
    # With clipping off, they pull the zero point.
    loose = zp.fit_zero_point(g, color, rate, error, cat_error, zp.ZeroPointOptions(clip_sigma=1e6))
    assert loose is not None
    assert loose.n_used == 100
    assert abs(loose.zero_point_mag - ZP_TRUE) > abs(clean.zero_point_mag - ZP_TRUE)
    assert loose.rms_mag > 2.0 * clean.rms_mag


def test_a_scatter_beyond_the_errors_becomes_an_intrinsic_term() -> None:
    g, color, rate, _, cat_error = make_stars(n=200, seed=4, scatter_mag=0.05)
    small = np.full(200, 0.01)  # the photometry claims 0.01 mag, but the stars scatter by 0.05
    fit = zp.fit_zero_point(g, color, rate, small, cat_error)
    assert fit is not None
    assert fit.intrinsic_mag == pytest.approx(0.05, rel=0.25)
    assert fit.zero_point_error_mag > 0.003  # the error grows with it: not 0.01 / sqrt(200)


def test_stars_without_a_color_help_the_zero_point_and_not_the_color_term() -> None:
    g, color, rate, error, cat_error = make_stars(n=150, seed=5)
    colorless = color.copy()
    colorless[::3] = np.nan
    fit = zp.fit_zero_point(g, colorless, rate, error, cat_error)
    assert fit is not None
    assert fit.n_used >= 145
    assert fit.color_term == pytest.approx(C_TRUE, abs=4.0 * fit.color_term_error)
    # The zero point is still good: the stand-in color is the median of the sample.
    assert fit.zero_point_mag == pytest.approx(ZP_TRUE, abs=0.02)


def test_a_catalog_error_lets_a_poor_magnitude_count_for_less() -> None:
    g, color, rate, error, _ = make_stars(n=100, seed=6)
    rate = rate.copy()
    rate[:10] *= 10.0 ** (-0.4 * 0.12)  # ten stars with a catalog G that is 0.12 mag off
    options = zp.ZeroPointOptions(clip_sigma=1e6)
    plain = zp.fit_zero_point(g, color, rate, error, np.full(100, 0.005), options)
    errors = np.full(100, 0.005)
    errors[:10] = 0.15
    weighted = zp.fit_zero_point(g, color, rate, error, errors, options)
    assert plain is not None
    assert weighted is not None
    assert abs(weighted.zero_point_mag - ZP_TRUE) < abs(plain.zero_point_mag - ZP_TRUE)


def test_catalog_errors_follow_the_kind_of_magnitude() -> None:
    g = np.array([2.0, 5.9, 6.5, 11.0, 11.0])
    estimated = np.array([True, False, False, False, True])
    error = zp.catalog_errors_mag(g, estimated, zp.ZeroPointOptions())
    np.testing.assert_allclose(error, [0.08, 0.02, 0.005, 0.005, 0.08])


def test_too_few_stars_give_no_zero_point() -> None:
    g, color, rate, error, cat_error = make_stars(n=7)
    assert zp.fit_zero_point(g, color, rate, error, cat_error) is None
    g, color, rate, error, cat_error = make_stars(n=9)
    bad = rate.copy()
    bad[:3] = -1.0  # rates that are not positive do not count
    assert zp.fit_zero_point(g, color, bad, error, cat_error) is None  # 6 usable
    assert zp.fit_zero_point(g, color, rate, error, cat_error) is not None


def test_the_options_refuse_nonsense() -> None:
    with pytest.raises(ValueError, match="invalid zero-point options"):
        zp.ZeroPointOptions(min_stars=2)


# --- On a rendered frame -------------------------------------------------------------------


def rendered_fit(
    *, zero_point: float, color_term: float, seed: int, width: int = 1600, height: int = 1200
) -> tuple[zp.ZeroPointFit, int]:
    """Render a frame, detect and measure its stars, and fit the zero point."""
    profile = synth.cropped_profile(width, height)
    catalog = synth.synthetic_catalog(cap_radius_deg=8.0, density_scale=1.0, seed=3)
    frame, truth = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        zero_point_mag=zero_point,
        color_term=color_term,
        star_scatter=0.01,
        sky_e_per_s_px=4.2,
        dark_e_per_s_px=0.7,
        seed=seed,
    )
    data = native_counts(frame)
    saturation = profile.saturation("bin2", 120).native_dn
    e_per_adu = profile.e_per_adu("bin2", 120)
    detections = detect_stars(data, saturation_dn=saturation, e_per_adu=e_per_adu)
    # Match every detection to the star that the renderer drew within 1.5 pixels.
    _, index = nearest(
        np.column_stack([truth.x, truth.y]), np.column_stack([detections.x, detections.y]), 1.5
    )
    cat_row = np.full(len(detections), -1, dtype=np.intp)
    found = index < truth.rows.size
    cat_row[found] = truth.rows[index[found]]
    stars = ph.measure_matched_stars(
        data,
        detections,
        cat_row,
        exposure_s=truth.exposure_s,
        e_per_adu=e_per_adu,
        saturation_dn=saturation,
        min_snr=20.0,
    )
    options = zp.ZeroPointOptions()
    g = catalog.g_mag[stars.cat_row]
    fit = zp.fit_zero_point(
        g,
        catalog.bp_rp[stars.cat_row],
        stars.rate_e_per_s,
        stars.mag_error,
        zp.catalog_errors_mag(g, np.zeros(g.size, dtype=bool), options),
        options,
    )
    assert fit is not None
    return fit, len(stars)


@pytest.mark.parametrize(
    ("zero_point", "color_term"), [(19.157, 0.0), (19.30, -0.08), (18.9, 0.12)]
)
def test_a_rendered_frame_gives_the_injected_zero_point_within_0_03_mag(
    zero_point: float, color_term: float
) -> None:
    """The done criterion: 0.03 mag. The sampling error of the fit is about 0.003 mag.

    The frame holds about 170 measurable stars that scatter by about 0.017 mag around the model
    (the photon noise of the fainter ones and 1% of flux noise), so the sampling error is
    0.017 / sqrt(170) = 0.0013 mag, and the color term costs a factor of 2 more. The tolerance of
    0.03 mag is ten times that, and the measured error stays under 0.005 mag.
    """
    fit, n_stars = rendered_fit(zero_point=zero_point, color_term=color_term, seed=5)
    assert n_stars > 100
    assert fit.n_used > 100
    assert fit.zero_point_error_mag < 0.006
    assert abs(fit.zero_point_mag - zero_point) < 0.03
    assert abs(fit.zero_point_mag - zero_point) < 4.0 * fit.zero_point_error_mag
    assert fit.color_term == pytest.approx(color_term, abs=0.02)
    assert fit.rms_mag < 0.03
