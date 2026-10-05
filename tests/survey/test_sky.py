"""The sky measurement, the magnitude, the V conversion, and the flat-field hook."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.solvers import fitsio
from seeingmon.survey import sky
from seeingmon.survey.detect import detect_stars, star_mask
from tests.survey import synth

E_PER_ADU = 0.88
READ_NOISE_E = 1.85
SATURATION_DN = 16383.0
BIAS_DN = 120.0
SCALE = 3.82  # arcsec per pixel in bin2
FloatArray = npt.NDArray[np.float64]


def sky_frame(
    *,
    sky_e_per_s_px: float = 4.2,
    dark_e_per_s_px: float = 0.7,
    exposure_s: float = 30.0,
    shape: tuple[int, int] = (400, 500),
    seed: int = 0,
    gradient: FloatArray | None = None,
) -> npt.NDArray[np.float32]:
    """Poisson and read noise on a uniform sky, in counts with the bias."""
    rng = np.random.default_rng(seed)
    sky_part = np.full(shape, sky_e_per_s_px * exposure_s)
    if gradient is not None:
        sky_part = sky_part * gradient  # the dark current is not vignetted
    mean = sky_part + dark_e_per_s_px * exposure_s
    electrons = rng.poisson(mean) + rng.normal(0.0, READ_NOISE_E, shape)
    return np.rint(electrons / E_PER_ADU + BIAS_DN).astype(np.float32)


def measure(
    data: npt.NDArray[np.float32],
    *,
    dark_dn: float = BIAS_DN + 0.7 * 30.0 / E_PER_ADU,
    **kwargs: object,
) -> sky.SkyMeasurement | None:
    base: dict[str, object] = {
        "star_mask": None,
        "flat": None,
        "exposure_s": 30.0,
        "e_per_adu": E_PER_ADU,
        "scale_arcsec_px": SCALE,
        "saturation_dn": SATURATION_DN,
    }
    base.update(kwargs)
    return sky.measure_sky(data, dark_level_dn=dark_dn, **base)  # type: ignore[arg-type]


# --- Magnitudes and colors -----------------------------------------------------------------


def test_a_pixel_of_bin2_adds_2_910_mag_to_a_surface_brightness() -> None:
    assert sky.pixel_solid_angle_mag(SCALE) == pytest.approx(2.910, abs=0.001)
    assert sky.pixel_solid_angle_mag(1.91) == pytest.approx(2.5 * math.log10(1.91**2))


def test_the_sky_magnitude_is_the_zero_point_minus_the_log_of_the_rate() -> None:
    assert sky.sky_magnitude(19.157, 0.29) == pytest.approx(19.157 - 2.5 * math.log10(0.29))
    assert sky.sky_magnitude(19.0, 0.0) is None
    assert sky.sky_magnitude(19.0, -1.0) is None
    assert sky.sky_magnitude(19.0, float("nan")) is None
    # One electron per second per square arcsecond is the zero point itself.
    assert sky.sky_magnitude(19.0, 1.0) == pytest.approx(19.0)


@pytest.mark.parametrize(
    ("bp_rp", "expected"),
    [
        (0.0, -0.02704),
        (0.8, -0.02704 + 0.01424 * 0.8 - 0.2156 * 0.64 + 0.01426 * 0.512),
        (1.0, -0.02704 + 0.01424 - 0.2156 + 0.01426),
        (3.0, -0.02704 + 0.01424 * 3 - 0.2156 * 9 + 0.01426 * 27),
    ],
)
def test_the_gaia_relation_between_g_and_v(bp_rp: float, expected: float) -> None:
    assert sky.g_minus_v(bp_rp) == pytest.approx(expected, abs=1e-9)
    assert sky.g_minus_v(1.0) == pytest.approx(-0.2141, abs=1e-4)


def test_the_relation_holds_its_value_outside_the_range() -> None:
    assert sky.g_minus_v(-3.0) == sky.g_minus_v(-0.5)
    assert sky.g_minus_v(9.0) == sky.g_minus_v(5.0)


def test_the_v_equivalent_adds_the_color_term_the_relation_and_the_offset() -> None:
    v = sky.v_equivalent(20.5, color_term=-0.08, bp_rp=1.0, offset_mag=0.12)
    assert v == pytest.approx(20.5 - 0.08 * 1.0 - sky.g_minus_v(1.0) + 0.12)
    # With no color term and no offset, V is G minus (G - V): a red sky is brighter in G than V.
    assert sky.v_equivalent(20.0, color_term=0.0, bp_rp=0.0) == pytest.approx(20.02704)


# --- Measuring the sky ---------------------------------------------------------------------


def test_the_sky_rate_is_recovered_below_one_percent() -> None:
    measurement = measure(sky_frame())
    assert measurement is not None
    assert measurement.rate_e_per_s_px == pytest.approx(4.2, rel=0.01)
    assert measurement.rate_e_per_s_arcsec2 == pytest.approx(4.2 / SCALE**2, rel=0.01)
    assert measurement.n_pixels > 150_000
    # Poisson noise of 4.9 e-/s/px * 30 s plus the read noise, in counts.
    expected = math.sqrt(4.9 * 30.0 + READ_NOISE_E**2) / E_PER_ADU
    assert measurement.noise_dn == pytest.approx(expected, rel=0.03)
    assert measurement.raw_dn == pytest.approx(BIAS_DN + 4.9 * 30.0 / E_PER_ADU, abs=0.7)


def test_a_wrong_dark_level_shifts_the_sky_by_the_difference() -> None:
    right = measure(sky_frame())
    wrong = measure(sky_frame(), dark_dn=BIAS_DN + 0.7 * 30.0 / E_PER_ADU + 5.0)
    assert right is not None
    assert wrong is not None
    assert right.level_dn - wrong.level_dn == pytest.approx(5.0, abs=0.01)


def test_stars_the_mask_hides_do_not_lift_the_sky() -> None:
    profile = synth.cropped_profile(1200, 900)
    catalog = synth.synthetic_catalog(cap_radius_deg=8.0, density_scale=1.0, seed=3)
    frame, _ = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        zero_point_mag=19.157,
        sky_e_per_s_px=4.2,
        dark_e_per_s_px=0.7,
        offset_dn=BIAS_DN,
        seed=2,
    )
    data = frame.data.astype(np.float32) / 4.0  # 14-bit counts from the 16-bit container
    detections = detect_stars(data, saturation_dn=SATURATION_DN, e_per_adu=E_PER_ADU)
    mask = star_mask(data.shape, detections)
    dark = BIAS_DN + 0.7 * 30.0 / E_PER_ADU
    masked = sky.measure_sky(
        data,
        star_mask=mask,
        dark_level_dn=dark,
        flat=None,
        exposure_s=30.0,
        e_per_adu=E_PER_ADU,
        scale_arcsec_px=SCALE,
        saturation_dn=SATURATION_DN,
    )
    assert masked is not None
    assert masked.rate_e_per_s_px == pytest.approx(4.2, rel=0.01)
    assert masked.n_pixels < data.size  # the mask and the edge took pixels away


def test_hot_pixels_saturated_pixels_and_the_edge_stay_out() -> None:
    data = sky_frame(seed=3)
    data[:8, :] = 9000.0  # the edge holds something else
    data[:, -8:] = 9000.0
    data[200:260, 100:160] = SATURATION_DN  # a saturated blob, bigger than 1% of the frame
    bad = np.zeros(data.shape, dtype=bool)
    bad[300:340, 300:360] = True
    data[bad] = 8000.0
    measurement = measure(data, bad=bad)
    assert measurement is not None
    assert measurement.rate_e_per_s_px == pytest.approx(4.2, rel=0.01)


def test_a_flat_field_removes_vignetting_from_the_sky() -> None:
    columns = np.linspace(-1.0, 1.0, 500)
    rows = np.linspace(-1.0, 1.0, 400)
    vignette = 1.0 - 0.125 * (columns[None, :] ** 2 + rows[:, None] ** 2)  # 25% less in the corners
    data = sky_frame(gradient=vignette, seed=4)
    uncorrected = measure(data)
    flat = (vignette / np.median(vignette)).astype(np.float32)
    corrected = measure(data, flat=flat)
    assert uncorrected is not None
    assert corrected is not None
    # The flat has median 1, so the corrected sky is the sky at the median sensitivity.
    assert corrected.rate_e_per_s_px == pytest.approx(4.2 * float(np.median(vignette)), rel=0.01)
    assert corrected.noise_dn < uncorrected.noise_dn  # the pattern is gone from the spread


def test_too_few_pixels_give_no_measurement() -> None:
    data = sky_frame(shape=(60, 80))
    assert measure(data) is None  # 44 x 64 pixels inside the edge
    tiny_ok = sky.measure_sky(
        data,
        star_mask=None,
        dark_level_dn=BIAS_DN + 0.7 * 30.0 / E_PER_ADU,
        flat=None,
        exposure_s=30.0,
        e_per_adu=E_PER_ADU,
        scale_arcsec_px=SCALE,
        saturation_dn=SATURATION_DN,
        options=sky.SkyOptions(edge_px=0, min_pixels=1000),
    )
    assert tiny_ok is not None


def test_the_measurement_does_not_change_between_calls() -> None:
    data = sky_frame(shape=(1500, 1500), seed=5)  # more pixels than the sample holds
    first = measure(data)
    second = measure(data)
    assert first == second
    assert first is not None
    assert first.n_pixels <= 1_000_000


def test_the_options_refuse_nonsense() -> None:
    with pytest.raises(ValueError, match="invalid sky options"):
        sky.SkyOptions(max_samples=10)


# --- The clipped median --------------------------------------------------------------------
#
# A sample of 100,000 values or more goes through a histogram. It may differ from the sorted
# computation by 0.2 of `sigma / sqrt(n)` for the median and 0.25 for the sigma, where `n` is the
# number of values. That is a fifth of the statistical error that the sample size gives anyway.

MEDIAN_TOLERANCE = 0.2
SIGMA_TOLERANCE = 0.25
SKIES = (
    "gaussian",
    "wide gaussian",
    "narrow gaussian",
    "skewed",
    "wings and hot pixels",
    "heavy tailed",
    "gradient",
    "dithered counts",
)


def made_up_sky(kind: str, n: int = 300_000) -> npt.NDArray[np.float32]:
    """The values of a sky that the clipping has to cope with, from a seed that the kind fixes."""
    rng = np.random.default_rng(100 + SKIES.index(kind))
    values: FloatArray
    if kind == "gaussian":
        values = 800.0 + 25.0 * rng.standard_normal(n)
    elif kind == "wide gaussian":
        values = 9000.0 + 100.0 * rng.standard_normal(n)
    elif kind == "narrow gaussian":
        values = 100.0 + 0.5 * rng.standard_normal(n)
    elif kind == "skewed":
        values = 200.0 + 12.0 * rng.gamma(4.0, 1.0, n)
    elif kind == "wings and hot pixels":  # 5% of faint star wings and 1% of hot pixels
        values = 800.0 + 25.0 * rng.standard_normal(n)
        values[: n // 20] += 250.0
        values[n // 20 : n // 20 + n // 100] += 25_000.0
        rng.shuffle(values)
    elif kind == "heavy tailed":
        values = 400.0 + 15.0 * rng.standard_t(3, n)
    elif kind == "gradient":
        values = 700.0 + 40.0 * rng.uniform(-1.0, 1.0, n) + 20.0 * rng.standard_normal(n)
    else:  # whole counts with the dither that `measure_sky` adds
        values = np.rint(300.0 + 1.2 * rng.standard_normal(n)) + rng.uniform(-0.5, 0.5, n)
    return values.astype(np.float32)


@pytest.mark.parametrize("kind", SKIES)
@pytest.mark.parametrize(
    ("clip_sigma", "iterations"), [(3.0, 4), (2.0, 4), (3.0, 1), (4.0, 0)], ids=str
)
def test_the_histogram_agrees_with_the_sorted_computation_to_a_fifth_of_the_error(
    kind: str, clip_sigma: float, iterations: int
) -> None:
    values = made_up_sky(kind)
    cfg = sky.SkyOptions(clip_sigma=clip_sigma, clip_iterations=iterations)
    exact_median, exact_sigma = sky._exact_clipped_median(values, cfg)
    found = sky._histogram_clipped_median(values, cfg)
    assert found is not None
    unit = exact_sigma / math.sqrt(values.size)
    assert abs(found[0] - exact_median) <= MEDIAN_TOLERANCE * unit
    assert abs(found[1] - exact_sigma) <= SIGMA_TOLERANCE * unit


def test_the_histogram_is_the_path_of_a_large_sample_and_the_sort_the_path_of_a_small_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = sky.SkyOptions()
    large = made_up_sky("gaussian", sky._HISTOGRAM_MIN_SAMPLES)
    small = large[:-1]
    assert sky._clipped_median(small, cfg) == sky._exact_clipped_median(small, cfg)  # to the bit
    assert sky._clipped_median(large, cfg) == sky._histogram_clipped_median(large, cfg)
    monkeypatch.setattr(sky, "_exact_clipped_median", lambda values, cfg: pytest.fail("sorted"))
    sky._clipped_median(large, cfg)  # a large sample never sorts the whole sample


def test_a_subsample_that_is_blind_to_the_values_gives_the_exact_result() -> None:
    values = made_up_sky("gaussian")
    values[:: values.size // sky._HISTOGRAM_ROUGH_SAMPLE] = 800.0  # what the rough step reads
    cfg = sky.SkyOptions()
    assert sky._histogram_clipped_median(values, cfg) is None  # a spread of zero
    assert sky._clipped_median(values, cfg) == sky._exact_clipped_median(values, cfg)


def test_bins_that_are_too_narrow_for_the_values_give_the_exact_result() -> None:
    values = made_up_sky("gaussian")
    rough = slice(None, None, values.size // sky._HISTOGRAM_ROUGH_SAMPLE)
    values[rough] = 800.0 + 0.1 * np.linspace(-1.0, 1.0, values[rough].size)  # a spread of 0.07
    cfg = sky.SkyOptions()  # the rest of the values have a spread of 25
    assert sky._histogram_clipped_median(values, cfg) is None  # the half-width lies out of reach
    assert sky._clipped_median(values, cfg) == sky._exact_clipped_median(values, cfg)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_value_that_is_not_finite_gives_what_the_sort_gives(bad: float) -> None:
    values = made_up_sky("gaussian")
    values[12_345] = bad
    cfg = sky.SkyOptions()
    assert sky._histogram_clipped_median(values, cfg) is None
    got = sky._clipped_median(values, cfg)
    expected = sky._exact_clipped_median(values, cfg)
    assert [math.isnan(v) for v in got] == [math.isnan(v) for v in expected]
    assert all(a == b for a, b in zip(got, expected, strict=True) if not math.isnan(a))


def test_a_constant_sample_has_a_median_and_no_spread() -> None:
    values = np.full(200_000, 812.5, dtype=np.float32)
    assert sky._clipped_median(values, sky.SkyOptions()) == (812.5, 0.0)


def test_the_cumulative_counts_place_the_median_and_the_half_width_of_a_flat_sample() -> None:
    # Ten bins of ten values each, from 0 to 10: the values lie evenly between 0 and 10.
    flat = sky._Cdf(np.arange(11.0), 10.0 * np.arange(11.0))
    assert flat.position_of(25.0) == pytest.approx(2.5)
    assert flat.position_of(50.0) == pytest.approx(5.0)
    assert flat.half_width(5.0, 0.0, 100.0) == pytest.approx(2.5)  # 20 values for each unit
    # The middle 60 values only: half of them (30) lie within 1.5 of the center.
    assert flat.half_width(5.0, 20.0, 80.0) == pytest.approx(1.5)
    # A center near the end of the bins leaves less than half of the values within reach.
    assert flat.half_width(2.0, 0.0, 100.0) is None  # half of the values need a width of 5 > 2


def test_the_counts_keep_the_values_outside_the_bins_in_the_ends() -> None:
    rng = np.random.default_rng(1)
    inside = (500.0 + 10.0 * rng.standard_normal(10_000)).astype(np.float32)
    values = np.concatenate([inside, np.full(30, -9000.0), np.full(7, 9000.0)]).astype(np.float32)
    cdf = sky._bin_counts(values, 500.0, 15.0)
    assert cdf.below[0] == 30  # under the first bin
    assert cdf.below[-1] == values.size - 7  # the 7 values over the last bin are not counted
    assert np.all(np.diff(cdf.below) >= 0.0)
    assert cdf.edges[0] == pytest.approx(500.0 - 8.0 * 15.0)
    assert cdf.edges[-1] == pytest.approx(500.0 + 8.0 * 15.0)


def test_the_sky_measurement_is_the_same_with_the_histogram_and_with_the_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = sky_frame(
        shape=(1500, 1500), seed=5
    )  # more than a million pixels: a sample of a million
    with_histogram = measure(data)
    monkeypatch.setattr(sky, "_HISTOGRAM_MIN_SAMPLES", 10**9)  # everything goes through the sort
    with_sort = measure(data)
    assert with_histogram is not None
    assert with_sort is not None
    unit = with_sort.noise_dn / math.sqrt(with_sort.n_pixels)
    assert abs(with_histogram.level_dn - with_sort.level_dn) <= MEDIAN_TOLERANCE * unit
    assert abs(with_histogram.noise_dn - with_sort.noise_dn) <= SIGMA_TOLERANCE * unit
    assert with_histogram.n_pixels == with_sort.n_pixels
    assert with_histogram.raw_dn == with_sort.raw_dn  # the median before the calibration
    assert with_histogram.rate_e_per_s_arcsec2 == pytest.approx(
        with_sort.rate_e_per_s_arcsec2, rel=1e-4
    )


# --- The flat model ------------------------------------------------------------------------


def test_a_unit_flat_changes_nothing() -> None:
    flat = sky.UnitFlat()
    assert flat.version == "unit"
    assert flat.image((10, 20)) is None
    np.testing.assert_array_equal(flat.at(np.array([1.0, 5.0]), np.array([2.0, 3.0])), [1.0, 1.0])


def test_an_array_flat_scales_to_median_one_and_cuts_a_window() -> None:
    raw = np.full((40, 60), 2.0)
    raw[:, 30:] = 3.0
    flat = sky.ArrayFlat(raw)
    assert flat.shape == (40, 60)
    assert float(np.median(flat.image((40, 60)))) == pytest.approx(1.0)  # type: ignore[arg-type]
    window = flat.image((10, 10), origin=(25, 5))
    assert window is not None
    assert window.shape == (10, 10)
    assert window[0, 0] == pytest.approx(0.8)  # columns 25 to 29 hold the low half
    assert window[0, 9] == pytest.approx(1.2)  # and columns 30 to 34 the high half
    np.testing.assert_allclose(flat.at(np.array([0.0, 59.0]), np.array([0.0, 39.0])), [0.8, 1.2])
    assert flat.version.startswith("flat-")
    assert sky.ArrayFlat(raw).version == flat.version  # the same data, the same version
    with pytest.raises(sky.SkyError, match="does not fit"):
        flat.image((50, 50))


def test_a_flat_must_be_a_positive_image() -> None:
    with pytest.raises(sky.SkyError, match="positive"):
        sky.ArrayFlat(np.zeros((4, 4)))
    with pytest.raises(sky.SkyError, match="positive"):
        sky.ArrayFlat(np.array([1.0, 2.0]))


def test_a_flat_loads_from_npy_or_fits_and_an_empty_path_gives_a_unit_flat(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(6)
    raw = rng.uniform(0.9, 1.1, (20, 30)).astype(np.float32)
    np.save(tmp_path / "flat.npy", raw)
    fitsio.write_image(tmp_path / "flat.fits", raw)
    from_npy = sky.load_flat(tmp_path / "flat.npy")
    from_fits = sky.load_flat(tmp_path / "flat.fits")
    assert from_npy.version == from_fits.version
    assert sky.load_flat("").version == "unit"
    with pytest.raises(sky.SkyError, match="cannot read the flat"):
        sky.load_flat(tmp_path / "missing.npy")
    (tmp_path / "bad.fits").write_bytes(b"not fits")
    with pytest.raises(sky.SkyError, match="cannot read the flat"):
        sky.load_flat(tmp_path / "bad.fits")
