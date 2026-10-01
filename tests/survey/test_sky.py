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
