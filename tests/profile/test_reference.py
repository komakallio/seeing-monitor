"""The reference profile reproduces the numbers in docs/research-notes.md.

Tolerances: the research notes print plate scales and fields of view to three decimals, so
`abs=5e-4` covers their rounding. They print Airy sizes and sampling ratios to two decimals,
and two of those entries round differently from the exact arithmetic (the Airy FWHM at 550 nm
is 2.335 arcsec, which the notes print as 2.34, and the bin2 sampling ratio at 650 nm is 1.425,
which they print as 1.43). The tolerance for those is `abs=0.01`.
"""

from __future__ import annotations

import pytest

from seeingmon.frames import PixelFormat
from seeingmon.profile import Profile, ProfileError
from seeingmon.profile.derived import (
    adc_full_scale,
    sensor_diagonal_mm,
    sensor_size_mm,
    usable_radius_px,
)

MODES = ["bin1", "bin2"]


# --- Identity and structure ----------------------------------------------------------------


def test_the_reference_profile_names_its_fast_and_survey_modes(reference: Profile) -> None:
    assert reference.id == "asi294mm-gs250"
    assert reference.fast_mode.mode == "bin1"
    assert reference.survey_mode.mode == "bin2"
    assert reference.fast_readout.name == "bin1"
    assert reference.survey_readout.name == "bin2"
    assert reference.fast_mode.pixel_format is PixelFormat.RAW16
    assert reference.survey_mode.pixel_format is PixelFormat.RAW16


def test_the_reference_camera_is_uncooled_with_a_temperature_sensor(reference: Profile) -> None:
    assert reference.sensor.cooled is False
    assert reference.sensor.has_temperature_sensor is True


def test_the_reference_profile_uses_the_full_sensor_as_the_image_circle(
    reference: Profile,
) -> None:
    assert reference.optics.image_circle_diameter_mm is None
    for mode in MODES:
        assert usable_radius_px(reference.mode(mode), reference.optics) is None


def test_the_reference_limits_follow_the_sdk_rules(reference: Profile) -> None:
    assert reference.limits.roi_width_multiple == 8
    assert reference.limits.roi_height_multiple == 2
    assert reference.limits.gain_range == (0, 570)
    assert reference.limits.exposure_us_range == (32, 2_000_000_000)  # 32 us to 2000 s


def test_the_readout_modes_carry_the_documented_values(reference: Profile) -> None:
    bin1, bin2 = reference.mode("bin1"), reference.mode("bin2")
    assert (bin1.sdk_bin, bin1.width_px, bin1.height_px, bin1.pixel_size_um) == (
        1,
        8288,
        5644,
        2.315,
    )
    assert (bin2.sdk_bin, bin2.width_px, bin2.height_px, bin2.pixel_size_um) == (
        2,
        4144,
        2822,
        4.63,
    )
    assert (bin1.adc_bits, bin1.adc_bits_high_speed) == (12, 10)
    assert (bin2.adc_bits, bin2.adc_bits_high_speed) == (14, 12)
    assert (bin1.full_well_gain0_e, bin1.e_per_adu_gain0, bin1.read_noise_gain0_e) == (
        14417,
        3.5,
        2.65,
    )
    assert (bin2.full_well_gain0_e, bin2.e_per_adu_gain0, bin2.read_noise_gain0_e) == (
        66387,
        4.05,
        8.0,
    )
    assert (bin1.row_time_us, bin1.frame_overhead_ms) == (37.6, 6.5)
    assert (bin2.row_time_us, bin2.frame_overhead_ms) == (21.3, 1.4)


def test_the_hcg_step_is_marked_at_gain_120_in_bin2_only(reference: Profile) -> None:
    steps = {
        mode.name: [point.gain for point in mode.gain_points if point.step]
        for mode in reference.readout_modes
    }
    assert steps == {"bin1": [], "bin2": [120]}


# --- Geometry ------------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "expected"), [("bin1", 1.910), ("bin2", 3.820)])
def test_plate_scale_matches_the_research_notes(
    reference: Profile, mode: str, expected: float
) -> None:
    assert reference.plate_scale_arcsec_per_px(mode) == pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize("mode", MODES)
def test_field_of_view_is_the_same_in_both_modes(reference: Profile, mode: str) -> None:
    fov = reference.field_of_view(mode)
    assert fov.width_deg == pytest.approx(4.395, abs=5e-4)
    assert fov.height_deg == pytest.approx(2.994, abs=5e-4)
    assert fov.diagonal_deg == pytest.approx(5.316, abs=5e-4)


def test_the_sensor_area_matches_the_research_notes(reference: Profile) -> None:
    for mode in MODES:
        width_mm, height_mm = sensor_size_mm(reference.mode(mode))
        assert width_mm == pytest.approx(19.187, abs=1e-3)
        assert height_mm == pytest.approx(13.066, abs=1e-3)
        assert sensor_diagonal_mm(reference.mode(mode)) == pytest.approx(23.213, abs=1e-3)


def test_a_4_1_arcmin_roi_is_128_pixels_in_bin1(reference: Profile) -> None:
    assert reference.roi_size_px("bin1", 4.1) == (128, 128)


def test_a_4_1_arcmin_roi_is_64_pixels_in_bin2(reference: Profile) -> None:
    """The architecture's fast-frame table lists a 64 x 64 bin2 ROI."""
    assert reference.roi_size_px("bin2", 4.1) == (64, 64)


# --- Sampling ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wavelength_nm", "expected"), [(550.0, 2.34), (700.0, 2.97)], ids=["550nm", "700nm"]
)
def test_airy_fwhm_in_arcsec_matches_the_research_notes(
    reference: Profile, wavelength_nm: float, expected: float
) -> None:
    assert reference.airy_fwhm_arcsec(wavelength_nm) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    ("mode", "wavelength_nm", "expected"),
    [
        ("bin1", 550.0, 1.22),
        ("bin1", 700.0, 1.56),
        ("bin2", 550.0, 0.61),
        ("bin2", 700.0, 0.78),
    ],
)
def test_airy_fwhm_in_pixels_matches_the_research_notes(
    reference: Profile, mode: str, wavelength_nm: float, expected: float
) -> None:
    assert reference.airy_fwhm_px(mode, wavelength_nm) == pytest.approx(expected, abs=0.01)


def test_the_airy_fwhm_defaults_to_the_effective_wavelength(reference: Profile) -> None:
    assert reference.optics.wavelength_nm == 600.0
    assert reference.airy_fwhm_arcsec() == pytest.approx(reference.airy_fwhm_arcsec(600.0))
    # Research notes, larger scopes table: 2.55 arcsec at 0.6 um; 1.34 and 0.67 pixels.
    assert reference.airy_fwhm_arcsec() == pytest.approx(2.55, abs=0.01)
    assert reference.airy_fwhm_px("bin1") == pytest.approx(1.34, abs=0.01)
    assert reference.airy_fwhm_px("bin2") == pytest.approx(0.67, abs=0.01)


@pytest.mark.parametrize(("mode", "expected"), [("bin1", 0.71), ("bin2", 1.43)])
def test_sampling_ratio_at_650_nm_matches_the_research_notes(
    reference: Profile, mode: str, expected: float
) -> None:
    assert reference.sampling_ratio(mode, 650.0) == pytest.approx(expected, abs=0.01)


def test_only_bin1_samples_below_the_centroid_phase_criterion(reference: Profile) -> None:
    assert reference.sampling_ratio("bin1", 650.0) < 1.0 < reference.sampling_ratio("bin2", 650.0)


# --- Frame rates ---------------------------------------------------------------------------


def test_a_bin1_roi_of_128_rows_takes_11_3_ms_and_runs_at_88_fps(reference: Profile) -> None:
    period = reference.frame_period_s("bin1", 128, exposure_us=2000)
    assert period == pytest.approx(11.3e-3, abs=0.05e-3)  # 6.5 ms + 128 x 37.6 us
    assert reference.max_frame_rate_hz("bin1", 128, exposure_us=2000) == pytest.approx(88, abs=0.5)


def test_a_bin2_roi_of_64_rows_takes_2_8_ms(reference: Profile) -> None:
    period = reference.frame_period_s("bin2", 64, exposure_us=1000)
    assert period == pytest.approx(2.8e-3, abs=0.05e-3)  # 1.4 ms + 64 x 21.3 us


def test_a_bin2_roi_of_64_rows_runs_at_360_fps_for_short_exposures(reference: Profile) -> None:
    rate = reference.max_frame_rate_hz("bin2", 64, exposure_us=1000)
    assert rate == pytest.approx(360, abs=3)  # 361.9: the notes round down


def test_a_bin2_roi_of_64_rows_runs_at_100_fps_for_a_10_ms_exposure(reference: Profile) -> None:
    assert reference.max_frame_rate_hz("bin2", 64, exposure_us=10_000) == pytest.approx(100)


def test_the_row_time_in_nanoseconds(reference: Profile) -> None:
    assert reference.row_time_ns("bin1") == 37_600
    assert reference.row_time_ns("bin2") == 21_300


# ZWO's USB 3.0 frame rates (research notes, "Frame rates and row timing"): (rows, bin1 12-bit,
# bin1 10-bit, bin2 14-bit, bin2 12-bit). The notes say the line model fits within 1 to 2%.
ZWO_FRAME_RATES = [
    (5644, 4.6, 5.7, None, None),
    (2822, None, None, 16.3, 19.0),
    (1080, 21.2, 26.6, 41.0, 47.9),
    (480, 40.8, 51.1, 86.0, 100.5),
    (240, 64.6, 80.9, 153.4, 179.3),
]


@pytest.mark.parametrize(
    ("mode", "high_speed", "column"),
    [("bin1", False, 1), ("bin1", True, 2), ("bin2", False, 3), ("bin2", True, 4)],
    ids=["bin1-12bit", "bin1-10bit", "bin2-14bit", "bin2-12bit"],
)
def test_frame_rates_match_zwos_published_table_within_2_percent(
    reference: Profile, mode: str, high_speed: bool, column: int
) -> None:
    readout = reference.mode(mode, high_speed=high_speed)
    for row in ZWO_FRAME_RATES:
        published = row[column]
        if published is None or row[0] > readout.height_px:
            continue
        rate = reference.max_frame_rate_hz(readout, row[0], exposure_us=100)
        assert rate == pytest.approx(published, rel=0.02)


def test_the_data_rate_of_a_fast_bin1_stream(reference: Profile) -> None:
    rate = reference.data_rate_bytes_per_s("bin1", 128, 128, exposure_us=2000)
    assert rate == pytest.approx(2.9e6, rel=0.01)  # 32 KB x 88 fps: "2.9 MB/s" in the architecture
    raw8 = reference.data_rate_bytes_per_s("bin1", 128, 128, 2000, pixel_format=PixelFormat.RAW8)
    assert raw8 == pytest.approx(rate / 2)


# --- ADC, gain, and saturation ----------------------------------------------------------------


def test_the_adc_full_scale(reference: Profile) -> None:
    assert adc_full_scale(reference.mode("bin1")) == 4095
    assert adc_full_scale(reference.mode("bin2")) == 16383


def test_the_high_speed_modes_have_fewer_adc_bits(reference: Profile) -> None:
    assert adc_full_scale(reference.mode("bin1", high_speed=True)) == 1023
    assert adc_full_scale(reference.mode("bin2", high_speed=True)) == 4095


def test_the_high_speed_variant_changes_the_timing(reference: Profile) -> None:
    fast = reference.mode("bin1", high_speed=True)
    assert (fast.row_time_us, fast.frame_overhead_ms) == (30.1, 5.0)
    assert reference.frame_period_s(fast, 128, exposure_us=2000) == pytest.approx(
        5.0e-3 + 128 * 30.1e-6
    )


def test_the_high_speed_variant_keeps_the_saturation_electrons_and_the_container_scale(
    reference: Profile,
) -> None:
    """Fewer ADC bits mean more electrons per ADU, so the well in electrons stays the same."""
    normal = reference.saturation("bin1", 0)
    fast = reference.saturation(reference.mode("bin1", high_speed=True), 0)
    assert fast.native_dn == 1023
    assert fast.container_dn == 1023 * 64  # RAW16 shifts a 10-bit value up by 6 bits
    assert fast.full_well_e == pytest.approx(normal.full_well_e, rel=0.01)


@pytest.mark.parametrize(
    ("mode", "gain", "full_well_e", "native_dn", "container_dn"),
    [
        # Bin1 at gain 0: the well would reach 4,119 ADU, so the 12-bit ADC (4,095) clips first.
        ("bin1", 0, 3.5 * 4095, 4095, 4095 * 16),
        # Bin1 gain 108 (unity gain): the ADC clips at 4,095 e-, far below the 14,417 e- well.
        ("bin1", 108, 4095.0, 4095, 4095 * 16),
        ("bin1", 270, 0.17 * 4095, 4095, 4095 * 16),
        ("bin2", 0, 4.05 * 16383, 16383, 16383 * 4),
        ("bin2", 119, 1.03 * 16383, 16383, 16383 * 4),
        # Bin2 gain 120: the HCG step lowers the full well in electrons from 16.9 ke- to 14.4 ke-.
        ("bin2", 120, 0.88 * 16383, 16383, 16383 * 4),
        ("bin2", 300, 0.11 * 16383, 16383, 16383 * 4),
    ],
)
def test_saturation_at_gain_0_and_at_the_gain_steps(
    reference: Profile,
    mode: str,
    gain: int,
    full_well_e: float,
    native_dn: float,
    container_dn: float,
) -> None:
    saturation = reference.saturation(mode, gain)
    assert saturation.limited_by == "adc"
    assert saturation.native_dn == pytest.approx(native_dn)
    assert saturation.container_dn == pytest.approx(container_dn)  # the SDK shifts the ADC value up
    assert saturation.full_well_e == pytest.approx(full_well_e)
    assert reference.full_well_e(mode, gain) == pytest.approx(full_well_e)


def test_bin1_gain_108_saturates_at_the_adc_full_scale_not_at_the_well(reference: Profile) -> None:
    saturation = reference.saturation("bin1", 108)
    assert saturation.limited_by == "adc"
    assert saturation.native_dn == adc_full_scale(reference.mode("bin1")) == 4095
    assert reference.full_well_e("bin1", 108) < reference.mode("bin1").full_well_gain0_e


def test_the_hcg_step_lowers_the_full_well_in_electrons(reference: Profile) -> None:
    assert reference.full_well_e("bin2", 119) == pytest.approx(16874, abs=1)
    assert reference.full_well_e("bin2", 120) == pytest.approx(14417, abs=1)


def test_the_gain_table_returns_its_own_rows_at_the_row_gains(reference: Profile) -> None:
    for mode in reference.readout_modes:
        assert reference.e_per_adu(mode, 0) == mode.e_per_adu_gain0
        assert reference.read_noise_e(mode, 0) == mode.read_noise_gain0_e
        for point in mode.gain_points:
            assert reference.e_per_adu(mode, point.gain) == point.e_per_adu
            assert reference.read_noise_e(mode, point.gain) == point.read_noise_e


def test_electrons_per_adu_interpolates_log_linearly_between_rows(reference: Profile) -> None:
    # Halfway in gain between gain 0 (3.5) and gain 108 (1.0) is the geometric mean.
    assert reference.e_per_adu("bin1", 54) == pytest.approx((3.5 * 1.0) ** 0.5)
    # The read noise interpolates linearly in gain: halfway between 2.65 and 1.8.
    assert reference.read_noise_e("bin1", 54) == pytest.approx((2.65 + 1.8) / 2)


def test_a_step_is_never_interpolated(reference: Profile) -> None:
    assert reference.e_per_adu("bin2", 119) == 1.03
    assert reference.e_per_adu("bin2", 119.5) == 1.03  # still on the low side of the step
    assert reference.e_per_adu("bin2", 120) == 0.88
    assert reference.read_noise_e("bin2", 119.5) == 6.2
    assert reference.read_noise_e("bin2", 120) == 1.85
    # Below the step, the segment from gain 0 to gain 119 interpolates as usual.
    assert reference.e_per_adu("bin2", 60) == pytest.approx(4.05 * (1.03 / 4.05) ** (60 / 119))


def test_the_values_hold_above_the_last_row_of_the_table(reference: Profile) -> None:
    """The table ends where the vendor's chart ends: gain 270 in bin1 and gain 300 in bin2."""
    assert reference.e_per_adu("bin1", 400) == 0.17
    assert reference.read_noise_e("bin1", 400) == 1.38
    assert reference.e_per_adu("bin2", 570) == 0.11
    assert reference.read_noise_e("bin2", 570) == 1.3
    assert reference.saturation("bin1", 570).native_dn == 4095


def test_a_negative_gain_is_an_error(reference: Profile) -> None:
    with pytest.raises(ValueError, match="gain must not be negative"):
        reference.e_per_adu("bin1", -1)


# --- Photometry ----------------------------------------------------------------------------


def test_the_magnitude_0_electron_rate_is_a_flagged_estimate(reference: Profile) -> None:
    assert reference.photometry is not None
    assert reference.photometry.mag0_electron_rate_e_per_s == 4.6e7
    assert reference.photometry.mag0_electron_rate_rel_uncertainty == 0.3


def test_star_rates_scale_by_the_magnitude(reference: Profile) -> None:
    assert reference.star_electron_rate_e_per_s(0.0) == pytest.approx(4.6e7)
    assert reference.star_electron_rate_e_per_s(5.0) == pytest.approx(4.6e7 / 100)  # 5 mag = 100x


def test_polaris_collects_about_the_photoelectrons_that_the_architecture_states(
    reference: Profile,
) -> None:
    """The architecture says 87,000 electrons in 10 ms, good to 30%. Polaris is V = 2.02."""
    electrons = reference.star_electron_rate_e_per_s(2.02) * 0.010
    assert electrons == pytest.approx(87_000, rel=0.30)


def test_the_dark_current_prior_returns_the_chart_points(reference: Profile) -> None:
    for temperature_c, rate in [(-20, 0.0022), (0, 0.019), (20, 0.20), (30, 0.70)]:
        assert reference.dark_current_e_per_s_per_px(temperature_c) == pytest.approx(rate)


def test_the_dark_current_prior_doubles_about_every_6_degrees(reference: Profile) -> None:
    ratio = reference.dark_current_e_per_s_per_px(26.0) / reference.dark_current_e_per_s_per_px(
        20.0
    )
    assert ratio == pytest.approx(2.0, rel=0.2)  # the notes give a doubling temperature of 5.8 C


def test_the_dark_current_prior_interpolates_log_linearly(reference: Profile) -> None:
    assert reference.dark_current_e_per_s_per_px(15.0) == pytest.approx((0.065 * 0.20) ** 0.5)


def test_the_dark_current_prior_continues_beyond_the_table(reference: Profile) -> None:
    hot = reference.dark_current_e_per_s_per_px(35.0)
    assert hot == pytest.approx(
        0.70 * (0.70 / 0.36)
    )  # one more step of the last segment (25 to 30 C)
    assert reference.dark_current_e_per_s_per_px(-30.0) < 0.0022


def test_a_profile_without_photometry_refuses_photometric_questions(reference: Profile) -> None:
    bare = reference.model_copy(update={"photometry": None})
    with pytest.raises(ProfileError, match="mag0_electron_rate_e_per_s"):
        bare.star_electron_rate_e_per_s(2.0)
    with pytest.raises(ProfileError, match="dark_current"):
        bare.dark_current_e_per_s_per_px(10.0)
