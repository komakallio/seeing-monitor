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
from seeingmon.profile import Profile, ProfileError, derived
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
    assert (bin1.row_time_us, bin1.frame_overhead_ms) == (37.6, 7.37)
    assert (bin2.row_time_us, bin2.frame_overhead_ms) == (18.5, 1.22)


def test_only_bin2_states_the_snapshot_model(reference: Profile) -> None:
    """From a Raspberry Pi 4 on October 3, 2026. Bin1 is unmeasured, so it states none."""
    bin1, bin2 = reference.mode("bin1"), reference.mode("bin2")
    assert (bin1.snapshot_overhead_s, bin1.snapshot_row_time_us) == (None, None)
    assert (bin2.snapshot_overhead_s, bin2.snapshot_row_time_us) == (0.27, 75.0)
    assert bin2.has_snapshot_model
    assert not bin1.has_snapshot_model


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


def test_a_bin1_roi_of_128_rows_takes_12_2_ms_and_runs_at_82_fps(reference: Profile) -> None:
    period = reference.frame_period_s("bin1", 128, exposure_us=2000)
    assert period == pytest.approx(12.18e-3, abs=0.05e-3)  # 7.37 ms + 128 x 37.6 us
    assert reference.max_frame_rate_hz("bin1", 128, exposure_us=2000) == pytest.approx(82, abs=0.5)


def test_a_bin2_roi_of_64_rows_takes_2_4_ms(reference: Profile) -> None:
    period = reference.frame_period_s("bin2", 64, exposure_us=1000)
    assert period == pytest.approx(2.4e-3, abs=0.05e-3)  # 1.22 ms + 64 x 18.5 us


def test_a_bin2_roi_of_64_rows_runs_at_416_fps_for_short_exposures(reference: Profile) -> None:
    rate = reference.max_frame_rate_hz("bin2", 64, exposure_us=1000)
    assert rate == pytest.approx(416, abs=3)  # the camera measured 417 at 0.5 ms


def test_a_bin2_roi_of_64_rows_runs_at_100_fps_for_a_10_ms_exposure(reference: Profile) -> None:
    assert reference.max_frame_rate_hz("bin2", 64, exposure_us=10_000) == pytest.approx(100)


def test_the_row_time_in_nanoseconds(reference: Profile) -> None:
    assert reference.row_time_ns("bin1") == 37_600
    assert reference.row_time_ns("bin2") == 18_500


# The frame rates of the real camera (`seeingmon camera rates` at USB bandwidth 100, October 2026,
# SDK 1.41, 150 frames per row): (mode, high speed, ROI rows, frames per second). The profile is a
# straight-line fit to these rows, so it must reproduce them. ZWO's published table, which the
# research notes quote, is slower by about 0.9 ms of overhead in the small ROIs of the fast
# streams, and in bin2 it has a larger row time for the 14-bit readout.
MEASURED_FRAME_RATES = [
    ("bin1", False, 32, 116.7),
    ("bin1", False, 64, 102.3),
    ("bin1", False, 128, 82.1),
    ("bin1", False, 256, 58.8),
    ("bin1", False, 512, 37.6),
    ("bin1", True, 32, 146.2),
    ("bin1", True, 64, 128.2),
    ("bin1", True, 128, 102.9),
    ("bin1", True, 256, 73.7),
    ("bin1", True, 512, 47.1),
    ("bin2", False, 64, 417.1),
    ("bin2", False, 128, 278.6),
    ("bin2", False, 256, 167.9),
    ("bin2", False, 512, 93.5),
]


@pytest.mark.parametrize(
    ("mode", "high_speed", "rows", "measured"),
    MEASURED_FRAME_RATES,
    ids=[f"{m}-{'hs' if h else 'normal'}-{r}" for m, h, r, _ in MEASURED_FRAME_RATES],
)
def test_frame_rates_match_the_measured_table_within_2_percent(
    reference: Profile, mode: str, high_speed: bool, rows: int, measured: float
) -> None:
    readout = reference.mode(mode, high_speed=high_speed)
    assert reference.max_frame_rate_hz(readout, rows, exposure_us=500) == pytest.approx(
        measured, rel=0.02
    )


# --- Single exposures ----------------------------------------------------------------------


def test_a_bin2_snapshot_of_the_watch_roi_takes_what_the_camera_measured(
    reference: Profile,
) -> None:
    """A 1 ms exposure of the 20 arcminute ROI (312 x 314 pixels in bin2) took 0.293 to 0.296 s on
    a Raspberry Pi 4. The video line gives 7 ms for it."""
    assert reference.roi_size_px("bin2", 20.0) == (312, 314)
    period = reference.snapshot_period_s("bin2", 314, exposure_us=1000)
    assert 0.293 <= period <= 0.296
    assert period == pytest.approx(0.001 + 0.27 + 314 * 75e-6)
    assert reference.frame_period_s("bin2", 314, exposure_us=1000) == pytest.approx(
        7.0e-3, abs=0.1e-3
    )


def test_a_bin2_snapshot_of_the_full_frame_is_not_shorter_than_the_camera_measured(
    reference: Profile,
) -> None:
    """A 1 ms exposure of the full frame took 0.480 to 0.532 s. The video line gives 53 ms, which
    left the scheduler about 0.1 s of its 0.61 s."""
    period = reference.snapshot_period_s("bin2", 2822, exposure_us=1000)
    assert period == pytest.approx(0.4827, abs=1e-4)
    assert 0.480 <= period <= 0.532
    assert reference.frame_period_s("bin2", 2822, exposure_us=1000) == pytest.approx(
        53.4e-3, abs=0.1e-3
    )


def test_a_2_s_snapshot_of_the_full_frame_takes_the_exposure_plus_the_readout(
    reference: Profile,
) -> None:
    """The camera took 2.52 s: the exposure plus about 0.52 s."""
    period = reference.snapshot_period_s("bin2", 2822, exposure_us=2_000_000)
    assert period == pytest.approx(2.0 + reference.snapshot_readout_time_s("bin2", 2822))
    assert period == pytest.approx(2.48, abs=0.01)


def test_the_snapshot_readout_grows_with_the_rows(reference: Profile) -> None:
    times = [reference.snapshot_readout_time_s("bin2", rows) for rows in (2, 314, 1000, 2822)]
    assert times == sorted(times)
    assert times[0] == pytest.approx(0.27 + 2 * 75e-6)


def test_bin1_takes_the_video_row_time_and_the_overhead_floor_for_a_snapshot(
    reference: Profile,
) -> None:
    bin1 = reference.mode("bin1")
    assert derived.snapshot_overhead_s(bin1) == 0.3 == derived.SNAPSHOT_OVERHEAD_FLOOR_S
    assert derived.snapshot_row_time_us(bin1) == 37.6
    assert reference.snapshot_readout_time_s("bin1", 128) == pytest.approx(0.3 + 128 * 37.6e-6)
    # Never shorter than the video model, and never below the overhead that the camera showed.
    assert reference.snapshot_readout_time_s("bin1", 128) >= derived.readout_time_s(bin1, 128)
    assert derived.snapshot_overhead_s(bin1) >= 0.27


def test_the_high_speed_variant_keeps_the_snapshot_model(reference: Profile) -> None:
    fast = reference.mode("bin2", high_speed=True)
    assert (fast.snapshot_overhead_s, fast.snapshot_row_time_us) == (0.27, 75.0)
    fast1 = reference.mode("bin1", high_speed=True)
    assert derived.snapshot_overhead_s(fast1) == 0.3  # the floor applies to the high-speed mode too
    assert derived.snapshot_row_time_us(fast1) == 30.0  # and the video row time of that mode


def test_a_snapshot_asks_for_a_valid_roi_height_and_exposure(reference: Profile) -> None:
    with pytest.raises(ValueError, match="ROI height must be 1 to 2822"):
        reference.snapshot_period_s("bin2", 2824, exposure_us=1000)
    with pytest.raises(ValueError, match="ROI height must be 1 to 2822"):
        reference.snapshot_readout_time_s("bin2", 0)
    with pytest.raises(ValueError, match="exposure_us must be positive"):
        reference.snapshot_period_s("bin2", 64, exposure_us=0)


def test_the_data_rate_of_a_fast_bin1_stream(reference: Profile) -> None:
    rate = reference.data_rate_bytes_per_s("bin1", 128, 128, exposure_us=2000)
    assert rate == pytest.approx(2.69e6, rel=0.01)  # 32 KB x 82 fps
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
    assert (fast.row_time_us, fast.frame_overhead_ms) == (30.0, 5.88)
    assert reference.frame_period_s(fast, 128, exposure_us=2000) == pytest.approx(
        5.88e-3 + 128 * 30.0e-6
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
