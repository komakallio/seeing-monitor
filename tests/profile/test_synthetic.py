"""Nothing is hard-coded: other sensors, pixel sizes, bit depths, and optics give other values.

The synthetic profile in `tests/profile/builders.py` shares no number with the reference
profile (a 7.5 x 5.625 mm sensor, 400 mm focal length, 80 mm aperture, and a 16-bit mode).
The expected values below come from the closed-form formulas, written out in each test.
"""

from __future__ import annotations

import math

import pytest

from seeingmon.frames import PixelFormat, Roi
from seeingmon.profile import Profile, parse_profile
from seeingmon.profile.derived import (
    adc_full_scale,
    sensor_diagonal_mm,
    sensor_size_mm,
    usable_radius_px,
)
from tests.profile.builders import with_optics

ARCSEC_PER_RAD = 180 * 3600 / math.pi


# --- The synthetic profile --------------------------------------------------------------------


def test_the_synthetic_profile_names_its_modes(synthetic: Profile) -> None:
    assert synthetic.id == "synthetic-cam"
    assert synthetic.fast_readout.name == "native"
    assert synthetic.survey_readout.name == "bin2"
    assert synthetic.fast_mode.pixel_format is None
    assert synthetic.survey_mode.pixel_format is PixelFormat.RAW8
    assert synthetic.sensor.cooled is True
    assert synthetic.limits.offset_range == (0, 255)
    assert synthetic.photometry is None


@pytest.mark.parametrize(
    ("mode", "pixel_um"), [("native", 3.75), ("bin2", 7.5)], ids=["native", "bin2"]
)
def test_the_plate_scale_follows_the_pixel_size_and_the_focal_length(
    synthetic: Profile, mode: str, pixel_um: float
) -> None:
    expected = (pixel_um * 1e-3 / 400.0) * ARCSEC_PER_RAD  # pixel and focal length in mm
    assert synthetic.plate_scale_arcsec_per_px(mode) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("mode", ["native", "bin2"])
def test_the_field_of_view_follows_the_sensor_size(synthetic: Profile, mode: str) -> None:
    fov = synthetic.field_of_view(mode)
    assert fov.width_deg == pytest.approx(math.degrees(2 * math.atan(7.5 / 800)), rel=1e-9)
    assert fov.height_deg == pytest.approx(math.degrees(2 * math.atan(5.625 / 800)), rel=1e-9)
    assert fov.diagonal_deg == pytest.approx(
        math.degrees(2 * math.atan(math.hypot(7.5, 5.625) / 800)), rel=1e-9
    )


def test_the_sensor_size_comes_from_the_resolution_and_the_pixel_size(synthetic: Profile) -> None:
    for mode in ("native", "bin2"):
        width_mm, height_mm = sensor_size_mm(synthetic.mode(mode))
        assert (width_mm, height_mm) == pytest.approx((7.5, 5.625))
        assert sensor_diagonal_mm(synthetic.mode(mode)) == pytest.approx(9.375)


def test_the_airy_size_follows_the_aperture_and_the_default_wavelength(synthetic: Profile) -> None:
    expected_arcsec = 1.029 * 550e-9 / 0.080 * ARCSEC_PER_RAD  # the default is 550 nm, not 600 nm
    assert synthetic.airy_fwhm_arcsec() == pytest.approx(expected_arcsec, rel=1e-9)
    assert synthetic.airy_fwhm_px("native") == pytest.approx(
        expected_arcsec / synthetic.plate_scale_arcsec_per_px("native"), rel=1e-9
    )


def test_the_sampling_ratio_follows_the_f_number(synthetic: Profile) -> None:
    assert synthetic.optics.f_number == pytest.approx(5.0)
    assert synthetic.sampling_ratio("native") == pytest.approx(3.75 / (0.550 * 5.0))
    assert synthetic.sampling_ratio("bin2", 650.0) == pytest.approx(7.5 / (0.650 * 5.0))


def test_the_image_circle_converts_to_pixels_of_each_mode(synthetic: Profile) -> None:
    """A 6 mm circle has a 3 mm radius: 800 pixels of 3.75 um, or 400 pixels of 7.5 um."""
    assert usable_radius_px(synthetic.mode("native"), synthetic.optics) == pytest.approx(800)
    assert usable_radius_px(synthetic.mode("bin2"), synthetic.optics) == pytest.approx(400)


def test_the_roi_rules_come_from_the_limits(synthetic: Profile) -> None:
    """The synthetic ROI steps are 4 x 4, where the reference steps are 8 x 2."""
    scale = 3.75e-3 / 400.0 * ARCSEC_PER_RAD
    pixels = 5 * 60 / scale  # 155.14 pixels for a 5 arcmin patch
    assert pixels / 4 == pytest.approx(38.785, abs=1e-3)
    assert synthetic.roi_size_px("native", 5.0) == (156, 156)  # 39 steps of 4
    assert synthetic.clamp_roi("native", 10, 10, 18, 18) == Roi(10, 10, 20, 20)
    assert synthetic.clamp_roi("native", 10, 10, 17, 17) == Roi(10, 10, 16, 16)


def test_the_roi_clamps_to_the_frame_of_the_mode(synthetic: Profile) -> None:
    assert synthetic.clamp_roi("bin2", -10, -10, 5000, 5000) == Roi(0, 0, 1000, 748)
    assert synthetic.clamp_roi("bin2", 990, 740, 100, 100) == Roi(900, 650, 100, 100)


def test_the_frame_period_follows_the_row_time_and_the_overhead(synthetic: Profile) -> None:
    assert synthetic.row_time_ns("native") == 10_000
    assert synthetic.row_time_ns("bin2") == 6_000
    assert synthetic.frame_period_s("native", 100, exposure_us=100) == pytest.approx(
        2.0e-3 + 100 * 10e-6
    )
    assert synthetic.max_frame_rate_hz("native", 100, exposure_us=100) == pytest.approx(1 / 3e-3)
    assert synthetic.frame_period_s("bin2", 50, exposure_us=5000) == pytest.approx(5e-3)  # exposure


def test_a_mode_without_a_snapshot_model_takes_its_video_row_time_and_the_overhead_floor(
    synthetic: Profile,
) -> None:
    """The synthetic modes state no snapshot values, and their video overheads (2 ms and 0.8 ms)
    sit below the floor of 0.3 s, so the floor applies."""
    assert synthetic.snapshot_readout_time_s("native", 100) == pytest.approx(0.3 + 100 * 10e-6)
    assert synthetic.snapshot_readout_time_s("bin2", 50) == pytest.approx(0.3 + 50 * 6e-6)
    assert synthetic.snapshot_period_s("native", 100, exposure_us=2000) == pytest.approx(
        0.002 + 0.3 + 100 * 10e-6
    )


def test_a_snapshot_model_in_another_profile_follows_that_profile(synthetic: Profile) -> None:
    slow = synthetic.model_copy(
        update={
            "readout_modes": tuple(
                mode.model_copy(update={"snapshot_overhead_s": 1.5, "snapshot_row_time_us": 200.0})
                for mode in synthetic.readout_modes
            )
        }
    )
    assert slow.snapshot_readout_time_s("native", 100) == pytest.approx(1.5 + 100 * 200e-6)
    assert slow.snapshot_period_s("bin2", 50, exposure_us=5000) == pytest.approx(
        0.005 + 1.5 + 50 * 200e-6
    )
    assert slow.frame_period_s("native", 100, exposure_us=100) == pytest.approx(
        2.0e-3 + 100 * 10e-6  # the video model does not change
    )


def test_the_data_rate_follows_the_pixel_format(synthetic: Profile) -> None:
    rate16 = synthetic.data_rate_bytes_per_s("native", 64, 64, 100)
    rate8 = synthetic.data_rate_bytes_per_s("native", 64, 64, 100, PixelFormat.RAW8)
    assert rate16 == pytest.approx(64 * 64 * 2 / (2.0e-3 + 64 * 10e-6))
    assert rate8 == pytest.approx(rate16 / 2)


def test_the_adc_depth_comes_from_the_mode(synthetic: Profile) -> None:
    assert adc_full_scale(synthetic.mode("native")) == 65535
    assert adc_full_scale(synthetic.mode("bin2")) == 16383


# --- Gain tables and saturation in the synthetic profile -----------------------------------------


def test_electrons_per_adu_interpolates_between_the_synthetic_rows(synthetic: Profile) -> None:
    assert synthetic.e_per_adu("native", 25) == pytest.approx(math.sqrt(0.5 * 0.25))
    assert synthetic.e_per_adu("native", 75) == pytest.approx(math.sqrt(0.25 * 0.125))
    assert synthetic.read_noise_e("native", 25) == pytest.approx((1.5 + 1.2) / 2)
    # A mode without gain rows keeps its gain 0 values at every gain.
    assert synthetic.e_per_adu("bin2", 100) == 2.0
    assert synthetic.read_noise_e("bin2", 100) == 3.0


def test_the_full_well_limits_saturation_when_it_fills_before_the_adc_clips(
    synthetic: Profile,
) -> None:
    """At gain 0 in the 16-bit mode, the 20,000 e- well reaches 40,000 ADU, below 65,535."""
    saturation = synthetic.saturation("native", 0)
    assert saturation.limited_by == "well"
    assert saturation.native_dn == pytest.approx(20000 / 0.5)
    assert saturation.full_well_e == pytest.approx(20000)
    assert saturation.container_dn == pytest.approx(saturation.native_dn)  # 16 bits: no shift
    assert synthetic.full_well_e("native", 0) == pytest.approx(20000)


def test_the_adc_limits_saturation_at_higher_gain(synthetic: Profile) -> None:
    saturation = synthetic.saturation("native", 50)  # 0.25 e-/ADU: the well would reach 80,000 ADU
    assert saturation.limited_by == "adc"
    assert saturation.native_dn == 65535
    assert saturation.full_well_e == pytest.approx(0.25 * 65535)


def test_the_limit_switches_where_the_well_and_the_adc_meet(synthetic: Profile) -> None:
    """The 20,000 e- well meets the 65,535 ADU full scale at 0.3052 e-/ADU."""
    switch = 20000 / 65535
    below = next(g for g in range(51) if synthetic.e_per_adu("native", g) < switch)
    assert synthetic.saturation("native", below - 1).limited_by == "well"
    assert synthetic.saturation("native", below).limited_by == "adc"


def test_the_container_level_shifts_the_adc_value_by_the_missing_bits(synthetic: Profile) -> None:
    saturation = synthetic.saturation("bin2", 0)  # a 14-bit ADC in a 16-bit container
    assert saturation.limited_by == "adc"
    assert saturation.native_dn == 16383
    assert saturation.container_dn == 16383 * 4


# --- The research notes' larger scopes ------------------------------------------------------------


@pytest.mark.parametrize(
    ("focal_mm", "aperture_mm", "scale_bin1", "scale_bin2", "fov_width", "fov_height"),
    [
        (300.0, 50.0, 1.592, 3.183, 3.663, 2.495),  # GS-300
        (350.0, 58.0, 1.364, 2.729, 3.140, 2.139),  # GS-350
    ],
    ids=["GS-300", "GS-350"],
)
def test_the_larger_scopes_reproduce_the_research_notes(
    focal_mm: float,
    aperture_mm: float,
    scale_bin1: float,
    scale_bin2: float,
    fov_width: float,
    fov_height: float,
) -> None:
    profile = parse_profile(with_optics(focal_mm, aperture_mm, "larger-scope"))
    assert profile.plate_scale_arcsec_per_px("bin1") == pytest.approx(scale_bin1, abs=5e-4)
    assert profile.plate_scale_arcsec_per_px("bin2") == pytest.approx(scale_bin2, abs=5e-4)
    for mode in ("bin1", "bin2"):
        fov = profile.field_of_view(mode)
        assert fov.width_deg == pytest.approx(fov_width, abs=5e-4)
        assert fov.height_deg == pytest.approx(fov_height, abs=5e-4)


@pytest.mark.parametrize(
    (
        "focal_mm",
        "aperture_mm",
        "airy_arcsec",
        "airy_bin1",
        "airy_bin2",
        "ratio_bin1",
        "ratio_bin2",
    ),
    [
        (
            300.0,
            50.0,
            2.55,
            1.60,
            0.80,
            0.59,
            1.19,
        ),  # GS-300 at 0.6 um, and at 0.65 um for the ratios
        (350.0, 58.0, 2.20, 1.61, 0.81, 0.59, 1.18),  # GS-350
    ],
    ids=["GS-300", "GS-350"],
)
def test_the_larger_scopes_reproduce_the_sampling_numbers(
    focal_mm: float,
    aperture_mm: float,
    airy_arcsec: float,
    airy_bin1: float,
    airy_bin2: float,
    ratio_bin1: float,
    ratio_bin2: float,
) -> None:
    """Research notes, larger-scopes table. They round to two decimals, so the tolerance is 0.01."""
    profile = parse_profile(with_optics(focal_mm, aperture_mm, "larger-scope"))
    assert profile.airy_fwhm_arcsec() == pytest.approx(airy_arcsec, abs=0.01)
    assert profile.airy_fwhm_px("bin1") == pytest.approx(airy_bin1, abs=0.01)
    assert profile.airy_fwhm_px("bin2") == pytest.approx(airy_bin2, abs=0.01)
    assert profile.sampling_ratio("bin1", 650.0) == pytest.approx(ratio_bin1, abs=0.01)
    assert profile.sampling_ratio("bin2", 650.0) == pytest.approx(ratio_bin2, abs=0.01)


def test_a_longer_focal_length_changes_the_roi_for_the_same_patch_of_sky(
    reference: Profile,
) -> None:
    """A 4.1 arcmin patch is 154.6 pixels with the GS-300 (300 mm) and 180.3 with the GS-350."""
    gs300 = parse_profile(with_optics(300.0, 50.0, "gs300"))
    gs350 = parse_profile(with_optics(350.0, 58.0, "gs350"))
    assert gs300.roi_size_px("bin1", 4.1) == (152, 154)  # 19 steps of 8, and 77 steps of 2
    assert gs350.roi_size_px("bin1", 4.1) == (184, 180)  # 23 steps of 8, and 90 steps of 2
    assert reference.roi_size_px("bin1", 4.1) == (128, 128)
