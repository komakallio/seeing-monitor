"""The profile summary that the API serves: plain JSON with every derived value."""

from __future__ import annotations

import json
from typing import Any

import pytest

from seeingmon.profile import Profile, profile_summary

MODE_DERIVED_KEYS = {
    "sensor_width_mm",
    "sensor_height_mm",
    "sensor_diagonal_mm",
    "plate_scale_arcsec_per_px",
    "field_of_view_width_deg",
    "field_of_view_height_deg",
    "field_of_view_diagonal_deg",
    "usable_radius_px",
    "airy_fwhm_px",
    "sampling_ratio",
    "adc_full_scale",
    "row_time_ns",
    "full_frame",
    "gain_curve",
    "high_speed",
}


def mode_of(summary: dict[str, Any], name: str) -> dict[str, Any]:
    return next(mode for mode in summary["readout_modes"] if mode["name"] == name)


def test_the_summary_is_plain_json(reference: Profile) -> None:
    summary = profile_summary(reference)
    assert json.loads(json.dumps(summary)) == summary


def test_the_summary_keeps_every_field_of_the_profile(reference: Profile) -> None:
    summary = profile_summary(reference)
    dump = reference.model_dump(mode="json")
    assert summary["id"] == "asi294mm-gs250"
    assert summary["sensor"] == dump["sensor"]
    assert summary["limits"] == dump["limits"]
    assert summary["fast_mode"] == {"mode": "bin1", "pixel_format": "RAW16"}
    assert summary["survey_mode"] == {"mode": "bin2", "pixel_format": "RAW16"}
    assert summary["photometry"] == dump["photometry"]
    for name in ("bin1", "bin2"):
        raw = {k: v for k, v in mode_of(summary, name).items() if k != "derived"}
        assert raw == next(m for m in dump["readout_modes"] if m["name"] == name)


def test_every_mode_has_the_same_derived_keys(reference: Profile, synthetic: Profile) -> None:
    for profile in (reference, synthetic):
        for mode in profile_summary(profile)["readout_modes"]:
            assert set(mode["derived"]) >= MODE_DERIVED_KEYS


def test_the_optics_gain_the_f_number_and_the_airy_size(reference: Profile) -> None:
    derived = profile_summary(reference)["optics"]["derived"]
    assert derived["f_number"] == pytest.approx(5.0)
    assert derived["airy_fwhm_arcsec"] == pytest.approx(2.55, abs=0.01)


@pytest.mark.parametrize(("name", "scale"), [("bin1", 1.910), ("bin2", 3.820)])
def test_the_derived_values_match_the_research_notes_for_each_mode(
    reference: Profile, name: str, scale: float
) -> None:
    derived = mode_of(profile_summary(reference), name)["derived"]
    assert derived["plate_scale_arcsec_per_px"] == pytest.approx(scale, abs=5e-4)
    assert derived["field_of_view_width_deg"] == pytest.approx(4.395, abs=5e-4)
    assert derived["field_of_view_height_deg"] == pytest.approx(2.994, abs=5e-4)
    assert derived["field_of_view_diagonal_deg"] == pytest.approx(5.316, abs=5e-4)
    assert derived["sensor_width_mm"] == pytest.approx(19.187, abs=1e-3)
    assert derived["usable_radius_px"] is None


def test_the_derived_values_equal_the_profile_methods(reference: Profile) -> None:
    summary = profile_summary(reference)
    for mode in reference.readout_modes:
        derived = mode_of(summary, mode.name)["derived"]
        assert derived["plate_scale_arcsec_per_px"] == reference.plate_scale_arcsec_per_px(mode)
        assert derived["airy_fwhm_px"] == reference.airy_fwhm_px(mode)
        assert derived["sampling_ratio"] == reference.sampling_ratio(mode)
        assert derived["row_time_ns"] == reference.row_time_ns(mode)


def test_the_adc_full_scales_and_the_full_frame_sizes(reference: Profile) -> None:
    summary = profile_summary(reference)
    bin1, bin2 = mode_of(summary, "bin1")["derived"], mode_of(summary, "bin2")["derived"]
    assert (bin1["adc_full_scale"], bin2["adc_full_scale"]) == (4095, 16383)
    assert bin1["full_frame"]["frame_bytes"] == {"RAW8": 8288 * 5644, "RAW16": 8288 * 5644 * 2}
    # The line model extrapolates the fit of the small square ROIs. ZWO publishes 16.3 fps for the
    # bin2 full frame, which the USB link limits (4144 x 2822 x 2 bytes at 16.3 fps is 380 MB/s),
    # so plan the large frames with the published rate.
    assert bin2["full_frame"]["max_frame_rate_hz"] == pytest.approx(18.7, abs=0.05)
    assert bin1["full_frame"]["max_frame_rate_hz"] == pytest.approx(4.6, abs=0.05)


def test_the_gain_curve_lists_gain_0_the_rows_and_the_end_of_the_range(reference: Profile) -> None:
    summary = profile_summary(reference)
    bin1 = mode_of(summary, "bin1")["derived"]["gain_curve"]
    bin2 = mode_of(summary, "bin2")["derived"]["gain_curve"]
    assert [row["gain"] for row in bin1] == [0, 108, 270, 570]
    assert [row["gain"] for row in bin2] == [0, 119, 120, 300, 570]
    assert [row["step"] for row in bin2] == [False, False, True, False, False]
    assert [row["above_table"] for row in bin2] == [False, False, False, False, True]
    assert not any(row["step"] for row in bin1)


def test_the_gain_curve_carries_the_noise_and_the_saturation(reference: Profile) -> None:
    rows = {
        row["gain"]: row
        for row in mode_of(profile_summary(reference), "bin1")["derived"]["gain_curve"]
    }
    assert rows[0]["e_per_adu"] == 3.5
    assert rows[0]["read_noise_e"] == 2.65
    assert rows[108]["saturation_native_dn"] == 4095
    assert rows[108]["saturation_container_dn"] == 65520
    assert rows[108]["saturation_limited_by"] == "adc"
    assert rows[108]["full_well_e"] == pytest.approx(4095)


def test_the_high_speed_variant_appears_only_for_a_mode_that_has_one(
    reference: Profile, synthetic: Profile
) -> None:
    bin1 = mode_of(profile_summary(reference), "bin1")["derived"]["high_speed"]
    assert bin1["adc_bits"] == 10
    assert bin1["adc_full_scale"] == 1023
    assert bin1["row_time_us"] == 30.0
    assert bin1["row_time_ns"] == 30_000
    assert bin1["full_frame_max_frame_rate_hz"] == pytest.approx(5.7, abs=0.05)
    for mode in profile_summary(synthetic)["readout_modes"]:
        assert mode["derived"]["high_speed"] is None


def test_the_summary_of_another_profile_follows_that_profile(synthetic: Profile) -> None:
    summary = profile_summary(synthetic)
    native = mode_of(summary, "native")["derived"]
    assert summary["id"] == "synthetic-cam"
    assert summary["photometry"] is None
    assert summary["optics"]["derived"]["f_number"] == pytest.approx(5.0)
    assert native["usable_radius_px"] == pytest.approx(800)
    assert native["adc_full_scale"] == 65535
    assert native["gain_curve"][0]["saturation_limited_by"] == "well"
    assert [row["gain"] for row in native["gain_curve"]] == [0, 50, 100]
    assert summary["survey_mode"]["pixel_format"] == "RAW8"
    assert summary["fast_mode"]["pixel_format"] is None


def test_each_call_returns_a_fresh_summary(reference: Profile) -> None:
    first = profile_summary(reference)
    first["optics"]["focal_length_mm"] = 1.0
    first["readout_modes"].clear()
    second = profile_summary(reference)
    assert second["optics"]["focal_length_mm"] == 250.0
    assert len(second["readout_modes"]) == 2
