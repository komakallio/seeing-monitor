"""Invalid profiles fail with messages that say what is wrong and where."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from seeingmon.frames import PixelFormat
from seeingmon.profile import ModeSelection, Profile, ProfileError, parse_profile
from tests.profile.builders import delete_path, set_path


def failure(data: dict[str, Any]) -> str:
    """The message of the `ProfileError` that validating `data` raises."""
    with pytest.raises(ProfileError) as error:
        parse_profile(data, source="test.toml")
    return str(error.value)


# --- A valid profile ------------------------------------------------------------------------


def test_the_reference_data_is_valid(data: dict[str, Any]) -> None:
    assert parse_profile(data).id == "asi294mm-gs250"


def test_the_error_message_names_the_source(data: dict[str, Any]) -> None:
    message = failure(set_path(data, "optics.focal_length_mm", -1))
    assert message.startswith("invalid profile test.toml:\n")


def test_profiles_are_frozen_and_hashable(reference: Profile) -> None:
    with pytest.raises(ValidationError, match="frozen"):
        reference.id = "other"  # type: ignore[misc]
    assert hash(reference) == hash(reference.model_copy())


# --- Mode names in fast_mode and survey_mode ------------------------------------------------------


@pytest.mark.parametrize("selection", ["fast_mode", "survey_mode"])
def test_an_unknown_mode_name_in_a_mode_selection_is_rejected(
    data: dict[str, Any], selection: str
) -> None:
    message = failure(set_path(data, f"{selection}.mode", "bin3"))
    assert f"{selection}: unknown readout mode 'bin3'" in message
    assert "this profile defines bin1, bin2" in message


def test_a_mode_name_is_case_sensitive(data: dict[str, Any]) -> None:
    assert "unknown readout mode 'BIN1'" in failure(set_path(data, "fast_mode.mode", "BIN1"))


@pytest.mark.parametrize(
    "name", ["", "bin 1", "x" * 17, "biné"], ids=["empty", "space", "long", "non-ascii"]
)
def test_a_mode_name_must_be_1_to_16_printable_ascii_characters(
    data: dict[str, Any], name: str
) -> None:
    message = failure(set_path(data, "readout_modes.0.name", name))
    assert "readout_modes[" in message
    assert "a mode name is 1 to 16 printable ASCII characters" in message


def test_a_mode_name_of_16_characters_is_valid(data: dict[str, Any]) -> None:
    data = set_path(data, "readout_modes.0.name", "bin1-0123456789a")
    data = set_path(data, "fast_mode.mode", "bin1-0123456789a")
    assert parse_profile(data).fast_readout.name == "bin1-0123456789a"


def test_duplicate_mode_names_are_rejected(data: dict[str, Any]) -> None:
    message = failure(set_path(data, "readout_modes.1.name", "bin1"))
    assert "readout mode names must be unique; repeated: bin1" in message


def test_a_profile_needs_at_least_one_readout_mode(data: dict[str, Any]) -> None:
    assert "readout_modes" in failure(set_path(data, "readout_modes", []))


# --- Gain points ---------------------------------------------------------------------------


def test_unsorted_gain_points_are_rejected(data: dict[str, Any]) -> None:
    points = [
        {"gain": 270, "e_per_adu": 0.17, "read_noise_e": 1.38},
        {"gain": 108, "e_per_adu": 1.0, "read_noise_e": 1.8},
    ]
    message = failure(set_path(data, "readout_modes.0.gain_points", points))
    assert "readout_modes[bin1]: gain_points must be strictly increasing" in message
    assert "gain_points[1] has gain 108 after gain 270" in message


def test_a_repeated_gain_is_rejected(data: dict[str, Any]) -> None:
    points = [
        {"gain": 108, "e_per_adu": 1.0, "read_noise_e": 1.8},
        {"gain": 108, "e_per_adu": 0.9, "read_noise_e": 1.7},
    ]
    message = failure(set_path(data, "readout_modes.0.gain_points", points))
    assert "gain_points[1] has gain 108 after gain 108" in message


def test_a_gain_point_at_gain_0_is_rejected_because_the_mode_states_gain_0(
    data: dict[str, Any],
) -> None:
    message = failure(set_path(data, "readout_modes.0.gain_points.0.gain", 0))
    assert "readout_modes[bin1].gain_points[0].gain: Input should be greater than 0" in message


def test_a_step_must_follow_the_previous_gain_directly(data: dict[str, Any]) -> None:
    message = failure(set_path(data, "readout_modes.1.gain_points.1.gain", 130))
    assert "gain_points[1] marks a step at gain 130" in message
    assert "the previous point must be at gain 129, not 119" in message


def test_a_gain_point_above_the_gain_range_is_rejected(data: dict[str, Any]) -> None:
    message = failure(set_path(data, "limits.gain_range", [0, 200]))
    assert "readout mode 'bin1' has a gain point at 270, above the gain range [0, 200]" in message


# --- Non-positive and non-finite values -------------------------------------------------------

POSITIVE_FIELDS = [
    "optics.focal_length_mm",
    "optics.aperture_mm",
    "optics.wavelength_nm",
    "optics.image_circle_diameter_mm",
    "readout_modes.0.sdk_bin",
    "readout_modes.0.width_px",
    "readout_modes.0.height_px",
    "readout_modes.0.pixel_size_um",
    "readout_modes.0.full_well_gain0_e",
    "readout_modes.0.read_noise_gain0_e",
    "readout_modes.0.e_per_adu_gain0",
    "readout_modes.0.row_time_us",
    "readout_modes.0.row_time_us_high_speed",
    "readout_modes.1.row_time_us",
    "readout_modes.1.snapshot_row_time_us",
    "readout_modes.0.gain_points.0.e_per_adu",
    "readout_modes.0.gain_points.1.read_noise_e",
    "limits.roi_width_multiple",
    "limits.roi_height_multiple",
    "photometry.mag0_electron_rate_e_per_s",
    "photometry.dark_current.0.e_per_s_per_px",
]


@pytest.mark.parametrize("bad", [0, -1])
@pytest.mark.parametrize("path", POSITIVE_FIELDS)
def test_non_positive_values_are_rejected(data: dict[str, Any], path: str, bad: float) -> None:
    message = failure(set_path(data, path, bad))
    assert "Input should be greater than 0" in message
    assert path.rsplit(".", maxsplit=1)[-1] in message


def test_a_non_positive_value_names_its_mode_and_field(data: dict[str, Any]) -> None:
    message = failure(set_path(data, "readout_modes.0.pixel_size_um", 0))
    assert "readout_modes[bin1].pixel_size_um: Input should be greater than 0" in message


def test_a_negative_frame_overhead_is_rejected_and_zero_is_allowed(data: dict[str, Any]) -> None:
    message = failure(set_path(data, "readout_modes.1.frame_overhead_ms", -0.1))
    assert (
        "readout_modes[bin2].frame_overhead_ms: Input should be greater than or equal to 0"
        in message
    )
    assert parse_profile(set_path(data, "readout_modes.1.frame_overhead_ms", 0)).id


@pytest.mark.parametrize("bad", [float("inf"), float("nan")], ids=["inf", "nan"])
def test_non_finite_values_are_rejected(data: dict[str, Any], bad: float) -> None:
    message = failure(set_path(data, "readout_modes.0.pixel_size_um", bad))
    assert "finite" in message


# --- The snapshot model ------------------------------------------------------------------------


def test_a_negative_snapshot_overhead_is_rejected_and_zero_is_allowed(data: dict[str, Any]) -> None:
    message = failure(set_path(data, "readout_modes.1.snapshot_overhead_s", -0.1))
    assert (
        "readout_modes[bin2].snapshot_overhead_s: Input should be greater than or equal to 0"
        in message
    )
    assert parse_profile(set_path(data, "readout_modes.1.snapshot_overhead_s", 0)).id


@pytest.mark.parametrize("bad", [float("inf"), float("nan")], ids=["inf", "nan"])
@pytest.mark.parametrize("path", ["snapshot_overhead_s", "snapshot_row_time_us"])
def test_a_non_finite_snapshot_value_is_rejected(
    data: dict[str, Any], path: str, bad: float
) -> None:
    assert "finite" in failure(set_path(data, f"readout_modes.1.{path}", bad))


@pytest.mark.parametrize("missing", ["snapshot_overhead_s", "snapshot_row_time_us"])
def test_the_two_snapshot_values_go_together(data: dict[str, Any], missing: str) -> None:
    message = failure(delete_path(data, f"readout_modes.1.{missing}"))
    assert "readout_modes[bin2]: " in message
    assert "snapshot_overhead_s and snapshot_row_time_us go together" in message


def test_a_mode_may_state_both_snapshot_values_or_neither(data: dict[str, Any]) -> None:
    bare = delete_path(
        delete_path(data, "readout_modes.1.snapshot_overhead_s"),
        "readout_modes.1.snapshot_row_time_us",
    )
    assert parse_profile(bare).mode("bin2").has_snapshot_model is False
    both = set_path(
        set_path(data, "readout_modes.0.snapshot_overhead_s", 1.0),
        "readout_modes.0.snapshot_row_time_us",
        90.0,
    )
    assert parse_profile(both).mode("bin1").has_snapshot_model is True


def test_the_snapshot_model_is_optional_in_a_profile(synthetic: Profile) -> None:
    assert not any(mode.has_snapshot_model for mode in synthetic.readout_modes)


@pytest.mark.parametrize(
    "path", ["readout_modes.0.adc_bits", "readout_modes.0.adc_bits_high_speed"]
)
@pytest.mark.parametrize("bad", [0, 17])
def test_adc_bits_must_fit_the_16_bit_container(data: dict[str, Any], path: str, bad: int) -> None:
    assert "adc_bits" in failure(set_path(data, path, bad))


def test_each_range_must_run_from_minimum_to_maximum(data: dict[str, Any]) -> None:
    assert "gain_range must be (minimum, maximum), got [10, 5]" in failure(
        set_path(data, "limits.gain_range", [10, 5])
    )
    assert "exposure_us_range must be (minimum, maximum)" in failure(
        set_path(data, "limits.exposure_us_range", [100, 10])
    )
    assert "offset_range must be (minimum, maximum)" in failure(
        set_path(data, "limits.offset_range", [50, 0])
    )


def test_a_negative_gain_in_the_range_is_rejected(data: dict[str, Any]) -> None:
    assert "gain_range" in failure(set_path(data, "limits.gain_range", [-1, 100]))


# --- Unknown, missing, and inconsistent keys -------------------------------------------------


def test_an_unknown_key_is_an_error(data: dict[str, Any]) -> None:
    message = failure(set_path(data, "optics.focal_lenth_mm", 250))
    assert "optics.focal_lenth_mm: Extra inputs are not permitted" in message


def test_an_unknown_top_level_key_is_an_error(data: dict[str, Any]) -> None:
    assert "colour: Extra inputs are not permitted" in failure({**data, "colour": "red"})


def test_a_missing_key_is_an_error(data: dict[str, Any]) -> None:
    message = failure(delete_path(data, "optics.focal_length_mm"))
    assert "optics.focal_length_mm: Field required" in message


def test_every_problem_is_reported_at_once(data: dict[str, Any]) -> None:
    broken = set_path(set_path(data, "optics.aperture_mm", 0), "readout_modes.1.row_time_us", -1)
    message = failure(broken)
    assert "optics.aperture_mm" in message
    assert "readout_modes[bin2].row_time_us" in message


def test_modes_that_cover_different_sensor_areas_are_rejected(data: dict[str, Any]) -> None:
    message = failure(set_path(data, "readout_modes.1.pixel_size_um", 9.26))
    assert "readout modes 'bin1' and 'bin2' cover different sensor areas" in message


def test_modes_that_cover_the_same_sensor_within_5_percent_are_accepted(
    data: dict[str, Any],
) -> None:
    assert parse_profile(set_path(data, "readout_modes.1.pixel_size_um", 4.63 * 1.04)).id


def test_a_frame_smaller_than_one_roi_step_is_rejected(data: dict[str, Any]) -> None:
    message = failure(set_path(data, "limits.roi_width_multiple", 9000))
    assert "smaller than one ROI step" in message


@pytest.mark.parametrize("bad", ["Has Space", "-leading-dash", "a/b", "", "x" * 65])
def test_a_profile_id_is_a_file_stem(data: dict[str, Any], bad: str) -> None:
    assert "\n  id: " in failure(set_path(data, "id", bad))


def test_the_description_is_required_and_not_empty(data: dict[str, Any]) -> None:
    assert "description" in failure(set_path(data, "description", ""))


@pytest.mark.parametrize("path", ["sensor.cooled", "sensor.has_temperature_sensor"])
def test_sensor_flags_must_be_booleans(data: dict[str, Any], path: str) -> None:
    assert "boolean" in failure(set_path(data, path, "no"))


# --- Pixel format -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("RAW16", PixelFormat.RAW16),
        ("raw8", PixelFormat.RAW8),
        (16, PixelFormat.RAW16),
        (None, None),
    ],
)
def test_a_pixel_format_accepts_a_name_or_a_number(
    value: object, expected: PixelFormat | None
) -> None:
    assert (
        ModeSelection.model_validate({"mode": "bin1", "pixel_format": value}).pixel_format
        is expected
    )


def test_the_pixel_format_is_optional() -> None:
    assert ModeSelection.model_validate({"mode": "bin1"}).pixel_format is None


def test_an_unknown_pixel_format_lists_the_choices(data: dict[str, Any]) -> None:
    message = failure(set_path(data, "fast_mode.pixel_format", "RAW12"))
    assert "fast_mode.pixel_format: pixel_format must be one of RAW8, RAW16, got 'RAW12'" in message


def test_the_pixel_format_dumps_as_its_name(reference: Profile) -> None:
    assert reference.model_dump(mode="json")["fast_mode"] == {
        "mode": "bin1",
        "pixel_format": "RAW16",
    }


# --- Photometry --------------------------------------------------------------------------------


def test_the_photometry_table_is_optional(data: dict[str, Any]) -> None:
    del data["photometry"]
    assert parse_profile(data).photometry is None


def test_a_rate_without_an_uncertainty_is_rejected(data: dict[str, Any]) -> None:
    message = failure(delete_path(data, "photometry.mag0_electron_rate_rel_uncertainty"))
    assert "go together" in message
    assert "plus or minus 30%" in message


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_the_uncertainty_is_a_fraction_between_0_and_1(data: dict[str, Any], bad: float) -> None:
    assert "mag0_electron_rate_rel_uncertainty" in failure(
        set_path(data, "photometry.mag0_electron_rate_rel_uncertainty", bad)
    )


def test_a_dark_current_table_needs_two_points_or_none(data: dict[str, Any]) -> None:
    one = [{"temperature_c": 0.0, "e_per_s_per_px": 0.01}]
    assert "at least two points" in failure(set_path(data, "photometry.dark_current", one))
    assert parse_profile(set_path(data, "photometry.dark_current", [])).photometry is not None


def test_a_dark_current_table_must_increase_in_temperature(data: dict[str, Any]) -> None:
    points = [
        {"temperature_c": 10.0, "e_per_s_per_px": 0.065},
        {"temperature_c": 0.0, "e_per_s_per_px": 0.019},
    ]
    message = failure(set_path(data, "photometry.dark_current", points))
    assert "strictly increasing temperature order" in message


def test_a_dark_current_table_that_cannot_extrapolate_is_reported(data: dict[str, Any]) -> None:
    """Two points one degree apart that differ by a factor of 1e8 overflow far from the table."""
    steep = [
        {"temperature_c": 0.0, "e_per_s_per_px": 1e-4},
        {"temperature_c": 1.0, "e_per_s_per_px": 1e4},
    ]
    profile = parse_profile(set_path(data, "photometry.dark_current", steep))
    with pytest.raises(ProfileError, match=r"does not extrapolate to 100\.0 C"):
        profile.dark_current_e_per_s_per_px(100.0)
    assert profile.dark_current_e_per_s_per_px(0.5) == pytest.approx(1.0)  # inside: fine


# --- Lookups ---------------------------------------------------------------------------------


def test_an_unknown_mode_lookup_lists_the_defined_modes(reference: Profile) -> None:
    with pytest.raises(
        ProfileError, match="unknown readout mode 'bin3'; this profile defines bin1, bin2"
    ):
        reference.mode("bin3")


def test_a_mode_without_a_high_speed_variant_refuses_one(synthetic: Profile) -> None:
    with pytest.raises(ProfileError, match="readout mode 'bin2' has no high-speed variant"):
        synthetic.mode("bin2", high_speed=True)


def test_a_derived_value_accepts_a_mode_name_or_a_mode(reference: Profile) -> None:
    mode = reference.mode("bin2")
    assert reference.plate_scale_arcsec_per_px("bin2") == reference.plate_scale_arcsec_per_px(mode)
