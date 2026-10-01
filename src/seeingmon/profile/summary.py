"""A JSON-able summary of a profile with every derived value.

`profile_summary` returns plain dictionaries, lists, strings, numbers, booleans, and `None`, so
`json.dumps` accepts the result. The `/api/v1/profile` endpoint serves it, and
`seeingmon profile show --json` prints it.
"""

from __future__ import annotations

from typing import Any

from seeingmon.frames import PixelFormat
from seeingmon.profile import derived
from seeingmon.profile.models import Profile, ReadoutMode


def profile_summary(profile: Profile) -> dict[str, Any]:
    """The profile as plain JSON types, with its derived values added.

    The result holds every field of the profile. `optics` gains a `derived` table with the
    f-number and the Airy FWHM at the effective wavelength. Each readout mode gains a
    `derived` table with its sensor size, plate scale, field of view, sampling, ADC full scale,
    row time, full-frame timing, a gain curve, and the high-speed variant when the mode has one.
    Every value that depends on a readout mode sits in that mode's table, so a plate scale
    always names its mode.
    """
    summary: dict[str, Any] = profile.model_dump(mode="json")
    summary["optics"]["derived"] = {
        "f_number": profile.optics.f_number,
        "airy_fwhm_arcsec": derived.airy_fwhm_arcsec(profile.optics),
    }
    for item, mode in zip(summary["readout_modes"], profile.readout_modes, strict=True):
        item["derived"] = _mode_summary(profile, mode)
    return summary


def _mode_summary(profile: Profile, mode: ReadoutMode) -> dict[str, Any]:
    optics = profile.optics
    width_mm, height_mm = derived.sensor_size_mm(mode)
    fov = derived.field_of_view(mode, optics)
    full_period_s = derived.frame_period_s(
        mode, mode.height_px, profile.limits.exposure_us_range[0]
    )
    return {
        "sensor_width_mm": width_mm,
        "sensor_height_mm": height_mm,
        "sensor_diagonal_mm": derived.sensor_diagonal_mm(mode),
        "plate_scale_arcsec_per_px": derived.plate_scale_arcsec_per_px(mode, optics),
        "field_of_view_width_deg": fov.width_deg,
        "field_of_view_height_deg": fov.height_deg,
        "field_of_view_diagonal_deg": fov.diagonal_deg,
        "usable_radius_px": derived.usable_radius_px(mode, optics),
        "airy_fwhm_px": derived.airy_fwhm_px(mode, optics),
        "sampling_ratio": derived.sampling_ratio(mode, optics),
        "adc_full_scale": derived.adc_full_scale(mode),
        "row_time_ns": derived.row_time_ns(mode),
        "full_frame": {
            "frame_bytes": {
                fmt.name: derived.frame_bytes(mode.width_px, mode.height_px, fmt)
                for fmt in PixelFormat
            },
            "frame_period_s": full_period_s,
            "max_frame_rate_hz": 1.0 / full_period_s,
        },
        "gain_curve": _gain_curve(profile, mode),
        "high_speed": _high_speed(profile, mode),
    }


def _gain_curve(profile: Profile, mode: ReadoutMode) -> list[dict[str, Any]]:
    """The gain-dependent values at gain 0, at every table row, and at the ends of the range.

    `step` marks a row where the values jump. `above_table` marks a gain above the last row,
    where the values stay at the last row.
    """
    step_gains = {point.gain for point in mode.gain_points if point.step}
    last_row = mode.gain_points[-1].gain if mode.gain_points else 0
    low, high = profile.limits.gain_range
    gains = sorted({0, low, high, *(point.gain for point in mode.gain_points)})
    curve = []
    for gain in gains:
        saturation = derived.saturation(mode, gain)
        curve.append(
            {
                "gain": gain,
                "e_per_adu": derived.e_per_adu(mode, gain),
                "read_noise_e": derived.read_noise_e(mode, gain),
                "full_well_e": saturation.full_well_e,
                "saturation_native_dn": saturation.native_dn,
                "saturation_container_dn": saturation.container_dn,
                "saturation_limited_by": saturation.limited_by,
                "step": gain in step_gains,
                "above_table": gain > last_row,
            }
        )
    return curve


def _high_speed(profile: Profile, mode: ReadoutMode) -> dict[str, Any] | None:
    """The values that change in high-speed mode, or `None` when the mode has no such variant."""
    if not mode.has_high_speed:
        return None
    fast = mode.high_speed_variant()
    period_s = derived.frame_period_s(fast, fast.height_px, profile.limits.exposure_us_range[0])
    return {
        "adc_bits": fast.adc_bits,
        "adc_full_scale": derived.adc_full_scale(fast),
        "e_per_adu_gain0": fast.e_per_adu_gain0,
        "row_time_us": fast.row_time_us,
        "row_time_ns": derived.row_time_ns(fast),
        "frame_overhead_ms": fast.frame_overhead_ms,
        "full_frame_max_frame_rate_hz": 1.0 / period_s,
    }
