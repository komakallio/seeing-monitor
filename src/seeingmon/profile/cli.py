"""The `seeingmon profile` commands.

    seeingmon profile list             the available profiles
    seeingmon profile show [NAME]      one profile with its derived values (add --json for JSON)

`NAME` is a profile name (a file stem in `profiles/`) or the path to a profile file. Without
it, `show` uses the profile that the configuration names.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from seeingmon.cli import CliError, Subparsers, add_command


def register(subparsers: Subparsers) -> None:
    profile = add_command(
        subparsers, "profile", help="List and show hardware profiles.", handler=_require_subcommand
    )
    commands = profile.add_subparsers(dest="profile_command", metavar="<subcommand>", required=True)
    add_command(commands, "list", help="List the available profiles.", handler=_list)
    show = add_command(
        commands, "show", help="Show a profile with its derived values.", handler=_show
    )
    show.add_argument(
        "name",
        nargs="?",
        metavar="NAME",
        help="a profile name or the path to a profile file (default: the configured profile)",
    )
    show.add_argument("--json", action="store_true", help="print the summary as JSON")


def _require_subcommand(args: argparse.Namespace) -> int:
    raise CliError("choose a subcommand: list or show", exit_code=2)


def _list(args: argparse.Namespace) -> int:
    from seeingmon import paths
    from seeingmon.profile import ProfileError, list_profiles, load_profile

    try:
        names = list_profiles()
    except paths.DataDirectoryError as error:
        raise CliError(str(error)) from error
    status = 0
    for name in names:
        try:
            print(f"{name}  {load_profile(name).description}")
        except ProfileError as error:
            first_line = str(error).splitlines()[0]
            print(f"{name}  (invalid: {first_line})")
            status = 1
    return status


def _show(args: argparse.Namespace) -> int:
    from seeingmon import paths
    from seeingmon.config import ConfigError, load_config
    from seeingmon.profile import ProfileError, load_profile
    from seeingmon.profile.summary import profile_summary

    try:
        name = args.name if args.name is not None else load_config().profile_name
        summary = profile_summary(load_profile(name))
    except (ConfigError, ProfileError, paths.DataDirectoryError) as error:
        raise CliError(str(error)) from error
    print(json.dumps(summary, indent=2) if args.json else render_summary(summary))
    return 0


# --- Text output ------------------------------------------------------------------------------


def _duration(us: float) -> str:
    if us >= 1e6:
        return f"{us / 1e6:g} s"
    if us >= 1e3:
        return f"{us / 1e3:g} ms"
    return f"{us:g} us"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _mode_label(summary: dict[str, Any], name: str) -> str:
    roles = [
        label
        for label, key in (("fast mode", "fast_mode"), ("survey mode", "survey_mode"))
        if summary[key]["mode"] == name
    ]
    return f", the {' and '.join(roles)}" if roles else ""


def _selection(selection: dict[str, Any]) -> str:
    fmt = selection["pixel_format"]
    return f"{selection['mode']} ({fmt})" if fmt else f"{selection['mode']}"


def render_summary(summary: dict[str, Any]) -> str:
    """Format a profile summary as plain ASCII text.

    Each value that depends on a readout mode names that mode.
    """
    sensor, optics, limits = summary["sensor"], summary["optics"], summary["limits"]
    lines = [f"Profile {summary['id']}", f"  {summary['description']}", ""]
    lines += [
        "Sensor",
        f"  {sensor['name']}; cooled: {_yes_no(sensor['cooled'])}; "
        f"temperature sensor: {_yes_no(sensor['has_temperature_sensor'])}",
        "Optics",
        f"  Focal length {optics['focal_length_mm']:g} mm, aperture {optics['aperture_mm']:g} mm, "
        f"f/{optics['derived']['f_number']:.2f}",
        f"  Effective wavelength {optics['wavelength_nm']:g} nm, "
        f"Airy FWHM {optics['derived']['airy_fwhm_arcsec']:.2f} arcsec",
    ]
    circle = optics.get("image_circle_diameter_mm")
    lines.append(
        "  Usable image circle: " + ("the full sensor" if circle is None else f"{circle:g} mm")
    )
    exposure_min, exposure_max = limits["exposure_us_range"]
    offset = limits["offset_range"]
    lines += [
        "Limits",
        f"  ROI width in multiples of {limits['roi_width_multiple']} px, "
        f"height in multiples of {limits['roi_height_multiple']} px (binned pixels)",
        f"  Gain {limits['gain_range'][0]} to {limits['gain_range'][1]}",
        f"  Exposure {_duration(exposure_min)} to {_duration(exposure_max)}",
        "  Offset range: " + ("not set" if offset is None else f"{offset[0]} to {offset[1]}"),
        "Fast mode: " + _selection(summary["fast_mode"]),
        "Survey mode: " + _selection(summary["survey_mode"]),
    ]
    for mode in summary["readout_modes"]:
        lines += ["", *_render_mode(summary, mode)]
    photometry = summary.get("photometry")
    if photometry:
        lines += ["", *_render_photometry(photometry)]
    return "\n".join(lines)


def _render_mode(summary: dict[str, Any], mode: dict[str, Any]) -> list[str]:
    name, d = mode["name"], mode["derived"]
    wavelength = summary["optics"]["wavelength_nm"]
    size_mm = f"{d['sensor_width_mm']:.2f} x {d['sensor_height_mm']:.2f} mm"
    fov = (
        f"{d['field_of_view_width_deg']:.3f} x {d['field_of_view_height_deg']:.3f} deg, "
        f"diagonal {d['field_of_view_diagonal_deg']:.3f} deg"
    )
    lines = [
        f"Readout mode {name} (SDK bin {mode['sdk_bin']}){_mode_label(summary, name)}",
        f"  Frame in {name}: {mode['width_px']} x {mode['height_px']} px of "
        f"{mode['pixel_size_um']:g} um; {size_mm}, diagonal {d['sensor_diagonal_mm']:.2f} mm",
        f"  Plate scale in {name}: {d['plate_scale_arcsec_per_px']:.3f} arcsec/px",
        f"  Field of view in {name}: {fov}",
        f"  Sampling in {name} at {wavelength:g} nm: Airy FWHM {d['airy_fwhm_px']:.2f} px, "
        f"pixel / (wavelength x f-number) = {d['sampling_ratio']:.2f}",
        f"  ADC: {mode['adc_bits']} bit, full scale {d['adc_full_scale']}",
        f"  Row time {mode['row_time_us']:g} us ({d['row_time_ns']} ns), "
        f"frame overhead {mode['frame_overhead_ms']:g} ms",
    ]
    full = d["full_frame"]
    lines.append(
        f"  Full frame: {full['max_frame_rate_hz']:.2f} fps at the shortest exposure, "
        f"{full['frame_bytes']['RAW16'] / 1e6:.1f} MB as RAW16"
    )
    snapshot = d["snapshot"]
    assumed = "" if snapshot["modeled"] else " (assumed: nobody measured this mode)"
    lines.append(
        f"  Single exposure in {name}: {snapshot['overhead_s']:g} s plus "
        f"{snapshot['row_time_us']:g} us per row beyond the exposure, "
        f"{snapshot['full_frame_readout_s']:.2f} s for the full frame{assumed}"
    )
    high_speed = d["high_speed"]
    if high_speed:
        lines.append(
            f"  High-speed mode: {high_speed['adc_bits']} bit (full scale "
            f"{high_speed['adc_full_scale']}), row time {high_speed['row_time_us']:g} us, "
            f"frame overhead {high_speed['frame_overhead_ms']:g} ms, "
            f"full frame {high_speed['full_frame_max_frame_rate_hz']:.2f} fps"
        )
    lines += [
        f"  Gain table for {name}:",
        "    gain  e-/ADU  noise (e-)  full well (e-)  saturation (ADC / RAW16)  limit",
    ]
    for row in d["gain_curve"]:
        notes = ["step"] if row["step"] else []
        if row["above_table"]:
            notes.append("held from the last row")
        saturation = f"{row['saturation_native_dn']:.0f} / {row['saturation_container_dn']:.0f}"
        lines.append(
            f"    {row['gain']:>4}  {row['e_per_adu']:>6.3f}  {row['read_noise_e']:>10.2f}  "
            f"{row['full_well_e']:>14.0f}  {saturation:<24}  {row['saturation_limited_by']:<5}"
            + (f"  ({', '.join(notes)})" if notes else "")
        )
    return [line.rstrip() for line in lines]


def _render_photometry(photometry: dict[str, Any]) -> list[str]:
    lines = ["Photometry"]
    rate = photometry.get("mag0_electron_rate_e_per_s")
    if rate is not None:
        uncertainty = photometry["mag0_electron_rate_rel_uncertainty"]
        lines.append(
            f"  Magnitude-0 star: {rate:.3g} e-/s, an estimate good to +-{uncertainty:.0%}"
        )
    points = photometry.get("dark_current") or []
    if points:
        lines.append("  Dark current prior (e-/s/px):")
        lines += [f"    {p['temperature_c']:>6g} C  {p['e_per_s_per_px']:g}" for p in points]
    return lines
