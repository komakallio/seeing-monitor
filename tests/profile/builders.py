"""Builders for profile data in tests.

The tests work on plain dictionaries, the shape that a profile file parses to, so a test can
change one value and validate the result.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILE = REPO_ROOT / "profiles" / "asi294mm-gs250.toml"


def reference_data() -> dict[str, Any]:
    """The contents of the reference profile file, as a fresh dictionary."""
    with REFERENCE_FILE.open("rb") as handle:
        return tomllib.load(handle)


def set_path(data: dict[str, Any], path: str, value: object) -> dict[str, Any]:
    """Return a deep copy of `data` with `value` at a dotted path. Digits index a list."""
    result = copy.deepcopy(data)
    node: Any = result
    *parents, last = path.split(".")
    for part in parents:
        node = node[int(part)] if isinstance(node, list) else node[part]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    return result


def delete_path(data: dict[str, Any], path: str) -> dict[str, Any]:
    """Return a deep copy of `data` without the key at a dotted path."""
    result = copy.deepcopy(data)
    node: Any = result
    *parents, last = path.split(".")
    for part in parents:
        node = node[int(part)] if isinstance(node, list) else node[part]
    del node[last]
    return result


def with_optics(focal_length_mm: float, aperture_mm: float, profile_id: str) -> dict[str, Any]:
    """The reference camera behind other optics, as the research notes' larger scopes."""
    data = reference_data()
    data["id"] = profile_id
    data["optics"]["focal_length_mm"] = focal_length_mm
    data["optics"]["aperture_mm"] = aperture_mm
    return data


def synthetic_data() -> dict[str, Any]:
    """A second, complete profile that shares no value with the reference profile.

    The sensor is 7.5 x 5.625 mm. The `native` mode has 2000 x 1500 pixels of 3.75 um and a
    16-bit ADC whose full scale (65,535) lies above the full well, so the well limits
    saturation. The `bin2` mode has 1000 x 750 pixels of 7.5 um and a 14-bit ADC.
    """
    return {
        "id": "synthetic-cam",
        "description": "A synthetic camera for tests",
        "sensor": {"name": "Synthetic CMOS", "cooled": True, "has_temperature_sensor": False},
        "optics": {
            "focal_length_mm": 400.0,
            "aperture_mm": 80.0,
            "wavelength_nm": 550.0,
            "image_circle_diameter_mm": 6.0,
        },
        "limits": {
            "roi_width_multiple": 4,
            "roi_height_multiple": 4,
            "gain_range": [0, 100],
            "exposure_us_range": [10, 10_000_000],
            "offset_range": [0, 255],
        },
        "fast_mode": {"mode": "native"},
        "survey_mode": {"mode": "bin2", "pixel_format": "RAW8"},
        "readout_modes": [
            {
                "name": "native",
                "sdk_bin": 1,
                "width_px": 2000,
                "height_px": 1500,
                "pixel_size_um": 3.75,
                "adc_bits": 16,
                "full_well_gain0_e": 20000.0,
                "read_noise_gain0_e": 1.5,
                "e_per_adu_gain0": 0.5,
                "row_time_us": 10.0,
                "frame_overhead_ms": 2.0,
                "gain_points": [
                    {"gain": 50, "e_per_adu": 0.25, "read_noise_e": 1.2},
                    {"gain": 100, "e_per_adu": 0.125, "read_noise_e": 1.0},
                ],
            },
            {
                "name": "bin2",
                "sdk_bin": 2,
                "width_px": 1000,
                "height_px": 750,
                "pixel_size_um": 7.5,
                "adc_bits": 14,
                "full_well_gain0_e": 80000.0,
                "read_noise_gain0_e": 3.0,
                "e_per_adu_gain0": 2.0,
                "row_time_us": 6.0,
                "frame_overhead_ms": 0.8,
            },
        ],
    }
