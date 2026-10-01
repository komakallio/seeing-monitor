"""The options of the simulated camera, and how to read them from a configuration table.

`SimOptions` gathers every choice that is not a hardware number: the seed, the atmosphere, the
stars and the pointing, the sky and the clouds, the sensor temperature, and the faults. All of
them have defaults, so `SimOptions()` gives a working camera.

`SimOptions.from_mapping` reads the same choices from a plain dictionary, such as a TOML table
in the configuration. It rejects unknown keys, so a typo does not pass silently.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from seeingmon.clock import DEFAULT_START_UTC_NS, NS_PER_S
from seeingmon.drivers.base import RecoveryLevel
from seeingmon.drivers.sim.detector import HotPixelConfig
from seeingmon.drivers.sim.faults import GeometryChange, SimFaults
from seeingmon.drivers.sim.optics import PsfConfig
from seeingmon.drivers.sim.sky import (
    SYNTHETIC_SITE,
    CloudEvent,
    Clouds,
    ScintillationConfig,
    Site,
)
from seeingmon.drivers.sim.stars import Pointing, StarField
from seeingmon.drivers.sim.turbulence import Layer, TurbulenceConfig


@dataclass(frozen=True, slots=True)
class SimOptions:
    """Everything that shapes a simulated camera besides its hardware.

    `turbulence=None` builds `TurbulenceConfig(seed=seed, zenith_angle_deg=...)` with the zenith
    angle of the celestial pole at the site, so that the line of sight matches the camera.
    `stars=None` builds `make_polar_field(seed)`, and `pointing=None` centres Polaris at the time
    when you create the driver. The turbulence has its own clock: its time is the seconds since
    `epoch_utc_ns`, so a given seed gives the same atmosphere at the same time, whatever the
    clock's start. The sensor reads `ambient_c + sensor_rise_c` degrees, because an uncooled
    camera runs warm. `max_lag_frames` is how many frames the camera buffers before a slow reader
    loses frames.
    `keep_truth_frames` bounds the per-frame truth that the driver keeps (`None` keeps all).
    """

    seed: int = 1
    psf: PsfConfig = field(default_factory=PsfConfig)
    turbulence: TurbulenceConfig | None = None
    stars: StarField | None = None
    pointing: Pointing | None = None
    site: Site = SYNTHETIC_SITE
    epoch_utc_ns: int = DEFAULT_START_UTC_NS
    sky_mag_arcsec2: float = 20.5
    twilight: bool = True
    clouds: Clouds = field(default_factory=Clouds)
    scintillation: ScintillationConfig = field(default_factory=ScintillationConfig)
    ambient_c: float = 15.0
    sensor_rise_c: float = 4.0
    ambient_drift_c_per_hour: float = 0.0
    hot_pixels: HotPixelConfig = field(default_factory=HotPixelConfig)
    faults: SimFaults = field(default_factory=SimFaults)
    time_error_ns: int = 1_000
    max_lag_frames: int = 3
    keep_truth_frames: int | None = 200_000

    def __post_init__(self) -> None:
        if self.max_lag_frames < 0 or self.time_error_ns < 0:
            raise ValueError("max_lag_frames and time_error_ns must not be negative")

    @classmethod
    def from_mapping(cls, table: Mapping[str, Any] | None) -> SimOptions:
        """Read options from a dictionary.

        The keys: `seed`, `psf_mode` (`wave` or `gaussian`), `bandwidth_fraction`,
        `sky_mag_arcsec2`, `twilight`, `ambient_c`, `sensor_rise_c`, `ambient_drift_c_per_hour`,
        `hot_pixels_per_mpix`, `time_error_ns`, `max_lag_frames`, and `keep_truth_frames`. Four
        nested tables:
        `[turbulence]` takes `r0_m`, `outer_scale_m`, `zenith_angle_deg`, `screen_points`,
        `boiling`, `wind_variability`, and `layers` (a list of tables with `cn2_fraction`,
        `wind_speed_m_s`, and `wind_direction_deg`). `[scintillation]` takes `sigma0` and
        `enabled`. `[faults]` takes the fields of `SimFaults`, with `scripted_drops` as a table
        of frame number to count.
        `[[clouds]]` entries take `start_s` (seconds after `epoch_s`, default 0), `duration_s`,
        `transmission`, and `ramp_s`.
        """
        data = dict(table or {})
        options = cls()
        simple = {
            "seed",
            "sky_mag_arcsec2",
            "twilight",
            "ambient_c",
            "sensor_rise_c",
            "ambient_drift_c_per_hour",
            "time_error_ns",
            "max_lag_frames",
            "keep_truth_frames",
        }
        kwargs: dict[str, Any] = {key: data.pop(key) for key in list(data) if key in simple}
        psf_kwargs: dict[str, Any] = {}
        if "psf_mode" in data:
            psf_kwargs["mode"] = data.pop("psf_mode")
        if "bandwidth_fraction" in data:
            psf_kwargs["bandwidth_fraction"] = data.pop("bandwidth_fraction")
        if psf_kwargs:
            kwargs["psf"] = replace(options.psf, **psf_kwargs)
        if "hot_pixels_per_mpix" in data:
            kwargs["hot_pixels"] = HotPixelConfig(
                density_per_mpix=float(data.pop("hot_pixels_per_mpix"))
            )
        if "turbulence" in data:
            kwargs["turbulence"] = _turbulence(
                data.pop("turbulence"), kwargs.get("seed", options.seed)
            )
        if "scintillation" in data:
            kwargs["scintillation"] = _scintillation(data.pop("scintillation"))
        if "faults" in data:
            kwargs["faults"] = _faults(data.pop("faults"))
        if "clouds" in data:
            kwargs["clouds"] = _clouds(
                data.pop("clouds"), kwargs.get("epoch_utc_ns", options.epoch_utc_ns)
            )
        if data:
            raise ValueError(f"unknown sim option(s): {', '.join(sorted(data))}")
        return replace(options, **kwargs)


def _check_keys(name: str, table: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"unknown key(s) in sim option {name}: {', '.join(unknown)}")


def _turbulence(table: Mapping[str, Any], seed: int) -> TurbulenceConfig:
    allowed = {
        "r0_m",
        "outer_scale_m",
        "zenith_angle_deg",
        "screen_points",
        "boiling",
        "wind_variability",
        "layers",
        "seed",
    }
    _check_keys("turbulence", table, allowed)
    kwargs: dict[str, Any] = {"seed": seed}
    for key in allowed - {"layers"}:
        if key in table:
            kwargs[key] = table[key]
    if "outer_scale_m" in kwargs and kwargs["outer_scale_m"] in ("inf", "infinity"):
        kwargs["outer_scale_m"] = math.inf
    if "layers" in table:
        layers = []
        for entry in table["layers"]:
            _check_keys(
                "turbulence.layers", entry, {"cn2_fraction", "wind_speed_m_s", "wind_direction_deg"}
            )
            layers.append(Layer(**entry))
        kwargs["layers"] = tuple(layers)
    return TurbulenceConfig(**kwargs)


def _scintillation(table: Mapping[str, Any]) -> ScintillationConfig:
    _check_keys("scintillation", table, {"sigma0", "enabled", "knee_s", "correlation_time_s"})
    return ScintillationConfig(**table)


def _faults(table: Mapping[str, Any]) -> SimFaults:
    allowed = {
        "drop_probability",
        "drop_burst",
        "scripted_drops",
        "timeout_probability",
        "scripted_timeouts",
        "slow_read_s",
        "slow_read_probability",
        "scripted_slow_reads",
        "disconnect_at_frame",
        "disconnect_clears_at",
        "stall_at_frame",
        "stall_clears_at",
        "geometry_change",
        "seed",
    }
    _check_keys("faults", table, allowed)
    kwargs: dict[str, Any] = dict(table)
    if "scripted_drops" in kwargs:
        drops = kwargs["scripted_drops"]
        pairs = drops.items() if isinstance(drops, Mapping) else drops
        kwargs["scripted_drops"] = tuple((int(frame), int(count)) for frame, count in pairs)
    for key in ("scripted_timeouts", "scripted_slow_reads"):
        if key in kwargs:
            kwargs[key] = frozenset(int(item) for item in kwargs[key])
    for key in ("disconnect_clears_at", "stall_clears_at"):
        if key in kwargs:
            value = kwargs[key]
            kwargs[key] = (
                RecoveryLevel[value.upper()] if isinstance(value, str) else RecoveryLevel(value)
            )
    if "geometry_change" in kwargs:
        _check_keys(
            "faults.geometry_change",
            kwargs["geometry_change"],
            {"dx", "dy", "d_width", "d_height", "times"},
        )
        kwargs["geometry_change"] = GeometryChange(**kwargs["geometry_change"])
    return SimFaults(**kwargs)


def _clouds(entries: Any, epoch_utc_ns: int) -> Clouds:
    events = []
    for entry in entries:
        _check_keys("clouds", entry, {"start_s", "duration_s", "transmission", "ramp_s"})
        events.append(
            CloudEvent(
                start_utc_ns=epoch_utc_ns + round(float(entry.get("start_s", 0.0)) * NS_PER_S),
                duration_s=float(entry["duration_s"]),
                transmission=float(entry.get("transmission", 0.1)),
                ramp_s=float(entry.get("ramp_s", 5.0)),
            )
        )
    return Clouds(events=tuple(events))
