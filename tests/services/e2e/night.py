"""A simulated night for `core` on a virtual clock: real analyzers, the sim camera, a real store.

`build_night` assembles a `CoreApp` in the way that `seeingmon core` does, with three changes that
make a night run in seconds and give the same answer every time:

- the clock is a `VirtualClock`, so reading a frame advances time and nothing sleeps;
- the camera is the `sim` driver in this process, built with the same options that `seeingmon dev`
  gives `acquire`, so the sky, the catalog, and the first pointing solution agree (see
  `seeingmon.services.simsky`);
- the survey analysis runs inline in the scheduler thread, through an `InlineExecutor`.

With `dark_set`, the data directory gets a dark set before `core` starts, recorded by the
production dark session on the same simulated camera with a cover on, so the survey frames get a
sky brightness, and `sky.dark` can fire.

The analyzers are the production ones: the fast path (`seeingmon.fastpath`) and the survey pipeline
(`seeingmon.survey`). A test injects the truth through the options of the simulator (the Fried
parameter, the clouds, the sky brightness) and reads what `core` stored.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seeingmon.clock import DEFAULT_START_UTC_NS, NS_PER_S, VirtualClock, iso_to_utc_ns
from seeingmon.config import Config, load_config
from seeingmon.drivers.sim import SimOptions
from seeingmon.drivers.sim import create as create_sim
from seeingmon.drivers.sim.stars import Pointing, StarField
from seeingmon.profile import Profile, load_profile
from seeingmon.records import Record
from seeingmon.scheduler import SchedulerConfig
from seeingmon.services.acquire.factory import create_camera_driver
from seeingmon.services.config import ServicesConfig
from seeingmon.services.core.app import CoreApp, CoreParts
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.notify import SystemdNotifier
from seeingmon.services.simsky import (
    seed_solution,
    sim_catalog,
    sim_field,
    write_seed,
    write_small_profile,
)
from seeingmon.store.layout import DataLayout
from seeingmon.survey.analyzer import InlineExecutor
from seeingmon.survey.catalog import write_catalog
from seeingmon.survey.dark import DarkLibrary
from seeingmon.survey.dark_session import DarkSessionOptions, run_dark_session

from ..addresses import unique_address
from ..core.rig import FakeRemote, NoSleepClock, read_all

SIM_LATITUDE_DEG = 55.0
REFERENCE_PROFILE = "asi294mm-gs250"  # the full sensor, as on the camera
KEY = ConnectionKey.from_text("an-e2e-key-of-more-than-32-characters-long")


@dataclass(slots=True)
class Night:
    """A running composition root and the truth that the sim was given."""

    app: CoreApp
    clock: VirtualClock
    directory: Path
    start_utc_ns: int
    r0_cm: float
    config: Config

    def run_for(self, seconds: float, *, tick_every_s: float = 1.0) -> None:
        """Step the scheduler for `seconds` of virtual time, and do the periodic work."""
        end = self.clock.monotonic_ns() + round(seconds * NS_PER_S)
        next_tick = 0
        while self.clock.monotonic_ns() < end:
            self.app.scheduler.step()
            now = self.clock.monotonic_ns()
            if now >= next_tick:
                self.app.tick()
                next_tick = now + round(tick_every_s * NS_PER_S)

    def run_until(
        self,
        condition: Any,
        *,
        limit_s: float = 1800.0,
        slice_s: float = 10.0,
    ) -> float:
        """Step in slices until `condition()` holds. Returns the virtual seconds that passed."""
        waited = 0.0
        while not condition():
            if waited >= limit_s:
                raise AssertionError(f"the condition did not hold within {limit_s:g} virtual s")
            self.run_for(slice_s)
            waited += slice_s
        return waited

    def records(self, record_type: str) -> list[Any]:
        """The stored records of a type, in row order (typed `Any`, so a test reads any field)."""
        assert self.app.storage is not None
        found: list[Record] = read_all(self.app.storage.store, record_type)
        return list(found)

    def events(self, kind: str | None = None) -> list[Any]:
        return [e for e in self.records("event") if kind is None or e.kind == kind]

    def seconds_since_start(self) -> float:
        return (self.clock.utc_ns() - self.start_utc_ns) / NS_PER_S


def cloud(
    start_s: float, duration_s: float, *, transmission: float, ramp_s: float = 5.0
) -> dict[str, float]:
    """One cloud event of the sim, `start_s` seconds after the start of the night."""
    return {
        "start_s": start_s,
        "duration_s": duration_s,
        "transmission": transmission,
        "ramp_s": ramp_s,
    }


def record_dark_set(data_dir: Path, profile: Profile, start_utc_ns: int, *, gain: int) -> None:
    """Record a dark set of the survey mode an hour before `start_utc_ns`, with a covered camera.

    The camera is the sim with the seed and the sensor of `build_night`, so the bias, the dark
    current, and the hot pixels match, with no stars and a sky far too faint to give an electron.
    """
    covered = SimOptions(
        seed=1, stars=StarField.from_arrays([], [], []), sky_mag_arcsec2=60.0, twilight=False
    )
    clock = VirtualClock(start_utc_ns - 3600 * NS_PER_S)
    run_dark_session(
        create_sim(profile=profile, clock=clock, options=covered),
        DarkLibrary.from_layout(DataLayout(data_dir)),
        profile,
        clock,
        DarkSessionOptions(
            mode=profile.survey_mode.mode, gain=gain, frames=3, bias_frames=3, wait=False
        ),
        say=lambda _: None,
    )


def build_night(
    directory: Path,
    *,
    start: str = "2026-01-01T18:10:00Z",
    r0_cm: float = 10.0,
    window_s: float = 6.0,
    fast_exposure_us: int = 250,
    cadence_s: float = 60.0,
    missing_star_frames: int = 90,
    polaris_mag: float | None = None,
    clouds: Sequence[Mapping[str, float]] = (),
    longitude_deg: float = 0.0,
    config_extra: str = "",
    sim_extra: Mapping[str, Any] | None = None,
    parts: Mapping[str, Any] | None = None,
    sensor: str = "small",
    night_split_utc_hour: float | None = None,
    dark_set: bool = False,
) -> Night:
    """Build and start a `CoreApp` on the simulated sky. Stop it with `night.app.stop()`.

    `sensor` is `small` for a sensor of 1280 by 960 bin1 pixels, which keeps the survey frames
    cheap, or `full` for the reference profile, whose survey frames take seconds of CPU each.
    `night_split_utc_hour` sets `[survey] night_split_utc_hour`, and `dark_set` records a dark set
    first (see the module text).
    """
    start_utc_ns = iso_to_utc_ns(start)
    clock = VirtualClock(start_utc_ns)
    if sensor == "small":
        profile_name = write_small_profile(directory).as_posix()
    elif sensor == "full":
        profile_name = REFERENCE_PROFILE
    else:
        raise ValueError(f"sensor must be small or full, not {sensor!r}")
    profile = load_profile(profile_name)
    catalog, _ = sim_catalog(1, polaris_mag=polaris_mag)
    catalog_path = directory / "catalog.bin"
    write_catalog(catalog_path, catalog)
    pointing = Pointing(t_ref_utc_ns=start_utc_ns)
    seed_path = directory / "seed.json"
    write_seed(
        seed_path,
        seed_solution(
            profile, sim_field(1, polaris_mag=polaris_mag), pointing, start_utc_ns
        ).solution,
    )
    lines = [
        'station_id = "e2e"',
        f'profile = "{profile_name}"',
        "[paths]",
        f'data_dir = "{(directory / "data").as_posix()}"',
        "[site]",
        f"latitude_deg = {SIM_LATITUDE_DEG}",
        f"longitude_deg = {longitude_deg}",
        "elevation_m = 0.0",
        "[fastpath]",
        f"window_s = {window_s}",
        f"min_window_s = {window_s / 2}",
        # One window in each fast period, a survey step every minute, and a period that ends after
        # a second without the star: a night of minutes holds many cycles, and a frame costs 9 ms
        # of CPU.
        "[scheduler.fast]",
        f"exposure_us = {fast_exposure_us}",
        f"analysis_window_s = {window_s}",
        f"window_s = {window_s}",
        f"missing_star_frames = {missing_star_frames}",
        "[scheduler.survey]",
        f"cadence_s = {cadence_s}",
        "[scheduler.cloud]",
        f"fast_window_s = {window_s}",
        f"survey_cadence_s = {cadence_s}",
        "[survey]",
        f'catalog_path = "{catalog_path.as_posix()}"',
        "solvers = []",
        *(
            []
            if night_split_utc_hour is None
            else [f"night_split_utc_hour = {night_split_utc_hour}"]
        ),
        "[survey.cloud]",
        "min_expected = 4",
        "expected_snr = 10.0",
        "mag_limit = 13.0",
        "[services.core]",
        f'seed_solution_file = "{seed_path.as_posix()}"',
    ]
    local = directory / "local.toml"
    local.write_text("\n".join(lines) + "\n" + config_extra, encoding="utf-8")
    config = load_config(local_file=local, env={})
    if dark_set:
        long_gain = config.section("scheduler", SchedulerConfig).survey.long_gain
        record_dark_set(directory / "data", profile, start_utc_ns, gain=long_gain)
    services = config.section("services", ServicesConfig).model_copy(
        update={
            "acquire_address": unique_address("acquire"),
            "core_address": unique_address("core"),
        }
    )
    options: dict[str, Any] = {
        "seed": 1,
        "psf_mode": "gaussian",
        "polaris": "real",
        **({} if polaris_mag is None else {"polaris_mag": polaris_mag}),
        "pointing": {"t_ref_utc_ns": start_utc_ns},
        "turbulence": {"r0_m": r0_cm / 100.0, "zenith_angle_deg": 35.0},
        **(sim_extra or {}),
    }
    if clouds:
        # The sim counts the start of a cloud from its own epoch, which is the start of 2026.
        shift_s = (start_utc_ns - DEFAULT_START_UTC_NS) / NS_PER_S
        options["clouds"] = [{**c, "start_s": c["start_s"] + shift_s} for c in clouds]
    driver = create_camera_driver("sim", profile=profile, clock=clock, options=options)
    core_parts = CoreParts(
        driver=driver,
        remote=FakeRemote(),
        survey_executor=InlineExecutor(),
        sinks=[],
        storage_clock=NoSleepClock(clock),
        notifier=SystemdNotifier(env={}),
    )
    for name, value in (parts or {}).items():
        setattr(core_parts, name, value)
    app = CoreApp(
        config,
        services,
        clock,
        KEY,
        parts=core_parts,
        endpoint=services.endpoint("core"),
        threads=False,
    )
    night = Night(app, clock, directory, start_utc_ns, r0_cm, config)
    app.start()
    return night


def wall_seconds() -> float:
    """The real time, for a test that reports how long a run took (never asserted on)."""
    return time.monotonic()
