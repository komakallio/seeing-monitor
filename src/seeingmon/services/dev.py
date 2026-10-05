"""`seeingmon dev`: the whole system on a simulated sky, started with one command.

The launcher starts `acquire` with the `sim` driver, `core`, and `web` as child processes of this
one. They talk over local sockets of a private temporary folder, keep their data in that folder,
and share a scaled clock, so a simulated evening passes in minutes. The launcher prints the URL of
the web UI, and stops every child when you press Ctrl+C.

**What a simulated run is.** The star field of the simulator, the cap catalog, and the first
pointing solution come from one seed (`seeingmon.services.simsky`), so `core` follows the simulated
Polaris from the first second with no plate solver. The sensor is small by default (`--sensor full`
makes the reference sensor). With the small sensor, Polaris starts 86 pixels left of the middle
of the 640 by 480 pixel field and drifts right by about 2.5 pixels a minute, so it leaves the field
after nearly three hours of simulated time, and a long run needs `--sensor full`. The fast stream
uses a longer exposure and a dimmer Polaris than a real night, because a scaled clock multiplies
the frame rate that the machine must sustain.

**The cover.** A simulated camera has no lens cap, so a dark session needs a stand-in. The
launcher gives the `sim` driver a cover file, named `cover` in the run folder. The camera shows
the sensor alone while that file exists. To take darks from the web UI, create the file (the
launcher prints its path), start the dark session on the Dark page, and delete the file when the
session says that the cover can come off. The simulator looks at the file twice a second. A dev
run also shortens the dark session to a few frames of each kind.

**The real camera.** `--driver asi` swaps the simulator for your ZWO camera, so that you can try
the web UI (the Dark page, the Align page) on the hardware. `acquire` runs the `asi` driver on the
full sensor (`asi294mm-gs250`, so `--sensor` does not apply), and all three processes run in real
time on the system clock: `--speed` must be 1, and `--start` does not apply. The fast stream takes
at most the 2 ms of `[scheduler.fast] exposure_us`, which the adaptive exposure shortens in a bright
scene, and the simulator gets no option. The vendor library comes from `--asi-library`, or from
`SEEINGMON_ASI__LIBRARY_PATH` in your environment, and never from a file. The launcher gives it to
`acquire` alone, through the environment of that child, and prints no path. `acquire` raises the
priority of its capture thread on the real camera, and on Windows it also asks for a 1 ms system
timer (the simulator never does, because it renders inside the read).
`--no-raise-priority` turns both off, so that you can compare two runs: the health line of
`acquire` in its log shows the priority, the timer, and the share of late and lost frames.
On Windows the launcher also holds a power request for the whole run (`KeepAwake`), so that the
machine does not sleep when it sits idle. Without it, Windows Modern Standby froze `acquire` and
`core` and cut the power of the camera at the first light. The request stops idle sleep only: the
display may turn off, and closing the lid still sleeps the laptop unless the power settings set
the lid action to Do nothing. `--no-keep-awake` turns the request off.
`--data-dir` keeps the store, the dark library, and the images in a folder that survives the run.
Without it, the run keeps its temporary folder. The sky catalog, the first pointing solution,
and the site stay synthetic, so a camera that sees a room or a dark reports no stars, and the
seeing windows and the sky quality stay empty.

**A real sky.** `--real-sky` (with `--driver asi` and `--data-dir`) drops the synthetic sky. The
launcher writes no simulated catalog and no seed solution. On a new data folder, `core` starts with
no pointing and solves the first survey frame and the alignment frames with your plate solvers, as a
first production start does. A later run on the same folder starts from the newest solution in the
store, as a production restart does. The site, the catalog, and the solvers come from your local
configuration: the launcher layers the `[site]`, `[survey]`, and `[alignment]` tables of
`local/config.toml` and the variables `SEEINGMON_SITE__*`, `SEEINGMON_SURVEY__*`, and
`SEEINGMON_ALIGNMENT__*`, the way that it layers `[web]` and `[auth]`, and it gives them to `core`
alone, in the environment of that child. `[site]` needs all three values, and `[survey]` needs a
catalog file that exists. The launcher refuses to start without them, and its message names the
table and the setting, never a value. The scheduler follows the real Sun at the real site. The
survey cloud limits are the production defaults, because the simulated limits suit a synthetic star
field only, and the windows (20 s) and the dark session (5 frames of each kind) stay short. The
banner says what is real, names the solvers that run, and warns about a solver program or a folder
that it does not find. It prints no coordinate and no path. The logs of the children go to the
folder `logs/<start time>` of your data folder, so that they outlive the run, and the banner names
that folder relative to your data folder. The level of the logs is `info`, so that they show each
solver run.

**Isolation.** A simulated run must never reach a real sink, device, or data directory. Each child
gets a clean environment: no `SEEINGMON_*` variable of yours reaches it, and `--local-config`
points the children that take it at a file that does not exist, so `local/config.toml` is not
read. The launcher sets every key that matters (the data folder, the profile, the survey catalog,
the services, and the clock) in the environment of the child. A real-sky run adds the three tables
above, and nothing else of your file: no sink, heater, SQM-LE, or power setting.

**The web settings are yours.** The `web` child is the only process that sees your `[web]` and
`[auth]` sections. The launcher layers `local/config.toml` and the variables `SEEINGMON_WEB__*`
and `SEEINGMON_AUTH__*`, and ignores every other key and variable. The address and the port come
from your `[web]` section, and default to the loopback interface and port 8080 when you set none.
The launcher prints one URL for each bind address (`bind_address` and `extra_bind_addresses`), and
writes no address to a file or a log. A wildcard (`0.0.0.0` or `::`) stands for every interface, so
it prints the loopback URL and, for `0.0.0.0`, one URL for each IPv4 address that this device has
now (a phone on the same network opens one of them). It passes the settings to the child in its
environment, and never on the command line.

**The API token.** When your `[auth]` section has no token hash, the launcher makes a random token
for the run, gives its hash (the one that `seeingmon web hash-token` makes) to the `web` child, and
prints the token once. The token goes to no file.

The children start with `sys.executable -m seeingmon`, so the launcher works on Windows. On
Windows each child has its own process group, and the launcher stops it with a console break.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, TypeVar

from seeingmon.cli import CliError
from seeingmon.clock import NS_PER_S, SystemClock, utc_ns_to_iso
from seeingmon.config import Config, ConfigError
from seeingmon.services.web.netaddr import (
    LOOPBACK_V6,
    connect_address,
    device_addresses,
    is_wildcard,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from seeingmon.scheduler.config import SearchConfig
    from seeingmon.survey.config import SurveyConfig

ModelT = TypeVar("ModelT", bound="BaseModel")

SENSORS = ("small", "full")
DEFAULT_START = "2026-01-01T19:00:00Z"
DEFAULT_PORT = 8080
LOOPBACK = "127.0.0.1"
READY_TIMEOUT_S = 120.0
STOP_TIMEOUT_S = 25.0
CLOCK = SystemClock()
WEB_TIMEOUT_S = 30.0

# The synthetic site of the simulator and of the scheduler: it is nobody's real site.
SIM_LATITUDE_DEG = 55.0
SIM_LONGITUDE_DEG = 0.0

# What a scaled clock allows: the simulator renders about 40 frames a second on one core (a frame of
# a 50 ms exposure costs about 20 ms), and the clock multiplies the frame rate of the stream. The
# fast stream of a dev run therefore takes a long exposure and a Polaris that is dim enough not to
# saturate it, and the default speed is real time. Twice real time is the most that a quiet machine
# sustains. The windows are short, so that the UI has data half a minute after the start.
DEV_FAST_EXPOSURE_US = 50_000
DEV_WINDOW_S = 20.0
# The fast period, in windows. A simulated run keeps three. A real camera runs seven (140 s), so
# that the fast period and the survey step (about 40 s) fill the 180 s cadence: the camera idles
# less, and a seeing reading is at most one window plus the survey step old, about a minute.
DEV_FAST_WINDOWS = 3
REAL_FAST_WINDOWS = 7
DEFAULT_SPEED = 1.0
DEV_POLARIS_MAG = 6.0
# The simulator of the full sensor renders a survey frame inside the camera read, which takes
# seconds, so the reads of that run wait longer than the default half second.
FULL_SENSOR_READ_MARGIN_S = 20.0
# The cover of the simulated camera is a file in the run folder (see the docstring of the module),
# and the dark session of a dev run is short: a few frames of each kind, and a quick look at the
# cover. The frames keep the exposure of the survey, so the session takes minutes at real time.
COVER_FILE_NAME = "cover"
DEV_DARK_FRAMES = 5
DEV_DARK_POLL_S = 2.0
# The real camera: the driver name, the profile of its full sensor, and the variable that names the
# vendor library. The launcher reads the variable from its own environment and gives it to
# `acquire`.
REAL_DRIVER = "asi"
REAL_PROFILE = "asi294mm-gs250"
ASI_LIBRARY_VARIABLE = "SEEINGMON_ASI__LIBRARY_PATH"
# The power request of a real-camera run on Windows: the flags of `SetThreadExecutionState`.
# `ES_CONTINUOUS` keeps the state until the next call, and `ES_SYSTEM_REQUIRED` says that the
# system is in use, so that Windows does not enter idle sleep. The display is not part of it.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
KEEP_AWAKE_NOTE = (
    "Windows stays awake for this run: it does not sleep when it sits idle, and the display may "
    "still turn off. Closing the lid sleeps the laptop unless the power settings set the lid "
    "action to Do nothing. --no-keep-awake turns the request off."
)
# The real sky (`--real-sky`): the tables of your local configuration that `core` needs, the keys
# of the site that all have to be there, and the folder of your data folder that holds the logs of
# the children, in one subfolder for each run.
REAL_SKY_SECTIONS = ("site", "survey", "alignment")
SITE_KEYS = ("latitude_deg", "longitude_deg", "elevation_m")
LOGS_DIRNAME = "logs"
# What a real-sky run keeps of the shortcuts of a dev run: a short dark session. The cloud limits
# of the survey stay at the production defaults, unless your `[survey]` table sets them.
REAL_SKY_SURVEY_DEFAULTS: dict[str, Any] = {
    "dark": {"frames": DEV_DARK_FRAMES, "bias_frames": DEV_DARK_FRAMES, "poll_s": DEV_DARK_POLL_S}
}


@dataclass(frozen=True, slots=True)
class DevOptions:
    """What the command line of `seeingmon dev` chooses."""

    speed: float = DEFAULT_SPEED
    window_s: float = DEV_WINDOW_S
    port: int | None = None
    sensor: str = "small"
    seed: int = 1
    start: str | None = None
    keep_data: bool = False
    log_level: str = "warning"
    fast_exposure_us: int = DEV_FAST_EXPOSURE_US
    polaris_mag: float | None = DEV_POLARIS_MAG
    # For the tests of the system: another driver for `acquire`, and settings that replace those of
    # the plan. A table merges into the plan key by key, and any other value replaces.
    acquire_driver: str = "sim"
    extra_sim: Mapping[str, Any] = field(default_factory=dict)
    core_overrides: Mapping[str, Any] = field(default_factory=dict)
    acquire_overrides: Mapping[str, Any] = field(default_factory=dict)
    # The real camera (`acquire_driver` is `asi`): where the vendor library is (without it, the
    # launcher reads the variable of its own environment), a data folder that survives the run, and
    # whether the person chose a sensor, which the real camera ignores.
    asi_library: str | None = None
    data_dir: Path | None = None
    sensor_given: bool = False
    # The real sky: the site, the catalog, and the solvers come from your local configuration, and
    # the run needs the real camera and a data folder that survives it.
    real_sky: bool = False
    # Whether acquire raises the priority of its capture thread on the real camera (and, on
    # Windows, asks for a 1 ms timer). A comparison turns it off. The simulator never raises it.
    raise_priority: bool = True
    # Whether a run on the real camera holds a Windows power request that stops idle sleep.
    keep_awake: bool = True


@dataclass(slots=True)
class ChildSpec:
    """One child process: its command line, its environment, and where its log goes."""

    name: str
    argv: list[str]
    env: dict[str, str]
    log: Path


@dataclass(slots=True)
class DevPlan:
    """Everything that a dev run needs, decided and written to the temporary folder."""

    options: DevOptions
    directory: Path
    key: str
    children: list[ChildSpec]
    bind_addresses: list[str]
    port: int
    start_utc_ns: int
    origin_real_ns: int
    token: str | None = None
    core_endpoint: str = ""
    acquire_endpoint: str = ""
    cover_file: Path | None = None  # the simulated camera is covered while this file exists
    real: bool = False  # the camera is the real one, and the clock is the system clock
    real_sky: bool = False  # the site, the catalog, and the solvers are yours, and no sky is seeded
    notes: list[str] = field(default_factory=list)  # lines for the banner
    # What the launcher found wrong with the setup of a real sky, for the banner.
    warnings: list[str] = field(default_factory=list)
    log_dir: Path | None = None  # a real-sky run keeps the logs of the children here
    log_folder: str = ""  # the same folder, relative to the data folder, for the banner

    def urls(self) -> list[str]:
        """One URL for each bind address, with an IPv6 address in brackets.

        A wildcard stands for every interface of its family: it gives the loopback URL, and
        `0.0.0.0` adds one URL for each IPv4 address that this device has now.
        """
        shown: list[str] = []
        for address in self.bind_addresses:
            if not is_wildcard(address):
                shown.append(address)
            elif ":" in address:  # the IPv6 wildcard
                shown.append(LOOPBACK_V6)
            else:  # the IPv4 wildcard
                shown.extend((connect_address(address), *device_addresses()))
        return [url_for(address, self.port) for address in dict.fromkeys(shown)]


# --- Settings ----------------------------------------------------------------------------------


def render_env_value(value: Any) -> str:
    """A configuration value as the text of an environment variable (a TOML scalar or array)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a TOML basic string
    if isinstance(value, list | tuple):
        return "[" + ", ".join(render_env_value(item) for item in value) + "]"
    raise TypeError(f"cannot put a {type(value).__name__} in an environment variable")


def flatten_env(table: Mapping[str, Any], prefix: str) -> dict[str, str]:
    """Turn nested configuration tables into `SEEINGMON_<SECTION>__<KEY>` variables."""
    result: dict[str, str] = {}
    for key, value in table.items():
        name = f"{prefix}__{str(key).upper()}"
        if isinstance(value, Mapping):
            result.update(flatten_env(value, name))
        else:
            result[name] = render_env_value(value)
    return result


def owner_tables(
    local_file: Path | str | None, env: Mapping[str, str], sections: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """The tables of the named sections that the owner set, and nothing else.

    The layers are the local configuration file and the variables `SEEINGMON_<SECTION>__*` of the
    named sections. The defaults of the repository are left out, so each table holds only what the
    owner chose, and a section that the owner did not set is an empty table. Every other key of the
    file and every other variable is ignored.
    """
    from seeingmon.config import load_config

    prefixes = tuple(f"SEEINGMON_{name.upper()}__" for name in sections)
    kept = {k: v for k, v in env.items() if k.startswith(prefixes)}
    with tempfile.TemporaryDirectory(prefix="smon-cfg-") as empty:
        Path(empty, "default.toml").write_text('profile = "none"\nstation_id = "none"\n')
        config = load_config(config_dir=empty, local_file=local_file, env=kept)
    data = config.effective(redact=False)
    return {name: dict(data[name]) if isinstance(data.get(name), dict) else {} for name in sections}


def owner_settings(
    local_file: Path | str | None, env: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The `[web]` and `[auth]` tables that the owner set, and nothing else.

    The layers are the local configuration file and the variables `SEEINGMON_WEB__*` and
    `SEEINGMON_AUTH__*` (see `owner_tables`).
    """
    tables = owner_tables(local_file, env, ("web", "auth"))
    return tables["web"], tables["auth"]


def url_for(address: str, port: int) -> str:
    """The URL of the web UI at one bind address. An IPv6 address goes in brackets."""
    host = f"[{address}]" if ":" in address and not address.startswith("[") else address
    return f"http://{host}:{port}/"


def child_environment(parent: Mapping[str, str]) -> dict[str, str]:
    """The environment of a child: yours, without any setting of the system, and without secrets."""
    blocked = ("SEEINGMON_",)
    env = {k: v for k, v in parent.items() if not k.startswith(blocked)}
    for name in ("CREDENTIALS_DIRECTORY", "NOTIFY_SOCKET"):
        env.pop(name, None)
    if sys.platform == "win32":
        # A Windows process without SYSTEMROOT cannot start Winsock, so asyncio fails to import.
        for name in ("SYSTEMROOT", "SystemRoot", "WINDIR"):
            if name not in env and name in os.environ:
                env[name] = os.environ[name]
    env["PYTHONUNBUFFERED"] = "1"
    return env


# --- The plan ----------------------------------------------------------------------------------


def _endpoint_text(directory: Path, role: str, token: str) -> str:
    """A fresh address of a service: a pipe name on Windows, a socket in the run folder elsewhere.

    The name holds a random token, so two runs, and a run and a real system on the same machine,
    never share an address. The default names (`seeingmon-core`) collide on Windows.
    """
    if os.name == "nt":
        return f"\\\\.\\pipe\\seeingmon-dev-{token}-{role}"
    return str(directory / f"{role}-{token}.sock")


def default_start_utc_ns() -> int:
    """The default start: five minutes after the Sun passes 18 degrees below the horizon.

    The windows of an earlier hour carry the `twilight` flag, and in a bright evening sky the
    scheduler waits in `safe` or searches for Polaris. Starting in the dark gives windows at once,
    without twilight flags. Pass `--start` to see the evening instead.
    """
    from seeingmon.clock import NS_PER_S, iso_to_utc_ns
    from seeingmon.scheduler.ephemeris import next_sun_crossing_utc_ns

    dark = next_sun_crossing_utc_ns(
        iso_to_utc_ns("2026-01-01T12:00:00Z"),
        SIM_LATITUDE_DEG,
        SIM_LONGITUDE_DEG,
        -18.0,
        rising=False,
    )
    if dark is None:
        return iso_to_utc_ns(DEFAULT_START)
    return dark + 300 * NS_PER_S


def build_plan(
    options: DevOptions,
    *,
    directory: Path,
    local_file: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    python: str | None = None,
    web_command: Sequence[str] | None = None,
    origin_real_ns: int | None = None,
) -> DevPlan:
    """Decide the settings of every child, and write the files that they need to `directory`.

    A real-sky plan (`options.real_sky`) reads and checks your site, survey, and alignment tables
    first, so that a missing value stops the launcher before it writes a file or starts a child.
    """
    from seeingmon.clock import iso_to_utc_ns
    from seeingmon.services.web.auth import hash_token

    _check_real_sky(options)
    parent_env = os.environ if env is None else env
    data_dir = directory / "data" if options.data_dir is None else options.data_dir
    real_sky = real_sky_settings(local_file, parent_env, data_dir) if options.real_sky else None
    interpreter = python or sys.executable
    token = uuid.uuid4().hex[:12]
    key = secrets.token_urlsafe(32)
    directory.mkdir(parents=True, exist_ok=True)
    real = options.acquire_driver == REAL_DRIVER
    origin_ns = CLOCK.utc_ns() if origin_real_ns is None else origin_real_ns
    if real:
        start_utc_ns = origin_ns  # real time: the simulated sky of the plan starts now
    else:
        start_utc_ns = iso_to_utc_ns(options.start) if options.start else default_start_utc_ns()

    # The sky: yours (the real sky), or a simulated one with a profile, a catalog, a first
    # solution, and the pointing of the camera.
    if real_sky is not None:
        profile_path: Path | str = REAL_PROFILE
        sky_tables = real_sky.core_tables
    else:
        if options.sensor == "small" and not real:
            from seeingmon.services.simsky import write_small_profile

            profile_path = write_small_profile(directory)
        else:
            profile_path = REAL_PROFILE
        sky_tables = _simulated_sky_tables(options, directory, profile_path, start_utc_ns)
    nowhere = directory / "no-local-config.toml"  # does not exist, so no local file is read

    # A real-sky run keeps the logs of the children in your data folder, where they outlive it.
    log_root, log_folder = directory, ""
    if real_sky is not None:
        stamp = utc_ns_to_iso(origin_ns, digits=0).replace("-", "").replace(":", "")
        log_folder = f"{LOGS_DIRNAME}/{stamp}"
        log_root = data_dir / LOGS_DIRNAME / stamp
        log_root.mkdir(parents=True, exist_ok=True)

    slow_reads: dict[str, Any] = (
        {}
        if options.sensor == "small" or real  # a real camera reads inside its own exposure
        else {"read_timeout_margin_s": FULL_SENSOR_READ_MARGIN_S}
    )
    acquire_address = _endpoint_text(directory, "acquire", token)
    core_address = _endpoint_text(directory, "core", token)
    cover_file = directory / COVER_FILE_NAME if options.acquire_driver == "sim" else None
    clock: dict[str, Any] = (
        {"kind": "system"}
        if real
        else {
            "kind": "scaled",
            "speed": options.speed,
            "start_utc_ns": start_utc_ns,
            "origin_real_ns": origin_ns,
        }
    )
    shared: dict[str, Any] = {
        "station_id": "dev",
        "profile": str(profile_path),
        "paths": {"data_dir": str(data_dir)},
        "services": {
            "connection_key": key,
            "acquire_address": acquire_address,
            "core_address": core_address,
            "clock": clock,
        },
    }
    sim_options: dict[str, Any] = {
        "seed": options.seed,
        "polaris": "real",
        **({} if options.polaris_mag is None else {"polaris_mag": options.polaris_mag}),
        "psf_mode": "gaussian",
        "pointing": {"t_ref_utc_ns": start_utc_ns},
        **({} if cover_file is None else {"cover_file": str(cover_file)}),
        **options.extra_sim,
    }
    acquire_table: dict[str, Any] = (
        # The real camera keeps the default frame gap, and its options are its own. Its capture
        # thread waits in the SDK, so a raised priority costs the machine nothing, and it keeps
        # the reads on time. The simulator renders inside the read, so a raised priority would
        # starve the other threads of the process.
        {
            "driver": options.acquire_driver,
            "raise_priority": options.raise_priority,
            "driver_options": {},
        }
        if real
        else {
            "driver": options.acquire_driver,
            "gap_factor": 1000.0,
            "raise_priority": False,
            "driver_options": sim_options if options.acquire_driver == "sim" else {},
            **slow_reads,
        }
    )
    acquire_settings = _merge(
        _merge(shared, {"services": {"acquire": acquire_table}}), options.acquire_overrides
    )
    fast_table: dict[str, Any] = {
        "analysis_window_s": options.window_s,
        "window_s": options.window_s * (REAL_FAST_WINDOWS if real else DEV_FAST_WINDOWS),
    }
    if not real:
        fast_table["exposure_us"] = options.fast_exposure_us  # the real one keeps its limit of 2 ms
    core_settings = _merge(
        _merge(
            shared,
            _merge(
                {
                    # Windows of 20 s, so that the UI has data half a minute after the start.
                    "fastpath": {
                        "window_s": options.window_s,
                        "min_window_s": options.window_s / 2,
                    },
                    "scheduler": {"fast": fast_table, "loop": dict(slow_reads)},
                },
                sky_tables,  # the site, the survey, and what else the sky brings
            ),
        ),
        options.core_overrides,
    )

    web_table, auth_table = owner_settings(local_file, parent_env)
    bind = str(web_table.get("bind_address", LOOPBACK))
    extra = web_table.get("extra_bind_addresses", [])
    extra_list = [str(item) for item in extra] if isinstance(extra, list) else []
    port = int(options.port or web_table.get("port", DEFAULT_PORT))
    web_table = {**web_table, "bind_address": bind, "port": port}
    generated: str | None = None
    if not auth_table.get("token_hash") and not auth_table.get("token_hash_file"):
        generated = secrets.token_urlsafe(18)
        auth_table = {**auth_table, "token_hash": hash_token(generated)}
    web_settings = _merge(shared, {"web": web_table, "auth": auth_table})

    base = child_environment(parent_env)
    base_args = [interpreter, "-m", "seeingmon"]

    def spec(
        name: str,
        argv: list[str],
        settings: Mapping[str, Any],
        extra_env: Mapping[str, str] | None = None,
    ) -> ChildSpec:
        env = dict(base)
        top = {k: v for k, v in settings.items() if not isinstance(v, Mapping)}
        env.update({f"SEEINGMON_{k.upper()}": render_env_value(v) for k, v in top.items()})
        for section, table in settings.items():
            if isinstance(table, Mapping):
                env.update(flatten_env(table, f"SEEINGMON_{section.upper()}"))
        env.update(extra_env or {})
        return ChildSpec(name, argv, env, log_root / f"{name}.log")

    # The one setting of the real camera that comes from the person: the vendor library. Only
    # `acquire` gets it, in its environment, and never on a command line.
    library = options.asi_library or parent_env.get(ASI_LIBRARY_VARIABLE) if real else None
    acquire_env = {ASI_LIBRARY_VARIABLE: str(Path(library).expanduser())} if library else {}
    level = ["--log-level", options.log_level]
    children = [
        spec(
            "acquire",
            [*base_args, "acquire", "--local-config", str(nowhere), *level],
            acquire_settings,
            acquire_env,
        ),
        spec("core", [*base_args, "core", "--local-config", str(nowhere), *level], core_settings),
        spec(
            "web",
            [*(web_command or [*base_args, "web", "--local-config", str(nowhere), *level])],
            web_settings,
        ),
    ]
    return DevPlan(
        options=options,
        directory=directory,
        key=key,
        children=children,
        bind_addresses=[bind, *extra_list],
        port=port,
        start_utc_ns=start_utc_ns,
        origin_real_ns=origin_ns,
        token=generated,
        core_endpoint=core_address,
        acquire_endpoint=acquire_address,
        cover_file=cover_file,
        real=real,
        real_sky=real_sky is not None,
        notes=(
            _real_sky_notes(options, real_sky.solvers, log_folder)
            if real_sky is not None
            else _real_notes(options)
            if real
            else []
        ),
        warnings=[] if real_sky is None else list(real_sky.warnings),
        log_dir=None if real_sky is None else log_root,
        log_folder=log_folder,
    )


def _simulated_sky_tables(
    options: DevOptions, directory: Path, profile_path: Path | str, start_utc_ns: int
) -> dict[str, Any]:
    """The tables of the simulated sky for `core`, and the files that they name.

    The simulated sky is a cap catalog, a first pointing solution (the seed), and a synthetic site,
    all from one seed (`seeingmon.services.simsky`), so `core` follows the simulated Polaris with no
    plate solver. The survey gets the cloud limits that suit a synthetic star field.
    """
    from seeingmon.drivers.sim.stars import Pointing
    from seeingmon.profile import load_profile
    from seeingmon.services.simsky import seed_solution, sim_catalog, write_seed
    from seeingmon.survey.catalog import write_catalog

    profile = load_profile(str(profile_path))
    catalog, field_ = sim_catalog(options.seed, polaris_mag=options.polaris_mag)
    catalog_path = directory / "catalog.bin"
    write_catalog(catalog_path, catalog)
    pointing = Pointing(t_ref_utc_ns=start_utc_ns)
    seed_path = directory / "seed.json"
    write_seed(seed_path, seed_solution(profile, field_, pointing, start_utc_ns).solution)
    return {
        "site": {
            "latitude_deg": SIM_LATITUDE_DEG,
            "longitude_deg": SIM_LONGITUDE_DEG,
            "elevation_m": 0.0,
        },
        "survey": {
            "catalog_path": str(catalog_path),
            "solvers": [],
            "cloud": {"min_expected": 4, "expected_snr": 10.0, "mag_limit": 13.0},
            "dark": {
                "frames": DEV_DARK_FRAMES,
                "bias_frames": DEV_DARK_FRAMES,
                "poll_s": DEV_DARK_POLL_S,
            },
        },
        "services": {"core": {"seed_solution_file": str(seed_path)}},
    }


@dataclass(frozen=True, slots=True)
class RealSky:
    """What a real-sky run takes from your local configuration, checked before anything starts.

    `core_tables` are the tables that `core` receives: the `site`, the `survey` (your table, with
    the shortcuts of a dev run under it, and the dark library of your data folder when you name
    none), and your `alignment` table when you set one. `solvers` are the plate solvers that run,
    in order. `warnings` say what the launcher found wrong with them. A warning names a setting and
    never a value.
    """

    core_tables: dict[str, dict[str, Any]]
    solvers: tuple[str, ...]
    warnings: list[str]


def real_sky_settings(
    local_file: Path | str | None, env: Mapping[str, str], data_dir: Path
) -> RealSky:
    """Read the site, survey, and alignment settings of a real-sky run, and check them.

    The layers are your local configuration and the variables `SEEINGMON_SITE__*`,
    `SEEINGMON_SURVEY__*`, and `SEEINGMON_ALIGNMENT__*` (see `owner_tables`). Raises `CliError`
    with the exit code 2 when the `[site]` table lacks one of its three values, when it still holds
    the placeholders of the template, when `[survey]` names no catalog file, a file that does not
    exist, or a file that is not a cap catalog (the check reads the header only), or when a table is
    not valid. The message names the table and the setting, and it never shows a value. A solver
    program or folder that the launcher does not find is a warning.
    """
    from seeingmon.scheduler.config import SiteConfig
    from seeingmon.services.core.settings import AlignmentSettings
    from seeingmon.survey.catalog import CatalogError, read_info
    from seeingmon.survey.config import SurveyConfig
    from seeingmon.survey.pipeline import solver_specs_from_config

    tables = owner_tables(local_file, env, REAL_SKY_SECTIONS)
    site_table, survey_table, alignment_table = (tables[name] for name in REAL_SKY_SECTIONS)

    missing = [key for key in SITE_KEYS if key not in site_table]
    if missing:
        raise CliError(
            f"the [site] table of your local configuration lacks {', '.join(missing)}. A real-sky "
            f"run needs {', '.join(SITE_KEYS)}, so that the scheduler follows the Sun at the site "
            "of your camera. Set them in the untracked local/config.toml, or in the variables "
            "SEEINGMON_SITE__<KEY>. The values stay in your local configuration and never go "
            "into the repository.",
            exit_code=2,
        )
    site = _validated(Config({"site": site_table}), "site", SiteConfig)
    if site.latitude_deg == 0.0 and site.longitude_deg == 0.0:
        raise CliError(
            "the [site] table of your local configuration still holds the placeholders of "
            "config/local.example.toml (a latitude and a longitude of 0). Set the site of your "
            "camera in the untracked local/config.toml.",
            exit_code=2,
        )

    survey = _validated(Config({"survey": survey_table}), "survey", SurveyConfig)
    if not survey.catalog_path:
        raise CliError(
            "the [survey] table of your local configuration has no catalog_path. A real-sky run "
            "needs the cap catalog that `seeingmon catalog build` writes (see the runbook, "
            '"First light on the dev machine").',
            exit_code=2,
        )
    if not Path(survey.catalog_path).is_file():
        raise CliError(
            "the catalog file that [survey] catalog_path names does not exist. Build it with "
            "`seeingmon catalog build`, or copy it there (see the runbook).",
            exit_code=2,
        )
    try:
        read_info(survey.catalog_path)  # the header only: it tells a catalog from any other file
    except CatalogError as error:
        raise CliError(
            f"the file that [survey] catalog_path names is not a cap catalog ({error}). Build it "
            "with `seeingmon catalog build`.",
            exit_code=2,
        ) from None
    try:
        solver_specs_from_config(survey)
    except ValueError as error:
        raise CliError(f"[survey] solvers is not valid: {error}", exit_code=2) from None
    _validated(Config({"alignment": alignment_table}), "alignment", AlignmentSettings)

    survey_settings = _merge(REAL_SKY_SURVEY_DEFAULTS, survey_table)
    if not survey.calibration_dir:  # the dark library goes where `core` would put it anyway
        survey_settings["calibration_dir"] = str(data_dir / "calibration")
    core_tables: dict[str, dict[str, Any]] = {"site": site_table, "survey": survey_settings}
    if alignment_table:
        core_tables["alignment"] = alignment_table
    return RealSky(core_tables, tuple(survey.solvers), _solver_warnings(survey))


def _validated(config: Config, name: str, model: type[ModelT]) -> ModelT:
    """Validate one table with the model of its owner, and turn a failure into a `CliError`."""
    try:
        return config.section(name, model)
    except ConfigError as error:
        raise CliError(
            f"your local configuration is not valid for a real-sky run: {error}", exit_code=2
        ) from None


def _finds_program(command: str) -> bool:
    """Whether the program of a solver command is on the PATH or at the path that the command gives.

    The check splits the command as the solver adapters do (`split_command`), so a command that the
    adapter cannot run does not pass here either.
    """
    from seeingmon.solvers.base import SolverError
    from seeingmon.solvers.process import split_command

    try:
        program = split_command(command)[0]
    except (SolverError, ValueError):
        return False
    return shutil.which(program) is not None or Path(program).is_file()


def _split_hint(command: str) -> str:
    """A sentence about how the adapters split a command, for a command that a split gets wrong.

    The adapters split a command like a POSIX shell line. That drops the backslashes of a Windows
    path and cuts a path at its spaces, which is the usual reason why a program that exists is not
    found. A command without a backslash or a space needs no hint.
    """
    if "\\" not in command and " " not in command.strip():
        return ""
    return (
        " The adapters split the command like a shell line, so write a Windows path with forward "
        "slashes, and put a path that contains a space in double quotes."
    )


def _solver_warnings(survey: SurveyConfig) -> list[str]:
    """What the launcher finds wrong with the solvers of `[survey]`: a missing program or folder.

    A warning names the solver and the setting, and never the value of the setting.
    """
    if not survey.solvers:
        return [
            "[survey] solvers is empty, so no plate solver runs and nothing can find the first "
            "pointing solution."
        ]
    found: list[str] = []
    for name in survey.solvers:
        if name == "astrometry.net":
            setting, command = "solve_field_command", survey.solve_field_command
        else:
            setting, command = "astap_command", survey.astap_command
        if not _finds_program(command):
            found.append(
                f"the solver {name} runs the program that [survey] {setting} names, and this "
                f"machine finds no such program on the PATH or at that path.{_split_hint(command)}"
            )
        if name == "astrometry.net":
            folder = Path(survey.index_dir) if survey.index_dir else None
            if folder is None or not any(folder.glob("*.fits")):
                found.append(
                    "the solver astrometry.net needs its index files, and [survey] index_dir names "
                    "no folder that holds *.fits files."
                )
        elif survey.astap_database_dir and not Path(survey.astap_database_dir).is_dir():
            found.append(
                "[survey] astap_database_dir is not a folder that exists, so the solver astap "
                "cannot read its star database."
            )
    return found


def _check_real_sky(options: DevOptions) -> None:
    """Refuse what a real-sky run cannot do, before anything starts. A message names no path."""
    if not options.real_sky:
        return
    if options.acquire_driver != REAL_DRIVER:
        raise CliError(
            "--real-sky needs --driver asi, because a real sky needs the real camera", exit_code=2
        )
    if options.data_dir is None:
        raise CliError(
            "--real-sky needs --data-dir, because a real-sky run keeps its data and its logs in "
            "a folder that survives the run",
            exit_code=2,
        )


def _real_sky_notes(options: DevOptions, solvers: Sequence[str], log_folder: str) -> list[str]:
    """The lines that the banner adds for a real sky: what is real, and what is not."""
    notes = []
    if options.sensor_given:
        notes.append("--sensor does not apply to the real camera, which has the full sensor.")
    notes.append(
        "Real: the camera, the system clock, the star catalog, the plate solvers, and the site "
        "(the last three come from your local configuration). Nothing about the sky is simulated."
    )
    notes.append(
        "No pointing solution is seeded. On a new data folder, the first survey frame goes to the "
        "plate solvers, in this order: "
        f"{', '.join(solvers) if solvers else 'none (the list is empty)'}. A later run starts from "
        "the newest solution in the store."
    )
    notes.append(
        "The scheduler follows the real Sun at your site (by the clock of this machine) and the "
        f"measured sky. {_search_note()} The Align page and a dark session run at any time."
    )
    notes.append(
        "Of your local configuration, only [site], [survey], [alignment], [web], and [auth] reach "
        "the system: no sink, heater, SQM-LE, or power setting does. As in every dev run, the "
        f"windows are {DEV_WINDOW_S:g} s and a dark session takes {DEV_DARK_FRAMES} frames of "
        "each kind."
    )
    notes.append(
        "Real star images ran through the detector, the pointing tracker, and the sky quality "
        "at the first light (October 4, 2026). ASTAP has solved real frames offline only, so "
        "watch the first solve of your run."
    )
    notes.append(f"The logs of the children are in the folder {log_folder} of your data folder.")
    notes.extend(_priority_notes(options))
    notes.append("Cover the camera by hand for a dark session.")
    return notes


def _real_notes(options: DevOptions) -> list[str]:
    """The lines that the banner adds for the real camera."""
    notes = []
    if options.sensor_given:
        notes.append("--sensor does not apply to the real camera, which has the full sensor.")
    notes.append(
        "The sky catalog and the first pointing solution are simulated, so a camera that sees a "
        "room or a dark reports no stars: the seeing windows and the sky quality stay empty."
    )
    notes.append(
        "The scheduler follows the Sun at the synthetic site (by the clock of this machine) and "
        f"the measured sky. {_search_note()} A dark session and the alignment run at any time."
    )
    notes.extend(_priority_notes(options))
    notes.append("Cover the camera by hand for a dark session.")
    return notes


def _search_note(search: SearchConfig | None = None) -> str:
    """What the Sun does to the search for Polaris, with the default settings of the search."""
    from seeingmon.scheduler.activity import duration_text
    from seeingmon.scheduler.config import SearchConfig

    search = SearchConfig() if search is None else search
    note = (
        "It searches for Polaris whenever the sky is not too bright for the camera, and it "
        f"records seeing windows once {search.confirm_bursts} search bursts in a row find the "
        "star."
    )
    if not search.limited:
        return f"{note} The height of the Sun does not limit the search."
    return (
        f"{note} While the Sun is above {search.max_sun_elevation_deg:g} degrees, one probe burst "
        f"every {duration_text(search.probe_interval_s)} looks instead."
    )


def _priority_notes(options: DevOptions) -> list[str]:
    """The line that says that acquire keeps its normal priority, for a comparison run."""
    if options.raise_priority:
        return []
    return [
        "--no-raise-priority: the capture thread of acquire keeps its normal priority, and on "
        "Windows the system timer keeps its default resolution."
    ]


def _merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two settings trees. Tables merge key by key, and any other value replaces."""
    result: dict[str, Any] = {
        k: (dict(v) if isinstance(v, Mapping) else v) for k, v in base.items()
    }
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = dict(value) if isinstance(value, Mapping) else value
    return result


# --- The power request -------------------------------------------------------------------------


if sys.platform == "win32":

    def _load_kernel32() -> Any:
        """The Windows library that holds `SetThreadExecutionState`, with the types of the call."""
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32")  # a private copy, so the argument types stay ours
        kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
        kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
        return kernel32

else:

    def _load_kernel32() -> Any:
        """Another platform has no such library, and needs no request."""
        raise OSError("this platform has no kernel32")


def keeps_awake(options: DevOptions, system: str) -> bool:
    """Whether the run holds the power request: the real camera on Windows, unless you opt out."""
    return options.keep_awake and options.acquire_driver == REAL_DRIVER and system == "win32"


class KeepAwake:
    """A Windows power request that stops idle sleep, held from `request` to `release`.

    `request` calls `SetThreadExecutionState` with `ES_CONTINUOUS | ES_SYSTEM_REQUIRED`, and
    `release` clears the state with `ES_CONTINUOUS` alone. The display may still turn off. Windows
    keeps the state for the thread that sets it, so call both methods from one thread. The state
    also ends with the thread, so a run that dies never leaves it behind.

    No call raises. `kernel32` is the library that holds the function: a test passes a stand-in,
    and without one the first `request` loads the real library through `ctypes`.
    """

    def __init__(self, *, kernel32: Any = None) -> None:
        self._kernel32 = kernel32
        self._held = False

    @property
    def held(self) -> bool:
        """Whether Windows accepted the request, and `release` has not cleared it."""
        return self._held

    def request(self) -> str | None:
        """Ask Windows to stay awake. Returns `None`, or a sentence that says why it could not."""
        try:
            kernel32 = self._kernel32 if self._kernel32 is not None else _load_kernel32()
            previous = kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        except Exception as error:  # a library that does not load, or a call that raises
            return (
                f"the request to keep Windows awake failed with {type(error).__name__}, so the "
                "machine may sleep during the run. Turn off sleep in the power settings."
            )
        if not previous:  # the function returns the previous state, and 0 for a failure
            return (
                "Windows refused the request to stay awake, so the machine may sleep during the "
                "run. Turn off sleep in the power settings."
            )
        self._kernel32 = kernel32
        self._held = True
        return None

    def release(self) -> None:
        """Clear the request. Safe to call twice, and when `request` held nothing."""
        if not self._held:
            return
        self._held = False
        with contextlib.suppress(Exception):  # the thread is ending, and Windows clears it anyway
            self._kernel32.SetThreadExecutionState(ES_CONTINUOUS)


# --- The children ------------------------------------------------------------------------------


class Child:
    """A running child process with a log file."""

    def __init__(self, spec: ChildSpec) -> None:
        self.spec = spec
        self.popen: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None

    def start(self) -> None:
        self._log = self.spec.log.open("ab")
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.popen = subprocess.Popen(
            self.spec.argv,
            env=self.spec.env,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )

    @property
    def running(self) -> bool:
        return self.popen is not None and self.popen.poll() is None

    @property
    def returncode(self) -> int | None:
        return None if self.popen is None else self.popen.poll()

    def stop(self, timeout_s: float = STOP_TIMEOUT_S) -> None:
        """Ask the process to stop as systemd does, and kill it when it does not."""
        popen = self.popen
        if popen is not None and popen.poll() is None:
            try:
                if sys.platform == "win32":
                    os.kill(popen.pid, signal.CTRL_BREAK_EVENT)
                else:
                    popen.send_signal(signal.SIGTERM)
                popen.wait(timeout_s)
            except (OSError, subprocess.TimeoutExpired):
                popen.kill()
                popen.wait(10.0)
        if self._log is not None:
            self._log.close()
            self._log = None

    def log_tail(self, lines: int = 15) -> str:
        try:
            text = self.spec.log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])


def wait_until_ready(
    plan: DevPlan,
    children: Mapping[str, Child],
    timeout_s: float,
    names: Sequence[str] = ("acquire", "core"),
) -> None:
    """Wait until the named children answer a ping. Fails at once when one of them exits."""
    from seeingmon.services.ipc.endpoint import Endpoint
    from seeingmon.services.ipc.errors import IpcError
    from seeingmon.services.ipc.keys import ConnectionKey
    from seeingmon.services.ipc.rpc import connect_rpc

    key = ConnectionKey.from_text(plan.key)
    addresses = {"acquire": plan.acquire_endpoint, "core": plan.core_endpoint}
    deadline_ns = CLOCK.monotonic_ns() + round(timeout_s * NS_PER_S)
    for name in names:
        endpoint = Endpoint.parse(addresses[name])
        while True:
            child = children[name]
            if not child.running:
                raise CliError(f"{name} exited with code {child.returncode}:\n{child.log_tail()}")
            try:
                client, _ = connect_rpc(
                    endpoint, key, {"role": "cli"}, connect_timeout_s=0.5, default_timeout_s=5.0
                )
            except IpcError:
                if CLOCK.monotonic_ns() > deadline_ns:
                    raise CliError(f"{name} did not answer within {timeout_s:g} s") from None
                CLOCK.sleep(0.2)
                continue
            try:
                client.call("ping")
            finally:
                client.close()
            break


def wait_for_web(plan: DevPlan, child: Child, timeout_s: float) -> bool:
    """Wait until the web child accepts a connection. Returns `False` when the child exits."""
    host = connect_address(plan.bind_addresses[0])
    deadline_ns = CLOCK.monotonic_ns() + round(timeout_s * NS_PER_S)
    while CLOCK.monotonic_ns() < deadline_ns:
        if not child.running:
            return False
        try:
            with socket.create_connection((host, plan.port), timeout=0.3):
                return True
        except OSError:
            CLOCK.sleep(0.3)
    return child.running


def banner(plan: DevPlan) -> list[str]:
    """The lines that tell you where the UI is. Nothing here is private except the token."""
    options = plan.options
    if plan.real_sky:
        lines = [
            f"Seeing monitor, real sky ({options.acquire_driver} driver): real time, full sensor."
        ]
    elif plan.real:
        lines = [
            f"Seeing monitor, real camera ({options.acquire_driver} driver): "
            "real time, full sensor."
        ]
    else:
        lines = [
            f"Seeing monitor, simulated sky: {options.speed:g}x speed, {options.sensor} sensor."
        ]
    lines.extend(f"Web UI: {url}" for url in plan.urls())
    lines.extend(plan.notes)
    lines.extend(f"Warning: {text}" for text in plan.warnings)
    if plan.cover_file is not None:
        lines.append(
            f"To cover the simulated camera for a dark session, create the file {plan.cover_file}. "
            "To uncover the camera, delete the file."
        )
    if plan.token is not None:
        lines.append(f"API token for this run (shown once, never stored): {plan.token}")
    lines.append("Press Ctrl+C to stop.")
    return lines


def options_from_args(args: argparse.Namespace) -> DevOptions:
    """The options of a run, from the command line. The data folder becomes an absolute path.

    A real sky logs at the level `info` unless you pass `--log-level`, so that the logs of the
    children show each solver run. Every other run logs at the level `warning`.
    """
    driver = getattr(args, "driver", None) or "sim"
    data_dir = getattr(args, "data_dir", None)
    real_sky = bool(getattr(args, "real_sky", False))
    return DevOptions(
        speed=args.speed,
        port=args.port,
        sensor=args.sensor or "small",
        sensor_given=args.sensor is not None,
        seed=args.seed,
        start=args.start,
        keep_data=args.keep_data,
        log_level=getattr(args, "log_level", None) or ("info" if real_sky else "warning"),
        acquire_driver=driver,
        asi_library=getattr(args, "asi_library", None),
        data_dir=None if not data_dir else Path(data_dir).expanduser().resolve(),
        real_sky=real_sky,
        raise_priority=bool(getattr(args, "raise_priority", True)),
        keep_awake=bool(getattr(args, "keep_awake", True)),
    )


def run_dev(
    args: argparse.Namespace,
    *,
    local_file: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    out: Callable[[str], None] = print,
    web_command: Sequence[str] | None = None,
    wait: Callable[[DevPlan, Mapping[str, Child]], None] | None = None,
    directory: Path | None = None,
    kernel32: Any = None,
    system: str = sys.platform,
) -> int:
    """Run the dev system until you press Ctrl+C. Returns the exit code.

    `wait` replaces the wait for Ctrl+C: a test passes a function that looks at the running system
    and returns. `directory` names the run folder, which a test wants to read, and the run removes
    it at the end unless `--keep-data` is set. A run on the real camera holds a Windows power
    request from the start of the children to the end of the run (`KeepAwake`): `system` is the
    platform, and `kernel32` the library that a test replaces with a stand-in.
    """
    options = options_from_args(args)
    real_sky = options.real_sky
    if options.speed <= 0:
        raise CliError("--speed must be positive", exit_code=2)
    _check_real_sky(options)
    _check_real_camera(options, os.environ if env is None else env)
    run_directory = directory or Path(tempfile.mkdtemp(prefix="smon-dev-"))
    plan: DevPlan | None = None
    children: dict[str, Child] = {}
    awake = KeepAwake(kernel32=kernel32) if keeps_awake(options, system) else None
    try:
        try:
            plan = build_plan(
                options,
                directory=run_directory,
                local_file=local_file,
                env=env,
                web_command=web_command,
            )
        except CliError:
            raise  # a refusal that names the setting to fix
        except Exception as error:
            raise CliError(
                f"cannot prepare the {'real-sky' if real_sky else 'simulated'} run: "
                f"{type(error).__name__}: {error}"
            ) from None
        if awake is not None:  # the banner says whether Windows accepted the request
            problem = awake.request()
            if problem is None:
                plan.notes.append(KEEP_AWAKE_NOTE)
            else:
                plan.warnings.append(problem)
        for spec in plan.children:
            children[spec.name] = Child(spec)
        for name in ("acquire", "core"):  # core connects to acquire, so acquire comes first
            children[name].start()
            wait_until_ready(plan, children, READY_TIMEOUT_S, (name,))
        children["web"].start()  # web starts as soon as core answers, and opens the store itself
        if not wait_for_web(plan, children["web"], WEB_TIMEOUT_S):
            raise CliError(
                f"web exited with code {children['web'].returncode}:\n{children['web'].log_tail()}"
            )
        for line in banner(plan):
            out(line)
        _wait_for_stop(plan, children, wait)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if awake is not None:
            awake.release()  # first, and it never raises, so nothing below can keep the request
        out("Stopping ...")
        for name in ("web", "core", "acquire"):
            if name in children:
                children[name].stop()
        if plan is not None and plan.real_sky:
            # The data and the logs are in your data folder, and the run folder holds nothing.
            out(f"The logs of this run stay in the folder {plan.log_folder} of your data folder.")
            shutil.rmtree(run_directory, ignore_errors=True)
        elif options.keep_data and options.data_dir is not None:
            out(f"The run folder stays: {run_directory}")  # the data is in your folder
        elif options.keep_data:
            out(f"The data folder stays: {run_directory}")
        else:
            shutil.rmtree(run_directory, ignore_errors=True)


def _check_real_camera(options: DevOptions, env: Mapping[str, str]) -> None:
    """Refuse what the real camera cannot do, before anything starts. A message names no path."""
    if options.acquire_driver != REAL_DRIVER:
        if options.asi_library:
            raise CliError("--asi-library needs --driver asi", exit_code=2)
        return
    if options.speed != DEFAULT_SPEED:
        raise CliError(
            f"--speed must be {DEFAULT_SPEED:g} with --driver asi, because the real camera runs "
            "in real time",
            exit_code=2,
        )
    if options.start:
        raise CliError(
            "--start does not apply with --driver asi, because the real camera runs in real time",
            exit_code=2,
        )
    library = options.asi_library or env.get(ASI_LIBRARY_VARIABLE)
    if library and not Path(library).expanduser().is_file():
        raise CliError(
            f"the ASI library file does not exist: check --asi-library and {ASI_LIBRARY_VARIABLE}",
            exit_code=2,
        )


def _wait_for_stop(
    plan: DevPlan,
    children: Mapping[str, Child],
    wait: Callable[[DevPlan, Mapping[str, Child]], None] | None,
) -> None:
    """Block until Ctrl+C, `wait` returns, or a child that matters dies."""
    if wait is not None:
        wait(plan, children)
        return
    while True:
        for name in ("acquire", "core", "web"):
            if not children[name].running:
                child = children[name]
                raise CliError(f"{name} stopped with code {child.returncode}:\n{child.log_tail()}")
        CLOCK.sleep(0.5)
