"""`seeingmon dev`: the whole system on a simulated sky, started with one command.

The launcher starts `acquire` with the `sim` driver, `core`, and `web` as child processes of this
one. They talk over local sockets of a private temporary folder, keep their data in that folder,
and share a scaled clock, so a simulated evening passes in minutes. The launcher prints the URL of
the web UI, and stops every child when you press Ctrl+C.

**What a simulated run is.** The star field of the simulator, the cap catalog, and the first
pointing solution come from one seed (`seeingmon.services.simsky`), so `core` follows the simulated
Polaris from the first second with no plate solver. The sensor is small by default (`--sensor full`
makes the reference sensor). With the small sensor, Polaris leaves the field after about two
hours of simulated time, because it drifts about 2.5 pixels a minute across 640 by 480 pixels, so a
long run needs `--sensor full`. The fast stream uses a longer exposure and a dimmer Polaris than a
real night, because a scaled clock multiplies the frame rate that the machine must sustain.

**The cover.** A simulated camera has no lens cap, so a dark session needs a stand-in. The
launcher gives the `sim` driver a cover file, named `cover` in the run folder. The camera shows
the sensor alone while that file exists. To take darks from the web UI, create the file (the
launcher prints its path), start the dark session on the Dark page, and delete the file when the
session says that the cover can come off. The simulator looks at the file twice a second. A dev
run also shortens the dark session to a few frames of each kind.

**Isolation.** A simulated run must never reach a real sink, device, or data directory. Each child
gets a clean environment: no `SEEINGMON_*` variable of yours reaches it, and `--local-config`
points the children that take it at a file that does not exist, so `local/config.toml` is not
read. The launcher sets every key that matters (the data folder, the profile, the survey catalog,
the services, and the clock) in the environment of the child.

**The web settings are yours.** The `web` child is the only process that sees your `[web]` and
`[auth]` sections. The launcher layers `local/config.toml` and the variables `SEEINGMON_WEB__*`
and `SEEINGMON_AUTH__*`, and ignores every other key and variable. The address and the port come
from your `[web]` section, and default to the loopback interface and port 8080 when you set none.
The launcher prints one URL for each bind address (`bind_address` and `extra_bind_addresses`), and
writes no address to a file or a log. It passes the settings to the child in its environment, and
never on the command line.

**The API token.** When your `[auth]` section has no token hash, the launcher makes a random token
for the run, gives its hash (the one that `seeingmon web hash-token` makes) to the `web` child, and
prints the token once. The token goes to no file.

The children start with `sys.executable -m seeingmon`, so the launcher works on Windows. On
Windows each child has its own process group, and the launcher stops it with a console break.
"""

from __future__ import annotations

import argparse
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
from typing import IO, Any

from seeingmon.cli import CliError
from seeingmon.clock import NS_PER_S, SystemClock

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

    def urls(self) -> list[str]:
        """One URL for each bind address, with an IPv6 address in brackets."""
        return [url_for(address, self.port) for address in self.bind_addresses]


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


def owner_settings(
    local_file: Path | str | None, env: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The `[web]` and `[auth]` tables that the owner set, and nothing else.

    The layers are the local configuration file and the variables `SEEINGMON_WEB__*` and
    `SEEINGMON_AUTH__*`. The defaults of the repository are left out, so the result holds only
    what the owner chose. Every other key of the file and every other variable is ignored.
    """
    from seeingmon.config import load_config

    kept = {k: v for k, v in env.items() if k.startswith(("SEEINGMON_WEB__", "SEEINGMON_AUTH__"))}
    with tempfile.TemporaryDirectory(prefix="smon-cfg-") as empty:
        Path(empty, "default.toml").write_text('profile = "none"\nstation_id = "none"\n')
        config = load_config(config_dir=empty, local_file=local_file, env=kept)
    data = config.effective(redact=False)
    web, auth = data.get("web", {}), data.get("auth", {})
    return (dict(web) if isinstance(web, dict) else {}), (
        dict(auth) if isinstance(auth, dict) else {}
    )


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

    The scheduler waits for the Sun to sink below its gate, and the windows of an earlier hour carry
    the `twilight` flag. Starting in the dark gives windows at once, without twilight flags. Pass
    `--start` to see the evening instead.
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
    """Decide the settings of every child, and write the files that they need to `directory`."""
    from seeingmon.clock import iso_to_utc_ns
    from seeingmon.drivers.sim.stars import Pointing
    from seeingmon.profile import load_profile
    from seeingmon.services.simsky import (
        seed_solution,
        sim_catalog,
        write_seed,
        write_small_profile,
    )
    from seeingmon.services.web.auth import hash_token
    from seeingmon.survey.catalog import write_catalog

    parent_env = os.environ if env is None else env
    interpreter = python or sys.executable
    token = uuid.uuid4().hex[:12]
    key = secrets.token_urlsafe(32)
    directory.mkdir(parents=True, exist_ok=True)
    start_utc_ns = iso_to_utc_ns(options.start) if options.start else default_start_utc_ns()
    origin_ns = CLOCK.utc_ns() if origin_real_ns is None else origin_real_ns

    # The simulated sky: a profile, a catalog, a first solution, and the pointing of the camera.
    if options.sensor == "small":
        profile_path: Path | str = write_small_profile(directory)
    else:
        profile_path = "asi294mm-gs250"
    profile = load_profile(str(profile_path))
    catalog, field_ = sim_catalog(options.seed, polaris_mag=options.polaris_mag)
    catalog_path = directory / "catalog.bin"
    write_catalog(catalog_path, catalog)
    pointing = Pointing(t_ref_utc_ns=start_utc_ns)
    seed_path = directory / "seed.json"
    write_seed(seed_path, seed_solution(profile, field_, pointing, start_utc_ns).solution)
    nowhere = directory / "no-local-config.toml"  # does not exist, so no local file is read

    slow_reads: dict[str, Any] = (
        {} if options.sensor == "small" else {"read_timeout_margin_s": FULL_SENSOR_READ_MARGIN_S}
    )
    acquire_address = _endpoint_text(directory, "acquire", token)
    core_address = _endpoint_text(directory, "core", token)
    cover_file = directory / COVER_FILE_NAME if options.acquire_driver == "sim" else None
    shared: dict[str, Any] = {
        "station_id": "dev",
        "profile": str(profile_path),
        "paths": {"data_dir": str(directory / "data")},
        "services": {
            "connection_key": key,
            "acquire_address": acquire_address,
            "core_address": core_address,
            "clock": {
                "kind": "scaled",
                "speed": options.speed,
                "start_utc_ns": start_utc_ns,
                "origin_real_ns": origin_ns,
            },
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
    acquire_settings = _merge(
        _merge(
            shared,
            {
                "services": {
                    "acquire": {
                        "driver": options.acquire_driver,
                        "gap_factor": 1000.0,
                        "raise_priority": False,
                        "driver_options": sim_options if options.acquire_driver == "sim" else {},
                        **slow_reads,
                    }
                }
            },
        ),
        options.acquire_overrides,
    )
    core_settings = _merge(
        _merge(
            shared,
            {
                "site": {
                    "latitude_deg": SIM_LATITUDE_DEG,
                    "longitude_deg": SIM_LONGITUDE_DEG,
                    "elevation_m": 0.0,
                },
                # Windows of 20 s, so that the UI has data half a minute after the start.
                "fastpath": {"window_s": options.window_s, "min_window_s": options.window_s / 2},
                "scheduler": {
                    "fast": {
                        "exposure_us": options.fast_exposure_us,
                        "analysis_window_s": options.window_s,
                        "window_s": options.window_s * 3,
                    },
                    "loop": dict(slow_reads),
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
            },
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

    def spec(name: str, argv: list[str], settings: Mapping[str, Any]) -> ChildSpec:
        env = dict(base)
        top = {k: v for k, v in settings.items() if not isinstance(v, Mapping)}
        env.update({f"SEEINGMON_{k.upper()}": render_env_value(v) for k, v in top.items()})
        for section, table in settings.items():
            if isinstance(table, Mapping):
                env.update(flatten_env(table, f"SEEINGMON_{section.upper()}"))
        return ChildSpec(name, argv, env, directory / f"{name}.log")

    level = ["--log-level", options.log_level]
    children = [
        spec(
            "acquire",
            [*base_args, "acquire", "--local-config", str(nowhere), *level],
            acquire_settings,
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
    )


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
    host = plan.bind_addresses[0]
    host = LOOPBACK if host == "localhost" else host
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
    lines = [f"Seeing monitor, simulated sky: {options.speed:g}x speed, {options.sensor} sensor."]
    lines.extend(f"Web UI: {url}" for url in plan.urls())
    if plan.cover_file is not None:
        lines.append(
            f"To cover the simulated camera for a dark session, create the file {plan.cover_file}. "
            "To uncover the camera, delete the file."
        )
    if plan.token is not None:
        lines.append(f"API token for this run (shown once, never stored): {plan.token}")
    lines.append("Press Ctrl+C to stop.")
    return lines


def run_dev(
    args: argparse.Namespace,
    *,
    local_file: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    out: Callable[[str], None] = print,
    web_command: Sequence[str] | None = None,
    wait: Callable[[DevPlan, Mapping[str, Child]], None] | None = None,
    directory: Path | None = None,
) -> int:
    """Run the dev system until you press Ctrl+C. Returns the exit code.

    `wait` replaces the wait for Ctrl+C: a test passes a function that looks at the running system
    and returns. `directory` names the run folder, which a test wants to read, and the run removes
    it at the end unless `--keep-data` is set.
    """
    options = DevOptions(
        speed=args.speed,
        port=args.port,
        sensor=args.sensor,
        seed=args.seed,
        start=args.start,
        keep_data=args.keep_data,
        log_level=args.log_level,
    )
    if options.speed <= 0:
        raise CliError("--speed must be positive", exit_code=2)
    run_directory = directory or Path(tempfile.mkdtemp(prefix="smon-dev-"))
    children: dict[str, Child] = {}
    try:
        try:
            plan = build_plan(
                options,
                directory=run_directory,
                local_file=local_file,
                env=env,
                web_command=web_command,
            )
        except Exception as error:
            raise CliError(
                f"cannot prepare the simulated run: {type(error).__name__}: {error}"
            ) from None
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
        out("Stopping ...")
        for name in ("web", "core", "acquire"):
            if name in children:
                children[name].stop()
        if options.keep_data:
            out(f"The data folder stays: {run_directory}")
        else:
            shutil.rmtree(run_directory, ignore_errors=True)


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
