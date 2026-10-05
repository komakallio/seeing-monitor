"""The commands of the services lane beyond `acquire`: `core`, the commissioning tools, and `dev`.

- `seeingmon core` runs the core process (`seeingmon.services.core.main`).
- `seeingmon heater-off` switches the heater outputs off. The unit of `core` runs it after the
  service stops, whatever the reason that it stopped.
- `seeingmon burst`, `sweep`, and `replay` queue a commissioning task in the running `core`
  through the RPC. They wait for the result and print it. With `--standalone`, they run the task
  in a private scheduler against the configured driver, for bench work (see
  `seeingmon.services.core.commissioning.standalone`).
- `seeingmon dark` (registered with the survey commands) queues a dark session the same way, and
  shows the progress that `core` reports (`run_dark_through_core`). With `--standalone`, it runs
  the session on the camera driver here, and `acquire` must not run.
- `seeingmon dev` starts the whole system on a simulated sky (see `seeingmon.services.dev`).

Exit codes of the commissioning commands: 0 when the task finished with the status `ok`, 1 when it
failed, was aborted, or did not finish in time, and 2 when `core` rejected the command.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seeingmon.cli import CliError, Subparsers, add_command

if TYPE_CHECKING:
    from seeingmon.scheduler.commands import Command
    from seeingmon.services.core.commissioning.client import CoreCommandClient

EXIT_FAILED = 1
EXIT_REJECTED = 2
DEFAULT_WAIT_S = 900.0
LOG_LEVELS = ("debug", "info", "warning", "error")


def _common(parser: argparse.ArgumentParser, *, standalone: bool = True) -> None:
    parser.add_argument(
        "--local-config",
        type=Path,
        help="read this file instead of local/config.toml (an absent file is ignored)",
    )
    parser.add_argument(
        "--address", help="the address of core (default: [services] core_address, or the default)"
    )
    parser.add_argument("--no-wait", action="store_true", help="queue the task and do not wait")
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=DEFAULT_WAIT_S,
        help=f"seconds to wait for the result (default {DEFAULT_WAIT_S:g})",
    )
    parser.add_argument("--priority", type=int, default=0, help="a higher number runs first")
    if standalone:
        parser.add_argument(
            "--standalone",
            action="store_true",
            help="run the task here, with the configured camera driver and without core. "
            "Stop acquire first, because one process may hold the camera.",
        )
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="warning")


def register(subparsers: Subparsers) -> None:
    from seeingmon.services.core.main import local_config_option

    core = add_command(
        subparsers,
        "core",
        help="Run the core process: scheduler, analysis, store, and the commands of web.",
        handler=_core,
    )
    local_config_option(core)
    core.add_argument(
        "--address", help="the address to listen at (default: [services] core_address)"
    )
    core.add_argument("--log-level", choices=LOG_LEVELS, default="info")

    heater = add_command(
        subparsers,
        "heater-off",
        help="Switch the dew-heater outputs off and exit.",
        handler=_heater_off,
    )
    local_config_option(heater)
    heater.add_argument("--log-level", choices=LOG_LEVELS, default="info")

    burst = add_command(
        subparsers,
        "burst",
        help="Record raw frames to a SER file with a JSON sidecar, and pin them.",
        handler=_burst,
    )
    burst.add_argument("--duration", type=float, default=10.0, help="seconds (default 10)")
    burst.add_argument("--label", default="", help="a short label for the folder of the burst")
    burst.add_argument("--exposure-us", type=int, help="stream settings: the exposure")
    burst.add_argument("--mode", help="stream settings: the readout mode (default: the fast mode)")
    burst.add_argument("--gain", type=int, default=0, help="stream settings: the gain")
    burst.add_argument(
        "--roi",
        metavar="X,Y,WIDTH,HEIGHT",
        help="stream settings: the ROI in pixels (default: the full frame)",
    )
    _common(burst)

    sweep = add_command(
        subparsers,
        "sweep",
        help="Run a short fast window for each cell of a grid, and print the table.",
        handler=_sweep,
    )
    sweep.add_argument("--exposure-us", help="exposures in microseconds, separated by commas")
    sweep.add_argument("--gain", help="gains, separated by commas")
    sweep.add_argument("--roi-arcmin", help="ROI sizes in arcminutes, separated by commas")
    sweep.add_argument("--mode", help="readout modes, separated by commas")
    sweep.add_argument("--window-s", type=float, help="seconds of each cell")
    _common(sweep)

    replay = add_command(
        subparsers,
        "replay",
        help="Replay a recording through the production analysis into a separate store.",
        handler=_replay,
    )
    replay.add_argument("source", help="the name of a recording, without a directory part")
    replay.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="a factor of the recorded rate; 0 is as fast as possible",
    )
    replay.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="start_frame, max_frames, mode, exposure_us, gain, or adc_bits. Repeatable.",
    )
    _common(replay)

    dev = add_command(
        subparsers,
        "dev",
        help="Run the whole system on a simulated sky, with one command (acquire, core, and web).",
        handler=_dev,
    )
    dev.description = (
        "Run the whole system on a simulated sky, with one command (acquire, core, and web). "
        "The simulated camera has no lens cap. To take darks from the web UI, create the cover "
        "file that the command prints, and delete the file to uncover the camera. With "
        "--driver asi, the system runs on your ZWO camera in real time instead: the full sensor, "
        "the system clock, and the exposure of the profile. The sky stays simulated, so a camera "
        "that sees a room or a dark reports no stars. Add --real-sky (with --data-dir) for a real "
        "night: the site, the star catalog, and the plate solvers come from your local "
        "configuration, and the first pointing solution comes from the first survey frame."
    )
    dev.add_argument(
        "--driver",
        choices=("sim", "asi"),
        default="sim",
        help="the camera: sim is the simulator (the default), and asi is a connected ZWO camera "
        "that runs in real time, so --speed must be 1 and --start does not apply",
    )
    dev.add_argument(
        "--asi-library",
        metavar="PATH",
        help="the vendor library of --driver asi (default: SEEINGMON_ASI__LIBRARY_PATH in your "
        "environment, then the system search path). The command gives it to acquire in its "
        "environment, and prints no path",
    )
    dev.add_argument(
        "--data-dir",
        metavar="PATH",
        help="keep the store, the dark library, and the images in this folder, which survives "
        "the run (default: a temporary folder that the run removes)",
    )
    dev.add_argument(
        "--real-sky",
        action="store_true",
        help="run on the real sky: no simulated catalog and no seed solution, so core solves the "
        "first survey frame with your plate solvers. It needs --driver asi and --data-dir. The "
        "[site] table (latitude_deg, longitude_deg, and elevation_m), the [survey] table (a "
        "catalog_path that exists, and the solvers), and the [alignment] table come from your "
        "local configuration and the variables SEEINGMON_SITE__*, SEEINGMON_SURVEY__*, and "
        "SEEINGMON_ALIGNMENT__*, and only core receives them. The children log at the level "
        "info into the folder logs of your data folder",
    )
    dev.add_argument(
        "--no-raise-priority",
        dest="raise_priority",
        action="store_false",
        help="with --driver asi, keep the capture thread of acquire at its normal priority, and "
        "on Windows the system timer at its default resolution. The default raises both, so use "
        "this option to compare a run with it. The simulator never raises them",
    )
    dev.add_argument(
        "--no-keep-awake",
        dest="keep_awake",
        action="store_false",
        help="with --driver asi on Windows, do not ask Windows to stay awake. The default holds a "
        "power request for the run that stops idle sleep, and the display may still turn off. "
        "Closing the lid still sleeps the laptop unless the power settings set the lid action to "
        "Do nothing",
    )
    dev.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="simulated seconds per real second (default 1). The simulator renders about 40 "
        "frames a second on one core, so 2 is the most that a quiet machine sustains; faster "
        "clocks drop frames, and the windows say so",
    )
    dev.add_argument(
        "--port", type=int, help="the port of the web UI (default: your [web] port, else 8080)"
    )
    dev.add_argument(
        "--sensor",
        choices=("small", "full"),
        default=None,
        help=(
            "small keeps the simulation fast, but Polaris leaves its field after about two hours "
            "of simulated time. full is the reference sensor, and it keeps Polaris in view for "
            "the whole run (default small). --driver asi ignores it: the real camera has the "
            "full sensor"
        ),
    )
    dev.add_argument("--seed", type=int, default=1, help="the seed of the simulated sky")
    dev.add_argument(
        "--start", help="the UTC start of the simulation, such as 2026-01-01T17:00:00Z"
    )
    dev.add_argument(
        "--keep-data", action="store_true", help="keep the temporary data folder when you stop"
    )
    dev.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=None,
        help="the level of the logs of the children (default warning, and info with --real-sky)",
    )


def _core(args: argparse.Namespace) -> int:
    from seeingmon.services.core.main import run_core

    return run_core(args)


def _heater_off(args: argparse.Namespace) -> int:
    from seeingmon.services.core.main import heater_off

    return heater_off(args)


def _dev(args: argparse.Namespace) -> int:
    from seeingmon.services.dev import run_dev

    return run_dev(args)


# --- The commissioning commands ----------------------------------------------------------------


def _split(text: str | None, convert: Callable[[str], Any], what: str) -> tuple[Any, ...]:
    if not text:
        return ()
    try:
        return tuple(convert(item.strip()) for item in text.split(",") if item.strip())
    except ValueError:
        raise CliError(
            f"{what} must be a list of numbers separated by commas", exit_code=2
        ) from None


def _burst(args: argparse.Namespace) -> int:
    from seeingmon.scheduler.commands import QueueBurst

    stream = None
    if args.exposure_us is not None or args.mode or args.roi:
        if args.exposure_us is None:
            raise CliError("stream settings need --exposure-us", exit_code=2)
        stream = _stream_from(args)
    return _queue(args, QueueBurst(args.duration, stream, args.label, args.priority))


def _stream_from(args: argparse.Namespace) -> Any:
    from seeingmon.config import ConfigError, load_config
    from seeingmon.frames import Roi, StreamConfig

    try:
        profile = load_config(local_file=args.local_config).profile
    except Exception as error:
        raise CliError(f"cannot read the profile: {error}") from None
    roi = None
    if args.roi:
        numbers = _split(args.roi, int, "--roi")
        if len(numbers) != 4:
            raise CliError("--roi wants X,Y,WIDTH,HEIGHT", exit_code=2)
        roi = Roi(*numbers)
    mode = args.mode or profile.fast_mode.mode
    pixel_format = profile.fast_mode.pixel_format
    try:
        kwargs: dict[str, Any] = {} if pixel_format is None else {"pixel_format": pixel_format}
        return StreamConfig(mode, args.exposure_us, args.gain, roi=roi, **kwargs)
    except (ValueError, ConfigError) as error:
        raise CliError(f"the stream settings are not valid: {error}", exit_code=2) from None


def _sweep(args: argparse.Namespace) -> int:
    from seeingmon.scheduler.commands import QueueSweep

    command = QueueSweep(
        exposure_us=_split(args.exposure_us, int, "--exposure-us"),
        gain=_split(args.gain, int, "--gain"),
        roi_arcmin=_split(args.roi_arcmin, float, "--roi-arcmin"),
        modes=tuple(m.strip() for m in (args.mode or "").split(",") if m.strip()),
        window_s=args.window_s,
        priority=args.priority,
    )
    return _queue(args, command)


def _replay(args: argparse.Namespace) -> int:
    from seeingmon.config.layers import parse_env_value
    from seeingmon.scheduler.commands import QueueReplay

    options: dict[str, Any] = {}
    for item in args.option:
        name, separator, text = item.partition("=")
        if not separator or not name.strip():
            raise CliError(f"--option wants KEY=VALUE, not {item!r}", exit_code=2)
        options[name.strip()] = parse_env_value(text)
    return _queue(args, QueueReplay(args.source, args.speed, options, args.priority))


def _queue(args: argparse.Namespace, command: Command) -> int:
    from seeingmon.config import ConfigError, load_config
    from seeingmon.services.config import ServicesConfig
    from seeingmon.services.core.main import setup_logging

    setup_logging(args.log_level)
    try:
        config = load_config(local_file=args.local_config)
        services = config.section("services", ServicesConfig)
    except (ConfigError, ValueError) as error:
        raise CliError(str(error)) from None
    if args.standalone:
        return _run_here(args, config, services, command)
    return _run_through_core(args, services, command)


def _connect(args: argparse.Namespace, services: Any) -> CoreCommandClient:
    """Connect to `core` at `--address`, or at the address of the configuration."""
    from seeingmon.services.core.commissioning.client import CoreCommandClient, CoreCommandError
    from seeingmon.services.ipc.endpoint import Endpoint
    from seeingmon.services.ipc.errors import IpcError

    try:
        endpoint = Endpoint.parse(args.address) if args.address else services.endpoint("core")
        key = services.load_key()
    except IpcError as error:
        raise CliError(str(error)) from None
    try:
        return CoreCommandClient(endpoint, key, connect_timeout_s=services.connect_timeout_s)
    except CoreCommandError as error:
        raise CliError(
            f"{error}. Start core with `seeingmon core`, or add --standalone to run on this "
            "machine without it."
        ) from None


def _run_through_core(args: argparse.Namespace, services: Any, command: Command) -> int:
    from seeingmon.services.core.commissioning.client import CoreCommandError

    client = _connect(args, services)
    try:
        answer = client.submit(command)
        if not answer.accepted:
            print(f"core rejected the command: {answer.message}")
            return EXIT_REJECTED
        print(answer.message)
        if args.no_wait or answer.task_id is None:
            return 0
        print(f"waiting for task {answer.task_id}, which runs at the next cycle boundary ...")
        outcome = client.wait_for(answer.task_id, timeout_s=args.wait_timeout)
    except CoreCommandError as error:
        raise CliError(str(error)) from None
    finally:
        client.close()
    if outcome.result is None:
        print(f"the task did not finish within {args.wait_timeout:g} s; it still runs in core")
        return EXIT_FAILED
    return _print_result(outcome.result)


def _run_here(args: argparse.Namespace, config: Any, services: Any, command: Command) -> int:
    import threading

    from seeingmon.services.core.commissioning.standalone import run_standalone

    stop = threading.Event()
    try:
        outcome = run_standalone(
            config,
            services,
            command,
            clock=services.clock.build(),
            show=lambda line: print(line, flush=True),
            timeout_s=args.wait_timeout,
            should_stop=stop.is_set,
        )
    except KeyboardInterrupt:
        stop.set()
        raise CliError("interrupted") from None
    except Exception as error:
        raise CliError(f"the standalone run failed: {type(error).__name__}: {error}") from None
    if not outcome.answer.accepted:
        print(f"the scheduler rejected the command: {outcome.answer.message}")
        return EXIT_REJECTED
    if outcome.result is None:
        print("the task did not finish")
        return EXIT_FAILED
    return _print_result(outcome.result.to_detail())


# The options of `seeingmon dark` that the run through `core` cannot take, because `core` has its
# own camera, its own data folder, and its own readout mode and gain ([survey.dark]).
DARK_STANDALONE_OPTIONS = (
    ("--driver", "driver"),
    ("--library", "library"),
    ("--mode", "mode"),
    ("--gain", "gain"),
)


def run_dark_through_core(args: argparse.Namespace) -> int:
    """`seeingmon dark` without `--standalone`: queue the session in `core`, and follow it.

    The command prints the progress that `core` reports, and then the result. It ends with the exit
    codes of the other commissioning commands. Ctrl+C stops the display, and the session goes on.
    """
    from seeingmon.config import ConfigError, load_config
    from seeingmon.scheduler.commands import QueueDark
    from seeingmon.services.config import ServicesConfig
    from seeingmon.services.core.commissioning.client import CoreCommandError

    for flag, name in DARK_STANDALONE_OPTIONS:
        if getattr(args, name, None) is not None:
            raise CliError(
                f"{flag} needs --standalone, because core runs the session with its own camera, "
                "library, readout mode, and gain",
                exit_code=2,
            )
    try:
        config = load_config(local_file=args.local_config)
        services = config.section("services", ServicesConfig)
    except (ConfigError, ValueError) as error:
        raise CliError(str(error)) from None
    command = QueueDark(
        exposure_s=args.exposure_s,
        frames=args.frames,
        bias_frames=args.bias_frames,
        wait_for_cover=not args.no_wait,
        wait_for_cover_timeout_s=args.wait_timeout,
    )
    client = _connect(args, services)
    try:
        answer = client.submit(command)
        if not answer.accepted:
            print(f"core rejected the command: {answer.message}")
            return EXIT_REJECTED
        print(answer.message, flush=True)
        if args.detach or answer.task_id is None:
            return 0
        print(
            f"following task {answer.task_id}. "
            "Press Ctrl+C to stop following: the session goes on in core.",
            flush=True,
        )
        outcome = client.follow_dark(answer.task_id, show=lambda line: print(line, flush=True))
    except CoreCommandError as error:
        raise CliError(str(error)) from None
    except KeyboardInterrupt:
        print("stopped following; the session goes on in core (Pause in the web UI stops it)")
        return EXIT_FAILED
    finally:
        client.close()
    if outcome.result is None:
        print("the session did not finish; it still runs in core")
        return EXIT_FAILED
    code = _print_result(outcome.result)
    data = outcome.result.get("data") or {}
    if data.get("remove_cover"):
        print("The scheduler waits in pause. Uncover the camera, then resume it from the web UI.")
    return code


def _print_result(result: Any) -> int:
    from seeingmon.scheduler.commission import format_sweep_table
    from seeingmon.services.core.commissioning.client import cells_from_result

    status = str(result.get("status"))
    print(f"{result.get('kind')} {result.get('task_id')}: {status}. {result.get('summary')}")
    data = result.get("data") or {}
    if result.get("kind") == "sweep":
        print(format_sweep_table(cells_from_result(data)))
    for artifact in result.get("artifacts") or ():
        print(f"  file: {artifact}")
    return 0 if status == "ok" else EXIT_FAILED
