"""The `seeingmon camera` and `seeingmon hardware` commands.

`seeingmon camera rates` measures the frame rates of the connected camera in a table: the exposure,
the ROI size, the pixel format, the USB bandwidth control, and the high-speed mode, one factor at a
time around the fast stream of the profile, and the second readout mode. It prints the measured rate
beside the rate of the profile's model and the fitted timing, and it can write the table as JSON
(see `seeingmon.hardware.rates`). The command puts every control that it changed back, so another
program that shares the camera finds it as it left it.

`seeingmon hardware sqm` reads the SQM-LE one time from the source that `[sqm]` names (the unit over
TCP, or the readings in InfluxDB), and it prints the magnitude, the temperature, and the age of the
reading. Use it to try the settings of `[sqm]` and `[sqm.influx]` (see `seeingmon.hardware.sqm`).

The handlers import the implementation on demand, so `seeingmon --help` stays fast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from seeingmon.cli import CliError, Subparsers, add_command

DEFAULT_JSON = Path("local") / "camera-rates.json"
DEFAULT_FRAMES = 150
DEFAULT_SETTLE = 10


def register(subparsers: Subparsers) -> None:
    camera = add_command(
        subparsers,
        "camera",
        help="Measure the connected camera.",
        handler=_missing_subcommand,
    )
    commands = camera.add_subparsers(dest="camera_command", metavar="<subcommand>", required=True)
    rates = commands.add_parser(
        "rates",
        help="Measure the frame rates in a table, and compare them with the profile.",
        description=(
            "Measure the frame rates of the connected camera, one factor at a time around the "
            "fast stream of the profile: the exposure, the ROI size, the pixel format, the USB "
            "bandwidth, and the high-speed mode, and then the second readout mode. Print the "
            "measured rate, the rate of the profile's model, the median, the jitter, and the "
            "maximum of the frame periods, the dropped frames, and the ADC depth. Fit the frame "
            "overhead and the row time of the profile to the rows at bandwidth 100. The command "
            "puts every control that it changed back, and it closes the camera, even when a row "
            "fails. Close other camera programs first. The command reads the driver options from "
            "[services.acquire.driver_options] of the configuration (the library path)."
        ),
    )
    rates.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"frames to measure in each row (default {DEFAULT_FRAMES})",
    )
    rates.add_argument(
        "--settle",
        type=int,
        default=DEFAULT_SETTLE,
        help=f"frames to read and drop before each measurement (default {DEFAULT_SETTLE})",
    )
    rates.add_argument("--gain", type=int, default=120, help="the gain of every row (default 120)")
    rates.add_argument(
        "--groups",
        help="the row groups to run, separated by commas, besides the baseline: "
        "exposure, roi, format, bandwidth, speed, and bin2 (default: all)",
    )
    rates.add_argument(
        "--json",
        nargs="?",
        const=DEFAULT_JSON,
        default=None,
        type=Path,
        metavar="PATH",
        help=f"also write the table as JSON (default path: {DEFAULT_JSON.as_posix()})",
    )
    rates.add_argument(
        "--local-config",
        type=Path,
        help="read this file instead of local/config.toml (an absent file is ignored)",
    )
    rates.set_defaults(handler=_rates)

    hardware = add_command(
        subparsers,
        "hardware",
        help="Try the readers of this machine, such as the SQM-LE.",
        handler=_missing_hardware_subcommand,
    )
    hardware_commands = hardware.add_subparsers(
        dest="hardware_command", metavar="<subcommand>", required=True
    )
    sqm = hardware_commands.add_parser(
        "sqm",
        help="Read the SQM-LE once from the source that [sqm] names, and print the reading.",
        description=(
            "Read the SQM-LE one time from the source that [sqm] names: the unit over TCP "
            '(source = "tcp") or the readings in InfluxDB (source = "influx", with the table '
            "[sqm.influx]). Print the magnitude, the temperature, and the age of the reading. "
            "The command ignores enabled, so you can try the settings before you turn the reader "
            "on. It prints no address and no name from the configuration. It exits with 0 when it "
            "has a reading, and with 1 and one line of text when the read fails or a setting is "
            "missing. A secret that token_env or password_env names must be in the environment."
        ),
    )
    sqm.add_argument(
        "--local-config",
        type=Path,
        help="read this file instead of local/config.toml (an absent file is ignored)",
    )
    sqm.set_defaults(handler=_sqm)


def _missing_subcommand(args: argparse.Namespace) -> int:
    # `required=True` makes argparse reject a missing subcommand before this runs.
    raise CliError("choose a subcommand: rates", exit_code=2)


def _missing_hardware_subcommand(args: argparse.Namespace) -> int:
    raise CliError("choose a subcommand: sqm", exit_code=2)


def _sqm(args: argparse.Namespace) -> int:
    from seeingmon.clock import SystemClock
    from seeingmon.config import ConfigError, load_config
    from seeingmon.hardware.sqm import SqmConfig, SqmError
    from seeingmon.hardware.sqm_factory import read_sqm_once

    try:
        config = load_config(local_file=args.local_config)
        sample = read_sqm_once(config.section("sqm", SqmConfig), clock=SystemClock())
    except (ConfigError, SqmError) as error:
        raise CliError(str(error)) from None
    print(sample.summary())
    return 0


def _rates(args: argparse.Namespace) -> int:
    from seeingmon.clock import SystemClock
    from seeingmon.config import ConfigError, load_config
    from seeingmon.drivers.base import CameraError
    from seeingmon.hardware import rates
    from seeingmon.profile import ProfileError

    if args.frames < rates.MIN_FRAMES:
        raise CliError(f"--frames must be at least {rates.MIN_FRAMES}", exit_code=2)
    if args.settle < 0:
        raise CliError("--settle must not be negative", exit_code=2)
    groups = (
        list(rates.GROUPS)
        if not args.groups
        else [name.strip() for name in args.groups.split(",") if name.strip()]
    )
    unknown = [name for name in groups if name not in rates.GROUPS]
    if unknown:
        raise CliError(
            f"unknown row group {unknown[0]!r}; the groups are {', '.join(rates.GROUPS)}",
            exit_code=2,
        )
    try:
        config = load_config(local_file=args.local_config)
        profile = config.profile
        clock = SystemClock()
        driver = rates.create_driver(profile, rates.driver_options(config), clock)
    except (ConfigError, ProfileError, CameraError) as error:
        raise CliError(str(error)) from None

    def on_start(conditions: dict[str, object]) -> None:
        print(rates.format_title(conditions))
        print()
        print(rates.format_header(), flush=True)

    def on_row(row: rates.RateRow) -> None:
        print(rates.format_row(row), flush=True)

    try:
        report = rates.run_table(
            driver,
            profile,
            frames=args.frames,
            settle=args.settle,
            gain=args.gain,
            groups=groups,
            on_start=on_start,
            on_row=on_row,
            clock=clock,
        )
    except CameraError as error:
        raise CliError(f"cannot measure the camera: {error}") from None
    print()
    print(rates.format_footer(report))
    if args.json is not None:
        try:
            rates.write_json(report, args.json)
        except OSError as error:
            # The table is on the screen already, so say where the file failed and go on.
            raise CliError(
                f"cannot write {args.json.as_posix()}: {error.strerror or error}; "
                "pass a folder that you can write with --json PATH"
            ) from None
        print(f"The table is in {args.json.as_posix()}.")
    return 1 if report.failed else 0
