"""The commands that start the three processes: `seeingmon acquire`, `core`, and `web`.

`seeingmon acquire` runs the process that owns the camera driver (see
`seeingmon.services.acquire`). Under systemd, its unit runs this command with `Type=notify`.
`seeingmon core` and the commissioning commands are in `seeingmon.services.commands`.
`seeingmon web` lives in `seeingmon.services.web.cli`, which this module registers.

`acquire` reads the `[services]` section of the configuration (see
`seeingmon.services.config`). The options override a few values for a single run:

    seeingmon acquire --driver fake --driver-option temperature_c=12.5

The connection key comes from the configuration, an environment variable, or a systemd
credential, and never from the command line, where `ps` would show it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from seeingmon.cli import CliError, Subparsers, add_command
from seeingmon.services.web.cli import register as register_web

LOG_LEVELS = ("debug", "info", "warning", "error")


def register(subparsers: Subparsers) -> None:
    acquire = add_command(
        subparsers,
        "acquire",
        help="Run the acquire process: it owns the camera driver and streams frames to core.",
        handler=_acquire,
    )
    acquire.add_argument("--driver", help="the camera driver (default: [services.acquire] driver)")
    acquire.add_argument(
        "--driver-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="an option of the driver. A number, true, or false parses as such. Repeatable.",
    )
    acquire.add_argument(
        "--address",
        help="the socket path (Linux) or pipe name (Windows) to listen at "
        "(default: [services] acquire_address, or the platform default)",
    )
    acquire.add_argument(
        "--local-config",
        type=Path,
        help="read this file instead of local/config.toml (an absent file is ignored)",
    )
    acquire.add_argument("--log-level", choices=LOG_LEVELS, default="info")
    from seeingmon.services import commands

    commands.register(subparsers)
    register_web(subparsers)


def _parse_options(items: list[str]) -> dict[str, Any]:
    from seeingmon.config.layers import parse_env_value

    options: dict[str, Any] = {}
    for item in items:
        name, separator, text = item.partition("=")
        if not separator or not name.strip():
            raise CliError(f"--driver-option wants KEY=VALUE, not {item!r}", exit_code=2)
        options[name.strip()] = parse_env_value(text)
    return options


def _acquire(args: argparse.Namespace) -> int:
    import logging
    import signal

    from seeingmon.config import ConfigError, load_config
    from seeingmon.profile import ProfileError
    from seeingmon.services.acquire.events import HardwareEventLog
    from seeingmon.services.acquire.factory import create_camera_driver
    from seeingmon.services.acquire.service import AcquireService
    from seeingmon.services.config import ServicesConfig
    from seeingmon.services.ipc.endpoint import Endpoint
    from seeingmon.services.ipc.errors import IpcError

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    options = _parse_options(args.driver_option)
    try:
        config = load_config(local_file=args.local_config)
        services = config.section("services", ServicesConfig)
        key = services.load_key()
        endpoint = Endpoint.parse(args.address) if args.address else services.endpoint("acquire")
        profile = config.profile
    except (ConfigError, ProfileError, IpcError) as error:
        raise CliError(str(error)) from None
    clock = services.clock.build()
    events = HardwareEventLog()
    name = args.driver or services.acquire.driver
    try:
        driver = create_camera_driver(
            name,
            profile=profile,
            clock=clock,
            options={**services.acquire.driver_options, **options},
            on_event=events.record,
        )
    except Exception as error:  # a driver can fail in its own ways, such as a missing library
        logging.getLogger(__name__).debug("the driver failed", exc_info=True)
        raise CliError(
            f"cannot create the driver {name!r}: {type(error).__name__}: {error}"
        ) from None
    service = AcquireService(driver, clock, endpoint, key, services, events=events)

    def request_stop(signum: int, frame: object) -> None:
        service.request_stop(f"signal {signum}")

    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        number = getattr(signal, signal_name, None)  # SIGBREAK exists on Windows only
        if number is not None:
            signal.signal(number, request_stop)
    try:
        return service.run()
    except IpcError as error:
        raise CliError(str(error)) from None
