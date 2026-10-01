"""The entry of `seeingmon core`: load the configuration, build `CoreApp`, and run it.

The function reads `[services]` and the profile, finds the connection key, builds the clock, and
runs the app until a signal or a fatal error. It turns every failure of the start into a message
and a non-zero exit code, so that systemd restarts the unit and the journal says why.

Exit codes of `seeingmon core`:

- 0: a clean stop (SIGTERM, SIGINT, or SIGBREAK).
- 1: the start failed: an invalid configuration, no connection key, a catalog that is not set, a
  store that cannot open, or the address of `core` already in use.
- 71: a thread that `core` cannot run without died, so the process ended and systemd restarts it.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from seeingmon.cli import CliError

_log = logging.getLogger(__name__)

SIGNALS = ("SIGINT", "SIGTERM", "SIGBREAK")  # SIGBREAK exists on Windows only


def setup_logging(level: str) -> None:
    """Send the log of a process to stderr, which the journal collects."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_core(args: argparse.Namespace) -> int:
    """Run `core` until it stops. Returns the exit code of the process."""
    from seeingmon.config import ConfigError, load_config
    from seeingmon.profile import ProfileError
    from seeingmon.services.config import ServicesConfig
    from seeingmon.services.core.app import CoreApp
    from seeingmon.services.ipc.endpoint import Endpoint
    from seeingmon.services.ipc.errors import IpcAddressInUseError, IpcError
    from seeingmon.store.db import StoreError

    setup_logging(args.log_level)
    try:
        config = load_config(local_file=getattr(args, "local_config", None))
        services = config.section("services", ServicesConfig)
        key = services.load_key()
        endpoint = (
            Endpoint.parse(args.address)
            if getattr(args, "address", None)
            else services.endpoint("core")
        )
        clock = services.clock.build()
        app = CoreApp(config, services, clock, key, endpoint=endpoint)
    except (ConfigError, ProfileError, IpcError, StoreError, ValueError, OSError) as error:
        _log.debug("core could not start", exc_info=True)
        raise CliError(f"core cannot start: {error}") from None

    def request_stop(signum: int, frame: object) -> None:
        app.request_stop(f"signal {signum}")

    for name in SIGNALS:
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, request_stop)
    try:
        return app.run()
    except IpcAddressInUseError as error:
        raise CliError(f"core cannot listen: {error}") from None
    except IpcError as error:
        raise CliError(f"core cannot listen: {error}") from None


def heater_off(args: argparse.Namespace) -> int:
    """Switch the heater outputs off and exit. The unit of `core` runs it after core stops.

    The work lives in `seeingmon.hardware.heater_off`: it reads only `[heater]`, switches off each
    output line by itself, names a failed output by its logical name, and ends within a time
    limit even when the GPIO library hangs. Exit code 0 means that the outputs are off or that the
    heater is not enabled, and 1 means that an output could not be switched off. With
    `--log-level debug`, the traceback of an unexpected error goes to the log.
    """
    from seeingmon.hardware.heater_off import run_heater_off

    setup_logging(getattr(args, "log_level", "info"))
    return run_heater_off(local_config=getattr(args, "local_config", None))


def local_config_option(parser: argparse.ArgumentParser) -> None:
    """Add `--local-config` to a parser."""
    parser.add_argument(
        "--local-config",
        type=Path,
        help="read this file instead of local/config.toml (an absent file is ignored)",
    )
