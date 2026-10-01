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
    """Switch the heater outputs off and exit. The unit of `core` runs it after core stops."""
    from seeingmon.clock import SystemClock
    from seeingmon.config import ConfigError, load_config
    from seeingmon.hardware.heater import HeaterConfig, create_heater

    setup_logging(getattr(args, "log_level", "info"))
    try:
        config = load_config(local_file=getattr(args, "local_config", None))
        heater_config = config.section("heater", HeaterConfig)
    except ConfigError as error:
        raise CliError(f"cannot read the heater configuration: {error}") from None
    if not heater_config.enabled:
        print("the heater is not enabled, so there is nothing to switch off")
        return 0
    try:
        controller = create_heater(heater_config, clock=SystemClock())
        controller.close()  # the constructor switches the output off, and close releases the lines
    except Exception as error:
        _log.debug("the heater could not be switched off", exc_info=True)
        raise CliError(f"cannot switch the heater off: {type(error).__name__}: {error}") from None
    print("the heater outputs are off")
    return 0


def local_config_option(parser: argparse.ArgumentParser) -> None:
    """Add `--local-config` to a parser."""
    parser.add_argument(
        "--local-config",
        type=Path,
        help="read this file instead of local/config.toml (an absent file is ignored)",
    )
