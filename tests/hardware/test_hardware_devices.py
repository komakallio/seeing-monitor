"""Checks against a GPIO board, an SQM-LE, and a power-cycle route. They skip unless you opt in.

Each check names what it needs. A check that lacks its configuration skips with the reason, so
`pytest --hardware` on a machine with one device runs only that device's checks.

- GPIO loopback: `SEEINGMON_HARDWARE_GPIO_OUT` and `SEEINGMON_HARDWARE_GPIO_IN`, each as
  `chip:line` (for example `gpiochip0:17`), with a jumper wire between the two lines.
- SQM-LE over TCP: `host` in the `[sqm]` section of `local/config.toml`, or `SEEINGMON_SQM__HOST`.
  Set `SEEINGMON_HARDWARE_SQM_DUMP` to a file path to save the raw responses.
- SQM-LE from InfluxDB: `source = "influx"` and the table `[sqm.influx]` in the same file, and
  the environment variable that `token_env` or `password_env` names.
- Power-cycle dry run: a route in the `[power]` section, and any environment variables that it
  names.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from seeingmon.clock import SystemClock
from seeingmon.config import Config, ConfigError
from seeingmon.hardware.io import IoError, LibgpiodIo
from seeingmon.hardware.power import PowerConfig, PowerCycle
from seeingmon.hardware.sqm import SqmConfig, SqmLeClient
from tests.hardware import device_checks as checks

pytestmark = pytest.mark.hardware


def test_a_gpio_loopback_follows_the_output() -> None:
    out_text = os.environ.get("SEEINGMON_HARDWARE_GPIO_OUT")
    in_text = os.environ.get("SEEINGMON_HARDWARE_GPIO_IN")
    if not out_text or not in_text:
        pytest.skip("set SEEINGMON_HARDWARE_GPIO_OUT and SEEINGMON_HARDWARE_GPIO_IN as chip:line")
    pins = {
        "out": checks.pin_from_text(out_text, "output"),
        "in": checks.pin_from_text(in_text, "input"),
    }
    try:
        io = LibgpiodIo(pins)
    except IoError as error:
        pytest.skip(f"GPIO is not available: {error}")
    try:
        print(checks.check_gpio_loopback(io, "out", "in"))
    finally:
        io.close()


def test_the_sqm_le_answers(local_config: Config) -> None:
    config = local_config.section("sqm", SqmConfig)
    if config.source != "tcp":
        pytest.skip('[sqm] source is not "tcp": the next check reads the unit from InfluxDB')
    if not config.host:
        pytest.skip("no SQM-LE host: set it in [sqm] of local/config.toml")
    client = SqmLeClient(
        config.host,
        config.port,
        connect_timeout_s=config.connect_timeout_s,
        read_timeout_s=config.read_timeout_s,
    )
    dump = os.environ.get("SEEINGMON_HARDWARE_SQM_DUMP")
    print(checks.check_sqm_unit(client, dump=Path(dump) if dump else None))


def test_the_sqm_le_readings_are_in_influxdb(local_config: Config) -> None:
    config = local_config.section("sqm", SqmConfig)
    if config.source != "influx" or config.influx is None:
        pytest.skip('[sqm] source is not "influx": set it, and [sqm.influx], in local/config.toml')
    try:
        report = checks.check_sqm_influx(config, clock=SystemClock())
    except ConfigError as error:  # a variable that token_env names is not set, for example
        pytest.skip(f"the settings of [sqm.influx] are incomplete: {error}")
    print(report)


def test_the_power_cycle_dry_run_passes(local_config: Config) -> None:
    config = local_config.section("power", PowerConfig)
    if config.route == "none":
        pytest.skip("no power-cycle route: set one in [power] of local/config.toml")
    rehearsal = config.model_copy(update={"dry_run": True, "state_file": None})
    print(checks.check_power_dry_run(PowerCycle(rehearsal, clock=SystemClock())))
