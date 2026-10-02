"""The GPIO, SQM-LE, and power-cycle checks that run on real hardware, as functions.

Like `camera_checks`, each function asserts what a working device must do and returns a short
report. `test_device_checks_on_fakes` runs them against fakes, so their code is tested first.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Mapping
from pathlib import Path

from seeingmon.clock import Clock
from seeingmon.hardware.io import Io, PinSpec
from seeingmon.hardware.power import PowerCycle, PowerOutcome
from seeingmon.hardware.sqm import (
    SqmConfig,
    SqmLeClient,
    parse_calibration,
    parse_info,
    parse_reading,
)
from seeingmon.hardware.sqm_factory import read_sqm_once

TOGGLES = 5


def pin_from_text(text: str, direction: str) -> PinSpec:
    """Read a pin from `chip:line`, such as `gpiochip0:17`."""
    chip, separator, line = text.rpartition(":")
    if not separator or not chip or not line.isdigit():
        raise ValueError("write a pin as chip:line, for example gpiochip0:17")
    return PinSpec(chip=chip, line=int(line), direction=direction)


def check_gpio_loopback(io: Io, output: str, input_name: str) -> str:
    """The input follows the output, which a jumper wire joins, in both directions."""
    io.set_output(output, False)
    assert io.read_input(input_name) is False, "the input reads high while the output is off"
    for _ in range(TOGGLES):
        io.set_output(output, True)
        assert io.read_input(input_name) is True, "the input does not follow the output going on"
        io.set_output(output, False)
        assert io.read_input(input_name) is False, "the input does not follow the output going off"
    return f"the input followed the output through {TOGGLES} on and off switches"


def check_sqm_unit(client: SqmLeClient, *, dump: Path | None = None) -> str:
    """The unit answers the information, reading, and calibration requests, and the parser reads
    the answers. With `dump`, the raw responses go to that file for the maintainer (blocker B5).

    The raw responses can hold the serial number of the unit, so keep the file out of the
    repository.
    """
    raw = {command: client.request(command) for command in ("ix", "rx", "ux", "cx")}
    if dump is not None:
        dump.write_text(
            "".join(f"{command} -> {text!r}\n" for command, text in raw.items()), encoding="utf-8"
        )
    info = parse_info(raw["ix"])
    reading = parse_reading(raw["rx"])
    unaveraged = parse_reading(raw["ux"])
    calibration = parse_calibration(raw["cx"])
    assert 5.0 < reading.magnitude < 25.0, f"the magnitude is implausible: {reading.magnitude}"
    assert abs(unaveraged.magnitude - reading.magnitude) < 2.0
    numbers = len(calibration.magnitudes) + len(calibration.periods_s)
    numbers += len(calibration.temperatures_c)
    return (
        f"protocol {info.protocol}, model {info.model}, feature {info.feature}; "
        f"{reading.magnitude:.2f} mag/arcsec2, temperature {reading.temperature_c} C, "
        f"frequency {reading.frequency_hz} Hz; {numbers} calibration numbers"
    )


def check_sqm_influx(
    config: SqmConfig,
    *,
    clock: Clock,
    env: Mapping[str, str] | None = None,
    opener: urllib.request.OpenerDirector | None = None,
) -> str:
    """InfluxDB holds a fresh, plausible reading of the SQM-LE, and the reader reads it.

    The check reads the newest point once, as `seeingmon hardware sqm` does. It fails with the
    error of the reader when the server does not answer, refuses the credentials, rejects the
    query, sends something that is no reading, holds no point, or holds only a point older than
    `max_age_s`. The report names no endpoint, bucket, database, measurement, field, or tag.
    """
    assert config.influx is not None, "the source influx needs the table [sqm.influx]"
    sample = read_sqm_once(config, clock=clock, env=env, opener=opener)
    assert 5.0 < sample.magnitude < 25.0, f"the magnitude is implausible: {sample.magnitude}"
    if sample.temperature_c is not None:
        assert -50.0 < sample.temperature_c < 70.0, "the temperature is implausible"
    return (
        f"{sample.summary()}; the point is within max_age_s ({config.influx.max_age_s:g} s), "
        f"and the server answered the query of version {config.influx.version}"
    )


def check_power_dry_run(power: PowerCycle) -> str:
    """The configured route expands from the environment and would run. Nothing runs."""
    result = power.request("hardware check")
    assert result.outcome is PowerOutcome.DRY_RUN, f"the dry run ended as {result.outcome}"
    return "the route is configured, the variables are set, and a dry run changed nothing"
