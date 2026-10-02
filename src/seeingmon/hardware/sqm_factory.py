"""Build the SQM-LE reader that `[sqm] source` names, and read one sample to try the settings.

`create_sqm_reader` returns an `SqmLeReader` for the source `tcp` and an `SqmInfluxReader` for the
source `influx`, as the `SqmReader` interface that `core` runs. For the source `influx`, it reads
the secrets with `resolve_credential`, as the sink factory does: a direct value in the
configuration, or an environment variable that `token_env` or `password_env` names. A variable that
is not set is a `ConfigError`, and the message names the variable and never a value.

`read_sqm_once` reads the configured source one time and returns an `SqmSample`, which
`seeingmon hardware sqm` prints. It ignores `enabled`, so you can try the settings before you turn
the reader on.
"""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass

from seeingmon.clock import Clock
from seeingmon.config import ConfigError
from seeingmon.hardware.events import EventCallback
from seeingmon.hardware.sqm import SqmConfig, SqmLeClient, SqmLeReader, SqmReader
from seeingmon.hardware.sqm_influx import SqmInfluxReader
from seeingmon.sinks.config import resolve_credential


def _influx_reader(
    config: SqmConfig,
    *,
    clock: Clock,
    station_id: str,
    profile_id: str,
    on_event: EventCallback | None,
    env: Mapping[str, str] | None,
    opener: urllib.request.OpenerDirector | None,
) -> SqmInfluxReader:
    influx = config.influx
    if influx is None:
        raise ConfigError('[sqm] source = "influx" needs the table [sqm.influx]')
    environment = os.environ if env is None else env
    token = resolve_credential(
        influx.token, influx.token_env, environment, "the token of [sqm.influx]"
    )
    password = resolve_credential(
        influx.password, influx.password_env, environment, "the password of [sqm.influx]"
    )
    return SqmInfluxReader(
        config,
        clock=clock,
        station_id=station_id,
        profile_id=profile_id,
        token=token,
        password=password,
        opener=opener,
        on_event=on_event,
    )


def create_sqm_reader(
    config: SqmConfig,
    *,
    clock: Clock,
    station_id: str,
    profile_id: str,
    on_event: EventCallback | None = None,
    env: Mapping[str, str] | None = None,
    opener: urllib.request.OpenerDirector | None = None,
) -> SqmReader:
    """Build the reader of the source that `config.source` names.

    `env` is where `token_env` and `password_env` look, and it defaults to `os.environ`. A test
    passes a dict. `opener` replaces the HTTP client of the `influx` source, and the `tcp` source
    ignores it. Raises `ConfigError` for a variable that is not set, or for a secret that a header
    cannot carry.
    """
    if config.source == "influx":
        return _influx_reader(
            config,
            clock=clock,
            station_id=station_id,
            profile_id=profile_id,
            on_event=on_event,
            env=env,
            opener=opener,
        )
    return SqmLeReader(
        config, clock=clock, station_id=station_id, profile_id=profile_id, on_event=on_event
    )


@dataclass(frozen=True, slots=True)
class SqmSample:
    """One reading for a person to look at.

    `age_s` is how long ago the reading was taken. It is 0 for the source `tcp`, because the unit
    answers with its current reading.
    """

    source: str
    magnitude: float
    temperature_c: float | None
    age_s: float

    def summary(self) -> str:
        """One line with the magnitude, the temperature, and the age. It names no setting."""
        temperature = "n/a" if self.temperature_c is None else f"{self.temperature_c:.1f} C"
        return (
            f"magnitude {self.magnitude:.2f} mag/arcsec^2, temperature {temperature}, "
            f"age {self.age_s:.1f} s (source {self.source})"
        )


def read_sqm_once(
    config: SqmConfig,
    *,
    clock: Clock,
    env: Mapping[str, str] | None = None,
    opener: urllib.request.OpenerDirector | None = None,
) -> SqmSample:
    """Read the configured source once, and return the sample. `enabled` plays no part.

    Raises `ConfigError` when the source lacks a setting. Raises an `SqmError` when the read fails:
    the TCP errors of `seeingmon.hardware.sqm`, or an `InfluxQueryError` of
    `seeingmon.hardware.sqm_influx`. No message holds a setting of the installation.
    """
    if config.source == "influx":
        reader = _influx_reader(
            config,
            clock=clock,
            station_id="probe",
            profile_id="probe",
            on_event=None,
            env=env,
            opener=opener,
        )
        reading = reader.read()
        return SqmSample("influx", reading.magnitude, reading.temperature_c, reading.age_s)
    if not config.host:
        raise ConfigError(
            '[sqm] host is not set. Set it, or set source = "influx" and the table [sqm.influx].'
        )
    client = SqmLeClient(
        config.host,
        config.port,
        connect_timeout_s=config.connect_timeout_s,
        read_timeout_s=config.read_timeout_s,
    )
    try:
        tcp_reading = client.read_reading(config.request)
    finally:
        client.close()
    return SqmSample("tcp", tcp_reading.magnitude, tcp_reading.temperature_c, 0.0)
