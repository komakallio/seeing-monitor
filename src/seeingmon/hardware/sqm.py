"""The SQM-LE reader: a fixed sky-brightness meter, read from the unit or from InfluxDB.

The Unihedron SQM-LE measures the sky brightness in magnitudes per square arcsecond and reports it
over Ethernet. The system reads it, and it compares the readings with its own sky brightness to
fit the loss of the dome and the difference in altitude (see `docs/architecture.md`, "Sky
quality"). Each reading becomes a `ReferenceRecord` (`instrument` `sqm_le` by default, `source`
`fixed`).

**Two sources.** `[sqm] source` says where the readings come from:

- `tcp` (the default) polls the unit over the LAN. `SqmLeReader` in this module does that, and
  the rest of this module describes it.
- `influx` reads the readings that another program wrote to InfluxDB. It suits a unit that this
  system cannot reach, for example one that a different computer polls and records.
  `SqmInfluxReader` in `seeingmon.hardware.sqm_influx` does that, and the table `[sqm.influx]`
  holds its settings (`SqmInfluxConfig` in this module).

Both readers have the interface `SqmReader`, so `core` runs either one, and
`seeingmon.hardware.sqm_factory` builds the one that `source` names. Both give the same events
(`sqm.read_failed` and `sqm.recovered`), the same backoff after a failure, and the same count of
failed polls in a row, which the `sqm` component of `health` reports.

**The protocol of the TCP source, as the owner's blocker B5 leaves it.** The reader uses the
ASCII protocol that Unihedron documents for the SQM-LE, from the maintainer's knowledge. **No
sample from a real unit has been checked yet**, so each fact below is unverified against hardware:

- The unit listens on TCP port 10001.
- A request is two ASCII characters with no terminator: `ix` (information), `rx` (the reading,
  averaged), `ux` (the reading, unaveraged), and `cx` (calibration).
- A response is one line that ends with a carriage return and a line feed.
- The `rx` response looks like `r, 06.70m,0000022921Hz,0000000020c,0000000.000s, 039.4C`: the
  magnitude in mag/arcsec2, the frequency in hertz, the period count, the period time in seconds,
  and the sensor temperature in degrees Celsius.
- The `ix` response looks like `i,00000002,00000003,00000001,<serial>`: the protocol, the model,
  and the feature number, then a serial number that this module never stores.
- The `cx` response carries calibration numbers. The module reads them by their units and keeps the
  order that they arrive in, because the meaning of each position is unverified.

**A tolerant parser.** `parse_reading` finds each quantity by its unit (`m`, `Hz`, `c`, `s`, `C`),
accepts spaces, signs, leading zeros, and a different letter in front, and falls back to the
position of the field when units are missing. It rejects a line without a magnitude in the range
-5 to 30, so a garbled line never becomes a record.

**Connection.** By default the reader opens a connection for each poll and closes it after the
answer, so another program on the LAN can read the unit between polls. Set `persistent` to keep
one connection. A failure closes the connection, and the reader backs off exponentially from
`backoff_initial_s` to `backoff_max_s` before it tries again. All waiting between polls goes
through the `Clock`. The socket timeouts are real time, because they bound a wait on the network.

**What the TCP reader stores.** A record has the time of the response (the unit may have measured
a little earlier, up to its sampling time), the magnitude, and the temperature. The pointing
(`altitude_deg`, `azimuth_deg`) comes from the configuration. The serial number and the address
never enter a record, an event, or a log line.
"""

from __future__ import annotations

import logging
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, Self, TypeVar

from pydantic import ConfigDict, Field, SecretStr, field_validator, model_validator

from seeingmon.clock import Clock
from seeingmon.config import SectionModel
from seeingmon.hardware.events import EventCallback, HardwareEvent, emit
from seeingmon.records.reference import ReferenceRecord
from seeingmon.sinks.config import check_endpoint, check_influx_connection

_log = logging.getLogger(__name__)

ParsedT = TypeVar("ParsedT")

MIN_MAGNITUDE = -5.0  # a reading outside this range is garbled, whatever the unit says
MAX_MAGNITUDE = 30.0
MAX_LINE_BYTES = 512
SLEEP_SLICE_S = 1.0  # `run` sleeps in slices of this length, so a stop request is prompt
MAX_NAME_CHARS = 256  # the longest name or tag value of `[sqm.influx]`
MAX_TAGS = 16
MAX_SECONDS = 30 * 86_400  # the longest `max_age_s` or `lookback_s`: 30 days
READER_VERSION = "sqm-le-1"
_NUMBER = r"[-+]?\s*\d+(?:\.\d*)?"
_FIELD = re.compile(rf"({_NUMBER})\s*(Hz|hz|HZ|m|c|s|C)(?![A-Za-z])")


class SqmError(Exception):
    """A read of the SQM-LE failed."""


class SqmConnectionError(SqmError):
    """The unit did not accept the connection, or the connection broke."""


class SqmTimeoutError(SqmError):
    """The unit did not answer in time."""


class SqmParseError(SqmError):
    """The response is not a reading that the parser can use."""


# --- Parsing -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SqmReading:
    """One reading. `kind` is the response letter: `r` (averaged) or `u` (unaveraged)."""

    kind: str
    magnitude: float
    frequency_hz: float | None = None
    period_count: float | None = None
    period_s: float | None = None
    temperature_c: float | None = None


@dataclass(frozen=True, slots=True)
class SqmInfo:
    """The unit information, without the serial number."""

    protocol: int
    model: int
    feature: int


@dataclass(frozen=True, slots=True)
class SqmCalibration:
    """The numbers of a calibration response, in the order that they arrive.

    The position of each number is unverified against a real unit, so the fields group them by
    unit and keep their order.
    """

    magnitudes: tuple[float, ...]
    periods_s: tuple[float, ...]
    temperatures_c: tuple[float, ...]


def _number(text: str) -> float:
    return float(text.replace(" ", ""))


def _lines(text: str) -> list[str]:
    return [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]


def _find_line(text: str, letters: str) -> str:
    """The last line that starts with one of `letters` and a comma, such as `r,`."""
    for line in reversed(_lines(text)):
        if re.match(rf"^[{letters}]\s*,", line):
            return line
    raise SqmParseError("the response has no line of the expected kind")


def parse_reading(text: str) -> SqmReading:
    """Parse the response to `rx` or `ux`. Raises `SqmParseError` for anything else.

    The parser looks for the letter `r` or `u` and a comma at the start of a line, and it reads each
    quantity by its unit. When no unit appears at all, it reads the fields by position: the
    magnitude, the frequency, the period count, the period time, and the temperature.
    """
    line = _find_line(text, "rRuU")
    kind = line[0].lower()
    body = line.split(",", 1)[1]
    by_unit: dict[str, float] = {}
    for match in _FIELD.finditer(body):
        unit = "Hz" if match.group(2).lower() == "hz" else match.group(2)
        by_unit.setdefault(unit, _number(match.group(1)))
    if not by_unit:  # no units at all: read the fields by position
        try:
            numbers = [
                _number(re.sub(r"[A-Za-z]+$", "", part.strip()))
                for part in body.split(",")
                if part.strip()
            ]
        except ValueError:
            raise SqmParseError("the reading has no magnitude") from None
        by_unit = dict(zip(["m", "Hz", "c", "s", "C"], numbers, strict=False))
    if "m" not in by_unit:
        raise SqmParseError("the reading has no magnitude")
    magnitude = by_unit["m"]
    if not MIN_MAGNITUDE <= magnitude <= MAX_MAGNITUDE:
        raise SqmParseError("the magnitude is out of range")
    return SqmReading(
        kind=kind,
        magnitude=magnitude,
        frequency_hz=by_unit.get("Hz"),
        period_count=by_unit.get("c"),
        period_s=by_unit.get("s"),
        temperature_c=by_unit.get("C"),
    )


def parse_info(text: str) -> SqmInfo:
    """Parse the response to `ix`. The serial number is read past and dropped."""
    line = _find_line(text, "iI")
    numbers = re.findall(r"\d+", line.split(",", 1)[1])
    if len(numbers) < 3:
        raise SqmParseError("the information response has fewer than three numbers")
    return SqmInfo(int(numbers[0]), int(numbers[1]), int(numbers[2]))


def parse_calibration(text: str) -> SqmCalibration:
    """Parse the response to `cx` into its numbers, grouped by unit."""
    line = _find_line(text, "cC")
    magnitudes: list[float] = []
    periods: list[float] = []
    temperatures: list[float] = []
    for match in _FIELD.finditer(line.split(",", 1)[1]):
        value, unit = _number(match.group(1)), match.group(2)
        if unit == "m":
            magnitudes.append(value)
        elif unit == "s":
            periods.append(value)
        elif unit == "C":
            temperatures.append(value)
    if not (magnitudes or periods or temperatures):
        raise SqmParseError("the calibration response has no numbers")
    return SqmCalibration(tuple(magnitudes), tuple(periods), tuple(temperatures))


# --- The connection ----------------------------------------------------------------------


class SqmLeClient:
    """A TCP client for the SQM-LE: one request, one response line.

    Args:
        host: The address of the unit.
        port: The TCP port. The documented default is 10001.
        connect_timeout_s: The longest wait for the connection, in real time.
        read_timeout_s: The longest wait for a response line, in real time.
        persistent: Keep the connection between requests. Otherwise each request opens one.
    """

    def __init__(
        self,
        host: str,
        port: int = 10001,
        *,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 10.0,
        persistent: bool = False,
    ) -> None:
        self._host = host
        self._port = port
        self._connect_timeout_s = connect_timeout_s
        self._read_timeout_s = read_timeout_s
        self._persistent = persistent
        self._socket: socket.socket | None = None

    def close(self) -> None:
        """Close the connection. Safe to call twice."""
        sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                _log.debug("closing the SQM-LE socket failed", exc_info=True)

    def _connect(self) -> socket.socket:
        if self._socket is not None:
            return self._socket
        try:
            sock = socket.create_connection(
                (self._host, self._port), timeout=self._connect_timeout_s
            )
        except TimeoutError:
            raise SqmTimeoutError("the unit did not accept the connection in time") from None
        except OSError as error:
            raise SqmConnectionError(
                f"the connection to the unit failed ({type(error).__name__})"
            ) from None
        sock.settimeout(self._read_timeout_s)
        self._socket = sock
        return sock

    def request(self, command: str) -> str:
        """Send a command (such as `rx`) and return the response text.

        The text holds the first response line, with its line end. A response that the unit cuts
        off before the line end is an error, because a partial line can hold a wrong number.
        Raises `SqmTimeoutError` when the unit does not answer in time, `SqmConnectionError` when
        the connection breaks or closes early, and `SqmParseError` for a line that does not end
        within `MAX_LINE_BYTES`. A failure closes the connection, and a success closes it too
        unless `persistent` is set.
        """
        sock = self._connect()
        try:
            sock.sendall(command.encode("ascii"))
            data = b""
            while b"\n" not in data:
                chunk = sock.recv(MAX_LINE_BYTES)
                if not chunk:
                    raise SqmConnectionError(
                        "the unit closed the connection before it ended the line"
                        if data
                        else "the unit closed the connection without an answer"
                    )
                data += chunk
                if len(data) > MAX_LINE_BYTES:
                    raise SqmParseError("the response line is too long")
        except TimeoutError:
            self.close()
            raise SqmTimeoutError("the unit did not answer in time") from None
        except OSError as error:
            self.close()
            raise SqmConnectionError(f"the connection broke ({type(error).__name__})") from None
        except SqmError:
            self.close()
            raise
        if not self._persistent:
            self.close()
        return data.decode("ascii", errors="replace")

    def _parse(self, parser: Callable[[str], ParsedT], text: str) -> ParsedT:
        """Parse a response. A response that does not parse closes the connection."""
        try:
            return parser(text)
        except SqmParseError:
            self.close()
            raise

    def read_info(self) -> SqmInfo:
        return self._parse(parse_info, self.request("ix"))

    def read_reading(self, command: str = "rx") -> SqmReading:
        return self._parse(parse_reading, self.request(command))

    def read_calibration(self) -> SqmCalibration:
        return self._parse(parse_calibration, self.request("cx"))


# --- The configuration -------------------------------------------------------------------


class SqmInfluxConfig(SectionModel):
    """The `[sqm.influx]` table: where InfluxDB keeps the readings of the SQM-LE, and their shape.

    Read the table with the `source` of `[sqm]` set to `influx`. Every key describes one
    installation, so keep the table in `local/config.toml` and give a secret with `token_env` or
    `password_env`, the names of environment variables. `seeingmon hardware sqm` reads one point
    with these settings and says what it found.

    **The connection.** The keys follow the InfluxDB sink (`seeingmon.sinks.config`). Version 2
    (Flux, `POST /api/v2/query`) needs `org`, `bucket`, and a `token`. Version 1 (InfluxQL,
    `GET /query`) needs `database`, and takes `retention_policy`, `username`, and a password.

    **The data.** `measurement` and `field` name the series with the magnitude in mag/arcsec^2.
    `temperature_field` names the field with the temperature in degrees Celsius. `tags` maps tag
    names to values, and it selects the unit when the measurement holds more than one.

    **Freshness.** A reading older than `max_age_s` is stale, and it counts as a failed poll.
    `lookback_s` is how far back the query looks, so it must be at least `max_age_s`. Give it more
    to tell a stale reading from no reading at all.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", coerce_numbers_to_str=True)

    endpoint: str = Field(
        description="The address of the server, such as https://influx.example.org."
    )
    version: Literal[1, 2] = Field(default=2, description="The query API: 1 or 2.")
    org: str | None = Field(default=None, min_length=1, description="The organization (version 2).")
    bucket: str | None = Field(default=None, min_length=1, description="The bucket (version 2).")
    database: str | None = Field(
        default=None, min_length=1, description="The database (version 1)."
    )
    retention_policy: str | None = Field(
        default=None, min_length=1, description="The retention policy (version 1)."
    )
    token: SecretStr | None = Field(default=None, description="The API token (version 2).")
    token_env: str | None = Field(
        default=None, description="The name of the environment variable that holds the token."
    )
    username: str | None = Field(default=None, description="The user name (version 1).")
    password: SecretStr | None = Field(default=None, description="The password (version 1).")
    password_env: str | None = Field(
        default=None, description="The name of the environment variable that holds the password."
    )
    timeout_s: float = Field(
        default=10.0, gt=0, le=300, description="How long to wait for the server, in seconds."
    )
    verify_tls: bool = Field(
        default=True, description="Whether to check the certificate of the server."
    )
    measurement: str = Field(
        min_length=1, max_length=MAX_NAME_CHARS, description="The measurement of the readings."
    )
    field: str = Field(
        min_length=1,
        max_length=MAX_NAME_CHARS,
        description="The field with the magnitude, in mag/arcsec^2.",
    )
    temperature_field: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_NAME_CHARS,
        description="The field with the temperature, in degrees Celsius.",
    )
    tags: dict[str, str] = Field(
        default_factory=dict, description="The tags that select the unit, as name and value."
    )
    max_age_s: float = Field(
        default=600.0,
        gt=0,
        le=MAX_SECONDS,
        description="A reading older than this is stale, in seconds.",
    )
    lookback_s: float = Field(
        default=3600.0,
        gt=0,
        le=MAX_SECONDS,
        description="How far back the query looks, in seconds. At least max_age_s.",
    )

    @field_validator("endpoint")
    @classmethod
    def _check_endpoint(cls, value: str) -> str:
        return check_endpoint(value)

    @field_validator("tags")
    @classmethod
    def _check_tags(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > MAX_TAGS:
            raise ValueError(f"use at most {MAX_TAGS} tags")
        for name, text in value.items():
            if not name or not text:
                raise ValueError("a tag needs a name and a value")
            if len(name) > MAX_NAME_CHARS or len(text) > MAX_NAME_CHARS:
                raise ValueError(f"a tag name or value has at most {MAX_NAME_CHARS} characters")
        return value

    @model_validator(mode="after")
    def _check_settings(self) -> Self:
        check_influx_connection(
            version=self.version,
            org=self.org,
            bucket=self.bucket,
            database=self.database,
            token=self.token,
            token_env=self.token_env,
            username=self.username,
            password=self.password,
            password_env=self.password_env,
        )
        if self.temperature_field == self.field:
            raise ValueError("temperature_field must differ from field")
        if self.lookback_s < self.max_age_s:
            raise ValueError("lookback_s must not be less than max_age_s")
        return self


class SqmConfig(SectionModel):
    """The `[sqm]` section. The reader stays off unless you enable it and name its source.

    `source` is `tcp` (the default) to poll the unit over the LAN, which needs `host`, or `influx`
    to read the readings from InfluxDB, which needs the table `[sqm.influx]`. A reader that you
    enable needs the keys of its source. `request` is `rx` for the averaged reading, or `ux` for
    the unaveraged one (a `tcp` key, like `host`, `port`, `persistent`, and the two timeouts).
    `poll_interval_s`, the backoff keys, `instrument`, `altitude_deg`, and `azimuth_deg` serve both
    sources. `altitude_deg` and `azimuth_deg` say where the unit points. They are optional, and
    the record carries them when you set them.
    """

    enabled: bool = False
    source: Literal["tcp", "influx"] = "tcp"
    host: str = ""
    port: int = Field(default=10001, ge=1, le=65535)
    poll_interval_s: float = Field(default=60.0, ge=1.0)
    connect_timeout_s: float = Field(default=5.0, gt=0.0)
    read_timeout_s: float = Field(default=10.0, gt=0.0)
    backoff_initial_s: float = Field(default=5.0, gt=0.0)
    backoff_max_s: float = Field(default=300.0, gt=0.0)
    persistent: bool = False
    request: Literal["rx", "ux"] = "rx"
    instrument: str = Field(default="sqm_le", min_length=1)
    altitude_deg: float | None = Field(default=None, ge=-90.0, le=90.0)
    azimuth_deg: float | None = Field(default=None, ge=0.0, le=360.0)
    influx: SqmInfluxConfig | None = None

    @model_validator(mode="after")
    def _check(self) -> SqmConfig:
        if self.enabled and self.source == "tcp" and not self.host:
            raise ValueError('an enabled SQM-LE reader with source = "tcp" needs a host')
        if self.enabled and self.source == "influx" and self.influx is None:
            raise ValueError('an enabled SQM-LE reader with source = "influx" needs [sqm.influx]')
        if self.backoff_max_s < self.backoff_initial_s:
            raise ValueError("backoff_max_s must not be less than backoff_initial_s")
        return self


# --- The readers -------------------------------------------------------------------------


class SqmReader(Protocol):
    """What `core` needs from an SQM-LE reader. `SqmLeReader` and `SqmInfluxReader` have it.

    `poll` reads once and returns the record, or `None` when the read failed or gave nothing new.
    It never raises. `failures` counts the failed polls in a row, and `delay_s` is the wait before
    the next poll. `run` polls until `should_stop()` returns true.
    """

    @property
    def failures(self) -> int: ...

    @property
    def delay_s(self) -> float: ...

    def poll(self) -> ReferenceRecord | None: ...

    def run(
        self, should_stop: Callable[[], bool], on_record: Callable[[ReferenceRecord], None]
    ) -> None: ...

    def close(self) -> None: ...


def backoff_delay_s(config: SqmConfig, failures: int) -> float:
    """The wait before the next poll: the poll interval, or the backoff after a failure.

    After `failures` failed polls in a row, the wait doubles from `backoff_initial_s` up to
    `backoff_max_s`. Both readers use it.
    """
    if failures == 0:
        return config.poll_interval_s
    doublings = min(failures - 1, 32)  # a long outage must not overflow the float
    delay: float = config.backoff_initial_s * 2.0**doublings
    return min(delay, config.backoff_max_s)


def run_polling(
    reader: SqmReader,
    clock: Clock,
    should_stop: Callable[[], bool],
    on_record: Callable[[ReferenceRecord], None],
) -> None:
    """Poll until `should_stop()` returns true, and hand each record to `on_record`.

    The loop sleeps on the clock in slices of one second, so a stop request takes effect within a
    second. Both readers run it.
    """
    while not should_stop():
        record = reader.poll()
        if record is not None:
            on_record(record)
        remaining_s = reader.delay_s
        while remaining_s > 0 and not should_stop():
            step = min(remaining_s, SLEEP_SLICE_S)
            clock.sleep(step)
            remaining_s -= step


class SqmLeReader:
    """Poll an SQM-LE over TCP and produce `ReferenceRecord`s. This is the `tcp` source.

    Args:
        config: The `[sqm]` section.
        clock: The time source for the time stamps and the waits between polls.
        station_id: The station ID from the configuration, for the record.
        profile_id: The profile ID, for the record.
        client: Replaces the TCP client, for a test.
        on_event: Receives `sqm.read_failed` for the first failure of a streak, and
            `sqm.recovered` when reads work again.
    """

    def __init__(
        self,
        config: SqmConfig,
        *,
        clock: Clock,
        station_id: str,
        profile_id: str,
        client: SqmLeClient | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self._cfg = config
        self._clock = clock
        self._station_id = station_id
        self._profile_id = profile_id
        self._client = client or SqmLeClient(
            config.host,
            config.port,
            connect_timeout_s=config.connect_timeout_s,
            read_timeout_s=config.read_timeout_s,
            persistent=config.persistent,
        )
        self._on_event = on_event
        self._failures = 0
        self._info: SqmInfo | None = None
        self._info_tried = False

    @property
    def failures(self) -> int:
        """The number of failed polls in a row."""
        return self._failures

    @property
    def delay_s(self) -> float:
        """The wait before the next poll: the poll interval, or the backoff after a failure."""
        return backoff_delay_s(self._cfg, self._failures)

    def _emit(self, level: str, kind: str, message: str, **detail: object) -> None:
        emit(
            self._on_event,
            HardwareEvent(level, kind, message, self._clock.utc_ns(), detail or None),
        )

    def _provenance(self) -> dict[str, str]:
        provenance = {"reader": READER_VERSION}
        if self._info is not None:
            provenance.update(
                protocol=str(self._info.protocol),
                model=str(self._info.model),
                feature=str(self._info.feature),
            )
        return provenance

    def poll(self) -> ReferenceRecord | None:
        """Read the unit once. Returns the record, or `None` when the read failed.

        A failure never raises. It counts toward the backoff (`delay_s`) and reports an event
        for the first failure of a streak.
        """
        try:
            reading = self._client.read_reading(self._cfg.request)
            t_utc_ns = self._clock.utc_ns()  # the time of the response
        except SqmError as error:
            self._client.close()
            self._failures += 1
            if self._failures == 1:
                self._emit(
                    "warning",
                    "sqm.read_failed",
                    "The SQM-LE did not give a usable reading.",
                    cause=type(error).__name__,
                )
            return None
        if self._failures:
            self._emit(
                "info", "sqm.recovered", "The SQM-LE answers again.", failures=self._failures
            )
            self._failures = 0
        if not self._info_tried:
            self._info_tried = True
            try:
                self._info = self._client.read_info()
            except SqmError:
                _log.debug("the SQM-LE information request failed", exc_info=True)
        return ReferenceRecord(
            station_id=self._station_id,
            t_utc_ns=t_utc_ns,
            profile_id=self._profile_id,
            provenance=self._provenance(),
            instrument=self._cfg.instrument,
            source="fixed",
            value_mag_arcsec2=reading.magnitude,
            temperature_c=reading.temperature_c,
            altitude_deg=self._cfg.altitude_deg,
            azimuth_deg=self._cfg.azimuth_deg,
        )

    def run(
        self, should_stop: Callable[[], bool], on_record: Callable[[ReferenceRecord], None]
    ) -> None:
        """Poll until `should_stop()` returns true. Each record goes to `on_record`.

        The loop sleeps on the clock in slices of one second, so a stop request takes effect
        within a second. It closes the connection when it ends.
        """
        try:
            run_polling(self, self._clock, should_stop, on_record)
        finally:
            self.close()

    def close(self) -> None:
        """Close the connection."""
        self._client.close()
