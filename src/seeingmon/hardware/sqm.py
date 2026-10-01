"""The SQM-LE reader: a fixed sky-brightness meter on the LAN.

The Unihedron SQM-LE measures the sky brightness in magnitudes per square arcsecond and reports it
over Ethernet. The system polls it, and it compares the readings with its own sky brightness to
fit the loss of the dome and the difference in altitude (see `docs/architecture.md`, "Sky
quality"). `SqmLeReader` turns each reading into a `ReferenceRecord` (`instrument` `sqm_le`,
`source` `fixed`).

**The protocol, as the owner's blocker B5 leaves it.** The reader uses the ASCII protocol that
Unihedron documents for the SQM-LE, from the maintainer's knowledge. **No sample from a real unit
has been checked yet**, so each fact below is unverified against hardware:

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

**What the reader stores.** A record has the time of the response (the unit may have measured a
little earlier, up to its sampling time), the magnitude, and the temperature. The pointing
(`altitude_deg`, `azimuth_deg`) comes from the configuration. The serial number and the address
never enter a record, an event, or a log line.
"""

from __future__ import annotations

import logging
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

from pydantic import Field, model_validator

from seeingmon.clock import Clock
from seeingmon.config import SectionModel
from seeingmon.hardware.events import EventCallback, HardwareEvent, emit
from seeingmon.records.reference import ReferenceRecord

_log = logging.getLogger(__name__)

ParsedT = TypeVar("ParsedT")

MIN_MAGNITUDE = -5.0  # a reading outside this range is garbled, whatever the unit says
MAX_MAGNITUDE = 30.0
MAX_LINE_BYTES = 512
SLEEP_SLICE_S = 1.0  # `run` sleeps in slices of this length, so a stop request is prompt
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


# --- The reader --------------------------------------------------------------------------


class SqmConfig(SectionModel):
    """The `[sqm]` section. The reader stays off unless you enable it and name the unit.

    `request` is `rx` for the averaged reading, or `ux` for the unaveraged one. `altitude_deg` and
    `azimuth_deg` say where the unit points. They are optional, and the record carries them when
    you set them.
    """

    enabled: bool = False
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

    @model_validator(mode="after")
    def _check(self) -> SqmConfig:
        if self.enabled and not self.host:
            raise ValueError("an enabled SQM-LE reader needs a host")
        if self.backoff_max_s < self.backoff_initial_s:
            raise ValueError("backoff_max_s must not be less than backoff_initial_s")
        return self


class SqmLeReader:
    """Poll an SQM-LE and produce `ReferenceRecord`s.

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
        if self._failures == 0:
            return self._cfg.poll_interval_s
        doublings = min(self._failures - 1, 32)  # a long outage must not overflow the float
        delay: float = self._cfg.backoff_initial_s * 2.0**doublings
        return min(delay, self._cfg.backoff_max_s)

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
            while not should_stop():
                record = self.poll()
                if record is not None:
                    on_record(record)
                remaining_s = self.delay_s
                while remaining_s > 0 and not should_stop():
                    step = min(remaining_s, SLEEP_SLICE_S)
                    self._clock.sleep(step)
                    remaining_s -= step
        finally:
            self.close()

    def close(self) -> None:
        """Close the connection."""
        self._client.close()
