"""The SQM-LE reader that reads from InfluxDB: the readings that another program wrote.

Some units cannot be reached over TCP. The SQM-LE may sit behind another computer, for example an
all-sky camera computer that polls it and writes each reading to InfluxDB. `SqmInfluxReader` asks
that database for the newest reading, and it gives `ReferenceRecord`s like the TCP reader
(`seeingmon.hardware.sqm`): `instrument` from the configuration, `source` `fixed`. Every detail of
the database is a setting of the table `[sqm.influx]` (`SqmInfluxConfig`), because the details of
an installation change.

**The queries.** Each poll asks for the newest point of the field, and the temperature field when
you name one. The reader supports two APIs, and it makes no claim about InfluxDB 3 (untested):

- **Version 1** sends `GET /query?db=&epoch=ns&q=`. The query is an InfluxQL `SELECT` of the field
  (and the temperature field) from the measurement, with the tag filters and a time lower bound,
  `ORDER BY time DESC LIMIT 1`. When you set a user name and a password, the reader sends HTTP
  basic authentication in a header, and never `u` and `p` in the query string.
- **Version 2** sends `POST /api/v2/query?org=` with `Content-Type: application/vnd.flux` and
  `Accept: application/csv`. The body is a Flux program with `range`, a `filter` for the
  measurement, the fields, and each tag, and `last()`. The header is `Authorization: Token
  <token>`.

The reader never follows a redirect. It reads the reply up to `MAX_REPLY_BYTES`: JSON for version
1, and annotated CSV for version 2 (`parse_flux_points` reads the annotation lines, the empty
line between tables, and one table for each field). A parser raises `InfluxParseError` for
anything that it cannot read, and a magnitude outside -5 to 30 is such a thing.

**Escaping.** The settings come from a file, and a name can hold any character, so every
identifier and string goes into a query escaped for its language (`influxql_identifier`,
`influxql_string`, `flux_string`). A quote, a backslash, or a line break in a setting cannot end
its literal and cannot add anything to the query, and the Flux escape stops the interpolation
`${...}` too.

**The time of a record.** `t_utc_ns` is the time stamp of the point, not the time of the poll. A
point with the time stamp of the last record gives no new record, and it is not a failure, so a
source that updates every few minutes works with a poll every minute. A point older than
`max_age_s` on the `Clock` is `Stale`, and a reply with no point in the `lookback_s` window is
`NoData`. Only the time stamp of the last record counts as a repeat. A point with an earlier time
stamp, such as one after a step of the clock of the writer, is a new reading, so a single point
with a wrong time stamp in the future cannot hide the readings that follow it.

**Failures.** A failed poll never raises. It counts toward the backoff (`delay_s`, with the keys
`poll_interval_s`, `backoff_initial_s`, and `backoff_max_s`) and toward `failures`, and the first
failure of a streak reports `sqm.read_failed` with one of these causes:

- `Unreachable`: the network failed, the server did not answer in `timeout_s`, or it answered 5xx,
  408, or 429.
- `Unauthorized`: the server answered 401 or 403.
- `BadRequest`: the server answered another 4xx or a redirect, or it reported an error in a reply
  of 200. The message holds the server message, shortened, as the sink does.
- `Parse`: the reply is not a reading that the parser can use.
- `NoData`: the lookback window holds no point of the field.
- `Stale`: the newest point is older than `max_age_s`.

Recovery reports `sqm.recovered`, as the TCP reader does.

**A range of readings.** `read_range` returns the points of a time range as records. The offset
fit of `seeingmon.survey.sqm_fit` can pair them with the survey sky brightness of a night that
the live reader missed.

**What stays out.** The endpoint, the bucket or database, the organization, the user name, the
measurement, the field and tag names and values, the token, and the password never enter a
record, an event, a log line, or an exception message. The reader replaces each of them in the
message of a failure, because a server can repeat a name that it did not find.

**Time and waiting.** The waits between polls go through the `Clock`, in slices of one second. The
timeout of a request (`timeout_s`) is real time, because it bounds a wait on the network.
"""

from __future__ import annotations

import base64
import csv
import http.client
import io
import json
import logging
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from seeingmon.clock import NS_PER_S, Clock, iso_to_utc_ns
from seeingmon.config import REDACTED, ConfigError
from seeingmon.hardware.events import EventCallback, HardwareEvent, emit
from seeingmon.hardware.sqm import (
    MAX_MAGNITUDE,
    MIN_MAGNITUDE,
    SqmConfig,
    SqmError,
    SqmInfluxConfig,
    backoff_delay_s,
    run_polling,
)
from seeingmon.records.reference import ReferenceRecord
from seeingmon.sinks.influx import error_reply_message, make_opener

_log = logging.getLogger(__name__)

READER_VERSION = "sqm-influx-1"
MAX_REPLY_BYTES = 1_048_576  # the reply to one poll is a few hundred bytes
MAX_RANGE_BYTES = 67_108_864  # the reply to a range read can hold a night of readings
MAX_RANGE_POINTS = 100_000
MAX_TEMPERATURE_C = 1000.0  # a temperature outside this range is garbled
MAX_TIME_NS = 2**63 - 1
NUMERIC_GUARD = "-1000"  # a magnitude is never below it, so a comparison keeps numeric values only
MAX_MESSAGE_CHARS = 200  # the words of the server that an error message keeps
MAX_SERVER_TEXT = 2000  # the words of the server that the reader keeps until it has cleaned them

_HINTS = {
    401: "check the token or the credentials",
    403: "the token or the user may not read this bucket or database",
    404: "check the endpoint, the organization, and the bucket or the database",
}


# --- Errors ------------------------------------------------------------------------------


class InfluxQueryError(SqmError):
    """A read of the readings from InfluxDB failed. `cause` names the kind for the event."""

    cause: ClassVar[str] = "Error"


class InfluxUnreachableError(InfluxQueryError):
    """The network failed, the server was too slow, or it answered 5xx, 408, or 429."""

    cause = "Unreachable"


class InfluxUnauthorizedError(InfluxQueryError):
    """The server refused the credentials: 401 or 403."""

    cause = "Unauthorized"


class InfluxBadRequestError(InfluxQueryError):
    """The server rejected the query: another 4xx, a redirect, or an error in a reply of 200."""

    cause = "BadRequest"


def _one_line(text: str) -> str:
    """The words of a server on one line, shortened for a message."""
    return " ".join(text.split())[:MAX_MESSAGE_CHARS]


class InfluxReportedError(InfluxBadRequestError):
    """A reply of 200 holds an error of the server. `server_message` holds its words.

    The parsers raise this error with the words of the server as they are, up to
    `MAX_SERVER_TEXT` characters. `SqmInfluxReader` replaces it with an `InfluxBadRequestError`
    that has the words cleaned of every setting, and then shortened.
    """

    def __init__(self, server_message: str) -> None:
        self.server_message = server_message[:MAX_SERVER_TEXT]
        super().__init__(f"InfluxDB reported an error: {_one_line(self.server_message)}")


class InfluxParseError(InfluxQueryError):
    """The reply is not a reading that the parser can use."""

    cause = "Parse"


class InfluxNoDataError(InfluxQueryError):
    """The lookback window holds no point of the field."""

    cause = "NoData"


class InfluxStaleError(InfluxQueryError):
    """The newest point is older than `max_age_s`."""

    cause = "Stale"


# --- Escaping and queries ----------------------------------------------------------------


def _escape_influxql(text: str, quote: str) -> str:
    """Escape a text for the inside of a quoted InfluxQL literal that `quote` delimits.

    InfluxQL reads a backslash, the quote, and the letter `n` after a backslash as escapes, and it
    refuses a raw line break inside a literal.
    """
    return text.replace("\\", "\\\\").replace(quote, "\\" + quote).replace("\n", "\\n")


def influxql_identifier(text: str) -> str:
    """Quote a measurement, a field, a tag, or a retention policy for InfluxQL.

    The result is in double quotes. A double quote, a backslash, and a line break inside it get an
    escape, so the text cannot end the identifier.
    """
    return '"' + _escape_influxql(text, '"') + '"'


def influxql_string(text: str) -> str:
    """Quote a string value for InfluxQL, such as the value of a tag filter.

    The result is in single quotes. A single quote, a backslash, and a line break inside it get an
    escape, so the text cannot end the string.
    """
    return "'" + _escape_influxql(text, "'") + "'"


_FLUX_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t", "$": "\\$"}
_FLUX_UNSAFE = re.compile(r'[\\"\x00-\x1f\x7f]|\$(?=\{)')


def flux_string(text: str) -> str:
    """Quote a string for Flux, such as a bucket name or the value of a tag filter.

    The result is in double quotes. A double quote, a backslash, a line break, a carriage return,
    and a tab get their escape, and any other control character becomes `\\xHH`. A `$` before a
    `{` becomes `\\$`, because Flux would otherwise read `${...}` as an expression and run it.
    """

    def escape(match: re.Match[str]) -> str:
        char = match.group()
        return _FLUX_ESCAPES.get(char) or f"\\x{ord(char):02x}"

    return '"' + _FLUX_UNSAFE.sub(escape, text) + '"'


def duration_literal(seconds: float) -> str:
    """A duration that InfluxQL and Flux both read: whole seconds, or else milliseconds."""
    if seconds == int(seconds):
        return f"{int(seconds)}s"
    return f"{max(1, round(seconds * 1000))}ms"


def build_influxql(config: SqmInfluxConfig, *, window: tuple[int, int] | None = None) -> str:
    """The InfluxQL query of version 1.

    Without `window`, it asks for the newest point in the last `lookback_s` seconds. With a
    `window` of `(start_ns, end_ns)`, it asks for the points from `start_ns` up to `end_ns`, the
    oldest first, and for one more than `MAX_RANGE_POINTS`, so that a caller sees an overflow.
    """
    columns = [influxql_identifier(config.field)]
    if config.temperature_field:
        columns.append(influxql_identifier(config.temperature_field))
    source = influxql_identifier(config.measurement)
    if config.retention_policy:
        source = influxql_identifier(config.retention_policy) + "." + source
    if window is None:
        conditions = [f"time > now() - {duration_literal(config.lookback_s)}"]
    else:
        conditions = [f"time >= {window[0]}ns", f"time < {window[1]}ns"]
    if config.temperature_field:
        # The newest row can hold the temperature alone. A comparison keeps the rows that have a
        # number in the field, because a comparison with a missing value is false.
        conditions.append(f"{influxql_identifier(config.field)} > {NUMERIC_GUARD}")
    conditions += [
        f"{influxql_identifier(name)} = {influxql_string(value)}"
        for name, value in config.tags.items()
    ]
    if window is None:
        order = "ORDER BY time DESC LIMIT 1"
    else:
        order = f"ORDER BY time ASC LIMIT {MAX_RANGE_POINTS + 1}"
    return f"SELECT {', '.join(columns)} FROM {source} WHERE {' AND '.join(conditions)} {order}"


def build_flux(config: SqmInfluxConfig, *, window: tuple[int, int] | None = None) -> str:
    """The Flux program of version 2.

    Without `window`, it asks for the last point of each field in the last `lookback_s` seconds.
    With a `window` of `(start_ns, end_ns)`, it asks for every point from `start_ns` up to
    `end_ns`.
    """
    fields = [config.field] + ([config.temperature_field] if config.temperature_field else [])
    field_test = " or ".join(f"r._field == {flux_string(name)}" for name in fields)
    if window is None:
        time_range = f"start: -{duration_literal(config.lookback_s)}"
    else:
        time_range = f"start: time(v: {window[0]}), stop: time(v: {window[1]})"
    lines = [
        f"from(bucket: {flux_string(config.bucket or '')})",
        f"  |> range({time_range})",
        f"  |> filter(fn: (r) => r._measurement == {flux_string(config.measurement)})",
        f"  |> filter(fn: (r) => {field_test})",
    ]
    lines += [
        f"  |> filter(fn: (r) => r[{flux_string(name)}] == {flux_string(value)})"
        for name, value in config.tags.items()
    ]
    if window is None:
        lines.append("  |> last()")
    return "\n".join(lines) + "\n"


# --- Parsing -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InfluxPoint:
    """One point of the field: its time stamp, the magnitude, and the temperature if it has one."""

    t_utc_ns: int
    magnitude: float
    temperature_c: float | None = None


@dataclass(frozen=True, slots=True)
class FluxRow:
    """One data row of a Flux reply: the field, the time stamp, and the value."""

    field: str
    t_utc_ns: int
    value: float


@dataclass(frozen=True, slots=True)
class InfluxReading:
    """The newest point, checked for age. `age_s` is how old it is on the clock of the reader."""

    t_utc_ns: int
    magnitude: float
    temperature_c: float | None
    age_s: float


def _make_point(t_utc_ns: int, magnitude: float, temperature_c: float | None) -> InfluxPoint:
    """Check one point. A value that no sky meter gives is a parse error."""
    if not 0 < t_utc_ns <= MAX_TIME_NS:
        raise InfluxParseError("a time stamp is out of range")
    if not MIN_MAGNITUDE <= magnitude <= MAX_MAGNITUDE:
        raise InfluxParseError("a magnitude is out of range")
    if temperature_c is not None and abs(temperature_c) > MAX_TEMPERATURE_C:
        raise InfluxParseError("a temperature is out of range")
    return InfluxPoint(t_utc_ns, magnitude, temperature_c)


def _json_number(value: object, what: str) -> float | None:
    """A number of a JSON reply as a float. Null gives `None`, and anything else is an error."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InfluxParseError(f"the {what} is not a number")
    try:
        number = float(value)
    except OverflowError:
        raise InfluxParseError(f"the {what} is not a number") from None
    if not math.isfinite(number):
        raise InfluxParseError(f"the {what} is not a number")
    return number


def _json_time_ns(value: object) -> int:
    """The time stamp of a row: an integer of nanoseconds (`epoch=ns`), or else a UTC text."""
    if isinstance(value, bool):
        raise InfluxParseError("a time stamp is not a time")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return iso_to_utc_ns(value)
        except ValueError:
            raise InfluxParseError("a time stamp is not a time") from None
    raise InfluxParseError("a time stamp is not a time")


def parse_influxql_points(
    text: str, field: str, temperature_field: str | None = None
) -> list[InfluxPoint]:
    """Read the reply of `GET /query`: the points that its first series holds, in reply order.

    Returns an empty list for a reply with no series (no point matched). A row where the field is
    null stays out. Raises `InfluxParseError` when the reply is not what InfluxDB sends,
    and `InfluxBadRequestError` when the reply holds an error of the server.
    """
    try:
        document = json.loads(text)
    except ValueError:
        raise InfluxParseError("the reply is not JSON") from None
    if not isinstance(document, dict):
        raise InfluxParseError("the reply is not a JSON object")
    if isinstance(document.get("error"), str):
        raise InfluxReportedError(document["error"])
    results = document.get("results")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise InfluxParseError("the reply has no results")
    result = results[0]
    if isinstance(result.get("error"), str):
        raise InfluxReportedError(result["error"])
    series = result.get("series")
    if series is None:
        return []
    if not isinstance(series, list):
        raise InfluxParseError("the reply has no series")
    if not series:
        return []
    if not isinstance(series[0], dict):
        raise InfluxParseError("the reply has no series")
    columns, rows = series[0].get("columns"), series[0].get("values")
    if (
        not isinstance(columns, list)
        or not isinstance(rows, list)
        or any(not isinstance(name, str) for name in columns)
    ):
        raise InfluxParseError("the series has no columns or no values")
    try:
        time_index, field_index = columns.index("time"), columns.index(field)
    except ValueError:
        raise InfluxParseError("the series lacks the time column or the field column") from None
    temperature_index = (
        columns.index(temperature_field)
        if temperature_field is not None and temperature_field in columns
        else None
    )
    points: list[InfluxPoint] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            raise InfluxParseError("a row does not match the columns")
        magnitude = _json_number(row[field_index], "magnitude")
        if magnitude is None:
            continue
        temperature = (
            None
            if temperature_index is None
            else _json_number(row[temperature_index], "temperature")
        )
        points.append(_make_point(_json_time_ns(row[time_index]), magnitude, temperature))
    return points


_NUMERIC_TYPES = frozenset({"double", "long", "unsignedLong"})


def parse_flux_rows(text: str) -> list[FluxRow]:
    """Read the annotated CSV of `POST /api/v2/query` into the rows of its tables.

    A reply holds one table for each series (so one for each field, with the right tags), and the
    tables follow one another under one header or under their own, with an empty line before a new
    header. The parser reads the annotation rows (`#datatype`, `#group`, and `#default`), the
    header, and the data rows, and it finds each column by its name. `_time` is RFC 3339 with up
    to nine fractional digits, and `_value` is a number. An empty reply, and a reply with no data
    row, give an empty list. Raises `InfluxParseError` for anything that it cannot read, and
    `InfluxBadRequestError` for the error table that a failed query leaves in a reply of 200.
    """
    rows: list[FluxRow] = []
    header: dict[str, int] | None = None
    width = 0
    datatypes: list[str] | None = None
    annotated = False  # a table has started with its annotation rows
    waiting_for_header = False
    try:
        for record in csv.reader(io.StringIO(text, newline=""), strict=True):
            if not record:  # an empty line: the next table has its own annotations and header
                header = datatypes = None
                continue
            if record[0].startswith("#"):
                if header is not None:  # annotations after data rows: a new table starts
                    header = datatypes = None
                annotated = waiting_for_header = True
                if record[0] == "#datatype":
                    datatypes = record
                continue
            if not annotated:
                raise InfluxParseError("the reply is not annotated CSV")
            if header is None:
                header = {name: index for index, name in enumerate(record) if name}
                width = len(record)
                waiting_for_header = False
                continue
            if len(record) != width:
                raise InfluxParseError("a row does not match the header")
            if "error" in header and "_value" not in header:
                raise InfluxReportedError(record[header["error"]])
            if any(name not in header for name in ("_time", "_value", "_field")):
                raise InfluxParseError("the header lacks _time, _value, or _field")
            if (
                datatypes is not None
                and len(datatypes) == width
                and datatypes[header["_value"]] not in _NUMERIC_TYPES
            ):
                raise InfluxParseError("the value is not a number")
            try:
                t_utc_ns = iso_to_utc_ns(record[header["_time"]])
                value = float(record[header["_value"]])
            except ValueError:
                raise InfluxParseError("a row has no valid time or value") from None
            if not math.isfinite(value) or not 0 < t_utc_ns <= MAX_TIME_NS:
                raise InfluxParseError("a row has no valid time or value")
            rows.append(FluxRow(record[header["_field"]], t_utc_ns, value))
    except csv.Error:
        raise InfluxParseError("the reply is not valid CSV") from None
    if waiting_for_header:
        raise InfluxParseError("the reply ends before its header")
    return rows


def parse_flux_points(
    text: str, field: str, temperature_field: str | None = None
) -> list[InfluxPoint]:
    """Read the reply of `POST /api/v2/query` into points, oldest first.

    A point exists for each time stamp where the `field` has a row. It carries the temperature
    when `temperature_field` has a row at the same time stamp. See `parse_flux_rows` for the
    format, and for the errors.
    """
    magnitudes: dict[int, float] = {}
    temperatures: dict[int, float] = {}
    for row in parse_flux_rows(text):
        if row.field == field:
            magnitudes[row.t_utc_ns] = row.value
        elif temperature_field is not None and row.field == temperature_field:
            temperatures[row.t_utc_ns] = row.value
    return [
        _make_point(t_utc_ns, magnitude, temperatures.get(t_utc_ns))
        for t_utc_ns, magnitude in sorted(magnitudes.items())
    ]


# --- The reader --------------------------------------------------------------------------


def _header_value(text: str, what: str) -> str:
    """A header value must be ASCII without a line break, or `http.client` refuses it."""
    if not text.isascii() or "\r" in text or "\n" in text:
        raise ConfigError(f"{what} must be ASCII text without a line break")
    return text


class SqmInfluxReader:
    """Read the SQM-LE from InfluxDB and produce `ReferenceRecord`s. This is the `influx` source.

    Args:
        config: The `[sqm]` section, with the table `[sqm.influx]`.
        clock: The time source for the age of a point, the event times, and the waits.
        station_id: The station ID from the configuration, for the record.
        profile_id: The profile ID, for the record.
        token: The token of version 2, after `resolve_credential` read it.
        password: The password of version 1, after `resolve_credential` read it.
        opener: Replaces the HTTP client, for a test. `make_opener` is the default.
        on_event: Receives `sqm.read_failed` for the first failure of a streak, and
            `sqm.recovered` when reads work again.

    Raises:
        ValueError: `config` has no `[sqm.influx]` table.
        ConfigError: The token or the user name holds a character that a header cannot carry.
    """

    def __init__(
        self,
        config: SqmConfig,
        *,
        clock: Clock,
        station_id: str,
        profile_id: str,
        token: str | None = None,
        password: str | None = None,
        opener: urllib.request.OpenerDirector | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        influx = config.influx
        if influx is None:
            raise ValueError('the "influx" source needs the [sqm.influx] table')
        self._cfg = config
        self._influx = influx
        self._clock = clock
        self._station_id = station_id
        self._profile_id = profile_id
        self._on_event = on_event
        self._timeout_s = influx.timeout_s
        self._max_age_ns = round(influx.max_age_s * NS_PER_S)
        self._failures = 0
        self._last_t_utc_ns: int | None = None
        self._opener = opener if opener is not None else make_opener(verify_tls=influx.verify_tls)
        self._headers = {"User-Agent": "seeingmon"}
        if influx.version == 2:
            self._headers["Content-Type"] = "application/vnd.flux"
            self._headers["Accept"] = "application/csv"
            if token:
                self._headers["Authorization"] = "Token " + _header_value(token, "the token")
        else:
            self._headers["Accept"] = "application/json"
            if influx.username and password:
                raw = f"{influx.username}:{password}".encode()
                self._headers["Authorization"] = "Basic " + base64.b64encode(raw).decode("ascii")
        self._private = self._private_pattern(token, password)

    # --- What the reader exposes -----------------------------------------------------------

    @property
    def failures(self) -> int:
        """The number of failed polls in a row."""
        return self._failures

    @property
    def delay_s(self) -> float:
        """The wait before the next poll: the poll interval, or the backoff after a failure."""
        return backoff_delay_s(self._cfg, self._failures)

    def close(self) -> None:
        """Do nothing. The reader keeps no connection, and the method completes the interface."""

    def run(
        self, should_stop: Callable[[], bool], on_record: Callable[[ReferenceRecord], None]
    ) -> None:
        """Poll until `should_stop()` returns true. Each new record goes to `on_record`.

        The loop sleeps on the clock in slices of one second, so a stop request takes effect
        within a second.
        """
        try:
            run_polling(self, self._clock, should_stop, on_record)
        finally:
            self.close()

    def poll(self) -> ReferenceRecord | None:
        """Ask InfluxDB for the newest point once.

        Returns the record, or `None` when the read failed or the point repeats the time stamp of
        the last record. A failure never raises. It counts toward the backoff (`delay_s`) and
        reports an event for the first failure of a streak. A repeated point is not a failure.
        """
        try:
            reading = self.read()
        except InfluxQueryError as error:
            self._fail(error.cause, str(error))
            return None
        except Exception as error:  # a bug must count as a failure, so that `health` shows it
            self._fail(InfluxQueryError.cause, f"an unexpected {type(error).__name__}")
            return None
        if self._failures:
            self._emit(
                "info", "sqm.recovered", "The SQM-LE answers again.", failures=self._failures
            )
            self._failures = 0
        if reading.t_utc_ns == self._last_t_utc_ns:
            return None
        self._last_t_utc_ns = reading.t_utc_ns
        return self._record(InfluxPoint(reading.t_utc_ns, reading.magnitude, reading.temperature_c))

    def _fail(self, cause: str, problem: str) -> None:
        """Count a failed poll, and report the first failure of a streak."""
        self._failures += 1
        if self._failures == 1:
            self._emit(
                "warning",
                "sqm.read_failed",
                f"The SQM-LE did not give a usable reading: {problem}.",
                cause=cause,
            )
        _log.debug("the SQM-LE read from InfluxDB failed: %s", cause)

    def read(self) -> InfluxReading:
        """Ask InfluxDB for the newest point, and check its age. A poll and a probe use it.

        Raises an `InfluxQueryError` for a failure: `InfluxUnreachableError`,
        `InfluxUnauthorizedError`, `InfluxBadRequestError`, `InfluxParseError`,
        `InfluxNoDataError` (no point in the lookback window), or `InfluxStaleError` (the newest
        point is older than `max_age_s`). A message never holds a setting of the installation: the
        words of the server have each setting replaced by a marker.
        """
        points = self._query(self._request(None), MAX_REPLY_BYTES)
        if not points:
            raise InfluxNoDataError(
                f"InfluxDB holds no reading of the SQM-LE from the last "
                f"{self._influx.lookback_s:g} s"
            )
        point = max(points, key=lambda candidate: candidate.t_utc_ns)
        age_ns = self._clock.utc_ns() - point.t_utc_ns
        if age_ns > self._max_age_ns:
            raise InfluxStaleError(
                f"the newest reading is {age_ns / NS_PER_S:.0f} s old, and max_age_s is "
                f"{self._influx.max_age_s:g} s"
            )
        return InfluxReading(
            point.t_utc_ns, point.magnitude, point.temperature_c, age_ns / NS_PER_S
        )

    def read_range(self, start_ns: int, end_ns: int) -> list[ReferenceRecord]:
        """Read the points from `start_ns` up to `end_ns` as records, the oldest first.

        The records have the same fields as the records of `poll`. The call changes neither the
        failure count nor the last time stamp, and it reports no event. The offset fit of
        `seeingmon.survey.sqm_fit` can pair these readings with the survey sky brightness of a
        night that the live reader missed. A range of more than `MAX_RANGE_POINTS` points is an
        error: ask for a shorter range. Raises `ValueError` when the range is empty, and an
        `InfluxQueryError` for a failure, as `read` does.
        """
        if end_ns <= start_ns:
            raise ValueError("end_ns must be greater than start_ns")
        points = self._query(self._request((start_ns, end_ns)), MAX_RANGE_BYTES)
        if len(points) > MAX_RANGE_POINTS:
            raise InfluxParseError(
                f"the range holds more than {MAX_RANGE_POINTS} points; ask for a shorter range"
            )
        return [self._record(point) for point in sorted(points, key=lambda p: p.t_utc_ns)]

    # --- The request -----------------------------------------------------------------------

    def _request(self, window: tuple[int, int] | None) -> urllib.request.Request:
        influx = self._influx
        if influx.version == 1:
            query = build_influxql(influx, window=window)
            parameters = {"db": influx.database or "", "epoch": "ns", "q": query}
            url = f"{influx.endpoint}/query?{urllib.parse.urlencode(parameters)}"
            return urllib.request.Request(url, headers=self._headers, method="GET")
        url = f"{influx.endpoint}/api/v2/query?{urllib.parse.urlencode({'org': influx.org or ''})}"
        program = build_flux(influx, window=window).encode("utf-8")
        return urllib.request.Request(url, data=program, headers=self._headers, method="POST")

    def _query(self, request: urllib.request.Request, limit_bytes: int) -> list[InfluxPoint]:
        """Send a request and parse the reply into points."""
        text = self._send(request, limit_bytes)
        parse = parse_influxql_points if self._influx.version == 1 else parse_flux_points
        try:
            return parse(text, self._influx.field, self._influx.temperature_field)
        except InfluxReportedError as error:
            raise InfluxBadRequestError(
                f"InfluxDB reported an error: {self._clean(error.server_message)}"
            ) from None

    def _send(self, request: urllib.request.Request, limit_bytes: int) -> str:
        data: bytes
        try:
            with self._opener.open(request, timeout=self._timeout_s) as response:
                data = response.read(limit_bytes + 1)
        except urllib.error.HTTPError as error:
            message = error_reply_message(error, limit=MAX_SERVER_TEXT)
            raise self._status_error(error.code, message) from None
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
            OSError,
        ) as error:
            reason = error.reason if isinstance(error, urllib.error.URLError) else error
            kind = type(reason if isinstance(reason, BaseException) else error).__name__
            raise InfluxUnreachableError(f"InfluxDB did not answer ({kind})") from None
        except ValueError:
            raise InfluxBadRequestError(
                "the request could not be sent; check the characters of the token and the user name"
            ) from None
        if len(data) > limit_bytes:
            raise InfluxParseError("the reply is too large")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise InfluxParseError("the reply is not text") from None

    def _status_error(self, status: int, message: str) -> InfluxQueryError:
        """The error for an HTTP status, with the cleaned server message and a hint."""
        text = f"InfluxDB answered HTTP {status}"
        if message:
            text += f": {self._clean(message)}"
        if status >= 500 or status in (408, 429):
            return InfluxUnreachableError(text)
        if 300 <= status < 400:
            hint = "the server redirected the query; set the endpoint to the final address"
        else:
            hint = _HINTS.get(status, "check the settings in [sqm.influx]")
        text += f" ({hint})"
        if status in (401, 403):
            return InfluxUnauthorizedError(text)
        return InfluxBadRequestError(text)

    # --- Records and events ----------------------------------------------------------------

    def _record(self, point: InfluxPoint) -> ReferenceRecord:
        return ReferenceRecord(
            station_id=self._station_id,
            t_utc_ns=point.t_utc_ns,
            profile_id=self._profile_id,
            provenance={"reader": READER_VERSION, "api": str(self._influx.version)},
            instrument=self._cfg.instrument,
            source="fixed",
            value_mag_arcsec2=point.magnitude,
            temperature_c=point.temperature_c,
            altitude_deg=self._cfg.altitude_deg,
            azimuth_deg=self._cfg.azimuth_deg,
        )

    def _emit(self, level: str, kind: str, message: str, **detail: object) -> None:
        emit(
            self._on_event,
            HardwareEvent(level, kind, message, self._clock.utc_ns(), detail or None),
        )

    # --- Privacy ---------------------------------------------------------------------------

    def _private_pattern(self, token: str | None, password: str | None) -> re.Pattern[str] | None:
        """A pattern for every setting of the installation, in each way that a server can echo it.

        The server may repeat a name that it did not find, such as a bucket, in its message. The
        pattern finds those names so that `_scrub` can replace them. A short value (up to two
        characters) matches as a whole word only, because it would otherwise garble the message.
        """
        influx = self._influx
        parts = urllib.parse.urlsplit(influx.endpoint)
        values = {
            influx.endpoint,
            parts.netloc,
            parts.hostname or "",
            influx.org or "",
            influx.bucket or "",
            influx.database or "",
            influx.retention_policy or "",
            influx.username or "",
            influx.measurement,
            influx.field,
            influx.temperature_field or "",
            token or "",
            password or "",
            *influx.tags,
            *influx.tags.values(),
        }
        forms: set[str] = set()
        for value in values:
            if not value:
                continue
            forms |= {
                value,
                " ".join(value.split()),  # the server text has its white space collapsed
                _escape_influxql(value, '"'),
                _escape_influxql(value, "'"),
                flux_string(value)[1:-1],
                urllib.parse.quote(value, safe=""),
                urllib.parse.quote_plus(value),
            }
        forms.discard("")
        ordered = sorted(forms, key=lambda form: (-len(form), form))
        long_forms = [re.escape(form) for form in ordered if len(form) > 2]
        short_forms = [re.escape(form) for form in ordered if len(form) <= 2]
        alternatives = []
        if long_forms:
            alternatives.append("|".join(long_forms))
        if short_forms:
            alternatives.append(r"(?<!\w)(?:" + "|".join(short_forms) + r")(?!\w)")
        return re.compile("|".join(alternatives)) if alternatives else None

    def _scrub(self, text: str) -> str:
        """Replace every setting of the installation in `text` with a marker."""
        if self._private is None:
            return text
        return self._private.sub(REDACTED, text)

    def _clean(self, text: str) -> str:
        """The words of the server, with every setting replaced, on one line and shortened.

        The reader cleans the whole text first and shortens it afterward, so the cut can never
        leave the start of a setting in the message.
        """
        return _one_line(self._scrub(text))
