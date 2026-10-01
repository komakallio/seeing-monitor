"""Read-only access to the results, for the REST API.

`StoreData` reads records through a `StoreSource`: a `StoreReader`, which opens the database
read-only, or a `ReopeningReader` (see `seeingmon.services.web.reader`), which opens it when it can.
This module never writes. It turns the stored rows into the JSON that the API serves, and it builds
the history of a record type.

**The record JSON.** A record is the `Record.to_row` dict, plus `t_utc`, the start time as an ISO
8601 UTC string (the integer `t_utc_ns` is too large for a JavaScript number to hold exactly).
A field that the settings withhold (`withhold_fields`) is `null` with the `quality` note
`withheld`. The text of an event goes through `seeingmon.services.web.privacy.scrub_text`.

**History.** `history` returns the records of a type between two times, oldest first. A range
is `from` (included) and `to` (excluded). Without `step`, each record is one item. With a step of
1 minute, 10 minutes, or 1 hour, the items are buckets that start at a multiple of the step in UTC:

- A numeric field is the mean of its values in the bucket, and `null` when the bucket has none.
  The fields that name a setting (`stream_id`, `gain`, `exposure_us`, and `revision`) keep the
  value of the last record.
- A list of flags is the union of the flags of the bucket, so a flag that applied to any record
  shows (the worst state). A boolean is true when any record is true.
- A list of numbers (a spectrum, the attitude) is `null`, and `quality` says that arrays are not
  aggregated when a request names the field.
- `n_samples` is the number of records in the bucket.
- `quality` says why a field is `null`. It keeps the first reason that a record of the bucket gave
  for that field, as the stored records do, and it has no entry for a field that no record
  explained.

A page holds at most `limit` items. `next_cursor` is `null` after the last page. Otherwise, pass
it back as `cursor` to read the next page. A page reads at most `max_scan_rows` rows.

**Fields.** Without `fields`, an item holds every field except the arrays of numbers. With
`fields`, it holds the named fields, `t_utc_ns`, `t_utc`, and `quality` (with the reasons for the
named fields only, or `null`).
"""

from __future__ import annotations

import base64
import binascii
import functools
import json
import math
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import Any, Protocol, TypeVar, get_args, get_origin

from seeingmon.clock import NS_PER_S, Clock, iso_to_utc_ns, utc_ns_to_iso
from seeingmon.config import ConfigError
from seeingmon.records.base import RECORD_TYPES, Record, field_specs, get_record_type
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.privacy import scrub_json, scrub_text
from seeingmon.sinks.base import StoredRow
from seeingmon.store.db import DEFAULT_LIMIT, StoreError

T = TypeVar("T")

MAX_T_NS = 4_102_444_800 * NS_PER_S  # 2100-01-01T00:00:00Z
WITHHELD = "withheld"
NOT_AGGREGATED = "arrays are not aggregated"
SETTING_FIELDS = frozenset({"revision", "stream_id", "gain", "exposure_us"})
EVENT_LEVEL_RANK = {"info": 0, "warning": 1, "error": 2}
ALWAYS_PRESENT = ("t_utc_ns", "t_utc")
MAX_CURSOR_CHARS = 128
AGGREGATE_CHUNK = 2000
_BASE64URL = re.compile(r"[A-Za-z0-9_-]+")
_FIELD_NAME = re.compile(r"[a-z][a-z0-9_]{0,62}")


class StoreSource(Protocol):
    """The reads that `StoreData` needs. `StoreReader` and `ReopeningReader` provide them."""

    def latest(self, record_type: str, *, station_id: str | None = None) -> StoredRow | None: ...

    def range(
        self,
        record_type: str,
        start_ns: int,
        end_ns: int,
        limit: int = DEFAULT_LIMIT,
        *,
        station_id: str | None = None,
        descending: bool = False,
        all_revisions: bool = False,
    ) -> list[StoredRow]: ...


class DataError(Exception):
    """Base class for the failures of the data layer."""


class InvalidQueryError(DataError, ValueError):
    """A parameter of a query is out of range or malformed. The message names the parameter."""


class StoreUnavailableError(DataError):
    """The store cannot be read: the file is missing, locked, or damaged."""


class Step(StrEnum):
    """The width of a history bucket. `raw` returns the stored records without aggregation."""

    RAW = "raw"
    MINUTE = "1m"
    TEN_MINUTES = "10m"
    HOUR = "1h"

    @property
    def seconds(self) -> int:
        """The width of the bucket in seconds, or 0 for `raw`."""
        return {"raw": 0, "1m": 60, "10m": 600, "1h": 3600}[self.value]


@dataclass(frozen=True, slots=True)
class TimeRange:
    """A range of times in nanoseconds: `start_ns` is included and `end_ns` is excluded."""

    start_ns: int
    end_ns: int


@dataclass(frozen=True, slots=True)
class Page:
    """One page of items and the cursor of the next page, or `None` after the last page."""

    items: list[dict[str, Any]]
    next_cursor: str | None
    time_range: TimeRange
    step: Step = Step.RAW


# --- Cursors and times -----------------------------------------------------------------------


def encode_token(payload: Mapping[str, Any]) -> str:
    """An opaque, URL-safe text for a small JSON object. A cursor is one."""
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_token(text: str) -> dict[str, Any]:
    """The object in a token. Raises `InvalidQueryError` for other text."""
    if not text or len(text) > MAX_CURSOR_CHARS or not _BASE64URL.fullmatch(text):
        raise InvalidQueryError("cursor is not a cursor that this server gave")
    try:
        data = json.loads(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)))
    except (binascii.Error, ValueError):
        raise InvalidQueryError("cursor is not a cursor that this server gave") from None
    if not isinstance(data, dict):
        raise InvalidQueryError("cursor is not a cursor that this server gave")
    return data


def encode_cursor(t_ns: int) -> str:
    """The opaque cursor for the page that starts or ends at `t_ns`."""
    return encode_token({"t": t_ns})


def decode_cursor(text: str) -> int:
    """The time in a cursor. Raises `InvalidQueryError` for anything else."""
    data = decode_token(text)
    value = data.get("t") if len(data) == 1 else None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_T_NS:
        raise InvalidQueryError("cursor is not a cursor that this server gave")
    return value


def parse_time(text: str, name: str) -> int:
    """An ISO 8601 UTC time as nanoseconds. Raises `InvalidQueryError`."""
    try:
        value = iso_to_utc_ns(text)
    except ValueError:
        raise InvalidQueryError(
            f"{name} must be an ISO 8601 UTC time, such as 2026-10-01T21:00:00Z"
        ) from None
    if not 0 <= value <= MAX_T_NS:
        raise InvalidQueryError(f"{name} must lie between 1970 and 2100")
    return value


def iso(t_ns: int) -> str:
    """The API form of a time: ISO 8601 UTC with six fractional digits."""
    return utc_ns_to_iso(t_ns, digits=6)


# --- Aggregation -----------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return float(f"{math.fsum(values) / len(values):.12g}")


@functools.cache
def array_fields(cls: type[Record]) -> frozenset[str]:
    """The fields that hold a list of numbers, such as a spectrum or the attitude."""
    return frozenset(
        spec.name
        for spec in field_specs(cls)
        if get_origin(spec.annotation) is list and get_args(spec.annotation)[0] in (int, float)
    )


def aggregate_rows(
    cls: type[Record], rows: Sequence[Mapping[str, Any]], bucket_start_ns: int
) -> dict[str, Any]:
    """Combine the rows of one bucket into one item. See the module documentation for the rules."""
    specs = field_specs(cls)
    out: dict[str, Any] = {}
    for spec in specs:
        name = spec.name
        values = [row[name] for row in rows if row.get(name) is not None]
        if name == "t_utc_ns":
            out[name] = bucket_start_ns
        elif name == "quality":
            continue
        elif name in ("station_id", "profile_id", "provenance") or name in SETTING_FIELDS:
            out[name] = rows[-1].get(name)
        elif spec.kind in ("float", "int"):
            out[name] = _mean(values) if values else None
        elif spec.kind == "bool":
            out[name] = any(values) if values else None
        elif spec.kind == "json" and get_origin(spec.annotation) is list:
            if spec.codes is not None:
                out[name] = sorted({code for value in values for code in value})
            else:
                out[name] = None
        else:
            out[name] = values[-1] if values else None
    quality: dict[str, str] = {}
    for spec in specs:
        if spec.name == "quality" or out.get(spec.name) is not None or not spec.nullable:
            continue
        for row in rows:
            reason = (row.get("quality") or {}).get(spec.name)
            if reason:
                quality[spec.name] = reason
                break
    out["quality"] = quality or None
    out["n_samples"] = len(rows)
    return out


# --- The store -------------------------------------------------------------------------------


class StoreData:
    """The reads of the API. It has no write path and keeps no state but its settings."""

    def __init__(
        self,
        store: StoreSource,
        settings: WebSettings,
        clock: Clock,
        *,
        station_id: str | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._clock = clock
        self._station_id = station_id
        self._withheld = frozenset(settings.withhold_fields)
        self.aggregate_chunk = AGGREGATE_CHUNK  # rows per read while a bucket builds
        self._check_withheld()

    def _check_withheld(self) -> None:
        known: dict[str, bool] = {}
        for cls in RECORD_TYPES.values():
            for spec in field_specs(cls):
                known[spec.name] = known.get(spec.name, True) and spec.nullable
        for name in sorted(self._withheld):
            if name not in known:
                raise ConfigError(f"[web] withhold_fields names {name}, which no record has")
            if not known[name]:
                raise ConfigError(f"[web] withhold_fields names {name}, which cannot be null")

    @property
    def settings(self) -> WebSettings:
        return self._settings

    def now_ns(self) -> int:
        """The time of the server, from its clock."""
        return self._clock.utc_ns()

    @staticmethod
    def record_class(record_type: str) -> type[Record]:
        """The class of a record type. Raises `KeyError` for an unknown name."""
        return get_record_type(record_type)

    def _call(self, read: Callable[[], T]) -> T:
        try:
            return read()
        except (sqlite3.Error, StoreError, OSError):
            raise StoreUnavailableError("the store cannot be read") from None

    # --- Parameters ------------------------------------------------------------------------

    def resolve_range(self, start: str | None, end: str | None) -> TimeRange:
        """The range of a query. A missing `to` is now, and a missing `from` is the default span."""
        end_ns = parse_time(end, "to") if end else self.now_ns() + 1
        if start:
            start_ns = parse_time(start, "from")
        else:
            span_ns = round(self._settings.paging.default_range_hours * 3600 * NS_PER_S)
            start_ns = max(0, end_ns - span_ns)
        if start_ns >= end_ns:
            raise InvalidQueryError("from must be earlier than to")
        return TimeRange(start_ns, end_ns)

    def resolve_limit(self, limit: int | None) -> int:
        """The page size: the default when absent, and an error when out of bounds."""
        paging = self._settings.paging
        if limit is None:
            return paging.default_limit
        if not 1 <= limit <= paging.max_limit:
            raise InvalidQueryError(f"limit must be between 1 and {paging.max_limit}")
        return limit

    @staticmethod
    def parse_fields(cls: type[Record], text: str | None) -> frozenset[str] | None:
        """The fields that a query asks for, or `None` for the default set."""
        if text is None:
            return None
        names = [name.strip() for name in text.split(",")]
        known = {spec.name for spec in field_specs(cls)}
        if not names or len(names) > len(known) or any(not _FIELD_NAME.fullmatch(n) for n in names):
            raise InvalidQueryError("fields must be a comma-separated list of field names")
        unknown = sorted(set(names) - known)
        if unknown:
            raise InvalidQueryError(f"fields names unknown fields: {', '.join(unknown)}")
        return frozenset(names)

    # --- Record JSON -----------------------------------------------------------------------

    def _public(self, cls: type[Record], values: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(values)
        row["t_utc"] = iso(int(row["t_utc_ns"]))
        withheld = sorted(name for name in self._withheld if name in row)
        if withheld:
            quality = dict(row.get("quality") or {})
            for name in withheld:
                row[name] = None
                quality[name] = WITHHELD
            row["quality"] = quality
        if row.get("quality"):
            row["quality"] = scrub_json(row["quality"])
        if row.get("provenance"):
            row["provenance"] = scrub_json(row["provenance"])
        if cls.record_type == "event":
            row["message"] = scrub_text(str(row["message"]))
            row["detail"] = scrub_json(row.get("detail"))
        return row

    @staticmethod
    def _project(
        cls: type[Record], row: dict[str, Any], wanted: frozenset[str] | None
    ) -> dict[str, Any]:
        if wanted is None:
            heavy = array_fields(cls)
            return {name: value for name, value in row.items() if name not in heavy}
        keep = wanted | set(ALWAYS_PRESENT) | {"n_samples", "quality"}
        shaped = {name: value for name, value in row.items() if name in keep}
        if "quality" in shaped:
            reasons = shaped["quality"] or {}
            quality = {name: why for name, why in reasons.items() if name in wanted}
            shaped["quality"] = quality or None
        return shaped

    # --- Reads -----------------------------------------------------------------------------

    def latest(self, record_type: str) -> dict[str, Any] | None:
        """The newest record of a type as JSON, or `None` when the store holds none."""
        cls = get_record_type(record_type)
        stored = self._call(lambda: self._store.latest(record_type, station_id=self._station_id))
        return None if stored is None else self._public(cls, stored.values)

    def history(
        self,
        record_type: str,
        *,
        time_range: TimeRange,
        step: Step,
        limit: int,
        cursor: str | None = None,
        wanted: frozenset[str] | None = None,
    ) -> Page:
        """One page of the history of a record type. See the module documentation."""
        cls = get_record_type(record_type)
        start_ns = time_range.start_ns
        if cursor is not None:
            start_ns = min(max(decode_cursor(cursor), start_ns), time_range.end_ns)
        if step is Step.RAW:
            rows, next_ns = self._scan(
                record_type, start_ns, time_range.end_ns, descending=False, limit=limit
            )
            items = [self._project(cls, self._public(cls, row), wanted) for row in rows]
        else:
            items, next_ns = self._aggregate(
                cls, start_ns, time_range.end_ns, step.seconds * NS_PER_S, limit, wanted
            )
        return Page(items, None if next_ns is None else encode_cursor(next_ns), time_range, step)

    def events(
        self,
        *,
        time_range: TimeRange,
        limit: int,
        cursor: str | None = None,
        min_level: str = "info",
        kind: str | None = None,
        descending: bool = True,
    ) -> Page:
        """One page of events, newest first by default, filtered by level and by kind prefix."""
        cls = get_record_type("event")
        rank = EVENT_LEVEL_RANK[min_level]
        start_ns, end_ns = time_range.start_ns, time_range.end_ns
        if cursor is not None:
            at = decode_cursor(cursor)
            if descending:
                end_ns = min(max(at, start_ns), end_ns)
            else:
                start_ns = min(max(at, start_ns), end_ns)

        def accept(row: Mapping[str, Any]) -> bool:
            if EVENT_LEVEL_RANK.get(str(row["level"]), 0) < rank:
                return False
            return kind is None or str(row["kind"]).startswith(kind)

        filtered = rank > 0 or kind is not None
        rows, next_ns = self._scan(
            "event",
            start_ns,
            end_ns,
            descending=descending,
            limit=limit,
            accept=accept if filtered else None,
        )
        items = [self._public(cls, row) for row in rows]
        return Page(items, None if next_ns is None else encode_cursor(next_ns), time_range)

    # --- Scans -----------------------------------------------------------------------------

    def _scan(
        self,
        record_type: str,
        start_ns: int,
        end_ns: int,
        *,
        descending: bool,
        limit: int,
        accept: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> tuple[list[Mapping[str, Any]], int | None]:
        """Read up to `limit` rows in time order, with the time that the next page continues at.

        The time is where an ascending page starts (included) or where a descending page ends
        (excluded), and `None` after the last page.
        """
        chunk = limit + 1 if accept is None else min(max(limit * 4, 200), 5000)
        found: list[Mapping[str, Any]] = []
        scanned = 0
        low, high = start_ns, end_ns
        while low < high:
            rows = self._call(
                partial(
                    self._store.range,
                    record_type,
                    low,
                    high,
                    chunk,
                    station_id=self._station_id,
                    descending=descending,
                )
            )
            for stored in rows:
                scanned += 1
                values = stored.values
                if accept is not None and not accept(values):
                    continue
                if len(found) == limit:
                    return found, (int(found[-1]["t_utc_ns"]) if descending else _t(values))
                found.append(values)
            if len(rows) < chunk:
                return found, None
            last_ns = _t(rows[-1].values)
            if scanned >= self._settings.paging.max_scan_rows:
                return found, last_ns if descending else last_ns + 1
            if descending:
                high = last_ns
            else:
                low = last_ns + 1
        return found, None

    def _aggregate(
        self,
        cls: type[Record],
        start_ns: int,
        end_ns: int,
        step_ns: int,
        limit: int,
        wanted: frozenset[str] | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        items: list[dict[str, Any]] = []
        pending: list[Mapping[str, Any]] = []
        pending_bucket: int | None = None
        scanned = 0
        chunk = self.aggregate_chunk
        low = start_ns

        def finish(bucket: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            item = aggregate_rows(cls, rows, bucket)
            if wanted is not None:
                arrays = sorted(wanted & array_fields(cls))
                if arrays:
                    item["quality"] = {
                        **(item["quality"] or {}),
                        **dict.fromkeys(arrays, NOT_AGGREGATED),
                    }
            return self._project(cls, self._public(cls, item), wanted)

        while low < end_ns:
            rows = self._call(
                partial(
                    self._store.range,
                    cls.record_type,
                    low,
                    end_ns,
                    chunk,
                    station_id=self._station_id,
                )
            )
            for stored in rows:
                values = stored.values
                bucket = _t(values) // step_ns * step_ns
                if pending_bucket is not None and bucket != pending_bucket:
                    items.append(finish(pending_bucket, pending))
                    pending = []
                    if len(items) == limit:
                        return items, bucket
                pending_bucket = bucket
                pending.append(values)
            scanned += len(rows)
            if len(rows) < chunk:
                break
            low = _t(rows[-1].values) + 1
            if scanned >= self._settings.paging.max_scan_rows and pending_bucket is not None:
                if not items:  # one bucket exceeds the budget, so emit it and move on
                    items.append(finish(pending_bucket, pending))
                    return items, pending_bucket + step_ns
                return items, pending_bucket  # the next page rebuilds the incomplete bucket
        if pending and pending_bucket is not None:
            items.append(finish(pending_bucket, pending))
        return items, None


def _t(values: Mapping[str, Any]) -> int:
    return int(values["t_utc_ns"])
