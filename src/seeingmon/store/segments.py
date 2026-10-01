"""Segment files of per-frame metrics.

A fast stream produces up to a few hundred metric rows a second, which SQLite handles poorly
and an SD card handles worse. `SegmentWriter` appends the rows of the `frame` record (44 bytes
each) to binary segment files of 10 minutes of frame time instead. It implements the
`MetricsWriter` protocol of `seeingmon.analysis.base`.

**Layout on disk.** A segment lives under a directory for the UTC date of its start:
`<root>/2026/10/01/frame_20261001T213000Z_s3.seg`. The name holds the record type, the start of
the 10-minute slot, and the stream ID (`s3`). A second file for the same slot and stream, which a
restart causes, gets a part number (`_s3_1`). A new slot or a new stream starts a new file.

**File format** (little-endian). The file starts with the magic bytes `\\x89SMSEG\\r\\n`, a `u16`
format version, a `u16` of flags (zero), and a `u32` length of the header. The header follows as
UTF-8 JSON, then a `u32` CRC-32 of the header bytes. Packed rows follow until the end of the file.
The JSON header holds the format version, the record type, the start of the slot, the time of the
first row, the time of creation, the row size, the row layout (`name` and NumPy type string for
each column, from `seeingmon.records.segments.segment_layout`), and the per-segment fields of the
record (`station_id`, `profile_id`, `provenance`, `revision`, `quality`, `stream_id`). The format
uses no pickle. The layout in the header lets a reader open an old file after the record gains a
column.

**Crash safety.** The writer creates the file under the name `<name>.seg.part`, writes the
header, and appends rows. It flushes to the operating system after each call, so a crash of the
process loses nothing, and it forces the file to disk at an interval, so a power cut loses
at most that interval. It renames the file to `<name>.seg` when it closes the segment. A crash
leaves a `.part` file that ends at the last complete row, or in a torn row. `read_segment` and
`SegmentReader` read such a file up to its last complete row, and they drop a torn row and any
zero-filled tail that a power cut can leave. `recover_orphans` repairs the files at the start of
the process that owns the writer: it cuts each file after its last complete row and renames it.
Run it before the first write, and never while another process writes.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
import struct
import threading
import zlib
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Self

import numpy as np
import numpy.typing as npt

import seeingmon
from seeingmon.clock import NS_PER_S, Clock, utc_ns_to_datetime
from seeingmon.records.base import Record, get_record_type
from seeingmon.records.segments import (
    from_segment,
    segment_dtype,
    segment_header_fields,
    segment_layout,
)
from seeingmon.store.config import SegmentsConfig

MAGIC = b"\x89SMSEG\r\n"
FORMAT_VERSION = 1
SEGMENT_SUFFIX = ".seg"
OPEN_SUFFIX = ".part"
CORRUPT_SUFFIX = ".corrupt"
DEFAULT_RECORD_TYPE = "frame"

_PREFIX = struct.Struct("<8sHHI")  # magic, format version, flags, header length
_CRC = struct.Struct("<I")
_MAX_HEADER_BYTES = 1 << 20
_REQUIRED_KEYS = ("record_type", "start_utc_ns", "layout", "row_bytes", "header")
_MAX_PARTS = 10_000
_STAMP = "%Y%m%dT%H%M%SZ"
_NAME = re.compile(
    r"(?P<type>[a-z][a-z0-9_]*?)_(?P<stamp>\d{8}T\d{6}Z)_s(?P<stream>\d+)"
    r"(?:_(?P<part>\d+))?\.seg(?P<open>\.part)?"
)


class SegmentFormatError(Exception):
    """The file is not a readable segment: wrong magic, a newer format, or a damaged header."""


@dataclass(frozen=True, slots=True)
class SegmentInfo:
    """What the name of a segment file says. `is_open` is true for a `.part` file."""

    path: Path
    record_type: str
    slot_start_utc_ns: int
    stream_id: int
    part: int
    is_open: bool


@dataclass(frozen=True, slots=True)
class SegmentData:
    """The contents of a segment file, up to its last complete row.

    `header` is the JSON header. `rows` is a structured array with the layout of the file.
    `closed` is false for a file that the writer did not close. `dropped_bytes` counts the bytes
    at the end that hold no complete row: a torn row or a zero-filled tail. `data_offset` is the
    byte at which the rows start.
    """

    path: Path
    header: dict[str, Any]
    rows: npt.NDArray[Any]
    closed: bool
    dropped_bytes: int
    data_offset: int

    @property
    def record_type(self) -> str:
        return str(self.header["record_type"])

    @property
    def record_header(self) -> dict[str, Any]:
        """The per-segment fields of the record: station, profile, provenance, stream ID, ..."""
        header = self.header["header"]
        assert isinstance(header, dict)
        return header

    @property
    def stream_id(self) -> int:
        return int(self.record_header["stream_id"])

    @property
    def start_utc_ns(self) -> int:
        """The time of the first row that the writer put in the segment."""
        return int(self.header["start_utc_ns"])

    def __len__(self) -> int:
        return len(self.rows)

    def records(self) -> list[Record]:
        """Build a record for each row, with the current declaration of the record type."""
        cls = get_record_type(self.record_type)
        return from_segment(cls, self.record_header, upgrade_rows(self.rows, segment_dtype(cls)))


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """What `recover_segment` did to one file.

    `action` is `renamed` (the file ends at its last complete row and has its final name),
    `removed` (the file held no complete row), or `quarantined` (the header is damaged, so the
    file got the `.corrupt` suffix and stays for a person to look at). `result` is the path
    after the action, or `None` for `removed`.
    """

    path: Path
    result: Path | None
    action: str
    rows: int
    dropped_bytes: int


def upgrade_rows(rows: npt.NDArray[Any], dtype: np.dtype[Any]) -> npt.NDArray[Any]:
    """Return the rows in a newer layout, with zeros (NaN for floats) in the new columns.

    A record type gains columns at the end, so an old segment has fewer columns than the
    current layout. The rows come back unchanged when the layout already matches.
    """
    if rows.dtype == dtype:
        return rows
    out = np.zeros(len(rows), dtype=dtype)
    have = rows.dtype.names or ()
    for name in dtype.names or ():
        if name in have:
            out[name] = rows[name]
        elif out.dtype[name].kind == "f":
            out[name] = np.nan
    return out


# --- names and directories ----------------------------------------------------------------------


def segment_name(record_type: str, slot_start_utc_ns: int, stream_id: int, part: int = 0) -> str:
    """The file name of a closed segment, for example `frame_20261001T213000Z_s3.seg`."""
    stamp = utc_ns_to_datetime(slot_start_utc_ns).strftime(_STAMP)
    suffix = f"_{part}" if part else ""
    return f"{record_type}_{stamp}_s{stream_id}{suffix}{SEGMENT_SUFFIX}"


def segment_directory(root: Path, slot_start_utc_ns: int) -> Path:
    """The directory for the UTC date of a slot: `<root>/YYYY/MM/DD`."""
    moment = utc_ns_to_datetime(slot_start_utc_ns)
    return root / f"{moment:%Y}" / f"{moment:%m}" / f"{moment:%d}"


def parse_segment_name(path: Path) -> SegmentInfo | None:
    """Read the name of a segment file. Returns `None` for a file that is not a segment."""
    match = _NAME.fullmatch(path.name)
    if match is None:
        return None
    stamp = datetime.strptime(match["stamp"], _STAMP).replace(tzinfo=UTC)
    return SegmentInfo(
        path=path,
        record_type=match["type"],
        slot_start_utc_ns=int(stamp.timestamp()) * NS_PER_S,
        stream_id=int(match["stream"]),
        part=int(match["part"] or 0),
        is_open=match["open"] is not None,
    )


def _fsync_directory(directory: Path) -> None:
    """Make a rename durable. Windows cannot open a directory, so it skips the step."""
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


# --- reading -----------------------------------------------------------------------------------


def _read_header(handle: BinaryIO, size: int) -> tuple[dict[str, Any], int]:
    prefix = handle.read(_PREFIX.size)
    if len(prefix) < _PREFIX.size:
        raise SegmentFormatError("the file ends before the header starts")
    magic, version, _flags, length = _PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise SegmentFormatError("the file is not a segment file")
    if version > FORMAT_VERSION:
        raise SegmentFormatError(
            f"the file has format {version}, and this software reads up to {FORMAT_VERSION}"
        )
    end = _PREFIX.size + length + _CRC.size
    if length > _MAX_HEADER_BYTES or end > size:
        raise SegmentFormatError("the file ends inside the header")
    body = handle.read(length)
    (checksum,) = _CRC.unpack(handle.read(_CRC.size))
    if zlib.crc32(body) != checksum:
        raise SegmentFormatError("the header checksum does not match")
    try:
        header = json.loads(body)
        missing = [key for key in _REQUIRED_KEYS if key not in header]
        if missing:
            raise KeyError(", ".join(missing))
        row_bytes = int(header["row_bytes"])
        record_header = header["header"]
        names = [(str(name), str(code)) for name, code in header["layout"]]
    except (ValueError, KeyError, TypeError) as exc:
        raise SegmentFormatError(f"the header is malformed ({exc})") from None
    if not isinstance(record_header, dict) or row_bytes < 1:
        raise SegmentFormatError("the header is malformed")
    try:
        dtype = np.dtype(names)
    except (TypeError, ValueError) as exc:
        raise SegmentFormatError(f"the row layout is malformed ({exc})") from None
    if dtype.itemsize != row_bytes:
        raise SegmentFormatError("the row size does not match the row layout")
    return header, end


def read_segment(path: Path | str) -> SegmentData:
    """Read a segment file up to its last complete row.

    The call works on a file that a writer still appends to: it reads the rows that were
    complete when it started. A torn row at the end, and a zero-filled tail, are dropped and
    counted in `dropped_bytes`. Raises `SegmentFormatError` when the header is unreadable.
    """
    file_path = Path(path)
    with file_path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        header, data_offset = _read_header(handle, size)
        dtype = np.dtype([(str(name), str(code)) for name, code in header["layout"]])
        count, torn = divmod(size - data_offset, dtype.itemsize)
        rows = np.frombuffer(bytearray(handle.read(count * dtype.itemsize)), dtype=dtype)
    keep = count
    if count:
        raw = rows.view(np.uint8).reshape(count, dtype.itemsize)
        filled = np.flatnonzero(raw.any(axis=1))
        keep = int(filled[-1]) + 1 if len(filled) else 0
    return SegmentData(
        path=file_path,
        header=header,
        rows=rows[:keep],
        closed=not file_path.name.endswith(OPEN_SUFFIX),
        dropped_bytes=torn + (count - keep) * dtype.itemsize,
        data_offset=data_offset,
    )


def _day_directories(root: Path) -> list[tuple[int, Path]]:
    """The date directories under a segment root, as `(start of the UTC day, path)`, sorted."""
    found: list[tuple[int, Path]] = []
    if not root.is_dir():
        return found
    for year in sorted(root.iterdir()):
        if not (year.is_dir() and re.fullmatch(r"\d{4}", year.name)):
            continue
        for month in sorted(year.iterdir()):
            if not (month.is_dir() and re.fullmatch(r"\d{2}", month.name)):
                continue
            for day in sorted(month.iterdir()):
                if not (day.is_dir() and re.fullmatch(r"\d{2}", day.name)):
                    continue
                try:
                    start = datetime(int(year.name), int(month.name), int(day.name), tzinfo=UTC)
                except ValueError:
                    continue
                found.append((int(start.timestamp()) * NS_PER_S, day))
    return found


def iter_segments(
    root: Path | str,
    start_utc_ns: int | None = None,
    end_utc_ns: int | None = None,
    *,
    record_type: str = DEFAULT_RECORD_TYPE,
    slot_s: int = 600,
    include_open: bool = True,
) -> Iterator[SegmentInfo]:
    """Yield the segments whose slot overlaps `[start_utc_ns, end_utc_ns)`, oldest first.

    The call reads directory and file names only. Leave a bound out to leave that side open.
    `slot_s` is the segment length that the writer used. A file that the writer still holds
    open (`is_open` is true) comes with `include_open=True`.
    """
    base = Path(root)
    slot_ns = slot_s * NS_PER_S
    day_ns = 86_400 * NS_PER_S
    for day_start, directory in _day_directories(base):
        if end_utc_ns is not None and day_start >= end_utc_ns:
            break
        if start_utc_ns is not None and day_start + day_ns + slot_ns <= start_utc_ns:
            continue
        infos: list[SegmentInfo] = []
        for entry in directory.iterdir():
            info = parse_segment_name(entry)
            if info is None or info.record_type != record_type:
                continue
            if info.is_open and not include_open:
                continue
            if end_utc_ns is not None and info.slot_start_utc_ns >= end_utc_ns:
                continue
            if start_utc_ns is not None and info.slot_start_utc_ns + slot_ns <= start_utc_ns:
                continue
            infos.append(info)
        infos.sort(
            key=lambda item: (item.slot_start_utc_ns, item.stream_id, item.part, item.is_open)
        )
        yield from infos


# --- recovery ----------------------------------------------------------------------------------


def _unused_name(path: Path) -> Path:
    """Return `path`, or the first of `path.1`, `path.2`, ... that does not exist."""
    candidate = path
    number = 0
    while candidate.exists():
        number += 1
        candidate = path.with_name(f"{path.name}.{number}")
    return candidate


def recover_segment(path: Path | str) -> RecoveryReport:
    """Repair one file that a crash left behind.

    The function cuts the file after its last complete row and renames a `.part` file to its
    final name. A file without any complete row is removed. A file with a damaged header gets
    the `.corrupt` suffix, so that no data disappears and a person can look at it. Run it only
    when no writer holds the file.
    """
    file_path = Path(path)
    try:
        data = read_segment(file_path)
    except SegmentFormatError:
        if file_path.stat().st_size < _PREFIX.size:
            file_path.unlink()
            return RecoveryReport(file_path, None, "removed", 0, 0)
        target = _unused_name(file_path.with_name(file_path.name + CORRUPT_SUFFIX))
        os.replace(file_path, target)
        return RecoveryReport(file_path, target, "quarantined", 0, 0)
    if len(data.rows) == 0:
        file_path.unlink()
        return RecoveryReport(file_path, None, "removed", 0, data.dropped_bytes)
    good = data.data_offset + len(data.rows) * data.rows.dtype.itemsize
    with file_path.open("r+b") as handle:
        handle.truncate(good)
        handle.flush()
        os.fsync(handle.fileno())
    final = file_path
    if file_path.name.endswith(OPEN_SUFFIX):
        final = _unused_name(file_path.with_name(file_path.name.removesuffix(OPEN_SUFFIX)))
        os.replace(file_path, final)
        _fsync_directory(final.parent)
    return RecoveryReport(file_path, final, "renamed", len(data.rows), data.dropped_bytes)


def recover_orphans(
    root: Path | str, *, skip: Collection[Path] = (), record_type: str | None = None
) -> list[RecoveryReport]:
    """Repair every `.part` file under a segment root. Call it once, at startup.

    `skip` names files to leave alone, such as the open file of a live writer. Pass
    `record_type` to repair the files of one record type only.
    """
    base = Path(root)
    skipped = {Path(item) for item in skip}
    reports: list[RecoveryReport] = []
    for _, directory in _day_directories(base):
        for entry in sorted(directory.iterdir()):
            info = parse_segment_name(entry)
            if info is None or not info.is_open or entry in skipped:
                continue
            if record_type is not None and info.record_type != record_type:
                continue
            reports.append(recover_segment(entry))
    return reports


class SegmentReader:
    """Reads the segments under one root, for one record type.

    `slot_s` is the segment length that the writer used, which the default config sets to 600.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        record_type: str = DEFAULT_RECORD_TYPE,
        slot_s: int = 600,
    ) -> None:
        self._root = Path(root)
        self._record_type = record_type
        self._slot_s = slot_s

    @property
    def root(self) -> Path:
        return self._root

    def iter_segments(
        self,
        start_utc_ns: int | None = None,
        end_utc_ns: int | None = None,
        *,
        include_open: bool = True,
    ) -> Iterator[SegmentInfo]:
        """Yield the segments that overlap `[start_utc_ns, end_utc_ns)`. See `iter_segments`."""
        return iter_segments(
            self._root,
            start_utc_ns,
            end_utc_ns,
            record_type=self._record_type,
            slot_s=self._slot_s,
            include_open=include_open,
        )

    @staticmethod
    def read(segment: Path | str | SegmentInfo) -> SegmentData:
        """Read one segment up to its last complete row. See `read_segment`."""
        return read_segment(segment.path if isinstance(segment, SegmentInfo) else segment)

    def read_range(self, start_utc_ns: int, end_utc_ns: int) -> npt.NDArray[Any]:
        """Return the rows with `start_utc_ns <= t_utc_ns < end_utc_ns`, ordered by time.

        The rows use the current layout of the record type, even when an older segment has
        fewer columns. The result is empty when no segment overlaps the range.
        """
        dtype = segment_dtype(self._record_type)
        pieces: list[npt.NDArray[Any]] = []
        for info in self.iter_segments(start_utc_ns, end_utc_ns):
            rows = upgrade_rows(read_segment(info.path).rows, dtype)
            times = rows["t_utc_ns"]
            pieces.append(rows[(times >= start_utc_ns) & (times < end_utc_ns)])
        if not pieces:
            return np.zeros(0, dtype=dtype)
        merged = np.concatenate(pieces)
        return merged[np.argsort(merged["t_utc_ns"], kind="stable")]

    def recover(self) -> list[RecoveryReport]:
        """Repair the orphaned files of the root. See `recover_orphans`."""
        return recover_orphans(self._root, record_type=self._record_type)


# --- writing -----------------------------------------------------------------------------------


@dataclass(slots=True)
class _OpenSegment:
    handle: BinaryIO
    partial: Path
    final: Path
    slot: int
    stream_id: int
    good_size: int
    rows: int = 0
    last_write_ns: int = 0
    last_sync_ns: int = 0


class SegmentWriter:
    """Appends metric rows to segment files. It implements `MetricsWriter`.

    Pass the root directory (`DataLayout.segments_dir`), a `Clock`, and the identity that goes
    in every header. The writer uses the clock for the fsync interval and for `tick`, and it
    assigns rows to slots by their own `t_utc_ns`, so a replay and a simulation behave like a
    live run. Any thread can call the methods, and a lock serializes them. A new writer does not
    repair old `.part` files: call `recover_orphans` before the first write.
    """

    def __init__(
        self,
        root: Path | str,
        clock: Clock,
        *,
        station_id: str,
        profile_id: str,
        provenance: Mapping[str, str] | None = None,
        config: SegmentsConfig | None = None,
        record_type: str = DEFAULT_RECORD_TYPE,
    ) -> None:
        settings = config if config is not None else SegmentsConfig()
        self._root = Path(root)
        self._clock = clock
        self._cls = get_record_type(record_type)
        self._dtype = segment_dtype(self._cls)  # raises for a record that is not a segment type
        self._layout = [list(item) for item in segment_layout(self._cls)]
        self._slot_ns = settings.segment_s * NS_PER_S
        self._fsync_ns = round(settings.fsync_interval_s * NS_PER_S)
        self._idle_ns = round(settings.idle_close_s * NS_PER_S)
        self._identity: dict[str, Any] = {
            "station_id": station_id,
            "profile_id": profile_id,
            "provenance": dict(
                provenance if provenance is not None else {"software": seeingmon.__version__}
            ),
            "revision": 0,
            "quality": None,
        }
        self._lock = threading.Lock()
        self._current: _OpenSegment | None = None

    @property
    def root(self) -> Path:
        return self._root

    @property
    def open_path(self) -> Path | None:
        """The `.part` file that the writer appends to now, or `None`."""
        with self._lock:
            return None if self._current is None else self._current.partial

    def write_metrics(self, stream_id: int, rows: npt.NDArray[Any]) -> None:
        """Append rows from one stream, which must have the record's segment dtype.

        The writer splits the rows at the slot boundaries by their `t_utc_ns`, closes a segment
        when a row belongs to another slot or stream, and opens the next one. An `OSError`
        (a full disk) closes the open segment at its last complete row and propagates.
        """
        if rows.dtype != self._dtype:
            raise ValueError(
                f"the rows do not have the {self._cls.record_type} layout "
                "(seeingmon.records.segments.segment_dtype)"
            )
        if rows.ndim != 1:
            raise ValueError("pass a one-dimensional array of rows")
        if not 0 <= stream_id <= 2**32 - 1:
            raise ValueError("stream_id must fit in 32 bits")
        if len(rows) == 0:
            return
        slots = rows["t_utc_ns"] // self._slot_ns
        changes = (np.flatnonzero(slots[1:] != slots[:-1]) + 1).tolist()
        bounds = [0, *changes, len(rows)]
        with self._lock:
            for start, stop in itertools.pairwise(bounds):
                self._append(stream_id, int(slots[start]), rows[start:stop])

    def flush(self) -> None:
        """Force the open segment to disk."""
        with self._lock:
            current = self._current
            if current is not None:
                current.handle.flush()
                os.fsync(current.handle.fileno())
                current.last_sync_ns = self._clock.monotonic_ns()

    def tick(self) -> bool:
        """Close the open segment when no rows arrived for `idle_close_s`. Return whether it did.

        Call it from a housekeeping loop, so that a stopped capture leaves a closed file and not
        a `.part` file.
        """
        with self._lock:
            current = self._current
            if current is None:
                return False
            if self._clock.monotonic_ns() - current.last_write_ns < self._idle_ns:
                return False
            self._close_current()
            return True

    def close(self) -> None:
        """Force the open segment to disk and rename it to its final name. Safe to repeat."""
        with self._lock:
            self._close_current()

    def recover_orphans(self) -> list[RecoveryReport]:
        """Repair the `.part` files that earlier runs left, except the one this writer holds."""
        with self._lock:
            skip = [] if self._current is None else [self._current.partial]
            return recover_orphans(self._root, skip=skip, record_type=self._cls.record_type)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- internals (the caller holds the lock) ---

    def _append(self, stream_id: int, slot: int, chunk: npt.NDArray[Any]) -> None:
        current = self._current
        if current is not None and (current.slot != slot or current.stream_id != stream_id):
            self._close_current()
            current = None
        if current is None:
            current = self._open(stream_id, slot, int(chunk["t_utc_ns"][0]))
            self._current = current
        payload = np.ascontiguousarray(chunk).tobytes()
        try:
            current.handle.write(payload)
            current.handle.flush()
        except OSError:
            self._abandon(current)
            raise
        current.rows += len(chunk)
        current.good_size += len(payload)
        now = self._clock.monotonic_ns()
        current.last_write_ns = now
        if now - current.last_sync_ns >= self._fsync_ns:
            os.fsync(current.handle.fileno())
            current.last_sync_ns = now

    def _header_bytes(self, stream_id: int, slot: int, first_t_utc_ns: int) -> bytes:
        values = {**self._identity, "stream_id": stream_id}
        record_header: dict[str, Any] = {}
        for spec in segment_header_fields(self._cls):
            if spec.name in values:
                record_header[spec.name] = values[spec.name]
            elif spec.has_default:
                record_header[spec.name] = spec.default
            else:
                raise ValueError(f"the segment header needs a value for {spec.name}")
        header = {
            "format": FORMAT_VERSION,
            "record_type": self._cls.record_type,
            "slot_start_utc_ns": slot * self._slot_ns,
            "slot_ns": self._slot_ns,
            "start_utc_ns": first_t_utc_ns,
            "created_utc_ns": self._clock.utc_ns(),
            "row_bytes": self._dtype.itemsize,
            "layout": self._layout,
            "header": record_header,
        }
        body = json.dumps(header, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
        return b"".join(
            [_PREFIX.pack(MAGIC, FORMAT_VERSION, 0, len(body)), body, _CRC.pack(zlib.crc32(body))]
        )

    def _open(self, stream_id: int, slot: int, first_t_utc_ns: int) -> _OpenSegment:
        start = slot * self._slot_ns
        directory = segment_directory(self._root, start)
        directory.mkdir(parents=True, exist_ok=True)
        header = self._header_bytes(stream_id, slot, first_t_utc_ns)
        for part in range(_MAX_PARTS):
            final = directory / segment_name(self._cls.record_type, start, stream_id, part)
            if final.exists():
                continue
            partial = final.with_name(final.name + OPEN_SUFFIX)
            try:
                handle = partial.open("xb")
            except FileExistsError:
                continue
            try:
                handle.write(header)
                handle.flush()
            except BaseException:
                handle.close()
                partial.unlink(missing_ok=True)
                raise
            now = self._clock.monotonic_ns()
            return _OpenSegment(
                handle=handle,
                partial=partial,
                final=final,
                slot=slot,
                stream_id=stream_id,
                good_size=len(header),
                last_write_ns=now,
                last_sync_ns=now,
            )
        raise OSError("there are too many segment files for one slot")

    def _close_current(self) -> None:
        current, self._current = self._current, None
        if current is None:
            return
        try:
            current.handle.flush()
            os.fsync(current.handle.fileno())
        finally:
            current.handle.close()
        os.replace(current.partial, current.final)
        _fsync_directory(current.final.parent)

    def _abandon(self, current: _OpenSegment) -> None:
        """Close a segment after a failed write: cut it at the last complete row and rename it."""
        self._current = None
        with contextlib.suppress(OSError):
            current.handle.close()
        with contextlib.suppress(OSError):
            with current.partial.open("r+b") as handle:
                handle.truncate(current.good_size)
            os.replace(current.partial, current.final)
