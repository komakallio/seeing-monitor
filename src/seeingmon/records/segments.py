"""The segment layout of a record type: a packed NumPy dtype that comes from the declaration.

A record type with `storage = "segment"` (such as `frame`) keeps its rows in binary segment
files of about 10 minutes, not in SQLite rows. Each row holds the fields that declare a
`dtype`, in declaration order, packed with no padding and in little-endian byte order. A
segment keeps every other field once, in its header: `station_id`, `profile_id`, `provenance`,
`revision`, `quality`, and the fields that declare no `dtype`, such as `stream_id`.

The layout is a file format. Append a new per-row field at the end of the class, and never
reorder, remove, or retype one, because that makes old segment files unreadable. A float field
that is `None` in a record is NaN in a row, and the reverse.

`segment_layout` returns the layout as plain data, so a writer can store it in the segment
header and a reader can rebuild the dtype of an old file with `np.dtype(list(layout))`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, overload

import numpy as np
import numpy.typing as npt

from seeingmon.records.base import FieldSpec, Record, field_specs, resolve_record_type

SegmentRows = npt.NDArray[np.void]
R = TypeVar("R", bound=Record)


def _require_segment(record: str | type[Record]) -> type[Record]:
    cls = resolve_record_type(record)
    if cls.storage != "segment":
        raise ValueError(f"{cls.record_type} is a {cls.storage} record, not a segment record")
    return cls


def segment_columns(record: str | type[Record]) -> tuple[FieldSpec, ...]:
    """The per-row fields of a segment record type, in the order of the dtype."""
    return tuple(spec for spec in field_specs(_require_segment(record)) if spec.dtype is not None)


def segment_header_fields(record: str | type[Record]) -> tuple[FieldSpec, ...]:
    """The fields that a segment stores once, in its header, in declaration order."""
    return tuple(spec for spec in field_specs(_require_segment(record)) if spec.dtype is None)


def segment_layout(record: str | type[Record]) -> tuple[tuple[str, str], ...]:
    """The layout as `(name, typestr)` pairs, such as `("seq", "<u4")`."""
    return tuple((spec.name, f"<{spec.dtype}") for spec in segment_columns(record))


def segment_dtype(record: str | type[Record]) -> np.dtype[np.void]:
    """The packed, little-endian structured dtype of the rows of a segment record type."""
    return np.dtype(list(segment_layout(record)))


def to_segment(records: Sequence[Record]) -> tuple[dict[str, Any], SegmentRows]:
    """Pack records of one segment record type into a header and an array of rows.

    The header is a dict of JSON-compatible values for the per-segment fields. Every record
    must have the same header values and the same record type. A float is stored as float32
    when the declaration says `f4`, so it can change in the last digit.
    """
    if not records:
        raise ValueError("a segment needs at least one record")
    cls = type(records[0])
    columns = segment_columns(cls)
    header_names = [spec.name for spec in segment_header_fields(cls)]
    header = {name: records[0].to_row()[name] for name in header_names}
    rows = []
    for record in records:
        if type(record) is not cls:
            raise TypeError(f"a segment holds {cls.__name__} records, not {type(record).__name__}")
        row = record.to_row()
        if {name: row[name] for name in header_names} != header:
            raise ValueError("every record of a segment must have the same header values")
        values = (getattr(record, spec.name) for spec in columns)
        rows.append(tuple(math.nan if value is None else value for value in values))
    return header, np.array(rows, dtype=segment_dtype(cls))


@overload
def from_segment(record: type[R], header: Mapping[str, Any], rows: SegmentRows) -> list[R]: ...


@overload
def from_segment(record: str, header: Mapping[str, Any], rows: SegmentRows) -> list[Record]: ...


def from_segment(
    record: str | type[Record], header: Mapping[str, Any], rows: SegmentRows
) -> list[Any]:
    """Build one record for each row of a segment. NaN in a float column becomes `None`.

    Pass the record class to get a list of that class, or the record type name.
    """
    cls = _require_segment(record)
    columns = segment_columns(cls)
    if rows.dtype != segment_dtype(cls):
        raise ValueError(f"the rows do not have the {cls.record_type} layout")
    lists = []
    for spec in columns:
        values = rows[spec.name].tolist()
        if spec.kind == "float":
            values = [None if math.isnan(value) else value for value in values]
        lists.append(values)
    return [
        cls(**header, **{spec.name: value for spec, value in zip(columns, row, strict=True)})
        for row in zip(*lists, strict=True)
    ]
