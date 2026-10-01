"""How a record type maps to InfluxDB and to TimescaleDB, generated from the declaration.

The sink adapters apply these mappings and decide nothing else about the layout. A sink receives
rows in the form of `Record.to_row` (JSON-compatible values), and it upserts by the key
`(station_id, t_utc_ns, revision)` of the record type, so it can receive a batch twice.

**InfluxDB.** A record type is a measurement with the name of the type. The tags are `station`
(the `station_id` field) and `profile` (the `profile_id` field). The timestamp of a point is
`t_utc_ns`, with nanosecond precision. Every other field is a field of the point, and a `None`
value leaves the field out. A `json` field is a JSON string, and a `bytes` field is a base64
string. The key is the series (the measurement and the tags) plus the timestamp, so a new
revision at the same time replaces the point.

**TimescaleDB.** A record type is a table with the name of the type, which the adapter turns into
a hypertable on `t_utc_ns` (a `BIGINT` of nanoseconds). The table has a unique index on the key
columns. `bytes` fields are `BYTEA` columns, so the adapter decodes the base64 text of a row, and
`json` fields are `JSONB` columns.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass

from seeingmon.records.base import (
    KEY_FIELDS,
    FieldKind,
    FieldSpec,
    Record,
    field_specs,
    resolve_record_type,
)

INFLUX_TAGS: Mapping[str, str] = {"station": "station_id", "profile": "profile_id"}
INFLUX_TIME_FIELD = "t_utc_ns"
INFLUX_RESERVED_FIELDS = frozenset({"time"})

_PG_TYPES: dict[FieldKind, str] = {
    "int": "BIGINT",
    "float": "DOUBLE PRECISION",
    "bool": "BOOLEAN",
    "str": "TEXT",
    "bytes": "BYTEA",
    "json": "JSONB",
}


@dataclass(frozen=True, slots=True)
class InfluxField:
    """A field of a point. `kind` is `float`, `int`, `bool`, `str`, `json`, or `bytes`.

    A `json` field travels as a JSON string and a `bytes` field as a base64 string. A value of
    `None` leaves the field out of the point, so a field with `nullable` set can be missing.
    """

    name: str
    kind: FieldKind
    nullable: bool


@dataclass(frozen=True, slots=True)
class InfluxMapping:
    """The measurement of a record type. `tags` maps each tag name to the field that supplies it."""

    measurement: str
    tags: Mapping[str, str]
    time_field: str
    fields: tuple[InfluxField, ...]


@dataclass(frozen=True, slots=True)
class TimescaleColumn:
    """A column of the table. `sql_type` is the PostgreSQL type that suits `kind`."""

    name: str
    sql_type: str
    kind: FieldKind
    nullable: bool


def _quote(name: str) -> str:
    return f'"{name}"'


@dataclass(frozen=True, slots=True)
class TimescaleMapping:
    """The hypertable of a record type. Every declared field is a column.

    `key_columns` identify a record, and `unique_index` is the name of the unique index on them.
    `time_column` is the hypertable dimension, in nanoseconds since the Unix epoch.
    """

    table: str
    columns: tuple[TimescaleColumn, ...]
    key_columns: tuple[str, ...]
    unique_index: str
    time_column: str

    def create_table_sql(self) -> str:
        """`CREATE TABLE IF NOT EXISTS`, with `NOT NULL` on the columns that cannot be null."""
        lines = [
            f"{_quote(column.name)} {column.sql_type}{'' if column.nullable else ' NOT NULL'}"
            for column in self.columns
        ]
        body = ",\n    ".join(lines)
        return f"CREATE TABLE IF NOT EXISTS {_quote(self.table)} (\n    {body}\n)"

    def unique_index_sql(self) -> str:
        """`CREATE UNIQUE INDEX IF NOT EXISTS` on the key columns."""
        columns = ", ".join(_quote(name) for name in self.key_columns)
        return (
            f"CREATE UNIQUE INDEX IF NOT EXISTS {_quote(self.unique_index)} "
            f"ON {_quote(self.table)} ({columns})"
        )

    def add_column_sql(self) -> list[str]:
        """`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for every column except the key columns.

        Run them after `create_table_sql` so that a table that an older version created gets the
        fields that a newer version added. The added columns allow null.
        """
        return [
            f"ALTER TABLE {_quote(self.table)} ADD COLUMN IF NOT EXISTS "
            f"{_quote(column.name)} {column.sql_type}"
            for column in self.columns
            if column.name not in self.key_columns
        ]


@dataclass(frozen=True, slots=True)
class SinkMapping:
    """Both mappings of one record type."""

    record_type: str
    influx: InfluxMapping
    timescale: TimescaleMapping


def _influx(cls: type[Record], specs: tuple[FieldSpec, ...]) -> InfluxMapping:
    taken = {*INFLUX_TAGS.values(), INFLUX_TIME_FIELD}
    fields = []
    for spec in specs:
        if spec.name in taken:
            continue
        if spec.name in INFLUX_RESERVED_FIELDS:
            raise ValueError(f"{cls.record_type}.{spec.name} is a reserved name in InfluxDB")
        fields.append(InfluxField(spec.name, spec.kind, spec.nullable))
    return InfluxMapping(
        measurement=cls.record_type,
        tags=dict(INFLUX_TAGS),
        time_field=INFLUX_TIME_FIELD,
        fields=tuple(fields),
    )


def _timescale(cls: type[Record], specs: tuple[FieldSpec, ...]) -> TimescaleMapping:
    columns = tuple(
        TimescaleColumn(spec.name, _PG_TYPES[spec.kind], spec.kind, spec.nullable) for spec in specs
    )
    return TimescaleMapping(
        table=cls.record_type,
        columns=columns,
        key_columns=KEY_FIELDS,
        unique_index=f"{cls.record_type}_key",
        time_column="t_utc_ns",
    )


@functools.cache
def _mapping_for(cls: type[Record]) -> SinkMapping:
    specs = field_specs(cls)
    return SinkMapping(cls.record_type, _influx(cls, specs), _timescale(cls, specs))


def sink_mapping(record: str | type[Record]) -> SinkMapping:
    """The InfluxDB and TimescaleDB mappings of a record type (a name or a class).

    Every record type has a mapping, including a `segment` type such as `frame`, so a sink can
    accept it. Raises `ValueError` when a field has a name that InfluxDB reserves.
    """
    return _mapping_for(resolve_record_type(record))
