from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator

import pytest

from seeingmon.records.base import Record, field_specs, quantity
from seeingmon.records.sink_mapping import (
    INFLUX_TAGS,
    sink_mapping,
)
from seeingmon.records.system import EventRecord
from tests.records.strategies import ALL_RECORD_TYPES, type_id

PG_TYPES = {
    "int": "BIGINT",
    "float": "DOUBLE PRECISION",
    "bool": "BOOLEAN",
    "str": "TEXT",
    "bytes": "BYTEA",
    "json": "JSONB",
}

EVENT_TABLE = """\
CREATE TABLE IF NOT EXISTS "event" (
    "station_id" TEXT NOT NULL,
    "t_utc_ns" BIGINT NOT NULL,
    "revision" BIGINT NOT NULL,
    "profile_id" TEXT NOT NULL,
    "provenance" JSONB NOT NULL,
    "quality" JSONB,
    "level" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "detail" JSONB
)"""


@pytest.mark.parametrize("cls", ALL_RECORD_TYPES, ids=type_id)
class TestEveryRecordType:
    def test_influx_gets_one_measurement_with_the_station_and_the_profile_as_tags(
        self, cls: type[Record]
    ) -> None:
        mapping = sink_mapping(cls)
        assert mapping.record_type == cls.record_type
        assert mapping.influx.measurement == cls.record_type
        assert mapping.influx.tags == {"station": "station_id", "profile": "profile_id"}
        assert mapping.influx.time_field == "t_utc_ns"

    def test_influx_covers_every_field_exactly_once(self, cls: type[Record]) -> None:
        influx = sink_mapping(cls).influx
        covered = [*influx.tags.values(), influx.time_field, *(f.name for f in influx.fields)]
        assert sorted(covered) == sorted(spec.name for spec in field_specs(cls))
        by_name = {spec.name: spec for spec in field_specs(cls)}
        for field in influx.fields:
            assert field.kind == by_name[field.name].kind
            assert field.nullable == by_name[field.name].nullable
            assert field.kind in {"float", "int", "bool", "str", "json", "bytes"}

    def test_timescale_gets_one_table_with_a_unique_index_on_the_key(
        self, cls: type[Record]
    ) -> None:
        timescale = sink_mapping(cls).timescale
        assert timescale.table == cls.record_type
        assert timescale.key_columns == ("station_id", "t_utc_ns", "revision")
        assert timescale.unique_index == f"{cls.record_type}_key"
        assert timescale.time_column == "t_utc_ns"
        specs = field_specs(cls)
        assert [c.name for c in timescale.columns] == [s.name for s in specs]
        for column, spec in zip(timescale.columns, specs, strict=True):
            assert column.sql_type == PG_TYPES[spec.kind]
            assert column.kind == spec.kind
            assert column.nullable == spec.nullable
        assert set(timescale.key_columns) <= {c.name for c in timescale.columns}

    def test_the_ddl_runs_and_the_index_enforces_the_key(self, cls: type[Record]) -> None:
        timescale = sink_mapping(cls).timescale
        with memory_db() as db:
            db.execute(timescale.create_table_sql())
            db.execute(timescale.unique_index_sql())
            names = [row[1] for row in db.execute(f'PRAGMA table_info("{cls.record_type}")')]
            assert names == [c.name for c in timescale.columns]
            indexes = db.execute(f'PRAGMA index_list("{cls.record_type}")').fetchall()
            unique = [row for row in indexes if row[1] == timescale.unique_index]
            assert len(unique) == 1
            assert unique[0][2] == 1  # the index is unique
            columns = [r[2] for r in db.execute(f'PRAGMA index_info("{timescale.unique_index}")')]
            assert columns == list(timescale.key_columns)

    def test_the_mapping_is_cached_and_accepts_the_name(self, cls: type[Record]) -> None:
        assert sink_mapping(cls) is sink_mapping(cls.record_type)


@contextlib.contextmanager
def memory_db() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


class TestInflux:
    def test_the_tags_are_fixed(self) -> None:
        assert dict(INFLUX_TAGS) == {"station": "station_id", "profile": "profile_id"}

    def test_the_fields_of_the_event_record(self) -> None:
        fields = [(f.name, f.kind, f.nullable) for f in sink_mapping("event").influx.fields]
        assert fields == [
            ("revision", "int", False),
            ("provenance", "json", False),
            ("quality", "json", True),
            ("level", "str", False),
            ("kind", "str", False),
            ("message", "str", False),
            ("detail", "json", True),
        ]

    def test_the_kinds_of_the_seeing_window_and_the_star_list(self) -> None:
        kinds = {f.name: f.kind for f in sink_mapping("seeing_window").influx.fields}
        assert kinds["n_frames"] == "int"
        assert kinds["seeing_fwhm_arcsec"] == "float"
        assert kinds["readout_mode"] == "str"
        assert kinds["flags"] == "json"
        assert kinds["motion_psd_freq_hz"] == "json"
        star_kinds = {f.name: f.kind for f in sink_mapping("star_list").influx.fields}
        assert star_kinds["data"] == "bytes"
        health_kinds = {f.name: f.kind for f in sink_mapping("health").influx.fields}
        assert health_kinds["degraded"] == "bool"
        assert health_kinds["time_synchronized"] == "bool"

    def test_a_segment_record_has_a_mapping_too(self) -> None:
        names = [f.name for f in sink_mapping("frame").influx.fields]
        assert names[:2] == ["revision", "provenance"]
        assert {"seq", "cx_px", "peak_dn", "stream_id"} <= set(names)

    def test_a_field_named_time_is_rejected(self) -> None:
        class WithTime(EventRecord, register=False):
            time: float | None = quantity(default=None, definition="A reserved name.")

        with pytest.raises(ValueError, match="reserved name in InfluxDB"):
            sink_mapping(WithTime)


class TestTimescale:
    def test_the_table_of_the_event_record(self) -> None:
        assert sink_mapping("event").timescale.create_table_sql() == EVENT_TABLE

    def test_the_unique_index(self) -> None:
        assert sink_mapping("event").timescale.unique_index_sql() == (
            'CREATE UNIQUE INDEX IF NOT EXISTS "event_key" '
            'ON "event" ("station_id", "t_utc_ns", "revision")'
        )

    def test_a_new_field_adds_columns_to_an_older_table(self) -> None:
        statements = sink_mapping("event").timescale.add_column_sql()
        assert statements[0] == ('ALTER TABLE "event" ADD COLUMN IF NOT EXISTS "profile_id" TEXT')
        assert len(statements) == len(field_specs("event")) - 3  # the key columns stay
        assert all("IF NOT EXISTS" in s and "NOT NULL" not in s for s in statements)
        assert not any('"station_id"' in s or '"revision"' in s for s in statements)

    def test_bytes_and_json_columns(self) -> None:
        types = {c.name: c.sql_type for c in sink_mapping("star_list").timescale.columns}
        assert types["data"] == "BYTEA"
        assert types["columns"] == "JSONB"
        assert types["catalog"] == "TEXT"
