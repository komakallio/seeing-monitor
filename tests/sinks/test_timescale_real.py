"""The PostgreSQL sink against a real server, when you give one.

Set `SEEINGMON_TEST_POSTGRES` to a libpq connection string for a test database, and run
`pytest --slow`. Each test makes a schema of its own, points `search_path` at it, and drops it
afterward. The tests need the `timescale` extra (`psycopg`), and they skip without the variable.
A server with the TimescaleDB extension also checks the hypertables.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from seeingmon.records.base import RECORD_TYPES
from seeingmon.records.samples import sample_record
from seeingmon.sinks.base import StoredRow
from seeingmon.sinks.timescale import TimescaleSink
from tests.store.builders import T0, make_health

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("SEEINGMON_TEST_POSTGRES"),
        reason="set SEEINGMON_TEST_POSTGRES to a PostgreSQL connection string to run this test",
    ),
]

Connect = Callable[[], Any]


@pytest.fixture
def database() -> Iterator[tuple[Connect, str, Any]]:
    """Yield a connection function that uses a private schema, the schema name, and `psycopg`."""
    psycopg = pytest.importorskip("psycopg")
    dsn = os.environ["SEEINGMON_TEST_POSTGRES"]
    schema = f"seeingmon_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')

    def connect() -> Any:
        return psycopg.connect(dsn, options=f"-c search_path={schema}")

    try:
        yield connect, schema, psycopg
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')


def query(connect: Connect, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    with connect() as connection:
        return [tuple(row) for row in connection.execute(sql, params).fetchall()]


def rows_of(*records: Any) -> list[StoredRow]:
    return [StoredRow(n + 1, record.to_row()) for n, record in enumerate(records)]


def test_a_batch_is_upserted_and_a_repeat_changes_nothing(
    database: tuple[Connect, str, Any],
) -> None:
    connect, _, _ = database
    sink = TimescaleSink("real", connect)
    rows = rows_of(make_health(T0), make_health(T0 + 1))
    sink.send("health", rows)
    sink.send("health", rows)
    assert query(connect, 'SELECT count(*) FROM "health"') == [(2,)]
    sink.close()


def test_a_corrected_row_replaces_the_old_one_with_the_same_key(
    database: tuple[Connect, str, Any],
) -> None:
    connect, _, _ = database
    sink = TimescaleSink("real", connect)
    sink.send("health", rows_of(make_health(T0, heater_duty=0.25)))
    sink.send("health", rows_of(make_health(T0, heater_duty=0.75)))
    assert query(connect, 'SELECT "heater_duty" FROM "health"') == [(0.75,)]
    sink.close()


def test_every_table_record_type_gets_a_table_that_takes_its_rows(
    database: tuple[Connect, str, Any],
) -> None:
    connect, _, _ = database
    sink = TimescaleSink("real", connect)
    for cls in RECORD_TYPES.values():
        if cls.storage != "table":
            continue
        sink.send(cls.record_type, rows_of(sample_record(cls)))
        assert query(connect, f'SELECT count(*) FROM "{cls.record_type}"') == [(1,)]
    sink.close()


def test_the_values_keep_their_types(database: tuple[Connect, str, Any]) -> None:
    connect, _, _ = database
    sink = TimescaleSink("real", connect)
    record = sample_record(
        "star_list", n_stars=1, columns=["x"], data=b"\x00\x00\x80\x3f", t_utc_ns=2**62
    )
    sink.send("star_list", rows_of(record))
    ((data, columns, t_utc_ns),) = query(
        connect, 'SELECT "data", "columns", "t_utc_ns" FROM "star_list"'
    )
    assert bytes(data) == b"\x00\x00\x80\x3f"  # BYTEA
    assert columns == ["x"]  # JSONB
    assert t_utc_ns == 2**62  # BIGINT keeps every nanosecond
    sink.close()


def test_the_key_has_a_unique_index(database: tuple[Connect, str, Any]) -> None:
    connect, schema, _ = database
    sink = TimescaleSink("real", connect)
    sink.send("health", rows_of(make_health(T0)))
    indexes = query(
        connect,
        "SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND indexname = %s",
        (schema, "health_key"),
    )
    assert len(indexes) == 1
    assert "UNIQUE" in indexes[0][0]
    sink.close()


def test_a_hypertable_exists_when_the_server_has_timescaledb(
    database: tuple[Connect, str, Any],
) -> None:
    connect, schema, _ = database
    if not query(connect, "SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'"):
        pytest.skip("the server has no TimescaleDB extension")
    sink = TimescaleSink("real", connect)
    sink.send("health", rows_of(make_health(T0)))
    found = query(
        connect,
        "SELECT count(*) FROM timescaledb_information.hypertables "
        "WHERE hypertable_schema = %s AND hypertable_name = 'health'",
        (schema,),
    )
    assert found == [(1,)]
    sink.close()
