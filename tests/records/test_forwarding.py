"""The records, the cursor, and the sink interface fit together.

These tests build the smallest possible forwarder from `fetch_after` and `StoredRow`, and they
run it against the scripted `FakeSink`. They document the contract that the storage lane and the
sink adapters rely on: the cursor is the last acknowledged row ID, a failed batch leaves the
cursor in place, a correction is a new row after the cursor, and a new sink starts at zero.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterator

import pytest

from seeingmon.records.samples import sample_record
from seeingmon.records.sqlite_schema import ensure_schema, fetch_after, insert_record
from seeingmon.records.system import EventRecord
from seeingmon.sinks.base import SinkError, StoredRow
from seeingmon.testing.fakes import FakeSink


@pytest.fixture
def db() -> Iterator[sqlite3.Connection]:
    with contextlib.closing(sqlite3.connect(":memory:")) as connection:
        ensure_schema(connection, [EventRecord])
        yield connection


def event(n: int, **overrides: object) -> EventRecord:
    return sample_record(EventRecord, t_utc_ns=1_800_000_000_000_000_000 + n, **overrides)


def forward_once(db: sqlite3.Connection, sink: FakeSink, cursor: int) -> int:
    """Send one batch of new rows to the sink. Return the new cursor."""
    pairs = fetch_after(db, "event", cursor, sink.max_batch_rows)
    if not pairs:
        return cursor
    sink.send("event", [StoredRow(row_id, values) for row_id, values in pairs])
    return pairs[-1][0]


def forward_all(db: sqlite3.Connection, sink: FakeSink, cursor: int = 0) -> int:
    while (next_cursor := forward_once(db, sink, cursor)) != cursor:
        cursor = next_cursor
    return cursor


def test_every_row_arrives_once_and_in_order(db: sqlite3.Connection) -> None:
    sent = [event(n) for n in range(7)]
    for record in sent:
        insert_record(db, record)
    sink = FakeSink(max_batch_rows=3)
    cursor = forward_all(db, sink)
    assert cursor == 7
    assert [len(batch) for _, batch in sink.sent] == [3, 3, 1]
    rows = sink.rows("event")
    assert [row.row_id for row in rows] == list(range(1, 8))
    assert [EventRecord.from_row(row.values, strict=True) for row in rows] == sent


def test_a_failed_batch_leaves_the_cursor_and_the_retry_loses_nothing(
    db: sqlite3.Connection,
) -> None:
    for n in range(5):
        insert_record(db, event(n))
    sink = FakeSink(max_batch_rows=2)
    cursor = forward_once(db, sink, 0)
    assert cursor == 2
    sink.fail_next(retryable=True)
    with pytest.raises(SinkError):
        forward_once(db, sink, cursor)  # the outage: the cursor stays at 2
    insert_record(db, event(5))  # new rows arrive during the outage
    cursor = forward_all(db, sink, cursor)
    assert cursor == 6
    assert [row.row_id for row in sink.rows("event")] == [1, 2, 3, 4, 5, 6]


def test_a_correction_is_a_new_row_after_the_cursor(db: sqlite3.Connection) -> None:
    insert_record(db, event(0, message="The first result."))
    sink = FakeSink()
    cursor = forward_all(db, sink)
    insert_record(db, event(0, message="The corrected result.", revision=1))
    cursor = forward_all(db, sink, cursor)
    rows = sink.rows("event")
    assert [(row.row_id, row.values["revision"]) for row in rows] == [(1, 0), (2, 1)]
    assert rows[1].values["message"] == "The corrected result."
    assert cursor == 2


def test_a_new_sink_backfills_from_row_zero(db: sqlite3.Connection) -> None:
    for n in range(4):
        insert_record(db, event(n))
    first, second = FakeSink("first"), FakeSink("second")
    forward_all(db, first)
    forward_all(db, second)  # a new sink keeps its own cursor and starts at zero
    assert [row.row_id for row in second.rows("event")] == [1, 2, 3, 4]
    assert [r.values for r in first.rows("event")] == [r.values for r in second.rows("event")]


def test_the_values_are_json_compatible_and_use_the_declared_names(
    db: sqlite3.Connection,
) -> None:
    insert_record(db, event(0, detail={"from": "safe", "to": "auto"}, quality={"detail": "x"}))
    ((row_id, values),) = fetch_after(db, "event")
    assert row_id == 1
    assert json.loads(json.dumps(values)) == values
    assert set(values) == set(event(0).to_row())
    assert values["detail"] == {"from": "safe", "to": "auto"}
