"""One writer and several readers share a store, and a second opener coexists with a writer."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from seeingmon.records.sqlite_schema import insert_record
from seeingmon.store.db import Store, StoreBusyError, StoreReader
from tests.store.builders import T0, make_event, make_health

TOTAL = 300


def run_threads(*targets: Callable[[], object]) -> list[BaseException]:
    """Run each callable in its own thread, and return the exceptions that they raised."""
    errors: list[BaseException] = []

    def guard(target: Callable[[], object]) -> None:
        try:
            target()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=guard, args=(target,)) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive(), "a thread did not finish"
    return errors


def test_readers_see_a_consistent_growing_table_while_one_thread_writes(
    store: Store, db_path: Path
) -> None:
    done = threading.Event()
    seen_final: list[int] = []

    def writer() -> None:
        try:
            for n in range(TOTAL):
                store.write(make_health(T0 + n))
        finally:
            done.set()

    def make_reader(read: Store | StoreReader) -> Callable[[], None]:
        def reader() -> None:
            last_count = 0
            while True:
                finished = done.is_set()  # read it first, so a last full pass follows the end
                # Two separate reads can see different commits, so they agree only in one
                # direction: the table only grows, and the second read happens later.
                count_first = read.count("health")
                rows_later = read.after("health", 0, TOTAL + 10)
                assert len(rows_later) >= count_first
                # The reads of one snapshot all see the same state of the database.
                with read.snapshot() as snap:
                    count = snap.count("health")
                    rows = snap.after("health", 0, TOTAL + 10)
                    latest = snap.latest("health")
                    last_id = snap.last_row_id("health")
                    behind = snap.count_after("health", 0)
                assert count == len(rows) == behind == last_id
                assert [row.row_id for row in rows] == list(range(1, count + 1))
                assert [row.values["t_utc_ns"] for row in rows] == [T0 + n for n in range(count)]
                assert (latest is None) == (count == 0)
                if latest is not None:
                    assert latest.row_id == count
                # Separate snapshots: the table never shrinks.
                assert count >= last_count
                last_count = count
                if finished:
                    seen_final.append(count)
                    return

        return reader

    with StoreReader.open(db_path) as web_reader:
        errors = run_threads(
            writer, make_reader(store), make_reader(store), make_reader(web_reader)
        )
    assert errors == []
    assert seen_final == [TOTAL, TOTAL, TOTAL]
    assert store.count("health") == TOTAL


def test_threads_that_write_at_once_take_turns(store: Store) -> None:
    per_thread = 150

    def write_events() -> None:
        for n in range(per_thread):
            store.write(make_event(T0, message=str(n)))  # every event shares one time

    errors = run_threads(write_events, write_events)
    assert errors == []
    rows = store.after("event", 0, 1000)
    assert [row.row_id for row in rows] == list(range(1, 2 * per_thread + 1))
    assert sorted(row.values["t_utc_ns"] for row in rows) == [T0 + n for n in range(2 * per_thread)]


def test_two_stores_on_one_file_write_in_turn_and_keep_events_apart(db_path: Path) -> None:
    with Store.open(db_path) as first, Store.open(db_path) as second:
        for n in range(20):
            (first if n % 2 == 0 else second).write(make_event(T0, message=str(n)))
        first.write(make_health(T0))
        assert second.count("health") == 1  # each sees the commits of the other
        rows = second.after("event", 0)
        assert [row.values["t_utc_ns"] for row in rows] == [T0 + n for n in range(20)]


def test_a_reader_never_sees_an_uncommitted_transaction(store: Store, db_path: Path) -> None:
    with StoreReader.open(db_path) as reader:
        with store._transaction() as connection:
            insert_record(connection, make_health(T0))
            assert store.count("health") == 0  # a read sees the snapshot without the open row
            assert reader.count("health") == 0
        assert store.count("health") == 1
        assert reader.count("health") == 1


def test_a_second_writer_gets_a_busy_error_when_the_first_holds_the_lock(db_path: Path) -> None:
    with (
        Store.open(db_path, busy_timeout_ms=50) as first,
        Store.open(db_path, busy_timeout_ms=50) as second,
    ):
        with first._transaction() as connection:
            insert_record(connection, make_health(T0))
            with pytest.raises(StoreBusyError):
                second.write(make_health(T0 + 1))
            with pytest.raises(StoreBusyError):
                Store.open(db_path, busy_timeout_ms=50)
            assert second.count("health") == 0
        assert second.write(make_health(T0 + 1)) == 2  # the lock is free again
        assert first.count("health") == 2
