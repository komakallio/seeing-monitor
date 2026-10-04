"""The store opens in WAL mode, keeps results immutable, and reads them back."""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from seeingmon.analysis.base import RecordWriter
from seeingmon.records.base import RECORD_TYPES, Record
from seeingmon.records.samples import sample_record
from seeingmon.records.sqlite_schema import SchemaError, table_record_types
from seeingmon.store.db import (
    APPLICATION_ID,
    STORE_SCHEMA_VERSION,
    DuplicateRecordError,
    SinkCursor,
    Store,
    StoreClosedError,
    StoreFormatError,
    StoreReader,
    UnknownRecordTypeError,
    record_from_row,
)
from tests.store.builders import (
    NS_PER_DAY,
    STATION,
    T0,
    make_event,
    make_health,
    make_star_list,
    make_window,
)

TABLE_TYPES = [cls for cls in RECORD_TYPES.values() if cls.storage == "table"]


def raw(path: Path) -> contextlib.closing[sqlite3.Connection]:
    return contextlib.closing(sqlite3.connect(path))


class TestOpen:
    def test_it_creates_a_wal_database_with_the_schema_and_the_cursor_table(
        self, store: Store, db_path: Path
    ) -> None:
        facts = store.diagnostics()
        assert facts["journal_mode"] == "wal"
        assert facts["synchronous"] == "normal"
        assert facts["busy_timeout_ms"] == 5000
        assert facts["application_id"] == APPLICATION_ID
        assert facts["user_version"] == STORE_SCHEMA_VERSION
        with raw(db_path) as connection:
            names = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        assert {cls.record_type for cls in table_record_types()} <= names
        assert "sink_cursor" in names
        assert "frame" not in names  # per-frame metrics live in segment files

    def test_it_creates_the_parent_directory(self, tmp_path: Path) -> None:
        with Store.open(tmp_path / "a" / "b" / "results.sqlite") as store:
            assert store.count("health") == 0

    def test_reopening_keeps_the_rows_and_changes_no_schema(self, db_path: Path) -> None:
        with Store.open(db_path) as first:
            first.write(make_health())
        with Store.open(db_path) as second:
            assert second.count("health") == 1
            assert second.last_row_id("health") == 1

    def test_it_refuses_an_unrelated_sqlite_file_and_leaves_it_alone(self, tmp_path: Path) -> None:
        path = tmp_path / "other.sqlite"
        with raw(path) as connection:
            connection.execute("CREATE TABLE notes (body TEXT)")
            connection.commit()
        with pytest.raises(StoreFormatError):
            Store.open(path)
        with raw(path) as connection:
            names = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        assert names == {"notes"}

    def test_it_refuses_a_database_that_newer_software_wrote(self, db_path: Path) -> None:
        with Store.open(db_path):
            pass
        with raw(db_path) as connection:
            connection.execute(f"PRAGMA user_version = {STORE_SCHEMA_VERSION + 1}")
            connection.commit()
        with pytest.raises(StoreFormatError, match="upgrade the software"):
            Store.open(db_path)

    def test_it_accepts_a_column_that_newer_software_added(self, db_path: Path) -> None:
        """An older release must open a database that a newer one migrated (a rollback)."""
        with Store.open(db_path):
            pass
        with raw(db_path) as connection:
            connection.execute('ALTER TABLE "health" ADD COLUMN "added_later" REAL')
            connection.commit()
        with Store.open(db_path) as store:
            row_id = store.write(make_health())
            latest = store.latest("health")
        assert latest is not None
        assert latest.row_id == row_id
        assert "added_later" not in latest.values

    def test_it_refuses_a_table_that_a_migration_cannot_fix(self, db_path: Path) -> None:
        with Store.open(db_path):
            pass
        with raw(db_path) as connection:  # retype a column, which no migration can undo
            connection.execute('ALTER TABLE "event" DROP COLUMN "message"')
            connection.execute('ALTER TABLE "event" ADD COLUMN "message" BLOB')
            connection.commit()
        with pytest.raises(SchemaError):
            Store.open(db_path)

    def test_a_missing_file_is_not_a_store_for_a_reader(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            StoreReader.open(tmp_path / "missing.sqlite")

    def test_an_empty_file_is_not_a_store_for_a_reader(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.sqlite"
        with raw(path) as connection:
            connection.execute("PRAGMA user_version = 0")
            connection.commit()
        with pytest.raises(StoreFormatError):
            StoreReader.open(path)

    def test_a_closed_store_raises_and_closing_twice_is_fine(self, db_path: Path) -> None:
        store = Store.open(db_path)
        store.close()
        store.close()
        with pytest.raises(StoreClosedError):
            store.write(make_health())
        with pytest.raises(StoreClosedError):
            store.count("health")


class TestWrite:
    def test_it_returns_growing_row_ids_and_reads_the_record_back(self, store: Store) -> None:
        first = make_health(T0)
        second = make_health(T0 + 60 * 10**9)
        assert store.write(first) == 1
        assert store.write(second) == 2
        rows = store.after("health", 0)
        assert [row.row_id for row in rows] == [1, 2]
        assert rows[0].values == first.to_row()
        assert record_from_row("health", rows[1]) == second

    def test_a_second_record_with_the_same_key_raises_and_changes_nothing(
        self, store: Store
    ) -> None:
        store.write(make_health(T0))
        with pytest.raises(DuplicateRecordError) as caught:
            store.write(make_health(T0, state="paused"))
        assert caught.value.key == (STATION, "health", T0, 0)
        assert store.count("health") == 1
        row = store.latest("health")
        assert row is not None
        assert row.values["state"] != "paused"
        assert store.write(make_health(T0 + 1)) == 2  # a failed insert burns no row ID

    def test_reprocessing_writes_the_next_revision_and_keeps_the_old_row(
        self, store: Store
    ) -> None:
        window = make_window(T0, n_frames=100)
        store.write(window)
        assert store.next_revision("seeing_window", STATION, T0) == 1
        corrected = make_window(T0, n_frames=120, revision=1)
        assert store.write(corrected) == 2
        assert store.next_revision("seeing_window", STATION, T0) == 2
        assert store.next_revision("seeing_window", STATION, T0 + 1) == 0
        assert store.count("seeing_window") == 2

    def test_write_correction_chooses_the_revision_itself(self, store: Store) -> None:
        store.write(make_window(T0, n_frames=1))
        store.write_correction(make_window(T0, n_frames=2, revision=7))  # the 7 is ignored
        store.write_correction(make_window(T0, n_frames=3))
        revisions = [row.values["revision"] for row in store.after("seeing_window", 0)]
        assert revisions == [0, 1, 2]
        latest = store.latest("seeing_window")
        assert latest is not None
        assert latest.values["n_frames"] == 3

    def test_an_event_is_never_corrected(self, store: Store) -> None:
        with pytest.raises(ValueError, match="never corrected"):
            store.write_correction(make_event())

    def test_write_many_is_one_transaction(self, store: Store) -> None:
        ids = store.write_many([make_health(T0 + n) for n in range(5)])
        assert ids == [1, 2, 3, 4, 5]
        assert store.write_many([]) == []

    def test_a_duplicate_in_a_batch_rolls_the_whole_batch_back(self, store: Store) -> None:
        store.write(make_health(T0))
        batch = [make_health(T0 + 1), make_health(T0 + 2), make_health(T0)]  # the last one repeats
        with pytest.raises(DuplicateRecordError):
            store.write_many(batch)
        assert store.count("health") == 1
        batch_with_internal_repeat = [make_health(T0 + 5), make_health(T0 + 5)]
        with pytest.raises(DuplicateRecordError):
            store.write_many(batch_with_internal_repeat)
        assert store.count("health") == 1

    def test_a_segment_record_goes_to_segment_files_not_to_the_store(self, store: Store) -> None:
        frame = sample_record("frame")
        with pytest.raises(ValueError, match="segment"):
            store.write(frame)
        with pytest.raises(ValueError, match="segment"):
            store.count("frame")

    def test_the_store_is_a_record_writer(self, store: Store) -> None:
        # The run-time protocol check. The static check fails, because `write` returns the row ID.
        assert isinstance(cast(object, store), RecordWriter)
        writer: RecordWriter = store.as_record_writer()
        writer.write(make_health(T0))
        assert store.count("health") == 1

    @pytest.mark.parametrize("cls", TABLE_TYPES, ids=lambda cls: cls.record_type)
    def test_every_table_type_round_trips(self, store: Store, cls: type[Record]) -> None:
        record = sample_record(cls)
        row_id = store.write(record)
        latest = store.latest(cls.record_type)
        assert latest is not None
        assert latest.row_id == row_id
        assert record_from_row(cls.record_type, latest) == record


class TestReads:
    def test_latest_takes_the_newest_time_and_then_the_highest_revision(self, store: Store) -> None:
        assert store.latest("seeing_window") is None
        store.write(make_window(T0 + 10, n_frames=1))
        store.write(make_window(T0, n_frames=2))  # older time, written later
        store.write(make_window(T0 + 10, n_frames=3, revision=1))
        store.write(make_window(T0 + 5, n_frames=4, revision=9))
        latest = store.latest("seeing_window")
        assert latest is not None
        assert (latest.values["t_utc_ns"], latest.values["revision"]) == (T0 + 10, 1)

    def test_latest_can_look_at_one_station(self, store: Store) -> None:
        store.write(make_health(T0, station_id="a"))
        store.write(make_health(T0 + 5, station_id="b"))
        by_station = store.latest("health", station_id="a")
        assert by_station is not None
        assert by_station.values["station_id"] == "a"
        assert store.latest("health", station_id="c") is None

    def test_range_is_half_open_and_ordered_by_time(self, store: Store) -> None:
        for offset in (30, 10, 20, 40, 50):  # written out of order
            store.write(make_health(T0 + offset))
        rows = store.range("health", T0 + 10, T0 + 40)
        assert [row.values["t_utc_ns"] - T0 for row in rows] == [10, 20, 30]
        newest_first = store.range("health", T0, T0 + 100, 2, descending=True)
        assert [row.values["t_utc_ns"] - T0 for row in newest_first] == [50, 40]
        oldest_first = store.range("health", T0, T0 + 100, 2)
        assert [row.values["t_utc_ns"] - T0 for row in oldest_first] == [10, 20]

    def test_range_hides_superseded_revisions_unless_you_ask_for_all(self, store: Store) -> None:
        store.write(make_window(T0, n_frames=1))
        store.write(make_window(T0, n_frames=2, revision=1))
        store.write(make_window(T0 + 60, n_frames=3))
        current = store.range("seeing_window", T0, T0 + 100)
        assert [(r.values["t_utc_ns"] - T0, r.values["revision"]) for r in current] == [
            (0, 1),
            (60, 0),
        ]
        everything = store.range("seeing_window", T0, T0 + 100, all_revisions=True)
        assert [r.values["revision"] for r in everything] == [0, 1, 0]

    def test_range_can_look_at_one_station(self, store: Store) -> None:
        store.write(make_health(T0, station_id="a"))
        store.write(make_health(T0 + 1, station_id="b"))
        rows = store.range("health", T0, T0 + 10, station_id="b")
        assert [row.values["station_id"] for row in rows] == ["b"]

    def test_after_reads_batches_in_row_order_and_includes_every_revision(
        self, store: Store
    ) -> None:
        for n in range(7):
            store.write(make_health(T0 + n))
        assert [row.row_id for row in store.after("health", 0, 3)] == [1, 2, 3]
        assert [row.row_id for row in store.after("health", 3, 3)] == [4, 5, 6]
        assert [row.row_id for row in store.after("health", 6, 3)] == [7]
        assert store.after("health", 7) == []

    def test_count_and_last_row_id(self, store: Store) -> None:
        assert (store.count("health"), store.last_row_id("health")) == (0, 0)
        store.write_many([make_health(T0 + n) for n in range(4)])
        assert (store.count("health"), store.last_row_id("health")) == (4, 4)
        assert store.count_after("health", 1) == 3
        assert store.count_after("health", 4) == 0
        assert store.last_row_id("event") == 0

    def test_limits_must_be_positive(self, store: Store) -> None:
        with pytest.raises(ValueError, match="limit"):
            store.after("health", 0, 0)
        with pytest.raises(ValueError, match="limit"):
            store.range("health", 0, 1, 0)

    def test_an_unknown_record_type_is_a_lookup_error(self, store: Store) -> None:
        with pytest.raises(UnknownRecordTypeError):
            store.count("no_such_type")
        with pytest.raises(LookupError):
            store.latest("no_such_type")

    def test_a_reader_sees_what_the_writer_committed(self, store: Store, db_path: Path) -> None:
        store.write(make_health(T0))
        with StoreReader.open(db_path) as reader:
            assert reader.count("health") == 1
            store.write(make_health(T0 + 1))
            assert reader.count("health") == 2
            latest = reader.latest("health")
            assert latest is not None
            assert latest.row_id == 2

    def test_separate_reads_can_disagree_while_a_writer_commits(self, store: Store) -> None:
        """Why `snapshot` exists: each plain read sees the commits made since the one before."""
        count = store.count("health")  # 0
        store.write(make_health(T0))  # the writer commits between the two reads
        latest = store.latest("health")
        assert (count, latest is None) == (0, False)  # a count of 0 and a latest row
        with store.snapshot() as snap:
            assert (snap.count("health") == 0) == (snap.latest("health") is None)

    def test_a_snapshot_shows_one_state_while_the_writer_commits(self, store: Store) -> None:
        store.write(make_health(T0))
        with store.snapshot() as snap:
            assert snap.count("health") == 1
            store.write(make_health(T0 + 1))  # a commit in the middle of the snapshot
            assert snap.count("health") == 1
            assert snap.last_row_id("health") == 1
            assert [row.row_id for row in snap.after("health", 0)] == [1]
            latest = snap.latest("health")
            assert latest is not None
            assert latest.row_id == 1
            assert store.count("health") == 2  # a read outside the snapshot sees the commit
        with store.snapshot() as snap:
            assert snap.count("health") == 2

    def test_a_snapshot_that_fails_gives_its_connection_back(self, db_path: Path) -> None:
        def fail_inside(store: Store) -> None:
            with store.snapshot() as snap:
                snap.count("health")
                raise RuntimeError("boom")

        with Store.open(db_path, max_readers=1) as store:
            with pytest.raises(RuntimeError, match="boom"):
                fail_inside(store)
            assert store.count("health") == 0  # the only pooled connection is free again

    def test_a_snapshot_reads_the_cursors_too(self, store: Store) -> None:
        store.advance_cursor("influx", "health", 3)
        with store.snapshot() as snap:
            assert snap.cursor("influx", "health") == 3  # the first read fixes the state
            store.advance_cursor("influx", "health", 9)
            assert snap.cursor("influx", "health") == 3
            assert [c.last_row_id for c in snap.cursors()] == [3]
        assert store.cursor("influx", "health") == 9

    def test_a_reader_cannot_write(self, store: Store, db_path: Path) -> None:
        store.write(make_health(T0))
        with StoreReader.open(db_path) as reader:
            assert not hasattr(reader, "write")
            with reader._pool.connection() as connection, pytest.raises(sqlite3.OperationalError):
                connection.execute('DELETE FROM "health"')

    def test_a_reader_opens_a_database_that_no_writer_holds(self, db_path: Path) -> None:
        with Store.open(db_path) as writer:
            writer.write(make_health(T0))
        with StoreReader.open(db_path) as reader:  # the WAL files are gone after a clean close
            assert reader.count("health") == 1


class TestAnOlderDatabase:
    """A database that an older release wrote lacks the columns of the fields added since.

    `web` and the tools open the store read-only, so they cannot migrate it. They read the old
    rows with `None` for the new fields. Only a writer (`core`) adds the columns.
    """

    @pytest.fixture
    def old_db(self, db_path: Path) -> Path:
        with Store.open(db_path) as writer:
            writer.write_many(
                [make_health(T0 + n, queue_depth=n, heater_duty=0.25) for n in range(3)]
            )
        with raw(db_path) as connection:  # what the table looked like before the fields
            connection.execute('ALTER TABLE "health" DROP COLUMN "queue_depth"')
            connection.execute('ALTER TABLE "health" DROP COLUMN "heater_duty"')
            connection.commit()
        return db_path

    def test_every_read_gives_none_for_a_column_that_the_table_lacks(self, old_db: Path) -> None:
        with StoreReader.open(old_db) as reader:
            latest = reader.latest("health")
            ranged = reader.range("health", T0, T0 + 10, descending=True)
            after = reader.after("health", 1)
            with reader.snapshot() as snapshot:
                snapped = snapshot.latest("health")
        assert latest is not None
        assert snapped is not None
        for row in [latest, snapped, *ranged, *after]:
            assert row.values["queue_depth"] is None
            assert row.values["heater_duty"] is None
            assert row.values["state"] == make_health().state  # the other columns read as before
        assert [row.values["t_utc_ns"] - T0 for row in ranged] == [2, 1, 0]
        assert [row.row_id for row in after] == [2, 3]

    def test_a_record_builds_from_a_row_of_the_old_table(self, old_db: Path) -> None:
        with StoreReader.open(old_db) as reader:
            latest = reader.latest("health")
        assert latest is not None
        record = record_from_row("health", latest)
        assert (record.queue_depth, record.heater_duty) == (None, None)  # type: ignore[attr-defined]
        assert record.t_utc_ns == T0 + 2

    def test_a_table_that_does_not_exist_still_fails_the_read(self, old_db: Path) -> None:
        with raw(old_db) as connection:
            connection.execute('DROP TABLE "event"')
            connection.commit()
        with StoreReader.open(old_db) as reader, pytest.raises(sqlite3.OperationalError):
            reader.latest("event")

    def test_a_writer_adds_the_columns_and_the_old_rows_keep_their_place(
        self, old_db: Path
    ) -> None:
        with Store.open(old_db) as store:
            store.write(make_health(T0 + 3, queue_depth=7, heater_duty=0.5))
            rows = store.range("health", T0, T0 + 10)
        assert [row.values["queue_depth"] for row in rows] == [None, None, None, 7]
        assert [row.values["heater_duty"] for row in rows] == [None, None, None, 0.5]
        with raw(old_db) as connection:
            names = [row[1] for row in connection.execute('PRAGMA table_info("health")')]
        assert set(names[-2:]) == {"queue_depth", "heater_duty"}  # the migration appends them

    def test_a_pointing_table_from_before_the_pole_fields_serves_them_as_none(
        self, db_path: Path
    ) -> None:
        with Store.open(db_path) as writer:
            writer.write(
                sample_record("pointing", t_utc_ns=T0, polaris_x_px=2690.5, polaris_y_px=1400.25)
            )
        with raw(db_path) as connection:
            connection.execute('ALTER TABLE "pointing" DROP COLUMN "pole_x_px"')
            connection.execute('ALTER TABLE "pointing" DROP COLUMN "pole_y_px"')
            connection.commit()
        with StoreReader.open(db_path) as reader:  # `web` before `core` has migrated the table
            old = reader.latest("pointing")
        assert old is not None
        assert (old.values["pole_x_px"], old.values["pole_y_px"]) == (None, None)
        assert old.values["polaris_x_px"] == 2690.5
        with Store.open(db_path) as store:  # `core` opens the store and adds the columns
            store.write(
                sample_record("pointing", t_utc_ns=T0 + 1, pole_x_px=2100.5, pole_y_px=1399.5)
            )
            rows = store.range("pointing", T0, T0 + 10)
        assert [(r.values["pole_x_px"], r.values["pole_y_px"]) for r in rows] == [
            (None, None),
            (2100.5, 1399.5),
        ]
        with raw(db_path) as connection:
            info = {row[1]: row for row in connection.execute('PRAGMA table_info("pointing")')}
        for name in ("pole_x_px", "pole_y_px"):
            assert (info[name][2], info[name][3]) == ("REAL", 0)  # a REAL column that allows NULL


class TestCursors:
    def test_a_sink_without_a_cursor_starts_at_zero(self, store: Store) -> None:
        assert store.cursor("influx", "health") == 0
        assert store.cursors() == []

    def test_advance_stores_the_last_acknowledged_row(self, store: Store) -> None:
        assert store.advance_cursor("influx", "health", 5, t_utc_ns=T0) == 5
        store.advance_cursor("influx", "event", 2)
        store.advance_cursor("archive", "health", 9, t_utc_ns=T0 + 1)
        assert store.cursor("influx", "health") == 5
        assert store.cursors() == [
            SinkCursor("archive", "health", 9, T0 + 1),
            SinkCursor("influx", "event", 2, None),
            SinkCursor("influx", "health", 5, T0),
        ]

    def test_the_cursor_never_moves_back(self, store: Store) -> None:
        store.advance_cursor("influx", "health", 5)
        store.advance_cursor("influx", "health", 5)  # the same value is fine
        with pytest.raises(ValueError, match="cannot move back"):
            store.advance_cursor("influx", "health", 4)
        with pytest.raises(ValueError, match="negative"):
            store.advance_cursor("influx", "health", -1)
        assert store.cursor("influx", "health") == 5

    def test_a_cursor_needs_a_table_record_type(self, store: Store) -> None:
        with pytest.raises(UnknownRecordTypeError):
            store.advance_cursor("influx", "no_such_type", 1)
        with pytest.raises(ValueError, match="segment"):
            store.advance_cursor("influx", "frame", 1)

    def test_reset_makes_a_sink_backfill_again(self, store: Store) -> None:
        store.advance_cursor("influx", "health", 5)
        store.advance_cursor("influx", "event", 3)
        store.advance_cursor("other", "health", 1)
        assert store.reset_cursor("influx", "health") == 1
        assert store.cursor("influx", "health") == 0
        assert store.cursor("influx", "event") == 3
        assert store.reset_cursor("influx") == 1
        assert store.cursors() == [SinkCursor("other", "health", 1, None)]
        assert store.reset_cursor("influx") == 0

    def test_cursors_survive_a_restart(self, db_path: Path) -> None:
        with Store.open(db_path) as first:
            first.advance_cursor("influx", "health", 12, t_utc_ns=T0)
        with Store.open(db_path) as second:
            assert second.cursor("influx", "health") == 12

    def test_a_reader_lists_the_cursors(self, store: Store, db_path: Path) -> None:
        store.advance_cursor("influx", "health", 3)
        with StoreReader.open(db_path) as reader:
            assert reader.cursor("influx", "health") == 3
            assert [c.sink for c in reader.cursors()] == ["influx"]


class TestExpire:
    def test_a_record_type_with_a_retention_expires_by_time(self, store: Store) -> None:
        star_lists = [make_star_list(T0 + n * NS_PER_DAY) for n in range(5)]
        store.write_many(star_lists)
        deleted = store.expire("star_list", T0 + 2 * NS_PER_DAY)
        assert deleted == 2
        assert store.count("star_list") == 3
        assert store.last_row_id("star_list") == 5  # the IDs of deleted rows never return
        assert store.write(make_star_list(T0 + 9 * NS_PER_DAY)) == 6

    def test_results_are_kept_forever(self, store: Store) -> None:
        store.write(make_health(T0))
        with pytest.raises(ValueError, match="kept forever"):
            store.expire("health", T0 + 10)
        assert store.count("health") == 1

    def test_a_sink_cursor_stays_valid_after_rows_expire(self, store: Store) -> None:
        store.write_many([make_star_list(T0 + n * NS_PER_DAY) for n in range(3)])
        store.advance_cursor("influx", "star_list", 1)
        store.expire("star_list", T0 + 2 * NS_PER_DAY)
        assert [
            row.row_id for row in store.after("star_list", store.cursor("influx", "star_list"))
        ] == [3]
