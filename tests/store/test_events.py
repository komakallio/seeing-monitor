"""Events that share a station and a timestamp never collide: the store moves them by 1 ns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from seeingmon.records.system import EventRecord
from seeingmon.store.db import DuplicateRecordError, Store, record_from_row
from tests.store.builders import T0, make_event, make_health, make_window


def times(store: Store, station_id: str | None = None) -> list[int]:
    rows = store.after("event", 0, 1000)
    return [
        row.values["t_utc_ns"]
        for row in rows
        if station_id is None or row.values["station_id"] == station_id
    ]


class TestEventsAtOneInstant:
    def test_five_events_at_one_time_land_one_nanosecond_apart_in_the_order_written(
        self, store: Store
    ) -> None:
        for n in range(5):
            store.write(make_event(T0, message=f"event {n}"))
        rows = store.after("event", 0)
        assert [row.row_id for row in rows] == [1, 2, 3, 4, 5]
        assert [row.values["t_utc_ns"] for row in rows] == [T0 + n for n in range(5)]
        assert [row.values["message"] for row in rows] == [f"event {n}" for n in range(5)]

    def test_a_batch_that_collides_with_the_rows_of_an_earlier_run_moves_past_them(
        self, db_path: Path
    ) -> None:
        with Store.open(db_path) as first_run:
            first_run.write_many([make_event(T0), make_event(T0), make_event(T0 + 1)])
            assert times(first_run) == [T0, T0 + 1, T0 + 2]
        with Store.open(db_path) as second_run:
            ids = second_run.write_many(
                [
                    make_event(T0, message="a"),
                    make_event(T0 + 1, message="b"),
                    make_event(T0 + 2, message="c"),
                    make_event(T0 + 10, message="d"),  # no collision
                ]
            )
            assert ids == [4, 5, 6, 7]
            assert times(second_run) == [T0, T0 + 1, T0 + 2, T0 + 3, T0 + 4, T0 + 5, T0 + 10]
            messages = [row.values["message"] for row in second_run.after("event", 3)]
            assert messages == ["a", "b", "c", "d"]

    def test_the_rule_repeats_until_the_time_is_free(self, store: Store) -> None:
        store.write(make_event(T0))
        store.write(make_event(T0))  # moves to T0 + 1
        store.write(make_event(T0 + 1))  # T0 + 1 is taken now, so it moves to T0 + 2
        assert times(store) == [T0, T0 + 1, T0 + 2]

    def test_a_gap_stays_free_for_an_event_that_asks_for_it(self, store: Store) -> None:
        store.write(make_event(T0))
        store.write(make_event(T0 + 5))
        store.write(make_event(T0 + 5))  # moves to T0 + 6
        store.write(make_event(T0 + 3))  # free, so it stays
        assert sorted(times(store)) == [T0, T0 + 3, T0 + 5, T0 + 6]

    def test_a_long_run_at_one_time_stays_strictly_increasing(self, store: Store) -> None:
        count = 300  # more than one chunk of the free-time scan
        store.write_many([make_event(T0, message=str(n)) for n in range(count)])
        assert times(store) == [T0 + n for n in range(count)]
        store.write(make_event(T0))
        assert times(store)[-1] == T0 + count

    def test_events_of_another_station_do_not_collide(self, store: Store) -> None:
        store.write(make_event(T0, station_id="a"))
        store.write(make_event(T0, station_id="b"))
        store.write(make_event(T0, station_id="a"))
        assert times(store, "a") == [T0, T0 + 1]
        assert times(store, "b") == [T0]

    def test_an_event_that_collides_in_time_with_another_revision_also_moves(
        self, store: Store
    ) -> None:
        store.write(make_event(T0))
        store.write(make_event(T0, revision=1))
        rows = store.after("event", 0)
        assert [(row.values["t_utc_ns"], row.values["revision"]) for row in rows] == [
            (T0, 0),
            (T0 + 1, 1),
        ]

    def test_a_moved_event_keeps_every_other_field(self, store: Store) -> None:
        original = make_event(
            T0,
            level="warning",
            kind="retention.early_delete",
            message="Deleted files.",
            detail={"tier": "metrics", "files": 3},
            quality={"detail": "estimated"},
            provenance={"software": "9.9"},
        )
        store.write(make_event(T0))
        store.write(original)
        row = store.latest("event")
        assert row is not None
        moved = record_from_row("event", row)
        assert isinstance(moved, EventRecord)
        assert moved == original.model_copy(update={"t_utc_ns": T0 + 1})

    def test_the_unique_key_of_every_stored_event_is_distinct(self, store: Store) -> None:
        store.write_many([make_event(T0) for _ in range(20)])
        keys = {
            (row.values["station_id"], row.values["t_utc_ns"], row.values["revision"])
            for row in store.after("event", 0)
        }
        assert len(keys) == 20


class TestOtherRecordTypesKeepTheStrictRule:
    @pytest.mark.parametrize("maker", [make_health, make_window], ids=["health", "seeing_window"])
    def test_a_colliding_key_is_a_duplicate_result(self, store: Store, maker: Any) -> None:
        store.write(maker(T0))
        with pytest.raises(DuplicateRecordError):
            store.write(maker(T0))
        with pytest.raises(DuplicateRecordError):
            store.write_many([maker(T0 + 1), maker(T0)])
        assert store.latest(maker(T0).record_type) is not None

    def test_the_station_and_time_rule_does_not_apply_to_a_new_revision(self, store: Store) -> None:
        store.write(make_window(T0))
        store.write(make_window(T0, revision=1))  # a correction keeps its time
        rows = store.range("seeing_window", T0, T0 + 1, all_revisions=True)
        assert [row.values["t_utc_ns"] for row in rows] == [T0, T0]
