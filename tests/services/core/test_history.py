"""The zero-point history over the store: the window, the new rows, and the restart of `core`."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from seeingmon.clock import NS_PER_S, VirtualClock, iso_to_utc_ns
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.services.core.history import StoreZeroPointHistory
from seeingmon.store.db import Store
from seeingmon.survey.transparency import MemoryHistory, reference_zero_point

NOW = iso_to_utc_ns("2026-03-01T12:00:00Z")
DAY_NS = 86_400 * NS_PER_S


def record(
    t_utc_ns: int,
    zero_point: float | None = 19.0,
    *,
    rms: float | None = 0.02,
    stars: int = 30,
    cloud: float | None = 0.05,
) -> SkyQualityRecord:
    return SkyQualityRecord(
        station_id="test",
        t_utc_ns=t_utc_ns,
        profile_id="profile",
        provenance={"algo": "sky-1"},
        zero_point_mag=zero_point,
        zero_point_rms_mag=rms,
        n_stars_used=stars,
        cloud_fraction=cloud,
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    opened = Store.open(tmp_path / "results.sqlite")
    try:
        yield opened
    finally:
        opened.close()


def history_of(store: Store, **options: object) -> StoreZeroPointHistory:
    return StoreZeroPointHistory(store, clock=VirtualClock(NOW), **options)  # type: ignore[arg-type]


class TestTheWindow:
    def test_the_history_starts_with_the_samples_of_the_window(self, store: Store) -> None:
        store.write(record(NOW - 90 * DAY_NS, 18.0))  # older than 60 days: not loaded
        store.write(record(NOW - 30 * DAY_NS, 19.1))
        store.write(record(NOW - 1 * DAY_NS, 19.2))
        history = history_of(store, window_days=60.0)
        assert len(history) == 2
        found = history.zero_points(NOW - 100 * DAY_NS, NOW)
        assert [round(sample.zero_point_mag, 1) for sample in found] == [19.1, 19.2]

    def test_a_record_without_a_zero_point_is_no_sample(self, store: Store) -> None:
        store.write(record(NOW - 3 * DAY_NS, None, rms=None, stars=0))
        store.write(record(NOW - 2 * DAY_NS, 19.0))
        assert len(history_of(store)) == 1

    def test_the_time_range_includes_the_start_and_excludes_the_end(self, store: Store) -> None:
        for day in (5, 4, 3):
            store.write(record(NOW - day * DAY_NS, 19.0 + day / 100))
        history = history_of(store)
        found = history.zero_points(NOW - 4 * DAY_NS, NOW - 3 * DAY_NS)
        assert [sample.t_utc_ns for sample in found] == [NOW - 4 * DAY_NS]

    def test_the_newest_samples_stay_when_there_are_more_than_the_limit(self, store: Store) -> None:
        for day in range(10, 0, -1):
            store.write(record(NOW - day * DAY_NS, 19.0 + day / 100))
        history = history_of(store, max_samples=3)
        found = history.zero_points(NOW - 20 * DAY_NS, NOW)
        assert [sample.t_utc_ns for sample in found] == [NOW - d * DAY_NS for d in (3, 2, 1)]

    def test_the_limits_are_positive(self, store: Store) -> None:
        with pytest.raises(ValueError, match="positive"):
            history_of(store, window_days=0)
        with pytest.raises(ValueError, match="positive"):
            history_of(store, max_samples=0)


class TestNewRows:
    def test_a_row_that_arrives_later_shows_up_at_the_next_question(self, store: Store) -> None:
        history = history_of(store)
        assert history.zero_points(0, NOW + DAY_NS) == ()
        store.write(record(NOW - 600 * NS_PER_S, 19.2))
        found = history.zero_points(0, NOW + DAY_NS)
        assert [round(sample.zero_point_mag, 1) for sample in found] == [19.2]
        store.write(record(NOW - 300 * NS_PER_S, 19.3))
        store.write(record(NOW - 200 * NS_PER_S, None, rms=None, stars=0))
        assert len(history.zero_points(0, NOW + DAY_NS)) == 2
        assert len(history) == 2

    def test_each_row_counts_once_however_often_the_history_is_asked(self, store: Store) -> None:
        history = history_of(store)
        store.write(record(NOW - 600 * NS_PER_S, 19.2))
        for _ in range(5):
            history.zero_points(0, NOW + DAY_NS)
        assert len(history) == 1
        assert history.refresh() == 0

    def test_a_large_arrival_is_read_in_batches(self, store: Store, monkeypatch: object) -> None:
        import seeingmon.services.core.history as module

        monkeypatch.setattr(module, "BATCH_ROWS", 7)  # type: ignore[attr-defined]
        history = history_of(store)
        for minute in range(30):
            store.write(record(NOW - (30 - minute) * 60 * NS_PER_S, 19.0 + minute / 1000))
        assert history.refresh() == 30
        assert len(history) == 30


class TestARestart:
    def test_the_reference_survives_a_restart_of_core(self, tmp_path: Path) -> None:
        path = tmp_path / "results.sqlite"
        zero_points = [19.0 + index / 100 for index in range(25)]
        with Store.open(path) as first:
            for index, value in enumerate(zero_points):
                first.write(record(NOW - (25 - index) * 3600 * NS_PER_S, value))
            before = StoreZeroPointHistory(first, clock=VirtualClock(NOW))
            reference = reference_zero_point(before, NOW)
        assert reference is not None
        assert reference.n_samples == 25
        with Store.open(path) as second:  # core starts again
            after = StoreZeroPointHistory(second, clock=VirtualClock(NOW))
            again = reference_zero_point(after, NOW)
        assert again == reference
        expected = MemoryHistory()
        for index, value in enumerate(zero_points):
            expected.add_record(record(NOW - (25 - index) * 3600 * NS_PER_S, value))
        assert reference == reference_zero_point(expected, NOW)

    def test_too_few_samples_give_no_reference_after_the_restart_either(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "results.sqlite"
        with Store.open(path) as first:
            for index in range(10):  # the rule asks for 20
                first.write(record(NOW - (10 - index) * 3600 * NS_PER_S, 19.0))
        with Store.open(path) as second:
            history = StoreZeroPointHistory(second, clock=VirtualClock(NOW))
            assert reference_zero_point(history, NOW) is None
