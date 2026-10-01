"""`ReopeningReader`: a store that opens when `core` has made it, and again after a failure."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.records.samples import sample_record
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.data import StoreData, StoreUnavailableError
from seeingmon.services.web.reader import ReopeningReader, StoreNotOpenError
from seeingmon.store.db import Store, StoreClosedError, StoreFormatError, StoreReader
from tests.services.web.client import TestClient
from tests.services.web.seed import NOW_NS, STATION, at, common


class CountingOpener:
    """Open the store as `StoreReader.open` does, and count the attempts."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, path: Path) -> StoreReader:
        self.calls += 1
        return StoreReader.open(path)


def is_open(reader: ReopeningReader) -> bool:
    """The `is_open` flag, read through a call so that a type checker does not narrow it."""
    return reader.is_open


@pytest.fixture
def opener() -> CountingOpener:
    return CountingOpener()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "db" / "results.sqlite"


def test_a_store_that_does_not_exist_yet_cannot_be_read_and_is_not_retried_at_once(
    db_path: Path, opener: CountingOpener
) -> None:
    clock = VirtualClock(NOW_NS)
    reader = ReopeningReader(db_path, clock, retry_s=5.0, opener=opener)
    with pytest.raises(FileNotFoundError):
        reader.latest("event")
    for _ in range(3):
        with pytest.raises(StoreNotOpenError):
            reader.latest("event")
    assert opener.calls == 1  # the next three reads waited for the retry time
    assert reader.opens == 0
    assert not reader.is_open


def test_the_wait_raises_an_error_that_the_data_layer_turns_into_one_answer(
    db_path: Path, opener: CountingOpener
) -> None:
    clock = VirtualClock(NOW_NS)
    reader = ReopeningReader(db_path, clock, retry_s=5.0, opener=opener)
    data = StoreData(reader, WebSettings(), clock, station_id=STATION)
    with pytest.raises(StoreUnavailableError):
        data.latest("health")  # the first attempt: the file is missing
    with pytest.raises(StoreUnavailableError):
        data.latest("health")  # the second: the reader waits
    assert opener.calls == 1


def test_the_reader_opens_the_store_once_core_has_made_it(
    db_path: Path, opener: CountingOpener
) -> None:
    clock = VirtualClock(NOW_NS)
    reader = ReopeningReader(db_path, clock, retry_s=5.0, opener=opener)
    with pytest.raises(FileNotFoundError):
        reader.latest("event")
    with Store.open(db_path) as writer:
        writer.write(
            sample_record("event", **common(at(1), level="info", kind="a.b", message="hello"))
        )
        clock.advance(4.9)
        with pytest.raises(StoreNotOpenError):
            reader.latest("event")
        clock.advance(0.2)
        found = reader.latest("event")
        assert found is not None
        assert found.values["message"] == "hello"
        assert reader.is_open
        assert reader.opens == 1
        assert [row.values["message"] for row in reader.range("event", 0, at(60))] == ["hello"]
        for _ in range(3):
            reader.latest("event")
        assert opener.calls == 2  # the failed first attempt, and the open one
    reader.close()


def test_a_file_that_is_not_a_store_is_refused_and_retried_later(
    db_path: Path, opener: CountingOpener
) -> None:
    db_path.parent.mkdir(parents=True)
    sqlite3.connect(db_path).close()  # an empty database: no store created it yet
    clock = VirtualClock(NOW_NS)
    reader = ReopeningReader(db_path, clock, retry_s=1.0, opener=opener)
    with pytest.raises(StoreFormatError):
        reader.latest("event")
    clock.advance(2)
    with Store.open(db_path):
        assert reader.latest("event") is None
        assert reader.is_open
        reader.close()


def test_a_failed_read_closes_the_reader_and_the_next_read_opens_it_again(
    db_path: Path, opener: CountingOpener
) -> None:
    clock = VirtualClock(NOW_NS)
    with Store.open(db_path):
        reader = ReopeningReader(db_path, clock, retry_s=1.0, opener=opener)
        assert reader.latest("event") is None
        assert reader.opens == 1
        underlying = reader._store  # the test looks inside on purpose
        assert underlying is not None
        underlying.close()  # the pool closes, as a store whose file went away would fail
        with pytest.raises(StoreClosedError):
            reader.latest("event")
        assert not is_open(reader)
        clock.advance(1.5)
        assert reader.latest("event") is None
        assert reader.opens == 2
        reader.close()


def test_close_is_safe_to_repeat_and_a_later_read_opens_again(
    db_path: Path, opener: CountingOpener
) -> None:
    clock = VirtualClock(NOW_NS)
    with Store.open(db_path):
        reader = ReopeningReader(db_path, clock, opener=opener)
        reader.latest("event")
        reader.close()
        reader.close()
        assert not is_open(reader)
        assert reader.latest("event") is None
        assert is_open(reader)
        reader.close()


def test_the_health_answer_goes_from_503_to_200_when_the_store_appears(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    db_path: Path,
    clock: VirtualClock,
    settings: WebSettings,
) -> None:
    """A web process that starts before `core` makes the store keeps running and recovers."""
    reader = ReopeningReader(db_path, clock, retry_s=5.0)
    client = open_client(make_app(store=reader))
    first = client.get("/api/v1/health")
    assert first.status_code == 503
    assert first.json()["reasons"] == ["store_unreadable"]
    assert client.get("/api/v1/status").status_code == 200  # the process stays up
    with Store.open(db_path) as writer:
        writer.write(
            sample_record(
                "health",
                **common(
                    NOW_NS - 10 * NS_PER_S,
                    state="auto",
                    degraded=False,
                    components={"core": "ok"},
                    dark_due=False,
                ),
            )
        )
        still = client.get("/api/v1/health")
        assert still.status_code == 503  # the reader waits before it tries again
        clock.advance(6)
        recovered = client.get("/api/v1/health")
        assert recovered.status_code == 200
        assert recovered.json()["status"] == "healthy"
    reader.close()


def test_a_reader_is_a_store_source_for_every_read_the_data_layer_makes(
    db_path: Path,
) -> None:
    clock = VirtualClock(NOW_NS)
    with Store.open(db_path) as writer:
        writer.write(sample_record("event", **common(at(1), level="info", kind="a.b", message="m")))
        reader = ReopeningReader(db_path, clock)
        data = StoreData(reader, WebSettings(), clock, station_id=STATION)
        page: Any = data.events(
            time_range=data.resolve_range("2026-10-01T02:00:00Z", "2026-10-01T03:00:00Z"),
            limit=10,
        )
        assert [e["message"] for e in page.items] == ["m"]
        reader.close()
