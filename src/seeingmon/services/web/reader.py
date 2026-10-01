"""A store reader that opens the database when it needs it, and again after a failure.

The web unit sees the data directory read-only. SQLite reads a database in WAL mode through a
read-only mount only while another process (`core`) keeps the database open, so a `web` process that
starts while `core` is down cannot open the store. If `web` exited, systemd would restart it, and a
few restarts in a row would stop the unit for good. So `web` starts without the store, answers
`GET /health` with 503 (`store_unreadable`), and opens the store when a read first succeeds.

`ReopeningReader` has the read methods that `StoreData` uses (`StoreSource`). A read opens the
database if it is not open. A failed open, and a failed read, close the reader and raise an error
that `StoreData` turns into `StoreUnavailableError`. The reader tries again after `retry_s` seconds,
so a page that polls does not open a database on every request while `core` is down.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.sinks.base import StoredRow
from seeingmon.store.db import DEFAULT_LIMIT, StoreError, StoreReader

T = TypeVar("T")


class StoreNotOpenError(OSError):
    """The store is not open, and the reader waits before it tries again."""


class ReopeningReader:
    """Read the store, and open it on the first read and after a failure."""

    def __init__(
        self,
        path: Path | str,
        clock: Clock,
        *,
        retry_s: float = 5.0,
        opener: Callable[[Path], StoreReader] = StoreReader.open,
    ) -> None:
        self._path = Path(path)
        self._clock = clock
        self._retry_ns = round(retry_s * NS_PER_S)
        self._opener = opener
        self._lock = threading.Lock()
        self._store: StoreReader | None = None
        self._retry_after_ns = 0
        self.opens = 0

    @property
    def is_open(self) -> bool:
        """Whether a database connection pool is open now."""
        with self._lock:
            return self._store is not None

    def _get(self) -> StoreReader:
        with self._lock:
            if self._store is not None:
                return self._store
            if self._clock.monotonic_ns() < self._retry_after_ns:
                raise StoreNotOpenError("the store is not open yet")
            try:
                self._store = self._opener(self._path)
            except (OSError, StoreError, sqlite3.Error):
                self._retry_after_ns = self._clock.monotonic_ns() + self._retry_ns
                raise
            self.opens += 1
            return self._store

    def _drop(self, store: StoreReader) -> None:
        with self._lock:
            if self._store is store:
                self._store = None
                self._retry_after_ns = self._clock.monotonic_ns() + self._retry_ns
        store.close()

    def _call(self, read: Callable[[StoreReader], T]) -> T:
        store = self._get()
        try:
            return read(store)
        except (OSError, StoreError, sqlite3.Error):
            self._drop(store)
            raise

    def latest(self, record_type: str, *, station_id: str | None = None) -> StoredRow | None:
        """The newest record of a type. See `StoreReader.latest`."""
        return self._call(lambda store: store.latest(record_type, station_id=station_id))

    def range(
        self,
        record_type: str,
        start_ns: int,
        end_ns: int,
        limit: int = DEFAULT_LIMIT,
        *,
        station_id: str | None = None,
        descending: bool = False,
        all_revisions: bool = False,
    ) -> list[StoredRow]:
        """The records of a time range. See `StoreReader.range`."""
        return self._call(
            lambda store: store.range(
                record_type,
                start_ns,
                end_ns,
                limit,
                station_id=station_id,
                descending=descending,
                all_revisions=all_revisions,
            )
        )

    def close(self) -> None:
        """Close the database. A later read opens it again."""
        with self._lock:
            store, self._store = self._store, None
        if store is not None:
            store.close()
