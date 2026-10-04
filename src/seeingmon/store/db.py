"""The SQLite store of the results.

`Store` is the only writer. It holds one write connection behind a lock, and a small pool of
read-only connections, so a reader never waits for the writer and the writer never waits for a
reader. `StoreReader` is the read-only half. The `web` process and tools such as
`seeingmon store info` open it with `StoreReader.open` while `core` writes.

**Database.** The store runs SQLite in WAL mode with `synchronous=NORMAL` and a busy timeout.
This keeps the writes few and small, which suits an SD card, and a crash loses at most the
last few commits and never corrupts the file. `Store.open` creates the tables and applies the
additive migrations of `seeingmon.records.sqlite_schema`. It also stamps the file with an
application ID, so the store refuses to create tables inside an unrelated SQLite file. A
database that a newer version of the software migrated still opens, because the store accepts
columns that its declarations lack. A reader that opens the database read-only cannot migrate
it, so a database that an older release wrote can lack a declared column until `core` opens it
for writing. The reader reads such a column as `None`.

**Records.** Every table record type is a table with an autoincrement row ID. Tables are
append-only, so the cursor of a sink is the last row ID that the sink acknowledged, and a new
sink backfills from row zero. A record is immutable. Writing a second record with the same key
`(station_id, t_utc_ns, revision)` raises `DuplicateRecordError`. A correction is a new record
with the next revision (`next_revision` and `write_correction`). Per-frame metrics are not rows:
`seeingmon.store.segments` writes them to segment files.

**Events.** Two events can share a station and a timestamp, for example in virtual time, with
a coarse clock, or when one fault writes several events in a row. InfluxDB identifies a point
by its measurement, tags, and timestamp, so the second event would overwrite the first. The
store therefore moves an event that collides with an existing event of the same station by
one nanosecond, and repeats until the time is free. The events keep the order in which you
wrote them, and a producer never handles the collision. Every other record type keeps the
strict rule: a colliding key is a duplicate result, and the store raises
`DuplicateRecordError`.

**Cursors.** The `sink_cursor` table holds the last acknowledged row ID for each sink and record
type. `advance_cursor` updates it in one transaction and never moves it backward.

**Threads.** Any thread can call the write methods, and the lock serializes them. Each read
borrows its own connection from the pool, so several threads can read at once. Call `close`
once, when no thread uses the store any more.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from seeingmon.records.base import Record, get_record_type
from seeingmon.records.sqlite_schema import (
    ROW_ID,
    ensure_schema,
    insert_record,
    quote,
    select_after_sql,
    select_columns_sql,
    sqlite_to_row,
    table_record_types,
)
from seeingmon.sinks.base import StoredRow

if TYPE_CHECKING:
    from seeingmon.analysis.base import RecordWriter

# The application ID in the file header (`PRAGMA application_id`): the ASCII bytes "SMON".
APPLICATION_ID = int.from_bytes(b"SMON", "big")
# The version of the tables that the store adds to the record tables (`PRAGMA user_version`).
STORE_SCHEMA_VERSION = 1

CURSOR_TABLE = "sink_cursor"
DEFAULT_BUSY_TIMEOUT_MS = 5000
DEFAULT_READERS = 4
DEFAULT_LIMIT = 10_000

# A record type in this set can share its station and timestamp with another record of the same
# type. The store moves the later record forward by one nanosecond instead of rejecting it.
SAME_INSTANT_TYPES = frozenset({"event"})

_WAL_SIZE_LIMIT_BYTES = 64 * 1024 * 1024
_SCAN_CHUNK = 64
_SYNCHRONOUS_NAMES = {0: "off", 1: "normal", 2: "full", 3: "extra"}

_CURSOR_DDL = f"""\
CREATE TABLE IF NOT EXISTS {quote(CURSOR_TABLE)} (
    "sink" TEXT NOT NULL,
    "record_type" TEXT NOT NULL,
    "last_row_id" INTEGER NOT NULL DEFAULT 0 CHECK ("last_row_id" >= 0),
    "updated_utc_ns" INTEGER,
    PRIMARY KEY ("sink", "record_type")
) WITHOUT ROWID"""


class StoreError(Exception):
    """A problem that the store reports in its own terms."""


class StoreClosedError(StoreError):
    """You used the store after `close`."""


class StoreFormatError(StoreError):
    """The file is not a seeing-monitor store, or newer software wrote it."""


class StoreBusyError(StoreError):
    """Another process holds the write lock for longer than the busy timeout."""


class UnknownRecordTypeError(StoreError, LookupError):
    """The name is not a declared record type."""


class DuplicateRecordError(StoreError):
    """The store already holds a record with this key. A result is immutable.

    `key` is `(station_id, record_type, t_utc_ns, revision)`. To correct a result, write it
    again with the next revision.
    """

    def __init__(self, key: tuple[str, str, int, int]) -> None:
        station_id, record_type, t_utc_ns, revision = key
        super().__init__(
            f"the store already holds a {record_type} record for station {station_id!r} "
            f"at {t_utc_ns} ns with revision {revision}; write a correction as the next revision"
        )
        self.key = key


@dataclass(frozen=True, slots=True)
class SinkCursor:
    """The last row that a sink acknowledged, for one record type.

    `updated_utc_ns` is the time of the last update, or `None` when the writer gave none.
    """

    sink: str
    record_type: str
    last_row_id: int
    updated_utc_ns: int | None


def _error_code(exc: sqlite3.Error) -> int | None:
    code = getattr(exc, "sqlite_errorcode", None)
    return code if isinstance(code, int) else None


def _is_busy(exc: sqlite3.Error) -> bool:
    code = _error_code(exc)
    return code is not None and (code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)


def _is_unique_violation(exc: sqlite3.IntegrityError) -> bool:
    code = _error_code(exc)
    if code is not None:
        return code in (sqlite3.SQLITE_CONSTRAINT_UNIQUE, sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY)
    return "UNIQUE constraint failed" in str(exc)


def _rollback(connection: sqlite3.Connection) -> None:
    if connection.in_transaction:
        # If the connection is broken, closing it drops the transaction.
        with contextlib.suppress(sqlite3.Error):
            connection.execute("ROLLBACK")


def _present_columns(connection: sqlite3.Connection, cls: type[Record]) -> frozenset[str]:
    """The columns that the table of a record type has now. It is empty for a missing table."""
    info = connection.execute(f"PRAGMA table_info({quote(cls.record_type)})").fetchall()
    return frozenset(str(row[1]) for row in info)


def _stored_rows(cursor: sqlite3.Cursor, cls: type[Record]) -> list[StoredRow]:
    names = [column[0] for column in cursor.description]
    rows: list[StoredRow] = []
    for fetched in cursor.fetchall():
        values = dict(zip(names, fetched, strict=True))
        rows.append(StoredRow(row_id=int(values[ROW_ID]), values=sqlite_to_row(cls, values)))
    return rows


def _check_identity(connection: sqlite3.Connection, *, writer: bool) -> None:
    """Refuse a file that is not a store, and (for a writer) a file that is too new."""
    application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
    names = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    ours = {cls.record_type for cls in table_record_types()}
    if application_id not in (0, APPLICATION_ID) or (names and not names & ours):
        raise StoreFormatError("the file is not a seeing-monitor store database")
    if not names and not writer:
        raise StoreFormatError("the database is empty, so no store created it yet")
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if writer and version > STORE_SCHEMA_VERSION:
        raise StoreFormatError(
            f"the database has store version {version}, and this software supports up to "
            f"{STORE_SCHEMA_VERSION}; upgrade the software before you write to it"
        )


def _free_instant(
    connection: sqlite3.Connection, table: str, station_id: str, t_utc_ns: int
) -> int:
    """Return the first time at or after `t_utc_ns` that no row of the station uses.

    The scan reads the taken times in chunks, so a long run of events at one time costs one
    query for every 64 events and not one for every event.
    """
    sql = (
        f'SELECT DISTINCT "t_utc_ns" FROM {quote(table)} '
        'WHERE "station_id" = ? AND "t_utc_ns" >= ? ORDER BY "t_utc_ns" LIMIT ?'
    )
    candidate = t_utc_ns
    while True:
        rows = connection.execute(sql, (station_id, candidate, _SCAN_CHUNK)).fetchall()
        for (taken,) in rows:
            if taken != candidate:
                return candidate
            candidate += 1
        if len(rows) < _SCAN_CHUNK:
            return candidate


def _insert(connection: sqlite3.Connection, record: Record) -> int:
    """Append one record inside the caller's transaction and return its row ID."""
    cls = type(record)
    if cls.record_type in SAME_INSTANT_TYPES:
        free = _free_instant(connection, cls.record_type, record.station_id, record.t_utc_ns)
        if free != record.t_utc_ns:
            record = record.model_copy(update={"t_utc_ns": free})
    try:
        return insert_record(connection, record)
    except sqlite3.IntegrityError as exc:
        if _is_unique_violation(exc):
            raise DuplicateRecordError(record.record_key) from None
        raise


class _ReaderPool:
    """A bounded pool of read-only connections that any thread can borrow from."""

    def __init__(self, connect: Callable[[], sqlite3.Connection], size: int) -> None:
        self._connect = connect
        self._slots = threading.BoundedSemaphore(size)
        self._lock = threading.Lock()
        self._idle: list[sqlite3.Connection] = []
        self._closed = False

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self._slots.acquire()
        try:
            connection = self._take()
            try:
                yield connection
            finally:
                self._give_back(connection)
        finally:
            self._slots.release()

    def _take(self) -> sqlite3.Connection:
        with self._lock:
            if self._closed:
                raise StoreClosedError("the store is closed")
            if self._idle:
                return self._idle.pop()
        connection = self._connect()
        if self._is_closed():
            connection.close()
            raise StoreClosedError("the store is closed")
        return connection

    def _is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def _give_back(self, connection: sqlite3.Connection) -> None:
        with self._lock:
            if not self._closed:
                self._idle.append(connection)
                return
        connection.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            idle, self._idle = self._idle, []
        for connection in idle:
            connection.close()


class _ReadView:
    """The read methods. A subclass says where its connection comes from."""

    def _connection(self) -> AbstractContextManager[sqlite3.Connection]:
        raise NotImplementedError

    @staticmethod
    def _table_type(record_type: str) -> type[Record]:
        try:
            cls = get_record_type(record_type)
        except KeyError as exc:
            raise UnknownRecordTypeError(str(exc.args[0])) from None
        if cls.storage != "table":
            raise ValueError(
                f"{record_type} is a {cls.storage} record: the store keeps it in segment files, "
                "not in a table"
            )
        return cls

    def _select(
        self,
        cls: type[Record],
        build_sql: Callable[[frozenset[str]], str],
        params: dict[str, Any],
    ) -> list[StoredRow]:
        """Read whole records. `build_sql` takes the columns of the table and returns the query.

        A table that an older release wrote can lack a declared column, and a reader cannot add
        it. The query reads such a column as `NULL`, so the value is `None` in every build of
        SQLite. A missing column name in double quotes is a text literal in some builds and an
        error in others.
        """
        with self._connection() as connection:
            sql = build_sql(_present_columns(connection, cls))
            return _stored_rows(connection.execute(sql, params), cls)

    def _scalar(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        with self._connection() as connection:
            row = connection.execute(sql, params).fetchone()
        return None if row is None else row[0]

    def latest(self, record_type: str, *, station_id: str | None = None) -> StoredRow | None:
        """Return the current result with the newest time, or `None` when there is none.

        The newest time wins, and among the revisions of that time the highest revision wins,
        so a correction of an old result never displaces a newer result. Pass `station_id` to
        look at one station.
        """
        cls = self._table_type(record_type)
        table = quote(cls.record_type)
        where = ' WHERE "station_id" = :station' if station_id is not None else ""

        def build(present: frozenset[str]) -> str:
            return (
                f"SELECT {select_columns_sql(cls, present)} FROM {table}{where} "
                f'ORDER BY "t_utc_ns" DESC, "revision" DESC, "{ROW_ID}" DESC LIMIT 1'
            )

        rows = self._select(cls, build, {"station": station_id} if station_id is not None else {})
        return rows[0] if rows else None

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
        """Return the rows with `start_ns <= t_utc_ns < end_ns`, ordered by time.

        By default the result holds the current revision of each result and hides the
        revisions that a correction replaced. Pass `all_revisions=True` to see every row.
        `limit` caps the number of rows. It keeps the oldest rows of the range, or the newest
        rows when you pass `descending=True`.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        cls = self._table_type(record_type)
        table = quote(cls.record_type)
        conditions = ['"t_utc_ns" >= :start', '"t_utc_ns" < :end']
        params: dict[str, Any] = {"start": start_ns, "end": end_ns, "limit": limit}
        if station_id is not None:
            conditions.append('"station_id" = :station')
            params["station"] = station_id
        if not all_revisions:
            conditions.append(
                f"NOT EXISTS (SELECT 1 FROM {table} AS newer "
                f'WHERE newer."station_id" = {table}."station_id" '
                f'AND newer."t_utc_ns" = {table}."t_utc_ns" '
                f'AND newer."revision" > {table}."revision")'
            )
        direction = "DESC" if descending else "ASC"

        def build(present: frozenset[str]) -> str:
            return (
                f"SELECT {select_columns_sql(cls, present)} FROM {table} "
                f"WHERE {' AND '.join(conditions)} "
                f'ORDER BY "t_utc_ns" {direction}, "revision" {direction}, '
                f'"{ROW_ID}" {direction} LIMIT :limit'
            )

        return self._select(cls, build, params)

    def after(self, record_type: str, row_id: int, limit: int = 1000) -> list[StoredRow]:
        """Return up to `limit` rows with a row ID greater than `row_id`, in row order.

        This is the feed of a sink: a sink that acknowledged row `n` reads the next batch with
        `after(record_type, n, limit)`, and a new sink starts at 0. The result includes every
        revision.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        cls = self._table_type(record_type)
        return self._select(
            cls, lambda present: select_after_sql(cls, present), {"after": row_id, "limit": limit}
        )

    def count(self, record_type: str) -> int:
        """Return the number of rows in the table of a record type."""
        cls = self._table_type(record_type)
        return int(self._scalar(f"SELECT COUNT(*) FROM {quote(cls.record_type)}"))

    def count_after(self, record_type: str, row_id: int) -> int:
        """Return the number of rows with a row ID greater than `row_id`."""
        cls = self._table_type(record_type)
        sql = f'SELECT COUNT(*) FROM {quote(cls.record_type)} WHERE "{ROW_ID}" > ?'
        return int(self._scalar(sql, (row_id,)))

    def last_row_id(self, record_type: str) -> int:
        """Return the highest row ID that the table ever assigned, or 0 for an empty table.

        A deleted row keeps its ID used (`AUTOINCREMENT`), so the value never goes down.
        """
        cls = self._table_type(record_type)
        value = self._scalar("SELECT seq FROM sqlite_sequence WHERE name = ?", (cls.record_type,))
        return 0 if value is None else int(value)

    def next_revision(self, record_type: str, station_id: str, t_utc_ns: int) -> int:
        """Return the revision for a correction: one more than the highest stored, or 0."""
        cls = self._table_type(record_type)
        sql = (
            f'SELECT MAX("revision") FROM {quote(cls.record_type)} '
            'WHERE "station_id" = ? AND "t_utc_ns" = ?'
        )
        highest = self._scalar(sql, (station_id, t_utc_ns))
        return 0 if highest is None else int(highest) + 1

    def cursor(self, sink: str, record_type: str) -> int:
        """Return the last row ID that a sink acknowledged for a record type, or 0."""
        sql = (
            f'SELECT "last_row_id" FROM {quote(CURSOR_TABLE)} '
            'WHERE "sink" = ? AND "record_type" = ?'
        )
        try:
            value = self._scalar(sql, (sink, record_type))
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return 0
            raise
        return 0 if value is None else int(value)

    def cursors(self) -> list[SinkCursor]:
        """Return every stored cursor, ordered by sink and record type."""
        sql = (
            'SELECT "sink", "record_type", "last_row_id", "updated_utc_ns" '
            f'FROM {quote(CURSOR_TABLE)} ORDER BY "sink", "record_type"'
        )
        try:
            with self._connection() as connection:
                fetched = connection.execute(sql).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return []
            raise
        return [
            SinkCursor(str(sink), str(kind), int(row_id), None if stamp is None else int(stamp))
            for sink, kind, row_id, stamp in fetched
        ]


class StoreSnapshot(_ReadView):
    """The reads of one read transaction: every call sees the same state of the database.

    Get it from `StoreReader.snapshot`. Each separate read of a `StoreReader` sees the commits
    that happened since the one before, so a count and a list that you read one after the other
    can disagree while a writer commits. Read them from one snapshot when they must agree. Use a
    snapshot in one thread, and finish it soon, because it holds one of the pooled connections.
    Read through the snapshot inside the block. A read through the store itself needs a second
    pooled connection, and it waits for ever when `max_readers` is 1.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._snapshot_connection = connection

    def _connection(self) -> AbstractContextManager[sqlite3.Connection]:
        return nullcontext(self._snapshot_connection)


class StoreReader(_ReadView):
    """Read access to a store database through a pool of read-only connections.

    Open it with `StoreReader.open`. Every read returns `StoredRow` values: the row ID and a
    dict of JSON-compatible values keyed by the declared field names (`Record.to_row` form).
    `record_from_row` turns a row back into a record. Each read borrows a connection and sees the
    latest commit. Use `snapshot` for reads that must agree with one another.
    """

    def __init__(self, path: Path, *, busy_timeout_ms: int, max_readers: int) -> None:
        if max_readers < 1:
            raise ValueError("max_readers must be at least 1")
        self._path = path
        self._busy_timeout_ms = busy_timeout_ms
        self._pool = _ReaderPool(self._connect_reader, max_readers)

    @classmethod
    def open(
        cls,
        path: Path | str,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        max_readers: int = DEFAULT_READERS,
    ) -> Self:
        """Open an existing store database for reading only.

        Raises `FileNotFoundError` when the file does not exist and `StoreFormatError` when it
        is not a store database. SQLite needs write access to the directory to create the
        shared-memory file of a WAL database, unless a writer already keeps it open.
        """
        database = Path(path)
        if not database.is_file():
            raise FileNotFoundError(f"there is no store database at {database}")
        reader = cls(database, busy_timeout_ms=busy_timeout_ms, max_readers=max_readers)
        try:
            with reader._pool.connection() as connection:
                _check_identity(connection, writer=False)
        except BaseException:
            reader.close()
            raise
        return reader

    @property
    def path(self) -> Path:
        return self._path

    def _connect_reader(self) -> sqlite3.Connection:
        uri = f"{self._path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            connection.execute("PRAGMA query_only = ON")
        except BaseException:
            connection.close()
            raise
        return connection

    def _connection(self) -> AbstractContextManager[sqlite3.Connection]:
        return self._pool.connection()

    @contextmanager
    def snapshot(self) -> Iterator[StoreSnapshot]:
        """Open a read transaction, so that every read inside the block sees one state.

        SQLite in WAL mode gives the transaction the database as of its first read, and later
        commits stay invisible until the block ends:

            with store.snapshot() as snap:
                count = snap.count("health")
                rows = snap.after("health", 0)  # holds the rows that `count` counted, no more
        """
        with self._pool.connection() as connection:
            connection.execute("BEGIN")
            try:
                yield StoreSnapshot(connection)
            finally:
                _rollback(connection)  # the transaction only read, so this just ends it

    def close(self) -> None:
        """Close every connection. Calling it again does nothing."""
        self._pool.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def diagnostics(self) -> dict[str, Any]:
        """Return facts about the database file: the journal mode, the IDs, and the size."""
        with self._pool.connection() as connection:
            pragma = {
                name: connection.execute(f"PRAGMA {name}").fetchone()[0]
                for name in (
                    "journal_mode",
                    "application_id",
                    "user_version",
                    "page_count",
                    "page_size",
                    "freelist_count",
                )
            }
        return {
            "journal_mode": str(pragma["journal_mode"]),
            "application_id": int(pragma["application_id"]),
            "user_version": int(pragma["user_version"]),
            "size_bytes": int(pragma["page_count"]) * int(pragma["page_size"]),
            "free_bytes": int(pragma["freelist_count"]) * int(pragma["page_size"]),
            "sqlite_version": sqlite3.sqlite_version,
        }


class Store(StoreReader):
    """The writer: appends records, keeps the sink cursors, and reads like `StoreReader`.

    Open it with `Store.open`. It satisfies the `RecordWriter` protocol of
    `seeingmon.analysis.base` at run time. The protocol types `write` as returning `None`,
    and this class returns the row ID, so for a static type check pass `as_record_writer()`.
    """

    def __init__(self, path: Path, *, busy_timeout_ms: int, max_readers: int) -> None:
        super().__init__(path, busy_timeout_ms=busy_timeout_ms, max_readers=max_readers)
        self._write_lock = threading.Lock()
        self._writer: sqlite3.Connection | None = None

    @classmethod
    def open(
        cls,
        path: Path | str,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        max_readers: int = DEFAULT_READERS,
    ) -> Self:
        """Open the database for writing, and create it with its schema when it is missing.

        Raises `StoreFormatError` when the file is not a store database or a newer version of
        the software wrote it, `SchemaError` (from `seeingmon.records.sqlite_schema`) when the
        tables do not fit the declarations and a migration cannot fix them, and
        `StoreBusyError` when another process holds the write lock past the busy timeout.
        """
        database = Path(path)
        database.parent.mkdir(parents=True, exist_ok=True)
        store = cls(database, busy_timeout_ms=busy_timeout_ms, max_readers=max_readers)
        try:
            store._writer = store._open_writer()
        except BaseException:
            store.close()
            raise
        return store

    def _open_writer(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self._path),
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0])
            if mode.lower() != "wal":
                raise StoreFormatError(
                    f"the file system does not support WAL mode (SQLite chose {mode!r})"
                )
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute(f"PRAGMA journal_size_limit = {_WAL_SIZE_LIMIT_BYTES}")
            connection.execute("PRAGMA temp_store = MEMORY")
            self._initialize(connection)
        except BaseException:
            connection.close()
            raise
        return connection

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            if _is_busy(exc):
                raise StoreBusyError("another process holds the write lock") from exc
            raise
        try:
            _check_identity(connection, writer=True)
            ensure_schema(connection, allow_extra_columns=True)
            connection.execute(_CURSOR_DDL)
            connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {STORE_SCHEMA_VERSION}")
            connection.execute("COMMIT")
        except BaseException:
            _rollback(connection)
            raise

    def close(self) -> None:
        """Roll back an open transaction, close every connection, and stop the pool.

        Calling it again does nothing. A transaction that did not commit is not part of the
        database.
        """
        with self._write_lock:
            connection, self._writer = self._writer, None
            if connection is not None:
                _rollback(connection)
                connection.close()
        super().close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            connection = self._writer
            if connection is None:
                raise StoreClosedError("the store is closed")
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                if _is_busy(exc):
                    raise StoreBusyError("another process holds the write lock") from exc
                raise
            try:
                yield connection
                connection.execute("COMMIT")
            except BaseException as exc:
                _rollback(connection)
                if isinstance(exc, sqlite3.Error) and _is_busy(exc):
                    raise StoreBusyError("another process holds the write lock") from exc
                raise

    # --- writes ------------------------------------------------------------------------------

    @staticmethod
    def _check_writable(record: Record) -> None:
        cls = type(record)
        if not hasattr(cls, "record_type"):
            raise TypeError("write a record of a declared type")
        if cls.storage != "table":
            raise ValueError(
                f"{cls.record_type} is a {cls.storage} record: write it to a segment file with "
                "SegmentWriter, not to the store"
            )

    def write(self, record: Record) -> int:
        """Append a record and return its row ID.

        The write is one transaction. Raises `DuplicateRecordError` when the store already
        holds a record with the same key, except for an event, which moves forward by one
        nanosecond until its time is free (see the module documentation).
        """
        self._check_writable(record)
        with self._transaction() as connection:
            return _insert(connection, record)

    def write_many(self, records: Iterable[Record]) -> list[int]:
        """Append records in one transaction and return their row IDs, in order.

        Either every record lands or none does. A duplicate key rolls the whole batch back.
        """
        items = list(records)
        for record in items:
            self._check_writable(record)
        if not items:
            return []
        with self._transaction() as connection:
            return [_insert(connection, record) for record in items]

    def write_correction(self, record: Record) -> int:
        """Append the record as the next revision of its station and time, and return its row ID.

        Use it to reprocess a result: the new row replaces the old one for readers that ask
        for the current revision, and the old row stays. The revision that the record carries
        is ignored. Events are never corrected, so this raises `ValueError` for an event.
        """
        self._check_writable(record)
        cls = type(record)
        if cls.record_type in SAME_INSTANT_TYPES:
            raise ValueError(f"a {cls.record_type} is never corrected; write a new one")
        with self._transaction() as connection:
            sql = (
                f'SELECT MAX("revision") FROM {quote(cls.record_type)} '
                'WHERE "station_id" = ? AND "t_utc_ns" = ?'
            )
            highest = connection.execute(sql, (record.station_id, record.t_utc_ns)).fetchone()[0]
            revision = 0 if highest is None else int(highest) + 1
            return _insert(connection, record.model_copy(update={"revision": revision}))

    def expire(self, record_type: str, before_t_utc_ns: int) -> int:
        """Delete the rows of a record type with `t_utc_ns` before the cutoff. Return the count.

        Only a record type that declares `retention_days` expires. Results are kept forever,
        so any other type raises `ValueError`. The row IDs of deleted rows never return.
        """
        cls = self._table_type(record_type)
        if cls.retention_days is None:
            raise ValueError(f"{record_type} records are kept forever and never expire")
        with self._transaction() as connection:
            cursor = connection.execute(
                f'DELETE FROM {quote(cls.record_type)} WHERE "t_utc_ns" < ?', (before_t_utc_ns,)
            )
            return cursor.rowcount

    def advance_cursor(
        self, sink: str, record_type: str, row_id: int, *, t_utc_ns: int | None = None
    ) -> int:
        """Store the last row ID that a sink acknowledged, and return it.

        The update is one transaction. A value below the stored cursor raises `ValueError`:
        a sink never moves back, except through `reset_cursor`. `t_utc_ns` records when the
        sink acknowledged the row.
        """
        cls = self._table_type(record_type)
        if row_id < 0:
            raise ValueError("a row ID is never negative")
        with self._transaction() as connection:
            row = connection.execute(
                f'SELECT "last_row_id" FROM {quote(CURSOR_TABLE)} '
                'WHERE "sink" = ? AND "record_type" = ?',
                (sink, cls.record_type),
            ).fetchone()
            if row is not None and row_id < int(row[0]):
                raise ValueError(
                    f"the cursor of sink {sink!r} for {record_type} is at {int(row[0])}, so it "
                    f"cannot move back to {row_id}; use reset_cursor to backfill again"
                )
            connection.execute(
                f"INSERT INTO {quote(CURSOR_TABLE)} "
                '("sink", "record_type", "last_row_id", "updated_utc_ns") VALUES (?, ?, ?, ?) '
                'ON CONFLICT ("sink", "record_type") DO UPDATE SET '
                '"last_row_id" = excluded."last_row_id", '
                '"updated_utc_ns" = excluded."updated_utc_ns"',
                (sink, cls.record_type, row_id, t_utc_ns),
            )
        return row_id

    def reset_cursor(self, sink: str, record_type: str | None = None) -> int:
        """Delete the cursors of a sink, so that it backfills from row zero. Return the count.

        Pass `record_type` to reset one record type, or leave it out to reset them all.
        """
        with self._transaction() as connection:
            if record_type is None:
                cursor = connection.execute(
                    f'DELETE FROM {quote(CURSOR_TABLE)} WHERE "sink" = ?', (sink,)
                )
            else:
                cursor = connection.execute(
                    f'DELETE FROM {quote(CURSOR_TABLE)} WHERE "sink" = ? AND "record_type" = ?',
                    (sink, record_type),
                )
            return cursor.rowcount

    def checkpoint(self) -> None:
        """Copy the WAL into the database file and shrink the WAL file.

        SQLite checkpoints on its own. Call this before you copy the database file.
        """
        with self._write_lock:
            if self._writer is None:
                raise StoreClosedError("the store is closed")
            self._writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()

    def diagnostics(self) -> dict[str, Any]:
        """Return the facts of `StoreReader.diagnostics`, plus the write connection settings."""
        facts = super().diagnostics()
        with self._write_lock:
            if self._writer is None:
                raise StoreClosedError("the store is closed")
            synchronous = int(self._writer.execute("PRAGMA synchronous").fetchone()[0])
            timeout = int(self._writer.execute("PRAGMA busy_timeout").fetchone()[0])
        facts["synchronous"] = _SYNCHRONOUS_NAMES.get(synchronous, str(synchronous))
        facts["busy_timeout_ms"] = timeout
        return facts

    def as_record_writer(self) -> RecordWriter:
        """Return a `RecordWriter` that writes to this store and discards the row ID.

        `RecordWriter.write` returns `None`, and `Store.write` returns the row ID, so a static
        type checker rejects a `Store` where a `RecordWriter` is expected. This wrapper fits.
        """
        return _StoreRecordWriter(self)


class _StoreRecordWriter:
    def __init__(self, store: Store) -> None:
        self._store = store

    def write(self, record: Record) -> None:
        self._store.write(record)


def record_from_row(record_type: str, row: StoredRow) -> Record:
    """Build the record of a stored row. The row holds `to_row` values, as every read returns."""
    try:
        cls = get_record_type(record_type)
    except KeyError as exc:
        raise UnknownRecordTypeError(str(exc.args[0])) from None
    return cls.from_row(row.values)
