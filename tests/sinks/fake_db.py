"""A fake DB-API connection that records every call, for the tests of the PostgreSQL sink.

The exception classes copy the names of the DB-API hierarchy (PEP 249), which is all the sink
reads. `FakeDatabase` also simulates upserts by key, so a test can compare what the sink wrote
with what it should have written.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from seeingmon.sinks.timescale import EXTENSION_QUERY


class Error(Exception):
    pass


class InterfaceError(Error):
    pass


class DatabaseError(Error):
    pass


class OperationalError(DatabaseError):
    pass


class InternalError(DatabaseError):
    pass


class ProgrammingError(DatabaseError):
    pass


class IntegrityError(DatabaseError):
    pass


class DataError(DatabaseError):
    pass


class NotSupportedError(DatabaseError):
    pass


class AdminShutdown(OperationalError):  # noqa: N818 - the name of the psycopg class
    """Like `psycopg.errors.AdminShutdown`: a subclass with a SQLSTATE."""

    sqlstate = "57P01"


@dataclass
class Diagnostics:
    message_primary: str


class UndefinedTable(ProgrammingError):  # noqa: N818 - the name of the psycopg class
    """Like `psycopg.errors.UndefinedTable`, with the server's diagnostics."""

    sqlstate = "42P01"

    def __init__(self, text: str, primary: str) -> None:
        super().__init__(text)
        self.diag = Diagnostics(primary)


@dataclass
class FakeDatabase:
    """The state behind the fake connections.

    `log` lists every call as a tuple, in order: `("connect",)`, `("execute", sql)`,
    `("executemany", sql, parameters)`, `("commit",)`, `("rollback",)`, and `("close",)`.
    `batch_failures` and `connect_failures` hold the errors that the next batches and the next
    connection attempts raise (`None` lets one through). `execute_failure` is raised by an
    `execute` whose statement contains the given text. `timescale` says whether the extension
    exists. `rows` holds the upserted rows by table and key.
    """

    timescale: bool = True
    log: list[tuple[Any, ...]] = field(default_factory=list)
    batch_failures: deque[BaseException | None] = field(default_factory=deque)
    connect_failures: deque[BaseException | None] = field(default_factory=deque)
    execute_failure: tuple[str, BaseException] | None = None
    rows: dict[str, dict[tuple[Any, ...], tuple[Any, ...]]] = field(default_factory=dict)
    connections: list[FakeConnection] = field(default_factory=list)

    def connect(self) -> FakeConnection:
        self.log.append(("connect",))
        if self.connect_failures:
            failure = self.connect_failures.popleft()
            if failure is not None:
                raise failure
        connection = FakeConnection(self)
        self.connections.append(connection)
        return connection

    def calls(self, kind: str) -> list[tuple[Any, ...]]:
        return [entry for entry in self.log if entry[0] == kind]

    def statements(self) -> list[str]:
        return [entry[1] for entry in self.log if entry[0] == "execute"]

    def upsert(self, sql: str, parameters: list[tuple[Any, ...]]) -> None:
        table = sql.split('"')[1]
        stored = self.rows.setdefault(table, {})
        for row in parameters:
            stored[row[:3]] = row  # the key is the first three columns: station, time, revision


class FakeCursor:
    def __init__(self, database: FakeDatabase) -> None:
        self._database = database
        self._row: tuple[Any, ...] | None = None
        self.closed = False

    def execute(self, sql: str, params: Any = None) -> None:
        self._database.log.append(("execute", sql))
        failure = self._database.execute_failure
        if failure is not None and failure[0] in sql:
            raise failure[1]
        if sql == EXTENSION_QUERY:
            self._row = (1,) if self._database.timescale else None

    def executemany(self, sql: str, params_seq: Any) -> None:
        parameters = [tuple(row) for row in params_seq]
        self._database.log.append(("executemany", sql, parameters))
        if self._database.batch_failures:
            failure = self._database.batch_failures.popleft()
            if failure is not None:
                raise failure
        self._database.upsert(sql, parameters)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, database: FakeDatabase) -> None:
        self._database = database
        self.cursors: list[FakeCursor] = []
        self.closed = False

    def cursor(self) -> FakeCursor:
        cursor = FakeCursor(self._database)
        self.cursors.append(cursor)
        return cursor

    def commit(self) -> None:
        self._database.log.append(("commit",))

    def rollback(self) -> None:
        self._database.log.append(("rollback",))

    def close(self) -> None:
        self.closed = True
        self._database.log.append(("close",))
