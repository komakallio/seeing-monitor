"""The PostgreSQL and TimescaleDB sink: one table for each record type, written with upserts.

`seeingmon.records.sink_mapping` decides the layout, and this module applies it. A record type is
a table with the name of the type. Every declared field is a column, and `t_utc_ns` (a `BIGINT` of
nanoseconds) is the time column. The table has a unique index on the key
`(station_id, t_utc_ns, revision)`. With the TimescaleDB extension, the sink also turns each table
into a hypertable on `t_utc_ns`, with a chunk interval of `chunk_days` days. Without it, the tables
are plain PostgreSQL tables.

**Schema.** The sink creates what it needs the first time it sends a record type over a new
connection: `CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS` for every column (so a table
that an older version made gains the fields of a newer one), the unique index, and the hypertable.
Every statement is idempotent. The user needs the right to create tables, or an administrator
makes the tables first with the same statements (`schema_statements`).

**Writes.** A batch is one transaction: `INSERT ... ON CONFLICT (key) DO UPDATE` for each row, then
a commit. A repeated batch changes nothing, and a corrected row replaces the old one. A `bytes`
field is `BYTEA`, and a list or a dict is `JSONB`. The statements use `%s` placeholders, which
`psycopg` and most PostgreSQL drivers accept.

**Connection.** The sink takes a function that returns a DB-API connection, and it keeps the
connection between batches. After a connection error it drops the connection and connects again
at the next batch. `make_psycopg_connect` builds that function and imports `psycopg` only when
you call it, so the rest of the package needs no PostgreSQL driver (install the `timescale` extra).

**Errors.** DB-API names the classes of error, and the sink reads the class names, so it needs no
driver import. A `ProgrammingError`, an `IntegrityError`, a `DataError`, or a `NotSupportedError`
means that retrying cannot help: the sink raises `SinkError(retryable=False)`. Every other error is
an operational problem (a lost connection, a restart, a lock timeout, a full disk), and the sink
raises `SinkError(retryable=True)`. A message never holds a query, a row value, a password, an
address, or a user name: the driver's message and the server's message can hold all of them, and
the message reaches the `event` table and every other sink. The sink reports the class of the
error and its SQLSTATE (for example `28P01` for a wrong password and `42501` for a missing
privilege), and it writes the primary message of the server to the debug log only.
"""

from __future__ import annotations

import base64
import importlib
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any

from seeingmon.config import ConfigError
from seeingmon.records.base import get_record_type
from seeingmon.records.sink_mapping import TimescaleMapping, sink_mapping
from seeingmon.sinks.base import SinkError, StoredRow
from seeingmon.sinks.config import TimescaleSinkConfig

logger = logging.getLogger(__name__)

NS_PER_DAY = 86_400 * 1_000_000_000
EXTENSION_QUERY = "SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'"

# The DB-API errors that retrying cannot fix. Any other error is operational, so the sink retries.
PERMANENT_ERRORS = frozenset(
    {"ProgrammingError", "IntegrityError", "DataError", "NotSupportedError"}
)
# The errors that leave the connection unusable, so the sink drops it and connects again.
CONNECTION_ERRORS = frozenset({"OperationalError", "InterfaceError", "InternalError"})
_MESSAGE_CHARS = 200

Connect = Callable[[], Any]


def _quote(name: str) -> str:
    return f'"{name}"'


# --- SQL ----------------------------------------------------------------------------------------


def hypertable_sql(mapping: TimescaleMapping, chunk_days: int = 7) -> str:
    """The call that turns a table into a hypertable on the time column.

    The time column holds nanoseconds, so the chunk interval is `chunk_days` days in nanoseconds.
    The call is idempotent (`if_not_exists`) and moves any existing rows (`migrate_data`).
    """
    return (
        f"SELECT create_hypertable('{_quote(mapping.table)}', '{mapping.time_column}', "
        f"chunk_time_interval => {chunk_days * NS_PER_DAY}, if_not_exists => TRUE, "
        "migrate_data => TRUE)"
    )


def schema_statements(record_type: str, *, hypertable: bool, chunk_days: int = 7) -> list[str]:
    """The statements that create a record type's table, its columns, its index, and its hypertable.

    Run them in order, each one on its own. Every statement is idempotent, so an administrator can
    run them by hand and the sink can run them again.
    """
    mapping = sink_mapping(record_type).timescale
    statements = [mapping.create_table_sql(), *mapping.add_column_sql(), mapping.unique_index_sql()]
    if hypertable:
        statements.append(hypertable_sql(mapping, chunk_days))
    return statements


def upsert_sql(record_type: str) -> str:
    """`INSERT ... ON CONFLICT (key) DO UPDATE` with one `%s` placeholder for each column.

    A JSON column takes its value as text and casts it (`%s::jsonb`), so a driver needs no JSON
    adapter.
    """
    mapping = sink_mapping(record_type).timescale
    names = [column.name for column in mapping.columns]
    placeholders = ["%s::jsonb" if column.kind == "json" else "%s" for column in mapping.columns]
    updates = ", ".join(
        f"{_quote(name)} = EXCLUDED.{_quote(name)}"
        for name in names
        if name not in mapping.key_columns
    )
    return (
        f"INSERT INTO {_quote(mapping.table)} ({', '.join(_quote(n) for n in names)}) "
        f"VALUES ({', '.join(placeholders)}) "
        f"ON CONFLICT ({', '.join(_quote(n) for n in mapping.key_columns)}) DO UPDATE SET {updates}"
    )


def row_parameters(record_type: str, values: Mapping[str, Any]) -> tuple[Any, ...]:
    """The parameters of one row, in the order of the columns of `upsert_sql`.

    A missing value (`None`) is NULL. `bytes` values arrive as base64 text and become `bytes`, a
    list or a dict becomes JSON text, and a NUL character leaves the text, because PostgreSQL
    cannot store it in `TEXT` or `JSONB`.
    """
    mapping = sink_mapping(record_type).timescale
    parameters: list[Any] = []
    for column in mapping.columns:
        value = values.get(column.name)
        if value is None:
            parameters.append(None)
        elif column.kind == "json":
            text = json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            parameters.append(text.replace("\\u0000", "").replace("\x00", ""))
        elif column.kind == "bytes":
            parameters.append(base64.b64decode(value, validate=True))
        elif column.kind == "bool":
            parameters.append(bool(value))
        elif column.kind == "int":
            parameters.append(int(value))
        elif column.kind == "float":
            number = float(value)
            parameters.append(number if math.isfinite(number) else None)
        else:
            parameters.append(str(value).replace("\x00", ""))
    return tuple(parameters)


# --- errors -------------------------------------------------------------------------------------


def is_permanent(error: BaseException) -> bool:
    """Whether a DB-API error is one that retrying cannot fix (by the name of its classes)."""
    return any(cls.__name__ in PERMANENT_ERRORS for cls in type(error).__mro__)


def _breaks_connection(error: BaseException) -> bool:
    return isinstance(error, OSError) or any(
        cls.__name__ in CONNECTION_ERRORS for cls in type(error).__mro__
    )


def _describe(error: BaseException) -> str:
    """Name the class of the error and its SQLSTATE. Never the query, a value, or a name."""
    parts = [type(error).__name__]
    state = getattr(error, "sqlstate", None)
    if isinstance(state, str) and state:
        parts.append(f"SQLSTATE {state}")
    return ", ".join(parts)


def _server_message(error: BaseException) -> str:
    """The primary message of the server, for the debug log only. It can name a user or a host."""
    primary = getattr(getattr(error, "diag", None), "message_primary", None)
    return " ".join(primary.split())[:_MESSAGE_CHARS] if isinstance(primary, str) else ""


# --- the sink -----------------------------------------------------------------------------------


class TimescaleSink:
    """Writes rows to PostgreSQL or TimescaleDB. See the module documentation.

    `connect` returns a DB-API connection with `autocommit` off. `hypertables` is `True` to
    require the TimescaleDB extension, `False` to make plain tables, and `None` to detect the
    extension on each new connection. `record_types` limits the record types, and `None` takes
    every table record type.
    """

    def __init__(
        self,
        name: str,
        connect: Connect,
        *,
        record_types: Sequence[str] | None = None,
        max_batch_rows: int = 1000,
        hypertables: bool | None = None,
        chunk_days: int = 7,
    ) -> None:
        self._name = name
        self._connect = connect
        self._record_types = None if record_types is None else frozenset(record_types)
        self._max_batch_rows = max_batch_rows
        self._hypertables = hypertables
        self._chunk_days = chunk_days
        self._connection: Any = None
        self._ready: set[str] = set()
        self._use_hypertables = False
        self._upserts: dict[str, str] = {}

    @classmethod
    def from_config(cls, name: str, config: TimescaleSinkConfig, connect: Connect) -> TimescaleSink:
        """Build the sink from its configuration and a connection function."""
        return cls(
            name,
            connect,
            record_types=config.record_types,
            max_batch_rows=config.max_batch_rows,
            hypertables=config.hypertables,
            chunk_days=config.chunk_days,
        )

    def __repr__(self) -> str:
        return f"TimescaleSink({self._name!r})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_batch_rows(self) -> int:
        return self._max_batch_rows

    def accepts(self, record_type: str) -> bool:
        if self._record_types is not None:
            return record_type in self._record_types
        try:
            return get_record_type(record_type).storage == "table"
        except KeyError:
            return False

    def close(self) -> None:
        """Close the connection. The next batch connects again."""
        self._drop_connection()

    # --- sending ---

    def send(self, record_type: str, rows: Sequence[StoredRow]) -> None:
        """Write a batch in one transaction. Returns when it commits, or raises `SinkError`."""
        try:
            statement = self._upserts.get(record_type)
            if statement is None:
                statement = self._upserts[record_type] = upsert_sql(record_type)
            parameters = [row_parameters(record_type, row.values) for row in rows]
        except (KeyError, ValueError, TypeError) as exc:
            # The row does not fit its declaration, and sending it again changes nothing.
            raise SinkError(
                f"sink {self._name} cannot build a {record_type} row ({type(exc).__name__})",
                retryable=False,
            ) from None
        try:
            connection = self._ready_connection(record_type)
            cursor = connection.cursor()
            try:
                cursor.executemany(statement, parameters)
            finally:
                cursor.close()
            connection.commit()
        except Exception as exc:
            self._fail(exc, record_type)
            raise self._to_sink_error(exc, record_type) from None

    def _ready_connection(self, record_type: str) -> Any:
        """Connect if needed, and make the table of the record type exist on this connection."""
        if self._connection is None:
            connection = self._connect()
            try:
                use_hypertables = self._detect_hypertables(connection)
            except BaseException:
                try:
                    connection.close()
                except Exception:
                    logger.debug("sink %s: closing the connection failed", self._name)
                raise
            self._connection = connection
            self._use_hypertables = use_hypertables
            self._ready = set()
        if record_type not in self._ready:
            cursor = self._connection.cursor()
            try:
                for statement in schema_statements(
                    record_type, hypertable=self._use_hypertables, chunk_days=self._chunk_days
                ):
                    cursor.execute(statement)
            finally:
                cursor.close()
            self._connection.commit()
            self._ready.add(record_type)
        return self._connection

    def _detect_hypertables(self, connection: Any) -> bool:
        if self._hypertables is not None:
            return self._hypertables
        cursor = connection.cursor()
        try:
            cursor.execute(EXTENSION_QUERY)
            found = cursor.fetchone() is not None
        finally:
            cursor.close()
        connection.commit()
        return found

    # --- failures ---

    def _drop_connection(self) -> None:
        connection, self._connection = self._connection, None
        self._ready = set()
        if connection is not None:
            try:
                connection.close()
            except Exception:
                logger.debug("sink %s: closing the connection failed", self._name, exc_info=True)

    def _fail(self, error: BaseException, record_type: str) -> None:
        logger.warning(
            "sink %s: a batch of %s rows failed: %s", self._name, record_type, _describe(error)
        )
        message = _server_message(error)
        if message:
            logger.debug("sink %s: the server said: %s", self._name, message)
        connection = self._connection
        if connection is not None and not _breaks_connection(error):
            try:
                connection.rollback()
            except Exception:
                self._drop_connection()
                return
        if _breaks_connection(error):
            self._drop_connection()

    def _to_sink_error(self, error: BaseException, record_type: str) -> SinkError:
        permanent = is_permanent(error)
        action = "refused" if permanent else "could not take"
        return SinkError(
            f"the database of sink {self._name} {action} a batch of {record_type} rows "
            f"({_describe(error)})",
            retryable=not permanent,
        )


# --- the driver ---------------------------------------------------------------------------------


def make_psycopg_connect(
    config: TimescaleSinkConfig,
    password: str | None,
    *,
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> Connect:
    """Build the connection function of a sink from its configuration, with `psycopg` (version 3).

    The function imports `psycopg` now, so a missing driver fails when the service starts and not
    at the first batch. Raises `ConfigError` when `psycopg` is not installed. The returned function
    connects with the host, port, database, user, SSL mode, and timeout of the configuration, and
    with `password`, and it never logs or prints them.
    """
    try:
        psycopg = import_module("psycopg")
    except ModuleNotFoundError:
        raise ConfigError(
            "the timescale sink needs the psycopg package: install the 'timescale' extra "
            "(pip install 'seeingmon[timescale]')"
        ) from None
    timeout = max(1, math.ceil(config.connect_timeout_s))

    def connect() -> Any:
        return psycopg.connect(
            host=config.host,
            port=config.port,
            dbname=config.database,
            user=config.user,
            password=password,
            sslmode=config.sslmode,
            connect_timeout=timeout,
        )

    return connect
