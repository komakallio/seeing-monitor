"""The SQLite schema, the statements, and the additive migrations of the record tables.

Every `table` record type gets one table that is named after the record type. A `segment`
record type (`frame`) gets none. A table has these columns:

- `row_id INTEGER PRIMARY KEY AUTOINCREMENT`. Tables are append-only, so a sink keeps the last
  row ID that it acknowledged as its cursor. `AUTOINCREMENT` makes sure that no row ID returns
  after a delete, and a new sink starts at row zero and backfills.
- One column for each declared field, in declaration order, with the key columns first.
  `int` becomes `INTEGER`, `float` becomes `REAL`, `bool` becomes `INTEGER` with a check for 0
  and 1, `str` becomes `TEXT`, `bytes` becomes `BLOB`, and a list or a dict becomes `TEXT` that
  holds JSON, with a `json_valid` check. A column is `NOT NULL` exactly when the field cannot
  be `None`. A field with a default has the same default in the column.
- `UNIQUE (station_id, t_utc_ns, revision)`. The record type is the table, so this is the key.
  A duplicate insert fails. A correction is a new row with the next `revision`.

A sink table that applies `upsert_sql` gets the same schema. The unique key makes the upsert
idempotent, so a repeated batch changes nothing.

**Migrations.** `migration_statements` compares the declaration with the output of
`PRAGMA table_info` and returns `ALTER TABLE ... ADD COLUMN` statements for the new fields.
A new field must be optional or must have a default, because SQLite cannot add a required
column without a value. A removed field, a retyped field, or a field that changed from required
to optional raises `SchemaError`, because SQLite cannot change those without rebuilding the
table. `ensure_schema` creates the missing tables and applies the migrations.

**Rows.** `row_to_sqlite` turns the output of `Record.to_row` into parameters for the
statements, and `sqlite_to_row` turns a fetched row back into the input of `Record.from_row`.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from seeingmon.records.base import (
    KEY_FIELDS,
    RECORD_TYPES,
    FieldKind,
    FieldSpec,
    Record,
    field_specs,
    resolve_record_type,
)

ROW_ID = "row_id"

_SQL_TYPES: dict[FieldKind, str] = {
    "int": "INTEGER",
    "float": "REAL",
    "bool": "INTEGER",
    "str": "TEXT",
    "bytes": "BLOB",
    "json": "TEXT",
}


class SchemaError(Exception):
    """The declaration does not fit the database, and a migration cannot fix that."""


def table_record_types() -> list[type[Record]]:
    """The record types that live in SQLite tables, in declaration order."""
    return [cls for cls in RECORD_TYPES.values() if cls.storage == "table"]


def _require_table(record: str | type[Record]) -> type[Record]:
    cls = resolve_record_type(record)
    if cls.storage != "table":
        raise ValueError(f"{cls.record_type} is a {cls.storage} record and has no SQLite table")
    return cls


def quote(name: str) -> str:
    """Quote an identifier. The declarations allow lowercase letters, digits, and underscores."""
    return f'"{name}"'


def _literal(spec: FieldSpec, value: Any) -> str:
    if spec.kind == "bool":
        return "1" if value else "0"
    if spec.kind in ("int", "float"):
        return repr(value)
    if spec.kind == "bytes":
        return f"X'{bytes(value).hex()}'"
    text = (
        json.dumps(value, separators=(",", ":"), allow_nan=False) if spec.kind == "json" else value
    )
    return "'" + str(text).replace("'", "''") + "'"


def column_sql(spec: FieldSpec) -> str:
    """The definition of one column, as it appears in `CREATE TABLE` and `ADD COLUMN`."""
    name = quote(spec.name)
    parts = [name, _SQL_TYPES[spec.kind]]
    if not spec.nullable:
        parts.append("NOT NULL")
    if spec.has_default and spec.default is not None:
        parts.append(f"DEFAULT {_literal(spec, spec.default)}")
    if spec.kind == "bool":
        parts.append(f"CHECK ({name} IN (0, 1))")
    elif spec.kind == "json":
        parts.append(f"CHECK (json_valid({name}))")
    return " ".join(parts)


def create_table_sql(record: str | type[Record]) -> str:
    """The `CREATE TABLE IF NOT EXISTS` statement of a record type."""
    cls = _require_table(record)
    lines = [f"{quote(ROW_ID)} INTEGER PRIMARY KEY AUTOINCREMENT"]
    lines += [column_sql(spec) for spec in field_specs(cls)]
    lines.append(f"UNIQUE ({', '.join(quote(name) for name in KEY_FIELDS)})")
    body = ",\n    ".join(lines)
    return f"CREATE TABLE IF NOT EXISTS {quote(cls.record_type)} (\n    {body}\n)"


def create_index_sql(record: str | type[Record]) -> list[str]:
    """The `CREATE INDEX IF NOT EXISTS` statements of a record type.

    The unique key already indexes the station and the time. The extra index on the time serves
    queries that span stations and the retention task.
    """
    cls = _require_table(record)
    table = cls.record_type
    return [
        f"CREATE INDEX IF NOT EXISTS {quote(f'ix_{table}_t_utc_ns')} "
        f"ON {quote(table)} ({quote('t_utc_ns')})"
    ]


def schema_statements(record_types: Iterable[str | type[Record]] | None = None) -> list[str]:
    """The statements that create the tables and the indexes. The default is every table type."""
    classes = (
        table_record_types()
        if record_types is None
        else [_require_table(record) for record in record_types]
    )
    statements: list[str] = []
    for cls in classes:
        statements.append(create_table_sql(cls))
        statements.extend(create_index_sql(cls))
    return statements


def schema_sql(record_types: Iterable[str | type[Record]] | None = None) -> str:
    """The schema as a script. Run it with `sqlite3.Connection.executescript`."""
    header = '-- Generated by "seeingmon records sqlite-schema" from the record declarations.\n'
    body = "\n\n".join(f"{statement};" for statement in schema_statements(record_types))
    return f"{header}\n{body}\n"


def migration_statements(
    record: str | type[Record],
    table_info: Iterable[Sequence[Any]],
    *,
    allow_extra_columns: bool = False,
) -> list[str]:
    """The statements that bring a table up to date with the declaration.

    `table_info` is the output of `PRAGMA table_info(<table>)`: one row of
    `(cid, name, type, notnull, dflt_value, pk)` for each column, and no rows when the table does
    not exist. In that case, the result creates the table and its indexes. Otherwise the result
    holds an `ALTER TABLE ... ADD COLUMN` statement for each declared field that the table lacks.

    Raises `SchemaError` when a new field is required and has no default, when the table has a
    column that no field declares (a removed field), when a column has another type than the
    field, or when a column is `NOT NULL` and the field can be `None`. Pass
    `allow_extra_columns=True` to accept columns that the declaration lacks, for example when an
    older version of the software opens a database that a newer version migrated.
    """
    cls = _require_table(record)
    table = cls.record_type
    existing = {str(row[1]): row for row in table_info}
    if not existing:
        return [create_table_sql(cls), *create_index_sql(cls)]

    declared = {spec.name: spec for spec in field_specs(cls)}
    problems: list[str] = []
    row_id = existing.get(ROW_ID)
    if row_id is None or str(row_id[2]).upper() != "INTEGER" or not row_id[5]:
        problems.append(f"{ROW_ID} must be the INTEGER primary key")
    for name, info in existing.items():
        if name == ROW_ID:
            continue
        spec = declared.get(name)
        if spec is None:
            if not allow_extra_columns:
                problems.append(f"column {name} has no field (was the field removed?)")
            continue
        if str(info[2]).upper() != _SQL_TYPES[spec.kind]:
            problems.append(
                f"column {name} is {info[2]}, but the field needs {_SQL_TYPES[spec.kind]} "
                "(was the field retyped?)"
            )
        elif info[3] and spec.nullable:
            problems.append(f"column {name} is NOT NULL, but the field can be None")

    statements: list[str] = []
    for spec in declared.values():
        if spec.name in existing:
            continue
        if not spec.nullable and not (spec.has_default and spec.default is not None):
            problems.append(f"new field {spec.name} must be optional or have a default")
            continue
        statements.append(f"ALTER TABLE {quote(table)} ADD COLUMN {column_sql(spec)}")
    if problems:
        raise SchemaError(f"cannot migrate {table}: " + "; ".join(problems))
    return statements


def ensure_schema(
    connection: sqlite3.Connection,
    record_types: Iterable[str | type[Record]] | None = None,
    *,
    allow_extra_columns: bool = False,
) -> list[str]:
    """Create the missing tables, apply the additive migrations, and create the indexes.

    Returns the statements that changed the schema. A second call returns an empty list. The
    caller owns the transaction. Raises `SchemaError` as `migration_statements` does.
    """
    classes = (
        table_record_types()
        if record_types is None
        else [_require_table(record) for record in record_types]
    )
    changes: list[str] = []
    for cls in classes:
        info = connection.execute(f"PRAGMA table_info({quote(cls.record_type)})").fetchall()
        for statement in migration_statements(cls, info, allow_extra_columns=allow_extra_columns):
            connection.execute(statement)
            changes.append(statement)
        for statement in create_index_sql(cls):
            connection.execute(statement)
    return changes


def insert_sql(record: str | type[Record]) -> str:
    """`INSERT` with one named parameter for each field. The row ID comes from the table."""
    cls = _require_table(record)
    names = [spec.name for spec in field_specs(cls)]
    columns = ", ".join(quote(name) for name in names)
    values = ", ".join(f":{name}" for name in names)
    return f"INSERT INTO {quote(cls.record_type)} ({columns}) VALUES ({values})"


def upsert_sql(record: str | type[Record]) -> str:
    """`INSERT ... ON CONFLICT DO UPDATE` on the key `(station_id, t_utc_ns, revision)`.

    A sink uses it, so a repeated record changes nothing and a changed record replaces the row.
    The local store never upserts: its tables are append-only, and a correction is a new
    revision. The syntax also works in PostgreSQL, with that database's parameter style.
    """
    cls = _require_table(record)
    updates = ", ".join(
        f"{quote(spec.name)} = excluded.{quote(spec.name)}"
        for spec in field_specs(cls)
        if spec.name not in KEY_FIELDS
    )
    conflict = ", ".join(quote(name) for name in KEY_FIELDS)
    return f"{insert_sql(cls)} ON CONFLICT ({conflict}) DO UPDATE SET {updates}"


def select_after_sql(record: str | type[Record]) -> str:
    """`SELECT` the rows after a cursor, in row order, with `:after` and `:limit` parameters."""
    cls = _require_table(record)
    columns = ", ".join(quote(name) for name in [ROW_ID, *(s.name for s in field_specs(cls))])
    return (
        f"SELECT {columns} FROM {quote(cls.record_type)} "
        f"WHERE {quote(ROW_ID)} > :after ORDER BY {quote(ROW_ID)} LIMIT :limit"
    )


def row_to_sqlite(record: str | type[Record], row: Mapping[str, Any]) -> dict[str, Any]:
    """Turn the output of `Record.to_row` into parameters for `insert_sql` and `upsert_sql`.

    A bool becomes 0 or 1, a list or a dict becomes JSON text, and base64 text becomes `bytes`.
    Raises `KeyError` for a missing field.
    """
    params: dict[str, Any] = {}
    for spec in field_specs(_require_table(record)):
        value = row[spec.name]
        if value is None:
            params[spec.name] = None
        elif spec.kind == "bool":
            params[spec.name] = int(value)
        elif spec.kind == "json":
            params[spec.name] = json.dumps(value, separators=(",", ":"), allow_nan=False)
        elif spec.kind == "bytes":
            params[spec.name] = base64.b64decode(value, validate=True)
        else:
            params[spec.name] = value
    return params


def sqlite_to_row(
    record: str | type[Record], stored: Mapping[str, Any] | sqlite3.Row
) -> dict[str, Any]:
    """Turn a fetched row into a dict that `Record.from_row` accepts.

    The result holds the declared fields that the fetched row has. It leaves out `row_id` and
    any column that the declaration lacks. Reverses `row_to_sqlite`.
    """
    if isinstance(stored, Mapping):
        values: Mapping[str, Any] = stored
    else:  # a sqlite3.Row iterates over its values, so read the names from keys()
        names = stored.keys()
        values = {name: stored[name] for name in names}
    row: dict[str, Any] = {}
    for spec in field_specs(_require_table(record)):
        if spec.name not in values:
            continue
        value = values[spec.name]
        if value is None:
            row[spec.name] = None
        elif spec.kind == "bool":
            row[spec.name] = bool(value)
        elif spec.kind == "json":
            row[spec.name] = json.loads(value)
        elif spec.kind == "bytes":
            row[spec.name] = base64.b64encode(value).decode("ascii")
        else:
            row[spec.name] = value
    return row
