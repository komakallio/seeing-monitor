"""The `seeingmon records` commands.

- `seeingmon records list` lists the record types.
- `seeingmon records sqlite-schema` prints the SQLite schema. With `--database`, it prints the
  statements that bring an existing database up to date, and it does not run them.
- `seeingmon records api-schema` prints the OpenAPI components of the records as JSON.
- `seeingmon records reference` prints the quantity reference. With `--output`, it writes the
  file (`docs/quantities.md`), and with `--check`, it fails when the file is out of date.

The handlers import the generators on demand, so `seeingmon --help` stays fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from seeingmon.cli import CliError, Subparsers, add_command

REFERENCE_PATH = "docs/quantities.md"


def register(subparsers: Subparsers) -> None:
    parser = add_command(
        subparsers,
        "records",
        help="List the record types and print the schemas that come from their declarations.",
        handler=_no_subcommand,
    )
    commands = parser.add_subparsers(dest="records_command", metavar="<subcommand>", required=True)

    add_command(commands, "list", help="List the record types.", handler=_list)

    schema = add_command(
        commands,
        "sqlite-schema",
        help="Print the SQLite schema, or the statements that migrate a database.",
        handler=_sqlite_schema,
    )
    schema.add_argument(
        "--database",
        type=Path,
        metavar="PATH",
        help="Print the statements that bring this database up to date. The command runs none.",
    )

    api = add_command(
        commands,
        "api-schema",
        help="Print the OpenAPI components of the records as JSON.",
        handler=_api_schema,
    )
    api.add_argument(
        "--record",
        action="append",
        metavar="TYPE",
        help="Print only this record type. Repeat the option to select several.",
    )

    reference = add_command(
        commands,
        "reference",
        help="Print the quantity reference, or write it to a file.",
        handler=_reference,
    )
    target = reference.add_mutually_exclusive_group()
    target.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help=f"Write the reference to this file, for example {REFERENCE_PATH}.",
    )
    target.add_argument(
        "--check",
        type=Path,
        metavar="PATH",
        help="Fail when this file differs from the generated reference.",
    )


def _no_subcommand(args: argparse.Namespace) -> int:
    raise CliError("choose a subcommand: list, sqlite-schema, api-schema, or reference", 2)


def _list(args: argparse.Namespace) -> int:
    from seeingmon.records.base import RECORD_TYPES, field_specs

    rows = []
    for name, cls in RECORD_TYPES.items():
        retention = "forever" if cls.retention_days is None else f"{cls.retention_days} days"
        rows.append((name, cls.storage, retention, f"{len(field_specs(cls))} fields"))
    widths = [max(len(row[column]) for row in rows) for column in range(4)]
    for row in rows:
        print(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        )
    return 0


def _sqlite_schema(args: argparse.Namespace) -> int:
    from seeingmon.records.sqlite_schema import schema_sql

    if args.database is None:
        sys.stdout.write(schema_sql())
        return 0
    return _migration(args.database)


def _migration(database: Path) -> int:
    import sqlite3

    from seeingmon.records.sqlite_schema import (
        SchemaError,
        migration_statements,
        quote,
        table_record_types,
    )

    if not database.is_file():
        raise CliError(f"there is no database file at {database}")
    uri = f"{database.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise CliError(f"cannot open the database: {exc}") from None
    statements: list[str] = []
    try:
        for cls in table_record_types():
            info = connection.execute(f"PRAGMA table_info({quote(cls.record_type)})").fetchall()
            try:
                statements += migration_statements(cls, info)
            except SchemaError as exc:
                raise CliError(str(exc)) from None
    except sqlite3.Error as exc:
        raise CliError(f"cannot read the database: {exc}") from None
    finally:
        connection.close()
    if not statements:
        print("-- The database is up to date.")
    for statement in statements:
        print(f"{statement};")
    return 0


def _api_schema(args: argparse.Namespace) -> int:
    from seeingmon.records.api_schema import api_schema

    try:
        document = api_schema(args.record)
    except KeyError as exc:
        raise CliError(str(exc.args[0])) from None
    print(json.dumps(document, indent=2))
    return 0


def _reference(args: argparse.Namespace) -> int:
    from seeingmon.records.quantity_reference import COMMAND, render_reference

    text = render_reference()
    if args.check is not None:
        if not args.check.is_file() or args.check.read_text(encoding="utf-8") != text:
            raise CliError(f"{args.check} is out of date. Run `{COMMAND} --output {args.check}`.")
        print(f"{args.check} is up to date.")
    elif args.output is not None:
        args.output.write_text(text, encoding="utf-8", newline="\n")
        print(f"Wrote {args.output}.")
    else:
        sys.stdout.write(text)
    return 0
