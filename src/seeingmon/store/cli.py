"""The `seeingmon store` commands.

`seeingmon store info DB` prints what a person needs to judge the health of a store database: the
journal mode, the size, the row count and the last row ID of every record type, the time of the
latest record, and the cursor of every sink with the number of rows that it still has to take. It
opens the file read-only, so it is safe to run while `core` writes. It prints no station ID, no
profile ID, no event text, no path, and no sink setting.

The handlers import the store on demand, so `seeingmon --help` stays fast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from seeingmon.cli import CliError, Subparsers, add_command


def register(subparsers: Subparsers) -> None:
    parser = add_command(
        subparsers,
        "store",
        help="Inspect the SQLite store of the results.",
        handler=_missing_subcommand,
    )
    commands = parser.add_subparsers(dest="store_command", metavar="<subcommand>", required=True)
    info = commands.add_parser(
        "info",
        help="Print the row counts, the last row IDs, and the sink cursors of a database.",
        description=(
            "Print the journal mode, the size, the row count and last row ID of each record "
            "type, and the cursor of each sink. The command opens the file read-only and prints "
            "no station, profile, event text, or path."
        ),
    )
    info.add_argument("database", type=Path, metavar="DB", help="the store database file")
    info.set_defaults(handler=_info)


def _missing_subcommand(args: argparse.Namespace) -> int:
    # `required=True` makes argparse reject a missing subcommand before this runs.
    raise CliError("choose a subcommand: info", exit_code=2)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[column]) for row in [headers, *rows]) for column in range(len(headers))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(headers, widths, strict=True))]
    lines += [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]
    return [line.rstrip() for line in lines]


def _size(size_bytes: int) -> str:
    if size_bytes >= 10**6:
        return f"{size_bytes / 10**6:.1f} MB"
    return f"{size_bytes / 10**3:.1f} kB"


def _info(args: argparse.Namespace) -> int:
    from seeingmon.clock import utc_ns_to_iso
    from seeingmon.records.sqlite_schema import table_record_types
    from seeingmon.store.db import STORE_SCHEMA_VERSION, StoreError, StoreReader

    try:
        reader = StoreReader.open(args.database)
    except FileNotFoundError:
        raise CliError("there is no database file at that path") from None
    except StoreError as exc:
        raise CliError(str(exc)) from None
    try:
        facts = reader.diagnostics()
        record_rows: list[list[str]] = []
        cursor_rows: list[list[str]]
        with reader.snapshot() as snapshot:
            for cls in table_record_types():
                latest = snapshot.latest(cls.record_type)
                record_rows.append(
                    [
                        cls.record_type,
                        str(snapshot.count(cls.record_type)),
                        str(snapshot.last_row_id(cls.record_type)),
                        "none"
                        if latest is None
                        else utc_ns_to_iso(int(latest.values["t_utc_ns"]), digits=0),
                    ]
                )
            cursor_rows = []
            for item in snapshot.cursors():
                try:
                    behind = str(snapshot.count_after(item.record_type, item.last_row_id))
                except (StoreError, ValueError):
                    behind = "unknown type"  # a cursor of a record type that no longer exists
                updated = (
                    "never"
                    if item.updated_utc_ns is None
                    else utc_ns_to_iso(item.updated_utc_ns, digits=0)
                )
                cursor_rows.append(
                    [item.sink, item.record_type, str(item.last_row_id), behind, updated]
                )
    except StoreError as exc:
        raise CliError(str(exc)) from None
    finally:
        reader.close()

    print(f"journal mode   {facts['journal_mode']}")
    print(f"store version  {facts['user_version']} (this software writes {STORE_SCHEMA_VERSION})")
    print(
        f"size           {_size(facts['size_bytes'])}, of which {_size(facts['free_bytes'])} free"
    )
    print(f"SQLite         {facts['sqlite_version']}")
    print()
    for line in _table(["record type", "rows", "last row ID", "latest record"], record_rows):
        print(line)
    print()
    if cursor_rows:
        for line in _table(
            ["sink", "record type", "cursor", "rows behind", "updated"], cursor_rows
        ):
            print(line)
    else:
        print("No sink has acknowledged a row yet.")
    return 0
