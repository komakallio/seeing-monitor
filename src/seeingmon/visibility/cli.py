"""The visibility command: `seeingmon visibility stats`.

`stats` reads the nightly `visibility_summary` records that `core` writes when the split hour
passes (`seeingmon.services.core.visibility`). It prints the newest nights, and then the Sun's
elevation at the first and the last detection of Polaris by month and by transparency bin, with
the censored nights counted apart. It opens the store read-only, so it is safe while `core` runs.
The statistics live in `seeingmon.visibility.stats`. This module defines the arguments, reads the
store, and prints.

The handler imports the heavy modules when it runs, so `seeingmon --help` stays fast.
"""

from __future__ import annotations

import argparse
import itertools
import math
import textwrap
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from seeingmon.cli import CliError, Subparsers, add_command

if TYPE_CHECKING:
    from seeingmon.records.visibility import VisibilitySummaryRecord
    from seeingmon.visibility.stats import Group

DEFAULT_NIGHTS = 10  # the newest nights that the table lists
DEFAULT_BINS = "0.6,0.8,0.9"  # stats.DEFAULT_TRANSPARENCY_EDGES, which the parser does not import
_BATCH = 1000  # rows for each read of the store
CENSORED_MARK = "*"
TEXT_WIDTH = 100  # the width of a paragraph of the output


def register(subparsers: Subparsers) -> None:
    parser = add_command(
        subparsers,
        "visibility",
        help="Show when Polaris was visible, from the nightly visibility summaries.",
        handler=_missing_subcommand,
    )
    commands = parser.add_subparsers(
        dest="visibility_command", metavar="<subcommand>", required=True
    )
    _register_stats(commands)


def _missing_subcommand(args: argparse.Namespace) -> int:
    # `required=True` makes argparse reject a missing subcommand before this runs.
    raise CliError("choose a subcommand: stats", exit_code=2)


def _register_stats(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    command = commands.add_parser(
        "stats",
        help="Print the Sun's elevation at the first and the last detection of Polaris.",
        description=(
            "Read the nightly visibility summaries from the store, and print the newest nights "
            "and the Sun's elevation at the first and the last detection of Polaris, by month and "
            "by the median transparency of the clear verdict. A censored detection is a bound: "
            "Polaris was visible before the first detection or after the last one, or the station "
            "did not watch the whole time. The tables count the censored nights apart and keep "
            "their values as bounds. The command opens the store read-only, so it is safe while "
            "core runs."
        ),
    )
    command.add_argument(
        "--data-dir",
        type=Path,
        metavar="PATH",
        help="the data directory with the store in db/results.sqlite "
        "(default: data_dir in [paths])",
    )
    command.add_argument(
        "--from",
        dest="first_night",
        type=_night,
        metavar="YYYY-MM-DD",
        help="the first night to include, by its label (default: the oldest)",
    )
    command.add_argument(
        "--to",
        dest="last_night",
        type=_night,
        metavar="YYYY-MM-DD",
        help="the last night to include, by its label (default: the newest)",
    )
    command.add_argument(
        "--nights",
        type=int,
        default=DEFAULT_NIGHTS,
        metavar="N",
        help=f"list the newest N nights, 0 for none (default {DEFAULT_NIGHTS})",
    )
    command.add_argument(
        "--transparency-bins",
        default=DEFAULT_BINS,
        metavar="EDGES",
        help=f"the rising edges of the transparency bins, separated by commas "
        f"(default {DEFAULT_BINS})",
    )
    command.add_argument(
        "--local-config",
        type=Path,
        help="read this file instead of local/config.toml (an absent file is ignored)",
    )
    command.set_defaults(handler=_stats)


def _night(text: str) -> str:
    """The label of a night in `YYYY-MM-DD` form, which the filters compare as text.

    Python 3.11 and later also read `20261002` and `2026-W40-5`, which become `2026-10-02`.
    """
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a night label (YYYY-MM-DD)") from None


def _edges(text: str) -> tuple[float, ...]:
    try:
        edges = tuple(float(part) for part in text.split(",") if part.strip())
    except ValueError:
        raise CliError(
            "--transparency-bins takes numbers separated by commas", exit_code=2
        ) from None
    if not all(math.isfinite(edge) for edge in edges) or any(
        high <= low for low, high in itertools.pairwise(edges)
    ):
        raise CliError("--transparency-bins takes rising numbers", exit_code=2)
    return edges


# --- Reading the store -------------------------------------------------------------------------


def _data_dir(args: argparse.Namespace) -> Path:
    from seeingmon.config import ConfigError, load_config
    from seeingmon.store.layout import DataLayout

    if args.data_dir is not None:
        return Path(args.data_dir)
    try:
        return DataLayout.from_config(load_config(local_file=args.local_config)).root
    except ConfigError as exc:
        raise CliError(
            f"pass --data-dir, or set data_dir in [paths] of local/config.toml ({exc})"
        ) from None


def read_summaries(data_dir: Path) -> list[VisibilitySummaryRecord]:
    """Every visibility summary of the store in a data directory, in time order."""
    import sqlite3

    from seeingmon.records.visibility import VisibilitySummaryRecord
    from seeingmon.store.db import StoreError, StoreReader, record_from_row
    from seeingmon.store.layout import DataLayout

    try:
        reader = StoreReader.open(DataLayout(data_dir).db_path)
    except FileNotFoundError:
        raise CliError("there is no store database in the data directory") from None
    except (StoreError, sqlite3.Error) as exc:
        raise CliError(f"cannot read the store: {exc}") from None
    found: list[VisibilitySummaryRecord] = []
    try:
        after = 0
        while True:
            rows = reader.after("visibility_summary", after, _BATCH)
            for row in rows:
                record = record_from_row("visibility_summary", row)
                if isinstance(record, VisibilitySummaryRecord):
                    found.append(record)
            if len(rows) < _BATCH:
                break
            after = rows[-1].row_id
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return []  # a store from before the visibility summary
        raise CliError(f"cannot read the store: {exc}") from None
    except StoreError as exc:
        raise CliError(f"cannot read the store: {exc}") from None
    finally:
        reader.close()
    return sorted(found, key=lambda record: (record.t_utc_ns, record.revision))


def newest_revisions(records: Sequence[VisibilitySummaryRecord]) -> list[VisibilitySummaryRecord]:
    """One record for each night and station: the newest revision, in time order."""
    newest: dict[tuple[str, int], VisibilitySummaryRecord] = {}
    for record in records:
        key = (record.station_id, record.t_utc_ns)
        if key not in newest or record.revision > newest[key].revision:
            newest[key] = record
    return sorted(newest.values(), key=lambda record: (record.t_utc_ns, record.station_id))


# --- Printing ----------------------------------------------------------------------------------


def _sun(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1f}"


def _clock(t_utc_ns: int | None, censored: bool) -> str:
    from seeingmon.clock import utc_ns_to_datetime

    if t_utc_ns is None:
        return (CENSORED_MARK if censored else "-").ljust(6)
    text = utc_ns_to_datetime(t_utc_ns).strftime("%H:%M")
    return text + (CENSORED_MARK if censored else " ")


def _number(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _say(text: str) -> None:
    """Print a paragraph, wrapped to the width of a terminal."""
    print(textwrap.fill(text, width=TEXT_WIDTH))


def print_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """Print rows under a header, with each column as wide as its widest cell."""
    widths = [len(name) for name in header]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]
    for row in (header, *rows):
        print(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        )


def _night_rows(records: Sequence[VisibilitySummaryRecord]) -> list[list[str]]:
    return [
        [
            record.night,
            _clock(record.first_visible_utc_ns, record.first_censored),
            _sun(record.first_visible_sun_deg),
            _clock(record.last_visible_utc_ns, record.last_censored),
            _sun(record.last_visible_sun_deg),
            _number(record.visible_hours, 2),
            _number(record.seeing_hours, 2),
            _sun(record.dark_sun_deg),
            _number(record.clear_share, 2),
            _number(record.transparency_median, 2),
            ",".join(record.flags) or "-",
        ]
        for record in records
    ]


def _range(values: Sequence[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return _sun(values[0])
    return f"{_sun(values[0])} to {_sun(values[-1])}"


def _group_rows(groups: Sequence[Group]) -> list[list[str]]:
    rows: list[list[str]] = []
    for group in groups:
        row = [group.key, str(group.nights), str(group.unseen)]
        for side in (group.first, group.last):
            row += [
                str(len(side.measured)),
                _sun(side.median),
                _range(side.measured),
                str(side.n_censored),
                _range(side.censored),
            ]
        rows.append(row)
    return rows


def without_sun_note(groups: Sequence[Group]) -> str | None:
    """A note on the detections that are not censored and have no Sun elevation, or `None`.

    The n columns leave them out, so the note accounts for every night of a group.
    """
    parts = [
        f"{group.key}: {group.first.without_sun} first and {group.last.without_sun} last"
        for group in groups
        if group.first.without_sun or group.last.without_sun
    ]
    if not parts:
        return None
    return (
        "Detections without a Sun elevation (no site, or a clock that was not synchronized) are "
        f"not in the n columns: {'; '.join(parts)}."
    )


def _print_groups(title: str, first_column: str, groups: Sequence[Group]) -> None:
    print(title)
    print_table((first_column, *GROUP_COLUMNS), _group_rows(groups))
    note = without_sun_note(groups)
    if note is not None:
        _say(note)


GROUP_COLUMNS = (
    "nights",
    "unseen",
    "first n",
    "median",
    "range",
    "censored",
    "bounds",
    "last n",
    "median",
    "range",
    "censored",
    "bounds",
)


def _stats(args: argparse.Namespace) -> int:
    from seeingmon.visibility import stats

    if args.nights < 0:
        raise CliError("--nights takes 0 or more", exit_code=2)
    edges = _edges(args.transparency_bins)
    records = newest_revisions(read_summaries(_data_dir(args)))
    if args.first_night is not None:
        records = [r for r in records if r.night >= args.first_night]
    if args.last_night is not None:
        records = [r for r in records if r.night <= args.last_night]
    if not records:
        print("No visibility summaries yet: core writes one when the split hour of a night passes.")
        return 0

    nights = len(records)
    _say(
        f"Visibility of Polaris: {nights} night{'s' if nights != 1 else ''} from "
        f"{records[0].night} to {records[-1].night}. The Sun's elevation is in degrees, and the "
        "times are UTC."
    )
    if args.nights:
        shown = records[-args.nights :]
        print()
        print("The newest night:" if len(shown) == 1 else f"The newest {len(shown)} nights:")
        print_table(
            (
                "night",
                "first",
                "sun",
                "last",
                "sun",
                "visible h",
                "seeing h",
                "dark sun",
                "clear",
                "transparency",
                "flags",
            ),
            _night_rows(shown),
        )
    print()
    _say(
        f"{CENSORED_MARK} A censored detection is a bound: Polaris was visible before the first "
        "detection or after the last one, or the station did not watch the whole time. The Sun "
        "stood at least as high as the bound when Polaris appeared or vanished. The tables count "
        "the censored nights apart, and the bounds column gives the range of their bounds."
    )
    print()
    _print_groups("By month:", "month", stats.by_month(records))
    print()
    _print_groups(
        "By the median transparency of the clear verdict:",
        "transparency",
        stats.by_transparency(records, edges),
    )
    return 0
