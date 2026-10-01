"""The `seeingmon catalog` commands.

`seeingmon catalog build` queries the Gaia archive and VizieR for the stars around the north
celestial pole, writes the cap catalog, and, when the astrometry.net tool
`build-astrometry-index` is installed, builds the solver index files. `seeingmon catalog info
PATH` prints the header of a catalog and checks the file.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from seeingmon.cli import CliError, Subparsers, add_command


def register(subparsers: Subparsers) -> None:
    parser = add_command(
        subparsers,
        "catalog",
        help="Build and inspect the cap catalog.",
        handler=_missing_subcommand,
    )
    commands = parser.add_subparsers(dest="catalog_command", metavar="<subcommand>", required=True)

    build = commands.add_parser(
        "build",
        help="Query Gaia and Tycho-2 and write the cap catalog and the solver index.",
        description=(
            "Query the Gaia archive (an asynchronous job, which can queue for many minutes) and "
            "VizieR for the stars within a radius of the north celestial pole, write the catalog, "
            "and build the astrometry.net index files when build-astrometry-index is installed. "
            "Run it on a machine with a network connection."
        ),
    )
    build.add_argument(
        "--output",
        type=Path,
        help="the catalog file to write (default: catalog_path in the [survey] configuration)",
    )
    build.add_argument("--radius-deg", type=float, default=15.0, help="cap radius (default 15)")
    build.add_argument(
        "--gaia-mag-limit", type=float, default=13.0, help="faintest Gaia G magnitude (default 13)"
    )
    build.add_argument(
        "--tycho-mag-limit", type=float, default=12.0, help="faintest Tycho-2 VT (default 12)"
    )
    build.add_argument("--gaia-url", help="the Gaia TAP service URL (default: the ESA archive)")
    build.add_argument("--vizier-url", help="the VizieR TAP service URL (default: CDS)")
    build.add_argument(
        "--gaia-csv", type=Path, help="read Gaia rows from this CSV, not the archive"
    )
    build.add_argument("--tycho-csv", type=Path, help="read Tycho-2 rows from this CSV")
    build.add_argument(
        "--poll-interval", type=float, default=10.0, help="seconds between job checks (default 10)"
    )
    build.add_argument(
        "--max-wait", type=float, default=7200.0, help="seconds to wait for the job (default 7200)"
    )
    build.add_argument(
        "--index-dir",
        type=Path,
        help="where to write the solver index (default: an index folder next to the catalog)",
    )
    build.add_argument("--no-index", action="store_true", help="do not build the solver index")
    build.add_argument(
        "--index-presets",
        default="8,9,10,11,12",
        help="build-astrometry-index scale presets, separated by commas (default 8,9,10,11,12)",
    )
    build.add_argument(
        "--build-index-command",
        default="build-astrometry-index",
        help="the index tool, with any leading arguments (default build-astrometry-index)",
    )
    build.set_defaults(handler=_build)

    info = commands.add_parser(
        "info",
        help="Print the header of a catalog file and check it.",
        description="Read a catalog file, check its CRC-32, and print its header.",
    )
    info.add_argument("path", type=Path, help="the catalog file")
    info.set_defaults(handler=_info)


def _missing_subcommand(args: argparse.Namespace) -> int:
    # `required=True` makes argparse reject a missing subcommand before this runs.
    raise CliError("choose a subcommand: build or info", exit_code=2)


def _configured_catalog_path() -> Path:
    from seeingmon.config import ConfigError, load_config
    from seeingmon.survey.config import SurveyConfig

    try:
        section = load_config().section("survey", SurveyConfig)
    except ConfigError as exc:
        raise CliError(str(exc)) from None
    if not section.catalog_path:
        raise CliError(
            "pass --output, or set catalog_path in the [survey] section of local/config.toml"
        )
    return Path(section.catalog_path)


def _parse_presets(text: str) -> list[int]:
    try:
        presets = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise CliError("--index-presets takes whole numbers separated by commas") from None
    if not presets or any(not -5 <= preset <= 19 for preset in presets):
        raise CliError("--index-presets takes numbers from -5 to 19")
    return presets


def _build(args: argparse.Namespace) -> int:
    from seeingmon.clock import SystemClock
    from seeingmon.survey.catalog import write_catalog
    from seeingmon.survey.catalog_build import (
        GAIA_TAP_URL,
        VIZIER_TAP_URL,
        BuildOptions,
        CatalogBuildError,
        build_catalog,
        build_solver_index,
        find_index_tool,
    )

    output: Path = args.output or _configured_catalog_path()
    presets = _parse_presets(args.index_presets)
    try:
        options = BuildOptions(
            radius_deg=args.radius_deg,
            gaia_mag_limit=args.gaia_mag_limit,
            tycho_mag_limit=args.tycho_mag_limit,
            gaia_url=args.gaia_url or GAIA_TAP_URL,
            vizier_url=args.vizier_url or VIZIER_TAP_URL,
            poll_interval_s=args.poll_interval,
            max_wait_s=args.max_wait,
        )
    except ValueError as exc:
        raise CliError(str(exc)) from None

    def say(message: str) -> None:
        print(message, flush=True)

    try:
        gaia_csv = args.gaia_csv.read_text(encoding="utf-8") if args.gaia_csv else None
        tycho_csv = args.tycho_csv.read_text(encoding="utf-8") if args.tycho_csv else None
    except OSError as exc:
        raise CliError(f"cannot read a CSV file: {exc.strerror or exc}") from None
    try:
        catalog = build_catalog(
            options, SystemClock(), gaia_csv=gaia_csv, tycho_csv=tycho_csv, progress=say
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_catalog(output, catalog)
    except (CatalogBuildError, OSError) as exc:
        raise CliError(str(exc)) from None
    size_mb = output.stat().st_size / 1e6
    say(f"wrote {len(catalog)} stars ({size_mb:.1f} MB), catalog id {catalog.content_id}")

    if args.no_index:
        return 0
    tool = find_index_tool(shlex.split(args.build_index_command))
    if tool is None:
        say("solver index: skipped, because the index tool is not installed")
        return 0
    index_dir: Path = args.index_dir or output.parent / "index"
    try:
        written = build_solver_index(
            catalog, index_dir, command=tool, presets=presets, progress=say
        )
    except CatalogBuildError as exc:
        raise CliError(str(exc)) from None
    say(f"solver index: {len(written)} files")
    return 0


def _info(args: argparse.Namespace) -> int:
    from seeingmon.survey.catalog import CatalogError, describe_flags, load_catalog

    try:
        catalog = load_catalog(args.path)
    except CatalogError as exc:
        raise CliError(str(exc)) from None
    info = catalog.info()
    flag_counts = {
        name: int(sum(1 for flags in catalog.flags if name in describe_flags(int(flags))))
        for name in ("tycho_only", "tycho_v", "no_proper_motion", "no_parallax", "no_color")
    }
    rows = [
        ("catalog id", catalog.content_id),
        ("format version", str(info.version)),
        ("stars", str(info.n_stars)),
        ("epoch", f"J{info.epoch_jyear:g}"),
        (
            "cap",
            f"{info.cap_radius_deg:g} degrees around ({info.cap_ra_deg:g}, {info.cap_dec_deg:g})",
        ),
        (
            "magnitude limits",
            f"Gaia G < {info.gaia_mag_limit:g}, Tycho-2 VT < {info.tycho_mag_limit:g}",
        ),
        ("brightest G", f"{float(catalog.g_mag.min()):.2f}" if len(catalog) else "none"),
        ("sources", info.notes or "not stated"),
        ("flags", ", ".join(f"{name} {count}" for name, count in flag_counts.items())),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")
    return 0
