"""The survey commands: `seeingmon catalog` and `seeingmon dark`.

`seeingmon catalog build` queries the Gaia archive and VizieR for the stars around the north
celestial pole, writes the cap catalog, and, when the astrometry.net tool
`build-astrometry-index` is installed, builds the solver index files. `seeingmon catalog info
PATH` prints the header of a catalog and checks the file.

`seeingmon dark` records a dark set with the camera covered and adds it to the dark library.
It takes the camera driver from the `[services.acquire]` configuration, so stop `acquire` first:
one process at a time can open the camera.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seeingmon.cli import CliError, Subparsers, add_command

if TYPE_CHECKING:
    from seeingmon.clock import Clock
    from seeingmon.config import Config
    from seeingmon.survey.config import SurveyConfig


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

    dark = add_command(
        subparsers,
        "dark",
        help="Record a dark set with the camera covered, and add it to the dark library.",
        handler=_dark,
    )
    dark.description = (
        "The camera has no lens cap, so cover it. The command takes bias frames, waits until a "
        "test frame is dark (skip the wait with --no-wait), records dark frames at the survey "
        "exposure, builds the master dark, and adds the set to the dark library. It reads the "
        "camera driver from [services.acquire], so stop acquire first. Defaults come from "
        "[survey.dark]."
    )
    dark.add_argument(
        "--no-wait", action="store_true", help="record at once, without waiting for the cover"
    )
    dark.add_argument("--frames", type=int, help="dark frames in the set")
    dark.add_argument("--bias-frames", type=int, help="bias frames at the shortest exposure")
    dark.add_argument("--exposure-s", type=float, help="the exposure of a dark frame, in seconds")
    dark.add_argument("--gain", type=int, help="the camera gain")
    dark.add_argument("--mode", help="the readout mode")
    dark.add_argument(
        "--wait-timeout", type=float, help="seconds to wait for the cover before giving up"
    )
    dark.add_argument("--driver", help="the camera driver (default: driver in [services.acquire])")
    dark.add_argument(
        "--library",
        type=Path,
        help="the dark library folder (default: darks/ in calibration_dir or the data directory)",
    )


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


def _make_clock(services: Any) -> Clock:
    """The clock of the dark session: the one that `[services.clock]` selects."""
    clock: Clock = services.clock.build()
    return clock


def _dark_library_dir(args: argparse.Namespace, config: Config, survey: SurveyConfig) -> Path:
    from seeingmon.config import ConfigError
    from seeingmon.store.layout import DataLayout
    from seeingmon.survey.dark import CALIBRATION_DIRNAME, DARKS_DIRNAME

    if args.library is not None:
        return Path(args.library)
    if survey.calibration_dir:
        return Path(survey.calibration_dir) / DARKS_DIRNAME
    try:
        layout = DataLayout.from_config(config)
    except ConfigError:
        raise CliError(
            "pass --library, or set calibration_dir in [survey] or data_dir in [paths] "
            "of local/config.toml"
        ) from None
    return layout.root / CALIBRATION_DIRNAME / DARKS_DIRNAME


def _dark(args: argparse.Namespace) -> int:
    from seeingmon.config import ConfigError, load_config
    from seeingmon.drivers import create_driver
    from seeingmon.drivers.base import CameraError
    from seeingmon.profile import ProfileError
    from seeingmon.services.config import ServicesConfig
    from seeingmon.survey.config import SurveyConfig
    from seeingmon.survey.dark import DarkCheckOptions, DarkError, DarkLibrary
    from seeingmon.survey.dark_session import DarkSessionOptions, run_dark_session

    try:
        config = load_config()
        survey = config.section("survey", SurveyConfig)
        services = config.section("services", ServicesConfig)
        profile = config.profile
    except (ConfigError, ProfileError) as exc:
        raise CliError(str(exc)) from None
    cfg = survey.dark
    library = DarkLibrary(_dark_library_dir(args, config, survey))
    try:
        options = DarkSessionOptions(
            mode=args.mode or cfg.mode,
            gain=cfg.gain if args.gain is None else args.gain,
            exposure_s=cfg.exposure_s if args.exposure_s is None else args.exposure_s,
            frames=cfg.frames if args.frames is None else args.frames,
            bias_frames=cfg.bias_frames if args.bias_frames is None else args.bias_frames,
            wait=not args.no_wait,
            test_exposure_s=cfg.test_exposure_s,
            poll_s=cfg.poll_s,
            stable_polls=cfg.stable_polls,
            wait_timeout_s=cfg.wait_timeout_s if args.wait_timeout is None else args.wait_timeout,
            max_temperature_spread_c=cfg.max_temperature_spread_c,
            check=DarkCheckOptions(
                rate_factor=cfg.rate_factor,
                min_rate_e_per_s=cfg.min_rate_e_per_s,
                noise_factor=cfg.noise_factor,
                max_tail_fraction=cfg.max_tail_fraction,
            ),
            hot_sigma=cfg.hot_sigma,
            hot_min_excess_dn=cfg.hot_min_excess_dn,
            prior_doubling_c=cfg.doubling_c,
            tolerance_c=cfg.temperature_tolerance_c,
            max_age_days=cfg.max_age_days,
        )
        profile.mode(options.mode)
    except (ValueError, ProfileError) as exc:
        raise CliError(str(exc)) from None
    clock = _make_clock(services)
    name = args.driver or services.acquire.driver
    try:
        driver = create_driver(
            name, profile=profile, clock=clock, options=dict(services.acquire.driver_options)
        )
    except Exception as exc:  # a driver can fail in its own ways, such as a missing library
        raise CliError(f"cannot create the driver {name!r}: {type(exc).__name__}: {exc}") from None

    def say(message: str) -> None:
        print(message, flush=True)

    try:
        run_dark_session(driver, library, profile, clock, options, say=say)
    except DarkError as exc:
        raise CliError(str(exc)) from None
    except CameraError as exc:
        raise CliError(
            f"the camera failed: {exc}. Stop acquire first: only one process can open the camera."
        ) from None
    return 0
