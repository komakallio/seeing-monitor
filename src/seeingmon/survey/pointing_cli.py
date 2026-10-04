"""The pointing commands: `seeingmon pointing set-reference` and `seeingmon pointing show`.

`set-reference` saves the newest good pointing solution of the store as the reference solution that
the Pointing card measures the offset from. Run it after you align the camera. It opens the store
read-only, so it is safe while `core` runs, and it writes the reference file atomically. It prints
the line to add to the `[survey.pointing]` table of the local configuration. `core` loads the file
when it starts (`seeingmon.survey.analyzer.SurveyPipelineAnalyzer`), so restart `core` afterwards.

`show` prints a reference file, and with `--data-dir` the offset of the newest stored solution from
it. The logic lives in `seeingmon.survey.reference`. This module defines the arguments, reads the
configuration, and prints.

The handlers import the heavy modules when they run, so `seeingmon --help` stays fast.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

from seeingmon.cli import CliError, Subparsers, add_command

if TYPE_CHECKING:
    from seeingmon.profile import Profile
    from seeingmon.store.db import StoreReader
    from seeingmon.survey.config import SurveyConfig

DEFAULT_MIN_MATCHED = 100  # a solution with fewer matched stars does not serve
DEFAULT_MAX_AGE_MIN = 60.0  # a solution that is older than this does not serve


def register_pointing(subparsers: Subparsers) -> None:
    parser = add_command(
        subparsers,
        "pointing",
        help="Save and show the reference solution of the pointing.",
        handler=_missing_subcommand,
    )
    commands = parser.add_subparsers(dest="pointing_command", metavar="<subcommand>", required=True)
    _register_set_reference(commands)
    _register_show(commands)


def _missing_subcommand(args: argparse.Namespace) -> int:
    # `required=True` makes argparse reject a missing subcommand before this runs.
    raise CliError("choose a subcommand: set-reference or show", exit_code=2)


def _add_local_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--local-config",
        type=Path,
        help="read this file instead of local/config.toml (an absent file is ignored)",
    )


def _register_set_reference(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    command = commands.add_parser(
        "set-reference",
        help="Save the latest good pointing solution as the reference solution.",
        description=(
            "Save the newest good pointing solution of the store as the reference solution that "
            "the Pointing card measures the offset from. Run it after you align the camera and "
            "the Align page shows Polaris on the aim. A good solution has a solver, enough "
            "matched stars, a finite residual, and no time_invalid flag, and it is recent. The "
            "command opens the store read-only, so it is safe while core runs. It prints the "
            "line to add to [survey.pointing] in local/config.toml. Restart core, because core "
            "loads the file when it starts."
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
        "--out",
        type=Path,
        metavar="FILE",
        help="the file to write (default: reference_file in [survey.pointing], else "
        "pointing-reference.json in calibration_dir of [survey], or in calibration/ of the "
        "data directory)",
    )
    command.add_argument(
        "--id",
        metavar="NAME",
        help="the ID that the pointing records carry (default: reference-YYYYMMDDTHHMMSSZ, the "
        "UTC time of the solution)",
    )
    command.add_argument(
        "--min-matched",
        type=int,
        default=DEFAULT_MIN_MATCHED,
        metavar="N",
        help=f"the fewest matched stars of a solution (default {DEFAULT_MIN_MATCHED})",
    )
    command.add_argument(
        "--max-age-min",
        type=float,
        default=DEFAULT_MAX_AGE_MIN,
        metavar="MINUTES",
        help=f"the oldest solution to use, in minutes (default {DEFAULT_MAX_AGE_MIN:g})",
    )
    command.add_argument(
        "--force", action="store_true", help="replace a reference file that exists"
    )
    _add_local_config(command)
    command.set_defaults(handler=_set_reference)


def _register_show(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    command = commands.add_parser(
        "show",
        help="Print a reference solution, and the offset of the newest solution from it.",
        description=(
            "Print the reference solution in a file. With --data-dir, also read the newest "
            "pointing solution from the store and print its offset from the reference."
        ),
    )
    command.add_argument(
        "--file",
        type=Path,
        metavar="FILE",
        help="the reference file (default: reference_file in [survey.pointing], else the file "
        "that set-reference writes by default)",
    )
    command.add_argument(
        "--data-dir",
        type=Path,
        metavar="PATH",
        help="also print the offset of the newest solution in db/results.sqlite of this "
        "data directory",
    )
    _add_local_config(command)
    command.set_defaults(handler=_show)


# --- Reading the configuration -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PointingContext:
    """What the commands read from the configuration.

    `data_dir` is `--data-dir` or `[paths] data_dir`, and `calibration_dir` is `[survey]
    calibration_dir` or `calibration/` in the data directory, as `core` chooses it. Each is `None`
    when the configuration gives no value.
    """

    profile: Profile
    survey: SurveyConfig
    data_dir: Path | None
    calibration_dir: Path | None

    @property
    def default_file(self) -> Path | None:
        """The reference file of the configuration, or the default place for one, or `None`."""
        from seeingmon.survey.reference import REFERENCE_FILENAME

        configured = self.survey.pointing.reference_file
        if configured:
            return Path(configured)
        if self.calibration_dir is None:
            return None
        return self.calibration_dir / REFERENCE_FILENAME


def _load_context(args: argparse.Namespace) -> PointingContext:
    from seeingmon.config import ConfigError, load_config
    from seeingmon.profile import ProfileError
    from seeingmon.store.layout import DataLayout
    from seeingmon.survey.config import SurveyConfig
    from seeingmon.survey.dark import CALIBRATION_DIRNAME

    try:
        config = load_config(local_file=args.local_config)
        survey = config.section("survey", SurveyConfig)
        profile = config.profile
    except (ConfigError, ProfileError) as exc:
        raise CliError(str(exc)) from None
    data_dir: Path | None = None if args.data_dir is None else Path(args.data_dir)
    if data_dir is None:
        try:
            data_dir = DataLayout.from_config(config).root
        except ConfigError:
            data_dir = None
    calibration: Path | None = None
    if survey.calibration_dir:
        calibration = Path(survey.calibration_dir)
    elif data_dir is not None:
        calibration = data_dir / CALIBRATION_DIRNAME
    return PointingContext(profile, survey, data_dir, calibration)


def _now_utc_ns() -> int:
    """The time now. A test replaces this function."""
    from seeingmon.clock import SystemClock

    return SystemClock().utc_ns()


def _open_store(context: PointingContext) -> StoreReader:
    """Open the store of the data directory read-only, as `seeingmon store info` does."""
    import sqlite3

    from seeingmon.store.db import StoreError, StoreReader
    from seeingmon.store.layout import DataLayout

    if context.data_dir is None:
        raise CliError("pass --data-dir, or set data_dir in [paths] of local/config.toml")
    try:
        return StoreReader.open(DataLayout(context.data_dir).db_path)
    except FileNotFoundError:
        raise CliError("there is no store database in the data directory") from None
    except (StoreError, sqlite3.Error) as exc:
        raise CliError(f"cannot read the store: {exc}") from None


def _file_to_use(chosen: Path | None, context: PointingContext) -> Path:
    """The absolute path of the file that you named, or of the one that the configuration gives."""
    target = chosen if chosen is not None else context.default_file
    if target is None:
        raise CliError(
            "name the file, or set reference_file in [survey.pointing], calibration_dir in "
            "[survey], or data_dir in [paths] of local/config.toml"
        )
    return Path(os.path.abspath(target))


def _print_rows(rows: list[tuple[str, str]]) -> None:
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")


# --- set-reference -----------------------------------------------------------------------------


def _same_path(first: Path, second: Path) -> bool:
    def normal(path: Path) -> str:
        return os.path.normcase(os.path.abspath(path))

    return normal(first) == normal(second)


def _toml_line(path: PurePath) -> str:
    """The line of the configuration that names `path`: forward slashes, and a TOML string."""
    return f"reference_file = {json.dumps(path.as_posix())}"


def _set_reference(args: argparse.Namespace) -> int:
    from seeingmon.survey import reference as ref
    from seeingmon.survey.pointing import ReferenceSolution, save_reference

    if args.min_matched < 1:
        raise CliError("--min-matched must be at least 1", exit_code=2)
    if not (math.isfinite(args.max_age_min) and args.max_age_min > 0.0):
        raise CliError("--max-age-min must be a number of minutes above 0", exit_code=2)
    if args.id is not None and not ref.is_valid_reference_id(args.id):
        raise CliError(
            "--id takes 1 to 64 letters, digits, '.', '_', or '-', starting with a letter or digit",
            exit_code=2,
        )
    context = _load_context(args)
    if context.data_dir is None:
        raise CliError("pass --data-dir, or set data_dir in [paths] of local/config.toml")
    out = _file_to_use(args.out, context)
    if out.exists() and not args.force:
        raise CliError(f"{out.name} already exists: add --force to replace it")

    reader = _open_store(context)
    now_utc_ns = _now_utc_ns()
    try:
        record = ref.newest_solution_record(
            reader,
            now_utc_ns=now_utc_ns,
            min_matched=args.min_matched,
            max_age_s=args.max_age_min * 60.0,
        )
    except ref.NoSolutionError as exc:
        raise CliError(str(exc)) from None
    finally:
        reader.close()
    try:
        solution = ref.solution_from_record(record, context.profile, dut1_s=context.survey.dut1_s)
    except ref.PointingReferenceError as exc:
        raise CliError(str(exc)) from None
    reference = ReferenceSolution(args.id or ref.default_reference_id(solution.t_utc_ns), solution)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        save_reference(out, reference)
    except OSError as exc:
        raise CliError(f"cannot write {out.name}: {exc.strerror or exc}") from None

    print("Saved the pointing reference.")
    _print_rows([*ref.summary_rows(reference, now_utc_ns=now_utc_ns), ("file", str(out))])
    print()
    configured = context.survey.pointing.reference_file
    if configured and _same_path(Path(configured), out):
        print("[survey.pointing] reference_file already names this file.")
        print("Restart core to load the new reference.")
    else:
        print("Add this to local/config.toml, then restart core to load the reference:")
        print()
        print("[survey.pointing]")
        print(_toml_line(out))
    return 0


# --- show --------------------------------------------------------------------------------------


def _show(args: argparse.Namespace) -> int:
    from seeingmon.survey import reference as ref
    from seeingmon.survey.pointing import load_reference, offset_between

    context = _load_context(args)
    path = _file_to_use(args.file, context)
    try:
        reference = load_reference(path)
    except OSError as exc:
        raise CliError(f"cannot read {path.name}: {exc.strerror or exc}") from None
    except (ValueError, KeyError, TypeError) as exc:
        raise CliError(
            f"{path.name} is not a pointing reference ({type(exc).__name__}: {exc})"
        ) from None
    if reference is None:
        raise CliError(f"there is no file {path.name}: run seeingmon pointing set-reference")
    _print_rows([*ref.summary_rows(reference), ("file", str(path))])
    if args.data_dir is None:
        return 0

    reader = _open_store(context)
    try:
        record = ref.newest_solution_record(
            reader, now_utc_ns=_now_utc_ns(), min_matched=1, max_age_s=None
        )
    except ref.NoSolutionError as exc:
        print()
        print(f"No offset, because {exc}.")
        return 0
    finally:
        reader.close()
    try:
        solution = ref.solution_from_record(record, context.profile, dut1_s=context.survey.dut1_s)
    except ref.PointingReferenceError as exc:
        raise CliError(str(exc)) from None
    offset = offset_between(solution, reference.solution)
    print()
    _print_rows(
        [
            (
                "newest solution",
                f"{ref.format_time(solution.t_utc_ns)}, {solution.n_matched} matched stars",
            ),
            ("boresight offset", f"{ref.format_number(offset.boresight_arcmin, 2)} arcmin"),
            ("roll offset", f"{ref.format_number(offset.roll_deg, 2)} degrees"),
        ]
    )
    return 0
