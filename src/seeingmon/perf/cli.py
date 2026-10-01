"""The `seeingmon perf` commands.

- `seeingmon perf run` runs the cases of the performance harness, each in a child process. Pass
  `--cases` to choose some, `--smoke` for a run that takes seconds and checks that every case
  works, `--json` to save the report, and `--label` to name the machine class (`pi4` for a
  Raspberry Pi 4).
- `seeingmon perf report PATH` prints a saved report. With `--budgets`, it also prints the verdict
  on each budget of the architecture. With `--baseline`, it also prints the ratio to another
  report.

The handlers import the harness on demand, so `seeingmon --help` stays fast. The page
`docs/performance.md` explains the cases and the budgets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from seeingmon.cli import CliError, Subparsers, add_command


def register(subparsers: Subparsers) -> None:
    parser = add_command(
        subparsers,
        "perf",
        help="Measure the figures of the performance gate: run the harness, and read a report.",
        handler=_no_subcommand,
    )
    commands = parser.add_subparsers(dest="perf_command", metavar="<subcommand>", required=True)

    run = add_command(
        commands,
        "run",
        help="Run the cases, each in a child process, and print the figures.",
        handler=_run,
    )
    run.add_argument(
        "--cases",
        metavar="NAME,...",
        help="run only these cases, separated by commas (default: all). See --list.",
    )
    run.add_argument(
        "--smoke",
        action="store_true",
        help="shrink every case to tens of milliseconds. The run checks that the cases work, and "
        "its figures say nothing about speed.",
    )
    run.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="write the report here, for example local/perf/dev.json (Git ignores local/)",
    )
    run.add_argument(
        "--label",
        default="",
        metavar="TEXT",
        help="name the machine class of the run: use pi4 on a Raspberry Pi 4, so that the report "
        "compares its figures with the budgets without scaling",
    )
    run.add_argument(
        "--quiet-wait",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="before each case, wait up to this long for the machine to be at most 15%% busy. "
        "Other work disturbs a measurement, and a report records how busy the machine was.",
    )
    run.add_argument("--list", action="store_true", help="list the cases and exit")

    report = add_command(
        commands,
        "report",
        help="Print a saved report, and with --budgets the verdict on each budget.",
        handler=_report,
    )
    report.add_argument("path", type=Path, help="the JSON file that `perf run --json` wrote")
    report.add_argument(
        "--budgets",
        action="store_true",
        help="print the verdict on each budget: this machine, the estimated Pi 4 range, and "
        "pass, marginal, or fail",
    )
    report.add_argument(
        "--baseline",
        type=Path,
        metavar="PATH",
        help="also print the ratio of each figure to the same figure in this report",
    )
    report.add_argument(
        "--details", action="store_true", help="also print the detail of each figure"
    )


def _no_subcommand(args: argparse.Namespace) -> int:
    raise CliError("choose a subcommand: run or report", exit_code=2)


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _run(args: argparse.Namespace) -> int:
    from seeingmon.perf.registry import UnknownCaseError, load_registry
    from seeingmon.perf.render import format_report
    from seeingmon.perf.report import write_report
    from seeingmon.perf.runner import run_cases

    registry = load_registry()
    if args.list:
        width = max(len(name) for name in registry.names())
        for case in registry.cases():
            print(f"{case.name.ljust(width)}  {case.summary}")
        return 0
    names = [name.strip() for name in args.cases.split(",") if name.strip()] if args.cases else None
    try:
        report = run_cases(
            names,
            smoke=args.smoke,
            label=args.label,
            registry=registry,
            quiet_wait_s=args.quiet_wait,
            progress=_progress,
        )
    except UnknownCaseError as error:
        raise CliError(str(error.args[0]), exit_code=2) from None
    print()
    print(format_report(report))
    if args.json is not None:
        written = write_report(args.json, report)
        print(f"Wrote the report to {written}")
        if not args.smoke:
            print(f"Next: seeingmon perf report {written} --budgets")
    return 1 if any(result.status == "failed" for result in report.cases) else 0


def _report(args: argparse.Namespace) -> int:
    from seeingmon.perf import budgets
    from seeingmon.perf.render import format_comparison, format_report
    from seeingmon.perf.report import ReportError, read_report
    from seeingmon.perf.scaling import PI4_SCALING

    try:
        report = read_report(args.path)
        baseline = read_report(args.baseline) if args.baseline is not None else None
    except ReportError as error:
        raise CliError(str(error), exit_code=2) from None
    print(format_report(report, details=args.details))
    if args.budgets:
        measured = budgets.is_pi4_measurement(report)
        print("Budgets")
        if report.smoke:
            print("This is a smoke report: its figures say nothing about speed.")
        if measured:
            print(
                "Basis: this report comes from a Pi 4 (label pi4, with a measured calibration "
                "case), so the figures are compared with the limits directly, without scaling."
            )
            if not report.environment.is_arm64:
                print(f"Warning: the report says {report.environment.machine}, not arm64.")
        else:
            ranges = ", ".join(
                f"{name} {item.low:g} to {item.high:g}"
                for name, item in PI4_SCALING.items()
                if name != "none"
            )
            print(
                "Basis: estimate. Each figure of this machine is multiplied by the range of its "
                f"class ({ranges}; see seeingmon.perf.scaling), so every Pi 4 figure is an "
                "estimate until a Pi 4 run replaces it."
            )
        print()
        print(budgets.format_verdicts(budgets.evaluate(report)))
        check = budgets.kernel_check(report)
        if check:
            print()
            print("\n".join(check))
    if baseline is not None:
        print()
        print(f"Ratio to the baseline report ({baseline.label or 'no label'}):")
        print(format_comparison(report, baseline))
    return 0
