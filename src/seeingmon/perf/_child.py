"""The child process of the harness: run one case and print its result.

    python -m seeingmon.perf._child <case> [--smoke] [--registry MODULE] [--quiet-wait SECONDS]

The runner starts this module once for each case, so that the peak memory of the process belongs
to that case alone. The process prints one line, the marker and the JSON of the `CaseResult`, and
exits with 0 whether the case succeeded, was skipped, or failed: the status is in the result.
The exit code is 2 for an unknown case, which is a mistake of the caller.

The module must import nothing heavy at the top, because a case may start worker processes
with the `spawn` method, and a spawned worker imports the main module again.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from seeingmon.perf.registry import UnknownCaseError, load_registry
from seeingmon.perf.runner import RESULT_MARKER, execute_case


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one case of the performance harness.")
    parser.add_argument("case", help="the name of the case")
    parser.add_argument("--smoke", action="store_true", help="use the tiny sizes of a smoke run")
    parser.add_argument(
        "--registry", metavar="MODULE", help="take the cases from the REGISTRY of this module"
    )
    parser.add_argument(
        "--quiet-wait",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="wait up to this long for the machine to be quiet before the case starts",
    )
    args = parser.parse_args(argv)
    try:
        case = load_registry(args.registry).get(args.case)
    except UnknownCaseError as error:
        print(f"error: {error.args[0]}", file=sys.stderr)
        return 2
    result = execute_case(case, smoke=args.smoke, quiet_wait_s=args.quiet_wait)
    sys.stdout.write(f"\n{RESULT_MARKER}{json.dumps(result.to_dict(), allow_nan=False)}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
