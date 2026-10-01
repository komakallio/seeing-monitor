#!/usr/bin/env python3
"""Run the secret scan that CI runs, on a clone.

CI scans every tracked file with `detect-secrets-hook --baseline .secrets.baseline`, and one
false positive turns `main` red on every runner. Run this before you push.

Usage:
    python tools/scan_secrets.py --repo PATH             every tracked file, as CI does
    python tools/scan_secrets.py --repo PATH --staged    only the files in the index

Mark a false positive with `# pragma: allowlist secret` on the same line.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

BASELINE = ".secrets.baseline"
# A Windows command line holds at most 32,767 characters, so scan the files in batches.
BATCH_SIZE = 100

_RUN_HOOK = (
    "import sys; from detect_secrets.pre_commit_hook import main; sys.exit(main(sys.argv[1:]))"
)


def list_files(repo: Path, *, staged: bool) -> list[str]:
    """The tracked files of `repo`, or only the added, copied, modified, and renamed staged ones."""
    command = ["git", "-C", str(repo)]
    if staged:
        command += ["diff", "--staged", "--name-only", "--diff-filter=ACMR", "-z"]
    else:
        command += ["ls-files", "-z"]
    output = subprocess.run(command, check=True, capture_output=True).stdout.decode("utf-8")
    return [name for name in output.split("\0") if name]


def batches(names: Sequence[str], size: int = BATCH_SIZE) -> Iterator[Sequence[str]]:
    for start in range(0, len(names), size):
        yield names[start : start + size]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CI secret scan on a clone.")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="the clone to scan")
    parser.add_argument("--staged", action="store_true", help="scan only the staged files")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    names = list_files(repo, staged=args.staged)
    status = 0
    for batch in batches(names):
        command = [sys.executable, "-c", _RUN_HOOK, "--baseline", BASELINE, *batch]
        status = max(status, subprocess.run(command, cwd=repo, check=False).returncode)
    if status == 0:
        print(f"scan_secrets: clean ({len(names)} files)")
    else:
        print("scan_secrets: the scan found a possible secret (see above)", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
