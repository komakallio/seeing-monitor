"""Helpers that run the deploy shell scripts against stubs of the programs they call.

The scripts need a POSIX shell and GNU tools, so the tests that use these helpers run on Linux
only. A stub is a tiny shell script in a directory that the test puts first on the `PATH`. It
logs its arguments, one word to a bracket, so that a test can check the exact argument list.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy"

linux_only = pytest.mark.skipif(
    sys.platform != "linux",
    reason="the deploy scripts need a POSIX shell and GNU tools: these tests run on Linux only",
)


@dataclass(frozen=True)
class Result:
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


def run_script(
    script: Path,
    arguments: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Result:
    """Run a script with bash. Its input is empty, so `[ -t 0 ]` is false."""
    result = subprocess.run(
        ["bash", str(script), *arguments],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=dict(env) if env is not None else None,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        timeout=300,
        check=False,
    )
    return Result(result.returncode, result.stdout, result.stderr)


class Stubs:
    """A directory of stub programs, and the log that they write."""

    def __init__(self, root: Path) -> None:
        self.bin = root / "stub-bin"
        self.bin.mkdir()
        self.log = root / "calls.log"
        self.log.write_text("", encoding="utf-8")

    def add(self, name: str, body: str = "") -> Path:
        """Add a stub that logs its call (as `name [arg] [arg]`) and then runs `body`."""
        path = self.bin / name
        path.write_text(
            "#!/bin/sh\n"
            f'printf "%s" "{name}" >> "{self.log}"\n'
            f'for argument in "$@"; do printf " [%s]" "$argument" >> "{self.log}"; done\n'
            f'printf "\\n" >> "{self.log}"\n' + body,
            encoding="utf-8",
            newline="\n",
        )
        path.chmod(0o755)
        return path

    def env(self, **extra: str) -> dict[str, str]:
        """An environment whose PATH starts with the stubs."""
        env = {key: value for key, value in os.environ.items() if key != "TMPDIR"}
        env["PATH"] = f"{self.bin}{os.pathsep}{env['PATH']}"
        env.update(extra)
        return env

    def calls(self) -> list[str]:
        """The logged calls, one string for each."""
        return self.log.read_text(encoding="utf-8").splitlines()

    def called(self, name: str) -> list[str]:
        """The logged calls of one stub, without the name."""
        return [call[len(name) + 1 :] for call in self.calls() if call.split(" ")[0] == name]

    def clear(self) -> None:
        self.log.write_text("", encoding="utf-8")
