"""Command-line entry point.

Any subpackage of `seeingmon` can add commands. Define `seeingmon.<subpackage>.cli` with
a `register(subparsers)` function that calls `add_command` once per command. The entry
point imports these modules by convention, so adding a command never edits a shared file.
Keep heavy imports inside the handler, so that `seeingmon --help` stays fast.

A handler takes the parsed `argparse.Namespace` and returns the process exit code. To
fail with a message, raise `CliError`.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from collections.abc import Callable, Sequence
from typing import TypeAlias

import seeingmon

Subparsers: TypeAlias = "argparse._SubParsersAction[argparse.ArgumentParser]"
Handler: TypeAlias = Callable[[argparse.Namespace], int]


class CliError(Exception):
    """A failure to report to the user without a traceback."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def add_command(
    subparsers: Subparsers, name: str, *, help: str, handler: Handler
) -> argparse.ArgumentParser:
    """Add a command and return its parser, so the caller can add arguments."""
    parser = subparsers.add_parser(name, help=help, description=help)
    parser.set_defaults(handler=handler)
    return parser


def _register_discovered_commands(subparsers: Subparsers) -> None:
    for module_info in sorted(pkgutil.iter_modules(seeingmon.__path__), key=lambda m: m.name):
        if not module_info.ispkg:
            continue
        module_name = f"seeingmon.{module_info.name}.cli"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:  # the subpackage has no commands
                continue
            raise
        module.register(subparsers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seeingmon",
        description="Seeing and sky-quality monitor for a fixed camera that points at Polaris.",
    )
    parser.add_argument("--version", action="version", version=f"seeingmon {seeingmon.__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>", required=True)
    _register_discovered_commands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except CliError as exc:
        print(f"seeingmon: error: {exc}", file=sys.stderr)
        return exc.exit_code
