from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import seeingmon
from seeingmon.cli import CliError, main

DEMO_CLI = """\
from seeingmon.cli import CliError, add_command


def _hello(args):
    print("hello from demo")
    return 7


def _fail(args):
    raise CliError("demo failure", exit_code=3)


def register(subparsers):
    add_command(subparsers, "demo-hello", help="Say hello.", handler=_hello)
    add_command(subparsers, "demo-fail", help="Fail on purpose.", handler=_fail)
"""


@pytest.fixture
def demo_subpackage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Add a throwaway subpackage with a `cli` module to the `seeingmon` search path."""
    package = tmp_path / "demo"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text(DEMO_CLI)
    monkeypatch.setattr(seeingmon, "__path__", [*seeingmon.__path__, str(tmp_path)])
    importlib.invalidate_caches()
    yield
    for name in [n for n in sys.modules if n.startswith("seeingmon.demo")]:
        del sys.modules[name]


def test_version_flag_prints_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"seeingmon {seeingmon.__version__}"


def test_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "usage: seeingmon" in capsys.readouterr().out


def test_missing_command_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2
    assert "required" in capsys.readouterr().err


@pytest.mark.usefixtures("demo_subpackage")
def test_commands_in_subpackages_are_discovered(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo-hello"]) == 7
    assert capsys.readouterr().out.strip() == "hello from demo"


@pytest.mark.usefixtures("demo_subpackage")
def test_cli_error_prints_a_message_and_returns_its_exit_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["demo-fail"]) == 3
    assert "demo failure" in capsys.readouterr().err


def test_discovery_does_not_import_a_subpackage_that_has_no_cli_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The `__init__` of a subpackage without commands can load numpy, which costs seconds on a
    # Raspberry Pi. This one fails when it runs, so the test fails if discovery imports it.
    quiet = tmp_path / "demo_quiet"
    quiet.mkdir()
    (quiet / "__init__.py").write_text("raise RuntimeError('imported without a cli module')\n")
    loud = tmp_path / "demo_loud"
    loud.mkdir()
    (loud / "__init__.py").write_text("")
    (loud / "cli.py").write_text(DEMO_CLI)
    monkeypatch.setattr(seeingmon, "__path__", [*seeingmon.__path__, str(tmp_path)])
    importlib.invalidate_caches()
    try:
        assert main(["demo-hello"]) == 7
    finally:
        for name in [n for n in sys.modules if n.startswith("seeingmon.demo_")]:
            del sys.modules[name]
    assert capsys.readouterr().out.strip() == "hello from demo"


def test_cli_error_keeps_its_message() -> None:
    assert str(CliError("broken")) == "broken"
    assert CliError("broken").exit_code == 1
