"""The commands of the services lane: `acquire`, `core`, and the commissioning tools."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from seeingmon.cli import main

KEY_VARIABLE = "SEEINGMON_SERVICES__CONNECTION_KEY"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("SEEINGMON_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)


def test_the_commands_are_listed(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    output = capsys.readouterr().out
    for name in ("acquire", "core", "web", "burst", "sweep", "replay", "dev", "heater-off"):
        assert name in output


def test_acquire_help_names_its_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["acquire", "--help"])
    output = capsys.readouterr().out
    for option in ("--driver", "--driver-option", "--address", "--local-config"):
        assert option in output


def test_acquire_without_a_key_says_where_to_set_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["acquire", "--local-config", str(tmp_path / "none.toml")]) == 1
    error = capsys.readouterr().err
    assert "no connection key" in error
    assert "SEEINGMON_SERVICES__CONNECTION_KEY" in error


def test_a_driver_option_needs_a_name_and_a_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["acquire", "--driver-option", "novalue", "--local-config", str(tmp_path / "x")])
    assert code == 2
    assert "KEY=VALUE" in capsys.readouterr().err


def test_an_unknown_driver_is_reported_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_VARIABLE, "a-test-key-with-32-characters-long")
    code = main(["acquire", "--driver", "nope", "--local-config", str(tmp_path / "none.toml")])
    assert code == 1
    error = capsys.readouterr().err
    assert "cannot create the driver 'nope'" in error
    assert "ValueError: unknown driver" in error
    assert "Traceback" not in error


def test_a_bad_option_of_the_fake_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_VARIABLE, "a-test-key-with-32-characters-long")
    code = main(
        [
            "acquire",
            "--driver",
            "fake",
            "--driver-option",
            "colour=1",
            "--local-config",
            str(tmp_path / "none.toml"),
        ]
    )
    assert code == 1
    assert "no option named 'colour'" in capsys.readouterr().err


def test_a_bad_address_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_VARIABLE, "a-test-key-with-32-characters-long")
    code = main(
        [
            "acquire",
            "--address",
            "a" * 300,
            "--driver",
            "fake",
            "--local-config",
            str(tmp_path / "n"),
        ]
    )
    assert code == 1
    assert capsys.readouterr().err.startswith("seeingmon: error:")
