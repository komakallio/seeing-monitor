"""The `seeingmon profile` commands and the lazy package imports."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import seeingmon.profile
from seeingmon import paths
from seeingmon.cli import main
from seeingmon.profile import Profile, profile_summary
from tests.profile.builders import REFERENCE_FILE


def profile_text(profile_id: str = "asi294mm-gs250", **replacements: str) -> str:
    """The reference profile file with another id and, optionally, other `old=new` text."""
    text = REFERENCE_FILE.read_text(encoding="utf-8")
    text = text.replace('id = "asi294mm-gs250"', f'id = "{profile_id}"', 1)
    for old, new in replacements.items():
        assert old in text
        text = text.replace(old, new)
    return text


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the developer's own configuration out of the commands that read it."""
    monkeypatch.setattr(paths, "local_config_file", lambda **_: tmp_path / "none.toml")
    for name in [name for name in os.environ if name.startswith("SEEINGMON_")]:
        monkeypatch.delenv(name)


# --- list ---------------------------------------------------------------------------------------


def test_list_prints_each_profile_with_its_description(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profile", "list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("asi294mm-gs250  ZWO ASI294MM")
    assert len(lines) == len(seeingmon.profile.list_profiles())


def test_list_flags_an_invalid_profile_and_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "asi294mm-gs250.toml").write_text(profile_text(), encoding="utf-8")
    (tmp_path / "broken.toml").write_text("x = = 1\n", encoding="utf-8")
    monkeypatch.setattr(paths, "profiles_dir", lambda **_: tmp_path)
    assert main(["profile", "list"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("asi294mm-gs250  ")
    assert lines[1].startswith("broken  (invalid: invalid profile ")
    assert "not valid TOML" in lines[1]


# --- show ---------------------------------------------------------------------------------------


def test_show_prints_the_derived_values_of_each_mode_by_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["profile", "show", "asi294mm-gs250"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Profile asi294mm-gs250\n")
    assert "Readout mode bin1 (SDK bin 1), the fast mode" in out
    assert "Readout mode bin2 (SDK bin 2), the survey mode" in out
    assert "Plate scale in bin1: 1.910 arcsec/px" in out
    assert "Plate scale in bin2: 3.820 arcsec/px" in out
    assert "Field of view in bin1: 4.395 x 2.994 deg, diagonal 5.316 deg" in out
    assert "Sampling in bin1 at 600 nm" in out
    assert "ADC: 12 bit, full scale 4095" in out
    assert "ADC: 14 bit, full scale 16383" in out
    assert "(step)" in out  # the HCG step at gain 120 in bin2
    assert "an estimate good to +-30%" in out


def test_show_prints_plain_ascii(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profile", "show", "asi294mm-gs250"]) == 0
    assert capsys.readouterr().out.isascii()


def test_show_names_the_readout_mode_with_every_plate_scale_and_pixel_size(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["profile", "show", "asi294mm-gs250"])
    quoted = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "arcsec/px" in line or " um;" in line
    ]
    assert len(quoted) == 4  # a pixel size and a plate scale for each of two modes
    assert all("bin1" in line or "bin2" in line for line in quoted)


def test_show_json_prints_the_summary(
    reference: Profile, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", "show", "asi294mm-gs250", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == profile_summary(reference)


def test_show_without_a_name_uses_the_profile_that_the_configuration_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["profile", "show", "--json"]) == 0  # the packaged default is the reference
    assert json.loads(capsys.readouterr().out)["id"] == "asi294mm-gs250"


def test_the_environment_can_choose_the_profile_that_show_uses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SEEINGMON_PROFILE", "other")
    assert main(["profile", "show"]) == 1
    assert "no profile named 'other'" in capsys.readouterr().err


def test_the_environment_can_point_show_at_a_profile_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "my-camera.toml"
    path.write_text(profile_text("my-camera"), encoding="utf-8")
    monkeypatch.setenv("SEEINGMON_PROFILE", f"'{path.as_posix()}'")
    assert main(["profile", "show", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "my-camera"


def test_show_accepts_a_path_to_a_profile_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "my-camera.toml"
    path.write_text(profile_text("my-camera"), encoding="utf-8")
    assert main(["profile", "show", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "my-camera"


def test_show_reports_an_unknown_profile_without_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["profile", "show", "nope"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("seeingmon: error: no profile named 'nope'")
    assert "available profiles: asi294mm-gs250" in captured.err


def test_show_reports_an_invalid_profile_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        profile_text("bad", **{"focal_length_mm = 250.0": "focal_length_mm = -1"}), encoding="utf-8"
    )
    assert main(["profile", "show", str(path)]) == 1
    assert "optics.focal_length_mm: Input should be greater than 0" in capsys.readouterr().err


def test_a_configuration_error_is_reported_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SEEINGMON_X__", "1")
    assert main(["profile", "show"]) == 1
    assert "malformed environment variable" in capsys.readouterr().err


# --- The command tree ----------------------------------------------------------------------------


def test_profile_without_a_subcommand_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["profile"])
    assert exit_info.value.code == 2
    assert "required" in capsys.readouterr().err


def test_the_help_lists_both_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["profile", "--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "list" in out
    assert "show" in out


def test_the_top_level_help_lists_the_profile_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "profile" in capsys.readouterr().out


# --- Lazy imports -------------------------------------------------------------------------------


def run_python(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    return result.stdout.strip()


def test_registering_the_commands_loads_neither_pydantic_nor_numpy() -> None:
    """`seeingmon --help` must stay fast, so the command module imports no heavy package."""
    code = (
        "import sys; import seeingmon.profile.cli; "
        "print('pydantic' in sys.modules or 'numpy' in sys.modules)"
    )
    assert run_python(code) == "False"


def test_the_package_names_import_on_first_use() -> None:
    code = (
        "import sys, seeingmon.profile as p; before = 'pydantic' in sys.modules; "
        "p.load_profile; print(before, 'pydantic' in sys.modules)"
    )
    assert run_python(code) == "False True"


def test_the_package_lists_its_names_and_rejects_others() -> None:
    assert "load_profile" in dir(seeingmon.profile)
    assert set(seeingmon.profile.__all__) <= set(dir(seeingmon.profile))
    with pytest.raises(AttributeError, match="no attribute 'nothing'"):
        seeingmon.profile.__getattr__("nothing")
