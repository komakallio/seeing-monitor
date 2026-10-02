"""The `seeingmon dark` command, against the simulated camera on a virtual clock.

The tests replace two things: the clock, so that no exposure and no wait takes real time, and
the driver factory, so that the command gets a sim camera that the test builds (a covered one,
or one with a cover that goes on after a few frames). The configuration comes from environment
variables, as it does in a real run. The tests run the command with `--standalone`, which opens the
camera in this process. The run through `core` has its tests in
`tests/services/core/test_dark_command.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from seeingmon.cli import main
from seeingmon.clock import VirtualClock
from seeingmon.drivers.base import CameraDisconnectedError, CameraDriver
from seeingmon.drivers.sim import SimDriver, SimOptions
from seeingmon.survey import cli as survey_cli
from seeingmon.survey.dark import DarkLibrary
from tests.survey import simfx, synth

PROFILE = synth.cropped_profile(512, 384)


class Setup:
    """The pieces that a test of the command shares."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.clock = VirtualClock()
        self.created: list[dict[str, Any]] = []
        self.driver: CameraDriver | None = None
        self.library_dir = tmp_path / "calibration" / "darks"
        monkeypatch.setattr(survey_cli, "_make_clock", lambda services: self.clock)
        monkeypatch.setattr("seeingmon.drivers.create_driver", self._create_driver)
        monkeypatch.setenv("SEEINGMON_SURVEY__CALIBRATION_DIR", str(tmp_path / "calibration"))
        monkeypatch.delenv("SEEINGMON_PATHS__DATA_DIR", raising=False)

    def _create_driver(self, name: str, *, profile: Any, clock: Any, options: Any) -> CameraDriver:
        self.created.append({"name": name, "options": options})
        if self.driver is None:
            self.driver = covered_driver(self.clock)
        return self.driver

    def library(self) -> DarkLibrary:
        return DarkLibrary(self.library_dir)


def covered_driver(clock: VirtualClock, *, seed: int = 3) -> SimDriver:
    options = simfx.covered_options(ambient_c=12.0, hot_pixels_per_mpix=200.0, seed=seed)
    return simfx.make_driver(PROFILE, options, clock)


def test_the_command_records_a_set_with_the_configured_driver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    setup = Setup(monkeypatch, tmp_path)
    code = main(["dark", "--standalone", "--no-wait", "--frames", "3", "--bias-frames", "3"])
    out = capsys.readouterr().out
    assert code == 0
    assert setup.created == [{"name": "sim", "options": {}}]  # the driver of [services.acquire]
    (dark_set,) = setup.library().sets()
    assert dark_set.mode == "bin2"
    assert dark_set.gain == 120
    assert dark_set.exposure_s == 30.0
    assert dark_set.n_frames == 3
    assert dark_set.temperature_c == pytest.approx(16.0)  # 12 C ambient and 4 C of self-heating
    assert "Taking 3 bias frames" in out
    assert "Recording 3 dark frames of 30 s." in out
    assert f"Added {dark_set.name} to the library" in out
    assert "Cover the camera now" not in out  # --no-wait skips the wait
    assert str(tmp_path) not in out  # the summary shows no path


def test_the_command_waits_for_the_cover_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    setup = Setup(monkeypatch, tmp_path)
    uncovered = simfx.make_driver(PROFILE, SimOptions(seed=5), setup.clock)
    setup.driver = simfx.CoverableDriver(
        uncovered, covered_driver(setup.clock, seed=5), cover_after_reads=3 + 2
    )
    code = main(["dark", "--standalone", "--frames", "3", "--bias-frames", "3"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Cover the camera now. Waiting for a dark frame." in out
    assert "The camera is dark." in out
    assert len(setup.library().sets()) == 1


def test_options_on_the_command_line_override_the_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    setup = Setup(monkeypatch, tmp_path)
    monkeypatch.setenv("SEEINGMON_SURVEY__DARK__EXPOSURE_S", "20")
    assert main(["dark", "--standalone", "--no-wait", "--frames", "3", "--bias-frames", "3"]) == 0
    assert setup.library().sets()[0].exposure_s == 20.0  # the configured exposure
    assert (
        main(
            [
                "dark",
                "--standalone",
                "--no-wait",
                "--frames",
                "4",
                "--bias-frames",
                "3",
                "--exposure-s",
                "10",
            ]
        )
        == 0
    )
    capsys.readouterr()
    newest = setup.library().sets()[-1]
    assert (newest.exposure_s, newest.n_frames) == (10.0, 4)


def test_a_failed_check_is_an_error_message_and_leaves_the_library_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    setup = Setup(monkeypatch, tmp_path)
    setup.driver = simfx.make_driver(PROFILE, SimOptions(seed=5), setup.clock)  # uncovered
    code = main(["dark", "--standalone", "--no-wait", "--frames", "3", "--bias-frames", "3"])
    captured = capsys.readouterr()
    assert code == 1
    assert "seeingmon: error: dark frame 1 of 3 is not dark" in captured.err
    assert "Traceback" not in captured.err
    assert setup.library().sets() == ()


def test_the_wait_gives_up_after_the_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    setup = Setup(monkeypatch, tmp_path)
    setup.driver = simfx.make_driver(PROFILE, SimOptions(seed=5), setup.clock)
    code = main(
        ["dark", "--standalone", "--wait-timeout", "30", "--bias-frames", "3", "--frames", "3"]
    )
    assert code == 1
    assert "not dark after 30 s" in capsys.readouterr().err


def test_a_camera_fault_names_the_likely_cause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    setup = Setup(monkeypatch, tmp_path)

    class Gone:
        def open(self) -> None:
            raise CameraDisconnectedError("no camera answers")

        def close(self) -> None:
            return None

    setup.driver = Gone()  # type: ignore[assignment]
    code = main(["dark", "--standalone", "--no-wait"])
    err = capsys.readouterr().err
    assert code == 1
    assert "the camera failed: no camera answers" in err
    assert "Stop acquire first" in err


def test_a_driver_that_cannot_be_created_is_an_error_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    Setup(monkeypatch, tmp_path)

    def broken(name: str, **kwargs: Any) -> CameraDriver:
        raise ImportError("the vendor library is missing")

    monkeypatch.setattr("seeingmon.drivers.create_driver", broken)
    code = main(["dark", "--standalone", "--no-wait", "--driver", "asi"])
    err = capsys.readouterr().err
    assert code == 1
    assert "cannot create the driver 'asi': ImportError: the vendor library is missing" in err


def test_the_library_folder_comes_from_the_option_or_the_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    setup = Setup(monkeypatch, tmp_path)
    elsewhere = tmp_path / "elsewhere"
    assert (
        main(
            [
                "dark",
                "--standalone",
                "--no-wait",
                "--frames",
                "3",
                "--bias-frames",
                "3",
                "--library",
                str(elsewhere),
            ]
        )
        == 0
    )
    assert len(DarkLibrary(elsewhere).sets()) == 1
    assert setup.library().sets() == ()  # the option wins over calibration_dir
    # With no option and no calibration_dir, the data directory holds the library.
    monkeypatch.delenv("SEEINGMON_SURVEY__CALIBRATION_DIR")
    monkeypatch.setenv("SEEINGMON_PATHS__DATA_DIR", str(tmp_path / "data"))
    assert main(["dark", "--standalone", "--no-wait", "--frames", "3", "--bias-frames", "3"]) == 0
    assert len(DarkLibrary(tmp_path / "data" / "calibration" / "darks").sets()) == 1
    capsys.readouterr()


def test_without_any_library_location_the_command_says_what_to_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    Setup(monkeypatch, tmp_path)
    monkeypatch.delenv("SEEINGMON_SURVEY__CALIBRATION_DIR")
    code = main(["dark", "--standalone", "--no-wait"])
    assert code == 1
    assert "pass --library, or set calibration_dir" in capsys.readouterr().err


def test_a_bad_mode_or_count_is_an_error_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    Setup(monkeypatch, tmp_path)
    assert main(["dark", "--standalone", "--no-wait", "--frames", "1"]) == 1
    assert "at least 3 dark frames" in capsys.readouterr().err
    assert main(["dark", "--standalone", "--no-wait", "--mode", "bin9"]) == 1
    assert "bin9" in capsys.readouterr().err


def test_the_help_lists_the_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["dark", "--help"])
    assert raised.value.code == 0
    text = capsys.readouterr().out
    for option in (
        "--no-wait",
        "--frames",
        "--bias-frames",
        "--exposure-s",
        "--library",
        "--standalone",
        "--detach",
        "--address",
        "--wait-timeout",
    ):
        assert option in text


def test_the_help_says_that_the_mode_and_the_gain_need_standalone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["dark", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "(only with --standalone; core takes the mode of [survey.dark])" in text
    assert "(only with --standalone; core takes the gain of [survey.dark])" in text
    assert "--wait-timeout" in text
    assert "(default: wait_timeout_s of [survey.dark])" in text  # a run through core takes it too
