"""The commissioning commands: `burst`, `sweep`, and `replay`, through `core` and on their own.

A command that goes through `core` connects to a running `CoreApp` over the real IPC layer. A thread
steps the scheduler of the stepped rig on a virtual clock, so a burst of seconds takes milliseconds.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("scipy", reason="the fast path needs the fast extra")

from seeingmon.cli import main
from seeingmon.clock import VirtualClock
from seeingmon.config import load_config
from seeingmon.hardware.io import FakeIo, PinSpec
from seeingmon.scheduler.commands import QueueBurst, QueueReplay, QueueSweep
from seeingmon.scheduler.commission import format_sweep_table
from seeingmon.services.config import ServicesConfig
from seeingmon.services.core.commissioning.client import cells_from_result
from seeingmon.services.core.commissioning.standalone import run_standalone
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.store.db import Store
from seeingmon.store.layout import PIN_MARKER
from seeingmon.testing import FakeCameraDriver
from tests.services.addresses import unique_address

from .rig import NIGHT, SMALL_BIN2, CoreRig, build_rig, local_config_text, write_small_profile
from .test_replay import write_recording

KEY_TEXT = "a-test-key-of-more-than-32-characters"
KEY_VARIABLE = "SEEINGMON_SERVICES__CONNECTION_KEY"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("SEEINGMON_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    monkeypatch.setenv(KEY_VARIABLE, KEY_TEXT)


class Stepper(threading.Thread):
    """Steps the scheduler of a stepped rig, and does its periodic work, until told to stop."""

    def __init__(self, rig: CoreRig) -> None:
        super().__init__(daemon=True)
        self.rig = rig
        self.stop_event = threading.Event()
        self.held = threading.Event()  # set while a test wants the scheduler to stand still
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                if not self.held.is_set():
                    self.rig.app.scheduler.step()
                    self.rig.app.tick()
                time.sleep(0.0005)
        except BaseException as error:
            self.error = error


@dataclass
class Served:
    rig: CoreRig
    stepper: Stepper
    address: str
    config: Path
    data: Path


@pytest.fixture
def served(tmp_path: Path) -> Iterator[Served]:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    write_recording(recordings / "night.ser", frames=400)
    rig = build_rig(
        tmp_path,
        key=ConnectionKey.from_text(KEY_TEXT),
        config_extra=f'[replay]\nrecordings_dir = "{recordings.as_posix()}"\n',
    )
    rig.app.start()
    stepper = Stepper(rig)
    stepper.start()
    yield Served(
        rig, stepper, str(rig.app.bound_endpoint), tmp_path / "local.toml", tmp_path / "data"
    )
    stepper.stop_event.set()
    stepper.join(30.0)
    rig.app.stop()
    assert stepper.error is None


def run(
    capsys: pytest.CaptureFixture[str], served: Served, *arguments: str
) -> tuple[int, str, str]:
    code = main(
        [
            *arguments,
            "--address",
            served.address,
            "--local-config",
            str(served.config),
            "--wait-timeout",
            "120",
        ]
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestThroughCore:
    def test_a_burst_is_recorded_pinned_and_reported(
        self, served: Served, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = run(capsys, served, "burst", "--duration", "1", "--label", "bench")
        assert code == 0, err
        assert "burst 1: ok" in out
        folders = [p for p in (served.data / "bursts").iterdir() if p.is_dir()]
        assert len(folders) == 1
        assert "bench" in folders[0].name
        names = {p.name for p in folders[0].iterdir()}
        assert any(n.endswith(".ser") for n in names)
        assert any(n.endswith(".json") for n in names)
        assert PIN_MARKER in names  # the burst is pinned, so retention keeps it
        assert f"file: bursts/{folders[0].name}" in out.replace("\\", "/")

    def test_no_wait_returns_after_the_answer(
        self, served: Served, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = run(capsys, served, "burst", "--duration", "1", "--no-wait")
        assert code == 0
        assert "burst" in out
        assert "waiting for task" not in out

    def test_a_sweep_prints_its_table(
        self, served: Served, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = run(
            capsys,
            served,
            "sweep",
            "--exposure-us",
            "2000,4000",
            "--gain",
            "100",
            "--roi-arcmin",
            "8",
            "--window-s",
            "2",
        )
        assert code == 0, err
        assert "sweep 1: ok" in out
        assert "exp_us" in out  # the header of the table
        assert "2000" in out
        assert "4000" in out

    def test_a_replay_goes_into_a_separate_store(
        self, served: Served, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = run(capsys, served, "replay", "night.ser", "--speed", "0")
        assert code == 0, err
        assert "replay 1: ok" in out
        assert "Replayed 400 frames" in out
        assert (served.data / "replays").is_dir()
        assert len(list((served.data / "replays").iterdir())) == 1

    def test_a_command_that_core_rejects_exits_with_two(
        self, served: Served, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = run(capsys, served, "replay", "no-such-recording.ser")
        assert code == 2
        assert "core rejected the command" in out
        assert "no recording" in out or "not found" in out

    def test_an_option_that_could_widen_the_access_is_rejected(
        self, served: Served, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = run(capsys, served, "replay", "night.ser", "--option", "path=other.ser")
        assert code == 2
        assert "core rejected the command" in out

    def test_a_task_that_does_not_finish_in_time_exits_with_one(
        self, served: Served, capsys: pytest.CaptureFixture[str]
    ) -> None:
        served.stepper.held.set()  # the scheduler stands still, so the task never starts
        time.sleep(0.05)  # a step in progress ends
        try:
            code = main(
                [
                    "burst",
                    "--duration",
                    "1",
                    "--address",
                    served.address,
                    "--local-config",
                    str(served.config),
                    "--wait-timeout",
                    "0.2",
                ]
            )
        finally:
            served.stepper.held.clear()
        out = capsys.readouterr().out
        assert code == 1
        assert "did not finish within 0.2 s" in out
        assert "still runs in core" in out


class TestWithoutCore:
    def test_nobody_listens_so_the_message_says_what_to_do(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = tmp_path / "local.toml"
        config.write_text(local_config_text(tmp_path / "data"), encoding="utf-8")
        code = main(
            [
                "burst",
                "--address",
                unique_address("core"),
                "--local-config",
                str(config),
            ]
        )
        error = capsys.readouterr().err
        assert code == 1
        assert "cannot reach core" in error
        assert "seeingmon core" in error
        assert "--standalone" in error

    def test_without_a_key_the_message_says_where_to_set_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(KEY_VARIABLE)
        config = tmp_path / "local.toml"
        config.write_text(local_config_text(tmp_path / "data"), encoding="utf-8")
        code = main(["sweep", "--local-config", str(config)])
        assert code == 1
        assert "no connection key" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (["burst", "--roi", "1,2,3,4"], "stream settings need --exposure-us"),
            (["burst", "--exposure-us", "500", "--roi", "1,2,3"], "--roi wants X,Y,WIDTH,HEIGHT"),
            (["sweep", "--exposure-us", "a,b"], "must be a list of numbers"),
            (["replay", "x.ser", "--option", "novalue"], "KEY=VALUE"),
        ],
    )
    def test_a_bad_argument_is_reported_before_anything_runs(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        arguments: list[str],
        message: str,
    ) -> None:
        config = tmp_path / "local.toml"
        config.write_text(local_config_text(tmp_path / "data"), encoding="utf-8")
        assert main([*arguments, "--local-config", str(config)]) == 2
        assert message in capsys.readouterr().err


def standalone_config(tmp_path: Path, extra: str = "") -> tuple[Path, Path]:
    """A local configuration for a run without core: a small sensor, a scaled night clock."""
    profile = write_small_profile(tmp_path)
    recordings = tmp_path / "recordings"
    recordings.mkdir(exist_ok=True)
    text = local_config_text(
        tmp_path / "data",
        profile,
        extra=(
            f'[replay]\nrecordings_dir = "{recordings.as_posix()}"\n'
            "[services.clock]\n"
            'kind = "scaled"\n'
            "speed = 30.0\n"
            f"start_utc_ns = {NIGHT}\n"
            f"origin_real_ns = {time.time_ns()}\n" + extra
        ),
    )
    path = tmp_path / "standalone.toml"
    path.write_text(text, encoding="utf-8")
    return path, recordings


class TestStandalone:
    def test_a_replay_needs_no_camera_and_no_core(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config, recordings = standalone_config(tmp_path)
        write_recording(recordings / "night.ser", frames=300)
        code = main(
            ["replay", "night.ser", "--standalone", "--local-config", str(config), "--speed", "0"]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "replay 1: ok" in out
        assert "Replayed 300 frames" in out

    def test_a_standalone_run_that_the_scheduler_rejects_exits_with_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config, _ = standalone_config(tmp_path)
        code = main(
            [
                "sweep",
                "--standalone",
                "--local-config",
                str(config),
                "--exposure-us",
                "1",  # below the lower limit of the camera
            ]
        )
        assert code == 2
        assert "the scheduler rejected the command" in capsys.readouterr().out


def make_config(tmp_path: Path) -> tuple[object, ServicesConfig]:
    path, _ = standalone_config(tmp_path)
    config = load_config(local_file=path, env={})
    return config, config.section("services", ServicesConfig)


class TestRunStandalone:
    """The function behind `--standalone`, on a fake camera and a virtual clock."""

    def test_a_burst_writes_its_files_and_pins_them(self, tmp_path: Path) -> None:
        config, services = make_config(tmp_path)
        clock = VirtualClock(NIGHT)
        shown: list[str] = []
        outcome = run_standalone(
            config,  # type: ignore[arg-type]
            services,
            QueueBurst(duration_s=2.0, label="bench"),
            clock=clock,
            driver=FakeCameraDriver(clock, full_frames={"bin1": (1280, 960), "bin2": SMALL_BIN2}),
            show=shown.append,
        )
        assert outcome.answer.accepted
        assert outcome.result is not None
        assert outcome.result.status == "ok"
        assert outcome.result.pinned is True
        (artifact,) = [a for a in outcome.result.artifacts if a.endswith(".ser")]
        assert (tmp_path / "data" / artifact).is_file()

    def test_a_sweep_runs_with_the_center_of_the_sensor_as_polaris(self, tmp_path: Path) -> None:
        config, services = make_config(tmp_path)
        clock = VirtualClock(NIGHT)
        outcome = run_standalone(
            config,  # type: ignore[arg-type]
            services,
            QueueSweep(exposure_us=(2000,), gain=(100,), roi_arcmin=(8.0,), window_s=2.0),
            clock=clock,
            driver=FakeCameraDriver(clock, full_frames={"bin1": (1280, 960), "bin2": SMALL_BIN2}),
        )
        assert outcome.result is not None
        assert outcome.result.status == "ok"
        (cell,) = outcome.result.data["cells"]
        assert cell["status"] == "ok"
        # The command prints the table from the JSON that core sends, so the roundtrip must hold.
        (rebuilt,) = cells_from_result(json.loads(json.dumps(outcome.result.data)))
        assert rebuilt.cell.exposure_us == 2000
        assert rebuilt.status == "ok"
        assert rebuilt.roi_px is not None
        assert "2000" in format_sweep_table([rebuilt])

    def test_a_stop_request_ends_the_run_without_a_result(self, tmp_path: Path) -> None:
        config, services = make_config(tmp_path)
        clock = VirtualClock(NIGHT)
        outcome = run_standalone(
            config,  # type: ignore[arg-type]
            services,
            QueueBurst(duration_s=500.0),
            clock=clock,
            driver=FakeCameraDriver(clock, full_frames={"bin1": (1280, 960), "bin2": SMALL_BIN2}),
            should_stop=lambda: True,
        )
        assert outcome.answer.accepted
        assert outcome.result is None

    def test_the_run_ends_when_the_timeout_passes_before_the_task_finishes(
        self, tmp_path: Path
    ) -> None:
        config, services = make_config(tmp_path)
        clock = VirtualClock(NIGHT)
        outcome = run_standalone(
            config,  # type: ignore[arg-type]
            services,
            QueueBurst(duration_s=5.0),
            clock=clock,
            driver=FakeCameraDriver(clock, full_frames={"bin1": (1280, 960), "bin2": SMALL_BIN2}),
            timeout_s=0.0,  # the clock has passed it after the first step, which only starts
        )
        assert outcome.answer.accepted
        assert outcome.result is None

    def test_a_replay_gets_a_fake_camera_when_none_is_given(self, tmp_path: Path) -> None:
        config, services = make_config(tmp_path)
        write_recording(tmp_path / "recordings" / "night.ser", frames=200)
        outcome = run_standalone(
            config,  # type: ignore[arg-type]
            services,
            QueueReplay("night.ser", 0.0, {}),
            clock=VirtualClock(NIGHT),
        )
        assert outcome.result is not None
        assert outcome.result.data["frames"] == 200


class TestTheCoreCommand:
    def test_without_a_key_the_message_says_where_to_set_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(KEY_VARIABLE)
        config = tmp_path / "local.toml"
        config.write_text(local_config_text(tmp_path / "data"), encoding="utf-8")
        assert main(["core", "--local-config", str(config)]) == 1
        error = capsys.readouterr().err
        assert "core cannot start" in error
        assert "no connection key" in error

    def test_a_missing_catalog_is_a_start_error_and_leaves_the_store_closed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = tmp_path / "local.toml"
        config.write_text(local_config_text(tmp_path / "data"), encoding="utf-8")
        code = main(["core", "--local-config", str(config), "--address", unique_address("core")])
        assert code == 1
        error = capsys.readouterr().err
        assert "core cannot start" in error
        assert "catalog_path is not set" in error
        Store.open(tmp_path / "data" / "db" / "results.sqlite").close()  # the start released it

    def test_a_bad_address_is_a_start_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = tmp_path / "local.toml"
        config.write_text(local_config_text(tmp_path / "data"), encoding="utf-8")
        assert main(["core", "--local-config", str(config), "--address", "a" * 300]) == 1
        assert "core cannot start" in capsys.readouterr().err


class TestHeaterOff:
    def test_a_station_without_a_heater_has_nothing_to_switch_off(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = tmp_path / "local.toml"
        config.write_text(local_config_text(tmp_path / "data"), encoding="utf-8")
        assert main(["heater-off", "--local-config", str(config)]) == 0
        assert "not enabled" in capsys.readouterr().out

    ENABLED = """
[heater]
enabled = true
[heater.pins.heater]
chip = "gpiochip0"
line = 1
direction = "output"
[heater.ambient]
kind = "fixed"
temperature_c = 5.0
"""

    def test_an_enabled_heater_is_switched_off_and_released(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        opened: list[FakeIo] = []

        def open_lines(pins: Mapping[str, PinSpec]) -> FakeIo:
            opened.append(FakeIo(outputs=pins))
            return opened[-1]

        monkeypatch.setattr("seeingmon.hardware.heater_off.open_lines", open_lines)
        config = tmp_path / "local.toml"
        config.write_text(
            local_config_text(tmp_path / "data", extra=self.ENABLED), encoding="utf-8"
        )
        assert main(["heater-off", "--local-config", str(config)]) == 0
        assert [(io.values, io.closed) for io in opened] == [({"heater": False}, True)]
        assert "outputs are off" in capsys.readouterr().out

    def test_a_heater_that_cannot_be_reached_is_an_error_with_a_message(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def open_lines(pins: Mapping[str, PinSpec]) -> FakeIo:
            raise OSError("no gpio here")

        monkeypatch.setattr("seeingmon.hardware.heater_off.open_lines", open_lines)
        config = tmp_path / "local.toml"
        config.write_text(
            local_config_text(tmp_path / "data", extra=self.ENABLED), encoding="utf-8"
        )
        assert main(["heater-off", "--local-config", str(config)]) == 1
        error = capsys.readouterr().err
        assert "cannot switch off heater: unexpected OSError" in error
        assert "no gpio here" not in error  # the journal gets the type of the error, not its text
