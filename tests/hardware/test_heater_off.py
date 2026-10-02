"""The `seeingmon heater-off` command: it switches the outputs off, and it ends in time."""

from __future__ import annotations

import inspect
import io
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pytest

from seeingmon.cli import build_parser, main
from seeingmon.config import ConfigError, load_config
from seeingmon.hardware import heater_off
from seeingmon.hardware.heater import HeaterConfig
from seeingmon.hardware.heater_off import (
    DEADLINE_S,
    HeaterLines,
    IoFactory,
    read_plan,
    run_heater_off,
)
from seeingmon.hardware.io import FakeIo, GpiodLibraryError, Io, IoError, LibgpiodIo, PinSpec
from tests.hardware.gpiod_fakes import FakeGpiodV1, FakeGpiodV2

CHIP = "gpiochip-test"  # a chip that no board has, so a real libgpiod finds nothing to drive
PREFIX = "seeingmon heater-off: "

HEATER_TOML = f"""\
[heater]
enabled = true
output = "heater"
keepalive = "watchdog"
fault_input = "fault"

[heater.ambient]
kind = "fixed"
temperature_c = 5.0
humidity_pct = 80.0

[heater.pins.heater]
chip = "{CHIP}"
line = 17

[heater.pins.watchdog]
chip = "{CHIP}"
line = 18

[heater.pins.fault]
chip = "{CHIP}"
line = 27
direction = "input"
"""


@pytest.fixture(autouse=True)
def _no_heater_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read no `[heater]` value from the environment of the person who runs the tests."""
    for name in [name for name in os.environ if name.upper().startswith("SEEINGMON_HEATER")]:
        monkeypatch.delenv(name)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(HEATER_TOML, encoding="utf-8")
    return path


class ScriptedIo(FakeIo):
    """A `FakeIo` that fails a write or the release when the test says so."""

    def __init__(
        self,
        outputs: Iterable[str],
        *,
        bad_writes: Iterable[str] = (),
        bad_close: Exception | None = None,
    ) -> None:
        super().__init__(outputs=outputs)
        self._bad_writes = frozenset(bad_writes)
        self._bad_close = bad_close

    def set_output(self, name: str, value: bool) -> None:
        if name in self._bad_writes:
            raise IoError("scripted write failure")
        super().set_output(name, value)

    def close(self) -> None:
        super().close()  # the fake still turns its values off
        if self._bad_close is not None:
            raise self._bad_close


class IoFarm:
    """An `IoFactory` that builds one fake `Io` for each request and remembers the requests."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, ...]] = []
        self.ios: list[ScriptedIo] = []
        self.refuse: dict[str, Exception] = {}  # an output name, and the error its request raises
        self.bad_writes: set[str] = set()
        self.bad_close: Exception | None = None
        self.start_on = False  # the lines were left on, as after a crash

    def __call__(self, pins: Mapping[str, PinSpec]) -> Io:
        names = tuple(pins)
        self.requests.append(names)
        for name in names:
            if name in self.refuse:
                raise self.refuse[name]
        fake = ScriptedIo(names, bad_writes=self.bad_writes, bad_close=self.bad_close)
        if self.start_on:
            fake.values.update(dict.fromkeys(names, True))
        self.ios.append(fake)
        return fake


def run(config: Path, open_io: IoFactory | None) -> tuple[int, str, str]:
    """Run the command on fakes. Returns the exit code, standard output, and standard error."""
    out, err = io.StringIO(), io.StringIO()
    code = run_heater_off(
        local_config=config,
        open_io=open_io,
        stdout=out,
        stderr=err,
        exit_process=lambda code: None,
    )
    return code, out.getvalue(), err.getvalue()


def lines(text: str) -> list[str]:
    return text.splitlines()


def writes(fake: FakeIo) -> list[tuple[str, bool]]:
    """The (name, value) pairs that a `FakeIo` recorded."""
    return [(name, value) for _, name, value in fake.history]


class TestSwitchingOff:
    def test_every_output_goes_off_and_the_lines_are_released(self, config_file: Path) -> None:
        farm = IoFarm()
        farm.start_on = True
        code, out, err = run(config_file, farm)
        assert code == 0
        assert farm.requests == [("heater", "watchdog")]  # the fault input is not requested
        fake = farm.ios[0]
        assert writes(fake)[:2] == [("heater", False), ("watchdog", False)]
        assert fake.values == {"heater": False, "watchdog": False}
        assert fake.closed
        assert lines(out) == [f"{PREFIX}the heater outputs are off: heater, watchdog"]
        assert err == ""

    def test_a_second_run_does_the_same(self, config_file: Path) -> None:
        farm = IoFarm()
        assert run(config_file, farm)[0] == 0
        assert run(config_file, farm)[0] == 0
        assert [fake.values for fake in farm.ios] == [{"heater": False, "watchdog": False}] * 2

    @pytest.mark.parametrize("library_type", [FakeGpiodV1, FakeGpiodV2])
    def test_active_low_is_honored(
        self, config_file: Path, library_type: type[FakeGpiodV1 | FakeGpiodV2]
    ) -> None:
        # The heater (line 17) is active low, and the watchdog (line 18) is active high.
        config_file.write_text(
            HEATER_TOML.replace("line = 17\n", "line = 17\nactive_low = true\n"), encoding="utf-8"
        )
        library = library_type()

        def open_lines(pins: Mapping[str, PinSpec]) -> Io:
            return LibgpiodIo(pins, library=library)

        code, _, err = run(config_file, open_lines)
        assert (code, err) == (0, "")
        assert library.levels[(f"/dev/{CHIP}", 17)] == 1  # active low: off is high
        assert library.levels[(f"/dev/{CHIP}", 18)] == 0  # active high: off is low
        set_value = (
            "gpiod_line_request_set_value"
            if library_type is FakeGpiodV2
            else "gpiod_line_set_value"
        )
        assert library.log.count(set_value) >= 2  # the command drives each output itself
        assert library.released == 2  # each output is released, and the input is never requested
        assert library.closed_chips >= 1

    def test_no_heater_configured_is_a_pass(self, tmp_path: Path) -> None:
        farm = IoFarm()
        code, out, err = run(tmp_path / "absent.toml", farm)
        assert code == 0
        assert lines(out) == [f"{PREFIX}the heater is not enabled, so there is nothing to do"]
        assert err == ""
        assert farm.requests == []

    def test_a_disabled_heater_keeps_its_pin_map_untouched(self, config_file: Path) -> None:
        config_file.write_text(
            HEATER_TOML.replace("enabled = true", "enabled = false"), encoding="utf-8"
        )
        farm = IoFarm()
        code, out, _ = run(config_file, farm)
        assert code == 0
        assert "nothing to do" in out
        assert farm.requests == []

    def test_every_output_line_of_the_pin_map_goes_off(self, config_file: Path) -> None:
        # No key names the third output, but it is in the pin map, so it is a heater output.
        config_file.write_text(
            HEATER_TOML + f'\n[heater.pins.lamp]\nchip = "{CHIP}"\nline = 19\n', encoding="utf-8"
        )
        farm = IoFarm()
        code, out, _ = run(config_file, farm)
        assert code == 0
        assert farm.requests == [("heater", "watchdog", "lamp")]
        assert lines(out) == [f"{PREFIX}the heater outputs are off: heater, watchdog, lamp"]

    def test_a_mistake_in_a_sensor_or_a_control_key_does_not_stop_the_outputs(
        self, config_file: Path
    ) -> None:
        # The controller refuses this section: a sysfs sensor needs a file, and a key is
        # misspelled. `core` would not start with it, but the outputs must still go off.
        text = HEATER_TOML.replace(
            'kind = "fixed"\ntemperature_c = 5.0\nhumidity_pct = 80.0\n', 'kind = "sysfs"\n'
        ).replace("enabled = true\n", "enabled = true\nmargin_celcius = 3.0\n")
        config_file.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(local_file=config_file, env={}).section("heater", HeaterConfig)
        farm = IoFarm()
        code, _, err = run(config_file, farm)
        assert (code, err) == (0, "")
        assert farm.ios[0].values == {"heater": False, "watchdog": False}


class TestFailures:
    def test_a_refused_line_fails_and_the_other_output_still_goes_off(
        self, config_file: Path
    ) -> None:
        farm = IoFarm()
        farm.start_on = True
        farm.refuse["heater"] = IoError("libgpiod could not request the line: the line is busy")
        code, out, err = run(config_file, farm)
        assert code == 1
        assert out == ""
        assert lines(err) == [
            f"{PREFIX}cannot switch off heater: "
            "libgpiod could not request the line: the line is busy"
        ]
        # The command asked for both lines at once, then for each by itself.
        assert farm.requests == [("heater", "watchdog"), ("heater",), ("watchdog",)]
        assert farm.ios[0].values == {"watchdog": False}
        assert farm.ios[0].closed

    def test_a_failing_library_is_reported_for_every_output_without_asking_line_by_line(
        self, config_file: Path
    ) -> None:
        # The lookup of the library can spawn programs, so a failure of the library must not
        # repeat it for each output.
        farm = IoFarm()
        reason = "libgpiod is not installed; install the libgpiod package of the OS"
        farm.refuse["heater"] = GpiodLibraryError(reason)
        code, out, err = run(config_file, farm)
        assert code == 1
        assert out == ""
        assert lines(err) == [
            f"{PREFIX}cannot switch off heater: {reason}",
            f"{PREFIX}cannot switch off watchdog: {reason}",
        ]
        assert farm.requests == [("heater", "watchdog")]

    def test_a_failing_write_fails_and_the_other_output_still_goes_off(
        self, config_file: Path
    ) -> None:
        farm = IoFarm()
        farm.start_on = True
        farm.bad_writes = {"heater"}
        code, _, err = run(config_file, farm)
        assert code == 1
        assert lines(err) == [f"{PREFIX}cannot switch off heater: scripted write failure"]
        assert farm.requests == [("heater", "watchdog")]
        assert ("watchdog", False) in writes(farm.ios[0])
        assert farm.ios[0].closed  # the lines are released even after a failed write

    def test_every_failing_output_gets_its_own_line(self, config_file: Path) -> None:
        farm = IoFarm()
        farm.refuse["heater"] = IoError("permission denied")
        farm.refuse["watchdog"] = IoError("the line is busy")
        code, out, err = run(config_file, farm)
        assert code == 1
        assert out == ""
        assert lines(err) == [
            f"{PREFIX}cannot switch off heater: permission denied",
            f"{PREFIX}cannot switch off watchdog: the line is busy",
        ]

    def test_a_failing_release_is_reported(self, config_file: Path) -> None:
        farm = IoFarm()
        farm.bad_close = IoError("scripted release failure")
        code, _, err = run(config_file, farm)
        assert code == 1
        reason = "could not release the line: scripted release failure"
        assert lines(err) == [
            f"{PREFIX}cannot switch off heater: {reason}",
            f"{PREFIX}cannot switch off watchdog: {reason}",
        ]

    def test_an_unexpected_error_names_its_type_and_leaks_no_text(self, config_file: Path) -> None:
        farm = IoFarm()
        farm.refuse["heater"] = OSError(f"/dev/{CHIP}: boom")
        code, _, err = run(config_file, farm)
        assert code == 1
        assert lines(err) == [f"{PREFIX}cannot switch off heater: unexpected OSError"]
        assert "/dev" not in err
        assert "boom" not in err

    def test_the_full_error_goes_to_the_debug_log_and_not_to_the_journal_line(
        self, config_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        farm = IoFarm()
        farm.refuse["heater"] = OSError(f"/dev/{CHIP}: boom")
        with caplog.at_level(logging.DEBUG, logger="seeingmon.hardware.heater_off"):
            _, _, err = run(config_file, farm)
        assert "boom" not in err
        assert any("boom" in str(record.exc_info) for record in caplog.records if record.exc_info)

    def test_a_message_carries_no_pin_map_device_path_or_address(self, config_file: Path) -> None:
        farm = IoFarm()
        farm.refuse["heater"] = IoError("cannot open a GPIO chip: permission denied")
        _, out, err = run(config_file, farm)
        for text in (out, err):
            assert CHIP not in text
            assert "17" not in text
            assert "/dev" not in text

    @pytest.mark.parametrize(
        ("old", "new", "name"),
        [
            ('keepalive = "watchdog"', 'keepalive = "spare"', "spare"),  # in no pin table
            ('keepalive = "watchdog"', 'keepalive = "fault"', "fault"),  # an input, not an output
            ('output = "heater"', 'output = "main"', "main"),
        ],
    )
    def test_a_named_output_without_an_output_line_is_reported(
        self, config_file: Path, old: str, new: str, name: str
    ) -> None:
        config_file.write_text(HEATER_TOML.replace(old, new), encoding="utf-8")
        farm = IoFarm()
        farm.start_on = True
        code, out, err = run(config_file, farm)
        assert code == 1
        assert out == ""
        assert lines(err) == [
            f"{PREFIX}cannot switch off {name}: the pin map has no output line of this name"
        ]
        # The outputs that the pin map does have still go off.
        assert farm.requests == [("heater", "watchdog")]
        assert farm.ios[0].values == {"heater": False, "watchdog": False}

    def test_a_bad_configuration_is_one_line_and_exit_1(self, config_file: Path) -> None:
        # The heater pin loses its chip.
        config_file.write_text(
            HEATER_TOML.replace(f'chip = "{CHIP}"\nline = 17\n', "line = 17\n"), encoding="utf-8"
        )
        farm = IoFarm()
        code, out, err = run(config_file, farm)
        assert code == 1
        assert out == ""
        assert len(lines(err)) == 1
        assert err.startswith(f"{PREFIX}cannot read the heater configuration: ")
        assert "pins.heater.chip" in err  # the key, and no value
        assert farm.requests == []

    def test_a_file_that_is_not_toml_is_one_line_and_exit_1(self, tmp_path: Path) -> None:
        broken = tmp_path / "config.toml"
        broken.write_text("[heater\nenabled = ", encoding="utf-8")
        code, _, err = run(broken, IoFarm())
        assert code == 1
        assert len(lines(err)) == 1
        assert "cannot read the heater configuration" in err


class TestReading:
    """The command reads a part of `[heater]`, and the controller reads all of it."""

    @pytest.mark.parametrize(
        "text",
        [
            "",
            HEATER_TOML,
            HEATER_TOML.replace("enabled = true", "enabled = false"),
            '[heater]\noutput = "relay"\n',
        ],
        ids=["defaults", "enabled", "disabled", "renamed output"],
    )
    def test_the_part_agrees_with_the_heater_config(self, tmp_path: Path, text: str) -> None:
        path = tmp_path / "config.toml"
        path.write_text(text, encoding="utf-8")
        config = load_config(local_file=path, env={})
        part = config.section("heater", HeaterLines)
        whole = config.section("heater", HeaterConfig)
        for field in HeaterLines.model_fields:
            assert getattr(part, field) == getattr(whole, field), field

    def test_the_part_has_only_keys_that_the_heater_config_has(self) -> None:
        assert set(HeaterLines.model_fields) <= set(HeaterConfig.model_fields)

    def test_the_plan_lists_the_output_lines(self, config_file: Path) -> None:
        plan = read_plan(config_file)
        assert plan is not None
        assert list(plan.outputs) == ["heater", "watchdog"]  # the input is not a heater output
        assert plan.outputs["heater"] == PinSpec(chip=CHIP, line=17)
        assert plan.missing == ()

    def test_a_missing_file_and_a_disabled_heater_have_no_plan(
        self, config_file: Path, tmp_path: Path
    ) -> None:
        assert read_plan(tmp_path / "absent.toml") is None
        config_file.write_text(
            HEATER_TOML.replace("enabled = true", "enabled = false"), encoding="utf-8"
        )
        assert read_plan(config_file) is None

    def test_an_environment_variable_overrides_the_file(
        self, config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The layers are the same as for `core`: defaults, then the file, then the environment.
        monkeypatch.setenv("SEEINGMON_HEATER__ENABLED", "false")
        assert read_plan(config_file) is None


class HangingIo(FakeIo):
    """An `Io` whose write blocks until the test releases it, as a GPIO library can hang."""

    def __init__(self, outputs: Iterable[str], release: threading.Event) -> None:
        super().__init__(outputs=outputs)
        self.release = release

    def set_output(self, name: str, value: bool) -> None:
        self.release.wait(30)  # a real hang never returns, and the test frees its thread at the end
        super().set_output(name, value)


class RecordingTimer:
    """A stand-in for `threading.Timer` that records what the command does with it."""

    created: list[RecordingTimer] = []  # noqa: RUF012 - a test double that the test resets

    def __init__(self, interval: float, function: Callable[[], None]) -> None:
        self.interval = interval
        self.function = function
        self.daemon = False
        self.started = False
        self.cancelled = False
        RecordingTimer.created.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


class TestTimeBound:
    def test_a_hanging_call_ends_the_process_in_time_with_a_failure(
        self, config_file: Path
    ) -> None:
        release = threading.Event()
        exits: list[int] = []
        blocked_at_exit: list[bool] = []

        def exit_process(code: int) -> None:
            exits.append(code)
            blocked_at_exit.append(not release.is_set())  # the call was still stuck
            release.set()  # the process ends, and the stuck call with it

        out, err = io.StringIO(), io.StringIO()
        started = time.perf_counter()
        code = run_heater_off(
            local_config=config_file,
            open_io=lambda pins: HangingIo(pins, release),
            deadline_s=0.3,
            exit_process=exit_process,
            stdout=out,
            stderr=err,
        )
        elapsed = time.perf_counter() - started
        assert exits == [1]
        assert blocked_at_exit == [True]
        assert code == 1  # a run that passed its deadline never reports success
        assert elapsed < 2.0
        assert lines(err.getvalue()) == [
            f"{PREFIX}the GPIO calls did not finish within 0.3 s; "
            "not confirmed off: heater, watchdog"
        ]
        assert out.getvalue() == ""

    def test_the_default_limit_leaves_room_below_two_seconds(self) -> None:
        assert 0 < DEADLINE_S < 2.0
        assert inspect.signature(run_heater_off).parameters["deadline_s"].default == DEADLINE_S

    def test_the_default_exit_ends_the_process_without_a_clean_shutdown(self) -> None:
        # A blocked native call would block a clean shutdown too, so the timer uses `os._exit`.
        assert inspect.signature(run_heater_off).parameters["exit_process"].default is os._exit

    def test_a_run_that_ends_in_time_cancels_its_timer(
        self, config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        RecordingTimer.created = []
        monkeypatch.setattr(threading, "Timer", RecordingTimer)
        code, _, _ = run(config_file, IoFarm())
        (timer,) = RecordingTimer.created
        assert code == 0
        assert timer.interval == DEADLINE_S
        assert (timer.started, timer.cancelled, timer.daemon) == (True, True, True)

    def test_a_timer_that_fires_after_the_work_does_nothing(
        self, config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        RecordingTimer.created = []
        monkeypatch.setattr(threading, "Timer", RecordingTimer)
        exits: list[int] = []
        out, err = io.StringIO(), io.StringIO()
        code = run_heater_off(
            local_config=config_file,
            open_io=IoFarm(),
            exit_process=exits.append,
            stdout=out,
            stderr=err,
        )
        RecordingTimer.created[0].function()  # the timer wakes after the work ended
        assert (code, exits) == (0, [])
        assert err.getvalue() == ""

    def test_a_timer_that_cannot_start_does_not_stop_the_work(
        self, config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_thread(self: threading.Thread) -> None:
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(threading.Timer, "start", no_thread)
        farm = IoFarm()
        code, out, _ = run(config_file, farm)
        assert code == 0
        assert "are off" in out
        assert farm.ios[0].values == {"heater": False, "watchdog": False}


class TestCommand:
    def test_the_parser_lists_heater_off(self) -> None:
        parser = build_parser()
        assert "heater-off" in parser.format_help()
        args = parser.parse_args(["heater-off"])
        assert args.command == "heater-off"
        assert args.local_config is None

    def test_the_command_runs_through_the_entry_point(
        self, config_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        farm = IoFarm()
        monkeypatch.setattr(heater_off, "open_lines", farm)
        assert main(["heater-off", "--local-config", str(config_file)]) == 0
        assert lines(capsys.readouterr().out) == [
            f"{PREFIX}the heater outputs are off: heater, watchdog"
        ]
        assert farm.requests == [("heater", "watchdog")]


HANG_SCRIPT = """import sys
import threading
from pathlib import Path

from seeingmon.hardware.heater_off import run_heater_off


def hang(pins):
    threading.Event().wait(600)  # a GPIO call that never returns
    raise SystemExit(0)  # not reached: the timer ends the process first


print("ready", flush=True)  # the parent times the run from here, not from the start of Python
sys.exit(run_heater_off(local_config=Path(sys.argv[1]), open_io=hang, deadline_s=0.5))
"""

# The real library lookup with a limit that a slow runner cannot use up. With the library missing,
# the lookup spawns up to three programs, and on a loaded arm64 runner that took longer than the
# 1 s limit of the command once. The limit itself is the subject of `HANG_SCRIPT`.
FAIL_SCRIPT = """import sys
from pathlib import Path

from seeingmon.hardware.heater_off import run_heater_off

sys.exit(run_heater_off(local_config=Path(sys.argv[1]), deadline_s=60.0))
"""

# How long after its first line the hanging child may take to end, besides its own 0.5 s limit.
# The slack covers a slow runner, and it stays far below the 600 s that the child would hang.
HANG_SLACK_S = 5.0


def process_environment() -> dict[str, str]:
    """The environment of the test run, without its `SEEINGMON_` settings."""
    return {k: v for k, v in os.environ.items() if not k.upper().startswith("SEEINGMON_")}


def run_process(tmp_path: Path, config: Path) -> subprocess.CompletedProcess[str]:
    """Run `python -m seeingmon heater-off` as systemd would, without the test's own settings."""
    return subprocess.run(
        [sys.executable, "-m", "seeingmon", "heater-off", "--local-config", str(config)],
        cwd=tmp_path,
        env=process_environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


class TestProcess:
    """The command as a process."""

    def test_a_station_without_a_heater_passes(self, tmp_path: Path) -> None:
        done = run_process(tmp_path, tmp_path / "absent.toml")
        assert done.returncode == 0
        assert done.stderr == ""
        assert lines(done.stdout) == [
            f"{PREFIX}the heater is not enabled, so there is nothing to do"
        ]

    def test_a_heater_that_cannot_be_driven_fails_with_one_line_for_each_output(
        self, config_file: Path, tmp_path: Path
    ) -> None:
        # The pin map names a chip that no board has, so every machine fails to drive it, each with
        # its own reason: not Linux, no libgpiod, or no such device.
        done = subprocess.run(
            [sys.executable, "-c", FAIL_SCRIPT, str(config_file)],
            cwd=tmp_path,
            env=process_environment(),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert done.returncode == 1
        assert done.stdout == ""
        error_lines = lines(done.stderr)
        assert len(error_lines) == 2
        assert error_lines[0].startswith(f"{PREFIX}cannot switch off heater: ")
        assert error_lines[1].startswith(f"{PREFIX}cannot switch off watchdog: ")
        assert CHIP not in done.stderr
        assert "/dev" not in done.stderr

    def test_a_hung_gpio_call_ends_the_process_with_one_line_and_exit_1(
        self, config_file: Path, tmp_path: Path
    ) -> None:
        # No stand-in ends the process here: the real `os._exit` does, from the timer thread,
        # while the main thread waits in a call that never returns. The test times the child from
        # its first line, so the start of Python and the import of the libraries do not count. The
        # child must take at least its own limit (its message says so) and end within the slack.
        process = subprocess.Popen(
            [sys.executable, "-c", HANG_SCRIPT, str(config_file)],
            cwd=tmp_path,
            env=process_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            first_line = process.stdout.readline() if process.stdout else ""
            ready_s = time.monotonic()
            rest, error_text = process.communicate(timeout=60)
            elapsed_s = time.monotonic() - ready_s
        finally:
            if process.poll() is None:
                process.kill()
        assert first_line == "ready\n"
        assert process.returncode == 1
        assert rest == ""
        assert lines(error_text) == [
            f"{PREFIX}the GPIO calls did not finish within 0.5 s; "
            "not confirmed off: heater, watchdog"
        ]
        assert 0.45 <= elapsed_s < 0.5 + HANG_SLACK_S
