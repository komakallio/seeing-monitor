"""GPIO lines and environment sensors: the fakes, `LibgpiodIo` on libgpiod stand-ins, and sysfs."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from seeingmon.clock import VirtualClock
from seeingmon.hardware.io import (
    EnvReading,
    EnvSensor,
    FakeEnvSensor,
    FakeIo,
    Io,
    IoError,
    LibgpiodIo,
    NullIo,
    PinSpec,
    SensorConfig,
    SensorError,
    SysfsEnvSensor,
    build_sensor,
    load_libgpiod,
)
from tests.hardware.gpiod_fakes import FakeGpiodV1, FakeGpiodV2

HEATER = PinSpec(chip="gpiochip0", line=17)
FAULT = PinSpec(chip="gpiochip0", line=27, direction="input")


class TestPinSpec:
    def test_a_device_name_lives_under_dev(self) -> None:
        assert PinSpec(chip="gpiochip4", line=1).device_path == "/dev/gpiochip4"
        assert PinSpec(chip="/dev/gpiochip0", line=1).device_path == "/dev/gpiochip0"

    def test_defaults(self) -> None:
        spec = PinSpec(chip="gpiochip0", line=5)
        assert (spec.direction, spec.active_low, spec.bias) == ("output", False, "none")

    @pytest.mark.parametrize(
        "values",
        [
            {"chip": "", "line": 1},
            {"chip": "gpiochip0", "line": -1},
            {"chip": "gpiochip0", "line": 1, "direction": "both"},
            {"chip": "gpiochip0", "line": 1, "bias": "pull_up"},  # bias on an output
            {"chip": "gpiochip0", "line": 1, "extra": True},
        ],
    )
    def test_a_bad_spec_is_refused(self, values: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            PinSpec(**values)


class TestFakeIo:
    def test_satisfies_the_protocol(self) -> None:
        io: Io = FakeIo()
        assert isinstance(io, Io)
        assert isinstance(NullIo(), Io)

    def test_outputs_are_recorded_with_their_time(self) -> None:
        clock = VirtualClock()
        io = FakeIo(clock, outputs=["heater"])
        io.set_output("heater", True)
        clock.advance(2.0)
        io.set_output("heater", False)
        assert io.values == {"heater": False}
        assert [(t, name, value) for t, name, value in io.history] == [
            (0, "heater", True),
            (2_000_000_000, "heater", False),
        ]

    def test_an_unknown_output_is_refused_when_names_are_declared(self) -> None:
        with pytest.raises(IoError, match="unknown output"):
            FakeIo(outputs=["heater"]).set_output("other", True)

    def test_a_loopback_wire_reads_back_the_output(self) -> None:
        io = FakeIo()
        io.link("out", "in")
        assert io.read_input("in") is False
        io.set_output("out", True)
        assert io.read_input("in") is True

    def test_a_scripted_input(self) -> None:
        io = FakeIo()
        io.set_input("fault", True)
        assert io.read_input("fault") is True

    def test_close_turns_the_outputs_off_and_refuses_later_calls(self) -> None:
        io = FakeIo(outputs=["heater"])
        io.set_output("heater", True)
        io.close()
        io.close()
        assert io.values == {"heater": False}
        assert io.closed
        with pytest.raises(IoError, match="closed"):
            io.set_output("heater", True)

    def test_a_scripted_failure_raises_once(self) -> None:
        io = FakeIo()
        io.fail_next()
        with pytest.raises(IoError, match="scripted"):
            io.set_output("heater", True)
        io.set_output("heater", True)


class TestLibgpiodV2:
    def make(self, **pins: PinSpec) -> tuple[LibgpiodIo, FakeGpiodV2]:
        library = FakeGpiodV2()
        return LibgpiodIo(pins or {"heater": HEATER, "fault": FAULT}, library=library), library

    def test_an_output_is_requested_off_and_an_input_as_an_input(self) -> None:
        io, library = self.make()
        assert library.opened == ["/dev/gpiochip0"]  # one chip serves both lines
        assert library.consumers == ["seeingmon", "seeingmon"]
        assert library.levels[("/dev/gpiochip0", 17)] == 0
        assert library.freed.count("settings") == 2  # the temporary objects are freed
        assert library.freed.count("line_config") == 2
        assert library.freed.count("request_config") == 2
        io.close()

    def test_set_output_drives_the_line(self) -> None:
        io, library = self.make()
        io.set_output("heater", True)
        assert library.levels[("/dev/gpiochip0", 17)] == 1
        io.set_output("heater", False)
        assert library.levels[("/dev/gpiochip0", 17)] == 0

    def test_an_active_low_output_starts_high_and_inverts(self) -> None:
        inverted = PinSpec(chip="gpiochip0", line=22, active_low=True)
        io, library = self.make(relay=inverted)
        assert library.levels[("/dev/gpiochip0", 22)] == 1  # off is high
        io.set_output("relay", True)
        assert library.levels[("/dev/gpiochip0", 22)] == 0
        io.close()
        assert library.levels[("/dev/gpiochip0", 22)] == 1  # off again before the release

    def test_a_loopback_reads_back(self) -> None:
        io, library = self.make()
        library.wires[("/dev/gpiochip0", 27)] = ("/dev/gpiochip0", 17)
        assert io.read_input("fault") is False
        io.set_output("heater", True)
        assert io.read_input("fault") is True

    def test_an_active_low_input_inverts(self) -> None:
        pin = PinSpec(chip="gpiochip0", line=27, direction="input", active_low=True)
        io, library = self.make(fault=pin)
        assert io.read_input("fault") is True  # the line is low, and low means asserted
        library.levels[("/dev/gpiochip0", 27)] = 1
        assert io.read_input("fault") is False

    def test_the_bias_reaches_the_library(self) -> None:
        pin = PinSpec(chip="gpiochip0", line=27, direction="input", bias="pull_up")
        library = FakeGpiodV2()
        LibgpiodIo({"fault": pin}, library=library)
        assert list(library.settings.values())[-1]["bias"] == 4  # pull-up in the enumeration

    def test_no_bias_leaves_the_pull_resistor_alone(self) -> None:
        library = FakeGpiodV2()
        LibgpiodIo({"fault": FAULT}, library=library)
        assert "gpiod_line_settings_set_bias" not in library.log
        pull_down = PinSpec(chip="gpiochip0", line=28, direction="input", bias="pull_down")
        LibgpiodIo({"other": pull_down}, library=library)
        assert list(library.settings.values())[-1]["bias"] == 5  # pull-down

    def test_close_turns_the_output_off_then_releases_every_line_and_the_chip(self) -> None:
        io, library = self.make()
        io.set_output("heater", True)
        io.close()
        io.close()
        assert library.levels[("/dev/gpiochip0", 17)] == 0
        assert library.released == 2
        assert library.closed_chips == 1
        with pytest.raises(IoError, match="closed"):
            io.set_output("heater", True)

    def test_a_name_or_direction_mistake_is_an_error(self) -> None:
        io, _ = self.make()
        with pytest.raises(IoError, match="unknown pin"):
            io.set_output("other", True)
        with pytest.raises(IoError, match="not an output"):
            io.set_output("fault", True)
        with pytest.raises(IoError, match="not an input"):
            io.read_input("heater")

    @pytest.mark.parametrize(
        "function",
        ["gpiod_chip_open", "gpiod_chip_request_lines", "gpiod_line_settings_new"],
    )
    def test_a_failing_request_raises_and_releases_what_it_took(self, function: str) -> None:
        library = FakeGpiodV2()
        library.fail.add(function)
        with pytest.raises(IoError):
            LibgpiodIo({"heater": HEATER}, library=library)
        assert library.released == 0

    def test_a_failing_second_line_releases_the_first(self) -> None:
        library = FakeGpiodV2()
        original = library.gpiod_chip_request_lines  # type: ignore[attr-defined]
        calls = []

        def failing_second(*args: object) -> object:
            calls.append(1)
            return None if len(calls) == 2 else original(*args)

        library.gpiod_chip_request_lines = failing_second  # type: ignore[attr-defined]
        with pytest.raises(IoError):
            LibgpiodIo({"heater": HEATER, "fault": FAULT}, library=library)
        assert library.released == 1
        assert library.levels[("/dev/gpiochip0", 17)] == 0

    def test_a_failing_write_and_read_raise(self) -> None:
        io, library = self.make()
        library.fail.update({"gpiod_line_request_set_value", "gpiod_line_request_get_value"})
        with pytest.raises(IoError, match="set the line"):
            io.set_output("heater", True)
        with pytest.raises(IoError, match="read the line"):
            io.read_input("fault")


class TestLibgpiodV1:
    def make(self, **pins: PinSpec) -> tuple[LibgpiodIo, FakeGpiodV1]:
        library = FakeGpiodV1()
        return LibgpiodIo(pins or {"heater": HEATER, "fault": FAULT}, library=library), library

    def test_an_output_is_requested_off_and_driven(self) -> None:
        io, library = self.make()
        assert library.levels[("/dev/gpiochip0", 17)] == 0
        io.set_output("heater", True)
        assert library.levels[("/dev/gpiochip0", 17)] == 1
        assert library.consumers == ["seeingmon", "seeingmon"]

    def test_an_input_reads_a_wired_line(self) -> None:
        io, library = self.make()
        library.wires[("/dev/gpiochip0", 27)] = ("/dev/gpiochip0", 17)
        io.set_output("heater", True)
        assert io.read_input("fault") is True

    def test_the_bias_becomes_a_request_flag(self) -> None:
        pin = PinSpec(chip="gpiochip0", line=27, direction="input", bias="pull_down")
        library = FakeGpiodV1()
        LibgpiodIo({"fault": pin}, library=library)
        assert library.flags == [1 << 4]

    def test_close_releases_every_line(self) -> None:
        io, library = self.make()
        io.set_output("heater", True)
        io.close()
        assert library.levels[("/dev/gpiochip0", 17)] == 0
        assert library.released == 2
        assert library.closed_chips == 1

    def test_a_missing_line_is_an_error(self) -> None:
        library = FakeGpiodV1()
        library.fail.add("gpiod_chip_get_line")
        with pytest.raises(IoError, match="no such line"):
            LibgpiodIo({"heater": HEATER}, library=library)


class TestLoading:
    def test_the_library_that_exports_settings_functions_is_version_2(self) -> None:
        assert type(load_libgpiod(FakeGpiodV2())).__name__ == "_GpiodV2"
        assert type(load_libgpiod(FakeGpiodV1())).__name__ == "_GpiodV1"

    def test_a_library_without_a_needed_function_is_refused(self) -> None:
        library = FakeGpiodV2()
        del library.gpiod_line_request_get_value  # type: ignore[attr-defined]
        with pytest.raises(IoError, match="gpiod_line_request_get_value"):
            load_libgpiod(library)

    def test_a_platform_other_than_linux_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("seeingmon.hardware.io._is_linux", lambda: False)
        with pytest.raises(IoError, match="needs Linux"):
            LibgpiodIo({"heater": HEATER})

    def test_the_loader_tries_the_system_library_and_then_the_usual_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("seeingmon.hardware.io._is_linux", lambda: True)
        tried: list[str] = []

        def loader(name: str) -> FakeGpiodV2:
            tried.append(name)
            if name != "libgpiod.so.2":
                raise OSError("no such library")
            return FakeGpiodV2()

        backend = load_libgpiod(loader=loader, find_library=lambda _: "libgpiod.so.99")
        assert type(backend).__name__ == "_GpiodV2"
        assert tried == ["libgpiod.so.99", "libgpiod.so.3", "libgpiod.so.2"]

    def test_a_path_that_you_give_is_the_only_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("seeingmon.hardware.io._is_linux", lambda: True)
        tried: list[str] = []

        def loader(name: str) -> FakeGpiodV1:
            tried.append(name)
            return FakeGpiodV1()

        load_libgpiod(path="gpiod-here", loader=loader, find_library=lambda _: None)
        assert tried == ["gpiod-here"]

    def test_no_library_is_an_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("seeingmon.hardware.io._is_linux", lambda: True)

        def loader(name: str) -> object:
            raise OSError(name)

        with pytest.raises(IoError, match="not installed"):
            load_libgpiod(loader=loader, find_library=lambda _: None)


class TestSensors:
    def test_the_fake_reads_and_changes(self) -> None:
        clock = VirtualClock()
        sensor = FakeEnvSensor(clock, temperature_c=5.0, humidity_pct=70.0)
        assert isinstance(sensor, EnvSensor)
        assert sensor.read() == EnvReading(clock.utc_ns(), 5.0, 70.0)
        sensor.set(temperature_c=-2.0)
        assert sensor.read().temperature_c == -2.0
        assert sensor.read().humidity_pct == 70.0
        assert sensor.reads == 3

    def test_the_fake_can_fail(self) -> None:
        sensor = FakeEnvSensor(VirtualClock())
        sensor.fail()
        with pytest.raises(SensorError):
            sensor.read()
        sensor.fail(False)
        sensor.read()

    def test_the_sysfs_sensor_scales_the_files(self, tmp_path: Path) -> None:
        (tmp_path / "temp1_input").write_text("-2500\n", encoding="ascii")
        (tmp_path / "humidity").write_text("63400\n", encoding="ascii")
        sensor = SysfsEnvSensor(
            VirtualClock(),
            temperature_file=str(tmp_path / "temp1_input"),
            humidity_file=str(tmp_path / "humidity"),
        )
        reading = sensor.read()
        assert reading.temperature_c == pytest.approx(-2.5)
        assert reading.humidity_pct == pytest.approx(63.4)

    def test_a_sensor_with_one_quantity_leaves_the_other_out(self, tmp_path: Path) -> None:
        (tmp_path / "t").write_text("21500", encoding="ascii")
        reading = SysfsEnvSensor(VirtualClock(), temperature_file=str(tmp_path / "t")).read()
        assert reading.temperature_c == pytest.approx(21.5)
        assert reading.humidity_pct is None

    def test_an_unreadable_file_is_a_sensor_error_without_the_path(self, tmp_path: Path) -> None:
        sensor = SysfsEnvSensor(VirtualClock(), temperature_file=str(tmp_path / "missing"))
        with pytest.raises(SensorError) as raised:
            sensor.read()
        assert str(tmp_path) not in str(raised.value)
        (tmp_path / "bad").write_text("not a number", encoding="ascii")
        with pytest.raises(SensorError):
            SysfsEnvSensor(VirtualClock(), temperature_file=str(tmp_path / "bad")).read()

    def test_the_sysfs_sensor_needs_a_file(self) -> None:
        with pytest.raises(ValueError, match="name a temperature file"):
            SysfsEnvSensor(VirtualClock())

    def test_the_config_builds_each_kind(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        assert build_sensor(SensorConfig(), clock) is None
        fixed = build_sensor(
            SensorConfig(kind="fixed", temperature_c=4.0, humidity_pct=80.0), clock
        )
        assert fixed is not None
        assert fixed.read().humidity_pct == 80.0
        (tmp_path / "t").write_text("1000", encoding="ascii")
        sysfs = build_sensor(
            SensorConfig(kind="sysfs", temperature_file=str(tmp_path / "t")), clock
        )
        assert sysfs is not None
        assert sysfs.read().temperature_c == pytest.approx(1.0)

    @pytest.mark.parametrize("values", [{"kind": "fixed"}, {"kind": "sysfs"}, {"kind": "gpio"}])
    def test_a_sensor_config_needs_what_its_kind_needs(self, values: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            SensorConfig(**values)
