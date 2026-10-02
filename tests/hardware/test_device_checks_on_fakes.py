"""The GPIO, SQM-LE, and power checks of `device_checks`, run against fakes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.hardware.io import FakeIo, LibgpiodIo, PinSpec
from seeingmon.hardware.power import CommandRoute, PowerConfig, PowerCycle
from seeingmon.hardware.sqm import SqmConfig, SqmInfluxConfig, SqmLeClient
from seeingmon.hardware.sqm_influx import InfluxStaleError
from seeingmon.sinks.influx import make_opener
from tests.hardware import device_checks as checks
from tests.hardware.gpiod_fakes import FakeGpiodV1, FakeGpiodV2
from tests.hardware.influx_replies import ago, v2_csv
from tests.hardware.sqm_server import SERIAL, FakeSqmServer
from tests.sinks.fake_influx import FakeInfluxServer, Reply


class TestGpioLoopback:
    def test_a_wired_fake_passes(self) -> None:
        io = FakeIo()
        io.link("out", "in")
        assert "5 on and off" in checks.check_gpio_loopback(io, "out", "in")

    def test_a_missing_wire_fails(self) -> None:
        with pytest.raises(AssertionError, match="going on"):
            checks.check_gpio_loopback(FakeIo(), "out", "in")

    @pytest.mark.parametrize("library", [FakeGpiodV2, FakeGpiodV1])
    def test_the_libgpiod_io_passes_on_a_wired_stand_in(self, library: type) -> None:
        stand_in = library()
        stand_in.wires[("/dev/gpiochip0", 27)] = ("/dev/gpiochip0", 17)
        pins = {
            "out": PinSpec(chip="gpiochip0", line=17),
            "in": PinSpec(chip="gpiochip0", line=27, direction="input"),
        }
        io = LibgpiodIo(pins, library=stand_in)
        checks.check_gpio_loopback(io, "out", "in")
        io.close()

    def test_a_pin_is_written_as_chip_and_line(self) -> None:
        pin = checks.pin_from_text("gpiochip0:17", "output")
        assert (pin.chip, pin.line, pin.direction) == ("gpiochip0", 17, "output")
        assert checks.pin_from_text("/dev/gpiochip4:3", "input").chip == "/dev/gpiochip4"
        for bad in ("gpiochip0", ":17", "gpiochip0:x"):
            with pytest.raises(ValueError, match="chip:line"):
                checks.pin_from_text(bad, "output")


@pytest.fixture
def unit() -> Iterator[FakeSqmServer]:
    server = FakeSqmServer()
    try:
        yield server
    finally:
        server.close()


class TestSqm:
    def test_the_fake_unit_passes_and_the_dump_holds_the_raw_responses(
        self, unit: FakeSqmServer, tmp_path: Path
    ) -> None:
        client = SqmLeClient("127.0.0.1", unit.port, read_timeout_s=1.0)
        dump = tmp_path / "sample.txt"
        report = checks.check_sqm_unit(client, dump=dump)
        assert "21.37 mag/arcsec2" in report
        assert SERIAL not in report
        text = dump.read_text(encoding="utf-8")
        assert "rx -> 'r, 21.37m" in text
        assert SERIAL in text  # the raw response holds it, which is why the file stays local

    def test_an_implausible_magnitude_fails(self, unit: FakeSqmServer) -> None:
        unit.magnitude = 2.0
        client = SqmLeClient("127.0.0.1", unit.port, read_timeout_s=1.0)
        with pytest.raises(AssertionError, match="implausible"):
            checks.check_sqm_unit(client)


class TestSqmInflux:
    @staticmethod
    def config(server: FakeInfluxServer) -> SqmConfig:
        table = SqmInfluxConfig(
            endpoint=server.url,
            org="example-org",
            bucket="example-bucket",
            token="example-token-value",
            measurement="sqm",
            field="mag",
            temperature_field="temp",
            timeout_s=5.0,
        )
        return SqmConfig(source="influx", influx=table)

    @staticmethod
    def point(clock: VirtualClock, magnitude: float, temperature: float, age_s: float) -> Reply:
        t_utc_ns = ago(clock.utc_ns(), age_s)
        return Reply(200, v2_csv([("mag", t_utc_ns, magnitude), ("temp", t_utc_ns, temperature)]))

    def run(self, server: FakeInfluxServer, clock: VirtualClock) -> str:
        return checks.check_sqm_influx(
            self.config(server),
            clock=clock,
            env={},
            opener=make_opener(use_environment_proxies=False),
        )

    def test_a_fresh_plausible_point_passes_and_the_report_names_nothing(self) -> None:
        clock = VirtualClock()
        with FakeInfluxServer() as server:
            server.script(self.point(clock, 21.37, 3.5, 20.0))
            report = self.run(server, clock)
            assert server.url not in report
        assert "magnitude 21.37 mag/arcsec^2, temperature 3.5 C, age 20.0 s" in report
        assert "max_age_s (600 s)" in report
        assert "example-bucket" not in report
        assert "example-token-value" not in report

    def test_an_implausible_magnitude_fails(self) -> None:
        clock = VirtualClock()
        with FakeInfluxServer() as server:
            server.script(self.point(clock, 2.0, 3.5, 20.0))
            with pytest.raises(AssertionError, match="implausible"):
                self.run(server, clock)

    def test_an_implausible_temperature_fails(self) -> None:
        clock = VirtualClock()
        with FakeInfluxServer() as server:
            server.script(self.point(clock, 21.37, 300.0, 20.0))
            with pytest.raises(AssertionError, match="temperature is implausible"):
                self.run(server, clock)

    def test_a_stale_point_fails_with_the_error_of_the_reader(self) -> None:
        clock = VirtualClock()
        with FakeInfluxServer() as server:
            server.script(self.point(clock, 21.37, 3.5, 3000.0))
            with pytest.raises(InfluxStaleError):
                self.run(server, clock)

    def test_the_source_tcp_has_no_influx_table_to_check(self) -> None:
        with pytest.raises(AssertionError, match=r"needs the table \[sqm.influx\]"):
            checks.check_sqm_influx(SqmConfig(), clock=VirtualClock())


class TestPower:
    def test_a_configured_route_passes_a_dry_run(self) -> None:
        config = PowerConfig(
            route="command", dry_run=True, command=CommandRoute(argv=["program", "${PLUG}"])
        )
        power = PowerCycle(config, clock=VirtualClock(), env={"PLUG": "plug-1"})
        assert "changed nothing" in checks.check_power_dry_run(power)

    def test_a_missing_variable_fails_the_check(self) -> None:
        config = PowerConfig(
            route="command", dry_run=True, command=CommandRoute(argv=["program", "${PLUG}"])
        )
        power = PowerCycle(config, clock=VirtualClock(), env={})
        with pytest.raises(AssertionError, match="failed"):
            checks.check_power_dry_run(power)
