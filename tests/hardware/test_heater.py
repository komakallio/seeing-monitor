"""The dew heater: the dew point, the time-proportional loop, the safety cutoffs, and the log."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.hardware.events import HardwareEvent
from seeingmon.hardware.heater import (
    HeaterConfig,
    HeaterController,
    HeaterStatus,
    create_heater,
    dew_point_c,
)
from seeingmon.hardware.io import FakeEnvSensor, FakeIo, IoError, NullIo, PinSpec, SensorConfig


class TestDewPoint:
    @pytest.mark.parametrize(
        ("temperature", "humidity", "expected"),
        [(20.0, 50.0, 9.26), (25.0, 60.0, 16.69), (0.0, 100.0, 0.0), (-10.0, 80.0, -12.79)],
    )
    def test_matches_published_values(
        self, temperature: float, humidity: float, expected: float
    ) -> None:
        assert dew_point_c(temperature, humidity) == pytest.approx(expected, abs=0.05)

    def test_agrees_with_the_sonntag_constants_within_a_tenth_of_a_degree(self) -> None:
        def sonntag(t: float, rh: float) -> float:
            gamma = math.log(rh / 100) + 17.62 * t / (243.12 + t)
            return 243.12 * gamma / (17.62 - gamma)

        for t in (-20.0, -5.0, 0.0, 10.0, 25.0, 40.0):
            for rh in (10.0, 40.0, 70.0, 95.0):
                assert dew_point_c(t, rh) == pytest.approx(sonntag(t, rh), abs=0.1)

    def test_saturated_air_has_its_dew_point_at_the_air_temperature(self) -> None:
        for t in (-30.0, 0.0, 12.5, 40.0):
            assert dew_point_c(t, 100.0) == pytest.approx(t, abs=1e-9)

    def test_dry_air_counts_as_one_percent(self) -> None:
        assert dew_point_c(10.0, 0.0) == dew_point_c(10.0, 1.0)

    @pytest.mark.parametrize(
        ("t", "rh"), [(-61.0, 50.0), (81.0, 50.0), (10.0, -1.0), (10.0, 100.5)]
    )
    def test_a_value_outside_the_range_is_refused(self, t: float, rh: float) -> None:
        with pytest.raises(ValueError, match=r"outside|must be"):
            dew_point_c(t, rh)

    @given(
        t=st.floats(min_value=-40, max_value=50),
        low=st.floats(min_value=1, max_value=100),
        high=st.floats(min_value=1, max_value=100),
    )
    def test_the_dew_point_never_exceeds_the_air_and_rises_with_humidity(
        self, t: float, low: float, high: float
    ) -> None:
        low, high = sorted((low, high))
        assert dew_point_c(t, high) <= t + 1e-9
        assert dew_point_c(t, low) <= dew_point_c(t, high) + 1e-9


PINS = {"heater": PinSpec(chip="gpiochip0", line=17)}


def config(**overrides: object) -> HeaterConfig:
    values: dict[str, object] = {
        "enabled": True,
        "pins": PINS,
        "ambient": SensorConfig(kind="fixed", temperature_c=5.0, humidity_pct=90.0),
        **overrides,
    }
    return HeaterConfig(**values)


def output_on(io: FakeIo) -> bool:
    return io.values["heater"]


@dataclass
class Rig:
    clock: VirtualClock
    io: FakeIo
    ambient: FakeEnvSensor
    optics: FakeEnvSensor | None
    heater: HeaterController
    events: list[HardwareEvent] = field(default_factory=list)

    def run(self, seconds: float) -> None:
        end = self.clock.monotonic_ns() + round(seconds * NS_PER_S)
        self.heater.run(lambda: self.clock.monotonic_ns() >= end)

    def advance_running(self, seconds: float) -> None:
        """Step through `seconds` of virtual time without stopping the heater."""
        end = self.clock.monotonic_ns() + round(seconds * NS_PER_S)
        while self.clock.monotonic_ns() < end:
            delay = self.heater.step()
            self.clock.sleep(min(delay, (end - self.clock.monotonic_ns()) / NS_PER_S))

    def duty(self, start_s: float, end_s: float, origin_ns: int) -> float | None:
        return self.heater.mean_duty(
            origin_ns + round(start_s * NS_PER_S), origin_ns + round(end_s * NS_PER_S)
        )

    def kinds(self) -> list[str]:
        return [event.kind for event in self.events]

    def on(self) -> bool:
        """Whether the heater output is on. A method, so the type checker keeps no narrowing."""
        return output_on(self.io)


def make(
    *,
    temperature_c: float = 5.0,
    humidity_pct: float = 90.0,
    optics_c: float | None = None,
    **overrides: object,
) -> Rig:
    clock = VirtualClock()
    io = FakeIo(clock, outputs=["heater", "keepalive"])
    ambient = FakeEnvSensor(clock, temperature_c=temperature_c, humidity_pct=humidity_pct)
    optics = (
        None
        if optics_c is None
        else FakeEnvSensor(clock, temperature_c=optics_c, humidity_pct=None)
    )
    events: list[HardwareEvent] = []
    heater = HeaterController(
        config(**overrides),
        io=io,
        clock=clock,
        ambient=ambient,
        optics=optics,
        on_event=events.append,
    )
    return Rig(clock, io, ambient, optics, heater, events)


class TestDefaults:
    def test_the_output_is_switched_off_at_construction(self) -> None:
        rig = make()
        assert rig.io.history == [(0, "heater", False)]
        assert rig.heater.status().state == "stopped"

    def test_nothing_heats_until_you_start(self) -> None:
        rig = make(temperature_c=-5.0, humidity_pct=100.0)
        for _ in range(10):
            rig.heater.step()
            rig.clock.advance(1.0)
        assert not rig.on()
        assert rig.heater.status().state == "stopped"

    def test_a_disabled_heater_never_starts(self) -> None:
        rig = make(enabled=False)
        rig.heater.start()
        rig.heater.step()
        assert not rig.heater.status().running
        assert rig.heater.status().state == "disabled"
        assert not rig.on()

    def test_stop_switches_the_heater_off_even_in_the_middle_of_a_pulse(self) -> None:
        rig = make(temperature_c=-5.0, humidity_pct=100.0)
        rig.heater.start()
        rig.heater.step()
        assert rig.on()
        rig.heater.stop()
        assert not rig.on()
        assert rig.heater.status().state == "stopped"

    def test_run_switches_the_heater_off_when_it_returns(self) -> None:
        rig = make(temperature_c=-5.0, humidity_pct=100.0)
        rig.run(10.0)
        assert not rig.on()
        assert not rig.heater.status().running

    def test_close_releases_the_lines(self) -> None:
        rig = make()
        rig.heater.close()
        assert rig.io.closed


class TestControl:
    def test_the_duty_follows_the_error_between_the_target_and_the_air(self) -> None:
        rig = make(temperature_c=5.0, humidity_pct=90.0)  # the dew point is 3.5, the target 6.5
        origin = rig.clock.utc_ns()
        rig.run(300.0)
        expected = (dew_point_c(5.0, 90.0) + 3.0 - 5.0) / 4.0
        assert 0.3 < expected < 0.4
        assert rig.duty(0, 300, origin) == pytest.approx(expected, abs=0.002)
        status = rig.heater.status()
        assert status.dew_point_c == pytest.approx(dew_point_c(5.0, 90.0))
        assert status.target_c == pytest.approx(dew_point_c(5.0, 90.0) + 3.0)

    def test_the_output_is_one_slow_pulse_per_period(self) -> None:
        rig = make(temperature_c=5.0, humidity_pct=90.0, period_s=30.0)
        rig.run(90.0)
        changes = [(t // NS_PER_S, value) for t, _, value in rig.io.history]
        on_s = 30 * ((dew_point_c(5.0, 90.0) + 3.0 - 5.0) / 4.0)
        assert [value for _, value in changes[:5]] == [False, True, False, True, False]
        # The pulses start at the period boundaries and last `on_s`.
        assert [t for t, value in changes if value] == [0, 30, 60]
        assert [t for t, value in changes if not value][1:3] == [int(on_s), 30 + int(on_s)]

    def test_air_above_the_target_never_heats(self) -> None:
        rig = make(temperature_c=20.0, humidity_pct=40.0)  # the dew point is 6, the target 9
        rig.heater.start()
        rig.advance_running(120.0)
        assert not any(value for _, _, value in rig.io.history)
        assert rig.heater.status().state == "off"

    def test_a_large_error_gives_full_duty(self) -> None:
        rig = make(temperature_c=0.0, humidity_pct=100.0, margin_c=5.0)
        origin = rig.clock.utc_ns()
        rig.heater.start()
        rig.advance_running(90.0)
        assert rig.heater.status().duty == 1.0
        assert rig.duty(0, 90, origin) == pytest.approx(1.0)

    def test_max_duty_caps_the_heating(self) -> None:
        rig = make(temperature_c=0.0, humidity_pct=100.0, margin_c=5.0, max_duty=0.5)
        origin = rig.clock.utc_ns()
        rig.run(300.0)
        assert rig.duty(0, 300, origin) == pytest.approx(0.5, abs=0.001)

    def test_a_pulse_shorter_than_the_minimum_is_skipped_and_a_gap_is_filled(self) -> None:
        # Error 0.08 degrees over a 4 degree band is a duty of 0.02, which is a 0.6 s pulse.
        short = make(temperature_c=6.4, humidity_pct=100.0, margin_c=0.08)
        short.run(120.0)
        assert not any(value for _, _, value in short.io.history)
        # Error 3.92 over the band is a duty of 0.98, which leaves a 0.6 s gap.
        long = make(temperature_c=0.0, humidity_pct=100.0, margin_c=3.92)
        long.run(120.0)
        assert long.io.history[-1][2] is False  # run() ends with an off
        assert sum(1 for _, _, value in long.io.history if not value) == 2  # start and end only

    def test_a_change_in_the_weather_takes_effect_at_the_next_period(self) -> None:
        rig = make(temperature_c=20.0, humidity_pct=40.0, sensor_interval_s=1.0)
        rig.heater.start()
        rig.advance_running(60.0)
        assert rig.heater.status().duty == 0.0
        rig.ambient.set(temperature_c=0.0, humidity_pct=100.0)
        rig.advance_running(35.0)
        assert rig.heater.status().duty == pytest.approx(0.75)  # error 3 over a 4 degree band

    def test_an_optics_sensor_closes_the_loop(self) -> None:
        rig = make(temperature_c=5.0, humidity_pct=90.0, optics_c=1.0)
        rig.heater.start()
        rig.heater.step()
        # The target is 6.5 and the optics read 1.0, so the error is 5.5 and the duty is full.
        assert rig.heater.status().duty == 1.0
        assert rig.heater.status().optics_c == 1.0
        rig.optics.set(temperature_c=8.0)  # type: ignore[union-attr]
        rig.advance_running(35.0)
        assert rig.heater.status().duty == 0.0

    def test_the_integral_term_removes_the_steady_state_error(self) -> None:
        rig = make(
            temperature_c=5.0,
            humidity_pct=90.0,
            optics_c=5.5,  # a constant error of about 1 degree
            integral_time_s=60.0,
            sensor_interval_s=1.0,
        )
        rig.heater.start()
        duties = []
        for _ in range(6):
            rig.advance_running(30.0)
            duties.append(rig.heater.status().duty)
        assert duties[0] == pytest.approx(0.25, abs=0.01)  # the proportional part alone
        assert all(b >= a for a, b in pairwise(duties))
        assert duties[-1] > duties[0] + 0.2

    def test_the_integral_stops_at_the_duty_limit(self) -> None:
        rig = make(
            temperature_c=5.0, humidity_pct=90.0, optics_c=5.5, integral_time_s=10.0, max_duty=0.6
        )
        rig.heater.start()
        rig.advance_running(600.0)
        assert rig.heater.status().duty == pytest.approx(0.6, abs=0.001)
        rig.optics.set(temperature_c=9.0)  # type: ignore[union-attr]  # now far above the target
        rig.advance_running(150.0)
        assert rig.heater.status().duty == 0.0  # the integral did not wind up past the limit

    def test_status_reports_the_readings(self) -> None:
        rig = make(temperature_c=5.0, humidity_pct=90.0)
        rig.heater.start()
        rig.heater.step()
        status = rig.heater.status()
        assert isinstance(status, HeaterStatus)
        assert (status.ambient_c, status.humidity_pct) == (5.0, 90.0)
        assert status.state == "heating"
        assert status.running
        assert status.enabled
        assert status.faults == ()


class TestSafety:
    def test_over_temperature_cuts_the_heater_at_once_and_latches(self) -> None:
        rig = make(temperature_c=0.0, humidity_pct=100.0, margin_c=5.0, sensor_interval_s=1.0)
        rig.heater.start()
        rig.advance_running(10.0)
        assert rig.on()
        rig.ambient.set(temperature_c=41.0)
        rig.advance_running(2.0)
        assert not rig.on()
        assert rig.heater.status().state == "fault"
        assert rig.heater.status().faults == ("over_temperature",)
        assert rig.kinds() == ["heater.over_temperature"]
        rig.ambient.set(temperature_c=39.0)  # below the limit but inside the hysteresis band
        rig.advance_running(5.0)
        assert rig.heater.status().state == "fault"
        rig.ambient.set(temperature_c=0.0)
        rig.advance_running(5.0)
        assert rig.heater.status().state == "heating"
        assert rig.kinds() == ["heater.over_temperature", "heater.fault_cleared"]

    def test_the_optics_sensor_has_its_own_limit(self) -> None:
        rig = make(optics_c=45.0, sensor_interval_s=1.0)
        rig.heater.start()
        rig.advance_running(5.0)
        assert rig.heater.status().faults == ("over_temperature",)
        assert not rig.on()

    def test_a_sensor_that_fails_is_tolerated_for_a_while_and_then_stops_the_heating(self) -> None:
        rig = make(
            temperature_c=0.0,
            humidity_pct=100.0,
            margin_c=5.0,
            sensor_interval_s=5.0,
            sensor_stale_s=60.0,
        )
        rig.heater.start()
        rig.advance_running(10.0)
        rig.ambient.fail()
        rig.advance_running(50.0)  # inside the 60 s of grace
        assert rig.on()
        assert rig.heater.status().faults == ()
        rig.advance_running(30.0)
        assert not rig.on()
        assert rig.heater.status().faults == ("sensor_stale",)
        rig.ambient.fail(False)
        rig.advance_running(35.0)
        assert rig.on()
        assert rig.heater.status().faults == ()
        assert rig.kinds() == ["heater.sensor_stale", "heater.fault_cleared"]

    def test_a_reading_outside_the_sensible_range_counts_as_a_failed_read(self) -> None:
        rig = make(sensor_stale_s=10.0, sensor_interval_s=1.0)
        rig.ambient.set(temperature_c=-127.0)  # a typical error code of a sensor
        rig.heater.start()
        rig.advance_running(15.0)
        assert rig.heater.status().faults == ("sensor_stale",)

    def test_a_missing_humidity_counts_as_a_failed_read(self) -> None:
        rig = make(sensor_stale_s=10.0, sensor_interval_s=1.0)
        rig.ambient.humidity_pct = None
        rig.heater.start()
        rig.advance_running(15.0)
        assert rig.heater.status().faults == ("sensor_stale",)

    def test_the_heater_stays_off_until_the_first_valid_reading(self) -> None:
        rig = make()
        rig.ambient.fail()
        rig.heater.start()
        rig.advance_running(5.0)
        assert not any(value for _, _, value in rig.io.history)

    def test_a_failing_optics_sensor_stops_the_heating_too(self) -> None:
        rig = make(optics_c=2.0, sensor_stale_s=20.0, sensor_interval_s=1.0)
        rig.heater.start()
        rig.advance_running(5.0)
        assert rig.on()
        assert rig.optics is not None
        rig.optics.fail()
        rig.advance_running(30.0)
        assert rig.heater.status().faults == ("sensor_stale",)
        assert not rig.on()

    def test_an_asserted_fault_input_stops_the_heating_until_it_clears(self) -> None:
        clock = VirtualClock()
        io = FakeIo(clock, outputs=["heater"])
        pins = {**PINS, "fault": PinSpec(chip="gpiochip0", line=27, direction="input")}
        heater = HeaterController(
            config(pins=pins, fault_input="fault"),
            io=io,
            clock=clock,
            ambient=FakeEnvSensor(clock, temperature_c=0.0, humidity_pct=100.0),
        )
        heater.start()
        heater.step()
        assert output_on(io)
        io.set_input("fault", True)
        heater.step()
        assert not output_on(io)
        assert heater.status().faults == ("fault_input",)
        io.set_input("fault", False)
        heater.step()
        heater.step()
        assert output_on(io)
        assert heater.status().faults == ()

    def test_an_unreadable_fault_line_counts_as_a_fault(self) -> None:
        clock = VirtualClock()
        io = FakeIo(clock, outputs=["heater"])
        pins = {**PINS, "fault": PinSpec(chip="gpiochip0", line=27, direction="input")}
        heater = HeaterController(
            config(pins=pins, fault_input="fault"),
            io=io,
            clock=clock,
            ambient=FakeEnvSensor(clock, temperature_c=0.0, humidity_pct=100.0),
        )
        heater.start()
        io.fail_next()  # the read of the fault line fails
        heater.step()
        assert heater.status().faults == ("fault_input",)
        assert not output_on(io)

    def test_a_failing_switch_on_is_a_fault_and_the_heater_stays_off(self) -> None:
        rig = make(temperature_c=0.0, humidity_pct=100.0, margin_c=5.0)
        rig.heater.start()
        rig.io.fail_next()
        rig.heater.step()
        assert not rig.on()
        assert rig.heater.status().faults == ("io_error",)
        rig.heater.step()  # the next step retries the output and clears the fault
        rig.heater.step()
        assert rig.heater.status().faults == ()
        assert rig.on()

    def test_a_failing_switch_off_is_retried(self) -> None:
        rig = make(temperature_c=0.0, humidity_pct=100.0, margin_c=5.0, sensor_interval_s=1.0)
        rig.heater.start()
        rig.heater.step()
        assert rig.on()
        rig.ambient.set(temperature_c=41.0)  # the cutoff
        rig.clock.advance(1.0)
        rig.io.fail_next()  # the switch-off fails
        rig.heater.step()
        assert rig.on()  # still on, and the controller knows it
        rig.clock.advance(1.0)
        rig.heater.step()
        assert not rig.on()

    def test_a_failing_step_in_run_switches_the_heater_off_and_carries_on(self) -> None:
        rig = make(temperature_c=0.0, humidity_pct=100.0, margin_c=5.0, sensor_interval_s=1.0)

        class Exploding(FakeEnvSensor):
            def read(self):  # type: ignore[no-untyped-def]
                raise RuntimeError("an adapter bug")

        # An adapter that raises is treated as a failed read, so the loop never sees it.
        rig.heater._ambient = Exploding(rig.clock)
        rig.run(200.0)
        assert not rig.on()
        assert rig.heater.status().faults == ("sensor_stale",)

    def test_the_keepalive_toggles_while_healthy_and_stops_on_a_fault(self) -> None:
        clock = VirtualClock()
        io = FakeIo(clock, outputs=["heater", "keepalive"])
        pins = {**PINS, "keepalive": PinSpec(chip="gpiochip0", line=23)}
        sensor = FakeEnvSensor(clock, temperature_c=0.0, humidity_pct=100.0)
        heater = HeaterController(
            config(pins=pins, keepalive="keepalive"), io=io, clock=clock, ambient=sensor
        )
        heater.start()
        for _ in range(4):
            heater.step()
        levels = [value for _, name, value in io.history if name == "keepalive"]
        assert levels[:4] == [True, False, True, False]
        sensor.set(temperature_c=41.0)
        clock.advance(10.0)
        heater.step()
        assert io.values["keepalive"] is False
        heater.step()
        assert io.values["keepalive"] is False


class TestLog:
    def test_the_mean_duty_of_a_window_is_the_share_of_time_on(self) -> None:
        rig = make(temperature_c=0.0, humidity_pct=100.0, margin_c=3.0)  # duty 0.75
        origin = rig.clock.utc_ns()
        rig.run(300.0)
        assert rig.duty(0, 300, origin) == pytest.approx(0.75, abs=0.001)
        assert rig.duty(0, 22.5, origin) == pytest.approx(1.0)  # the first pulse
        assert rig.duty(22.5, 30.0, origin) == pytest.approx(0.0)  # the gap
        assert rig.duty(15, 45, origin) == pytest.approx((7.5 + 15.0) / 30.0)

    def test_a_window_before_the_controller_existed_has_no_duty(self) -> None:
        rig = make()
        origin = rig.clock.utc_ns()
        rig.run(60.0)
        assert rig.duty(-10, 30, origin) is None

    def test_an_empty_or_future_window_has_no_duty(self) -> None:
        rig = make()
        origin = rig.clock.utc_ns()
        rig.run(60.0)
        assert rig.duty(30, 30, origin) is None
        assert rig.duty(40, 20, origin) is None
        assert rig.duty(120, 180, origin) is None

    def test_a_window_that_ends_in_the_future_counts_up_to_now(self) -> None:
        rig = make(temperature_c=0.0, humidity_pct=100.0, margin_c=5.0)  # always on
        origin = rig.clock.utc_ns()
        rig.run(60.0)
        assert rig.duty(30, 600, origin) == pytest.approx(1.0)

    def test_a_heater_that_stopped_is_off_in_later_windows(self) -> None:
        rig = make(temperature_c=0.0, humidity_pct=100.0, margin_c=5.0)
        origin = rig.clock.utc_ns()
        rig.run(60.0)
        rig.clock.advance(60.0)
        assert rig.duty(0, 60, origin) == pytest.approx(1.0)
        assert rig.duty(60, 120, origin) == pytest.approx(0.0)

    def test_recent_duty_looks_back_from_now(self) -> None:
        rig = make(temperature_c=0.0, humidity_pct=100.0, margin_c=3.0)  # duty 0.75
        rig.heater.start()
        rig.advance_running(60.0)
        assert rig.heater.recent_duty(60.0) == pytest.approx(0.75, abs=0.001)
        assert rig.heater.recent_duty(10.0) is not None
        assert rig.heater.recent_duty(3600.0) is None  # the log starts at construction

    def test_a_wall_clock_step_back_keeps_the_log_in_order(self) -> None:
        rig = make(temperature_c=0.0, humidity_pct=100.0, margin_c=5.0)
        origin = rig.clock.utc_ns()
        rig.heater.start()
        rig.heater.step()
        rig.clock.step_utc_ns(-3600 * NS_PER_S)
        rig.advance_running(30.0)
        rig.heater.stop()
        times = [t for t, _ in rig.heater._log]
        assert times == sorted(times)  # a step of the wall clock never reorders the log
        assert rig.duty(-3600, -3590, origin) is None  # before the log, so no value


class TestConfig:
    def test_the_defaults_are_a_disabled_heater(self) -> None:
        defaults = HeaterConfig()
        assert not defaults.enabled
        assert defaults.margin_c == 3.0
        assert defaults.pins == {}

    @pytest.mark.parametrize(
        "values",
        [
            {"margin_c": -1},
            {"band_c": 0},
            {"max_duty": 1.5},
            {"period_s": 0},
            {"min_pulse_s": 20.0},  # at least half of the period
            {"unknown": 1},
        ],
    )
    def test_invalid_values_are_refused(self, values: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            HeaterConfig(**values)

    def test_an_enabled_heater_needs_its_pin_and_an_ambient_sensor(self) -> None:
        with pytest.raises(ValidationError, match="pins has no entry"):
            HeaterConfig(enabled=True, ambient=SensorConfig(kind="fixed", temperature_c=1.0))
        with pytest.raises(ValidationError, match="ambient sensor"):
            HeaterConfig(enabled=True, pins=PINS)

    def test_the_named_pins_must_have_the_right_direction(self) -> None:
        pins = {"heater": PinSpec(chip="gpiochip0", line=1, direction="input")}
        with pytest.raises(ValidationError, match="must be an output"):
            config(pins=pins)
        with pytest.raises(ValidationError, match="must be an input"):
            config(fault_input="heater")

    def test_a_disabled_heater_needs_no_pins(self) -> None:
        HeaterConfig(enabled=False)


class TestCreate:
    def test_a_disabled_heater_gets_no_gpio_and_no_sensors(self) -> None:
        clock = VirtualClock()
        heater = create_heater(HeaterConfig(), clock=clock)
        heater.start()
        heater.step()
        assert heater.status().state == "disabled"

    def test_an_enabled_heater_builds_its_sensors_from_the_configuration(self) -> None:
        clock = VirtualClock()
        io = FakeIo(clock, outputs=["heater"])
        heater = create_heater(config(), clock=clock, io=io)
        heater.start()
        heater.step()
        assert heater.status().ambient_c == 5.0
        assert output_on(io)

    def test_the_replacements_win(self) -> None:
        clock = VirtualClock()
        sensor = FakeEnvSensor(clock, temperature_c=30.0, humidity_pct=30.0)
        heater = create_heater(config(), clock=clock, io=NullIo(), ambient=sensor)
        heater.start()
        heater.step()
        assert heater.status().ambient_c == 30.0
        assert heater.status().state == "off"

    def test_an_enabled_heater_without_a_gpio_library_fails_loudly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("seeingmon.hardware.io._is_linux", lambda: False)
        with pytest.raises(IoError, match="needs Linux"):
            create_heater(config(), clock=VirtualClock())
