"""The dew-heater controller.

A fixed camera behind a telescope collects dew on its optics, and dew ruins the frames. A small
heater on the dew shield or lens cell keeps the glass a margin above the dew point. The heater also
creates convection that can add image motion, so the controller heats as little as it can, logs the
duty so that every seeing window can carry it, and defaults to off.

**Control.** The controller computes the dew point from the ambient temperature and humidity (the
Magnus formula), and it aims at `dew point + margin_c`. The error is the target minus a reference
temperature: the optics sensor when you configure one, or the ambient temperature otherwise. A
proportional band turns the error into a duty from 0 to `max_duty`: no error or a negative error
gives 0, and an error of `band_c` or more gives full duty. With an optics sensor, an optional
integral term (`integral_time_s`) removes the steady-state error.

**Time-proportional output.** The output is a slow pulse, not a fast one. Each `period_s` the
controller computes a duty and switches the heater on for that share of the period, then off. A
pulse shorter than `min_pulse_s` is skipped, and a gap shorter than `min_pulse_s` is filled, so a
relay never sees a short pulse. The controller reads the sensors every `sensor_interval_s`.

**Safety.** The heater is off at construction, after `stop`, and after `close`. The controller
switches it off at once, and it reports an event, in these cases:

- *Over-temperature.* A temperature above `over_temp_c` latches a cutoff until every temperature
  falls `over_temp_hysteresis_c` below it.
- *Stale sensors.* A sensor that gives no valid reading for `sensor_stale_s` stops the heating.
  Off is the safe state for the equipment, and the dew risk shows in the data flags.
- *Fault input.* A line that you name in `fault_input` (a HAT's own fault output) stops the heating
  while it is asserted.
- *Output failure.* A failing GPIO write stops the heating, and the controller retries to switch
  the output off.

If your HAT has a watchdog that cuts the heater when a heartbeat stops, name an output in
`keepalive`. The controller toggles it at each step while it runs without a fault, and it stops
toggling on a fault, so the HAT's own watchdog also cuts the heater.

**Threads and time.** A worker thread calls `run`, or any caller calls `step` at its own pace.
`status` and `mean_duty` are safe to call from other threads. All timing goes through the `Clock`,
so a test runs a night in seconds.

**Not verified on hardware.** The control constants (`margin_c`, `band_c`, the period) are
placeholders that suit a small dew shield. Commissioning tunes them (phase 3). The real HAT is
undecided: the pin map, the sensors, and the failsafe come from the owner's configuration.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

from pydantic import Field, model_validator

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.config import SectionModel
from seeingmon.hardware.events import EventCallback, HardwareEvent, emit
from seeingmon.hardware.io import (
    EnvReading,
    EnvSensor,
    Io,
    IoError,
    LibgpiodIo,
    NullIo,
    PinSpec,
    SensorConfig,
    build_sensor,
)

_log = logging.getLogger(__name__)

# The Magnus constants of Alduchov and Eskridge (1996), for the range -40 to 50 degrees Celsius.
MAGNUS_B = 17.625
MAGNUS_C = 243.04
MIN_TEMPERATURE_C = -60.0  # a reading outside the sensible range is a sensor fault
MAX_TEMPERATURE_C = 80.0
MIN_STEP_S = 0.001  # the shortest delay that `step` asks for, so a loop never spins


def dew_point_c(temperature_c: float, humidity_pct: float) -> float:
    """The dew point in degrees Celsius, from the Magnus formula.

    The humidity is a percentage. A humidity below 1% counts as 1%, because the dew point of dry
    air is far below any temperature that matters here. Raises `ValueError` for a temperature
    outside -60 to 80 degrees or a humidity outside 0 to 100%.
    """
    if not MIN_TEMPERATURE_C <= temperature_c <= MAX_TEMPERATURE_C:
        raise ValueError(f"the temperature is outside the sensible range: {temperature_c}")
    if not 0.0 <= humidity_pct <= 100.0:
        raise ValueError(f"the humidity must be 0 to 100 percent: {humidity_pct}")
    humidity = max(humidity_pct, 1.0) / 100.0
    gamma = math.log(humidity) + MAGNUS_B * temperature_c / (MAGNUS_C + temperature_c)
    return MAGNUS_C * gamma / (MAGNUS_B - gamma)


class HeaterConfig(SectionModel):
    """The `[heater]` section. The heater stays off unless you set `enabled`.

    `pins` maps logical names to lines (see `seeingmon.hardware.io.PinSpec`). `output` names the
    line that switches the heater, `keepalive` names an optional watchdog output, and
    `fault_input` names an optional fault input. `ambient` describes the air sensor (temperature
    and humidity), and `optics` describes the optional sensor on the dew shield or lens cell.
    """

    enabled: bool = False
    output: str = Field(default="heater", min_length=1)
    keepalive: str | None = None
    fault_input: str | None = None
    margin_c: float = Field(default=3.0, ge=0.0, le=20.0)
    band_c: float = Field(default=4.0, gt=0.0)
    integral_time_s: float = Field(default=0.0, ge=0.0)
    period_s: float = Field(default=30.0, gt=0.0)
    max_duty: float = Field(default=1.0, gt=0.0, le=1.0)
    min_pulse_s: float = Field(default=1.0, ge=0.0)
    over_temp_c: float = 40.0
    over_temp_hysteresis_c: float = Field(default=2.0, ge=0.0)
    sensor_interval_s: float = Field(default=5.0, gt=0.0)
    sensor_stale_s: float = Field(default=120.0, gt=0.0)
    tick_s: float = Field(default=1.0, gt=0.0)
    history_h: float = Field(default=48.0, gt=0.0)
    pins: dict[str, PinSpec] = Field(default_factory=dict)
    ambient: SensorConfig = Field(default_factory=SensorConfig)
    optics: SensorConfig = Field(default_factory=SensorConfig)

    @model_validator(mode="after")
    def _check_enabled_heater(self) -> HeaterConfig:
        if 2 * self.min_pulse_s >= self.period_s:
            raise ValueError("min_pulse_s must be less than half of period_s")
        if not self.enabled:
            return self
        wanted = [(self.output, "output"), (self.keepalive, "output"), (self.fault_input, "input")]
        for name, direction in wanted:
            if name is None:
                continue
            pin = self.pins.get(name)
            if pin is None:
                raise ValueError(f"pins has no entry for {name!r}")
            if pin.direction != direction:
                raise ValueError(f"pin {name!r} must be an {direction}")
        if self.ambient.kind == "none":
            raise ValueError("an enabled heater needs an ambient sensor")
        return self


@dataclass(frozen=True, slots=True)
class HeaterStatus:
    """What the controller reports to the health record and the API.

    `state` is `disabled`, `stopped`, `off` (running with no heating), `heating`, or `fault`.
    `duty` is the share of the current period that the heater is on, after the pulse rule.
    """

    state: str
    enabled: bool
    running: bool
    output_on: bool
    duty: float
    dew_point_c: float | None
    target_c: float | None
    ambient_c: float | None
    humidity_pct: float | None
    optics_c: float | None
    faults: tuple[str, ...]
    t_utc_ns: int


class HeaterController:
    """A time-proportional dew-heater controller.

    Args:
        config: The `[heater]` section.
        io: The lines. The constructor switches the output off.
        clock: The time source.
        ambient: The air sensor. A disabled controller needs none.
        optics: The optional optics sensor. It gives closed-loop control and a second
            over-temperature check.
        on_event: Receives faults and their clearing.
    """

    def __init__(
        self,
        config: HeaterConfig,
        *,
        io: Io,
        clock: Clock,
        ambient: EnvSensor | None = None,
        optics: EnvSensor | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self._cfg = config
        self._io = io
        self._clock = clock
        self._ambient = ambient
        self._optics = optics
        self._on_event = on_event
        self._lock = threading.RLock()
        self._running = False
        self._output_on = False
        self._duty = 0.0
        self._period_start_ns: int | None = None
        self._period_end_ns = 0
        self._on_ns = 0
        self._integral = 0.0
        self._last_sensor_ns: int | None = None
        self._ambient_reading: EnvReading | None = None
        self._optics_reading: EnvReading | None = None
        self._ambient_valid_ns: int | None = None
        self._optics_valid_ns: int | None = None
        self._faults: set[str] = set()
        self._over_temp_latched = False
        self._keepalive_level = False
        self._dew_point: float | None = None
        self._target: float | None = None
        history = round(config.history_h * 3600 * 2 / max(config.period_s, 1.0)) + 64
        self._log: deque[tuple[int, bool]] = deque(maxlen=max(history, 1024))
        self._log.append((clock.utc_ns(), False))
        self._set_output(False, force=True)  # off at construction, whatever the line held before

    # --- Output and log ---

    def _set_output(self, on: bool, *, force: bool = False) -> bool:
        """Switch the heater. Returns whether the output is now in the wanted state.

        The call skips the GPIO write when the output already has that state, unless you `force`.
        """
        if on == self._output_on and not force:
            return True
        try:
            self._io.set_output(self._cfg.output, on)
        except IoError:
            self._raise_fault("io_error", "The heater output could not be switched.")
            return False
        self._clear_fault("io_error")
        if on != self._output_on or not self._log:
            self._output_on = on
            now = self._clock.utc_ns()
            if self._log and now < self._log[-1][0]:
                now = self._log[-1][0]  # the wall clock stepped back, so keep the log in order
            self._log.append((now, on))
        return True

    def _force_off(self, *, force: bool = False) -> None:
        self._duty = 0.0
        self._on_ns = 0
        self._set_output(False, force=force)

    def _toggle_keepalive(self, healthy: bool) -> None:
        name = self._cfg.keepalive
        if name is None:
            return
        self._keepalive_level = (not self._keepalive_level) if healthy else False
        try:
            self._io.set_output(name, self._keepalive_level)
        except IoError:
            _log.debug("the keepalive output failed", exc_info=True)

    # --- Faults ---

    def _emit(self, level: str, kind: str, message: str, **detail: object) -> None:
        event = HardwareEvent(level, kind, message, self._clock.utc_ns(), detail or None)
        emit(self._on_event, event)

    def _raise_fault(self, fault: str, message: str) -> None:
        if fault not in self._faults:
            self._faults.add(fault)
            self._emit("warning", f"heater.{fault}", message)

    def _clear_fault(self, fault: str) -> None:
        if fault in self._faults:
            self._faults.discard(fault)
            self._emit("info", "heater.fault_cleared", "A heater fault cleared.", fault=fault)

    def _set_fault(self, fault: str, active: bool, message: str) -> None:
        if active:
            self._raise_fault(fault, message)
        else:
            self._clear_fault(fault)

    # --- Sensors ---

    def _read(self, sensor: EnvSensor) -> EnvReading | None:
        try:
            return sensor.read()
        except Exception:  # a sensor adapter is third-party code, so any failure means no reading
            _log.debug("a sensor read failed", exc_info=True)
            return None

    def _read_sensors(self, now_ns: int) -> None:
        self._last_sensor_ns = now_ns
        if self._ambient is not None:
            reading = self._read(self._ambient)
            if (
                reading is not None
                and reading.temperature_c is not None
                and reading.humidity_pct is not None
                and MIN_TEMPERATURE_C <= reading.temperature_c <= MAX_TEMPERATURE_C
                and 0.0 <= reading.humidity_pct <= 100.0
            ):
                self._ambient_reading = reading
                self._ambient_valid_ns = now_ns
        if self._optics is not None:
            reading = self._read(self._optics)
            if (
                reading is not None
                and reading.temperature_c is not None
                and MIN_TEMPERATURE_C <= reading.temperature_c <= MAX_TEMPERATURE_C
            ):
                self._optics_reading = reading
                self._optics_valid_ns = now_ns

    def _stale(self, valid_ns: int | None, now_ns: int) -> bool:
        return valid_ns is None or now_ns - valid_ns > round(self._cfg.sensor_stale_s * NS_PER_S)

    def _temperatures(self) -> list[float]:
        found: list[float] = []
        if self._ambient_reading is not None and self._ambient_reading.temperature_c is not None:
            found.append(self._ambient_reading.temperature_c)
        if self._optics_reading is not None and self._optics_reading.temperature_c is not None:
            found.append(self._optics_reading.temperature_c)
        return found

    def _update_over_temperature(self) -> None:
        temperatures = self._temperatures()
        limit = self._cfg.over_temp_c
        if any(value > limit for value in temperatures):
            self._over_temp_latched = True
        elif self._over_temp_latched and all(
            value < limit - self._cfg.over_temp_hysteresis_c for value in temperatures
        ):
            self._over_temp_latched = False
        self._set_fault(
            "over_temperature",
            self._over_temp_latched,
            "A temperature passed the limit, so the heater is off.",
        )

    # --- Control ---

    def _start_period(
        self, now_ns: int, temperature_c: float, humidity_pct: float, optics_c: float | None
    ) -> None:
        """Compute the duty for a new period from the ambient air and the optional optics."""
        cfg = self._cfg
        self._dew_point = dew_point_c(temperature_c, humidity_pct)
        self._target = self._dew_point + cfg.margin_c
        closed_loop = optics_c is not None
        reference = optics_c if optics_c is not None else temperature_c
        error = self._target - reference
        period_s = cfg.period_s
        if closed_loop and cfg.integral_time_s > 0:
            previous = self._period_start_ns
            elapsed_s = 0.0 if previous is None else (now_ns - previous) / NS_PER_S
            ceiling = cfg.max_duty * cfg.band_c * cfg.integral_time_s
            self._integral = min(
                max(self._integral + error * min(elapsed_s, 2 * period_s), 0.0), ceiling
            )
            integral_term = self._integral / (cfg.integral_time_s * cfg.band_c)
        else:
            self._integral = 0.0
            integral_term = 0.0
        duty = min(max(error / cfg.band_c + integral_term, 0.0), cfg.max_duty)
        period_ns = round(period_s * NS_PER_S)
        on_ns = round(duty * period_ns)
        pulse_ns = round(cfg.min_pulse_s * NS_PER_S)
        if on_ns < pulse_ns:
            on_ns = 0
        elif period_ns - on_ns < pulse_ns:
            on_ns = period_ns
        self._period_start_ns = now_ns
        self._period_end_ns = now_ns + period_ns
        self._on_ns = on_ns
        self._duty = on_ns / period_ns

    def step(self) -> float:
        """Run one control step. Returns the seconds until the next step that does something.

        The delay never exceeds `tick_s` and never falls below a millisecond. Call `step` from a
        loop that sleeps on the clock for the returned time, or use `run`.
        """
        with self._lock:
            now = self._clock.monotonic_ns()
            tick = self._cfg.tick_s
            if not self._running:
                self._force_off()
                return tick
            self._check_fault_input()
            if self._last_sensor_ns is None or now - self._last_sensor_ns >= round(
                self._cfg.sensor_interval_s * NS_PER_S
            ):
                self._read_sensors(now)
            self._set_fault(
                "sensor_stale",
                self._stale(self._ambient_valid_ns, now)
                or (self._optics is not None and self._stale(self._optics_valid_ns, now)),
                "A sensor gave no valid reading for too long, so the heater is off.",
            )
            self._update_over_temperature()
            if self._faults:
                self._force_off(force="io_error" in self._faults)
                self._toggle_keepalive(healthy=False)
                self._period_end_ns = 0  # start a new period when the fault clears
                return tick
            if now >= self._period_end_ns:
                ambient = self._ambient_reading
                if ambient is None or ambient.temperature_c is None or ambient.humidity_pct is None:
                    self._force_off()  # no reading to control from, which the stale fault covers
                    return tick
                optics = self._optics_reading
                self._start_period(
                    now,
                    ambient.temperature_c,
                    ambient.humidity_pct,
                    None if optics is None else optics.temperature_c,
                )
            started = self._period_start_ns if self._period_start_ns is not None else now
            self._set_output(now - started < self._on_ns)
            self._toggle_keepalive(healthy=True)
            next_event = self._period_end_ns
            if self._output_on:
                next_event = min(next_event, started + self._on_ns)
            return min(max((next_event - now) / NS_PER_S, MIN_STEP_S), tick)

    def _check_fault_input(self) -> None:
        name = self._cfg.fault_input
        if name is None:
            return
        try:
            asserted = self._io.read_input(name)
        except IoError:
            asserted = True  # a fault line that cannot be read is a fault
        self._set_fault(
            "fault_input", asserted, "The heater hardware reports a fault, so the heater is off."
        )

    # --- Lifecycle ---

    def start(self) -> None:
        """Begin to control the heater. Does nothing for a disabled heater."""
        with self._lock:
            if not self._cfg.enabled:
                return
            self._running = True
            self._period_end_ns = 0

    def stop(self) -> None:
        """Switch the heater off and stop controlling it."""
        with self._lock:
            self._running = False
            self._force_off(force=True)
            self._toggle_keepalive(healthy=False)

    def close(self) -> None:
        """Stop, and release the lines."""
        self.stop()
        with self._lock:
            self._io.close()

    def run(self, should_stop: Callable[[], bool]) -> None:
        """Control the heater until `should_stop()` returns true, and then switch it off.

        A failure of `step` switches the heater off and the loop carries on, so a faulty sensor
        adapter never stops the control loop or leaves the heater on.
        """
        self.start()
        try:
            while not should_stop():
                try:
                    delay = self.step()
                except Exception:
                    _log.exception("the heater step failed, so the heater goes off")
                    with self._lock:
                        self._force_off()
                    delay = self._cfg.tick_s
                self._clock.sleep(delay)
        finally:
            self.stop()

    # --- What the controller reports ---

    def mean_duty(self, t_start_ns: int, t_end_ns: int) -> float | None:
        """The share of `[t_start_ns, t_end_ns]` (UTC) that the heater was on, from 0 to 1.

        Returns `None` when the log does not cover the window: the window starts before the
        controller existed, it is empty, or it lies in the future. A window that ends after now
        counts up to now.
        """
        with self._lock:
            if t_end_ns <= t_start_ns:
                return None
            now = self._clock.utc_ns()
            end = min(t_end_ns, now)
            if end <= t_start_ns or t_start_ns < self._log[0][0]:
                return None
            on_ns = 0
            entries = [*self._log, (max(now, self._log[-1][0]), self._output_on)]
            for (since, on), (until, _) in pairwise(entries):
                if on:
                    on_ns += max(0, min(until, end) - max(since, t_start_ns))
            return on_ns / (end - t_start_ns)

    def recent_duty(self, window_s: float) -> float | None:
        """The share of the last `window_s` seconds that the heater was on, or `None` when the log
        does not reach back that far. Use it to fill `FastContext.heater_duty` for the window
        that closes now."""
        now = self._clock.utc_ns()
        return self.mean_duty(now - round(window_s * NS_PER_S), now)

    def status(self) -> HeaterStatus:
        """A snapshot of the controller for the health record and the API."""
        with self._lock:
            ambient, optics = self._ambient_reading, self._optics_reading
            if not self._cfg.enabled:
                state = "disabled"
            elif self._faults:
                state = "fault"
            elif not self._running:
                state = "stopped"
            else:
                state = "heating" if self._output_on or self._duty > 0 else "off"
            return HeaterStatus(
                state=state,
                enabled=self._cfg.enabled,
                running=self._running,
                output_on=self._output_on,
                duty=self._duty,
                dew_point_c=self._dew_point,
                target_c=self._target,
                ambient_c=None if ambient is None else ambient.temperature_c,
                humidity_pct=None if ambient is None else ambient.humidity_pct,
                optics_c=None if optics is None else optics.temperature_c,
                faults=tuple(sorted(self._faults)),
                t_utc_ns=self._clock.utc_ns(),
            )


def create_heater(
    config: HeaterConfig,
    *,
    clock: Clock,
    io: Io | None = None,
    ambient: EnvSensor | None = None,
    optics: EnvSensor | None = None,
    on_event: EventCallback | None = None,
) -> HeaterController:
    """Build the controller that `config` describes.

    A disabled heater gets no GPIO and no sensors, so it never touches a pin. An enabled heater
    requests its lines through libgpiod and builds its sensors from the configuration. Pass `io`,
    `ambient`, or `optics` to replace any of them, for a test or a different adapter.
    """
    if io is None:
        io = LibgpiodIo(config.pins) if config.enabled else NullIo()
    if config.enabled:
        ambient = ambient or build_sensor(config.ambient, clock)
        optics = optics or build_sensor(config.optics, clock)
    return HeaterController(
        config, io=io, clock=clock, ambient=ambient, optics=optics, on_event=on_event
    )
