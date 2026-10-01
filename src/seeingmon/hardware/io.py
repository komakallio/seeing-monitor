"""GPIO lines and environment sensors behind small interfaces.

The dew heater needs to switch a line and read two kinds of sensor. The real HAT is undecided, so
the heater code talks to two interfaces and knows no board:

- `Io` switches named lines. `set_output(name, value)` drives an output, `read_input(name)` reads
  an input, and `close()` releases every line. A value of `True` means "on": the heater is
  energized, or a fault is asserted. A pin that is active low flips the level in software.
- `EnvSensor` reads the ambient temperature and relative humidity, or the temperature of the
  optics.

**Implementations.** `FakeIo` and `FakeEnvSensor` run everywhere, so the tests need no board.
`LibgpiodIo` drives real lines through the kernel's `/dev/gpiochip` interface with `libgpiod` and
`ctypes`. It works on the Pi 4 and the Pi 5 (`RPi.GPIO` does not work on the Pi 5), and it supports
libgpiod 1.x (Raspberry Pi OS based on Debian 12) and 2.x (Debian 13). The module loads the library
only when you create a `LibgpiodIo`, and it refuses to load it on a platform other than Linux.
`SysfsEnvSensor` reads a sensor through a file that a kernel driver exposes (`hwmon` or
`iio`), which covers the common I2C temperature and humidity sensors without code for each model.

**Names, not pins.** Code refers to a line by a logical name (`heater`), and the configuration maps
the name to a chip and a line offset (`PinSpec`). The pin map is deployment-specific, so it lives in
the local configuration and never in the repository.

**Safety.** `LibgpiodIo` requests an output in the "off" state, so the line never glitches on. It
drives every output off before it releases the lines. A line that nothing holds falls back to the
board's default, so a HAT with its own failsafe should default to off.

**Not verified on hardware.** `LibgpiodIo` is tested against a stand-in for each libgpiod version.
`docs/hardware-checks.md` lists the loopback test that confirms it on a real board.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import Field, model_validator

from seeingmon.clock import Clock
from seeingmon.config import SectionModel

DEFAULT_CONSUMER = "seeingmon"
LIBGPIOD_NAMES = ("libgpiod.so.3", "libgpiod.so.2", "libgpiod.so")


class IoError(Exception):
    """A GPIO call failed, or a name does not exist."""


class SensorError(Exception):
    """A sensor read failed or returned nothing usable."""


# --- Lines -------------------------------------------------------------------------------


@runtime_checkable
class Io(Protocol):
    def set_output(self, name: str, value: bool) -> None:
        """Drive the output called `name`. `True` means on."""
        ...

    def read_input(self, name: str) -> bool:
        """Read the input called `name`. `True` means asserted."""
        ...

    def close(self) -> None:
        """Turn every output off and release the lines. Safe to call twice."""
        ...


class PinSpec(SectionModel):
    """Where a logical line lives: a chip, a line offset, and a direction.

    `chip` is a device name such as `gpiochip0` (found under `/dev`) or an absolute device path.
    The name of the chip that carries the header pins depends on the Pi model and the kernel, so
    check it with `gpiodetect`. `active_low` says that the level is low when the line is on.
    `bias` sets a pull resistor on an input.
    """

    chip: str = Field(min_length=1)
    line: int = Field(ge=0)
    direction: Literal["output", "input"] = "output"
    active_low: bool = False
    bias: Literal["none", "pull_up", "pull_down"] = "none"

    @model_validator(mode="after")
    def _bias_belongs_to_inputs(self) -> PinSpec:
        if self.bias != "none" and self.direction != "input":
            raise ValueError("bias applies only to an input")
        return self

    @property
    def device_path(self) -> str:
        """The path of the chip's device node."""
        if self.chip.startswith("/") or "\\" in self.chip:
            return self.chip
        return f"/dev/{self.chip}"


class NullIo:
    """An `Io` that does nothing. A disabled heater uses it, so it never touches a pin."""

    def set_output(self, name: str, value: bool) -> None:
        return None

    def read_input(self, name: str) -> bool:
        return False

    def close(self) -> None:
        return None


class FakeIo:
    """An `Io` for tests. It records every change and can wire an output to an input.

    `outputs` lists the names that exist. With none listed, any name is accepted as an output.
    `history` holds (monotonic time in ns, name, value) for each call of `set_output`. Use
    `link` for a loopback wire, `set_input` for a scripted input, and `fail_next` for a fault.
    """

    def __init__(self, clock: Clock | None = None, *, outputs: Iterable[str] = ()) -> None:
        self._clock = clock
        self._allowed = frozenset(outputs)
        self.values: dict[str, bool] = dict.fromkeys(self._allowed, False)
        self.history: list[tuple[int, str, bool]] = []
        self._inputs: dict[str, bool] = {}
        self._links: dict[str, str] = {}
        self._failures: list[IoError] = []
        self.closed = False

    def link(self, output: str, input_name: str) -> None:
        """Wire `output` to `input_name`, so the input reads what the output drives."""
        self._links[input_name] = output

    def set_input(self, name: str, value: bool) -> None:
        self._inputs[name] = value

    def fail_next(self, error: IoError | None = None) -> None:
        """Make the next call raise `error`."""
        self._failures.append(error or IoError("scripted GPIO failure"))

    def _check(self) -> None:
        if self._failures:
            raise self._failures.pop(0)
        if self.closed:
            raise IoError("the Io is closed")

    def set_output(self, name: str, value: bool) -> None:
        self._check()
        if self._allowed and name not in self._allowed:
            raise IoError(f"unknown output {name!r}")
        self.values[name] = value
        now = self._clock.monotonic_ns() if self._clock is not None else 0
        self.history.append((now, name, value))

    def read_input(self, name: str) -> bool:
        self._check()
        if name in self._links:
            return self.values.get(self._links[name], False)
        return self._inputs.get(name, False)

    def close(self) -> None:
        if self.closed:
            return
        for name in list(self.values):
            self.values[name] = False
            now = self._clock.monotonic_ns() if self._clock is not None else 0
            self.history.append((now, name, False))
        self.closed = True


# --- libgpiod ----------------------------------------------------------------------------


class _Backend(Protocol):
    """One API generation of libgpiod, reduced to the five operations that `LibgpiodIo` needs."""

    def request(self, spec: PinSpec, physical: bool, consumer: str) -> object: ...
    def write(self, handle: object, spec: PinSpec, physical: bool) -> None: ...
    def read(self, handle: object, spec: PinSpec) -> bool: ...
    def release(self, handle: object) -> None: ...
    def close(self) -> None: ...


class _GpiodV2:
    """libgpiod 2.x: a request object holds the lines, and settings objects describe them."""

    DIRECTION_INPUT = 2
    DIRECTION_OUTPUT = 3
    BIAS: ClassVar[Mapping[str, int]] = {"none": 1, "pull_up": 3, "pull_down": 4}

    def __init__(self, lib: Any) -> None:
        self._lib = lib
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        signatures: dict[str, tuple[Any, list[Any]]] = {
            "gpiod_chip_open": (pointer, [ctypes.c_char_p]),
            "gpiod_chip_close": (None, [pointer]),
            "gpiod_line_settings_new": (pointer, []),
            "gpiod_line_settings_free": (None, [pointer]),
            "gpiod_line_settings_set_direction": (integer, [pointer, integer]),
            "gpiod_line_settings_set_output_value": (integer, [pointer, integer]),
            "gpiod_line_settings_set_bias": (integer, [pointer, integer]),
            "gpiod_line_config_new": (pointer, []),
            "gpiod_line_config_free": (None, [pointer]),
            "gpiod_line_config_add_line_settings": (
                integer,
                [pointer, ctypes.POINTER(ctypes.c_uint), ctypes.c_size_t, pointer],
            ),
            "gpiod_request_config_new": (pointer, []),
            "gpiod_request_config_free": (None, [pointer]),
            "gpiod_request_config_set_consumer": (None, [pointer, ctypes.c_char_p]),
            "gpiod_chip_request_lines": (pointer, [pointer, pointer, pointer]),
            "gpiod_line_request_release": (None, [pointer]),
            "gpiod_line_request_set_value": (integer, [pointer, ctypes.c_uint, integer]),
            "gpiod_line_request_get_value": (integer, [pointer, ctypes.c_uint]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(lib, name)
            function.restype = restype
            function.argtypes = argtypes
        self._chips: dict[str, Any] = {}

    def _chip(self, path: str) -> Any:
        if path not in self._chips:
            chip = self._lib.gpiod_chip_open(path.encode())
            if not chip:
                raise IoError("cannot open a GPIO chip")
            self._chips[path] = chip
        return self._chips[path]

    def request(self, spec: PinSpec, physical: bool, consumer: str) -> object:
        lib = self._lib
        chip = self._chip(spec.device_path)
        settings, line_config, request_config = None, None, None
        try:
            settings = lib.gpiod_line_settings_new()
            line_config = lib.gpiod_line_config_new()
            request_config = lib.gpiod_request_config_new()
            if not (settings and line_config and request_config):
                raise IoError("libgpiod could not allocate a request")
            if spec.direction == "output":
                lib.gpiod_line_settings_set_direction(settings, self.DIRECTION_OUTPUT)
                lib.gpiod_line_settings_set_output_value(settings, int(physical))
            else:
                lib.gpiod_line_settings_set_direction(settings, self.DIRECTION_INPUT)
                lib.gpiod_line_settings_set_bias(settings, self.BIAS[spec.bias])
            offsets = (ctypes.c_uint * 1)(spec.line)
            if lib.gpiod_line_config_add_line_settings(line_config, offsets, 1, settings) != 0:
                raise IoError("libgpiod rejected the line settings")
            lib.gpiod_request_config_set_consumer(request_config, consumer.encode())
            request = lib.gpiod_chip_request_lines(chip, request_config, line_config)
            if not request:
                raise IoError("libgpiod could not request the line")
            return request
        finally:
            for pointer, free in (
                (settings, lib.gpiod_line_settings_free),
                (line_config, lib.gpiod_line_config_free),
                (request_config, lib.gpiod_request_config_free),
            ):
                if pointer:
                    free(pointer)

    def write(self, handle: object, spec: PinSpec, physical: bool) -> None:
        if self._lib.gpiod_line_request_set_value(handle, spec.line, int(physical)) != 0:
            raise IoError("libgpiod could not set the line")

    def read(self, handle: object, spec: PinSpec) -> bool:
        value = self._lib.gpiod_line_request_get_value(handle, spec.line)
        if value < 0:
            raise IoError("libgpiod could not read the line")
        return bool(value)

    def release(self, handle: object) -> None:
        self._lib.gpiod_line_request_release(handle)

    def close(self) -> None:
        for chip in self._chips.values():
            self._lib.gpiod_chip_close(chip)
        self._chips.clear()


class _GpiodV1:
    """libgpiod 1.x: a line object per offset, requested as an input or an output."""

    FLAG_BIAS: ClassVar[Mapping[str, int]] = {"none": 0, "pull_down": 1 << 4, "pull_up": 1 << 5}

    def __init__(self, lib: Any) -> None:
        self._lib = lib
        pointer, integer, text = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
        signatures: dict[str, tuple[Any, list[Any]]] = {
            "gpiod_chip_open": (pointer, [text]),
            "gpiod_chip_close": (None, [pointer]),
            "gpiod_chip_get_line": (pointer, [pointer, ctypes.c_uint]),
            "gpiod_line_request_output": (integer, [pointer, text, integer]),
            "gpiod_line_request_input_flags": (integer, [pointer, text, integer]),
            "gpiod_line_set_value": (integer, [pointer, integer]),
            "gpiod_line_get_value": (integer, [pointer]),
            "gpiod_line_release": (None, [pointer]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(lib, name)
            function.restype = restype
            function.argtypes = argtypes
        self._chips: dict[str, Any] = {}

    def request(self, spec: PinSpec, physical: bool, consumer: str) -> object:
        path = spec.device_path
        if path not in self._chips:
            chip = self._lib.gpiod_chip_open(path.encode())
            if not chip:
                raise IoError("cannot open a GPIO chip")
            self._chips[path] = chip
        line = self._lib.gpiod_chip_get_line(self._chips[path], spec.line)
        if not line:
            raise IoError("libgpiod has no such line")
        if spec.direction == "output":
            status = self._lib.gpiod_line_request_output(line, consumer.encode(), int(physical))
        else:
            flags = self.FLAG_BIAS[spec.bias]
            status = self._lib.gpiod_line_request_input_flags(line, consumer.encode(), flags)
        if status != 0:
            raise IoError("libgpiod could not request the line")
        return line

    def write(self, handle: object, spec: PinSpec, physical: bool) -> None:
        if self._lib.gpiod_line_set_value(handle, int(physical)) != 0:
            raise IoError("libgpiod could not set the line")

    def read(self, handle: object, spec: PinSpec) -> bool:
        value = self._lib.gpiod_line_get_value(handle)
        if value < 0:
            raise IoError("libgpiod could not read the line")
        return bool(value)

    def release(self, handle: object) -> None:
        self._lib.gpiod_line_release(handle)

    def close(self) -> None:
        for chip in self._chips.values():
            self._lib.gpiod_chip_close(chip)
        self._chips.clear()


def _is_linux() -> bool:
    """Whether this is Linux. A function, so the type checker does not fold the platform test."""
    return sys.platform.startswith("linux")


def load_libgpiod(
    library: Any | None = None,
    *,
    path: str | None = None,
    loader: Callable[[str], Any] = ctypes.CDLL,
    find_library: Callable[[str], str | None] = ctypes.util.find_library,
) -> _Backend:
    """Load libgpiod and pick the API generation by the functions that the library exports.

    Pass `library` to use a loaded library (a test passes a stand-in). Otherwise the function
    loads `path`, then the system's `gpiod` library, then the usual file names. It raises
    `IoError` on a platform other than Linux, or when no library loads.
    """
    if library is None:
        if not _is_linux():
            raise IoError("libgpiod needs Linux")
        candidates = [path] if path else [find_library("gpiod"), *LIBGPIOD_NAMES]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                library = loader(candidate)
                break
            except OSError:
                continue
        if library is None:
            raise IoError("libgpiod is not installed; install the libgpiod package of the OS")
    try:
        if hasattr(library, "gpiod_line_settings_new"):
            return _GpiodV2(library)
        return _GpiodV1(library)
    except AttributeError as error:
        raise IoError(f"the libgpiod library lacks a function ({error.name})") from None


class LibgpiodIo:
    """Real GPIO lines through libgpiod.

    The constructor requests every line in `pins`. An output starts off, and an input starts as
    an input. Pass `library` only in a test.
    """

    def __init__(
        self,
        pins: Mapping[str, PinSpec],
        *,
        consumer: str = DEFAULT_CONSUMER,
        library: Any | None = None,
        library_path: str | None = None,
    ) -> None:
        self._pins = dict(pins)
        self._backend = load_libgpiod(library, path=library_path)
        self._handles: dict[str, object] = {}
        self._closed = False
        try:
            for name, spec in self._pins.items():
                physical_off = spec.active_low  # the level of an output that is off
                self._handles[name] = self._backend.request(spec, physical_off, consumer)
        except IoError:
            self.close()
            raise

    def _spec(self, name: str, direction: str) -> PinSpec:
        if self._closed:
            raise IoError("the Io is closed")
        spec = self._pins.get(name)
        if spec is None:
            raise IoError(f"unknown pin {name!r}")
        if spec.direction != direction:
            raise IoError(f"pin {name!r} is not an {direction}")
        return spec

    def set_output(self, name: str, value: bool) -> None:
        spec = self._spec(name, "output")
        self._backend.write(self._handles[name], spec, value != spec.active_low)

    def read_input(self, name: str) -> bool:
        spec = self._spec(name, "input")
        return self._backend.read(self._handles[name], spec) != spec.active_low

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name, spec in self._pins.items():
            handle = self._handles.get(name)
            if handle is None:
                continue
            if spec.direction == "output":
                with contextlib.suppress(IoError):
                    self._backend.write(handle, spec, spec.active_low)  # off, before the release
            self._backend.release(handle)
        self._handles.clear()
        self._backend.close()


# --- Environment sensors -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnvReading:
    """One sensor reading. A sensor leaves out the quantity that it does not measure."""

    t_utc_ns: int
    temperature_c: float | None = None
    humidity_pct: float | None = None


@runtime_checkable
class EnvSensor(Protocol):
    def read(self) -> EnvReading:
        """Read the sensor. Raises `SensorError` when the read fails."""
        ...


class FakeEnvSensor:
    """An `EnvSensor` for tests. Change its values with `set`, and script a fault with `fail`."""

    def __init__(
        self,
        clock: Clock,
        *,
        temperature_c: float | None = 10.0,
        humidity_pct: float | None = 60.0,
    ) -> None:
        self._clock = clock
        self.temperature_c = temperature_c
        self.humidity_pct = humidity_pct
        self.reads = 0
        self._failing = False

    def set(self, *, temperature_c: float | None = None, humidity_pct: float | None = None) -> None:
        if temperature_c is not None:
            self.temperature_c = temperature_c
        if humidity_pct is not None:
            self.humidity_pct = humidity_pct

    def fail(self, failing: bool = True) -> None:
        """Make every read raise `SensorError` until you call `fail(False)`."""
        self._failing = failing

    def read(self) -> EnvReading:
        self.reads += 1
        if self._failing:
            raise SensorError("scripted sensor failure")
        return EnvReading(self._clock.utc_ns(), self.temperature_c, self.humidity_pct)


class SysfsEnvSensor:
    """An `EnvSensor` that reads a number from a file that a kernel driver exposes.

    Linux drivers publish temperature and humidity through `hwmon` (`temp1_input`, in thousandths
    of a degree) and `iio` (`in_temp_input` and `in_humidityrelative_input`, in thousandths). Name
    the files in the configuration, and set the scale to match the driver. Enable the driver with a
    device-tree overlay for the sensor on your board.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        temperature_file: str | None = None,
        humidity_file: str | None = None,
        temperature_scale: float = 0.001,
        humidity_scale: float = 0.001,
    ) -> None:
        if temperature_file is None and humidity_file is None:
            raise ValueError("name a temperature file, a humidity file, or both")
        self._clock = clock
        self._temperature_file = temperature_file
        self._humidity_file = humidity_file
        self._temperature_scale = temperature_scale
        self._humidity_scale = humidity_scale

    @staticmethod
    def _number(path: str, scale: float) -> float:
        try:
            return float(Path(path).read_text(encoding="ascii").strip()) * scale
        except (OSError, ValueError, UnicodeDecodeError):
            raise SensorError("the sensor file could not be read") from None

    def read(self) -> EnvReading:
        temperature = (
            None
            if self._temperature_file is None
            else self._number(self._temperature_file, self._temperature_scale)
        )
        humidity = (
            None
            if self._humidity_file is None
            else self._number(self._humidity_file, self._humidity_scale)
        )
        return EnvReading(self._clock.utc_ns(), temperature, humidity)


class SensorConfig(SectionModel):
    """The configuration of one sensor. `none` means that the role has no sensor.

    `fixed` always returns the values that you give (a dry run, or a simulated station), and
    `sysfs` reads the files that you name.
    """

    kind: Literal["none", "fixed", "sysfs"] = "none"
    temperature_c: float | None = None
    humidity_pct: float | None = None
    temperature_file: str | None = None
    humidity_file: str | None = None
    temperature_scale: float = 0.001
    humidity_scale: float = 0.001

    @model_validator(mode="after")
    def _check_kind(self) -> SensorConfig:
        if self.kind == "fixed" and self.temperature_c is None and self.humidity_pct is None:
            raise ValueError("a fixed sensor needs temperature_c, humidity_pct, or both")
        if self.kind == "sysfs" and self.temperature_file is None and self.humidity_file is None:
            raise ValueError("a sysfs sensor needs temperature_file, humidity_file, or both")
        return self


def build_sensor(config: SensorConfig, clock: Clock) -> EnvSensor | None:
    """The sensor that `config` describes, or `None` for the kind `none`."""
    if config.kind == "fixed":
        return FakeEnvSensor(
            clock, temperature_c=config.temperature_c, humidity_pct=config.humidity_pct
        )
    if config.kind == "sysfs":
        return SysfsEnvSensor(
            clock,
            temperature_file=config.temperature_file,
            humidity_file=config.humidity_file,
            temperature_scale=config.temperature_scale,
            humidity_scale=config.humidity_scale,
        )
    return None
