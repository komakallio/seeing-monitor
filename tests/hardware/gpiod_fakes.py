"""Stand-ins for libgpiod 1.x and 2.x: Python functions with the C signatures.

Each fake keeps the state of the lines it grants, so a test checks the calls that `LibgpiodIo`
makes (the direction, the initial level, the consumer, the release) and the levels that result.
A level lives in `levels`, keyed by (device path, line offset), and `wires` connects an input to an
output so that a loopback test reads back what it wrote. Every function is a plain Python
function, so the code under test can set `argtypes` and `restype` on it.
"""

from __future__ import annotations

import ctypes
from typing import Any


class _FakeGpiod:
    def __init__(self) -> None:
        self.levels: dict[tuple[str, int], int] = {}
        self.wires: dict[tuple[str, int], tuple[str, int]] = {}  # input line -> output line
        self.fail: set[str] = set()  # function names that report a failure
        self.errno = 0  # the errno that a failing function leaves, as the C library does
        self.log: list[str] = []
        self.opened: list[str] = []
        self.closed_chips = 0
        self.released = 0
        self.consumers: list[str] = []
        self._next = 100

    def _handle(self) -> int:
        self._next += 1
        return self._next

    def _level(self, key: tuple[str, int]) -> int:
        return self.levels.get(self.wires.get(key, key), 0)

    def _define(self, name: str, function: Any) -> None:
        def called(*args: Any) -> Any:
            self.log.append(name)
            if name in self.fail:
                ctypes.set_errno(self.errno)
                return None if name.endswith(("_new", "_open", "request_lines", "get_line")) else -1
            return function(*args)

        setattr(self, name, called)


class FakeGpiodV2(_FakeGpiod):
    """libgpiod 2.x. A request holds one line here, which is how `LibgpiodIo` uses it."""

    def __init__(self) -> None:
        super().__init__()
        self.chips: dict[int, str] = {}
        self.settings: dict[int, dict[str, int]] = {}
        self.line_configs: dict[int, list[tuple[list[int], int]]] = {}
        self.request_configs: dict[int, str] = {}
        self.requests: dict[int, tuple[str, int]] = {}
        self.freed: list[str] = []
        define = self._define

        def chip_open(path: bytes) -> int:
            handle = self._handle()
            self.chips[handle] = path.decode()
            self.opened.append(path.decode())
            return handle

        def chip_close(handle: int) -> None:
            self.closed_chips += 1

        def settings_new() -> int:
            handle = self._handle()
            self.settings[handle] = {}
            return handle

        def set_value(key: str) -> Any:
            def setter(handle: int, value: int) -> int:
                self.settings[handle][key] = value
                return 0

            return setter

        def config_new() -> int:
            handle = self._handle()
            self.line_configs[handle] = []
            return handle

        def add_line_settings(config: int, offsets: Any, count: int, settings: int) -> int:
            self.line_configs[config].append((list(offsets)[:count], settings))
            return 0

        def request_config_new() -> int:
            handle = self._handle()
            self.request_configs[handle] = ""
            return handle

        def set_consumer(config: int, consumer: bytes) -> None:
            self.request_configs[config] = consumer.decode()

        def request_lines(chip: int, request_config: int, line_config: int) -> int:
            offsets, settings_handle = self.line_configs[line_config][0]
            settings = self.settings[settings_handle]
            path = self.chips[chip]
            handle = self._handle()
            self.requests[handle] = (path, offsets[0])
            self.consumers.append(self.request_configs[request_config])
            if settings["direction"] == 3:  # output: the initial level
                self.levels[(path, offsets[0])] = settings.get("output_value", 0)
            self.settings[handle] = dict(settings)  # remember how the line was requested
            return handle

        def request_release(handle: int) -> None:
            self.released += 1

        def request_set_value(handle: int, offset: int, value: int) -> int:
            self.levels[(self.requests[handle][0], offset)] = value
            return 0

        def request_get_value(handle: int, offset: int) -> int:
            return self._level((self.requests[handle][0], offset))

        define("gpiod_chip_open", chip_open)
        define("gpiod_chip_close", chip_close)
        define("gpiod_line_settings_new", settings_new)
        define("gpiod_line_settings_free", lambda handle: self.freed.append("settings"))
        define("gpiod_line_settings_set_direction", set_value("direction"))
        define("gpiod_line_settings_set_output_value", set_value("output_value"))
        define("gpiod_line_settings_set_bias", set_value("bias"))
        define("gpiod_line_config_new", config_new)
        define("gpiod_line_config_free", lambda handle: self.freed.append("line_config"))
        define("gpiod_line_config_add_line_settings", add_line_settings)
        define("gpiod_request_config_new", request_config_new)
        define("gpiod_request_config_free", lambda handle: self.freed.append("request_config"))
        define("gpiod_request_config_set_consumer", set_consumer)
        define("gpiod_chip_request_lines", request_lines)
        define("gpiod_line_request_release", request_release)
        define("gpiod_line_request_set_value", request_set_value)
        define("gpiod_line_request_get_value", request_get_value)


class FakeGpiodV1(_FakeGpiod):
    """libgpiod 1.x: a line object per offset."""

    def __init__(self) -> None:
        super().__init__()
        self.chips: dict[int, str] = {}
        self.lines: dict[int, tuple[str, int]] = {}
        self.flags: list[int] = []
        define = self._define

        def chip_open(path: bytes) -> int:
            handle = self._handle()
            self.chips[handle] = path.decode()
            self.opened.append(path.decode())
            return handle

        def chip_close(handle: int) -> None:
            self.closed_chips += 1

        def get_line(chip: int, offset: int) -> int:
            handle = self._handle()
            self.lines[handle] = (self.chips[chip], offset)
            return handle

        def request_output(line: int, consumer: bytes, default: int) -> int:
            self.consumers.append(consumer.decode())
            self.levels[self.lines[line]] = default
            return 0

        def request_input_flags(line: int, consumer: bytes, flags: int) -> int:
            self.consumers.append(consumer.decode())
            self.flags.append(flags)
            return 0

        def set_value(line: int, value: int) -> int:
            self.levels[self.lines[line]] = value
            return 0

        def line_release(line: int) -> None:
            self.released += 1

        define("gpiod_chip_open", chip_open)
        define("gpiod_chip_close", chip_close)
        define("gpiod_chip_get_line", get_line)
        define("gpiod_line_request_output", request_output)
        define("gpiod_line_request_input_flags", request_input_flags)
        define("gpiod_line_set_value", set_value)
        define("gpiod_line_get_value", lambda line: self._level(self.lines[line]))
        define("gpiod_line_release", line_release)
