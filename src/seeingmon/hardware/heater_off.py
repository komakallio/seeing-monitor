"""Switch every heater output off and release the lines: the `seeingmon heater-off` command.

The `core` unit runs this command after `core` stops, whatever the reason (a crash, a kill, the
watchdog, a reboot), so that the heater is off whenever `core` does not run. The command is safe to
run at any time, and a second run does the same as the first.

**What it does.** The command reads the `[heater]` section of the layered configuration and
nothing else. When the heater is enabled, it requests each output line of the pin map in its off
state, drives it to its inactive level (an `active_low` line goes high), and releases it. It
leaves the inputs alone, and it opens no camera, network connection, or store. A line that another
process holds, such as the line of a running `core`, cannot be requested, so the command reports
that line and leaves it alone. The command validates only the keys that it needs (`enabled`,
`output`, `keepalive`, and `pins`) and ignores the rest of `[heater]`, so a mistake in a sensor
setting cannot stop it from switching the outputs off. It also does not import the controller.

**Exit codes.**

- `0`: every heater output is off, or no heater is configured (the section is missing, or
  `enabled` is false). A station without a heater passes.
- `1`: an output cannot be switched off, the configuration cannot be read, or the time limit
  passed. The command still tries the other outputs when one fails.

**Messages.** Each result is one line, prefixed with `seeingmon heater-off:`. A success goes to
standard output, and a failure goes to standard error, one line for each output that failed. A line
names an output by its configured logical name and gives the reason, such as "libgpiod is not
installed", "permission denied", or "the line is busy". It carries no pin map, device path, or
address, because the journal keeps it.

**Time.** The GPIO work ends within `DEADLINE_S` (1 s), even when the GPIO library hangs, because
systemd waits for `ExecStopPost`. A hung native call cannot be cancelled from Python, so a timer
thread ends the process with `os._exit` and the exit code `1`. The limit starts when the command
turns to the GPIO lines, after it has read the configuration. The start of the interpreter and the
import of the configuration come before it, because they run for a bounded time and cannot hang,
and a limit that covered them could end a healthy run on a slow board before it switched anything
off. The limit is real time by design (a hang is a real-time fault), so it does not use a `Clock`.
A call stuck in the kernel can still delay the exit, and nothing in user space bounds that.

**What software cannot do.** After a process exits, the kernel returns a released GPIO line to its
default, usually an input. The command drives the outputs off before the release, and it cannot hold
them there. A heater driver with no pull resistor that keeps it off, and no failsafe of its own,
can float on. See the heater paragraph of `docs/architecture.md`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, TypeAlias

from pydantic import ConfigDict, Field

from seeingmon.config import ConfigError, SectionModel, load_config
from seeingmon.hardware.io import GpiodLibraryError, Io, IoError, LibgpiodIo, PinSpec

DEADLINE_S = 1.0
EXIT_OFF = 0
EXIT_FAILED = 1
_PREFIX = "seeingmon heater-off: "
_log = logging.getLogger(__name__)

IoFactory: TypeAlias = Callable[[Mapping[str, PinSpec]], Io]
Failure: TypeAlias = tuple[str, str]  # an output name and the reason


class HeaterLines(SectionModel):
    """The part of `[heater]` that this command reads. It ignores every other key.

    The fields and defaults match `seeingmon.hardware.heater.HeaterConfig`, and a test keeps them
    in step. The command does not use that class, because it validates the sensors and the control
    constants too, and a mistake there must not stop the outputs from going off.
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    output: str = Field(default="heater", min_length=1)
    keepalive: str | None = None
    pins: dict[str, PinSpec] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Plan:
    """What to switch off: the output lines to request, and the names that have no such line."""

    outputs: dict[str, PinSpec]
    missing: tuple[str, ...]


def open_lines(pins: Mapping[str, PinSpec]) -> Io:
    """Request `pins` through libgpiod, as `create_heater` does. Each output starts off."""
    return LibgpiodIo(pins)


class _Progress:
    """What the deadline handler reports. The main thread writes it, and the timer reads it."""

    __slots__ = ("expired", "finished", "off", "outputs")

    def __init__(self) -> None:
        self.outputs: tuple[str, ...] = ()  # every output to switch off
        self.off: list[str] = []  # the outputs that were driven off
        self.finished = False  # the work ended, so a late timer does nothing
        self.expired = False  # the timer fired


def _say(stream: TextIO, text: str) -> None:
    """Write one line and flush it, because the timer can end the process without a clean exit."""
    with contextlib.suppress(OSError, ValueError):  # a closed pipe must not change the result
        stream.write(f"{_PREFIX}{' '.join(text.split())}\n")
        stream.flush()


def _reason(error: Exception) -> str:
    """Why a call failed, in words that carry no path, pin, or address.

    The full error, with its traceback, goes to the log at the debug level for a person who runs
    the command by hand with `--log-level debug`. The journal gets only the short reason.
    """
    _log.debug("a GPIO call failed", exc_info=error)
    if isinstance(error, IoError):
        return str(error)  # the messages of IoError name no device path
    return f"unexpected {type(error).__name__}"  # the text of another error can hold a path


def read_plan(local_config: Path | None = None) -> Plan | None:
    """Read the pin map. Returns the lines to switch off, or `None` when no heater is configured.

    Raises `ConfigError` when the configuration cannot be read. The heater counts as configured
    when `enabled` is true. Every output line of the pin map goes into the plan, and the plan
    lists the names of `output` and `keepalive` that have no output line.
    """
    heater = load_config(local_file=local_config).section("heater", HeaterLines)
    if not heater.enabled:
        return None
    outputs = {name: spec for name, spec in heater.pins.items() if spec.direction == "output"}
    named = dict.fromkeys([heater.output, *([heater.keepalive] if heater.keepalive else [])])
    return Plan(outputs, tuple(name for name in named if name not in outputs))


def _drive_off(
    group: Mapping[str, PinSpec], open_io: IoFactory, progress: _Progress
) -> list[Failure]:
    """Switch the outputs of `group` off and release their lines. Returns the failures.

    The function asks for all lines at once. When that fails and the group has more than one
    output, it asks for each output by itself, so that one line that cannot be had neither stops
    the others nor hides its name. A library that fails (`GpiodLibraryError`) fails for every
    output, so the function gives each output that reason and does not ask again: each new request
    would look for the library once more, and the lookup can spawn programs.
    """
    try:
        io = open_io(group)
    except Exception as error:
        if len(group) == 1 or isinstance(error, GpiodLibraryError):
            reason = _reason(error)
            return [(name, reason) for name in group]
        failures: list[Failure] = []
        for name, spec in group.items():
            failures += _drive_off({name: spec}, open_io, progress)
        return failures
    failures = []
    for name in group:
        try:
            io.set_output(name, False)
        except Exception as error:
            failures.append((name, _reason(error)))
        else:
            progress.off.append(name)
    try:
        io.close()
    except Exception as error:
        failed = {name for name, _ in failures}
        reason = f"could not release the line: {_reason(error)}"
        failures += [(name, reason) for name in group if name not in failed]
    return failures


def run_heater_off(
    *,
    local_config: Path | None = None,
    open_io: IoFactory | None = None,
    deadline_s: float = DEADLINE_S,
    exit_process: Callable[[int], object] = os._exit,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Switch every heater output off, release the lines, and return the exit code.

    Args:
        local_config: Read this file instead of `local/config.toml`.
        open_io: Builds the `Io` for a group of pins. The default requests them through libgpiod.
        deadline_s: The time limit of the GPIO work. When it passes, the function writes a line
            and calls `exit_process(1)`, which ends the process even when a GPIO call has blocked.
        exit_process: Ends the process at once. A test passes a stand-in that records the call.
        stdout: Where a success goes (default: standard output).
        stderr: Where a failure goes (default: standard error).
    """
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    try:
        plan = read_plan(local_config)
    except ConfigError as error:
        _say(err, f"cannot read the heater configuration: {error}")
        return EXIT_FAILED
    if plan is None:
        _say(out, "the heater is not enabled, so there is nothing to do")
        return EXIT_OFF
    progress = _Progress()
    progress.outputs = tuple(plan.outputs)

    def expire() -> None:
        if progress.finished:
            return
        progress.expired = True
        pending = [name for name in progress.outputs if name not in progress.off]
        text = f"the GPIO calls did not finish within {deadline_s:g} s"
        if pending:
            text += f"; not confirmed off: {', '.join(pending)}"
        try:
            _say(err, text)
        finally:
            exit_process(EXIT_FAILED)

    failures = [(name, "the pin map has no output line of this name") for name in plan.missing]
    if plan.outputs:
        timer = threading.Timer(deadline_s, expire)
        timer.daemon = True
        with contextlib.suppress(RuntimeError):  # no thread to spare: work without a limit
            timer.start()
        try:
            failures += _drive_off(plan.outputs, open_io or open_lines, progress)
        finally:
            progress.finished = True
            timer.cancel()
        if progress.expired:  # the timer wrote the failure, and the process is about to end
            return EXIT_FAILED
    for name, reason in failures:
        _say(err, f"cannot switch off {name}: {reason}")
    if failures:
        return EXIT_FAILED
    _say(out, f"the heater outputs are off: {', '.join(plan.outputs)}")
    return EXIT_OFF
