"""The steps of the recovery ladder that only a supervisor can do.

The scheduler climbs the ladder after a camera fault. The driver performs the first three steps
(restart the capture, reopen the camera, reset the USB device). For the last three, the scheduler
calls the `escalate` callback that `core` provides, with an `EscalationLevel`:

1. **Restart `acquire`.** `core` asks `acquire` to exit (`RemoteCameraDriver.request_restart`).
   The process stops with exit code 75, and systemd starts a new one, so the vendor SDK starts
   clean. The step is done when the connection to the old process is gone.
2. **Reboot.** `core` runs the command of `[services.core.escalation] reboot_command`, as an
   argument list without a shell. The default is no command, and then the step writes an event and
   does nothing, so no machine reboots until you name the command in the local configuration.
3. **Power cycle.** `core` asks the hook of `seeingmon.hardware.power` for a cycle. The hook has its
   own limits (a minimum interval and a daily maximum) and its own events.

Every step writes an `escalation.*` event first, because the last two can end the process. An event
never holds the command, an address, or an output. The callback runs on the scheduler thread, so
it returns when the step is done or has failed. It never raises.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Protocol

from seeingmon.clock import Clock
from seeingmon.drivers.base import CameraError
from seeingmon.hardware.power import CommandResult, CommandRunner, PowerResult, SubprocessRunner
from seeingmon.scheduler.levels import EscalationLevel, step_name
from seeingmon.services.core.events import EventWriter
from seeingmon.services.core.settings import EscalationSettings

_log = logging.getLogger(__name__)

POLL_S = 0.2


class PowerHook(Protocol):
    """`seeingmon.hardware.power.PowerCycle` fits."""

    def request(self, reason: str) -> PowerResult: ...


class Escalator:
    """Performs `EscalationLevel` steps. Pass an instance as `escalate` to the scheduler.

    `restart_acquire` asks the process to exit, and `acquire_connected` says whether the session
    with it is still alive. A part that is `None` makes its step report that it is unavailable.
    """

    def __init__(
        self,
        *,
        writer: EventWriter,
        clock: Clock,
        settings: EscalationSettings,
        restart_acquire: Callable[[str], None] | None = None,
        acquire_connected: Callable[[], bool] | None = None,
        power: PowerHook | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        self._writer = writer
        self._clock = clock
        self._settings = settings
        self._restart_acquire = restart_acquire
        self._acquire_connected = acquire_connected
        self._power = power
        self._runner = runner or SubprocessRunner()
        self.performed: list[EscalationLevel] = []

    def __call__(self, level: EscalationLevel) -> None:
        steps: dict[EscalationLevel, Callable[[str], None]] = {
            EscalationLevel.RESTART_ACQUIRE: self._restart,
            EscalationLevel.REBOOT: self._reboot,
            EscalationLevel.POWER_CYCLE: self._power_cycle,
        }
        step = steps.get(level)
        if step is None:  # a step that this class does not know
            self._writer.emit(
                "error",
                "escalation.unknown_step",
                "The scheduler asked for a recovery step that core does not know.",
                {"level": int(level)},
            )
            return
        name = step_name(level)
        self.performed.append(level)
        try:
            step(f"recovery ladder: {name}")
        except Exception:  # the ladder must go on, whatever a step does
            _log.exception("the escalation step %s failed", name)
            self._writer.emit(
                "error", "escalation.failed", f"The step {name} raised an error.", {"step": name}
            )

    # --- The steps -------------------------------------------------------------------------

    def _restart(self, reason: str) -> None:
        if self._restart_acquire is None:
            self._writer.emit(
                "warning",
                "escalation.restart_acquire_unavailable",
                "Core cannot ask acquire to restart, so the step did nothing.",
            )
            return
        self._writer.emit(
            "warning",
            "escalation.restart_acquire",
            "Core asks acquire to restart, so that the camera driver starts clean.",
        )
        try:
            self._restart_acquire(reason)
        except CameraError as error:
            self._writer.emit(
                "error",
                "escalation.restart_acquire_failed",
                "Acquire did not accept the restart request.",
                {"error": type(error).__name__},
            )
            return
        if self._wait_until_gone():
            self._writer.emit("info", "escalation.acquire_stopped", "Acquire stopped.")
        else:
            self._writer.emit(
                "warning",
                "escalation.acquire_still_connected",
                "Acquire did not stop in time.",
                {"waited_s": self._settings.restart_acquire_wait_s},
            )

    def _wait_until_gone(self) -> bool:
        if self._acquire_connected is None:
            return True
        waited = 0.0
        while self._acquire_connected():
            if waited >= self._settings.restart_acquire_wait_s:
                return False
            self._clock.sleep(POLL_S)
            waited += POLL_S
        return True

    def _reboot(self, reason: str) -> None:
        argv: Sequence[str] = self._settings.reboot_command
        if not argv:
            self._writer.emit(
                "warning",
                "escalation.reboot_unavailable",
                "No reboot command is configured, so the reboot step did nothing.",
            )
            return
        self._writer.emit(
            "warning", "escalation.reboot", "Core runs the reboot command.", {"reason": reason}
        )
        result: CommandResult = self._runner.run(argv, self._settings.command_timeout_s)
        if result.timed_out or result.returncode != 0:
            self._writer.emit(
                "error",
                "escalation.reboot_failed",
                "The reboot command failed.",
                {"timed_out": result.timed_out, "returncode": result.returncode},
            )

    def _power_cycle(self, reason: str) -> None:
        if self._power is None:
            self._writer.emit(
                "warning",
                "escalation.power_cycle_unavailable",
                "No power-cycle hook is configured, so the step did nothing.",
            )
            return
        self._writer.emit(
            "warning", "escalation.power_cycle", "Core asks for a power cycle.", {"reason": reason}
        )
        result = self._power.request(reason)
        self._writer.emit(
            "info" if result.outcome.value == "done" else "warning",
            "escalation.power_cycle_result",
            "The power-cycle hook answered.",
            {"outcome": result.outcome.value},
        )
