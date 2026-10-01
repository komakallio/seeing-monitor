"""The recovery ladder: the steps that the scheduler climbs after a camera fault.

The first three steps belong to the driver (`RecoveryLevel` in `seeingmon.drivers.base`), and the
scheduler performs them with `CameraDriver.recover`. The last three steps need the supervisor,
because only it can restart the `acquire` process, reboot the Pi, or cut its power. The
scheduler asks for them through the `escalate` callback that you pass to it, with an
`EscalationLevel`. The numbers continue the driver's numbering, so one integer names a step.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TypeAlias

from seeingmon.drivers.base import RecoveryLevel


class EscalationLevel(IntEnum):
    """Recovery steps that only the supervisor can do, from mildest to most drastic."""

    RESTART_ACQUIRE = 4  # restart the `acquire` process, so the SDK starts clean
    REBOOT = 5  # reboot the Pi
    POWER_CYCLE = 6  # cut and restore the power of the whole Pi


LadderStep: TypeAlias = RecoveryLevel | EscalationLevel

# Every step in the order the scheduler tries them.
LADDER: tuple[LadderStep, ...] = (
    RecoveryLevel.RESTART_CAPTURE,
    RecoveryLevel.REOPEN,
    RecoveryLevel.USB_RESET,
    EscalationLevel.RESTART_ACQUIRE,
    EscalationLevel.REBOOT,
    EscalationLevel.POWER_CYCLE,
)

# The steps that interrupt the whole Pi. The scheduler asks for them rarely.
DESTRUCTIVE_STEPS: frozenset[LadderStep] = frozenset(
    {EscalationLevel.REBOOT, EscalationLevel.POWER_CYCLE}
)

STEP_NAMES: tuple[str, ...] = tuple(step.name.lower() for step in LADDER)


def step_name(step: LadderStep) -> str:
    """The lowercase name of a step, such as `usb_reset`, as it appears in the configuration."""
    return step.name.lower()


def parse_step(name: str) -> LadderStep:
    """Return the step with this lowercase name. Raises `ValueError` for an unknown name."""
    for step in LADDER:
        if step_name(step) == name:
            return step
    raise ValueError(f"unknown recovery step {name!r}; use one of {', '.join(STEP_NAMES)}")
