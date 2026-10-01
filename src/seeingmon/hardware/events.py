"""The event that a hardware component reports to its caller.

A component takes an optional `EventCallback` and calls it for an occurrence that the operator
may need to see, such as a power cycle or an over-temperature cutoff. The services lane turns a
`HardwareEvent` into an `event` record, so the fields match `EventRecord`: a level, a dotted
kind, a message, and a detail object.

An event never carries a host name, an address, a secret, or a serial number. Put the facts
that matter (the route, the outcome, a count) in `detail`, and leave deployment values out.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from seeingmon.records.system import EVENT_KIND_PATTERN, EVENT_LEVELS

_KIND = re.compile(EVENT_KIND_PATTERN)
_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HardwareEvent:
    """An occurrence that a component reports.

    `level` is `info`, `warning`, or `error`. `kind` is a dotted code such as
    `power.cycle_done`. `t_utc_ns` is the time of the occurrence, and the component fills it
    from its clock.
    """

    level: str
    kind: str
    message: str
    t_utc_ns: int
    detail: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.level not in EVENT_LEVELS:
            raise ValueError(
                f"event level must be one of {', '.join(EVENT_LEVELS)}: {self.level!r}"
            )
        if not _KIND.fullmatch(self.kind):
            raise ValueError(
                f"event kind must be a dotted code such as 'power.cycle_done': {self.kind!r}"
            )
        if not self.message:
            raise ValueError("an event needs a message")


EventCallback = Callable[[HardwareEvent], None]


def emit(callback: EventCallback | None, event: HardwareEvent) -> None:
    """Pass `event` to `callback`. A failing callback never disturbs the component.

    The function logs the failure and returns, because a control loop must not stop when the
    code that stores events fails.
    """
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        _log.exception("an event callback failed for %s", event.kind)
