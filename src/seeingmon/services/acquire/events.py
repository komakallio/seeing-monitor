"""The events that the camera driver reports, kept for `core` to collect.

A driver such as `asi` reports occurrences that an operator needs to see: a recovery step, a camera
that changed its geometry, a profile that does not match the sensor (see
`seeingmon.hardware.events`). The driver lives in `acquire`, and the `event` records live in
`core`, so `acquire` keeps the events in a bounded log, and `core` collects them with the `events`
call, which works whether or not a stream runs.

Each event gets a sequence number. `core` asks for the events after the last number that it saw,
and the answer says how many events the log lost because nobody collected them in time. The log
keeps the last 256 events by default.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from seeingmon.hardware.events import HardwareEvent
from seeingmon.services.ipc.codec import (
    CodecError,
    as_mapping,
    get_int,
    get_str,
)

DEFAULT_LOG_SIZE = 256
_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoggedEvent:
    """An event and its sequence number in the log."""

    seq: int
    event: HardwareEvent


@dataclass(frozen=True, slots=True)
class EventBatch:
    """The events after a sequence number, the last number in the log, and how many were lost."""

    events: tuple[LoggedEvent, ...]
    last: int
    lost: int


class HardwareEventLog:
    """A bounded, thread-safe log of driver events. `record` is the driver's `on_event`."""

    def __init__(self, size: int = DEFAULT_LOG_SIZE) -> None:
        if size < 1:
            raise ValueError("the log holds at least one event")
        self._lock = threading.Lock()
        self._events: deque[LoggedEvent] = deque(maxlen=size)
        self._last = 0

    @property
    def last(self) -> int:
        """The sequence number of the newest event, or 0 when the log has none."""
        with self._lock:
            return self._last

    def record(self, event: HardwareEvent) -> None:
        """Add an event, and write it to the log of the process at its own level."""
        with self._lock:
            self._last += 1
            self._events.append(LoggedEvent(self._last, event))
        _log.log(_LEVELS.get(event.level, logging.INFO), "%s: %s", event.kind, event.message)

    def since(self, after: int = 0) -> EventBatch:
        """The events with a sequence number above `after`, oldest first."""
        with self._lock:
            newer = tuple(item for item in self._events if item.seq > after)
            first = self._events[0].seq if self._events else self._last + 1
            last = self._last
        lost = max(0, first - after - 1) if after < last else 0
        return EventBatch(newer, last, lost)


# --- JSON --------------------------------------------------------------------------------------


def encode_batch(batch: EventBatch) -> dict[str, Any]:
    """An `EventBatch` as a JSON object."""
    return {
        "last": batch.last,
        "lost": batch.lost,
        "events": [
            {
                "seq": item.seq,
                "level": item.event.level,
                "kind": item.event.kind,
                "message": item.event.message,
                "t_utc_ns": item.event.t_utc_ns,
                "detail": None if item.event.detail is None else dict(item.event.detail),
            }
            for item in batch.events
        ],
    }


def _decode_event(value: Any) -> LoggedEvent:
    data = as_mapping(value, "event")
    detail = data.get("detail")
    try:
        event = HardwareEvent(
            level=get_str(data, "level", "event"),
            kind=get_str(data, "kind", "event"),
            message=get_str(data, "message", "event"),
            t_utc_ns=get_int(data, "t_utc_ns", "event"),
            detail=None if detail is None else dict(as_mapping(detail, "event.detail")),
        )
    except ValueError as error:
        raise CodecError(f"event is not valid: {error}") from None
    return LoggedEvent(get_int(data, "seq", "event"), event)


def decode_batch(value: Any) -> EventBatch:
    """The inverse of `encode_batch`. Raises `CodecError` for a malformed batch."""
    data = as_mapping(value, "event batch")
    events = data.get("events")
    if not isinstance(events, list):
        raise CodecError("event batch.events must be a list")
    return EventBatch(
        events=tuple(_decode_event(item) for item in events),
        last=get_int(data, "last", "event batch"),
        lost=get_int(data, "lost", "event batch"),
    )


def after_of(params: Mapping[str, Any]) -> int:
    """The `after` parameter of an `events` call. Absent means 0. A negative number is refused."""
    after = 0 if "after" not in params else get_int(params, "after", "events")
    if after < 0:
        raise CodecError("events.after must not be negative")
    return after


__all__ = [
    "DEFAULT_LOG_SIZE",
    "EventBatch",
    "HardwareEventLog",
    "LoggedEvent",
    "after_of",
    "decode_batch",
    "encode_batch",
]
