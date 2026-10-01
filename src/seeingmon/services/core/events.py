"""Events in `core`: hardware events become `event` records, and `acquire` is polled for its own.

The heater, the SQM-LE reader, the power-cycle hook, and the driver in `acquire` report
occurrences through an `on_event` callback (see `seeingmon.hardware.events`). `EventWriter` turns
each `HardwareEvent` into an `EventRecord` with the time of the occurrence, and writes it to the
store. The driver lives in another process, so `acquire` keeps its events in a log, and `EventPump`
collects them with the `events` call. A restart of `acquire` starts a new log with new numbers, and
the pump notices the new process, writes an `acquire.restarted` event, and starts at zero again.

A failing store never disturbs a control loop: the writer logs the failure and drops the event.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import seeingmon
from seeingmon.clock import Clock
from seeingmon.drivers.base import CameraError
from seeingmon.hardware.events import HardwareEvent
from seeingmon.records import EventRecord, Record
from seeingmon.services.acquire.events import EventBatch

_log = logging.getLogger(__name__)


class EventWriter:
    """Writes events as records. Pass it as the `on_event` of a hardware component.

    `write` stores a record, such as `Store.write`. `source` names the producer in the
    provenance of each record (`local` for the hardware of this machine, `acquire` for the driver).
    """

    def __init__(
        self,
        write: Callable[[Record], object],
        *,
        station_id: str,
        profile_id: str,
        clock: Clock,
        source: str = "local",
    ) -> None:
        self._write = write
        self._station_id = station_id
        self._profile_id = profile_id
        self._clock = clock
        self._provenance = {"software": seeingmon.__version__, "source": source}
        self.written = 0
        self.failed = 0

    def __call__(self, event: HardwareEvent) -> None:
        self._store(
            event.level,
            event.kind,
            event.message,
            event.t_utc_ns,
            None if event.detail is None else dict(event.detail),
        )

    def emit(
        self,
        level: str,
        kind: str,
        message: str,
        detail: Mapping[str, Any] | None = None,
        *,
        t_utc_ns: int | None = None,
    ) -> None:
        """Write an event of `core` itself, stamped with the clock unless you give a time."""
        self._store(
            level,
            kind,
            message,
            self._clock.utc_ns() if t_utc_ns is None else t_utc_ns,
            None if detail is None else dict(detail),
        )

    def _store(
        self, level: str, kind: str, message: str, t_utc_ns: int, detail: dict[str, Any] | None
    ) -> None:
        try:
            record = EventRecord(
                station_id=self._station_id,
                t_utc_ns=t_utc_ns,
                profile_id=self._profile_id,
                provenance=dict(self._provenance),
                level=level,
                kind=kind,
                message=message,
                detail=detail,
            )
            self._write(record)
        except Exception:
            self.failed += 1
            _log.exception("could not store the %s event", kind)
            return
        self.written += 1


class EventSource(Protocol):
    """What the pump needs from the driver: `RemoteCameraDriver` fits."""

    @property
    def instance(self) -> str | None: ...

    def events(self, after: int = 0) -> EventBatch: ...


class EventPump:
    """Collects the events that the driver in `acquire` reported, and writes them.

    Call `poll` now and then, from one thread. It never raises: an `acquire` that does not
    answer gives no events this time, and the next call tries again.
    """

    def __init__(self, source: EventSource, writer: EventWriter) -> None:
        self._source = source
        self._writer = writer
        self._instance: str | None = None
        self._after = 0
        self.collected = 0

    @property
    def after(self) -> int:
        """The number of the last event that the pump took from the current process."""
        return self._after

    def poll(self) -> int:
        """Collect the events that came since the last call. Returns how many it wrote."""
        instance = self._source.instance
        if instance is None:
            return 0  # no session, so nothing to ask
        if self._instance is not None and instance != self._instance:
            self._writer.emit(
                "warning",
                "acquire.restarted",
                "A new acquire process answers, so the camera driver started again.",
                {"events_seen_from_the_old_process": self._after},
            )
            self._after = 0
        self._instance = instance
        try:
            batch = self._source.events(after=self._after)
            if batch.last < self._after:  # the log restarted, and the instance did not show it
                self._after = 0
                batch = self._source.events(after=0)
        except CameraError:
            _log.debug("acquire gave no events", exc_info=True)
            return 0
        for item in batch.events:
            self._writer(item.event)
        if batch.lost:
            self._writer.emit(
                "warning",
                "acquire.events_lost",
                "The log of acquire dropped events before core collected them.",
                {"lost": batch.lost},
            )
        self._after = batch.last
        self.collected += len(batch.events)
        return len(batch.events)
