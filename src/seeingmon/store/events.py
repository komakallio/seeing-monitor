"""Events that the storage lane writes: retention actions and sink problems.

`EventEmitter` builds an `EventRecord` with the station, the profile, the provenance, and the
time of a `Clock`, and passes it to a callable that stores it. In production that callable is
`Store.write`. The store moves an event that shares its station and timestamp with another
event, so an emitter never handles a collision.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import seeingmon
from seeingmon.clock import Clock
from seeingmon.records.system import EventRecord

# A callable that takes an `EventRecord` and stores it, such as `Store.write`.
EventSink = Callable[[EventRecord], object]

logger = logging.getLogger(__name__)


class EventEmitter:
    """Builds events and hands them to a sink.

    `provenance` defaults to the version of the `seeingmon` package.
    """

    def __init__(
        self,
        sink: EventSink,
        clock: Clock,
        *,
        station_id: str,
        profile_id: str,
        provenance: Mapping[str, str] | None = None,
    ) -> None:
        self._sink = sink
        self._clock = clock
        self._station_id = station_id
        self._profile_id = profile_id
        self._provenance = dict(
            provenance if provenance is not None else {"software": seeingmon.__version__}
        )

    def emit(
        self,
        level: str,
        kind: str,
        message: str,
        detail: Mapping[str, Any] | None = None,
    ) -> EventRecord | None:
        """Write one event and return it, or return `None` when the sink fails.

        `level` is `info`, `warning`, or `error`. `kind` is a dotted code such as
        `retention.early_delete`. A failing sink never stops the caller: the emitter logs the
        failure and carries on, because a retention pass or a forwarder must go on when the
        database is busy. A malformed `level` or `kind` raises `pydantic.ValidationError`.
        """
        record = EventRecord(
            station_id=self._station_id,
            t_utc_ns=self._clock.utc_ns(),
            profile_id=self._profile_id,
            provenance=dict(self._provenance),
            level=level,
            kind=kind,
            message=message,
            detail=None if detail is None else dict(detail),
        )
        try:
            self._sink(record)
        except Exception:
            logger.exception("could not store the %s event", kind)
            return None
        return record
