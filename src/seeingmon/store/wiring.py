"""Open the whole storage side of `core` in one call.

`open_storage(config, clock)` builds what `core` needs, in the right order: the data layout, the
store, the event emitter, the segment writer (after it repaired the files that a crash left), the
sinks of the `[sinks]` section, the forwarder, and the retention manager. The `Storage` object
holds them, runs the housekeeping loop, and closes everything in the right order.

```python
config = load_config()
storage = open_storage(config, SystemClock())
scheduler = Scheduler(..., records=storage.store.as_record_writer(), metrics=storage.segments)
threading.Thread(target=storage.run_housekeeping, args=(stop.is_set,), name="housekeeping").start()
...
health = HealthRecord(..., **storage.health_fields())
...
storage.close()  # after the housekeeping thread stopped
```

**Housekeeping.** `run_housekeeping` runs on one thread. It forwards rows to the sinks all the
time, closes an idle segment every minute, and runs the retention task at the start and then every
`interval_s`. A failure in one of these steps is logged, and the loop goes on.

**Health.** `health_fields` returns the storage fields of a `health` record: the free space, the
space that the data directory uses, the backlog of each sink, and the flags `low_space` and
`sink_backlog`. Merge them into the record that the scheduler builds.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.config import Config
from seeingmon.sinks.base import Sink
from seeingmon.sinks.factory import build_sinks
from seeingmon.sinks.forwarder import Forwarder
from seeingmon.store.config import GB, StoreConfig
from seeingmon.store.db import Store
from seeingmon.store.events import EventEmitter
from seeingmon.store.layout import DataLayout
from seeingmon.store.retention import DiskProbe, RetentionManager, system_disk_usage
from seeingmon.store.segments import SegmentWriter

logger = logging.getLogger(__name__)

SEGMENT_TICK_S = 60.0


@dataclass(slots=True)
class Storage:
    """The storage side of `core`. Build it with `open_storage`."""

    layout: DataLayout
    store: Store
    events: EventEmitter
    segments: SegmentWriter
    retention: RetentionManager
    forwarder: Forwarder
    config: StoreConfig
    clock: Clock
    _next_tick_ns: int = field(default=-1, init=False, repr=False)  # -1: due at once
    _next_retention_ns: int = field(default=-1, init=False, repr=False)

    def run_housekeeping(
        self, should_stop: Callable[[], bool], *, max_passes: int | None = None
    ) -> int:
        """Forward, tick the segments, and run retention, until `should_stop()` is true.

        Call it from one thread. It sleeps on the `Clock` when no sink has rows to send. The
        schedule of the segment tick and of retention lives in this object, so a second call
        carries on where the first stopped. `max_passes` ends the loop after that many passes,
        which suits a test. Returns the number of passes.
        """
        clock = self.clock
        passes = 0
        while not should_stop():
            report = self.forwarder.run_once()
            passes += 1
            now_ns = clock.monotonic_ns()
            if now_ns >= self._next_tick_ns:
                self._next_tick_ns = now_ns + round(SEGMENT_TICK_S * NS_PER_S)
                self._guard("closing idle segments", self.segments.tick)
            if now_ns >= self._next_retention_ns:
                self._next_retention_ns = now_ns + round(
                    self.config.retention.interval_s * NS_PER_S
                )
                self._guard("the retention pass", self.retention.run_once)
            if max_passes is not None and passes >= max_passes:
                break
            if not report.more_pending:
                clock.sleep(self.config.forwarder.poll_interval_s)
        return passes

    @staticmethod
    def _guard(what: str, step: Callable[[], object]) -> None:
        try:
            step()
        except Exception:
            logger.exception("%s failed; the housekeeping loop goes on", what)

    def health_fields(self) -> dict[str, Any]:
        """The storage fields of a `health` record, as keyword arguments for `HealthRecord`.

        `free_space_gb` is the free space on the data partition now. `data_used_gb` is what the data
        directory used at the last retention pass, or `None` before the first. `sink_backlog` maps
        each sink to its unsent rows. `flags` holds `low_space` while the free space is below the
        limit, and `sink_backlog` while a sink has more unsent rows than the configured limit.
        """
        status = self.retention.status()
        flags: list[str] = []
        if status.low_space:
            flags.append("low_space")
        if self.forwarder.backlog_exceeded():
            flags.append("sink_backlog")
        return {
            "free_space_gb": status.free_bytes / GB,
            "data_used_gb": None if status.usage_bytes is None else status.usage_bytes / GB,
            "sink_backlog": self.forwarder.sink_backlog(),
            "flags": flags,
        }

    def capture_allowed(self) -> bool:
        """Whether raw capture may run. It is `RetentionManager.capture_allowed`."""
        return self.retention.capture_allowed()

    def close(self) -> None:
        """Close the sinks, finish the open segment, and close the store, in that order.

        Stop the housekeeping thread first. Calling `close` twice is safe.
        """
        try:
            self.forwarder.close()
        finally:
            try:
                self.segments.close()
            finally:
                self.store.close()


def open_storage(
    config: Config,
    clock: Clock,
    *,
    sinks: Sequence[Sink] | None = None,
    disk_usage: DiskProbe = system_disk_usage,
) -> Storage:
    """Open the data directory, the store, the segment writer, the forwarder, and retention.

    The function creates the folders, opens the database (and migrates it), and repairs the
    segment files that an unclean stop left. It writes one `storage.recovered` event when it
    repaired any. By default it builds the sinks from the `[sinks]` section (`build_sinks`). Pass
    `sinks` to use your own, and `disk_usage` to replace the probe of the partition.

    Raises `ConfigError` for an invalid `paths`, `store`, or `sinks` section or a missing secret,
    and `StoreFormatError` or `SchemaError` when the database does not fit.
    """
    store_config = config.section("store", StoreConfig)
    layout = DataLayout.from_config(config)
    layout.create()
    chosen = build_sinks(config) if sinks is None else list(sinks)
    station_id = config.station_id
    profile_id = config.profile.id
    store = Store.open(
        layout.db_path,
        busy_timeout_ms=store_config.busy_timeout_ms,
        max_readers=store_config.reader_connections,
    )
    try:
        events = EventEmitter(store.write, clock, station_id=station_id, profile_id=profile_id)
        segments = SegmentWriter(
            layout.segments_dir,
            clock,
            station_id=station_id,
            profile_id=profile_id,
            config=store_config.segments,
        )
        repaired = segments.recover_orphans()
        if repaired:
            events.emit(
                "warning",
                "storage.recovered",
                f"Repaired {len(repaired)} segment files that an unclean stop left behind.",
                {
                    "files": len(repaired),
                    "actions": sorted({report.action for report in repaired}),
                    "rows": sum(report.rows for report in repaired),
                    "dropped_bytes": sum(report.dropped_bytes for report in repaired),
                },
            )
        forwarder = Forwarder(store, chosen, clock, store_config.forwarder, events=events)
        retention = RetentionManager(
            layout,
            store_config.retention,
            clock,
            events,
            store=store,
            disk_usage=disk_usage,
        )
    except BaseException:
        store.close()
        raise
    return Storage(
        layout=layout,
        store=store,
        events=events,
        segments=segments,
        retention=retention,
        forwarder=forwarder,
        config=store_config,
        clock=clock,
    )
