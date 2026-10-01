"""The store: SQLite results, segment files of per-frame metrics, retention, and the data layout.

Import the classes from here, for example `from seeingmon.store import Store`. The package
loads each name on first use, so importing `seeingmon.store` (and `seeingmon --help`) stays
fast.

- `db`: `Store` (the writer), `StoreReader` (read-only access for `web` and tools), the sink
  cursors, and the immutability and event rules.
- `segments`: `SegmentWriter` and `SegmentReader` for the 10-minute files of per-frame metrics.
- `layout`: `DataLayout` for the data directory, atomic writes, and the pin marker of a burst.
- `retention`: `RetentionManager`, which keeps the data directory within its quotas and decides
  whether raw capture may run.
- `events`: `EventEmitter`, which writes the events of the storage lane.
- `config`: the configuration section models (`[store]` in `config/default.d/store.toml`).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from seeingmon.store.config import (
        ForwarderConfig,
        RetentionConfig,
        SegmentsConfig,
        StoreConfig,
    )
    from seeingmon.store.db import (
        DuplicateRecordError,
        SinkCursor,
        Store,
        StoreBusyError,
        StoreClosedError,
        StoreError,
        StoreFormatError,
        StoreReader,
        StoreSnapshot,
        UnknownRecordTypeError,
        record_from_row,
    )
    from seeingmon.store.events import EventEmitter, EventSink
    from seeingmon.store.layout import DataLayout, PathsConfig, burst_name, write_atomic
    from seeingmon.store.retention import (
        Deletion,
        DiskUsage,
        RetentionManager,
        RetentionReport,
        RetentionStatus,
        system_disk_usage,
    )
    from seeingmon.store.segments import (
        RecoveryReport,
        SegmentData,
        SegmentFormatError,
        SegmentInfo,
        SegmentReader,
        SegmentWriter,
        iter_segments,
        read_segment,
        recover_orphans,
        recover_segment,
    )

# The module that defines each public name.
_EXPORTS: dict[str, str] = {
    "DuplicateRecordError": "db",
    "SinkCursor": "db",
    "Store": "db",
    "StoreBusyError": "db",
    "StoreClosedError": "db",
    "StoreError": "db",
    "StoreFormatError": "db",
    "StoreReader": "db",
    "StoreSnapshot": "db",
    "UnknownRecordTypeError": "db",
    "record_from_row": "db",
    "EventEmitter": "events",
    "EventSink": "events",
    "DataLayout": "layout",
    "PathsConfig": "layout",
    "burst_name": "layout",
    "write_atomic": "layout",
    "ForwarderConfig": "config",
    "RetentionConfig": "config",
    "SegmentsConfig": "config",
    "StoreConfig": "config",
    "Deletion": "retention",
    "DiskUsage": "retention",
    "RetentionManager": "retention",
    "RetentionReport": "retention",
    "RetentionStatus": "retention",
    "system_disk_usage": "retention",
    "RecoveryReport": "segments",
    "SegmentData": "segments",
    "SegmentFormatError": "segments",
    "SegmentInfo": "segments",
    "SegmentReader": "segments",
    "SegmentWriter": "segments",
    "iter_segments": "segments",
    "read_segment": "segments",
    "recover_orphans": "segments",
    "recover_segment": "segments",
}

__all__ = [
    "DataLayout",
    "Deletion",
    "DiskUsage",
    "DuplicateRecordError",
    "EventEmitter",
    "EventSink",
    "ForwarderConfig",
    "PathsConfig",
    "RecoveryReport",
    "RetentionConfig",
    "RetentionManager",
    "RetentionReport",
    "RetentionStatus",
    "SegmentData",
    "SegmentFormatError",
    "SegmentInfo",
    "SegmentReader",
    "SegmentWriter",
    "SegmentsConfig",
    "SinkCursor",
    "Store",
    "StoreBusyError",
    "StoreClosedError",
    "StoreConfig",
    "StoreError",
    "StoreFormatError",
    "StoreReader",
    "StoreSnapshot",
    "UnknownRecordTypeError",
    "burst_name",
    "iter_segments",
    "read_segment",
    "record_from_row",
    "recover_orphans",
    "recover_segment",
    "system_disk_usage",
    "write_atomic",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
