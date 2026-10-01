"""The store: SQLite results, segment files of per-frame metrics, retention, and the data layout.

Import the classes from here, for example `from seeingmon.store import Store`. The package
loads each name on first use, so importing `seeingmon.store` (and `seeingmon --help`) stays
fast.

- `db`: `Store` (the writer), `StoreReader` (read-only access for `web` and tools), the sink
  cursors, and the immutability and event rules.
- `segments`: `SegmentWriter` and `SegmentReader` for the 10-minute files of per-frame metrics.
- `layout`: `DataLayout` for the data directory, atomic writes, and the pin marker of a burst.
- `retention`: `RetentionManager`, which keeps the data directory within its quotas.
- `events`: `EventEmitter`, which writes the events of the storage lane.
- `config`: the configuration section models (`[store]` in `config/default.d/store.toml`).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from seeingmon.store.db import (
        DuplicateRecordError,
        SinkCursor,
        Store,
        StoreBusyError,
        StoreClosedError,
        StoreError,
        StoreFormatError,
        StoreReader,
        UnknownRecordTypeError,
        record_from_row,
    )
    from seeingmon.store.events import EventEmitter, EventSink

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
    "UnknownRecordTypeError": "db",
    "record_from_row": "db",
    "EventEmitter": "events",
    "EventSink": "events",
}

__all__ = [
    "DuplicateRecordError",
    "EventEmitter",
    "EventSink",
    "SinkCursor",
    "Store",
    "StoreBusyError",
    "StoreClosedError",
    "StoreError",
    "StoreFormatError",
    "StoreReader",
    "UnknownRecordTypeError",
    "record_from_row",
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
