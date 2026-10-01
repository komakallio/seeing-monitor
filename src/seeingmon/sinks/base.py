"""The sink interface: where results go after the local store.

The core stores every record in SQLite first. A forwarder then reads the new rows of each
record type and hands them to each sink. Every sink keeps its own cursor, which is the last
row ID it acknowledged, so a slow or broken sink never blocks another. Tables are
append-only, and a new sink starts at row zero and backfills.

**Delivery.** Delivery is at least once and in row order. A sink must be idempotent: it
upserts by the key `(station_id, t_utc_ns, revision)`, so a repeated batch changes nothing.
`send` returns only when the sink has durably accepted every row in the batch. If it
raises `SinkError(retryable=True)`, the forwarder retries the same batch with backoff and
keeps the cursor where it is. A `SinkError(retryable=False)` means retrying cannot help, so
the forwarder stops that sink, writes an `event`, and leaves the cursor in place.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class StoredRow:
    """One stored record.

    `values` holds JSON-compatible values keyed by the declared field names.
    """

    row_id: int
    values: Mapping[str, Any]


class SinkError(Exception):
    """A sink could not take a batch."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@runtime_checkable
class Sink(Protocol):
    @property
    def name(self) -> str:
        """A unique, stable name. The cursor is stored under it."""
        ...

    @property
    def max_batch_rows(self) -> int:
        """The largest batch `send` takes."""
        ...

    def accepts(self, record_type: str) -> bool:
        """Whether this sink wants records of the type."""
        ...

    def send(self, record_type: str, rows: Sequence[StoredRow]) -> None:
        """Take a non-empty batch of one record type, ordered by `row_id`.

        Raises `SinkError` when the batch is not accepted.
        """
        ...
