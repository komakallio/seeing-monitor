"""Sink doubles for the forwarder tests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence

from seeingmon.clock import Clock
from seeingmon.sinks.base import SinkError, StoredRow


class OutageSink:
    """A sink that goes down and comes back, and that upserts what it accepts, as a real one does.

    While `up` is false, every `send` raises a retryable `SinkError`. Set `permanent` to make the
    failure permanent. The sink keeps the first copy of each row by row ID, counts the rows that
    it received more than once, and records the order of delivery.
    """

    def __init__(
        self,
        name: str,
        clock: Clock,
        *,
        record_types: Collection[str] | None = None,
        max_batch_rows: int = 100,
    ) -> None:
        self._name = name
        self._clock = clock
        self._record_types = None if record_types is None else frozenset(record_types)
        self._max_batch_rows = max_batch_rows
        self.up = True
        self.permanent = False
        self.attempts = 0
        self.attempt_times_ns: list[int] = []
        self.batches: list[tuple[str, int, int]] = []  # record type, first row ID, last row ID
        self.received: dict[str, list[int]] = defaultdict(list)  # every row ID, in delivery order
        self.stored: dict[str, dict[int, StoredRow]] = defaultdict(dict)
        self.repeated = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_batch_rows(self) -> int:
        return self._max_batch_rows

    def accepts(self, record_type: str) -> bool:
        return self._record_types is None or record_type in self._record_types

    def send(self, record_type: str, rows: Sequence[StoredRow]) -> None:
        assert rows, "the forwarder never sends an empty batch"
        assert len(rows) <= self._max_batch_rows, "the batch is larger than max_batch_rows"
        ids = [row.row_id for row in rows]
        assert ids == sorted(ids), "a batch is ordered by row ID"
        self.attempts += 1
        self.attempt_times_ns.append(self._clock.monotonic_ns())
        if self.permanent:
            raise SinkError("the endpoint refused the credentials", retryable=False)
        if not self.up:
            raise SinkError("the endpoint is down", retryable=True)
        self.batches.append((record_type, ids[0], ids[-1]))
        for row in rows:
            if row.row_id in self.stored[record_type]:
                self.repeated += 1
            self.stored[record_type].setdefault(row.row_id, row)
            self.received[record_type].append(row.row_id)

    def row_ids(self, record_type: str) -> list[int]:
        """The IDs of the rows that the sink holds, in order."""
        return sorted(self.stored[record_type])
