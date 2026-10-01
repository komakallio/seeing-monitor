"""The zero-point history of the survey analysis, read from the store.

The transparency of a survey frame compares its zero point with a reference: the 90th percentile
of the usable zero points of the last 60 days, from at least 20 frames
(`seeingmon.survey.transparency`). The zero points are in the `sky_quality` records of the store,
so the reference must come from there, and a restart of `core` must not forget it.
`StoreZeroPointHistory` implements the `ZeroPointHistory` protocol of the survey analysis over the
store:

- The constructor loads the window once, from the newest rows backward, so a station that has run
  for a year loads two months and not the year.
- Each later question reads only the rows that arrived since the last one, by row ID. That is one
  small query per survey frame. `core` writes the rows itself (the scheduler hands the survey
  results to the store), so nobody else needs to feed the history.
- The samples stay in memory, in a `MemoryHistory` that answers the time-range questions.

The survey worker process builds its own pipeline and never reads this history: the analyzer in
`core` reads it when it submits a frame, and hands the worker one number, the reference.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.store.db import StoreReader, record_from_row
from seeingmon.survey.transparency import MemoryHistory, ZeroPointSample

_log = logging.getLogger(__name__)

RECORD_TYPE = "sky_quality"
SECONDS_PER_DAY = 86_400.0
BATCH_ROWS = 2_000


class StoreZeroPointHistory:
    """A `ZeroPointHistory` over the `sky_quality` table of a store."""

    def __init__(
        self,
        store: StoreReader,
        *,
        clock: Clock,
        window_days: float = 60.0,
        max_samples: int = 50_000,
    ) -> None:
        if window_days <= 0 or max_samples < 1:
            raise ValueError("the window and the sample limit are positive")
        self._store = store
        self._lock = threading.Lock()
        self._memory = MemoryHistory(max_samples=max_samples)
        self._last_row_id = 0
        self._load(clock.utc_ns(), window_days, max_samples)

    def __len__(self) -> int:
        with self._lock:
            return len(self._memory)

    def _load(self, now_ns: int, window_days: float, max_samples: int) -> None:
        """Load the newest samples of the window, and remember where the table ended."""
        self._last_row_id = self._store.last_row_id(RECORD_TYPE)
        start_ns = now_ns - round(window_days * SECONDS_PER_DAY * NS_PER_S)
        rows = self._store.range(
            RECORD_TYPE, start_ns, now_ns + NS_PER_S, limit=max_samples, descending=True
        )
        loaded = 0
        for row in rows:
            record = record_from_row(RECORD_TYPE, row)
            if isinstance(record, SkyQualityRecord) and self._memory.add_record(record):
                loaded += 1
        _log.info(
            "the zero-point history starts with %d samples of the last %g days", loaded, window_days
        )

    def refresh(self) -> int:
        """Read the rows that arrived since the last read. Returns the number of samples added."""
        added = 0
        with self._lock:
            while True:
                rows = self._store.after(RECORD_TYPE, self._last_row_id, BATCH_ROWS)
                if not rows:
                    return added
                for row in rows:
                    record = record_from_row(RECORD_TYPE, row)
                    if isinstance(record, SkyQualityRecord) and self._memory.add_record(record):
                        added += 1
                self._last_row_id = rows[-1].row_id

    def zero_points(self, since_utc_ns: int, until_utc_ns: int) -> Sequence[ZeroPointSample]:
        """The samples with `since_utc_ns <= t_utc_ns < until_utc_ns`, oldest first."""
        self.refresh()
        with self._lock:
            return self._memory.zero_points(since_utc_ns, until_utc_ns)
