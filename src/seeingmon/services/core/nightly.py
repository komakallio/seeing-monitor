"""The nightly star summary: close each night on time, and close the last one at shutdown.

The survey analysis averages the matched stars of the frames of one night into one `star_epoch`
record (`seeingmon.survey.star_epoch`). A night runs from `night_split_utc_hour` (12:00 UTC by
default) to the same hour of the next day. The analyzer closes a night when a frame of the next
night arrives, but no survey frame arrives in the daytime, so without help the record of a night
would wait for the next evening, and a process that stops would lose the night in memory.
`NightlySummary` wraps the analyzer and does the two things that it cannot do itself:

- **At the end of each night.** `flush_due` (a periodic task of `core`) closes the open night once
  the clock has passed the split hour, and it writes the record to the store. It acts only when the
  analyzer holds no frame in flight and no frame of the new night has arrived, because a frame of
  the new night has closed the old night already, and closing again would cut the new one short.
- **At shutdown.** `flush` closes the open night whatever the time. Call it after the scheduler has
  stopped, and before the store closes.

The analyzer touches its accumulator when `poll` collects a result, and `flush_night` touches it
too, so a lock keeps the scheduler thread (`poll`) and the supervisor thread (`flush_due`) apart.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from seeingmon.analysis import SurveyOutput
from seeingmon.clock import Clock
from seeingmon.frames import Frame
from seeingmon.records import Record
from seeingmon.survey.nights import night_label

_log = logging.getLogger(__name__)


class NightAnalyzer(Protocol):
    """A survey analyzer that keeps a nightly summary (`SurveyPipelineAnalyzer` fits)."""

    def submit(self, frame: Frame) -> None: ...

    def poll(self) -> tuple[SurveyOutput, ...]: ...

    def pending(self) -> int: ...

    def flush_night(self) -> Sequence[Record]: ...


class NightlySummary:
    """A `SurveyAnalyzer` that wraps another one and closes its nights. See the module text."""

    def __init__(
        self,
        analyzer: NightAnalyzer,
        *,
        write: Callable[[Record], Any],
        clock: Clock,
        split_utc_hour: float = 12.0,
    ) -> None:
        self._analyzer = analyzer
        self._write = write
        self._clock = clock
        self._split = split_utc_hour
        self._lock = threading.Lock()
        self._night = night_label(clock.utc_ns(), split_utc_hour)  # the night of the last check
        self._frame_night: str | None = None  # the night of the newest frame that came in
        self.written = 0

    # --- The SurveyAnalyzer interface ------------------------------------------------------

    def submit(self, frame: Frame) -> None:
        self._frame_night = night_label(frame.t_utc_ns, self._split)
        self._analyzer.submit(frame)

    def poll(self) -> tuple[SurveyOutput, ...]:
        with self._lock:
            return self._analyzer.poll()

    def pending(self) -> int:
        return self._analyzer.pending()

    # --- The nights ------------------------------------------------------------------------

    def flush_due(self) -> int:
        """Close the open night when the clock has passed the end of it. Returns the records."""
        label = night_label(self._clock.utc_ns(), self._split)
        if label == self._night:
            return 0
        if self._analyzer.pending() > 0:
            return 0  # a frame of the old night is still on its way: try again at the next call
        self._night = label
        if self._frame_night == label:
            return 0  # a frame of the new night has closed the old one already
        return self.flush()

    def flush(self) -> int:
        """Close the open night now, and write its record. Returns the number of records."""
        with self._lock:
            records = self._analyzer.flush_night()
        count = 0
        for record in records:
            try:
                self._write(record)
            except Exception:
                _log.exception("could not store the nightly star summary")
                continue
            count += 1
        self.written += count
        return count

    def __getattr__(self, name: str) -> Any:
        """Everything else (the tracker, the history, `close`) is the analyzer's."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._analyzer, name)
