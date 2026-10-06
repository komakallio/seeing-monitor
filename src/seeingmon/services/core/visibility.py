"""The nightly visibility summary of Polaris: one `visibility_summary` record when a night ends.

`VisibilitySummary.write_due` is a periodic task of `core`. Once the split hour of a night
(`[survey] night_split_utc_hour`) and one analysis window have passed, it reads the night back from
the store and writes its record. The window lets the seeing window that began before the split hour
reach the store. The values come from `seeingmon.visibility.summary`, which says how the events, the
health records, and the seeing windows of the night become the record.

**From the store, written once.** The summary reads the store and keeps no state of the night, so a
restart of `core` in the middle of a night loses nothing. The record of a night has a fixed key: its
`t_utc_ns` is the start of the night. A second write of the same night raises
`DuplicateRecordError`, which means that the record is there already. `core` writes no summary at
shutdown, because the night is not over, and the fixed key would block the real record at the
split hour.

**Catch-up.** The first call after a start of `core` checks the last `[survey.visibility]
catch_up_nights` nights, the newest night that ended included, and writes each one that has health
records and no summary, oldest first, so that each night finds the summary of the night before. A
night that has its summary, or no health record, costs one or two reads of the store. A call writes
one night at most, and the next calls of the task write the rest, so the first start after an
upgrade, with a week of health records and no summary, never holds the supervisor thread, which
also sends the heartbeat to systemd, for more than one night. Each later call handles the nights
that ended since the call before.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from typing import Any

import seeingmon
from seeingmon.clock import NS_PER_S, Clock
from seeingmon.records.visibility import VisibilitySummaryRecord
from seeingmon.scheduler.config import SiteConfig
from seeingmon.scheduler.ephemeris import sun_elevation_deg
from seeingmon.services.core.settings import SkyFlagSettings
from seeingmon.services.core.skyflags import moon_flag
from seeingmon.store.db import DuplicateRecordError, Store, record_from_row
from seeingmon.survey.config import VisibilityConfig
from seeingmon.survey.nights import night_label, night_start_utc_ns
from seeingmon.visibility.summary import (
    LOOKBACK_NS,
    NIGHT_NS,
    SUMMARY_EVENT_KINDS,
    Event,
    HealthMark,
    NightData,
    SummaryRules,
    WindowMark,
    summarize_night,
)

_log = logging.getLogger(__name__)

_BATCH = 2000  # rows for each read of the store


class VisibilitySummary:
    """Writes the `visibility_summary` record of each night once. See the module text.

    `delay_s` is how long after the split hour the summary waits, which should be at least the
    length of an analysis window. `health_interval_s` is `[services.core] health_interval_s`, and a
    health record vouches for `settings.health_span_intervals` of them.
    """

    def __init__(
        self,
        store: Store,
        *,
        clock: Clock,
        station_id: str,
        profile_id: str,
        split_utc_hour: float,
        settings: VisibilityConfig,
        health_interval_s: float,
        delay_s: float,
        site: SiteConfig | None,
        sky_flags: SkyFlagSettings,
    ) -> None:
        self._store = store
        self._clock = clock
        self._station_id = station_id
        self._profile_id = profile_id
        self._split = split_utc_hour
        self._rules = SummaryRules(
            max_gap_s=settings.max_gap_s,
            health_span_s=settings.health_span_intervals * health_interval_s,
        )
        self._catch_up_nights = settings.catch_up_nights
        self._delay_ns = round(delay_s * NS_PER_S)
        self._site = site
        self._sky_flags = sky_flags
        self._done: str | None = None  # the label of the newest night that came due
        self._pending: list[str] = []  # the nights that came due and wait, oldest first
        self.written = 0

    def due_night(self) -> str:
        """The label of the newest night whose end, plus the delay, lies in the past."""
        current = night_label(self._clock.utc_ns() - self._delay_ns, self._split)
        return night_label(night_start_utc_ns(current, self._split) - 1, self._split)

    def due_nights(self) -> list[str]:
        """The labels of the nights that the next call of `write_due` handles, oldest first.

        They are the nights that ended since the last call, and at most `catch_up_nights` of them,
        which on the first call after a start are the last `catch_up_nights`.
        """
        due = self.due_night()
        if due == self._done:
            return []
        newest = night_start_utc_ns(due, self._split)
        oldest = newest - (self._catch_up_nights - 1) * NIGHT_NS
        if self._done is not None:
            oldest = max(oldest, night_start_utc_ns(self._done, self._split) + NIGHT_NS)
        return [night_label(start, self._split) for start in range(oldest, newest + 1, NIGHT_NS)]

    @property
    def pending(self) -> tuple[str, ...]:
        """The labels of the nights that are due and wait for a later call, oldest first."""
        return tuple(self._pending)

    def write_due(self) -> VisibilitySummaryRecord | None:
        """Write the record of the oldest due night that has none (see `due_nights`).

        A call reads the rows of one night at most, so the catch-up after a start spreads over the
        calls of the task, and no call holds the supervisor thread for longer than the summary of
        one night. A night that has its summary, or no health record, costs one or two reads of the
        store, and the call moves on to the next. Returns the record that the call wrote, or `None`.
        A failure is logged, and the night counts as handled, so a broken night costs one log entry.
        A restart of `core` tries the nights again.
        """
        nights = self.due_nights()
        if nights:
            self._done = nights[-1]
            self._pending += [label for label in nights if label not in self._pending]
        while self._pending:
            read, record = self._write_night(self._pending.pop(0))
            if read:
                return record
        return None

    def _write_night(self, label: str) -> tuple[bool, VisibilitySummaryRecord | None]:
        """Write the record of a night.

        Returns whether the night cost more than the two checks, and the record that it wrote.
        """
        try:
            if self._has_summary(night_start_utc_ns(label, self._split)):
                return False, None
            record = self.summarize(label)
            if record is None:
                return False, None
            self._store.write(record)
        except DuplicateRecordError:
            return True, None  # written by another process since the check
        except Exception:
            _log.exception("could not write the visibility summary of the night %s", label)
            return True, None
        self.written += 1
        _log.info("wrote the visibility summary of the night %s", label)
        return True, record

    def _has_summary(self, start_ns: int) -> bool:
        rows = self._store.range(
            "visibility_summary", start_ns, start_ns + 1, limit=1, station_id=self._station_id
        )
        return bool(rows)

    def summarize(self, label: str) -> VisibilitySummaryRecord | None:
        """The record of a night, from the store, or `None` when no health record lies in it.

        A night without a health record is one in which `core` never ran, so it has no record.
        """
        start = night_start_utc_ns(label, self._split)
        end = start + NIGHT_NS
        if not self._store.range("health", start, end, limit=1, station_id=self._station_id):
            return None
        health = [HealthMark.from_values(v) for v in self._rows("health", start - LOOKBACK_NS, end)]
        events = [
            Event(
                t_utc_ns=int(values["t_utc_ns"]),
                kind=str(values["kind"]),
                detail=values.get("detail") or {},
            )
            for values in self._rows("event", start - LOOKBACK_NS, end)
            if values.get("kind") in SUMMARY_EVENT_KINDS
        ]
        windows = [WindowMark.from_values(v) for v in self._rows("seeing_window", start, end)]
        data = NightData(
            label=label,
            start_utc_ns=start,
            end_utc_ns=end,
            events=events,
            health=health,
            windows=windows,
            visible_before=self._visible_before(start),
        )
        summary = summarize_night(data, self._rules, sun_at=self._sun_at, moon_at=self._moon_at)
        return VisibilitySummaryRecord(
            station_id=self._station_id,
            t_utc_ns=start,
            profile_id=self._profile_id,
            provenance={"software": seeingmon.__version__},
            **summary.fields(),
        )

    def _visible_before(self, start_ns: int) -> bool | None:
        """Whether the summary of the previous night says that Polaris was visible at its end."""
        rows = self._store.range(
            "visibility_summary",
            start_ns - NIGHT_NS,
            start_ns - NIGHT_NS + 1,
            limit=1,
            station_id=self._station_id,
            descending=True,
        )
        if not rows:
            return None
        record = record_from_row("visibility_summary", rows[0])
        if not isinstance(record, VisibilitySummaryRecord):
            return None
        return record.last_visible_utc_ns == start_ns

    def _rows(self, record_type: str, start_ns: int, end_ns: int) -> Iterator[Mapping[str, Any]]:
        """The values of the station's rows of a type in `[start_ns, end_ns)`, in time order."""
        while start_ns < end_ns:
            rows = self._store.range(
                record_type, start_ns, end_ns, limit=_BATCH, station_id=self._station_id
            )
            for row in rows:
                yield row.values
            if len(rows) < _BATCH:
                return
            start_ns = int(rows[-1].values["t_utc_ns"]) + 1

    def _sun_at(self, t_utc_ns: int) -> float | None:
        if self._site is None:
            return None
        elevation = sun_elevation_deg(t_utc_ns, self._site.latitude_deg, self._site.longitude_deg)
        return round(elevation, 2)

    def _moon_at(self, t_utc_ns: int) -> bool | None:
        if self._site is None:
            return None
        return moon_flag(t_utc_ns, self._site, self._sky_flags)
