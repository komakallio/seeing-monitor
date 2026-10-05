"""The darkness of the sky and the clear verdict, from the results of the survey frames.

`core` writes two events of the sky from the long survey frames (the frames with a `sky_quality`
record), each at most once a night. The night is the one of `[survey] night_split_utc_hour` (see
`seeingmon.survey.nights`).

- **`sky.dark`** when the sky stops getting darker. The watch fits a line to the sky brightness
  (`sky_mag_arcsec2`) of the last `[survey.darkness] frames` frames that solved, and the sky counts
  as dark when the line changes by less than `max_slope_mag_per_hour`. The rule measures the change
  per hour and not per degree of the Sun, so it also fires on a summer night where the Sun stays
  above -18 degrees: the sky stops changing near the darkest time of every night. A long frame
  that did not solve, or that has no sky brightness, starts the run again, and so does a gap of
  more than `max_gap_s` since the previous frame, so that the line never spans a cloudy stretch or
  hours without frames (a pause, a degraded camera). The detail holds the sky brightness of the
  line at the newest frame (`sky_mag_arcsec2`), the slope (`slope_mag_per_hour`), the number of
  frames (`frames`), and the Sun's elevation at the site at the newest frame
  (`sun_elevation_deg`). The elevation is `null` without a `[site]`, and for a frame whose clock
  was not synchronized. The web API serves it as `null` too, because the Sun's elevation at a
  known time shows the site (`seeingmon.services.web.privacy`).
- **`sky.clear_verdict`** when `verdict_frames` long frames with a cloud fraction have followed
  `sky.dark`. The detail holds the share of them with a cloud fraction at or below
  `[scheduler.cloud] clear_threshold` (`clear_share`), the number of frames (`frames`), the
  threshold (`clear_threshold`), and the median transparency of the frames that have one
  (`transparency_median`, `null` when none has). The verdict counts a frame whether it solved or
  not. A frame that does not solve gets its cloud fraction from the stored pointing, and thick
  clouds are what keeps a frame from solving, so a verdict over the solved frames only would wait
  for the gaps between the clouds and call a cloudy night clear.

Each event carries the time of the frame that completed it. `sky.dark` needs a sky brightness,
which needs a dark model and the sensor temperature, so a station without a dark set writes neither
event.

`DarknessWatch` holds the rule and the state of the night. `SkyDarkness` wraps the survey analyzer
that the scheduler polls, and it hands each result to the watch on its way, on the scheduler thread.

**A restart.** `DarknessWatch.restore` reads the events of the night from the store, so a restart
of `core` writes neither event twice. After a `sky.dark` without a verdict, it also reads the
`sky_quality` rows that followed, so the verdict keeps its frames. The run of `sky.dark` starts
empty, which delays the event by up to `frames - 1` frames after a restart in the twilight.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from seeingmon.analysis import SurveyAnalyzer, SurveyOutput
from seeingmon.clock import NS_PER_S
from seeingmon.frames import Frame
from seeingmon.records import EventRecord
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.scheduler.config import SiteConfig
from seeingmon.scheduler.ephemeris import sun_elevation_deg
from seeingmon.store.db import StoreReader, record_from_row
from seeingmon.survey.config import DarknessConfig
from seeingmon.survey.nights import night_label, night_start_utc_ns

_log = logging.getLogger(__name__)

DARK_EVENT = "sky.dark"
CLEAR_VERDICT_EVENT = "sky.clear_verdict"

_NS_PER_HOUR = 3600 * NS_PER_S
_EVENT_BATCH = 500  # the event rows that one read of the store returns at most
_MAX_VERDICT_ROWS = 2000  # more than the survey frames of the longest night


@dataclass(frozen=True, slots=True)
class SkyFrame:
    """What the watch needs of one long survey frame.

    `time_valid` is false when the clock was not synchronized (the `time_invalid` flag).
    """

    t_utc_ns: int
    solved: bool
    sky_mag_arcsec2: float | None = None
    cloud_fraction: float | None = None
    transparency: float | None = None
    time_valid: bool = True

    @classmethod
    def from_record(cls, record: SkyQualityRecord, *, solved: bool) -> SkyFrame:
        return cls(
            t_utc_ns=record.t_utc_ns,
            solved=solved,
            sky_mag_arcsec2=record.sky_mag_arcsec2,
            cloud_fraction=record.cloud_fraction,
            transparency=record.transparency,
            time_valid="time_invalid" not in record.flags,
        )

    @classmethod
    def from_output(cls, output: SurveyOutput) -> SkyFrame | None:
        """The frame of a survey result, or `None` for a frame without sky quality (a short one)."""
        for record in output.records:
            if isinstance(record, SkyQualityRecord):
                return cls.from_record(record, solved=output.solved)
        return None


@dataclass(frozen=True, slots=True)
class SkyEvent:
    """An event that the watch asks to write."""

    kind: str
    t_utc_ns: int
    message: str
    detail: dict[str, Any]


def fit_line(points: list[tuple[int, float]]) -> tuple[float, float]:
    """A least-squares line through `(t_utc_ns, value)` points, in time order.

    Returns the slope per hour and the value of the line at the time of the last point. The times
    must differ.
    """
    last = points[-1][0]
    hours = [(t - last) / _NS_PER_HOUR for t, _ in points]
    slope, at_last = statistics.linear_regression(hours, [value for _, value in points])
    return slope, at_last


class DarknessWatch:
    """The rule of `sky.dark` and `sky.clear_verdict`, and the state of one night.

    Feed it the long survey frames in time order with `update`. `clear_threshold` is the
    `[scheduler.cloud]` value, and `site` gives the Sun's elevation in the detail of `sky.dark`.
    """

    def __init__(
        self,
        settings: DarknessConfig,
        *,
        clear_threshold: float,
        split_utc_hour: float = 12.0,
        site: SiteConfig | None = None,
    ) -> None:
        self._settings = settings
        self._clear_threshold = clear_threshold
        self._split = split_utc_hour
        self._site = site
        self._night: str | None = None
        self._run: deque[tuple[int, float]] = deque(maxlen=settings.frames)
        self._max_gap_ns = round(settings.max_gap_s * NS_PER_S)
        self._dark_utc_ns: int | None = None
        self._verdict_given = False
        # The frames after `sky.dark` with a cloud fraction: (time, cloud fraction, transparency).
        self._after_dark: list[tuple[int, float, float | None]] = []

    @property
    def night(self) -> str | None:
        """The label of the night that the watch is in, or `None` before the first frame."""
        return self._night

    @property
    def dark_utc_ns(self) -> int | None:
        """The time of the `sky.dark` of the night, or `None` while the sky is not dark yet."""
        return self._dark_utc_ns

    @property
    def verdict_given(self) -> bool:
        return self._verdict_given

    def update(self, frame: SkyFrame) -> list[SkyEvent]:
        """Take the next long survey frame. Returns the events that the frame completes."""
        self._enter(night_label(frame.t_utc_ns, self._split))
        if self._verdict_given:
            return []
        if self._dark_utc_ns is None:
            event = self._watch_darkness(frame)
            return [] if event is None else [event]
        if frame.t_utc_ns > self._dark_utc_ns and frame.cloud_fraction is not None:
            self._after_dark.append((frame.t_utc_ns, frame.cloud_fraction, frame.transparency))
        if len(self._after_dark) < self._settings.verdict_frames:
            return []
        return [self._verdict()]

    def restore(self, store: StoreReader, now_utc_ns: int) -> None:
        """Take up the night of `now_utc_ns` where a restart left it. See the module text."""
        label = night_label(now_utc_ns, self._split)
        self._enter(label)
        end = now_utc_ns + NS_PER_S
        for event in _events(store, night_start_utc_ns(label, self._split), end):
            if event.kind == DARK_EVENT and self._dark_utc_ns is None:
                self._dark_utc_ns = event.t_utc_ns
            elif event.kind == CLEAR_VERDICT_EVENT:
                self._verdict_given = True
        if self._dark_utc_ns is None or self._verdict_given:
            return
        # The store moves an event that collides with another by 1 ns, so the frame of `sky.dark`
        # lies at or before its time, and the frames after it start 1 ns later.
        rows = store.range("sky_quality", self._dark_utc_ns + 1, end, limit=_MAX_VERDICT_ROWS)
        for row in rows:
            record = record_from_row("sky_quality", row)
            if isinstance(record, SkyQualityRecord) and record.cloud_fraction is not None:
                self._after_dark.append(
                    (record.t_utc_ns, record.cloud_fraction, record.transparency)
                )

    # --- The rule --------------------------------------------------------------------------

    def _enter(self, label: str) -> None:
        """Start the state of a new night when `label` differs from the current one."""
        if label == self._night:
            return
        self._night = label
        self._run.clear()
        self._dark_utc_ns = None
        self._verdict_given = False
        self._after_dark = []

    def _watch_darkness(self, frame: SkyFrame) -> SkyEvent | None:
        settings = self._settings
        if self._run and not 0 < frame.t_utc_ns - self._run[-1][0] <= self._max_gap_ns:
            # The clock went back, or no long frame came for a while: the line would span a
            # stretch of the sky that the run never saw.
            self._run.clear()
        if not frame.solved or frame.sky_mag_arcsec2 is None:
            self._run.clear()
            return None
        self._run.append((frame.t_utc_ns, frame.sky_mag_arcsec2))
        if len(self._run) < settings.frames:
            return None
        slope, level = fit_line(list(self._run))
        if abs(slope) >= settings.max_slope_mag_per_hour:
            return None
        self._dark_utc_ns = frame.t_utc_ns
        return SkyEvent(
            kind=DARK_EVENT,
            t_utc_ns=frame.t_utc_ns,
            message=(
                f"The sky is dark: {level:.2f} mag/arcsec^2, and it changes by {slope:+.2f} mag "
                "an hour."
            ),
            detail={
                "sky_mag_arcsec2": round(level, 3),
                "slope_mag_per_hour": round(slope, 3),
                "frames": len(self._run),
                "sun_elevation_deg": self._sun_elevation(frame),
            },
        )

    def _verdict(self) -> SkyEvent:
        frames = self._after_dark[: self._settings.verdict_frames]
        clear = sum(1 for _, cloud, _ in frames if cloud <= self._clear_threshold)
        transparencies = [value for _, _, value in frames if value is not None]
        median = statistics.median(transparencies) if transparencies else None
        self._verdict_given = True
        return SkyEvent(
            kind=CLEAR_VERDICT_EVENT,
            t_utc_ns=frames[-1][0],
            message=f"{clear} of {len(frames)} frames after the sky got dark were clear.",
            detail={
                "clear_share": round(clear / len(frames), 3),
                "frames": len(frames),
                "clear_threshold": self._clear_threshold,
                "transparency_median": None if median is None else round(median, 3),
            },
        )

    def _sun_elevation(self, frame: SkyFrame) -> float | None:
        if self._site is None or not frame.time_valid:
            return None
        elevation = sun_elevation_deg(
            frame.t_utc_ns, self._site.latitude_deg, self._site.longitude_deg
        )
        return round(elevation, 2)


def _events(store: StoreReader, start_ns: int, end_ns: int) -> Iterator[EventRecord]:
    """The stored events with `start_ns <= t_utc_ns < end_ns`, in time order."""
    while True:
        rows = store.range("event", start_ns, end_ns, limit=_EVENT_BATCH)
        for row in rows:
            record = record_from_row("event", row)
            if isinstance(record, EventRecord):
                yield record
        if len(rows) < _EVENT_BATCH:
            return
        start_ns = int(rows[-1].values["t_utc_ns"]) + 1


class EmitEvent(Protocol):
    """What writes an event of `core`: `EventWriter.emit` fits."""

    def __call__(
        self,
        level: str,
        kind: str,
        message: str,
        detail: Mapping[str, Any] | None = None,
        *,
        t_utc_ns: int | None = None,
    ) -> None: ...


class SkyDarkness:
    """A `SurveyAnalyzer` that wraps another one and writes the events of a `DarknessWatch`.

    The results pass through unchanged. A failure of the watch is logged and never reaches the
    scheduler, so the records of the frame still reach the store.
    """

    def __init__(self, analyzer: SurveyAnalyzer, watch: DarknessWatch, *, emit: EmitEvent) -> None:
        self._analyzer = analyzer
        self.watch = watch
        self._emit = emit
        self.written = 0

    def submit(self, frame: Frame) -> None:
        self._analyzer.submit(frame)

    def poll(self) -> tuple[SurveyOutput, ...]:
        outputs = self._analyzer.poll()
        for output in outputs:
            self._observe(output)
        return outputs

    def pending(self) -> int:
        return self._analyzer.pending()

    def _observe(self, output: SurveyOutput) -> None:
        frame = SkyFrame.from_output(output)
        if frame is None:
            return
        try:
            events = self.watch.update(frame)
        except Exception:
            _log.exception("the darkness watch failed on the survey frame at %d", output.t_utc_ns)
            return
        for event in events:
            self._emit("info", event.kind, event.message, event.detail, t_utc_ns=event.t_utc_ns)
            self.written += 1

    def __getattr__(self, name: str) -> Any:
        """Everything else is the analyzer's."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._analyzer, name)
