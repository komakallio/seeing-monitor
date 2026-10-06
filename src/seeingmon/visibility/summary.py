"""The visibility of Polaris in one night, from the rows that the store holds.

A night runs from `[survey] night_split_utc_hour` to the same hour of the next day
(`seeingmon.survey.nights`). `summarize_night` takes the rows of the night, and of the
`LOOKBACK_NS` before it, and returns the values of the night's `visibility_summary` record:

- **Events.** `polaris.visible` and `polaris.hidden`, which the scheduler writes at every start and
  every end of measure, `sky.dark` and `sky.clear_verdict` (`seeingmon.services.core.darkness`),
  `core.started`, `scheduler.clock_unsynchronized`, and `scheduler.solve_requested`.
- **Health records.** `core` writes one every `[services.core] health_interval_s` and at every
  change of the state, so they tell when the station watched the sky.
- **Seeing windows.** The windows that start in the night give the seeing hours.

**When Polaris was visible.** Polaris counts as visible from each `polaris.visible` to the next
`polaris.hidden`. The scheduler writes `polaris.hidden` at every end of measure, also when the
state changes or the camera fails, so that event ends the measure even when the station stopped
watching before it. A crash of `core` writes no `polaris.hidden`, so a measure without one (closed
by the next `core.started`, or still open at the end of the rows) ends where the station stopped
watching: at the end of the health records that cover it. Polaris was visible at the start of the
night when a measure ran across it. When the lookback holds no `polaris.*` event, the summary of
the previous night decides (`NightData.visible_before`).

**When the station watched.** The station watches the sky while `core` runs and the scheduler is
in `auto` or `safe` without a fault of the camera (`HealthMark.from_values`). In `safe` the sky is
too bright to measure, which is a limit of the detection, so it counts as watching. A fault of
another component, such as the heater or the SQM-LE reader, sets the record's `degraded` too, but
the search and the measure go on, so it counts as watching. A health record vouches for the time
until the next record, and for at most `SummaryRules.health_span_s`. The rest is a gap: a stop or
a crash of `core`, a pause, an alignment session, a commissioning task, or a camera that failed
repeatedly.

**The first and the last detection.** The first detection is the first time in the night that
Polaris was visible, and the last detection is the last time. Each is a bound, and counts as
censored, when the station did not see the change: Polaris was visible at the start of the night
(or at its end), or a gap longer than `SummaryRules.max_gap_s` lies between the start of the night
and the first detection (or between the last detection and the end). A night without a detection
is censored at both ends when such a gap lies anywhere in it. The Sun's elevation of a detection
comes from the detail of its event, where the scheduler leaves it `null` without a site or with a
clock that is not synchronized. A bound at the start or the end of the night, and an end of measure
that a crash cut short, take it from `sun_at`.

**Every value set or explained.** A value that the night cannot give is `None`, and `quality` says
why.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from seeingmon.clock import NS_PER_S

VISIBLE_EVENT = "polaris.visible"
HIDDEN_EVENT = "polaris.hidden"
DARK_EVENT = "sky.dark"
CLEAR_VERDICT_EVENT = "sky.clear_verdict"
CORE_STARTED_EVENT = "core.started"
CLOCK_UNSYNCHRONIZED_EVENT = "scheduler.clock_unsynchronized"
SOLVE_REQUESTED_EVENT = "scheduler.solve_requested"

# The event kinds that a summary reads. Every other kind stays out of memory.
SUMMARY_EVENT_KINDS: frozenset[str] = frozenset(
    {
        VISIBLE_EVENT,
        HIDDEN_EVENT,
        DARK_EVENT,
        CLEAR_VERDICT_EVENT,
        CORE_STARTED_EVENT,
        CLOCK_UNSYNCHRONIZED_EVENT,
        SOLVE_REQUESTED_EVENT,
    }
)

# The states of the scheduler in which the station watches the sky (see the module text).
WATCHING_STATES: frozenset[str] = frozenset({"auto", "safe"})

NIGHT_NS = 24 * 3600 * NS_PER_S
# How far before the night the rows reach: one night, so that a measure that runs across the start
# shows, and the health records that cover it.
LOOKBACK_NS = NIGHT_NS
_NS_PER_HOUR = 3600 * NS_PER_S

NO_DETECTION = "Polaris was not detected in this night"
NO_SITE = "no site is configured, so the Sun's elevation is unknown"
NO_EVENT_SUN = (
    "the event has no Sun elevation: no site is configured, or the clock was not synchronized"
)
NO_DARK = (
    "no sky.dark event in this night: it needs a dark set and solved survey frames whose sky "
    "brightness stops changing"
)
NO_VERDICT = "no sky.clear_verdict event in this night: the verdict follows sky.dark"
NO_TRANSPARENCY = "no frame of the clear verdict had a transparency"
MOON_UNKNOWN = "no site is configured, so the moon flag is unknown"


@dataclass(frozen=True, slots=True)
class Event:
    """What a summary needs of one `event` record."""

    t_utc_ns: int
    kind: str
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HealthMark:
    """What a summary needs of one `health` record.

    `watching` is true when the scheduler was in a state of `WATCHING_STATES` without a fault of
    the camera, and `synchronized` is the record's `time_synchronized`.
    """

    t_utc_ns: int
    watching: bool
    synchronized: bool | None = None

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> HealthMark:
        """The mark of a `health` record, from the values of its row.

        The camera has a fault when the scheduler's component is `degraded`: the camera failed
        repeatedly, and the scheduler only retries. The record's `degraded` also holds the faults
        of the heater, `acquire`, and the SQM-LE reader, which do not stop the search or the
        measure, so it decides only for a row without the scheduler's component.
        """
        components = values.get("components")
        if isinstance(components, Mapping) and "scheduler" in components:
            fault = components["scheduler"] == "degraded"
        else:
            fault = bool(values.get("degraded"))
        synchronized = values.get("time_synchronized")
        return cls(
            t_utc_ns=int(values["t_utc_ns"]),
            watching=values.get("state") in WATCHING_STATES and not fault,
            synchronized=synchronized if isinstance(synchronized, bool) else None,
        )


@dataclass(frozen=True, slots=True)
class WindowMark:
    """What a summary needs of one `seeing_window` record."""

    t_utc_ns: int
    duration_s: float
    has_seeing: bool
    time_invalid: bool = False

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> WindowMark:
        """The mark of a `seeing_window` record, from the values of its row.

        A window has seeing when it holds `r0_cm` or `seeing_fwhm_arcsec`.
        """
        return cls(
            t_utc_ns=int(values["t_utc_ns"]),
            duration_s=float(values["duration_s"]),
            has_seeing=values.get("r0_cm") is not None
            or values.get("seeing_fwhm_arcsec") is not None,
            time_invalid="time_invalid" in (values.get("flags") or ()),
        )


@dataclass(frozen=True, slots=True)
class NightData:
    """The rows of one night, and of `LOOKBACK_NS` before it, each in time order.

    `visible_before` says whether Polaris was visible at the start of the night, from the summary
    of the previous night, or `None` when that summary does not exist. It counts only when the
    lookback holds no `polaris.*` event.
    """

    label: str
    start_utc_ns: int
    end_utc_ns: int
    events: Sequence[Event] = ()
    health: Sequence[HealthMark] = ()
    windows: Sequence[WindowMark] = ()
    visible_before: bool | None = None


@dataclass(frozen=True, slots=True)
class SummaryRules:
    """How a summary judges the gaps. See the module text."""

    max_gap_s: float = 300.0
    health_span_s: float = 120.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.max_gap_s) and self.max_gap_s >= 0):
            raise ValueError("max_gap_s must be a number of seconds, 0 or more")
        if not (math.isfinite(self.health_span_s) and self.health_span_s > 0):
            raise ValueError("health_span_s must be a number of seconds above 0")


@dataclass(frozen=True, slots=True)
class NightSummary:
    """The values of a `visibility_summary` record, without the key and the provenance."""

    night: str
    first_visible_utc_ns: int | None
    first_visible_sun_deg: float | None
    last_visible_utc_ns: int | None
    last_visible_sun_deg: float | None
    visible_hours: float
    seeing_hours: float
    first_censored: bool
    last_censored: bool
    dark_utc_ns: int | None
    dark_sun_deg: float | None
    dark_sky_mag_arcsec2: float | None
    clear_share: float | None
    transparency_median: float | None
    flags: tuple[str, ...]
    quality: dict[str, str]

    def fields(self) -> dict[str, Any]:
        """The values as keyword arguments of `VisibilitySummaryRecord`, `quality` included."""
        return {
            "night": self.night,
            "first_visible_utc_ns": self.first_visible_utc_ns,
            "first_visible_sun_deg": self.first_visible_sun_deg,
            "last_visible_utc_ns": self.last_visible_utc_ns,
            "last_visible_sun_deg": self.last_visible_sun_deg,
            "visible_hours": self.visible_hours,
            "seeing_hours": self.seeing_hours,
            "first_censored": self.first_censored,
            "last_censored": self.last_censored,
            "dark_utc_ns": self.dark_utc_ns,
            "dark_sun_deg": self.dark_sun_deg,
            "dark_sky_mag_arcsec2": self.dark_sky_mag_arcsec2,
            "clear_share": self.clear_share,
            "transparency_median": self.transparency_median,
            "flags": list(self.flags),
            "quality": dict(self.quality) or None,
        }


@dataclass(frozen=True, slots=True)
class _Measure:
    """One stretch of measure: from its start to its end, with the events that mark them."""

    start_ns: int
    end_ns: int
    start_event: Event | None  # None for a measure that the summary of the night before opened
    end_event: Event | None  # None for a measure that a crash, or the end of the rows, cut short


# --- When the station watched ------------------------------------------------------------------


def watched_stretches(health: Sequence[HealthMark], span_ns: int) -> list[tuple[int, int]]:
    """The stretches `[start, end)` in which the station watched the sky, merged, in time order."""
    stretches: list[tuple[int, int]] = []
    for index, mark in enumerate(health):
        if not mark.watching:
            continue
        end = mark.t_utc_ns + span_ns
        if index + 1 < len(health):
            end = min(end, health[index + 1].t_utc_ns)
        if end <= mark.t_utc_ns:
            continue
        if stretches and mark.t_utc_ns <= stretches[-1][1]:
            stretches[-1] = (stretches[-1][0], max(stretches[-1][1], end))
        else:
            stretches.append((mark.t_utc_ns, end))
    return stretches


def longest_gap_ns(stretches: Sequence[tuple[int, int]], start_ns: int, end_ns: int) -> int:
    """The longest part of `[start_ns, end_ns)` that no watched stretch covers."""
    longest = 0
    cursor = start_ns
    for first, last in stretches:
        if last <= cursor:
            continue
        if first >= end_ns:
            break
        longest = max(longest, min(first, end_ns) - cursor)
        cursor = max(cursor, last)
        if cursor >= end_ns:
            return longest
    return max(longest, end_ns - cursor)


def _watched_until(stretches: Sequence[tuple[int, int]], t_ns: int) -> int | None:
    """The end of the watched stretch that holds `t_ns`, or `None` when no stretch holds it."""
    for first, last in stretches:
        if first <= t_ns < last:
            return last
    return None


# --- When Polaris was visible ------------------------------------------------------------------


def _measures(data: NightData, stretches: Sequence[tuple[int, int]]) -> list[_Measure]:
    """The stretches of measure in the rows.

    A measure without a `polaris.hidden` ends where the station stopped watching (see the module
    text).
    """
    events = [e for e in data.events if e.t_utc_ns < data.end_utc_ns]
    polaris_before = any(
        e.kind in (VISIBLE_EVENT, HIDDEN_EVENT) and e.t_utc_ns < data.start_utc_ns for e in events
    )
    found: list[_Measure] = []
    opened: tuple[int, Event | None] | None = None
    if not polaris_before and data.visible_before:
        opened = (data.start_utc_ns, None)

    def close(end_ns: int, event: Event | None) -> None:
        nonlocal opened
        assert opened is not None
        start_ns, start_event = opened
        if event is None:
            # No polaris.hidden ended the measure: core crashed, or the rows end. The measure ended
            # where the station stopped watching, if it stopped before.
            watched = _watched_until(stretches, start_ns)
            if watched is not None and watched < end_ns:
                end_ns = watched
        found.append(_Measure(start_ns, max(start_ns, end_ns), start_event, event))
        opened = None

    for event in events:
        if event.kind == VISIBLE_EVENT:
            if opened is None:
                opened = (event.t_utc_ns, event)
        elif event.kind == HIDDEN_EVENT:
            if opened is not None:
                close(event.t_utc_ns, event)
        elif event.kind == CORE_STARTED_EVENT and opened is not None:
            # A start of core with a measure open: the one before crashed. The measure ended where
            # the health records stopped covering it, or at the start without them.
            close(event.t_utc_ns, None)
    if opened is not None:
        close(data.end_utc_ns, None)
    return found


# --- Values of the events ----------------------------------------------------------------------


def _number(detail: Mapping[str, Any], key: str, low: float, high: float) -> float | None:
    """A finite number of an event detail within `[low, high]`, or `None`."""
    value = detail.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        return None
    return number


def _sun_of(event: Event) -> float | None:
    return _number(event.detail, "sun_elevation_deg", -90.0, 90.0)


def _first_of(events: Sequence[Event], kind: str, start_ns: int, end_ns: int) -> Event | None:
    return next((e for e in events if e.kind == kind and start_ns <= e.t_utc_ns < end_ns), None)


# --- The summary -------------------------------------------------------------------------------


def summarize_night(
    data: NightData,
    rules: SummaryRules,
    *,
    sun_at: Callable[[int], float | None],
    moon_at: Callable[[int], bool | None],
) -> NightSummary:
    """The visibility summary of one night. See the module text.

    `sun_at` gives the Sun's elevation at a time, or `None` without a site. `moon_at` says whether
    the Moon was up and lit enough at a time to brighten the sky, or `None` without a site.
    """
    start, end = data.start_utc_ns, data.end_utc_ns
    span_ns = round(rules.health_span_s * NS_PER_S)
    gap_ns = round(rules.max_gap_s * NS_PER_S)
    stretches = watched_stretches(data.health, span_ns)
    quality: dict[str, str] = {}

    # The stretches of measure, cut to the night.
    in_night = [
        (max(m.start_ns, start), min(m.end_ns, end), m)
        for m in _measures(data, stretches)
        if m.end_ns > start and m.start_ns < end
    ]
    in_night = [(a, b, m) for a, b, m in in_night if b > a]
    visible_ns = sum(b - a for a, b, _ in in_night)

    first_ns: int | None = None
    first_sun: float | None = None
    first_event: Event | None = None
    last_ns: int | None = None
    last_sun: float | None = None
    last_event: Event | None = None
    visible_at_start = visible_at_end = False
    if in_night:
        a, _, measure = in_night[0]
        visible_at_start = measure.start_ns < start or measure.start_event is None
        first_ns = a
        first_event = None if visible_at_start else measure.start_event
        _, b, measure = max(in_night, key=lambda item: item[1])
        visible_at_end = measure.end_ns >= end and measure.end_event is None
        last_ns = b
        last_event = None if visible_at_end else measure.end_event
        first_sun = _sun_of(first_event) if first_event is not None else sun_at(first_ns)
        last_sun = _sun_of(last_event) if last_event is not None else sun_at(last_ns)
        for name, event, value in (
            ("first_visible_sun_deg", first_event, first_sun),
            ("last_visible_sun_deg", last_event, last_sun),
        ):
            if value is None:
                quality[name] = NO_EVENT_SUN if event is not None else NO_SITE
    else:
        for name in (
            "first_visible_utc_ns",
            "first_visible_sun_deg",
            "last_visible_utc_ns",
            "last_visible_sun_deg",
        ):
            quality[name] = NO_DETECTION

    first_censored = visible_at_start or (
        longest_gap_ns(stretches, start, end if first_ns is None else first_ns) > gap_ns
    )
    last_censored = visible_at_end or (
        longest_gap_ns(stretches, start if last_ns is None else last_ns, end) > gap_ns
    )

    seeing_s = sum(
        w.duration_s
        for w in data.windows
        if w.has_seeing and start <= w.t_utc_ns < end and math.isfinite(w.duration_s)
    )

    # The darkness and the clear verdict.
    dark = _first_of(data.events, DARK_EVENT, start, end)
    dark_sun = dark_mag = None
    if dark is None:
        for name in ("dark_utc_ns", "dark_sun_deg", "dark_sky_mag_arcsec2"):
            quality[name] = NO_DARK
    else:
        dark_sun = _sun_of(dark)
        if dark_sun is None:
            quality["dark_sun_deg"] = NO_EVENT_SUN
        dark_mag = _number(dark.detail, "sky_mag_arcsec2", -math.inf, math.inf)
        if dark_mag is None:
            quality["dark_sky_mag_arcsec2"] = "the sky.dark event holds no sky brightness"
    verdict = _first_of(data.events, CLEAR_VERDICT_EVENT, start, end)
    clear_share = transparency = None
    if verdict is None:
        quality["clear_share"] = quality["transparency_median"] = NO_VERDICT
    else:
        clear_share = _number(verdict.detail, "clear_share", 0.0, 1.0)
        if clear_share is None:
            quality["clear_share"] = "the sky.clear_verdict event holds no clear share"
        transparency = _number(verdict.detail, "transparency_median", 0.0, math.inf)
        if transparency is None:
            quality["transparency_median"] = NO_TRANSPARENCY

    # The flags. The Moon counts at the detections that events mark, and at `sky.dark`. A bound at
    # the start or the end of the night is no detection, and the split hour lies in the daytime.
    flags: list[str] = []
    moons = [moon_at(e.t_utc_ns) for e in (first_event, last_event, dark) if e is not None]
    if any(moons):
        flags.append("moon")
    elif any(value is None for value in moons):
        quality["flags"] = MOON_UNKNOWN
    night_events = [e for e in data.events if start <= e.t_utc_ns < end]
    if (
        any(e.kind == CLOCK_UNSYNCHRONIZED_EVENT for e in night_events)
        or any(m.synchronized is False for m in data.health if start <= m.t_utc_ns < end)
        or any(w.time_invalid for w in data.windows if start <= w.t_utc_ns < end)
    ):
        flags.append("time_invalid")
    if any(e.kind == SOLVE_REQUESTED_EVENT for e in night_events):
        flags.append("no_pointing")

    return NightSummary(
        night=data.label,
        first_visible_utc_ns=first_ns,
        first_visible_sun_deg=first_sun,
        last_visible_utc_ns=last_ns,
        last_visible_sun_deg=last_sun,
        visible_hours=round(visible_ns / _NS_PER_HOUR, 6),
        seeing_hours=round(seeing_s / 3600.0, 6),
        first_censored=first_censored,
        last_censored=last_censored,
        dark_utc_ns=None if dark is None else dark.t_utc_ns,
        dark_sun_deg=dark_sun,
        dark_sky_mag_arcsec2=dark_mag,
        clear_share=clear_share,
        transparency_median=transparency,
        flags=tuple(flags),
        quality=quality,
    )
