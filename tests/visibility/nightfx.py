"""Rows of a synthetic night for the tests of the visibility summary.

The night is `2026-01-01` with the default split hour (12:00 UTC), so it runs from 12:00 UTC on 1
January to 12:00 UTC on 2 January. `at("17:30")` is a time in the evening, and
`at("06:00", day=2)` is one in the morning. The station writes a health record every minute, as
`core` does.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.visibility.summary import (
    CORE_STARTED_EVENT,
    HIDDEN_EVENT,
    VISIBLE_EVENT,
    Event,
    HealthMark,
    NightData,
    SummaryRules,
    WindowMark,
)

LABEL = "2026-01-01"
START = iso_to_utc_ns("2026-01-01T12:00:00Z")
END = iso_to_utc_ns("2026-01-02T12:00:00Z")
MINUTE = 60 * NS_PER_S
HOUR = 60 * MINUTE
RULES = SummaryRules(max_gap_s=300.0, health_span_s=120.0)


def at(clock: str, *, day: int = 1) -> int:
    """A UTC time of the night, such as `at("17:30")` or `at("06:00", day=2)`."""
    return iso_to_utc_ns(f"2026-01-0{day}T{clock}:00Z")


def visible(t_ns: int, sun: float | None = -5.0) -> Event:
    return Event(t_ns, VISIBLE_EVENT, {"sun_elevation_deg": sun})


def hidden(t_ns: int, sun: float | None = -6.0, reason: str = "star_missing") -> Event:
    return Event(t_ns, HIDDEN_EVENT, {"sun_elevation_deg": sun, "reason": reason})


def started(t_ns: int) -> Event:
    return Event(t_ns, CORE_STARTED_EVENT, {"instance": "abc"})


def health_values(
    t_ns: int,
    *,
    state: str = "auto",
    synchronized: bool | None = True,
    components: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The values of a `health` row, with the components that `core` writes (all `ok`)."""
    parts = {"core": "ok", "scheduler": "ok", "camera": "ok", **(components or {})}
    return {
        "t_utc_ns": t_ns,
        "state": state,
        "degraded": "failed" in parts.values() or parts["scheduler"] == "degraded",
        "components": parts,
        "time_synchronized": synchronized,
    }


def health(
    start_ns: int,
    end_ns: int,
    *,
    state: str = "auto",
    synchronized: bool | None = True,
    components: dict[str, str] | None = None,
) -> list[HealthMark]:
    """A health record every minute in `[start_ns, end_ns)`, mapped as `core` maps the rows."""
    return [
        HealthMark.from_values(
            health_values(t, state=state, synchronized=synchronized, components=components)
        )
        for t in range(start_ns, end_ns, MINUTE)
    ]


def windows(start_ns: int, end_ns: int, *, seconds: float = 60.0) -> list[WindowMark]:
    """Seeing windows with a value that cover `[start_ns, end_ns)`."""
    step = round(seconds * NS_PER_S)
    return [WindowMark(t, seconds, True) for t in range(start_ns, end_ns, step)]


def night(
    events: Iterable[Event] = (),
    marks: Sequence[HealthMark] | None = None,
    window_marks: Sequence[WindowMark] = (),
    **more: Any,
) -> NightData:
    """The data of the night. Without `marks`, the station watches the whole day before and the
    whole night."""
    if marks is None:
        marks = health(START - 24 * HOUR, END)
    return NightData(
        label=LABEL,
        start_utc_ns=START,
        end_utc_ns=END,
        events=sorted(events, key=lambda e: e.t_utc_ns),
        health=sorted(marks, key=lambda m: m.t_utc_ns),
        windows=window_marks,
        **more,
    )
