"""Statistics of the nightly visibility summaries: the Sun's elevation at the first and the last
detection of Polaris, by month and by transparency.

A censored detection is a bound, not the moment when Polaris appeared or vanished (see
`seeingmon.visibility.summary`). Polaris showed before a censored first detection of the evening,
when the Sun stood higher, and stayed after a censored last detection of the morning, until the Sun
stood higher. So the Sun's elevation of a censored detection is a lower bound of the elevation at
which Polaris shows. Dropping the censored nights would bias the statistics toward the nights that
the station watched from end to end, so each group keeps them, apart from the measured values.

The functions take any objects with the fields of a `visibility_summary` record, such as the
records themselves.
"""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Which = Literal["first", "last"]

# The default edges of the transparency bins. 0.6 is `[survey.transparency] transparency_flag`, the
# transparency below which a survey frame gets the `cloud` flag.
DEFAULT_TRANSPARENCY_EDGES: tuple[float, ...] = (0.6, 0.8, 0.9)
UNKNOWN_BIN = "unknown"


class NightValues(Protocol):
    """The fields of a `visibility_summary` record that the statistics read."""

    @property
    def night(self) -> str: ...

    @property
    def first_visible_utc_ns(self) -> int | None: ...

    @property
    def first_visible_sun_deg(self) -> float | None: ...

    @property
    def first_censored(self) -> bool: ...

    @property
    def last_visible_utc_ns(self) -> int | None: ...

    @property
    def last_visible_sun_deg(self) -> float | None: ...

    @property
    def last_censored(self) -> bool: ...

    @property
    def transparency_median(self) -> float | None: ...


@dataclass(frozen=True, slots=True)
class Side:
    """The Sun's elevation at one end of the visibility (the first or the last detection).

    `measured` holds the values of the nights whose detection is not censored, and `censored` the
    bounds of the censored ones, each sorted. `censored_without_value` counts the censored nights
    without a value (no detection, or no Sun elevation), and `without_sun` the detections that are
    not censored but have no Sun elevation (no site, or a clock that was not synchronized).
    """

    measured: tuple[float, ...] = ()
    censored: tuple[float, ...] = ()
    censored_without_value: int = 0
    without_sun: int = 0

    @property
    def n_censored(self) -> int:
        return len(self.censored) + self.censored_without_value

    @property
    def median(self) -> float | None:
        return statistics.median(self.measured) if self.measured else None


@dataclass(frozen=True, slots=True)
class Group:
    """The nights of one month or one transparency bin.

    `unseen` counts the nights without any detection that the station watched (neither end
    censored): Polaris stayed hidden, for example behind clouds.
    """

    key: str
    nights: int
    unseen: int
    first: Side
    last: Side


def _values(night: NightValues, which: Which) -> tuple[int | None, float | None, bool]:
    if which == "first":
        return night.first_visible_utc_ns, night.first_visible_sun_deg, night.first_censored
    return night.last_visible_utc_ns, night.last_visible_sun_deg, night.last_censored


def side_of(nights: Iterable[NightValues], which: Which) -> Side:
    """The Sun's elevation at the first or the last detection of the nights."""
    measured: list[float] = []
    censored: list[float] = []
    censored_without_value = without_sun = 0
    for night in nights:
        time_ns, sun, is_censored = _values(night, which)
        if is_censored:
            if time_ns is not None and sun is not None:
                censored.append(sun)
            else:
                censored_without_value += 1
        elif time_ns is not None:
            if sun is None:
                without_sun += 1
            else:
                measured.append(sun)
    return Side(
        tuple(sorted(measured)), tuple(sorted(censored)), censored_without_value, without_sun
    )


def is_unseen(night: NightValues) -> bool:
    """Whether the station watched the night and never detected Polaris."""
    return (
        night.first_visible_utc_ns is None and not night.first_censored and not night.last_censored
    )


def group_nights(nights: Iterable[NightValues], key: Callable[[NightValues], str]) -> list[Group]:
    """The nights grouped by `key`, in the order of the keys."""
    groups: dict[str, list[NightValues]] = {}
    for night in nights:
        groups.setdefault(key(night), []).append(night)
    return [
        Group(
            key=name,
            nights=len(members),
            unseen=sum(1 for night in members if is_unseen(night)),
            first=side_of(members, "first"),
            last=side_of(members, "last"),
        )
        for name, members in groups.items()
    ]


def by_month(nights: Iterable[NightValues]) -> list[Group]:
    """The nights grouped by the month of their label (`YYYY-MM`), in time order."""
    return sorted(group_nights(nights, lambda night: night.night[:7]), key=lambda g: g.key)


def transparency_bins(edges: Sequence[float]) -> list[str]:
    """The labels of the bins that `edges` make, from the most opaque, then `unknown`."""
    if not edges:
        return ["all", UNKNOWN_BIN]
    labels = [f"< {edges[0]:g}"]
    labels += [f"{low:g} to {high:g}" for low, high in itertools.pairwise(edges)]
    labels.append(f">= {edges[-1]:g}")
    return [*labels, UNKNOWN_BIN]


def transparency_bin(value: float | None, edges: Sequence[float]) -> str:
    """The label of the bin of a median transparency. `None` falls in `unknown`."""
    labels = transparency_bins(edges)
    if value is None:
        return UNKNOWN_BIN
    index = sum(1 for edge in edges if value >= edge)
    return labels[index]


def by_transparency(nights: Iterable[NightValues], edges: Sequence[float]) -> list[Group]:
    """The nights grouped by the bin of their median transparency, from the most opaque.

    `edges` must rise. A night without a clear verdict, or whose verdict had no transparency, falls
    in `unknown`.
    """
    if any(high <= low for low, high in itertools.pairwise(edges)):
        raise ValueError("the edges of the transparency bins must rise")
    order = {label: index for index, label in enumerate(transparency_bins(edges))}
    groups = group_nights(nights, lambda night: transparency_bin(night.transparency_median, edges))
    return sorted(groups, key=lambda g: order[g.key])
