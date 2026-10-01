"""Which night a time belongs to.

The survey groups its results by night: the star epochs summarize one night, and the transparency
reference counts nights. A night runs from one hour of the UTC day to the same hour of the next
day. The hour (`night_split_utc_hour`, 12 by default) must fall in the daytime of your site,
where nobody takes survey frames. The code needs no site: you choose the hour in the
configuration, and no latitude or longitude enters the repository.
"""

from __future__ import annotations

from datetime import UTC, datetime

from seeingmon.clock import NS_PER_S, utc_ns_to_datetime

_SECONDS_PER_HOUR = 3600


def night_label(t_utc_ns: int, split_utc_hour: float = 12.0) -> str:
    """The `YYYY-MM-DD` label of the night that holds a time: the UTC date of its start.

    With the default split at 12:00 UTC, the evening of 1 October and the early morning of 2
    October both belong to the night `2026-10-01`.
    """
    shifted = t_utc_ns - round(split_utc_hour * _SECONDS_PER_HOUR * NS_PER_S)
    return utc_ns_to_datetime(shifted).strftime("%Y-%m-%d")


def night_start_utc_ns(label: str, split_utc_hour: float = 12.0) -> int:
    """The first instant of the night with a label, in nanoseconds since the Unix epoch."""
    date = datetime.strptime(label, "%Y-%m-%d").replace(tzinfo=UTC)
    return (round(date.timestamp()) + round(split_utc_hour * _SECONDS_PER_HOUR)) * NS_PER_S
