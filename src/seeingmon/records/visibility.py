"""The record of the visibility of Polaris: one summary for each night.

The visibility lane owns this module. Add a field at the end of a class, and keep the rules in the
module documentation of `seeingmon.records.base`.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar

from pydantic import field_validator

from seeingmon.records.base import Record, quantity

TIME_INVALID = "The clock was not synchronized for part of the night, so the times may be off."

VISIBILITY_FLAGS: dict[str, str] = {
    "moon": (
        "The Moon was up and lit enough to brighten the sky at the first detection, at the last "
        "detection, or at `sky.dark`."
    ),
    "time_invalid": TIME_INVALID,
    "no_pointing": (
        "No pointing solution existed for part of the night, so the search could not run."
    ),
}


class VisibilitySummaryRecord(Record):
    """When Polaris was visible in one night, and how dark and clear the night was.

    `t_utc_ns` is the start of the night, at `[survey] night_split_utc_hour`, so the key of a
    night never changes. A censored detection is a bound: Polaris was already visible when the
    station began to watch, or still visible when it stopped. A missing value is `null`, and
    `quality` says why.
    """

    record_type: ClassVar[str] = "visibility_summary"

    night: str = quantity(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        example="2026-10-01",
        definition="The UTC date that labels the night, in `YYYY-MM-DD` form.",
    )
    first_visible_utc_ns: int | None = quantity(
        unit="ns",
        default=None,
        definition=(
            "The first time in the night that Polaris was visible: the first `polaris.visible`, or "
            "the start of the night when Polaris was visible then, in nanoseconds since the Unix "
            "epoch."
        ),
    )
    first_visible_sun_deg: float | None = quantity(
        unit="deg",
        ge=-90,
        le=90,
        default=None,
        definition="The Sun's elevation at `first_visible_utc_ns`, in degrees.",
    )
    last_visible_utc_ns: int | None = quantity(
        unit="ns",
        default=None,
        definition=(
            "The last time in the night that Polaris was visible: the last end of measure, or the "
            "end of the night when Polaris was visible then, in nanoseconds since the Unix epoch."
        ),
    )
    last_visible_sun_deg: float | None = quantity(
        unit="deg",
        ge=-90,
        le=90,
        default=None,
        definition="The Sun's elevation at `last_visible_utc_ns`, in degrees.",
    )
    visible_hours: float = quantity(
        unit="h",
        ge=0,
        definition="The time in the night that the fast stream measured Polaris, in hours.",
    )
    seeing_hours: float = quantity(
        unit="h",
        ge=0,
        definition="The time in the night that seeing windows with a seeing value cover, in hours.",
    )
    first_censored: bool = quantity(
        definition=(
            "Whether the first detection is a bound, because Polaris was visible when the night "
            "started or the station did not watch the whole time before the detection."
        ),
    )
    last_censored: bool = quantity(
        definition=(
            "Whether the last detection is a bound, because Polaris was visible when the night "
            "ended or the station did not watch the whole time after the detection."
        ),
    )
    dark_utc_ns: int | None = quantity(
        unit="ns",
        default=None,
        definition="The time of the `sky.dark` event of the night, in nanoseconds since the epoch.",
    )
    dark_sun_deg: float | None = quantity(
        unit="deg",
        ge=-90,
        le=90,
        default=None,
        definition="The Sun's elevation at `sky.dark`, in degrees.",
    )
    dark_sky_mag_arcsec2: float | None = quantity(
        unit="mag/arcsec^2",
        default=None,
        definition=(
            "The sky brightness that `sky.dark` reports, in magnitudes per square arcsecond."
        ),
    )
    clear_share: float | None = quantity(
        ge=0,
        le=1,
        default=None,
        definition="The share of clear frames that the `sky.clear_verdict` of the night reports.",
    )
    transparency_median: float | None = quantity(
        ge=0,
        default=None,
        definition="The median transparency that the `sky.clear_verdict` of the night reports.",
    )
    flags: list[str] = quantity(
        default_factory=list,
        codes=VISIBILITY_FLAGS,
        definition="The conditions that apply to the night, as documented codes.",
    )

    @field_validator("night")
    @classmethod
    def _check_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value
