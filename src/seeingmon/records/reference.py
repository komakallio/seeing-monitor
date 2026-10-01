"""The record of a reference instrument: a reading of the fixed SQM-LE or a handheld meter.

The lane that owns the reference readers owns this module. Add a field at the end of a class,
and keep the rules in the module documentation of `seeingmon.records.base`.
"""

from __future__ import annotations

from typing import ClassVar

from seeingmon.records.base import Record, quantity

REFERENCE_SOURCES: dict[str, str] = {
    "fixed": "A fixed instrument that the system polls, such as the SQM-LE.",
    "manual": "A reading that you enter by hand, such as one from a handheld SQM-L.",
}


class ReferenceRecord(Record):
    """One reading of a reference instrument.

    `t_utc_ns` is the UTC time of the reading. The system compares the readings with its own
    sky brightness to fit the dome loss and the difference in altitude.
    """

    record_type: ClassVar[str] = "reference"

    instrument: str = quantity(
        min_length=1,
        example="sqm-le",
        definition="The name of the instrument that took the reading.",
    )
    source: str = quantity(
        codes=REFERENCE_SOURCES,
        definition="Whether the reading comes from a fixed instrument or from a manual entry.",
    )
    value_mag_arcsec2: float = quantity(
        unit="mag/arcsec^2",
        definition=(
            "The sky brightness that the instrument reports, in magnitudes per square arcsecond."
        ),
    )
    temperature_c: float | None = quantity(
        unit="degC",
        default=None,
        definition="The temperature that the instrument reports, in degrees Celsius.",
    )
    altitude_deg: float | None = quantity(
        unit="deg",
        ge=-90,
        le=90,
        default=None,
        definition="The altitude that the instrument pointed at, in degrees.",
    )
    azimuth_deg: float | None = quantity(
        unit="deg",
        ge=0,
        le=360,
        default=None,
        definition=(
            "The azimuth that the instrument pointed at, in degrees from north through east."
        ),
    )
    note: str | None = quantity(default=None, definition="A free-text note about the reading.")
