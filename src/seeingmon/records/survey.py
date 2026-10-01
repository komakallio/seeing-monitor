"""The records of the survey path: survey frames, sky quality, pointing, and star lists.

The survey-path lane owns this module. Add a field at the end of a class, and keep the rules in
the module documentation of `seeingmon.records.base`.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import PurePosixPath
from typing import ClassVar, Self

from pydantic import field_validator, model_validator

from seeingmon.records.base import Record, quantity

TIME_INVALID = "The clock was not synchronized, so `t_utc_ns` is not trustworthy."

SKY_QUALITY_FLAGS: dict[str, str] = {
    "cloud": "Clouds reduce the number of detected stars.",
    "twilight": "The Sun was less than 18 degrees below the horizon.",
    "moon": "The Moon was above the horizon and bright enough to raise the sky background.",
    "dew": "The star widths and the transparency indicate dew on the optics.",
    "dark_due": "The dark library misses the current temperature or is older than 6 months.",
    "time_invalid": TIME_INVALID,
}

POINTING_FLAGS: dict[str, str] = {
    "unsolved": "The solver found no solution, so the geometry fields are `null`.",
    "few_stars": "Too few stars matched for a reliable fit.",
    "moved": "The offset from the reference solution exceeds the configured limit.",
    "roll_undefined": "The center lies on the pole, so the roll is undefined.",
    "time_invalid": TIME_INVALID,
}

_DRIVE_LETTER = re.compile(r"[A-Za-z]:")
_STAR_VALUE_BYTES = 4  # one little-endian float32


class SurveyFrameRecord(Record):
    """One survey exposure.

    `t_utc_ns` is the UTC time at the middle of the exposure. The other survey records
    (`sky_quality`, `pointing`, and `star_list`) refer to the same time.
    """

    record_type: ClassVar[str] = "survey_frame"

    exposure_s: float = quantity(unit="s", gt=0, definition="The exposure time, in seconds.")
    gain: int = quantity(
        ge=0, definition="The camera gain setting, in the units of the camera driver."
    )
    readout_mode: str = quantity(
        min_length=1,
        definition="The name of the readout mode in the hardware profile, such as `bin2`.",
    )
    sensor_temperature_c: float | None = quantity(
        unit="degC",
        default=None,
        definition="The sensor temperature during the exposure, in degrees Celsius.",
    )
    image_ref: str | None = quantity(
        default=None,
        min_length=1,
        definition=(
            "A reference to the stored image file, as a path relative to the data directory, "
            "or `null` when the system did not store the frame."
        ),
    )
    n_detected: int | None = quantity(
        ge=0, default=None, definition="The number of stars that detection found in the frame."
    )
    n_saturated: int | None = quantity(
        ge=0,
        default=None,
        definition="The number of detected stars that have saturated pixels.",
    )
    background_dn: float | None = quantity(
        unit="DN",
        default=None,
        definition="The sigma-clipped median of the sky background, in digital numbers.",
    )

    @field_validator("image_ref")
    @classmethod
    def _check_relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return value
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value or _DRIVE_LETTER.match(value):
            raise ValueError("image_ref must be a relative path with forward slashes and no `..`")
        return value


class SkyQualityRecord(Record):
    """The sky brightness and the transparency from one survey frame.

    `t_utc_ns` is the UTC time at the middle of the exposure. A value that the frame cannot
    support is `null`, and `quality` says why.
    """

    record_type: ClassVar[str] = "sky_quality"

    sky_mag_arcsec2: float | None = quantity(
        unit="mag/arcsec^2",
        default=None,
        definition=(
            "The sky surface brightness in the camera band, in magnitudes per square arcsecond."
        ),
    )
    sky_mag_arcsec2_v: float | None = quantity(
        unit="mag/arcsec^2",
        default=None,
        definition=(
            "The sky surface brightness converted to the V band, in magnitudes per square "
            "arcsecond."
        ),
    )
    zero_point_mag: float | None = quantity(
        unit="mag",
        default=None,
        definition=(
            "The photometric zero point, which is the magnitude of a star that gives one "
            "electron per second, in magnitudes."
        ),
    )
    zero_point_rms_mag: float | None = quantity(
        unit="mag",
        ge=0,
        default=None,
        definition="The scatter of the matched stars around the zero-point fit, in magnitudes.",
    )
    color_term: float | None = quantity(
        default=None,
        definition=(
            "The fitted coefficient of the color term, in magnitudes per magnitude of the "
            "BP-RP color."
        ),
    )
    transparency: float | None = quantity(
        ge=0,
        default=None,
        definition=(
            "The atmospheric transmission relative to the clearest nights, normally from 0 to 1."
        ),
    )
    cloud_fraction: float | None = quantity(
        ge=0,
        le=1,
        default=None,
        definition="The share of the expected catalog stars that the frame missed, from 0 to 1.",
    )
    limiting_mag: float | None = quantity(
        unit="mag",
        default=None,
        definition="The magnitude at which the frame detects half of the stars, in magnitudes.",
    )
    n_stars_used: int = quantity(
        ge=0, definition="The number of matched stars in the zero-point fit."
    )
    sky_rate_e_per_s_arcsec2: float | None = quantity(
        unit="e-/s/arcsec^2",
        ge=0,
        default=None,
        definition="The sky signal, in electrons per second per square arcsecond.",
    )
    dark_model_version: str | None = quantity(
        default=None,
        min_length=1,
        definition="The version of the dark model that the pipeline subtracted.",
    )
    flags: list[str] = quantity(
        default_factory=list,
        codes=SKY_QUALITY_FLAGS,
        definition="The conditions that apply to the measurement, as documented codes.",
    )


class PointingRecord(Record):
    """The pointing solution of one survey frame.

    `t_utc_ns` is the UTC time at the middle of the exposure. The attitude is a rotation matrix,
    so analysis can express it in ICRS or in the frame of date. The plate scale belongs to the
    readout mode in `readout_mode`. A frame that the solver cannot solve has `null` in every
    geometry field and the `unsolved` flag.
    """

    record_type: ClassVar[str] = "pointing"

    center_ra_deg: float | None = quantity(
        unit="deg",
        ge=0,
        le=360,
        default=None,
        definition="The right ascension of the field center, in degrees.",
    )
    center_dec_deg: float | None = quantity(
        unit="deg",
        ge=-90,
        le=90,
        default=None,
        definition="The declination of the field center, in degrees.",
    )
    roll_deg: float | None = quantity(
        unit="deg",
        default=None,
        definition=(
            "The roll angle, which is the position angle of the direction to the pole, in degrees."
        ),
    )
    plate_scale_arcsec_px: float | None = quantity(
        unit="arcsec/px",
        gt=0,
        default=None,
        definition="The plate scale in the readout mode of the frame, in arcseconds per pixel.",
    )
    attitude: list[float] | None = quantity(
        min_length=9,
        max_length=9,
        default=None,
        definition="The rotation matrix of the attitude, as nine numbers in row-major order.",
    )
    offset_arcmin: float | None = quantity(
        unit="arcmin",
        ge=0,
        default=None,
        definition="The offset of the field center from the reference solution, in arcminutes.",
    )
    solve_rms_arcsec: float | None = quantity(
        unit="arcsec",
        ge=0,
        default=None,
        definition="The RMS residual of the matched stars after the fit, in arcseconds.",
    )
    n_matched: int = quantity(
        ge=0, definition="The number of stars that the fit matched to the catalog."
    )
    focus_fwhm_px: float | None = quantity(
        unit="px",
        ge=0,
        default=None,
        definition="The median FWHM of the unsaturated stars, in pixels.",
    )
    polaris_x_px: float | None = quantity(
        unit="px",
        default=None,
        definition="The x coordinate of Polaris predicted at `t_utc_ns`, in sensor pixels.",
    )
    polaris_y_px: float | None = quantity(
        unit="px",
        default=None,
        definition="The y coordinate of Polaris predicted at `t_utc_ns`, in sensor pixels.",
    )
    readout_mode: str = quantity(
        min_length=1,
        definition="The name of the readout mode in the hardware profile, such as `bin2`.",
    )
    solver: str = quantity(
        min_length=1,
        definition="The name of the plate solver that produced the initial solution.",
    )
    solve_time_s: float | None = quantity(
        unit="s",
        ge=0,
        default=None,
        definition="The time that the solver took, in seconds.",
    )
    reference_id: str | None = quantity(
        default=None,
        min_length=1,
        definition="The ID of the reference solution that `offset_arcmin` refers to.",
    )
    flags: list[str] = quantity(
        default_factory=list,
        codes=POINTING_FLAGS,
        definition="The conditions that apply to the solution, as documented codes.",
    )


def _check_star_rows(n_stars: int, columns: list[str], data: bytes) -> None:
    """Check that the columns are distinct and that the data holds the rows that it claims."""
    if len(set(columns)) != len(columns) or "" in columns:
        raise ValueError("columns must be distinct and not empty")
    expected = n_stars * len(columns) * _STAR_VALUE_BYTES
    if len(data) != expected:
        raise ValueError(
            f"data needs {expected} bytes for {n_stars} stars and {len(columns)} columns "
            f"of float32, not {len(data)}"
        )


class StarListRecord(Record):
    """The stars that one survey step detected.

    The list holds the matched stars brighter than G = 11 and all unmatched detections.
    `t_utc_ns` is the UTC time at the middle of the exposure. The system keeps star lists for
    1 year, so a better algorithm can reprocess the photometry and the astrometry.
    """

    record_type: ClassVar[str] = "star_list"
    retention_days: ClassVar[int | None] = 365

    n_stars: int = quantity(ge=0, definition="The number of stars in the list.")
    columns: list[str] = quantity(
        definition=(
            "The names of the columns of `data`, in order, where every value is a little-endian "
            "float32."
        ),
    )
    data: bytes = quantity(
        definition=(
            "The star rows as little-endian float32 values in row-major order, with one value "
            "for each name in `columns`."
        ),
    )
    catalog: str | None = quantity(
        default=None,
        min_length=1,
        definition="The catalog and release that the matched stars come from, such as `gaia-dr3`.",
    )

    @model_validator(mode="after")
    def _check_rows(self) -> Self:
        _check_star_rows(self.n_stars, self.columns, self.data)
        return self


class StarEpochRecord(Record):
    """The nightly summary of the stars that the survey matched.

    `t_utc_ns` is the start of the night. Each row of `data` describes one star: its mean
    position offset, its mean magnitude, its scatter, and the number of frames that contribute.
    """

    record_type: ClassVar[str] = "star_epoch"

    night: str = quantity(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        definition="The UTC date that labels the night, in `YYYY-MM-DD` form.",
    )
    n_stars: int = quantity(ge=0, definition="The number of stars in the summary.")
    n_frames: int = quantity(
        ge=0, definition="The number of survey frames that the summary averages."
    )
    columns: list[str] = quantity(
        definition=(
            "The names of the columns of `data`, in order, where every value is a little-endian "
            "float32."
        ),
    )
    data: bytes = quantity(
        definition=(
            "The star rows as little-endian float32 values in row-major order, with one value "
            "for each name in `columns`."
        ),
    )

    @field_validator("night")
    @classmethod
    def _check_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _check_rows(self) -> Self:
        _check_star_rows(self.n_stars, self.columns, self.data)
        return self
