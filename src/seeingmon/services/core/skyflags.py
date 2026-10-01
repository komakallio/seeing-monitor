"""The flags that only `core` can set on a `sky_quality` record: the Moon and the dew.

A `sky_quality` record carries the flags `cloud`, `twilight`, `moon`, `dew`, `dark_due`, and
`time_invalid`. The survey pipeline sets `cloud`, `dark_due`, and `time_invalid`, and the scheduler
sets `twilight` (and `cloud` and `time_invalid` again) from its own gates when it takes a survey
result. Two flags need knowledge that neither has:

- **`moon`.** The survey path knows no site, and the scheduler has no Moon. `core` has the site and
  the Moon (`seeingmon.services.core.moon`), so it sets the flag when the Moon was above the
  horizon and lit enough to raise the sky background at the time of the frame. A frame with
  `time_invalid` gets no `moon` flag, because the time of the frame is wrong then.
- **`dew`.** Only the heater controller knows the dew point. The flag says that the optics were
  within a margin of the dew point, so dew on the lens or the window can have dimmed the stars. A
  station without a heater controller, or one whose sensors report nothing, never sets it.

`SkyFlagWriter` wraps the record writer that `core` gives the scheduler. It passes every record on
unchanged except a `sky_quality` record, which gets the flags that apply, sorted with the ones that
it has. The flags describe the moment of the frame, and the survey analysis finishes a frame within
seconds, so the heater state at the time of writing stands for the state at the time of the frame.
"""

from __future__ import annotations

import logging
from typing import Protocol

from seeingmon.analysis import RecordWriter
from seeingmon.records import Record
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.scheduler.config import SiteConfig
from seeingmon.services.core.moon import moon_elevation_deg, moon_illumination
from seeingmon.services.core.settings import SkyFlagSettings

_log = logging.getLogger(__name__)


class DewReading(Protocol):
    """What the flag needs from the status of the heater controller (`HeaterStatus` fits)."""

    @property
    def enabled(self) -> bool: ...

    @property
    def dew_point_c(self) -> float | None: ...

    @property
    def ambient_c(self) -> float | None: ...

    @property
    def optics_c(self) -> float | None: ...


class DewSource(Protocol):
    """The heater controller, as far as the flag needs it."""

    def status(self) -> DewReading: ...


def moon_flag(t_utc_ns: int, site: SiteConfig | None, settings: SkyFlagSettings) -> bool:
    """Whether the Moon was above the horizon and lit enough at a time. `False` without a site."""
    if site is None:
        return False
    if moon_elevation_deg(t_utc_ns, site.latitude_deg, site.longitude_deg) <= (
        settings.moon_min_elevation_deg
    ):
        return False
    return moon_illumination(t_utc_ns) >= settings.moon_min_illumination


def dew_flag(reading: DewReading | None, settings: SkyFlagSettings) -> bool:
    """Whether the optics are within the margin of the dew point. `False` when it is unknown."""
    if reading is None or not reading.enabled or reading.dew_point_c is None:
        return False
    surface = reading.optics_c if reading.optics_c is not None else reading.ambient_c
    if surface is None:
        return False
    return surface - reading.dew_point_c <= settings.dew_margin_c


class SkyFlagWriter:
    """A `RecordWriter` that adds the `moon` and `dew` flags to `sky_quality` records."""

    def __init__(
        self,
        writer: RecordWriter,
        *,
        site: SiteConfig | None,
        heater: DewSource | None,
        settings: SkyFlagSettings,
    ) -> None:
        self._writer = writer
        self._site = site
        self._heater = heater
        self._settings = settings

    def write(self, record: Record) -> None:
        if isinstance(record, SkyQualityRecord):
            record = self.flagged(record)
        self._writer.write(record)

    def flagged(self, record: SkyQualityRecord) -> SkyQualityRecord:
        """The record with the flags that apply added to the ones it has."""
        added: list[str] = []
        if "time_invalid" not in record.flags and moon_flag(
            record.t_utc_ns, self._site, self._settings
        ):
            added.append("moon")
        if dew_flag(self._reading(), self._settings):
            added.append("dew")
        merged = sorted({*record.flags, *added})
        if merged == sorted(record.flags):
            return record
        return record.model_copy(update={"flags": merged})

    def _reading(self) -> DewReading | None:
        if self._heater is None:
            return None
        try:
            return self._heater.status()
        except Exception:
            _log.exception("the heater status is not available, so there is no dew flag")
            return None
