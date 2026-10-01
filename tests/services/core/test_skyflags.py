"""The `moon` and `dew` flags that `core` adds to the sky quality records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from seeingmon.clock import iso_to_utc_ns
from seeingmon.records import EventRecord, Record
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.scheduler.config import SiteConfig
from seeingmon.services.core.settings import SkyFlagSettings
from seeingmon.services.core.skyflags import SkyFlagWriter, dew_flag, moon_flag

SITE = SiteConfig(latitude_deg=55.0, longitude_deg=0.0)
SETTINGS = SkyFlagSettings()

# The full Moon of 2026-01-03 stands high at midnight on the prime meridian, and the new Moon of
# 2026-01-18 is at the Sun's side of the sky, so it is never up at night.
FULL_MOON_NIGHT = iso_to_utc_ns("2026-01-04T00:00:00Z")
NEW_MOON_NIGHT = iso_to_utc_ns("2026-01-19T00:00:00Z")
FULL_MOON_BELOW_HORIZON = iso_to_utc_ns("2026-01-03T14:00:00Z")  # the Moon is below the horizon


@dataclass(frozen=True)
class Reading:
    enabled: bool = True
    dew_point_c: float | None = 2.0
    ambient_c: float | None = 6.0
    optics_c: float | None = None


class Heater:
    def __init__(self, reading: Reading | Exception) -> None:
        self.reading = reading

    def status(self) -> Reading:
        if isinstance(self.reading, Exception):
            raise self.reading
        return self.reading


class Sink:
    def __init__(self) -> None:
        self.records: list[Record] = []

    def write(self, record: Record) -> None:
        self.records.append(record)


def sky_quality(t_utc_ns: int, flags: list[str] | None = None) -> SkyQualityRecord:
    return SkyQualityRecord(
        station_id="test",
        t_utc_ns=t_utc_ns,
        profile_id="profile",
        provenance={"algo": "sky-1"},
        n_stars_used=20,
        flags=flags or [],
    )


def writer(
    sink: Sink,
    *,
    site: SiteConfig | None = SITE,
    heater: Heater | None = None,
    settings: SkyFlagSettings = SETTINGS,
) -> SkyFlagWriter:
    return SkyFlagWriter(sink, site=site, heater=heater, settings=settings)


class TestMoonFlag:
    def test_a_bright_moon_above_the_horizon_counts(self) -> None:
        assert moon_flag(FULL_MOON_NIGHT, SITE, SETTINGS)

    def test_a_new_moon_gives_no_flag_however_high_it_stands(self) -> None:
        assert not moon_flag(NEW_MOON_NIGHT, SITE, SETTINGS)

    def test_a_moon_below_the_horizon_gives_no_flag_however_bright_it_is(self) -> None:
        assert not moon_flag(FULL_MOON_BELOW_HORIZON, SITE, SETTINGS)

    def test_without_a_site_there_is_no_flag(self) -> None:
        assert not moon_flag(FULL_MOON_NIGHT, None, SETTINGS)

    def test_the_limits_are_settings(self) -> None:
        strict = SkyFlagSettings(moon_min_illumination=1.0, moon_min_elevation_deg=89.0)
        assert not moon_flag(FULL_MOON_NIGHT, SITE, strict)
        lenient = SkyFlagSettings(moon_min_illumination=0.0, moon_min_elevation_deg=-90.0)
        assert moon_flag(NEW_MOON_NIGHT, SITE, lenient)


class TestDewFlag:
    def test_optics_close_to_the_dew_point_count(self) -> None:
        assert dew_flag(Reading(dew_point_c=5.5, ambient_c=6.0), SETTINGS)  # 0.5 above it
        assert dew_flag(Reading(dew_point_c=7.0, ambient_c=6.0), SETTINGS)  # below it

    def test_optics_well_above_the_dew_point_do_not(self) -> None:
        assert not dew_flag(Reading(dew_point_c=2.0, ambient_c=6.0), SETTINGS)

    def test_the_optics_sensor_counts_before_the_air(self) -> None:
        warm_air_cold_optics = Reading(dew_point_c=4.0, ambient_c=9.0, optics_c=4.5)
        assert dew_flag(warm_air_cold_optics, SETTINGS)
        cold_air_warm_optics = Reading(dew_point_c=4.0, ambient_c=4.2, optics_c=9.0)
        assert not dew_flag(cold_air_warm_optics, SETTINGS)

    @pytest.mark.parametrize(
        "reading",
        [
            None,
            Reading(enabled=False),
            Reading(dew_point_c=None),
            Reading(ambient_c=None, optics_c=None),
        ],
    )
    def test_unknown_is_not_dew(self, reading: Reading | None) -> None:
        assert not dew_flag(reading, SETTINGS)


class TestTheWriter:
    def test_a_sky_quality_record_gets_the_flags_that_apply_in_sorted_order(self) -> None:
        sink = Sink()
        heater = Heater(Reading(dew_point_c=5.8, ambient_c=6.0))
        writer(sink, heater=heater).write(sky_quality(FULL_MOON_NIGHT, ["twilight", "cloud"]))
        (record,) = sink.records
        assert isinstance(record, SkyQualityRecord)
        assert record.flags == ["cloud", "dew", "moon", "twilight"]

    def test_a_record_without_a_reason_for_a_flag_passes_unchanged(self) -> None:
        sink = Sink()
        original = sky_quality(NEW_MOON_NIGHT, ["cloud"])
        writer(sink, heater=Heater(Reading())).write(original)
        assert sink.records == [original]
        assert sink.records[0] is original

    def test_other_records_pass_through_untouched(self) -> None:
        sink = Sink()
        event = EventRecord(
            station_id="test",
            t_utc_ns=FULL_MOON_NIGHT,
            profile_id="profile",
            provenance={"source": "local"},
            level="info",
            kind="test.event",
            message="An event.",
        )
        writer(sink, heater=Heater(Reading(dew_point_c=6.0))).write(event)
        assert sink.records == [event]

    def test_a_frame_with_an_untrustworthy_time_gets_no_moon_flag(self) -> None:
        sink = Sink()
        writer(sink).write(sky_quality(FULL_MOON_NIGHT, ["time_invalid"]))
        (record,) = sink.records
        assert isinstance(record, SkyQualityRecord)
        assert record.flags == ["time_invalid"]

    def test_a_heater_that_raises_does_not_stop_the_record(self) -> None:
        sink = Sink()
        writer(sink, heater=Heater(RuntimeError("the sensor is gone"))).write(
            sky_quality(FULL_MOON_NIGHT)
        )
        (record,) = sink.records
        assert isinstance(record, SkyQualityRecord)
        assert record.flags == ["moon"]

    def test_a_station_without_a_site_or_a_heater_adds_nothing(self) -> None:
        sink = Sink()
        original = sky_quality(FULL_MOON_NIGHT)
        writer(sink, site=None, heater=None).write(original)
        assert sink.records[0] is original

    def test_the_flags_are_valid_codes_of_the_record(self) -> None:
        # The record rejects a code that it does not declare, so a rename would fail here.
        flagged: Any = writer(Sink(), heater=Heater(Reading(dew_point_c=6.0))).flagged(
            sky_quality(FULL_MOON_NIGHT)
        )
        assert set(flagged.flags) == {"moon", "dew"}
