"""The ephemeris against published values, analytic limits, and its own consistency.

The published rise, set, and twilight times come from the rise, set, and transit service of the
U.S. Naval Observatory (Astronomical Applications Department), queried in October 2026 for the
synthetic sites below. The service rounds each time to the minute, so the tolerance is one
minute. The other published values are the dates of the equinoxes and solstices, the obliquity
of the ecliptic, and the extremes of the equation of time. `test_ephemeris_astropy.py` compares
with astropy.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.scheduler.ephemeris import (
    SUNRISE_ELEVATION_DEG,
    julian_day,
    next_sun_crossing_utc_ns,
    polaris_zenith_angle_deg,
    sun_elevation_deg,
    sun_position,
)

# Synthetic sites. Neither is anyone's real site.
NORTH = (55.0, 0.0)
SOUTH_EAST = (-35.0, 150.0)

OBLIQUITY_DEG = 23.44  # the published obliquity of the ecliptic, 23 degrees 26 arcmin
MINUTE_S = 60.0
MINUTE_NS = 60 * NS_PER_S


def utc(text: str) -> int:
    return iso_to_utc_ns(text)


# --- The Sun's place ---------------------------------------------------------------------------


def test_julian_day_of_the_epochs() -> None:
    assert julian_day(utc("1970-01-01T00:00:00Z")) == 2440587.5
    assert julian_day(utc("2000-01-01T12:00:00Z")) == pytest.approx(2451545.0, abs=1e-9)
    day_later = julian_day(utc("2000-01-02T12:00:00Z")) - julian_day(utc("2000-01-01T12:00:00Z"))
    assert day_later == pytest.approx(1.0, abs=1e-9)


# The instants of the equinoxes and solstices in UTC, as the U.S. Naval Observatory publishes them,
# with the Sun's declination at each: zero at an equinox, plus or minus the obliquity at a solstice.
SEASON_INSTANTS = [
    ("2024-03-20T03:06:00Z", 0.0),
    ("2024-06-20T20:51:00Z", OBLIQUITY_DEG),
    ("2024-09-22T12:44:00Z", 0.0),
    ("2024-12-21T09:21:00Z", -OBLIQUITY_DEG),
    ("2025-03-20T09:01:00Z", 0.0),
    ("2025-06-21T02:42:00Z", OBLIQUITY_DEG),
    ("2025-09-22T18:19:00Z", 0.0),
    ("2025-12-21T15:03:00Z", -OBLIQUITY_DEG),
]


@pytest.mark.parametrize(("instant", "declination"), SEASON_INSTANTS)
def test_the_declination_at_the_published_equinox_and_solstice_instants(
    instant: str, declination: float
) -> None:
    # The Sun moves 0.4 degrees a day at an equinox, so a published minute is 0.0003 degrees.
    # The 0.02 degree tolerance also covers the rounding of the obliquity.
    assert sun_position(utc(instant)).declination_deg == pytest.approx(declination, abs=0.02)


@pytest.mark.parametrize(
    ("day", "minutes"),
    [
        ("2025-02-11T12:00:00Z", -14.25),
        ("2025-05-14T12:00:00Z", 3.67),
        ("2025-07-26T12:00:00Z", -6.5),
        ("2025-11-03T12:00:00Z", 16.42),
    ],
)
def test_the_equation_of_time_at_its_published_extremes(day: str, minutes: float) -> None:
    """The four extremes of the year, to 15 seconds."""
    assert sun_position(utc(day)).equation_of_time_min == pytest.approx(minutes, abs=0.25)


def test_the_right_ascension_is_zero_at_the_march_equinox() -> None:
    right_ascension = sun_position(utc("2025-03-20T09:01:00Z")).right_ascension_deg
    assert min(right_ascension, 360.0 - right_ascension) < 0.02


# --- Noon elevations ---------------------------------------------------------------------------


def peak_elevation_deg(day: str, latitude: float, longitude: float) -> float:
    """The highest elevation on the solar day that is centered on the local solar noon."""
    noon_ns = utc(f"{day}T12:00:00Z") - round(longitude / 15.0 * 3600 * NS_PER_S)
    return max(
        sun_elevation_deg(noon_ns + minute * MINUTE_NS, latitude, longitude)
        for minute in range(-12 * 60, 12 * 60)
    )


@pytest.mark.parametrize(
    ("day", "site", "expected"),
    [
        # At a solstice the noon elevation is 90 degrees minus the distance to the Sun's latitude.
        ("2024-06-20", NORTH, 90 - 55 + OBLIQUITY_DEG),
        ("2024-12-21", NORTH, 90 - 55 - OBLIQUITY_DEG),
        ("2024-12-21", SOUTH_EAST, 90 - (35 - OBLIQUITY_DEG)),
        ("2024-06-20", SOUTH_EAST, 90 - (35 + OBLIQUITY_DEG)),
        # At an equinox it is 90 degrees minus the latitude. These days sit next to the instants.
        ("2024-09-22", NORTH, 90 - 55),
        ("2025-03-20", NORTH, 90 - 55),
        ("2024-03-20", SOUTH_EAST, 90 - 35),
    ],
)
def test_the_noon_elevation_at_the_solstices_and_equinoxes(
    day: str, site: tuple[float, float], expected: float
) -> None:
    assert peak_elevation_deg(day, *site) == pytest.approx(expected, abs=0.1)


def test_the_midnight_sun_stays_above_the_twilight_limit_at_55_degrees_but_not_at_40() -> None:
    """The architecture notes weeks with no astronomical night. At 55 degrees north the Sun
    reaches only 11.6 degrees below the horizon at midnight on the longest day."""
    midnight_ns = utc("2024-06-20T00:00:00Z")
    assert sun_elevation_deg(midnight_ns, 55.0, 0.0) == pytest.approx(
        -(90 - 55 - OBLIQUITY_DEG), abs=0.2
    )
    assert sun_elevation_deg(midnight_ns, 55.0, 0.0) > -18.0
    assert sun_elevation_deg(midnight_ns, 40.0, 0.0) < -18.0


# --- Rise, set, and twilight against the U.S. Naval Observatory --------------------------------

# For each date and site: the UTC times of the beginning of civil twilight, sunrise, upper transit,
# sunset, and the end of civil twilight. A trailing plus means the next day. The southern site's
# entry is the published local times for 2024-12-21 (UTC+10), minus 10 hours.
USNO_DAYS = [
    ("2024-06-20", NORTH, ["02:22", "03:20", "12:02", "20:43", "21:41"]),
    ("2024-12-21", NORTH, ["07:38", "08:23", "11:58", "15:33", "16:19"]),
    ("2024-09-22", NORTH, ["05:10", "05:46", "11:53", "17:58", "18:34"]),
    ("2025-03-20", NORTH, ["05:26", "06:02", "12:07", "18:14", "18:50"]),
    ("2024-12-20", SOUTH_EAST, ["18:13", "18:43", "01:58+", "09:14+", "09:43+"]),
]
USNO_IDS = ["june-solstice", "december-solstice", "september-equinox", "march-equinox", "southeast"]


def published_times(day: str, texts: list[str]) -> list[int]:
    """Parse each `HH:MM` as a UTC time on `day`, or on the next day when it ends with `+`."""
    start = datetime.fromisoformat(day).replace(tzinfo=UTC)
    times = []
    for text in texts:
        hour, minute = (int(part) for part in text.rstrip("+").split(":"))
        moment = start + timedelta(hours=hour, minutes=minute, days=1 if text.endswith("+") else 0)
        times.append(round(moment.timestamp()) * NS_PER_S)
    return times


@pytest.mark.parametrize(("day", "site", "texts"), USNO_DAYS, ids=USNO_IDS)
def test_the_rise_set_and_civil_twilight_times_match_the_usno_to_a_minute(
    day: str, site: tuple[float, float], texts: list[str]
) -> None:
    twilight_begin, rise, transit, sunset, twilight_end = published_times(day, texts)
    search_from = twilight_begin - 3600 * NS_PER_S
    latitude, longitude = site

    def crossing(elevation: float, *, rising: bool) -> int:
        found = next_sun_crossing_utc_ns(search_from, latitude, longitude, elevation, rising=rising)
        assert found is not None
        return found

    computed = {
        "twilight begin": (crossing(-6.0, rising=True), twilight_begin),
        "sunrise": (crossing(SUNRISE_ELEVATION_DEG, rising=True), rise),
        "sunset": (crossing(SUNRISE_ELEVATION_DEG, rising=False), sunset),
        "twilight end": (crossing(-6.0, rising=False), twilight_end),
    }
    for name, (ours, published) in computed.items():
        assert abs(ours - published) / NS_PER_S <= MINUTE_S, name
    # The upper transit is the middle of the day between sunrise and sunset, to about a minute.
    middle = (computed["sunrise"][0] + computed["sunset"][0]) // 2
    assert abs(middle - transit) / NS_PER_S <= MINUTE_S


def test_the_crossing_search_returns_the_first_crossing_in_the_requested_direction() -> None:
    start = utc("2024-06-20T00:00:00Z")
    rising = next_sun_crossing_utc_ns(start, *NORTH, 0.0, rising=True)
    falling = next_sun_crossing_utc_ns(start, *NORTH, 0.0, rising=False)
    assert rising is not None
    assert falling is not None
    assert start < rising < falling
    # The Sun is above the limit just after the rising time, and below it just before.
    assert sun_elevation_deg(rising + 5 * NS_PER_S, *NORTH) > 0.0
    assert sun_elevation_deg(rising - 5 * NS_PER_S, *NORTH) < 0.0
    assert sun_elevation_deg(falling - 5 * NS_PER_S, *NORTH) > 0.0


def test_the_crossing_search_returns_none_in_the_polar_day_and_the_polar_night() -> None:
    """At 75 degrees north the Sun stays above the horizon in June and below it in December."""
    june = utc("2024-06-20T00:00:00Z")
    december = utc("2024-12-20T00:00:00Z")
    assert next_sun_crossing_utc_ns(june, 75.0, 0.0, SUNRISE_ELEVATION_DEG, rising=False) is None
    assert next_sun_crossing_utc_ns(december, 75.0, 0.0, SUNRISE_ELEVATION_DEG, rising=True) is None


# --- Polaris -----------------------------------------------------------------------------------


@pytest.mark.parametrize("latitude", [-10.0, 0.0, 35.0, 55.0, 70.0])
def test_the_polaris_zenith_angle_circles_90_degrees_minus_the_latitude(latitude: float) -> None:
    """In 2026 Polaris is about 0.64 degrees from the pole, so the angle swings by twice that."""
    start = utc("2026-01-01T00:00:00Z")
    angles = [
        polaris_zenith_angle_deg(start + minute * MINUTE_NS, latitude, 10.0)
        for minute in range(0, 24 * 60, 5)
    ]
    swing = max(angles) - min(angles)
    assert 1.2 < swing < 1.35
    assert (max(angles) + min(angles)) / 2 == pytest.approx(90.0 - latitude, abs=0.02)


# --- Properties --------------------------------------------------------------------------------

times = st.integers(min_value=utc("2000-01-01T00:00:00Z"), max_value=utc("2040-01-01T00:00:00Z"))
latitudes = st.floats(min_value=-89.9, max_value=89.9, allow_nan=False)
longitudes = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False)


@given(times, latitudes, longitudes)
def test_the_elevation_is_a_valid_angle(t_ns: int, latitude: float, longitude: float) -> None:
    assert -90.0 <= sun_elevation_deg(t_ns, latitude, longitude) <= 90.0


@given(times, latitudes, longitudes)
def test_the_elevation_changes_slowly(t_ns: int, latitude: float, longitude: float) -> None:
    """The Sun climbs at most 15 degrees an hour, so one minute moves it less than 0.26 degrees."""
    later = sun_elevation_deg(t_ns + MINUTE_NS, latitude, longitude)
    assert abs(later - sun_elevation_deg(t_ns, latitude, longitude)) < 0.26


@given(times, latitudes, longitudes)
def test_the_elevation_stays_below_the_highest_possible_noon(
    t_ns: int, latitude: float, longitude: float
) -> None:
    highest = 90.0 - abs(latitude) + OBLIQUITY_DEG + 0.05
    assert sun_elevation_deg(t_ns, latitude, longitude) <= min(90.0, highest)


@given(times, latitudes, longitudes)
def test_the_antipode_sees_the_opposite_elevation(
    t_ns: int, latitude: float, longitude: float
) -> None:
    """At the antipodal point the Sun is as far below the horizon as it is above here."""
    antipode_longitude = longitude + 180.0 if longitude < 0 else longitude - 180.0
    assert sun_elevation_deg(t_ns, -latitude, antipode_longitude) == pytest.approx(
        -sun_elevation_deg(t_ns, latitude, longitude), abs=1e-9
    )


@given(times, st.floats(min_value=-80.0, max_value=80.0), longitudes)
def test_the_polaris_zenith_angle_never_strays_from_the_pole_circle(
    t_ns: int, latitude: float, longitude: float
) -> None:
    """Polaris is never more than 0.8 degrees from the pole between 2000 and 2040."""
    angle = polaris_zenith_angle_deg(t_ns, latitude, longitude)
    assert math.isfinite(angle)
    assert abs(angle - (90.0 - latitude)) <= 0.8
