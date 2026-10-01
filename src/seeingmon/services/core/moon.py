"""A low-precision position of the Moon, for the `moon` flag of the sky quality.

The `moon` flag of a `sky_quality` record says that the Moon was above the horizon and bright enough
to raise the sky background. The decision needs the altitude to a fraction of a degree and the lit
fraction to a few percent. The low-precision series of the Astronomical Almanac keeps the largest
terms of the lunar theory and has an error of about 0.3 degrees in the position, which is enough.
The ephemeris of the scheduler (`seeingmon.scheduler.ephemeris`) has the Sun and Polaris only, so
the Moon lives here.

Every time is UTC nanoseconds since the Unix epoch, as everywhere else in the package. Latitude is
north-positive and longitude is east-positive, in degrees. The difference between UTC and the
uniform time scales is under 70 seconds, which the Moon (0.5 degrees an hour) does not notice. The
functions read no clock and no file, and they keep no state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from seeingmon.scheduler.ephemeris import julian_day, sun_position

_J2000_JD = 2451545.0


@dataclass(frozen=True, slots=True)
class MoonPosition:
    """The Moon's geocentric place of date: the equatorial angles and the horizontal parallax."""

    right_ascension_deg: float
    declination_deg: float
    parallax_deg: float


def _sin(degrees: float) -> float:
    return math.sin(math.radians(degrees))


def _cos(degrees: float) -> float:
    return math.cos(math.radians(degrees))


def moon_position(t_utc_ns: int) -> MoonPosition:
    """The Moon's right ascension and declination of date, and its horizontal parallax."""
    t = (julian_day(t_utc_ns) - _J2000_JD) / 36525.0
    longitude = (
        218.32
        + 481267.881 * t
        + 6.29 * _sin(135.0 + 477198.87 * t)
        - 1.27 * _sin(259.3 - 413335.36 * t)
        + 0.66 * _sin(235.7 + 890534.22 * t)
        + 0.21 * _sin(269.9 + 954397.74 * t)
        - 0.19 * _sin(357.5 + 35999.05 * t)
        - 0.11 * _sin(186.5 + 966404.03 * t)
    )
    latitude = (
        5.13 * _sin(93.3 + 483202.02 * t)
        + 0.28 * _sin(228.2 + 960400.89 * t)
        - 0.28 * _sin(318.3 + 6003.15 * t)
        - 0.17 * _sin(217.6 - 407332.21 * t)
    )
    parallax = (
        0.9508
        + 0.0518 * _cos(135.0 + 477198.87 * t)
        + 0.0095 * _cos(259.3 - 413335.36 * t)
        + 0.0078 * _cos(235.7 + 890534.22 * t)
        + 0.0028 * _cos(269.9 + 954397.74 * t)
    )
    # Direction cosines in the equatorial frame (the obliquity is 23.44 degrees).
    x = _cos(latitude) * _cos(longitude)
    y = 0.9175 * _cos(latitude) * _sin(longitude) - 0.3978 * _sin(latitude)
    z = 0.3978 * _cos(latitude) * _sin(longitude) + 0.9175 * _sin(latitude)
    return MoonPosition(
        right_ascension_deg=math.degrees(math.atan2(y, x)) % 360.0,
        declination_deg=math.degrees(math.asin(max(-1.0, min(1.0, z)))),
        parallax_deg=parallax,
    )


def _sidereal_deg(t_utc_ns: int) -> float:
    """The mean sidereal time at Greenwich, in degrees."""
    return (280.46061837 + 360.98564736629 * (julian_day(t_utc_ns) - _J2000_JD)) % 360.0


def moon_elevation_deg(t_utc_ns: int, latitude_deg: float, longitude_deg: float) -> float:
    """The elevation of the Moon's center above the horizon, as seen from the site.

    The elevation is geometric (no refraction) and topocentric: the parallax of the Moon (up to a
    degree) lowers it, which matters for a flag at the horizon.
    """
    moon = moon_position(t_utc_ns)
    hour_angle = _sidereal_deg(t_utc_ns) + longitude_deg - moon.right_ascension_deg
    latitude = math.radians(latitude_deg)
    declination = math.radians(moon.declination_deg)
    sine = math.sin(latitude) * math.sin(declination) + math.cos(latitude) * math.cos(
        declination
    ) * math.cos(math.radians(hour_angle))
    geocentric = math.degrees(math.asin(max(-1.0, min(1.0, sine))))
    return geocentric - moon.parallax_deg * math.cos(math.radians(geocentric))


def moon_illumination(t_utc_ns: int) -> float:
    """The lit fraction of the Moon's disk, from 0 at the new Moon to 1 at the full Moon.

    The fraction follows from the angle between the Moon and the Sun in the sky. The distances
    shift it by less than 0.5 percent.
    """
    moon = moon_position(t_utc_ns)
    sun = sun_position(t_utc_ns)
    moon_ra, moon_dec = math.radians(moon.right_ascension_deg), math.radians(moon.declination_deg)
    sun_ra, sun_dec = math.radians(sun.right_ascension_deg), math.radians(sun.declination_deg)
    cosine = math.sin(moon_dec) * math.sin(sun_dec) + math.cos(moon_dec) * math.cos(
        sun_dec
    ) * math.cos(moon_ra - sun_ra)
    return (1.0 - max(-1.0, min(1.0, cosine))) / 2.0
