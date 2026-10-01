"""A low-precision ephemeris for the daylight gate and the twilight flag.

The Sun's position follows the NOAA solar calculator, which is the Meeus algorithm without
the small terms. It agrees with a full ephemeris to about 0.01 degrees in elevation between the
years 1900 and 2100, which is far better than the gate needs: the thresholds are whole degrees.
`polaris_zenith_angle_deg` adds the zenith angle of Polaris for the fast analysis. It precesses
the catalog position to the date and ignores nutation and aberration (about 0.005 degrees).

All times are UTC nanoseconds since the Unix epoch, as everywhere else in the package. Latitude
is north-positive and longitude is east-positive, in degrees. The elevations are geometric: they
carry no atmospheric refraction, which is how the twilight limits (such as -18 degrees) are
defined. The functions read no clock and no file, and they keep no state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from seeingmon.clock import NS_PER_S

_UNIX_EPOCH_JD = 2440587.5
_J2000_JD = 2451545.0
_S_PER_DAY = 86_400.0

# Polaris (HIP 11767) at epoch J2000, in the ICRS. The proper motion moves it by about
# 0.3 arcsec on the sky in 26 years, which does not matter for a zenith angle.
POLARIS_RA_J2000_DEG = 37.95456067
POLARIS_DEC_J2000_DEG = 89.26410897

# The elevation of the Sun's center at sunrise and sunset: refraction (34 arcmin) plus the
# Sun's radius (16 arcmin).
SUNRISE_ELEVATION_DEG = -0.833


@dataclass(frozen=True, slots=True)
class SunPosition:
    """The Sun's apparent place.

    `declination_deg` and `right_ascension_deg` are of date. `equation_of_time_min` is apparent
    minus mean solar time, in minutes.
    """

    declination_deg: float
    right_ascension_deg: float
    equation_of_time_min: float


def julian_day(t_utc_ns: int) -> float:
    """The Julian day of a UTC time.

    The algorithms below want a uniform time scale, and UTC differs from it by about 69 seconds.
    The Sun moves 0.003 degrees in that time, so the code ignores the difference.
    """
    return t_utc_ns / (NS_PER_S * _S_PER_DAY) + _UNIX_EPOCH_JD


def _julian_centuries(t_utc_ns: int) -> float:
    return (julian_day(t_utc_ns) - _J2000_JD) / 36525.0


def sun_position(t_utc_ns: int) -> SunPosition:
    """Compute the Sun's declination, right ascension, and equation of time."""
    t = _julian_centuries(t_utc_ns)
    mean_longitude = math.radians((280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0)
    mean_anomaly = math.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    center = (
        math.sin(mean_anomaly) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2.0 * mean_anomaly) * (0.019993 - 0.000101 * t)
        + math.sin(3.0 * mean_anomaly) * 0.000289
    )
    true_longitude = math.degrees(mean_longitude) + center
    node = math.radians(125.04 - 1934.136 * t)
    apparent_longitude = math.radians(true_longitude - 0.00569 - 0.00478 * math.sin(node))
    mean_obliquity_arcsec = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))
    obliquity = math.radians(
        23.0 + (26.0 + mean_obliquity_arcsec / 60.0) / 60.0 + 0.00256 * math.cos(node)
    )
    declination = math.asin(math.sin(obliquity) * math.sin(apparent_longitude))
    right_ascension = math.atan2(
        math.cos(obliquity) * math.sin(apparent_longitude), math.cos(apparent_longitude)
    )
    y = math.tan(obliquity / 2.0) ** 2
    equation_of_time = 4.0 * math.degrees(
        y * math.sin(2.0 * mean_longitude)
        - 2.0 * eccentricity * math.sin(mean_anomaly)
        + 4.0 * eccentricity * y * math.sin(mean_anomaly) * math.cos(2.0 * mean_longitude)
        - 0.5 * y * y * math.sin(4.0 * mean_longitude)
        - 1.25 * eccentricity * eccentricity * math.sin(2.0 * mean_anomaly)
    )
    return SunPosition(
        declination_deg=math.degrees(declination),
        right_ascension_deg=math.degrees(right_ascension) % 360.0,
        equation_of_time_min=equation_of_time,
    )


def _elevation_deg(latitude_deg: float, declination_deg: float, hour_angle_deg: float) -> float:
    latitude = math.radians(latitude_deg)
    declination = math.radians(declination_deg)
    sine = math.sin(latitude) * math.sin(declination) + math.cos(latitude) * math.cos(
        declination
    ) * math.cos(math.radians(hour_angle_deg))
    return math.degrees(math.asin(max(-1.0, min(1.0, sine))))


def sun_elevation_deg(t_utc_ns: int, latitude_deg: float, longitude_deg: float) -> float:
    """The geometric elevation of the Sun's center above the horizon, in degrees."""
    sun = sun_position(t_utc_ns)
    # True solar time in minutes: the UTC time of day, the equation of time, and the longitude.
    minutes_of_day = (t_utc_ns % (86_400 * NS_PER_S)) / (NS_PER_S * 60.0)
    solar_time_min = (minutes_of_day + sun.equation_of_time_min + 4.0 * longitude_deg) % 1440.0
    return _elevation_deg(latitude_deg, sun.declination_deg, solar_time_min / 4.0 - 180.0)


def next_sun_crossing_utc_ns(
    t_utc_ns: int,
    latitude_deg: float,
    longitude_deg: float,
    elevation_deg: float,
    *,
    rising: bool,
    horizon_days: float = 2.0,
    precision_s: float = 1.0,
) -> int | None:
    """Find when the Sun's elevation next crosses `elevation_deg`, going up or down.

    The search starts at `t_utc_ns` and looks `horizon_days` ahead. It returns `None` when the
    Sun does not cross in that time, as in the polar day or the polar night. Pass
    `SUNRISE_ELEVATION_DEG` for the times of sunrise and sunset, `-6.0` for civil twilight, and
    `-18.0` for the start and end of astronomical night.
    """
    step_ns = 10 * 60 * NS_PER_S

    def above(t_ns: int) -> bool:
        return sun_elevation_deg(t_ns, latitude_deg, longitude_deg) > elevation_deg

    end_ns = t_utc_ns + round(horizon_days * _S_PER_DAY * NS_PER_S)
    low_ns = t_utc_ns
    low_above = above(low_ns)
    while low_ns < end_ns:
        high_ns = min(low_ns + step_ns, end_ns)
        high_above = above(high_ns)
        if low_above != high_above and high_above == rising:
            while high_ns - low_ns > precision_s * NS_PER_S:
                middle_ns = (low_ns + high_ns) // 2
                if above(middle_ns) == high_above:
                    high_ns = middle_ns
                else:
                    low_ns = middle_ns
            return high_ns
        low_ns, low_above = high_ns, high_above
    return None


def _precess_to_date(t_centuries: float, ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Precess a J2000 position to the mean equator and equinox of date (Meeus, chapter 21)."""
    t = t_centuries
    zeta = math.radians((2306.2181 * t + 0.30188 * t**2 + 0.017998 * t**3) / 3600.0)
    z = math.radians((2306.2181 * t + 1.09468 * t**2 + 0.018203 * t**3) / 3600.0)
    theta = math.radians((2004.3109 * t - 0.42665 * t**2 - 0.041833 * t**3) / 3600.0)
    ra0 = math.radians(ra_deg)
    dec0 = math.radians(dec_deg)
    a = math.cos(dec0) * math.sin(ra0 + zeta)
    b = math.cos(theta) * math.cos(dec0) * math.cos(ra0 + zeta) - math.sin(theta) * math.sin(dec0)
    c = math.sin(theta) * math.cos(dec0) * math.cos(ra0 + zeta) + math.cos(theta) * math.sin(dec0)
    ra = (math.atan2(a, b) + z) % (2.0 * math.pi)
    # `atan2` stays accurate near the pole, where `asin` of a value close to 1 loses digits.
    dec = math.atan2(c, math.hypot(a, b))
    return math.degrees(ra), math.degrees(dec)


def polaris_zenith_angle_deg(t_utc_ns: int, latitude_deg: float, longitude_deg: float) -> float:
    """The geometric zenith angle of Polaris in degrees: 90 minus its elevation.

    Polaris circles the pole at about 0.65 degrees, so its zenith angle swings by that much
    around 90 degrees minus the latitude. The result carries no refraction.
    """
    t = _julian_centuries(t_utc_ns)
    ra_deg, dec_deg = _precess_to_date(t, POLARIS_RA_J2000_DEG, POLARIS_DEC_J2000_DEG)
    days = julian_day(t_utc_ns) - _J2000_JD
    gmst_deg = (
        280.46061837 + 360.98564736629 * days + 0.000387933 * t * t - t**3 / 38710000.0
    ) % 360.0
    elevation = _elevation_deg(latitude_deg, dec_deg, gmst_deg + longitude_deg - ra_deg)
    return 90.0 - elevation
