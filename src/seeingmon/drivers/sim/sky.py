"""Time-dependent sky conditions: the sun, the sky background, clouds, and scintillation.

The simulator needs a place to compute the sun's altitude and the airmass of a star. `Site`
holds a latitude and a longitude, and the default is a synthetic round-number position in the
North Sea. It is not a real station. Replace it in `SimOptions` when you need another latitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S

FloatArray = npt.NDArray[np.float64]

_UNIX_EPOCH_JD = 2_440_587.5
_J2000_JD = 2_451_545.0


@dataclass(frozen=True, slots=True)
class Site:
    """A place on Earth, for the sun and the airmass. The default is synthetic, not a station."""

    latitude_deg: float = 55.0
    longitude_deg: float = 0.0

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude_deg <= 90.0 or not -180.0 <= self.longitude_deg <= 360.0:
            raise ValueError("latitude must be within 90 degrees and longitude within 180")


SYNTHETIC_SITE = Site()


def julian_day(t_utc_ns: npt.ArrayLike) -> float | FloatArray:
    """The Julian date of a time in nanoseconds since the Unix epoch."""
    return _UNIX_EPOCH_JD + np.asarray(t_utc_ns, dtype=np.float64) / (NS_PER_S * 86400.0)


def sun_altitude_deg(site: Site, t_utc_ns: npt.ArrayLike) -> float | FloatArray:
    """The altitude of the sun above the horizon, in degrees.

    The formula is the low-precision solar position of the Astronomical Almanac, good to about
    0.01 degrees, which is far more than the twilight model needs.
    """
    n = np.asarray(julian_day(t_utc_ns)) - _J2000_JD
    mean_longitude = np.radians((280.460 + 0.9856474 * n) % 360.0)
    anomaly = np.radians((357.528 + 0.9856003 * n) % 360.0)
    ecliptic_longitude = mean_longitude + np.radians(
        1.915 * np.sin(anomaly) + 0.020 * np.sin(2.0 * anomaly)
    )
    obliquity = np.radians(23.439 - 0.0000004 * n)
    right_ascension = np.arctan2(
        np.cos(obliquity) * np.sin(ecliptic_longitude), np.cos(ecliptic_longitude)
    )
    declination = np.arcsin(np.sin(obliquity) * np.sin(ecliptic_longitude))
    gmst_deg = (280.46061837 + 360.98564736629 * n) % 360.0
    hour_angle = np.radians(gmst_deg + site.longitude_deg) - right_ascension
    latitude = math.radians(site.latitude_deg)
    sin_alt = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(
        declination
    ) * np.cos(hour_angle)
    result = np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))
    return float(result) if result.ndim == 0 else np.asarray(result, dtype=np.float64)


def airmass(zenith_angle_deg: float) -> float:
    """The relative airmass at a zenith angle, from the formula of Kasten and Young (1989)."""
    z = min(max(zenith_angle_deg, 0.0), 89.0)
    return 1.0 / (math.cos(math.radians(z)) + 0.50572 * math.pow(96.07995 - z, -1.6364))


DAYLIGHT_SKY_MAG_ARCSEC2 = 4.2
"""The daylight sky near the celestial pole, in V mag/arcsec^2.

It is the median of the V-band sky that Nickel and Calderwood (2021, JAAVSO 49, 269, Figure 2)
measured 66 to 96 degrees from the sun with the sun 10 to 52 degrees high. The pole is always 66.6
to 113.4 degrees from the sun. `docs/research-notes.md` ("Polaris in a bright sky") has the source
and what the model leaves out.
"""

_DEFAULT_DARK_SKY_MAG_ARCSEC2 = 20.5

# The sky near the pole with the default dark sky, in magnitudes per square arcsecond, against
# the altitude of the sun. A dark sky needs a sun below -18 degrees. Up to 0 degrees, the values
# follow typical twilight curves, as a brightening of the default dark sky by 0, 0.5, 2.5, 5.5,
# 8.5, 11.5, and 14.5 mag. From +10 degrees up, the sky is the daylight sky near the pole.
_TWILIGHT_ALTITUDE_DEG = (-18.0, -15.0, -12.0, -9.0, -6.0, -3.0, 0.0, 10.0, 90.0)
_TWILIGHT_SKY_MAG_ARCSEC2 = (
    20.5,
    20.0,
    18.0,
    15.0,
    12.0,
    9.0,
    6.0,
    DAYLIGHT_SKY_MAG_ARCSEC2,
    DAYLIGHT_SKY_MAG_ARCSEC2,
)


def _flux(mag: float | FloatArray) -> FloatArray:
    return np.asarray(np.power(10.0, -0.4 * np.asarray(mag, dtype=np.float64)))


def sky_brightness_mag_arcsec2(
    dark_sky_mag_arcsec2: float, sun_altitude: float | FloatArray, *, twilight: bool = True
) -> float | FloatArray:
    """The surface brightness of the sky for a dark-sky value and the altitude of the sun.

    The result is in mag/arcsec^2, so a smaller number means a brighter sky. The sky is the dark
    sky of the site plus the sunlight that the air scatters, and the scattered light depends on
    the altitude of the sun alone: it is the table above less the default dark sky of 20.5. A
    site with another dark sky therefore has the same daylight sky, `DAYLIGHT_SKY_MAG_ARCSEC2`.
    With `twilight=False`, the sky stays at its dark value at every time.
    """
    if not twilight:
        return (
            dark_sky_mag_arcsec2
            if np.isscalar(sun_altitude)
            else np.full_like(np.asarray(sun_altitude, dtype=np.float64), dark_sky_mag_arcsec2)
        )
    table = np.interp(sun_altitude, _TWILIGHT_ALTITUDE_DEG, _TWILIGHT_SKY_MAG_ARCSEC2)
    sunlight = np.maximum(_flux(table) - _flux(_DEFAULT_DARK_SKY_MAG_ARCSEC2), 0.0)
    result = -2.5 * np.log10(_flux(dark_sky_mag_arcsec2) + sunlight)
    return float(result) if np.ndim(result) == 0 else np.asarray(result, dtype=np.float64)


# --- clouds ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CloudEvent:
    """A cloud that crosses the field.

    The transparency falls from 1 to `transmission` over `ramp_s`, holds, and rises again over
    `ramp_s`, so that the whole event lasts `duration_s`. `transmission` is the smallest
    transparency, from 0 (opaque) to 1 (no effect).
    """

    start_utc_ns: int
    duration_s: float
    transmission: float = 0.1
    ramp_s: float = 5.0

    def __post_init__(self) -> None:
        if self.duration_s <= 0 or self.ramp_s <= 0:
            raise ValueError("duration_s and ramp_s must be positive")
        if not 0.0 <= self.transmission <= 1.0:
            raise ValueError("transmission must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Clouds:
    """The transparency of the sky as a function of time: a baseline and a list of events."""

    baseline: float = 1.0
    events: tuple[CloudEvent, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 < self.baseline <= 1.0:
            raise ValueError("baseline must be between 0 (exclusive) and 1")

    def transparency(self, t_utc_ns: npt.ArrayLike) -> float | FloatArray:
        """The transparency, from 0 to 1, at a time or an array of times."""
        times = np.asarray(t_utc_ns, dtype=np.float64)
        result = np.full(times.shape, self.baseline, dtype=np.float64)
        for event in self.events:
            x = (times - event.start_utc_ns) / NS_PER_S
            rise = np.clip(x / event.ramp_s, 0.0, 1.0)
            fall = np.clip((event.duration_s - x) / event.ramp_s, 0.0, 1.0)
            level = np.minimum(rise, fall)
            shape = 0.5 * (1.0 - np.cos(math.pi * level))
            result *= 1.0 - (1.0 - event.transmission) * shape
        return float(result) if result.ndim == 0 else result

    def mean_transparency(self, t_start_utc_ns: int, duration_s: float, samples: int = 16) -> float:
        """The mean transparency over an interval."""
        times = t_start_utc_ns + (np.arange(samples) + 0.5) * duration_s * NS_PER_S / samples
        return float(np.mean(self.transparency(times)))


# --- scintillation --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScintillationConfig:
    """Scintillation as multiplicative flux noise.

    `sigma0` is the relative rms of the flux for a very short exposure at the reference airmass,
    and `reference_airmass` names that airmass. The rms scales as `airmass^1.5` (Young's law) and
    falls as `1 / sqrt(1 + T / knee_s)` with the exposure `T`, which gives a variance that falls as
    `1 / T` above the knee. For Polaris at an airmass of 1.155, the defaults give an rms of 0.39 for
    2 ms and 0.24 for 10 ms, in the range that the research notes give. `correlation_time_s` is
    the e-folding time of the flux fluctuation.
    """

    sigma0: float = 0.50
    reference_airmass: float = 1.155
    knee_s: float = 3.0e-3
    correlation_time_s: float = 3.0e-3
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.sigma0 < 0 or self.knee_s <= 0 or self.correlation_time_s <= 0:
            raise ValueError(
                "sigma0 must not be negative; knee_s and correlation_time_s must be positive"
            )

    def index(self, exposure_s: float, airmass_value: float) -> float:
        """The relative rms of the flux (the square root of the scintillation index)."""
        if not self.enabled:
            return 0.0
        scale = math.pow(airmass_value / self.reference_airmass, 1.5)
        return self.sigma0 * scale / math.sqrt(1.0 + exposure_s / self.knee_s)


class ScintillationProcess:
    """A stationary Gaussian process with unit variance and an exponential correlation.

    The process is a sum of cosines with random phases and Cauchy-distributed frequencies,
    which gives the correlation `exp(-|dt| / tau)`. It is a pure function of the time, so
    frames in any order see the same flux.
    """

    def __init__(self, seed: int, correlation_time_s: float, components: int = 96) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5C1]))
        self._rate = rng.standard_cauchy(components) / (2.0 * math.pi * correlation_time_s)
        # Cap the rare enormous frequencies, which would only add white noise.
        self._rate = np.clip(self._rate, -2e4, 2e4)
        self._phase = rng.random(components) * 2.0 * math.pi
        self._norm = math.sqrt(2.0 / components)

    def values(self, t_s: npt.ArrayLike) -> FloatArray:
        """The process at times in seconds."""
        arg = 2.0 * math.pi * np.outer(np.asarray(t_s, dtype=np.float64), self._rate) + self._phase
        return np.asarray(self._norm * np.cos(arg).sum(axis=1), dtype=np.float64)

    def value(self, t_s: float) -> float:
        return float(self.values(np.asarray([t_s]))[0])


def flux_factor(relative_rms: float, gaussian: float | FloatArray) -> float | FloatArray:
    """A log-normal flux factor with mean 1 and the given relative rms, from a unit Gaussian."""
    if relative_rms <= 0.0:
        return (
            1.0 if np.isscalar(gaussian) else np.ones_like(np.asarray(gaussian, dtype=np.float64))
        )
    sigma = math.sqrt(math.log1p(relative_rms**2))
    result = np.exp(sigma * np.asarray(gaussian) - 0.5 * sigma**2)
    return float(result) if result.ndim == 0 else np.asarray(result, dtype=np.float64)
