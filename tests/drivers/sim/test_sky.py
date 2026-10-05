"""The sun, the sky background, clouds, and scintillation."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from seeingmon.clock import DEFAULT_START_UTC_NS, NS_PER_S
from seeingmon.drivers.sim.options import SimOptions
from seeingmon.drivers.sim.sky import (
    DAYLIGHT_SKY_MAG_ARCSEC2,
    SYNTHETIC_SITE,
    CloudEvent,
    Clouds,
    ScintillationConfig,
    ScintillationProcess,
    Site,
    airmass,
    flux_factor,
    sky_brightness_mag_arcsec2,
    sun_altitude_deg,
)


def utc_ns(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    moment = datetime(year, month, day, hour, minute, tzinfo=UTC)
    return round(moment.timestamp()) * NS_PER_S


def test_the_default_site_is_synthetic() -> None:
    assert (SYNTHETIC_SITE.latitude_deg, SYNTHETIC_SITE.longitude_deg) == (55.0, 0.0)
    with pytest.raises(ValueError, match="latitude"):
        Site(latitude_deg=95.0)


def test_the_sun_follows_the_seasons() -> None:
    # Summer solstice at noon: altitude 90 - 55 + 23.44 = 58.4 degrees.
    noon = float(sun_altitude_deg(SYNTHETIC_SITE, utc_ns(2026, 6, 21, 12, 2)))
    assert noon == pytest.approx(58.4, abs=0.4)
    # Winter solstice at midnight: the sun is 90 - 55 + 23.44 = 58.4 degrees below the horizon.
    midnight = float(sun_altitude_deg(SYNTHETIC_SITE, utc_ns(2026, 12, 21, 0, 0)))
    assert midnight == pytest.approx(-58.4, abs=0.4)
    # The default virtual start (1 January 00:00 UTC) is deep night.
    assert float(sun_altitude_deg(SYNTHETIC_SITE, DEFAULT_START_UTC_NS)) < -50
    # Arrays work and the sun rises through the morning.
    hours = np.arange(6, 12) * 3600 * NS_PER_S + utc_ns(2026, 3, 20)
    altitude = np.asarray(sun_altitude_deg(SYNTHETIC_SITE, hours))
    assert np.all(np.diff(altitude) > 0)


def test_the_sky_brightens_through_twilight() -> None:
    dark = 20.5
    altitudes = np.array([-30.0, -18.0, -12.0, -6.0, 0.0, 20.0])
    sky = np.asarray(sky_brightness_mag_arcsec2(dark, altitudes))
    assert sky[0] == pytest.approx(dark)
    assert sky[1] == pytest.approx(dark)
    assert np.all(np.diff(sky) <= 0)  # smaller magnitudes are brighter
    assert sky[3] < dark - 7  # at -6 degrees, the sky is many magnitudes brighter
    assert sky[-1] < 5  # daylight near the pole is 4.2 mag/arcsec^2
    assert float(sky_brightness_mag_arcsec2(dark, -5.0, twilight=False)) == dark
    flat = np.asarray(sky_brightness_mag_arcsec2(dark, altitudes, twilight=False))
    assert np.all(flat == dark)


def test_the_daylight_sky_near_the_pole_is_the_measured_value() -> None:
    # Nickel and Calderwood (2021), Figure 2: the median V sky 66 to 96 degrees from the sun, with
    # the sun 10 to 52 degrees high, is 4.2 mag/arcsec^2 (docs/research-notes.md).
    dark = SimOptions().sky_mag_arcsec2
    assert pytest.approx(4.2) == DAYLIGHT_SKY_MAG_ARCSEC2
    daylight = np.asarray(
        sky_brightness_mag_arcsec2(dark, np.array([10.0, 20.0, 35.0, 58.4, 90.0]))
    )
    assert daylight == pytest.approx(np.full(5, 4.2), abs=1e-9)
    # From sunset to +10 degrees, the sky brightens linearly from the twilight table's 6.0.
    assert float(sky_brightness_mag_arcsec2(dark, 0.0)) == pytest.approx(6.0, abs=1e-9)
    assert float(sky_brightness_mag_arcsec2(dark, 3.0)) == pytest.approx(5.46, abs=1e-9)
    # The sun adds its light to the dark sky, so every site has the same day, and a darker site
    # keeps its darker night.
    for site_dark in (18.0, 21.0, 22.0):
        day = float(sky_brightness_mag_arcsec2(site_dark, 30.0))
        assert day == pytest.approx(DAYLIGHT_SKY_MAG_ARCSEC2, abs=1e-4)  # mag/arcsec^2
        night = float(sky_brightness_mag_arcsec2(site_dark, -20.0))
        assert night == pytest.approx(site_dark, abs=1e-9)
    # In twilight, the sunlight of the default site (its sky is 20.0 at -15 degrees) adds to a
    # darker sky: -2.5 log10(10^(-0.4 * 22) + 10^(-0.4 * 20.0) - 10^(-0.4 * 20.5)) is 20.69.
    twilight = float(sky_brightness_mag_arcsec2(22.0, -15.0))
    assert twilight == pytest.approx(20.69, abs=0.005)  # mag/arcsec^2


def test_airmass() -> None:
    assert airmass(0.0) == pytest.approx(1.0, abs=1e-3)
    assert airmass(60.0) == pytest.approx(1.995, abs=0.01)
    # The celestial pole at a latitude of 55 degrees is 35 degrees from the zenith.
    assert airmass(35.0) == pytest.approx(1.22, abs=0.01)
    assert airmass(30.0) == pytest.approx(1.155, abs=0.002)  # the research notes use this value


# --- clouds -----------------------------------------------------------------------------


def test_a_cloud_event_dips_and_recovers() -> None:
    start = DEFAULT_START_UTC_NS + 100 * NS_PER_S
    clouds = Clouds(events=(CloudEvent(start, duration_s=60.0, transmission=0.2, ramp_s=10.0),))
    assert clouds.transparency(start - NS_PER_S) == 1.0
    assert clouds.transparency(start + 30 * NS_PER_S) == pytest.approx(0.2)
    assert clouds.transparency(start + 5 * NS_PER_S) == pytest.approx(0.6)  # half way down
    assert clouds.transparency(start + 61 * NS_PER_S) == 1.0
    times = start + np.arange(0, 61, 5) * NS_PER_S
    series = np.asarray(clouds.transparency(times))
    assert series.min() == pytest.approx(0.2)
    assert np.all(np.diff(series[:3]) < 0)
    assert clouds.mean_transparency(start + 20 * NS_PER_S, 20.0) == pytest.approx(0.2)


def test_events_multiply_and_the_baseline_scales() -> None:
    start = DEFAULT_START_UTC_NS
    clouds = Clouds(
        baseline=0.9,
        events=(
            CloudEvent(start, 100.0, 0.5, ramp_s=1.0),
            CloudEvent(start, 100.0, 0.5, ramp_s=1.0),
        ),
    )
    assert clouds.transparency(start + 50 * NS_PER_S) == pytest.approx(0.9 * 0.25)
    with pytest.raises(ValueError, match="transmission"):
        CloudEvent(start, 10.0, transmission=1.5)
    with pytest.raises(ValueError, match="baseline"):
        Clouds(baseline=0.0)


# --- scintillation ----------------------------------------------------------------------


def test_scintillation_index_matches_the_research_notes() -> None:
    config = ScintillationConfig()
    # "Polaris at airmass 1.155: rms 0.30 to 0.45 per 2 ms frame, falling about as 1/T above 3 ms."
    assert 0.30 < config.index(0.002, 1.155) < 0.45
    assert config.index(0.010, 1.155) == pytest.approx(0.24, abs=0.03)
    # Above the knee, the variance falls as 1/T.
    ratio = config.index(0.100, 1.155) ** 2 / config.index(0.050, 1.155) ** 2
    assert ratio == pytest.approx(0.5, abs=0.04)
    # Young's law: the rms grows as airmass^1.5.
    assert config.index(0.002, 2.0) / config.index(0.002, 1.155) == pytest.approx(
        (2.0 / 1.155) ** 1.5
    )
    assert ScintillationConfig(enabled=False).index(0.002, 1.5) == 0.0
    with pytest.raises(ValueError, match="knee_s"):
        ScintillationConfig(knee_s=0.0)


def test_the_scintillation_process_is_gaussian_with_an_exponential_correlation() -> None:
    process = ScintillationProcess(seed=1, correlation_time_s=0.003)
    t = np.arange(40_000) * 0.0005  # 20 s at 2 kHz
    values = process.values(t)
    assert values.mean() == pytest.approx(0.0, abs=0.1)
    assert values.std() == pytest.approx(1.0, abs=0.1)
    # The correlation falls to exp(-1) at one correlation time (6 samples).
    centred = values - values.mean()
    lag = 6
    correlation = float(np.mean(centred[:-lag] * centred[lag:]) / np.mean(centred**2))
    assert correlation == pytest.approx(math.exp(-1), abs=0.12)
    # A pure function of the time: the order of the calls does not matter.
    assert process.value(1.234) == process.value(1.234)
    assert np.array_equal(process.values(t[:10]), values[:10])
    assert ScintillationProcess(seed=2, correlation_time_s=0.003).value(1.234) != process.value(
        1.234
    )


def test_the_flux_factor_has_unit_mean_and_the_requested_rms() -> None:
    gaussian = np.random.default_rng(0).standard_normal(200_000)
    factors = np.asarray(flux_factor(0.3, gaussian))
    assert factors.mean() == pytest.approx(1.0, abs=0.005)
    assert factors.std() == pytest.approx(0.3, abs=0.005)
    assert float(flux_factor(0.0, 1.5)) == 1.0
    assert np.all(factors > 0)
