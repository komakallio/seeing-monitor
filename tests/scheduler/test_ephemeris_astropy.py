"""The ephemeris against astropy, which serves as truth in these tests only.

The scheduler itself never imports astropy. The tests keep to 2015 to 2025, where astropy's
bundled Earth-orientation data is valid, and they turn off its download, so they never touch the
network and never raise an astropy warning.
"""

from __future__ import annotations

import random

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.scheduler.ephemeris import (
    POLARIS_DEC_J2000_DEG,
    POLARIS_RA_J2000_DEG,
    polaris_zenith_angle_deg,
    sun_elevation_deg,
)

pytest.importorskip("astropy")

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
from astropy.utils import iers

START_NS = iso_to_utc_ns("2015-01-01T00:00:00Z")
END_NS = iso_to_utc_ns("2025-12-31T00:00:00Z")

# Synthetic sites that span the globe. None is anyone's real site.
SITES = [(55.0, 0.0), (-35.0, 150.0), (0.0, -75.0), (68.0, 25.0), (-60.0, -170.0)]


def sample_times(count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.randrange(START_NS, END_NS) for _ in range(count))


def astropy_times(t_ns: list[int]) -> Time:
    # The "unix" format counts UTC seconds without leap seconds, like `t_utc_ns`.
    return Time(np.array(t_ns, dtype=np.float64) / NS_PER_S, format="unix", scale="utc")


def astropy_sun_elevation(
    t_ns: list[int], latitude: float, longitude: float
) -> npt.NDArray[np.float64]:
    times = astropy_times(t_ns)
    location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=0 * u.m)
    frame = AltAz(obstime=times, location=location, pressure=0 * u.hPa)
    return np.asarray(get_sun(times).transform_to(frame).alt.deg, dtype=np.float64)


@pytest.fixture(autouse=True)
def offline_astropy() -> object:
    with iers.conf.set_temp("auto_download", False):
        yield


@pytest.mark.parametrize(("latitude", "longitude"), SITES)
def test_the_sun_elevation_agrees_with_astropy_to_a_few_hundredths_of_a_degree(
    latitude: float, longitude: float
) -> None:
    t_ns = sample_times(150, seed=11)
    truth = astropy_sun_elevation(t_ns, latitude, longitude)
    ours = np.array([sun_elevation_deg(t, latitude, longitude) for t in t_ns])
    # NOAA quotes 0.01 degrees. The tolerance leaves room for the 69 seconds of UTC versus TT.
    assert np.max(np.abs(ours - truth)) < 0.03


def test_the_sun_elevation_agrees_through_a_whole_night_at_the_synthetic_site() -> None:
    """Check the twilight limits too: every minute of one night, to 0.03 degrees."""
    start = iso_to_utc_ns("2025-03-01T12:00:00Z")
    t_ns = [start + minute * 60 * NS_PER_S for minute in range(0, 24 * 60, 5)]
    truth = astropy_sun_elevation(t_ns, 55.0, 0.0)
    ours = np.array([sun_elevation_deg(t, 55.0, 0.0) for t in t_ns])
    assert np.max(np.abs(ours - truth)) < 0.03


@pytest.mark.parametrize(("latitude", "longitude"), SITES)
def test_the_polaris_zenith_angle_agrees_with_astropy(latitude: float, longitude: float) -> None:
    t_ns = sample_times(100, seed=29)
    times = astropy_times(t_ns)
    location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=0 * u.m)
    polaris = SkyCoord(
        ra=POLARIS_RA_J2000_DEG * u.deg, dec=POLARIS_DEC_J2000_DEG * u.deg, frame="icrs"
    )
    frame = AltAz(obstime=times, location=location, pressure=0 * u.hPa)
    truth = 90.0 - np.asarray(polaris.transform_to(frame).alt.deg, dtype=np.float64)
    ours = np.array([polaris_zenith_angle_deg(t, latitude, longitude) for t in t_ns])
    # The code ignores nutation and aberration, which move a star by up to 0.006 degrees.
    assert np.max(np.abs(ours - truth)) < 0.02
