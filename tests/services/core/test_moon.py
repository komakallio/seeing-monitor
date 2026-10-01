"""The low-precision Moon: phases at known lunations, and the elevation against astropy."""

from __future__ import annotations

import math
import warnings
from itertools import pairwise

import pytest

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.services.core.moon import moon_elevation_deg, moon_illumination, moon_position

# Moon phases of 2026 (the instant of the phase, UTC), and the lit fraction that they mean.
PHASES = [
    ("2026-01-03T10:03:00Z", 1.0),  # full
    ("2026-01-10T15:48:00Z", 0.5),  # last quarter
    ("2026-01-18T19:52:00Z", 0.0),  # new
    ("2026-01-26T04:47:00Z", 0.5),  # first quarter
    ("2026-02-01T22:09:00Z", 1.0),  # full
    ("2026-02-17T12:01:00Z", 0.0),  # new
]


class TestPhase:
    @pytest.mark.parametrize(("when", "lit"), PHASES)
    def test_the_lit_fraction_follows_the_phases_of_2026(self, when: str, lit: float) -> None:
        assert moon_illumination(iso_to_utc_ns(when)) == pytest.approx(lit, abs=0.02)

    def test_the_fraction_stays_between_zero_and_one(self) -> None:
        start = iso_to_utc_ns("2026-01-01T00:00:00Z")
        for hour in range(0, 24 * 60, 7):
            value = moon_illumination(start + hour * 3600 * NS_PER_S)
            assert 0.0 <= value <= 1.0


class TestPosition:
    def test_the_moon_moves_east_by_thirteen_degrees_a_day(self) -> None:
        start = iso_to_utc_ns("2026-03-10T00:00:00Z")
        first = moon_position(start).right_ascension_deg
        second = moon_position(start + 86_400 * NS_PER_S).right_ascension_deg
        assert (second - first) % 360.0 == pytest.approx(13.2, abs=2.0)

    def test_the_horizontal_parallax_is_about_a_degree(self) -> None:
        for day in range(0, 60, 5):
            parallax = moon_position(
                iso_to_utc_ns("2026-01-01T00:00:00Z") + day * 86_400 * NS_PER_S
            )
            assert 0.88 < parallax.parallax_deg < 1.03

    def test_a_full_moon_in_winter_climbs_high_at_midnight_on_the_prime_meridian(self) -> None:
        # The full Moon is opposite the Sun, so it stands at 90 - 55 + 24 degrees at midnight.
        elevation = moon_elevation_deg(iso_to_utc_ns("2026-01-03T23:55:00Z"), 55.0, 0.0)
        assert elevation == pytest.approx(58.7, abs=1.5)

    def test_the_moon_rises_and_sets_once_a_day_at_the_equator(self) -> None:
        start = iso_to_utc_ns("2026-05-01T00:00:00Z")
        signs = [
            moon_elevation_deg(start + minute * 60 * NS_PER_S, 0.0, 30.0) > 0
            for minute in range(0, 24 * 60, 10)
        ]
        crossings = sum(1 for a, b in pairwise(signs) if a != b)
        assert crossings in (1, 2, 3)  # a rise and a set, and the day may hold only one of them


class TestAgainstAstropy:
    def test_the_elevation_agrees_within_a_third_of_a_degree(self) -> None:
        astropy = pytest.importorskip("astropy")
        del astropy
        from astropy import units as u
        from astropy.coordinates import AltAz, EarthLocation, get_body
        from astropy.time import Time

        latitude, longitude = 55.0, 20.0
        place = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=0 * u.m)
        start = iso_to_utc_ns("2026-01-01T00:00:00Z")
        worst = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index in range(40):
                t_ns = start + round(index * 8.3 * 86_400 * NS_PER_S / 1.37)
                when = Time(t_ns / 1e9, format="unix", scale="utc")
                moon = get_body("moon", when, place)
                reference = float(moon.transform_to(AltAz(obstime=when, location=place)).alt.deg)
                worst = max(worst, abs(moon_elevation_deg(t_ns, latitude, longitude) - reference))
        assert worst < 0.33
        assert not math.isnan(worst)
