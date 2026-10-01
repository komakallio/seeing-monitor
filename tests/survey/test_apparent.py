"""Apparent places against `astropy` coordinate transformations, which serve as the truth."""

from __future__ import annotations

import warnings

import erfa
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.clock import NS_PER_S
from seeingmon.survey import apparent
from seeingmon.survey.geometry import (
    ARCSEC_PER_RAD,
    FloatArray,
    angular_separation,
    normalize,
    radec_to_vector,
    rot_z,
)

# 2026-01-01T00:00:00Z. Monthly dates sample the whole annual aberration cycle.
START_UTC_NS = 1_767_225_600 * NS_PER_S
MONTHLY_UTC_NS = [START_UTC_NS + round(month * 30.4375 * 86_400 * NS_PER_S) for month in range(12)]

# The agreement that the pointing fit needs: 0.01 arcsec is 0.003 pixel in bin2.
TOLERANCE_ARCSEC = 0.01


def random_cap_stars(
    count: int, seed: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Stars uniformly distributed within 15 degrees of the pole, with proper motions."""
    rng = np.random.default_rng(seed)
    # Uniform on the sphere in the cap: cos(polar distance) is uniform.
    cos_polar = rng.uniform(np.cos(np.radians(15.0)), 1.0, count)
    dec = 90.0 - np.degrees(np.arccos(cos_polar))
    ra = rng.uniform(0.0, 360.0, count)
    pm_ra = rng.normal(0.0, 60.0, count)  # mas/yr
    pm_dec = rng.normal(0.0, 60.0, count)
    parallax = rng.uniform(0.2, 25.0, count)  # mas
    return ra, dec, pm_ra, pm_dec, parallax


def astropy_cirs_vectors(
    ra: FloatArray,
    dec: FloatArray,
    pm_ra: FloatArray,
    pm_dec: FloatArray,
    parallax: FloatArray,
    t_utc_ns: int,
) -> FloatArray:
    """The CIRS unit vectors of the same stars from `astropy`."""
    units = pytest.importorskip("astropy.units")
    coordinates = pytest.importorskip("astropy.coordinates")
    time = pytest.importorskip("astropy.time")

    stars = coordinates.SkyCoord(
        ra=ra * units.deg,
        dec=dec * units.deg,
        pm_ra_cosdec=pm_ra * units.mas / units.yr,
        pm_dec=pm_dec * units.mas / units.yr,
        distance=(1000.0 / parallax) * units.pc,
        radial_velocity=np.zeros(ra.shape) * units.km / units.s,
        frame="icrs",
        obstime=time.Time("J2016.0"),
    )
    observed = time.Time(t_utc_ns / NS_PER_S, format="unix", scale="utc")
    exceptions = pytest.importorskip("astropy.utils.exceptions")
    with warnings.catch_warnings():
        # ERFA calls a date more than five years past its leap-second table "dubious", and
        # astropy warns about polar motion after the end of its bundled IERS table. Neither
        # affects the CIRS direction of a star.
        warnings.simplefilter("ignore", erfa.ErfaWarning)
        warnings.simplefilter("ignore", exceptions.AstropyWarning)
        moved = stars.apply_space_motion(new_obstime=observed)
        cirs = moved.transform_to(coordinates.CIRS(obstime=observed))
    return radec_to_vector(cirs.ra.deg, cirs.dec.deg)


def max_separation_arcsec(a: FloatArray, b: FloatArray) -> float:
    return float(np.max(angular_separation(a, b)) * ARCSEC_PER_RAD)


@pytest.mark.parametrize("t_utc_ns", MONTHLY_UTC_NS)
def test_apparent_places_match_astropy_across_the_year(t_utc_ns: int) -> None:
    ra, dec, pm_ra, pm_dec, parallax = random_cap_stars(400, seed=1)
    epoch = apparent.epoch_from_utc_ns(t_utc_ns)
    mine = apparent.apparent_vectors(ra, dec, pm_ra, pm_dec, parallax, epoch)
    truth = astropy_cirs_vectors(ra, dec, pm_ra, pm_dec, parallax, t_utc_ns)
    assert max_separation_arcsec(mine, truth) < TOLERANCE_ARCSEC


def test_apparent_places_match_astropy_for_fast_movers_years_later() -> None:
    # Twelve years past the catalog epoch, with proper motions of up to 1 arcsec per year.
    t_utc_ns = START_UTC_NS + 2 * 365 * 86_400 * NS_PER_S
    ra, dec, _, _, parallax = random_cap_stars(200, seed=2)
    rng = np.random.default_rng(3)
    pm_ra = rng.uniform(-1000.0, 1000.0, 200)
    pm_dec = rng.uniform(-1000.0, 1000.0, 200)
    epoch = apparent.epoch_from_utc_ns(t_utc_ns)
    mine = apparent.apparent_vectors(ra, dec, pm_ra, pm_dec, parallax, epoch)
    truth = astropy_cirs_vectors(ra, dec, pm_ra, pm_dec, parallax, t_utc_ns)
    assert max_separation_arcsec(mine, truth) < TOLERANCE_ARCSEC


def test_stars_without_parallax_match_astropy() -> None:
    # The pipeline gives stars without a parallax the value 0. Astropy needs a distance, so
    # use a very large one.
    ra, dec, pm_ra, pm_dec, _ = random_cap_stars(100, seed=4)
    parallax = np.full(100, 1e-4)
    t_utc_ns = MONTHLY_UTC_NS[5]
    epoch = apparent.epoch_from_utc_ns(t_utc_ns)
    mine = apparent.apparent_vectors(ra, dec, pm_ra, pm_dec, np.zeros(100), epoch)
    truth = astropy_cirs_vectors(ra, dec, pm_ra, pm_dec, parallax, t_utc_ns)
    assert max_separation_arcsec(mine, truth) < TOLERANCE_ARCSEC


def test_stars_at_and_near_the_pole_have_no_singularity() -> None:
    # Declinations from 0.1 degree to 1 micro-arcsecond from the pole, and the pole itself.
    dec = np.array([89.9, 89.99, 89.9999, 90.0 - 1e-9, 90.0 - 1e-12, 90.0])
    ra = np.array([10.0, 100.0, 200.0, 300.0, 0.0, 0.0])
    zeros = np.zeros_like(dec)
    parallax = np.full_like(dec, 5.0)
    t_utc_ns = MONTHLY_UTC_NS[8]
    epoch = apparent.epoch_from_utc_ns(t_utc_ns)
    mine = apparent.apparent_vectors(ra, dec, zeros, zeros, parallax, epoch)
    assert np.all(np.isfinite(mine))
    np.testing.assert_allclose(np.linalg.norm(mine, axis=-1), 1.0, atol=1e-14)
    truth = astropy_cirs_vectors(ra, dec, zeros, zeros, parallax, t_utc_ns)
    assert max_separation_arcsec(mine, truth) < TOLERANCE_ARCSEC


def test_the_pole_has_the_same_apparent_place_for_every_right_ascension() -> None:
    epoch = apparent.epoch_from_utc_ns(MONTHLY_UTC_NS[3])
    ra = np.array([0.0, 17.0, 123.4, 359.9])
    dec = np.full(4, 90.0)
    zeros = np.zeros(4)
    mine = apparent.apparent_vectors(ra, dec, zeros, zeros, zeros, epoch)
    assert max_separation_arcsec(mine, np.broadcast_to(mine[0], mine.shape)) < 1e-6


@given(
    ra=st.floats(0.0, 360.0, exclude_max=True),
    polar_distance_arcsec=st.floats(0.0, 3600.0),
    month=st.integers(0, 11),
)
def test_a_star_near_the_pole_moves_by_less_than_the_aberration(
    ra: float, polar_distance_arcsec: float, month: int
) -> None:
    dec = 90.0 - polar_distance_arcsec / 3600.0
    epoch = apparent.epoch_from_utc_ns(MONTHLY_UTC_NS[month])
    apparent_place = apparent.apparent_vectors(ra, dec, 0.0, 0.0, 0.0, epoch)
    # Without aberration or deflection, the place is the catalog direction turned by the
    # bias-precession-nutation matrix.
    rotated = epoch.npb @ radec_to_vector(ra, dec)
    shift = float(angular_separation(apparent_place, rotated)) * ARCSEC_PER_RAD
    assert 15.0 < shift < 21.0  # 20.5 arcsec times the sine of the angle to the apex


def test_the_aberration_ellipse_has_its_published_size() -> None:
    # Over a year, a star near the pole traces an ellipse of 20.5 and 20.5 sin(66.56 deg)
    # = 18.8 arcsec semi-axes (research notes, "Long-term science estimates").
    shifts = []
    for t_utc_ns in MONTHLY_UTC_NS:
        epoch = apparent.epoch_from_utc_ns(t_utc_ns)
        place = apparent.apparent_vectors(0.0, 89.5, 0.0, 0.0, 0.0, epoch)
        shifts.append(float(angular_separation(place, epoch.npb @ radec_to_vector(0.0, 89.5))))
    arcsec = np.array(shifts) * ARCSEC_PER_RAD
    assert arcsec.max() == pytest.approx(20.5, abs=0.3)
    assert arcsec.min() == pytest.approx(18.8, abs=0.3)


def test_astrometric_direction_inverts_the_apparent_place() -> None:
    ra, dec, _, _, _ = random_cap_stars(300, seed=5)
    zeros = np.zeros(300)
    epoch = apparent.epoch_from_utc_ns(MONTHLY_UTC_NS[2])
    cirs = apparent.apparent_vectors(ra, dec, zeros, zeros, zeros, epoch)
    back = apparent.astrometric_from_apparent(cirs, epoch)
    assert max_separation_arcsec(back, radec_to_vector(ra, dec)) < 1e-6


def test_earth_rotation_angle_advances_by_a_turn_each_sidereal_day() -> None:
    sidereal_day_s = 86_164.0905
    t0 = MONTHLY_UTC_NS[0]
    t1 = t0 + round(sidereal_day_s * NS_PER_S)
    difference = (apparent.earth_rotation_angle(t1) - apparent.earth_rotation_angle(t0)) % (
        2.0 * np.pi
    )
    assert min(difference, 2.0 * np.pi - difference) < 2e-6  # 0.4 arcsec
    rate = (apparent.earth_rotation_angle(t0 + NS_PER_S) - apparent.earth_rotation_angle(t0)) % (
        2.0 * np.pi
    )
    assert rate == pytest.approx(apparent.EARTH_ROTATION_RATE_RAD_S, rel=1e-6)


def test_dut1_shifts_the_earth_rotation_angle() -> None:
    t = MONTHLY_UTC_NS[4]
    shift = apparent.earth_rotation_angle(t, dut1_s=0.5) - apparent.earth_rotation_angle(t)
    assert shift == pytest.approx(0.5 * apparent.EARTH_ROTATION_RATE_RAD_S, rel=1e-6)


def test_earth_fixed_matrix_agrees_with_the_erfa_celestial_to_terrestrial_matrix() -> None:
    t_utc_ns = MONTHLY_UTC_NS[6]
    epoch = apparent.epoch_from_utc_ns(t_utc_ns)
    utc1, utc2 = apparent.utc_two_part_jd(t_utc_ns)
    tt1, tt2 = erfa.taitt(*erfa.utctai(utc1, utc2))
    # GCRS to the terrestrial intermediate frame (no polar motion), IAU 2006/2000A.
    reference = erfa.c2t06a(tt1, tt2, utc1, utc2, 0.0, 0.0)
    mine = apparent.cirs_to_earth_fixed(epoch.era_rad) @ epoch.npb
    assert np.max(np.abs(mine - reference)) < 1e-9  # 0.0002 arcsec


def test_a_point_fixed_on_the_earth_rotates_in_cirs() -> None:
    # A point on the equator with longitude 0 is [1, 0, 0] in the Earth-fixed frame, so its
    # CIRS direction is [cos era, sin era, 0].
    era = 1.234
    fixed_to_cirs = apparent.cirs_to_earth_fixed(era).T
    np.testing.assert_allclose(fixed_to_cirs @ [1.0, 0.0, 0.0], [np.cos(era), np.sin(era), 0.0])
    np.testing.assert_allclose(
        apparent.cirs_to_earth_fixed(era) @ [np.cos(era), np.sin(era), 0.0],
        [1.0, 0.0, 0.0],
        atol=1e-15,
    )
    np.testing.assert_allclose(apparent.cirs_to_earth_fixed(era), rot_z(-era))


def test_polaris_is_where_the_literature_puts_it() -> None:
    star = apparent.POLARIS
    assert star.ra_deg == pytest.approx(37.9546, abs=1e-4)
    assert star.dec_deg == pytest.approx(89.2641, abs=1e-4)
    # Polaris lies 37.1 arcmin from the mean pole of date in late 2026 (research notes,
    # "Calculations", row "Polaris separation from the pole"). The apparent place adds the
    # aberration (up to 0.34 arcmin) and the nutation (up to 0.3 arcmin) to that.
    epoch = apparent.epoch_from_utc_ns(MONTHLY_UTC_NS[9])
    place = apparent.apparent_vectors_for(star, epoch)
    separation_arcmin = float(angular_separation(place, [0.0, 0.0, 1.0])) * ARCSEC_PER_RAD / 60.0
    assert separation_arcmin == pytest.approx(37.1, abs=0.7)


def test_normalize_keeps_unit_vectors_unit() -> None:
    v = normalize([[3.0, 4.0, 0.0]])
    np.testing.assert_allclose(v, [[0.6, 0.8, 0.0]])
