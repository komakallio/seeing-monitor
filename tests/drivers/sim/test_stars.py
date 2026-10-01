"""Star fields, the pointing model, and the rotation of the sky."""

from __future__ import annotations

import math

import numpy as np
import pytest

from seeingmon.clock import DEFAULT_START_UTC_NS, NS_PER_S
from seeingmon.drivers.sim.params import SimParams
from seeingmon.drivers.sim.stars import (
    POLARIS_DEC_DEG,
    POLARIS_MAG,
    POLARIS_RA_DEG,
    SIDEREAL_RATE_RAD_PER_S,
    Pointing,
    SkyProjector,
    StarField,
    make_polar_field,
    polaris_field,
    sidereal_motion_arcsec_per_s,
)

BIN1 = SimParams.reference("bin1")
BIN2 = SimParams.reference("bin2")
T0 = DEFAULT_START_UTC_NS


def separation_deg(field: StarField, ra_deg: float, dec_deg: float) -> np.ndarray:
    ra, dec = np.radians(field.ra_deg), np.radians(field.dec_deg)
    ra0, dec0 = math.radians(ra_deg), math.radians(dec_deg)
    cosine = np.sin(dec) * math.sin(dec0) + np.cos(dec) * math.cos(dec0) * np.cos(ra - ra0)
    return np.asarray(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


@pytest.fixture(scope="module")
def field() -> StarField:
    return make_polar_field(seed=3)


@pytest.fixture(scope="module")
def projector(field: StarField) -> SkyProjector:
    return SkyProjector(field, Pointing())


# --- the catalog --------------------------------------------------------------------------


def test_the_sidereal_rate_is_15_arcsec_per_second() -> None:
    assert pytest.approx(15.041, abs=0.001) == SIDEREAL_RATE_RAD_PER_S * 206_264.806
    # "A star at angular distance theta from the pole moves 15.041 x sin(theta) arcsec per second."
    assert sidereal_motion_arcsec_per_s(0.618) == pytest.approx(0.1622, abs=0.0005)
    assert sidereal_motion_arcsec_per_s(3.3) == pytest.approx(0.866, abs=0.002)


def test_the_polar_field_is_deterministic_and_has_polaris(field: StarField) -> None:
    again = make_polar_field(seed=3)
    assert np.array_equal(field.ra_deg, again.ra_deg)
    assert np.array_equal(field.mag, again.mag)
    other = make_polar_field(seed=4)
    assert len(other) != len(field) or not np.array_equal(field.mag, other.mag)
    brightest = int(np.argmin(field.mag))
    assert field.mag[brightest] == pytest.approx(POLARIS_MAG)
    assert field.ra_deg[brightest] == pytest.approx(POLARIS_RA_DEG)
    assert 90.0 - field.dec_deg[brightest] == pytest.approx(0.618, abs=1e-6)


def test_the_star_counts_match_the_research_notes(field: StarField) -> None:
    """233, 510, and 1,085 stars to magnitudes 11, 12, and 13 in the 1.83 degree design circle."""
    inside = separation_deg(field, POLARIS_RA_DEG, POLARIS_DEC_DEG) < 1.833
    for limit, expected in ((11, 233), (12, 510), (13, 1085)):
        count = int(np.sum(inside & (field.mag < limit)))
        # Poisson counts: four standard deviations.
        assert abs(count - expected) < 4 * math.sqrt(expected), (limit, count)
    assert np.all(field.dec_deg >= 75.0 - 1e-9)  # inside the 15 degree cap
    assert np.all(np.diff(field.mag[:-2]) >= 0)  # sorted by brightness (Polaris is appended)


def test_the_field_validates_its_input() -> None:
    with pytest.raises(ValueError, match="one length"):
        StarField.from_arrays([1.0, 2.0], [80.0], [5.0])
    with pytest.raises(ValueError, match="declinations"):
        StarField.from_arrays([1.0], [95.0], [5.0])
    with pytest.raises(ValueError, match="finite"):
        StarField.from_arrays([math.nan], [80.0], [5.0])
    with pytest.raises(ValueError, match="cap_radius"):
        make_polar_field(cap_radius_deg=0.1)
    only = polaris_field(include_polaris_b=False)
    assert len(only) == 1
    assert len(polaris_field()) == 2
    subset = polaris_field().subset(np.array([False, True]))
    assert float(subset.mag[0]) == pytest.approx(8.7)


# --- projection and the rotation of the sky ---------------------------------------------


def test_polaris_starts_at_the_centre_with_the_pole_above(projector: SkyProjector) -> None:
    x, y = projector.project(T0, BIN1.pixel_rad, BIN1.width, BIN1.height)
    brightest = int(np.argmin(projector.stars.mag))
    assert (x[brightest], y[brightest]) == pytest.approx(((8288 - 1) / 2, (5644 - 1) / 2), abs=1e-6)
    pole_x, pole_y = projector.pole_pixel(T0, BIN1.pixel_rad, BIN1.width, BIN1.height)
    # The pole is 0.618 degrees (2,225 arcsec) above Polaris: 1,165 pixels in bin1.
    assert pole_x == pytest.approx((8288 - 1) / 2, abs=1e-3)
    assert (5644 - 1) / 2 - pole_y == pytest.approx(0.618 * 3600 / 1.910, rel=1e-3)
    bin2_x, bin2_y = projector.pole_pixel(T0, BIN2.pixel_rad, BIN2.width, BIN2.height)
    assert (2822 - 1) / 2 - bin2_y == pytest.approx(0.618 * 3600 / 3.820, rel=1e-3)
    assert bin2_x == pytest.approx((4144 - 1) / 2, abs=1e-3)


@pytest.mark.parametrize(("seconds", "arcsec"), [(60, 9.7), (120, 19.5), (600, 97.3)])
def test_polaris_drifts_as_the_research_notes_say(
    projector: SkyProjector, seconds: int, arcsec: float
) -> None:
    index = int(np.argmin(projector.stars.mag))
    x0, y0 = projector.project(T0, BIN1.pixel_rad, BIN1.width, BIN1.height)
    x1, y1 = projector.project(T0 + seconds * NS_PER_S, BIN1.pixel_rad, BIN1.width, BIN1.height)
    moved_px = math.hypot(x1[index] - x0[index], y1[index] - y0[index])
    assert moved_px * BIN1.plate_scale_arcsec_per_px == pytest.approx(arcsec, rel=0.01)
    # Polaris sits below the pole, and the sky turns counter-clockwise, so Polaris moves right.
    assert x1[index] > x0[index]


def test_far_stars_trail_faster(projector: SkyProjector) -> None:
    """A star 3 degrees from the pole moves 15.041 sin(theta) arcsec per second."""
    polar = 90.0 - projector.stars.dec_deg
    selected = np.flatnonzero((np.abs(polar - 3.0) < 0.05) & (projector.stars.mag < 12))
    assert len(selected) > 0
    index = int(selected[0])
    x0, y0 = projector.project(T0, BIN1.pixel_rad, BIN1.width, BIN1.height)
    x1, y1 = projector.project(T0 + 5 * NS_PER_S, BIN1.pixel_rad, BIN1.width, BIN1.height)
    moved = (
        math.hypot(x1[index] - x0[index], y1[index] - y0[index]) * BIN1.plate_scale_arcsec_per_px
    )
    assert moved == pytest.approx(5 * sidereal_motion_arcsec_per_s(float(polar[index])), rel=0.01)


def test_the_projection_is_gnomonic_about_the_optical_axis() -> None:
    """Polaris B lies 18.3 arcsec from Polaris, which is 9.6 pixels in bin1."""
    both = SkyProjector(polaris_field(), Pointing())
    x, y = both.project(T0, BIN1.pixel_rad, BIN1.width, BIN1.height)
    assert math.hypot(x[1] - x[0], y[1] - y[0]) == pytest.approx(18.3 / 1.910, rel=0.002)
    # The pattern rotates rigidly: the separation does not change as the sky turns.
    x2, y2 = both.project(T0 + 3600 * NS_PER_S, BIN1.pixel_rad, BIN1.width, BIN1.height)
    assert math.hypot(x2[1] - x2[0], y2[1] - y2[0]) == pytest.approx(
        math.hypot(x[1] - x[0], y[1] - y[0]), rel=2e-3
    )


@pytest.mark.parametrize(
    ("roll", "direction"), [(0.0, (0, -1)), (90.0, (-1, 0)), (180.0, (0, 1)), (270.0, (1, 0))]
)
def test_roll_is_the_position_angle_of_north(
    field: StarField, roll: float, direction: tuple[int, int]
) -> None:
    """With roll 0 the pole is above the centre. Roll turns it counter-clockwise on the image."""
    projector = SkyProjector(field, Pointing(roll_deg=roll))
    pole_x, pole_y = projector.pole_pixel(T0, BIN1.pixel_rad, BIN1.width, BIN1.height)
    centre_x, centre_y = (BIN1.width - 1) / 2, (BIN1.height - 1) / 2
    offset = math.hypot(pole_x - centre_x, pole_y - centre_y)
    assert offset == pytest.approx(0.618 * 3600 / 1.910, rel=1e-3)
    assert (pole_x - centre_x) / offset == pytest.approx(direction[0], abs=1e-3)
    assert (pole_y - centre_y) / offset == pytest.approx(direction[1], abs=1e-3)
    assert projector.roll_deg() == roll


def test_the_projector_keeps_only_stars_that_can_reach_the_sensor(
    field: StarField, projector: SkyProjector
) -> None:
    assert 3000 < len(projector.stars) < 12_000
    polar = 90.0 - projector.stars.dec_deg
    assert polar.max() < 0.618 + 2.9 + 0.5 + 1e-9
    # Every star of the full field that lands on the sensor at some time is kept.
    for hours in (0.0, 5.0, 11.0):
        full = SkyProjector(field, Pointing(), max_field_radius_deg=90.0)
        t = T0 + round(hours * 3600 * NS_PER_S)
        fx, fy = full.project(t, BIN1.pixel_rad, BIN1.width, BIN1.height)
        on_sensor = (fx >= 0) & (fx < BIN1.width) & (fy >= 0) & (fy < BIN1.height)
        kept_ids = {int(i) for i in projector.indices}
        on_sensor_ids = {int(i) for i in full.indices[on_sensor]}
        assert on_sensor_ids <= kept_ids
        assert 600 < int(np.sum(on_sensor & (full.stars.mag < 13))) < 1700


def test_a_subset_projects_like_the_whole(projector: SkyProjector) -> None:
    indices = np.array([0, 5, 17])
    x_all, y_all = projector.project(T0, BIN2.pixel_rad, BIN2.width, BIN2.height)
    x, y = projector.project_stars(indices, T0, BIN2.pixel_rad, BIN2.width, BIN2.height)
    assert np.allclose(x, x_all[indices])
    assert np.allclose(y, y_all[indices])


def test_the_pointing_validates() -> None:
    with pytest.raises(ValueError, match="pole"):
        Pointing(dec_deg=90.0)
    assert Pointing().pole_distance_deg == pytest.approx(0.618)
