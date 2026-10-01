"""Vector and rotation helpers: conventions, round trips, and the pole."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.survey import geometry as g


def random_rotation(rng: np.random.Generator) -> g.FloatArray:
    return g.exp_so3(rng.normal(size=3))


def test_rot_z_turns_x_toward_y() -> None:
    np.testing.assert_allclose(g.rot_z(np.pi / 2) @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(g.rot_x(np.pi / 2) @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(g.rot_y(np.pi / 2) @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], atol=1e-15)


@given(
    ra=st.floats(0.0, 360.0, exclude_max=True),
    dec=st.floats(-90.0, 90.0),
)
def test_radec_and_vector_round_trip(ra: float, dec: float) -> None:
    vector = g.radec_to_vector(ra, dec)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-15)
    ra_back, dec_back = g.vector_to_radec(vector)
    assert float(dec_back) == pytest.approx(dec, abs=1e-9)
    if abs(dec) < 89.999999:  # the right ascension of a vector at the pole is arbitrary
        assert float(((ra_back - ra + 180.0) % 360.0) - 180.0) == pytest.approx(0.0, abs=1e-8)


def test_the_pole_has_right_ascension_zero_by_convention() -> None:
    ra, dec = g.vector_to_radec([0.0, 0.0, 1.0])
    assert (float(ra), float(dec)) == (0.0, 90.0)


def test_angular_separation_is_accurate_for_tiny_and_opposite_angles() -> None:
    tiny = 1e-9  # 0.2 milliarcseconds
    a = g.radec_to_vector(10.0, 89.0)
    b = g.normalize(a + tiny * np.array([0.0, 0.0, 1.0]) - tiny * a * a[2])
    assert float(g.angular_separation(a, b)) == pytest.approx(
        tiny * np.sqrt(1 - a[2] ** 2), rel=1e-3
    )
    assert float(g.angular_separation([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])) == pytest.approx(np.pi)


@given(
    x=st.floats(-3.0, 3.0), y=st.floats(-3.0, 3.0), z=st.floats(-3.0, 3.0)
)  # up to just below pi in angle
def test_exp_and_log_round_trip(x: float, y: float, z: float) -> None:
    rotation_vector = np.array([x, y, z])
    angle = float(np.linalg.norm(rotation_vector))
    if angle > 3.1:
        rotation_vector = rotation_vector / angle * 3.1
    rotation = g.exp_so3(rotation_vector)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(g.log_so3(rotation), rotation_vector, atol=1e-9)


def test_log_of_a_half_turn_gives_a_vector_of_length_pi() -> None:
    rotation = g.rot_z(np.pi)
    vector = g.log_so3(rotation)
    assert float(np.linalg.norm(vector)) == pytest.approx(np.pi)
    np.testing.assert_allclose(g.exp_so3(vector), rotation, atol=1e-12)


def test_best_rotation_recovers_a_known_rotation() -> None:
    rng = np.random.default_rng(10)
    truth = random_rotation(rng)
    source = g.normalize(rng.normal(size=(200, 3)))
    target = source @ truth.T + rng.normal(scale=1e-6, size=(200, 3))
    estimate = g.best_rotation(source, target)
    np.testing.assert_allclose(estimate, truth, atol=1e-5)


def test_best_rotation_honors_weights() -> None:
    rng = np.random.default_rng(11)
    truth = random_rotation(rng)
    source = g.normalize(rng.normal(size=(50, 3)))
    target = source @ truth.T
    target[:5] += 0.5  # five bad pairs
    weights = np.ones(50)
    weights[:5] = 1e-9
    estimate = g.best_rotation(source, target, weights)
    np.testing.assert_allclose(estimate, truth, atol=1e-5)


def test_nearest_rotation_projects_a_perturbed_matrix() -> None:
    rng = np.random.default_rng(12)
    rotation = random_rotation(rng)
    nearest = g.nearest_rotation(rotation + rng.normal(scale=1e-4, size=(3, 3)))
    assert np.linalg.det(nearest) == pytest.approx(1.0)
    np.testing.assert_allclose(nearest, rotation, atol=5e-4)


def test_nearest_rotation_never_returns_a_reflection() -> None:
    reflection = np.diag([1.0, 1.0, -1.0])
    assert np.linalg.det(g.nearest_rotation(reflection)) == pytest.approx(1.0)


@given(ra=st.floats(0.0, 360.0, exclude_max=True), dec=st.floats(-90.0, 90.0))
def test_tangent_basis_is_a_right_handed_frame(ra: float, dec: float) -> None:
    east, north, outward = g.tangent_basis(ra, dec)
    basis = np.stack([east, north, outward])
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(east, north), outward, atol=1e-12)
    np.testing.assert_allclose(outward, g.radec_to_vector(ra, dec), atol=1e-12)


def test_tangent_basis_at_the_pole_points_north_away_from_the_given_right_ascension() -> None:
    # At the pole with right ascension 0, north (the direction of increasing declination)
    # points along -x, the continuation of the meridian through the pole.
    east, north, outward = g.tangent_basis(0.0, 90.0)
    np.testing.assert_allclose(east, [0.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(north, [-1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(outward, [0.0, 0.0, 1.0], atol=1e-15)
