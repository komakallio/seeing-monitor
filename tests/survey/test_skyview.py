"""The sky view: the pole, its offset from the frame center, and the margin of the orbit."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from seeingmon.survey import apparent, skyview
from seeingmon.survey.geometry import ARCSEC_PER_RAD, rot_z
from seeingmon.survey.pointing import PointingSolution, polaris_colatitude_deg
from seeingmon.survey.skyview import build_sky_view, frame_center, position_angle_deg
from seeingmon.survey.wcs_fit import ROLL_MIN_DISTANCE_PX, CameraAttitude, pixel_center
from tests.survey.synth import NIGHT_UTC_NS, make_attitude

WIDTH, HEIGHT = 4144, 2822
SCALE_ARCSEC = 3.82
SCALE_RAD = SCALE_ARCSEC / ARCSEC_PER_RAD
COLATITUDE = 0.62  # degrees: about the colatitude of Polaris
PX_PER_DEG = 3600.0 / SCALE_ARCSEC


def attitude(
    distance_deg: float,
    roll_deg: float = 0.0,
    *,
    azimuth_deg: float = 0.0,
    parity: int = 1,
    center: tuple[float, float] | None = None,
) -> CameraAttitude:
    """A camera whose pole lies `distance_deg` from the frame center, at the angle `roll_deg`."""
    return CameraAttitude(
        rotation=make_attitude(distance_deg, azimuth_deg, roll_deg),
        scale_rad_px=SCALE_RAD,
        parity=parity,
        center_px=center or pixel_center(WIDTH, HEIGHT),
    )


def brute_margin(camera: CameraAttitude, colatitude_deg: float, width: int, height: int) -> float:
    """The distance from the orbit to the nearest frame edge, from 7200 points."""
    rho = math.radians(colatitude_deg)
    angles = np.linspace(0.0, 2.0 * math.pi, 7200, endpoint=False)
    vectors = np.stack(
        [
            math.sin(rho) * np.cos(angles),
            math.sin(rho) * np.sin(angles),
            np.full(7200, math.cos(rho)),
        ],
        axis=-1,
    )
    x, y, front = camera.project(vectors)
    assert front.all()
    return float(
        np.min([x.min() + 0.5, width - 0.5 - x.max(), y.min() + 0.5, height - 0.5 - y.max()])
    )


class TestPole:
    def test_a_pole_at_the_center_has_no_offset_and_no_roll(self) -> None:
        view = build_sky_view(attitude(0.0), WIDTH, HEIGHT, COLATITUDE)
        pole = view.pole
        assert pole.in_front
        assert pole.inside_frame
        assert pole.dx_px == pytest.approx(0.0, abs=1e-3)
        assert pole.dy_px == pytest.approx(0.0, abs=1e-3)
        assert pole.distance_px == pytest.approx(0.0, abs=1e-3)
        assert pole.distance_arcmin == pytest.approx(0.0, abs=1e-3)
        assert pole.roll_deg is None

    def test_a_pole_off_center_has_an_offset_a_distance_and_a_roll(self) -> None:
        # The roll is the position angle from image up toward image left, so a roll of 60 degrees
        # puts the pole up and to the left of the center.
        view = build_sky_view(attitude(1.0, 60.0), WIDTH, HEIGHT, COLATITUDE)
        pole = view.pole
        distance_px = math.tan(math.radians(1.0)) / SCALE_RAD
        assert pole.distance_px == pytest.approx(distance_px, abs=0.01)
        assert pole.dx_px == pytest.approx(-distance_px * math.sin(math.radians(60.0)), abs=0.01)
        assert pole.dy_px == pytest.approx(-distance_px * math.cos(math.radians(60.0)), abs=0.01)
        assert pole.distance_arcmin == pytest.approx(60.0, abs=0.01)  # the exact angle
        assert pole.roll_deg == pytest.approx(60.0, abs=0.01)
        assert pole.inside_frame

    def test_the_pixel_is_the_center_plus_the_offset(self) -> None:
        camera = attitude(0.7, -130.0)
        pole = build_sky_view(camera, WIDTH, HEIGHT, None).pole
        assert (pole.x_px, pole.y_px) == (
            pytest.approx(camera.pole_pixel()[0], abs=1e-3),  # type: ignore[index]
            pytest.approx(camera.pole_pixel()[1], abs=1e-3),  # type: ignore[index]
        )
        center_x, center_y = frame_center(WIDTH, HEIGHT)
        assert pole.x_px == pytest.approx(center_x + pole.dx_px, abs=2e-3)  # type: ignore[operator]
        assert pole.y_px == pytest.approx(center_y + pole.dy_px, abs=2e-3)  # type: ignore[operator]

    def test_the_roll_agrees_with_the_camera_model(self) -> None:
        for roll in (-170.0, -90.0, -10.0, 0.0, 45.0, 135.0, 179.0):
            camera = attitude(0.8, roll)
            assert build_sky_view(camera, WIDTH, HEIGHT, None).pole.roll_deg == pytest.approx(
                camera.roll_deg(), abs=1e-3
            )

    def test_a_pole_outside_the_frame_is_in_front_and_not_inside(self) -> None:
        # 2 degrees above the top edge of a frame that is 1.5 degrees tall.
        view = build_sky_view(attitude(2.5, 0.0), WIDTH, HEIGHT, COLATITUDE)
        pole = view.pole
        assert pole.in_front
        assert not pole.inside_frame
        assert pole.dy_px is not None
        assert pole.dy_px < -HEIGHT / 2
        assert pole.dx_px == pytest.approx(0.0, abs=0.01)
        assert pole.roll_deg == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.parametrize(
        ("roll", "inside"),
        [(0.0, True), (90.0, True), (180.0, True), (-90.0, True)],
    )
    def test_a_pole_just_inside_each_edge_is_inside(self, roll: float, inside: bool) -> None:
        # The half height is 1411 px = 1.497 degrees, and the half width is 2072 px = 2.199 degrees.
        half = {0.0: HEIGHT / 2, 180.0: HEIGHT / 2, 90.0: WIDTH / 2, -90.0: WIDTH / 2}[roll]
        distance_deg = math.degrees(math.atan((half - 1.0) * SCALE_RAD))
        view = build_sky_view(attitude(distance_deg, roll), WIDTH, HEIGHT, None)
        assert view.pole.inside_frame is inside
        beyond = math.degrees(math.atan((half + 1.0) * SCALE_RAD))
        assert not build_sky_view(attitude(beyond, roll), WIDTH, HEIGHT, None).pole.inside_frame

    def test_a_pole_behind_the_camera_has_no_pixel_but_a_distance(self) -> None:
        view = build_sky_view(attitude(120.0, 30.0), WIDTH, HEIGHT, COLATITUDE)
        pole = view.pole
        assert not pole.in_front
        assert (pole.x_px, pole.y_px, pole.dx_px, pole.dy_px, pole.distance_px) == (
            None,
            None,
            None,
            None,
            None,
        )
        assert not pole.inside_frame
        assert pole.distance_arcmin == pytest.approx(120.0 * 60.0, abs=0.01)
        assert pole.roll_deg is not None

    def test_a_mirrored_image_flips_the_vertical_offset(self) -> None:
        normal = build_sky_view(attitude(0.9, 40.0), WIDTH, HEIGHT, None).pole
        mirrored = build_sky_view(attitude(0.9, 40.0, parity=-1), WIDTH, HEIGHT, None).pole
        assert mirrored.dx_px == pytest.approx(normal.dx_px, abs=1e-3)
        assert mirrored.dy_px == pytest.approx(-normal.dy_px, abs=1e-3)  # type: ignore[operator]
        assert mirrored.distance_px == pytest.approx(normal.distance_px, abs=1e-3)
        camera = attitude(0.9, 40.0, parity=-1)
        assert mirrored.x_px == pytest.approx(camera.pole_pixel()[0], abs=1e-3)  # type: ignore[index]
        assert mirrored.y_px == pytest.approx(camera.pole_pixel()[1], abs=1e-3)  # type: ignore[index]

    def test_the_offset_is_measured_from_the_frame_that_is_given_not_from_the_principal_point(
        self,
    ) -> None:
        camera = attitude(0.5, 30.0)  # the principal point is the center of a 4144 x 2822 frame
        pole = camera.pole_pixel()
        assert pole is not None
        view = build_sky_view(camera, 3000, 2000, None).pole  # a smaller frame
        center_x, center_y = frame_center(3000, 2000)
        assert view.dx_px == pytest.approx(pole[0] - center_x, abs=2e-3)
        assert view.dy_px == pytest.approx(pole[1] - center_y, abs=2e-3)
        assert view.inside_frame == (-0.5 <= pole[0] <= 2999.5 and -0.5 <= pole[1] <= 1999.5)
        # The camera keeps the principal point of the attitude, whatever the frame size.
        assert build_sky_view(camera, 3000, 2000, None).camera.center_x_px == pytest.approx(2071.5)

    def test_the_angular_distance_is_exact_for_a_far_pole(self) -> None:
        # At 4 degrees the pixel distance (a tangent) and the angle differ by 0.1 percent.
        view = build_sky_view(attitude(4.0, 10.0), WIDTH, HEIGHT, None).pole
        assert view.distance_arcmin == pytest.approx(240.0, abs=1e-3)

    def test_the_position_angle_has_the_documented_directions(self) -> None:
        assert position_angle_deg(0.0, -10.0) == pytest.approx(0.0)  # up
        assert position_angle_deg(-10.0, 0.0) == pytest.approx(90.0)  # left
        assert position_angle_deg(0.0, 10.0) == pytest.approx(180.0)  # down
        assert position_angle_deg(10.0, 0.0) == pytest.approx(-90.0)  # right
        assert position_angle_deg(0.5, 0.5) is None
        assert ROLL_MIN_DISTANCE_PX == skyview.ROLL_MIN_DISTANCE_PX


class TestCamera:
    def test_the_camera_projects_like_the_attitude_to_a_tenth_of_a_milli_pixel(self) -> None:
        camera = attitude(1.1, 25.0, azimuth_deg=70.0)
        geometry = build_sky_view(camera, WIDTH, HEIGHT, None).camera
        rebuilt = CameraAttitude(
            rotation=np.array(geometry.rotation).reshape(3, 3),
            scale_rad_px=geometry.scale_arcsec_px / ARCSEC_PER_RAD,
            parity=geometry.parity,
            center_px=(geometry.center_x_px, geometry.center_y_px),
        )
        rng = np.random.default_rng(3)
        angle = rng.uniform(0.0, 2.0 * math.pi, 200)
        rho = np.radians(rng.uniform(0.0, 2.5, 200))
        vectors = np.stack(
            [np.sin(rho) * np.cos(angle), np.sin(rho) * np.sin(angle), np.cos(rho)], axis=-1
        )
        x0, y0, _ = camera.project(vectors)
        x1, y1, _ = rebuilt.project(vectors)
        assert np.max(np.abs(x1 - x0)) < 1e-3
        assert np.max(np.abs(y1 - y0)) < 1e-3

    def test_the_camera_carries_nine_numbers_row_by_row(self) -> None:
        camera = attitude(0.4, 10.0)
        geometry = build_sky_view(camera, WIDTH, HEIGHT, None).camera
        assert len(geometry.rotation) == 9
        assert geometry.rotation == pytest.approx(list(camera.rotation.reshape(-1)), abs=1e-9)
        assert geometry.parity == 1
        assert geometry.scale_arcsec_px == pytest.approx(SCALE_ARCSEC)
        assert (geometry.center_x_px, geometry.center_y_px) == (2071.5, 1410.5)

    def test_the_view_stays_small(self) -> None:
        view = build_sky_view(attitude(0.9, -65.0), WIDTH, HEIGHT, COLATITUDE)
        assert len(json.dumps(dataclasses.asdict(view))) < 1300


class TestOrbit:
    def test_without_a_colatitude_there_is_no_orbit(self) -> None:
        assert build_sky_view(attitude(0.5), WIDTH, HEIGHT, None).orbit is None
        view = build_sky_view(attitude(0.5), WIDTH, HEIGHT, None)
        assert view.polaris_colatitude_deg is None

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 90.0, 120.0])
    def test_a_colatitude_that_makes_no_sense_counts_as_unknown(self, value: float) -> None:
        view = build_sky_view(attitude(0.5), WIDTH, HEIGHT, value)
        assert view.orbit is None
        assert view.polaris_colatitude_deg is None

    def test_a_centered_orbit_fits_with_the_margin_of_the_short_side(self) -> None:
        view = build_sky_view(attitude(0.0), WIDTH, HEIGHT, COLATITUDE)
        assert view.polaris_colatitude_deg == COLATITUDE
        orbit = view.orbit
        assert orbit is not None
        radius = math.tan(math.radians(COLATITUDE)) / SCALE_RAD
        assert orbit.fits
        assert orbit.margin_px == pytest.approx(HEIGHT / 2 - radius, abs=0.1)
        assert orbit.margin_arcmin == pytest.approx(orbit.margin_px * SCALE_ARCSEC / 60.0, abs=0.01)
        assert orbit.margin_arcmin == pytest.approx((1.4967 - COLATITUDE) * 60.0, abs=0.3)

    def test_an_orbit_that_crosses_an_edge_has_a_negative_margin(self) -> None:
        # The pole is 1.9 degrees left of the center, and the left edge is 2.199 degrees from the
        # center. The circle reaches 0.62 degrees beyond the pole, so it leaves by 0.32 degrees.
        view = build_sky_view(attitude(1.9, 90.0), WIDTH, HEIGHT, COLATITUDE)
        orbit = view.orbit
        assert orbit is not None
        assert not orbit.fits
        assert view.pole.dx_px is not None
        assert view.pole.dx_px < 0.0
        assert orbit.margin_arcmin == pytest.approx((2.199 - 1.9 - COLATITUDE) * 60.0, abs=1.0)
        assert orbit.margin_px == pytest.approx(
            (2.199 - 1.9 - COLATITUDE) * PX_PER_DEG, abs=PX_PER_DEG / 60.0
        )

    def test_an_orbit_that_leaves_through_the_top_has_a_negative_margin(self) -> None:
        # The pole sits 1.347 degrees above the center. The circle reaches 1.967 degrees, which is
        # 0.47 degrees beyond the top edge at 1.497 degrees.
        view = build_sky_view(attitude(1.347, 0.0), WIDTH, HEIGHT, COLATITUDE)
        orbit = view.orbit
        assert orbit is not None
        assert not orbit.fits
        assert orbit.margin_arcmin == pytest.approx((1.4967 - 1.347 - COLATITUDE) * 60.0, abs=0.5)

    @pytest.mark.parametrize("seed", range(12))
    def test_the_margin_matches_a_brute_force_search(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        distance = float(rng.uniform(0.0, 1.6))
        roll = float(rng.uniform(-180.0, 180.0))
        azimuth = float(rng.uniform(0.0, 360.0))
        parity = int(rng.choice([1, -1]))
        colatitude = float(rng.uniform(0.3, 0.9))
        camera = attitude(distance, roll, azimuth_deg=azimuth, parity=parity)
        orbit = build_sky_view(camera, WIDTH, HEIGHT, colatitude).orbit
        assert orbit is not None
        expected = brute_margin(camera, colatitude, WIDTH, HEIGHT)
        assert orbit.margin_px == pytest.approx(expected, abs=0.25)  # the sampling error is 0.02
        assert orbit.fits == (expected >= 0.0) or abs(expected) < 0.25

    def test_a_circle_that_reaches_behind_the_camera_does_not_fit(self) -> None:
        view = build_sky_view(attitude(120.0, 30.0), WIDTH, HEIGHT, COLATITUDE)
        orbit = view.orbit
        assert orbit is not None
        assert not orbit.fits
        assert orbit.margin_px == -WIDTH
        assert orbit.margin_px < 0.0
        assert orbit.margin_arcmin < 0.0

    def test_the_orbit_of_a_cropped_frame_uses_that_frame(self) -> None:
        camera = attitude(0.0)
        full = build_sky_view(camera, WIDTH, HEIGHT, COLATITUDE).orbit
        cropped = build_sky_view(camera, 2000, 1000, COLATITUDE).orbit
        assert full is not None
        assert cropped is not None
        assert cropped.margin_px < full.margin_px
        radius = math.tan(math.radians(COLATITUDE)) / SCALE_RAD
        # The 2000 x 1000 frame is centered on its own center, 72 px and 205 px from the principal
        # point, so its margin is the distance from the circle to its nearest edge.
        center_x, center_y = frame_center(2000, 1000)
        pole_x, pole_y = camera.pole_pixel()  # type: ignore[misc]
        expected = min(
            pole_x - radius + 0.5,
            1999.5 - pole_x - radius,
            pole_y - radius + 0.5,
            999.5 - pole_y - radius,
        )
        assert cropped.margin_px == pytest.approx(expected, abs=0.3)
        assert (center_x, center_y) == (999.5, 499.5)


def test_the_earth_turns_the_view_about_the_pole_and_leaves_the_pole_where_it_is() -> None:
    """The sky rotates about the pole: the pole pixel stays put and the right ascension moves."""
    camera = attitude(0.9, 40.0)
    turned = CameraAttitude(
        rotation=camera.rotation @ rot_z(math.radians(37.0)),
        scale_rad_px=camera.scale_rad_px,
        parity=camera.parity,
        center_px=camera.center_px,
    )
    first = build_sky_view(camera, WIDTH, HEIGHT, COLATITUDE)
    second = build_sky_view(turned, WIDTH, HEIGHT, COLATITUDE)
    assert second.pole == first.pole
    assert second.orbit == first.orbit
    assert second.camera.rotation != first.camera.rotation


class TestPolarisColatitude:
    """`PointingSolution.polaris_colatitude_deg` feeds the orbit, so it must match the pixels."""

    def solution(self) -> PointingSolution:
        epoch = apparent.epoch_from_utc_ns(NIGHT_UTC_NS)
        return PointingSolution.from_attitude(
            attitude(0.9, 20.0, azimuth_deg=30.0),
            epoch,
            mode="bin2",
            width_px=WIDTH,
            height_px=HEIGHT,
        )

    def test_the_colatitude_of_polaris_in_2026_is_about_0_63_degrees(self) -> None:
        colatitude = self.solution().polaris_colatitude_deg(NIGHT_UTC_NS)
        assert colatitude == pytest.approx(0.6265, abs=0.002)

    def test_the_colatitude_agrees_with_the_pixels_of_polaris_and_the_pole(self) -> None:
        solution = self.solution()
        polaris = solution.polaris_pixel(NIGHT_UTC_NS)
        pole = solution.attitude_at(NIGHT_UTC_NS).pole_pixel()
        assert polaris is not None
        assert pole is not None
        distance_px = math.hypot(polaris[0] - pole[0], polaris[1] - pole[1])
        colatitude = solution.polaris_colatitude_deg(NIGHT_UTC_NS)
        assert distance_px == pytest.approx(colatitude * PX_PER_DEG, rel=2e-3)

    def test_the_colatitude_moves_slowly_over_a_day(self) -> None:
        # Aberration, nutation, and precession change it by arcseconds per day, not by minutes.
        solution = self.solution()
        later = solution.polaris_colatitude_deg(NIGHT_UTC_NS + 12 * 3600 * 1_000_000_000)
        assert later == pytest.approx(solution.polaris_colatitude_deg(NIGHT_UTC_NS), abs=0.002)

    def test_the_function_needs_no_solution_and_agrees_with_the_method(self) -> None:
        solution = self.solution()
        for hours in (0, 5, 17):
            t = NIGHT_UTC_NS + hours * 3600 * 1_000_000_000
            assert polaris_colatitude_deg(t) == pytest.approx(
                solution.polaris_colatitude_deg(t), abs=1e-12
            )

    def test_the_apparent_colatitude_shifts_with_the_season_by_the_aberration(self) -> None:
        # Annual aberration moves the apparent place of Polaris by up to 20 arcseconds, which is
        # 0.0057 degrees, and nutation adds a few arcseconds.
        day = 24 * 3600 * 1_000_000_000
        values = [polaris_colatitude_deg(NIGHT_UTC_NS + k * 30 * day) for k in range(12)]
        assert max(values) - min(values) > 0.002
        assert max(values) - min(values) < 0.02
