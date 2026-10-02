"""The reticle, the aim ring, and the adjustment in altitude and azimuth.

The owner moves the mount until the pole sits at the aim, and the mount moves in altitude and in
azimuth. These tests build cameras from altitude and azimuth (`altaz_attitude`), raise, lower, and
turn them by known steps, and check that the view says to move them back, with the right sign.
They also check the two layers of the picture: the reticle holds still (it knows only the frame
size, the plate scale, and Polaris), and the aim ring lies on it, and they compare the aim ring
with a full reprojection.

The site below is a synthetic round-number position. It is not a real station.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from seeingmon.survey import skyview
from seeingmon.survey.geometry import ARCSEC_PER_RAD, nearest_rotation
from seeingmon.survey.skyview import (
    MIN_ZENITH_SEPARATION,
    FloatArray,
    SkyGeometry,
    build_sky_view,
    frame_center,
    reticle_geometry,
    zenith_vector,
)
from seeingmon.survey.wcs_fit import CameraAttitude, pixel_center
from tests.survey.synth import make_attitude

WIDTH, HEIGHT = 4144, 2822
SCALE_ARCSEC = 3.82
SCALE_RAD = SCALE_ARCSEC / ARCSEC_PER_RAD
COLATITUDE = 0.6265  # degrees: about the colatitude of Polaris
PX_PER_DEG = 3600.0 / SCALE_ARCSEC
RADIUS_PX = math.tan(math.radians(COLATITUDE)) / SCALE_RAD  # the orbit on the optical axis
LATITUDE = 50.0  # a synthetic site
LONGITUDE = 10.0
ERA = 1.2345  # radians: any value, because the cameras below are built from the same zenith
Z = np.array([0.0, 0.0, 1.0])


def normalize(vector: FloatArray) -> FloatArray:
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float64)


def local_directions() -> tuple[FloatArray, FloatArray, FloatArray]:
    """North, east, and up at the site, as CIRS unit vectors."""
    up = zenith_vector(LATITUDE, LONGITUDE, ERA)
    north = normalize(Z - up * float(Z @ up))
    return north, np.cross(north, up), up


def altaz_attitude(
    altitude_deg: float, azimuth_deg: float, *, roll_deg: float = 0.0, parity: int = 1
) -> CameraAttitude:
    """A camera that looks at an altitude and an azimuth, with the zenith up in the image.

    The azimuth runs from north through east. A camera that looks north has the east to the right
    of the image. `roll_deg` turns the camera about its optical axis.
    """
    north, east, up = local_directions()
    alt, az = math.radians(altitude_deg), math.radians(azimuth_deg)
    boresight = math.cos(alt) * (math.cos(az) * north + math.sin(az) * east) + math.sin(alt) * up
    toward_up = normalize(up - boresight * float(up @ boresight))
    down = -toward_up
    right = np.cross(down, boresight)
    turn = math.radians(roll_deg)
    x_axis = math.cos(turn) * right + math.sin(turn) * down
    y_axis = -math.sin(turn) * right + math.cos(turn) * down
    return CameraAttitude(
        rotation=nearest_rotation(np.stack([x_axis, y_axis, boresight])),
        scale_rad_px=SCALE_RAD,
        parity=parity,
        center_px=pixel_center(WIDTH, HEIGHT),
    )


def view(camera: CameraAttitude, **options: object) -> SkyGeometry:
    options.setdefault("zenith", zenith_vector(LATITUDE, LONGITUDE, ERA))
    return build_sky_view(camera, WIDTH, HEIGHT, COLATITUDE, **options)  # type: ignore[arg-type]


def east_azimuth_deg(arc_arcmin: float, altitude_deg: float) -> float:
    """The change of azimuth that moves a camera by an arc of the sky along the horizon."""
    return arc_arcmin / 60.0 / math.cos(math.radians(altitude_deg))


def rodrigues(axis: FloatArray, angle: float) -> FloatArray:
    unit = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]])
    return np.asarray(np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * cross @ cross)


def polaris_vector(hour_angle_deg: float) -> FloatArray:
    rho, angle = math.radians(COLATITUDE), math.radians(hour_angle_deg)
    return np.array(
        [math.sin(rho) * math.cos(angle), math.sin(rho) * math.sin(angle), math.cos(rho)]
    )


def pixel_of(camera: CameraAttitude, vector: FloatArray) -> tuple[float, float]:
    x, y, front = camera.project(vector)
    assert front[0]
    return float(x[0]), float(y[0])


class TestZenith:
    def test_the_zenith_has_the_declination_of_the_latitude_and_the_right_ascension_of_the_site(
        self,
    ) -> None:
        zenith = zenith_vector(LATITUDE, LONGITUDE, ERA)
        assert float(np.linalg.norm(zenith)) == pytest.approx(1.0)
        assert zenith[2] == pytest.approx(math.sin(math.radians(LATITUDE)))
        right_ascension = math.atan2(zenith[1], zenith[0]) % (2.0 * math.pi)
        assert right_ascension == pytest.approx((ERA + math.radians(LONGITUDE)) % (2.0 * math.pi))

    def test_the_pole_stands_at_the_altitude_of_the_latitude(self) -> None:
        _, _, up = local_directions()
        assert math.degrees(math.asin(float(Z @ up))) == pytest.approx(LATITUDE)

    def test_the_directions_of_the_site_are_orthonormal_and_the_east_is_east(self) -> None:
        north, east, up = local_directions()
        for a, b in ((north, east), (north, up), (east, up)):
            assert float(a @ b) == pytest.approx(0.0, abs=1e-12)
        # The Earth turns toward the east, so a star east of the meridian has the larger
        # right ascension: the east direction points toward the increasing right ascension.
        assert float(east @ np.array([-up[1], up[0], 0.0])) > 0.0

    def test_the_constant_of_the_module_is_the_one_of_the_geometry_module(self) -> None:
        assert pytest.approx(ARCSEC_PER_RAD) == skyview.ARCSEC_PER_RAD


class TestNoMoveNeeded:
    def test_a_camera_that_points_at_the_pole_needs_no_move(self) -> None:
        sky = view(altaz_attitude(LATITUDE, 0.0))
        assert sky.altitude_arcmin == pytest.approx(0.0, abs=0.01)
        assert sky.azimuth_arcmin == pytest.approx(0.0, abs=0.01)
        assert sky.aim.distance_arcmin == pytest.approx(0.0, abs=0.01)
        assert sky.pole.distance_arcmin == pytest.approx(0.0, abs=0.01)


class TestSigns:
    """The numbers are the move of the camera's pointing that brings the pole to the aim."""

    def test_a_camera_that_is_too_high_must_be_lowered(self) -> None:
        sky = view(altaz_attitude(LATITUDE + 0.5, 0.0))  # raised by 30 arcminutes
        assert sky.altitude_arcmin == pytest.approx(-30.0, abs=0.05)
        assert sky.azimuth_arcmin == pytest.approx(0.0, abs=0.05)
        assert sky.aim.distance_arcmin == pytest.approx(30.0, abs=0.05)

    def test_a_camera_that_is_too_low_must_be_raised(self) -> None:
        sky = view(altaz_attitude(LATITUDE - 20.0 / 60.0, 0.0))
        assert sky.altitude_arcmin == pytest.approx(20.0, abs=0.05)
        assert sky.azimuth_arcmin == pytest.approx(0.0, abs=0.05)

    def test_a_camera_that_is_east_of_the_pole_must_turn_west(self) -> None:
        azimuth = east_azimuth_deg(10.0, LATITUDE)
        sky = view(altaz_attitude(LATITUDE, azimuth))  # 10 arcminutes of arc east of the pole
        assert sky.azimuth_arcmin == pytest.approx(-10.0, abs=0.05)
        assert sky.altitude_arcmin == pytest.approx(0.0, abs=0.05)

    def test_a_camera_that_is_west_of_the_pole_must_turn_east(self) -> None:
        sky = view(altaz_attitude(LATITUDE, -east_azimuth_deg(12.0, LATITUDE)))
        assert sky.azimuth_arcmin == pytest.approx(12.0, abs=0.05)
        assert sky.altitude_arcmin == pytest.approx(0.0, abs=0.05)

    def test_both_parts_come_out_together(self) -> None:
        altitude = LATITUDE + 15.0 / 60.0
        sky = view(altaz_attitude(altitude, east_azimuth_deg(8.0, altitude)))
        assert sky.altitude_arcmin == pytest.approx(-15.0, abs=0.1)
        assert sky.azimuth_arcmin == pytest.approx(-8.0, abs=0.1)

    @pytest.mark.parametrize(
        ("raised", "east"),
        [(30.0, 0.0), (-20.0, 0.0), (0.0, 10.0), (0.0, -14.0), (25.0, 18.0), (-12.0, -9.0)],
    )
    def test_making_the_move_brings_the_pole_to_the_aim(self, raised: float, east: float) -> None:
        """Raise or lower the camera and turn it as the view says, and the view is satisfied."""
        altitude = LATITUDE + raised / 60.0
        azimuth = east_azimuth_deg(east, altitude)
        first = view(altaz_attitude(altitude, azimuth))
        assert first.altitude_arcmin is not None
        assert first.azimuth_arcmin is not None
        moved_altitude = altitude + first.altitude_arcmin / 60.0
        moved_azimuth = azimuth + east_azimuth_deg(first.azimuth_arcmin, moved_altitude)
        second = view(altaz_attitude(moved_altitude, moved_azimuth))
        assert second.aim.distance_arcmin < 0.1
        assert second.altitude_arcmin == pytest.approx(0.0, abs=0.1)
        assert second.azimuth_arcmin == pytest.approx(0.0, abs=0.1)

    def test_the_numbers_do_not_depend_on_the_roll_of_the_camera(self) -> None:
        altitude = LATITUDE + 0.3
        base = view(altaz_attitude(altitude, east_azimuth_deg(6.0, altitude)))
        for roll in (-170.0, -40.0, 15.0, 90.0, 133.0):
            rolled = view(altaz_attitude(altitude, east_azimuth_deg(6.0, altitude), roll_deg=roll))
            assert rolled.altitude_arcmin == pytest.approx(base.altitude_arcmin, abs=0.01)
            assert rolled.azimuth_arcmin == pytest.approx(base.azimuth_arcmin, abs=0.01)
            assert rolled.aim.distance_arcmin == pytest.approx(base.aim.distance_arcmin, abs=0.01)

    def test_a_mirrored_image_gives_the_same_move_and_mirrored_arrows(self) -> None:
        altitude = LATITUDE + 0.2
        azimuth = east_azimuth_deg(9.0, altitude)
        normal = view(altaz_attitude(altitude, azimuth))
        mirrored = view(altaz_attitude(altitude, azimuth, parity=-1))
        assert mirrored.altitude_arcmin == pytest.approx(normal.altitude_arcmin, abs=0.01)
        assert mirrored.azimuth_arcmin == pytest.approx(normal.azimuth_arcmin, abs=0.01)
        assert normal.axes is not None
        assert mirrored.axes is not None
        assert mirrored.axes.altitude_dx == pytest.approx(normal.axes.altitude_dx, abs=1e-3)
        assert mirrored.axes.altitude_dy == pytest.approx(-normal.axes.altitude_dy, abs=1e-3)
        assert mirrored.axes.azimuth_dx == pytest.approx(normal.axes.azimuth_dx, abs=1e-3)
        assert mirrored.axes.azimuth_dy == pytest.approx(-normal.axes.azimuth_dy, abs=1e-3)


class TestAxes:
    def test_with_the_zenith_up_the_altitude_arrow_points_up_and_the_azimuth_arrow_points_right(
        self,
    ) -> None:
        sky = view(altaz_attitude(LATITUDE, 0.0))
        assert sky.axes is not None
        assert sky.axes.altitude_dx == pytest.approx(0.0, abs=0.02)
        assert sky.axes.altitude_dy == pytest.approx(-1.0, abs=0.001)  # y grows downward
        assert sky.axes.azimuth_dx == pytest.approx(1.0, abs=0.001)
        assert sky.axes.azimuth_dy == pytest.approx(0.0, abs=0.02)

    def test_a_rolled_camera_turns_both_arrows(self) -> None:
        sky = view(altaz_attitude(LATITUDE, 0.0, roll_deg=90.0))
        assert sky.axes is not None
        # Turning the camera by 90 degrees about its axis turns the picture the other way.
        assert sky.axes.altitude_dx == pytest.approx(-1.0, abs=0.001)
        assert sky.axes.altitude_dy == pytest.approx(0.0, abs=0.02)

    @pytest.mark.parametrize("roll", [0.0, 37.0, -120.0, 200.0])
    def test_the_arrows_are_unit_vectors_at_right_angles(self, roll: float) -> None:
        sky = view(altaz_attitude(LATITUDE + 0.1, 0.05, roll_deg=roll))
        assert sky.axes is not None
        altitude = np.array([sky.axes.altitude_dx, sky.axes.altitude_dy])
        azimuth = np.array([sky.axes.azimuth_dx, sky.axes.azimuth_dy])
        assert float(np.linalg.norm(altitude)) == pytest.approx(1.0, abs=1e-3)
        assert float(np.linalg.norm(azimuth)) == pytest.approx(1.0, abs=1e-3)
        assert float(altitude @ azimuth) == pytest.approx(0.0, abs=0.01)

    def test_the_pole_lies_on_the_side_that_the_arrows_name(self) -> None:
        """A pole above the aim in the altitude direction shows as a positive altitude."""
        sky = view(altaz_attitude(LATITUDE - 0.4, east_azimuth_deg(-6.0, LATITUDE)))
        assert sky.axes is not None
        assert sky.altitude_arcmin is not None
        assert sky.azimuth_arcmin is not None
        assert sky.aim.dx_px is not None
        assert sky.aim.dy_px is not None
        offset = np.array([sky.aim.dx_px, sky.aim.dy_px])
        altitude = np.array([sky.axes.altitude_dx, sky.axes.altitude_dy])
        azimuth = np.array([sky.axes.azimuth_dx, sky.axes.azimuth_dy])
        assert float(offset @ altitude) * sky.altitude_arcmin > 0.0
        assert float(offset @ azimuth) * sky.azimuth_arcmin > 0.0
        # The parts of the offset along the arrows are the parts of the move, in arcminutes.
        assert float(offset @ altitude) * SCALE_ARCSEC / 60.0 == pytest.approx(
            sky.altitude_arcmin, abs=0.15
        )
        assert float(offset @ azimuth) * SCALE_ARCSEC / 60.0 == pytest.approx(
            sky.azimuth_arcmin, abs=0.15
        )


class TestWithoutTheSite:
    def test_no_zenith_means_no_move_and_no_arrows_but_the_aim_stays(self) -> None:
        camera = altaz_attitude(LATITUDE + 0.3, 0.0)
        sky = build_sky_view(camera, WIDTH, HEIGHT, COLATITUDE)
        assert sky.altitude_arcmin is None
        assert sky.azimuth_arcmin is None
        assert sky.axes is None
        assert sky.aim.distance_arcmin == pytest.approx(18.0, abs=0.05)
        assert sky.pole.in_front

    def test_a_camera_that_looks_almost_at_the_zenith_has_no_defined_directions(self) -> None:
        sky = view(altaz_attitude(89.9, 0.0))
        assert sky.altitude_arcmin is None
        assert sky.azimuth_arcmin is None
        assert sky.axes is None

    def test_the_limit_is_about_a_degree_from_the_zenith(self) -> None:
        limit_deg = math.degrees(math.asin(MIN_ZENITH_SEPARATION))
        inside = view(altaz_attitude(90.0 - limit_deg + 0.2, 0.0))
        outside = view(altaz_attitude(90.0 - limit_deg - 0.2, 0.0))
        assert inside.altitude_arcmin is None
        assert outside.altitude_arcmin is not None

    def test_a_pole_behind_the_camera_has_no_move(self) -> None:
        camera = CameraAttitude(
            rotation=make_attitude(120.0, 30.0, 0.0),
            scale_rad_px=SCALE_RAD,
            parity=1,
            center_px=pixel_center(WIDTH, HEIGHT),
        )
        sky = view(camera, polaris_xy=(1000.0, 1000.0))
        assert sky.altitude_arcmin is None
        assert sky.axes is None
        assert sky.aim_ring is None
        assert sky.aim.dx_px is None
        assert sky.aim.distance_arcmin == pytest.approx(120.0 * 60.0, abs=0.01)


class TestAim:
    def test_the_aim_is_the_frame_center_by_default(self) -> None:
        sky = view(altaz_attitude(LATITUDE + 0.2, 0.1))
        center_x, center_y = frame_center(WIDTH, HEIGHT)
        assert sky.aim.x_px == pytest.approx(center_x)
        assert sky.aim.y_px == pytest.approx(center_y)
        assert sky.aim.dx_px == pytest.approx(sky.pole.dx_px, abs=1e-3)
        assert sky.aim.dy_px == pytest.approx(sky.pole.dy_px, abs=1e-3)
        assert sky.aim.distance_arcmin == pytest.approx(sky.pole.distance_arcmin, abs=1e-3)

    def test_the_aim_can_be_another_pixel_and_the_offsets_follow_it(self) -> None:
        camera = altaz_attitude(LATITUDE + 0.2, 0.1)
        pole = view(camera).pole
        assert pole.x_px is not None
        assert pole.y_px is not None
        sky = view(camera, aim_xy=(2500.0, 1700.0))
        assert (sky.aim.x_px, sky.aim.y_px) == (2500.0, 1700.0)
        assert sky.aim.dx_px == pytest.approx(pole.x_px - 2500.0, abs=2e-3)
        assert sky.aim.dy_px == pytest.approx(pole.y_px - 1700.0, abs=2e-3)

    def test_the_move_is_measured_from_the_aim_and_not_from_the_center(self) -> None:
        """Put the aim where the pole is, and nothing is left to move."""
        camera = altaz_attitude(LATITUDE + 0.4, east_azimuth_deg(5.0, LATITUDE))
        pole = view(camera).pole
        assert pole.x_px is not None
        assert pole.y_px is not None
        sky = view(camera, aim_xy=(pole.x_px, pole.y_px))
        assert sky.aim.distance_arcmin == pytest.approx(0.0, abs=0.01)
        assert sky.altitude_arcmin == pytest.approx(0.0, abs=0.01)
        assert sky.azimuth_arcmin == pytest.approx(0.0, abs=0.01)


class TestReticle:
    """Layer A: a circle that depends on the frame size, the plate scale, and Polaris only."""

    def test_the_reticle_is_a_circle_around_the_frame_center_with_the_radius_of_the_orbit(
        self,
    ) -> None:
        reticle = reticle_geometry(WIDTH, HEIGHT, SCALE_ARCSEC, COLATITUDE)
        assert reticle is not None
        center_x, center_y = frame_center(WIDTH, HEIGHT)
        assert (reticle.x_px, reticle.y_px) == (center_x, center_y)
        assert reticle.radius_px == pytest.approx(RADIUS_PX, abs=1e-3)
        assert reticle.radius_px == pytest.approx(COLATITUDE * PX_PER_DEG, rel=1e-3)

    def test_the_reticle_follows_the_aim_and_the_frame_size(self) -> None:
        moved = reticle_geometry(WIDTH, HEIGHT, SCALE_ARCSEC, COLATITUDE, aim_xy=(1500.0, 900.0))
        assert moved is not None
        assert (moved.x_px, moved.y_px) == (1500.0, 900.0)
        assert moved.radius_px == pytest.approx(RADIUS_PX, abs=1e-3)
        small = reticle_geometry(2072, 1411, SCALE_ARCSEC, COLATITUDE)
        assert small is not None
        assert (small.x_px, small.y_px) == (1035.5, 705.0)
        assert small.radius_px == pytest.approx(RADIUS_PX, abs=1e-3)  # the scale is the same

    def test_a_coarser_scale_gives_a_smaller_circle(self) -> None:
        coarse = reticle_geometry(WIDTH, HEIGHT, 2 * SCALE_ARCSEC, COLATITUDE)
        assert coarse is not None
        assert coarse.radius_px == pytest.approx(RADIUS_PX / 2, abs=1e-2)

    @pytest.mark.parametrize(
        ("scale", "colatitude"),
        [(0.0, COLATITUDE), (-1.0, COLATITUDE), (float("nan"), COLATITUDE), (3.8, None)],
    )
    def test_without_a_scale_or_a_colatitude_there_is_no_reticle(
        self, scale: float, colatitude: float | None
    ) -> None:
        assert reticle_geometry(WIDTH, HEIGHT, scale, colatitude) is None

    def test_the_reticle_does_not_know_the_attitude_and_holds_still_for_five_poles(self) -> None:
        """Five attitudes of the same frame give the same circle, while the pole moves about."""
        circles = set()
        poles = set()
        for altitude, azimuth in ((0.0, 0.0), (0.4, 0.1), (-0.3, -0.2), (0.7, 0.3), (-0.1, 0.5)):
            camera = altaz_attitude(LATITUDE + altitude, azimuth)
            sky = view(camera)
            assert sky.pole.x_px is not None
            poles.add((sky.pole.x_px, sky.pole.y_px))
            reticle = reticle_geometry(WIDTH, HEIGHT, sky.camera.scale_arcsec_px, COLATITUDE)
            circles.add(dataclasses.astuple(reticle) if reticle else None)
        assert len(circles) == 1
        assert len(poles) == 5


class TestAimRing:
    """The aim ring is the real Polaris pixel plus the aim minus the pole."""

    def test_the_ring_is_polaris_plus_the_offset_from_the_pole_to_the_aim(self) -> None:
        camera = altaz_attitude(LATITUDE + 0.3, east_azimuth_deg(4.0, LATITUDE))
        polaris = (2200.0, 2100.0)
        sky = view(camera, polaris_xy=polaris)
        assert sky.aim_ring is not None
        assert sky.pole.x_px is not None
        assert sky.pole.y_px is not None
        center_x, center_y = frame_center(WIDTH, HEIGHT)
        assert sky.aim_ring.x_px == pytest.approx(polaris[0] + center_x - sky.pole.x_px, abs=2e-3)
        assert sky.aim_ring.y_px == pytest.approx(polaris[1] + center_y - sky.pole.y_px, abs=2e-3)

    def test_the_ring_follows_a_moved_aim(self) -> None:
        camera = altaz_attitude(LATITUDE + 0.3, 0.0)
        polaris = (2200.0, 2100.0)
        base = view(camera, polaris_xy=polaris)
        moved = view(camera, polaris_xy=polaris, aim_xy=(1500.0, 900.0))
        assert base.aim_ring is not None
        assert moved.aim_ring is not None
        center_x, center_y = frame_center(WIDTH, HEIGHT)
        assert moved.aim_ring.x_px - base.aim_ring.x_px == pytest.approx(
            1500.0 - center_x, abs=2e-3
        )
        assert moved.aim_ring.y_px - base.aim_ring.y_px == pytest.approx(900.0 - center_y, abs=2e-3)

    def test_the_ring_lies_on_the_circle_of_the_reticle_for_ten_random_attitudes(self) -> None:
        """The distance from the aim to the ring is the radius of the reticle, to half a pixel."""
        rng = np.random.default_rng(2026)
        reticle = reticle_geometry(WIDTH, HEIGHT, SCALE_ARCSEC, COLATITUDE)
        assert reticle is not None
        for _ in range(10):
            camera = CameraAttitude(
                rotation=make_attitude(
                    float(rng.uniform(0.0, 1.2)),
                    float(rng.uniform(0.0, 360.0)),
                    float(rng.uniform(-180.0, 180.0)),
                ),
                scale_rad_px=SCALE_RAD,
                parity=int(rng.choice([1, -1])),
                center_px=pixel_center(WIDTH, HEIGHT),
            )
            polaris = pixel_of(camera, polaris_vector(float(rng.uniform(0.0, 360.0))))
            sky = view(camera, polaris_xy=polaris)
            assert sky.aim_ring is not None
            distance = math.hypot(
                sky.aim_ring.x_px - reticle.x_px, sky.aim_ring.y_px - reticle.y_px
            )
            assert distance == pytest.approx(reticle.radius_px, abs=0.5)

    def test_with_the_pole_at_the_aim_the_ring_is_polaris_and_the_declination_ring_is_the_reticle(
        self,
    ) -> None:
        """A camera that looks at the pole: Polaris is in the ring, and its orbit is the circle."""
        camera = altaz_attitude(LATITUDE, 0.0, roll_deg=33.0)
        reticle = reticle_geometry(WIDTH, HEIGHT, SCALE_ARCSEC, COLATITUDE)
        assert reticle is not None
        for hour_angle in range(0, 360, 20):
            polaris = pixel_of(camera, polaris_vector(float(hour_angle)))
            sky = view(camera, polaris_xy=polaris)
            assert sky.aim_ring is not None
            assert sky.aim_ring.x_px == pytest.approx(polaris[0], abs=0.01)
            assert sky.aim_ring.y_px == pytest.approx(polaris[1], abs=0.01)
            # The circle of the declination of Polaris around the pole is the dashed circle.
            radius = math.hypot(polaris[0] - reticle.x_px, polaris[1] - reticle.y_px)
            assert radius == pytest.approx(reticle.radius_px, abs=1.0)

    def test_the_sky_layer_moves_with_the_attitude_and_the_reticle_does_not(self) -> None:
        first = view(altaz_attitude(LATITUDE + 0.3, 0.1), polaris_xy=(2000.0, 2000.0))
        second = view(altaz_attitude(LATITUDE - 0.2, -0.2), polaris_xy=(2100.0, 1900.0))
        assert first.pole != second.pole
        assert first.camera.rotation != second.camera.rotation
        assert first.aim == dataclasses.replace(
            second.aim,
            dx_px=first.aim.dx_px,
            dy_px=first.aim.dy_px,
            distance_arcmin=first.aim.distance_arcmin,
        )  # the aim pixel is the same
        assert reticle_geometry(WIDTH, HEIGHT, SCALE_ARCSEC, COLATITUDE) == reticle_geometry(
            WIDTH, HEIGHT, SCALE_ARCSEC, COLATITUDE
        )

    def test_without_a_solved_polaris_there_is_no_ring(self) -> None:
        assert view(altaz_attitude(LATITUDE + 0.3, 0.0)).aim_ring is None


class TestTheShortcut:
    """The ring against a full reprojection of the sky after the move.

    After the mount brings the pole to the aim, the camera has turned by a rotation that takes the
    sky direction at the aim pixel to the pole, and Polaris is where the old attitude projects the
    turned Polaris. The page draws the old Polaris pixel plus the aim minus the pole instead.
    """

    @staticmethod
    def minimal_turn(camera: CameraAttitude) -> FloatArray:
        aim_x, aim_y = pixel_center(WIDTH, HEIGHT)
        aim = camera.unproject(aim_x, aim_y)[0]
        axis = np.cross(aim, Z)  # the turn of the pointing that carries the aim onto the pole
        return rodrigues(axis, math.acos(float(np.clip(Z @ aim, -1.0, 1.0))))

    @staticmethod
    def altaz_turn(camera: CameraAttitude) -> FloatArray:
        """The turn of an altitude move and an azimuth move, with the horizon kept level."""
        north, east, up = local_directions()
        aim_x, aim_y = pixel_center(WIDTH, HEIGHT)
        aim = camera.unproject(aim_x, aim_y)[0]

        def altitude_azimuth(vector: FloatArray) -> tuple[float, float]:
            return math.asin(float(np.clip(vector @ up, -1.0, 1.0))), math.atan2(
                float(vector @ east), float(vector @ north)
            )

        alt_aim, az_aim = altitude_azimuth(aim)
        alt_pole, az_pole = altitude_azimuth(Z)
        # The azimuth grows clockwise seen from above, which is a turn about minus the vertical.
        about_vertical = rodrigues(up, -(az_pole - az_aim))
        turned = about_vertical @ aim
        raise_it = rodrigues(np.cross(turned, up), alt_pole - alt_aim)
        return np.asarray(raise_it @ about_vertical)

    @classmethod
    def worst_error_px(cls, distance_deg: float, *, altaz: bool) -> float:
        worst = 0.0
        rng = np.random.default_rng(7)
        for _ in range(60):
            direction = float(rng.uniform(0.0, 2.0 * math.pi))
            if altaz:
                altitude = LATITUDE + distance_deg * math.cos(direction)
                azimuth = distance_deg * math.sin(direction) / math.cos(math.radians(LATITUDE))
                camera = altaz_attitude(altitude, azimuth)
                turn = cls.altaz_turn(camera)
            else:
                camera = CameraAttitude(
                    rotation=make_attitude(
                        distance_deg, float(rng.uniform(0.0, 360.0)), float(rng.uniform(-180, 180))
                    ),
                    scale_rad_px=SCALE_RAD,
                    parity=int(rng.choice([1, -1])),
                    center_px=pixel_center(WIDTH, HEIGHT),
                )
                turn = cls.minimal_turn(camera)
            aim_x, aim_y = pixel_center(WIDTH, HEIGHT)
            pole_x, pole_y = pixel_of(camera, Z)
            polaris = polaris_vector(float(rng.uniform(0.0, 360.0)))
            x, y = pixel_of(camera, polaris)
            ring = np.array([x + aim_x - pole_x, y + aim_y - pole_y])
            full = np.array(pixel_of(camera, turn.T @ polaris))
            worst = max(worst, float(np.hypot(*(full - ring))))
        return worst

    @pytest.mark.parametrize(
        ("distance_deg", "bound_px"),
        [(0.1, 0.03), (0.3, 0.1), (0.6, 0.2), (1.0, 0.4), (1.5, 0.7), (2.0, 1.2)],
    )
    def test_against_the_shortest_turn_the_ring_is_within_a_pixel(
        self, distance_deg: float, bound_px: float
    ) -> None:
        # A pixel is 3.82 arcseconds, and the alignment aims at 2 arcminutes, which is 31 pixels.
        assert self.worst_error_px(distance_deg, altaz=False) < bound_px

    @pytest.mark.parametrize(
        ("distance_deg", "bound_px"), [(0.2, 3.0), (0.5, 7.0), (1.0, 14.0), (1.5, 21.0)]
    )
    def test_against_an_altitude_and_an_azimuth_move_the_ring_is_off_by_the_twist_of_the_azimuth(
        self, distance_deg: float, bound_px: float
    ) -> None:
        """The azimuth move turns the picture about the axis, so Polaris slides along the circle.

        At this site, 12 pixels (47 arcseconds) for a pole 1 degree from the aim: below the
        2 arcminutes of the alignment, and zero when the pole reaches the aim.
        """
        error = self.worst_error_px(distance_deg, altaz=True)
        assert error < bound_px
        assert error > 0.2 * bound_px  # the bound is not loose by a factor of five

    def test_the_two_errors_grow_with_the_distance(self) -> None:
        near, far = self.worst_error_px(0.5, altaz=False), self.worst_error_px(1.0, altaz=False)
        assert 2.0 < far / near < 4.5


def test_the_view_stays_small_with_the_aim_the_ring_and_the_arrows() -> None:
    sky = view(
        altaz_attitude(LATITUDE + 0.3, east_azimuth_deg(8.0, LATITUDE)), polaris_xy=(2100.0, 2000.0)
    )
    assert len(json.dumps(dataclasses.asdict(sky))) < 1300
