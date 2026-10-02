"""The sky view of a frame: the reticle, the pole, the aim ring, and the adjustment to make.

A fixed camera that points near the pole sees Polaris go round the pole once a day, on a circle of
about 0.6 degrees. A first alignment has one job: move the camera until the pole sits at the aim
(the center of the sensor, unless `[alignment]` names another pixel), so that Polaris stays in the
frame all day. The mount of the camera moves in altitude and in azimuth, and it cannot change the
roll about the optical axis, so the view gives the roll for information only. The alignment page
shows two layers, and this module computes the numbers for both.

**Layer A, the reticle.** It is fixed in the picture, and it never moves with the mount. It is a
function of the frame size, the plate scale, and the colatitude of Polaris (`reticle_geometry`): a
circle centered on the aim, with the radius of the orbit of Polaris in image pixels,
`tan(colatitude) / scale_rad_px`. The orbit is a circle of constant colatitude around the pole, and
around the optical axis it projects to an exact circle of that radius.

**Layer B, the sky layer.** It is fixed to the stars and drawn from the solution of the very frame,
so it moves in the picture as the mount moves. `build_sky_view` turns a camera attitude and the
frame size into its numbers:

- **The camera.** The rotation (row by row), the plate scale, the parity, and the principal point,
  rounded to what a pixel needs, so that a browser can project sky points with the same formula as
  `CameraAttitude.project` (`w = R u`, then `x = x_c + w_x / w_z / s` and `y = y_c + parity *
  w_y / w_z / s`). The browser draws the declination rings, the right-ascension lines, and the ring
  of the declination of Polaris (the real orbit) from it.
- **The pole.** The celestial pole of date is the z axis of CIRS, the apparent frame of the
  survey path. The view gives its pixel (when it is in front of the camera), whether it lies in
  the frame, its offset from the frame center, its distance from the frame center, and the
  position angle of its direction.
- **The orbit.** The circle at the colatitude of Polaris around the pole, sampled at
  `ORBIT_SAMPLES` points and projected. The margin is the distance from the circle to the
  nearest frame edge in pixels, and it is negative when the circle leaves the frame.
- **The aim.** The pixel where the pole should go, and the offset and the exact angle from the pole
  to it.
- **The aim ring.** Where Polaris belongs now: the pixel on the reticle circle that the real
  Polaris would take if the pole sat at the aim, with the same roll and at the same time. The
  moves of an altitude-azimuth mount translate the picture, so it is the detected Polaris pixel
  plus the aim minus the detected pole. It lies on the reticle circle to 0.3 pixel for a pole 1
  degree from the axis and to 0.9 pixel for one 2 degrees away. A test compares it with a full
  reprojection of the sky after an altitude move and an azimuth move that keep the horizon level.
  The azimuth move also twists the picture about the optical axis (by about the azimuth turn times
  the sine of the altitude), which slides Polaris along the circle: 2.5 pixels for a pole 0.2
  degree from the aim, 6 pixels for 0.5 degree, and 12 pixels (47 arcseconds) for 1 degree, at a
  site at 50 degrees of latitude. That is less than the 2 arcminutes of the alignment, and it
  falls to zero as the pole reaches the aim: then the ring and the real Polaris coincide, whatever
  the roll.
- **The adjustment.** With the zenith of the site in CIRS (`zenith_vector`), the view splits the
  move of the camera's pointing that brings the pole to the aim into an altitude part and an
  azimuth part, as arcs on the sky: positive altitude means raise the camera (toward the zenith),
  and positive azimuth means turn it toward the east. It also gives the directions in the image of
  the two moves (`axes`), so a page can draw them. Both are `None` without a zenith, and when the
  zenith lies within about a degree of the aim, where the two directions are not defined.

Every coordinate is of date, as in `seeingmon.survey.wcs_fit`. The frame center is the middle of
the frame as the page shows it, `((width - 1) / 2, (height - 1) / 2)` in the pixel convention of
the survey path (the center of the first pixel is at 0). It equals the principal point for a
full frame and differs from it for a cropped one, and the view measures offsets from the frame
center.

The module needs only NumPy, so the `web` process can import it without the survey extra. It
returns plain dataclasses, and the contract of `web` and `core`
(`seeingmon.services.web.contract.SkyView`) wraps them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from seeingmon.survey.wcs_fit import CameraAttitude

FloatArray = npt.NDArray[np.float64]

# Arcseconds in a radian, as in `seeingmon.survey.geometry.ARCSEC_PER_RAD` (a test checks).
ARCSEC_PER_RAD = 180.0 * 3600.0 / math.pi

# The same limit as `seeingmon.survey.wcs_fit.ROLL_MIN_DISTANCE_PX`, which a test checks.
ROLL_MIN_DISTANCE_PX = 1.0
# The points of the orbit. A step of 1 degree puts the sampled margin within 0.02 pixel of the
# true one for a circle of 600 pixels.
ORBIT_SAMPLES = 360
# The sine of the smallest angle between the aim and the zenith for which the directions of the
# altitude and the azimuth are defined (1.1 degrees). Nearer the zenith, they turn too fast.
MIN_ZENITH_SEPARATION = 0.02
# How far from the aim the view looks to find the image direction of a move (0.25 degrees).
_AXIS_PROBE_RAD = math.radians(0.25)
_Z = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True, slots=True)
class CameraGeometry:
    """The camera model in the form that a page projects with."""

    rotation: list[float]  # nine numbers, row by row: the CIRS-to-camera rotation
    scale_arcsec_px: float
    parity: int
    center_x_px: float
    center_y_px: float


@dataclass(frozen=True, slots=True)
class PoleGeometry:
    """Where the celestial pole falls in the frame. Pixels follow the survey convention.

    `x_px`, `y_px`, `dx_px`, `dy_px`, and `distance_px` are `None` when the pole lies behind
    the camera. `dx_px` and `dy_px` are the pole minus the frame center, and `roll_deg` is the
    position angle of the direction to the pole, from image up toward image left (`None`
    within `ROLL_MIN_DISTANCE_PX` of the center). The roll is information only: a mount that
    moves in altitude and azimuth cannot change it.
    """

    in_front: bool
    x_px: float | None
    y_px: float | None
    inside_frame: bool
    dx_px: float | None
    dy_px: float | None
    distance_px: float | None
    distance_arcmin: float
    roll_deg: float | None


@dataclass(frozen=True, slots=True)
class OrbitGeometry:
    """How the circle of Polaris sits in the frame.

    `margin_px` is the distance from the circle to the nearest frame edge, negative when the
    circle leaves the frame. When a point of the circle lies behind the camera, the circle does
    not fit, and the margin is minus the longer side of the frame.
    """

    fits: bool
    margin_px: float
    margin_arcmin: float


@dataclass(frozen=True, slots=True)
class PointGeometry:
    """A pixel of the frame."""

    x_px: float
    y_px: float


@dataclass(frozen=True, slots=True)
class ReticleGeometry:
    """The fixed circle of the reticle: its center, the aim, and its radius in pixels."""

    x_px: float
    y_px: float
    radius_px: float


@dataclass(frozen=True, slots=True)
class AimGeometry:
    """Where the pole should go, and how far it is from there.

    `dx_px` and `dy_px` are the pole minus the aim, and they are `None` when the pole lies
    behind the camera. `distance_arcmin` is the exact angle between the pole and the sky
    direction at the aim pixel.
    """

    x_px: float
    y_px: float
    dx_px: float | None
    dy_px: float | None
    distance_arcmin: float


@dataclass(frozen=True, slots=True)
class AxesGeometry:
    """The image directions of the two moves of the mount, as unit vectors (x right, y down).

    `altitude_*` is where the camera looks when you raise it, and `azimuth_*` is where it looks
    when you turn it toward the east. The pole lies on the side of the aim in which the camera
    has to move, so the two vectors read like the arrows of a compass drawn on the picture.
    """

    altitude_dx: float
    altitude_dy: float
    azimuth_dx: float
    azimuth_dy: float


@dataclass(frozen=True, slots=True)
class SkyGeometry:
    """The sky view of one frame.

    `altitude_arcmin` and `azimuth_arcmin` are the move of the camera's pointing that brings the
    pole to the aim: positive altitude means raise the camera, and positive azimuth means turn it
    toward the east. They are arcs on the sky, so a turn of the azimuth axis is the azimuth arc
    divided by the cosine of the altitude of the camera.
    """

    camera: CameraGeometry
    pole: PoleGeometry
    polaris_colatitude_deg: float | None
    orbit: OrbitGeometry | None
    aim: AimGeometry
    aim_ring: PointGeometry | None = None
    altitude_arcmin: float | None = None
    azimuth_arcmin: float | None = None
    axes: AxesGeometry | None = None


def frame_center(width_px: int, height_px: int) -> tuple[float, float]:
    """The center of a frame, where the center of the first pixel is (0, 0)."""
    return (width_px - 1) / 2.0, (height_px - 1) / 2.0


def position_angle_deg(dx_px: float, dy_px: float) -> float | None:
    """The angle of a pixel offset from image up toward image left, in degrees in (-180, 180].

    The result is `None` for an offset shorter than `ROLL_MIN_DISTANCE_PX`.
    """
    if math.hypot(dx_px, dy_px) < ROLL_MIN_DISTANCE_PX:
        return None
    return math.degrees(math.atan2(0.0 - dx_px, 0.0 - dy_px))  # 0.0 - x keeps -0.0 out


def zenith_vector(latitude_deg: float, longitude_deg: float, era_rad: float) -> FloatArray:
    """The local zenith as a CIRS unit vector.

    The Earth turns the site about the pole by the Earth rotation angle, so the zenith has the
    right ascension `era_rad + longitude` and the declination of the latitude. Latitude is
    north-positive and longitude is east-positive. The plumb line gives the astronomical latitude,
    which the geodetic latitude of the site matches to the deflection of the vertical, a few
    arcseconds.
    """
    latitude = math.radians(latitude_deg)
    angle = era_rad + math.radians(longitude_deg)
    return np.array(
        [
            math.cos(latitude) * math.cos(angle),
            math.cos(latitude) * math.sin(angle),
            math.sin(latitude),
        ]
    )


def reticle_geometry(
    width_px: int,
    height_px: int,
    scale_arcsec_px: float,
    polaris_colatitude_deg: float | None,
    aim_xy: tuple[float, float] | None = None,
) -> ReticleGeometry | None:
    """The fixed reticle of a frame, or `None` when the radius is not known.

    The reticle is a circle centered on the aim (the frame center by default) with the radius of
    the orbit of Polaris in pixels, `tan(colatitude) / scale_rad_px`. It depends on the frame size
    and the plate scale only, plus the colatitude of Polaris, which changes by arcseconds in a
    year, so the circle holds still while the mount moves.
    """
    colatitude = _valid_colatitude(polaris_colatitude_deg)
    if colatitude is None or not scale_arcsec_px > 0.0 or not math.isfinite(scale_arcsec_px):
        return None
    center = frame_center(width_px, height_px) if aim_xy is None else aim_xy
    scale_rad = scale_arcsec_px / ARCSEC_PER_RAD
    return ReticleGeometry(
        x_px=_round(center[0], 3),
        y_px=_round(center[1], 3),
        radius_px=_round(math.tan(math.radians(colatitude)) / scale_rad, 3),
    )


def build_sky_view(
    attitude: CameraAttitude,
    width_px: int,
    height_px: int,
    polaris_colatitude_deg: float | None,
    *,
    polaris_xy: tuple[float, float] | None = None,
    aim_xy: tuple[float, float] | None = None,
    zenith: npt.ArrayLike | None = None,
) -> SkyGeometry:
    """The sky view of a frame of `width_px` by `height_px` pixels.

    `attitude` is the camera model in the apparent frame at the time of the frame, and
    `polaris_colatitude_deg` is the angle between Polaris and the pole at that time
    (`PointingSolution.polaris_colatitude_deg`), or `None` when it is not known.

    `polaris_xy` is where the solver found Polaris. `aim_xy` is the pixel where the pole should
    go, and it defaults to the frame center. `zenith` is the CIRS unit vector of the zenith of the
    site at the time of the frame (`zenith_vector`), or `None` when the site is not known.
    """
    colatitude = _valid_colatitude(polaris_colatitude_deg)
    x, y, front = attitude.project(_Z)
    pole_xy = (float(x[0]), float(y[0])) if front[0] else None
    pole = _pole(attitude, width_px, height_px, pole_xy)
    orbit = None if colatitude is None else _orbit(attitude, width_px, height_px, colatitude)

    aim_point = frame_center(width_px, height_px) if aim_xy is None else aim_xy
    aim_direction = attitude.unproject(aim_point[0], aim_point[1])[0]
    aim = _aim(pole_xy, aim_point, aim_direction)
    ring = None
    if pole_xy is not None and polaris_xy is not None:
        ring = PointGeometry(
            x_px=_round(polaris_xy[0] + aim_point[0] - pole_xy[0], 3),
            y_px=_round(polaris_xy[1] + aim_point[1] - pole_xy[1], 3),
        )
    altitude = azimuth = None
    axes = None
    if pole_xy is not None and zenith is not None:
        altitude, azimuth, axes = _adjustment(attitude, aim_point, aim_direction, zenith)

    camera = CameraGeometry(
        rotation=[round(float(value), 9) for value in attitude.rotation.reshape(-1)],
        scale_arcsec_px=round(attitude.scale_arcsec_px, 6),
        parity=attitude.parity,
        center_x_px=round(float(attitude.center_px[0]), 3),
        center_y_px=round(float(attitude.center_px[1]), 3),
    )
    return SkyGeometry(
        camera=camera,
        pole=pole,
        polaris_colatitude_deg=None if colatitude is None else round(colatitude, 5),
        orbit=orbit,
        aim=aim,
        aim_ring=ring,
        altitude_arcmin=altitude,
        azimuth_arcmin=azimuth,
        axes=axes,
    )


def _valid_colatitude(value: float | None) -> float | None:
    if value is None or not math.isfinite(value) or not 0.0 <= value < 90.0:
        return None
    return float(value)


def _round(value: float, digits: int) -> float:
    return round(value, digits) + 0.0  # + 0.0 turns -0.0 into 0.0


def _angle_arcmin(vector: FloatArray) -> float:
    """The angle between a unit vector and the pole, in arcminutes."""
    return (
        math.degrees(math.atan2(math.hypot(float(vector[0]), float(vector[1])), vector[2])) * 60.0
    )


def _pole(
    attitude: CameraAttitude, width_px: int, height_px: int, pole_xy: tuple[float, float] | None
) -> PoleGeometry:
    center_x, center_y = frame_center(width_px, height_px)
    # The angular distance from the frame center to the pole is exact, and it also holds when
    # the pole lies behind the camera.
    distance_arcmin = _angle_arcmin(attitude.unproject(center_x, center_y)[0])
    if pole_xy is None:
        # Behind the camera there is no pixel. The roll uses the camera-frame direction, as
        # `CameraAttitude.roll_deg` does.
        direction = attitude.rotation @ _Z
        away_x, away_y = float(direction[0]), float(attitude.parity * direction[1])
        roll = (
            None
            if math.hypot(away_x, away_y) < 1e-9
            else math.degrees(math.atan2(-away_x, -away_y))
        )
        return PoleGeometry(
            in_front=False,
            x_px=None,
            y_px=None,
            inside_frame=False,
            dx_px=None,
            dy_px=None,
            distance_px=None,
            distance_arcmin=_round(distance_arcmin, 3),
            roll_deg=None if roll is None else _round(roll, 3),
        )
    pole_x, pole_y = pole_xy
    dx, dy = pole_x - center_x, pole_y - center_y
    roll = position_angle_deg(dx, dy)
    return PoleGeometry(
        in_front=True,
        x_px=_round(pole_x, 3),
        y_px=_round(pole_y, 3),
        inside_frame=_inside(pole_x, pole_y, width_px, height_px),
        dx_px=_round(dx, 3),
        dy_px=_round(dy, 3),
        distance_px=_round(math.hypot(dx, dy), 3),
        distance_arcmin=_round(distance_arcmin, 3),
        roll_deg=None if roll is None else _round(roll, 3),
    )


def _inside(x: float, y: float, width_px: int, height_px: int) -> bool:
    return -0.5 <= x <= width_px - 0.5 and -0.5 <= y <= height_px - 0.5


def _aim(
    pole_xy: tuple[float, float] | None, aim_xy: tuple[float, float], direction: FloatArray
) -> AimGeometry:
    return AimGeometry(
        x_px=_round(aim_xy[0], 3),
        y_px=_round(aim_xy[1], 3),
        dx_px=None if pole_xy is None else _round(pole_xy[0] - aim_xy[0], 3),
        dy_px=None if pole_xy is None else _round(pole_xy[1] - aim_xy[1], 3),
        distance_arcmin=_round(_angle_arcmin(direction), 3),
    )


def _adjustment(
    attitude: CameraAttitude,
    aim_xy: tuple[float, float],
    aim_direction: FloatArray,
    zenith: npt.ArrayLike,
) -> tuple[float | None, float | None, AxesGeometry | None]:
    """The altitude part and the azimuth part of the move that brings the pole to the aim.

    At the sky direction `a` of the aim, the altitude moves the pointing toward the zenith `u`
    (along `u - (u . a) a`), and the azimuth moves it toward the east (along `a x u`, which is
    horizontal and points to the east for a camera that looks north). The move that carries the
    aim onto the pole is the part of the pole direction that is perpendicular to `a`, and its
    components along the two directions are the two parts.
    """
    up = np.asarray(zenith, dtype=np.float64)
    up = up / float(np.linalg.norm(up))
    across = np.cross(aim_direction, up)
    separation = float(np.linalg.norm(across))
    if separation < MIN_ZENITH_SEPARATION:
        return None, None, None
    east = across / separation
    toward_zenith = (up - aim_direction * float(up @ aim_direction)) / separation
    toward_pole = _Z - aim_direction * float(_Z @ aim_direction)
    altitude = (
        math.degrees(math.asin(max(-1.0, min(1.0, float(toward_pole @ toward_zenith))))) * 60.0
    )
    azimuth = math.degrees(math.asin(max(-1.0, min(1.0, float(toward_pole @ east))))) * 60.0
    return (
        _round(altitude, 3),
        _round(azimuth, 3),
        _axes(attitude, aim_xy, aim_direction, toward_zenith, east),
    )


def _axes(
    attitude: CameraAttitude,
    aim_xy: tuple[float, float],
    aim_direction: FloatArray,
    toward_zenith: FloatArray,
    east: FloatArray,
) -> AxesGeometry | None:
    """The image directions of the two moves, from where a probe point lands next to the aim."""
    cosine, sine = math.cos(_AXIS_PROBE_RAD), math.sin(_AXIS_PROBE_RAD)
    probes = np.stack(
        [aim_direction * cosine + toward_zenith * sine, aim_direction * cosine + east * sine]
    )
    x, y, front = attitude.project(probes)
    if not bool(front.all()):
        return None
    directions = []
    for index in range(2):
        dx, dy = float(x[index]) - aim_xy[0], float(y[index]) - aim_xy[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return None
        directions.append((dx / length, dy / length))
    return AxesGeometry(
        altitude_dx=_round(directions[0][0], 4),
        altitude_dy=_round(directions[0][1], 4),
        azimuth_dx=_round(directions[1][0], 4),
        azimuth_dy=_round(directions[1][1], 4),
    )


def _orbit(
    attitude: CameraAttitude, width_px: int, height_px: int, colatitude_deg: float
) -> OrbitGeometry:
    rho = math.radians(colatitude_deg)
    angles = np.linspace(0.0, 2.0 * math.pi, ORBIT_SAMPLES, endpoint=False)
    vectors = np.stack(
        [
            math.sin(rho) * np.cos(angles),
            math.sin(rho) * np.sin(angles),
            np.full(ORBIT_SAMPLES, math.cos(rho)),
        ],
        axis=-1,
    )
    x, y, front = attitude.project(vectors)
    if not bool(front.all()):
        margin = -float(max(width_px, height_px))
    else:
        edge = np.minimum.reduce([x + 0.5, width_px - 0.5 - x, y + 0.5, height_px - 0.5 - y])
        margin = float(edge.min())
    return OrbitGeometry(
        fits=margin >= 0.0,
        margin_px=_round(margin, 3),
        margin_arcmin=_round(margin * attitude.scale_arcsec_px / 60.0, 3),
    )
