"""The sky view of a frame: where the celestial pole falls, and whether the orbit of Polaris fits.

A fixed camera that points near the pole sees Polaris go round the pole once a day, on a circle of
about 0.6 degrees. A first alignment has one job: put the pole near the center of the sensor, so
that the whole circle stays in the frame all day. `build_sky_view` turns a camera attitude and the
frame size into the numbers that the alignment page needs for that job.

- **The camera.** The rotation (row by row), the plate scale, the parity, and the principal point,
  rounded to what a pixel needs, so that a browser can project sky points with the same formula as
  `CameraAttitude.project` (`w = R u`, then `x = x_c + w_x / w_z / s` and `y = y_c + parity *
  w_y / w_z / s`).
- **The pole.** The celestial pole of date is the z axis of CIRS, the apparent frame of the
  survey path. The view gives its pixel (when it is in front of the camera), whether it lies in
  the frame, its offset from the frame center, its distance from the frame center, and the
  position angle of its direction.
- **The orbit.** The circle at the colatitude of Polaris around the pole, sampled at
  `ORBIT_SAMPLES` points and projected. The margin is the distance from the circle to the
  nearest frame edge in pixels, and it is negative when the circle leaves the frame.

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

if TYPE_CHECKING:
    from seeingmon.survey.wcs_fit import CameraAttitude

# The same limit as `seeingmon.survey.wcs_fit.ROLL_MIN_DISTANCE_PX`, which a test checks.
ROLL_MIN_DISTANCE_PX = 1.0
# The points of the orbit. A step of 1 degree puts the sampled margin within 0.02 pixel of the
# true one for a circle of 600 pixels.
ORBIT_SAMPLES = 360
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
    within `ROLL_MIN_DISTANCE_PX` of the center).
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
class SkyGeometry:
    """The sky view of one frame."""

    camera: CameraGeometry
    pole: PoleGeometry
    polaris_colatitude_deg: float | None
    orbit: OrbitGeometry | None


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


def build_sky_view(
    attitude: CameraAttitude,
    width_px: int,
    height_px: int,
    polaris_colatitude_deg: float | None,
) -> SkyGeometry:
    """The sky view of a frame of `width_px` by `height_px` pixels.

    `attitude` is the camera model in the apparent frame at the time of the frame, and
    `polaris_colatitude_deg` is the angle between Polaris and the pole at that time
    (`PointingSolution.polaris_colatitude_deg`), or `None` when it is not known.
    """
    colatitude = _valid_colatitude(polaris_colatitude_deg)
    pole = _pole(attitude, width_px, height_px)
    orbit = None if colatitude is None else _orbit(attitude, width_px, height_px, colatitude)
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
    )


def _valid_colatitude(value: float | None) -> float | None:
    if value is None or not math.isfinite(value) or not 0.0 <= value < 90.0:
        return None
    return float(value)


def _pole(attitude: CameraAttitude, width_px: int, height_px: int) -> PoleGeometry:
    center_x, center_y = frame_center(width_px, height_px)
    # The angular distance from the frame center to the pole is exact, and it also holds when
    # the pole lies behind the camera.
    toward_center = attitude.unproject(center_x, center_y)[0]
    distance_arcmin = (
        math.degrees(
            math.atan2(
                math.hypot(float(toward_center[0]), float(toward_center[1])), toward_center[2]
            )
        )
        * 60.0
    )
    x, y, front = attitude.project(_Z)
    if not front[0]:
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
            distance_arcmin=round(distance_arcmin, 3),
            roll_deg=None if roll is None else round(roll, 3),
        )
    pole_x, pole_y = float(x[0]), float(y[0])
    dx, dy = pole_x - center_x, pole_y - center_y
    roll = position_angle_deg(dx, dy)
    return PoleGeometry(
        in_front=True,
        x_px=round(pole_x, 3),
        y_px=round(pole_y, 3),
        inside_frame=_inside(pole_x, pole_y, width_px, height_px),
        dx_px=round(dx, 3),
        dy_px=round(dy, 3),
        distance_px=round(math.hypot(dx, dy), 3),
        distance_arcmin=round(distance_arcmin, 3),
        roll_deg=None if roll is None else round(roll, 3),
    )


def _inside(x: float, y: float, width_px: int, height_px: int) -> bool:
    return -0.5 <= x <= width_px - 0.5 and -0.5 <= y <= height_px - 0.5


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
        margin_px=round(margin, 3),
        margin_arcmin=round(margin * attitude.scale_arcsec_px / 60.0, 3),
    )
