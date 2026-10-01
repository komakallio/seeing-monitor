"""Star trails: how far the sky turns during an exposure, and what that does to a star.

A camera that is fixed to the Earth sees the sky turn about the celestial pole at 15.04
arcsec per second of time. A star at angular distance `d` from the pole moves at
`15.04 arcsec/s * sin(d)`, so a 30 s exposure trails a star 3 degrees from the pole by 25
arcsec, which is 6.6 pixels in bin2. The trail is a short arc that looks like a straight line,
perpendicular to the direction to the pole.

`TrailModel` predicts the trail vector of a star at any pixel from the pixel position of the
pole. The detector uses it to set the stamp size and to fit the centroid of a trailed star with
the right profile (a Gaussian smeared along a line), and the analysis uses it to tell a
star from a satellite streak. Without a solution, `trail_from_moments` estimates the trail of one
star from its second moments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.survey.apparent import EARTH_ROTATION_RATE_RAD_S
from seeingmon.survey.geometry import ARCSEC_PER_RAD, FloatArray

SIDEREAL_RATE_ARCSEC_S = EARTH_ROTATION_RATE_RAD_S * ARCSEC_PER_RAD


def analytic_trail_length_px(
    polar_distance_rad: npt.ArrayLike, exposure_s: float, scale_arcsec_px: float
) -> FloatArray:
    """The trail of a star at `polar_distance_rad` from the pole, in pixels.

    This is the exact length of the arc for a star on a circle of radius
    `sin(polar_distance)` around the pole, in the plane of the sky, divided by the plate
    scale: `15.04 arcsec/s * exposure * sin(distance) / scale`.
    """
    distance = np.asarray(polar_distance_rad, dtype=np.float64)
    return np.asarray(
        SIDEREAL_RATE_ARCSEC_S * exposure_s * np.sin(distance) / scale_arcsec_px,
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class TrailModel:
    """The trail of every star in a frame, from the pixel position of the celestial pole.

    `rotation_rad` is the angle that the Earth turns during the exposure. A star at the pixel
    `(x, y)` trails along the tangent of the circle around the pole, by `rotation_rad` times
    its distance from the pole in pixels. The small-angle form is good to 0.3% within 3.3
    degrees of the pole.
    """

    pole_x: float
    pole_y: float
    rotation_rad: float

    @classmethod
    def for_exposure(cls, pole_x: float, pole_y: float, exposure_s: float) -> TrailModel:
        return cls(pole_x, pole_y, EARTH_ROTATION_RATE_RAD_S * exposure_s)

    def vectors(self, x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """The trail vector `(dx, dy)` of each star, in pixels. The sign is arbitrary."""
        rx = np.asarray(x, dtype=np.float64) - self.pole_x
        ry = np.asarray(y, dtype=np.float64) - self.pole_y
        return -self.rotation_rad * ry, self.rotation_rad * rx

    def length(self, x: npt.ArrayLike, y: npt.ArrayLike) -> FloatArray:
        """The trail length of each star, in pixels."""
        dx, dy = self.vectors(x, y)
        return np.asarray(np.hypot(dx, dy), dtype=np.float64)

    def angle(self, x: npt.ArrayLike, y: npt.ArrayLike) -> FloatArray:
        """The direction of the trail in radians, measured from the x axis toward the y axis.

        The value lies in (-pi/2, pi/2], because a trail has no sign.
        """
        dx, dy = self.vectors(x, y)
        angle = np.arctan2(dy, dx)
        angle = np.where(angle > np.pi / 2, angle - np.pi, angle)
        return np.asarray(np.where(angle <= -np.pi / 2, angle + np.pi, angle), dtype=np.float64)


def trail_from_moments(
    major: npt.ArrayLike, minor: npt.ArrayLike, theta: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Estimate each star's trail length and angle from its second moments.

    A Gaussian of width `s` smeared along a line of length `L` has the rms width
    `sqrt(s^2 + L^2 / 12)` along the line and `s` across it. With `major` and `minor` the rms
    widths along and across the principal axes (as SEP reports them) and `theta` the angle of
    the major axis, the length is `sqrt(12 (major^2 - minor^2))`. The estimate carries a bias
    when the detection threshold truncates the profile, so use it only as a starting point or
    when there is no pole position yet.
    """
    a = np.asarray(major, dtype=np.float64)
    b = np.asarray(minor, dtype=np.float64)
    length = np.sqrt(12.0 * np.clip(a**2 - b**2, 0.0, None))
    return length, np.asarray(theta, dtype=np.float64)
