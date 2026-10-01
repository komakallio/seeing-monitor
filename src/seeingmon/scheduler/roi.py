"""ROI placement through the profile's helpers.

The profile knows the pixel size, the plate scale, and the ROI rules of each readout mode, so
the scheduler never computes a size or a rounding itself. A center is a position in sensor pixels
of the readout mode, as the `PointingProvider` and the analysis report it.
"""

from __future__ import annotations

from seeingmon.frames import Roi
from seeingmon.profile import Profile


def roi_centered_on(profile: Profile, mode: str, center: tuple[float, float], arcmin: float) -> Roi:
    """A square ROI of `arcmin` full width around `center`, rounded and clamped to the frame."""
    width, height = profile.roi_size_px(mode, arcmin)
    x, y = center
    return profile.clamp_roi(mode, x - width / 2.0, y - height / 2.0, width, height)


def roi_at_sensor_center(profile: Profile, mode: str, arcmin: float) -> Roi | None:
    """A square ROI of `arcmin` full width at the middle of the sensor.

    A width of 0 means the full frame, and the function returns `None`.
    """
    if arcmin <= 0:
        return None
    readout = profile.mode(mode)
    return roi_centered_on(profile, mode, (readout.width_px / 2.0, readout.height_px / 2.0), arcmin)
