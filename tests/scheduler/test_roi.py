"""ROI placement through the profile's helpers."""

from __future__ import annotations

import pytest

from seeingmon.profile import load_profile
from seeingmon.scheduler.roi import roi_at_sensor_center, roi_centered_on

PROFILE = load_profile("asi294mm-gs250")
BIN1 = PROFILE.mode("bin1")


def test_4_1_arcmin_in_bin1_is_a_128_pixel_roi_centered_on_the_star() -> None:
    roi = roi_centered_on(PROFILE, "bin1", (4144.0, 2822.0), 4.1)
    assert (roi.width, roi.height) == (128, 128)  # the architecture's 128 x 128 ROI
    assert (roi.x, roi.y) == (4144 - 64, 2822 - 64)
    assert roi.contains(4144.0, 2822.0)
    assert roi.distance_to_edge(4144.0, 2822.0) == 64


def test_the_roi_follows_the_profile_rules_for_the_origin_and_the_size() -> None:
    roi = roi_centered_on(PROFILE, "bin1", (1000.4, 2000.6), 4.1)
    limits = PROFILE.limits
    assert roi.width % limits.roi_width_multiple == 0
    assert roi.height % limits.roi_height_multiple == 0
    assert (roi.x, roi.y) == (936, 1937)  # the origin rounds to the nearest pixel


@pytest.mark.parametrize(
    ("center", "origin"),
    [
        ((10.0, 10.0), (0, 0)),  # near the top left corner
        ((BIN1.width_px - 5.0, BIN1.height_px - 5.0), (BIN1.width_px - 128, BIN1.height_px - 128)),
        ((-500.0, 3000.0), (0, 3000 - 64)),  # a prediction off the sensor
    ],
)
def test_the_roi_clamps_to_the_sensor(center: tuple[float, float], origin: tuple[int, int]) -> None:
    roi = roi_centered_on(PROFILE, "bin1", center, 4.1)
    assert (roi.x, roi.y) == origin
    assert 0 <= roi.x <= BIN1.width_px - roi.width
    assert 0 <= roi.y <= BIN1.height_px - roi.height


def test_the_same_angle_is_half_the_pixels_in_bin2() -> None:
    roi = roi_centered_on(PROFILE, "bin2", (2072.0, 1411.0), 4.1)
    assert (roi.width, roi.height) == (64, 64)  # bin2 pixels are 3.82 arcsec


def test_the_watch_roi_sits_at_the_middle_of_the_sensor() -> None:
    roi = roi_at_sensor_center(PROFILE, "bin2", 20.0)
    assert roi is not None
    assert (roi.width, roi.height) == (312, 314)
    mode = PROFILE.mode("bin2")
    assert abs((roi.x + roi.width / 2) - mode.width_px / 2) <= 1
    assert abs((roi.y + roi.height / 2) - mode.height_px / 2) <= 1


def test_a_width_of_zero_means_the_full_frame() -> None:
    assert roi_at_sensor_center(PROFILE, "bin2", 0.0) is None
