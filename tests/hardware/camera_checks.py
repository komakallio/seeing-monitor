"""The camera checks that run on real hardware, written as functions of a driver.

Each `check_*` function takes an open `CameraDriver` (the `asi` driver in `test_hardware_camera`)
and the profile, asserts what a working camera must do, and returns a short report that the test
prints. `test_camera_checks_on_fakes` runs the same functions against the fake SDK, so the code of
a check is tested before it meets a camera. The bounds are loose on purpose: a check finds a
camera that does not work, and it does not tune one.
"""

from __future__ import annotations

import statistics
from itertools import pairwise

from seeingmon.drivers.base import CameraDriver, RecoveryLevel
from seeingmon.frames import FrameFlag, PixelFormat, Roi, StreamConfig
from seeingmon.hardware.asi.api import AsiApi
from seeingmon.profile import Profile

FRAME_TIMEOUT_S = 5.0


def fast_config(profile: Profile, roi: Roi | None = None) -> StreamConfig:
    """The fast-mode stream of the profile: its readout mode, 2 ms, gain 120, a 128-pixel ROI."""
    mode = profile.fast_readout
    return StreamConfig(
        mode=mode.name,
        exposure_us=2000,
        gain=120,
        pixel_format=profile.fast_mode.pixel_format or PixelFormat.RAW16,
        roi=roi or Roi(mode.width_px // 2 - 64, mode.height_px // 2 - 64, 128, 128),
    )


def check_enumerate_and_open(api: AsiApi, driver: CameraDriver) -> str:
    """The SDK lists a camera, and the driver opens it and reports it without a serial number."""
    count = api.get_connected_camera_count()
    assert count >= 1, "the SDK lists no camera"
    info = driver.open()
    assert info.driver == "asi"
    assert info.model, "the camera reports no model name"
    assert info.max_width > 0
    assert info.max_height > 0
    return (
        f"{count} camera(s), model {info.model}, SDK {info.sdk_version}, "
        f"{info.max_width} x {info.max_height}, temperature sensor: {info.has_temperature}"
    )


def check_capabilities_match_the_profile(driver: CameraDriver, profile: Profile) -> str:
    """The camera reports the limits that the profile states. A difference means that the profile
    holds a wrong number or that the camera is another model."""
    info = driver.open()
    caps = driver.capabilities()
    limits = profile.limits
    problems: list[str] = []
    if caps.gain_range != tuple(limits.gain_range):
        problems.append(f"gain range {caps.gain_range}, profile {tuple(limits.gain_range)}")
    if caps.exposure_us_range != tuple(limits.exposure_us_range):
        problems.append(
            f"exposure range {caps.exposure_us_range}, profile {tuple(limits.exposure_us_range)}"
        )
    wanted_bins = {mode.sdk_bin for mode in profile.readout_modes}
    if not wanted_bins <= set(caps.bins):
        problems.append(
            f"bins {caps.bins} lack a binning that the profile uses: {sorted(wanted_bins)}"
        )
    problems.extend(
        f"the camera lacks {pixel_format.name}"
        for pixel_format in (PixelFormat.RAW8, PixelFormat.RAW16)
        if pixel_format not in caps.pixel_formats
    )
    finest = min(profile.readout_modes, key=lambda mode: mode.sdk_bin)
    if (info.max_width, info.max_height) != (finest.width_px, finest.height_px):
        problems.append(
            f"sensor {info.max_width} x {info.max_height}, profile {finest.width_px} x "
            f"{finest.height_px}"
        )
    if info.has_temperature != profile.sensor.has_temperature_sensor:
        problems.append(f"temperature sensor {info.has_temperature}, profile says otherwise")
    if limits.offset_range is not None and caps.offset_range != tuple(limits.offset_range):
        problems.append(f"offset range {caps.offset_range}, profile {tuple(limits.offset_range)}")
    assert not problems, "; ".join(problems)
    return f"offset range {caps.offset_range} (the profile leaves it out when it is unknown)"


def check_roi_round_trip(driver: CameraDriver, profile: Profile) -> str:
    """The camera applies the ROI that the driver asks for, and `move_roi` lands where it says."""
    mode = profile.fast_readout
    active = driver.configure(fast_config(profile))
    applied = active.config.roi
    assert applied is not None
    driver.start()
    first = driver.read_frame(FRAME_TIMEOUT_S)
    assert first.roi == applied
    assert first.data.shape == (applied.height, applied.width)
    lines = [f"applied {applied}"]
    for x, y in [(1001, 801), (3000, 2000), (0, 0), (mode.width_px, mode.height_px)]:
        moved = driver.move_roi(x, y)
        frame = driver.read_frame(FRAME_TIMEOUT_S)
        assert frame.roi == moved
        assert (moved.width, moved.height) == (applied.width, applied.height)
        assert 0 <= moved.x <= mode.width_px - moved.width
        assert 0 <= moved.y <= mode.height_px - moved.height
        lines.append(f"asked ({x}, {y}), the camera applied ({moved.x}, {moved.y})")
    driver.stop()
    return "; ".join(lines)


def check_stream(driver: CameraDriver, profile: Profile, *, frames: int = 100) -> str:
    """The camera streams the fast-mode ROI with steady timing, a consistent drop counter, and
    increasing time stamps."""
    active = driver.configure(fast_config(profile))
    driver.start()
    received = [driver.read_frame(FRAME_TIMEOUT_S) for _ in range(frames)]
    driver.stop()
    assert [frame.seq for frame in received] == list(range(frames))
    arrivals = [frame.t_arrival_ns for frame in received]
    utc = [frame.t_utc_ns for frame in received]
    assert all(b > a for a, b in pairwise(arrivals)), "arrival times repeat"
    assert all(b > a for a, b in pairwise(utc)), "UTC times do not increase"
    periods_s = [(b - a) / 1e9 for a, b in pairwise(arrivals)]
    expected_s = active.frame_period_s
    assert expected_s is not None
    median_s = statistics.median(periods_s)
    assert 0.5 * expected_s < median_s < 2.0 * expected_s, (
        f"the median period is {median_s * 1e3:.2f} ms, "
        f"the profile predicts {expected_s * 1e3:.2f} ms"
    )
    dropped = sum(frame.dropped_before for frame in received)
    assert dropped <= frames // 20, f"{dropped} of {frames} frames were dropped"
    assert driver.dropped_frames() >= dropped
    jitter_ms = statistics.pstdev(periods_s) * 1e3
    return (
        f"{frames} frames at {1 / statistics.mean(periods_s):.1f} fps (the profile predicts "
        f"{1 / expected_s:.1f}), jitter {jitter_ms:.2f} ms, {dropped} dropped, "
        f"t_err {received[0].t_err_ns / 1e6:.1f} ms, time quality {received[0].t_quality.name}"
    )


def check_temperature(driver: CameraDriver, profile: Profile) -> str:
    """The sensor temperature is plausible, and the frames carry it."""
    value = driver.read_temperature_c()
    if not profile.sensor.has_temperature_sensor:
        assert value is None
        return "the profile states no temperature sensor, and the driver reports none"
    assert value is not None, "the profile states a temperature sensor, and the driver reports none"
    assert -30.0 < value < 70.0, f"the sensor temperature is implausible: {value}"
    driver.configure(fast_config(profile))
    driver.start()
    frame = driver.read_frame(FRAME_TIMEOUT_S)
    driver.stop()
    assert frame.temperature_c is None or -30.0 < frame.temperature_c < 70.0
    return f"{value:.1f} degrees Celsius, first frame carries {frame.temperature_c}"


def check_recovery_restart(driver: CameraDriver, profile: Profile) -> str:
    """Recovery step 1 restarts capture, and the stream continues with the RECOVERED flag."""
    driver.configure(fast_config(profile))
    driver.start()
    before = [driver.read_frame(FRAME_TIMEOUT_S) for _ in range(5)]
    driver.recover(RecoveryLevel.RESTART_CAPTURE)
    first = driver.read_frame(FRAME_TIMEOUT_S)
    assert first.flags & FrameFlag.RECOVERED, "the first frame after the step lacks RECOVERED"
    after = [driver.read_frame(FRAME_TIMEOUT_S) for _ in range(5)]
    driver.stop()
    assert first.stream_id == before[0].stream_id
    assert [frame.seq for frame in after] == list(range(first.seq + 1, first.seq + 6))
    assert not any(frame.flags & FrameFlag.RECOVERED for frame in after)
    return "capture restarted, and the stream continued"


def check_usb_reset(driver: CameraDriver, profile: Profile) -> str:
    """Recovery step 3 resets the USB device, and the camera comes back with the stream."""
    driver.configure(fast_config(profile))
    driver.start()
    driver.read_frame(FRAME_TIMEOUT_S)
    driver.recover(RecoveryLevel.USB_RESET)
    frame = driver.read_frame(FRAME_TIMEOUT_S)
    assert frame.flags & FrameFlag.RECOVERED
    driver.stop()
    return "the camera reappeared and streamed again"
