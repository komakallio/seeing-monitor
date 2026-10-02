"""The camera checks that run on real hardware, written as functions of a driver.

Each `check_*` function takes an open `CameraDriver` (the `asi` driver in `test_hardware_camera`)
and the profile, asserts what a working camera must do, and returns a short report that the test
prints. `test_camera_checks_on_fakes` runs the same functions against the fake SDK, so the code of
a check is tested before it meets a camera. The bounds are loose on purpose: a check finds a
camera that does not work, and it does not tune one.
"""

from __future__ import annotations

import statistics
from dataclasses import replace
from itertools import pairwise

from seeingmon.drivers.asi import AsiDriver
from seeingmon.drivers.base import CameraDriver, RecoveryLevel
from seeingmon.frames import FrameFlag, PixelFormat, Roi, StreamConfig
from seeingmon.hardware.asi.api import AsiApi, AsiControl
from seeingmon.profile import Profile

FRAME_TIMEOUT_S = 5.0

# The sequence that showed the defect on the bench: n128 / h128 / h8_128 / h128 / n128 / n8_128 /
# n128. Each step is a label, the high-speed flag, and the pixel format, at 128 x 128 pixels (n is
# the normal flag, h is the high-speed flag, and 8 is RAW8; the others are RAW16). The second and
# the fifth step change the flag alone, which the camera ignores unless the driver changes the
# image format with it.
HIGH_SPEED_SEQUENCE: tuple[tuple[str, bool, PixelFormat], ...] = (
    ("n128", False, PixelFormat.RAW16),
    ("h128", True, PixelFormat.RAW16),
    ("h8_128", True, PixelFormat.RAW8),
    ("h128", True, PixelFormat.RAW16),
    ("n128", False, PixelFormat.RAW16),
    ("n8_128", False, PixelFormat.RAW8),
    ("n128", False, PixelFormat.RAW16),
)
SEQUENCE_SETTLE_FRAMES = 3  # frames that a step reads before it measures
REGIME_AGREEMENT = 0.05  # the steps of one regime agree in their frame period within this much


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


def check_stale_controls_are_overridden(api: AsiApi, driver: AsiDriver, profile: Profile) -> str:
    """The driver sets the controls that the camera keeps, over a camera that another program left
    in another state, and the camera is back as it was at the end.

    The camera keeps its USB bandwidth, flip, offset, and high-speed mode between processes. The
    check makes them stale (half the bandwidth, a flip, another offset, high-speed mode), configures
    the fast stream, and reads the controls back through the SDK. It also streams, because a stale
    bandwidth halves the frame rate.
    """
    driver.open()
    saved = driver.save_settings()
    try:
        camera_id = api.get_camera_property(0).camera_id
        offset_range = driver.capabilities().offset_range
        stale_offset = 20 if offset_range is None else min(20, offset_range[1])
        for control, value in (
            (AsiControl.BANDWIDTH_OVERLOAD, 50),
            (AsiControl.FLIP, 3),
            (AsiControl.OFFSET, stale_offset),
            (AsiControl.HIGH_SPEED_MODE, 1),
        ):
            api.set_control_value(camera_id, control, value)
        active = driver.configure(fast_config(profile))
        applied = {
            control.name.lower(): api.get_control_value(camera_id, control)
            for control in (
                AsiControl.BANDWIDTH_OVERLOAD,
                AsiControl.FLIP,
                AsiControl.OFFSET,
                AsiControl.HIGH_SPEED_MODE,
            )
        }
        assert applied["bandwidth_overload"] == (100, False), applied
        assert applied["flip"] == (0, False), applied
        assert applied["high_speed_mode"] == (0, False), applied
        assert applied["offset"][0] == active.config.offset, applied
        assert not applied["offset"][1], applied
        driver.start()
        arrivals = [driver.read_frame(FRAME_TIMEOUT_S).t_arrival_ns for _ in range(41)]
        driver.stop()
        fps = 40 / ((arrivals[-1] - arrivals[0]) / 1e9)
        expected_s = active.frame_period_s
        assert expected_s is not None
        assert fps > 0.7 / expected_s, (
            f"the stream runs at {fps:.1f} fps and the profile predicts {1 / expected_s:.1f}: a "
            "bandwidth that the driver left as it found it would give about half"
        )
    finally:
        problems = driver.restore_settings(saved)
    assert not problems, f"the camera could not be restored: {problems}"
    return (
        f"over a stale camera (bandwidth 50, flip 3, offset {stale_offset}, high-speed on) the "
        f"driver applied bandwidth 100, flip 0, offset {active.config.offset}, and normal speed, "
        f"and the stream ran at {fps:.1f} fps (the profile predicts {1 / expected_s:.1f}); the "
        "camera is back as it was"
    )


def check_high_speed_follows_the_flag(
    driver: AsiDriver, profile: Profile, *, frames: int = 40
) -> str:
    """A change of the high-speed flag changes the readout regime of the camera, in every order.

    The ASI294MM takes up the flag only at the first ROI format after it opens and when the image
    format changes. A driver that sets the flag alone leaves the camera in the old regime: the frame
    rate is off by 20 to 25%, the ADC depth is another, and the camera reports no error. The check
    runs the bench sequence (`HIGH_SPEED_SEQUENCE`) on the fast stream of the profile, takes the
    median frame period of each step, and asserts that the steps of one regime agree and that the
    high-speed steps are faster than the normal ones by about what the profile predicts. It needs
    no number of the profile but that ratio, so it works before the profile holds measured timing.
    The camera is back as it was at the end.
    """
    mode = profile.fast_readout
    if not mode.has_high_speed:
        return "the fast readout mode of the profile has no high-speed variant: nothing to check"
    base = fast_config(profile)
    assert base.roi is not None
    rows, exposure_us = base.roi.height, base.exposure_us
    predicted = profile.frame_period_s(
        mode.high_speed_variant(), rows, exposure_us
    ) / profile.frame_period_s(mode, rows, exposure_us)
    if predicted > 0.95:
        return "the profile predicts under 5% between the regimes: nothing to tell them apart by"
    driver.open()
    saved = driver.save_settings()
    try:
        steps: list[tuple[str, bool, float]] = []
        for label, high_speed, pixel_format in HIGH_SPEED_SEQUENCE:
            driver.configure(replace(base, high_speed=high_speed, pixel_format=pixel_format))
            driver.start()
            received = [
                driver.read_frame(FRAME_TIMEOUT_S) for _ in range(SEQUENCE_SETTLE_FRAMES + frames)
            ]
            driver.stop()
            arrivals = [frame.t_arrival_ns for frame in received[SEQUENCE_SETTLE_FRAMES:]]
            period_s = statistics.median((b - a) / 1e9 for a, b in pairwise(arrivals))
            steps.append((label, high_speed, period_s))
    finally:
        problems = driver.restore_settings(saved)
    assert not problems, f"the camera could not be restored: {problems}"
    normal_s = statistics.median(period for _, high_speed, period in steps if not high_speed)
    fast_s = statistics.median(period for _, high_speed, period in steps if high_speed)
    listing = ", ".join(f"{label} {1 / period:.1f}" for label, _, period in steps)
    for number, (label, high_speed, period_s) in enumerate(steps, start=1):
        regime_s = fast_s if high_speed else normal_s
        assert abs(period_s / regime_s - 1.0) < REGIME_AGREEMENT, (
            f"step {number} ({label}) ran at {1 / period_s:.1f} fps, and the median of its regime "
            f"is {1 / regime_s:.1f} fps: the camera did not take up the flag that the step asked "
            f"for. The steps ran at {listing} fps."
        )
    assert fast_s < normal_s * (1.0 + predicted) / 2.0, (
        f"the high-speed steps ran at {1 / fast_s:.1f} fps and the normal steps at "
        f"{1 / normal_s:.1f}: the camera shows no difference between the regimes, and the profile "
        f"predicts {1 / predicted:.2f} times the rate. The steps ran at {listing} fps."
    )
    return (
        f"{listing} fps; the high-speed regime runs {normal_s / fast_s:.2f} times as fast as the "
        f"normal one (the profile predicts {1 / predicted:.2f}); the camera is back as it was"
    )
