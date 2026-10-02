"""The camera checks of `camera_checks`, run against the fake SDK so that their code is tested."""

from __future__ import annotations

import re

import pytest

from seeingmon.drivers.asi import AsiDriver
from seeingmon.drivers.base import CameraDriver
from seeingmon.hardware.asi.api import AsiControl
from seeingmon.hardware.asi.fake import DEFAULT_TIMING
from tests.hardware import camera_checks as checks
from tests.hardware.asi_support import Rig, make_rig, reference_profile


@pytest.fixture
def rig() -> Rig:
    return make_rig()


def test_enumerate_and_open(rig: Rig) -> None:
    report = checks.check_enumerate_and_open(rig.sdk, rig.driver)
    assert "ZWO ASI294MM" in report
    assert "serial" not in report.lower()


def test_the_fake_camera_matches_the_reference_profile(rig: Rig) -> None:
    checks.check_capabilities_match_the_profile(rig.driver, reference_profile())


def test_a_camera_that_differs_from_the_profile_fails_the_check() -> None:
    rig = make_rig(sdk={"max_width": 4144, "max_height": 2822})
    with pytest.raises(AssertionError, match="sensor 4144 x 2822"):
        checks.check_capabilities_match_the_profile(rig.driver, reference_profile())


def test_roi_round_trip(rig: Rig) -> None:
    rig.driver.open()
    report = checks.check_roi_round_trip(rig.driver, reference_profile())
    assert "asked (1001, 801)" in report


def test_the_roi_check_notices_a_camera_that_aligns_the_start_position(rig: Rig) -> None:
    aligned = make_rig(sdk={"start_alignment": 4})
    aligned.driver.open()
    report = checks.check_roi_round_trip(aligned.driver, reference_profile())
    assert "asked (1001, 801), the camera applied (1000, 800)" in report


def test_stream(rig: Rig) -> None:
    rig.driver.open()
    report = checks.check_stream(rig.driver, reference_profile())
    assert "100 frames" in report
    assert "0 dropped" in report


def test_the_stream_check_fails_when_the_camera_drops_frames(rig: Rig) -> None:
    rig.driver.open()
    original = rig.sdk.get_video_data
    state = {"reads": 0}

    def lossy(camera_id: int, buffer: bytearray, wait_ms: int) -> None:
        state["reads"] += 1
        if state["reads"] % 3 == 0:
            rig.sdk.lose_frames(1)
        original(camera_id, buffer, wait_ms)

    rig.sdk.get_video_data = lossy  # type: ignore[method-assign]
    with pytest.raises(AssertionError, match="dropped"):
        checks.check_stream(rig.driver, reference_profile())


def test_temperature(rig: Rig) -> None:
    rig.driver.open()
    report = checks.check_temperature(rig.driver, reference_profile())
    assert "18.3" in report


def test_recovery_step_one(rig: Rig) -> None:
    rig.driver.open()
    checks.check_recovery_restart(rig.driver, reference_profile())


def test_recovery_step_three(rig: Rig) -> None:
    rig.driver.open()
    checks.check_usb_reset(rig.driver, reference_profile())
    assert rig.resetter.count == 1


def test_every_check_works_through_the_driver_protocol(rig: Rig) -> None:
    driver: CameraDriver = rig.driver
    rig.driver.open()
    checks.check_recovery_restart(driver, reference_profile())


def test_stale_controls_are_overridden_and_the_camera_comes_back(rig: Rig) -> None:
    state = rig.sdk.state
    before = (dict(state.controls), set(state.automatic))
    report = checks.check_stale_controls_are_overridden(rig.sdk, rig.driver, reference_profile())
    assert "applied bandwidth 100, flip 0" in report
    assert "camera is back as it was" in report
    assert (dict(state.controls), set(state.automatic)) == before


def test_the_stale_controls_check_fails_for_a_driver_that_leaves_the_bandwidth_alone() -> None:
    rig = make_rig(bandwidth_pct=None)  # the option that leaves the control: the old behavior
    with pytest.raises(AssertionError, match="bandwidth_overload"):
        checks.check_stale_controls_are_overridden(rig.sdk, rig.driver, reference_profile())
    assert rig.sdk.state.controls[6] == 50  # the check still put the camera back


def test_the_high_speed_flag_takes_effect_in_every_order_and_the_camera_comes_back(
    rig: Rig,
) -> None:
    state = rig.sdk.state
    before = (dict(state.controls), set(state.automatic))
    report = checks.check_high_speed_follows_the_flag(rig.driver, reference_profile())
    assert "n128 " in report
    assert "h8_128 " in report
    assert "camera is back as it was" in report
    assert (dict(state.controls), set(state.automatic)) == before
    assert not rig.sdk.video_active


def test_the_high_speed_check_runs_the_bench_sequence_on_a_camera_that_latches_late(
    rig: Rig,
) -> None:
    report = checks.check_high_speed_follows_the_flag(rig.driver, reference_profile())
    rates = [float(rate) for rate in re.findall(r"[nh]\d+(?:_\d+)? (\d+\.\d)", report)]
    assert len(rates) == len(checks.HIGH_SPEED_SEQUENCE)
    normal, fast = rates[0], rates[2]
    assert fast > 1.2 * normal
    assert rates == [normal, fast, fast, fast, normal, normal, normal]


def test_the_high_speed_check_fails_for_a_driver_that_sets_the_flag_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The driver of the first run: it sets the flag, and the camera keeps the old regime."""
    rig = make_rig()
    monkeypatch.setattr(AsiDriver, "_latch_high_speed", lambda self, plan: None)
    state = rig.sdk.state
    with pytest.raises(AssertionError, match=r"step 2 \(h128\) ran at \d+\.\d fps"):
        checks.check_high_speed_follows_the_flag(rig.driver, reference_profile())
    assert state.controls[AsiControl.HIGH_SPEED_MODE] == 0  # the check still put the camera back


def test_the_high_speed_check_fails_for_a_camera_without_a_regime_change() -> None:
    same = {**DEFAULT_TIMING, (1, True): DEFAULT_TIMING[(1, False)]}
    rig = make_rig(sdk={"timing": same})
    with pytest.raises(AssertionError, match="no difference between the regimes"):
        checks.check_high_speed_follows_the_flag(rig.driver, reference_profile())


def test_the_high_speed_check_has_nothing_to_do_for_a_profile_without_a_high_speed_mode() -> None:
    profile = reference_profile()
    plain = {
        "adc_bits_high_speed": None,
        "row_time_us_high_speed": None,
        "frame_overhead_ms_high_speed": None,
    }
    profile = profile.model_copy(
        update={"readout_modes": [m.model_copy(update=plain) for m in profile.readout_modes]}
    )
    rig = make_rig()
    assert "nothing to check" in checks.check_high_speed_follows_the_flag(rig.driver, profile)
    assert rig.sdk.calls == []  # it never touched the camera
