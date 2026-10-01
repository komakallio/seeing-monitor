"""The recovery ladder, the reader and stop rules, and the watchdog around SDK calls."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from seeingmon.clock import NS_PER_S
from seeingmon.drivers import (
    CameraDisconnectedError,
    CameraError,
    CameraStateError,
    CameraTimeoutError,
    RecoveryLevel,
)
from seeingmon.frames import FrameFlag, Roi
from seeingmon.hardware.asi.api import AsiControl, AsiErrorCode
from seeingmon.hardware.asi.fake import FakeAsiSdk
from seeingmon.hardware.asi.usb import UsbResetError
from tests.hardware.asi_support import FAST, TINY, Rig, make_rig, run_in_thread, wait_until


def names(rig: Rig) -> list[str]:
    return [name for name, _ in rig.sdk.calls]


class TestRestartCapture:
    def test_a_stalled_stream_restarts_and_marks_the_first_frame(self) -> None:
        rig = make_rig().streaming(TINY)
        stream_id = rig.driver.read_frame(1.0).stream_id
        rig.sdk.stall_reads(1)
        with pytest.raises(CameraTimeoutError):
            rig.driver.read_frame(1.0)
        rig.driver.recover(RecoveryLevel.RESTART_CAPTURE)
        first = rig.driver.read_frame(1.0)
        second = rig.driver.read_frame(1.0)
        assert first.flags & FrameFlag.RECOVERED
        assert not second.flags & FrameFlag.RECOVERED
        assert (first.stream_id, first.seq) == (stream_id, 1)  # the stream continues
        assert second.seq == 2

    def test_the_step_stops_capture_before_it_applies_the_settings_and_starts_again(self) -> None:
        rig = make_rig().streaming(TINY)
        rig.sdk.calls.clear()
        rig.driver.recover(RecoveryLevel.RESTART_CAPTURE)
        order = [
            "stop_video_capture",
            "set_control_value",
            "set_roi_format",
            "set_start_position",
            "get_roi_format",
            "start_video_capture",
        ]
        called = names(rig)
        positions = [called.index(name) for name in order]
        assert positions == sorted(positions)
        assert "close_camera" not in called

    def test_the_step_keeps_a_roi_that_move_roi_set(self) -> None:
        rig = make_rig().streaming(TINY)
        rig.driver.move_roi(40, 20)
        rig.driver.recover(RecoveryLevel.RESTART_CAPTURE)
        assert rig.sdk.roi == (40, 20, 16, 8)
        assert rig.driver.read_frame(1.0).roi == Roi(40, 20, 16, 8)

    def test_a_stopped_stream_stays_stopped(self) -> None:
        rig = make_rig().streaming(TINY)
        rig.driver.stop()
        rig.driver.recover(RecoveryLevel.RESTART_CAPTURE)
        assert not rig.sdk.video_active
        with pytest.raises(CameraStateError):
            rig.driver.read_frame(1.0)

    def test_the_caller_may_start_again_after_a_step(self) -> None:
        rig = make_rig().streaming(TINY)
        rig.driver.recover(RecoveryLevel.RESTART_CAPTURE)
        rig.driver.start()  # does nothing when capture runs
        assert len(rig.sdk.calls_named("start_video_capture")) == 2
        rig.driver.read_frame(1.0)

    def test_a_driver_that_was_never_configured_has_nothing_to_reapply(self) -> None:
        rig = make_rig().opened()
        rig.driver.recover(RecoveryLevel.RESTART_CAPTURE)
        assert rig.sdk.calls_named("set_roi_format") == []

    def test_a_snapshot_stays_stopped_after_a_step(self) -> None:
        from seeingmon.frames import StreamConfig, StreamKind

        rig = make_rig().opened()
        rig.driver.configure(
            StreamConfig(
                mode="bin2",
                exposure_us=1_000_000,
                gain=1,
                kind=StreamKind.SNAPSHOT,
                roi=Roi(0, 0, 64, 64),
            )
        )
        rig.driver.start()
        rig.driver.recover(RecoveryLevel.RESTART_CAPTURE)
        assert len(rig.sdk.calls_named("start_exposure")) == 2  # the step started a new one
        assert rig.driver.read_frame(10.0).flags & FrameFlag.RECOVERED


class TestReopen:
    def test_the_step_closes_and_opens_the_camera_and_reapplies_the_settings(self) -> None:
        rig = make_rig().streaming(FAST)
        rig.sdk.set_control_value(0, AsiControl.GAIN, 0)  # something changed the camera
        rig.sdk.calls.clear()
        rig.driver.recover(RecoveryLevel.REOPEN)
        called = names(rig)
        assert called.index("close_camera") < called.index("open_camera")
        assert called.index("init_camera") < called.index("start_video_capture")
        assert rig.sdk.control(AsiControl.GAIN) == 120
        frame = rig.driver.read_frame(1.0)
        assert frame.flags & FrameFlag.RECOVERED
        assert frame.roi == Roi(100, 200, 128, 128)

    def test_the_step_fails_when_the_camera_is_gone_and_works_after_it_returns(self) -> None:
        rig = make_rig().streaming(TINY)
        rig.sdk.disconnect()
        with pytest.raises(CameraDisconnectedError):
            rig.driver.recover(RecoveryLevel.REOPEN)
        rig.sdk.reconnect()
        rig.driver.recover(RecoveryLevel.REOPEN)
        assert rig.driver.read_frame(1.0).flags & FrameFlag.RECOVERED

    def test_a_lower_step_reopens_a_camera_that_a_failed_step_left_closed(self) -> None:
        rig = make_rig().streaming(TINY)
        rig.sdk.disconnect()
        with pytest.raises(CameraDisconnectedError):
            rig.driver.recover(RecoveryLevel.REOPEN)
        rig.sdk.reconnect()
        rig.driver.recover(RecoveryLevel.RESTART_CAPTURE)
        rig.driver.read_frame(1.0)

    def test_recovery_needs_an_open_camera(self) -> None:
        rig = make_rig().streaming(TINY)
        rig.driver.close()
        with pytest.raises(CameraStateError):
            rig.driver.recover(RecoveryLevel.REOPEN)
        with pytest.raises(CameraStateError):
            make_rig().driver.recover(RecoveryLevel.RESTART_CAPTURE)

    def test_a_driver_that_was_never_configured_only_reopens(self) -> None:
        rig = make_rig().opened()
        rig.driver.recover(RecoveryLevel.REOPEN)
        assert len(rig.sdk.calls_named("open_camera")) == 2
        assert rig.sdk.calls_named("set_roi_format") == []


class TestUsbReset:
    def test_the_step_resets_waits_for_the_camera_and_starts_from_power_on_state(self) -> None:
        rig = make_rig(reappear_after_s=2.0).streaming(FAST)
        started = rig.clock.monotonic_ns()
        rig.driver.recover(RecoveryLevel.USB_RESET)
        assert rig.resetter.count == 1
        waited_s = (rig.clock.monotonic_ns() - started) / NS_PER_S
        assert 2.0 <= waited_s < 2.5  # the camera was gone for 2 s of the clock's time
        assert (
            rig.sdk.control(AsiControl.GAIN) == 120
        )  # the reset cleared it, and the driver set it
        frame = rig.driver.read_frame(1.0)
        assert frame.flags & FrameFlag.RECOVERED
        assert frame.roi == Roi(100, 200, 128, 128)

    def test_a_camera_that_never_returns_ends_the_step_after_the_timeout(self) -> None:
        rig = make_rig(reappear_after_s=1e6, usb_reenumerate_timeout_s=5.0).streaming(TINY)
        started = rig.clock.monotonic_ns()
        with pytest.raises(CameraDisconnectedError, match="did not come back"):
            rig.driver.recover(RecoveryLevel.USB_RESET)
        assert 5.0 <= (rig.clock.monotonic_ns() - started) / NS_PER_S < 6.0

    def test_a_failing_reset_is_a_camera_error(self) -> None:
        rig = make_rig().streaming(TINY)
        rig.resetter.failure = UsbResetError("no device")
        with pytest.raises(CameraError):
            rig.driver.recover(RecoveryLevel.USB_RESET)

    def test_a_driver_without_a_resetter_refuses_the_step(self) -> None:
        from seeingmon.drivers.asi import AsiDriver
        from tests.hardware.asi_support import reference_profile

        rig = make_rig()
        driver = AsiDriver(api=rig.sdk, profile=reference_profile(), clock=rig.clock)
        driver.open()
        with pytest.raises(CameraError, match="resetter"):
            driver.recover(RecoveryLevel.USB_RESET)


class TestReaderAndStop:
    def hang(
        self, rig: Rig, timeout_s: float = 1.0
    ) -> tuple[threading.Event, threading.Thread, list[object]]:
        """Block a read inside the SDK, and wait until it is blocked."""
        release = rig.sdk.hang_next("get_video_data")
        thread, outcome = run_in_thread(lambda: rig.driver.read_frame(timeout_s))
        assert wait_until(lambda: rig.watchdog.armed == 1)
        return release, thread, outcome

    def test_stop_waits_for_the_reader_before_it_stops_capture(self) -> None:
        rig = make_rig().streaming(TINY)
        rig.sdk.calls.clear()  # forget the calls that opened the camera
        release, reader, _ = self.hang(rig)
        stopper, stopped = run_in_thread(rig.driver.stop)
        assert wait_until(lambda: rig.driver._stopping)
        stopper.join(0.2)
        assert stopper.is_alive()  # waiting for the reader
        assert "stop_video_capture" not in names(rig)
        release.set()
        reader.join(5.0)
        stopper.join(5.0)
        assert not stopper.is_alive()
        assert stopped == [None]
        called = names(rig)
        assert called.index("get_video_data_returned") < called.index("stop_video_capture")

    def test_configure_waits_for_the_reader_too(self) -> None:
        rig = make_rig().streaming(TINY)
        release, reader, _ = self.hang(rig)
        configurer, outcome = run_in_thread(lambda: rig.driver.configure(FAST))
        assert wait_until(lambda: rig.driver._stopping)
        configurer.join(0.2)
        assert configurer.is_alive()
        assert "set_roi_format" not in names(rig)[names(rig).index("get_video_data") :]
        release.set()
        reader.join(5.0)
        configurer.join(5.0)
        assert not isinstance(outcome[0], BaseException)

    def test_a_reader_that_never_returns_makes_stop_raise_and_leaves_the_sdk_alone(self) -> None:
        rig = make_rig(call_timeout_s=0.2).streaming(TINY)
        rig.sdk.calls.clear()  # forget the calls that opened the camera
        release, reader, _ = self.hang(rig, timeout_s=0.05)
        with pytest.raises(CameraTimeoutError, match="did not return"):
            rig.driver.stop()
        assert "stop_video_capture" not in names(rig)
        release.set()
        reader.join(5.0)

    def test_close_with_a_stuck_reader_leaves_the_camera_open_and_returns(self) -> None:
        rig = make_rig(call_timeout_s=0.2).streaming(TINY)
        rig.sdk.calls.clear()  # forget the calls that opened the camera
        release, reader, _ = self.hang(rig, timeout_s=0.05)
        rig.driver.close()
        assert "close_camera" not in names(rig)
        release.set()
        reader.join(5.0)

    def test_other_calls_run_while_a_read_waits(self) -> None:
        rig = make_rig().streaming(TINY)
        release, reader, _ = self.hang(rig)
        assert rig.driver.move_roi(40, 20) == Roi(40, 20, 16, 8)
        assert rig.driver.read_temperature_c() is not None
        assert rig.driver.dropped_frames() >= 0  # the warm-up sleep advanced virtual time
        release.set()
        reader.join(5.0)

    def test_a_second_reader_is_refused(self) -> None:
        rig = make_rig().streaming(TINY)
        release, reader, _ = self.hang(rig)
        with pytest.raises(CameraStateError, match="another read_frame"):
            rig.driver.read_frame(1.0)
        release.set()
        reader.join(5.0)

    def test_a_read_after_stop_is_refused(self) -> None:
        rig = make_rig().streaming(TINY)
        rig.driver.stop()
        with pytest.raises(CameraStateError):
            rig.driver.read_frame(1.0)
        rig.driver.start()
        rig.driver.read_frame(1.0)


class TestWatchdog:
    def test_a_hung_read_is_reported_with_its_name_and_deadline(self) -> None:
        rig = make_rig().streaming(TINY)
        release = rig.sdk.hang_next("get_video_data")
        reader, _ = run_in_thread(lambda: rig.driver.read_frame(1.0))
        assert wait_until(lambda: rig.watchdog.armed == 1)
        rig.clock.advance(10.0)
        reports = rig.watchdog.check()
        assert [(r.name, round(r.timeout_s, 3)) for r in reports] == [
            ("get_video_data", 2.514)
        ]  # 514 ms wait + 2 s margin
        release.set()
        reader.join(5.0)
        assert len(rig.hangs) == 1

    def test_every_sdk_call_runs_under_a_guard(self) -> None:
        rig = make_rig()
        guarded: list[str] = []
        original = rig.watchdog.arm

        def recording(name: str, timeout_s: float) -> int:
            guarded.append(name)
            return original(name, timeout_s)

        rig.watchdog.arm = recording  # type: ignore[method-assign]
        rig.driver.open()
        rig.driver.configure(FAST)
        rig.driver.start()
        rig.driver.read_frame(1.0)
        rig.driver.move_roi(120, 220)
        rig.driver.read_temperature_c()
        rig.driver.dropped_frames()
        for level in RecoveryLevel:
            rig.driver.recover(level)
        rig.driver.stop()
        rig.driver.close()
        called = {name for name, _ in rig.sdk.calls if name != "get_video_data_returned"}
        assert called <= set(guarded)
        assert rig.watchdog.armed == 0

    def test_a_slow_call_that_returns_is_reported_late(self) -> None:
        class SlowOpen:
            def __init__(self, sdk: FakeAsiSdk) -> None:
                self._sdk = sdk

            def __getattr__(self, name: str) -> Any:
                method = getattr(self._sdk, name)

                def call(*args: Any, **kwargs: Any) -> Any:
                    result = method(*args, **kwargs)
                    if name == "init_camera":
                        self._sdk._clock.advance(60.0)  # type: ignore[attr-defined]
                    return result

                return call

        rig = make_rig(api=SlowOpen)
        rig.driver.open()
        assert [report.name for report in rig.hangs] == ["init_camera"]

    def test_an_sdk_error_ends_the_guard(self) -> None:
        rig = make_rig().opened()
        rig.sdk.fail_next("get_roi_format", AsiErrorCode.GENERAL_ERROR)
        with pytest.raises(CameraError):
            rig.driver.configure(FAST)
        assert rig.watchdog.armed == 0
        assert rig.hangs == []
