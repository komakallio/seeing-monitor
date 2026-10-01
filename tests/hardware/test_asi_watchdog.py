"""The call watchdog: deadlines, hang reports, the polling thread, and the exit handler."""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path

import pytest

from seeingmon.clock import SystemClock, VirtualClock
from seeingmon.hardware.asi.watchdog import (
    HANG_EXIT_CODE,
    CallWatchdog,
    HangReport,
    dump_thread_stacks,
    make_exit_handler,
)


class Recorder:
    def __init__(self) -> None:
        self.reports: list[HangReport] = []
        self.called = threading.Event()

    def __call__(self, report: HangReport) -> None:
        self.reports.append(report)
        self.called.set()


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock()


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def watchdog(clock: VirtualClock, recorder: Recorder) -> CallWatchdog:
    return CallWatchdog(clock, recorder)


class TestDeadlines:
    def test_a_call_inside_its_deadline_reports_nothing(
        self, watchdog: CallWatchdog, clock: VirtualClock, recorder: Recorder
    ) -> None:
        with watchdog.guard("get_video_data", 0.5):
            clock.advance(0.5)  # the deadline itself is not an overrun
            assert watchdog.check() == []
        assert recorder.reports == []
        assert watchdog.hang_count == 0

    def test_a_call_past_its_deadline_reports_once(
        self, watchdog: CallWatchdog, clock: VirtualClock, recorder: Recorder
    ) -> None:
        token = watchdog.arm("get_video_data", 0.5)
        clock.advance(0.8)
        first = watchdog.check()
        assert [(r.name, r.timeout_s, round(r.elapsed_s, 3)) for r in first] == [
            ("get_video_data", 0.5, 0.8)
        ]
        clock.advance(5.0)
        assert watchdog.check() == []  # a call that stays blocked reports once
        watchdog.disarm(token)
        assert len(recorder.reports) == 1
        assert watchdog.hang_count == 1

    def test_a_call_that_returns_late_reports_when_the_guard_ends(
        self, watchdog: CallWatchdog, clock: VirtualClock, recorder: Recorder
    ) -> None:
        with watchdog.guard("stop_video_capture", 1.0):
            clock.advance(2.5)  # nothing polled meanwhile
        assert [r.name for r in recorder.reports] == ["stop_video_capture"]
        assert watchdog.armed == 0

    def test_a_report_made_by_check_is_not_repeated_when_the_guard_ends(
        self, watchdog: CallWatchdog, clock: VirtualClock, recorder: Recorder
    ) -> None:
        with watchdog.guard("open_camera", 1.0):
            clock.advance(2.0)
            watchdog.check()
        assert len(recorder.reports) == 1

    def test_calls_on_several_threads_report_separately(
        self, watchdog: CallWatchdog, clock: VirtualClock, recorder: Recorder
    ) -> None:
        reader = watchdog.arm("get_video_data", 0.5)
        mover = watchdog.arm("set_start_position", 5.0)
        assert watchdog.armed == 2
        clock.advance(1.0)
        assert [r.name for r in watchdog.check()] == ["get_video_data"]
        clock.advance(10.0)
        assert [r.name for r in watchdog.check()] == ["set_start_position"]
        watchdog.disarm(reader)
        watchdog.disarm(mover)
        assert sorted(r.name for r in recorder.reports) == ["get_video_data", "set_start_position"]

    def test_a_guard_ends_when_the_call_raises(
        self, watchdog: CallWatchdog, recorder: Recorder
    ) -> None:
        with pytest.raises(RuntimeError), watchdog.guard("init_camera", 1.0):
            raise RuntimeError("the call failed")
        assert watchdog.armed == 0
        assert recorder.reports == []

    def test_a_failing_handler_is_logged_and_does_not_disturb_the_call(
        self, clock: VirtualClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken(_: HangReport) -> None:
            raise RuntimeError("the handler failed")

        watchdog = CallWatchdog(clock, broken)
        with caplog.at_level(logging.ERROR):
            with watchdog.guard("get_video_data", 0.1):
                clock.advance(1.0)
            assert watchdog.check() == []
        assert "get_video_data" in caplog.text

    def test_a_negative_timeout_is_refused(self, watchdog: CallWatchdog) -> None:
        with pytest.raises(ValueError, match="negative"):
            watchdog.arm("get_video_data", -1.0)


class TestThread:
    def test_a_virtual_clock_cannot_drive_the_thread(self, watchdog: CallWatchdog) -> None:
        with pytest.raises(ValueError, match="virtual clock"):
            watchdog.start()

    def test_the_thread_reports_a_hung_call(self, recorder: Recorder) -> None:
        watchdog = CallWatchdog(SystemClock(), recorder, poll_interval_s=0.01)
        watchdog.start()
        watchdog.start()  # a second start does nothing
        try:
            with watchdog.guard("get_video_data", 0.05):
                assert recorder.called.wait(10.0)
        finally:
            watchdog.stop()
        assert [r.name for r in recorder.reports] == ["get_video_data"]
        assert recorder.reports[0].elapsed_s >= 0.05

    def test_the_thread_leaves_a_call_in_time_alone(self, recorder: Recorder) -> None:
        watchdog = CallWatchdog(SystemClock(), recorder, poll_interval_s=0.01)
        watchdog.start()
        try:
            with watchdog.guard("get_video_data", 60.0):
                recorder.called.wait(0.1)
        finally:
            watchdog.stop()
        assert recorder.reports == []

    def test_stop_without_start_is_harmless(self, watchdog: CallWatchdog) -> None:
        watchdog.stop()


class TestExitHandler:
    report = HangReport("get_video_data", 0.5, 3.2)

    def test_the_handler_writes_the_stacks_and_ends_the_process(self) -> None:
        exits: list[int] = []
        stream = io.StringIO()
        dumps: list[object] = []
        handler = make_exit_handler(
            exit_process=exits.append, dump_stacks=dumps.append, stream=stream
        )
        handler(self.report)
        assert exits == [HANG_EXIT_CODE]
        assert dumps == [stream]
        assert "get_video_data" in stream.getvalue()
        assert "3.2 s" in stream.getvalue()

    def test_a_failing_stack_dump_does_not_stop_the_exit(self) -> None:
        exits: list[int] = []

        def broken(_: object) -> None:
            raise OSError("no file descriptor")

        handler = make_exit_handler(
            exit_process=exits.append, dump_stacks=broken, stream=io.StringIO()
        )
        handler(self.report)
        assert exits == [HANG_EXIT_CODE]

    def test_the_stack_dump_names_the_threads_of_a_stream_without_a_descriptor(self) -> None:
        stream = io.StringIO()
        dump_thread_stacks(stream)
        assert "MainThread" in stream.getvalue() or "Thread" in stream.getvalue()
        assert "test_the_stack_dump_names_the_threads" in stream.getvalue()

    def test_the_stack_dump_uses_faulthandler_for_a_real_file(self, tmp_path: Path) -> None:
        with (tmp_path / "stacks.txt").open("w+") as handle:
            dump_thread_stacks(handle)
            handle.flush()
            handle.seek(0)
            text = handle.read()
        assert "test_the_stack_dump_uses_faulthandler" in text
