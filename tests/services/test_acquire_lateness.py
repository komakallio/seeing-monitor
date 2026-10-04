"""Late reads at the capture thread: what the drop accounting makes of them.

The tests hand frames to an `AcquireService` that nobody started, at times that they choose, and
read the frames that reach the queue. The service sees the same calls that its capture thread makes.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import replace
from typing import Any, ClassVar

import pytest

from seeingmon.clock import DEFAULT_START_UTC_NS, NS_PER_S, VirtualClock
from seeingmon.frames import ActiveStream, Frame
from seeingmon.services.acquire import service as service_module
from seeingmon.services.acquire.service import AcquireService, timing_config
from seeingmon.services.acquire.timing import TimeStamper
from seeingmon.services.config import AcquireSettings, ServicesConfig
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.testing import FakeCameraDriver

from .rig import FAST


class Bench:
    """An `AcquireService` with a stream that it believes runs, and frames that arrive on demand."""

    def __init__(self, **acquire: object) -> None:
        self.clock = VirtualClock(start_utc_ns=DEFAULT_START_UTC_NS)
        driver = FakeCameraDriver(self.clock)
        options: dict[str, object] = {"raise_priority": False, "time_source": "stamp", **acquire}
        settings = ServicesConfig(acquire=AcquireSettings(**options))
        self.service = AcquireService(
            driver,
            self.clock,
            Endpoint.loopback(0),
            ConnectionKey.from_text(secrets.token_urlsafe(24)),
            settings,
            priority_hook=lambda: "test",
        )
        driver.open()
        self.active: ActiveStream = driver.configure(FAST)
        service = self.service
        service._stamper.configure(service._stream_timing(self.active))
        service._drops.reset(self.active.frame_period_s)
        service._timing_stream_id = self.active.stream_id
        service._active = self.active
        service._capturing = True
        self.epoch = service._new_epoch()
        driver.start()
        self.template = driver.read_frame(1.0)
        assert self.active.frame_period_s
        self.period_ns = round(self.active.frame_period_s * 1e9)
        self.frames: list[Frame] = []
        original = service._queue.put_frame

        def record(frame: Frame, epoch: int) -> int:
            self.frames.append(frame)
            return original(frame, epoch)

        service._queue.put_frame = record  # type: ignore[method-assign]
        self.at_ns = self.clock.monotonic_ns()
        self.seq = 0

    def arrive(self, after_periods: float, *, counted: int = 0) -> None:
        """A frame comes `after_periods` frame periods after the previous one.

        `counted` is what the driver reports in `dropped_before`, as the SDK counter would.
        """
        delta_ns = round(after_periods * self.period_ns)
        self.at_ns += delta_ns
        self.clock.advance_ns(delta_ns)  # the service reads the same clock for its health
        frame = replace(self.template, seq=self.seq, t_arrival_ns=0, dropped_before=counted)
        self.seq += 1
        self.service._on_frame(
            frame, DEFAULT_START_UTC_NS + self.at_ns, self.at_ns, self.active, self.epoch
        )
        self.service._queue.clear()  # nobody reads it, and a full queue would count as lost frames

    @property
    def lost(self) -> list[int]:
        """The `dropped_before` of each frame that reached the queue."""
        return [frame.dropped_before for frame in self.frames]


def test_the_gap_rule_ignores_the_period_of_the_time_fit() -> None:
    bench = Bench()

    class ShrunkenFit(TimeStamper):
        """What the fit reads after phantom drops have counted into its frame numbers."""

        @property
        def period_s(self) -> float | None:
            return 0.001

    shrunken = ShrunkenFit(bench.clock, timing_config(bench.service._cfg))
    shrunken.configure(bench.service._stream_timing(bench.active))
    bench.service._stamper = shrunken
    for _ in range(60):
        bench.arrive(1.0)
    assert bench.lost == [0] * 60
    assert bench.service.health().dropped_gap == 0


def test_late_reads_followed_by_catch_ups_lose_no_frame_and_leave_the_fit_alone() -> None:
    bench = Bench()
    bench.arrive(1.0)
    for _ in range(100):  # three frames in three periods: one read that is 1.6 periods late
        for step in (2.6, 0.2, 0.2):
            bench.arrive(step)
    assert bench.lost == [0] * 301
    health = bench.service.health()
    assert (health.dropped_driver, health.dropped_gap, health.dropped_queue) == (0, 0, 0)
    assert health.late_reads == 100
    # The fit counts frames as they came, so its period stays the period of the camera.
    assert bench.service._stamper.period_s == pytest.approx(bench.period_ns / 1e9, rel=0.02)
    assert health.time_resets == 0


def test_a_real_gap_reaches_the_frame_after_it_and_the_fit_follows() -> None:
    bench = Bench()
    for _ in range(40):
        bench.arrive(1.0)
    bench.arrive(3.0)  # two frames never came
    for _ in range(40):
        bench.arrive(1.0)
    assert bench.lost == [0] * 41 + [2] + [0] * 39
    health = bench.service.health()
    assert (health.dropped_driver, health.dropped_gap) == (0, 2)
    assert health.time_resets == 0


def test_a_loss_that_the_driver_counted_reaches_the_frame_at_once() -> None:
    bench = Bench()
    for _ in range(40):
        bench.arrive(1.0)
    bench.arrive(2.0, counted=1)  # one frame lost, and the SDK counted it
    bench.arrive(1.0)
    assert bench.lost == [0] * 40 + [1, 0]
    assert bench.service.health().dropped_driver == 1


def run_pattern(bench: Bench, seconds: float, pattern: tuple[float, ...]) -> None:
    """Feed frames at the intervals of the pattern, over and over, for `seconds` of the clock.

    The watchdog thread notes the counters each tick, so the test notes them after each frame.
    """
    end_ns = bench.at_ns + round(seconds * NS_PER_S)
    while bench.at_ns < end_ns:
        for step in pattern:
            bench.arrive(step)
            bench.service._note_recent()


class TestTheLastMinute:
    def short_minute(self) -> Bench:
        bench = Bench()
        bench.service._recent_ns = 10 * NS_PER_S  # ten seconds stand for the minute
        return bench

    def test_the_figures_follow_the_stream_and_forget_what_is_older_than_the_minute(self) -> None:
        bench = self.short_minute()
        run_pattern(bench, 12.0, (1.0,))
        clean = bench.service.health()
        assert (clean.recent_lost, clean.recent_late) == (0, 0)
        assert clean.recent_lost_percent == 0.0
        assert 10.0 <= clean.recent_s <= 11.1
        assert clean.recent_frames == pytest.approx(10.5 * NS_PER_S / bench.period_ns, rel=0.1)

        run_pattern(bench, 12.0, (*[1.0] * 8, 3.0, 1.0))  # two frames lost in twelve periods
        lossy = bench.service.health()
        assert lossy.recent_lost_percent == pytest.approx(100 * 2 / 12, abs=1.5)
        assert lossy.recent_late == 0

        run_pattern(bench, 12.0, (1.0,))  # the loss is a minute old now
        assert bench.service.health().recent_lost == 0

        run_pattern(bench, 12.0, (2.6, 0.2, 0.2))  # one late read in three frames
        late = bench.service.health()
        assert late.recent_lost == 0
        assert late.recent_late_percent == pytest.approx(100 / 3, abs=2.0)

    def test_the_totals_keep_counting_when_the_minute_moves_on(self) -> None:
        bench = self.short_minute()
        run_pattern(bench, 12.0, (*[1.0] * 8, 3.0, 1.0))
        run_pattern(bench, 12.0, (1.0,))
        health = bench.service.health()
        assert health.recent_lost == 0
        assert health.dropped_gap > 150  # the first part of the run lost 2 frames in 12 periods

    def test_a_stream_that_has_not_run_has_no_figures(self) -> None:
        health = Bench().service.health()
        assert (health.recent_frames, health.recent_lost, health.recent_late) == (0, 0, 0)
        assert health.recent_lost_percent is None
        assert health.recent_late_percent is None
        assert "in the last" not in health.summary()

    def test_the_notes_stay_few(self) -> None:
        bench = Bench()
        run_pattern(bench, 200.0, (1.0,))
        assert len(bench.service._recent) <= 62  # one a second for a minute, and the one before


class TestTheSummaryLine:
    def health(self, **changes: Any) -> Any:
        return replace(Bench().service.health(), **changes)

    def test_it_names_the_sources_the_last_minute_and_the_platform_calls(self) -> None:
        health = self.health(
            state="streaming",
            capturing=True,
            frame_rate_hz=82.1,
            frames_captured=9413,
            dropped_gap=1101,
            priority="highest thread priority",
            timer="1 ms resolution",
            recent_s=60.2,
            recent_frames=4939,
            recent_lost=598,
            recent_late=300,
        )
        assert health.summary() == (
            "streaming, 82.1 fps, 9413 frames, 1101 dropped (driver 0, gap 1101, queue 0), "
            "10.8% lost and 6.1% late in the last minute, priority: highest thread priority, "
            "timer: 1 ms resolution"
        )

    def test_a_span_of_less_than_a_minute_says_how_long_it_is(self) -> None:
        health = self.health(recent_s=23.4, recent_frames=100, recent_lost=0, recent_late=0)
        assert "0.0% lost and 0.0% late in the last 23 s" in health.summary()

    def test_a_platform_without_a_timer_request_says_nothing_about_it(self) -> None:
        summary = self.health(priority="nice -10", timer="").summary()
        assert "priority: nice -10" in summary
        assert "timer" not in summary

    def test_the_last_error_comes_last(self) -> None:
        summary = self.health(last_error="boom", priority="disabled").summary()
        assert summary.endswith("priority: disabled, last error: boom")

    def test_the_json_carries_the_figures_and_the_two_shares(self) -> None:
        health = self.health(recent_s=60.0, recent_frames=900, recent_lost=100, recent_late=90)
        data = json.loads(json.dumps(health.to_json()))
        assert data["recent_lost_percent"] == pytest.approx(10.0)
        assert data["recent_late_percent"] == pytest.approx(10.0)
        assert (data["recent_s"], data["late_reads"], data["timer"]) == (60.0, 0, "")
        assert self.health().to_json()["recent_lost_percent"] is None


class RecordingTimer:
    """Stands in for `TimerResolution`, and notes what the service asks of it."""

    made: ClassVar[list[RecordingTimer]] = []

    def __init__(self) -> None:
        self.calls: list[str] = []
        RecordingTimer.made.append(self)

    def request(self) -> str:
        self.calls.append("request")
        return "1 ms resolution"

    def release(self) -> None:
        self.calls.append("release")


class TestTheTimerRequest:
    @pytest.fixture(autouse=True)
    def recording(self, monkeypatch: pytest.MonkeyPatch) -> None:
        RecordingTimer.made = []
        monkeypatch.setattr(service_module, "TimerResolution", RecordingTimer)

    def service(self, *, raise_priority: bool, **options: Any) -> AcquireService:
        clock = VirtualClock(start_utc_ns=DEFAULT_START_UTC_NS)
        settings = ServicesConfig(acquire=AcquireSettings(raise_priority=raise_priority))
        return AcquireService(
            FakeCameraDriver(clock),
            clock,
            Endpoint.loopback(0),
            ConnectionKey.from_text(secrets.token_urlsafe(24)),
            settings,
            **options,
        )

    def test_the_service_asks_at_the_start_and_gives_back_at_the_stop(self) -> None:
        service = self.service(raise_priority=True)
        (timer,) = RecordingTimer.made
        assert timer.calls == []
        service.start()
        try:
            assert timer.calls == ["request"]
            assert service.health().timer == "1 ms resolution"
        finally:
            service.stop()
        assert timer.calls == ["request", "release"]
        service.stop()  # a second stop gives back nothing more
        assert timer.calls == ["request", "release"]

    def test_a_service_that_does_not_raise_the_priority_leaves_the_timer_alone(self) -> None:
        service = self.service(raise_priority=False)
        assert RecordingTimer.made == []
        service.start()
        try:
            assert service.health().timer == ""
        finally:
            service.stop()

    def test_a_test_hook_for_the_priority_stands_for_the_platform_calls_and_the_timer(
        self,
    ) -> None:
        self.service(raise_priority=True, priority_hook=lambda: "test")
        assert RecordingTimer.made == []

    def test_a_timer_request_that_raises_does_not_stop_the_service(self) -> None:
        class BrokenTimer(RecordingTimer):
            def request(self) -> str:
                raise RuntimeError("no such library")

        service = self.service(raise_priority=False, timer=BrokenTimer())
        service.start()
        try:
            assert service.health().timer == (
                "the request failed, so the timer keeps its default resolution"
            )
            assert service.health().threads_alive
        finally:
            service.stop()

    def test_a_timer_that_the_caller_passes_is_the_one_that_runs(self) -> None:
        mine = RecordingTimer()
        service = self.service(raise_priority=False, timer=mine)
        service.start()
        service.stop()
        assert mine.calls == ["request", "release"]
