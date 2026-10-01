"""The per-thread CPU clock, and the way the ipc case turns thread times into figures."""

from __future__ import annotations

import itertools
import threading
import time

import pytest

from seeingmon.perf import _acquire as acquire_helper
from seeingmon.perf.cases.fastmodes import FAST_MODES
from seeingmon.perf.cases.ipc import (
    RunResult,
    burst_figures,
    figures,
    split_work_and_wakeups,
)
from seeingmon.perf.report import Measurement
from seeingmon.perf.threadcpu import thread_cpu_ns, threads_cpu_ns


def burn(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


class TestThreadCpu:
    def test_a_busy_thread_uses_cpu_time_that_another_thread_can_read(self) -> None:
        if thread_cpu_ns(threading.current_thread()) is None:
            pytest.skip("this platform gives no thread CPU time")
        started = threading.Event()
        stop = threading.Event()

        def work() -> None:
            started.set()
            burn(0.4)  # long enough for the 15.6 ms CPU clock of Windows
            stop.wait(5.0)

        worker = threading.Thread(target=work, name="burner")
        worker.start()
        try:
            started.wait(5.0)
            time.sleep(0.6)
            used = thread_cpu_ns(worker)
        finally:
            stop.set()
            worker.join()
        assert used is not None
        assert 0.2e9 < used < 1.5e9

    def test_the_current_thread_agrees_with_the_standard_clock(self) -> None:
        if thread_cpu_ns(threading.current_thread()) is None:
            pytest.skip("this platform gives no thread CPU time")
        before = thread_cpu_ns(threading.current_thread())
        burn(0.3)
        after = thread_cpu_ns(threading.current_thread())
        assert before is not None
        assert after is not None
        assert 0.15e9 < after - before < 1.0e9

    def test_a_thread_that_has_not_started_has_no_reading(self) -> None:
        assert thread_cpu_ns(threading.Thread(target=lambda: None)) is None

    def test_every_live_thread_appears_under_its_name_and_a_repeated_name_gets_its_id(
        self,
    ) -> None:
        if thread_cpu_ns(threading.current_thread()) is None:
            pytest.skip("this platform gives no thread CPU time")
        stop = threading.Event()
        twins = [threading.Thread(target=stop.wait, name="twin") for _ in range(2)]
        for twin in twins:
            twin.start()
        try:
            found = threads_cpu_ns()
        finally:
            stop.set()
            for twin in twins:
                twin.join()
        assert "MainThread" in found
        assert "twin" in found
        assert sum(name.startswith("twin-") for name in found) == 1
        assert all(used >= 0 for used in found.values())


def run_result(
    sent: int = 400,
    acquire_cpu_ms: float = 200.0,
    core_cpu_ms: float = 120.0,
    captured: int | None = None,
    capture_ms: float = 80.0,
    sender_ms: float = 90.0,
) -> RunResult:
    return RunResult(
        received=sent,
        wall_s=4.0,
        core_cpu_ns=round(core_cpu_ms * 1e6),
        acquire={
            "cpu_ns": round(acquire_cpu_ms * 1e6),
            "frames_sent": sent,
            "frames_captured": sent + 3 if captured is None else captured,
            "dropped_queue": 0,
            "dropped_gap": 1,
            "queue_peak_frames": 4,
            "peak_rss_bytes": 50_000_000,
            "threads_cpu_ns": {
                "acquire-capture": round(capture_ms * 1e6),
                "acquire-sender": round(sender_ms * 1e6),
                "acquire-control": 6_000_000,
                "acquire-watchdog": 4_000_000,
            },
        },
    )


class TestRunResult:
    def test_the_cpu_time_per_frame_is_the_process_time_over_the_frames(self) -> None:
        run = run_result()
        assert run.acquire_us == pytest.approx(500.0)  # 200 ms over 400 frames
        assert run.core_us == pytest.approx(300.0)

    def test_the_threads_split_into_capture_sender_and_the_rest(self) -> None:
        split = run_result().thread_us()
        assert split == {"capture_us": 200.0, "sender_us": 225.0, "other_us": 25.0}

    def test_a_time_below_the_clock_resolution_stays_positive(self) -> None:
        assert run_result(acquire_cpu_ms=0.0).acquire_us > 0


class TestFigures:
    def test_each_side_gets_a_cost_per_frame_and_a_share_at_the_rate(self) -> None:
        runs = [run_result(acquire_cpu_ms=200.0), run_result(acquire_cpu_ms=300.0)]
        found = {item.name: item for item in figures("", 98.0, runs, threads=True)}
        assert set(found) == {
            "acquire.cpu_per_frame",
            "acquire.share",
            "core_rx.cpu_per_frame",
            "core_rx.share",
        }
        cost = found["acquire.cpu_per_frame"]
        assert cost.unit == "us/frame"
        assert cost.value == pytest.approx(625.0)  # the median of 500 and 750 microseconds
        assert cost.stats is not None
        assert (cost.stats.min, cost.stats.max) == pytest.approx((500.0, 750.0))
        share = found["acquire.share"]
        assert share.unit == "percent"
        assert share.value == pytest.approx(625.0 * 98.0 / 1e4)
        assert share.detail["share_at_hz"] == 98.0
        assert cost.detail["runs"] == 2
        assert cost.detail["capture_us"] == 200.0

    def test_the_second_mode_has_no_acquire_figure_and_uses_its_own_rate(self) -> None:
        found = {
            item.name: item for item in figures("bin2.", 360.0, [run_result()], with_acquire=False)
        }
        assert set(found) == {"bin2.core_rx.cpu_per_frame", "bin2.core_rx.share"}
        assert found["bin2.core_rx.share"].value == pytest.approx(300.0 * 360.0 / 1e4)

    def test_the_figures_are_in_the_classes_that_the_budgets_scale(self) -> None:
        assert {item.scale for item in figures("", 98.0, [run_result()])} == {"interpreter"}
        assert FAST_MODES[0].shape == (128, 128)


class TestSplit:
    """A paced frame costs work + wake-ups. A frame of a burst of n costs work + wake-ups / n."""

    def test_two_runs_give_the_work_and_the_wake_ups(self) -> None:
        # work 300, wake-ups 700: paced 1,000, and a burst of 10 costs 300 + 70 = 370.
        work, wakeups = split_work_and_wakeups(1000.0, 370.0, 10)
        assert work == pytest.approx(300.0)
        assert wakeups == pytest.approx(700.0)
        assert work + wakeups == pytest.approx(1000.0)

    def test_a_burst_that_costs_as_much_as_the_paced_run_shows_no_wake_ups(self) -> None:
        assert split_work_and_wakeups(500.0, 500.0, 10) == (500.0, 0.0)
        assert split_work_and_wakeups(500.0, 650.0, 10) == (500.0, 0.0)

    def test_a_burst_of_one_frame_cannot_tell_the_parts_apart(self) -> None:
        assert split_work_and_wakeups(500.0, 200.0, 1) == (500.0, 0.0)

    def test_the_work_is_never_negative(self) -> None:
        work, wakeups = split_work_and_wakeups(100.0, 1.0, 2)  # an extreme burst run
        assert work == 0.0
        assert wakeups == 100.0


class TestBurstFigures:
    """The burst runs and the paced runs give each side a work share and a wake-up share."""

    def figures(self) -> dict[str, Measurement]:
        paced = [run_result(acquire_cpu_ms=400.0, core_cpu_ms=240.0)]  # 1,000 and 600 us
        bursts = [run_result(acquire_cpu_ms=148.0, core_cpu_ms=192.0)]  # 370 and 480 us
        return {item.name: item for item in burst_figures(bursts, paced, 98.0, 10)}

    def test_the_work_and_the_wake_ups_of_acquire_come_from_the_two_runs(self) -> None:
        found = self.figures()
        assert found["acquire.compute_share"].value == pytest.approx(300.0 * 98 / 1e4)
        assert found["acquire.wakeup_share"].value == pytest.approx(700.0 * 98 / 1e4)

    def test_the_same_holds_for_the_receive_side_of_core(self) -> None:
        found = self.figures()
        # paced 600, burst 480: wake-ups (600 - 480) * 10 / 9 = 133.3, work 466.7
        assert found["core_rx.compute_share"].value == pytest.approx(466.667 * 98 / 1e4, rel=1e-4)
        assert found["core_rx.wakeup_share"].value == pytest.approx(133.333 * 98 / 1e4, rel=1e-4)

    def test_each_part_is_in_its_own_scale_class(self) -> None:
        found = self.figures()
        assert found["acquire.compute_share"].scale == "interpreter"
        assert found["core_rx.compute_share"].scale == "interpreter"
        assert found["acquire.wakeup_share"].scale == "scheduler"
        assert found["core_rx.wakeup_share"].scale == "scheduler"

    def test_the_burst_costs_are_reported_with_the_numbers_they_came_from(self) -> None:
        found = self.figures()
        assert found["burst.acquire.cpu_per_frame"].value == pytest.approx(370.0)
        share = found["acquire.wakeup_share"]
        assert share.detail["paced_us"] == 1000.0
        assert share.detail["burst_us"] == 370.0
        assert share.detail["burst_frames"] == 10

    def test_a_burst_run_costlier_than_the_paced_run_leaves_a_tiny_positive_wake_up_share(
        self,
    ) -> None:
        paced = [run_result(acquire_cpu_ms=100.0, core_cpu_ms=100.0)]
        bursts = [run_result(acquire_cpu_ms=400.0, core_cpu_ms=400.0)]
        found = {item.name: item for item in burst_figures(bursts, paced, 98.0, 10)}
        assert 0 < found["acquire.wakeup_share"].value < 1e-3
        assert found["acquire.compute_share"].value == pytest.approx(250.0 * 98 / 1e4)


class TestBurstClock:
    """The clock of the acquire helper wakes the capture thread once per burst."""

    def sleeps(self, burst: int, calls: int, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        recorded: list[float] = []
        monkeypatch.setattr("seeingmon.perf._acquire.time.sleep", recorded.append)
        clock = acquire_helper.BurstClock(burst)
        for _ in range(calls):
            clock.sleep(0.01)
        return recorded

    def test_the_first_call_of_each_burst_sleeps_for_the_whole_burst(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self.sleeps(3, 7, monkeypatch) == pytest.approx([0.03, 0.03, 0.03])

    def test_a_burst_of_one_sleeps_for_every_frame(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self.sleeps(1, 4, monkeypatch) == pytest.approx([0.01] * 4)

    def test_a_frame_of_a_burst_is_as_old_as_if_the_camera_had_delivered_it_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("seeingmon.perf._acquire.time.sleep", lambda seconds: None)
        clock = acquire_helper.BurstClock(3)
        shifts = []
        for _ in range(6):
            clock.sleep(0.01)
            shifts.append(clock._shift_ns())
        # The first frame of a burst is two periods old, the next one period, and the last none.
        assert shifts == [20_000_000, 10_000_000, 0, 20_000_000, 10_000_000, 0]

    def test_the_time_stamps_of_a_burst_follow_a_regular_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("seeingmon.perf._acquire.time.sleep", lambda seconds: None)
        clock = acquire_helper.BurstClock(4)
        stamps = []
        for _ in range(4):
            clock.sleep(0.010)
            stamps.append(clock.monotonic_ns())
        gaps = [later - earlier for earlier, later in itertools.pairwise(stamps)]
        # Each gap is a period, plus the few microseconds that the loop takes.
        assert all(9_000_000 < gap < 11_000_000 for gap in gaps)

    def test_a_burst_has_at_least_one_frame(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            acquire_helper.BurstClock(0)
