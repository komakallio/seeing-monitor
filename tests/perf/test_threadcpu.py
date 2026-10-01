"""The per-thread CPU clock, and the way the ipc case turns thread times into figures."""

from __future__ import annotations

import threading
import time

import pytest

from seeingmon.perf.cases.fastmodes import FAST_MODES
from seeingmon.perf.cases.ipc import RunResult, figures
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
    sent: int = 400, acquire_cpu_ms: float = 200.0, core_cpu_ms: float = 120.0
) -> RunResult:
    return RunResult(
        received=sent,
        wall_s=4.0,
        core_cpu_ns=round(core_cpu_ms * 1e6),
        acquire={
            "cpu_ns": round(acquire_cpu_ms * 1e6),
            "frames_sent": sent,
            "frames_captured": sent + 3,
            "dropped_queue": 0,
            "dropped_gap": 1,
            "queue_peak_frames": 4,
            "peak_rss_bytes": 50_000_000,
            "threads_cpu_ns": {
                "acquire-capture": 80_000_000,
                "acquire-sender": 90_000_000,
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
