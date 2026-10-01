"""The per-thread CPU clock, and the way the ipc case turns thread times into figures."""

from __future__ import annotations

import threading
import time

import pytest

from seeingmon.perf.cases.fastmodes import FAST_MODES
from seeingmon.perf.cases.ipc import RunResult, figures, saturated_figures
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


class TestSaturatedFigures:
    """The saturated run gives the work without wake-ups, and the nominal run the rest."""

    def figures(self, **saturated: float) -> dict[str, Measurement]:
        nominal = [run_result(acquire_cpu_ms=400.0, core_cpu_ms=240.0)]  # 1,000 and 600 us
        run = run_result(sent=1000, captured=3000, **saturated)
        return {item.name: item for item in saturated_figures(run, nominal, 98.0)}

    def test_the_work_is_the_thread_time_per_frame_that_each_thread_handled(self) -> None:
        found = self.figures(capture_ms=300.0, sender_ms=150.0, acquire_cpu_ms=500.0)
        # The capture thread took 300 ms for 3,000 captured frames (100 us each). The sender took
        # 150 ms and the control and watchdog threads 10 ms for 1,000 sent frames (160 us each).
        assert found["saturated.acquire.compute_us"].value == pytest.approx(260.0)
        assert found["saturated.core_rx.compute_us"].value == pytest.approx(
            120.0
        )  # 120 ms over 1,000
        assert found["saturated.acquire.compute_us"].detail["frames_captured"] == 3000

    def test_the_wake_ups_are_what_the_paced_run_costs_beyond_the_work(self) -> None:
        found = self.figures(capture_ms=300.0, sender_ms=150.0, acquire_cpu_ms=500.0)
        assert found["acquire.compute_share"].value == pytest.approx(260.0 * 98 / 1e4)
        assert found["acquire.wakeup_share"].value == pytest.approx((1000.0 - 260.0) * 98 / 1e4)
        assert found["core_rx.wakeup_share"].value == pytest.approx((600.0 - 120.0) * 98 / 1e4)

    def test_each_part_is_in_its_own_scale_class(self) -> None:
        found = self.figures()
        assert found["acquire.compute_share"].scale == "interpreter"
        assert found["core_rx.compute_share"].scale == "interpreter"
        assert found["acquire.wakeup_share"].scale == "scheduler"
        assert found["core_rx.wakeup_share"].scale == "scheduler"

    def test_a_paced_run_cheaper_than_the_saturated_one_leaves_a_tiny_positive_wake_up_share(
        self,
    ) -> None:
        nominal = [run_result(acquire_cpu_ms=10.0, core_cpu_ms=10.0)]  # 25 us per frame
        run = run_result(sent=1000, captured=1000, capture_ms=500.0, sender_ms=500.0)
        found = {item.name: item for item in saturated_figures(run, nominal, 98.0)}
        assert 0 < found["acquire.wakeup_share"].value < 1e-3

    def test_without_thread_clocks_the_whole_process_is_the_upper_bound(self) -> None:
        run = run_result(sent=1000, captured=1000, acquire_cpu_ms=300.0)
        run.acquire["threads_cpu_ns"] = {}
        found = {item.name: item for item in saturated_figures(run, [run_result()], 98.0)}
        assert found["saturated.acquire.compute_us"].value == pytest.approx(300.0)
        assert "no thread clocks" in str(found["saturated.acquire.compute_us"].detail["basis"])
