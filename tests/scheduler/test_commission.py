"""The commissioning queue, the frame statistics, and the sweep, apart from the scheduler."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from seeingmon.clock import Clock, VirtualClock
from seeingmon.drivers.base import CameraConfigError
from seeingmon.frames import ActiveStream, Frame, Roi, StreamConfig
from seeingmon.profile import Profile, load_profile
from seeingmon.records import SeeingWindowRecord
from seeingmon.scheduler.commands import Command, QueueBurst, QueueSweep
from seeingmon.scheduler.commission import (
    CommissionContext,
    CommissionHandler,
    CommissionResult,
    CommissionTask,
    FastWindowSample,
    FrameStatsAccumulator,
    FrameStatsSummary,
    SweepCell,
    SweepCellResult,
    SweepHandler,
    SweepPlan,
    TaskQueue,
    border_pixels,
    format_sweep_table,
    summarize_cell,
)
from seeingmon.scheduler.config import SchedulerConfig, SweepConfig
from tests.scheduler.helpers import make_frame

PROFILE = load_profile("asi294mm-gs250")
SATURATION = 65_520.0  # the bin1 saturation level in 16-bit counts


def task(task_id: int, priority: int = 0, command: Command | None = None) -> CommissionTask:
    return CommissionTask(
        task_id=task_id,
        kind="sweep",
        command=command or QueueSweep(),
        submitted_utc_ns=task_id,
        priority=priority,
    )


class TestTaskQueue:
    def test_tasks_run_by_priority_and_then_in_order_of_arrival(self) -> None:
        queue = TaskQueue(max_queued=10)
        for task_id, priority in [(1, 0), (2, 5), (3, 0), (4, 5), (5, -1)]:
            assert queue.push(task(task_id, priority))
        assert [t.task_id for t in queue.tasks()] == [2, 4, 1, 3, 5]
        popped = []
        while (next_task := queue.pop()) is not None:
            popped.append(next_task.task_id)
        assert popped == [2, 4, 1, 3, 5]
        assert len(queue) == 0

    def test_a_full_queue_refuses_a_task(self) -> None:
        queue = TaskQueue(max_queued=2)
        results = [(queue.push(task(task_id)), queue.full, len(queue)) for task_id in (1, 2, 3)]
        assert results == [(True, False, 1), (True, True, 2), (False, True, 2)]
        popped = queue.pop()
        assert popped is not None
        assert popped.task_id == 1
        assert queue.push(task(3))  # room again after a pop


class TestFrameStats:
    @staticmethod
    def noisy_frame(rng: np.random.Generator, *, star: int, sigma: float = 10.0) -> Frame:
        data = rng.normal(1000.0, sigma, size=(64, 64)).round().clip(0, 65535).astype(np.uint16)
        data[32, 32] = 1000 + star
        return make_frame(data, mode="bin1", gain=0, adc_bits=12)

    def test_the_border_holds_only_the_ring(self) -> None:
        data = np.zeros((64, 64), dtype=np.uint16)
        data[8:-8, 8:-8] = 999  # everything inside the ring of width 4
        border = border_pixels(data)
        assert border.max() == 0.0  # no inner pixel leaks into the background estimate
        assert border.size == 64 * 64 - 56 * 56

    def test_a_tiny_frame_uses_every_pixel_as_border(self) -> None:
        assert border_pixels(np.ones((2, 2), dtype=np.uint8)).size == 4
        assert border_pixels(np.ones((3, 3), dtype=np.uint8)).size == 8  # the ring, not the middle

    def test_the_snr_is_the_peak_over_the_background_over_the_noise_of_the_border(self) -> None:
        rng = np.random.default_rng(5)
        stats = FrameStatsAccumulator()
        for _ in range(40):
            stats.add(self.noisy_frame(rng, star=500), SATURATION, star_found=True)
        summary = stats.summary()
        # The peak is 500 counts above the background, and the noise is 10 counts per pixel, so
        # the ratio is 50. The estimate of the noise from 4 percent of the pixels is good to 10%.
        assert summary.snr_median == pytest.approx(50.0, rel=0.1)
        assert summary.n_frames == 40
        assert summary.star_found_fraction == 1.0
        assert summary.background_fraction == pytest.approx(1000 / SATURATION, rel=0.01)
        assert summary.peak_fraction_max == pytest.approx(1500 / SATURATION, rel=0.01)

    def test_saturated_frames_are_counted_at_98_percent_of_the_level(self) -> None:
        stats = FrameStatsAccumulator()
        for peak in (60_000, 64_300, 64_400, 65_520):  # 98% of 65,520 is 64,209.6
            data = np.full((32, 32), 1000, dtype=np.uint16)
            data[5, 5] = peak
            stats.add(make_frame(data, mode="bin1", gain=0), SATURATION, star_found=False)
        summary = stats.summary()
        assert summary.saturated_fraction == 0.75
        assert summary.star_found_fraction == 0.0

    def test_a_flat_frame_has_no_snr(self) -> None:
        """A background without noise gives no ratio, and the summary says `None`."""
        stats = FrameStatsAccumulator()
        stats.add(make_frame(np.full((32, 32), 100, dtype=np.uint16)), SATURATION, star_found=False)
        assert stats.summary().snr_median is None

    def test_no_frames_give_an_empty_summary(self) -> None:
        assert FrameStatsAccumulator().summary() == FrameStatsSummary(
            0, None, None, None, None, None, None
        )


class TestSweepPlan:
    def plan(self, command: QueueSweep | None = None, **sweep: Any) -> SweepPlan:
        return SweepPlan.resolve(command or QueueSweep(), SweepConfig(**sweep), PROFILE)

    def test_the_default_grid_comes_from_the_configuration(self) -> None:
        plan = self.plan()
        config = SweepConfig()
        assert len(plan.cells) == len(config.exposure_us) * len(config.gain) * len(
            config.roi_arcmin
        )
        assert plan.window_s == config.window_s
        assert {cell.mode for cell in plan.cells} == {PROFILE.fast_mode.mode}

    def test_a_command_overrides_each_axis_and_the_window(self) -> None:
        command = QueueSweep(
            exposure_us=(1000, 2000),
            gain=(0, 120),
            roi_arcmin=(2.0,),
            modes=("bin1", "bin2"),
            window_s=4.0,
        )
        plan = self.plan(command)
        assert plan.window_s == 4.0
        assert len(plan.cells) == 2 * 2 * 1 * 2
        # The exposure changes fastest, so a run steps through exposures at one gain first.
        assert [c.exposure_us for c in plan.cells[:4]] == [1000, 2000, 1000, 2000]
        assert [c.gain for c in plan.cells[:4]] == [0, 0, 120, 120]
        assert plan.cells[0] == SweepCell("bin1", 1000, 0, 2.0)
        assert plan.cells[-1] == SweepCell("bin2", 2000, 120, 2.0)

    @pytest.mark.parametrize(
        ("command", "message"),
        [
            (QueueSweep(exposure_us=(1,)), "exposure 1 us"),
            (QueueSweep(exposure_us=(3_000_000_000,)), "exposure"),
            (QueueSweep(gain=(600,)), "gain 600"),
            (QueueSweep(roi_arcmin=(0.0,)), "positive angle"),
            (QueueSweep(roi_arcmin=(float("nan"),)), "positive angle"),
            (QueueSweep(modes=("bin9",)), "unknown readout mode"),
            (QueueSweep(window_s=0.0), "window_s"),
            (QueueSweep(window_s=float("inf")), "window_s"),
        ],
    )
    def test_a_value_outside_the_profile_is_an_error_that_says_which(
        self, command: QueueSweep, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            self.plan(command)

    def test_a_grid_larger_than_the_limit_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="3 cells, and the limit is 2"):
            self.plan(QueueSweep(exposure_us=(1000, 2000, 3000), gain=(0,)), max_cells=2)


def window(**fields: Any) -> SeeingWindowRecord:
    base: dict[str, Any] = {
        "station_id": "test",
        "t_utc_ns": 1_000,
        "profile_id": "test",
        "provenance": {"algo": "test"},
        "duration_s": 5.0,
        "stream_id": 1,
        "readout_mode": "bin1",
        "exposure_us": 2000,
        "gain": 0,
        "n_frames": 400,
        "n_dropped": 0,
        "valid_fraction": 1.0,
    }
    return SeeingWindowRecord(**{**base, **fields})


def sample(
    *,
    windows: tuple[SeeingWindowRecord, ...] = (),
    n_frames: int = 400,
    n_dropped: int = 0,
    duration_s: float = 4.0,
) -> FastWindowSample:
    return FastWindowSample(
        config=StreamConfig(mode="bin1", exposure_us=2000, gain=0, roi=Roi(10, 10, 128, 128)),
        stream_id=3,
        duration_s=duration_s,
        n_frames=n_frames,
        n_dropped=n_dropped,
        windows=windows,
        stats=FrameStatsSummary(n_frames, 0.25, 0.4, 0.9, 0.02, 33.0, 1.0),
    )


class TestCellSummary:
    cell = SweepCell("bin1", 2000, 0, 4.1)

    def test_the_cell_combines_the_frame_statistics_and_the_window_records(self) -> None:
        windows = (
            window(
                centroid_noise_px=0.02, image_motion_rms_x_arcsec=0.4, image_motion_rms_y_arcsec=0.5
            ),
            window(centroid_noise_px=0.04, image_motion_rms_x_arcsec=0.6, seeing_fwhm_arcsec=1.2),
        )
        result = summarize_cell(self.cell, sample(windows=windows, n_frames=400, n_dropped=100))
        assert result.status == "ok"
        assert result.roi_px == (128, 128)
        assert result.frame_rate_hz == pytest.approx(100.0)  # 400 frames in 4 s
        assert result.drop_rate == pytest.approx(0.2)  # 100 of 500
        assert result.saturated_fraction == 0.25
        assert result.snr_median == 33.0
        assert result.n_windows == 2
        assert result.centroid_noise_px == pytest.approx(0.03)
        assert result.image_motion_rms_arcsec == pytest.approx((0.4 + 0.5 + 0.6) / 3)
        assert result.seeing_fwhm_arcsec == pytest.approx(1.2)

    def test_a_window_that_reports_no_estimate_leaves_the_value_empty(self) -> None:
        result = summarize_cell(self.cell, sample(windows=(window(),)))
        assert result.centroid_noise_px is None
        assert result.image_motion_rms_arcsec is None
        assert result.seeing_fwhm_arcsec is None

    def test_a_cell_without_frames_has_no_rates(self) -> None:
        result = summarize_cell(self.cell, sample(n_frames=0, n_dropped=0, duration_s=0.0))
        assert result.frame_rate_hz is None
        assert result.drop_rate is None


class FakeContext:
    """Just enough of a `CommissionContext` to run the sweep handler on canned answers."""

    def __init__(
        self, *, no_solution_for: tuple[str, ...] = (), stop_after: int | None = None
    ) -> None:
        self._clock = VirtualClock()
        self._no_solution_for = no_solution_for
        self._stop_after = stop_after
        self.windows_run: list[tuple[StreamConfig, float]] = []
        self.events: list[tuple[str, str]] = []
        self.rejected_gain: int | None = None

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def profile(self) -> Profile:
        return PROFILE

    @property
    def config(self) -> SchedulerConfig:
        return SchedulerConfig()

    def should_stop(self) -> bool:
        return self._stop_after is not None and len(self.windows_run) >= self._stop_after

    def emit_event(
        self, level: str, kind: str, message: str, detail: Mapping[str, Any] | None = None
    ) -> None:
        self.events.append((level, kind))

    def fast_stream_config(
        self,
        *,
        mode: str | None = None,
        exposure_us: int | None = None,
        gain: int | None = None,
        roi_arcmin: float | None = None,
    ) -> StreamConfig | None:
        if mode in self._no_solution_for:
            return None
        return StreamConfig(
            mode=mode or "bin1",
            exposure_us=exposure_us or 2000,
            gain=gain or 0,
            roi=Roi(0, 0, 8, 8),
        )

    def configure(self, config: StreamConfig) -> ActiveStream:
        raise AssertionError("the sweep runs its windows through run_fast_window")

    def start(self) -> None:
        raise AssertionError("the sweep does not start a stream itself")

    def read_frame(self, timeout_s: float | None = None) -> Frame:
        raise AssertionError("the sweep does not read frames itself")

    def stop(self) -> None:
        raise AssertionError("the sweep does not stop a stream itself")

    def run_fast_window(self, config: StreamConfig, duration_s: float) -> FastWindowSample:
        self.windows_run.append((config, duration_s))
        if config.gain == self.rejected_gain:
            raise CameraConfigError("gain not supported")
        return sample(n_frames=int(duration_s * 100), duration_s=duration_s)


class TestSweepHandler:
    def run(self, command: QueueSweep, context: FakeContext) -> CommissionResult:
        return SweepHandler().run(task(7, command=command), context)

    def test_the_handler_and_the_context_satisfy_their_protocols(self) -> None:
        assert isinstance(SweepHandler(), CommissionHandler)
        assert isinstance(FakeContext(), CommissionContext)

    def test_it_runs_one_window_for_each_cell_and_reports_a_pinned_result(self) -> None:
        context = FakeContext()
        command = QueueSweep(
            exposure_us=(1000, 2000), gain=(0, 60), roi_arcmin=(4.1,), window_s=3.0
        )
        result = self.run(command, context)
        assert [(c.exposure_us, c.gain) for c, _ in context.windows_run] == [
            (1000, 0),
            (2000, 0),
            (1000, 60),
            (2000, 60),
        ]
        assert {duration for _, duration in context.windows_run} == {3.0}
        assert (result.task_id, result.kind, result.status) == (7, "sweep", "ok")
        assert result.pinned is True
        assert result.summary == "4 of 4 cells measured"
        assert len(result.data["cells"]) == 4
        json.dumps(result.to_detail(), allow_nan=False)  # an event can carry it

    def test_a_cell_without_a_pointing_solution_is_skipped_with_the_reason(self) -> None:
        context = FakeContext(no_solution_for=("bin2",))
        result = self.run(
            QueueSweep(exposure_us=(2000,), gain=(0,), modes=("bin1", "bin2")), context
        )
        statuses = [(c["cell"]["mode"], c["status"]) for c in result.data["cells"]]
        assert statuses == [("bin1", "ok"), ("bin2", "skipped")]
        assert "pointing" in result.data["cells"][1]["note"]
        assert result.summary == "1 of 2 cells measured"
        assert result.status == "ok"

    def test_a_cell_that_the_camera_rejects_is_reported_and_the_sweep_goes_on(self) -> None:
        context = FakeContext()
        context.rejected_gain = 60
        result = self.run(QueueSweep(exposure_us=(2000,), gain=(0, 60, 120)), context)
        statuses = [(c["cell"]["gain"], c["status"]) for c in result.data["cells"]]
        assert statuses == [(0, "ok"), (60, "failed"), (120, "ok")]
        assert "gain not supported" in result.data["cells"][1]["note"]
        assert len(context.windows_run) == 3

    def test_a_stop_request_ends_the_sweep_early_with_the_aborted_status(self) -> None:
        context = FakeContext(stop_after=2)
        result = self.run(QueueSweep(exposure_us=(1000, 2000, 5000, 10000), gain=(0,)), context)
        assert result.status == "aborted"
        assert len(result.data["cells"]) == 2
        assert result.summary.endswith("then the sweep stopped early")

    def test_a_command_of_the_wrong_kind_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="QueueSweep"):
            SweepHandler().run(task(1, command=QueueBurst()), FakeContext())


def test_the_table_lists_a_row_for_each_cell_with_dashes_for_missing_values() -> None:
    ok = summarize_cell(SweepCell("bin1", 2000, 0, 4.1), sample())
    skipped = SweepCellResult(
        cell=SweepCell("bin2", 5000, 120, 4.1), status="skipped", note="no pointing solution"
    )
    lines = format_sweep_table([ok, skipped]).splitlines()
    assert lines[0].split()[:4] == ["mode", "exp_us", "gain", "roi_px"]
    assert len(lines) == 3
    assert "128x128" in lines[1]
    assert "100.0" in lines[1]  # the frame rate
    assert lines[2].split()[:3] == ["bin2", "5000", "120"]
    assert lines[2].endswith("no pointing solution")
    assert " - " in lines[2]


def test_a_result_converts_to_the_detail_of_an_event() -> None:
    result = CommissionResult(
        task_id=3,
        kind="burst",
        status="ok",
        summary="recorded 100 frames",
        started_utc_ns=10,
        finished_utc_ns=20,
        data={"frames": 100},
        artifacts=("bursts/burst-3.ser",),
    )
    detail = result.to_detail()
    assert detail["pinned"] is True  # commissioning results are pinned unless the handler says not
    assert detail["artifacts"] == ["bursts/burst-3.ser"]
    assert detail["data"] == {"frames": 100}
    json.dumps(detail, allow_nan=False)
