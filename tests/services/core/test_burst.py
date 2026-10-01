"""The burst handler: a SER file, a JSON sidecar, a pin, and the ways that a burst ends early."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.drivers.base import CameraConfigError, CameraTimeoutError
from seeingmon.frames import ActiveStream, Frame, Roi, StreamConfig, TimeQuality
from seeingmon.profile import load_profile
from seeingmon.recordings.ser import SerFile
from seeingmon.recordings.sidecar import burst_sidecar_path, read_burst_sidecar
from seeingmon.scheduler.commands import QueueBurst, QueueSweep
from seeingmon.scheduler.commission import CommissionTask
from seeingmon.services.core.commissioning.burst import BurstHandler
from seeingmon.store.layout import DataLayout

from .context import FakeContext

START = 1_800_000_000 * NS_PER_S


def task_for(command: QueueBurst, task_id: int = 1) -> CommissionTask:
    return CommissionTask(task_id, "burst", command, START)


@pytest.fixture
def layout(tmp_path: Path) -> DataLayout:
    data = DataLayout(tmp_path / "data")
    data.create()
    return data


Build = Callable[..., BurstHandler]


@pytest.fixture
def build(layout: DataLayout) -> Build:
    def make(clock: VirtualClock, **parts: object) -> BurstHandler:
        defaults: dict[str, object] = {
            "layout": layout,
            "profile": load_profile("asi294mm-gs250"),
            "clock": clock,
            "capture_allowed": lambda: True,
            "max_duration_s": 60.0,
        }
        defaults.update(parts)
        return BurstHandler(**defaults)  # type: ignore[arg-type]

    return make


class TestARecording:
    def test_the_frames_go_to_a_ser_file_with_a_sidecar_and_a_pin(
        self, build: Build, layout: DataLayout
    ) -> None:
        clock = VirtualClock(START)
        context = FakeContext(clock)
        result = build(clock).run(task_for(QueueBurst(duration_s=1.0, label="Focus Test")), context)
        assert result.status == "ok"
        assert result.pinned is True
        assert result.kind == "burst"
        frames = int(result.data["frames"])
        assert frames > 50  # one second at about 90 frames per second
        assert f"Recorded {frames} frames" in result.summary

        (burst,) = [p for p in layout.bursts_dir.iterdir() if p.is_dir()]
        assert burst.name.endswith("-focus-test")
        assert layout.is_pinned(burst)
        ser_path = burst / f"{burst.name}.ser"
        with SerFile(ser_path) as ser:
            assert ser.frame_count == frames
            assert (ser.header.width, ser.header.height) == (32, 32)
            assert ser.header.container_bits == 16
        sidecar = read_burst_sidecar(burst_sidecar_path(ser_path))
        assert sidecar.frame_count == frames
        assert sidecar.stream.mode == "bin1"
        assert sidecar.stream.roi == Roi(100, 100, 32, 32)
        assert sidecar.adc_bits == 14
        assert sidecar.time_quality is TimeQuality.EXACT
        assert sidecar.profile_id == "asi294mm-gs250"
        assert sidecar.start_utc_ns is not None
        assert result.artifacts == (
            layout.relative(ser_path),
            layout.relative(burst_sidecar_path(ser_path)),
        )

    def test_the_sidecar_holds_no_host_path_or_serial(
        self, build: Build, layout: DataLayout
    ) -> None:
        clock = VirtualClock(START)
        build(clock).run(task_for(QueueBurst(duration_s=0.2)), FakeContext(clock))
        (burst,) = [p for p in layout.bursts_dir.iterdir() if p.is_dir()]
        text = (burst / f"{burst.name}.json").read_text()
        assert json.loads(text)["schema_version"] >= 1
        assert str(layout.root) not in text
        assert "serial" not in text.lower()

    def test_the_stream_of_the_command_wins_over_the_fast_stream(self, build: Build) -> None:
        clock = VirtualClock(START)
        context = FakeContext(clock)
        own = StreamConfig("bin2", 1000, 120, roi=Roi(0, 0, 64, 64))
        result = build(clock).run(task_for(QueueBurst(duration_s=0.5, stream=own)), context)
        assert context.configs == [own]
        assert result.data["stream"] == {
            "mode": "bin2",
            "exposure_us": 1000,
            "gain": 120,
            "roi": [0, 0, 64, 64],
        }

    def test_dropped_frames_are_summed(self, build: Build) -> None:
        clock = VirtualClock(START)
        context = FakeContext(clock, drop_at=5)
        result = build(clock).run(task_for(QueueBurst(duration_s=0.2)), context)
        assert result.data["dropped"] == 7

    def test_two_bursts_in_the_same_second_get_two_folders(
        self, build: Build, layout: DataLayout
    ) -> None:
        clock = VirtualClock(START)
        handler = build(clock)
        handler.run(task_for(QueueBurst(duration_s=0.1)), FakeContext(clock))
        clock.step_utc_ns(-int(clock.utc_ns() - START))  # the same wall second again
        handler.run(task_for(QueueBurst(duration_s=0.1), 2), FakeContext(clock))
        assert len([p for p in layout.bursts_dir.iterdir() if p.is_dir()]) == 2


class TestRefusals:
    def test_a_stopped_capture_refuses_the_burst_and_leaves_nothing(
        self, build: Build, layout: DataLayout
    ) -> None:
        clock = VirtualClock(START)
        result = build(clock, capture_allowed=lambda: False).run(
            task_for(QueueBurst()), FakeContext(clock)
        )
        assert result.status == "failed"
        assert "raw capture is stopped" in result.summary
        assert result.pinned is False
        assert list(layout.bursts_dir.iterdir()) == []

    @pytest.mark.parametrize("duration", [0.0, -1.0, 61.0])
    def test_a_duration_out_of_range_is_refused(self, build: Build, duration: float) -> None:
        clock = VirtualClock(START)
        result = build(clock).run(task_for(QueueBurst(duration_s=duration)), FakeContext(clock))
        assert result.status == "failed"
        assert "at most 60 s" in result.summary

    def test_a_task_of_another_kind_is_refused(self, build: Build) -> None:
        clock = VirtualClock(START)
        result = build(clock).run(task_for(QueueSweep()), FakeContext(clock))  # type: ignore[arg-type]
        assert result.status == "failed"
        assert "no burst command" in result.summary

    def test_without_a_pointing_solution_the_fast_stream_cannot_be_made(self, build: Build) -> None:
        clock = VirtualClock(START)
        context = FakeContext(clock)
        context.pointing = None
        result = build(clock).run(task_for(QueueBurst()), context)
        assert result.status == "failed"
        assert "no pointing solution" in result.summary

    def test_a_stream_that_the_camera_refuses_is_a_failed_task_and_not_a_fault(
        self, build: Build, layout: DataLayout
    ) -> None:
        clock = VirtualClock(START)
        context = FakeContext(clock)

        def refuse(config: StreamConfig) -> ActiveStream:
            raise CameraConfigError("the ROI does not fit")

        context.configure = refuse  # type: ignore[method-assign]
        result = build(clock).run(task_for(QueueBurst()), context)
        assert result.status == "failed"
        assert "the ROI does not fit" in result.summary
        assert list(layout.bursts_dir.iterdir()) == []


class TestEndingEarly:
    def test_a_preempting_command_stops_the_burst_and_keeps_the_frames(
        self, build: Build, layout: DataLayout
    ) -> None:
        clock = VirtualClock(START)
        context = FakeContext(clock, stop_after=10)
        result = build(clock).run(task_for(QueueBurst(duration_s=30.0)), context)
        assert result.status == "aborted"
        assert result.data["frames"] == 10
        assert "another command took the camera" in result.summary
        (burst,) = [p for p in layout.bursts_dir.iterdir() if p.is_dir()]
        assert layout.is_pinned(burst)  # the partial burst still counts as data
        with SerFile(burst / f"{burst.name}.ser") as ser:
            assert ser.frame_count == 10

    def test_a_capture_that_stops_in_the_middle_ends_the_burst(self, build: Build) -> None:
        clock = VirtualClock(START)
        context = FakeContext(clock)
        allowed = {"value": True}

        def gate() -> bool:
            if context.reads >= 5:
                allowed["value"] = False
            return allowed["value"]

        result = build(clock, capture_allowed=gate).run(
            task_for(QueueBurst(duration_s=30.0)), context
        )
        assert result.status == "aborted"
        assert "free space fell below the limit" in result.summary
        assert result.data["frames"] == 5

    def test_a_burst_that_stops_before_a_frame_leaves_no_folder(
        self, build: Build, layout: DataLayout
    ) -> None:
        clock = VirtualClock(START)
        result = build(clock).run(
            task_for(QueueBurst(duration_s=5.0)), FakeContext(clock, stop_after=0)
        )
        assert result.status == "failed"
        assert "no frame" in result.summary
        assert list(layout.bursts_dir.iterdir()) == []

    def test_a_camera_error_reaches_the_scheduler_after_the_file_is_closed(
        self, build: Build, layout: DataLayout
    ) -> None:
        clock = VirtualClock(START)
        context = FakeContext(clock)
        handler = build(clock)

        original = context.read_frame

        def flaky(timeout_s: float | None = None) -> Frame:
            if context.reads >= 3:
                raise CameraTimeoutError("the camera stopped")
            return original(timeout_s)

        context.read_frame = flaky  # type: ignore[method-assign]
        with pytest.raises(CameraTimeoutError):
            handler.run(task_for(QueueBurst(duration_s=5.0)), context)
        (burst,) = [p for p in layout.bursts_dir.iterdir() if p.is_dir()]
        with SerFile(burst / f"{burst.name}.ser") as ser:  # a valid file, with the three frames
            assert ser.frame_count == 3
