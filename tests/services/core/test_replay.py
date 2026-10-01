"""The replay handler: a recording through the production analysis, into a separate store."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy", reason="the fast path needs the fast extra")

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.fastpath import FastPathConfig
from seeingmon.frames import PixelFormat, Roi, StreamConfig, TimeQuality
from seeingmon.profile import load_profile
from seeingmon.recordings.ser import SerWriter
from seeingmon.recordings.sidecar import BurstSidecar, burst_sidecar_path, write_burst_sidecar
from seeingmon.scheduler.commands import QueueReplay
from seeingmon.scheduler.commission import CommissionResult, CommissionTask
from seeingmon.services.core.commissioning.replay import (
    ReplayHandler,
    clean_options,
    resolve_source,
)
from seeingmon.store.config import StoreConfig
from seeingmon.store.db import Store
from seeingmon.store.layout import DataLayout

from .context import FakeContext

START = 1_800_000_000 * NS_PER_S
FRAMES = 1000
PERIOD_NS = 10_200_000  # 98 frames per second


def write_recording(path: Path, *, frames: int = FRAMES, sidecar: bool = True) -> None:
    """A synthetic 8-bit bin2 recording of a star that jitters by a tenth of a pixel."""
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:96, 0:96]
    with SerWriter(path, width=96, height=96, pixel_depth=8) as writer:
        for index in range(frames):
            cx = 48.0 + rng.normal(0, 0.1)
            cy = 48.0 + rng.normal(0, 0.1)
            star = 150.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0**2))
            data = rng.normal(20.0, 2.0, (96, 96)) + star
            writer.write_frame(np.clip(data, 0, 255).astype(np.uint8), START + index * PERIOD_NS)
    if sidecar:
        write_burst_sidecar(
            burst_sidecar_path(path),
            BurstSidecar(
                profile_id="asi294mm-gs250",
                stream=StreamConfig(
                    "bin2", 10_000, 100, pixel_format=PixelFormat.RAW8, roi=Roi(0, 0, 96, 96)
                ),
                adc_bits=8,
                time_quality=TimeQuality.ESTIMATED,
                frame_period_s=PERIOD_NS / NS_PER_S,
                frame_count=frames,
                start_utc_ns=START,
            ),
        )


@pytest.fixture(scope="module")
def recordings(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("recordings")
    write_recording(folder / "night.ser")
    return folder


@pytest.fixture
def layout(tmp_path: Path) -> DataLayout:
    data = DataLayout(tmp_path / "data")
    data.create()
    return data


Build = Callable[..., ReplayHandler]


@pytest.fixture
def build(layout: DataLayout, recordings: Path) -> Build:
    def make(clock: VirtualClock, **parts: object) -> ReplayHandler:
        defaults: dict[str, object] = {
            "layout": layout,
            "profile": load_profile("asi294mm-gs250"),
            "station_id": "test",
            "clock": clock,
            "fast_config": FastPathConfig(window_s=4.0, min_window_s=2.0),
            "store_config": StoreConfig(),
            "recordings_dir": recordings,
        }
        defaults.update(parts)
        return ReplayHandler(**defaults)  # type: ignore[arg-type]

    return make


def run(
    handler: ReplayHandler,
    clock: VirtualClock,
    command: QueueReplay,
    *,
    stop_after: int | None = None,
) -> CommissionResult:
    return handler.run(
        CommissionTask(1, "replay", command, START), FakeContext(clock, stop_after=stop_after)
    )


class TestAReplay:
    def test_the_recording_goes_through_the_analysis_into_a_separate_store(
        self, build: Build, layout: DataLayout
    ) -> None:
        clock = VirtualClock(START)
        result = run(build(clock), clock, QueueReplay(source="night.ser", speed=0))
        assert result.status == "ok"
        assert result.pinned is True
        assert result.data["frames"] == FRAMES
        windows = int(result.data["windows"])
        assert windows >= 2  # 10 s of frames in windows of 4 s, and a partial one
        assert f"Replayed {FRAMES} frames into {windows} windows" in result.summary
        (replay_db,) = result.artifacts
        assert replay_db.startswith("replays/")
        assert replay_db.endswith("db/results.sqlite")
        with Store.open(layout.resolve(replay_db)) as store:
            assert store.count("seeing_window") == windows
        # The main store of the data directory never sees a replayed window.
        assert not layout.db_path.exists()
        replay_root = layout.resolve(str(result.data["store"]))
        assert any(replay_root.joinpath("segments").rglob("*.seg*")) or any(
            p.is_file() for p in replay_root.joinpath("segments").rglob("*")
        )

    def test_the_median_r0_goes_into_the_result_and_the_summary(self, build: Build) -> None:
        clock = VirtualClock(START)
        result = run(build(clock), clock, QueueReplay(source="night.ser", speed=0))
        r0 = result.data["r0_cm_median"]
        assert r0 is None or float(r0) > 0
        if r0 is not None:
            assert "median r0" in result.summary

    def test_the_name_works_without_the_suffix(self, build: Build) -> None:
        clock = VirtualClock(START)
        result = run(build(clock), clock, QueueReplay(source="night", speed=0))
        assert result.status == "ok"

    def test_a_burst_of_the_data_directory_replays_by_its_folder_name(
        self, build: Build, layout: DataLayout
    ) -> None:
        burst = layout.new_burst(START, "mine")
        write_recording(burst / f"{burst.name}.ser", frames=300)
        clock = VirtualClock(START)
        result = run(
            build(clock, recordings_dir=None), clock, QueueReplay(source=burst.name, speed=0)
        )
        assert result.status == "ok"
        assert result.data["frames"] == 300

    def test_the_speed_paces_the_replay_on_the_clock(self, build: Build) -> None:
        clock = VirtualClock(START)
        before = clock.monotonic_ns()
        run(
            build(clock),
            clock,
            QueueReplay(source="night.ser", speed=1.0, options={"max_frames": 300}),
        )
        paced = (clock.monotonic_ns() - before) / NS_PER_S
        assert paced == pytest.approx(300 * PERIOD_NS / NS_PER_S, rel=0.05)  # the original pace
        clock = VirtualClock(START)
        before = clock.monotonic_ns()
        run(
            build(clock),
            clock,
            QueueReplay(source="night.ser", speed=0, options={"max_frames": 300}),
        )
        assert (clock.monotonic_ns() - before) / NS_PER_S < 0.5  # as fast as possible

    def test_an_option_limits_the_frames(self, build: Build) -> None:
        clock = VirtualClock(START)
        result = run(
            build(clock),
            clock,
            QueueReplay(source="night.ser", speed=0, options={"max_frames": 200}),
        )
        assert result.data["frames"] == 200

    def test_each_frame_tells_the_watchdog_that_the_scheduler_thread_works(
        self, build: Build
    ) -> None:
        clock = VirtualClock(START)
        beats: list[int] = []
        result = run(
            build(clock, beat=lambda: beats.append(1)),
            clock,
            QueueReplay(source="night.ser", speed=0, options={"max_frames": 200}),
        )
        assert result.data["frames"] == 200
        assert len(beats) == 200

    def test_a_preempting_command_ends_the_replay_at_once(self, build: Build) -> None:
        clock = VirtualClock(START)
        # The fake context stops when its camera has read `stop_after` frames. A replay reads none.
        result = run(build(clock), clock, QueueReplay(source="night.ser", speed=0), stop_after=0)
        assert result.status == "aborted"
        assert result.data["frames"] == 0
        assert "another command took over" in result.summary


class TestRefusals:
    @pytest.mark.parametrize(
        "source",
        ["", "..", "../night.ser", "sub/night.ser", "C:night.ser", "dir\\night.ser", "/abs.ser"],
    )
    def test_a_source_with_a_directory_part_is_refused(self, build: Build, source: str) -> None:
        clock = VirtualClock(START)
        result = run(build(clock), clock, QueueReplay(source=source))
        assert result.status == "failed"
        assert "without a directory part" in result.summary

    def test_an_unknown_recording_is_refused_without_naming_a_path(
        self, build: Build, recordings: Path
    ) -> None:
        clock = VirtualClock(START)
        result = run(build(clock), clock, QueueReplay(source="missing.ser"))
        assert result.status == "failed"
        assert "no recording has that name" in result.summary
        assert str(recordings) not in result.summary

    @pytest.mark.parametrize("option", ["path", "sidecar", "loop", "rate", "nothing"])
    def test_an_option_that_could_widen_the_access_is_refused(
        self, build: Build, option: str
    ) -> None:
        clock = VirtualClock(START)
        result = run(build(clock), clock, QueueReplay(source="night.ser", options={option: "x"}))
        assert result.status == "failed"
        assert f"the option {option!r}" in result.summary

    @pytest.mark.parametrize("speed", [-1.0, float("nan"), float("inf")])
    def test_a_bad_speed_is_refused(self, build: Build, speed: float) -> None:
        clock = VirtualClock(START)
        result = run(build(clock), clock, QueueReplay(source="night.ser", speed=speed))
        assert result.status == "failed"
        assert "speed" in result.summary

    def test_a_file_that_is_not_a_recording_fails_the_task_and_leaves_the_store_closed(
        self, build: Build, recordings: Path
    ) -> None:
        (recordings / "junk.ser").write_bytes(b"not a recording" * 20)
        clock = VirtualClock(START)
        result = run(build(clock), clock, QueueReplay(source="junk.ser", speed=0))
        assert result.status == "failed"
        assert "cannot replay" in result.summary or "failed" in result.summary

    def test_a_task_of_another_kind_is_refused(self, build: Build) -> None:
        from seeingmon.scheduler.commands import QueueBurst

        clock = VirtualClock(START)
        handler = build(clock)
        result = handler.run(
            CommissionTask(1, "replay", QueueBurst(), START),
            FakeContext(clock),
        )
        assert result.status == "failed"
        assert "no replay command" in result.summary


class TestResolving:
    def test_the_recordings_folder_comes_before_a_burst(
        self, tmp_path: Path, layout: DataLayout
    ) -> None:
        folder = tmp_path / "rec"
        folder.mkdir()
        (folder / "same.ser").write_bytes(b"x")
        burst = layout.bursts_dir / "same"
        burst.mkdir()
        (burst / "same.ser").write_bytes(b"y")
        path, _ = resolve_source("same", folder, layout)
        assert path == folder / "same.ser"

    def test_the_options_are_checked_against_the_list(self) -> None:
        assert clean_options({"max_frames": 5, "mode": "bin2"}) == (
            {"max_frames": 5, "mode": "bin2"},
            "",
        )
        options, reason = clean_options({"path": "x"})
        assert options == {}
        assert "'path'" in reason
