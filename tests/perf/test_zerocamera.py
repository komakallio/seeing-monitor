"""The camera that costs nothing per frame: it waits for the frame period and returns a frame that
exists already."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.drivers.base import CameraStateError
from seeingmon.frames import Roi, StreamConfig
from seeingmon.perf import _zerocamera
from seeingmon.perf._zerocamera import ZeroCostCamera

EXPOSURE_US = 10_000


def pool_of(count: int = 8) -> npt.NDArray[np.uint16]:
    return np.arange(count * 16 * 16, dtype=np.uint16).reshape(count, 16, 16)


def started_camera(
    pool: npt.NDArray[np.uint16] | None = None,
) -> tuple[VirtualClock, ZeroCostCamera, npt.NDArray[np.uint16]]:
    clock = VirtualClock()
    frames = pool_of() if pool is None else pool
    camera = ZeroCostCamera(clock, {"f16x16": frames})
    camera.open()
    camera.configure(StreamConfig("bin1", EXPOSURE_US, 120, roi=Roi(8, 8, 16, 16)))
    camera.start()
    return clock, camera, frames


class TestReads:
    def test_a_read_waits_for_one_frame_period(self) -> None:
        clock, camera, _ = started_camera()
        start = clock.monotonic_ns()
        for count in range(1, 4):
            camera.read_frame(1.0)
            assert clock.monotonic_ns() - start == count * EXPOSURE_US * 1_000

    def test_the_period_is_the_exposure_because_the_camera_has_no_readout_time(self) -> None:
        clock, camera, _ = started_camera()
        camera.read_frame(1.0)
        assert clock.monotonic_ns() == round(EXPOSURE_US / 1e6 * NS_PER_S)

    def test_the_frames_count_up_and_share_the_memory_of_the_pool(self) -> None:
        _, camera, pool = started_camera()
        frames = [camera.read_frame(1.0) for _ in range(5)]
        assert [frame.seq for frame in frames] == [0, 1, 2, 3, 4]
        for index, frame in enumerate(frames):
            assert np.shares_memory(frame.data, pool)
            np.testing.assert_array_equal(frame.data, pool[index])

    def test_the_frames_describe_the_stream(self) -> None:
        _, camera, _ = started_camera()
        frame = camera.read_frame(1.0)
        assert (frame.mode, frame.gain, frame.exposure_us) == ("bin1", 120, EXPOSURE_US)
        assert frame.roi == Roi(8, 8, 16, 16)
        assert frame.dropped_before == 0

    def test_a_frame_has_no_arrival_time_so_that_acquire_takes_its_own(self) -> None:
        _, camera, _ = started_camera()
        assert camera.read_frame(1.0).t_arrival_ns == 0

    def test_the_pool_repeats_when_the_stream_is_longer_than_it(self) -> None:
        _, camera, pool = started_camera(pool_of(3))
        frames = [camera.read_frame(1.0) for _ in range(4)]
        np.testing.assert_array_equal(frames[3].data, pool[0])
        assert frames[3].seq == 3  # the sequence number keeps counting

    def test_the_ring_comes_around_after_its_length(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_zerocamera, "RING_FRAMES", 4)
        _, camera, _ = started_camera()
        frames = [camera.read_frame(1.0) for _ in range(5)]
        assert frames[4] is frames[0]
        assert len({id(frame) for frame in frames[:4]}) == 4

    def test_a_read_leaves_no_entry_in_the_call_log_of_the_fake(self) -> None:
        _, camera, _ = started_camera()
        for _ in range(3):
            camera.read_frame(1.0)
        assert [name for name, _ in camera.calls] == ["open", "configure", "start"]

    def test_a_new_stream_has_its_own_frames(self) -> None:
        _, camera, _ = started_camera()
        first = camera.read_frame(1.0)
        camera.configure(StreamConfig("bin1", EXPOSURE_US, 120, roi=Roi(8, 8, 16, 16)))
        camera.start()
        second = camera.read_frame(1.0)
        assert second.stream_id != first.stream_id
        assert second.seq == 0

    def test_a_read_when_the_camera_does_not_run_raises(self) -> None:
        _, camera, _ = started_camera()
        camera.stop()
        with pytest.raises(CameraStateError):
            camera.read_frame(1.0)
