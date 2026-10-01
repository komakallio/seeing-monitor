"""The lifecycle tests of `FakeCameraDriver`, run against the fake and against the remote driver.

`tests/test_fakes.py` checks the lifecycle that every `CameraDriver` follows. This module runs
the same scenarios twice: on a `FakeCameraDriver` directly, and on a `RemoteCameraDriver` that
talks to an `acquire` service, which runs a `FakeCameraDriver`. A scenario passes on the remote
driver only if the network hop is invisible to the scheduler.

Scripted faults (`fail_reads`, `drop_frames`) go to the fake in both cases, before `start`,
because `acquire` reads ahead of the caller and a fault that comes after `start` meets a frame
that is already in the queue.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seeingmon.clock import DEFAULT_START_UTC_NS
from seeingmon.drivers.base import (
    CameraConfigError,
    CameraDisconnectedError,
    CameraDriver,
    CameraError,
    CameraStateError,
    CameraTimeoutError,
    RecoveryLevel,
)
from seeingmon.frames import (
    FrameFlag,
    PixelFormat,
    Roi,
    StreamConfig,
    StreamKind,
    TimeQuality,
    decode_frame,
    encode_frame,
    frames_equal,
)
from seeingmon.services.ipc.keys import ConnectionKey

from .conftest import native_endpoint
from .rig import FAST, StatusClock, TracingFake, make_rig


@dataclass
class Camera:
    """The driver under test, and the fake behind it (the same object when there is no hop)."""

    driver: CameraDriver
    fake: TracingFake
    remote: bool


MakeCamera = Callable[..., Camera]


@pytest.fixture(params=["direct", "remote"])
def make_camera(
    request: pytest.FixtureRequest, short_dir: Path, key: ConnectionKey
) -> Iterator[MakeCamera]:
    cleanups: list[Callable[[], None]] = []

    def make(speed: float = 5.0, **fake_options: Any) -> Camera:
        if request.param == "direct":
            clock = StatusClock(
                start_utc_ns=DEFAULT_START_UTC_NS, origin_real_ns=time.time_ns(), speed=speed
            )
            fake = TracingFake(clock, **fake_options)
            return Camera(fake, fake, remote=False)
        endpoint = native_endpoint(short_dir, f"camera{len(cleanups)}")
        rig = make_rig(endpoint, key, speed=speed, fake_options=fake_options)
        cleanups.append(rig.close)
        return Camera(rig.driver(), rig.fake, remote=True)

    yield make
    for cleanup in cleanups:
        cleanup()


@pytest.fixture
def camera(make_camera: MakeCamera) -> Camera:
    return make_camera()


def test_it_satisfies_the_protocol(camera: Camera) -> None:
    assert isinstance(camera.driver, CameraDriver)


def test_streams_frames_in_order_with_their_metadata(camera: Camera) -> None:
    driver = camera.driver
    driver.open()
    active = driver.configure(FAST)
    driver.start()
    frames = [driver.read_frame(timeout_s=10.0) for _ in range(5)]
    assert [f.seq for f in frames] == [0, 1, 2, 3, 4]
    assert all(f.stream_id == active.stream_id for f in frames)
    assert all(f.t_quality is TimeQuality.EXACT for f in frames)
    assert all(f.flags & FrameFlag.SIMULATED for f in frames)
    assert frames[0].data.shape == (128, 128)
    assert frames[0].t_utc_ns < frames[0].t_arrival_ns
    assert frames[1].t_arrival_ns > frames[0].t_arrival_ns


def test_frames_survive_the_wire_format(camera: Camera) -> None:
    driver = camera.driver
    driver.open()
    driver.configure(FAST)
    driver.start()
    frame = driver.read_frame(timeout_s=10.0)
    assert frames_equal(decode_frame(encode_frame(frame)), frame)


def test_roi_follows_the_vendor_rules_and_stays_inside_the_frame(camera: Camera) -> None:
    driver = camera.driver
    driver.open()
    odd = StreamConfig(mode="bin1", exposure_us=2000, gain=1, roi=Roi(8200, 5600, 131, 125))
    roi = driver.configure(odd).config.roi
    assert roi is not None
    assert (roi.width, roi.height) == (128, 124)
    assert roi.x_end <= 8288
    assert roi.y_end <= 5644


def test_the_full_frame_is_the_default_roi(camera: Camera) -> None:
    driver = camera.driver
    driver.open()
    active = driver.configure(StreamConfig(mode="bin2", exposure_us=1000, gain=1))
    assert active.frame_shape == (2822, 4144)


def test_each_configure_starts_a_new_stream(camera: Camera) -> None:
    driver = camera.driver
    driver.open()
    first = driver.configure(FAST)
    driver.start()
    driver.read_frame(10.0)
    second = driver.configure(FAST)
    assert second.stream_id == first.stream_id + 1
    with pytest.raises(CameraStateError):  # configure stops capture
        driver.read_frame(1.0)
    driver.start()
    frame = driver.read_frame(10.0)
    assert frame.seq == 0
    assert frame.stream_id == second.stream_id


def test_move_roi_keeps_the_stream_and_clamps(camera: Camera) -> None:
    driver = camera.driver
    driver.open()
    active = driver.configure(FAST)
    driver.start()
    moved = driver.move_roi(-50, 10_000)
    assert moved == Roi(0, 5644 - 128, 128, 128)
    deadline = time.monotonic() + 10.0
    frame = driver.read_frame(10.0)
    while frame.roi != moved and time.monotonic() < deadline:  # frames before the move drain first
        frame = driver.read_frame(10.0)
    assert frame.roi == moved
    assert frame.stream_id == active.stream_id


def test_snapshot_returns_one_frame_per_start(camera: Camera) -> None:
    driver = camera.driver
    driver.open()
    driver.configure(
        StreamConfig(
            mode="bin2",
            exposure_us=250_000,
            gain=120,
            kind=StreamKind.SNAPSHOT,
            roi=Roi(0, 0, 64, 64),
        )
    )
    driver.start()
    frame = driver.read_frame(timeout_s=30.0)
    assert frame.exposure_us == 250_000
    with pytest.raises(CameraStateError):
        driver.read_frame(1.0)
    driver.start()
    assert driver.read_frame(30.0).seq == 1


def test_a_short_timeout_expires(camera: Camera) -> None:
    driver = camera.driver
    driver.open()
    driver.configure(StreamConfig(mode="bin1", exposure_us=5_000_000, gain=1, roi=Roi(0, 0, 8, 2)))
    driver.start()
    with pytest.raises(CameraTimeoutError):
        driver.read_frame(timeout_s=0.1)


def test_scripted_read_failures_come_one_per_read(camera: Camera) -> None:
    driver = camera.driver
    driver.open()
    driver.configure(FAST)
    camera.fake.fail_reads(CameraTimeoutError("stall"), CameraDisconnectedError("unplugged"))
    driver.start()
    with pytest.raises(CameraTimeoutError):
        driver.read_frame(10.0)
    with pytest.raises(CameraDisconnectedError):
        driver.read_frame(10.0)
    assert driver.read_frame(10.0).seq == 0


def test_dropped_frames_are_reported_once(camera: Camera) -> None:
    driver = camera.driver
    driver.open()
    driver.configure(FAST)
    camera.fake.drop_frames(3)
    driver.start()
    assert driver.dropped_frames() == 3
    assert driver.read_frame(10.0).dropped_before == 3
    assert driver.read_frame(10.0).dropped_before == 0
    assert driver.dropped_frames() == 3


def test_lifecycle_errors(camera: Camera) -> None:
    driver = camera.driver
    with pytest.raises(CameraStateError):
        driver.configure(FAST)
    driver.open()
    with pytest.raises(CameraStateError):
        driver.start()
    with pytest.raises(CameraConfigError):
        driver.configure(StreamConfig(mode="bin9", exposure_us=1, gain=0))
    with pytest.raises(CameraConfigError):
        driver.configure(StreamConfig(mode="bin1", exposure_us=1, gain=0, roi=Roi(0, 0, 9000, 8)))


def test_a_failing_open_raises_the_camera_error(camera: Camera) -> None:
    camera.fake.fail_open()
    with pytest.raises(CameraDisconnectedError):
        camera.driver.open()


def test_recovery_and_close(camera: Camera) -> None:
    driver = camera.driver
    info = driver.open()
    assert (info.driver, info.has_temperature) == ("fake", True)
    driver.recover(RecoveryLevel.RESTART_CAPTURE)
    camera.fake.fail_recover()
    with pytest.raises(CameraError):
        driver.recover(RecoveryLevel.USB_RESET)
    driver.close()
    driver.close()
    assert [call for call, _ in camera.fake.calls if call == "recover"] == ["recover", "recover"]


def test_the_temperature_and_the_capabilities(camera: Camera) -> None:
    driver = camera.driver
    driver.open()
    assert driver.read_temperature_c() == 18.0
    assert driver.capabilities() == camera.fake.capabilities()


def test_a_custom_frame_factory_and_pixel_scaling(make_camera: MakeCamera) -> None:
    def star(config: StreamConfig, roi: Roi, seq: int) -> np.ndarray[Any, Any]:
        data = np.zeros((roi.height, roi.width), dtype=config.pixel_format.dtype)
        data[roi.height // 2, roi.width // 2] = 1000 + seq
        return data

    custom = make_camera(frame_factory=star)
    custom.driver.open()
    custom.driver.configure(
        StreamConfig(mode="bin1", exposure_us=1000, gain=0, roi=Roi(0, 0, 16, 16))
    )
    custom.driver.start()
    assert custom.driver.read_frame(10.0).data[8, 8] == 1000
    assert custom.driver.read_frame(10.0).data[8, 8] == 1001

    plain = make_camera(adc_bits=12)
    plain.driver.open()
    plain.driver.configure(
        StreamConfig(
            mode="bin1",
            exposure_us=1000,
            gain=0,
            roi=Roi(0, 0, 8, 2),
            pixel_format=PixelFormat.RAW16,
        )
    )
    plain.driver.start()
    assert plain.driver.read_frame(10.0).data[0, 0] == 100 << 4
