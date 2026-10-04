"""`acquire` waits for a single exposure as long as the driver says that the exposure takes.

The capture thread reads a frame with a timeout of `read_timeout_factor` frame periods plus
`read_timeout_margin_s`, and the period comes from the driver (`ActiveStream.frame_period_s`). A
snapshot stream reports the snapshot model of the profile, so the wait of the capture thread follows
it, and a video stream keeps its short wait.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from seeingmon.frames import Frame, Roi, StreamConfig, StreamKind
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.keys import ConnectionKey

from .rig import Rig, TracingFake, make_rig

# The numbers of the snapshot model of the reference profile in bin2.
SNAPSHOT_OVERHEAD_S = 0.27
SNAPSHOT_ROW_TIME_S = 75e-6


class TimeoutNotingFake(TracingFake):
    """A fake that notes the timeout of each read."""

    def __init__(self, clock: Any, **options: Any) -> None:
        super().__init__(clock, **options)
        self.timeouts: list[float] = []

    def read_frame(self, timeout_s: float) -> Frame:
        self.timeouts.append(timeout_s)
        return super().read_frame(timeout_s)


@pytest.fixture
def rig(native: Endpoint, key: ConnectionKey) -> Iterator[Rig]:
    made = make_rig(
        native,
        key,
        speed=20.0,  # the 0.3 s of a snapshot passes in 15 ms of real time
        fake_class=TimeoutNotingFake,
        fake_options={
            "snapshot_overhead_s": SNAPSHOT_OVERHEAD_S,
            "snapshot_row_time_s": SNAPSHOT_ROW_TIME_S,
        },
    )
    yield made
    made.close()


def noted(rig: Rig) -> list[float]:
    assert isinstance(rig.fake, TimeoutNotingFake)
    return rig.fake.timeouts


def test_a_snapshot_read_waits_twice_the_snapshot_period_plus_the_margin(rig: Rig) -> None:
    driver = rig.driver()
    driver.open()
    config = StreamConfig(
        mode="bin2", exposure_us=1000, gain=1, kind=StreamKind.SNAPSHOT, roi=Roi(0, 0, 4144, 2822)
    )
    active = driver.configure(config)
    period_s = 0.001 + SNAPSHOT_OVERHEAD_S + 2822 * SNAPSHOT_ROW_TIME_S  # 0.48 s for a full frame
    assert active.frame_period_s == pytest.approx(period_s)
    driver.start()
    assert driver.read_frame(10.0).seq == 0
    (timeout_s,) = noted(rig)
    assert timeout_s == pytest.approx(2 * period_s + 0.5)
    assert timeout_s > 1.0  # the video line of the same ROI gives 0.6 s, and the frame needs 0.5 s


def test_a_video_read_keeps_its_short_wait(rig: Rig) -> None:
    driver = rig.driver()
    driver.open()
    active = driver.configure(
        StreamConfig(mode="bin2", exposure_us=1000, gain=1, roi=Roi(0, 0, 64, 64))
    )
    video_period_s = 0.0065 + 64 * 37.6e-6  # the numbers of the fake for a video frame
    assert active.frame_period_s == pytest.approx(video_period_s)
    driver.start()
    for _ in range(3):
        driver.read_frame(10.0)
    driver.stop()
    assert min(noted(rig)) == pytest.approx(2 * video_period_s + 0.5)
    assert max(noted(rig)) < 0.6  # a snapshot model does not reach a stream
