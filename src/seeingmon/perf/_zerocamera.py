"""A camera that costs nothing per frame, for the `acquire` side of the `ipc` case.

The fake camera of `seeingmon.testing` builds a frame on every read: it works out the frame
period, takes a time stamp, builds a `Frame` and runs its checks, and records the call in a list.
That work belongs to the fake, and a real camera does not do it in the same way, so the CPU time
of the capture thread with that fake overstates what `acquire` itself costs.

`ZeroCostCamera` builds its frames when you configure a stream, which is before the case starts
its clock. A read then waits for the next frame, as the blocking call of the vendor SDK does, and
returns a frame that exists already. The frames share the memory of the pool that the case gives
them, so a read copies nothing. What the capture thread spends per frame is the capture code of
`acquire`: the gate, the guard, the time stamper, the queue, and the bookkeeping.

The camera leaves out what the real driver does in Python for each frame: the guard around the SDK
calls, the check of the geometry, the copy of the buffer, and the `Frame` with its checks. A
figure that this camera gives is therefore a lower bound of the capture cost.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from seeingmon.clock import Clock
from seeingmon.drivers.base import CameraStateError
from seeingmon.frames import ActiveStream, Frame, FrameFlag, StreamConfig, TimeQuality
from seeingmon.testing import FakeCameraDriver

# The frames of one stream, which repeat after this many. Sequence numbers stay unique for 40 s at
# 98 frames per second, which is longer than any run of the case.
RING_FRAMES = 4096


class ZeroCostCamera(FakeCameraDriver):
    """A `CameraDriver` whose read waits for the frame period and returns a prebuilt frame.

    `pools` maps `f"f{height}x{width}"` to an array of frames with shape `(count, height, width)`,
    the same arrays that `seeingmon.perf._acquire` gives the fake camera. The camera takes the
    pool that matches the ROI of the stream and keeps a ring of `Frame` objects over its rows. The
    frames carry no arrival time, so `acquire` takes the time of the return from the read.
    """

    def __init__(self, clock: Clock, pools: Mapping[str, npt.NDArray[np.generic]]) -> None:
        super().__init__(clock, overhead_s=0.0, row_time_s=0.0)
        self._pools = pools
        self._ring: list[Frame] = []
        self._next = 0
        self._wait_s = 0.0

    def configure(self, config: StreamConfig) -> ActiveStream:
        active = super().configure(config)
        roi = active.config.roi
        assert roi is not None
        pool = self._pools[f"f{roi.height}x{roi.width}"]
        self._wait_s = self._frame_period_s(active.config, roi)
        self._ring = [
            Frame(
                data=pool[index % len(pool)],
                stream_id=active.stream_id,
                seq=index,
                t_arrival_ns=0,  # the service takes the time of the return from the read
                t_utc_ns=0,
                t_err_ns=1_000,
                t_quality=TimeQuality.ESTIMATED,
                dropped_before=0,
                exposure_us=active.config.exposure_us,
                gain=active.config.gain,
                mode=active.config.mode,
                roi=roi,
                adc_bits=active.adc_bits,
                temperature_c=self._temperature_c,
                flags=FrameFlag.SIMULATED,
            )
            for index in range(RING_FRAMES)
        ]
        self._next = 0
        return active

    def read_frame(self, timeout_s: float) -> Frame:
        if not self._running:
            raise CameraStateError("read_frame while not capturing")
        self._clock.sleep(self._wait_s)
        frame = self._ring[self._next]
        self._next = (self._next + 1) % RING_FRAMES
        return frame
