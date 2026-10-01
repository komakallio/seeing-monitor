"""The production ASI driver on a stub SDK, for the camera costs of the `ipc` case.

`seeingmon.drivers.asi.AsiDriver` is the driver that `acquire` runs on a Raspberry Pi. For each
frame it takes the lock, arms the call watchdog, reads the SDK buffer and the drop counter, copies
the pixels, and builds a `Frame` with its checks. `StubSdk` replaces the vendor library with the
least that this driver needs for a frame: its video read waits for the frame period, as the real
call blocks until the camera delivers, and then copies a stored frame into the buffer, as the real
call does. The rest of the SDK is the fake of `seeingmon.hardware.asi.fake`, which the driver
accepts for its set-up calls.

The CPU time of a `read_frame` call on this driver is the cost of the production Python code for a
frame, from the capture thread down to the SDK call. It leaves out what the vendor library and the
USB stack of the kernel use for a frame.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from seeingmon.clock import Clock
from seeingmon.drivers.asi.driver import AsiDriver
from seeingmon.drivers.asi.options import AsiOptions
from seeingmon.hardware.asi.fake import DEFAULT_TIMING, FakeAsiSdk, FakeTiming
from seeingmon.hardware.asi.watchdog import CallWatchdog
from seeingmon.profile import load_profile


class StubSdk(FakeAsiSdk):
    """A fake SDK whose video read waits for the frame period and copies a stored frame."""

    def __init__(self, clock: Clock, pools: Mapping[str, npt.NDArray[np.generic]]) -> None:
        # No readout time, so that the frame period is the exposure, as in the other cameras.
        super().__init__(
            clock,
            timing={key: FakeTiming(0.0, 0.0) for key in DEFAULT_TIMING},
            temperature_warmup_s=0.0,
        )
        self._pools = pools
        self._payloads: list[bytes] = []
        self._index = 0
        self._wait_s = 0.0

    def start_video_capture(self, camera_id: int) -> None:
        super().start_video_capture(camera_id)
        pool = self._pools[f"f{self._height}x{self._width}"]  # the ROI that is now set
        self._payloads = [frame.astype("<u2").tobytes() for frame in pool]
        self._wait_s = self.frame_period_s()
        self._index = 0

    def get_video_data(self, camera_id: int, buffer: bytearray, wait_ms: int) -> None:
        self._clock.sleep(self._wait_s)
        payload = self._payloads[self._index % len(self._payloads)]
        self._index += 1
        buffer[: len(payload)] = payload

    def get_dropped_frames(self, camera_id: int) -> int:
        return 0


def create_asi_driver(
    clock: Clock, pools: Mapping[str, npt.NDArray[np.generic]], *, watchdog_thread: bool = True
) -> AsiDriver:
    """The production driver, with the watchdog that production uses, on a `StubSdk`.

    Production starts a thread for the watchdog. Pass `watchdog_thread=False` for a clock that must
    not drive a thread, such as a `VirtualClock`.
    """
    return AsiDriver(
        api=StubSdk(clock, pools),
        profile=load_profile("asi294mm-gs250"),
        clock=clock,
        options=AsiOptions(),
        usb_resetter=None,
        watchdog=CallWatchdog(clock, lambda report: None),
        watchdog_thread=watchdog_thread,
        on_event=None,
    )
