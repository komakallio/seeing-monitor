"""A thin proxy of the camera driver that reports what `open` returns, and how long calls take.

The scheduler opens the camera on its own thread and discards the `CameraInfo` that `open` returns.
The `run` record needs it (the model and the SDK version of the camera), so `core` gives the
scheduler an `InfoDriver`. The proxy forwards every call unchanged and passes the info of each
successful `open` to a callback. A callback that raises never disturbs the camera call.

The proxy also tells `Liveness` about the calls of the scheduler thread. A call into the camera may
block for a long time while the camera recovers, and the watchdog must not take that for a hang.
`read_frame` may block for its timeout, and the other calls for `call_limit_s`. A call that takes
longer is a hang, and the watchdog then stops.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from seeingmon.drivers.base import CameraCaps, CameraDriver, CameraInfo, RecoveryLevel
from seeingmon.frames import ActiveStream, Frame, Roi, StreamConfig
from seeingmon.services.core.liveness import Liveness

_log = logging.getLogger(__name__)

DEFAULT_CALL_LIMIT_S = 240.0
READ_MARGIN_S = 5.0


class InfoDriver:
    """A `CameraDriver` that forwards to another one, and reports the info of each `open`."""

    def __init__(
        self,
        driver: CameraDriver,
        on_open: Callable[[CameraInfo], None],
        *,
        liveness: Liveness | None = None,
        call_limit_s: float = DEFAULT_CALL_LIMIT_S,
        read_margin_s: float = READ_MARGIN_S,
    ) -> None:
        self._driver = driver
        self._on_open = on_open
        self._liveness = liveness
        self._call_limit_s = call_limit_s
        self._read_margin_s = read_margin_s

    @property
    def wrapped(self) -> CameraDriver:
        """The driver behind the proxy."""
        return self._driver

    @property
    def name(self) -> str:
        return self._driver.name

    @contextmanager
    def _within(self, seconds: float) -> Iterator[None]:
        """Tell the liveness that this thread may block for `seconds` inside the call."""
        liveness = self._liveness
        if liveness is None or not liveness.counts_here():
            yield
            return
        liveness.expect(seconds)
        try:
            yield
        finally:
            liveness.leave()

    def open(self) -> CameraInfo:
        with self._within(self._call_limit_s):
            info = self._driver.open()
        try:
            self._on_open(info)
        except Exception:
            _log.exception("the callback for the camera info failed")
        return info

    def close(self) -> None:
        with self._within(self._call_limit_s):
            self._driver.close()

    def capabilities(self) -> CameraCaps:
        with self._within(self._call_limit_s):
            return self._driver.capabilities()

    def configure(self, config: StreamConfig) -> ActiveStream:
        with self._within(self._call_limit_s):
            return self._driver.configure(config)

    def start(self) -> None:
        with self._within(self._call_limit_s):
            self._driver.start()

    def read_frame(self, timeout_s: float) -> Frame:
        with self._within(timeout_s + self._read_margin_s):
            return self._driver.read_frame(timeout_s)

    def stop(self) -> None:
        with self._within(self._call_limit_s):
            self._driver.stop()

    def move_roi(self, x: int, y: int) -> Roi:
        with self._within(self._call_limit_s):
            return self._driver.move_roi(x, y)

    def read_temperature_c(self) -> float | None:
        with self._within(self._call_limit_s):
            return self._driver.read_temperature_c()

    def dropped_frames(self) -> int:
        with self._within(self._call_limit_s):
            return self._driver.dropped_frames()

    def recover(self, level: RecoveryLevel) -> None:
        with self._within(self._call_limit_s):
            self._driver.recover(level)
