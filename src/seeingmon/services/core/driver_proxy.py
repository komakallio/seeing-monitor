"""A thin proxy of the camera driver that reports what `open` returns.

The scheduler opens the camera on its own thread and discards the `CameraInfo` that `open` returns.
The `run` record needs it (the model and the SDK version of the camera), so `core` gives the
scheduler an `InfoDriver`. The proxy forwards every call unchanged and passes the info of each
successful `open` to a callback. A callback that raises never disturbs the camera call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from seeingmon.drivers.base import CameraCaps, CameraDriver, CameraInfo, RecoveryLevel
from seeingmon.frames import ActiveStream, Frame, Roi, StreamConfig

_log = logging.getLogger(__name__)


class InfoDriver:
    """A `CameraDriver` that forwards to another one, and reports the info of each `open`."""

    def __init__(self, driver: CameraDriver, on_open: Callable[[CameraInfo], None]) -> None:
        self._driver = driver
        self._on_open = on_open

    @property
    def wrapped(self) -> CameraDriver:
        """The driver behind the proxy."""
        return self._driver

    @property
    def name(self) -> str:
        return self._driver.name

    def open(self) -> CameraInfo:
        info = self._driver.open()
        try:
            self._on_open(info)
        except Exception:
            _log.exception("the callback for the camera info failed")
        return info

    def close(self) -> None:
        self._driver.close()

    def capabilities(self) -> CameraCaps:
        return self._driver.capabilities()

    def configure(self, config: StreamConfig) -> ActiveStream:
        return self._driver.configure(config)

    def start(self) -> None:
        self._driver.start()

    def read_frame(self, timeout_s: float) -> Frame:
        return self._driver.read_frame(timeout_s)

    def stop(self) -> None:
        self._driver.stop()

    def move_roi(self, x: int, y: int) -> Roi:
        return self._driver.move_roi(x, y)

    def read_temperature_c(self) -> float | None:
        return self._driver.read_temperature_c()

    def dropped_frames(self) -> int:
        return self._driver.dropped_frames()

    def recover(self, level: RecoveryLevel) -> None:
        self._driver.recover(level)
