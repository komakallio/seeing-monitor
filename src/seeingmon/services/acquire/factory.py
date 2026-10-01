"""Create the camera driver that `acquire` owns, from its configuration.

`create_camera_driver` hands every name to `seeingmon.drivers.create_driver`, which imports
`seeingmon.drivers.<name>` and calls its `create(profile=..., clock=..., options=...)`. One name
is special: `fake` builds the scripted `FakeCameraDriver` of `seeingmon.testing`, so that the
end-to-end tests and a development machine without a camera can run `acquire` with a driver that
needs nothing. The options of the fake are `adc_bits`, `overhead_s`, `row_time_s`, and
`temperature_c`. The option `hang_in` (`open`, `configure`, `start`, or `read_frame`) makes that
call block for good, so that a test can show that a hung driver call ends the process.

This module imports the fakes, so import it only when you create a driver.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from seeingmon.clock import Clock
from seeingmon.drivers import CameraConfigError, CameraDriver
from seeingmon.drivers.base import CameraInfo
from seeingmon.frames import ActiveStream, Frame, StreamConfig
from seeingmon.testing import FakeCameraDriver

FAKE_OPTIONS = {"adc_bits": int, "overhead_s": float, "row_time_s": float, "temperature_c": float}
HANGABLE = ("open", "configure", "start", "read_frame")


class HangingFakeDriver(FakeCameraDriver):
    """A fake whose `hang_in` call blocks for good, as an SDK call can after a USB fault."""

    def __init__(self, clock: Clock, *, hang_in: str, **options: Any) -> None:
        super().__init__(clock, **options)
        self._hang_in = hang_in

    def _hang(self, name: str) -> None:
        if name == self._hang_in:
            threading.Event().wait()

    def open(self) -> CameraInfo:
        self._hang("open")
        return super().open()

    def configure(self, config: StreamConfig) -> ActiveStream:
        self._hang("configure")
        return super().configure(config)

    def start(self) -> None:
        self._hang("start")
        super().start()

    def read_frame(self, timeout_s: float) -> Frame:
        self._hang("read_frame")
        return super().read_frame(timeout_s)


def create_camera_driver(
    name: str, *, profile: Any, clock: Clock, options: Mapping[str, Any]
) -> CameraDriver:
    """Build the driver `name`. Raises `CameraConfigError` for an option it does not know."""
    if name == "fake":
        return _create_fake(clock, options)
    from seeingmon.drivers import create_driver

    return create_driver(name, profile=profile, clock=clock, options=dict(options))


def _create_fake(clock: Clock, options: Mapping[str, Any]) -> CameraDriver:
    rest = dict(options)
    hang_in = rest.pop("hang_in", None)
    unknown = set(rest) - set(FAKE_OPTIONS)
    if unknown:
        raise CameraConfigError(f"the fake driver has no option named {sorted(unknown)[0]!r}")
    kwargs: dict[str, Any] = {}
    for key, value in rest.items():
        kind = FAKE_OPTIONS[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CameraConfigError(f"the option {key!r} of the fake driver must be a number")
        kwargs[key] = kind(value)
    if hang_in is None:
        return FakeCameraDriver(clock, **kwargs)
    if hang_in not in HANGABLE:
        raise CameraConfigError(f"hang_in is one of {', '.join(HANGABLE)}")
    return HangingFakeDriver(clock, hang_in=str(hang_in), **kwargs)
