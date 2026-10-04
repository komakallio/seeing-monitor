"""The camera driver interface.

A driver wraps one camera backend: `asi` (the vendor SDK), `sim` (synthetic), or `replay`
(recorded). The `acquire` process owns one driver and calls it from a capture thread, so a
driver is synchronous and blocking, and it needs no queue of its own.

**Lifecycle.** `open`, then `configure`, `start`, `read_frame` (repeatedly), `stop`, and
`close`. `configure` stops a running capture first, and it is the only call that changes
geometry or exposure. It must apply the settings, read them back, discard stale frames,
and return what the camera confirmed, because the vendor SDK can change geometry silently.
`move_roi` is the one exception: it moves a running ROI without a restart.

**Bounded waits.** Nothing blocks forever. `read_frame` takes a timeout and raises
`CameraTimeoutError` when it expires. A driver that cannot return from a backend call within its
own bound raises `CameraTimeoutError` too, and `acquire` then climbs the recovery ladder.

**Video and snapshot.** With `StreamKind.VIDEO`, frames flow after `start` until `stop`.
With `StreamKind.SNAPSHOT`, each `start` takes one exposure, and one `read_frame` returns it.
`ActiveStream.frame_period_s` of a snapshot stream is the time from `start` to the frame: the
exposure plus a readout time that can be far longer than a video frame of the same ROI (the `asi`
driver takes it from the snapshot model of the profile). The scheduler and `acquire` derive their
read timeouts from it.

**Factory convention.** `seeingmon.drivers.<name>` defines
`create(*, profile, clock, options) -> CameraDriver`, and `seeingmon.drivers.create_driver`
imports it by name, so adding a driver edits no shared file.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable

from seeingmon.frames import ActiveStream, Frame, PixelFormat, Roi, StreamConfig


class CameraError(Exception):
    """Base class for camera failures."""


class CameraTimeoutError(CameraError):
    """A bounded wait expired: no frame arrived, or a backend call did not return."""


class CameraDisconnectedError(CameraError):
    """The camera is gone, for example after a USB fault."""


class CameraConfigError(CameraError):
    """The camera rejected a setting, or silently applied a different geometry."""


class CameraStateError(CameraError):
    """A call came in the wrong state, such as `read_frame` before `start`."""


class RecoveryLevel(IntEnum):
    """Recovery steps that a driver performs itself, from mildest to most drastic.

    Later steps (restarting the `acquire` process, rebooting, and a hard power cycle) belong
    to the supervisor, not to the driver.
    """

    RESTART_CAPTURE = 1  # stop the reader, restart capture, and count a gap
    REOPEN = 2  # close and reopen the camera, then reapply the stream settings
    USB_RESET = 3  # reset the USB device (sysfs or `USBDEVFS_RESET`)


@dataclass(frozen=True, slots=True)
class CameraInfo:
    """Identity of the connected camera. It carries no serial number, which stays private."""

    model: str
    driver: str
    sdk_version: str | None
    max_width: int  # pixels at the finest readout (SDK bin 1)
    max_height: int
    is_color: bool = False
    has_temperature: bool = False


@dataclass(frozen=True, slots=True)
class CameraCaps:
    """What the camera reports about itself. `acquire` compares this with the profile."""

    gain_range: tuple[int, int]
    exposure_us_range: tuple[int, int]
    bins: tuple[int, ...]  # SDK binning factors
    pixel_formats: tuple[PixelFormat, ...]
    offset_range: tuple[int, int] | None = None
    roi_width_multiple: int = 8
    roi_height_multiple: int = 2
    supports_video: bool = True
    supports_snapshot: bool = True


@runtime_checkable
class CameraDriver(Protocol):
    @property
    def name(self) -> str:
        """The backend name: `asi`, `sim`, `replay`, or a test name."""
        ...

    def open(self) -> CameraInfo:
        """Connect to the camera. Raises `CameraDisconnectedError` when none answers."""
        ...

    def close(self) -> None:
        """Stop any capture and release the camera. Safe to call twice."""
        ...

    def capabilities(self) -> CameraCaps:
        """The limits the camera reports. Valid after `open`."""
        ...

    def configure(self, config: StreamConfig) -> ActiveStream:
        """Stop capture, apply `config`, read it back, and discard stale frames.

        Raises `CameraConfigError` when the camera rejects a setting or the read-back
        differs from the request in a way the driver cannot correct. Every call returns a
        new `stream_id`.
        """
        ...

    def start(self) -> None:
        """Begin capture for the configured stream."""
        ...

    def read_frame(self, timeout_s: float) -> Frame:
        """Block for the next frame, up to `timeout_s`. Raises `CameraTimeoutError` on expiry.

        The driver stamps `t_arrival_ns` right after the backend read returns, and it sets
        `dropped_before` from the backend's counter when it has one (otherwise 0).
        """
        ...

    def stop(self) -> None:
        """Stop capture. Safe to call when already stopped."""
        ...

    def move_roi(self, x: int, y: int) -> Roi:
        """Move the ROI origin while streaming. Returns the ROI the camera applied."""
        ...

    def read_temperature_c(self) -> float | None:
        """The sensor temperature, or `None` when the camera has no sensor."""
        ...

    def dropped_frames(self) -> int:
        """The backend's drop counter since capture started. 0 when it has none."""
        ...

    def recover(self, level: RecoveryLevel) -> None:
        """Perform one recovery step. Raises `CameraError` if the step does not work."""
        ...
