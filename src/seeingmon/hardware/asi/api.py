"""The ZWO ASI SDK as the camera driver sees it.

This module holds the types that cross the SDK boundary and the `AsiApi` protocol. Two classes
implement the protocol: `CtypesAsiApi` calls the vendor library through `ctypes`, and
`FakeAsiSdk` imitates it in pure Python on a `Clock`. The driver depends on the protocol only,
so tests run the driver on the fake and production runs it on the library.

**Calls.** Each method mirrors one SDK function, with Python types in place of out-parameters.
A method returns when the SDK reports success, and it raises an `AsiError` otherwise. The
subclasses of `AsiError` also derive from the matching `CameraError` class of the driver
contract, so a caller that catches `CameraTimeoutError` also catches an SDK timeout.

**Error codes.** `error_for` maps an SDK error code to an exception class. A code that this
module does not know maps to a plain `AsiError`, so a newer SDK never crashes the mapping.

**Structures.** `AsiCameraInfo`, `AsiControlCaps`, and `AsiRoiFormat` carry the SDK structures
as immutable Python values. `AsiCameraInfo` leaves out the serial number and the camera ID
string on purpose, because the repository and its logs must stay free of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from seeingmon.drivers.base import (
    CameraConfigError,
    CameraDisconnectedError,
    CameraError,
    CameraStateError,
    CameraTimeoutError,
)

# --- Constants ---------------------------------------------------------------------------


class AsiErrorCode(IntEnum):
    """The status that an SDK function returns. `SUCCESS` is 0."""

    SUCCESS = 0
    INVALID_INDEX = 1
    INVALID_ID = 2
    INVALID_CONTROL_TYPE = 3
    CAMERA_CLOSED = 4
    CAMERA_REMOVED = 5
    INVALID_PATH = 6
    INVALID_FILE_FORMAT = 7
    INVALID_SIZE = 8
    INVALID_IMAGE_TYPE = 9
    OUT_OF_BOUNDARY = 10
    TIMEOUT = 11
    INVALID_SEQUENCE = 12
    BUFFER_TOO_SMALL = 13
    VIDEO_MODE_ACTIVE = 14
    EXPOSURE_IN_PROGRESS = 15
    GENERAL_ERROR = 16
    INVALID_MODE = 17
    GPS_NOT_SUPPORTED = 18
    GPS_VERSION_ERROR = 19
    GPS_FPGA_ERROR = 20
    GPS_PARAMETER_OUT_OF_RANGE = 21
    GPS_DATA_INVALID = 22


class AsiImageType(IntEnum):
    """The pixel format of a frame. The driver uses `RAW8` and `RAW16`."""

    RAW8 = 0
    RGB24 = 1
    RAW16 = 2
    Y8 = 3
    END = -1  # ends the list of supported formats


class AsiControl(IntEnum):
    """The adjustable settings and read-only values of a camera, by SDK control number.

    A camera supports a subset. `AsiControlCaps` lists what the connected camera offers.
    """

    GAIN = 0
    EXPOSURE = 1  # in microseconds
    GAMMA = 2
    WB_R = 3
    WB_B = 4
    OFFSET = 5
    BANDWIDTH_OVERLOAD = 6  # the share of the USB bandwidth, in percent
    OVERCLOCK = 7
    TEMPERATURE = 8  # in tenths of a degree Celsius, read-only
    FLIP = 9
    AUTO_MAX_GAIN = 10
    AUTO_MAX_EXPOSURE = 11
    AUTO_TARGET_BRIGHTNESS = 12
    HARDWARE_BIN = 13
    HIGH_SPEED_MODE = 14
    COOLER_POWER_PERCENT = 15
    TARGET_TEMPERATURE = 16
    COOLER_ON = 17
    MONO_BIN = 18
    FAN_ON = 19
    PATTERN_ADJUST = 20
    ANTI_DEW_HEATER = 21


class AsiExposureStatus(IntEnum):
    """The state of a single exposure."""

    IDLE = 0
    WORKING = 1
    SUCCESS = 2
    FAILED = 3


# --- Errors ------------------------------------------------------------------------------

_MEANINGS: dict[int, str] = {
    AsiErrorCode.INVALID_INDEX: "the camera index is out of range",
    AsiErrorCode.INVALID_ID: "the camera ID is not valid",
    AsiErrorCode.INVALID_CONTROL_TYPE: "the camera does not have this control",
    AsiErrorCode.CAMERA_CLOSED: "the camera is not open",
    AsiErrorCode.CAMERA_REMOVED: "the camera is gone",
    AsiErrorCode.INVALID_PATH: "the file path is not valid",
    AsiErrorCode.INVALID_FILE_FORMAT: "the file format is not valid",
    AsiErrorCode.INVALID_SIZE: "the image size is not valid",
    AsiErrorCode.INVALID_IMAGE_TYPE: "the pixel format is not valid",
    AsiErrorCode.OUT_OF_BOUNDARY: "the value or position is out of bounds",
    AsiErrorCode.TIMEOUT: "no frame arrived in time",
    AsiErrorCode.INVALID_SEQUENCE: "the call is out of sequence",
    AsiErrorCode.BUFFER_TOO_SMALL: "the buffer is too small for the frame",
    AsiErrorCode.VIDEO_MODE_ACTIVE: "video capture is active",
    AsiErrorCode.EXPOSURE_IN_PROGRESS: "an exposure is in progress",
    AsiErrorCode.GENERAL_ERROR: "the SDK reports a general error",
    AsiErrorCode.INVALID_MODE: "the camera mode is not valid",
}


class AsiError(CameraError):
    """An SDK call failed. `function` names the call and `code` is the SDK status."""

    def __init__(self, function: str, code: int, message: str | None = None) -> None:
        self.function = function
        self.code = code
        self._message = message
        try:
            name = AsiErrorCode(code).name
        except ValueError:
            name = "UNKNOWN"
        detail = message or _MEANINGS.get(code, "the SDK reports an error")
        super().__init__(f"{function} failed with {name} ({code}): {detail}")

    def __reduce__(self) -> tuple[type[AsiError], tuple[str, int, str | None]]:
        return (type(self), (self.function, self.code, self._message))


class AsiTimeoutError(AsiError, CameraTimeoutError):
    """The SDK waited for a frame and none arrived."""


class AsiDisconnectedError(AsiError, CameraDisconnectedError):
    """The camera is gone or the SDK no longer knows its ID."""


class AsiConfigError(AsiError, CameraConfigError):
    """The SDK rejected a setting, a size, or a position."""


class AsiStateError(AsiError, CameraStateError):
    """The call came in the wrong state, such as an exposure during video capture."""


class AsiLibraryError(CameraError):
    """The vendor library is missing or does not match the binding."""


_CODE_CLASSES: dict[int, type[AsiError]] = {
    AsiErrorCode.TIMEOUT: AsiTimeoutError,
    AsiErrorCode.CAMERA_REMOVED: AsiDisconnectedError,
    AsiErrorCode.INVALID_ID: AsiDisconnectedError,
    AsiErrorCode.INVALID_INDEX: AsiConfigError,
    AsiErrorCode.INVALID_CONTROL_TYPE: AsiConfigError,
    AsiErrorCode.INVALID_PATH: AsiConfigError,
    AsiErrorCode.INVALID_FILE_FORMAT: AsiConfigError,
    AsiErrorCode.INVALID_SIZE: AsiConfigError,
    AsiErrorCode.INVALID_IMAGE_TYPE: AsiConfigError,
    AsiErrorCode.OUT_OF_BOUNDARY: AsiConfigError,
    AsiErrorCode.BUFFER_TOO_SMALL: AsiConfigError,
    AsiErrorCode.INVALID_MODE: AsiConfigError,
    AsiErrorCode.CAMERA_CLOSED: AsiStateError,
    AsiErrorCode.INVALID_SEQUENCE: AsiStateError,
    AsiErrorCode.VIDEO_MODE_ACTIVE: AsiStateError,
    AsiErrorCode.EXPOSURE_IN_PROGRESS: AsiStateError,
}


def error_for(function: str, code: int) -> AsiError:
    """The exception for a failing status. Unknown codes give a plain `AsiError`."""
    return _CODE_CLASSES.get(code, AsiError)(function, code)


def check(function: str, code: int) -> None:
    """Raise the exception for `code`, or return when the call succeeded."""
    if code != AsiErrorCode.SUCCESS:
        raise error_for(function, code)


# --- Structures --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AsiCameraInfo:
    """What the SDK reports about a connected camera.

    `camera_id` is the handle that every other call takes. The SDK numbers cameras from 0 in
    the order that it enumerates them. `max_width` and `max_height` describe the finest readout
    (SDK binning 1). `supported_bins` and `supported_formats` list only the entries that the
    camera supports.
    """

    name: str
    camera_id: int
    max_width: int
    max_height: int
    is_color: bool
    supported_bins: tuple[int, ...]
    supported_formats: tuple[AsiImageType, ...]
    pixel_size_um: float
    is_cooled: bool
    is_usb3_host: bool
    is_usb3_camera: bool
    electrons_per_adu: float
    bit_depth: int


@dataclass(frozen=True, slots=True)
class AsiControlCaps:
    """The range and defaults of one control, from `ASIGetControlCaps`.

    `control` is the SDK control number. It stays an `int`, because a newer camera can report
    a control that `AsiControl` does not list.
    """

    name: str
    description: str
    control: int
    min_value: int
    max_value: int
    default_value: int
    is_auto_supported: bool
    is_writable: bool


@dataclass(frozen=True, slots=True)
class AsiRoiFormat:
    """The ROI size, the binning, and the pixel format that the camera applies."""

    width: int
    height: int
    binning: int
    image_type: AsiImageType


# --- The protocol ------------------------------------------------------------------------


class AsiApi(Protocol):
    """The SDK functions that the driver uses. The methods mirror the vendor functions.

    The driver passes the camera ID that `get_camera_property` returned. The video and
    snapshot calls fill a `bytearray` that the caller sizes for the frame.
    """

    def get_sdk_version(self) -> str:
        """The version string that the library reports."""
        ...

    def get_connected_camera_count(self) -> int:
        """The number of connected cameras."""
        ...

    def get_camera_property(self, index: int) -> AsiCameraInfo:
        """The description of the camera at `index`, counted from 0."""
        ...

    def open_camera(self, camera_id: int) -> None:
        """Open the camera for use."""
        ...

    def init_camera(self, camera_id: int) -> None:
        """Initialize an open camera. Call it once after `open_camera`."""
        ...

    def close_camera(self, camera_id: int) -> None:
        """Close the camera."""
        ...

    def get_control_count(self, camera_id: int) -> int:
        """The number of controls that the camera offers."""
        ...

    def get_control_caps(self, camera_id: int, index: int) -> AsiControlCaps:
        """The caps of the control at `index`, counted from 0."""
        ...

    def get_control_value(self, camera_id: int, control: int) -> tuple[int, bool]:
        """The value of a control, and whether the camera sets it automatically."""
        ...

    def set_control_value(
        self, camera_id: int, control: int, value: int, *, auto: bool = False
    ) -> None:
        """Set a control. `auto` turns on automatic control for the controls that support it."""
        ...

    def set_roi_format(
        self, camera_id: int, width: int, height: int, binning: int, image_type: AsiImageType
    ) -> None:
        """Set the ROI size, binning, and pixel format, and recenter the ROI.

        The camera must not capture. Call `set_start_position` afterward, because this call
        moves the ROI to the center.
        """
        ...

    def get_roi_format(self, camera_id: int) -> AsiRoiFormat:
        """The ROI size, binning, and pixel format that the camera applies."""
        ...

    def set_start_position(self, camera_id: int, x: int, y: int) -> None:
        """Move the ROI origin, in binned pixels. The call also works during video capture."""
        ...

    def get_start_position(self, camera_id: int) -> tuple[int, int]:
        """The ROI origin that the camera applies, in binned pixels."""
        ...

    def get_dropped_frames(self, camera_id: int) -> int:
        """The frames lost since video capture started. Stopping capture resets the count."""
        ...

    def start_video_capture(self, camera_id: int) -> None:
        """Start continuous capture."""
        ...

    def stop_video_capture(self, camera_id: int) -> None:
        """Stop continuous capture. A read that is still blocked cannot be cancelled."""
        ...

    def get_video_data(self, camera_id: int, buffer: bytearray, wait_ms: int) -> None:
        """Fill `buffer` with the next frame, and wait at most `wait_ms` milliseconds for it.

        Raises `AsiTimeoutError` when no frame arrives in time.
        """
        ...

    def start_exposure(self, camera_id: int, *, dark: bool = False) -> None:
        """Start one exposure. `dark` keeps a mechanical shutter closed, where one exists."""
        ...

    def stop_exposure(self, camera_id: int) -> None:
        """Abort the exposure."""
        ...

    def get_exposure_status(self, camera_id: int) -> AsiExposureStatus:
        """The state of the single exposure."""
        ...

    def get_data_after_exposure(self, camera_id: int, buffer: bytearray) -> None:
        """Fill `buffer` with the finished exposure."""
        ...
