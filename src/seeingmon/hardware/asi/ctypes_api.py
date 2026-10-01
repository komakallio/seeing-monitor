"""The ZWO ASI SDK through `ctypes`.

`CtypesAsiApi` implements `AsiApi` on the vendor shared library. The vendor library, its header,
and its binaries never enter this repository. This module describes the public C interface in its
own words: the function names, the argument types, and the layout of the two structures that the
SDK fills. You tell the binding where the library lives, and it loads the library when you ask.

**Where the library lives.** `resolve_library` looks in three places, in this order: the path that
you pass (the `library_path` driver option), the environment variable `SEEINGMON_ASI__LIBRARY_PATH`
(the same key in the configuration scheme), and the system search path (`ASICamera2`). The
installer takes the SDK archive from a path that you give it, installs the library privately,
and sets the path.

**Types.** C `long` is 32 bits on Windows and 64 bits on Linux, so the binding uses `c_long` where
the interface says `long`. The camera ID, the sizes, and the binning are `int`. Status codes are
`int`, and each method turns a failing status into an exception (see `seeingmon.hardware.asi.api`).

**Threads.** `ctypes` releases the GIL during a foreign call, so one thread can block in
`get_video_data` while another thread runs. The binding keeps no state between calls.

**Verification.** The layout of the structures comes from the public interface description and is
checked here only for size and field offsets. A real library on real hardware is the final check
(see `docs/hardware-checks.md`).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from collections.abc import Callable, Mapping
from typing import Any

from seeingmon.hardware.asi.api import (
    AsiCameraInfo,
    AsiControlCaps,
    AsiExposureStatus,
    AsiImageType,
    AsiLibraryError,
    AsiRoiFormat,
    check,
)

LIBRARY_ENV = "SEEINGMON_ASI__LIBRARY_PATH"
LIBRARY_NAME = "ASICamera2"

_NAME_BYTES = 64
_DESCRIPTION_BYTES = 128
_MAX_BINS = 16
_MAX_FORMATS = 8


class _CameraInfoStruct(ctypes.Structure):
    """The camera description that `ASIGetCameraProperty` fills."""

    _fields_ = [
        ("name", ctypes.c_char * _NAME_BYTES),
        ("camera_id", ctypes.c_int),
        ("max_height", ctypes.c_long),
        ("max_width", ctypes.c_long),
        ("is_color", ctypes.c_int),
        ("bayer_pattern", ctypes.c_int),
        ("supported_bins", ctypes.c_int * _MAX_BINS),
        ("supported_formats", ctypes.c_int * _MAX_FORMATS),
        ("pixel_size_um", ctypes.c_double),
        ("mechanical_shutter", ctypes.c_int),
        ("st4_port", ctypes.c_int),
        ("is_cooled", ctypes.c_int),
        ("is_usb3_host", ctypes.c_int),
        ("is_usb3_camera", ctypes.c_int),
        ("electrons_per_adu", ctypes.c_float),
        ("bit_depth", ctypes.c_int),
        ("is_trigger_camera", ctypes.c_int),
        ("reserved", ctypes.c_char * 16),
    ]


class _ControlCapsStruct(ctypes.Structure):
    """The description of one control that `ASIGetControlCaps` fills."""

    _fields_ = [
        ("name", ctypes.c_char * _NAME_BYTES),
        ("description", ctypes.c_char * _DESCRIPTION_BYTES),
        ("max_value", ctypes.c_long),
        ("min_value", ctypes.c_long),
        ("default_value", ctypes.c_long),
        ("is_auto_supported", ctypes.c_int),
        ("is_writable", ctypes.c_int),
        ("control_type", ctypes.c_int),
        ("reserved", ctypes.c_char * 32),
    ]


_int = ctypes.c_int
_long = ctypes.c_long
_int_ptr = ctypes.POINTER(ctypes.c_int)
_long_ptr = ctypes.POINTER(ctypes.c_long)
_buffer = ctypes.c_void_p

# The C functions that the binding calls: (return type, argument types). Every function returns a
# status code except the two that this table marks with another return type.
_FUNCTIONS: dict[str, tuple[Any, list[Any]]] = {
    "ASIGetNumOfConnectedCameras": (_int, []),
    "ASIGetSDKVersion": (ctypes.c_char_p, []),
    "ASIGetCameraProperty": (_int, [ctypes.POINTER(_CameraInfoStruct), _int]),
    "ASIOpenCamera": (_int, [_int]),
    "ASIInitCamera": (_int, [_int]),
    "ASICloseCamera": (_int, [_int]),
    "ASIGetNumOfControls": (_int, [_int, _int_ptr]),
    "ASIGetControlCaps": (_int, [_int, _int, ctypes.POINTER(_ControlCapsStruct)]),
    "ASIGetControlValue": (_int, [_int, _int, _long_ptr, _int_ptr]),
    "ASISetControlValue": (_int, [_int, _int, _long, _int]),
    "ASISetROIFormat": (_int, [_int, _int, _int, _int, _int]),
    "ASIGetROIFormat": (_int, [_int, _int_ptr, _int_ptr, _int_ptr, _int_ptr]),
    "ASISetStartPos": (_int, [_int, _int, _int]),
    "ASIGetStartPos": (_int, [_int, _int_ptr, _int_ptr]),
    "ASIGetDroppedFrames": (_int, [_int, _int_ptr]),
    "ASIStartVideoCapture": (_int, [_int]),
    "ASIStopVideoCapture": (_int, [_int]),
    "ASIGetVideoData": (_int, [_int, _buffer, _long, _int]),
    "ASIStartExposure": (_int, [_int, _int]),
    "ASIStopExposure": (_int, [_int]),
    "ASIGetExpStatus": (_int, [_int, _int_ptr]),
    "ASIGetDataAfterExp": (_int, [_int, _buffer, _long]),
}


def _text(raw: bytes) -> str:
    """Decode a C string. The SDK pads fixed-size fields with zero bytes."""
    return raw.split(b"\0", 1)[0].decode("ascii", errors="replace")


def _until(values: Any, stop: Callable[[int], bool]) -> list[int]:
    found: list[int] = []
    for value in values:
        if stop(value):
            break
        found.append(int(value))
    return found


class CtypesAsiApi:
    """`AsiApi` on a loaded vendor library.

    Pass a `ctypes` library object (see `load_asi_api`). The constructor declares the signature of
    each function, and it raises `AsiLibraryError` when the library lacks one.
    """

    def __init__(self, library: Any) -> None:
        self._lib = library
        for name, (restype, argtypes) in _FUNCTIONS.items():
            try:
                function = getattr(library, name)
            except AttributeError:
                raise AsiLibraryError(f"the ASI library has no function {name}") from None
            function.restype = restype
            function.argtypes = argtypes

    def _call(self, name: str, *args: Any) -> None:
        check(name, getattr(self._lib, name)(*args))

    # --- Enumeration and lifecycle ---

    def get_sdk_version(self) -> str:
        raw = self._lib.ASIGetSDKVersion()
        return "" if raw is None else bytes(raw).decode("ascii", errors="replace")

    def get_connected_camera_count(self) -> int:
        return int(self._lib.ASIGetNumOfConnectedCameras())

    def get_camera_property(self, index: int) -> AsiCameraInfo:
        info = _CameraInfoStruct()
        self._call("ASIGetCameraProperty", ctypes.pointer(info), index)
        formats = []
        for value in _until(info.supported_formats, lambda v: v == AsiImageType.END):
            try:
                formats.append(AsiImageType(value))
            except ValueError:
                continue  # a format that this binding does not know
        return AsiCameraInfo(
            name=_text(info.name),
            camera_id=info.camera_id,
            max_width=info.max_width,
            max_height=info.max_height,
            is_color=bool(info.is_color),
            supported_bins=tuple(_until(info.supported_bins, lambda v: v == 0)),
            supported_formats=tuple(formats),
            pixel_size_um=info.pixel_size_um,
            is_cooled=bool(info.is_cooled),
            is_usb3_host=bool(info.is_usb3_host),
            is_usb3_camera=bool(info.is_usb3_camera),
            electrons_per_adu=info.electrons_per_adu,
            bit_depth=info.bit_depth,
        )

    def open_camera(self, camera_id: int) -> None:
        self._call("ASIOpenCamera", camera_id)

    def init_camera(self, camera_id: int) -> None:
        self._call("ASIInitCamera", camera_id)

    def close_camera(self, camera_id: int) -> None:
        self._call("ASICloseCamera", camera_id)

    # --- Controls ---

    def get_control_count(self, camera_id: int) -> int:
        count = ctypes.c_int()
        self._call("ASIGetNumOfControls", camera_id, ctypes.pointer(count))
        return count.value

    def get_control_caps(self, camera_id: int, index: int) -> AsiControlCaps:
        caps = _ControlCapsStruct()
        self._call("ASIGetControlCaps", camera_id, index, ctypes.pointer(caps))
        return AsiControlCaps(
            name=_text(caps.name),
            description=_text(caps.description),
            control=caps.control_type,
            min_value=caps.min_value,
            max_value=caps.max_value,
            default_value=caps.default_value,
            is_auto_supported=bool(caps.is_auto_supported),
            is_writable=bool(caps.is_writable),
        )

    def get_control_value(self, camera_id: int, control: int) -> tuple[int, bool]:
        value, auto = ctypes.c_long(), ctypes.c_int()
        self._call(
            "ASIGetControlValue", camera_id, control, ctypes.pointer(value), ctypes.pointer(auto)
        )
        return value.value, bool(auto.value)

    def set_control_value(
        self, camera_id: int, control: int, value: int, *, auto: bool = False
    ) -> None:
        self._call("ASISetControlValue", camera_id, control, value, int(auto))

    # --- Geometry ---

    def set_roi_format(
        self, camera_id: int, width: int, height: int, binning: int, image_type: AsiImageType
    ) -> None:
        self._call("ASISetROIFormat", camera_id, width, height, binning, int(image_type))

    def get_roi_format(self, camera_id: int) -> AsiRoiFormat:
        width, height, binning, image_type = (ctypes.c_int() for _ in range(4))
        self._call(
            "ASIGetROIFormat",
            camera_id,
            ctypes.pointer(width),
            ctypes.pointer(height),
            ctypes.pointer(binning),
            ctypes.pointer(image_type),
        )
        try:
            kind = AsiImageType(image_type.value)
        except ValueError:
            kind = AsiImageType.END  # a format that this binding does not know
        return AsiRoiFormat(width.value, height.value, binning.value, kind)

    def set_start_position(self, camera_id: int, x: int, y: int) -> None:
        self._call("ASISetStartPos", camera_id, x, y)

    def get_start_position(self, camera_id: int) -> tuple[int, int]:
        x, y = ctypes.c_int(), ctypes.c_int()
        self._call("ASIGetStartPos", camera_id, ctypes.pointer(x), ctypes.pointer(y))
        return x.value, y.value

    # --- Video ---

    def get_dropped_frames(self, camera_id: int) -> int:
        dropped = ctypes.c_int()
        self._call("ASIGetDroppedFrames", camera_id, ctypes.pointer(dropped))
        return dropped.value

    def start_video_capture(self, camera_id: int) -> None:
        self._call("ASIStartVideoCapture", camera_id)

    def stop_video_capture(self, camera_id: int) -> None:
        self._call("ASIStopVideoCapture", camera_id)

    def get_video_data(self, camera_id: int, buffer: bytearray, wait_ms: int) -> None:
        view = (ctypes.c_char * len(buffer)).from_buffer(buffer)
        self._call("ASIGetVideoData", camera_id, view, len(buffer), wait_ms)

    # --- Single exposures ---

    def start_exposure(self, camera_id: int, *, dark: bool = False) -> None:
        self._call("ASIStartExposure", camera_id, int(dark))

    def stop_exposure(self, camera_id: int) -> None:
        self._call("ASIStopExposure", camera_id)

    def get_exposure_status(self, camera_id: int) -> AsiExposureStatus:
        status = ctypes.c_int()
        self._call("ASIGetExpStatus", camera_id, ctypes.pointer(status))
        try:
            return AsiExposureStatus(status.value)
        except ValueError:
            return AsiExposureStatus.FAILED  # a state that this binding does not know

    def get_data_after_exposure(self, camera_id: int, buffer: bytearray) -> None:
        view = (ctypes.c_char * len(buffer)).from_buffer(buffer)
        self._call("ASIGetDataAfterExp", camera_id, view, len(buffer))


# --- Loading the library -----------------------------------------------------------------


def _is_windows() -> bool:
    """Whether this is Windows. A function, so the type checker does not fold the platform test."""
    return sys.platform == "win32"


def resolve_library(
    path: str | None = None,
    env: Mapping[str, str] | None = None,
    *,
    find_library: Callable[[str], str | None] = ctypes.util.find_library,
) -> str:
    """The location of the vendor library, from the path, the environment, or the system.

    Raises `AsiLibraryError` when none of them names a library. A path that you give must be an
    existing file, so a typo fails here and not deep inside the loader.
    """
    environment = os.environ if env is None else env
    configured = path or environment.get(LIBRARY_ENV) or None
    if configured is not None:
        if not os.path.isfile(configured):
            raise AsiLibraryError(
                "the configured ASI library does not exist: check the library_path option "
                f"or {LIBRARY_ENV}"
            )
        return configured
    found = find_library(LIBRARY_NAME)
    if found is None:
        raise AsiLibraryError(
            f"the ASI library is not installed or not on the search path: set the library_path "
            f"option or {LIBRARY_ENV}"
        )
    return found


def load_asi_api(
    path: str | None = None,
    env: Mapping[str, str] | None = None,
    *,
    loader: Callable[[str], Any] | None = None,
) -> CtypesAsiApi:
    """Load the vendor library and wrap it. Raises `AsiLibraryError` when that fails.

    `loader` replaces `ctypes.CDLL`, so a test loads a stand-in library.
    """
    location = resolve_library(path, env)
    add_dll_directory = getattr(os, "add_dll_directory", None)  # Windows only
    if _is_windows() and add_dll_directory is not None and os.path.isabs(location):
        add_dll_directory(os.path.dirname(location))  # the library finds its own dependencies
    try:
        library = (loader or ctypes.CDLL)(location)
    except OSError as error:
        raise AsiLibraryError(
            f"the ASI library did not load ({error.__class__.__name__})"
        ) from None
    return CtypesAsiApi(library)
