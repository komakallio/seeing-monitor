"""The ZWO ASI SDK behind a protocol: types, a `ctypes` binding, a fake, and fault handling.

`seeingmon.hardware.asi.api` defines `AsiApi`. The vendor library implements it
(`seeingmon.hardware.asi.ctypes_api`), and a pure-Python fake implements it
(`seeingmon.hardware.asi.fake`), so the driver in `seeingmon.drivers.asi` runs on either.
This package imports no vendor library and no platform-specific module until you use one.
"""

from __future__ import annotations

from seeingmon.hardware.asi.api import (
    AsiApi,
    AsiCameraInfo,
    AsiConfigError,
    AsiControl,
    AsiControlCaps,
    AsiDisconnectedError,
    AsiError,
    AsiErrorCode,
    AsiExposureStatus,
    AsiImageType,
    AsiLibraryError,
    AsiRoiFormat,
    AsiStateError,
    AsiTimeoutError,
)

__all__ = [
    "AsiApi",
    "AsiCameraInfo",
    "AsiConfigError",
    "AsiControl",
    "AsiControlCaps",
    "AsiDisconnectedError",
    "AsiError",
    "AsiErrorCode",
    "AsiExposureStatus",
    "AsiImageType",
    "AsiLibraryError",
    "AsiRoiFormat",
    "AsiStateError",
    "AsiTimeoutError",
]
