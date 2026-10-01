"""A stand-in for the vendor library: Python functions with the C signatures.

`FakeCLibrary` receives what `ctypes` would pass to a C function (pointers to structures and to
integers, and a buffer) and answers from a `FakeAsiSdk`. A test drives `CtypesAsiApi` through it,
which exercises the argument handling, the structure fields, and the status codes of the binding
without a real library. Each function is a plain Python function, so the binding can set
`argtypes` and `restype` on it, as it does on a real foreign function.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import Any

from seeingmon.hardware.asi.api import AsiError, AsiImageType
from seeingmon.hardware.asi.fake import FakeAsiSdk


class FakeCLibrary:
    def __init__(self, sdk: FakeAsiSdk, *, missing: tuple[str, ...] = ()) -> None:
        self.sdk = sdk
        self.calls: list[str] = []

        def status(name: str, body: Callable[..., None]) -> None:
            if name in missing:
                return

            def c_function(*args: Any) -> int:
                self.calls.append(name)
                try:
                    body(*args)
                except AsiError as error:
                    return int(error.code)
                return 0

            setattr(self, name, c_function)

        def value(name: str, body: Callable[..., Any]) -> None:
            def c_function(*args: Any) -> Any:
                self.calls.append(name)
                return body(*args)

            setattr(self, name, c_function)

        def get_camera_property(info_ptr: Any, index: int) -> None:
            info = sdk.get_camera_property(index)
            out = info_ptr.contents
            out.name = info.name.encode("ascii")
            out.camera_id = info.camera_id
            out.max_width, out.max_height = info.max_width, info.max_height
            out.is_color = int(info.is_color)
            for slot in range(16):
                out.supported_bins[slot] = (
                    info.supported_bins[slot] if slot < len(info.supported_bins) else 0
                )
            formats = [int(f) for f in info.supported_formats] + [int(AsiImageType.END)]
            for slot in range(8):
                out.supported_formats[slot] = formats[slot] if slot < len(formats) else -1
            out.pixel_size_um = info.pixel_size_um
            out.is_cooled = int(info.is_cooled)
            out.is_usb3_host = int(info.is_usb3_host)
            out.is_usb3_camera = int(info.is_usb3_camera)
            out.electrons_per_adu = info.electrons_per_adu
            out.bit_depth = info.bit_depth

        def get_control_count(camera: int, count_ptr: Any) -> None:
            count_ptr.contents.value = sdk.get_control_count(camera)

        def get_control_caps(camera: int, index: int, caps_ptr: Any) -> None:
            caps = sdk.get_control_caps(camera, index)
            out = caps_ptr.contents
            out.name = caps.name.encode("ascii")
            out.description = caps.description.encode("ascii")
            out.max_value, out.min_value, out.default_value = (
                caps.max_value,
                caps.min_value,
                caps.default_value,
            )
            out.is_auto_supported = int(caps.is_auto_supported)
            out.is_writable = int(caps.is_writable)
            out.control_type = caps.control

        def get_control_value(camera: int, control: int, value_ptr: Any, auto_ptr: Any) -> None:
            number, auto = sdk.get_control_value(camera, control)
            value_ptr.contents.value = number
            auto_ptr.contents.value = int(auto)

        def get_roi_format(camera: int, w_ptr: Any, h_ptr: Any, b_ptr: Any, t_ptr: Any) -> None:
            fmt = sdk.get_roi_format(camera)
            w_ptr.contents.value, h_ptr.contents.value = fmt.width, fmt.height
            b_ptr.contents.value, t_ptr.contents.value = fmt.binning, int(fmt.image_type)

        def get_start_position(camera: int, x_ptr: Any, y_ptr: Any) -> None:
            x, y = sdk.get_start_position(camera)
            x_ptr.contents.value, y_ptr.contents.value = x, y

        def get_dropped_frames(camera: int, dropped_ptr: Any) -> None:
            dropped_ptr.contents.value = sdk.get_dropped_frames(camera)

        def get_video_data(camera: int, buffer: Any, size: int, wait_ms: int) -> None:
            data = bytearray(size)
            sdk.get_video_data(camera, data, wait_ms)
            ctypes.memmove(buffer, bytes(data), size)

        def get_data_after_exposure(camera: int, buffer: Any, size: int) -> None:
            data = bytearray(size)
            sdk.get_data_after_exposure(camera, data)
            ctypes.memmove(buffer, bytes(data), size)

        def get_exposure_status(camera: int, status_ptr: Any) -> None:
            status_ptr.contents.value = int(sdk.get_exposure_status(camera))

        value("ASIGetNumOfConnectedCameras", lambda: sdk.get_connected_camera_count())
        value("ASIGetSDKVersion", lambda: sdk.get_sdk_version().encode("ascii"))
        status("ASIGetCameraProperty", get_camera_property)
        status("ASIOpenCamera", lambda camera: sdk.open_camera(camera))
        status("ASIInitCamera", lambda camera: sdk.init_camera(camera))
        status("ASICloseCamera", lambda camera: sdk.close_camera(camera))
        status("ASIGetNumOfControls", get_control_count)
        status("ASIGetControlCaps", get_control_caps)
        status("ASIGetControlValue", get_control_value)
        status(
            "ASISetControlValue",
            lambda camera, control, number, auto: sdk.set_control_value(
                camera, control, number, auto=bool(auto)
            ),
        )
        status(
            "ASISetROIFormat",
            lambda camera, w, h, b, t: sdk.set_roi_format(camera, w, h, b, AsiImageType(t)),
        )
        status("ASIGetROIFormat", get_roi_format)
        status("ASISetStartPos", lambda camera, x, y: sdk.set_start_position(camera, x, y))
        status("ASIGetStartPos", get_start_position)
        status("ASIGetDroppedFrames", get_dropped_frames)
        status("ASIStartVideoCapture", lambda camera: sdk.start_video_capture(camera))
        status("ASIStopVideoCapture", lambda camera: sdk.stop_video_capture(camera))
        status("ASIGetVideoData", get_video_data)
        status("ASIStartExposure", lambda camera, dark: sdk.start_exposure(camera, dark=bool(dark)))
        status("ASIStopExposure", lambda camera: sdk.stop_exposure(camera))
        status("ASIGetExpStatus", get_exposure_status)
        status("ASIGetDataAfterExp", get_data_after_exposure)
