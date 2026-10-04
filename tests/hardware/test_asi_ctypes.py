"""The `ctypes` binding, driven through a Python stand-in for the vendor library."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seeingmon.clock import VirtualClock
from seeingmon.hardware.asi import ctypes_api
from seeingmon.hardware.asi.api import (
    AsiApi,
    AsiConfigError,
    AsiControl,
    AsiDisconnectedError,
    AsiExposureStatus,
    AsiImageType,
    AsiLibraryError,
    AsiTimeoutError,
)
from seeingmon.hardware.asi.ctypes_api import (
    LIBRARY_ENV,
    CtypesAsiApi,
    load_asi_api,
    resolve_library,
)
from seeingmon.hardware.asi.fake import FakeAsiSdk, FakeFrameInfo, default_pixels
from tests.hardware.asi_clib import FakeCLibrary

RAW16 = AsiImageType.RAW16


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock()


@pytest.fixture
def sdk(clock: VirtualClock) -> FakeAsiSdk:
    return FakeAsiSdk(clock)


@pytest.fixture
def library(sdk: FakeAsiSdk) -> FakeCLibrary:
    return FakeCLibrary(sdk)


@pytest.fixture
def api(library: FakeCLibrary) -> CtypesAsiApi:
    return CtypesAsiApi(library)


class TestStructures:
    """The layout of the two structures that the SDK fills.

    The expected offsets follow from the C types: 4-byte `int` and `float`, 8-byte `double`,
    and `long` of 8 bytes on LP64 (Linux) or 4 bytes on Windows.
    """

    long_size = ctypes.sizeof(ctypes.c_long)

    def test_the_camera_info_layout(self) -> None:
        info: Any = ctypes_api._CameraInfoStruct
        assert ctypes.sizeof(info) == (248 if self.long_size == 8 else 240)
        assert info.camera_id.offset == 64
        assert info.max_height.offset == (72 if self.long_size == 8 else 68)
        assert info.supported_bins.size == 16 * 4
        assert info.supported_formats.size == 8 * 4
        assert info.pixel_size_um.offset == (192 if self.long_size == 8 else 184)
        assert info.pixel_size_um.offset % 8 == 0
        assert info.reserved.size == 16

    def test_the_control_caps_layout(self) -> None:
        caps: Any = ctypes_api._ControlCapsStruct
        assert ctypes.sizeof(caps) == (264 if self.long_size == 8 else 248)
        assert caps.max_value.offset == 192
        assert caps.control_type.offset == (224 if self.long_size == 8 else 212)
        assert caps.reserved.size == 32


class TestBinding:
    def test_the_binding_satisfies_the_protocol(self, api: CtypesAsiApi) -> None:
        protocol: AsiApi = api
        assert protocol.get_sdk_version() == "1, 41, 0, 0"

    def test_each_function_gets_its_signature(self, library: FakeCLibrary) -> None:
        CtypesAsiApi(library)
        read: Any = library.ASIGetVideoData  # type: ignore[attr-defined]
        assert read.argtypes[2] is ctypes.c_long
        assert read.argtypes[3] is ctypes.c_int
        assert read.restype is ctypes.c_int
        version: Any = library.ASIGetSDKVersion  # type: ignore[attr-defined]
        assert version.restype is ctypes.c_char_p

    def test_a_library_without_a_function_is_refused(self, sdk: FakeAsiSdk) -> None:
        with pytest.raises(AsiLibraryError, match="ASIGetDroppedFrames"):
            CtypesAsiApi(FakeCLibrary(sdk, missing=("ASIGetDroppedFrames",)))

    def test_enumeration_and_camera_properties(self, api: CtypesAsiApi, sdk: FakeAsiSdk) -> None:
        assert api.get_connected_camera_count() == 1
        info = api.get_camera_property(0)
        assert info == sdk.get_camera_property(0)
        assert info.supported_bins == (1, 2)
        assert info.supported_formats == (AsiImageType.RAW8, AsiImageType.RAW16)
        assert info.name == "ZWO ASI294MM (fake)"
        with pytest.raises(AsiConfigError):
            api.get_camera_property(3)

    def test_controls(self, api: CtypesAsiApi, sdk: FakeAsiSdk, clock: VirtualClock) -> None:
        api.open_camera(0)
        api.init_camera(0)
        count = api.get_control_count(0)
        listed = [api.get_control_caps(0, index) for index in range(count)]
        assert listed == [sdk.get_control_caps(0, index) for index in range(count)]
        gain = next(caps for caps in listed if caps.control == AsiControl.GAIN)
        assert (gain.min_value, gain.max_value, gain.is_writable) == (0, 570, True)
        api.set_control_value(0, AsiControl.GAIN, 120)
        assert api.get_control_value(0, AsiControl.GAIN) == (120, False)
        clock.advance(1.0)
        assert api.get_control_value(0, AsiControl.TEMPERATURE) == (183, False)

    def test_geometry(self, api: CtypesAsiApi) -> None:
        api.open_camera(0)
        api.init_camera(0)
        api.set_roi_format(0, 128, 64, 2, RAW16)
        api.set_start_position(0, 100, 50)
        fmt = api.get_roi_format(0)
        assert (fmt.width, fmt.height, fmt.binning, fmt.image_type) == (128, 64, 2, RAW16)
        assert api.get_start_position(0) == (100, 50)
        with pytest.raises(AsiConfigError) as raised:
            api.set_roi_format(0, 130, 64, 2, RAW16)
        assert raised.value.function == "ASISetROIFormat"

    def test_video_frames_and_the_drop_counter(
        self, api: CtypesAsiApi, sdk: FakeAsiSdk, clock: VirtualClock
    ) -> None:
        api.open_camera(0)
        api.init_camera(0)
        api.set_control_value(0, AsiControl.EXPOSURE, 2000)
        api.set_roi_format(0, 16, 8, 1, RAW16)
        api.set_start_position(0, 8, 4)
        api.start_video_capture(0)
        buffer = bytearray(16 * 8 * 2)
        api.get_video_data(0, buffer, 500)
        expected = default_pixels(FakeFrameInfo(0, 16, 8, 1, RAW16, 8, 4, 2000, 0, 12)) << 4
        np.testing.assert_array_equal(np.frombuffer(buffer, dtype="<u2").reshape(8, 16), expected)
        clock.advance(1.0)
        assert api.get_dropped_frames(0) == sdk.get_dropped_frames(0) > 0
        api.stop_video_capture(0)
        assert api.get_dropped_frames(0) == 0

    def test_a_timeout_status_raises_a_timeout_error(self, api: CtypesAsiApi) -> None:
        api.open_camera(0)
        api.init_camera(0)
        api.set_control_value(0, AsiControl.EXPOSURE, 500_000)
        api.set_roi_format(0, 16, 8, 1, RAW16)
        api.start_video_capture(0)
        with pytest.raises(AsiTimeoutError) as raised:
            api.get_video_data(0, bytearray(256), 100)
        assert raised.value.function == "ASIGetVideoData"

    def test_single_exposures(
        self, api: CtypesAsiApi, clock: VirtualClock, sdk: FakeAsiSdk
    ) -> None:
        api.open_camera(0)
        api.init_camera(0)
        api.set_control_value(0, AsiControl.EXPOSURE, 1_000_000)
        api.set_roi_format(0, 16, 8, 2, RAW16)
        api.start_exposure(0, dark=False)
        assert api.get_exposure_status(0) is AsiExposureStatus.WORKING
        clock.advance(1.01)  # the exposure is over, and the camera still reads the frame out
        assert api.get_exposure_status(0) is AsiExposureStatus.WORKING
        clock.advance(sdk.snapshot_period_s())
        assert api.get_exposure_status(0) is AsiExposureStatus.SUCCESS
        buffer = bytearray(256)
        api.get_data_after_exposure(0, buffer)
        assert any(buffer)
        api.start_exposure(0)
        api.stop_exposure(0)
        assert api.get_exposure_status(0) is AsiExposureStatus.IDLE

    def test_a_removed_camera_raises_a_disconnect_error(
        self, api: CtypesAsiApi, sdk: FakeAsiSdk
    ) -> None:
        api.open_camera(0)
        api.init_camera(0)
        sdk.disconnect()
        with pytest.raises(AsiDisconnectedError):
            api.get_control_count(0)

    def test_an_unknown_format_or_status_does_not_crash_the_binding(
        self, library: FakeCLibrary
    ) -> None:
        def odd_format(camera: int, w: Any, h: Any, b: Any, t_ptr: Any) -> int:
            t_ptr.contents.value = 77
            return 0

        def odd_status(camera: int, status_ptr: Any) -> int:
            status_ptr.contents.value = 99
            return 0

        library.ASIGetROIFormat = odd_format  # type: ignore[attr-defined]
        library.ASIGetExpStatus = odd_status  # type: ignore[attr-defined]
        api = CtypesAsiApi(library)
        assert api.get_roi_format(0).image_type is AsiImageType.END
        assert api.get_exposure_status(0) is AsiExposureStatus.FAILED


class TestLoading:
    def test_a_path_that_you_pass_wins(self, tmp_path: Path) -> None:
        explicit, from_env = tmp_path / "explicit.lib", tmp_path / "env.lib"
        explicit.write_bytes(b"x")
        from_env.write_bytes(b"x")
        found = resolve_library(str(explicit), {LIBRARY_ENV: str(from_env)})
        assert found == str(explicit)

    def test_the_environment_variable_is_next(self, tmp_path: Path) -> None:
        from_env = tmp_path / "env.lib"
        from_env.write_bytes(b"x")
        assert resolve_library(None, {LIBRARY_ENV: str(from_env)}) == str(from_env)

    def test_the_system_search_path_is_last(self) -> None:
        names: list[str] = []

        def finder(name: str) -> str | None:
            names.append(name)
            return "libASICamera2.so"

        assert resolve_library(None, {}, find_library=finder) == "libASICamera2.so"
        assert names == ["ASICamera2"]

    def test_a_missing_library_names_the_two_places_to_set_it(self, tmp_path: Path) -> None:
        with pytest.raises(AsiLibraryError, match=LIBRARY_ENV):
            resolve_library(None, {}, find_library=lambda _: None)
        with pytest.raises(AsiLibraryError, match="library_path"):
            resolve_library(str(tmp_path / "missing.lib"), {})

    def test_load_wraps_what_the_loader_returns(self, tmp_path: Path, sdk: FakeAsiSdk) -> None:
        location = tmp_path / "asi.lib"
        location.write_bytes(b"x")
        loaded: list[str] = []

        def loader(path: str) -> FakeCLibrary:
            loaded.append(path)
            return FakeCLibrary(sdk)

        api = load_asi_api(str(location), {}, loader=loader)
        assert loaded == [str(location)]
        assert api.get_connected_camera_count() == 1

    def test_a_loader_failure_becomes_a_library_error_without_the_path(
        self, tmp_path: Path
    ) -> None:
        location = tmp_path / "asi.lib"
        location.write_bytes(b"x")

        def loader(path: str) -> object:
            raise OSError(f"cannot load {path}")

        with pytest.raises(AsiLibraryError) as raised:
            load_asi_api(str(location), {}, loader=loader)
        assert str(location) not in str(raised.value)
