"""The `asi` driver: a ZWO ASI camera through the vendor SDK.

`AsiDriver` implements `CameraDriver` on an `AsiApi`, which is the `ctypes` binding in production
and `FakeAsiSdk` in tests. Build it with `create` (see `seeingmon.drivers.asi`).

**Four rules** keep the closed SDK from stalling the system (see `docs/architecture.md`, "Camera
access"):

1. *One owner.* The driver is the only code that calls the SDK. A lock serializes the calls, except
   the blocking frame read, so `move_roi`, `read_temperature_c`, and `dropped_frames` run while a
   read waits.
2. *Time stamps.* `read_frame` stamps `t_arrival_ns` with `Clock.utc_ns` in the statement that
   follows the SDK read. The SDK returns no frame time, so `t_utc_ns` is the arrival time minus
   the expected frame period plus half the exposure (`TimeQuality.ESTIMATED`). The analysis fits
   a line through the arrival times, and commissioning measures the latency.
3. *Bounded waits.* A frame read waits at most twice the longer of the exposure and the expected
   frame period, plus 500 ms, and never longer than the caller's timeout. A blocked read cannot be
   cancelled, so `stop`, `configure`, `recover`, and `close` wait for the reader to return before
   they call `StopVideoCapture`. A `CallWatchdog` guards every SDK call and reports one that
   outlives its deadline. In production that ends the process.
4. *One mode-change function.* `configure` stops capture, sets the controls, sets the ROI and
   binning, sets the start position, reads the geometry back, and drops stale state. The SDK can
   change geometry silently, so the function compares the read-back with the request, corrects a
   mismatch once, and raises `CameraConfigError` when it persists. `start` restarts capture. While
   a stream runs, the driver checks the geometry every `geometry_check_interval` frames and stops
   with `CameraConfigError` when it differs.

**Frames.** A `RAW8` buffer becomes a `uint8` array, and a `RAW16` buffer becomes a `uint16` array
with the ADC value in the high bits, as the SDK delivers it. `dropped_before` is the change of the
SDK drop counter since the previous frame. The first frame after a recovery step carries
`FrameFlag.RECOVERED`. A frame carries `FrameFlag.TIME_INVALID` and `TimeQuality.INVALID` while the
clock reports that it is not synchronized.

**Recovery.** `recover` climbs the first three steps of the ladder (`RecoveryLevel`): restart
capture, close and reopen the camera, and reset the USB device. Each step reapplies the stream
settings and leaves the driver ready to read. Restarting the process, rebooting, and the power
cycle belong to the supervisor.

**Threads.** Any thread can call any method. One thread at a time can read frames, and a second
`read_frame` raises `CameraStateError`. The calls that stop capture wait for an in-flight read.

**Not verified on hardware.** The driver passes the lifecycle tests of `FakeCameraDriver` on the
fake SDK. The behavior of the real SDK that the research notes describe is modeled in the fake, and
the checks in `docs/hardware-checks.md` confirm it on a camera.
"""

from __future__ import annotations

import logging
import math
import threading
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import NoReturn

import numpy as np

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.drivers.asi.options import AsiOptions
from seeingmon.drivers.base import (
    CameraCaps,
    CameraConfigError,
    CameraDisconnectedError,
    CameraError,
    CameraInfo,
    CameraStateError,
    CameraTimeoutError,
    RecoveryLevel,
)
from seeingmon.frames import (
    ActiveStream,
    Frame,
    FrameData,
    FrameFlag,
    PixelFormat,
    Roi,
    StreamConfig,
    StreamKind,
    TimeQuality,
)
from seeingmon.hardware.asi.api import (
    AsiApi,
    AsiCameraInfo,
    AsiConfigError,
    AsiControl,
    AsiControlCaps,
    AsiError,
    AsiExposureStatus,
    AsiImageType,
)
from seeingmon.hardware.asi.usb import UsbResetter
from seeingmon.hardware.asi.watchdog import CallWatchdog
from seeingmon.hardware.events import EventCallback, HardwareEvent, emit
from seeingmon.profile import derived
from seeingmon.profile.errors import ProfileError
from seeingmon.profile.models import Profile, ReadoutMode

_log = logging.getLogger(__name__)

VENDOR_WAIT_FLOOR_S = 0.5  # the vendor advice: twice the exposure, plus this much
SNAPSHOT_POLL_MAX_S = 0.25  # the longest sleep between two status polls of a single exposure
REOPEN_POLL_S = 0.25  # how often the USB reset step checks whether the camera is back
EXPOSURE_TOLERANCE = 0.01  # a read-back exposure within 1% of the request counts as applied


@dataclass(slots=True)
class _Plan:
    """A validated request: the readout mode and the ROI after the ROI rules."""

    config: StreamConfig
    mode: ReadoutMode
    image_type: AsiImageType
    roi: Roi


@dataclass(slots=True)
class _Stream:
    """The stream that the camera runs: what it confirmed, and what the reader needs."""

    stream_id: int
    request: StreamConfig  # the request, which a recovery step applies again
    config: StreamConfig  # as applied
    mode: ReadoutMode
    image_type: AsiImageType
    roi: Roi
    adc_bits: int
    period_s: float
    buffer: bytearray
    seq: int = 0

    @property
    def frame_bytes(self) -> int:
        return len(self.buffer)


class AsiDriver:
    """A `CameraDriver` for a ZWO ASI camera.

    Args:
        api: The SDK. Pass `CtypesAsiApi` in production and `FakeAsiSdk` in tests.
        profile: The hardware profile. The driver reads the readout modes, the ROI rules, and the
            timing from it.
        clock: The time source for time stamps, waits, and deadlines.
        options: The driver options. The defaults apply when you pass none.
        usb_resetter: Resets the USB device in the third recovery step. Without one, that step
            raises `CameraError`.
        watchdog: Guards every SDK call. Without one, calls run unguarded.
        watchdog_thread: Start and stop the watchdog's polling thread with `open` and `close`.
            Leave it off for a `VirtualClock`.
        on_event: Receives notable occurrences, such as a corrected geometry.
    """

    def __init__(
        self,
        *,
        api: AsiApi,
        profile: Profile,
        clock: Clock,
        options: AsiOptions | None = None,
        usb_resetter: UsbResetter | None = None,
        watchdog: CallWatchdog | None = None,
        watchdog_thread: bool = False,
        on_event: EventCallback | None = None,
    ) -> None:
        self._api = api
        self._profile = profile
        self._clock = clock
        self._opts = options or AsiOptions()
        self._resetter = usb_resetter
        self._watchdog = watchdog
        self._watchdog_thread = watchdog_thread
        self._on_event = on_event
        # State. `_lock` guards it, except the fields that only the reader thread touches.
        self._lock = threading.RLock()
        self._open = False
        self._wanted_open = False  # the caller opened the camera and has not closed it
        self._camera_id = 0
        self._info: AsiCameraInfo | None = None
        self._sdk_version: str | None = None
        self._caps: dict[int, AsiControlCaps] = {}
        self._init_ns = 0
        self._stream: _Stream | None = None
        self._next_stream_id = 0
        self._capturing = False  # SDK video capture runs
        self._exposing = False  # a single exposure is in flight
        self._intent_running = False  # the caller started the stream and has not stopped it
        self._stopping = False  # a stop is waiting for the reader
        self._reader_busy = False
        self._reader_idle = threading.Event()
        self._reader_idle.set()
        self._read_bound_s = 0.0
        self._counter_base = 0
        self._last_counter = 0
        self._discard_left = 0
        self._since_check = 0
        self._pending_flags = FrameFlag.NONE
        self._exposure_started_utc_ns = 0
        self._temperature_c: float | None = None
        self._temperature_ns: int | None = None
        self._time_checked_ns: int | None = None
        self._time_quality = TimeQuality.ESTIMATED
        self._time_error_ns = 0

    # --- Plumbing ---

    @property
    def name(self) -> str:
        return "asi"

    def _guard(self, call: str, timeout_s: float | None = None) -> AbstractContextManager[None]:
        if self._watchdog is None:
            return nullcontext()
        return self._watchdog.guard(
            call, self._opts.call_timeout_s if timeout_s is None else timeout_s
        )

    def _emit(self, level: str, kind: str, message: str, **detail: object) -> None:
        event = HardwareEvent(level, kind, message, self._clock.utc_ns(), detail or None)
        emit(self._on_event, event)

    def _require_open(self) -> AsiCameraInfo:
        if not self._open or self._info is None:
            raise CameraStateError("the camera is not open")
        return self._info

    # --- Open and close ---

    def open(self) -> CameraInfo:
        with self._lock:
            if self._open and self._info is not None:
                return self._camera_info()
            self._open_camera()
            self._wanted_open = True
            if self._watchdog is not None and self._watchdog_thread:
                self._watchdog.start()
            self._check_against_profile()
            return self._camera_info()

    def _camera_info(self) -> CameraInfo:
        info = self._require_open()
        return CameraInfo(
            model=info.name,
            driver=self.name,
            sdk_version=self._sdk_version,
            max_width=info.max_width,
            max_height=info.max_height,
            is_color=info.is_color,
            has_temperature=int(AsiControl.TEMPERATURE) in self._caps,
        )

    def _open_camera(self) -> None:
        """Find the camera, open and initialize it, and read its controls."""
        api = self._api
        with self._guard("get_connected_camera_count"):
            count = api.get_connected_camera_count()
        if count <= 0:
            raise CameraDisconnectedError("no ASI camera is connected")
        if self._opts.camera_index >= count:
            raise CameraDisconnectedError(
                f"camera index {self._opts.camera_index} is not connected ({count} found)"
            )
        with self._guard("get_camera_property"):
            info = api.get_camera_property(self._opts.camera_index)
        with self._guard("open_camera"):
            api.open_camera(info.camera_id)
        try:
            with self._guard("init_camera"):
                api.init_camera(info.camera_id)
            self._init_ns = self._clock.monotonic_ns()
            self._camera_id = info.camera_id
            self._caps = self._load_caps(info.camera_id)
            self._quiesce(info.camera_id)
        except CameraError:
            self._close_quietly(info.camera_id)
            raise
        if self._sdk_version is None:
            with self._guard("get_sdk_version"):
                self._sdk_version = api.get_sdk_version() or None
        self._info = info
        self._open = True
        self._temperature_c = None
        self._temperature_ns = None

    def _quiesce(self, camera_id: int) -> None:
        """Stop a capture or an exposure that a previous process left running.

        A process that the watchdog ended cannot stop the camera, and `set_roi_format` needs a
        camera that does not capture. Errors do not matter here, because an idle camera may
        refuse the calls.
        """
        for call, function in (
            ("stop_video_capture", self._api.stop_video_capture),
            ("stop_exposure", self._api.stop_exposure),
        ):
            try:
                with self._guard(call):
                    function(camera_id)
            except AsiError:
                _log.debug("%s failed while quieting a new camera", call, exc_info=True)

    def _load_caps(self, camera_id: int) -> dict[int, AsiControlCaps]:
        api = self._api
        with self._guard("get_control_count"):
            count = api.get_control_count(camera_id)
        caps: dict[int, AsiControlCaps] = {}
        for index in range(count):
            with self._guard("get_control_caps"):
                entry = api.get_control_caps(camera_id, index)
            caps[entry.control] = entry
        for control in (AsiControl.GAIN, AsiControl.EXPOSURE):
            if control not in caps:
                raise CameraError(f"the camera has no {control.name.lower()} control")
        return caps

    def _check_against_profile(self) -> None:
        """Report a camera that does not match the profile. `acquire` decides what to do."""
        info = self._require_open()
        finest = next((m for m in self._profile.readout_modes if m.sdk_bin == 1), None)
        if finest is not None and (info.max_width, info.max_height) != (
            finest.width_px,
            finest.height_px,
        ):
            self._emit(
                "warning",
                "camera.profile_mismatch",
                "The camera reports a different sensor size than the profile.",
                camera_size=[info.max_width, info.max_height],
                profile_size=[finest.width_px, finest.height_px],
            )
        has_sensor = int(AsiControl.TEMPERATURE) in self._caps
        if has_sensor != self._profile.sensor.has_temperature_sensor:
            self._emit(
                "warning",
                "camera.profile_mismatch",
                "The camera and the profile disagree about the temperature sensor.",
                camera_has_sensor=has_sensor,
            )

    def _close_quietly(self, camera_id: int) -> None:
        try:
            with self._guard("close_camera"):
                self._api.close_camera(camera_id)
        except AsiError:
            _log.debug("close_camera failed while cleaning up", exc_info=True)

    def close(self) -> None:
        reader_hung = False
        try:
            self._stop_capture()
        except CameraTimeoutError:
            reader_hung = True
            _log.warning("the reader did not return while closing, so the camera stays open")
        except CameraError:
            _log.warning("stopping capture failed while closing", exc_info=True)
        with self._lock:
            if self._open and not reader_hung:
                self._close_quietly(self._camera_id)
            self._open = False
            self._wanted_open = False
            self._stream = None
            self._intent_running = False
            self._capturing = False
            self._exposing = False
            self._info = None
            self._caps = {}
        if self._watchdog is not None and self._watchdog_thread:
            self._watchdog.stop()

    def capabilities(self) -> CameraCaps:
        with self._lock:
            info = self._require_open()
            gain, exposure = self._caps[AsiControl.GAIN], self._caps[AsiControl.EXPOSURE]
            offset = self._caps.get(AsiControl.OFFSET)
            formats = tuple(
                fmt
                for fmt, native in (
                    (PixelFormat.RAW8, AsiImageType.RAW8),
                    (PixelFormat.RAW16, AsiImageType.RAW16),
                )
                if native in info.supported_formats
            )
            limits = self._profile.limits
            return CameraCaps(
                gain_range=(gain.min_value, gain.max_value),
                exposure_us_range=(exposure.min_value, exposure.max_value),
                bins=info.supported_bins,
                pixel_formats=formats,
                offset_range=None if offset is None else (offset.min_value, offset.max_value),
                roi_width_multiple=limits.roi_width_multiple,
                roi_height_multiple=limits.roi_height_multiple,
            )

    # --- Stopping ---

    def _stop_capture(self) -> None:
        """Wait for the reader, then stop video capture and any exposure.

        A reader that does not return in time raises `CameraTimeoutError`, and the SDK keeps
        running, because stopping capture under a blocked read can crash the SDK.
        """
        with self._lock:
            self._stopping = True
            wait_s = self._read_bound_s + self._opts.call_timeout_s
        try:
            # A real-time wait on another thread. It cannot run on the `Clock`.
            if not self._reader_idle.wait(timeout=wait_s):
                raise CameraTimeoutError(
                    "the reader did not return from the SDK; the call may be hung"
                )
            with self._lock:
                self._stop_sdk()
        finally:
            with self._lock:
                self._stopping = False

    def _stop_sdk(self) -> None:
        """Stop capture and exposure in the SDK. The caller holds the lock and owns the reader."""
        if self._capturing:
            self._capturing = False
            try:
                with self._guard("stop_video_capture"):
                    self._api.stop_video_capture(self._camera_id)
            except AsiError as error:
                _log.debug("stop_video_capture failed: %s", error)
                if not isinstance(error, CameraDisconnectedError):
                    raise
        if self._exposing:
            self._exposing = False
            try:
                with self._guard("stop_exposure"):
                    self._api.stop_exposure(self._camera_id)
            except AsiError as error:
                _log.debug("stop_exposure failed: %s", error)
                if not isinstance(error, CameraDisconnectedError):
                    raise

    def stop(self) -> None:
        with self._lock:
            if not self._open:
                self._intent_running = False
                return
        self._stop_capture()
        with self._lock:
            self._intent_running = False

    # --- Configure ---

    def configure(self, config: StreamConfig) -> ActiveStream:
        with self._lock:
            self._require_open()
        self._stop_capture()
        with self._lock:
            self._intent_running = False
            self._stream = None
            plan = self._plan(config)
            stream = self._build_stream(plan, request=config)
            return self._active_stream(stream)

    def _plan(self, config: StreamConfig) -> _Plan:
        info = self._require_open()
        try:
            mode = self._profile.mode(config.mode, high_speed=config.high_speed)
        except ProfileError as error:
            raise CameraConfigError(str(error)) from None
        if mode.sdk_bin not in info.supported_bins:
            raise CameraConfigError(f"the camera does not support binning {mode.sdk_bin}")
        image_type = (
            AsiImageType.RAW16 if config.pixel_format is PixelFormat.RAW16 else AsiImageType.RAW8
        )
        if image_type not in info.supported_formats:
            raise CameraConfigError(f"the camera does not support {config.pixel_format.name}")
        limits = self._profile.limits
        wanted = config.roi or Roi(0, 0, mode.width_px, mode.height_px)
        width = max(
            limits.roi_width_multiple,
            wanted.width // limits.roi_width_multiple * limits.roi_width_multiple,
        )
        height = max(
            limits.roi_height_multiple,
            wanted.height // limits.roi_height_multiple * limits.roi_height_multiple,
        )
        if width > mode.width_px or height > mode.height_px:
            raise CameraConfigError("the ROI is larger than the frame")
        roi = Roi(
            min(wanted.x, mode.width_px - width),
            min(wanted.y, mode.height_px - height),
            width,
            height,
        )
        return _Plan(config, mode, image_type, roi)

    def _build_stream(self, plan: _Plan, *, request: StreamConfig) -> _Stream:
        """Run the mode-change function and install the stream that the camera confirmed."""
        applied, roi = self._apply_stream(plan)
        bytes_per_pixel = 2 if plan.image_type is AsiImageType.RAW16 else 1
        if applied.kind is StreamKind.SNAPSHOT:
            period_s = applied.exposure_us / 1e6 + derived.readout_time_s(plan.mode, roi.height)
        else:
            period_s = derived.frame_period_s(plan.mode, roi.height, applied.exposure_us)
        self._next_stream_id += 1
        stream = _Stream(
            stream_id=self._next_stream_id,
            request=request,
            config=applied,
            mode=plan.mode,
            image_type=plan.image_type,
            roi=roi,
            adc_bits=plan.mode.adc_bits,
            period_s=period_s,
            buffer=bytearray(roi.width * roi.height * bytes_per_pixel),
        )
        self._stream = stream
        return stream

    def _active_stream(self, stream: _Stream) -> ActiveStream:
        return ActiveStream(
            stream_id=stream.stream_id,
            config=stream.config,
            frame_shape=(stream.roi.height, stream.roi.width),
            adc_bits=stream.adc_bits,
            frame_period_s=stream.period_s,
        )

    # --- The mode-change function ---

    def _apply_stream(self, plan: _Plan) -> tuple[StreamConfig, Roi]:
        """Stop, set the controls, set ROI and binning, set the start position, read back, drop
        stale state. The caller holds the lock and starts capture afterward.

        The SDK can apply a different geometry without an error. The function compares the
        read-back with the request, applies the geometry once more when they differ, and raises
        `CameraConfigError` when the second read-back differs too.
        """
        self._stop_sdk()
        applied = self._apply_controls(plan)
        roi = self._apply_geometry(plan)
        self._counter_base = 0
        self._last_counter = 0
        self._discard_left = 0
        self._since_check = 0
        self._pending_flags = FrameFlag.NONE
        return replace(applied, roi=roi), roi

    def _apply_controls(self, plan: _Plan) -> StreamConfig:
        config = plan.config
        if int(AsiControl.HIGH_SPEED_MODE) in self._caps:
            self._set_checked(AsiControl.HIGH_SPEED_MODE, int(config.high_speed), "high_speed")
        elif config.high_speed:
            raise CameraConfigError("this camera has no high-speed mode")
        bandwidth = config.bandwidth_pct
        if bandwidth is not None:
            bandwidth = self._set_checked(AsiControl.BANDWIDTH_OVERLOAD, bandwidth, "bandwidth_pct")
        elif int(AsiControl.BANDWIDTH_OVERLOAD) in self._caps:
            bandwidth = self._get_control(AsiControl.BANDWIDTH_OVERLOAD)
        offset = config.offset
        if offset is not None:
            offset = self._set_checked(AsiControl.OFFSET, offset, "offset")
        elif int(AsiControl.OFFSET) in self._caps:
            offset = self._get_control(AsiControl.OFFSET)
        gain = self._set_checked(AsiControl.GAIN, config.gain, "gain")
        exposure = self._set_checked(
            AsiControl.EXPOSURE, config.exposure_us, "exposure_us", tolerance=EXPOSURE_TOLERANCE
        )
        return replace(
            config, gain=gain, exposure_us=exposure, offset=offset, bandwidth_pct=bandwidth
        )

    def _get_control(self, control: AsiControl) -> int:
        with self._guard("get_control_value"):
            return self._api.get_control_value(self._camera_id, int(control))[0]

    def _set_checked(
        self, control: AsiControl, value: int, label: str, *, tolerance: float = 0.0
    ) -> int:
        """Set a control and read it back. Raises `CameraConfigError` for a value that the
        camera's range excludes, or that the camera changed without an error."""
        caps = self._caps.get(int(control))
        if caps is None:
            raise CameraConfigError(f"this camera has no {label} control")
        if not caps.min_value <= value <= caps.max_value:
            raise CameraConfigError(
                f"{label} {value} is outside the camera's range "
                f"{caps.min_value} to {caps.max_value}"
            )
        with self._guard("set_control_value"):
            self._api.set_control_value(self._camera_id, int(control), value)
        applied = self._get_control(control)
        allowed = max(1.0, abs(value) * tolerance) if tolerance else 0.0
        if abs(applied - value) > allowed:
            raise CameraConfigError(f"the camera applied {label} {applied} for the request {value}")
        return applied

    def _apply_geometry(self, plan: _Plan) -> Roi:
        roi, mode = plan.roi, plan.mode
        mismatch = ""
        for attempt in (1, 2):
            with self._guard("set_roi_format"):
                self._api.set_roi_format(
                    self._camera_id, roi.width, roi.height, mode.sdk_bin, plan.image_type
                )
            with self._guard("set_start_position"):
                self._api.set_start_position(self._camera_id, roi.x, roi.y)
            with self._guard("get_roi_format"):
                fmt = self._api.get_roi_format(self._camera_id)
            with self._guard("get_start_position"):
                x, y = self._api.get_start_position(self._camera_id)
            asked = (roi.width, roi.height, mode.sdk_bin, plan.image_type)
            got = (fmt.width, fmt.height, fmt.binning, fmt.image_type)
            inside = 0 <= x <= mode.width_px - roi.width and 0 <= y <= mode.height_px - roi.height
            if got == asked and inside:
                if attempt == 2:
                    self._emit(
                        "warning",
                        "camera.geometry_corrected",
                        "The camera applied a different geometry. Applying it again fixed it.",
                    )
                return Roi(x, y, roi.width, roi.height)
            mismatch = (
                f"asked for {asked[0]} x {asked[1]} at binning {asked[2]}, "
                f"got {got[0]} x {got[1]} at binning {got[2]}"
            )
        raise CameraConfigError(f"the camera applied a different geometry ({mismatch})")

    # --- Start and read ---

    def start(self) -> None:
        with self._lock:
            self._require_open()
            stream = self._stream
            if stream is None:
                raise CameraStateError("start before configure")
            if self._stopping:
                raise CameraStateError("capture is stopping")
            self._intent_running = True
            self._start_capture(stream)

    def _start_capture(self, stream: _Stream) -> None:
        """Start capture or one exposure. The caller holds the lock. Does nothing if it runs."""
        if stream.config.kind is StreamKind.SNAPSHOT:
            if self._exposing:
                return
            with self._guard("start_exposure"):
                self._api.start_exposure(self._camera_id, dark=False)
            self._exposing = True
            self._exposure_started_utc_ns = self._clock.utc_ns()
            return
        if self._capturing:
            return
        with self._guard("start_video_capture"):
            self._api.start_video_capture(self._camera_id)
        self._capturing = True
        self._counter_base = 0
        self._last_counter = 0
        for _ in range(self._opts.discard_frames):
            self._read_raw(stream, self._vendor_bound_s(stream))
        if self._opts.discard_frames:
            self._counter_base = self._read_counter()

    def _vendor_bound_s(self, stream: _Stream) -> float:
        exposure_s = stream.config.exposure_us / 1e6
        return 2.0 * max(exposure_s, stream.period_s) + VENDOR_WAIT_FLOOR_S

    def _read_counter(self) -> int:
        with self._guard("get_dropped_frames"):
            counter = self._api.get_dropped_frames(self._camera_id)
        self._last_counter = counter
        return counter

    def _read_raw(self, stream: _Stream, wait_s: float) -> int:
        """One bounded SDK read into the stream's buffer. Returns the arrival time in UTC ns."""
        wait_ms = max(0, math.ceil(wait_s * 1000 - 1e-9))
        with self._guard("get_video_data", wait_ms / 1000 + self._opts.read_margin_s):
            self._api.get_video_data(self._camera_id, stream.buffer, wait_ms)
            return self._clock.utc_ns()  # the statement right after the read returns

    def read_frame(self, timeout_s: float) -> Frame:
        with self._lock:
            stream = self._stream
            if stream is None or self._stopping or not (self._capturing or self._exposing):
                raise CameraStateError("read_frame while not capturing")
            if self._reader_busy:
                raise CameraStateError("another read_frame is in progress")
            self._reader_busy = True
            self._reader_idle.clear()
            self._read_bound_s = min(max(timeout_s, 0.0), self._vendor_bound_s(stream))
        try:
            if stream.config.kind is StreamKind.SNAPSHOT:
                return self._read_snapshot(stream, timeout_s)
            return self._read_video(stream, timeout_s)
        finally:
            with self._lock:
                self._reader_busy = False
                self._reader_idle.set()

    def _read_video(self, stream: _Stream, timeout_s: float) -> Frame:
        vendor_bound_s = self._vendor_bound_s(stream)
        started_ns = self._clock.monotonic_ns()
        while True:
            remaining_s = timeout_s - (self._clock.monotonic_ns() - started_ns) / NS_PER_S
            try:
                t_arrival_ns = self._read_raw(stream, max(0.0, min(vendor_bound_s, remaining_s)))
            except AsiConfigError:
                self._geometry_fault("the frame size no longer fits the buffer")
            with self._lock:
                if self._discard_left > 0:
                    self._discard_left -= 1
                    continue
            break
        self._check_geometry(stream)
        counter = self._read_counter()
        with self._lock:
            dropped = counter - self._counter_base if counter >= self._counter_base else counter
            self._counter_base = counter
            flags, self._pending_flags = self._pending_flags, FrameFlag.NONE
        exposure_ns = stream.config.exposure_us * 1000
        t_utc_ns = t_arrival_ns - round(stream.period_s * NS_PER_S) + exposure_ns // 2
        return self._make_frame(
            stream, t_arrival_ns, t_utc_ns, dropped, flags, fresh_temperature=False
        )

    def _read_snapshot(self, stream: _Stream, timeout_s: float) -> Frame:
        exposure_s = stream.config.exposure_us / 1e6
        bound_s = min(max(timeout_s, 0.0), 2.0 * stream.period_s + VENDOR_WAIT_FLOOR_S)
        poll_s = min(max(exposure_s / 50, 0.002), SNAPSHOT_POLL_MAX_S)
        started_ns = self._clock.monotonic_ns()
        while True:
            with self._guard("get_exposure_status"):
                status = self._api.get_exposure_status(self._camera_id)
            if status is AsiExposureStatus.SUCCESS:
                break
            if status in (AsiExposureStatus.FAILED, AsiExposureStatus.IDLE):
                with self._lock:
                    self._exposing = False
                    self._intent_running = False
                raise CameraError(
                    "the exposure failed"
                    if status is AsiExposureStatus.FAILED
                    else "the exposure ended without data"
                )
            if self._stopping:
                raise CameraStateError("capture was stopped")
            elapsed_s = (self._clock.monotonic_ns() - started_ns) / NS_PER_S
            if elapsed_s >= bound_s:
                raise CameraTimeoutError(f"the exposure did not finish within {bound_s:.1f} s")
            self._clock.sleep(min(poll_s, bound_s - elapsed_s))
        with self._guard("get_data_after_exposure"):
            self._api.get_data_after_exposure(self._camera_id, stream.buffer)
            t_arrival_ns = self._clock.utc_ns()
        with self._lock:
            self._exposing = False
            self._intent_running = False
            flags, self._pending_flags = self._pending_flags, FrameFlag.NONE
        t_utc_ns = self._exposure_started_utc_ns + stream.config.exposure_us * 1000 // 2
        return self._make_frame(stream, t_arrival_ns, t_utc_ns, 0, flags, fresh_temperature=True)

    def _make_frame(
        self,
        stream: _Stream,
        t_arrival_ns: int,
        t_utc_ns: int,
        dropped_before: int,
        flags: FrameFlag,
        *,
        fresh_temperature: bool,
    ) -> Frame:
        roi = stream.roi
        count = roi.width * roi.height
        pixels: FrameData
        if stream.image_type is AsiImageType.RAW16:
            pixels = np.frombuffer(stream.buffer, dtype="<u2", count=count).astype(np.uint16)
        else:
            pixels = np.frombuffer(stream.buffer, dtype=np.uint8, count=count).copy()
        quality, error_ns, time_flags = self._time_info()
        frame = Frame(
            data=pixels.reshape(roi.height, roi.width),
            stream_id=stream.stream_id,
            seq=stream.seq,
            t_arrival_ns=t_arrival_ns,
            t_utc_ns=t_utc_ns,
            t_err_ns=error_ns,
            t_quality=quality,
            dropped_before=dropped_before,
            exposure_us=stream.config.exposure_us,
            gain=stream.config.gain,
            mode=stream.config.mode,
            roi=roi,
            adc_bits=stream.adc_bits,
            temperature_c=self._frame_temperature(fresh_temperature),
            flags=flags | time_flags,
        )
        stream.seq += 1
        return frame

    def _time_info(self) -> tuple[TimeQuality, int, FrameFlag]:
        """The time quality, the 1-sigma error, and the flags that the clock state implies.

        The clock status can cost a subprocess call, so the driver reads it every
        `status_interval_s` and not for each frame.
        """
        now = self._clock.monotonic_ns()
        interval_ns = round(self._opts.status_interval_s * NS_PER_S)
        if self._time_checked_ns is None or now - self._time_checked_ns >= interval_ns:
            status = self._clock.status()
            self._time_checked_ns = now
            self._time_quality = (
                TimeQuality.INVALID if status.synchronized is False else TimeQuality.ESTIMATED
            )
            self._time_error_ns = (status.error_bound_ns or 0) + round(
                self._opts.time_error_ms * 1e6
            )
        flags = (
            FrameFlag.TIME_INVALID if self._time_quality is TimeQuality.INVALID else FrameFlag.NONE
        )
        return self._time_quality, self._time_error_ns, flags

    # --- Geometry check ---

    def _check_geometry(self, stream: _Stream) -> None:
        interval = self._opts.geometry_check_interval
        if not interval:
            return
        self._since_check += 1
        if self._since_check < interval:
            return
        self._since_check = 0
        with self._lock:  # `move_roi` holds the lock, so a move never looks like a change
            with self._guard("get_roi_format"):
                fmt = self._api.get_roi_format(self._camera_id)
            with self._guard("get_start_position"):
                position = self._api.get_start_position(self._camera_id)
            roi = stream.roi
            changed = (fmt.width, fmt.height, fmt.binning, fmt.image_type) != (
                roi.width,
                roi.height,
                stream.mode.sdk_bin,
                stream.image_type,
            ) or position != (roi.x, roi.y)
        if changed:
            self._geometry_fault("the camera reports a different ROI, binning, or position")

    def _geometry_fault(self, detail: str) -> NoReturn:
        """Stop capture and raise: the frames no longer match the stream the driver reported."""
        self._emit(
            "error", "camera.geometry_changed", f"The camera changed its geometry: {detail}."
        )
        with self._lock:
            try:
                self._stop_sdk()
            except CameraError:
                _log.exception("stopping capture after a geometry change failed")
        raise CameraConfigError(
            f"the camera changed its geometry in the middle of a stream ({detail})"
        )

    # --- Controls while running ---

    def move_roi(self, x: int, y: int) -> Roi:
        with self._lock:
            stream = self._stream
            if stream is None:
                raise CameraStateError("move_roi before configure")
            mode, roi = stream.mode, stream.roi
            target_x = min(max(x, 0), mode.width_px - roi.width)
            target_y = min(max(y, 0), mode.height_px - roi.height)
            with self._guard("set_start_position"):
                self._api.set_start_position(self._camera_id, target_x, target_y)
            with self._guard("get_start_position"):
                applied_x, applied_y = self._api.get_start_position(self._camera_id)
            if not (
                0 <= applied_x <= mode.width_px - roi.width
                and 0 <= applied_y <= mode.height_px - roi.height
            ):
                raise CameraConfigError("the camera applied a start position outside the frame")
            stream.roi = Roi(applied_x, applied_y, roi.width, roi.height)
            if self._capturing:
                self._discard_left = self._opts.roi_move_discard_frames
            return stream.roi

    def read_temperature_c(self) -> float | None:
        with self._lock:
            if not self._open or int(AsiControl.TEMPERATURE) not in self._caps:
                return None
            return self._read_temperature(wait_for_warmup=True)

    def _read_temperature(self, *, wait_for_warmup: bool) -> float | None:
        """Read the sensor. Exactly 0 shortly after initialization is not a reading yet."""
        for _ in range(2):
            tenths = self._get_control(AsiControl.TEMPERATURE)
            warmup_left_s = (
                self._opts.temperature_warmup_s
                - (self._clock.monotonic_ns() - self._init_ns) / NS_PER_S
            )
            if tenths != 0 or warmup_left_s <= 0:
                self._temperature_c = tenths / 10
                self._temperature_ns = self._clock.monotonic_ns()
                return self._temperature_c
            if not wait_for_warmup:
                return self._temperature_c
            self._clock.sleep(warmup_left_s)
        return self._temperature_c

    def _frame_temperature(self, fresh: bool) -> float | None:
        if int(AsiControl.TEMPERATURE) not in self._caps:
            return None
        now = self._clock.monotonic_ns()
        stale = self._temperature_ns is None or (
            now - self._temperature_ns >= round(self._opts.temperature_interval_s * NS_PER_S)
        )
        if fresh or stale:
            try:
                with self._lock:
                    self._read_temperature(wait_for_warmup=False)
            except AsiError:
                _log.debug("the temperature read failed", exc_info=True)
        return self._temperature_c

    def dropped_frames(self) -> int:
        with self._lock:
            if self._open and self._capturing:
                return self._read_counter()
            return self._last_counter

    # --- Recovery ---

    def recover(self, level: RecoveryLevel) -> None:
        try:
            self._stop_capture()
        except CameraTimeoutError:
            raise  # a reader that is stuck in the SDK needs the supervisor, not another step
        except CameraError:
            if level < RecoveryLevel.REOPEN:
                raise
            _log.warning("stopping capture failed, so the step reopens the camera", exc_info=True)
        with self._lock:
            if not self._wanted_open:
                raise CameraStateError("the camera is not open")
            self._emit(
                "info",
                "camera.recovery",
                f"The driver runs recovery step {level.name}.",
                step=int(level),
            )
            if level >= RecoveryLevel.REOPEN or not self._open:
                self._reopen(reset_usb=level >= RecoveryLevel.USB_RESET)
            stream = self._stream
            if stream is not None:
                plan = self._plan(replace(stream.request, roi=stream.roi))
                stream.config, stream.roi = self._apply_stream(plan)
                self._pending_flags |= FrameFlag.RECOVERED
                if self._intent_running:
                    self._start_capture(stream)

    def _reopen(self, *, reset_usb: bool) -> None:
        """Close the camera, optionally reset its USB device, and open it again."""
        self._close_quietly(self._camera_id)
        self._open = False
        if reset_usb:
            if self._resetter is None:
                raise CameraError("no USB resetter is configured")
            self._resetter.reset()
            self._wait_for_camera()
        self._open_camera()

    def _wait_for_camera(self) -> None:
        """Wait for the camera to enumerate again after a reset, on the clock."""
        waited_s = 0.0
        while waited_s <= self._opts.usb_reenumerate_timeout_s:
            try:
                with self._guard("get_connected_camera_count"):
                    if self._api.get_connected_camera_count() > self._opts.camera_index:
                        return
            except AsiError:
                _log.debug("enumeration failed while waiting for the camera", exc_info=True)
            self._clock.sleep(REOPEN_POLL_S)
            waited_s += REOPEN_POLL_S
        raise CameraDisconnectedError("the camera did not come back after the USB reset")
