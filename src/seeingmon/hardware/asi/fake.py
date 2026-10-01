"""A pure-Python imitation of the ZWO ASI SDK, for tests and dry runs.

`FakeAsiSdk` implements `AsiApi` on a `Clock`. With a `VirtualClock`, a blocking read advances
time instead of waiting, so a night of frames runs in seconds. The fake models what the research
notes (`docs/research-notes.md`, "Camera access options") report about the real behavior:

- **ROI rules.** The width is a multiple of 8, the height is a multiple of 2, and both fit the
  binned frame. `set_roi_format` recenters the ROI, so a caller must follow it with
  `set_start_position`.
- **Frame timing.** Video frames complete one frame period apart, where the period is the larger
  of the exposure and the readout time (an overhead plus the rows times the row time).
- **A small buffer.** The camera keeps the newest `buffer_frames` frames. A late reader loses the
  oldest frames, and the drop counter counts them. Stopping capture resets the counter.
- **Silent geometry changes.** `set_roi_format` during capture succeeds without effect, and you
  can script a geometry that differs from the request (`corrupt_next_roi`) or a change in the
  middle of a stream (`change_geometry`).
- **The first temperature read.** The temperature control returns 0 for the first 250 ms after
  `init_camera`.
- **Faults.** Reads that time out (`stall_reads`), a call that hangs (`hang_next`), errors for any
  call (`fail_next`), and a camera that disconnects and returns (`disconnect`, `reconnect`).

Pixels are deterministic: `default_pixels` returns the ADC counts of a frame, so a test can compute
what a frame must hold. A 16-bit frame carries the ADC value in the high bits, as the real SDK
delivers it, and an 8-bit frame carries the top 8 bits.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.hardware.asi.api import (
    AsiCameraInfo,
    AsiControl,
    AsiControlCaps,
    AsiErrorCode,
    AsiExposureStatus,
    AsiImageType,
    AsiRoiFormat,
    AsiTimeoutError,
    error_for,
)

NS_PER_MS = 1_000_000
HANG_LIMIT_S = 30.0  # a scripted hang releases itself after this much real time, so a test ends


@dataclass(frozen=True, slots=True)
class FakeTiming:
    """The readout timing of one binning: the time per row and the overhead per frame."""

    row_time_s: float
    overhead_s: float


# The reference camera, from the research notes ("Frame rates and row timing"). The key is
# (SDK binning, high-speed mode).
DEFAULT_TIMING: Mapping[tuple[int, bool], FakeTiming] = {
    (1, False): FakeTiming(37.6e-6, 6.5e-3),
    (1, True): FakeTiming(30.1e-6, 5.0e-3),
    (2, False): FakeTiming(21.3e-6, 1.4e-3),
    (2, True): FakeTiming(18.2e-6, 1.2e-3),
}
DEFAULT_ADC_BITS: Mapping[tuple[int, bool], int] = {
    (1, False): 12,
    (1, True): 10,
    (2, False): 14,
    (2, True): 12,
}


@dataclass(frozen=True, slots=True)
class FakeFrameInfo:
    """Everything that a pixel factory needs to render one frame."""

    index: int
    width: int
    height: int
    binning: int
    image_type: AsiImageType
    start_x: int
    start_y: int
    exposure_us: int
    gain: int
    adc_bits: int


PixelFactory = Callable[[FakeFrameInfo], npt.NDArray[np.uint32]]


def default_pixels(info: FakeFrameInfo) -> npt.NDArray[np.uint32]:
    """The ADC counts of a frame, below 1000 so that every ADC depth holds them.

    The value of a pixel depends on its position on the sensor and on the frame index, so a
    test can tell which frame and which part of the sensor a buffer came from.
    """
    rows = np.arange(info.height, dtype=np.uint32)[:, None] + np.uint32(info.start_y)
    cols = np.arange(info.width, dtype=np.uint32)[None, :] + np.uint32(info.start_x)
    counts: npt.NDArray[np.uint32] = (cols + np.uint32(7) * rows + np.uint32(31 * info.index)) % (
        np.uint32(1000)
    )
    return counts


def pixel_bytes(counts: npt.NDArray[np.uint32], image_type: AsiImageType, adc_bits: int) -> bytes:
    """Pack ADC counts the way the SDK delivers them: 16-bit values with the ADC value in the
    high bits (little-endian), or the top 8 bits of the ADC value."""
    if image_type is AsiImageType.RAW16:
        shifted = counts << np.uint32(16 - adc_bits)
        return bytes(shifted.astype("<u2").tobytes())
    if image_type is AsiImageType.RAW8:
        return bytes((counts >> np.uint32(max(adc_bits - 8, 0))).astype(np.uint8).tobytes())
    raise ValueError(f"the fake renders only RAW8 and RAW16, not {image_type!r}")


@dataclass(slots=True)
class _Frame:
    index: int
    data: bytes


class FakeAsiSdk:
    """An `AsiApi` with one camera. Script faults with the methods under "Scripting".

    Args:
        clock: The time source. Reads wait through `clock.sleep`.
        model: The camera name that `get_camera_property` reports.
        max_width: The width of the finest readout (SDK binning 1).
        max_height: The height of the finest readout.
        bins: The supported SDK binning factors.
        temperature_c: The sensor temperature, or `None` for a camera without a sensor.
        buffer_frames: How many frames the camera holds before it overwrites the oldest.
        timing: The readout timing, by (binning, high-speed mode).
        adc_bits: The ADC depth, by (binning, high-speed mode).
        pixel_factory: Renders the ADC counts of a frame. The default is `default_pixels`.
        start_alignment: The ROI origin rounds down to a multiple of this many pixels.
        temperature_warmup_s: How long the temperature control reads 0 after `init_camera`.
        silent_gain_limit: If set, the camera applies at most this gain without an error,
            although its caps report a higher maximum.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        model: str = "ZWO ASI294MM (fake)",
        max_width: int = 8288,
        max_height: int = 5644,
        bins: tuple[int, ...] = (1, 2),
        temperature_c: float | None = 18.3,
        buffer_frames: int = 3,
        timing: Mapping[tuple[int, bool], FakeTiming] | None = None,
        adc_bits: Mapping[tuple[int, bool], int] | None = None,
        pixel_factory: PixelFactory | None = None,
        start_alignment: int = 1,
        temperature_warmup_s: float = 0.25,
        silent_gain_limit: int | None = None,
        sdk_version: str = "1, 41, 0, 0",
    ) -> None:
        self._clock = clock
        self._model = model
        self._max_width = max_width
        self._max_height = max_height
        self._bins = bins
        self._buffer_frames = buffer_frames
        self._timing = dict(timing or DEFAULT_TIMING)
        self._adc_bits = dict(adc_bits or DEFAULT_ADC_BITS)
        self._pixel_factory = pixel_factory or default_pixels
        self._start_alignment = start_alignment
        self._warmup_ns = round(temperature_warmup_s * NS_PER_S)
        self._silent_gain_limit = silent_gain_limit
        self._sdk_version = sdk_version
        self.temperature_c = temperature_c
        self._lock = threading.RLock()
        # Every call in order, as (function, arguments). A blocking read adds a second entry,
        # `get_video_data_returned`, when it returns.
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._failures: dict[str, deque[int]] = {}
        self._hangs: dict[str, threading.Event] = {}
        self._stalled_reads = 0
        self._lose_next = 0
        self._corrupt_roi: tuple[int, int, bool] | None = None
        self._exposure_failure = False
        self._present = True
        self._reappear_ns = 0
        self._reset_camera()

    # --- Scripting ---

    def fail_next(self, function: str, code: AsiErrorCode, count: int = 1) -> None:
        """Make the next `count` calls of `function` (such as `"set_roi_format"`) fail."""
        with self._lock:
            self._failures.setdefault(function, deque()).extend([int(code)] * count)

    def stall_reads(self, count: int = 1) -> None:
        """Make the next `count` blocking reads wait out their timeout and return no frame."""
        with self._lock:
            self._stalled_reads += count

    def hang_next(self, function: str = "get_video_data") -> threading.Event:
        """Make the next call of `function` block until you set the returned event.

        The block uses real time, so a test runs the call in a thread. The call raises an
        `AsiTimeoutError` after the release, or after `HANG_LIMIT_S` seconds without one.
        """
        event = threading.Event()
        with self._lock:
            self._hangs[function] = event
        return event

    def lose_frames(self, count: int) -> None:
        """Lose the next `count` frames: the drop counter rises now, and the next frame arrives
        `count` frame periods late."""
        with self._lock:
            self._advance(self._clock.monotonic_ns())
            self._dropped += count
            if self._video:
                self._next_complete_ns += count * self._period_ns()

    def corrupt_next_roi(self, width: int, height: int, *, always: bool = False) -> None:
        """Make `set_roi_format` succeed but apply this size instead of the requested one.

        The default corrupts one call. With `always`, every call applies this size.
        """
        with self._lock:
            self._corrupt_roi = (width, height, always)

    def change_geometry(self, width: int, height: int) -> None:
        """Change the ROI size without a call, as the SDK does when it changes geometry silently
        in the middle of a stream."""
        with self._lock:
            self._advance(self._clock.monotonic_ns())
            self._width, self._height = width, height
            self._reschedule()

    def fail_next_exposure(self) -> None:
        """Make the next single exposure end in the failed state."""
        with self._lock:
            self._exposure_failure = True

    def disconnect(self) -> None:
        """Unplug the camera. Calls fail with `CAMERA_REMOVED` until `reconnect`."""
        with self._lock:
            self._present = False
            self._reset_camera()

    def reconnect(self, after_s: float = 0.0) -> None:
        """Plug the camera in again, `after_s` seconds from now. It starts with power-on state,
        and the caller must open and initialize it again."""
        with self._lock:
            self._present = True
            self._reappear_ns = self._clock.monotonic_ns() + round(after_s * NS_PER_S)
            self._reset_camera()

    # --- Inspection ---

    @property
    def video_active(self) -> bool:
        """Whether video capture runs."""
        return self._video

    @property
    def roi(self) -> tuple[int, int, int, int]:
        """The applied ROI as (x, y, width, height)."""
        return (self._x, self._y, self._width, self._height)

    def calls_named(self, function: str) -> list[tuple[Any, ...]]:
        """The arguments of every call to `function`."""
        return [args for name, args in self.calls if name == function]

    def control(self, control: AsiControl) -> int:
        """The stored value of a control, without the temperature warm-up rule."""
        return self._controls[int(control)]

    def frame_period_s(self) -> float:
        """The frame period of the current settings, in seconds."""
        return self._period_ns() / NS_PER_S

    # --- The camera state ---

    def _reset_camera(self) -> None:
        self._opened = False
        self._initialized = False
        self._init_ns = 0
        self._controls: dict[int, int] = {caps.control: caps.default_value for caps in self._caps()}
        self._bin = 1
        self._image_type = AsiImageType.RAW8
        self._width = self._max_width
        self._height = self._max_height
        self._x = 0
        self._y = 0
        self._video = False
        self._queue: deque[_Frame] = deque()
        self._next_complete_ns = 0
        self._produced = 0
        self._dropped = 0
        self._exposure_status = AsiExposureStatus.IDLE
        self._exposure_done_ns = 0
        self._exposure_info: FakeFrameInfo | None = None
        self._snapshots = 0

    def _caps(self) -> list[AsiControlCaps]:
        def caps(
            control: AsiControl, name: str, low: int, high: int, default: int
        ) -> AsiControlCaps:
            writable = control is not AsiControl.TEMPERATURE
            return AsiControlCaps(name, name, int(control), low, high, default, False, writable)

        listed = [
            caps(AsiControl.GAIN, "Gain", 0, 570, 0),
            caps(AsiControl.EXPOSURE, "Exposure", 32, 2_000_000_000, 10_000),
            caps(AsiControl.GAMMA, "Gamma", 1, 100, 50),
            caps(AsiControl.OFFSET, "Offset", 0, 255, 10),
            caps(AsiControl.BANDWIDTH_OVERLOAD, "BandWidth", 40, 100, 50),
            caps(AsiControl.HIGH_SPEED_MODE, "HighSpeedMode", 0, 1, 0),
            caps(AsiControl.FLIP, "Flip", 0, 3, 0),
        ]
        if self.temperature_c is not None:
            listed.append(caps(AsiControl.TEMPERATURE, "Temperature", -500, 1000, 200))
        return listed

    def _high_speed(self) -> bool:
        return bool(self._controls[AsiControl.HIGH_SPEED_MODE])

    def _frame_adc_bits(self) -> int:
        return self._adc_bits[(self._bin, self._high_speed())]

    def _readout_ns(self) -> int:
        timing = self._timing[(self._bin, self._high_speed())]
        return round(timing.overhead_s * NS_PER_S) + self._height * round(
            timing.row_time_s * NS_PER_S
        )

    def _period_ns(self) -> int:
        exposure_ns = self._controls[AsiControl.EXPOSURE] * 1000
        return max(exposure_ns, self._readout_ns())

    def _reschedule(self) -> None:
        """Keep the schedule consistent after a change of the settings. Call with the lock."""
        now = self._clock.monotonic_ns()
        if self._video and self._next_complete_ns < now:
            self._next_complete_ns = now + self._period_ns()

    # --- Frame production (lock held) ---

    def _frame_info(self, index: int) -> FakeFrameInfo:
        return FakeFrameInfo(
            index=index,
            width=self._width,
            height=self._height,
            binning=self._bin,
            image_type=self._image_type,
            start_x=self._x,
            start_y=self._y,
            exposure_us=self._controls[AsiControl.EXPOSURE],
            gain=self._controls[AsiControl.GAIN],
            adc_bits=self._frame_adc_bits(),
        )

    def _render(self, info: FakeFrameInfo) -> bytes:
        counts = self._pixel_factory(info)
        return pixel_bytes(counts, info.image_type, info.adc_bits)

    def _advance(self, now_ns: int) -> None:
        """Complete the frames that are due by `now_ns` and apply the buffer limit."""
        if not self._video or now_ns < self._next_complete_ns:
            return
        period = self._period_ns()
        count = (now_ns - self._next_complete_ns) // period + 1
        first = self._produced
        self._produced += count
        lost = max(0, len(self._queue) + count - self._buffer_frames)
        self._dropped += lost
        evicted = min(lost, len(self._queue))
        for _ in range(evicted):
            self._queue.popleft()
        skipped = lost - evicted
        for index in range(first + skipped, first + count):
            self._queue.append(_Frame(index, self._render(self._frame_info(index))))
        self._next_complete_ns += count * period

    # --- Call plumbing ---

    def _enter(self, function: str, *args: Any) -> None:
        with self._lock:
            self.calls.append((function, args))
            hang = self._hangs.pop(function, None)
            queue = self._failures.get(function)
            code = queue.popleft() if queue else None
        if hang is not None:
            hang.wait(HANG_LIMIT_S)
            raise AsiTimeoutError(function, AsiErrorCode.TIMEOUT)
        if code is not None:
            raise error_for(function, code)

    def _require_camera(self, function: str, camera_id: int, *, need: str = "ready") -> None:
        """Check the handle. `need` is `present`, `opened`, or `ready` (opened and initialized).

        Call with the lock held.
        """
        if not self._visible():
            raise error_for(function, AsiErrorCode.CAMERA_REMOVED)
        if camera_id != 0:
            raise error_for(function, AsiErrorCode.INVALID_ID)
        if need != "present" and not self._opened:
            raise error_for(function, AsiErrorCode.CAMERA_CLOSED)
        if need == "ready" and not self._initialized:
            raise error_for(function, AsiErrorCode.CAMERA_CLOSED)

    # --- AsiApi: enumeration and lifecycle ---

    def get_sdk_version(self) -> str:
        self._enter("get_sdk_version")
        return self._sdk_version

    def _visible(self) -> bool:
        return self._present and self._clock.monotonic_ns() >= self._reappear_ns

    def get_connected_camera_count(self) -> int:
        self._enter("get_connected_camera_count")
        with self._lock:
            return 1 if self._visible() else 0

    def get_camera_property(self, index: int) -> AsiCameraInfo:
        self._enter("get_camera_property", index)
        with self._lock:
            if not self._visible() or index != 0:
                raise error_for("get_camera_property", AsiErrorCode.INVALID_INDEX)
            return AsiCameraInfo(
                name=self._model,
                camera_id=0,
                max_width=self._max_width,
                max_height=self._max_height,
                is_color=False,
                supported_bins=self._bins,
                supported_formats=(AsiImageType.RAW8, AsiImageType.RAW16),
                pixel_size_um=2.315,
                is_cooled=False,
                is_usb3_host=True,
                is_usb3_camera=True,
                electrons_per_adu=3.5,
                bit_depth=14,
            )

    def open_camera(self, camera_id: int) -> None:
        self._enter("open_camera", camera_id)
        with self._lock:
            self._require_camera("open_camera", camera_id, need="present")
            self._opened = True

    def init_camera(self, camera_id: int) -> None:
        self._enter("init_camera", camera_id)
        with self._lock:
            self._require_camera("init_camera", camera_id, need="opened")
            self._initialized = True
            self._init_ns = self._clock.monotonic_ns()

    def close_camera(self, camera_id: int) -> None:
        self._enter("close_camera", camera_id)
        with self._lock:
            if not self._present:
                raise error_for("close_camera", AsiErrorCode.CAMERA_REMOVED)
            self._opened = False
            self._initialized = False
            self._video = False
            self._queue.clear()
            self._exposure_status = AsiExposureStatus.IDLE

    # --- AsiApi: controls ---

    def get_control_count(self, camera_id: int) -> int:
        self._enter("get_control_count", camera_id)
        with self._lock:
            self._require_camera("get_control_count", camera_id)
            return len(self._caps())

    def get_control_caps(self, camera_id: int, index: int) -> AsiControlCaps:
        self._enter("get_control_caps", camera_id, index)
        with self._lock:
            self._require_camera("get_control_caps", camera_id)
            caps = self._caps()
            if not 0 <= index < len(caps):
                raise error_for("get_control_caps", AsiErrorCode.INVALID_INDEX)
            return caps[index]

    def get_control_value(self, camera_id: int, control: int) -> tuple[int, bool]:
        self._enter("get_control_value", camera_id, control)
        with self._lock:
            self._require_camera("get_control_value", camera_id)
            if control == AsiControl.TEMPERATURE and self.temperature_c is not None:
                if self._clock.monotonic_ns() - self._init_ns < self._warmup_ns:
                    return 0, False
                return round(self.temperature_c * 10), False
            if control not in self._controls:
                raise error_for("get_control_value", AsiErrorCode.INVALID_CONTROL_TYPE)
            return self._controls[control], False

    def set_control_value(
        self, camera_id: int, control: int, value: int, *, auto: bool = False
    ) -> None:
        self._enter("set_control_value", camera_id, control, value, auto)
        with self._lock:
            self._require_camera("set_control_value", camera_id)
            caps = {c.control: c for c in self._caps()}
            if control not in caps or not caps[control].is_writable:
                raise error_for("set_control_value", AsiErrorCode.INVALID_CONTROL_TYPE)
            limits = caps[control]
            applied = min(max(value, limits.min_value), limits.max_value)  # the SDK clamps
            if control == AsiControl.GAIN and self._silent_gain_limit is not None:
                applied = min(applied, self._silent_gain_limit)
            self._advance(self._clock.monotonic_ns())
            self._controls[control] = applied
            self._reschedule()

    # --- AsiApi: geometry ---

    def _binned_size(self, binning: int) -> tuple[int, int]:
        return self._max_width // binning, self._max_height // binning

    def set_roi_format(
        self, camera_id: int, width: int, height: int, binning: int, image_type: AsiImageType
    ) -> None:
        self._enter("set_roi_format", camera_id, width, height, binning, int(image_type))
        with self._lock:
            self._require_camera("set_roi_format", camera_id)
            if self._exposure_status is AsiExposureStatus.WORKING:
                raise error_for("set_roi_format", AsiErrorCode.EXPOSURE_IN_PROGRESS)
            if binning not in self._bins:
                raise error_for("set_roi_format", AsiErrorCode.INVALID_SIZE)
            if image_type not in (AsiImageType.RAW8, AsiImageType.RAW16):
                raise error_for("set_roi_format", AsiErrorCode.INVALID_IMAGE_TYPE)
            full_width, full_height = self._binned_size(binning)
            if not (
                0 < width <= full_width
                and 0 < height <= full_height
                and width % 8 == 0
                and height % 2 == 0
            ):
                raise error_for("set_roi_format", AsiErrorCode.INVALID_SIZE)
            self._advance(self._clock.monotonic_ns())
            if self._video:
                # The real SDK can accept this call during capture and keep the old size.
                width, height, binning = self._width, self._height, self._bin
            elif self._corrupt_roi is not None:
                width, height, always = self._corrupt_roi
                if not always:
                    self._corrupt_roi = None
            self._bin, self._image_type = binning, image_type
            self._width, self._height = width, height
            full_width, full_height = self._binned_size(self._bin)
            self._x = self._align((full_width - self._width) // 2)  # recenter, as the SDK does
            self._y = (full_height - self._height) // 2
            self._reschedule()

    def get_roi_format(self, camera_id: int) -> AsiRoiFormat:
        self._enter("get_roi_format", camera_id)
        with self._lock:
            self._require_camera("get_roi_format", camera_id)
            return AsiRoiFormat(self._width, self._height, self._bin, self._image_type)

    def _align(self, value: int) -> int:
        return value - value % self._start_alignment

    def set_start_position(self, camera_id: int, x: int, y: int) -> None:
        self._enter("set_start_position", camera_id, x, y)
        with self._lock:
            self._require_camera("set_start_position", camera_id)
            if self._exposure_status is AsiExposureStatus.WORKING:
                raise error_for("set_start_position", AsiErrorCode.EXPOSURE_IN_PROGRESS)
            full_width, full_height = self._binned_size(self._bin)
            if not (0 <= x <= full_width - self._width and 0 <= y <= full_height - self._height):
                raise error_for("set_start_position", AsiErrorCode.OUT_OF_BOUNDARY)
            self._advance(self._clock.monotonic_ns())
            self._x, self._y = self._align(x), self._align(y)

    def get_start_position(self, camera_id: int) -> tuple[int, int]:
        self._enter("get_start_position", camera_id)
        with self._lock:
            self._require_camera("get_start_position", camera_id)
            return self._x, self._y

    # --- AsiApi: video ---

    def get_dropped_frames(self, camera_id: int) -> int:
        self._enter("get_dropped_frames", camera_id)
        with self._lock:
            self._require_camera("get_dropped_frames", camera_id)
            self._advance(self._clock.monotonic_ns())
            return self._dropped

    def start_video_capture(self, camera_id: int) -> None:
        self._enter("start_video_capture", camera_id)
        with self._lock:
            self._require_camera("start_video_capture", camera_id)
            if self._exposure_status is AsiExposureStatus.WORKING:
                raise error_for("start_video_capture", AsiErrorCode.EXPOSURE_IN_PROGRESS)
            if not self._video:
                self._video = True
                self._queue.clear()
                self._dropped = 0
                self._next_complete_ns = self._clock.monotonic_ns() + self._period_ns()

    def stop_video_capture(self, camera_id: int) -> None:
        self._enter("stop_video_capture", camera_id)
        with self._lock:
            self._require_camera("stop_video_capture", camera_id)
            self._video = False
            self._queue.clear()
            self._dropped = 0

    def get_video_data(self, camera_id: int, buffer: bytearray, wait_ms: int) -> None:
        self._enter("get_video_data", camera_id, len(buffer), wait_ms)
        deadline_ns = self._clock.monotonic_ns() + wait_ms * NS_PER_MS
        try:
            while True:
                with self._lock:
                    now = self._clock.monotonic_ns()
                    self._require_camera("get_video_data", camera_id)
                    if not self._video:
                        raise error_for("get_video_data", AsiErrorCode.INVALID_SEQUENCE)
                    stalled = self._stalled_reads > 0
                    if not stalled:
                        self._advance(now)
                        if self._queue:
                            self._deliver("get_video_data", self._queue[0], buffer)
                            self._queue.popleft()
                            return
                    next_ns = self._next_complete_ns
                if stalled or next_ns > deadline_ns:
                    self._clock.sleep(max(deadline_ns - now, 0) / NS_PER_S)
                    with self._lock:
                        if stalled:
                            self._stalled_reads -= 1
                            self._queue.clear()
                            self._next_complete_ns = self._clock.monotonic_ns() + self._period_ns()
                    raise error_for("get_video_data", AsiErrorCode.TIMEOUT)
                self._clock.sleep((next_ns - now) / NS_PER_S)
        finally:
            with self._lock:
                self.calls.append(("get_video_data_returned", ()))

    def _deliver(self, function: str, frame: _Frame, buffer: bytearray) -> None:
        if len(buffer) < len(frame.data):
            raise error_for(function, AsiErrorCode.BUFFER_TOO_SMALL)
        buffer[: len(frame.data)] = frame.data

    # --- AsiApi: single exposures ---

    def start_exposure(self, camera_id: int, *, dark: bool = False) -> None:
        self._enter("start_exposure", camera_id, dark)
        with self._lock:
            self._require_camera("start_exposure", camera_id)
            if self._video:
                raise error_for("start_exposure", AsiErrorCode.VIDEO_MODE_ACTIVE)
            if self._exposure_status is AsiExposureStatus.WORKING:
                raise error_for("start_exposure", AsiErrorCode.EXPOSURE_IN_PROGRESS)
            exposure_ns = self._controls[AsiControl.EXPOSURE] * 1000
            self._exposure_status = AsiExposureStatus.WORKING
            self._exposure_done_ns = self._clock.monotonic_ns() + exposure_ns + self._readout_ns()
            self._exposure_info = self._frame_info(self._snapshots)

    def stop_exposure(self, camera_id: int) -> None:
        self._enter("stop_exposure", camera_id)
        with self._lock:
            self._require_camera("stop_exposure", camera_id)
            self._exposure_status = AsiExposureStatus.IDLE

    def get_exposure_status(self, camera_id: int) -> AsiExposureStatus:
        self._enter("get_exposure_status", camera_id)
        with self._lock:
            self._require_camera("get_exposure_status", camera_id)
            if (
                self._exposure_status is AsiExposureStatus.WORKING
                and self._clock.monotonic_ns() >= self._exposure_done_ns
            ):
                failed = self._exposure_failure
                self._exposure_failure = False
                self._exposure_status = (
                    AsiExposureStatus.FAILED if failed else AsiExposureStatus.SUCCESS
                )
            return self._exposure_status

    def get_data_after_exposure(self, camera_id: int, buffer: bytearray) -> None:
        self._enter("get_data_after_exposure", camera_id, len(buffer))
        with self._lock:
            self._require_camera("get_data_after_exposure", camera_id)
            info = self._exposure_info
            if self._exposure_status is not AsiExposureStatus.SUCCESS or info is None:
                raise error_for("get_data_after_exposure", AsiErrorCode.INVALID_SEQUENCE)
            self._deliver("get_data_after_exposure", _Frame(info.index, self._render(info)), buffer)
            self._snapshots += 1
            self._exposure_status = AsiExposureStatus.IDLE


__all__ = [
    "DEFAULT_ADC_BITS",
    "DEFAULT_TIMING",
    "FakeAsiSdk",
    "FakeFrameInfo",
    "FakeTiming",
    "PixelFactory",
    "default_pixels",
    "pixel_bytes",
]
