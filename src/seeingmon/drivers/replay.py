"""The `replay` driver: a recorded SER file as a camera.

The driver feeds a recording through the camera interface, so `acquire` and the analysis run
unchanged. Build it with `create(profile=..., clock=..., options={...})`, or let
`seeingmon.drivers.create_driver` do it.

**Options.** `path` names the SER file and is required. The others are optional.

- `rate`: `"original"` (the default) paces frames by their recorded timestamps, `"max"` never
  waits, and a number is a speed factor (2 replays twice as fast as recorded).
- `loop`: start over at `start_frame` when the recording ends, with a continuous timeline.
  Without it, the driver raises `ReplayFinishedError` after the last frame.
- `start_frame` and `max_frames`: replay only a window of the recording.
- `mode`: the readout mode name of the recording. The default comes from the sidecar (a burst's
  JSON sidecar, or the read mode and binning of a SharpCap sidecar), else `bin2`.
- `exposure_us` and `gain`: the settings of the recording. They override the sidecar, and they
  are required when no sidecar gives them.
- `sidecar`: a path to the sidecar to read (`.json` for a burst, any other name for a SharpCap
  settings file), or `false` to read none. By default the driver looks next to the SER file for
  `<capture>.json`, then `<capture>.CameraSettings.txt`.
- `byte_order` (`"little"` or `"big"`): overrides the header flag of a 16-bit file.
- `adc_bits`: the ADC depth. The default is the pixel depth of the file.

**Frames.** A frame carries the recorded exposure, gain, pixel format, and ADC depth, whatever
`configure` asks for. `t_arrival_ns` is the recorded timestamp (the PC clock at arrival).
`t_utc_ns` is the middle of the first row's exposure, as arrival time minus the frame period
plus half the exposure, with `TimeQuality.ESTIMATED`. After a software crop, `t_utc_ns` still
refers to the first row of the recorded frame, because the recording holds no row time.
`dropped_before` reports gaps in the recorded timestamps that exceed 1.5 frame periods.

**Timeline.** `rate="original"` waits through `Clock.sleep`, so a `VirtualClock` replays at
once. The first frame after `start` is due immediately, and later frames are due by their
recorded spacing. A consumer that falls behind gets frames at once until it catches up. A file
without timestamps gets synthetic ones from the start time (`StartCapture` of a SharpCap
sidecar, else the burst sidecar or the header) and the frame period.

**Lifecycle.** `open` rewinds to `start_frame`. `configure` starts a new stream (a new
`stream_id` and a `seq` of 0) and continues from the current position, like a live camera.
`stop` and `start` pause and resume.

**Privacy.** No message of this driver contains the path of the recording or any text from its
header or sidecar.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, Clock
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
    FrameFlag,
    PixelFormat,
    Roi,
    StreamConfig,
    StreamKind,
    TimeQuality,
)
from seeingmon.recordings.ser import ByteOrder, ColorId, SerError, SerFile, SerFormatError
from seeingmon.recordings.sidecar import (
    BurstSidecar,
    SidecarError,
    SidecarInfo,
    burst_sidecar_path,
    read_burst_sidecar,
    read_sharpcap_sidecar,
    sharpcap_sidecar_path,
)

if TYPE_CHECKING:
    from seeingmon.profile.models import Profile

__all__ = ["ReplayDriver", "ReplayFinishedError", "ReplayOptions", "create"]

DEFAULT_MODE = "bin2"
TIME_ERROR_NS = 2_000_000  # 1 sigma of t_utc_ns: PC-clock arrival times jitter by milliseconds

_PERIOD_BASELINE = 64  # frames between the timestamps that estimate the frame period
_MODE_NAME = re.compile(r"^[\x21-\x7e]{1,16}$")
_SDK_BIN = re.compile(r"^bin(\d+)$")
_KNOWN_OPTIONS = frozenset(
    {
        "path",
        "rate",
        "loop",
        "start_frame",
        "max_frames",
        "mode",
        "exposure_us",
        "gain",
        "sidecar",
        "byte_order",
        "adc_bits",
    }
)
_T = TypeVar("_T")


class ReplayFinishedError(CameraError):
    """The recording has no more frames and `loop` is off."""


def _first_not_none(*values: _T | None) -> _T | None:
    return next((value for value in values if value is not None), None)


def _parse_rate(value: object) -> float | None:
    """`None` means no waiting. Otherwise the speed factor."""
    if value is None:
        return 1.0
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "original":
            return 1.0
        if text == "max":
            return None
        try:
            speed = float(text.removesuffix("x"))
        except ValueError:
            raise CameraConfigError("rate must be 'original', 'max', or a number") from None
    elif isinstance(value, bool) or not isinstance(value, int | float):
        raise CameraConfigError("rate must be 'original', 'max', or a number")
    else:
        speed = float(value)
    if not math.isfinite(speed) or speed <= 0:
        raise CameraConfigError("rate must be positive")
    return speed


def _int_option(
    options: Mapping[str, object], name: str, *, minimum: int, maximum: int | None = None
) -> int | None:
    value = options.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CameraConfigError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise CameraConfigError(f"{name} must be {bound}")
    return value


def _bool_option(options: Mapping[str, object], name: str, *, default: bool) -> bool:
    value = options.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise CameraConfigError(f"{name} must be true or false")
    return value


def _parse_sidecar(value: object) -> Path | bool | None:
    if value is None or value is True:
        return None  # look next to the SER file
    if value is False:
        return False
    if isinstance(value, str | os.PathLike) and str(value):
        return Path(value)
    raise CameraConfigError("sidecar must be a path or false")


@dataclass(frozen=True, slots=True)
class ReplayOptions:
    """The validated options of the replay driver. See the module docstring."""

    path: Path
    rate: float | None = 1.0  # a speed factor, or None for no waiting
    loop: bool = False
    start_frame: int = 0
    max_frames: int | None = None
    mode: str | None = None
    exposure_us: int | None = None
    gain: int | None = None
    sidecar: Path | bool | None = None  # None: look next to the SER file. False: read none.
    byte_order: ByteOrder | None = None
    adc_bits: int | None = None

    @classmethod
    def from_mapping(cls, options: Mapping[str, object]) -> ReplayOptions:
        """Validate a mapping of options. Raises `CameraConfigError` for a bad one."""
        unknown = sorted(set(options) - _KNOWN_OPTIONS)
        if unknown:
            raise CameraConfigError(f"unknown replay options: {', '.join(unknown)}")
        raw_path = options.get("path")
        if not isinstance(raw_path, str | os.PathLike) or not str(raw_path):
            raise CameraConfigError("the path option must name a SER file")
        mode = options.get("mode")
        if mode is not None and (not isinstance(mode, str) or not _MODE_NAME.match(mode)):
            raise CameraConfigError("mode must be 1 to 16 printable ASCII characters")
        raw_order = options.get("byte_order")
        if raw_order not in (None, "little", "big"):
            raise CameraConfigError("byte_order must be 'little' or 'big'")
        byte_order: ByteOrder | None = (
            None if raw_order is None else ("little" if raw_order == "little" else "big")
        )
        return cls(
            path=Path(raw_path),
            rate=_parse_rate(options.get("rate")),
            loop=_bool_option(options, "loop", default=False),
            start_frame=_int_option(options, "start_frame", minimum=0) or 0,
            max_frames=_int_option(options, "max_frames", minimum=1),
            mode=mode,
            exposure_us=_int_option(options, "exposure_us", minimum=1),
            gain=_int_option(options, "gain", minimum=0),
            sidecar=_parse_sidecar(options.get("sidecar")),
            byte_order=byte_order,
            adc_bits=_int_option(options, "adc_bits", minimum=1, maximum=16),
        )


@dataclass(frozen=True, slots=True)
class _Recording:
    """Everything that `open` works out about the recording."""

    mode: str
    exposure_us: int
    gain: int
    pixel_format: PixelFormat
    adc_bits: int
    temperature_c: float | None
    frame_roi: Roi  # where the recorded frame sits, in the pixels of the mode
    sdk_bin: int
    is_color: bool
    stamps_ns: npt.NDArray[np.int64]  # arrival time of each frame, Unix nanoseconds
    drops: npt.NDArray[np.int64]  # frames lost immediately before each frame
    period_ns: int


def _estimate_period_ns(stamps_ns: npt.NDArray[np.int64]) -> int | None:
    """The median spacing over a baseline of up to 64 frames: jitter and a few gaps drop out."""
    if len(stamps_ns) < 2:
        return None
    baseline = max(1, min(_PERIOD_BASELINE, (len(stamps_ns) - 1) // 4))
    spans = stamps_ns[baseline:] - stamps_ns[:-baseline]
    period = round(float(np.median(spans)) / baseline)
    return period if period > 0 else None


def _find_drops(stamps_ns: npt.NDArray[np.int64], period_ns: int) -> npt.NDArray[np.int64]:
    """Frames lost before each frame: round(gap / period) - 1 for gaps over 1.5 periods."""
    drops = np.zeros(len(stamps_ns), dtype=np.int64)
    if len(stamps_ns) > 1:
        gaps = np.diff(stamps_ns)
        lost = (2 * gaps + period_ns) // (2 * period_ns) - 1  # round half up
        drops[1:] = np.where(2 * gaps > 3 * period_ns, lost, 0)
    return drops


class ReplayDriver:
    """A `CameraDriver` that replays a SER recording. Build it with `create`."""

    def __init__(self, options: ReplayOptions, clock: Clock) -> None:
        self._options = options
        self._clock = clock
        self._ser: SerFile | None = None
        self._rec: _Recording | None = None
        self._active: ActiveStream | None = None
        self._roi: Roi | None = None
        self._running = False
        self._stream_id = 0
        self._seq = 0
        self._dropped_total = 0
        self._first_of_stream = True
        self._next = 0  # index of the next frame in the file
        self._end = 0  # one past the last frame of the replayed window
        self._pass = 0  # how many times the window has wrapped
        self._wrapped = False
        self._pass_ns = 0
        self._epoch_clock_ns = 0
        self._epoch_stamp_ns: int | None = None

    @property
    def name(self) -> str:
        return "replay"

    # --- opening ---

    def open(self) -> CameraInfo:
        """Open the recording and its sidecar. Rewinds to `start_frame`."""
        self.close()
        options = self._options
        try:
            ser = SerFile(options.path, byte_order=options.byte_order)
        except SerFormatError as exc:
            raise CameraConfigError(f"the recording is not a valid SER file: {exc}") from None
        except SerError as exc:
            raise CameraDisconnectedError(f"cannot open the recording: {exc}") from None
        try:
            rec = self._inspect(ser)
        except BaseException:
            ser.close()
            raise
        self._ser, self._rec = ser, rec
        count = ser.frame_count
        self._next = options.start_frame
        self._end = (
            count if options.max_frames is None else min(count, self._next + options.max_frames)
        )
        window = rec.stamps_ns[self._next : self._end]
        self._pass_ns = int(window[-1] - window[0]) + rec.period_ns
        self._pass = 0
        self._wrapped = False
        return CameraInfo(
            model="SER recording",
            driver=self.name,
            sdk_version=None,
            max_width=rec.frame_roi.x_end * rec.sdk_bin,
            max_height=rec.frame_roi.y_end * rec.sdk_bin,
            is_color=rec.is_color,
            has_temperature=rec.temperature_c is not None,
        )

    def _load_sidecars(self) -> tuple[BurstSidecar | None, SidecarInfo | None]:
        options = self._options
        if options.sidecar is False:
            return None, None
        try:
            if isinstance(options.sidecar, Path):
                if options.sidecar.suffix.lower() == ".json":
                    return read_burst_sidecar(options.sidecar), None
                return None, read_sharpcap_sidecar(options.sidecar)
            burst_path = burst_sidecar_path(options.path)
            if burst_path.is_file():
                return read_burst_sidecar(burst_path), None
            sharpcap_path = sharpcap_sidecar_path(options.path)
            if sharpcap_path.is_file():
                return None, read_sharpcap_sidecar(sharpcap_path)
        except SidecarError as exc:
            raise CameraConfigError(f"the sidecar cannot be used: {exc}") from None
        return None, None

    def _inspect(self, ser: SerFile) -> _Recording:
        options = self._options
        header = ser.header
        if header.planes != 1:
            raise CameraConfigError("the replay driver reads mono and raw mosaic recordings only")
        if ser.frame_count == 0:
            raise CameraConfigError("the recording has no frames")
        if options.start_frame >= ser.frame_count:
            raise CameraConfigError("start_frame is past the last frame of the recording")
        burst, sharpcap = self._load_sidecars()
        stream = None if burst is None else burst.stream

        mode = (
            _first_not_none(
                options.mode,
                None if stream is None else stream.mode,
                None if sharpcap is None else sharpcap.readout_mode,
            )
            or DEFAULT_MODE
        )
        exposure_us = _first_not_none(
            options.exposure_us,
            None if stream is None else stream.exposure_us,
            None if sharpcap is None else sharpcap.exposure_us,
        )
        gain = _first_not_none(
            options.gain,
            None if stream is None else stream.gain,
            None if sharpcap is None else sharpcap.gain,
        )
        if exposure_us is None or gain is None:
            raise CameraConfigError(
                "the exposure and gain of the recording are unknown: "
                "pass the exposure_us and gain options, or put a sidecar next to the file"
            )
        adc_bits = (
            _first_not_none(options.adc_bits, None if burst is None else burst.adc_bits)
            or header.pixel_depth
        )
        temperature = _first_not_none(
            None if burst is None else burst.temperature_c,
            None if sharpcap is None else sharpcap.sensor_temperature_c,
        )
        recorded_at = None if stream is None else stream.roi  # where the burst put its ROI
        frame_roi = Roi(
            0 if recorded_at is None else recorded_at.x,
            0 if recorded_at is None else recorded_at.y,
            header.width,
            header.height,
        )
        match = _SDK_BIN.match(mode)
        sdk_bin = int(match[1]) if match is not None and int(match[1]) > 0 else 1

        try:
            stamps = ser.timestamps_utc_ns()
        except SerFormatError as exc:
            raise CameraConfigError(f"the recording has unusable timestamps: {exc}") from None
        period_ns = None if stamps is None else _estimate_period_ns(stamps)
        if period_ns is None:
            rate_hz = None if sharpcap is None else sharpcap.fps
            if rate_hz is not None:
                period_ns = round(NS_PER_S / rate_hz)
            elif burst is not None and burst.frame_period_s is not None:
                period_ns = round(burst.frame_period_s * NS_PER_S)
            else:
                period_ns = exposure_us * 1000  # an exposure-limited stream
        if stamps is None:
            start_ns = _first_not_none(
                None if sharpcap is None else sharpcap.start_utc_ns,
                None if burst is None else burst.start_utc_ns,
                header.start_utc_ns,
            )
            if start_ns is None:
                start_ns = self._clock.utc_ns()
            stamps = start_ns + np.arange(ser.frame_count, dtype=np.int64) * period_ns
        return _Recording(
            mode=mode,
            exposure_us=exposure_us,
            gain=gain,
            pixel_format=PixelFormat.RAW8 if header.container_bits == 8 else PixelFormat.RAW16,
            adc_bits=adc_bits,
            temperature_c=temperature,
            frame_roi=frame_roi,
            sdk_bin=sdk_bin,
            is_color=header.color_id != ColorId.MONO,
            stamps_ns=stamps,
            drops=_find_drops(stamps, period_ns),
            period_ns=period_ns,
        )

    # --- CameraDriver ---

    def close(self) -> None:
        self._running = False
        self._active = None
        self._roi = None
        if self._ser is not None:
            self._ser.close()
        self._ser = None
        self._rec = None

    def capabilities(self) -> CameraCaps:
        """The limits of the ASI294 that the recordings come from. The replay accepts any request
        that fits the recording, whatever its exposure and gain."""
        rec = self._require_open()
        return CameraCaps(
            gain_range=(0, 570),
            exposure_us_range=(32, 2_000_000_000),
            bins=(rec.sdk_bin,),
            pixel_formats=(rec.pixel_format,),
            offset_range=(0, 255),
        )

    def configure(self, config: StreamConfig) -> ActiveStream:
        """Start a new stream. The request must name the recorded mode and fit the recording.

        The returned configuration shows what the frames carry: the recorded exposure, gain,
        and pixel format, and the ROI after the software crop.
        """
        rec = self._require_open()
        self._running = False
        if config.mode != rec.mode:
            raise CameraConfigError(
                f"the recording is {rec.mode}, but the request names another mode"
            )
        roi = self._fit_roi(rec.frame_roi, config.roi)
        self._stream_id += 1
        self._seq = 0
        self._dropped_total = 0
        self._first_of_stream = True
        self._roi = roi
        applied = replace(
            config,
            exposure_us=rec.exposure_us,
            gain=rec.gain,
            pixel_format=rec.pixel_format,
            roi=roi,
        )
        self._active = ActiveStream(
            stream_id=self._stream_id,
            config=applied,
            frame_shape=(roi.height, roi.width),
            adc_bits=rec.adc_bits,
            frame_period_s=rec.period_ns / NS_PER_S,
        )
        return self._active

    @staticmethod
    def _fit_roi(frame: Roi, request: Roi | None) -> Roi:
        """Crop rules of the vendor SDK: width to a multiple of 8, height to a multiple of 2."""
        if request is None:
            return frame
        width = max(8, request.width // 8 * 8)
        height = max(2, request.height // 2 * 2)
        if width > frame.width or height > frame.height:
            raise CameraConfigError("the ROI is larger than the recording")
        return Roi(
            min(max(request.x, frame.x), frame.x_end - width),
            min(max(request.y, frame.y), frame.y_end - height),
            width,
            height,
        )

    def start(self) -> None:
        if self._active is None or self._rec is None:
            raise CameraStateError("start before configure")
        self._running = True
        self._epoch_clock_ns = self._clock.monotonic_ns()
        self._epoch_stamp_ns = None

    def read_frame(self, timeout_s: float) -> Frame:
        rec, ser, active, roi = self._rec, self._ser, self._active, self._roi
        if not self._running or rec is None or ser is None or active is None or roi is None:
            raise CameraStateError("read_frame while not capturing")
        if self._next >= self._end:
            if not self._options.loop:
                raise ReplayFinishedError("the recording has no more frames")
            self._next = self._options.start_frame
            self._pass += 1
            self._wrapped = True
        index = self._next
        arrival_ns = int(rec.stamps_ns[index]) + self._pass * self._pass_ns
        self._wait_until_due(arrival_ns, timeout_s)

        data = ser.frame(index)
        if roi != rec.frame_roi:
            top, left = roi.y - rec.frame_roi.y, roi.x - rec.frame_roi.x
            data = data[top : top + roi.height, left : left + roi.width].copy()
            data.flags.writeable = False
        exposure_ns = rec.exposure_us * 1000
        period_ns = max(rec.period_ns, exposure_ns)
        dropped = 0 if self._first_of_stream or self._wrapped else int(rec.drops[index])
        frame = Frame(
            data=data,
            stream_id=active.stream_id,
            seq=self._seq,
            t_arrival_ns=arrival_ns,
            t_utc_ns=arrival_ns - (period_ns - exposure_ns // 2),
            t_err_ns=TIME_ERROR_NS,
            t_quality=TimeQuality.ESTIMATED,
            dropped_before=dropped,
            exposure_us=rec.exposure_us,
            gain=rec.gain,
            mode=rec.mode,
            roi=roi,
            adc_bits=rec.adc_bits,
            temperature_c=rec.temperature_c,
            flags=FrameFlag.REPLAYED,
        )
        self._next = index + 1
        self._seq += 1
        self._dropped_total += dropped
        self._first_of_stream = False
        self._wrapped = False
        if active.config.kind is StreamKind.SNAPSHOT:
            self._running = False
        return frame

    def _wait_until_due(self, arrival_ns: int, timeout_s: float) -> None:
        """Sleep until the frame is due. Raises `CameraTimeoutError` if that takes too long."""
        speed = self._options.rate
        if speed is None:
            return
        if self._epoch_stamp_ns is None:
            self._epoch_stamp_ns = arrival_ns  # the first frame after `start` is due at once
        recorded_ns = arrival_ns - self._epoch_stamp_ns
        elapsed_ns = recorded_ns if speed == 1.0 else round(recorded_ns / speed)
        wait_ns = self._epoch_clock_ns + elapsed_ns - self._clock.monotonic_ns()
        if wait_ns <= 0:
            return
        if wait_ns > round(timeout_s * NS_PER_S):
            self._clock.sleep(max(timeout_s, 0.0))
            raise CameraTimeoutError(f"no frame within {timeout_s} s")
        self._clock.sleep(wait_ns / NS_PER_S)

    def stop(self) -> None:
        self._running = False

    def move_roi(self, x: int, y: int) -> Roi:
        """Move the crop window, clamped to the recorded frame."""
        rec = self._require_open()
        if self._active is None or self._roi is None:
            raise CameraStateError("move_roi before configure")
        frame, roi = rec.frame_roi, self._roi
        self._roi = Roi(
            min(max(x, frame.x), frame.x_end - roi.width),
            min(max(y, frame.y), frame.y_end - roi.height),
            roi.width,
            roi.height,
        )
        return self._roi

    def read_temperature_c(self) -> float | None:
        return None if self._rec is None else self._rec.temperature_c

    def dropped_frames(self) -> int:
        """The frames that the gaps in the recorded timestamps cost, summed over the frames read
        since `configure`."""
        return self._dropped_total

    def recover(self, level: RecoveryLevel) -> None:
        """Stop the capture. A recording has nothing else to recover."""
        self._running = False

    def _require_open(self) -> _Recording:
        if self._rec is None:
            raise CameraStateError("the recording is not open")
        return self._rec


def create(*, profile: Profile | None, clock: Clock, options: Mapping[str, object]) -> ReplayDriver:
    """Build a replay driver. The driver reads nothing from `profile`: the recording and the
    options give it the readout mode, the exposure, and the gain."""
    return ReplayDriver(ReplayOptions.from_mapping(options), clock)
