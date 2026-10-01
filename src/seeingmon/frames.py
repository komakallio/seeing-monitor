"""Frames, stream settings, and the wire format between processes.

A `Frame` is one image from a camera with the metadata that analysis needs. A
`StreamConfig` is what the scheduler asks a driver for, and an `ActiveStream` is what the
driver confirms after it reads the geometry back.

**Time.** `t_arrival_ns` is the real-time clock reading right after the read returned.
`t_utc_ns` is the best estimate of UTC at the middle of the exposure of the first row of
the ROI. With a rolling shutter, row `r` of the ROI exposes `r * row_time` later (the row
time comes from the profile's readout mode). `t_err_ns` is the 1-sigma error of `t_utc_ns`,
and `t_quality` says how the time was derived.

**Wire format.** Frames cross process boundaries as bytes with a fixed 96-byte header. No
pickle crosses a boundary. `encode_frame` and `decode_frame` are the only code that knows
the layout. Transports add their own length prefix (`multiprocessing.connection` does). A message
may hold several frames, one after the other: each header states the size of its frame, so
`decode_frames` finds where the next one starts. A message of one frame is the message of
`encode_frame`.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, IntFlag, StrEnum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FrameData: TypeAlias = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]

MODE_NAME_BYTES = 16
NO_TEMPERATURE = -(2**31)


def _check_mode_name(mode: str) -> None:
    if not mode or not mode.isascii() or len(mode) > MODE_NAME_BYTES:
        raise ValueError(f"mode must be 1 to {MODE_NAME_BYTES} ASCII characters: {mode!r}")


class PixelFormat(IntEnum):
    """How a driver delivers pixels. The value is the container size in bits."""

    RAW8 = 8
    RAW16 = 16

    @property
    def dtype(self) -> np.dtype[np.uint8] | np.dtype[np.uint16]:
        return np.dtype(np.uint8) if self is PixelFormat.RAW8 else np.dtype(np.uint16)


class StreamKind(StrEnum):
    """`VIDEO` streams frames continuously. `SNAPSHOT` takes one exposure per `start`."""

    VIDEO = "video"
    SNAPSHOT = "snapshot"


class TimeQuality(IntEnum):
    """How `Frame.t_utc_ns` was derived."""

    INVALID = 0  # the host clock was not synchronized
    ESTIMATED = 1  # arrival time minus a latency estimate
    FITTED = 2  # smoothed by a fit of arrival time against frame number
    EXACT = 3  # simulated, or stamped by hardware


class FrameFlag(IntFlag):
    """Facts about how a frame was produced. Analysis stores them with its results."""

    NONE = 0
    TIME_INVALID = 1
    RECOVERED = 2  # the first frame after a driver recovery step
    INCOMPLETE = 4  # the driver reported a short or corrupt transfer
    SIMULATED = 8
    REPLAYED = 16


@dataclass(frozen=True, slots=True)
class Roi:
    """A rectangle of the sensor, in pixels of the active readout mode (after binning)."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError(f"invalid ROI: {self}")

    @property
    def x_end(self) -> int:
        return self.x + self.width

    @property
    def y_end(self) -> int:
        return self.y + self.height

    def contains(self, x: float, y: float) -> bool:
        """Whether the point lies inside the ROI, in sensor pixel coordinates."""
        return self.x <= x < self.x_end and self.y <= y < self.y_end

    def distance_to_edge(self, x: float, y: float) -> float:
        """Distance from a point to the nearest ROI edge. Negative outside the ROI."""
        return min(x - self.x, self.x_end - x, y - self.y, self.y_end - y)


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """What the scheduler asks a driver for. `mode` names a readout mode in the profile."""

    mode: str
    exposure_us: int
    gain: int
    pixel_format: PixelFormat = PixelFormat.RAW16
    roi: Roi | None = None  # None means the full frame of the mode
    kind: StreamKind = StreamKind.VIDEO
    offset: int | None = None
    bandwidth_pct: int | None = None
    high_speed: bool = False

    def __post_init__(self) -> None:
        _check_mode_name(self.mode)
        if self.exposure_us <= 0:
            raise ValueError("exposure_us must be positive")
        if self.gain < 0:
            raise ValueError("gain must not be negative")


@dataclass(frozen=True, slots=True)
class ActiveStream:
    """What the camera confirmed after `configure`: the settings it applied and the geometry.

    `config` carries the settings as applied, with `roi` filled in. `stream_id` is new for
    every `configure`, so a window of analysis never spans a reconfiguration.
    """

    stream_id: int
    config: StreamConfig
    frame_shape: tuple[int, int]  # (height, width)
    adc_bits: int
    frame_period_s: float | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """One image and its metadata. `eq=False` because array equality is elementwise.

    `data` is a 2-D array of `uint8` or `uint16` with shape `(roi.height, roi.width)`. For
    16-bit frames, the ADC value sits in the high bits, as the vendor SDK delivers it.
    `dropped_before` counts the frames lost immediately before this one, as far as the
    producer knows. `adc_bits` is the sensor's ADC depth.
    """

    data: FrameData
    stream_id: int
    seq: int
    t_arrival_ns: int
    t_utc_ns: int
    t_err_ns: int
    t_quality: TimeQuality
    dropped_before: int
    exposure_us: int
    gain: int
    mode: str
    roi: Roi
    adc_bits: int
    temperature_c: float | None = None
    flags: FrameFlag = FrameFlag.NONE

    def __post_init__(self) -> None:
        if self.data.ndim != 2 or self.data.dtype not in (np.uint8, np.uint16):
            raise ValueError("data must be a 2-D uint8 or uint16 array")
        if self.data.shape != (self.roi.height, self.roi.width):
            raise ValueError(f"data shape {self.data.shape} does not match {self.roi}")
        if self.t_err_ns < 0 or self.dropped_before < 0 or self.seq < 0 or self.stream_id < 0:
            raise ValueError("t_err_ns, dropped_before, seq, and stream_id must not be negative")
        _check_mode_name(self.mode)

    @property
    def pixel_format(self) -> PixelFormat:
        return PixelFormat.RAW8 if self.data.dtype == np.uint8 else PixelFormat.RAW16

    @property
    def shape(self) -> tuple[int, int]:
        return (self.roi.height, self.roi.width)


def _scalar_fields(frame: Frame) -> tuple[object, ...]:
    return (
        frame.stream_id,
        frame.seq,
        frame.t_arrival_ns,
        frame.t_utc_ns,
        frame.t_err_ns,
        frame.t_quality,
        frame.dropped_before,
        frame.exposure_us,
        frame.gain,
        frame.mode,
        frame.roi,
        frame.adc_bits,
        frame.flags,
    )


def frames_equal(a: Frame, b: Frame) -> bool:
    """Compare two frames field by field, including the pixels. Temperatures compare to 1 mK."""
    if _scalar_fields(a) != _scalar_fields(b):
        return False
    if (a.temperature_c is None) != (b.temperature_c is None):
        return False
    if (
        a.temperature_c is not None
        and b.temperature_c is not None
        and abs(a.temperature_c - b.temperature_c) > 5e-4
    ):
        return False
    return a.data.dtype == b.data.dtype and bool(np.array_equal(a.data, b.data))


# --- Wire format -------------------------------------------------------------------------

FRAME_MAGIC = b"SMFR"
FRAME_WIRE_VERSION = 1
_HEADER = struct.Struct("<4sBBBBHHIQqqqIIIHHHHi16sI4x")
FRAME_HEADER_SIZE = _HEADER.size
if FRAME_HEADER_SIZE != 96:  # the layout is part of the contract
    raise RuntimeError(f"frame header is {FRAME_HEADER_SIZE} bytes, expected 96")
# Where a header states its own size and the size of its pixels, for `decode_frames`.
_HEADER_SIZE_AT = 8
_HEADER_SIZE_FIELD = struct.Struct("<H")
_PAYLOAD_SIZE_AT = 88
_PAYLOAD_SIZE_FIELD = struct.Struct("<I")


class FrameDecodeError(ValueError):
    """The bytes are not a valid frame message."""


def frame_wire_size(frame: Frame) -> int:
    """The number of bytes that `encode_frame` and `encode_frame_into` write for a frame."""
    return FRAME_HEADER_SIZE + int(frame.data.nbytes)


def encode_frame_into(out: bytearray | memoryview, frame: Frame) -> None:
    """Write a frame in the wire format into `out`, which holds `frame_wire_size(frame)` bytes.

    The header is packed in place and the pixels are copied once, so the call makes no
    temporary buffer. `encode_frame` makes the same bytes.
    """
    size = frame_wire_size(frame)
    if len(out) != size:
        raise ValueError(f"the buffer holds {len(out)} bytes and the frame needs {size}")
    temperature_mc = (
        NO_TEMPERATURE if frame.temperature_c is None else round(frame.temperature_c * 1000)
    )
    _HEADER.pack_into(
        out,
        0,
        FRAME_MAGIC,
        FRAME_WIRE_VERSION,
        frame.pixel_format.value,
        int(frame.t_quality),
        frame.adc_bits,
        FRAME_HEADER_SIZE,
        int(frame.flags) & 0xFFFF,
        frame.stream_id,
        frame.seq,
        frame.t_arrival_ns,
        frame.t_utc_ns,
        frame.t_err_ns,
        frame.dropped_before,
        frame.exposure_us,
        frame.gain,
        frame.roi.x,
        frame.roi.y,
        frame.roi.width,
        frame.roi.height,
        temperature_mc,
        frame.mode.encode("ascii"),
        frame.data.nbytes,
    )
    data = frame.data if frame.data.flags.c_contiguous else np.ascontiguousarray(frame.data)
    memoryview(out)[FRAME_HEADER_SIZE:] = memoryview(data).cast("B")


def encode_frames_into(out: bytearray | memoryview, frames: Sequence[Frame]) -> None:
    """Write several frames, one after the other, into `out`.

    `out` holds the sum of `frame_wire_size` over `frames`. `decode_frames` reads them back.
    """
    view = memoryview(out)
    offset = 0
    for frame in frames:
        end = offset + frame_wire_size(frame)
        encode_frame_into(view[offset:end], frame)
        offset = end
    if offset != view.nbytes:
        raise ValueError(f"the buffer holds {view.nbytes} bytes and the frames need {offset}")


def encode_frame(frame: Frame) -> bytes:
    """Serialize a frame: the 96-byte header, then the pixels in row-major order."""
    if frame.temperature_c is None:
        temperature_mc = NO_TEMPERATURE
    else:
        temperature_mc = round(frame.temperature_c * 1000)
    header = _HEADER.pack(
        FRAME_MAGIC,
        FRAME_WIRE_VERSION,
        frame.pixel_format.value,
        int(frame.t_quality),
        frame.adc_bits,
        FRAME_HEADER_SIZE,
        int(frame.flags) & 0xFFFF,
        frame.stream_id,
        frame.seq,
        frame.t_arrival_ns,
        frame.t_utc_ns,
        frame.t_err_ns,
        frame.dropped_before,
        frame.exposure_us,
        frame.gain,
        frame.roi.x,
        frame.roi.y,
        frame.roi.width,
        frame.roi.height,
        temperature_mc,
        frame.mode.encode("ascii"),
        frame.data.nbytes,
    )
    return header + frame.data.tobytes()


def decode_frame(buffer: bytes | bytearray | memoryview) -> Frame:
    """Parse a frame message. Raises `FrameDecodeError` for anything malformed.

    The returned array shares memory with `buffer` and is read-only. Copy it
    (`numpy.array(frame.data)`) if you keep it after the buffer is reused.
    """
    view = memoryview(buffer).cast("B")
    if view.nbytes < FRAME_HEADER_SIZE:
        raise FrameDecodeError("message is shorter than the header")
    (
        magic,
        version,
        pixel_bits,
        t_quality,
        adc_bits,
        header_size,
        flags,
        stream_id,
        seq,
        t_arrival_ns,
        t_utc_ns,
        t_err_ns,
        dropped_before,
        exposure_us,
        gain,
        roi_x,
        roi_y,
        roi_width,
        roi_height,
        temperature_mc,
        raw_mode,
        payload_size,
    ) = _HEADER.unpack_from(view)
    if magic != FRAME_MAGIC:
        raise FrameDecodeError("bad magic")
    if version != FRAME_WIRE_VERSION:
        raise FrameDecodeError(f"unsupported wire version {version}")
    if header_size < FRAME_HEADER_SIZE:
        raise FrameDecodeError("header size is smaller than the version 1 header")
    if pixel_bits not in (8, 16):
        raise FrameDecodeError(f"unsupported pixel size {pixel_bits}")
    if roi_width == 0 or roi_height == 0:
        raise FrameDecodeError("empty ROI")
    if payload_size != roi_width * roi_height * (pixel_bits // 8):
        raise FrameDecodeError("payload size does not match the ROI")
    if view.nbytes != header_size + payload_size:
        raise FrameDecodeError("message length does not match the header")
    if t_err_ns < 0:
        raise FrameDecodeError("negative time error")
    try:
        quality = TimeQuality(t_quality)
        mode = raw_mode.rstrip(b"\0").decode("ascii")
        count = roi_width * roi_height
        pixels: FrameData
        if pixel_bits == 8:
            pixels = np.frombuffer(view, dtype=np.uint8, count=count, offset=header_size)
        else:
            pixels = np.frombuffer(view, dtype=np.uint16, count=count, offset=header_size)
        pixels = pixels.reshape(roi_height, roi_width)
        pixels.flags.writeable = False
        return Frame(
            data=pixels,
            stream_id=stream_id,
            seq=seq,
            t_arrival_ns=t_arrival_ns,
            t_utc_ns=t_utc_ns,
            t_err_ns=t_err_ns,
            t_quality=quality,
            dropped_before=dropped_before,
            exposure_us=exposure_us,
            gain=gain,
            mode=mode,
            roi=Roi(roi_x, roi_y, roi_width, roi_height),
            adc_bits=adc_bits,
            temperature_c=None if temperature_mc == NO_TEMPERATURE else temperature_mc / 1000,
            flags=FrameFlag(flags),
        )
    except ValueError as exc:  # includes UnicodeDecodeError, an invalid enum, or a bad frame
        raise FrameDecodeError(str(exc)) from exc


def decode_frames(buffer: bytes | bytearray | memoryview) -> list[Frame]:
    """Parse a message that holds one or more frames, one after the other.

    A message of one frame is the message of `encode_frame`, so `decode_frames` reads it too.
    Raises `FrameDecodeError` for anything malformed, including a frame that the message cuts
    short and bytes after the last frame. Each frame shares memory with `buffer`, as in
    `decode_frame`.
    """
    view = memoryview(buffer).cast("B")
    end = view.nbytes
    frames: list[Frame] = []
    offset = 0
    while offset < end:
        if end - offset < FRAME_HEADER_SIZE:
            raise FrameDecodeError("message is shorter than the header")
        (header_size,) = _HEADER_SIZE_FIELD.unpack_from(view, offset + _HEADER_SIZE_AT)
        (payload_size,) = _PAYLOAD_SIZE_FIELD.unpack_from(view, offset + _PAYLOAD_SIZE_AT)
        size = header_size + payload_size
        if size > end - offset:
            raise FrameDecodeError("message length does not match the header")
        frames.append(decode_frame(view[offset : offset + size]))
        offset += size
    if not frames:
        raise FrameDecodeError("message is shorter than the header")
    return frames
