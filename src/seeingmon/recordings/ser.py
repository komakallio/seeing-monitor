"""Read and write SER video files.

SER is the format that planetary and high-speed capture programs such as SharpCap and
FireCapture write. A version 3 file holds a 178-byte header, the frames in order, and an
optional trailer with one 8-byte timestamp per frame. The timestamps count 100 ns ticks since
0001-01-01 (the .NET `DateTime` tick), and the capture program takes them from the PC clock
when a frame arrives.

`SerFile` memory-maps the file, so opening a 2 GiB recording costs almost nothing and each
`frame` call reads one frame. `SerWriter` writes a file frame by frame.

**Privacy.** The header carries free text (observer, instrument, telescope) that can identify a
person or a camera. `SerHeader` keeps it out of its `repr`, and no message in this module
contains it or the path of a file.

**Byte order.** The header has a flag for the byte order of 16-bit pixels, and many programs
write it inverted. SharpCap, for example, writes 0 for the little-endian data that it
produces. `SerFile` honors the flag unless you pass `byte_order`.
"""

from __future__ import annotations

import mmap
import operator
import os
import struct
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from enum import IntEnum
from types import TracebackType
from typing import BinaryIO, Literal, Self, TypeAlias

import numpy as np
import numpy.typing as npt

from seeingmon.frames import FrameData

FILE_ID = b"LUCAM-RECORDER"
HEADER_SIZE = 178
TIMESTAMP_SIZE = 8
STRING_FIELD_SIZE = 40
NS_PER_TICK = 100
TICKS_AT_UNIX_EPOCH = 621_355_968_000_000_000  # 1970-01-01T00:00:00Z in .NET ticks

# The tick range whose Unix time in nanoseconds still fits in a signed 64-bit integer.
_TICK_RANGE = (2**63 - 1) // NS_PER_TICK
MIN_VALID_TICKS = TICKS_AT_UNIX_EPOCH - _TICK_RANGE
MAX_VALID_TICKS = TICKS_AT_UNIX_EPOCH + _TICK_RANGE

ByteOrder: TypeAlias = Literal["little", "big"]

_HEADER = struct.Struct("<14s7i40s40s40s2q")
if _HEADER.size != HEADER_SIZE:  # the layout is part of the format
    raise RuntimeError(f"SER header is {_HEADER.size} bytes, expected {HEADER_SIZE}")


class SerError(Exception):
    """A SER file cannot be opened, read, or written."""


class SerFormatError(SerError, ValueError):
    """The content is not a valid SER file, or a value does not fit the format."""


class ColorId(IntEnum):
    """The `ColorID` header field: how to read the pixels of a frame."""

    MONO = 0
    BAYER_RGGB = 8
    BAYER_GRBG = 9
    BAYER_GBRG = 10
    BAYER_BGGR = 11
    BAYER_CYYM = 16
    BAYER_YCMY = 17
    BAYER_YMCY = 18
    BAYER_MYYC = 19
    RGB = 100
    BGR = 101

    @property
    def planes(self) -> int:
        """Samples per pixel: 3 for RGB and BGR, 1 for mono and for raw mosaics."""
        return 3 if self in (ColorId.RGB, ColorId.BGR) else 1


def ticks_to_unix_ns(ticks: int) -> int:
    """Convert SER ticks (100 ns since 0001-01-01) to nanoseconds since the Unix epoch. Exact."""
    return (ticks - TICKS_AT_UNIX_EPOCH) * NS_PER_TICK


def unix_ns_to_ticks(unix_ns: int) -> int:
    """Convert Unix nanoseconds to SER ticks. The format resolves 100 ns, so this rounds down."""
    return unix_ns // NS_PER_TICK + TICKS_AT_UNIX_EPOCH


def _text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("latin-1").rstrip()


def _encode_text(text: str, name: str) -> bytes:
    try:
        encoded = text.encode("latin-1")
    except UnicodeEncodeError:
        raise SerFormatError(f"{name} must use Latin-1 characters") from None
    if len(encoded) > STRING_FIELD_SIZE:
        raise SerFormatError(f"{name} must have at most {STRING_FIELD_SIZE} characters")
    return encoded


@dataclass(frozen=True, slots=True)
class SerHeader:
    """The SER header. `observer`, `instrument`, and `telescope` stay out of `repr`.

    `endian_flag` is the `LittleEndian` field as written: 1 means little-endian pixels and 0
    means big-endian. `datetime_ticks` is the local start time and `datetime_utc_ticks` is the
    UTC start time, both as SER ticks, and 0 means the writer left the field empty.
    """

    width: int
    height: int
    pixel_depth: int
    frame_count: int
    color_id: ColorId = ColorId.MONO
    endian_flag: int = 1
    lu_id: int = 0
    observer: str = field(default="", repr=False)
    instrument: str = field(default="", repr=False)
    telescope: str = field(default="", repr=False)
    datetime_ticks: int = 0
    datetime_utc_ticks: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise SerFormatError(f"the image size {self.width} x {self.height} is not valid")
        if not 1 <= self.pixel_depth <= 16:
            raise SerFormatError(f"the pixel depth {self.pixel_depth} is not between 1 and 16")
        if not 0 <= self.frame_count < 2**31:
            raise SerFormatError(f"the frame count {self.frame_count} is not valid")

    @property
    def planes(self) -> int:
        """Samples per pixel (3 for RGB and BGR, otherwise 1)."""
        return self.color_id.planes

    @property
    def container_bits(self) -> int:
        """The size of one sample in the file: 8 for depths up to 8, otherwise 16."""
        return 8 if self.pixel_depth <= 8 else 16

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * self.planes * (self.container_bits // 8)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of one frame: `(height, width)`, or `(height, width, 3)` for RGB."""
        base = (self.height, self.width)
        return base if self.planes == 1 else (*base, self.planes)

    @property
    def start_utc_ns(self) -> int | None:
        """The UTC start time in Unix nanoseconds, or `None` when the field is empty or invalid."""
        ticks = self.datetime_utc_ticks
        if ticks == 0 or not MIN_VALID_TICKS <= ticks <= MAX_VALID_TICKS:
            return None
        return ticks_to_unix_ns(ticks)

    def file_size(self, *, trailer: bool) -> int:
        """The size in bytes of a file with this header, with or without a timestamp trailer."""
        size = HEADER_SIZE + self.frame_count * self.frame_bytes
        return size + self.frame_count * TIMESTAMP_SIZE if trailer else size

    def pack(self) -> bytes:
        """Serialize the header to its 178 bytes."""
        return _HEADER.pack(
            FILE_ID,
            self.lu_id,
            int(self.color_id),
            self.endian_flag,
            self.width,
            self.height,
            self.pixel_depth,
            self.frame_count,
            _encode_text(self.observer, "observer").ljust(STRING_FIELD_SIZE, b"\0"),
            _encode_text(self.instrument, "instrument").ljust(STRING_FIELD_SIZE, b"\0"),
            _encode_text(self.telescope, "telescope").ljust(STRING_FIELD_SIZE, b"\0"),
            self.datetime_ticks,
            self.datetime_utc_ticks,
        )

    @classmethod
    def unpack(cls, raw: bytes | bytearray | memoryview) -> SerHeader:
        """Parse the first 178 bytes of a file. Raises `SerFormatError` for anything invalid."""
        if len(raw) < HEADER_SIZE:
            raise SerFormatError(f"the data is shorter than the {HEADER_SIZE}-byte SER header")
        (
            file_id,
            lu_id,
            color_id,
            endian_flag,
            width,
            height,
            pixel_depth,
            frame_count,
            observer,
            instrument,
            telescope,
            datetime_ticks,
            datetime_utc_ticks,
        ) = _HEADER.unpack_from(raw)
        if file_id != FILE_ID:
            raise SerFormatError("the file does not start with the SER signature")
        try:
            color = ColorId(color_id)
        except ValueError:
            raise SerFormatError(f"the ColorID {color_id} is not supported") from None
        return cls(
            width=width,
            height=height,
            pixel_depth=pixel_depth,
            frame_count=frame_count,
            color_id=color,
            endian_flag=endian_flag,
            lu_id=lu_id,
            observer=_text(observer),
            instrument=_text(instrument),
            telescope=_text(telescope),
            datetime_ticks=datetime_ticks,
            datetime_utc_ticks=datetime_utc_ticks,
        )


def _has_trailer(header: SerHeader, size: int) -> bool:
    """Whether a file of `size` bytes has a timestamp trailer. Raises when the size is wrong."""
    plain = header.file_size(trailer=False)
    full = header.file_size(trailer=True)
    if size == plain:
        return False
    if size == full:
        return True
    if size < plain:
        raise SerFormatError(
            f"the file is truncated: the header declares {header.frame_count} frames and "
            f"needs {plain} bytes, but the file has {size} bytes"
        )
    if size < full:
        raise SerFormatError(
            f"the file has {size - plain} bytes after the last frame, but a complete "
            f"timestamp trailer has {full - plain} bytes"
        )
    raise SerFormatError(
        f"the file is {size - full} bytes longer than the header declares ({full} bytes)"
    )


class SerFile:
    """A SER file, memory-mapped and read-only. Use it as a context manager.

    The constructor validates the header and checks that the file size matches the header
    exactly: the header, the frames, and either no trailer or a complete one. It raises
    `SerFormatError` otherwise. `byte_order` overrides the endian flag of the header for 16-bit
    pixels.

    `frame` returns a read-only array that does not depend on the file, so it stays valid
    after you close the `SerFile`.
    """

    def __init__(
        self, path: str | os.PathLike[str], *, byte_order: ByteOrder | None = None
    ) -> None:
        if byte_order not in (None, "little", "big"):
            raise ValueError("byte_order must be None, 'little', or 'big'")
        self._map: mmap.mmap | None = None
        try:
            with open(path, "rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                if size < HEADER_SIZE:
                    raise SerFormatError(
                        f"the file has {size} bytes, fewer than the {HEADER_SIZE}-byte header"
                    )
                header = SerHeader.unpack(handle.read(HEADER_SIZE))
                self._trailer = _has_trailer(header, size)
                self._map = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        except OSError as exc:
            # The message omits the path on purpose: recording locations stay out of logs.
            raise SerError(f"cannot read the file: {exc.strerror or type(exc).__name__}") from None
        self._header = header
        self._size = size
        self._byte_order: ByteOrder = byte_order or ("little" if header.endian_flag else "big")
        self._timestamps: npt.NDArray[np.int64] | None = None
        self._timestamps_read = False

    def __repr__(self) -> str:
        header = self._header
        return (
            f"<SerFile {header.width}x{header.height}, depth {header.pixel_depth}, "
            f"{header.frame_count} frames>"
        )

    # --- properties ---

    @property
    def header(self) -> SerHeader:
        return self._header

    @property
    def frame_count(self) -> int:
        return self._header.frame_count

    @property
    def width(self) -> int:
        return self._header.width

    @property
    def height(self) -> int:
        return self._header.height

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of one frame: `(height, width)`, or `(height, width, 3)` for RGB."""
        return self._header.shape

    @property
    def pixel_depth(self) -> int:
        return self._header.pixel_depth

    @property
    def dtype(self) -> np.dtype[np.uint8] | np.dtype[np.uint16]:
        """The dtype that `frame` returns: `uint8` for depths up to 8, otherwise `uint16`."""
        return np.dtype(np.uint8) if self._header.container_bits == 8 else np.dtype(np.uint16)

    @property
    def byte_order(self) -> ByteOrder:
        """The byte order in effect for 16-bit pixels: the override, or the header flag."""
        return self._byte_order

    @property
    def has_trailer(self) -> bool:
        """Whether the file ends with a timestamp trailer."""
        return self._trailer

    @property
    def file_size(self) -> int:
        return self._size

    @property
    def closed(self) -> bool:
        return self._map is None

    # --- reading ---

    def __len__(self) -> int:
        return self._header.frame_count

    def __iter__(self) -> Iterator[FrameData]:
        return self.frames()

    def frames(self, start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[FrameData]:
        """Iterate over frames `start` to `stop` (exclusive), with the usual slice rules."""
        for index in range(*slice(start, stop, step).indices(self.frame_count)):
            yield self.frame(index)

    def frame(self, index: int) -> FrameData:
        """Read one frame. A negative index counts from the end. Raises `IndexError` past it.

        The array is read-only and holds a copy, so it stays valid after `close`. Frames of
        RGB files have shape `(height, width, 3)`.
        """
        mapped = self._require_open()
        position = operator.index(index)
        if position < 0:
            position += self.frame_count
        if not 0 <= position < self.frame_count:
            raise IndexError(f"frame {index} is out of range for {self.frame_count} frames")
        frame_bytes = self._header.frame_bytes
        start = HEADER_SIZE + position * frame_bytes
        raw = mapped[start : start + frame_bytes]
        shape = self._header.shape
        if self._header.container_bits == 8:
            pixels8 = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            pixels8.flags.writeable = False
            return pixels8
        order = "<u2" if self._byte_order == "little" else ">u2"
        pixels16 = np.frombuffer(raw, dtype=order).astype(np.uint16, copy=False).reshape(shape)
        pixels16.flags.writeable = False
        return pixels16

    def timestamps_utc_ns(self) -> npt.NDArray[np.int64] | None:
        """The per-frame timestamps as Unix nanoseconds, or `None` when there are none.

        The result is `None` when the file has no trailer and when the trailer is all zeros,
        which some writers leave when they have no clock. The array is read-only, and the
        conversion from ticks is exact. A trailer with any other value outside the range of
        valid times raises `SerFormatError`.
        """
        if not self._timestamps_read:
            self._timestamps = self._load_timestamps()
            self._timestamps_read = True
        return self._timestamps

    def _load_timestamps(self) -> npt.NDArray[np.int64] | None:
        mapped = self._require_open()
        if not self._trailer:
            return None
        start = HEADER_SIZE + self.frame_count * self._header.frame_bytes
        raw = mapped[start : start + self.frame_count * TIMESTAMP_SIZE]
        ticks = np.frombuffer(raw, dtype="<i8")
        if not ticks.any():
            return None
        if int(ticks.min()) < MIN_VALID_TICKS or int(ticks.max()) > MAX_VALID_TICKS:
            raise SerFormatError("a timestamp in the trailer is not a valid time")
        unix_ns = (ticks - TICKS_AT_UNIX_EPOCH) * NS_PER_TICK
        unix_ns.flags.writeable = False
        return unix_ns

    # --- lifecycle ---

    def _require_open(self) -> mmap.mmap:
        if self._map is None:
            raise SerError("the file is closed")
        return self._map

    def close(self) -> None:
        """Release the memory map. Safe to call twice."""
        if self._map is not None:
            self._map.close()
            self._map = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class SerWriter:
    """Write a SER file frame by frame. Use it as a context manager.

    The writer creates the file when you construct it, and it refuses to replace an existing
    file unless you pass `overwrite=True`. Each `write_frame` takes an array with the shape
    and dtype of the file (`uint8` for depths up to 8, `uint16` above). With `timestamps=True`
    (the default) every frame needs a Unix time in nanoseconds, and the writer stores them in
    the trailer, rounded down to the 100 ns tick of the format. `close` writes the trailer and
    the final header, so the frame count and the start time are correct even when your code
    raised inside the `with` block. A start time that you do not give becomes the first
    timestamp.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        width: int,
        height: int,
        pixel_depth: int = 8,
        color: ColorId = ColorId.MONO,
        byte_order: ByteOrder = "little",
        timestamps: bool = True,
        start_utc_ns: int | None = None,
        local_offset_s: int = 0,
        observer: str = "",
        instrument: str = "",
        telescope: str = "",
        overwrite: bool = False,
    ) -> None:
        if byte_order not in ("little", "big"):
            raise ValueError("byte_order must be 'little' or 'big'")
        self._header = SerHeader(
            width=width,
            height=height,
            pixel_depth=pixel_depth,
            frame_count=0,
            color_id=color,
            endian_flag=1 if byte_order == "little" else 0,
            observer=observer,
            instrument=instrument,
            telescope=telescope,
        )
        self._header.pack()  # validates the text fields before the file exists
        self._byte_order = byte_order
        self._with_timestamps = timestamps
        self._start_utc_ns = start_utc_ns
        self._local_offset_ns = local_offset_s * 1_000_000_000
        self._ticks = array("q")
        self._count = 0
        self._closed = False
        try:
            self._file: BinaryIO = open(path, "wb" if overwrite else "xb")  # noqa: SIM115
        except FileExistsError:  # the message omits the path on purpose
            raise SerError("the file already exists") from None
        except OSError as exc:
            raise SerError(
                f"cannot create the file: {exc.strerror or type(exc).__name__}"
            ) from None
        self._file.write(self._header.pack())

    @property
    def frames_written(self) -> int:
        return self._count

    def write_frame(
        self, data: npt.NDArray[np.uint8] | npt.NDArray[np.uint16], t_utc_ns: int | None = None
    ) -> None:
        """Append a frame. `t_utc_ns` is required with timestamps and not allowed without."""
        if self._closed:
            raise SerError("the writer is closed")
        header = self._header
        if data.shape != header.shape:
            raise ValueError(f"the frame has shape {data.shape}, but the file needs {header.shape}")
        expected = np.uint8 if header.container_bits == 8 else np.uint16
        if data.dtype != expected:
            raise ValueError(
                f"the frame has dtype {data.dtype}, but the file needs {np.dtype(expected)}"
            )
        if self._with_timestamps and t_utc_ns is None:
            raise ValueError("this file stores timestamps, so every frame needs t_utc_ns")
        if not self._with_timestamps and t_utc_ns is not None:
            raise ValueError("this file stores no timestamps, so t_utc_ns is not allowed")
        if t_utc_ns is not None:
            ticks = unix_ns_to_ticks(t_utc_ns)
            if not MIN_VALID_TICKS <= ticks <= MAX_VALID_TICKS:
                raise ValueError("t_utc_ns is outside the range that SER can store")
        pixels = np.ascontiguousarray(data)
        if header.container_bits == 16:
            pixels = pixels.astype("<u2" if self._byte_order == "little" else ">u2", copy=False)
        self._file.write(pixels.tobytes())
        if t_utc_ns is not None:
            self._ticks.append(unix_ns_to_ticks(t_utc_ns))
        self._count += 1

    def close(self) -> None:
        """Write the trailer and the final header, then close the file. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._with_timestamps and self._count > 0:
                self._file.write(np.array(self._ticks, dtype="<i8").tobytes())
            start_ns = self._start_utc_ns
            if start_ns is None and self._ticks:
                start_ns = ticks_to_unix_ns(self._ticks[0])
            utc_ticks = 0 if start_ns is None else unix_ns_to_ticks(start_ns)
            local_ticks = (
                0 if start_ns is None else unix_ns_to_ticks(start_ns + self._local_offset_ns)
            )
            final = replace(
                self._header,
                frame_count=self._count,
                datetime_ticks=local_ticks,
                datetime_utc_ticks=utc_ticks,
            )
            self._file.seek(0)
            self._file.write(final.pack())
        finally:
            self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
