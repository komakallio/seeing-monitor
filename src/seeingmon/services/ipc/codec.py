"""JSON for the messages of the connection layer.

Every message that is not a frame is UTF-8 JSON. This module holds the JSON helpers and the
codecs for the types that cross the boundary between `core` and `acquire`: `Roi`,
`StreamConfig`, `ActiveStream`, `CameraInfo`, `CameraCaps`, and the camera exceptions.

A decoder checks every field, because the bytes come from another process. It rejects a
`bool` where a number belongs, a float where an integer belongs, a missing or extra field,
and `NaN` or infinity. A bad message raises `CodecError`, which names the field and never
echoes a value.

**Exceptions.** `encode_exception` sends the name of the nearest registered class and the
message. `decode_exception` builds an instance of the registered class, and a name that the
receiver does not know becomes a `RemoteError`. The receiver never imports or instantiates a
class by a name that the peer sends: only registered classes come back.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

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
from seeingmon.frames import ActiveStream, PixelFormat, Roi, StreamConfig, StreamKind
from seeingmon.services.ipc.errors import (
    IpcProtocolError,
    RemoteError,
    RpcInvalidParamsError,
    RpcMethodNotFoundError,
)

MAX_MESSAGE_TEXT = 2000  # characters of an exception message that cross the boundary


class CodecError(IpcProtocolError, ValueError):
    """A message does not have the shape that the codec expects."""


# --- JSON ----------------------------------------------------------------------------------


def encode_json(value: Any) -> bytes:
    """Encode a JSON value as compact UTF-8. Raises `ValueError` for `NaN` and infinity."""
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not allowed")


def decode_json(data: bytes | bytearray | memoryview) -> Any:
    """Decode UTF-8 JSON. Raises `CodecError` for anything else, including `NaN`."""
    try:
        return json.loads(bytes(data).decode("utf-8"), parse_constant=_reject_constant)
    except (ValueError, RecursionError) as error:  # includes JSONDecodeError and UnicodeError
        raise CodecError("the message is not valid JSON") from error


# --- Field checks --------------------------------------------------------------------------


def as_mapping(value: Any, what: str) -> Mapping[str, Any]:
    """The value as a JSON object, or `CodecError`."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CodecError(f"{what} must be a JSON object")
    return value


def _check_keys(
    data: Mapping[str, Any], what: str, required: Iterable[str], optional: Iterable[str] = ()
) -> None:
    required_set, optional_set = set(required), set(optional)
    missing = required_set - data.keys()
    if missing:
        raise CodecError(f"{what} lacks {', '.join(sorted(missing))}")
    unknown = data.keys() - required_set - optional_set
    if unknown:
        raise CodecError(f"{what} has unknown fields: {', '.join(sorted(unknown))}")


def get_int(data: Mapping[str, Any], key: str, what: str) -> int:
    """A required integer field. A bool or a float does not count."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CodecError(f"{what}.{key} must be an integer")
    return value


def get_opt_int(data: Mapping[str, Any], key: str, what: str) -> int | None:
    """An integer field that may be `null` or absent."""
    return None if data.get(key) is None else get_int(data, key, what)


def get_float(data: Mapping[str, Any], key: str, what: str) -> float:
    """A required finite number."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise CodecError(f"{what}.{key} must be a finite number")
    return float(value)


def get_opt_float(data: Mapping[str, Any], key: str, what: str) -> float | None:
    """A number field that may be `null` or absent."""
    return None if data.get(key) is None else get_float(data, key, what)


def get_str(data: Mapping[str, Any], key: str, what: str) -> str:
    """A required string."""
    value = data.get(key)
    if not isinstance(value, str):
        raise CodecError(f"{what}.{key} must be a string")
    return value


def get_opt_str(data: Mapping[str, Any], key: str, what: str) -> str | None:
    """A string field that may be `null` or absent."""
    return None if data.get(key) is None else get_str(data, key, what)


def get_bool(data: Mapping[str, Any], key: str, what: str) -> bool:
    """A required boolean."""
    value = data.get(key)
    if not isinstance(value, bool):
        raise CodecError(f"{what}.{key} must be true or false")
    return value


def _int_pair(value: Any, what: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise CodecError(f"{what} must be a pair of integers")
    return int(value[0]), int(value[1])


_T = TypeVar("_T")


def _build(what: str, build: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Call a constructor, and turn its `ValueError` into a `CodecError`."""
    try:
        return build(*args, **kwargs)
    except ValueError as error:
        raise CodecError(f"{what} is not valid: {error}") from None


# --- Roi -----------------------------------------------------------------------------------


def encode_roi(roi: Roi) -> dict[str, Any]:
    """`Roi` as a JSON object."""
    return {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height}


def decode_roi(value: Any) -> Roi:
    """The inverse of `encode_roi`. Raises `CodecError`."""
    data = as_mapping(value, "roi")
    _check_keys(data, "roi", ("x", "y", "width", "height"))
    return _build("roi", Roi, *(get_int(data, key, "roi") for key in ("x", "y", "width", "height")))


# --- Pixel formats and stream kinds --------------------------------------------------------


def _pixel_format(value: Any, what: str) -> PixelFormat:
    if not isinstance(value, str) or value not in PixelFormat.__members__:
        raise CodecError(f"{what} must be one of {', '.join(PixelFormat.__members__)}")
    return PixelFormat[value]


def _stream_kind(value: Any, what: str) -> StreamKind:
    try:
        return StreamKind(value)
    except ValueError:
        options = ", ".join(kind.value for kind in StreamKind)
        raise CodecError(f"{what} must be one of {options}") from None


# --- StreamConfig and ActiveStream ---------------------------------------------------------


def encode_stream_config(config: StreamConfig) -> dict[str, Any]:
    """`StreamConfig` as a JSON object."""
    return {
        "mode": config.mode,
        "exposure_us": config.exposure_us,
        "gain": config.gain,
        "pixel_format": config.pixel_format.name,
        "roi": None if config.roi is None else encode_roi(config.roi),
        "kind": config.kind.value,
        "offset": config.offset,
        "bandwidth_pct": config.bandwidth_pct,
        "high_speed": config.high_speed,
    }


def decode_stream_config(value: Any) -> StreamConfig:
    """The inverse of `encode_stream_config`. Raises `CodecError`."""
    data = as_mapping(value, "stream config")
    _check_keys(
        data,
        "stream config",
        ("mode", "exposure_us", "gain"),
        ("pixel_format", "roi", "kind", "offset", "bandwidth_pct", "high_speed"),
    )
    roi = data.get("roi")
    return _build(
        "stream config",
        StreamConfig,
        mode=get_str(data, "mode", "stream config"),
        exposure_us=get_int(data, "exposure_us", "stream config"),
        gain=get_int(data, "gain", "stream config"),
        pixel_format=_pixel_format(data.get("pixel_format", "RAW16"), "stream config.pixel_format"),
        roi=None if roi is None else decode_roi(roi),
        kind=_stream_kind(data.get("kind", "video"), "stream config.kind"),
        offset=get_opt_int(data, "offset", "stream config"),
        bandwidth_pct=get_opt_int(data, "bandwidth_pct", "stream config"),
        high_speed=get_bool(data, "high_speed", "stream config") if "high_speed" in data else False,
    )


def encode_active_stream(stream: ActiveStream) -> dict[str, Any]:
    """`ActiveStream` as a JSON object."""
    return {
        "stream_id": stream.stream_id,
        "config": encode_stream_config(stream.config),
        "frame_shape": [stream.frame_shape[0], stream.frame_shape[1]],
        "adc_bits": stream.adc_bits,
        "frame_period_s": stream.frame_period_s,
    }


def decode_active_stream(value: Any) -> ActiveStream:
    """The inverse of `encode_active_stream`. Raises `CodecError`."""
    data = as_mapping(value, "active stream")
    _check_keys(
        data,
        "active stream",
        ("stream_id", "config", "frame_shape", "adc_bits"),
        ("frame_period_s",),
    )
    height, width = _int_pair(data.get("frame_shape"), "active stream.frame_shape")
    stream_id = get_int(data, "stream_id", "active stream")
    if stream_id < 0:
        raise CodecError("active stream.stream_id must not be negative")
    return ActiveStream(
        stream_id=stream_id,
        config=decode_stream_config(data.get("config")),
        frame_shape=(height, width),
        adc_bits=get_int(data, "adc_bits", "active stream"),
        frame_period_s=get_opt_float(data, "frame_period_s", "active stream"),
    )


# --- CameraInfo and CameraCaps -------------------------------------------------------------


def encode_camera_info(info: CameraInfo) -> dict[str, Any]:
    """`CameraInfo` as a JSON object."""
    return {
        "model": info.model,
        "driver": info.driver,
        "sdk_version": info.sdk_version,
        "max_width": info.max_width,
        "max_height": info.max_height,
        "is_color": info.is_color,
        "has_temperature": info.has_temperature,
    }


def decode_camera_info(value: Any) -> CameraInfo:
    """The inverse of `encode_camera_info`. Raises `CodecError`."""
    data = as_mapping(value, "camera info")
    _check_keys(
        data,
        "camera info",
        ("model", "driver", "max_width", "max_height"),
        ("sdk_version", "is_color", "has_temperature"),
    )
    return CameraInfo(
        model=get_str(data, "model", "camera info"),
        driver=get_str(data, "driver", "camera info"),
        sdk_version=get_opt_str(data, "sdk_version", "camera info"),
        max_width=get_int(data, "max_width", "camera info"),
        max_height=get_int(data, "max_height", "camera info"),
        is_color=get_bool(data, "is_color", "camera info") if "is_color" in data else False,
        has_temperature=(
            get_bool(data, "has_temperature", "camera info") if "has_temperature" in data else False
        ),
    )


def encode_camera_caps(caps: CameraCaps) -> dict[str, Any]:
    """`CameraCaps` as a JSON object."""
    return {
        "gain_range": list(caps.gain_range),
        "exposure_us_range": list(caps.exposure_us_range),
        "bins": list(caps.bins),
        "pixel_formats": [fmt.name for fmt in caps.pixel_formats],
        "offset_range": None if caps.offset_range is None else list(caps.offset_range),
        "roi_width_multiple": caps.roi_width_multiple,
        "roi_height_multiple": caps.roi_height_multiple,
        "supports_video": caps.supports_video,
        "supports_snapshot": caps.supports_snapshot,
    }


def decode_camera_caps(value: Any) -> CameraCaps:
    """The inverse of `encode_camera_caps`. Raises `CodecError`."""
    data = as_mapping(value, "camera caps")
    _check_keys(
        data,
        "camera caps",
        ("gain_range", "exposure_us_range", "bins", "pixel_formats"),
        (
            "offset_range",
            "roi_width_multiple",
            "roi_height_multiple",
            "supports_video",
            "supports_snapshot",
        ),
    )
    bins = data.get("bins")
    formats = data.get("pixel_formats")
    if not isinstance(bins, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in bins
    ):
        raise CodecError("camera caps.bins must be a list of integers")
    if not isinstance(formats, list):
        raise CodecError("camera caps.pixel_formats must be a list")
    offset = data.get("offset_range")
    defaults = CameraCaps((0, 0), (0, 0), (), ())
    return CameraCaps(
        gain_range=_int_pair(data.get("gain_range"), "camera caps.gain_range"),
        exposure_us_range=_int_pair(data.get("exposure_us_range"), "camera caps.exposure_us_range"),
        bins=tuple(bins),
        pixel_formats=tuple(_pixel_format(item, "camera caps.pixel_formats") for item in formats),
        offset_range=None if offset is None else _int_pair(offset, "camera caps.offset_range"),
        roi_width_multiple=(
            get_int(data, "roi_width_multiple", "camera caps")
            if "roi_width_multiple" in data
            else defaults.roi_width_multiple
        ),
        roi_height_multiple=(
            get_int(data, "roi_height_multiple", "camera caps")
            if "roi_height_multiple" in data
            else defaults.roi_height_multiple
        ),
        supports_video=(
            get_bool(data, "supports_video", "camera caps")
            if "supports_video" in data
            else defaults.supports_video
        ),
        supports_snapshot=(
            get_bool(data, "supports_snapshot", "camera caps")
            if "supports_snapshot" in data
            else defaults.supports_snapshot
        ),
    )


def decode_recovery_level(value: Any) -> RecoveryLevel:
    """A `RecoveryLevel` from its integer value. Raises `CodecError`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CodecError("the recovery level must be an integer")
    try:
        return RecoveryLevel(value)
    except ValueError:
        raise CodecError("the recovery level is not known") from None


# --- Exceptions ----------------------------------------------------------------------------


class ErrorRegistry:
    """The exception classes that may cross the boundary, by name.

    Register a class with `register`. A registered class must build from a single message
    argument. `name_of` finds the nearest registered class of an exception, so a driver's own
    subclass of `CameraTimeoutError` still arrives as `CameraTimeoutError`.
    """

    def __init__(self, classes: Iterable[type[Exception]] = ()) -> None:
        self._by_name: dict[str, type[Exception]] = {}
        self._by_class: dict[type[BaseException], str] = {}
        for cls in classes:
            self.register(cls)

    def register(self, cls: type[Exception], name: str | None = None) -> None:
        """Make `cls` cross the boundary under `name` (default: the class name)."""
        key = name or cls.__name__
        if self._by_name.get(key, cls) is not cls:
            raise ValueError(f"the name {key!r} belongs to another class")
        self._by_name[key] = cls
        self._by_class[cls] = key

    def name_of(self, error: BaseException) -> str | None:
        """The name of the nearest registered class in the MRO of `error`, if any."""
        for cls in type(error).__mro__:
            name = self._by_class.get(cls)
            if name is not None:
                return name
        return None

    def build(self, name: str, message: str) -> Exception:
        """An instance of the class registered as `name`, or a `RemoteError` for another name."""
        cls = self._by_name.get(name)
        if cls is None:
            return RemoteError(name, message)
        return cls(message)

    def copy(self) -> ErrorRegistry:
        """A registry with the same classes, which you can extend without changing this one."""
        other = ErrorRegistry()
        other._by_name = dict(self._by_name)
        other._by_class = dict(self._by_class)
        return other


INTERNAL_ERROR = "InternalError"

DEFAULT_ERRORS = ErrorRegistry(
    [
        CameraError,
        CameraTimeoutError,
        CameraDisconnectedError,
        CameraConfigError,
        CameraStateError,
        ValueError,
        TypeError,
    ]
)
DEFAULT_ERRORS.register(RpcMethodNotFoundError, "MethodNotFound")
DEFAULT_ERRORS.register(RpcInvalidParamsError, "InvalidParams")


def encode_exception(
    error: BaseException, registry: ErrorRegistry = DEFAULT_ERRORS
) -> dict[str, Any]:
    """An exception as `{"type": name, "message": text}`.

    An exception without a registered ancestor arrives as `InternalError`, with the class name
    in the message, so the receiver sees what failed and no stack trace leaves the process.
    """
    name = registry.name_of(error)
    text = str(error)
    if name is None:
        name = INTERNAL_ERROR
        text = f"{type(error).__name__}: {text}" if text else type(error).__name__
    return {"type": name, "message": text[:MAX_MESSAGE_TEXT]}


def decode_exception(value: Any, registry: ErrorRegistry = DEFAULT_ERRORS) -> Exception:
    """The inverse of `encode_exception`. An unknown type becomes a `RemoteError`."""
    data = as_mapping(value, "error")
    _check_keys(data, "error", ("type", "message"))
    name = get_str(data, "type", "error")
    message = get_str(data, "message", "error")
    if len(name) > 100:
        raise CodecError("error.type is too long")
    return registry.build(name, message[:MAX_MESSAGE_TEXT])
