"""The JSON codecs of the connection layer."""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

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
from seeingmon.services.ipc.codec import (
    DEFAULT_ERRORS,
    MAX_MESSAGE_TEXT,
    CodecError,
    ErrorRegistry,
    decode_active_stream,
    decode_camera_caps,
    decode_camera_info,
    decode_exception,
    decode_json,
    decode_recovery_level,
    decode_roi,
    decode_stream_config,
    encode_active_stream,
    encode_camera_caps,
    encode_camera_info,
    encode_exception,
    encode_json,
    encode_roi,
    encode_stream_config,
)
from seeingmon.services.ipc.errors import (
    IpcProtocolError,
    RemoteError,
    RpcInvalidParamsError,
    RpcMethodNotFoundError,
)

rois = st.builds(
    Roi,
    x=st.integers(0, 9000),
    y=st.integers(0, 6000),
    width=st.integers(1, 9000),
    height=st.integers(1, 6000),
)
stream_configs = st.builds(
    StreamConfig,
    mode=st.text(
        alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=16
    ),
    exposure_us=st.integers(1, 2_000_000_000),
    gain=st.integers(0, 570),
    pixel_format=st.sampled_from(list(PixelFormat)),
    roi=st.one_of(st.none(), rois),
    kind=st.sampled_from(list(StreamKind)),
    offset=st.one_of(st.none(), st.integers(0, 255)),
    bandwidth_pct=st.one_of(st.none(), st.integers(40, 100)),
    high_speed=st.booleans(),
)


class TestJson:
    def test_round_trip_is_compact_utf8(self) -> None:
        data = encode_json({"a": [1, 2.5, None, True], "b": "x"})
        assert data == b'{"a":[1,2.5,null,true],"b":"x"}'
        assert decode_json(data) == {"a": [1, 2.5, None, True], "b": "x"}

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_encoding_refuses_non_finite_numbers(self, value: float) -> None:
        with pytest.raises(ValueError, match="JSON"):
            encode_json({"v": value})

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param(b"", id="empty"),
            pytest.param(b"{", id="truncated"),
            pytest.param(b"\xff\xfe", id="not-utf8"),
            pytest.param(b'{"v": NaN}', id="nan"),
            pytest.param(b'{"v": Infinity}', id="infinity"),
            pytest.param(b"[" * 100_000 + b"]" * 100_000, id="deeply-nested"),
        ],
    )
    def test_decoding_refuses_what_is_not_strict_json(self, raw: bytes) -> None:
        with pytest.raises(CodecError):
            decode_json(raw)

    def test_a_codec_error_is_a_protocol_error_and_a_value_error(self) -> None:
        assert issubclass(CodecError, IpcProtocolError)
        assert issubclass(CodecError, ValueError)

    def test_decoding_takes_a_memoryview(self) -> None:
        assert decode_json(memoryview(b"[1]")) == [1]


class TestRoi:
    @given(rois)
    def test_round_trip(self, roi: Roi) -> None:
        assert decode_roi(json.loads(encode_json(encode_roi(roi)))) == roi

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            [],
            {"x": 1, "y": 2, "width": 3},
            {"x": 1, "y": 2, "width": 3, "height": 4, "z": 5},
            {"x": 1.0, "y": 2, "width": 3, "height": 4},
            {"x": True, "y": 2, "width": 3, "height": 4},
            {"x": "1", "y": 2, "width": 3, "height": 4},
            {"x": -1, "y": 2, "width": 3, "height": 4},
            {"x": 1, "y": 2, "width": 0, "height": 4},
        ],
    )
    def test_malformed_input_is_a_codec_error(self, bad: Any) -> None:
        with pytest.raises(CodecError):
            decode_roi(bad)

    def test_an_error_names_the_field_and_not_the_value(self) -> None:
        with pytest.raises(CodecError) as raised:
            decode_roi({"x": "secret-looking", "y": 2, "width": 3, "height": 4})
        assert "roi.x" in str(raised.value)
        assert "secret-looking" not in str(raised.value)


class TestStreamConfig:
    @given(stream_configs)
    def test_round_trip(self, config: StreamConfig) -> None:
        assert decode_stream_config(json.loads(encode_json(encode_stream_config(config)))) == config

    def test_optional_fields_default(self) -> None:
        config = decode_stream_config({"mode": "bin1", "exposure_us": 2000, "gain": 120})
        assert config == StreamConfig(mode="bin1", exposure_us=2000, gain=120)

    @pytest.mark.parametrize(
        "patch",
        [
            {"mode": ""},
            {"mode": "x" * 17},
            {"mode": "café"},
            {"exposure_us": 0},
            {"gain": -1},
            {"pixel_format": "RAW12"},
            {"pixel_format": 16},
            {"kind": "stream"},
            {"roi": {"x": 0}},
            {"high_speed": 1},
            {"offset": 1.5},
            {"surprise": 1},
        ],
    )
    def test_malformed_input_is_a_codec_error(self, patch: dict[str, Any]) -> None:
        data = {"mode": "bin1", "exposure_us": 2000, "gain": 120, **patch}
        with pytest.raises(CodecError):
            decode_stream_config(data)


class TestActiveStream:
    def test_round_trip(self) -> None:
        stream = ActiveStream(
            stream_id=7,
            config=StreamConfig(mode="bin1", exposure_us=2000, gain=120, roi=Roi(8, 8, 128, 128)),
            frame_shape=(128, 128),
            adc_bits=12,
            frame_period_s=0.0113,
        )
        assert decode_active_stream(json.loads(encode_json(encode_active_stream(stream)))) == stream

    def test_the_frame_period_may_be_unknown(self) -> None:
        stream = ActiveStream(
            stream_id=0,
            config=StreamConfig(mode="bin2", exposure_us=10, gain=0),
            frame_shape=(2, 8),
            adc_bits=14,
        )
        wire = json.loads(encode_json(encode_active_stream(stream)))
        assert wire["frame_period_s"] is None
        assert decode_active_stream(wire) == stream

    @pytest.mark.parametrize(
        "patch",
        [
            {"frame_shape": [1, 2, 3]},
            {"frame_shape": "ab"},
            {"frame_shape": [1.5, 2]},
            {"stream_id": -1},
            {"stream_id": True},
            {"adc_bits": "14"},
            {"frame_period_s": "fast"},
            {"frame_period_s": float("inf")},
            {"config": None},
        ],
    )
    def test_malformed_input_is_a_codec_error(self, patch: dict[str, Any]) -> None:
        stream = ActiveStream(
            stream_id=1,
            config=StreamConfig(mode="bin1", exposure_us=1, gain=0),
            frame_shape=(2, 8),
            adc_bits=14,
        )
        wire = {**encode_active_stream(stream), **patch}
        with pytest.raises(CodecError):
            decode_active_stream(wire)


class TestCameraInfoAndCaps:
    def test_info_round_trip(self) -> None:
        info = CameraInfo(
            model="Model X",
            driver="sim",
            sdk_version="1.2.3",
            max_width=8288,
            max_height=5644,
            is_color=True,
            has_temperature=True,
        )
        assert decode_camera_info(json.loads(encode_json(encode_camera_info(info)))) == info

    def test_info_defaults_and_unknown_fields(self) -> None:
        info = decode_camera_info(
            {"model": "m", "driver": "d", "max_width": 1, "max_height": 2, "sdk_version": None}
        )
        assert (info.is_color, info.has_temperature, info.sdk_version) == (False, False, None)
        with pytest.raises(CodecError, match="unknown"):
            decode_camera_info(
                {"model": "m", "driver": "d", "max_width": 1, "max_height": 2, "serial": "x"}
            )

    def test_caps_round_trip(self) -> None:
        caps = CameraCaps(
            gain_range=(0, 570),
            exposure_us_range=(32, 2_000_000_000),
            bins=(1, 2),
            pixel_formats=(PixelFormat.RAW8, PixelFormat.RAW16),
            offset_range=(0, 255),
            roi_width_multiple=8,
            roi_height_multiple=2,
            supports_video=True,
            supports_snapshot=False,
        )
        assert decode_camera_caps(json.loads(encode_json(encode_camera_caps(caps)))) == caps

    def test_caps_without_an_offset_range(self) -> None:
        caps = CameraCaps((0, 1), (1, 2), (1,), (PixelFormat.RAW16,))
        assert decode_camera_caps(encode_camera_caps(caps)) == caps

    @pytest.mark.parametrize(
        "patch",
        [
            {"gain_range": [0]},
            {"bins": [1, "2"]},
            {"bins": "12"},
            {"pixel_formats": ["RAW9"]},
            {"pixel_formats": "RAW8"},
            {"offset_range": [1]},
            {"supports_video": "yes"},
        ],
    )
    def test_malformed_caps_are_codec_errors(self, patch: dict[str, Any]) -> None:
        wire = {
            **encode_camera_caps(CameraCaps((0, 1), (1, 2), (1,), (PixelFormat.RAW16,))),
            **patch,
        }
        with pytest.raises(CodecError):
            decode_camera_caps(wire)


class TestRecoveryLevel:
    @pytest.mark.parametrize("level", list(RecoveryLevel))
    def test_every_level_round_trips(self, level: RecoveryLevel) -> None:
        assert decode_recovery_level(int(level)) is level

    @pytest.mark.parametrize("bad", [0, 99, True, "1", 1.0, None])
    def test_other_values_are_refused(self, bad: Any) -> None:
        with pytest.raises(CodecError):
            decode_recovery_level(bad)


class TestExceptions:
    @pytest.mark.parametrize(
        "cls",
        [
            CameraError,
            CameraTimeoutError,
            CameraDisconnectedError,
            CameraConfigError,
            CameraStateError,
            ValueError,
            TypeError,
            RpcMethodNotFoundError,
            RpcInvalidParamsError,
        ],
    )
    def test_registered_classes_come_back_as_themselves(self, cls: type[Exception]) -> None:
        wire = json.loads(encode_json(encode_exception(cls("it broke"))))
        restored = decode_exception(wire)
        assert type(restored) is cls
        assert str(restored) == "it broke"

    def test_a_subclass_arrives_as_its_nearest_registered_parent(self) -> None:
        class UsbStallError(CameraTimeoutError):
            pass

        wire = encode_exception(UsbStallError("stalled"))
        assert wire["type"] == "CameraTimeoutError"
        assert isinstance(decode_exception(wire), CameraTimeoutError)

    def test_the_names_are_stable(self) -> None:
        assert encode_exception(CameraStateError("x"))["type"] == "CameraStateError"
        assert encode_exception(RpcMethodNotFoundError("x"))["type"] == "MethodNotFound"
        assert encode_exception(RpcInvalidParamsError("x"))["type"] == "InvalidParams"

    def test_an_unregistered_exception_is_an_internal_error_without_a_traceback(self) -> None:
        wire = encode_exception(ZeroDivisionError("division by zero"))
        assert wire == {"type": "InternalError", "message": "ZeroDivisionError: division by zero"}
        restored = decode_exception(wire)
        assert isinstance(restored, RemoteError)
        assert restored.remote_type == "InternalError"

    def test_an_unknown_name_never_imports_or_builds_a_class(self) -> None:
        restored = decode_exception({"type": "os.system", "message": "echo hello"})
        assert isinstance(restored, RemoteError)
        assert restored.remote_type == "os.system"

    def test_a_long_message_is_cut(self) -> None:
        wire = encode_exception(CameraError("x" * 10_000))
        assert len(wire["message"]) == MAX_MESSAGE_TEXT
        restored = decode_exception({"type": "CameraError", "message": "y" * 10_000})
        assert len(str(restored)) == MAX_MESSAGE_TEXT

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            {"type": "CameraError"},
            {"type": 1, "message": "m"},
            {"type": "A" * 200, "message": ""},
        ],
    )
    def test_a_malformed_error_is_a_codec_error(self, bad: Any) -> None:
        with pytest.raises(CodecError):
            decode_exception(bad)

    def test_a_registry_can_be_extended_without_changing_the_default(self) -> None:
        class SinkFullError(Exception):
            pass

        registry = DEFAULT_ERRORS.copy()
        registry.register(SinkFullError)
        wire = encode_exception(SinkFullError("full"), registry)
        assert wire["type"] == "SinkFullError"
        assert type(decode_exception(wire, registry)) is SinkFullError
        assert encode_exception(SinkFullError("full"))["type"] == "InternalError"

    def test_one_name_cannot_serve_two_classes(self) -> None:
        registry = ErrorRegistry([CameraError])
        with pytest.raises(ValueError, match="another class"):
            registry.register(KeyError, "CameraError")
