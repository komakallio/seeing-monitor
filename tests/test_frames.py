from __future__ import annotations

import contextlib
import struct
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.frames import (
    FRAME_HEADER_SIZE,
    ActiveStream,
    Frame,
    FrameDecodeError,
    FrameFlag,
    PixelFormat,
    Roi,
    StreamConfig,
    StreamKind,
    TimeQuality,
    decode_frame,
    encode_frame,
    frames_equal,
)

# Byte offsets in the version 1 header, from the struct layout in `seeingmon.frames`.
OFFSET_VERSION = 4
OFFSET_PIXEL_BITS = 5
OFFSET_T_QUALITY = 6
OFFSET_HEADER_SIZE = 8
OFFSET_T_ERR = 40
OFFSET_ROI_WIDTH = 64
OFFSET_MODE = 72
OFFSET_PAYLOAD_SIZE = 88


def make_frame(
    width: int = 8,
    height: int = 4,
    pixel_format: PixelFormat = PixelFormat.RAW16,
    **overrides: object,
) -> Frame:
    rng = np.random.default_rng(7)
    dtype = pixel_format.dtype
    data = rng.integers(0, np.iinfo(dtype).max, size=(height, width), dtype=dtype, endpoint=True)
    fields: dict[str, object] = {
        "data": data,
        "stream_id": 3,
        "seq": 41,
        "t_arrival_ns": 1_700_000_000_000_000_123,
        "t_utc_ns": 1_700_000_000_000_000_000,
        "t_err_ns": 2_500_000,
        "t_quality": TimeQuality.FITTED,
        "dropped_before": 2,
        "exposure_us": 2_000,
        "gain": 120,
        "mode": "bin1",
        "roi": Roi(16, 24, width, height),
        "adc_bits": 12,
        "temperature_c": 18.3,
        "flags": FrameFlag.RECOVERED,
    }
    fields.update(overrides)
    return Frame(**fields)  # type: ignore[arg-type]


@st.composite
def frames(draw: st.DrawFn) -> Frame:
    return make_frame(
        width=draw(st.integers(1, 40)),
        height=draw(st.integers(1, 40)),
        pixel_format=draw(st.sampled_from(list(PixelFormat))),
        stream_id=draw(st.integers(0, 2**32 - 1)),
        seq=draw(st.integers(0, 2**64 - 1)),
        t_arrival_ns=draw(st.integers(-(2**62), 2**62)),
        t_utc_ns=draw(st.integers(-(2**62), 2**62)),
        t_err_ns=draw(st.integers(0, 2**62)),
        t_quality=draw(st.sampled_from(list(TimeQuality))),
        dropped_before=draw(st.integers(0, 2**32 - 1)),
        exposure_us=draw(st.integers(1, 2**32 - 1)),
        gain=draw(st.integers(0, 2**32 - 1)),
        mode=draw(
            st.text(
                alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=16
            )
        ),
        adc_bits=draw(st.integers(0, 255)),
        temperature_c=draw(st.none() | st.integers(-300_000, 300_000).map(lambda mc: mc / 1000)),
        flags=FrameFlag(draw(st.integers(0, 0xFFFF))),
    )


class TestRoi:
    def test_edges_and_containment(self) -> None:
        roi = Roi(10, 20, 8, 4)
        assert (roi.x_end, roi.y_end) == (18, 24)
        assert roi.contains(10, 20)
        assert roi.contains(17.9, 23.9)
        assert not roi.contains(18, 20)

    def test_distance_to_the_nearest_edge(self) -> None:
        roi = Roi(0, 0, 100, 50)
        assert roi.distance_to_edge(50, 25) == 25
        assert roi.distance_to_edge(3, 25) == 3
        assert roi.distance_to_edge(-2, 25) == -2

    @pytest.mark.parametrize("args", [(-1, 0, 8, 2), (0, -1, 8, 2), (0, 0, 0, 2), (0, 0, 8, 0)])
    def test_rejects_invalid_geometry(self, args: tuple[int, int, int, int]) -> None:
        with pytest.raises(ValueError, match="invalid ROI"):
            Roi(*args)


class TestStreamConfig:
    def test_defaults(self) -> None:
        config = StreamConfig(mode="bin1", exposure_us=2000, gain=120)
        assert config.pixel_format is PixelFormat.RAW16
        assert config.kind is StreamKind.VIDEO
        assert config.roi is None

    @pytest.mark.parametrize(
        "overrides",
        [{"exposure_us": 0}, {"gain": -1}, {"mode": ""}, {"mode": "x" * 17}, {"mode": "bïn"}],
    )
    def test_rejects_invalid_settings(self, overrides: dict[str, object]) -> None:
        fields: dict[str, object] = {"mode": "bin1", "exposure_us": 2000, "gain": 1}
        fields.update(overrides)
        with pytest.raises(ValueError, match=r"exposure|gain|mode"):
            StreamConfig(**fields)  # type: ignore[arg-type]


class TestFrame:
    def test_shape_and_format_follow_the_data(self) -> None:
        frame = make_frame(width=16, height=6, pixel_format=PixelFormat.RAW8)
        assert frame.shape == (6, 16)
        assert frame.pixel_format is PixelFormat.RAW8

    @pytest.mark.parametrize(
        "overrides",
        [
            {"data": np.zeros((4, 8), dtype=np.float32)},
            {"data": np.zeros(32, dtype=np.uint16)},
            {"data": np.zeros((5, 8), dtype=np.uint16)},
            {"t_err_ns": -1},
            {"dropped_before": -1},
            {"seq": -1},
            {"mode": ""},
        ],
    )
    def test_rejects_inconsistent_frames(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValueError, match=r"data|negative|mode"):
            make_frame(**overrides)  # type: ignore[arg-type]

    def test_active_stream_holds_the_confirmed_geometry(self) -> None:
        config = StreamConfig(mode="bin1", exposure_us=2000, gain=120, roi=Roi(0, 0, 128, 128))
        active = ActiveStream(stream_id=1, config=config, frame_shape=(128, 128), adc_bits=12)
        assert replace(active, stream_id=2).stream_id == 2
        assert active.frame_period_s is None


class TestWireFormat:
    def test_header_is_96_bytes_and_precedes_the_pixels(self) -> None:
        message = encode_frame(make_frame())
        assert FRAME_HEADER_SIZE == 96
        assert len(message) == 96 + 8 * 4 * 2
        assert message[:4] == b"SMFR"

    @pytest.mark.parametrize("pixel_format", list(PixelFormat))
    def test_round_trip(self, pixel_format: PixelFormat) -> None:
        frame = make_frame(width=24, height=10, pixel_format=pixel_format)
        assert frames_equal(decode_frame(encode_frame(frame)), frame)

    def test_round_trip_without_a_temperature(self) -> None:
        frame = make_frame(temperature_c=None)
        decoded = decode_frame(encode_frame(frame))
        assert decoded.temperature_c is None
        assert frames_equal(decoded, frame)

    def test_temperature_keeps_millikelvin_resolution(self) -> None:
        decoded = decode_frame(encode_frame(make_frame(temperature_c=-12.3456)))
        assert decoded.temperature_c == pytest.approx(-12.346, abs=1e-9)

    @given(frames())
    def test_round_trip_property(self, frame: Frame) -> None:
        assert frames_equal(decode_frame(encode_frame(frame)), frame)

    def test_decoded_pixels_share_the_buffer_and_are_read_only(self) -> None:
        decoded = decode_frame(encode_frame(make_frame()))
        assert not decoded.data.flags.writeable
        assert not decoded.data.flags.owndata
        with pytest.raises(ValueError, match="read-only"):
            decoded.data[0, 0] = 1

    def test_accepts_bytearray_and_memoryview(self) -> None:
        frame = make_frame()
        message = encode_frame(frame)
        assert frames_equal(decode_frame(bytearray(message)), frame)
        assert frames_equal(decode_frame(memoryview(message)), frame)

    def test_a_longer_header_from_a_newer_writer_is_skipped(self) -> None:
        frame = make_frame()
        message = bytearray(encode_frame(frame))
        message[FRAME_HEADER_SIZE:FRAME_HEADER_SIZE] = b"\xaa" * 8
        struct.pack_into("<H", message, OFFSET_HEADER_SIZE, FRAME_HEADER_SIZE + 8)
        assert frames_equal(decode_frame(message), frame)

    def test_every_truncation_is_rejected(self) -> None:
        message = encode_frame(make_frame(width=8, height=2))
        for length in range(len(message)):
            with pytest.raises(FrameDecodeError):
                decode_frame(message[:length])

    def test_trailing_bytes_are_rejected(self) -> None:
        with pytest.raises(FrameDecodeError, match="length"):
            decode_frame(encode_frame(make_frame()) + b"\0")

    @pytest.mark.parametrize(
        ("offset", "fmt", "value", "match"),
        [
            (0, "<4s", b"XXXX", "magic"),
            (OFFSET_VERSION, "<B", 2, "version"),
            (OFFSET_PIXEL_BITS, "<B", 12, "pixel size"),
            (OFFSET_T_QUALITY, "<B", 9, "not a valid"),
            (OFFSET_HEADER_SIZE, "<H", 40, "header size"),
            (OFFSET_ROI_WIDTH, "<H", 0, "ROI"),
            (OFFSET_PAYLOAD_SIZE, "<I", 7, "payload"),
            (OFFSET_T_ERR, "<q", -1, "negative"),
        ],
    )
    def test_corrupt_headers_are_rejected(
        self, offset: int, fmt: str, value: object, match: str
    ) -> None:
        message = bytearray(encode_frame(make_frame()))
        struct.pack_into(fmt, message, offset, value)
        with pytest.raises(FrameDecodeError, match=match):
            decode_frame(message)

    @pytest.mark.parametrize("mode_bytes", [b"\xff" + b"\0" * 15, b"\0" * 16])
    def test_a_non_ascii_or_empty_mode_is_rejected(self, mode_bytes: bytes) -> None:
        message = bytearray(encode_frame(make_frame()))
        message[OFFSET_MODE : OFFSET_MODE + 16] = mode_bytes
        with pytest.raises(FrameDecodeError):
            decode_frame(message)

    @given(st.binary(max_size=300))
    def test_random_bytes_never_crash_the_decoder(self, blob: bytes) -> None:
        with contextlib.suppress(FrameDecodeError):
            decode_frame(blob)

    @given(st.data())
    def test_flipping_a_header_byte_decodes_or_raises_decode_error(
        self, data: st.DataObject
    ) -> None:
        message = bytearray(encode_frame(make_frame()))
        position = data.draw(st.integers(0, FRAME_HEADER_SIZE - 1))
        message[position] ^= data.draw(st.integers(1, 255))
        with contextlib.suppress(FrameDecodeError):
            decode_frame(message)
