"""Several frames in one message: `encode_frames_into` and `decode_frames`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from seeingmon.frames import (
    FRAME_HEADER_SIZE,
    Frame,
    FrameDecodeError,
    FrameFlag,
    Roi,
    TimeQuality,
    decode_frame,
    decode_frames,
    encode_frame,
    encode_frames_into,
    frame_wire_size,
    frames_equal,
)


def make_frame(seq: int, *, width: int = 8, height: int = 6, dtype: type = np.uint16) -> Frame:
    grid = np.arange(width * height, dtype=np.uint32).reshape(height, width) + seq
    pixels: Any = grid.astype(dtype)
    return Frame(
        data=pixels,
        stream_id=3,
        seq=seq,
        t_arrival_ns=10**18 + seq,
        t_utc_ns=10**18 + seq * 1000,
        t_err_ns=500,
        t_quality=TimeQuality.FITTED,
        dropped_before=seq % 3,
        exposure_us=2000,
        gain=7,
        mode="bin1",
        roi=Roi(10, 20, width, height),
        adc_bits=14,
        temperature_c=18.5,
        flags=FrameFlag.SIMULATED,
    )


def pack(frames: list[Frame]) -> bytearray:
    out = bytearray(sum(frame_wire_size(frame) for frame in frames))
    encode_frames_into(out, frames)
    return out


class TestEncode:
    def test_the_frames_follow_one_another_in_the_format_of_a_single_frame(self) -> None:
        frames = [make_frame(seq) for seq in range(4)]
        assert bytes(pack(frames)) == b"".join(encode_frame(frame) for frame in frames)

    def test_a_buffer_of_the_wrong_size_is_refused(self) -> None:
        frames = [make_frame(0), make_frame(1)]
        too_big = bytearray(sum(frame_wire_size(frame) for frame in frames) + 1)
        with pytest.raises(ValueError, match="need"):
            encode_frames_into(too_big, frames)
        with pytest.raises(ValueError, match="needs"):
            encode_frames_into(bytearray(10), frames)

    def test_a_list_of_no_frames_needs_an_empty_buffer(self) -> None:
        encode_frames_into(bytearray(), [])
        with pytest.raises(ValueError, match="need"):
            encode_frames_into(bytearray(3), [])


class TestDecode:
    def test_a_message_of_one_frame_is_the_message_of_decode_frame(self) -> None:
        frame = make_frame(5)
        decoded = decode_frames(encode_frame(frame))
        assert len(decoded) == 1
        assert frames_equal(decoded[0], frame)
        assert frames_equal(decoded[0], decode_frame(encode_frame(frame)))

    def test_every_frame_of_a_batch_comes_back_whole_and_in_order(self) -> None:
        frames = [make_frame(seq) for seq in range(7)]
        decoded = decode_frames(pack(frames))
        assert len(decoded) == len(frames)
        assert all(frames_equal(a, b) for a, b in zip(decoded, frames, strict=True))

    def test_frames_of_different_shapes_and_depths_share_a_message(self) -> None:
        frames = [
            make_frame(0, width=4, height=4),
            make_frame(1, width=16, height=2, dtype=np.uint8),
            make_frame(2, width=3, height=9),
        ]
        decoded = decode_frames(pack(frames))
        assert [frame.data.shape for frame in decoded] == [(4, 4), (2, 16), (9, 3)]
        assert [frame.data.dtype for frame in decoded] == [np.uint16, np.uint8, np.uint16]
        assert all(frames_equal(a, b) for a, b in zip(decoded, frames, strict=True))

    def test_the_frames_share_the_memory_of_the_message(self) -> None:
        buffer = bytearray(pack([make_frame(0), make_frame(1)]))
        first, second = decode_frames(memoryview(buffer))
        assert not first.data.flags.writeable
        assert np.shares_memory(first.data, np.frombuffer(buffer, dtype=np.uint8))
        assert np.shares_memory(second.data, np.frombuffer(buffer, dtype=np.uint8))

    def test_an_empty_message_is_refused(self) -> None:
        with pytest.raises(FrameDecodeError):
            decode_frames(b"")

    def test_a_message_that_stops_inside_a_header_is_refused(self) -> None:
        data = bytes(pack([make_frame(0), make_frame(1)]))
        first = frame_wire_size(make_frame(0))
        with pytest.raises(FrameDecodeError, match="shorter than the header"):
            decode_frames(data[: first + FRAME_HEADER_SIZE - 1])

    def test_a_message_that_stops_inside_the_pixels_is_refused(self) -> None:
        data = bytes(pack([make_frame(0), make_frame(1)]))
        with pytest.raises(FrameDecodeError, match="length"):
            decode_frames(data[:-1])

    def test_bytes_after_the_last_frame_are_refused(self) -> None:
        data = bytes(pack([make_frame(0)])) + b"\x00" * 7
        with pytest.raises(FrameDecodeError):
            decode_frames(data)

    def test_a_damaged_header_is_refused_in_the_middle_of_a_batch(self) -> None:
        data = bytearray(pack([make_frame(0), make_frame(1), make_frame(2)]))
        second = frame_wire_size(make_frame(0))
        data[second : second + 4] = b"XXXX"  # the magic of the second frame
        with pytest.raises(FrameDecodeError, match="magic"):
            decode_frames(data)

    def test_a_header_that_states_no_size_does_not_loop(self) -> None:
        data = bytearray(pack([make_frame(0), make_frame(1)]))
        second = frame_wire_size(make_frame(0))
        data[second + 8 : second + 10] = b"\x00\x00"  # header_size = 0
        data[second + 88 : second + 92] = b"\x00\x00\x00\x00"  # payload_size = 0
        with pytest.raises(FrameDecodeError):
            decode_frames(data)
