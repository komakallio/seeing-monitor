"""The SER reader and writer, on small synthetic files."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from typing import get_args

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.frames import FrameData
from seeingmon.recordings.ser import (
    HEADER_SIZE,
    TICKS_AT_UNIX_EPOCH,
    ByteOrder,
    ColorId,
    SerError,
    SerFile,
    SerFormatError,
    SerHeader,
    SerWriter,
    ticks_to_unix_ns,
    unix_ns_to_ticks,
)
from tests.recordings.synthetic import (
    PERIOD_NS,
    START_UTC_NS,
    make_ser,
    regular_timestamps,
    synthetic_frame,
)

ENDIAN_OFFSET = 22  # the LittleEndian field of the header
FRAME_COUNT_OFFSET = 38


def random_frame(rng: np.random.Generator, width: int, height: int, depth: int) -> FrameData:
    values = rng.integers(0, 1 << depth, size=(height, width))
    if depth <= 8:
        return values.astype(np.uint8)
    return values.astype(np.uint16)


def patch(path: Path, offset: int, data: bytes) -> None:
    with path.open("r+b") as handle:
        handle.seek(offset)
        handle.write(data)


class TestRoundTrip:
    def test_8_bit_frames_and_timestamps(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", count=6, width=16, height=8)
        with SerFile(recording.path) as ser:
            assert (ser.width, ser.height, ser.pixel_depth, ser.frame_count) == (16, 8, 8, 6)
            assert ser.shape == (8, 16)
            assert ser.dtype == np.uint8
            assert ser.has_trailer
            assert ser.file_size == HEADER_SIZE + 6 * 128 + 6 * 8
            for i, expected in enumerate(recording.frames):
                frame = ser.frame(i)
                assert frame.dtype == np.uint8
                assert np.array_equal(frame, expected)
            stamps = ser.timestamps_utc_ns()
            assert stamps is not None
            assert stamps.dtype == np.int64
            assert stamps.tolist() == recording.timestamps_ns

    @pytest.mark.parametrize("byte_order", ["little", "big"])
    def test_16_bit_frames_in_either_byte_order(
        self, tmp_path: Path, byte_order: ByteOrder
    ) -> None:
        recording = make_ser(tmp_path / "a.ser", depth=16, byte_order=byte_order)
        assert any(int(frame.max()) > 255 for frame in recording.frames)
        with SerFile(recording.path) as ser:
            assert ser.dtype == np.uint16
            assert ser.byte_order == byte_order
            assert ser.header.endian_flag == (1 if byte_order == "little" else 0)
            for i, expected in enumerate(recording.frames):
                assert np.array_equal(ser.frame(i), expected)

    def test_depths_above_8_use_a_16_bit_container(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", depth=12)
        with SerFile(recording.path) as ser:
            assert ser.pixel_depth == 12
            assert ser.dtype == np.uint16
            assert np.array_equal(ser.frame(3), recording.frames[3])

    def test_a_file_without_a_trailer_has_no_timestamps(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", timestamps=False)
        assert recording.path.stat().st_size == HEADER_SIZE + 8 * 128
        with SerFile(recording.path) as ser:
            assert not ser.has_trailer
            assert ser.timestamps_utc_ns() is None
            assert np.array_equal(ser.frame(7), recording.frames[7])

    def test_rgb_frames_have_three_planes(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ser"
        pixels = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        with SerWriter(path, width=5, height=4, color=ColorId.RGB, timestamps=False) as writer:
            writer.write_frame(pixels)
        with SerFile(path) as ser:
            assert ser.shape == (4, 5, 3)
            assert np.array_equal(ser.frame(0), pixels)

    def test_a_file_without_frames(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ser"
        with SerWriter(path, width=4, height=2):
            pass
        assert path.stat().st_size == HEADER_SIZE
        with SerFile(path) as ser:
            assert ser.frame_count == 0
            assert list(ser) == []
            assert ser.timestamps_utc_ns() is None
            with pytest.raises(IndexError):
                ser.frame(0)

    @given(
        width=st.integers(1, 12),
        height=st.integers(1, 12),
        depth=st.sampled_from([1, 8, 12, 16]),
        count=st.integers(0, 4),
        byte_order=st.sampled_from(get_args(ByteOrder)),
        with_timestamps=st.booleans(),
        seed=st.integers(0, 2**32 - 1),
    )
    def test_any_geometry_round_trips(
        self,
        width: int,
        height: int,
        depth: int,
        count: int,
        byte_order: ByteOrder,
        with_timestamps: bool,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        frames = [random_frame(rng, width, height, depth) for _ in range(count)]
        stamps = regular_timestamps(count)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.ser"
            with SerWriter(
                path,
                width=width,
                height=height,
                pixel_depth=depth,
                byte_order=byte_order,
                timestamps=with_timestamps,
            ) as writer:
                for i, frame in enumerate(frames):
                    writer.write_frame(frame, stamps[i] if with_timestamps else None)
            with SerFile(path) as ser:
                assert ser.frame_count == count
                for i, frame in enumerate(frames):
                    assert np.array_equal(ser.frame(i), frame)
                read = ser.timestamps_utc_ns()
                if with_timestamps and count > 0:
                    assert read is not None
                    assert read.tolist() == stamps
                else:
                    assert read is None


class TestByteOrder:
    def test_an_inverted_flag_needs_the_override(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", depth=16, byte_order="big")
        patch(recording.path, ENDIAN_OFFSET, struct.pack("<i", 1))  # a writer that lies
        with SerFile(recording.path) as honest:
            assert not np.array_equal(honest.frame(2), recording.frames[2])
        with SerFile(recording.path, byte_order="big") as corrected:
            assert corrected.byte_order == "big"
            assert np.array_equal(corrected.frame(2), recording.frames[2])

    def test_the_flag_does_not_matter_for_8_bit_frames(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser")
        patch(recording.path, ENDIAN_OFFSET, struct.pack("<i", 0))
        with SerFile(recording.path) as ser:
            assert np.array_equal(ser.frame(1), recording.frames[1])

    def test_an_unknown_byte_order_is_refused(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser")
        with pytest.raises(ValueError, match="byte_order"):
            SerFile(recording.path, byte_order="middle")  # type: ignore[arg-type]


class TestTimestamps:
    def test_ticks_convert_exactly(self) -> None:
        assert ticks_to_unix_ns(TICKS_AT_UNIX_EPOCH) == 0
        assert ticks_to_unix_ns(TICKS_AT_UNIX_EPOCH + 1) == 100
        assert ticks_to_unix_ns(TICKS_AT_UNIX_EPOCH - 1) == -100
        assert unix_ns_to_ticks(0) == 621_355_968_000_000_000
        assert unix_ns_to_ticks(START_UTC_NS) == 639_028_224_000_000_000  # 2026-01-01T00:00:00Z
        assert ticks_to_unix_ns(639_028_224_000_000_000) == START_UTC_NS

    def test_the_writer_rounds_down_to_a_tick(self) -> None:
        assert unix_ns_to_ticks(199) == TICKS_AT_UNIX_EPOCH + 1
        assert unix_ns_to_ticks(-1) == TICKS_AT_UNIX_EPOCH - 1
        assert ticks_to_unix_ns(unix_ns_to_ticks(1_234_567_891)) == 1_234_567_800

    def test_nanosecond_timestamps_survive_the_file(self, tmp_path: Path) -> None:
        stamps = [START_UTC_NS + 123_456_700, START_UTC_NS + 10_219_600, START_UTC_NS + 2**40 * 100]
        stamps.sort()
        recording = make_ser(tmp_path / "a.ser", count=3, timestamps=stamps)
        with SerFile(recording.path) as ser:
            read = ser.timestamps_utc_ns()
            assert read is not None
            assert read.tolist() == stamps

    def test_the_trailer_converts_from_raw_ticks(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", count=2)
        trailer = HEADER_SIZE + 2 * 128
        patch(
            recording.path,
            trailer,
            struct.pack("<2q", TICKS_AT_UNIX_EPOCH + 5, 639_028_224_000_000_001),
        )
        with SerFile(recording.path) as ser:
            read = ser.timestamps_utc_ns()
            assert read is not None
            assert read.tolist() == [500, START_UTC_NS + 100]

    def test_an_all_zero_trailer_means_no_timestamps(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", count=3)
        patch(recording.path, HEADER_SIZE + 3 * 128, bytes(24))
        with SerFile(recording.path) as ser:
            assert ser.has_trailer
            assert ser.timestamps_utc_ns() is None

    def test_a_trailer_value_outside_the_time_range_is_an_error(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", count=3)
        patch(recording.path, HEADER_SIZE + 3 * 128 + 8, struct.pack("<q", 12345))
        with (
            SerFile(recording.path) as ser,
            pytest.raises(SerFormatError, match="not a valid time"),
        ):
            ser.timestamps_utc_ns()

    def test_the_array_is_read_only(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser")
        with SerFile(recording.path) as ser:
            stamps = ser.timestamps_utc_ns()
            assert stamps is not None
            assert not stamps.flags.writeable
            assert ser.timestamps_utc_ns() is stamps

    def test_the_header_start_time(self, tmp_path: Path) -> None:
        explicit = make_ser(tmp_path / "a.ser", header_start_utc_ns=START_UTC_NS + 42_000_000)
        with SerFile(explicit.path) as ser:
            assert ser.header.start_utc_ns == START_UTC_NS + 42_000_000
        derived = make_ser(tmp_path / "b.ser")
        with SerFile(derived.path) as ser:
            assert ser.header.start_utc_ns == START_UTC_NS  # the first timestamp
        empty = make_ser(tmp_path / "c.ser", timestamps=False)
        with SerFile(empty.path) as ser:
            assert ser.header.start_utc_ns is None
            assert ser.header.datetime_utc_ticks == 0

    def test_the_local_start_time_follows_the_offset(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ser"
        with SerWriter(path, width=2, height=2, start_utc_ns=START_UTC_NS, local_offset_s=10_800):
            pass
        with SerFile(path) as ser:
            local = ticks_to_unix_ns(ser.header.datetime_ticks)
            assert local - START_UTC_NS == 10_800 * 1_000_000_000


class TestInvalidFiles:
    def test_a_truncated_file_is_rejected(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", count=4)
        data = recording.path.read_bytes()
        frames_end = HEADER_SIZE + 4 * 128
        for cut in (frames_end - 1, frames_end - 128, HEADER_SIZE + 1):
            path = tmp_path / f"cut{cut}.ser"
            path.write_bytes(data[:cut])
            with pytest.raises(SerFormatError, match="truncated"):
                SerFile(path)

    def test_a_partial_trailer_is_rejected(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", count=4)
        data = recording.path.read_bytes()
        path = tmp_path / "partial.ser"
        path.write_bytes(data[:-3])
        with pytest.raises(SerFormatError, match="trailer"):
            SerFile(path)

    def test_an_oversized_file_is_rejected(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", count=4)
        with recording.path.open("ab") as handle:
            handle.write(b"\0" * 5)
        with pytest.raises(SerFormatError, match="longer than the header declares"):
            SerFile(recording.path)

    def test_a_file_that_gained_a_frame_is_rejected(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", count=4, timestamps=False)
        with recording.path.open("ab") as handle:
            handle.write(bytes(128))
        with pytest.raises(SerFormatError):
            SerFile(recording.path)

    def test_a_wrong_frame_count_is_rejected(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", count=4)
        patch(recording.path, FRAME_COUNT_OFFSET, struct.pack("<i", 5))
        with pytest.raises(SerFormatError, match="truncated"):
            SerFile(recording.path)

    def test_a_file_shorter_than_the_header_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ser"
        path.write_bytes(b"LUCAM-RECORDER" + bytes(20))
        with pytest.raises(SerFormatError, match="fewer than the 178-byte header"):
            SerFile(path)

    def test_an_empty_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ser"
        path.write_bytes(b"")
        with pytest.raises(SerFormatError):
            SerFile(path)

    def test_the_signature_is_checked(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser")
        patch(recording.path, 0, b"NOT-A-SER-FILE!")
        with pytest.raises(SerFormatError, match="signature"):
            SerFile(recording.path)

    @pytest.mark.parametrize(
        ("offset", "value", "message"),
        [
            (18, 7, "ColorID"),  # not a defined color id
            (26, 0, "image size"),
            (30, -4, "image size"),
            (34, 0, "pixel depth"),
            (34, 17, "pixel depth"),
            (38, -1, "frame count"),
        ],
    )
    def test_invalid_header_values_are_rejected(
        self, tmp_path: Path, offset: int, value: int, message: str
    ) -> None:
        recording = make_ser(tmp_path / "a.ser")
        patch(recording.path, offset, struct.pack("<i", value))
        with pytest.raises(SerFormatError, match=message):
            SerFile(recording.path)

    def test_a_missing_file_is_a_ser_error(self, tmp_path: Path) -> None:
        with pytest.raises(SerError, match="cannot read the file"):
            SerFile(tmp_path / "missing.ser")

    def test_errors_never_name_the_file(self, tmp_path: Path) -> None:
        marker = "distinctive-folder-name"
        folder = tmp_path / marker
        folder.mkdir()
        messages = []
        with pytest.raises(SerError) as missing:
            SerFile(folder / "missing.ser")
        messages.append(missing.value)
        bad = folder / "bad.ser"
        bad.write_bytes(b"x" * 300)
        with pytest.raises(SerError) as invalid:
            SerFile(bad)
        messages.append(invalid.value)
        with pytest.raises(SerError) as exists:
            SerWriter(bad, width=2, height=2)
        messages.append(exists.value)
        for error in messages:
            assert marker not in str(error)
            assert error.__cause__ is None
            assert error.__context__ is None or error.__suppress_context__


class TestPrivacy:
    def test_header_text_is_readable_but_never_printed(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ser"
        with SerWriter(
            path,
            width=4,
            height=2,
            observer="OBSERVER-SECRET",
            instrument="INSTRUMENT-SECRET",
            telescope="TELESCOPE-SECRET",
        ) as writer:
            writer.write_frame(np.zeros((2, 4), dtype=np.uint8), START_UTC_NS)
        with SerFile(path) as ser:
            header = ser.header
            assert (header.observer, header.instrument, header.telescope) == (
                "OBSERVER-SECRET",
                "INSTRUMENT-SECRET",
                "TELESCOPE-SECRET",
            )
            shown = repr(header) + repr(ser) + str(header)
        assert "SECRET" not in shown

    def test_format_errors_do_not_quote_the_header_text(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ser"
        with SerWriter(path, width=4, height=2, observer="OBSERVER-SECRET") as writer:
            writer.write_frame(np.zeros((2, 4), dtype=np.uint8), START_UTC_NS)
        path.write_bytes(path.read_bytes()[:-20])
        with pytest.raises(SerFormatError) as raised:
            SerFile(path)
        assert "SECRET" not in str(raised.value)


class TestAccess:
    def test_indexing_and_iteration(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", count=5)
        with SerFile(recording.path) as ser:
            assert len(ser) == 5
            assert np.array_equal(ser.frame(-1), recording.frames[4])
            assert np.array_equal(ser.frame(-5), recording.frames[0])
            for bad in (5, -6):
                with pytest.raises(IndexError):
                    ser.frame(bad)
            with pytest.raises(TypeError):
                ser.frame(1.5)  # type: ignore[arg-type]
            assert all(np.array_equal(a, b) for a, b in zip(ser, recording.frames, strict=True))
            picked = list(ser.frames(1, 5, 2))
            assert len(picked) == 2
            assert np.array_equal(picked[1], recording.frames[3])
            assert len(list(ser.frames(stop=2))) == 2

    def test_frames_are_read_only_and_outlive_the_file(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", depth=16, byte_order="big")
        with SerFile(recording.path, byte_order="big") as ser:
            frame = ser.frame(1)
        assert not frame.flags.writeable
        assert np.array_equal(frame, recording.frames[1])

    def test_a_closed_file_cannot_be_read(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser")
        ser = SerFile(recording.path)
        states = [ser.closed]
        ser.close()
        ser.close()
        states.append(ser.closed)
        assert states == [False, True]
        with pytest.raises(SerError, match="closed"):
            ser.frame(0)
        with pytest.raises(SerError, match="closed"):
            ser.timestamps_utc_ns()

    def test_a_recording_can_be_deleted_after_closing(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser")
        with SerFile(recording.path) as ser:
            ser.frame(0)
        recording.path.unlink()  # fails on Windows while the file is still mapped
        assert not recording.path.exists()


class TestWriter:
    def test_it_refuses_to_replace_a_file_unless_told_to(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser")
        with pytest.raises(SerError, match="already exists"):
            SerWriter(recording.path, width=2, height=2)
        with SerWriter(recording.path, width=2, height=2, overwrite=True):
            pass
        assert recording.path.stat().st_size == HEADER_SIZE

    def test_it_checks_each_frame(self, tmp_path: Path) -> None:
        with SerWriter(tmp_path / "a.ser", width=4, height=2) as writer:
            good = np.zeros((2, 4), dtype=np.uint8)
            with pytest.raises(ValueError, match="shape"):
                writer.write_frame(np.zeros((4, 2), dtype=np.uint8), START_UTC_NS)
            with pytest.raises(ValueError, match="dtype"):
                writer.write_frame(np.zeros((2, 4), dtype=np.uint16), START_UTC_NS)
            with pytest.raises(ValueError, match="needs t_utc_ns"):
                writer.write_frame(good)
            with pytest.raises(ValueError, match="outside the range"):
                writer.write_frame(good, -(2**63))
            writer.write_frame(good, START_UTC_NS)
            assert writer.frames_written == 1

    def test_without_timestamps_a_timestamp_is_an_error(self, tmp_path: Path) -> None:
        with (
            SerWriter(tmp_path / "a.ser", width=4, height=2, timestamps=False) as writer,
            pytest.raises(ValueError, match="no timestamps"),
        ):
            writer.write_frame(np.zeros((2, 4), dtype=np.uint8), START_UTC_NS)

    def test_a_closed_writer_refuses_frames(self, tmp_path: Path) -> None:
        writer = SerWriter(tmp_path / "a.ser", width=4, height=2, timestamps=False)
        writer.close()
        writer.close()
        with pytest.raises(SerError, match="closed"):
            writer.write_frame(np.zeros((2, 4), dtype=np.uint8))

    def test_the_file_is_valid_after_an_error_inside_the_block(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ser"

        def write_and_fail() -> None:
            with SerWriter(path, width=4, height=2) as writer:
                for i in range(3):
                    writer.write_frame(synthetic_frame(i, 4, 2), START_UTC_NS + i * PERIOD_NS)
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            write_and_fail()
        with SerFile(path) as ser:
            assert ser.frame_count == 3
            assert np.array_equal(ser.frame(2), synthetic_frame(2, 4, 2))

    def test_header_text_must_fit_the_format(self, tmp_path: Path) -> None:
        with pytest.raises(SerFormatError, match="at most 40"):
            SerWriter(tmp_path / "a.ser", width=2, height=2, observer="x" * 41)
        with pytest.raises(SerFormatError, match="Latin-1"):
            SerWriter(tmp_path / "b.ser", width=2, height=2, instrument="\N{SNOWMAN}")
        assert not (tmp_path / "a.ser").exists()

    def test_it_validates_the_geometry(self, tmp_path: Path) -> None:
        with pytest.raises(SerFormatError):
            SerWriter(tmp_path / "a.ser", width=0, height=2)
        with pytest.raises(SerFormatError):
            SerWriter(tmp_path / "b.ser", width=2, height=2, pixel_depth=24)
        with pytest.raises(ValueError, match="byte_order"):
            SerWriter(tmp_path / "c.ser", width=2, height=2, byte_order="middle")  # type: ignore[arg-type]


def test_the_header_round_trips_through_bytes() -> None:
    header = SerHeader(
        width=320,
        height=240,
        pixel_depth=8,
        frame_count=12,
        color_id=ColorId.BAYER_RGGB,
        endian_flag=0,
        lu_id=4660,
        observer="a",
        instrument="b",
        telescope="c",
        datetime_ticks=5,
        datetime_utc_ticks=6,
    )
    raw = header.pack()
    assert len(raw) == HEADER_SIZE
    assert SerHeader.unpack(raw) == header
    assert header.frame_bytes == 76_800
    assert header.file_size(trailer=True) == HEADER_SIZE + 12 * (76_800 + 8)
