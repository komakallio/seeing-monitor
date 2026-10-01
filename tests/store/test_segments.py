"""Segment files: layout, crash recovery, and reading."""

from __future__ import annotations

import json
import os
import shutil
import struct
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from seeingmon.clock import VirtualClock
from seeingmon.records.segments import from_segment, segment_layout
from seeingmon.store.config import SegmentsConfig
from seeingmon.store.segments import (
    MAGIC,
    SegmentFormatError,
    SegmentReader,
    SegmentWriter,
    iter_segments,
    parse_segment_name,
    read_segment,
    recover_orphans,
    recover_segment,
    segment_name,
    upgrade_rows,
)
from tests.store.builders import FRAME_DTYPE, NS_PER_S, STATION, T0, make_rows

SLOT_NS = 600 * NS_PER_S


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock(T0)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "segments"


@pytest.fixture
def writer(root: Path, clock: VirtualClock) -> Iterator[SegmentWriter]:
    with SegmentWriter(
        root, clock, station_id=STATION, profile_id="profile-1", provenance={"software": "9.9"}
    ) as opened:
        yield opened


def files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def same_rows(left: npt.NDArray[Any], right: npt.NDArray[Any]) -> None:
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    for name in left.dtype.names or ():  # field by field, because NaN must equal NaN
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


class TestWriting:
    def test_25_minutes_of_frames_make_three_segments_under_a_date_directory(
        self, writer: SegmentWriter, root: Path
    ) -> None:
        rows = make_rows(T0 + 300 * NS_PER_S, 1500)  # 00:05:00 to 00:30:00, one row a second
        writer.write_metrics(1, rows)
        writer.close()
        assert files(root) == [
            "2026/01/01/frame_20260101T000000Z_s1.seg",
            "2026/01/01/frame_20260101T001000Z_s1.seg",
            "2026/01/01/frame_20260101T002000Z_s1.seg",
        ]
        pieces = [read_segment(root / name).rows for name in files(root)]
        assert [len(piece) for piece in pieces] == [300, 600, 600]
        same_rows(np.concatenate(pieces), rows)

    def test_the_header_holds_the_identity_the_layout_and_the_times(
        self, writer: SegmentWriter, root: Path, clock: VirtualClock
    ) -> None:
        clock.advance(5)
        writer.write_metrics(7, make_rows(T0 + 90 * NS_PER_S, 10))
        writer.close()
        (name,) = files(root)
        data = read_segment(root / name)
        assert data.closed
        assert data.dropped_bytes == 0
        assert data.record_type == "frame"
        assert data.stream_id == 7
        assert data.start_utc_ns == T0 + 90 * NS_PER_S
        assert data.header["format"] == 1
        assert data.header["slot_start_utc_ns"] == T0
        assert data.header["slot_ns"] == SLOT_NS
        assert data.header["created_utc_ns"] == T0 + 5 * NS_PER_S
        assert data.header["row_bytes"] == 44 == FRAME_DTYPE.itemsize
        assert [tuple(item) for item in data.header["layout"]] == list(segment_layout("frame"))
        assert data.record_header == {
            "station_id": STATION,
            "profile_id": "profile-1",
            "provenance": {"software": "9.9"},
            "revision": 0,
            "quality": None,
            "stream_id": 7,
        }

    def test_the_header_is_json_with_a_magic_and_a_checksum_and_no_pickle(
        self, writer: SegmentWriter, root: Path
    ) -> None:
        writer.write_metrics(1, make_rows(T0, 3))
        writer.close()
        raw = (root / files(root)[0]).read_bytes()
        assert raw.startswith(MAGIC)
        magic, version, flags, length = struct.unpack_from("<8sHHI", raw)
        assert (magic, version, flags) == (MAGIC, 1, 0)
        body = raw[16 : 16 + length]
        assert json.loads(body)["record_type"] == "frame"
        (checksum,) = struct.unpack_from("<I", raw, 16 + length)
        assert checksum == zlib.crc32(body)
        assert len(raw) == 16 + length + 4 + 3 * 44

    def test_the_file_has_the_part_name_until_the_segment_closes(
        self, writer: SegmentWriter, root: Path
    ) -> None:
        writer.write_metrics(1, make_rows(T0, 5))
        assert files(root) == ["2026/01/01/frame_20260101T000000Z_s1.seg.part"]
        held = writer.open_path
        assert held == root / "2026/01/01/frame_20260101T000000Z_s1.seg.part"
        writer.close()
        assert files(root) == ["2026/01/01/frame_20260101T000000Z_s1.seg"]
        assert writer.open_path is None
        writer.close()  # closing again changes nothing

    def test_a_reader_sees_the_complete_rows_of_a_segment_that_is_still_open(
        self, writer: SegmentWriter, root: Path
    ) -> None:
        rows = make_rows(T0, 20)
        writer.write_metrics(1, rows[:12])
        data = read_segment(root / files(root)[0])
        assert not data.closed
        same_rows(data.rows, rows[:12])
        writer.write_metrics(1, rows[12:])
        same_rows(read_segment(root / files(root)[0]).rows, rows)

    def test_a_call_that_spans_a_slot_boundary_is_split_there(
        self, writer: SegmentWriter, root: Path
    ) -> None:
        writer.write_metrics(1, make_rows(T0 + SLOT_NS - 3 * NS_PER_S, 6))  # 3 rows on each side
        writer.close()
        lengths = [len(read_segment(root / name)) for name in files(root)]
        assert lengths == [3, 3]

    def test_rows_that_arrive_in_several_calls_extend_one_segment(
        self, writer: SegmentWriter, root: Path
    ) -> None:
        for n in range(5):
            writer.write_metrics(1, make_rows(T0 + n * 10 * NS_PER_S, 10, seq0=n * 10))
        writer.close()
        assert len(files(root)) == 1
        assert len(read_segment(root / files(root)[0])) == 50

    def test_a_new_stream_starts_a_new_file_and_a_repeated_stream_gets_a_part_number(
        self, writer: SegmentWriter, root: Path
    ) -> None:
        writer.write_metrics(1, make_rows(T0, 5))
        writer.write_metrics(2, make_rows(T0 + 10 * NS_PER_S, 5))
        writer.write_metrics(1, make_rows(T0 + 20 * NS_PER_S, 5))  # the stream ID came back
        writer.close()
        assert files(root) == [
            "2026/01/01/frame_20260101T000000Z_s1.seg",
            "2026/01/01/frame_20260101T000000Z_s1_1.seg",
            "2026/01/01/frame_20260101T000000Z_s2.seg",
        ]

    def test_a_restart_in_the_same_slot_never_overwrites_an_earlier_file(
        self, root: Path, clock: VirtualClock
    ) -> None:
        first = make_rows(T0, 4)
        second = make_rows(T0 + 100 * NS_PER_S, 4)
        for rows in (first, second):
            with SegmentWriter(root, clock, station_id=STATION, profile_id="p") as run:
                run.write_metrics(1, rows)
        names = files(root)
        assert len(names) == 2
        same_rows(read_segment(root / names[0]).rows, first)
        same_rows(read_segment(root / names[1]).rows, second)

    def test_rows_with_the_wrong_layout_or_shape_are_rejected(
        self, writer: SegmentWriter, root: Path
    ) -> None:
        with pytest.raises(ValueError, match="layout"):
            writer.write_metrics(1, np.zeros(3, dtype=[("t_utc_ns", "<i8")]))
        with pytest.raises(ValueError, match="one-dimensional"):
            writer.write_metrics(1, np.zeros((2, 2), dtype=FRAME_DTYPE))
        with pytest.raises(ValueError, match="32 bits"):
            writer.write_metrics(-1, make_rows(T0, 1))
        assert not root.exists() or files(root) == []

    def test_no_rows_make_no_file(self, writer: SegmentWriter, root: Path) -> None:
        writer.write_metrics(1, np.zeros(0, dtype=FRAME_DTYPE))
        assert not root.exists()

    def test_only_a_segment_record_type_can_be_written(
        self, root: Path, clock: VirtualClock
    ) -> None:
        with pytest.raises(ValueError, match="not a segment record"):
            SegmentWriter(root, clock, station_id=STATION, profile_id="p", record_type="health")

    def test_the_segment_length_comes_from_the_configuration(
        self, root: Path, clock: VirtualClock
    ) -> None:
        config = SegmentsConfig(segment_s=60)
        with SegmentWriter(root, clock, station_id=STATION, profile_id="p", config=config) as short:
            short.write_metrics(1, make_rows(T0, 150))  # 150 s at one row a second: 3 slots
        assert len(files(root)) == 3


class TestIdleAndSync:
    def test_tick_closes_a_segment_that_sat_idle(
        self, writer: SegmentWriter, root: Path, clock: VirtualClock
    ) -> None:
        writer.write_metrics(1, make_rows(T0, 5))
        clock.advance(119)
        assert writer.tick() is False
        assert files(root)[0].endswith(".part")
        clock.advance(1)
        assert writer.tick() is True
        assert files(root) == ["2026/01/01/frame_20260101T000000Z_s1.seg"]
        assert writer.tick() is False  # nothing is open

    def test_a_write_restarts_the_idle_time(
        self, writer: SegmentWriter, clock: VirtualClock
    ) -> None:
        writer.write_metrics(1, make_rows(T0, 5))
        clock.advance(100)
        writer.write_metrics(1, make_rows(T0 + 100 * NS_PER_S, 5))
        clock.advance(100)
        assert writer.tick() is False

    def test_the_file_is_forced_to_disk_at_the_interval_and_at_close(
        self, writer: SegmentWriter, clock: VirtualClock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        monkeypatch.setattr(os, "fsync", calls.append)
        # POSIX also forces the directory after the rename, and Windows does not.
        monkeypatch.setattr("seeingmon.store.segments.fsync_directory", lambda directory: None)
        writer.write_metrics(1, make_rows(T0, 5))
        clock.advance(30)
        writer.write_metrics(1, make_rows(T0 + 30 * NS_PER_S, 5))
        assert calls == []  # less than the 60 s interval since the file opened
        clock.advance(40)
        writer.write_metrics(1, make_rows(T0 + 70 * NS_PER_S, 5))
        assert len(calls) == 1
        writer.close()
        assert len(calls) == 2  # the close forces the disk before the rename

    def test_the_directory_is_forced_after_the_rename(
        self, writer: SegmentWriter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rename is durable only after the directory reaches the disk (a no-op on Windows)."""
        order: list[str] = []
        real_replace = os.replace

        def record_replace(source: Any, destination: Any) -> None:
            order.append("rename")
            real_replace(source, destination)

        monkeypatch.setattr(os, "replace", record_replace)
        monkeypatch.setattr(
            "seeingmon.store.segments.fsync_directory", lambda directory: order.append("directory")
        )
        writer.write_metrics(1, make_rows(T0, 3))
        assert order == []  # nothing renames while the segment is open
        writer.close()
        assert order == ["rename", "directory"]

    def test_flush_forces_the_open_segment_to_disk(
        self, writer: SegmentWriter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        monkeypatch.setattr(os, "fsync", calls.append)
        # POSIX also forces the directory after the rename, and Windows does not.
        monkeypatch.setattr("seeingmon.store.segments.fsync_directory", lambda directory: None)
        writer.flush()
        assert calls == []
        writer.write_metrics(1, make_rows(T0, 2))
        writer.flush()
        assert len(calls) == 1


class FailingHandle:
    """Stands in for the open file, and raises on the next write as a full disk does."""

    def __init__(self, real: Any) -> None:
        self.real = real

    def write(self, data: bytes) -> int:
        raise OSError("no space left on the device")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real, name)


class TestWriteFailure:
    def test_a_failed_write_closes_the_segment_at_its_last_complete_row(
        self, writer: SegmentWriter, root: Path
    ) -> None:
        good = make_rows(T0, 5)
        writer.write_metrics(1, good)
        current = writer._current
        assert current is not None
        current.handle.write(b"\x01\x02\x03")  # a torn row that an interrupted write left
        current.handle.flush()
        current.handle = FailingHandle(current.handle)  # type: ignore[assignment]
        with pytest.raises(OSError, match="no space"):
            writer.write_metrics(1, make_rows(T0 + 10 * NS_PER_S, 5))
        assert writer.open_path is None
        (name,) = files(root)
        assert not name.endswith(".part")
        same_rows(read_segment(root / name).rows, good)  # the torn bytes are gone
        writer.write_metrics(1, make_rows(T0 + 20 * NS_PER_S, 2))  # the next call opens a new file
        writer.close()
        assert len(files(root)) == 2


class TestCrashRecovery:
    def make_open_segment(self, root: Path, clock: VirtualClock, rows: Any) -> Path:
        """Write rows, flush, and copy the `.part` file to a second root, as a crash leaves it."""
        crashed = root.parent / "crashed"
        with SegmentWriter(root, clock, station_id=STATION, profile_id="p") as run:
            run.write_metrics(1, rows)
            run.flush()
            part = run.open_path
            assert part is not None
            target = crashed / part.relative_to(root)
            target.parent.mkdir(parents=True)
            shutil.copyfile(part, target)
        return crashed

    def part_file(self, root: Path) -> Path:
        (path,) = list(root.rglob("*.seg.part"))
        return path

    def test_a_segment_that_stays_open_is_readable_up_to_its_last_row(
        self, root: Path, clock: VirtualClock
    ) -> None:
        rows = make_rows(T0, 30)
        crashed = self.make_open_segment(root, clock, rows)
        data = read_segment(self.part_file(crashed))
        assert not data.closed
        same_rows(data.rows, rows)

    def test_a_torn_row_at_the_end_is_dropped(self, root: Path, clock: VirtualClock) -> None:
        rows = make_rows(T0, 30)
        crashed = self.make_open_segment(root, clock, rows)
        part = self.part_file(crashed)
        size = part.stat().st_size
        with part.open("r+b") as handle:
            handle.truncate(size - 10)  # the last row lost 10 of its 44 bytes
        data = read_segment(part)
        same_rows(data.rows, rows[:29])
        assert data.dropped_bytes == 34

    def test_a_zero_filled_tail_from_a_power_cut_is_dropped(
        self, root: Path, clock: VirtualClock
    ) -> None:
        rows = make_rows(T0, 30)
        crashed = self.make_open_segment(root, clock, rows)
        part = self.part_file(crashed)
        with part.open("ab") as handle:
            handle.write(bytes(100))  # two zero rows and a torn zero row
        data = read_segment(part)
        same_rows(data.rows, rows)
        assert data.dropped_bytes == 100

    def test_recover_orphans_cuts_the_file_and_gives_it_its_final_name(
        self, root: Path, clock: VirtualClock
    ) -> None:
        rows = make_rows(T0, 30)
        crashed = self.make_open_segment(root, clock, rows)
        part = self.part_file(crashed)
        with part.open("r+b") as handle:
            handle.truncate(part.stat().st_size - 10)
        (report,) = recover_orphans(crashed)
        assert (report.action, report.rows, report.dropped_bytes) == ("renamed", 29, 34)
        assert report.result is not None
        assert report.result.name == "frame_20260101T000000Z_s1.seg"
        assert files(crashed) == ["2026/01/01/frame_20260101T000000Z_s1.seg"]
        data = read_segment(report.result)
        assert data.closed
        assert data.dropped_bytes == 0
        same_rows(data.rows, rows[:29])
        assert recover_orphans(crashed) == []  # a second run finds nothing

    def test_a_segment_with_no_complete_row_is_removed(
        self, root: Path, clock: VirtualClock
    ) -> None:
        crashed = self.make_open_segment(root, clock, make_rows(T0, 1))
        part = self.part_file(crashed)
        with part.open("r+b") as handle:
            handle.truncate(part.stat().st_size - 1)
        (report,) = recover_orphans(crashed)
        assert (report.action, report.result) == ("removed", None)
        assert files(crashed) == []

    def test_a_file_with_a_damaged_header_is_kept_under_the_corrupt_suffix(
        self, root: Path, clock: VirtualClock
    ) -> None:
        crashed = self.make_open_segment(root, clock, make_rows(T0, 10))
        part = self.part_file(crashed)
        raw = bytearray(part.read_bytes())
        raw[20] ^= 0xFF  # a byte inside the JSON header
        part.write_bytes(bytes(raw))
        (report,) = recover_orphans(crashed)
        assert report.action == "quarantined"
        assert files(crashed) == ["2026/01/01/frame_20260101T000000Z_s1.seg.part.corrupt"]

    def test_a_file_that_ends_before_its_header_is_removed(self, tmp_path: Path) -> None:
        path = tmp_path / "2026" / "01" / "01" / "frame_20260101T000000Z_s1.seg.part"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"\x89SM")
        report = recover_segment(path)
        assert report.action == "removed"
        assert not path.exists()

    def test_recovery_leaves_the_file_of_a_live_writer_alone(
        self, writer: SegmentWriter, root: Path
    ) -> None:
        writer.write_metrics(1, make_rows(T0, 5))
        assert writer.recover_orphans() == []  # the only .part file is the writer's own
        assert writer.open_path is not None
        assert writer.open_path.exists()

    def test_a_writer_repairs_what_an_earlier_run_left(
        self, root: Path, clock: VirtualClock
    ) -> None:
        crashed = self.make_open_segment(root, clock, make_rows(T0, 8))
        with SegmentWriter(crashed, clock, station_id=STATION, profile_id="p") as next_run:
            (report,) = next_run.recover_orphans()
            assert report.rows == 8
            next_run.write_metrics(1, make_rows(T0 + 100 * NS_PER_S, 4))  # same slot
        assert len(files(crashed)) == 2


class TestFormatErrors:
    def write(self, path: Path, data: bytes) -> Path:
        path.write_bytes(data)
        return path

    def test_a_file_with_another_magic_is_not_a_segment(self, tmp_path: Path) -> None:
        path = self.write(tmp_path / "x.seg", b"NOTASEGMENTFILE_" * 4)
        with pytest.raises(SegmentFormatError, match="not a segment"):
            read_segment(path)

    def test_a_truncated_header_is_an_error(
        self, tmp_path: Path, root: Path, clock: VirtualClock
    ) -> None:
        with SegmentWriter(root, clock, station_id=STATION, profile_id="p") as run:
            run.write_metrics(1, make_rows(T0, 3))
        raw = next(root.rglob("*.seg")).read_bytes()
        path = self.write(tmp_path / "short.seg", raw[:30])
        with pytest.raises(SegmentFormatError, match="ends inside the header"):
            read_segment(path)
        path = self.write(tmp_path / "tiny.seg", raw[:5])
        with pytest.raises(SegmentFormatError, match="before the header"):
            read_segment(path)

    def test_a_damaged_header_fails_its_checksum(
        self, tmp_path: Path, root: Path, clock: VirtualClock
    ) -> None:
        with SegmentWriter(root, clock, station_id=STATION, profile_id="p") as run:
            run.write_metrics(1, make_rows(T0, 3))
        raw = bytearray(next(root.rglob("*.seg")).read_bytes())
        raw[30] ^= 0x01
        with pytest.raises(SegmentFormatError, match="checksum"):
            read_segment(self.write(tmp_path / "bad.seg", bytes(raw)))

    def test_a_newer_format_is_refused(
        self, tmp_path: Path, root: Path, clock: VirtualClock
    ) -> None:
        with SegmentWriter(root, clock, station_id=STATION, profile_id="p") as run:
            run.write_metrics(1, make_rows(T0, 3))
        raw = bytearray(next(root.rglob("*.seg")).read_bytes())
        struct.pack_into("<H", raw, 8, 99)
        with pytest.raises(SegmentFormatError, match="format 99"):
            read_segment(self.write(tmp_path / "new.seg", bytes(raw)))

    def test_a_header_with_a_lying_row_size_is_refused(self, tmp_path: Path) -> None:
        header = {
            "format": 1,
            "record_type": "frame",
            "start_utc_ns": 0,
            "layout": [["t_utc_ns", "<i8"]],
            "row_bytes": 44,
            "header": {},
        }
        body = json.dumps(header).encode()
        raw = (
            struct.pack("<8sHHI", MAGIC, 1, 0, len(body))
            + body
            + struct.pack("<I", zlib.crc32(body))
        )
        with pytest.raises(SegmentFormatError, match="row size"):
            read_segment(self.write(tmp_path / "lie.seg", raw))


class TestNamesAndListing:
    def test_a_name_round_trips(self) -> None:
        name = segment_name("frame", T0 + SLOT_NS, 12, 3)
        assert name == "frame_20260101T001000Z_s12_3.seg"
        info = parse_segment_name(Path(name + ".part"))
        assert info is not None
        assert (info.record_type, info.slot_start_utc_ns, info.stream_id, info.part) == (
            "frame",
            T0 + SLOT_NS,
            12,
            3,
        )
        assert info.is_open

    @pytest.mark.parametrize(
        "name", ["notes.txt", "frame_2026_s1.seg", "frame_20260101T000000Z.seg"]
    )
    def test_a_file_that_is_not_a_segment_has_no_info(self, name: str) -> None:
        assert parse_segment_name(Path(name)) is None

    def build(self, root: Path, clock: VirtualClock) -> None:
        with SegmentWriter(root, clock, station_id=STATION, profile_id="p") as run:
            for day in range(3):
                for slot in (0, 1, 5):
                    start = T0 + day * 86_400 * NS_PER_S + slot * SLOT_NS
                    run.write_metrics(1, make_rows(start, 3))
            run.write_metrics(2, make_rows(T0 + 5 * SLOT_NS + 30 * NS_PER_S, 3, seq0=100))

    def test_iter_segments_lists_the_overlapping_slots_in_time_order(
        self, root: Path, clock: VirtualClock
    ) -> None:
        self.build(root, clock)
        everything = list(iter_segments(root))
        assert len(everything) == 10
        keys = [(i.slot_start_utc_ns, i.stream_id) for i in everything]
        assert keys == sorted(keys)
        window = list(iter_segments(root, T0 + SLOT_NS, T0 + 2 * SLOT_NS))
        assert [i.slot_start_utc_ns for i in window] == [T0 + SLOT_NS]
        edge = list(iter_segments(root, T0 + SLOT_NS - 1, T0 + SLOT_NS + 1))
        assert [i.slot_start_utc_ns for i in edge] == [T0, T0 + SLOT_NS]  # a slot is half open
        later = list(iter_segments(root, T0 + 86_400 * NS_PER_S, None))
        assert len(later) == 6
        assert list(iter_segments(root, T0 + 10 * 86_400 * NS_PER_S, None)) == []

    def test_iter_segments_can_leave_out_the_open_file(
        self, writer: SegmentWriter, root: Path
    ) -> None:
        writer.write_metrics(1, make_rows(T0, 3))
        assert [i.is_open for i in iter_segments(root)] == [True]
        assert list(iter_segments(root, include_open=False)) == []

    def test_iter_segments_of_a_missing_root_is_empty(self, tmp_path: Path) -> None:
        assert list(iter_segments(tmp_path / "nothing")) == []

    def test_iter_segments_ignores_other_files_and_other_record_types(
        self, root: Path, clock: VirtualClock
    ) -> None:
        self.build(root, clock)
        (root / "2026" / "01" / "01" / "notes.txt").write_text("hi", encoding="utf-8")
        (root / "2026" / "01" / "01" / "seeing_window_20260101T000000Z_s1.seg").write_bytes(b"")
        (root / "stray").mkdir()
        assert len(list(iter_segments(root))) == 10

    def test_the_reader_reads_a_range_in_time_order_across_segments(
        self, root: Path, clock: VirtualClock
    ) -> None:
        with SegmentWriter(root, clock, station_id=STATION, profile_id="p") as run:
            rows = make_rows(T0, 3000)  # 50 minutes, one row a second
            run.write_metrics(1, rows)
        reader = SegmentReader(root)
        got = reader.read_range(T0 + 590 * NS_PER_S, T0 + 1210 * NS_PER_S)
        same_rows(got, rows[590:1210])
        assert len(reader.read_range(T0 + 10**6 * NS_PER_S, T0 + 2 * 10**6 * NS_PER_S)) == 0
        empty = reader.read_range(T0 + 10**6 * NS_PER_S, T0 + 2 * 10**6 * NS_PER_S)
        assert empty.dtype == FRAME_DTYPE
        assert [info.slot_start_utc_ns for info in reader.iter_segments(T0, T0 + SLOT_NS)] == [T0]

    def test_the_reader_repairs_the_orphans_of_its_root(
        self, root: Path, clock: VirtualClock
    ) -> None:
        writer = SegmentWriter(root, clock, station_id=STATION, profile_id="p")
        writer.write_metrics(1, make_rows(T0, 5))
        assert writer._current is not None
        part = writer._current.partial
        copy = root.parent / "copy" / part.relative_to(root)
        copy.parent.mkdir(parents=True)
        shutil.copyfile(part, copy)
        writer.close()
        reports = SegmentReader(root.parent / "copy").recover()
        assert [r.action for r in reports] == ["renamed"]


class TestOlderLayouts:
    """A record type gains columns at the end. Old segments must stay readable."""

    def write_old_segment(self, root: Path, clock: VirtualClock) -> npt.NDArray[Any]:
        old = np.dtype(list(segment_layout("frame"))[:5])  # t_utc_ns, seq, t_err_us, cx, cy
        writer = SegmentWriter(root, clock, station_id=STATION, profile_id="p")
        writer._dtype = old
        writer._layout = [list(item) for item in segment_layout("frame")[:5]]
        rows = np.zeros(6, dtype=old)
        rows["t_utc_ns"] = T0 + np.arange(6) * NS_PER_S
        rows["seq"] = np.arange(6)
        rows["t_err_us"] = 3
        rows["cx_px"] = 1.5
        rows["cy_px"] = np.nan
        writer.write_metrics(1, rows)
        writer.close()
        return rows

    def test_an_old_segment_reads_in_its_own_layout(self, root: Path, clock: VirtualClock) -> None:
        old = self.write_old_segment(root, clock)
        data = read_segment(next(root.rglob("*.seg")))
        assert data.rows.dtype.names == old.dtype.names
        same_rows(data.rows, old)

    def test_an_old_segment_upgrades_to_the_current_layout(
        self, root: Path, clock: VirtualClock
    ) -> None:
        self.write_old_segment(root, clock)
        got = SegmentReader(root).read_range(T0, T0 + 10 * NS_PER_S)
        assert got.dtype == FRAME_DTYPE
        assert got["seq"].tolist() == list(range(6))
        assert np.isnan(got["width_x_px"]).all()  # a new float column is NaN
        assert (got["peak_dn"] == 0).all()  # a new integer column is zero

    def test_an_old_segment_builds_records_with_the_current_declaration(
        self, root: Path, clock: VirtualClock
    ) -> None:
        self.write_old_segment(root, clock)
        records = read_segment(next(root.rglob("*.seg"))).records()
        assert len(records) == 6
        assert records[0].station_id == STATION
        assert records[0].cx_px == 1.5  # type: ignore[attr-defined]
        assert records[0].cy_px is None  # type: ignore[attr-defined]  # NaN reads as None

    def test_upgrade_rows_returns_the_same_rows_for_the_same_layout(self) -> None:
        rows = make_rows(T0, 3)
        assert upgrade_rows(rows, FRAME_DTYPE) is rows


class TestRecords:
    def test_a_segment_builds_the_same_records_as_the_records_helper(
        self, writer: SegmentWriter, root: Path
    ) -> None:
        rows = make_rows(T0, 8)
        rows["cy_px"][2] = np.nan
        writer.write_metrics(4, rows)
        writer.close()
        data = read_segment(next(root.rglob("*.seg")))
        records = data.records()
        assert records == from_segment("frame", data.record_header, rows)
        assert len(records) == 8
        assert records[2].cy_px is None  # type: ignore[attr-defined]
        assert records[0].stream_id == 4  # type: ignore[attr-defined]


@settings(max_examples=25, deadline=None)
@given(
    count=st.integers(1, 40),
    offsets=st.lists(st.integers(0, 3 * 86_400), min_size=1, max_size=40),
    seed=st.integers(0, 2**32 - 1),
)
def test_any_rows_round_trip_through_the_files(
    tmp_path_factory: pytest.TempPathFactory, count: int, offsets: list[int], seed: int
) -> None:
    root = tmp_path_factory.mktemp("roundtrip")
    clock = VirtualClock(T0)
    rows = make_rows(T0, len(offsets), seed=seed)
    rows["t_utc_ns"] = np.sort(np.array(offsets, dtype=np.int64)) * NS_PER_S + T0
    rows["cx_px"][::3] = np.nan
    with SegmentWriter(root, clock, station_id=STATION, profile_id="p") as run:
        for start in range(0, len(rows), max(count, 1)):
            run.write_metrics(1, rows[start : start + count])
    got = SegmentReader(root).read_range(T0 - 1, T0 + 4 * 86_400 * NS_PER_S)
    assert len(got) == len(rows)
    same_rows(got, rows[np.argsort(rows["t_utc_ns"], kind="stable")])
    assert os.listdir(root)  # the date directories exist
    shutil.rmtree(root)
