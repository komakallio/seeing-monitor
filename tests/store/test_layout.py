"""The data layout: names, references, atomic writes, temporary files, and the pin marker."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from seeingmon.config import Config, ConfigError
from seeingmon.store import layout as layout_module
from seeingmon.store.layout import (
    DB_FILENAME,
    PIN_MARKER,
    DataLayout,
    burst_name,
    fsync_directory,
    is_temp_file,
    write_atomic,
)
from tests.store.builders import NS_PER_S, T0


@pytest.fixture
def data(tmp_path: Path) -> DataLayout:
    layout = DataLayout(tmp_path / "data")
    layout.create()
    return layout


def listing(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


class TestFolders:
    def test_create_makes_the_folder_of_every_tier_and_can_repeat(self, tmp_path: Path) -> None:
        layout = DataLayout(tmp_path / "data")
        layout.create()
        layout.create()
        for directory in (
            layout.db_dir,
            layout.segments_dir,
            layout.survey_dir,
            layout.previews_dir,
            layout.bursts_dir,
        ):
            assert directory.is_dir()
            assert directory.parent == layout.root
        assert layout.db_path == layout.db_dir / DB_FILENAME

    def test_the_layout_reads_data_dir_from_the_paths_section(self, tmp_path: Path) -> None:
        config = Config({"paths": {"data_dir": str(tmp_path / "x"), "other_lane_key": 1}})
        assert DataLayout.from_config(config).root == tmp_path / "x"

    def test_a_missing_data_dir_names_the_variable_that_sets_it(self) -> None:
        with pytest.raises(ConfigError, match="SEEINGMON_PATHS__DATA_DIR"):
            DataLayout.from_config(Config({}))


class TestNames:
    def test_a_survey_frame_goes_under_its_date_with_milliseconds(self, data: DataLayout) -> None:
        path = data.survey_path(T0 + 123_000_000)
        assert path == data.survey_dir / "2026" / "01" / "01" / "20260101T000000.123Z.fits"
        assert data.survey_path(T0, suffix=".fits.fz").name == "20260101T000000.000Z.fits.fz"
        assert not path.exists()  # naming creates nothing

    def test_a_preview_has_its_kind_in_the_name(self, data: DataLayout) -> None:
        path = data.preview_path(T0 + 86_400 * NS_PER_S, kind="live")
        assert path == data.previews_dir / "2026" / "01" / "02" / "live-20260102T000000.000Z.jpg"

    def test_a_burst_name_is_the_start_time_and_a_clean_label(self) -> None:
        assert burst_name(T0) == "20260101T000000Z"
        assert burst_name(T0, "Focus sweep #2") == "20260101T000000Z-focus-sweep-2"
        assert burst_name(T0, "///") == "20260101T000000Z"
        assert burst_name(T0, "x" * 80) == "20260101T000000Z-" + "x" * 40
        assert burst_name(T0, "a_b-c") == "20260101T000000Z-a_b-c"

    def test_new_burst_creates_the_directory_and_never_reuses_a_name(
        self, data: DataLayout
    ) -> None:
        first = data.new_burst(T0, "dark")
        second = data.new_burst(T0, "dark")
        third = data.new_burst(T0, "dark")
        assert [p.name for p in (first, second, third)] == [
            "20260101T000000Z-dark",
            "20260101T000000Z-dark-2",
            "20260101T000000Z-dark-3",
        ]
        assert all(p.is_dir() and p.parent == data.bursts_dir for p in (first, second, third))


class TestReferences:
    def test_a_reference_is_relative_with_forward_slashes_and_round_trips(
        self, data: DataLayout
    ) -> None:
        path = data.survey_path(T0)
        reference = data.relative(path)
        assert reference == "survey/2026/01/01/20260101T000000.000Z.fits"
        assert data.resolve(reference) == path.resolve()
        assert data.relative(Path(reference)) == reference  # a relative path is taken as is

    @pytest.mark.parametrize(
        "reference",
        [
            "",
            "../outside.txt",
            "survey/../../outside.txt",
            "/etc/passwd",
            "C:/Windows/system.ini",
            "C:\\Windows\\system.ini",
            "survey\\x.fits",
        ],
    )
    def test_a_reference_cannot_leave_the_data_directory(
        self, data: DataLayout, reference: str
    ) -> None:
        with pytest.raises(ValueError, match="data directory"):
            data.resolve(reference)

    def test_a_symbolic_link_cannot_carry_a_reference_outside(
        self, data: DataLayout, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("x", encoding="utf-8")
        try:
            (data.root / "link").symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("this system does not allow symbolic links")
        with pytest.raises(ValueError, match="leaves the data directory"):
            data.resolve("link/secret.txt")

    def test_relative_rejects_a_path_outside(self, data: DataLayout, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="outside"):
            data.relative(tmp_path / "elsewhere.txt")


class TestAtomicWrites:
    def test_it_writes_bytes_and_creates_the_folders(self, data: DataLayout) -> None:
        target = data.previews_dir / "a" / "b" / "image.jpg"
        assert write_atomic(target, b"jpeg bytes") == target
        assert target.read_bytes() == b"jpeg bytes"
        assert listing(data.root) == ["previews/a/b/image.jpg"]  # no temporary file remains

    def test_it_writes_through_a_callable_for_a_large_file(self, data: DataLayout) -> None:
        def produce(handle: BinaryIO) -> None:
            for _ in range(4):
                handle.write(b"x" * 1000)

        target = data.survey_dir / "frame.fits"
        write_atomic(target, produce)
        assert target.read_bytes() == b"x" * 4000

    def test_it_accepts_a_memoryview_and_a_bytearray(self, data: DataLayout) -> None:
        write_atomic(data.root / "a.bin", memoryview(b"abc"))
        write_atomic(data.root / "b.bin", bytearray(b"def"))
        assert (data.root / "a.bin").read_bytes() == b"abc"
        assert (data.root / "b.bin").read_bytes() == b"def"

    def test_the_target_keeps_its_old_content_until_the_write_completes(
        self, data: DataLayout
    ) -> None:
        target = data.root / "latest.jpg"
        target.write_bytes(b"old")
        seen: list[tuple[bytes, list[str]]] = []

        def produce(handle: BinaryIO) -> None:
            handle.write(b"new, part one ")
            handle.flush()
            seen.append((target.read_bytes(), listing(data.root)))
            handle.write(b"and part two")

        write_atomic(target, produce)
        assert seen[0][0] == b"old"  # a reader sees the old file in the middle of the write
        assert any(name.startswith(".latest.jpg.") for name in seen[0][1])  # a hidden temp file
        assert target.read_bytes() == b"new, part one and part two"

    def test_a_failed_write_leaves_the_target_and_no_temporary_file(self, data: DataLayout) -> None:
        target = data.root / "latest.jpg"
        target.write_bytes(b"old")

        def explode(handle: BinaryIO) -> None:
            handle.write(b"partial")
            raise RuntimeError("the encoder failed")

        with pytest.raises(RuntimeError, match="encoder"):
            write_atomic(target, explode)
        assert target.read_bytes() == b"old"
        assert listing(data.root) == ["latest.jpg"]

    def test_a_failed_rename_leaves_the_target_and_no_temporary_file(
        self, data: DataLayout, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = data.root / "latest.jpg"
        target.write_bytes(b"old")

        def refuse(source: object, destination: object) -> None:
            raise OSError("the disk is gone")

        monkeypatch.setattr(os, "replace", refuse)
        with pytest.raises(OSError, match="disk is gone"):
            write_atomic(target, b"new")
        monkeypatch.undo()
        assert target.read_bytes() == b"old"
        assert listing(data.root) == ["latest.jpg"]

    def test_the_directory_is_forced_after_the_rename(
        self, data: DataLayout, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order: list[str] = []
        real_replace = os.replace

        def record_replace(source: Any, destination: Any) -> None:
            order.append("rename")
            real_replace(source, destination)

        monkeypatch.setattr(os, "replace", record_replace)
        monkeypatch.setattr(
            "seeingmon.store.layout.fsync_directory", lambda directory: order.append("directory")
        )
        write_atomic(data.root / "a.bin", b"x")
        assert order == ["rename", "directory"]

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows cannot open a directory")
    def test_fsync_directory_forces_the_directory_itself(
        self, data: DataLayout, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        forced: list[int] = []
        monkeypatch.setattr(os, "fsync", forced.append)
        fsync_directory(data.root)
        assert len(forced) == 1

    def test_the_layout_method_refuses_a_path_outside_the_data_directory(
        self, data: DataLayout, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="outside"):
            data.write_atomic(tmp_path / "elsewhere.bin", b"x")
        assert not (tmp_path / "elsewhere.bin").exists()
        assert data.write_atomic(data.root / "inside.bin", b"x").read_bytes() == b"x"

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Windows cannot replace a file that a reader holds open"
    )
    def test_a_reader_never_sees_a_mixture_of_two_writes(self, data: DataLayout) -> None:
        target = data.root / "shared.bin"
        write_atomic(target, b"0" * 20_000)
        stop = threading.Event()
        problems: list[object] = []

        def read() -> None:
            try:
                while not stop.is_set():
                    content = target.read_bytes()
                    if len(set(content)) != 1 or len(content) != 20_000:
                        problems.append(content[:20])
            except BaseException as exc:
                problems.append(exc)

        def write(fill: bytes) -> None:
            try:
                for _ in range(40):
                    write_atomic(target, fill * 20_000)
            except BaseException as exc:
                problems.append(exc)

        reader = threading.Thread(target=read)
        reader.start()
        writers = [threading.Thread(target=write, args=(bytes([49 + n]),)) for n in range(3)]
        for thread in writers:
            thread.start()
        for thread in writers:
            thread.join(timeout=120)
        stop.set()
        reader.join(timeout=120)
        assert problems == []
        assert not any(thread.is_alive() for thread in (reader, *writers))

    def test_a_temporary_name_is_hidden_and_unique(self, data: DataLayout) -> None:
        names: list[str] = []

        def spy(handle: BinaryIO) -> None:
            temp = [p.name for p in data.root.iterdir() if is_temp_file(p)]
            assert len(temp) == 1
            names.extend(temp)

        write_atomic(data.root / "a.bin", spy)
        write_atomic(data.root / "a.bin", spy)
        assert len(set(names)) == 2
        assert all(name.startswith(".a.bin.") for name in names)
        assert not is_temp_file(Path("a.bin"))
        assert not is_temp_file(Path(".hidden"))


class TestTemporaryFiles:
    def test_clean_temp_files_deletes_the_stale_leftovers_only(self, data: DataLayout) -> None:
        now = T0
        stale = data.previews_dir / ".live.jpg.123.0.tmp"
        recent = data.survey_dir / ".frame.fits.123.1.tmp"
        keep = data.survey_dir / "frame.fits"
        visible_tmp = data.survey_dir / "notes.tmp"
        for path in (stale, recent, keep, visible_tmp):
            path.write_bytes(b"x")
        age = {stale: 7200, recent: 60, keep: 7200, visible_tmp: 7200}
        for path, seconds in age.items():
            stamp = now - seconds * NS_PER_S
            os.utime(path, ns=(stamp, stamp))
        assert data.clean_temp_files(now, max_age_s=3600) == 1
        assert not stale.exists()
        assert recent.exists()
        assert keep.exists()
        assert visible_tmp.exists()

    def test_clean_temp_files_of_a_missing_directory_does_nothing(self, tmp_path: Path) -> None:
        assert DataLayout(tmp_path / "nothing").clean_temp_files(T0, 1) == 0


class TestPins:
    def test_a_pinned_burst_carries_a_marker_file(self, data: DataLayout) -> None:
        burst = data.new_burst(T0, "keep")
        assert data.is_pinned(burst) is False
        marker = data.pin_burst(burst)
        assert marker == burst.resolve() / PIN_MARKER
        assert data.is_pinned(burst) is True
        assert data.is_pinned(burst.name) is True  # the name works as well as the path
        assert data.unpin_burst(burst) is True
        assert data.is_pinned(burst) is False
        assert data.unpin_burst(burst) is False

    def test_pinning_twice_is_fine(self, data: DataLayout) -> None:
        burst = data.new_burst(T0)
        data.pin_burst(burst.name)
        data.pin_burst(burst.name)
        assert data.is_pinned(burst)

    def test_a_missing_burst_cannot_be_pinned(self, data: DataLayout) -> None:
        with pytest.raises(FileNotFoundError):
            data.pin_burst("20260101T000000Z")

    def test_only_a_burst_directory_can_be_pinned(self, data: DataLayout) -> None:
        nested = data.new_burst(T0) / "inner"
        nested.mkdir()
        with pytest.raises(ValueError, match="directly under"):
            data.pin_burst(nested)
        with pytest.raises(ValueError, match="directly under"):
            data.pin_burst(data.segments_dir)


def test_the_module_exports_what_the_other_lanes_need() -> None:
    for name in ("DataLayout", "write_atomic", "burst_name", "PIN_MARKER", "fsync_directory"):
        assert hasattr(layout_module, name)
