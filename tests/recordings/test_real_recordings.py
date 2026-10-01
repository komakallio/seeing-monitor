"""Checks of the owner's real SER files. They skip when the recordings are not configured.

The reader, the command, and the replay driver run on the real files here, without their
sidecars (the sidecar checks are in `test_real_sidecars.py`). Nothing in this module prints a
path, a name, or header text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seeingmon.cli import main
from seeingmon.clock import VirtualClock
from seeingmon.drivers.replay import create
from seeingmon.frames import StreamConfig
from seeingmon.recordings.ser import HEADER_SIZE, TIMESTAMP_SIZE, SerFile
from tests.recordings.real import is_owner_capture, ser_files_or_skip

pytestmark = pytest.mark.recordings


@pytest.fixture(scope="module")
def ser_paths(recordings_dir: Path) -> list[Path]:
    return ser_files_or_skip(recordings_dir)


@pytest.fixture(scope="module")
def captures(ser_paths: list[Path]) -> list[Path]:
    """The files that have the layout of the owner's SharpCap captures."""
    found = []
    for path in ser_paths:
        with SerFile(path) as ser:
            if is_owner_capture(ser):
                found.append(path)
    if not found:
        pytest.skip("no recording has the 8-bit mono 320 x 240 layout")
    return found


def test_every_file_opens_and_its_size_matches_the_header(ser_paths: list[Path]) -> None:
    for path in ser_paths:
        with SerFile(path) as ser:  # the constructor checks the size against the header
            header = ser.header
            exact = ser.file_size == header.file_size(trailer=ser.has_trailer)
            assert exact, "a file size does not match its header"
            assert ser.frame_count > 0, "a recording has no frames"


def test_the_captures_have_the_documented_layout(captures: list[Path]) -> None:
    for path in captures:
        with SerFile(path) as ser:
            frame_bytes = ser.header.frame_bytes
            layout = (
                frame_bytes == 76_800
                and ser.has_trailer
                and ser.file_size == HEADER_SIZE + ser.frame_count * (frame_bytes + TIMESTAMP_SIZE)
            )
            assert layout, "a capture does not have the documented layout"


def test_timestamps_increase_and_imply_the_documented_frame_rate(captures: list[Path]) -> None:
    for path in captures:
        with SerFile(path) as ser:
            stamps = ser.timestamps_utc_ns()
            assert stamps is not None, "a capture has no timestamps"
            increasing = bool(np.all(np.diff(stamps) > 0))
            assert increasing, "the timestamps of a capture do not increase"
            rate = (len(stamps) - 1) / ((int(stamps[-1]) - int(stamps[0])) / 1e9)
            near = 97.0 <= rate <= 98.8
            assert near, "the implied frame rate of a capture is not near 97.9 fps"


def test_frames_read_from_the_start_the_middle_and_the_end(captures: list[Path]) -> None:
    for path in captures:
        with SerFile(path) as ser:
            for index in (0, ser.frame_count // 2, ser.frame_count - 1):
                frame = ser.frame(index)
                ok = frame.shape == (240, 320) and frame.dtype == np.uint8
                assert ok, "a frame has the wrong shape or type"
                assert not frame.flags.writeable


def test_the_info_command_runs_and_names_nothing(
    ser_paths: list[Path], capsys: pytest.CaptureFixture[str]
) -> None:
    for path in ser_paths:
        code = main(["recordings", "info", str(path)])
        shown = capsys.readouterr().out
        assert code == 0
        assert path.name not in shown
        assert path.parent.name not in shown


def test_the_replay_driver_streams_the_start_of_each_capture(captures: list[Path]) -> None:
    for path in captures:
        driver = create(
            profile=None,
            clock=VirtualClock(),
            options={
                "path": path,
                "rate": "max",
                "max_frames": 50,
                "sidecar": False,  # this module reads no sidecar
                "mode": "bin2",
                "exposure_us": 10_000,
                "gain": 100,
            },
        )
        driver.open()
        driver.configure(StreamConfig(mode="bin2", exposure_us=10_000, gain=100))
        driver.start()
        with SerFile(path) as ser:
            stamps = ser.timestamps_utc_ns()
            assert stamps is not None
            for index in range(50):
                frame = driver.read_frame(1.0)
                same_shape = frame.shape == (240, 320)
                same_time = frame.t_arrival_ns == int(stamps[index])
                same_pixels = bool(np.array_equal(frame.data, ser.frame(index)))
                assert same_shape, "a replayed frame has the wrong shape"
                assert same_time, "a replayed frame has the wrong arrival time"
                assert same_pixels, "a replayed frame has the wrong pixels"
                assert frame.seq == index
        driver.close()
