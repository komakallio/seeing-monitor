"""Checks of the owner's real SharpCap sidecars against their SER files.

These tests skip when the recordings are not configured, and for each capture whose sidecar
file is missing. They assert only that the sidecar and the SER file agree, and they build every
failure message from fixed words and booleans, so no value of a real file reaches the output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.drivers.replay import create
from seeingmon.frames import StreamConfig
from seeingmon.recordings.ser import SerFile
from seeingmon.recordings.sidecar import (
    SidecarInfo,
    is_private_key,
    read_sharpcap_sidecar,
    sharpcap_sidecar_path,
)
from tests.recordings.real import ser_files_or_skip

pytestmark = pytest.mark.recordings


@pytest.fixture(scope="module")
def pairs(recordings_dir: Path) -> list[tuple[Path, SidecarInfo]]:
    """Each SER file that has a sidecar, with the parsed sidecar."""
    found = [
        (path, read_sharpcap_sidecar(sharpcap_sidecar_path(path)))
        for path in ser_files_or_skip(recordings_dir)
        if sharpcap_sidecar_path(path).is_file()
    ]
    if not found:
        pytest.skip("no recording has a SharpCap sidecar")
    return found


def test_the_sidecars_parse_without_problems(pairs: list[tuple[Path, SidecarInfo]]) -> None:
    for _, info in pairs:
        assert info.problems == (), "a sidecar has lines that the parser could not read"
        assert info.entries, "a sidecar parsed to nothing"


def test_the_sidecars_keep_no_serial_number_or_id(pairs: list[tuple[Path, SidecarInfo]]) -> None:
    for _, info in pairs:
        assert info.dropped > 0, "the privacy filter dropped nothing from a sidecar"
        assert not any(is_private_key(key) for key in info.values), "a private key survived"


def test_the_frame_count_matches_the_ser_header(pairs: list[tuple[Path, SidecarInfo]]) -> None:
    for path, info in pairs:
        with SerFile(path) as ser:
            same = info.frame_count is not None and info.frame_count == ser.frame_count
        assert same, "the sidecar frame count differs from the SER header"


def test_the_resolution_matches_the_ser_header(pairs: list[tuple[Path, SidecarInfo]]) -> None:
    for path, info in pairs:
        with SerFile(path) as ser:
            same = info.resolution == (ser.width, ser.height)
        assert same, "the sidecar resolution differs from the SER header"


def test_the_frame_rate_agrees_with_the_count_and_the_duration(
    pairs: list[tuple[Path, SidecarInfo]],
) -> None:
    for _, info in pairs:
        assert info.frame_count is not None
        assert info.duration_s is not None
        assert info.fps is not None
        ratio = info.frame_count / info.duration_s / info.fps
        within = abs(ratio - 1) < 0.01
        assert within, "FrameCount / Duration is not within 1% of ActualFrameRate"


def test_the_capture_times_bracket_the_ser_timestamps(
    pairs: list[tuple[Path, SidecarInfo]],
) -> None:
    for path, info in pairs:
        assert info.start_utc_ns is not None
        assert info.end_utc_ns is not None
        with SerFile(path) as ser:
            stamps = ser.timestamps_utc_ns()
            assert stamps is not None
            starts_before = info.start_utc_ns <= int(stamps[0])
            ends_after = int(stamps[-1]) <= info.end_utc_ns
        assert starts_before, "StartCapture is after the first SER timestamp"
        assert ends_after, "EndCapture is before the last SER timestamp"


def test_the_replay_driver_takes_its_settings_from_the_sidecar(
    pairs: list[tuple[Path, SidecarInfo]],
) -> None:
    for path, info in pairs:
        mode = info.readout_mode or "bin2"
        driver = create(
            profile=None,
            clock=VirtualClock(),
            options={"path": path, "rate": "max", "max_frames": 5},
        )
        driver.open()
        driver.configure(StreamConfig(mode=mode, exposure_us=1, gain=0))
        driver.start()
        frame = driver.read_frame(1.0)
        driver.close()
        agrees = (
            frame.exposure_us == info.exposure_us
            and frame.gain == info.gain
            and frame.temperature_c == info.sensor_temperature_c
            and frame.mode == mode
        )
        assert agrees, "the replayed frame does not carry the settings of the sidecar"
