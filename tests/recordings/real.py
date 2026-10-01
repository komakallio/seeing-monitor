"""Helpers for the tests of the owner's real recordings.

These tests print nothing about the files: no path, no name, no header text, and no sidecar
text. They build each failure message from fixed words and from booleans, because pytest shows
the values of a failed comparison.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from seeingmon.recordings.ser import SerFile


def find_ser_files(folder: Path) -> list[Path]:
    """The SER files in the folder, or in its subfolders when the folder itself has none."""
    candidates = sorted(folder.iterdir())
    files = [path for path in candidates if path.is_file() and path.suffix.lower() == ".ser"]
    if files:
        return files
    return sorted(
        path
        for sub in candidates
        if sub.is_dir()
        for path in sub.iterdir()
        if path.is_file() and path.suffix.lower() == ".ser"
    )


def ser_files_or_skip(folder: Path) -> list[Path]:
    files = find_ser_files(folder)
    if not files:
        pytest.skip("the recordings folder holds no SER files")
    return files


def is_owner_capture(ser: SerFile) -> bool:
    """Whether the file has the layout of the owner's SharpCap captures (8-bit mono 320 x 240)."""
    header = ser.header
    return (header.width, header.height, header.pixel_depth) == (320, 240, 8) and (
        header.planes == 1
    )
