"""Synthetic SER recordings for tests. Each frame is a small, deterministic pattern."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seeingmon.frames import FrameData
from seeingmon.recordings.ser import ByteOrder, SerWriter

START_UTC_NS = 1_767_225_600 * 1_000_000_000  # 2026-01-01T00:00:00Z
PERIOD_NS = 10_000_000


def synthetic_frame(index: int, width: int, height: int, depth: int = 8) -> FrameData:
    """A frame whose pixel values depend on the position and the frame index."""
    values = np.arange(width * height, dtype=np.int64).reshape(height, width) * 257 + 1009 * index
    values %= 1 << depth
    if depth <= 8:
        return values.astype(np.uint8)
    return values.astype(np.uint16)


@dataclass(frozen=True)
class Recording:
    """A synthetic recording on disk and the data that went into it."""

    path: Path
    frames: list[FrameData]
    timestamps_ns: list[int] | None


def regular_timestamps(
    count: int, *, start_ns: int = START_UTC_NS, period_ns: int = PERIOD_NS
) -> list[int]:
    return [start_ns + i * period_ns for i in range(count)]


def make_ser(
    path: Path,
    *,
    count: int = 8,
    width: int = 16,
    height: int = 8,
    depth: int = 8,
    byte_order: ByteOrder = "little",
    timestamps: Sequence[int] | bool = True,
    header_start_utc_ns: int | None = None,
) -> Recording:
    """Write a recording of `count` frames. `timestamps` is a list, or a flag for regular ones."""
    if isinstance(timestamps, bool):
        stamps = regular_timestamps(count) if timestamps else None
    else:
        stamps = list(timestamps)
        if len(stamps) != count:
            raise ValueError("one timestamp per frame")
    frames = [synthetic_frame(i, width, height, depth) for i in range(count)]
    with SerWriter(
        path,
        width=width,
        height=height,
        pixel_depth=depth,
        byte_order=byte_order,
        timestamps=stamps is not None,
        start_utc_ns=header_start_utc_ns,
    ) as writer:
        for i, frame in enumerate(frames):
            writer.write_frame(frame, None if stamps is None else stamps[i])
    return Recording(path, frames, stamps)
