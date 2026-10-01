"""Scripted fakes of the interfaces, for tests of the code that uses them.

These fakes are small and deterministic. They are not the simulator. The `sim` driver
renders stars and turbulence, and these fakes return flat frames and let a test script faults.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from seeingmon.clock import NS_PER_S, Clock
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
from seeingmon.frames import (
    ActiveStream,
    Frame,
    FrameData,
    FrameFlag,
    PixelFormat,
    Roi,
    StreamConfig,
    StreamKind,
    TimeQuality,
)
from seeingmon.sinks.base import SinkError, StoredRow
from seeingmon.solvers.base import SolveRequest, SolverError, SolveResult

DEFAULT_FULL_FRAMES: Mapping[str, tuple[int, int]] = {"bin1": (8288, 5644), "bin2": (4144, 2822)}

FrameFactory = Callable[[StreamConfig, Roi, int], FrameData]


class FakeCameraDriver:
    """A `CameraDriver` on a `Clock`. With a `VirtualClock`, reading a frame advances time.

    The fake enforces the lifecycle, rounds the ROI to the ROI rules (width to a multiple of 8,
    height to a multiple of 2, as the vendor SDK does), and records every call in `calls`.
    Script failures with `fail_reads`, `drop_frames`, `fail_open`, and `fail_recover`.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        full_frames: Mapping[str, tuple[int, int]] | None = None,
        adc_bits: int = 14,
        overhead_s: float = 0.0065,
        row_time_s: float = 37.6e-6,
        temperature_c: float | None = 18.0,
        frame_factory: FrameFactory | None = None,
    ) -> None:
        self._clock = clock
        self._full_frames = dict(full_frames or DEFAULT_FULL_FRAMES)
        self._adc_bits = adc_bits
        self._overhead_s = overhead_s
        self._row_time_s = row_time_s
        self._temperature_c = temperature_c
        self._frame_factory = frame_factory
        self.calls: list[tuple[str, Any]] = []
        self._opened = False
        self._active: ActiveStream | None = None
        self._roi: Roi | None = None
        self._running = False
        self._stream_id = 0
        self._seq = 0
        self._dropped_counter = 0
        self._pending_drops = 0
        self._read_failures: list[CameraError] = []
        self._open_failure: CameraError | None = None
        self._recover_failure: CameraError | None = None

    # --- scripting ---

    def fail_reads(self, *errors: CameraError) -> None:
        """Make the next reads raise these errors, one per read."""
        self._read_failures.extend(errors)

    def drop_frames(self, count: int) -> None:
        """Lose `count` frames before the next one: the next frame reports them."""
        self._pending_drops += count
        self._dropped_counter += count

    def fail_open(self, error: CameraError | None = None) -> None:
        self._open_failure = error or CameraDisconnectedError("no camera")

    def fail_recover(self, error: CameraError | None = None) -> None:
        self._recover_failure = error or CameraError("recovery did not work")

    # --- CameraDriver ---

    @property
    def name(self) -> str:
        return "fake"

    def open(self) -> CameraInfo:
        self.calls.append(("open", None))
        if self._open_failure is not None:
            raise self._open_failure
        self._opened = True
        width, height = max(self._full_frames.values())
        return CameraInfo(
            model="Fake camera",
            driver=self.name,
            sdk_version=None,
            max_width=width,
            max_height=height,
            has_temperature=self._temperature_c is not None,
        )

    def close(self) -> None:
        self.calls.append(("close", None))
        self._running = False
        self._opened = False

    def capabilities(self) -> CameraCaps:
        return CameraCaps(
            gain_range=(0, 570),
            exposure_us_range=(32, 2_000_000_000),
            bins=(1, 2),
            pixel_formats=(PixelFormat.RAW8, PixelFormat.RAW16),
            offset_range=(0, 255),
        )

    def configure(self, config: StreamConfig) -> ActiveStream:
        self.calls.append(("configure", config))
        if not self._opened:
            raise CameraStateError("configure before open")
        self._running = False
        if config.mode not in self._full_frames:
            raise CameraConfigError(f"unknown readout mode {config.mode!r}")
        width, height = self._full_frames[config.mode]
        request = config.roi or Roi(0, 0, width, height)
        roi_width = max(8, request.width // 8 * 8)
        roi_height = max(2, request.height // 2 * 2)
        if roi_width > width or roi_height > height:
            raise CameraConfigError("ROI is larger than the frame")
        roi = Roi(
            min(request.x, width - roi_width),
            min(request.y, height - roi_height),
            roi_width,
            roi_height,
        )
        self._stream_id += 1
        self._seq = 0
        self._pending_drops = 0
        self._dropped_counter = 0
        self._roi = roi
        applied = replace(config, roi=roi)
        self._active = ActiveStream(
            stream_id=self._stream_id,
            config=applied,
            frame_shape=(roi.height, roi.width),
            adc_bits=self._adc_bits,
            frame_period_s=self._frame_period_s(applied, roi),
        )
        return self._active

    def start(self) -> None:
        self.calls.append(("start", None))
        if self._active is None:
            raise CameraStateError("start before configure")
        self._running = True

    def read_frame(self, timeout_s: float) -> Frame:
        self.calls.append(("read_frame", timeout_s))
        if not self._running or self._active is None or self._roi is None:
            raise CameraStateError("read_frame while not capturing")
        if self._read_failures:
            self._clock.sleep(timeout_s)
            raise self._read_failures.pop(0)
        config = self._active.config
        period_s = self._frame_period_s(config, self._roi)
        if period_s > timeout_s:
            self._clock.sleep(timeout_s)
            raise CameraTimeoutError(f"no frame within {timeout_s} s")
        self._clock.sleep(period_s)
        t_arrival_ns = self._clock.utc_ns()
        dtype = config.pixel_format.dtype
        if self._frame_factory is not None:
            data = self._frame_factory(config, self._roi, self._seq)
        else:
            shift = 16 - self._adc_bits if config.pixel_format is PixelFormat.RAW16 else 0
            data = np.full((self._roi.height, self._roi.width), 100 << shift, dtype=dtype)
        dropped, self._pending_drops = self._pending_drops, 0
        frame = Frame(
            data=data,
            stream_id=self._active.stream_id,
            seq=self._seq,
            t_arrival_ns=t_arrival_ns,
            t_utc_ns=t_arrival_ns - round((period_s - config.exposure_us / 1e6 / 2) * NS_PER_S),
            t_err_ns=1_000,
            t_quality=TimeQuality.EXACT,
            dropped_before=dropped,
            exposure_us=config.exposure_us,
            gain=config.gain,
            mode=config.mode,
            roi=self._roi,
            adc_bits=self._adc_bits,
            temperature_c=self._temperature_c,
            flags=FrameFlag.SIMULATED,
        )
        self._seq += 1
        if config.kind is StreamKind.SNAPSHOT:
            self._running = False
        return frame

    def stop(self) -> None:
        self.calls.append(("stop", None))
        self._running = False

    def move_roi(self, x: int, y: int) -> Roi:
        self.calls.append(("move_roi", (x, y)))
        if self._active is None or self._roi is None:
            raise CameraStateError("move_roi before configure")
        width, height = self._full_frames[self._active.config.mode]
        self._roi = Roi(
            min(max(x, 0), width - self._roi.width),
            min(max(y, 0), height - self._roi.height),
            self._roi.width,
            self._roi.height,
        )
        return self._roi

    def read_temperature_c(self) -> float | None:
        return self._temperature_c

    def dropped_frames(self) -> int:
        return self._dropped_counter

    def recover(self, level: RecoveryLevel) -> None:
        self.calls.append(("recover", level))
        if self._recover_failure is not None:
            raise self._recover_failure
        self._running = False

    def _frame_period_s(self, config: StreamConfig, roi: Roi) -> float:
        readout_s = self._overhead_s + roi.height * self._row_time_s
        exposure_s = config.exposure_us / 1e6
        return (
            exposure_s + readout_s
            if config.kind is StreamKind.SNAPSHOT
            else max(exposure_s, readout_s)
        )


class FakeSink:
    """A `Sink` that keeps what it receives. Script failures with `fail_next`."""

    def __init__(
        self,
        name: str = "fake",
        *,
        record_types: Collection[str] | None = None,
        max_batch_rows: int = 100,
    ) -> None:
        self._name = name
        self._record_types = None if record_types is None else frozenset(record_types)
        self._max_batch_rows = max_batch_rows
        self._failures: list[SinkError] = []
        self.sent: list[tuple[str, tuple[StoredRow, ...]]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_batch_rows(self) -> int:
        return self._max_batch_rows

    def fail_next(self, count: int = 1, *, retryable: bool = True) -> None:
        self._failures.extend(
            SinkError("scripted failure", retryable=retryable) for _ in range(count)
        )

    def accepts(self, record_type: str) -> bool:
        return self._record_types is None or record_type in self._record_types

    def send(self, record_type: str, rows: Sequence[StoredRow]) -> None:
        if not rows or len(rows) > self._max_batch_rows:
            raise ValueError("batch is empty or larger than max_batch_rows")
        if self._failures:
            raise self._failures.pop(0)
        self.sent.append((record_type, tuple(rows)))

    def rows(self, record_type: str) -> list[StoredRow]:
        """Every row received for a record type, in order."""
        return [row for kind, batch in self.sent for row in batch if kind == record_type]


class FakeSolver:
    """A `PlateSolver` that returns a fixed result or raises a fixed error."""

    def __init__(self, result: SolveResult | None = None, error: SolverError | None = None) -> None:
        self._result = result or SolveResult(solved=False, solver="fake", elapsed_s=0.0)
        self._error = error
        self.requests: list[SolveRequest] = []

    @property
    def name(self) -> str:
        return "fake"

    def solve(self, request: SolveRequest) -> SolveResult:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._result
