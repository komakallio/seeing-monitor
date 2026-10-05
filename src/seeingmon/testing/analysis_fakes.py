"""Scripted fakes of the analysis interfaces, for tests of the scheduler and the services.

`FakeFastAnalyzer` finds the brightest pixel and groups frames into windows by frame time.
It reports no seeing values: the real estimators live in `seeingmon.fastpath`. `FakeFocusSink`
stands in for the consumer of the rapid focus frames.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from seeingmon.analysis.base import NO_STAR, FastContext, FastUpdate, StarState, SurveyOutput
from seeingmon.frames import ActiveStream, Frame
from seeingmon.records import Record, SeeingWindowRecord, SurveyFrameRecord
from seeingmon.records.segments import segment_dtype

_NS_PER_S = 1_000_000_000
_UINT16_MAX = 65_535


@dataclass(slots=True)
class _OpenWindow:
    t_start_ns: int
    t_last_ns: int
    stream_id: int
    mode: str
    exposure_us: int
    gain: int
    n_frames: int = 0
    n_dropped: int = 0
    temperature_sum: float = 0.0
    temperature_count: int = 0


class FakeFastAnalyzer:
    """A `FastAnalyzer` that tracks the brightest pixel and closes a window every `window_s`.

    A star counts as found when the brightest pixel exceeds the median by `min_contrast_dn`.
    A window shorter than `partial_below` of `window_s` closes with the `partial` flag, and a
    window with more than 5% dropped frames carries `degraded`.
    """

    def __init__(
        self,
        *,
        station_id: str = "test",
        profile_id: str = "test",
        window_s: float = 60.0,
        partial_below: float = 0.5,
        min_contrast_dn: float = 50.0,
    ) -> None:
        self._station_id = station_id
        self._profile_id = profile_id
        self._window_ns = round(window_s * _NS_PER_S)
        self._partial_ns = round(window_s * partial_below * _NS_PER_S)
        self._min_contrast_dn = min_contrast_dn
        self._context = FastContext()
        self._stream_id: int | None = None
        self._window: _OpenWindow | None = None
        self._rows: list[tuple[Any, ...]] = []
        self.star = NO_STAR
        self.frames_pushed = 0

    def begin_stream(self, stream: ActiveStream) -> tuple[SeeingWindowRecord, ...]:
        closed = self._close_window(partial=True)
        self._stream_id = stream.stream_id
        self.star = NO_STAR
        return closed

    def set_context(self, context: FastContext) -> None:
        self._context = context

    def push(self, frame: Frame) -> FastUpdate:
        closed: tuple[SeeingWindowRecord, ...] = ()
        if frame.stream_id != self._stream_id:
            closed = self._close_window(partial=True)
            self._stream_id = frame.stream_id
        window = self._window
        if window is not None and frame.t_utc_ns - window.t_start_ns >= self._window_ns:
            closed += self._close_window(partial=False)
            window = None
        if window is None:
            window = _OpenWindow(
                t_start_ns=frame.t_utc_ns,
                t_last_ns=frame.t_utc_ns,
                stream_id=frame.stream_id,
                mode=frame.mode,
                exposure_us=frame.exposure_us,
                gain=frame.gain,
            )
            self._window = window
        window.t_last_ns = frame.t_utc_ns
        window.n_frames += 1
        window.n_dropped += frame.dropped_before
        if frame.temperature_c is not None:
            window.temperature_sum += frame.temperature_c
            window.temperature_count += 1
        self.star = self._track(frame)
        self.frames_pushed += 1
        return FastUpdate(star=self.star, windows=closed)

    def flush(self, reason: str = "end") -> tuple[SeeingWindowRecord, ...]:
        return self._close_window(partial=True)

    def drain_metrics(self) -> npt.NDArray[Any] | None:
        if not self._rows:
            return None
        rows = np.zeros(len(self._rows), dtype=segment_dtype("frame"))
        for index, row in enumerate(self._rows):
            rows[index] = row
        self._rows.clear()
        return rows

    def _track(self, frame: Frame) -> StarState:
        data = frame.data
        background = float(np.median(data))
        row, column = np.unravel_index(int(np.argmax(data)), data.shape)
        peak = float(data[row, column])
        found = peak - background >= self._min_contrast_dn
        x = float(frame.roi.x + column)
        y = float(frame.roi.y + row)
        self._rows.append(
            (
                frame.t_utc_ns,
                frame.seq,
                min(frame.t_err_ns // 1000, _UINT16_MAX),
                x if found else np.nan,
                y if found else np.nan,
                np.nan,
                np.nan,
                min(int(peak), _UINT16_MAX),
                np.nan,
                background,
                int(frame.flags) & _UINT16_MAX,
                min(frame.dropped_before, _UINT16_MAX),
            )
        )
        if not found:
            return NO_STAR
        return StarState(
            found=True,
            x_px=x,
            y_px=y,
            peak_fraction=peak / float(np.iinfo(data.dtype).max),
            edge_distance_px=frame.roi.distance_to_edge(x, y),
        )

    def _close_window(self, *, partial: bool) -> tuple[SeeingWindowRecord, ...]:
        window, self._window = self._window, None
        if window is None:
            return ()
        interval_ns = (window.t_last_ns - window.t_start_ns) / max(window.n_frames - 1, 1)
        duration_ns = window.t_last_ns - window.t_start_ns + interval_ns
        duration_s = max(duration_ns / _NS_PER_S, 1e-6)
        expected = window.n_frames + window.n_dropped
        flags = set(self._context.flags)
        if window.n_dropped > 0.05 * expected:
            flags.add("degraded")
        if partial and duration_ns < self._partial_ns:
            flags.add("partial")
        temperature = (
            window.temperature_sum / window.temperature_count if window.temperature_count else None
        )
        record = SeeingWindowRecord(
            station_id=self._station_id,
            t_utc_ns=window.t_start_ns,
            profile_id=self._profile_id,
            provenance={"algo": "fake"},
            duration_s=duration_s,
            stream_id=window.stream_id,
            readout_mode=window.mode,
            exposure_us=window.exposure_us,
            gain=window.gain,
            n_frames=window.n_frames,
            n_dropped=window.n_dropped,
            valid_fraction=window.n_frames / expected,
            frame_rate_hz=window.n_frames / duration_s if window.n_frames > 1 else None,
            heater_duty=self._context.heater_duty,
            zenith_angle_deg=self._context.zenith_angle_deg,
            sensor_temperature_c=temperature,
            flags=sorted(flags),
        )
        return (record,)


class FakeFocusSink:
    """A `FocusSink` that finds the brightest pixel and keeps a log of what the scheduler did.

    A star counts as found when the brightest pixel exceeds the median by `min_contrast_dn`, as in
    `FakeFastAnalyzer`. `frames` holds every frame that arrived, `begun` counts the sessions, and
    `ended` holds the reason of each ended session. Set `failing` to make `push` raise.
    """

    def __init__(self, *, min_contrast_dn: float = 50.0) -> None:
        self._min_contrast_dn = min_contrast_dn
        self.frames: list[Frame] = []
        self.begun = 0
        self.ended: list[str] = []
        self.failing = False
        self.star = NO_STAR

    def begin_session(self) -> None:
        self.begun += 1

    def end_session(self, reason: str) -> None:
        self.ended.append(reason)

    def push(self, frame: Frame) -> StarState | None:
        if self.failing:
            raise RuntimeError("the fake focus sink fails")
        self.frames.append(frame)
        data = frame.data
        background = float(np.median(data))
        row, column = np.unravel_index(int(np.argmax(data)), data.shape)
        peak = float(data[row, column])
        if peak - background < self._min_contrast_dn:
            self.star = NO_STAR
            return self.star
        x = float(frame.roi.x + column)
        y = float(frame.roi.y + row)
        self.star = StarState(
            found=True,
            x_px=x,
            y_px=y,
            peak_fraction=peak / float(np.iinfo(data.dtype).max),
            edge_distance_px=frame.roi.distance_to_edge(x, y),
        )
        return self.star


@dataclass(slots=True)
class _Waiting:
    polls_left: int
    frame: Frame
    solved: bool
    cloud_fraction: float | None


class FakeSurveyAnalyzer:
    """A `SurveyAnalyzer` that returns a `survey_frame` record for each frame it receives.

    Set `solved` and `cloud_fraction` (at any time) to script the outcome of later frames.
    With `polls_until_ready` set to N, the first N `poll` calls after a `submit` return
    nothing, and the next one returns the result.
    """

    def __init__(
        self,
        *,
        station_id: str = "test",
        profile_id: str = "test",
        solved: bool = True,
        cloud_fraction: float | None = 0.0,
        polls_until_ready: int = 0,
    ) -> None:
        self._station_id = station_id
        self._profile_id = profile_id
        self.solved = solved
        self.cloud_fraction = cloud_fraction
        self.polls_until_ready = polls_until_ready
        self.submitted: list[Frame] = []
        self._waiting: list[_Waiting] = []

    def submit(self, frame: Frame) -> None:
        self.submitted.append(frame)
        self._waiting.append(
            _Waiting(self.polls_until_ready, frame, self.solved, self.cloud_fraction)
        )

    def poll(self) -> tuple[SurveyOutput, ...]:
        for item in self._waiting:
            item.polls_left -= 1
        ready: list[SurveyOutput] = []
        while self._waiting and self._waiting[0].polls_left < 0:
            ready.append(self._output(self._waiting.pop(0)))
        return tuple(ready)

    def pending(self) -> int:
        return len(self._waiting)

    def _output(self, item: _Waiting) -> SurveyOutput:
        frame = item.frame
        record = SurveyFrameRecord(
            station_id=self._station_id,
            t_utc_ns=frame.t_utc_ns,
            profile_id=self._profile_id,
            provenance={"algo": "fake"},
            exposure_s=frame.exposure_us / 1e6,
            gain=frame.gain,
            readout_mode=frame.mode,
            sensor_temperature_c=frame.temperature_c,
            n_detected=0,
            n_saturated=0,
            background_dn=float(np.median(frame.data)),
        )
        return SurveyOutput(
            t_utc_ns=frame.t_utc_ns,
            records=(record,),
            solved=item.solved,
            cloud_fraction=item.cloud_fraction,
        )


class FakePointingProvider:
    """A `PointingProvider` with a scripted Polaris position per readout mode.

    The position drifts linearly: `x + vx * (t - t0)`. Call `clear` to simulate a lost
    solution, which makes `polaris_position` return `None`.
    """

    def __init__(self, positions: Mapping[str, tuple[float, float]] | None = None) -> None:
        self._solutions: dict[str, tuple[float, float, int, float, float]] = {
            mode: (x, y, 0, 0.0, 0.0) for mode, (x, y) in (positions or {}).items()
        }

    def set_solution(
        self,
        mode: str,
        x: float,
        y: float,
        *,
        t0_utc_ns: int = 0,
        drift_px_per_s: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self._solutions[mode] = (x, y, t0_utc_ns, *drift_px_per_s)

    def clear(self) -> None:
        self._solutions.clear()

    def polaris_position(self, t_utc_ns: int, mode: str) -> tuple[float, float] | None:
        solution = self._solutions.get(mode)
        if solution is None:
            return None
        x, y, t0_ns, vx, vy = solution
        elapsed_s = (t_utc_ns - t0_ns) / _NS_PER_S
        return (x + vx * elapsed_s, y + vy * elapsed_s)


class ListRecordWriter:
    """A `RecordWriter` and `MetricsWriter` that keeps everything in lists."""

    def __init__(self) -> None:
        self.records: list[Record] = []
        self.metrics: list[tuple[int, npt.NDArray[Any]]] = []

    def write(self, record: Record) -> None:
        self.records.append(record)

    def write_metrics(self, stream_id: int, rows: npt.NDArray[Any]) -> None:
        self.metrics.append((stream_id, rows.copy()))

    def of_type(self, record_type: str) -> list[Record]:
        """Every record written so far with this record type, in order."""
        return [record for record in self.records if record.record_type == record_type]
