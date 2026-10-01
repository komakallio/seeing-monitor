"""The `fastpath` case: the whole per-frame path, and its share of one core at the frame rate.

The case runs `FastPathAnalyzer` the way `core` does. For every frame it calls `push` (the kernel,
the metrics row, and the window bookkeeping), and once per second of frame time it drains the
metrics rows and appends them to a segment file with `SegmentWriter` (the scheduler drains at that
interval). It times each call, and it runs whole windows of 60 s, so that the cost of closing a
window (the estimator, the scintillation index, and the spectrum) appears too.

For each mode the case reports

- `push`: the time of one frame that does not close a window, in microseconds;
- `close_window`: the extra time of the frame that closes a window, in milliseconds;
- `drain_append`: the drain and the segment append, in microseconds per frame;
- `per_frame`: the mean cost of a frame with everything amortized, in microseconds;
- `push_share`, `close_share`, `append_share`, and `share`: the part of one core that each cost
  takes at the frame rate of the mode, in percent. The budget is 25%.

The shares use the median of each cost, which a busy machine disturbs less than a mean. The
segment append runs on a virtual clock, so its fsync interval counts frame time and not
the speed of this machine. The fsync wait itself is not processor time, and the median leaves it
out. The `store` case measures the append with and without the fsync.
"""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from seeingmon.perf.cases.fastmodes import FAST_MODES, REFERENCE_PROFILE, FastMode, pool_frames
from seeingmon.perf.registry import REGISTRY, CaseContext
from seeingmon.perf.report import Measurement
from seeingmon.perf.timing import TimingStats

if TYPE_CHECKING:
    from seeingmon.fastpath.analyzer import FastPathAnalyzer
    from seeingmon.frames import Frame
    from seeingmon.profile import Profile

_NS_PER_S = 1_000_000_000
_EPOCH_NS = 1_800_000_000 * _NS_PER_S
_STREAM_ID = 1
_WARMUP_WINDOW_S = 2.5


class FrameSource:
    """Builds the frames of one mode, one at a time, outside the timed calls."""

    def __init__(self, profile: Profile, mode: FastMode, smoke: bool) -> None:
        from seeingmon.frames import ActiveStream, PixelFormat, Roi, StreamConfig

        self.mode = mode
        self.adc_bits = mode.adc_bits(profile)
        self.pool = pool_frames(profile, mode, smoke)
        height, width = mode.shape_for(smoke)
        self.period_ns = mode.period_ns
        self.roi = Roi(100, 200, width, height)
        pixel_format = PixelFormat.RAW16 if mode.container_bits == 16 else PixelFormat.RAW8
        self.config = StreamConfig(
            mode.mode, mode.exposure_us, mode.gain, roi=self.roi, pixel_format=pixel_format
        )
        self.stream = ActiveStream(
            _STREAM_ID, self.config, (height, width), self.adc_bits, self.period_ns / _NS_PER_S
        )

    def frame(self, index: int) -> Frame:
        from seeingmon.frames import Frame, FrameFlag, TimeQuality

        t_ns = _EPOCH_NS + index * self.period_ns
        return Frame(
            data=self.pool[index % len(self.pool)],
            stream_id=_STREAM_ID,
            seq=index,
            t_arrival_ns=t_ns,
            t_utc_ns=t_ns,
            t_err_ns=1_000,
            t_quality=TimeQuality.EXACT,
            dropped_before=0,
            exposure_us=self.config.exposure_us,
            gain=self.config.gain,
            mode=self.config.mode,
            roi=self.roi,
            adc_bits=self.adc_bits,
            temperature_c=15.0,
            flags=FrameFlag.SIMULATED,
        )


def new_analyzer(profile: Profile, source: FrameSource, window_s: float) -> FastPathAnalyzer:
    from seeingmon.fastpath.analyzer import FastPathAnalyzer
    from seeingmon.fastpath.config import FastPathConfig

    config = FastPathConfig(window_s=window_s, min_window_s=min(5.0, window_s - 0.1))
    analyzer = FastPathAnalyzer(profile, config)
    analyzer.begin_stream(source.stream)
    return analyzer


def warm_up(profile: Profile, source: FrameSource) -> None:
    """Close one short window, so that the first timed window pays no one-time cost."""
    analyzer = new_analyzer(profile, source, _WARMUP_WINDOW_S)
    frames = round(_WARMUP_WINDOW_S * source.mode.rate_hz) + 2
    for index in range(frames):
        analyzer.push(source.frame(index))
    analyzer.drain_metrics()


def _us(values_ns: list[float]) -> TimingStats:
    return TimingStats.from_samples(values_ns).scaled(1e-3)


def time_mode(ctx: CaseContext, profile: Profile, mode: FastMode) -> list[Measurement]:
    """Time `windows` windows of frames in one mode, and turn the costs into shares of a core."""
    from seeingmon.clock import VirtualClock
    from seeingmon.store.segments import SegmentWriter

    window_s = ctx.pick(60.0, _WARMUP_WINDOW_S)
    windows = ctx.pick(3, 1)
    source = FrameSource(profile, mode, ctx.smoke)
    warm_up(profile, source)
    analyzer = new_analyzer(profile, source, window_s)
    per_window = round(window_s * mode.rate_hz)
    total = per_window * windows + 2  # the next frame closes the last window
    drain_every = max(
        1, round(mode.rate_hz)
    )  # once per second of frame time, as the scheduler does
    clock = VirtualClock()
    push_ns: list[float] = []
    close_ns: list[float] = []
    drain_ns: list[float] = []
    timer = time.perf_counter_ns
    with tempfile.TemporaryDirectory(prefix="smon-perf-", ignore_cleanup_errors=True) as folder:
        writer = SegmentWriter(Path(folder), clock, station_id="perf", profile_id=profile.id)
        for index in range(total):
            frame = source.frame(index)
            start = timer()
            update = analyzer.push(frame)
            elapsed = timer() - start
            (close_ns if update.windows else push_ns).append(elapsed)
            if (index + 1) % drain_every == 0:
                start = timer()
                rows = analyzer.drain_metrics()
                if rows is not None:
                    writer.write_metrics(_STREAM_ID, rows)
                drain_ns.append(timer() - start)
                clock.advance(1.0)
        analyzer.flush()
        writer.close()
    if not push_ns or not close_ns or not drain_ns:
        raise RuntimeError("the run closed no window or drained no rows")

    rate = mode.rate_hz
    push = _us(push_ns)
    push_median_ns = statistics.median(push_ns)
    extra = TimingStats.from_samples(
        [max((value - push_median_ns) / 1e6, 1e-6) for value in close_ns]
    )
    drain = _us([value / drain_every for value in drain_ns])
    frames = len(push_ns) + len(close_ns)
    mean_us = (sum(push_ns) + sum(close_ns) + sum(drain_ns)) / frames / 1e3
    push_share = push.median * rate / 1e4
    close_share = extra.median / (window_s * 10.0)
    append_share = drain.median * rate / 1e4
    detail: dict[str, float | int | str] = {
        "mode": mode.summary,
        "rate_hz": rate,
        "window_s": window_s,
        "windows": len(close_ns),
        "frames": frames,
    }
    return [
        Measurement(f"{mode.key}.push", "us/frame", push.median, push, "numpy", dict(detail)),
        Measurement(f"{mode.key}.close_window", "ms", extra.median, extra, "numpy", dict(detail)),
        Measurement(
            f"{mode.key}.drain_append", "us/frame", drain.median, drain, "interpreter", dict(detail)
        ),
        Measurement(f"{mode.key}.per_frame", "us/frame", mean_us, None, "numpy", dict(detail)),
        Measurement(f"{mode.key}.push_share", "percent", push_share, None, "numpy", dict(detail)),
        Measurement(f"{mode.key}.close_share", "percent", close_share, None, "numpy", dict(detail)),
        Measurement(
            f"{mode.key}.append_share", "percent", append_share, None, "interpreter", dict(detail)
        ),
        Measurement(
            f"{mode.key}.share",
            "percent",
            push_share + close_share + append_share,
            None,
            "numpy",
            dict(detail),
        ),
    ]


@REGISTRY.case(
    "fastpath", summary="The whole per-frame path of the fast path, as a share of a core"
)
def fastpath(ctx: CaseContext) -> list[Measurement]:
    from seeingmon.profile import load_profile

    profile = load_profile(REFERENCE_PROFILE)
    ctx.mark_baseline()
    measurements: list[Measurement] = []
    for mode in FAST_MODES:
        measurements.extend(time_mode(ctx, profile, mode))
    ctx.note(
        "Each mode runs whole windows of frames through FastPathAnalyzer.push, drains the metrics "
        "once per second of frame time, and appends them to a segment file in a folder that the "
        "case deletes. A share is the median cost of one frame, times the frame rate of the mode."
    )
    return measurements
