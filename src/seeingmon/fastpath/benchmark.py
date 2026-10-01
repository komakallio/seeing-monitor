"""Benchmark of the fast path: microseconds per frame, and the cost of closing a window.

    python -m seeingmon.fastpath.benchmark [--frames N] [--repeats R] [--json]

The benchmark times two cases of the reference hardware: a bin1 ROI of 128 x 128 pixels (RAW16
with a 12-bit ADC, the planned fast mode) and a bin2 ROI of 64 x 64 pixels (RAW16 with a 14-bit
ADC). Each case uses a pool of rendered star frames with photon and read noise, so the cache
cannot hide the cost of fresh pixels. The benchmark reports

- `kernel_us`: `measure_frame` alone, following the star from frame to frame;
- `stack_us`: `measure_stack` over the same frames;
- `push_us`: the whole `FastPathAnalyzer.push`, which adds the metrics row, the window
  bookkeeping, and the star state to the kernel;
- `close_ms`: the time that `flush` takes to close one full window of frames (the fits, the
  spectrum, and the corrections). It runs inside the `push` that closes a window.

Each figure is the median over `repeats` of the mean time of a batch of frames, and the report
also gives the best batch. The Raspberry Pi 4 budget is 0.2 to 0.4 ms for the kernel of a bin1
frame, which is 5 to 10 times the figure of a modern desktop.

The function takes a `Clock`, so a test can run it on a clock that it controls. The default is the
system clock.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.fastpath.analyzer import FastPathAnalyzer
from seeingmon.fastpath.config import FastPathConfig
from seeingmon.fastpath.kernel import measure_frame, measure_stack
from seeingmon.frames import (
    ActiveStream,
    Frame,
    FrameFlag,
    PixelFormat,
    Roi,
    StreamConfig,
    TimeQuality,
)
from seeingmon.profile import Profile, load_profile

POOL = 64  # distinct frames that the benchmark cycles through
_REFERENCE_PROFILE = "asi294mm-gs250"
_CASES = (
    ("bin1_128x128_uint16", "bin1", (128, 128), 2000, 0),
    ("bin2_64x64_uint16", "bin2", (64, 64), 2000, 0),
)
_EPOCH_NS = 1_800_000_000 * NS_PER_S

UInt16Frames = npt.NDArray[np.uint16]


@dataclass(frozen=True, slots=True)
class CaseResult:
    """The timings of one case, in microseconds per frame unless the name says otherwise."""

    name: str
    mode: str
    shape: tuple[int, int]
    frames: int
    kernel_us: float
    kernel_best_us: float
    stack_us: float
    push_us: float
    push_best_us: float
    close_ms: float


def star_frames(
    shape: tuple[int, int], e_per_adu: float, read_noise_e: float, adc_bits: int, seed: int = 1
) -> UInt16Frames:
    """A pool of RAW16 frames with a Polaris-like star that wanders by a quarter of a pixel.

    The star has a core of 0.8 pixel sigma and a halo of 3 pixels, with 14,000 electrons (Polaris
    at 2 ms), photon noise, read noise, and a black level of 30 ADU.
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float64)
    frames = np.empty((POOL, height, width), dtype=np.uint16)
    full_scale = (1 << adc_bits) - 1
    for index in range(POOL):
        x = width / 2 + rng.normal(0.0, 0.25)
        y = height / 2 + rng.normal(0.0, 0.25)
        r2 = (columns - x) ** 2 + (rows - y) ** 2
        core = np.exp(-r2 / (2 * 0.8**2)) / (2 * math.pi * 0.8**2)
        halo = np.exp(-r2 / (2 * 3.0**2)) / (2 * math.pi * 3.0**2)
        electrons = 14_000.0 * (0.8 * core + 0.2 * halo)
        noisy = rng.poisson(electrons).astype(np.float64) + rng.normal(0.0, read_noise_e, shape)
        counts = np.clip(np.rint(noisy / e_per_adu + 30.0), 0, full_scale)
        frames[index] = counts.astype(np.uint16) << (16 - adc_bits)
    return frames


def _frame(
    data: UInt16Frames, seq: int, period_ns: int, roi: Roi, config: StreamConfig, adc_bits: int
) -> Frame:
    t = _EPOCH_NS + seq * period_ns
    return Frame(
        data=data,
        stream_id=1,
        seq=seq,
        t_arrival_ns=t,
        t_utc_ns=t,
        t_err_ns=1_000,
        t_quality=TimeQuality.EXACT,
        dropped_before=0,
        exposure_us=config.exposure_us,
        gain=config.gain,
        mode=config.mode,
        roi=roi,
        adc_bits=adc_bits,
        temperature_c=15.0,
        flags=FrameFlag.SIMULATED,
    )


def _microseconds_per_frame(clock: Clock, count: int, work: Callable[[], object]) -> float:
    start = clock.monotonic_ns()
    work()
    return (clock.monotonic_ns() - start) / 1000.0 / count


def run_case(
    profile: Profile,
    name: str,
    mode: str,
    shape: tuple[int, int],
    exposure_us: int,
    gain: int,
    *,
    frames: int,
    repeats: int,
    clock: Clock,
    close_window_s: float = 60.0,
) -> CaseResult:
    """Time one case. `frames` is the number of frames in a batch, and `repeats` the batches.

    The cost of closing a window comes from `close_window_s` seconds of frames.
    """
    adc_bits = profile.mode(mode).adc_bits
    pool = star_frames(
        shape, profile.e_per_adu(mode, gain), profile.read_noise_e(mode, gain), adc_bits
    )
    roi = Roi(100, 200, shape[1], shape[0])
    config = StreamConfig(mode, exposure_us, gain, roi=roi, pixel_format=PixelFormat.RAW16)
    period_ns = round(profile.frame_period_s(mode, shape[0], exposure_us) * NS_PER_S)
    stream = ActiveStream(1, config, shape, adc_bits, period_ns / NS_PER_S)
    arrays = [pool[i % POOL] for i in range(frames)]
    stack = np.stack(arrays)

    def new_analyzer() -> FastPathAnalyzer:
        analyzer = FastPathAnalyzer(profile, FastPathConfig())
        analyzer.begin_stream(stream)
        return analyzer

    params, calibration = new_analyzer().kernel_setup(mode, gain, exposure_us, adc_bits, 16)

    def kernel() -> None:
        guess: tuple[float, float] | None = None
        for data in arrays:
            m = measure_frame(data, roi.x, roi.y, params, calibration, guess)
            guess = (m.x, m.y) if m.found else None

    kernel_runs: list[float] = []
    stack_runs: list[float] = []
    push_runs: list[float] = []
    for _ in range(repeats):
        kernel_runs.append(_microseconds_per_frame(clock, frames, kernel))
        stack_runs.append(
            _microseconds_per_frame(
                clock, frames, lambda: measure_stack(stack, roi.x, roi.y, params, calibration)
            )
        )
        analyzer = new_analyzer()
        batch = [_frame(arrays[i], i, period_ns, roi, config, adc_bits) for i in range(frames)]

        def push(target: FastPathAnalyzer = analyzer, items: list[Frame] = batch) -> None:
            for item in items:
                target.push(item)

        push_runs.append(_microseconds_per_frame(clock, frames, push))
    # Fill a full window and time how long `flush` takes to close it.
    analyzer = new_analyzer()
    for i in range(max(1, round(close_window_s * NS_PER_S / period_ns))):
        analyzer.push(_frame(pool[i % POOL], i, period_ns, roi, config, adc_bits))
    start = clock.monotonic_ns()
    analyzer.flush()
    close_ms = (clock.monotonic_ns() - start) / 1e6
    return CaseResult(
        name=name,
        mode=mode,
        shape=shape,
        frames=frames,
        kernel_us=statistics.median(kernel_runs),
        kernel_best_us=min(kernel_runs),
        stack_us=statistics.median(stack_runs),
        push_us=statistics.median(push_runs),
        push_best_us=min(push_runs),
        close_ms=close_ms,
    )


def run_benchmark(
    *,
    frames: int = 1000,
    repeats: int = 7,
    profile: Profile | None = None,
    clock: Clock | None = None,
    close_window_s: float = 60.0,
) -> list[CaseResult]:
    """Run both cases and return their timings.

    `frames` is the size of a timed batch, and `repeats` the number of batches of each kind.
    `close_window_s` is the length of the window whose closing the benchmark times. The benchmark
    reads the reference profile unless you pass one.
    """
    profile = profile or load_profile(_REFERENCE_PROFILE)
    clock = clock or SystemClock()
    return [
        run_case(
            profile,
            *case,
            frames=frames,
            repeats=repeats,
            clock=clock,
            close_window_s=close_window_s,
        )
        for case in _CASES
    ]


def format_report(results: Sequence[CaseResult]) -> str:
    """The report as text, one line per case."""
    return "\n".join(
        f"{r.name}: kernel {r.kernel_us:.1f} us/frame (best {r.kernel_best_us:.1f}), "
        f"stack {r.stack_us:.1f}, push {r.push_us:.1f} (best {r.push_best_us:.1f}), "
        f"close one window {r.close_ms:.1f} ms"
        for r in results
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and print the report. Returns the exit status."""
    parser = argparse.ArgumentParser(description="Time the fast path in microseconds per frame.")
    parser.add_argument("--frames", type=int, default=1000, help="frames in a timed batch")
    parser.add_argument("--repeats", type=int, default=7, help="batches of each kind")
    parser.add_argument(
        "--close-window-s", type=float, default=60.0, help="window whose closing is timed"
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)
    results = run_benchmark(
        frames=args.frames, repeats=args.repeats, close_window_s=args.close_window_s
    )
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print(format_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
