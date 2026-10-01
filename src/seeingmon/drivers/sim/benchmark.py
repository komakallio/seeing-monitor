"""How fast the simulator renders frames.

Run `python -m seeingmon.drivers.sim.benchmark` to print the table, or call `run_benchmarks`.
The benchmark reads frames from a `VirtualClock`, so it measures the cost of rendering only. A
real run on a `SystemClock` also waits for each frame to be due, and the simulator keeps up in real
time when it renders faster than the camera would deliver: 88 fps for bin1 with a 128-row ROI, and
360 fps for bin2 with a 64-row ROI at a short exposure. On a development machine (one core of a
desktop processor), bin1 keeps up with about 100 fps for the wave optics and 170 fps for the
Gaussian mixture. Bin2 does not: it renders 160 and 270 fps, so a real-time run delivers fewer
frames than the camera would.

Each case reads frames of Polaris with the default atmosphere (three layers, an outer scale of
20 m) and a synthetic star field. `wall` times use the clock on the wall, and `cpu` times use the
processor time of this process, which other programs on the machine do not disturb.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass

from seeingmon.clock import VirtualClock
from seeingmon.drivers.sim.driver import SimDriver
from seeingmon.drivers.sim.optics import PsfConfig
from seeingmon.drivers.sim.options import SimOptions
from seeingmon.drivers.sim.params import reference_modes
from seeingmon.frames import Roi, StreamConfig


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One thing to time: a readout mode, a ROI, an exposure, and a PSF mode."""

    name: str
    mode: str
    roi: Roi
    exposure_us: int
    psf_mode: str


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """The speed of one case."""

    name: str
    frames: int
    wall_s: float
    cpu_s: float

    @property
    def wall_fps(self) -> float:
        return self.frames / self.wall_s

    @property
    def cpu_fps(self) -> float:
        return self.frames / self.cpu_s


# Polaris sits at the centre of the full frame at the reference time.
CASES = (
    BenchmarkCase("bin1 128x128 wave", "bin1", Roi(4080, 2758, 128, 128), 2000, "wave"),
    BenchmarkCase("bin1 128x128 gaussian", "bin1", Roi(4080, 2758, 128, 128), 2000, "gaussian"),
    BenchmarkCase("bin2 64x64 wave", "bin2", Roi(2040, 1378, 64, 64), 1000, "wave"),
    BenchmarkCase("bin2 64x64 gaussian", "bin2", Roi(2040, 1378, 64, 64), 1000, "gaussian"),
)


def _driver_for(case: BenchmarkCase) -> SimDriver:
    options = SimOptions(psf=PsfConfig(mode=case.psf_mode))  # type: ignore[arg-type]
    driver = SimDriver(reference_modes(), VirtualClock(), options)
    driver.open()
    driver.configure(StreamConfig(case.mode, case.exposure_us, 0, roi=case.roi))
    driver.start()
    return driver


def run_case(case: BenchmarkCase, frames: int = 200, warmup: int = 20) -> BenchmarkResult:
    """Time `frames` frames of one case, after `warmup` frames that build the caches."""
    driver = _driver_for(case)
    for _ in range(warmup):
        driver.read_frame(1.0)
    wall = time.perf_counter()
    cpu = time.process_time()
    for _ in range(frames):
        driver.read_frame(1.0)
    return BenchmarkResult(case.name, frames, time.perf_counter() - wall, time.process_time() - cpu)


def run_benchmarks(frames: int = 200, warmup: int = 20) -> list[BenchmarkResult]:
    """Time every case and return the results."""
    return [run_case(case, frames, warmup) for case in CASES]


def format_results(results: Sequence[BenchmarkResult]) -> str:
    """The results as a table of frames per second."""
    lines = [f"{'case':26} {'frames':>7} {'wall fps':>9} {'cpu fps':>9} {'ms/frame':>9}"]
    lines.extend(
        f"{result.name:26} {result.frames:7d} {result.wall_fps:9.1f} {result.cpu_fps:9.1f} "
        f"{1000.0 / result.cpu_fps:9.2f}"
        for result in results
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time the sim driver.")
    parser.add_argument("--frames", type=int, default=200, help="frames to time per case")
    parser.add_argument("--warmup", type=int, default=20, help="frames to skip first")
    args = parser.parse_args(argv)
    print(format_results(run_benchmarks(args.frames, args.warmup)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
