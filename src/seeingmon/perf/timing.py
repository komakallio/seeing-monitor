"""A timer for repeated measurements: a warm-up, repeats, and the min, median, p95, and max.

`Timer.measure` calls a function, groups the calls into batches, and reports the time of one call
for each batch. A batch of one call shows the spread of single calls. A batch of many calls keeps
the clock resolution out of a figure for a function that takes a few microseconds. The timer reads
`time.perf_counter_ns` by default, which ticks every 100 ns or finer on Windows and Linux. A test
passes its own clock, so that the statistics are exact.

The module uses the standard library only, so the report command and the calibration case load
it without NumPy.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

NS_PER_S = 1_000_000_000
_MAX_AUTORANGE = 10**6

NsClock = Callable[[], int]


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """The `q`-th percentile (0 to 100) of an ascending sequence, by linear interpolation.

    The method is the default of NumPy's `percentile`. Raises `ValueError` for an empty sequence.
    """
    if not sorted_values:
        raise ValueError("a percentile needs at least one value")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be between 0 and 100")
    position = (len(sorted_values) - 1) * q / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


@dataclass(frozen=True, slots=True)
class TimingStats:
    """The summary of repeated timings. The unit is the one of whoever made the samples."""

    samples: int
    min: float
    median: float
    p95: float
    max: float
    mean: float

    def __post_init__(self) -> None:
        values = (self.min, self.median, self.p95, self.max, self.mean)
        if self.samples < 1 or not all(math.isfinite(value) for value in values):
            raise ValueError("timing statistics need at least one sample and finite values")

    @classmethod
    def from_samples(cls, samples: Iterable[float]) -> TimingStats:
        """Summarize samples. Raises `ValueError` when there are none."""
        ordered = sorted(float(value) for value in samples)
        if not ordered:
            raise ValueError("timing statistics need at least one sample")
        return cls(
            samples=len(ordered),
            min=ordered[0],
            median=percentile(ordered, 50.0),
            p95=percentile(ordered, 95.0),
            max=ordered[-1],
            mean=math.fsum(ordered) / len(ordered),
        )

    def scaled(self, factor: float) -> TimingStats:
        """The same statistics in another unit: every time is multiplied by `factor`."""
        return TimingStats(
            samples=self.samples,
            min=self.min * factor,
            median=self.median * factor,
            p95=self.p95 * factor,
            max=self.max * factor,
            mean=self.mean * factor,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "samples": self.samples,
            "min": self.min,
            "median": self.median,
            "p95": self.p95,
            "max": self.max,
            "mean": self.mean,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TimingStats:
        """Read statistics from `to_dict`. Raises `ValueError` for a missing or bad value."""
        try:
            return cls(
                samples=int(data["samples"]),
                min=float(data["min"]),
                median=float(data["median"]),
                p95=float(data["p95"]),
                max=float(data["max"]),
                mean=float(data["mean"]),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"the timing statistics are incomplete or malformed ({error})"
            ) from None

    def ordered(self) -> bool:
        """Whether `min <= median <= p95 <= max`, which every real summary satisfies."""
        return self.min <= self.median <= self.p95 <= self.max


@dataclass(frozen=True, slots=True)
class Timer:
    """Repeated timing with a warm-up. The statistics are in seconds per call.

    `warmup` calls run first and are not timed. Then `repeats` batches run, each of `number`
    calls, and each batch yields one sample: its time divided by `number`.
    """

    repeats: int = 30
    number: int = 1
    warmup: int = 3
    clock: NsClock = time.perf_counter_ns

    def __post_init__(self) -> None:
        if self.repeats < 1 or self.number < 1 or self.warmup < 0:
            raise ValueError("repeats and number must be at least 1, and warmup at least 0")

    def measure(self, work: Callable[[], object]) -> TimingStats:
        """Time `work` and return the seconds per call."""
        for _ in range(self.warmup):
            work()
        clock = self.clock
        samples: list[float] = []
        number = self.number
        for _ in range(self.repeats):
            start = clock()
            for _ in range(number):
                work()
            samples.append((clock() - start) / NS_PER_S / number)
        return TimingStats.from_samples(samples)


def autorange(
    work: Callable[[], object], target_s: float = 0.01, *, clock: NsClock = time.perf_counter_ns
) -> int:
    """The number of calls per batch that makes a batch take at least `target_s` seconds.

    The function tries 1, 2, 5, 10, 20, 50, ... calls and returns the first that is long enough,
    like `timeit.Timer.autorange`. It runs `work` as many times as it tries.
    """
    if target_s <= 0:
        raise ValueError("target_s must be positive")
    number = 1
    while number < _MAX_AUTORANGE:
        for count in (number, number * 2, number * 5):
            start = clock()
            for _ in range(count):
                work()
            if (clock() - start) / NS_PER_S >= target_s:
                return count
        number *= 10
    return _MAX_AUTORANGE  # a clock that never advances stops here, and not in a loop
