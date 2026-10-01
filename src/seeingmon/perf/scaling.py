"""The assumed scaling from this machine to a Raspberry Pi 4, in one place.

The harness runs on a dev machine, and the budgets of the architecture are for a Pi 4. Until a real
Pi 4 run exists, the report estimates the Pi 4 figure of a measurement by multiplying it by a
range, and it labels the result an estimate. This module holds every range, with the reason for it
and its sources. Edit the table here, and nothing else changes.

A real Pi 4 run replaces the table. The `calibration` case runs the same fixed workloads on both
machines, and `seeingmon perf report --baseline` prints their ratios. Compare the ratios with the
ranges below. A ratio outside its range means that the range is wrong, so correct it here.

**The reference machine.** The ranges assume a dev machine with a recent x86-64 laptop or
desktop core (a Geekbench 6 single-core score of 2,100 to 2,600). A machine that is much faster or
slower than that needs its own ranges.

**Sources.**

- Geekbench Browser, Raspberry Pi 4 Model B results: single-core scores of about 250 at 1.5 GHz and
  about 300 at 1.8 GHz (https://browser.geekbench.com/search?q=raspberry+pi+4).
- Geekbench Browser processor charts: single-core scores of 2,100 to 2,600 for recent x86-64
  laptop cores (https://browser.geekbench.com/processor-benchmarks).
- Arm, Cortex-A72 technical reference manual: two 128-bit Advanced SIMD pipelines. A Pi 4 runs it
  at 1.5 to 1.8 GHz.
- The architecture, `docs/architecture.md`: the Pi 4 estimates that the architecture states, and
  the check below.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScaleRange:
    """The factor from this machine to a Pi 4 for one class of code, as a range."""

    name: str
    low: float
    high: float
    applies_to: str
    basis: str

    def __post_init__(self) -> None:
        if not 0 < self.low <= self.high:
            raise ValueError(f"the range of {self.name!r} must satisfy 0 < low <= high")


PI4_SCALING: Mapping[str, ScaleRange] = {
    "interpreter": ScaleRange(
        "interpreter",
        7.0,
        11.0,
        "Python bytecode, SQLite, and the bookkeeping of `acquire` and the store",
        "The Geekbench 6 single-core ratio of the two processor classes: about 250 to 300 for a "
        "Pi 4, and 2,100 to 2,600 for the reference machine, which is 7 to 11 times. Interpreter "
        "code is integer and branch heavy, which that score weighs.",
    ),
    "numpy": ScaleRange(
        "numpy",
        5.0,
        11.0,
        "NumPy and SciPy code on arrays that fit in the cache: the kernel and the estimators",
        "Assumption. One core has about 5 to 7 times the vector throughput on a modern laptop "
        "than on a Pi 4 (two 128-bit NEON pipelines at 1.5 to 1.8 GHz against two 256-bit AVX2 "
        "pipelines at 4 to 5 GHz), and about 4 to 6 times the memory bandwidth. A NumPy call that "
        "does little work costs the interpreter ratio, which is 7 to 11 times. The range runs "
        "from the vector-bound end to the call-bound end.",
    ),
    "scheduler": ScaleRange(
        "scheduler",
        1.0,
        4.0,
        "Thread wake-ups and system calls: the part of a stream that a sleeping thread costs",
        "Assumption. A wake-up costs the scheduler of the kernel and the memory system, and not "
        "the instruction stream. On a Pi 4 with Linux, a Python thread hand-off takes about as "
        "long as it takes on the reference machine in a virtual machine on a hybrid laptop "
        "core, which has expensive wake-ups, so the range starts at 1. The Pi's slower core and "
        "memory make it up to 4 times longer. The `thread_handoff` workload of the `calibration` "
        "case measures the ratio.",
    ),
    "memory": ScaleRange(
        "memory",
        0.7,
        1.3,
        "The resident size of a process",
        "Assumption. The Python and NumPy objects have the same size on both machines. The "
        "resident size counts the shared libraries that the process touched, and Windows and "
        "Linux load different ones, so the figure moves by tens of percent.",
    ),
    "none": ScaleRange(
        "none",
        1.0,
        1.0,
        "A figure that does not depend on the speed of the processor",
        "Not scaled.",
    ),
}

# The estimate that the architecture states for the kernel on a Pi 4 (docs/architecture.md,
# "Processes, data rates, and storage"), in milliseconds per 128 x 128 frame.
ARCHITECTURE_KERNEL_ESTIMATE_MS = (0.2, 0.4)

# The operating system and its services on a Raspberry Pi OS Lite image, before the services of
# this project start, in megabytes. It is an assumption. The `free` command on the Pi settles it.
OS_MEMORY_MB = (150.0, 300.0)


def scale_range(name: str) -> ScaleRange:
    """The range of a class. Raises `KeyError` for a class that the table does not hold."""
    return PI4_SCALING[name]


def implied_factor(measured_ms: float) -> tuple[float, float]:
    """The factor that the architecture's kernel estimate implies for a measured kernel time.

    The architecture says that the kernel takes 0.2 to 0.4 ms on a Pi 4. Divide those figures by
    the time that this machine needs, and you get the factor from this machine to a Pi 4 that
    the architecture assumed.
    """
    if measured_ms <= 0:
        raise ValueError("the measured time must be positive")
    low, high = ARCHITECTURE_KERNEL_ESTIMATE_MS
    return low / measured_ms, high / measured_ms
