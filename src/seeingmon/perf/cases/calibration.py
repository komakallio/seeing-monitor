"""The `calibration` case: fixed workloads that make a later Raspberry Pi 4 run comparable.

The workloads do not depend on the rest of the code base, and they never change, so the ratio of
a Pi 4 run to a dev-machine run on them is a measured scaling factor. `seeingmon perf report
--baseline` prints the ratios. The factor of the interpreter workloads replaces the assumed
`interpreter` range of `seeingmon.perf.scaling`, and the factor of the NumPy workloads replaces the
`numpy` range.

| Workload | What it exercises | Class |
|---|---|---|
| `python_loop` | The interpreter: integer arithmetic and a branch in a loop. | `interpreter` |
| `numpy_matmul` | A BLAS call on one thread: a 384 x 384 float64 matrix product. | `numpy` |
| `numpy_fft` | A NumPy FFT: a real transform of 131,072 samples. | `numpy` |
| `sqlite_insert` | SQLite: 20,000 rows in one transaction, in memory. | `interpreter` |
"""

from __future__ import annotations

from seeingmon.perf.registry import REGISTRY, CaseContext
from seeingmon.perf.report import Measurement
from seeingmon.perf.timing import TimingStats

_MS = 1e3


def python_loop(iterations: int) -> int:
    """Integer arithmetic and a branch in a pure-Python loop. The result checks the work."""
    acc = 0
    for i in range(iterations):
        acc = (acc * 31 + i) & 0xFFFF_FFFF
        if acc & 1:
            acc ^= 0x5BD1E995
    return acc


def _measurement(
    name: str, stats: TimingStats, scale: str, detail: dict[str, float | int | str]
) -> Measurement:
    in_ms = stats.scaled(_MS)
    return Measurement(name, "ms", in_ms.median, stats=in_ms, scale=scale, detail=detail)


@REGISTRY.case("calibration", summary="Fixed workloads that make a Pi 4 run comparable")
def calibration(ctx: CaseContext) -> list[Measurement]:
    import sqlite3

    import numpy as np

    ctx.mark_baseline()
    iterations = ctx.pick(500_000, 5_000)
    matrix = ctx.pick(384, 32)
    samples = ctx.pick(131_072, 1_024)
    rows = ctx.pick(20_000, 500)

    rng = np.random.default_rng(1)
    left = rng.standard_normal((matrix, matrix))
    right = rng.standard_normal((matrix, matrix))
    signal = rng.standard_normal(samples)
    table = [(i, i * 1_000_003, float(i) * 0.5, float(i) * 0.25, f"row {i}") for i in range(rows)]

    def insert_rows() -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, t_ns INTEGER, a REAL, b REAL, label TEXT)"
            )
            with connection:
                connection.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?)", table)
        finally:
            connection.close()

    results = [
        _measurement(
            "python_loop",
            ctx.timer(11, warmup=1).measure(lambda: python_loop(iterations)),
            "interpreter",
            {"iterations": iterations},
        ),
        _measurement(
            "numpy_matmul",
            ctx.timer(41).measure(lambda: left @ right),
            "numpy",
            {"matrix": matrix, "dtype": "float64"},
        ),
        _measurement(
            "numpy_fft",
            ctx.timer(41).measure(lambda: np.fft.rfft(signal)),
            "numpy",
            {"samples": samples},
        ),
        _measurement(
            "sqlite_insert",
            ctx.timer(11, warmup=1).measure(insert_rows),
            "interpreter",
            {"rows": rows},
        ),
    ]
    ctx.note(
        "Each workload runs on one thread. The ratio of a Pi 4 run to a dev-machine run on the "
        "same workload is the measured scaling factor of its class."
    )
    return results
