"""Reports with fixed numbers, so that the verdict logic is deterministic."""

from __future__ import annotations

from seeingmon.perf.environment import Environment
from seeingmon.perf.report import CaseResult, Measurement, Report
from seeingmon.perf.timing import TimingStats

MB = 1_000_000


def environment(machine: str = "x86-64") -> Environment:
    return Environment(
        machine=machine,
        os_family="Linux",
        python="3.13.0",
        python_implementation="CPython",
        cpu_count=4,
        cpu_model="Test processor",
        packages={"numpy": "2.0.0"},
        git_commit="0123456789abcdef0123456789abcdef01234567",
        git_dirty=False,
    )


def figure(
    name: str, value: float, unit: str, scale: str = "none", spread: float = 0.0
) -> Measurement:
    """A measurement with statistics that surround `value` by `spread` (a fraction)."""
    stats = TimingStats(
        samples=5,
        min=value * (1 - spread),
        median=value,
        p95=value * (1 + spread),
        max=value * (1 + 2 * spread),
        mean=value,
    )
    return Measurement(name, unit, value, stats=stats, scale=scale)


def case(
    name: str,
    *measurements: Measurement,
    peak_mb: float | None = None,
    baseline_mb: float | None = None,
) -> CaseResult:
    return CaseResult(
        name=name,
        status="ok",
        duration_s=1.5,
        cpu_s=1.4,
        baseline_rss_bytes=None if baseline_mb is None else round(baseline_mb * MB),
        peak_rss_bytes=None if peak_mb is None else round(peak_mb * MB),
        system_busy_percent=3.0,
        measurements=tuple(measurements),
        notes=("a note",),
    )


def shares(mode: str, push: float, close: float, append: float) -> list[Measurement]:
    return [
        figure(f"{mode}.push_share", push, "percent", "numpy"),
        figure(f"{mode}.close_share", close, "percent", "numpy"),
        figure(f"{mode}.append_share", append, "percent", "interpreter"),
    ]


def fixture_report(
    *,
    bin1_push: float = 1.0,
    bin2_push: float = 2.0,
    acquire_compute: float = 0.3,
    acquire_wakeup: float = 0.2,
    rx_compute: float = 0.3,
    rx_wakeup: float = 0.1,
    label: str = "dev",
    machine: str = "x86-64",
    calibration: bool = True,
    survey_peak_mb: float = 100.0,
    survey_seconds: float = 2.0,
) -> Report:
    """A report with fixed numbers.

    With the defaults and the scaling table of this commit, the budgets come out as: `acquire`
    passes (0.5% against 10%), the bin1 fast path passes (about 1.1% against 25%), the bin2 fast
    path is marginal (2.3% on this machine, so 11.7 to 25.3% on a Pi 4), and the memory budgets
    pass. Pass larger numbers to make a budget fail.
    """
    cases = [
        case(
            "ipc",
            figure("acquire.compute_share", acquire_compute, "percent", "interpreter"),
            figure("acquire.wakeup_share", acquire_wakeup, "percent", "scheduler"),
            figure("core_rx.compute_share", rx_compute, "percent", "interpreter"),
            figure("core_rx.wakeup_share", rx_wakeup, "percent", "scheduler"),
            figure("acquire.peak_rss", 60 * MB, "bytes", "memory"),
        ),
        case(
            "fastpath",
            *shares("bin1_128x128_u16", bin1_push, 0.05, 0.03),
            *shares("bin2_64x64_u16", bin2_push, 0.2, 0.1),
            peak_mb=120,
            baseline_mb=100,
        ),
        case("store", figure("results.write", 150.0, "us", "interpreter"), peak_mb=80),
        case(
            "survey",
            figure("frame.total", survey_seconds, "s", "numpy"),
            figure("worker.peak_rss", survey_peak_mb * MB, "bytes", "memory"),
            peak_mb=200,
        ),
        case("memory", figure("baseline.web", 90 * MB, "bytes", "memory")),
        case("kernel", figure("bin1_128x128_u16.kernel", 100.0, "us/frame", "numpy")),
    ]
    if calibration:
        cases.append(case("calibration", figure("python_loop", 100.0, "ms", "interpreter")))
    return Report(
        label=label,
        smoke=False,
        created_utc="2026-10-01T12:00:00Z",
        environment=environment(machine),
        cases=tuple(cases),
    )
