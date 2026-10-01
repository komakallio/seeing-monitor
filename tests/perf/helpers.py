"""Reports and runs with fixed numbers, so that the verdict logic is deterministic."""

from __future__ import annotations

from typing import Any

from seeingmon.perf.environment import Environment
from seeingmon.perf.report import CaseResult, Measurement, Report
from seeingmon.perf.sysrun import (
    SMOKE_PLAN,
    RunPlan,
    Snapshot,
    SystemRun,
    split_phases,
)
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
    bin2_rx: float = 1.0,
    label: str = "dev",
    machine: str = "x86-64",
    calibration: bool = True,
    survey_peak_mb: float = 100.0,
    survey_seconds: float = 2.0,
) -> Report:
    """A report with fixed numbers.

    With the defaults and the scaling table of this commit, the budgets come out as: `acquire`
    passes (0.5% against 10%), the bin1 fast path passes (about 1.1% against 25%), the bin2 fast
    path is marginal (2.3% on this machine, so 11.7 to 25.3% on a Pi 4), the bin2 fast path with
    its receive is marginal too (3.3%, so 18.7 to 36.3%), and the memory budgets pass. Pass larger
    numbers to make a budget fail.
    """
    cases = [
        case(
            "ipc",
            figure("acquire.compute_share", acquire_compute, "percent", "interpreter"),
            figure("acquire.wakeup_share", acquire_wakeup, "percent", "scheduler"),
            figure("core_rx.compute_share", rx_compute, "percent", "interpreter"),
            figure("core_rx.wakeup_share", rx_wakeup, "percent", "scheduler"),
            figure("bin2.core_rx.share", bin2_rx, "percent", "interpreter"),
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


def core_sim_case(
    *,
    core_peak_mb: float = 140.0,
    worker_peak_mb: float | None = 450.0,
    web_peak_mb: float = 90.0,
    other_peak_mb: float = 8.0,
    fastpath_receive: float | None = 3.0,
) -> CaseResult:
    """The `core-sim` case with the figures that the budgets read. `None` leaves a figure out."""
    figures = [
        figure("core.peak_rss", core_peak_mb * MB, "bytes", "memory"),
        figure("web.peak_rss", web_peak_mb * MB, "bytes", "memory"),
        figure("other.peak_rss", other_peak_mb * MB, "bytes", "memory"),
    ]
    if worker_peak_mb is not None:
        figures.append(figure("survey_worker.peak_rss", worker_peak_mb * MB, "bytes", "memory"))
    if fastpath_receive is not None:
        figures.append(
            figure("core.fastpath_receive_share", fastpath_receive, "percent", "interpreter")
        )
    return case("core-sim", *figures)


def with_case(report: Report, result: CaseResult) -> Report:
    """The report with one more case at the end."""
    return Report(
        report.label,
        report.smoke,
        report.created_utc,
        report.environment,
        (*report.cases, result),
    )


class Timeline:
    """Samples at a fixed interval, whose counters and CPU times add up as the real ones do.

    Each `tick` adds one interval: the frames that arrived, the CPU time (in milliseconds) that each
    role used, and the state of the scheduler at the end of the interval.
    """

    def __init__(self, interval_s: float = 1.0, *, windows: int = 1) -> None:
        self.interval_s = interval_s
        self.samples: list[Snapshot] = []
        self._t = 100.0
        self._frames = 0
        self._cpu = {"acquire": 0, "core": 0, "web": 0}
        self._threads: dict[str, dict[int, int]] = {}
        self.tick(windows=windows)  # the first sample, before any interval

    def tick(
        self,
        *,
        frames: int = 0,
        core_ms: float = 0.0,
        acquire_ms: float = 0.0,
        web_ms: float = 0.0,
        core_threads_ms: dict[int, float] | None = None,
        rss_mb: dict[str, float] | None = None,
        state: str = "auto",
        purpose: str | None = "fast",
        stream_id: int | None = 1,
        windows: int = 1,
        steps: int = 0,
        results: int = 0,
        pending: int = 0,
    ) -> Timeline:
        if self.samples:
            self._t += self.interval_s
        self._frames += frames
        for role, used in (("core", core_ms), ("acquire", acquire_ms), ("web", web_ms)):
            self._cpu[role] += round(used * 1e6)
        for tid, used in (core_threads_ms or {}).items():
            by_thread = self._threads.setdefault("core", {})
            by_thread[tid] = by_thread.get(tid, 0) + round(used * 1e6)
        self.samples.append(
            Snapshot(
                t=self._t,
                state=state,
                purpose=purpose,
                stream_id=stream_id,
                frames=self._frames,
                windows=windows,
                survey_steps=steps,
                survey_results=results,
                survey_pending=pending,
                cpu_ns=dict(self._cpu),
                threads={role: dict(found) for role, found in self._threads.items()},
                rss={role: round(size * MB) for role, size in (rss_mb or {}).items()},
            )
        )
        return self

    def repeat(self, count: int, **kwargs: Any) -> Timeline:
        for _ in range(count):
            self.tick(**kwargs)
        return self


def fabricated_run(
    *,
    plan: RunPlan = SMOKE_PLAN,
    with_worker: bool = True,
    fast_frames: int = 100,
    machine_busy: float | None = 12.5,
) -> SystemRun:
    """A run with round numbers: 4 s of fast at 100 fps and 2 s idle (the first paused sample ends
    the fast phase, and the interval after it is the first idle one).

    In the fast phase `core` uses 50 ms per second (5% of a core) and `acquire` uses 400 ms. When
    idle, `core` uses 10 ms per second (1%). A frame costs `core` 400 us, so the share at 98 fps is
    3.92%.
    """
    clock = Timeline()
    clock.repeat(
        4,
        frames=fast_frames,
        core_ms=50,
        acquire_ms=400,
        web_ms=2,
        rss_mb={"core": 130, "acquire": 120, "web": 80},
    )
    clock.repeat(3, state="paused", purpose=None, stream_id=None, core_ms=10, acquire_ms=20)
    samples = clock.samples
    fast, idle = split_phases(samples, warmup_windows=0, min_fps=1.0)
    peaks = {
        "acquire": 138 * MB,
        "core": 132 * MB,
        "web": 85 * MB,
        "other": 5 * MB,
    }
    if with_worker:
        peaks["survey_worker"] = 460 * MB
    return SystemRun(
        plan=plan,
        fast=fast,
        idle=idle,
        peaks=peaks,
        snapshots=tuple(samples),
        startup_s=11.0,
        sampling_s=samples[-1].t - samples[0].t,
        survey_steps=2 if with_worker else 0,
        survey_results=4 if with_worker else 0,
        stream={"mode": "bin1_128x128_u16", "exposure_us": 2000, "gain": 100},
        worker_cpu_ns=6_000_000_000 if with_worker else 0,
        logical_cpus=8,
        machine_busy_percent=machine_busy,
        own_busy_percent=1.5,
        notes=(),
    )
