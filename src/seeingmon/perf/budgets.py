"""The budgets of the architecture, and the verdict for each on a report.

The Pi 4 budget of the architecture ("Processes, data rates, and storage") is 10% of one core for
`acquire`, 25% of one core for the fast path, one core in bursts for the survey worker, and about
1.4 GB of memory at peak, of which the survey worker takes 550 MB. More than 1.6 GB at peak means
the 4 GB model. Each budget below sums a few figures of a report and compares the sum with its
limit.

**Two bases.** A report from the dev machine gives an estimate: each figure is multiplied by the
range of its class (`seeingmon.perf.scaling`), and the verdict is `pass` when the whole range is
within the limit, `fail` when the whole range is above it, and `marginal` when the range straddles
the limit. A report with the label `pi4` and a measured `calibration` case gives a measurement:
the figures are compared with the limits directly, and the verdict is `pass` or `fail`.

**Rows.** The fast path has two rows for each mode: the fast path as the brief defines it (the
kernel, the window, and the segment append), and the fast path with the cost of receiving the
frames from `acquire`, which the same consumer in `core` pays. A figure that mixes work and thread
wake-ups, such as the stream between the two processes, is split in the `ipc` case, so that the
work scales with the speed of the processor and the wake-ups with the `scheduler` range. The
second mode, bin2 at 360 fps, has no split: its receive figure scales with the interpreter range.

A budget whose figures are missing, because a case skipped or failed, has the verdict `n/a`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from seeingmon.perf.report import Report
from seeingmon.perf.scaling import OS_MEMORY_MB, implied_factor, scale_range

Verdict = Literal["pass", "marginal", "fail", "n/a"]

PEAK = "@peak_rss_bytes"  # the peak memory of the child that ran the case
BASELINE = "@baseline_rss_bytes"  # its resident size before the work started

FAST_RATE_BIN1_HZ = 98.0
FAST_RATE_BIN2_HZ = 360.0
MB = 1e6

BIN1 = "bin1_128x128_u16"
BIN2 = "bin2_64x64_u16"


@dataclass(frozen=True, slots=True)
class Term:
    """One figure of a budget: a measurement of a case, or a constant range.

    `measurement` is the name of a figure of `case`, or `PEAK` or `BASELINE` for the memory of the
    process that ran the case. `to_budget` converts the figure into the unit of the budget.
    `scale` overrides the class of the figure. A term with `constant` is an assumed range in the
    unit of the budget, such as the operating system's share of the memory.
    """

    label: str
    case: str = ""
    measurement: str = ""
    to_budget: float = 1.0
    scale: str | None = None
    constant: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class Budget:
    """A limit and the figures that add up to the quantity that it limits."""

    key: str
    title: str
    limit: float
    unit: str
    terms: tuple[Term, ...]
    gate: bool = True  # false for a row that the harness derives, which gates nothing


@dataclass(frozen=True, slots=True)
class BudgetVerdict:
    """The verdict on one budget.

    `value` is the sum of the measured figures on the machine of the report. `low` and `high`
    are the estimated range for a Pi 4, or the measured value twice when the report comes from a
    Pi 4. `missing` lists the figures that the report lacks.
    """

    budget: Budget
    verdict: Verdict
    value: float | None
    low: float | None
    high: float | None
    measured: bool
    missing: tuple[str, ...] = ()


RECEIVE_TERMS: dict[str, tuple[Term, ...]] = {
    BIN1: (
        Term("receive, the work", "ipc", "core_rx.compute_share"),
        Term("receive, the wake-ups", "ipc", "core_rx.wakeup_share"),
    ),
    # The bin2 run of the `ipc` case has no burst run to split with, so it gives the two parts
    # together, and the budget scales them with the interpreter range.
    BIN2: (Term("receive, the work and the wake-ups", "ipc", "bin2.core_rx.share"),),
}


def fast_path_budget(
    mode: str, rate_hz: float, title: str, key: str, *, receive: bool = False
) -> Budget:
    """The 25% budget of the fast path for a mode at a frame rate (per frame: 25% of the period).

    With `receive`, the budget also counts the cost of receiving the frames from `acquire`, which
    the `ipc` case measures (`RECEIVE_TERMS`). The fast-path consumer of `core` pays it.
    """
    per_frame_ms = 0.25 / rate_hz * 1e3
    terms = [
        Term("push", "fastpath", f"{mode}.push_share"),
        Term("window close", "fastpath", f"{mode}.close_share"),
        Term("segment append", "fastpath", f"{mode}.append_share"),
    ]
    if receive:
        terms.extend(RECEIVE_TERMS[mode])
    return Budget(
        key=key,
        title=f"{title}, {rate_hz:g} fps ({per_frame_ms:.2f} ms per frame)",
        limit=25.0,
        unit="% of one core",
        terms=tuple(terms),
    )


def build_budgets(report: Report) -> list[Budget]:
    """The budgets for a report. The core process of the memory budget depends on the cases that
    ran: `core-sim` replaces the stand-in of the fast path plus the store when it ran."""
    core_sim = report.case("core-sim")
    core_terms: tuple[Term, ...]
    if core_sim is not None and core_sim.ok:
        core_terms = (Term("core process peak", "core-sim", PEAK, 1 / MB, "memory"),)
    else:
        core_terms = (
            Term("core, fast path peak (stand-in)", "fastpath", PEAK, 1 / MB, "memory"),
            Term("core, store peak (stand-in)", "store", PEAK, 1 / MB, "memory"),
        )
    memory_terms = (
        Term("acquire peak", "ipc", "acquire.peak_rss", 1 / MB, "memory"),
        *core_terms,
        Term("survey worker peak", "survey", "worker.peak_rss", 1 / MB, "memory"),
        Term("web process, imports only (lower bound)", "memory", "baseline.web", 1 / MB, "memory"),
        Term("operating system (assumption)", constant=OS_MEMORY_MB),
    )
    return [
        Budget(
            "acquire-cpu",
            f"acquire, bin1 128 x 128 at {FAST_RATE_BIN1_HZ:g} fps",
            10.0,
            "% of one core",
            (
                Term("acquire, the work", "ipc", "acquire.compute_share"),
                Term("acquire, the wake-ups", "ipc", "acquire.wakeup_share"),
            ),
        ),
        fast_path_budget(BIN1, FAST_RATE_BIN1_HZ, "Fast path, bin1 128 x 128", "fast-bin1"),
        fast_path_budget(
            BIN1,
            FAST_RATE_BIN1_HZ,
            "Fast path and receive, bin1 128 x 128",
            "core-bin1",
            receive=True,
        ),
        fast_path_budget(BIN2, FAST_RATE_BIN2_HZ, "Fast path, bin2 64 x 64", "fast-bin2"),
        fast_path_budget(
            BIN2,
            FAST_RATE_BIN2_HZ,
            "Fast path and receive, bin2 64 x 64",
            "core-bin2",
            receive=True,
        ),
        Budget(
            "survey-time",
            "Survey frame, bin2 (derived: it must end before the next frame, every 180 s)",
            180.0,
            "s",
            (Term("frame through the worker", "survey", "frame.total"),),
            gate=False,
        ),
        Budget(
            "survey-memory",
            "Survey worker, peak memory",
            550.0,
            "MB",
            (Term("survey worker peak", "survey", "worker.peak_rss", 1 / MB, "memory"),),
        ),
        Budget("memory-1.4", "All processes, peak memory (the budget)", 1400.0, "MB", memory_terms),
        Budget(
            "memory-1.6", "All processes, peak memory (the 2 GB gate)", 1600.0, "MB", memory_terms
        ),
    ]


def is_pi4_measurement(report: Report) -> bool:
    """Whether a report is a real Pi 4 run: the label `pi4` and a measured `calibration`."""
    calibration = report.case("calibration")
    return report.label.strip().lower() == "pi4" and calibration is not None and calibration.ok


def _resolve(report: Report, term: Term) -> tuple[float, str] | None:
    """The figure of a term in the unit of the budget, and its class, or `None` when absent."""
    result = report.case(term.case)
    if result is None or not result.ok:
        return None
    if term.measurement in (PEAK, BASELINE):
        raw = result.peak_rss_bytes if term.measurement == PEAK else result.baseline_rss_bytes
        return None if raw is None else (raw * term.to_budget, term.scale or "memory")
    figure = result.measurement(term.measurement)
    if figure is None:
        return None
    return figure.value * term.to_budget, term.scale or figure.scale


def evaluate_budget(report: Report, budget: Budget, *, measured: bool) -> BudgetVerdict:
    """The verdict on one budget. With `measured`, the figures are not scaled."""
    value = 0.0
    low = 0.0
    high = 0.0
    missing: list[str] = []
    for term in budget.terms:
        if term.constant is not None:
            if not measured:  # an assumption, and a real Pi 4 run has measured the memory itself
                low += term.constant[0]
                high += term.constant[1]
            continue
        resolved = _resolve(report, term)
        if resolved is None:
            missing.append(f"{term.case}: {term.measurement}")
            continue
        figure, scale = resolved
        factors = scale_range(scale)
        value += figure
        low += figure if measured else figure * factors.low
        high += figure if measured else figure * factors.high
    if missing:
        return BudgetVerdict(budget, "n/a", None, None, None, measured, tuple(missing))
    return BudgetVerdict(budget, classify(low, high, budget.limit), value, low, high, measured)


def classify(low: float, high: float, limit: float) -> Verdict:
    """`pass` when the whole range is within the limit, `fail` when all of it is above, and
    `marginal` when the range straddles the limit."""
    if high <= limit:
        return "pass"
    if low > limit:
        return "fail"
    return "marginal"


def evaluate(report: Report) -> list[BudgetVerdict]:
    """The verdicts on every budget, measured for a Pi 4 report and estimated for any other."""
    measured = is_pi4_measurement(report)
    return [evaluate_budget(report, budget, measured=measured) for budget in build_budgets(report)]


def kernel_check(report: Report, mode: str = "bin1_128x128_u16") -> list[str]:
    """The lines that compare the architecture's kernel estimate with the measurement.

    The architecture estimates 0.2 to 0.4 ms for the kernel on a 128 x 128 frame on a Pi 4. The
    check divides that estimate by the time that this machine needs, which gives the factor that
    the architecture assumed, and it compares the factor with the scaling table. Returns no lines
    when the report has no kernel figure.
    """
    figure = report.measurement("kernel", f"{mode}.kernel")
    if figure is None or figure.unit != "us/frame":
        return []
    measured_ms = figure.value / 1e3
    low, high = implied_factor(measured_ms)
    table = scale_range("numpy")
    time = f"{figure.value:.0f} us" if figure.value >= 10 else f"{figure.value:.1f} us"
    if report.label.strip().lower() == "pi4":
        return [
            f"Kernel on this machine: {time} per 128 x 128 frame (measured, not scaled). "
            "The architecture estimates 0.2 to 0.4 ms."
        ]
    if high < table.low:
        verdict = "less cautious than the table, because it assumes a faster Pi 4"
    elif low > table.high:
        verdict = "more cautious than the table, because it assumes a slower Pi 4"
    else:
        verdict = "consistent with the table"
    return [
        f"Kernel on this machine: {time} per 128 x 128 frame (median).",
        f"The architecture estimates 0.2 to 0.4 ms on a Pi 4, which implies a factor of "
        f"{low:.1f} to {high:.1f} from this machine.",
        f"The scaling table assumes {table.low:g} to {table.high:g} for NumPy code, so the "
        f"architecture's estimate is {verdict}.",
        f"With the table, the kernel takes an estimated {measured_ms * table.low:.2f} to "
        f"{measured_ms * table.high:.2f} ms on a Pi 4.",
    ]


def format_verdicts(verdicts: Iterable[BudgetVerdict]) -> str:
    """The verdicts as a text table."""
    rows: list[tuple[str, str, str, str, str]] = []
    for item in verdicts:
        budget = item.budget
        limit = f"{budget.limit:g} {budget.unit}"
        if item.value is None or item.low is None or item.high is None:
            cases = sorted({figure.split(":")[0] for figure in item.missing})
            rows.append((budget.title, limit, "n/a", "n/a", f"n/a (needs {', '.join(cases)})"))
            continue
        here = f"{_number(item.value)}"
        if item.measured:
            pi4 = f"{_number(item.value)} (measured)"
        else:
            pi4 = f"{_number(item.low)} to {_number(item.high)} (estimate)"
        verdict = item.verdict if budget.gate else f"{item.verdict} (not a gate)"
        rows.append((budget.title, limit, here, pi4, verdict))
    header = ("Budget", "Limit", "This machine", "Pi 4", "Verdict")
    table = [header, *rows]
    widths = [max(len(row[column]) for row in table) for column in range(len(header))]
    lines = [
        "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(row)) for row in table
    ]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(line.rstrip() for line in lines)


def _number(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"
