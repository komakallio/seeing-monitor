"""Text tables for a report: the figures of every case, and the ratio to a baseline report."""

from __future__ import annotations

from collections.abc import Sequence

from seeingmon.perf.report import CaseResult, Measurement, Report
from seeingmon.perf.scaling import PI4_SCALING

MB = 1e6


def format_number(value: float) -> str:
    """A number for a table: thousands separators from 1,000, then one to three decimals."""
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"


def format_size(value: float | None) -> str:
    """A size in bytes as megabytes, or `-` when it is unknown."""
    return "-" if value is None else f"{value / MB:,.0f} MB"


def _table(rows: Sequence[Sequence[str]], *, indent: str = "") -> list[str]:
    """Align columns: the first column left, the others right."""
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = []
    for row in rows:
        cells = [row[0].ljust(widths[0])]
        cells.extend(cell.rjust(widths[column]) for column, cell in enumerate(row[1:], start=1))
        lines.append((indent + "  ".join(cells)).rstrip())
    return lines


def _measurement_row(item: Measurement) -> tuple[str, str, str, str, str, str]:
    unit = item.unit
    if unit == "bytes":
        value = f"{item.value / MB:,.0f}"
        unit = "MB"
        stats = ("", "", "")
    elif item.stats is not None:
        stats = (
            format_number(item.stats.min),
            format_number(item.stats.p95),
            format_number(item.stats.max),
        )
    else:
        stats = ("", "", "")
    if unit != "MB":
        value = format_number(item.value)
    return (item.name, unit, value, *stats)


def format_case(result: CaseResult, *, details: bool = False) -> str:
    """One case: its status line, its figures, and its notes."""
    busy = (
        ""
        if result.system_busy_percent is None
        else f", machine {result.system_busy_percent:.0f}% busy"
    )
    head = f"{result.name}: {result.status}"
    if result.reason:
        head += f" ({result.reason})"
    head += f", {result.duration_s:.1f} s, peak {format_size(result.peak_rss_bytes)}{busy}"
    lines = [head]
    if result.measurements:
        rows = [("figure", "unit", "median", "min", "p95", "max")]
        rows.extend(_measurement_row(item) for item in result.measurements)
        lines.extend(_table(rows, indent="  "))
        if details:
            for item in result.measurements:
                if item.detail:
                    facts = ", ".join(f"{key}={value}" for key, value in item.detail.items())
                    lines.append(f"  {item.name}: {facts}")
    lines.extend(f"  note: {note}" for note in result.notes)
    return "\n".join(lines)


def format_environment(report: Report) -> str:
    """The environment of a report on two lines."""
    env = report.environment
    packages = ", ".join(
        f"{name} {env.packages[name]}"
        for name in ("numpy", "scipy", "pydantic")
        if name in env.packages
    )
    commit = "no git checkout"
    if env.git_commit:
        commit = f"commit {env.git_commit[:10]}" + (" (modified)" if env.git_dirty else "")
    cpu = f", {env.cpu_model}" if env.cpu_model else ""
    return (
        f"{env.machine}, {env.os_family}, {env.python_implementation} {env.python}, "
        f"{env.cpu_count} logical processors{cpu}\n{packages}; {commit}"
    )


def format_report(report: Report, *, details: bool = False) -> str:
    """The whole report as text."""
    label = report.label or "no label"
    title = f"Performance report: {label}, {report.created_utc}"
    if report.smoke:
        title += " (smoke run: the sizes are tiny, so the figures say nothing about speed)"
    parts = [title, format_environment(report), ""]
    parts.extend(format_case(result, details=details) + "\n" for result in report.cases)
    return "\n".join(parts).rstrip() + "\n"


def format_comparison(report: Report, baseline: Report) -> str:
    """The ratio of each figure of `report` to the same figure of `baseline`.

    For a figure of a class with an assumed range (`seeingmon.perf.scaling`), the table marks
    whether the ratio is inside it. Run it with a Pi 4 report and a dev report to check the
    assumed scaling against the measured one.
    """
    rows = [("figure", "baseline", "report", "ratio", "assumed range")]
    for result in report.cases:
        if not result.ok:
            continue
        for item in result.measurements:
            other = baseline.measurement(result.name, item.name)
            if other is None or other.value <= 0 or other.unit != item.unit:
                continue
            ratio = item.value / other.value
            factors = PI4_SCALING[item.scale]
            if item.scale == "none":
                assumed = "-"
            else:
                inside = factors.low <= ratio <= factors.high
                assumed = (
                    f"{factors.low:g} to {factors.high:g} ({'inside' if inside else 'outside'})"
                )
            rows.append(
                (
                    f"{result.name}: {item.name}",
                    format_number(other.value),
                    format_number(item.value),
                    f"{ratio:.2f}",
                    assumed,
                )
            )
    if len(rows) == 1:
        return "The two reports have no figure in common."
    return "\n".join(_table(rows))
