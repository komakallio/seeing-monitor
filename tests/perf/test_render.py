"""The text tables of a report."""

from __future__ import annotations

import pytest

from seeingmon.perf.render import (
    format_case,
    format_comparison,
    format_environment,
    format_number,
    format_report,
    format_size,
)
from seeingmon.perf.report import CaseResult, Measurement, Report

from .helpers import case, environment, figure, fixture_report


def report_with(item: Measurement) -> Report:
    return Report("dev", False, "2026-10-01T12:00:00Z", environment(), (case("calibration", item),))


class TestNumbers:
    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (12_345.6, "12,346"),
            (1_000.0, "1,000"),
            (250.4, "250"),
            (42.37, "42.4"),
            (3.14159, "3.14"),
            (0.0123456, "0.0123"),
            (0.0, "0"),
            (-5.5, "-5.50"),
        ],
    )
    def test_a_number_gets_the_decimals_that_its_size_needs(self, value: float, text: str) -> None:
        assert format_number(value) == text

    def test_a_size_in_bytes_is_shown_in_megabytes_and_an_unknown_size_as_a_dash(self) -> None:
        assert format_size(123_456_789) == "123 MB"
        assert format_size(None) == "-"


class TestCase:
    def test_the_status_line_names_the_duration_the_peak_and_the_load(self) -> None:
        result = case("kernel", figure("a", 1.0, "ms"), peak_mb=120)
        first = format_case(result).splitlines()[0]
        assert first == "kernel: ok, 1.5 s, peak 120 MB, machine 3% busy"

    def test_a_skipped_case_shows_its_reason_and_no_table(self) -> None:
        text = format_case(CaseResult("core-sim", "skipped", "the core process is not on main"))
        assert text.splitlines()[0].startswith(
            "core-sim: skipped (the core process is not on main)"
        )
        assert "figure" not in text

    def test_the_table_has_the_median_and_the_spread_of_each_figure(self) -> None:
        result = case("kernel", figure("bin1.kernel", 100.0, "us/frame", spread=0.5))
        row = format_case(result).splitlines()[2]
        assert row.split() == ["bin1.kernel", "us/frame", "100", "50.0", "150", "200"]

    def test_a_memory_figure_is_shown_in_megabytes_without_a_spread(self) -> None:
        result = case("memory", figure("baseline.web", 54_000_000, "bytes"))
        row = format_case(result).splitlines()[2]
        assert row.split() == ["baseline.web", "MB", "54"]

    def test_the_details_and_the_notes_follow_the_table_when_asked(self) -> None:
        item = figure("a", 1.0, "ms")
        item.detail["frames"] = 500
        result = case("x", item)
        plain = format_case(result)
        detailed = format_case(result, details=True)
        assert "frames=500" not in plain
        assert "  a: frames=500" in detailed
        assert "  note: a note" in plain


class TestReport:
    def test_the_title_names_the_label_and_the_time_and_marks_a_smoke_run(self) -> None:
        text = format_report(fixture_report(label="dev"))
        assert text.startswith("Performance report: dev, 2026-10-01T12:00:00Z\n")
        assert "smoke run" not in text
        full = fixture_report(label="")
        assert "no label" in format_report(full)

    def test_the_environment_lists_the_machine_the_versions_and_the_commit(self) -> None:
        text = format_environment(fixture_report())
        assert "x86-64, Linux, CPython 3.13.0, 4 logical processors, Test processor" in text
        assert "numpy 2.0.0" in text
        assert "commit 0123456789" in text

    def test_every_case_appears(self) -> None:
        text = format_report(fixture_report())
        for name in ("ipc", "fastpath", "store", "survey", "memory", "kernel", "calibration"):
            assert f"\n{name}: ok" in text


class TestComparison:
    def test_it_prints_the_ratio_and_whether_it_lies_in_the_assumed_range(self) -> None:
        dev = fixture_report()
        same = fixture_report(label="pi4", machine="arm64")
        # The fixture holds the same figures in both, so every ratio is 1, which is below the
        # range that the scaling table assumes for a Pi 4.
        text = format_comparison(same, dev)
        assert "1.00" in text
        assert "calibration: python_loop" in text
        row = next(line for line in text.splitlines() if "calibration: python_loop" in line)
        assert row.rstrip().endswith("7 to 11 (outside)")

    def test_a_ratio_inside_the_range_is_marked_inside(self) -> None:
        dev = report_with(figure("python_loop", 100.0, "ms", "interpreter"))
        pi = report_with(figure("python_loop", 850.0, "ms", "interpreter"))
        row = format_comparison(pi, dev).splitlines()[1]
        assert row.split()[-7:] == ["100", "850", "8.50", "7", "to", "11", "(inside)"]

    def test_a_figure_that_does_not_scale_has_no_assumed_range(self) -> None:
        dev = report_with(figure("count", 2.0, "ms", "none"))
        pi = report_with(figure("count", 6.0, "ms", "none"))
        row = format_comparison(pi, dev).splitlines()[1]
        assert "3.00" in row
        assert row.rstrip().endswith("-")

    def test_a_figure_in_another_unit_is_left_out(self) -> None:
        dev = report_with(figure("x", 2.0, "ms", "none"))
        pi = report_with(figure("x", 6.0, "us", "none"))
        assert format_comparison(pi, dev) == "The two reports have no figure in common."

    def test_reports_with_nothing_in_common_say_so(self) -> None:
        empty = Report("dev", False, "2026-10-01T12:00:00Z", environment(), ())
        assert format_comparison(fixture_report(), empty) == (
            "The two reports have no figure in common."
        )
