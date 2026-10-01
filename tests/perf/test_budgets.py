"""The budget verdicts, on reports with fixed numbers."""

from __future__ import annotations

import pytest

from seeingmon.perf.budgets import (
    BudgetVerdict,
    build_budgets,
    classify,
    evaluate,
    format_verdicts,
    is_pi4_measurement,
    kernel_check,
)
from seeingmon.perf.report import CaseResult, Report
from seeingmon.perf.scaling import (
    ARCHITECTURE_KERNEL_ESTIMATE_MS,
    OS_MEMORY_MB,
    PI4_SCALING,
    ScaleRange,
    implied_factor,
)

from .helpers import case, figure, fixture_report


def verdicts(report: Report) -> dict[str, BudgetVerdict]:
    return {item.budget.key: item for item in evaluate(report)}


class TestClassify:
    @pytest.mark.parametrize(
        ("low", "high", "limit", "expected"),
        [
            (1.0, 5.0, 10.0, "pass"),
            (1.0, 10.0, 10.0, "pass"),  # the limit itself passes
            (1.0, 10.1, 10.0, "marginal"),
            (10.0, 12.0, 10.0, "marginal"),  # a range that starts at the limit straddles it
            (10.1, 12.0, 10.0, "fail"),
        ],
    )
    def test_the_whole_range_decides_the_verdict(
        self, low: float, high: float, limit: float, expected: str
    ) -> None:
        assert classify(low, high, limit) == expected


class TestEstimates:
    """A dev-machine report gives pass, marginal, and fail from the scaling table."""

    def test_the_default_fixture_shows_a_pass_and_a_marginal_verdict(self) -> None:
        found = verdicts(fixture_report())
        assert found["acquire-cpu"].verdict == "pass"
        assert found["fast-bin1"].verdict == "pass"
        assert found["fast-bin2"].verdict == "marginal"
        assert found["memory-1.4"].verdict == "pass"
        assert found["survey-memory"].verdict == "pass"

    def test_a_larger_figure_makes_the_budget_fail(self) -> None:
        found = verdicts(fixture_report(bin2_push=6.0, survey_peak_mb=2000.0))
        assert found["fast-bin2"].verdict == "fail"
        assert found["survey-memory"].verdict == "fail"
        assert found["memory-1.4"].verdict == "fail"
        assert found["memory-1.6"].verdict == "fail"

    def test_each_figure_is_scaled_by_the_range_of_its_class(self) -> None:
        item = verdicts(fixture_report())["fast-bin2"]
        interpreter, numpy = PI4_SCALING["interpreter"], PI4_SCALING["numpy"]
        # Push 2.0 and close 0.2 are NumPy figures, and the append of 0.1 is an interpreter one.
        assert item.value == pytest.approx(2.3)
        assert item.low == pytest.approx(2.2 * numpy.low + 0.1 * interpreter.low)
        assert item.high == pytest.approx(2.2 * numpy.high + 0.1 * interpreter.high)
        assert not item.measured

    def test_the_work_and_the_wake_ups_of_acquire_are_scaled_apart(self) -> None:
        item = verdicts(fixture_report())["acquire-cpu"]
        interpreter, scheduler = PI4_SCALING["interpreter"], PI4_SCALING["scheduler"]
        assert item.value == pytest.approx(0.5)  # 0.3 for the work, 0.2 for the wake-ups
        assert item.low == pytest.approx(0.3 * interpreter.low + 0.2 * scheduler.low)
        assert item.high == pytest.approx(0.3 * interpreter.high + 0.2 * scheduler.high)

    def test_the_receive_adds_to_the_fast_path_in_the_core_row_only(self) -> None:
        found = verdicts(fixture_report())
        plain, core = found["fast-bin1"], found["core-bin1"]
        interpreter, scheduler = PI4_SCALING["interpreter"], PI4_SCALING["scheduler"]
        assert plain.value is not None
        assert plain.low is not None
        assert plain.high is not None
        assert core.value == pytest.approx(plain.value + 0.3 + 0.1)
        assert core.low == pytest.approx(plain.low + 0.3 * interpreter.low + 0.1 * scheduler.low)
        assert core.high == pytest.approx(
            plain.high + 0.3 * interpreter.high + 0.1 * scheduler.high
        )
        assert "core-bin2" not in found  # only the planned mode counts the receive

    def test_a_costly_receive_makes_the_core_row_fail_while_the_fast_path_passes(self) -> None:
        found = verdicts(fixture_report(rx_compute=4.0, rx_wakeup=1.0))
        assert found["fast-bin1"].verdict == "pass"
        assert found["core-bin1"].verdict == "fail"

    def test_the_memory_budget_adds_the_assumed_share_of_the_operating_system(self) -> None:
        item = verdicts(fixture_report())["memory-1.4"]
        memory = PI4_SCALING["memory"]
        os_low, os_high = OS_MEMORY_MB
        measured = 60 + 120 + 80 + 100 + 90  # acquire, fast path, store, survey worker, web
        assert item.value == pytest.approx(measured)
        assert item.low == pytest.approx(measured * memory.low + os_low)
        assert item.high == pytest.approx(measured * memory.high + os_high)

    def test_the_two_memory_budgets_differ_only_in_their_limit(self) -> None:
        found = verdicts(fixture_report(survey_peak_mb=900.0))
        assert found["memory-1.4"].budget.limit == 1400.0
        assert found["memory-1.6"].budget.limit == 1600.0
        assert (found["memory-1.4"].low, found["memory-1.4"].high) == (
            found["memory-1.6"].low,
            found["memory-1.6"].high,
        )
        # 1,000 MB is the measured sum, and the range spans both limits.
        assert (found["memory-1.4"].verdict, found["memory-1.6"].verdict) == (
            "marginal",
            "marginal",
        )

    def test_the_survey_time_is_derived_and_gates_nothing(self) -> None:
        item = verdicts(fixture_report())["survey-time"]
        assert not item.budget.gate
        assert item.verdict == "pass"

    def test_a_missing_case_gives_n_a_and_names_what_is_missing(self) -> None:
        report = fixture_report()
        without_ipc = Report(
            report.label,
            report.smoke,
            report.created_utc,
            report.environment,
            tuple(result for result in report.cases if result.name != "ipc"),
        )
        found = verdicts(without_ipc)
        assert found["acquire-cpu"].verdict == "n/a"
        assert found["acquire-cpu"].value is None
        assert found["acquire-cpu"].missing == (
            "ipc: acquire.compute_share",
            "ipc: acquire.wakeup_share",
        )
        assert found["memory-1.4"].verdict == "n/a"
        assert found["fast-bin1"].verdict == "pass"  # the other budgets stand

    def test_a_skipped_case_counts_as_missing(self) -> None:
        report = fixture_report()
        cases = tuple(
            CaseResult("survey", "skipped", "the survey extra is not installed")
            if result.name == "survey"
            else result
            for result in report.cases
        )
        found = verdicts(Report("dev", False, report.created_utc, report.environment, cases))
        assert found["survey-memory"].verdict == "n/a"


class TestCoreSim:
    def test_a_core_sim_result_replaces_the_upper_bound_of_the_fast_path_and_the_store(
        self,
    ) -> None:
        report = fixture_report()
        with_core = Report(
            report.label,
            report.smoke,
            report.created_utc,
            report.environment,
            (*report.cases, case("core-sim", figure("cpu_share", 5.0, "percent"), peak_mb=300)),
        )
        terms = [term.label for term in build_budgets(with_core)[-1].terms]
        assert "core process peak" in terms
        assert not any("upper bound" in label for label in terms)
        found = verdicts(with_core)["memory-1.4"]
        assert found.value == pytest.approx(60 + 300 + 100 + 90)

    def test_a_skipped_core_sim_keeps_the_upper_bound(self) -> None:
        report = fixture_report()
        with_skip = Report(
            report.label,
            report.smoke,
            report.created_utc,
            report.environment,
            (*report.cases, CaseResult("core-sim", "skipped", "the core process is not on main")),
        )
        terms = [term.label for term in build_budgets(with_skip)[-1].terms]
        assert any("upper bound" in label for label in terms)


class TestPi4Measurement:
    def test_a_pi4_label_with_a_calibration_compares_without_scaling(self) -> None:
        report = fixture_report(label="pi4", machine="arm64", bin2_push=2.0)
        assert is_pi4_measurement(report)
        found = verdicts(report)
        item = found["fast-bin2"]
        assert item.measured
        assert (item.low, item.high) == (item.value, item.value)
        assert item.verdict == "pass"  # 2.3% of a core against 25%
        assert found["memory-1.4"].value == found["memory-1.4"].low  # no assumed OS share either

    def test_a_measured_figure_above_the_limit_fails_and_is_never_marginal(self) -> None:
        item = verdicts(fixture_report(label="pi4", machine="arm64", bin1_push=30.0))["fast-bin1"]
        assert item.verdict == "fail"
        assert item.measured

    def test_the_label_is_not_case_sensitive(self) -> None:
        assert is_pi4_measurement(fixture_report(label=" PI4 "))

    def test_the_label_alone_is_not_enough_without_a_measured_calibration(self) -> None:
        report = fixture_report(label="pi4", calibration=False)
        assert not is_pi4_measurement(report)
        assert not verdicts(report)["fast-bin2"].measured

    def test_a_dev_label_stays_an_estimate_even_with_a_calibration(self) -> None:
        assert not is_pi4_measurement(fixture_report(label="dev"))


class TestKernelCheck:
    def test_it_compares_the_architecture_estimate_with_the_table(self) -> None:
        lines = kernel_check(fixture_report())  # 100 us on this machine
        text = "\n".join(lines)
        low, high = implied_factor(0.1)
        assert (low, high) == pytest.approx((2.0, 4.0))
        assert "factor of 2.0 to 4.0" in text
        assert "assumes 5 to 11" in text
        assert "outside the table" in text
        assert "0.50 to 1.10 ms" in text
        assert "100 us per 128 x 128 frame" in text

    def test_a_slower_machine_makes_the_estimate_consistent(self) -> None:
        report = fixture_report()
        slow = Report(
            report.label,
            report.smoke,
            report.created_utc,
            report.environment,
            (case("kernel", figure("bin1_128x128_u16.kernel", 40.0, "us/frame", "numpy")),),
        )
        assert "consistent with the table" in "\n".join(kernel_check(slow))

    def test_a_pi4_report_prints_the_measured_time_and_no_factor(self) -> None:
        lines = kernel_check(fixture_report(label="pi4", machine="arm64"))
        assert len(lines) == 1
        assert "measured, not scaled" in lines[0]

    def test_a_report_without_the_kernel_case_gives_no_lines(self) -> None:
        report = fixture_report()
        bare = Report(report.label, report.smoke, report.created_utc, report.environment, ())
        assert kernel_check(bare) == []

    def test_the_architecture_estimate_is_the_one_in_the_architecture(self) -> None:
        assert ARCHITECTURE_KERNEL_ESTIMATE_MS == (0.2, 0.4)


class TestScalingTable:
    def test_every_range_is_ordered_and_positive(self) -> None:
        for name, item in PI4_SCALING.items():
            assert item.name == name
            assert 0 < item.low <= item.high

    def test_the_none_class_does_not_scale(self) -> None:
        assert (PI4_SCALING["none"].low, PI4_SCALING["none"].high) == (1.0, 1.0)

    def test_a_pi4_is_slower_than_the_dev_machine_for_every_cpu_class(self) -> None:
        for name in ("interpreter", "numpy"):
            assert PI4_SCALING[name].low > 1.0

    def test_a_bad_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="low <= high"):
            ScaleRange("x", 3.0, 2.0, "a", "b")
        with pytest.raises(ValueError, match="low <= high"):
            ScaleRange("x", 0.0, 2.0, "a", "b")

    def test_the_implied_factor_needs_a_positive_time(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            implied_factor(0.0)


class TestFormat:
    def test_the_table_shows_every_verdict_word_and_labels_estimates(self) -> None:
        report = fixture_report(bin2_push=2.0)
        text = format_verdicts(evaluate(report))
        assert "pass" in text
        assert "marginal" in text
        assert "(estimate)" in text
        assert "(not a gate)" in text
        assert "(measured)" not in text

    def test_a_pi4_table_says_measured(self) -> None:
        text = format_verdicts(evaluate(fixture_report(label="pi4", machine="arm64")))
        assert "(measured)" in text
        assert "(estimate)" not in text

    def test_a_table_with_a_failure_shows_the_word_fail(self) -> None:
        text = format_verdicts(evaluate(fixture_report(bin2_push=6.0)))
        assert "fail" in text

    def test_an_n_a_row_names_the_missing_figures(self) -> None:
        report = fixture_report()
        bare = Report(report.label, report.smoke, report.created_utc, report.environment, ())
        text = format_verdicts(evaluate(bare))
        assert "n/a" in text
        assert "n/a (needs ipc)" in text
