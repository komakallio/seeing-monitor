"""The budget verdicts, on reports with fixed numbers."""

from __future__ import annotations

import pytest

from seeingmon.perf.budgets import (
    Budget,
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

from .helpers import (
    case,
    cloudy_sim_case,
    core_sim_case,
    day_sim_case,
    figure,
    fixture_report,
    with_case,
)


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

    def test_the_bin2_row_adds_the_receive_as_one_interpreter_figure(self) -> None:
        found = verdicts(fixture_report(bin2_rx=3.0))
        plain, core = found["fast-bin2"], found["core-bin2"]
        interpreter = PI4_SCALING["interpreter"]
        assert plain.value is not None
        assert plain.low is not None
        assert plain.high is not None
        assert core.value == pytest.approx(plain.value + 3.0)
        assert core.low == pytest.approx(plain.low + 3.0 * interpreter.low)
        assert core.high == pytest.approx(plain.high + 3.0 * interpreter.high)

    def test_a_costly_receive_makes_the_core_row_fail_while_the_fast_path_passes(self) -> None:
        found = verdicts(fixture_report(rx_compute=4.0, rx_wakeup=1.0, bin2_rx=6.0))
        assert found["fast-bin1"].verdict == "pass"
        assert found["core-bin1"].verdict == "fail"
        assert found["fast-bin2"].verdict == "marginal"
        assert found["core-bin2"].verdict == "fail"  # 6% on this machine is 42% or more

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
    """The `core-sim` case replaces the stand-ins with figures from the whole system."""

    @staticmethod
    def budget(report: Report, key: str) -> Budget:
        return next(item for item in build_budgets(report) if item.key == key)

    def test_the_measured_peaks_replace_the_stand_ins_of_the_memory_rows(self) -> None:
        report = with_case(fixture_report(), core_sim_case())
        labels = [term.label for term in self.budget(report, "memory-1.4").terms]
        assert "core process peak (measured)" in labels
        assert "web process peak (measured)" in labels
        assert "survey worker peak (measured in the system)" in labels
        assert "other children of core (measured)" in labels
        assert "acquire peak (prebuilt frames)" in labels
        assert not any("stand-in" in label or "imports only" in label for label in labels)

    def test_the_all_processes_figure_is_the_sum_of_the_measured_peaks(self) -> None:
        report = with_case(fixture_report(), core_sim_case())
        found = verdicts(report)
        # acquire 60 (the ipc case), core 140, the worker 450, web 90, the other children 8
        assert found["memory-1.4"].value == pytest.approx(60 + 140 + 450 + 90 + 8)
        assert found["memory-1.6"].value == found["memory-1.4"].value

    def test_the_sum_scales_with_the_memory_range_and_adds_the_share_of_the_system(self) -> None:
        found = verdicts(with_case(fixture_report(), core_sim_case()))["memory-1.4"]
        memory = PI4_SCALING["memory"]
        os_low, os_high = OS_MEMORY_MB
        total = 60 + 140 + 450 + 90 + 8
        assert found.low == pytest.approx(total * memory.low + os_low)
        assert found.high == pytest.approx(total * memory.high + os_high)

    def test_the_measured_share_replaces_the_two_parts_of_the_core_row(self) -> None:
        report = with_case(fixture_report(), core_sim_case(fastpath_receive=3.0))
        row = self.budget(report, "core-bin1")
        assert [(term.case, term.measurement) for term in row.terms] == [
            ("core-sim", "core.fastpath_receive_share")
        ]
        assert row.title.endswith("measured in core")
        assert (row.limit, row.unit) == (25.0, "% of one core")
        found = verdicts(report)["core-bin1"]
        interpreter = PI4_SCALING["interpreter"]
        assert found.value == pytest.approx(3.0)
        assert found.low == pytest.approx(3.0 * interpreter.low)
        assert found.high == pytest.approx(3.0 * interpreter.high)

    def test_a_share_in_the_range_of_the_limit_is_marginal_and_a_large_one_fails(self) -> None:
        interpreter = PI4_SCALING["interpreter"]
        straddles = (25.0 / interpreter.low + 25.0 / interpreter.high) / 2
        marginal = verdicts(with_case(fixture_report(), core_sim_case(fastpath_receive=straddles)))
        assert marginal["core-bin1"].verdict == "marginal"
        too_much = verdicts(with_case(fixture_report(), core_sim_case(fastpath_receive=25.0)))
        assert too_much["core-bin1"].verdict == "fail"

    def test_a_pi4_report_compares_the_measured_share_without_scaling(self) -> None:
        report = with_case(
            fixture_report(label="pi4", machine="arm64"), core_sim_case(fastpath_receive=3.0)
        )
        found = verdicts(report)["core-bin1"]
        assert (found.value, found.low, found.high) == (3.0, 3.0, 3.0)
        assert found.measured
        assert found.verdict == "pass"

    def test_the_other_rows_do_not_change(self) -> None:
        report = with_case(fixture_report(), core_sim_case())
        with_system = {item.key: item for item in build_budgets(report)}
        without = {item.key: item for item in build_budgets(fixture_report())}
        for key in ("acquire-cpu", "fast-bin1", "fast-bin2", "core-bin2", "survey-time"):
            assert with_system[key] == without[key], key

    def test_the_survey_memory_row_reads_the_measured_worker(self) -> None:
        found = verdicts(with_case(fixture_report(survey_peak_mb=100.0), core_sim_case()))
        assert found["survey-memory"].value == pytest.approx(450.0)  # not the 100 MB of `survey`

    def test_a_system_without_the_worker_reads_the_worker_of_the_survey_case(self) -> None:
        report = with_case(fixture_report(survey_peak_mb=120.0), core_sim_case(worker_peak_mb=None))
        found = verdicts(report)
        assert found["survey-memory"].value == pytest.approx(120.0)
        assert found["memory-1.4"].value == pytest.approx(60 + 140 + 120 + 90 + 8)

    def test_a_system_without_the_share_keeps_the_two_parts_of_the_core_row(self) -> None:
        report = with_case(fixture_report(), core_sim_case(fastpath_receive=None))
        row = self.budget(report, "core-bin1")
        assert [term.case for term in row.terms] == ["fastpath"] * 3 + ["ipc"] * 2
        assert "measured in core" not in row.title
        # The memory rows still read the measured peaks.
        memory = [term.label for term in self.budget(report, "memory-1.4").terms]
        assert "core process peak (measured)" in memory

    def test_a_system_without_the_core_peak_keeps_the_stand_ins_of_the_memory_rows(self) -> None:
        bare = core_sim_case()
        without = CaseResult(
            bare.name,
            "ok",
            measurements=tuple(item for item in bare.measurements if item.name != "core.peak_rss"),
        )
        report = with_case(fixture_report(), without)
        assert any("stand-in" in term.label for term in self.budget(report, "memory-1.4").terms)

    @pytest.mark.parametrize(
        "result",
        [
            CaseResult("core-sim", "skipped", "the core process is not on main"),
            CaseResult("core-sim", "failed", "RuntimeError: core stopped with code 3"),
        ],
    )
    def test_a_case_that_did_not_run_keeps_the_stand_ins(self, result: CaseResult) -> None:
        report = with_case(fixture_report(), result)
        labels = [term.label for term in self.budget(report, "memory-1.4").terms]
        assert any("stand-in" in label for label in labels)
        cases = [term.case for term in self.budget(report, "core-bin1").terms]
        assert cases == ["fastpath"] * 3 + ["ipc"] * 2
        assert verdicts(report)["memory-1.4"].value == pytest.approx(60 + 120 + 80 + 100 + 90)


class TestVisibilityRows:
    """The rows of a day of measuring, of a cloudy night of searching, and of the weighted centroid.

    The fixture's kernel case holds a search frame of 200 us and a Gaussian-weighted centroid that
    costs 50 us beyond the aperture, in both fast modes.
    """

    @staticmethod
    def budget(report: Report, key: str) -> Budget:
        return next(item for item in build_budgets(report) if item.key == key)

    @staticmethod
    def report() -> Report:
        return with_case(with_case(fixture_report(), day_sim_case()), cloudy_sim_case())

    def test_the_day_row_reads_the_share_that_day_sim_measured_in_core(self) -> None:
        row = self.budget(self.report(), "day-core")
        assert [(term.case, term.measurement) for term in row.terms] == [
            ("day-sim", "core.fastpath_receive_share")
        ]
        assert (row.limit, row.unit) == (25.0, "% of one core")
        assert row.title.endswith("measured in core")
        found = verdicts(self.report())["day-core"]
        interpreter = PI4_SCALING["interpreter"]
        assert found.value == pytest.approx(2.5)
        assert (found.low, found.high) == pytest.approx(
            (2.5 * interpreter.low, 2.5 * interpreter.high)
        )
        assert found.verdict == "marginal"  # 17.5 to 27.5% against 25%

    def test_a_day_share_beyond_the_limit_on_every_estimate_fails(self) -> None:
        report = with_case(fixture_report(), day_sim_case(fastpath_receive=4.0))
        assert verdicts(report)["day-core"].verdict == "fail"  # 28% or more
        report = with_case(fixture_report(), day_sim_case(fastpath_receive=1.0))
        assert verdicts(report)["day-core"].verdict == "pass"  # 11% at most

    def test_without_the_runs_their_rows_are_n_a_and_name_the_case(self) -> None:
        found = verdicts(fixture_report())
        assert found["day-core"].verdict == "n/a"
        assert found["day-core"].missing == ("day-sim: core.fastpath_receive_share",)
        for key in ("day-memory-1.4", "day-memory-1.6", "cloudy-memory-1.4", "cloudy-memory-1.6"):
            assert found[key].verdict == "n/a", key
        text = format_verdicts(evaluate(fixture_report()))
        assert "n/a (needs day-sim)" in text
        assert "n/a (needs cloudy-sim)" in text

    @pytest.mark.parametrize(("key", "case_name"), [("day", "day-sim"), ("cloudy", "cloudy-sim")])
    def test_the_memory_rows_of_a_run_add_its_peaks_to_acquire_of_the_ipc_case(
        self, key: str, case_name: str
    ) -> None:
        found = verdicts(self.report())
        budget, gate = found[f"{key}-memory-1.4"], found[f"{key}-memory-1.6"]
        memory = PI4_SCALING["memory"]
        os_low, os_high = OS_MEMORY_MB
        total = 60 + 150 + 300 + 90 + 8  # acquire (ipc), core, the worker, web, the others
        assert budget.value == pytest.approx(total)
        assert budget.low == pytest.approx(total * memory.low + os_low)
        assert budget.high == pytest.approx(total * memory.high + os_high)
        assert (budget.budget.limit, gate.budget.limit) == (1400.0, 1600.0)
        assert (gate.value, gate.low, gate.high) == (budget.value, budget.low, budget.high)
        cases = {term.case for term in budget.budget.terms if term.constant is None}
        assert cases == {"ipc", case_name}

    def test_each_run_has_its_own_memory_rows(self) -> None:
        report = with_case(
            with_case(fixture_report(), day_sim_case(core_peak_mb=200.0)),
            cloudy_sim_case(core_peak_mb=100.0),
        )
        found = verdicts(report)
        day, cloudy = found["day-memory-1.4"].value, found["cloudy-memory-1.4"].value
        assert day is not None
        assert cloudy is not None
        assert day - cloudy == pytest.approx(100.0)

    def test_a_run_without_a_worker_takes_the_worker_of_the_survey_case(self) -> None:
        report = with_case(fixture_report(survey_peak_mb=120.0), day_sim_case(worker_peak_mb=None))
        found = verdicts(report)
        assert found["day-memory-1.4"].value == pytest.approx(60 + 150 + 120 + 90 + 8)
        labels = [term.label for term in found["day-memory-1.4"].budget.terms]
        assert "survey worker peak" in labels
        assert found["day-core"].verdict != "n/a"

    def test_without_cloudy_sim_the_search_rows_add_the_kernel_frame_to_the_receive(self) -> None:
        found = verdicts(fixture_report())
        numpy, interpreter = PI4_SCALING["numpy"], PI4_SCALING["interpreter"]
        scheduler = PI4_SCALING["scheduler"]
        bin1, bin2 = found["search-bin1"], found["search-bin2"]
        # 200 us at 98 fps is 1.96% of a core, and the receive of the fixture adds 0.3 and 0.1.
        assert bin1.value == pytest.approx(1.96 + 0.3 + 0.1)
        assert bin1.low == pytest.approx(
            1.96 * numpy.low + 0.3 * interpreter.low + 0.1 * scheduler.low
        )
        assert bin1.high == pytest.approx(
            1.96 * numpy.high + 0.3 * interpreter.high + 0.1 * scheduler.high
        )
        # 200 us at 360 fps is 7.2%, and the bin2 receive of the fixture adds 1.0.
        assert bin2.value == pytest.approx(7.2 + 1.0)
        assert bin2.low == pytest.approx(7.2 * numpy.low + 1.0 * interpreter.low)
        assert "measured in core" not in bin1.budget.title
        assert (bin1.budget.limit, bin2.budget.limit) == (25.0, 25.0)

    def test_cloudy_sim_replaces_the_bin1_search_row_with_the_burst_measured_in_core(
        self,
    ) -> None:
        report = with_case(fixture_report(), cloudy_sim_case(burst_share=6.0))
        row = self.budget(report, "search-bin1")
        assert [(term.case, term.measurement) for term in row.terms] == [
            ("cloudy-sim", "core.search_burst_share")
        ]
        assert row.title.endswith("measured in core")
        found = verdicts(report)
        interpreter = PI4_SCALING["interpreter"]
        assert found["search-bin1"].value == pytest.approx(6.0)
        assert found["search-bin1"].low == pytest.approx(6.0 * interpreter.low)
        assert found["search-bin1"].verdict == "fail"  # 42% or more of a core
        # The bin2 mode has no run of the whole system, so its row keeps the kernel.
        assert self.budget(report, "search-bin2") == self.budget(fixture_report(), "search-bin2")

    def test_a_cloudy_run_without_the_burst_share_keeps_the_kernel_row(self) -> None:
        report = with_case(fixture_report(), cloudy_sim_case(burst_share=None))
        row = self.budget(report, "search-bin1")
        assert [term.case for term in row.terms] == ["kernel", "ipc", "ipc"]
        assert verdicts(report)["cloudy-memory-1.4"].verdict != "n/a"

    def test_a_search_without_its_kernel_figure_is_n_a(self) -> None:
        found = verdicts(fixture_report(search=False))
        assert found["search-bin1"].verdict == "n/a"
        assert "kernel: bin1_128x128_u16.search" in found["search-bin1"].missing
        assert found["gaussian-bin2"].verdict == "n/a"

    @pytest.mark.parametrize(
        ("key", "plain", "rate_hz"),
        [("gaussian-bin1", "fast-bin1", 98.0), ("gaussian-bin2", "fast-bin2", 360.0)],
    )
    def test_the_gaussian_rows_add_the_extra_of_the_centroid_to_the_fast_path(
        self, key: str, plain: str, rate_hz: float
    ) -> None:
        found = verdicts(fixture_report())
        numpy = PI4_SCALING["numpy"]
        row, base = found[key], found[plain]
        assert base.value is not None
        assert base.low is not None
        assert base.high is not None
        extra = 50.0 * rate_hz / 1e4  # 50 us a frame at the rate of the mode, in percent
        assert row.value == pytest.approx(base.value + extra)
        assert row.low == pytest.approx(base.low + extra * numpy.low)
        assert row.high == pytest.approx(base.high + extra * numpy.high)
        assert not row.budget.gate
        assert row.budget.title.endswith("(information)")

    def test_a_pi4_report_compares_the_new_rows_without_scaling(self) -> None:
        report = with_case(
            with_case(
                fixture_report(label="pi4", machine="arm64"), day_sim_case(fastpath_receive=8.0)
            ),
            cloudy_sim_case(burst_share=30.0),
        )
        found = verdicts(report)
        assert (found["day-core"].value, found["day-core"].low) == (8.0, 8.0)
        assert found["day-core"].verdict == "pass"
        assert found["search-bin1"].verdict == "fail"  # measured, so never marginal
        total = 60 + 150 + 300 + 90 + 8
        assert found["cloudy-memory-1.4"].value == pytest.approx(total)
        assert found["cloudy-memory-1.4"].low == pytest.approx(total)  # no assumed OS share

    def test_the_rows_of_the_night_do_not_change_with_the_new_runs(self) -> None:
        with_runs = {item.key: item for item in build_budgets(self.report())}
        without = {item.key: item for item in build_budgets(fixture_report())}
        for key in (
            "acquire-cpu",
            "fast-bin1",
            "core-bin1",
            "fast-bin2",
            "core-bin2",
            "survey-time",
            "survey-memory",
            "memory-1.4",
            "memory-1.6",
        ):
            assert with_runs[key] == without[key], key

    def test_the_new_rows_come_after_the_rows_of_the_night_with_unique_keys(self) -> None:
        keys = [item.key for item in build_budgets(self.report())]
        assert len(keys) == len(set(keys))
        assert keys[-9:] == [
            "day-core",
            "day-memory-1.4",
            "day-memory-1.6",
            "search-bin1",
            "search-bin2",
            "cloudy-memory-1.4",
            "cloudy-memory-1.6",
            "gaussian-bin1",
            "gaussian-bin2",
        ]


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
        assert "less cautious than the table, because it assumes a faster Pi 4" in text
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

    def test_a_much_faster_machine_makes_the_estimate_more_cautious_than_the_table(self) -> None:
        report = fixture_report()
        fast = Report(
            report.label,
            report.smoke,
            report.created_utc,
            report.environment,
            (case("kernel", figure("bin1_128x128_u16.kernel", 10.0, "us/frame", "numpy")),),
        )
        text = "\n".join(kernel_check(fast))
        assert "factor of 20.0 to 40.0" in text
        assert "more cautious than the table, because it assumes a slower Pi 4" in text

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
