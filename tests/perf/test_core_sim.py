"""The `core-sim` case: the figures that a run gives, and how the budgets read them.

The tests that start the system are the smoke test in `test_cases.py`, which runs every case, and
the slow test at the end of this file. The others feed the case with a made-up run.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import pytest

from seeingmon.perf.budgets import build_budgets, evaluate
from seeingmon.perf.cases import core_sim
from seeingmon.perf.cases.core_sim import BUDGET_FPS, measurements_from
from seeingmon.perf.registry import REGISTRY
from seeingmon.perf.report import SCALES, Measurement, Report
from seeingmon.perf.runner import execute_case
from seeingmon.perf.sysrun import FULL_PLAN, SMOKE_PLAN, RunPlan, SystemRun, run_system

from .helpers import MB, case, fabricated_run, fixture_report, with_case


def figures(run: SystemRun) -> dict[str, Measurement]:
    return {item.name: item for item in measurements_from(run)}


class TestFigures:
    def test_a_run_gives_every_figure_that_the_budgets_and_the_page_read(self) -> None:
        assert set(figures(fabricated_run())) == {
            "core.peak_rss",
            "survey_worker.peak_rss",
            "web.peak_rss",
            "acquire.peak_rss",
            "other.peak_rss",
            "all.peak_rss_sum",
            "core.fast_share",
            "core.idle_share",
            "acquire.fast_share",
            "acquire.idle_share",
            "web.fast_share",
            "web.idle_share",
            "core.frame_cost",
            "core.fastpath_receive_share",
            "survey_worker.cpu",
            "run.frame_rate",
            "run.length",
            "run.machine_busy",
        }

    def test_every_figure_is_finite_and_positive_with_a_known_scale(self) -> None:
        for item in measurements_from(fabricated_run()):
            assert item.unit
            assert item.scale in SCALES
            assert math.isfinite(item.value)
            assert item.value > 0, item.name

    def test_the_peaks_are_in_bytes_and_scale_as_memory(self) -> None:
        found = figures(fabricated_run())
        for role, peak in (("core", 132), ("web", 85), ("acquire", 138), ("survey_worker", 460)):
            item = found[f"{role}.peak_rss"]
            assert (item.value, item.unit, item.scale) == (peak * MB, "bytes", "memory")

    def test_the_sum_of_the_peaks_adds_every_process_and_names_the_parts(self) -> None:
        item = figures(fabricated_run())["all.peak_rss_sum"]
        assert (item.value, item.unit, item.scale) == (820 * MB, "bytes", "memory")
        assert item.detail["acquire"] == "138 MB"
        assert item.detail["survey_worker"] == "460 MB"
        assert item.detail["includes"] == "the simulator in acquire"

    def test_the_peak_of_acquire_says_that_it_includes_the_simulator(self) -> None:
        found = figures(fabricated_run())
        assert found["acquire.peak_rss"].detail["includes"] == "the simulator"
        assert found["acquire.fast_share"].detail["includes"] == "the simulator"
        assert "includes" not in found["core.peak_rss"].detail
        assert "includes" not in found["core.fast_share"].detail

    def test_the_shares_are_percentages_of_one_core_in_each_phase(self) -> None:
        found = figures(fabricated_run())
        assert found["core.fast_share"].value == pytest.approx(5.0)
        assert found["core.idle_share"].value == pytest.approx(1.0)
        assert found["acquire.fast_share"].value == pytest.approx(40.0)
        assert found["acquire.idle_share"].value == pytest.approx(2.0)
        assert found["web.fast_share"].value == pytest.approx(0.2)
        assert found["core.fast_share"].unit == "percent"
        assert found["core.fast_share"].detail["fast_seconds"] == 4.0
        assert found["core.fast_share"].detail["idle_seconds"] == 2.0

    def test_a_frame_costs_the_difference_of_the_shares_over_the_frame_rate(self) -> None:
        found = figures(fabricated_run())
        cost = found["core.frame_cost"]
        assert cost.value == pytest.approx(400.0)
        assert cost.unit == "us/frame"
        assert cost.detail["frames"] == 400
        assert cost.detail["fps"] == 100.0

    def test_the_share_of_the_budget_is_the_cost_of_a_frame_at_98_frames_per_second(self) -> None:
        share = figures(fabricated_run())["core.fastpath_receive_share"]
        assert BUDGET_FPS == 98.0  # the rate of the budget of the bin1 mode
        assert share.value == pytest.approx(400.0 * 98.0 / 1e6 * 100.0)  # 3.92% of a core
        assert share.unit == "percent"
        assert share.scale == "interpreter"
        assert share.detail["share_at_hz"] == 98.0

    def test_the_worker_gives_its_cpu_time_in_seconds(self) -> None:
        item = figures(fabricated_run())["survey_worker.cpu"]
        assert (item.value, item.unit) == (6.0, "s")
        assert item.detail["survey_results"] == 2

    def test_the_length_of_the_run_comes_with_the_facts_of_the_method(self) -> None:
        item = figures(fabricated_run())["run.length"]
        assert (item.value, item.unit) == (7.0, "s")  # eight samples, one second apart
        assert item.detail["sample_interval_s"] == 1.0
        assert item.detail["sensor"] == SMOKE_PLAN.sensor
        assert item.detail["speed"] == SMOKE_PLAN.speed
        assert item.detail["fast_s"] == 4.0
        assert item.detail["idle_s"] == 2.0
        assert item.detail["samples"] == 8
        assert item.detail["startup_s"] == 11.0
        assert item.detail["mode"] == "bin1_128x128_u16"
        assert item.detail["exposure_us"] == "2000"

    def test_the_load_of_the_machine_comes_with_the_share_of_the_system(self) -> None:
        item = figures(fabricated_run())["run.machine_busy"]
        assert (item.value, item.unit) == (12.5, "percent")
        assert item.detail["system_share_percent"] == 1.5
        assert item.detail["logical_cpus"] == 8

    def test_a_run_without_a_load_reading_leaves_the_figure_out(self) -> None:
        assert "run.machine_busy" not in figures(fabricated_run(machine_busy=None))

    def test_a_run_without_a_worker_leaves_out_the_figures_of_the_worker(self) -> None:
        found = figures(fabricated_run(with_worker=False))
        assert "survey_worker.peak_rss" not in found
        assert "survey_worker.cpu" not in found
        assert "core.peak_rss" in found

    def test_the_other_peak_is_always_there_and_positive(self) -> None:
        run = fabricated_run()
        quiet = replace(run, peaks={k: v for k, v in run.peaks.items() if k != "other"})
        item = figures(quiet)["other.peak_rss"]
        assert 0 < item.value < 1.0  # no other process: the floor of a figure

    def test_a_run_without_frames_gives_no_cost_and_no_share_for_the_budget(self) -> None:
        found = figures(fabricated_run(fast_frames=0))
        assert "core.frame_cost" not in found
        assert "core.fastpath_receive_share" not in found
        assert "core.fast_share" not in found
        assert "core.peak_rss" in found  # the peaks still stand


class TestBudgetsReadTheFigures:
    """The names of the case and the names that the budgets read must stay the same.

    A budget falls back to its stand-in when a figure is missing, so a renamed figure would change
    the verdicts without an error. These tests turn the change into a failure.
    """

    @staticmethod
    def report(*, with_worker: bool = True) -> Report:
        run = fabricated_run(with_worker=with_worker)
        return with_case(fixture_report(), case("core-sim", *measurements_from(run)))

    def test_the_memory_rows_read_every_measured_process_of_the_case(self) -> None:
        budget = next(item for item in build_budgets(self.report()) if item.key == "memory-1.4")
        labels = [term.label for term in budget.terms]
        assert "core process peak (measured)" in labels
        assert "web process peak (measured)" in labels
        assert "survey worker peak (measured in the system)" in labels
        assert "other children of core (measured)" in labels
        assert not any("stand-in" in label for label in labels)

    def test_the_core_row_reads_the_share_of_the_case(self) -> None:
        budgets = {item.key: item for item in build_budgets(self.report())}
        assert [term.measurement for term in budgets["core-bin1"].terms] == [
            "core.fastpath_receive_share"
        ]
        assert budgets["core-bin1"].title.endswith("measured in core")

    def test_no_row_misses_a_figure_of_the_case(self) -> None:
        found = {item.budget.key: item for item in evaluate(self.report())}
        for key in ("memory-1.4", "memory-1.6", "core-bin1", "survey-memory"):
            assert found[key].verdict != "n/a", key
            assert found[key].missing == ()
        assert found["memory-1.4"].value == pytest.approx(60 + 132 + 460 + 85 + 5)
        assert found["core-bin1"].value == pytest.approx(3.92)

    def test_a_run_without_a_worker_leaves_the_survey_row_to_the_survey_case(self) -> None:
        found = {item.budget.key: item for item in evaluate(self.report(with_worker=False))}
        assert found["survey-memory"].value == pytest.approx(
            100.0
        )  # the survey case of the fixture


class TestTheCase:
    def test_the_core_process_is_on_main_so_the_case_does_not_skip(self) -> None:
        assert core_sim.core_process_exists()

    def test_a_missing_module_gives_a_skip_with_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(core_sim, "CORE_MODULE", "seeingmon.perf.not_there")
        result = execute_case(REGISTRY.get("core-sim"), smoke=True)
        assert (result.status, result.reason) == ("skipped", "the core process is not on main")

    def test_a_module_without_the_entry_function_does_not_count_as_the_core_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(core_sim, "CORE_MODULE", "seeingmon.perf.timing")
        monkeypatch.setattr(core_sim, "CORE_ENTRY", "no_such_function")
        assert not core_sim.core_process_exists()

    def test_the_mode_picks_the_plan_and_the_case_returns_the_figures_of_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plans: list[RunPlan] = []

        def fake(plan: RunPlan, log: Callable[[str], None] | None = None) -> SystemRun:
            plans.append(plan)
            return fabricated_run(plan=plan)

        monkeypatch.setattr(core_sim, "run_system", fake)
        smoke = execute_case(REGISTRY.get("core-sim"), smoke=True)
        full = execute_case(REGISTRY.get("core-sim"), smoke=False)
        assert plans == [SMOKE_PLAN, FULL_PLAN]
        for result in (smoke, full):
            assert result.status == "ok"
            assert {item.name for item in result.measurements} == set(figures(fabricated_run()))

    def test_the_notes_state_the_method_and_what_the_figures_include(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            core_sim,
            "run_system",
            lambda plan, log=None: replace(
                fabricated_run(plan=plan), notes=("the sampling reached its limit of 5 s",)
            ),
        )
        result = execute_case(REGISTRY.get("core-sim"), smoke=True)
        text = "\n".join(result.notes)
        assert "simulator renders inside acquire" in text
        assert "one sample every 1 s" in text
        assert "One client polled web every 5 s" in text
        assert "12% busy" in text or "13% busy" in text
        assert "the sampling reached its limit of 5 s" in text

    def test_a_system_that_fails_to_start_fails_the_case_with_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(plan: RunPlan, log: Callable[[str], None] | None = None) -> SystemRun:
            raise RuntimeError("core stopped with code 3")

        monkeypatch.setattr(core_sim, "run_system", broken)
        result = execute_case(REGISTRY.get("core-sim"), smoke=True)
        assert result.status == "failed"
        assert "core stopped with code 3" in (result.reason or "")


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_the_full_run_reaches_the_steady_state_and_two_survey_results() -> None:
    """The run of the page: the `full` sensor, speed 1, until two survey frames have a result."""
    run = run_system(FULL_PLAN)
    assert run.fast.seconds >= FULL_PLAN.fast_seconds
    assert run.idle.seconds >= FULL_PLAN.idle_seconds
    assert run.survey_results >= FULL_PLAN.survey_results
    assert run.stream.get("mode")
    names = set(figures(run))
    assert {
        "core.peak_rss",
        "survey_worker.peak_rss",
        "web.peak_rss",
        "acquire.peak_rss",
        "core.frame_cost",
        "core.fastpath_receive_share",
        "survey_worker.cpu",
    } <= names
