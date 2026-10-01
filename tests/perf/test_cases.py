"""The cases: every case runs in smoke mode and returns figures or a skip marker.

This is the CI smoke test. It checks that each case runs, that its figures are finite and
positive, that its statistics are in order, and that the budgets find the figures that they read.
It checks nothing about how large a figure is, because the speed of a runner says nothing.
"""

from __future__ import annotations

import math
import subprocess
import sys
from typing import Any

import pytest

from seeingmon.perf.budgets import BASELINE, PEAK, build_budgets
from seeingmon.perf.cases.fastmodes import FAST_MODES
from seeingmon.perf.cases.survey import check_sky_step
from seeingmon.perf.registry import REGISTRY, load_registry
from seeingmon.perf.report import SCALES, Measurement, Report
from seeingmon.perf.runner import execute_case

from .helpers import environment

CASE_NAMES = load_registry().names()


def empty_report() -> Report:
    return Report("dev", True, "2026-10-01T12:00:00Z", environment(), ())


def check_measurement(item: Measurement) -> None:
    assert item.name
    assert item.unit
    assert item.scale in SCALES
    assert math.isfinite(item.value)
    assert item.value > 0, f"{item.name} is not positive"
    if item.stats is not None:
        stats = item.stats
        assert stats.samples >= 1
        assert stats.ordered(), f"the statistics of {item.name} are out of order"
        assert stats.min > 0
        # The value is the median, or the mean for a figure where a rare slow call is the point.
        assert any(
            math.isclose(item.value, kept, rel_tol=1e-9) for kept in (stats.median, stats.mean)
        )
    for key, value in item.detail.items():
        assert key
        if isinstance(value, float):
            assert math.isfinite(value)


class TestRegistryOfCases:
    def test_the_cases_come_in_the_order_of_the_budgets(self) -> None:
        assert CASE_NAMES == [
            "calibration",
            "kernel",
            "fastpath",
            "ipc",
            "survey",
            "store",
            "memory",
            "core-sim",
        ]

    def test_importing_the_cases_loads_no_code_under_test(self) -> None:
        script = (
            "import sys\n"
            "import seeingmon.perf.cases\n"
            "heavy = {'numpy', 'scipy', 'sep', 'astropy', 'erfa', 'pydantic', 'fastapi', 'PIL'}\n"
            "loaded = sorted(heavy & set(sys.modules))\n"
            "assert not loaded, loaded\n"
            "assert not any(name.startswith('seeingmon.fastpath') for name in sys.modules)\n"
            "assert not any(name.startswith('seeingmon.survey') for name in sys.modules)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False
        )
        assert completed.returncode == 0, completed.stderr

    def test_every_case_has_a_summary(self) -> None:
        for case in REGISTRY.cases():
            assert case.summary


class TestFastModes:
    def test_the_modes_are_the_three_of_the_brief(self) -> None:
        assert [mode.key for mode in FAST_MODES] == [
            "bin1_128x128_u16",
            "bin2_64x64_u16",
            "bin2_320x240_u8",
        ]
        rates = {mode.key: mode.rate_hz for mode in FAST_MODES}
        assert rates["bin1_128x128_u16"] == 98.0  # the budget of 2.55 ms per frame
        assert rates["bin2_64x64_u16"] == 360.0  # the budget of 0.69 ms per frame
        assert [mode.container_bits for mode in FAST_MODES] == [16, 16, 8]

    def test_the_period_follows_the_rate_and_never_falls_short(self) -> None:
        for mode in FAST_MODES:
            assert mode.period_ns * mode.rate_hz >= 1e9


@pytest.mark.parametrize("name", CASE_NAMES)
def test_every_case_runs_in_smoke_mode_and_returns_figures_or_a_skip(name: str) -> None:
    result = execute_case(REGISTRY.get(name), smoke=True)
    assert result.status in ("ok", "skipped"), f"{name} {result.status}: {result.reason}"
    assert result.name == name
    assert result.duration_s >= 0
    if result.status == "skipped":
        assert result.reason, "a skipped case says why"
        assert result.measurements == ()
        return
    assert result.reason is None
    assert result.measurements, f"{name} returned no figure"
    names = [item.name for item in result.measurements]
    assert len(names) == len(set(names)), f"{name} repeats a figure"
    for item in result.measurements:
        check_measurement(item)
    if result.peak_rss_bytes is not None:
        assert result.peak_rss_bytes > 0
    if name == "survey":
        # The sky quality step is part of every frame, so its stage must not go missing.
        assert "stage.quality" in names
    if name == "ipc":
        # The runs with the fake camera show what that fake adds, and no budget reads them.
        assert "fake_camera.acquire.cpu_per_frame" in names
    if name == "core-sim":
        # The system ran for real, so each process of it has a peak and the fast phase has a cost.
        # A budget falls back to its stand-in when one of these is missing, so the test names them.
        assert {
            "core.peak_rss",
            "web.peak_rss",
            "acquire.peak_rss",
            "other.peak_rss",
            "all.peak_rss_sum",
            "core.fast_share",
            "core.idle_share",
            "core.frame_cost",
            "core.fastpath_receive_share",
            "run.frame_rate",
            "run.length",
        } <= set(names)
    # The budgets read figures by name. A renamed figure must break this test and not the verdict.
    # With this case in the report, the budgets of `core-sim` read their measured terms.
    with_result = Report("dev", True, "2026-10-01T12:00:00Z", environment(), (result,))
    wanted = {
        term.measurement
        for budget in (*build_budgets(empty_report()), *build_budgets(with_result))
        for term in budget.terms
        if term.case == name and term.measurement not in (PEAK, BASELINE) and term.constant is None
    }
    assert wanted <= set(names), f"{name} lacks {sorted(wanted - set(names))}"


class TestSurveyCase:
    @staticmethod
    def job(
        stages: tuple[str, ...], rate: float | None, reasons: dict[str, str] | None = None
    ) -> dict[str, Any]:
        row = {"sky_rate_e_per_s_arcsec2": rate, "quality": reasons, "n_stars_used": 20}
        return {
            "timings": dict.fromkeys(stages, 0.1),
            "records": [
                {"record_type": "survey_frame", "row": {}},
                {"record_type": "sky_quality", "row": row},
            ],
        }

    def test_a_job_with_the_sky_step_gives_the_sky_row(self) -> None:
        row = check_sky_step(self.job(("detect", "quality"), 3.5))
        assert row["n_stars_used"] == 20

    def test_a_job_without_the_sky_step_fails(self) -> None:
        with pytest.raises(RuntimeError, match="skipped the sky quality step"):
            check_sky_step(self.job(("detect", "records"), 3.5))

    def test_a_step_that_measures_no_sky_fails_and_gives_the_reason(self) -> None:
        job = self.job(("quality",), None, {"sky_mag_arcsec2": "no dark model"})
        with pytest.raises(RuntimeError, match="no dark model"):
            check_sky_step(job)

    def test_a_job_without_a_sky_record_fails(self) -> None:
        job = {"timings": {"quality": 0.1}, "records": []}
        with pytest.raises(RuntimeError, match="measured no sky"):
            check_sky_step(job)
