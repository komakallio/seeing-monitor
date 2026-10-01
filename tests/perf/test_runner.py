"""The runner: a case in a fresh child process, the pinned threads, and the assembled report."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.perf.registry import Registry, UnknownCaseError
from seeingmon.perf.report import CaseResult, Measurement
from seeingmon.perf.runner import (
    RESULT_MARKER,
    THREAD_VARIABLES,
    child_environment,
    parse_child_output,
    run_case_in_child,
    run_cases,
)

from . import probe_cases

REGISTRY_MODULE = "tests.perf.probe_cases"
MB = 1024 * 1024


@pytest.fixture(scope="module")
def repo_path(repo_root: Path) -> dict[str, str]:
    """The environment that lets a child import this test package, and keeps the path it has."""
    parts = [str(repo_root), os.environ.get("PYTHONPATH", "")]
    return {"PYTHONPATH": os.pathsep.join(part for part in parts if part)}


def child(
    name: str, repo_path: dict[str, str], log: Callable[[str], None] | None = None
) -> CaseResult:
    return run_case_in_child(
        name,
        smoke=True,
        registry_module=REGISTRY_MODULE,
        extra_env=repo_path,
        timeout_s=120.0,
        log=log,
    )


class TestChildEnvironment:
    def test_every_math_library_gets_one_thread(self) -> None:
        env = child_environment()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            assert env[name] == "1"
        assert set(THREAD_VARIABLES) <= set(env)

    def test_extra_variables_override_and_the_parent_environment_is_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEEINGMON_PERF_TEST_VALUE", "kept")
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        env = child_environment({"EXTRA": "yes"})
        assert env["SEEINGMON_PERF_TEST_VALUE"] == "kept"
        assert env["EXTRA"] == "yes"
        assert env["OMP_NUM_THREADS"] == "1"


class TestParseChildOutput:
    def test_it_reads_the_last_marker_line_and_ignores_other_output(self) -> None:
        payload = CaseResult("x", "skipped", "because").to_dict()
        text = f"banner\n{RESULT_MARKER}{json.dumps(payload)}\n"
        result = parse_child_output(text)
        assert result is not None
        assert (result.name, result.status) == ("x", "skipped")

    @pytest.mark.parametrize("text", ["", "no marker here\n", f"{RESULT_MARKER}{{not json\n"])
    def test_output_without_a_readable_result_gives_none(self, text: str) -> None:
        assert parse_child_output(text) is None

    def test_a_result_that_is_not_a_case_result_gives_none(self) -> None:
        assert parse_child_output(f'{RESULT_MARKER}{{"name": "x"}}\n') is None


class TestChildProcess:
    def test_a_case_runs_in_a_child_and_returns_its_figures(
        self, repo_path: dict[str, str]
    ) -> None:
        result = child("probe-small", repo_path)
        assert result.status == "ok"
        assert [(m.name, m.value) for m in result.measurements] == [("answer", 42.0)]

    def test_the_child_sees_one_thread_for_every_math_library(
        self, repo_path: dict[str, str]
    ) -> None:
        result = child("probe-threads", repo_path)
        assert result.status == "ok"
        assert [m.value for m in result.measurements] == [1.0, 1.0, 1.0]

    def test_a_skip_and_a_failure_travel_through_the_child(self, repo_path: dict[str, str]) -> None:
        skipped = child("probe-skip", repo_path)
        assert (skipped.status, skipped.reason) == ("skipped", "the component is not on main")
        failed = child("probe-fail", repo_path)
        assert failed.status == "failed"
        assert failed.reason == "FileNotFoundError: no such file: <path>"

    def test_what_a_case_prints_does_not_hide_its_result(self, repo_path: dict[str, str]) -> None:
        result = child("probe-noisy", repo_path)
        assert result.status == "ok"
        assert [m.value for m in result.measurements] == [1.0]

    def test_a_child_that_dies_without_a_result_is_a_failed_case(
        self, repo_path: dict[str, str]
    ) -> None:
        lines: list[str] = []
        result = child("probe-crash", repo_path, log=lines.append)
        assert result.status == "failed"
        assert result.reason == "the child process exited with code 3 and no result"

    def test_an_unknown_case_is_a_failed_case_with_the_exit_code_of_the_child(
        self, repo_path: dict[str, str]
    ) -> None:
        result = child("probe-nothing", repo_path)
        assert result.status == "failed"
        assert "exited with code 2" in (result.reason or "")

    def test_a_child_that_takes_too_long_is_a_failed_case(self, repo_path: dict[str, str]) -> None:
        result = run_case_in_child(
            "probe-small",
            smoke=True,
            registry_module=REGISTRY_MODULE,
            extra_env=repo_path,
            timeout_s=0.01,
        )
        assert result.status == "failed"
        assert "did not finish" in (result.reason or "")

    def test_the_peak_memory_of_one_case_does_not_include_another(
        self, repo_path: dict[str, str]
    ) -> None:
        heavy = child("probe-alloc", repo_path)
        light = child("probe-small", repo_path)
        assert heavy.status == light.status == "ok"
        if heavy.peak_rss_bytes is None or light.peak_rss_bytes is None:
            pytest.skip("this platform gives no memory reading")
        allocated = probe_cases.ALLOCATED_MB * MB
        # The heavy case holds 150 MB, and the light case that ran after it holds none of it.
        assert heavy.peak_rss_bytes > allocated * 0.8
        assert light.peak_rss_bytes < heavy.peak_rss_bytes - allocated * 0.6
        assert heavy.baseline_rss_bytes is not None
        assert heavy.peak_rss_bytes - heavy.baseline_rss_bytes > allocated * 0.6


class TestRunCases:
    def test_it_assembles_a_report_in_the_order_of_the_names(self) -> None:
        clock = VirtualClock()
        lines: list[str] = []
        report = run_cases(
            ["probe-skip", "probe-small"],
            smoke=True,
            label="test",
            isolate=False,
            registry=probe_cases.REGISTRY,
            progress=lines.append,
            clock=clock,
        )
        assert [result.name for result in report.cases] == ["probe-skip", "probe-small"]
        assert (report.label, report.smoke) == ("test", True)
        assert report.created_utc.endswith("Z")
        assert report.environment.python
        assert lines[0] == "probe-skip: running"
        assert lines[1].startswith("probe-skip: skipped in ")
        assert "the component is not on main" in lines[1]

    def test_no_names_means_every_case_of_the_registry(self) -> None:
        registry = Registry()
        for name in ("one", "two", "three"):
            registry.case(name, summary="x")(lambda ctx: [Measurement("v", "count", 1.0)])
        report = run_cases(None, smoke=True, isolate=False, registry=registry)
        assert [result.name for result in report.cases] == ["one", "two", "three"]
        assert all(result.ok for result in report.cases)

    def test_an_unknown_name_fails_before_anything_runs(self) -> None:
        lines: list[str] = []
        with pytest.raises(UnknownCaseError, match="probe-nothing"):
            run_cases(
                ["probe-small", "probe-nothing"],
                isolate=False,
                registry=probe_cases.REGISTRY,
                progress=lines.append,
            )
        assert lines == []

    def test_with_isolation_each_case_runs_in_its_own_child(
        self, repo_path: dict[str, str]
    ) -> None:
        report = run_cases(
            ["probe-small", "probe-threads"],
            smoke=True,
            registry_module=REGISTRY_MODULE,
            extra_env=repo_path,
            timeout_s=120.0,
        )
        assert [result.status for result in report.cases] == ["ok", "ok"]
