"""The case registry, the case context, and how a case turns into a result."""

from __future__ import annotations

import importlib

import pytest

from seeingmon.perf.registry import (
    REGISTRY,
    CaseContext,
    Registry,
    UnknownCaseError,
    load_registry,
)
from seeingmon.perf.report import Measurement
from seeingmon.perf.runner import execute_case, scrub_paths

from . import probe_cases


def make_registry() -> Registry:
    registry = Registry()

    @registry.case("first", summary="The first case")
    def first(ctx: CaseContext) -> list[Measurement]:
        return [Measurement("a", "ms", 1.0)]

    @registry.case("second-case", summary="The second case")
    def second(ctx: CaseContext) -> list[Measurement]:
        return []

    return registry


class TestRegistry:
    def test_it_keeps_the_order_of_registration(self) -> None:
        registry = make_registry()
        assert registry.names() == ["first", "second-case"]
        assert [case.summary for case in registry.cases()] == ["The first case", "The second case"]
        assert len(registry) == 2
        assert "first" in registry
        assert "third" not in registry

    def test_the_decorator_returns_the_function_unchanged(self) -> None:
        registry = Registry()

        def work(ctx: CaseContext) -> list[Measurement]:
            return []

        assert registry.case("work", summary="x")(work) is work

    def test_a_name_registered_twice_is_an_error(self) -> None:
        registry = make_registry()
        with pytest.raises(ValueError, match="registered twice"):
            registry.case("first", summary="again")(lambda ctx: [])

    @pytest.mark.parametrize("name", ["", "Kernel", "two words", "-lead", "trail-", "a--b", "1x"])
    def test_a_name_must_be_lowercase_words_joined_by_hyphens(self, name: str) -> None:
        with pytest.raises(ValueError, match="lowercase words"):
            Registry().case(name, summary="x")

    def test_an_unknown_name_lists_the_known_cases(self) -> None:
        with pytest.raises(UnknownCaseError, match="first, second-case"):
            make_registry().get("third")

    def test_the_harness_registers_the_calibration_case_first(self) -> None:
        registry = load_registry()
        assert registry is REGISTRY
        assert registry.names()[0] == "calibration"

    def test_a_registry_can_come_from_another_module(self) -> None:
        assert load_registry("tests.perf.probe_cases") is probe_cases.REGISTRY
        with pytest.raises(TypeError, match="REGISTRY"):
            load_registry("seeingmon.perf.timing")


class TestCaseContext:
    def test_pick_chooses_by_the_smoke_flag(self) -> None:
        assert CaseContext(smoke=False).pick(1000, 10) == 1000
        assert CaseContext(smoke=True).pick(1000, 10) == 10

    def test_a_smoke_timer_is_tiny(self) -> None:
        timer = CaseContext(smoke=True).timer(500, number=1000, warmup=20)
        assert (timer.repeats, timer.number, timer.warmup) == (2, 4, 1)

    def test_a_full_timer_keeps_its_settings(self) -> None:
        timer = CaseContext().timer(30, number=7, warmup=2)
        assert (timer.repeats, timer.number, timer.warmup) == (30, 7, 2)

    def test_the_baseline_is_the_resident_size_when_marked(self) -> None:
        assert CaseContext().baseline_rss_bytes is None
        marked = CaseContext()
        marked.mark_baseline()
        baseline = marked.baseline_rss_bytes
        assert baseline is None or baseline > 0

    def test_notes_collect_in_order(self) -> None:
        context = CaseContext()
        context.note("one")
        context.note("two")
        assert context.notes == ["one", "two"]


class TestExecuteCase:
    def test_a_case_that_returns_figures_is_ok(self) -> None:
        result = execute_case(probe_cases.REGISTRY.get("probe-small"), smoke=True)
        assert result.status == "ok"
        assert result.reason is None
        assert [item.name for item in result.measurements] == ["answer"]
        assert result.duration_s >= 0
        assert result.system_busy_percent is None or 0 <= result.system_busy_percent <= 100

    def test_skip_case_gives_a_skipped_result_with_its_reason(self) -> None:
        result = execute_case(probe_cases.REGISTRY.get("probe-skip"), smoke=True)
        assert (result.status, result.reason) == ("skipped", "the component is not on main")
        assert result.measurements == ()

    def test_a_missing_module_of_the_project_means_not_on_main_yet(self) -> None:
        result = execute_case(probe_cases.REGISTRY.get("probe-missing"), smoke=True)
        assert result.status == "skipped"
        assert result.reason == "the module seeingmon.not_on_main_yet is not on main yet"

    def test_a_missing_third_party_package_means_not_installed(self) -> None:
        registry = Registry()

        @registry.case("needs-package", summary="x")
        def needs_package(ctx: CaseContext) -> list[Measurement]:
            importlib.import_module("a_package_that_does_not_exist")
            return []

        result = execute_case(registry.get("needs-package"), smoke=True)
        assert result.status == "skipped"
        assert result.reason is not None
        assert "a_package_that_does_not_exist is not installed" in result.reason

    def test_a_failure_is_a_failed_result_with_no_path_in_it(self) -> None:
        result = execute_case(probe_cases.REGISTRY.get("probe-fail"), smoke=True)
        assert result.status == "failed"
        assert result.reason == "FileNotFoundError: no such file: <path>"
        assert result.measurements == ()
        assert any("probe_cases.py" in line for line in result.notes)  # a file name, no folder
        assert not any("\\" in line or "/" in line for line in result.notes)

    def test_the_peak_memory_and_the_baseline_are_recorded_where_the_platform_gives_them(
        self,
    ) -> None:
        result = execute_case(probe_cases.REGISTRY.get("probe-small"), smoke=True)
        if result.peak_rss_bytes is None:
            pytest.skip("this platform gives no memory reading")
        assert result.baseline_rss_bytes is not None
        assert result.peak_rss_bytes > 0


# Each text holds an absolute path on purpose, so each line carries the allow marker.
WINDOWS_PATH = r"missing C:\Users\someone\frames.ser here"  # repo-check: allow
DRIVE_PATH = "missing D:/data/frames.ser"  # repo-check: allow
HOME_PATH = "cannot open '/home/someone/data/frames.ser'"  # repo-check: allow
USERS_PATH = "under /Users/someone/Library"  # repo-check: allow


class TestQuietWait:
    def test_a_case_that_never_finds_a_quiet_machine_says_so_in_its_notes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("seeingmon.perf.runner.wait_for_quiet", lambda *a, **k: (80.0, 3.0))
        result = execute_case(probe_cases.REGISTRY.get("probe-small"), smoke=True, quiet_wait_s=3.0)
        assert result.status == "ok"
        assert result.system_busy_percent == 80.0
        assert any("stayed more than 15% busy for 3 s" in note for note in result.notes)

    def test_a_quiet_machine_adds_no_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("seeingmon.perf.runner.wait_for_quiet", lambda *a, **k: (4.0, 0.5))
        result = execute_case(probe_cases.REGISTRY.get("probe-small"), smoke=True, quiet_wait_s=3.0)
        assert result.system_busy_percent == 4.0
        assert result.notes == ()

    def test_without_a_wait_a_busy_machine_adds_no_note(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("seeingmon.perf.runner.wait_for_quiet", lambda *a, **k: (90.0, 0.05))
        result = execute_case(probe_cases.REGISTRY.get("probe-small"), smoke=True)
        assert result.notes == ()
        assert result.system_busy_percent == 90.0


class TestScrubPaths:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (WINDOWS_PATH, "missing <path> here"),
            (DRIVE_PATH, "missing <path>"),
            (HOME_PATH, "cannot open '<path>'"),
            (USERS_PATH, "under <path>"),
            ("a ratio of 3/4 and the unit us/frame", "a ratio of 3/4 and the unit us/frame"),
            ("the module seeingmon.perf.timing", "the module seeingmon.perf.timing"),
        ],
    )
    def test_it_replaces_absolute_paths(self, text: str, expected: str) -> None:
        assert scrub_paths(text) == expected
