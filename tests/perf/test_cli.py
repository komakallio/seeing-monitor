"""The `seeingmon perf` commands: registration, the report printout, and the errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seeingmon.cli import build_parser, main
from seeingmon.perf.report import Report, write_report

from .helpers import fixture_report


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def saved(tmp_path: Path) -> Path:
    return write_report(tmp_path / "dev.json", fixture_report(bin2_push=2.0))


class TestRegistration:
    def test_perf_is_a_command_with_two_subcommands(self) -> None:
        parser = build_parser()
        run_args = parser.parse_args(["perf", "run", "--smoke", "--cases", "kernel,ipc"])
        assert (run_args.perf_command, run_args.smoke, run_args.cases) == (
            "run",
            True,
            "kernel,ipc",
        )
        report_args = parser.parse_args(["perf", "report", "x.json", "--budgets"])
        assert (report_args.perf_command, report_args.budgets) == ("report", True)

    def test_run_has_the_options_of_the_brief(self) -> None:
        args = build_parser().parse_args(
            ["perf", "run", "--json", "local/perf/dev.json", "--label", "pi4"]
        )
        assert args.json == Path("local/perf/dev.json")
        assert args.label == "pi4"
        assert args.smoke is False

    def test_run_takes_a_wait_for_a_quiet_machine(self) -> None:
        parser = build_parser()
        assert parser.parse_args(["perf", "run", "--quiet-wait", "90"]).quiet_wait == 90.0
        assert parser.parse_args(["perf", "run"]).quiet_wait == 0.0

    def test_perf_without_a_subcommand_is_an_error(self) -> None:
        with pytest.raises(SystemExit) as raised:
            build_parser().parse_args(["perf"])
        assert raised.value.code == 2

    def test_the_command_appears_in_the_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main(["--help"])
        assert "perf" in capsys.readouterr().out


class TestRunList:
    def test_it_lists_the_cases_with_their_summaries(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = run(["perf", "run", "--list"], capsys)
        assert code == 0
        assert out.splitlines()[0].startswith("calibration")
        assert "Fixed workloads" in out

    def test_an_unknown_case_is_a_usage_error_that_lists_the_cases(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = run(["perf", "run", "--cases", "calibration,nothing"], capsys)
        assert code == 2
        assert "there is no case 'nothing'" in err
        assert "calibration" in err


class TestRunSmoke:
    def test_a_smoke_run_of_one_case_prints_the_table_and_writes_the_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "out" / "smoke.json"
        code, out, err = run(
            ["perf", "run", "--cases", "calibration", "--smoke", "--json", str(target)], capsys
        )
        assert code == 0
        assert "calibration: running" in err
        assert "calibration: ok in" in err
        assert "smoke run" in out
        assert "python_loop" in out
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["smoke"] is True
        assert [item["name"] for item in data["cases"]] == ["calibration"]


class TestReport:
    def test_it_prints_the_table(self, saved: Path, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(["perf", "report", str(saved)], capsys)
        assert code == 0
        assert "Performance report: dev" in out
        assert "fastpath: ok" in out
        assert "Budgets" not in out

    def test_budgets_print_the_verdicts_and_the_basis(
        self, saved: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = run(["perf", "report", str(saved), "--budgets"], capsys)
        assert code == 0
        assert "Basis: estimate" in out
        assert "estimate" in out
        for word in ("pass", "marginal"):
            assert word in out
        assert "Kernel on this machine" in out

    def test_a_pi4_report_compares_directly_and_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_report(tmp_path / "pi4.json", fixture_report(label="pi4", machine="arm64"))
        code, out, _ = run(["perf", "report", str(path), "--budgets"], capsys)
        assert code == 0
        assert "comes from a Pi 4" in out
        assert "without scaling" in out
        assert "(measured)" in out
        assert "Warning" not in out

    def test_a_pi4_label_on_an_x86_report_warns(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_report(tmp_path / "odd.json", fixture_report(label="pi4", machine="x86-64"))
        _, out, _ = run(["perf", "report", str(path), "--budgets"], capsys)
        assert "Warning: the report says x86-64, not arm64." in out

    def test_a_smoke_report_warns_that_its_figures_say_nothing_about_speed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        full = fixture_report()
        smoke = Report(full.label, True, full.created_utc, full.environment, full.cases)
        path = write_report(tmp_path / "smoke.json", smoke)
        _, out, _ = run(["perf", "report", str(path), "--budgets"], capsys)
        assert "smoke report" in out

    def test_a_baseline_prints_the_ratios_and_whether_they_are_in_range(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dev = write_report(tmp_path / "dev.json", fixture_report())
        pi = write_report(tmp_path / "pi.json", fixture_report(label="pi4", machine="arm64"))
        code, out, _ = run(["perf", "report", str(pi), "--baseline", str(dev)], capsys)
        assert code == 0
        assert "Ratio to the baseline report (dev)" in out
        assert "calibration: python_loop" in out

    def test_a_missing_file_is_a_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = run(["perf", "report", str(tmp_path / "missing.json")], capsys)
        assert code == 2
        assert "cannot read the report" in err

    def test_a_bad_baseline_is_a_usage_error(
        self, saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        code, _, err = run(["perf", "report", str(saved), "--baseline", str(bad)], capsys)
        assert code == 2
        assert "schema version" in err
