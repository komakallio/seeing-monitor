"""The report schema, its round trip through JSON, and the environment facts."""

from __future__ import annotations

import getpass
import json
import math
import platform
from pathlib import Path

import pytest

from seeingmon.perf.environment import (
    Environment,
    collect_environment,
    normalize_machine,
    parse_cpuinfo,
)
from seeingmon.perf.report import (
    SCALES,
    SCHEMA_VERSION,
    CaseResult,
    Measurement,
    Report,
    ReportError,
    read_report,
    write_report,
)
from seeingmon.perf.scaling import PI4_SCALING

from .helpers import case, environment, figure, fixture_report

# The board identifier is a made-up value. The tests assert that it never reaches a name.
FAKE_BOARD_ID = "feedface"
PI_CPUINFO = (
    "processor\t: 0\n"
    "BogoMIPS\t: 108.00\n"
    "Features\t: fp asimd evtstrm crc32 cpuid\n"
    "CPU implementer\t: 0x41\n"
    "CPU architecture: 8\n"
    "CPU variant\t: 0x0\n"
    "CPU part\t: 0xd08\n"
    "CPU revision\t: 3\n"
    "\n"
    "Hardware\t: BCM2711\n"
    "Revision\t: c03114\n"
    "Serial\t\t: " + FAKE_BOARD_ID + "\n"
    "Model\t\t: Raspberry Pi 4 Model B Rev 1.4\n"
)

X86_CPUINFO = """\
processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Test(R) Core(TM) Processor
cpu MHz\t\t: 2400.000
"""


class TestMeasurement:
    def test_it_round_trips_through_a_dict(self) -> None:
        item = figure("kernel", 120.0, "us/frame", "numpy", spread=0.1)
        item.detail["frames"] = 500
        assert Measurement.from_dict(item.to_dict()) == item

    @pytest.mark.parametrize("value", [math.nan, math.inf, -1.0])
    def test_it_rejects_a_value_that_is_not_finite_and_non_negative(self, value: float) -> None:
        with pytest.raises(ValueError, match="finite, non-negative"):
            Measurement("x", "ms", value)

    def test_it_rejects_an_unknown_scale_and_a_non_finite_detail(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            Measurement("x", "ms", 1.0, scale="gpu")
        with pytest.raises(ValueError, match="detail"):
            Measurement("x", "ms", 1.0, detail={"rate": math.inf})

    def test_every_scale_has_a_range_in_the_scaling_table(self) -> None:
        assert set(SCALES) == set(PI4_SCALING)


class TestCaseResult:
    def test_a_skipped_or_failed_case_needs_a_reason(self) -> None:
        with pytest.raises(ValueError, match="needs a reason"):
            CaseResult("x", "skipped")
        assert CaseResult("x", "skipped", "not on main").reason == "not on main"

    def test_an_unknown_status_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="status"):
            CaseResult("x", "weird")  # type: ignore[arg-type]

    def test_it_finds_a_figure_by_name(self) -> None:
        result = case("k", figure("a", 1.0, "ms"), figure("b", 2.0, "ms"))
        found = result.measurement("b")
        assert found is not None
        assert found.value == 2.0
        assert result.measurement("c") is None


class TestReportRoundTrip:
    def test_a_report_survives_a_write_and_a_read(self, tmp_path: Path) -> None:
        report = fixture_report()
        path = write_report(tmp_path / "nested" / "report.json", report)
        assert path.is_file()
        assert read_report(path) == report

    def test_the_file_is_json_with_a_schema_version(self, tmp_path: Path) -> None:
        path = write_report(tmp_path / "r.json", fixture_report())
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == SCHEMA_VERSION
        assert {"label", "smoke", "created_utc", "environment", "cases"} <= set(data)

    def test_a_skipped_case_keeps_its_reason(self, tmp_path: Path) -> None:
        report = Report(
            "dev",
            True,
            "2026-10-01T12:00:00Z",
            environment(),
            (CaseResult("core-sim", "skipped", "the core process is not on main"),),
        )
        again = read_report(write_report(tmp_path / "r.json", report))
        skipped = again.case("core-sim")
        assert skipped is not None
        assert (skipped.status, skipped.reason) == ("skipped", "the core process is not on main")
        assert again.measurement("core-sim", "anything") is None

    def test_an_unknown_schema_version_is_refused_with_both_versions(self, tmp_path: Path) -> None:
        data = fixture_report().to_dict()
        data["schema_version"] = SCHEMA_VERSION + 1
        path = tmp_path / "r.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(
            ReportError, match=rf"version {SCHEMA_VERSION + 1}.*version {SCHEMA_VERSION}"
        ):
            read_report(path)

    def test_a_file_that_is_not_a_report_is_refused(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.json"
        with pytest.raises(ReportError, match="cannot read"):
            read_report(missing)
        text = tmp_path / "text.json"
        text.write_text("not json", encoding="utf-8")
        with pytest.raises(ReportError, match="not valid JSON"):
            read_report(text)
        listing = tmp_path / "list.json"
        listing.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ReportError, match="does not hold a report"):
            read_report(listing)
        partial = tmp_path / "partial.json"
        partial.write_text(json.dumps({"schema_version": SCHEMA_VERSION}), encoding="utf-8")
        with pytest.raises(ReportError, match="incomplete"):
            read_report(partial)

    def test_the_writer_refuses_a_number_that_is_not_finite(self, tmp_path: Path) -> None:
        report = Report(
            "dev",
            False,
            "2026-10-01T12:00:00Z",
            environment(),
            (CaseResult("x", "ok", duration_s=math.nan),),
        )
        with pytest.raises(ValueError, match="JSON"):
            write_report(tmp_path / "r.json", report)


class TestEnvironment:
    def test_it_reports_the_architecture_the_os_family_and_the_versions(self) -> None:
        env = collect_environment()
        assert env.machine
        assert env.os_family == platform.system()
        assert env.python == platform.python_version()
        assert "numpy" in env.packages
        assert "seeingmon" in env.packages

    def test_it_survives_a_round_trip_through_a_dict(self) -> None:
        env = collect_environment()
        assert Environment.from_dict(env.to_dict()) == env

    def test_a_malformed_dict_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="incomplete"):
            Environment.from_dict({"machine": "x86-64"})

    def test_it_records_no_host_name_and_no_user_name(self) -> None:
        text = json.dumps(collect_environment().to_dict()).lower()
        for secret in (platform.node(), getpass.getuser()):
            if len(secret) >= 4:
                assert secret.lower() not in text

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("AMD64", "x86-64"),
            ("x86_64", "x86-64"),
            ("aarch64", "arm64"),
            ("arm64", "arm64"),
            ("riscv64", "riscv64"),
            ("", "unknown"),
        ],
    )
    def test_it_normalizes_the_machine_name(self, raw: str, expected: str) -> None:
        assert normalize_machine(raw) == expected


class TestCpuInfo:
    def test_a_raspberry_pi_gives_its_board_and_core_and_never_its_serial(self) -> None:
        name = parse_cpuinfo(PI_CPUINFO)
        assert name == "Raspberry Pi 4 Model B Rev 1.4 (Cortex-A72)"
        assert FAKE_BOARD_ID not in name

    def test_an_x86_processor_gives_its_model_name(self) -> None:
        assert parse_cpuinfo(X86_CPUINFO) == "Test(R) Core(TM) Processor"

    def test_a_text_without_the_lines_gives_nothing(self) -> None:
        assert parse_cpuinfo("processor\t: 0\n") is None
