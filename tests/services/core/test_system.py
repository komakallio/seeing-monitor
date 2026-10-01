"""The reads of the machine: the load average and the memory in use."""

from __future__ import annotations

from pathlib import Path

import pytest

from seeingmon.services.core.system import (
    SystemStats,
    parse_loadavg,
    parse_meminfo,
    read_system_stats,
)

MEMINFO = """\
MemTotal:        4046344 kB
MemFree:          187184 kB
MemAvailable:    2097152 kB
Buffers:          123456 kB
"""


class TestLoadAverage:
    def test_the_first_field_is_the_last_minute(self) -> None:
        assert parse_loadavg("0.52 0.58 0.59 1/467 12345\n") == 0.52

    @pytest.mark.parametrize("text", ["", "   ", "x 1 2", "nan 1 2", "-1 0 0", "inf 0 0"])
    def test_an_unreadable_text_is_unknown(self, text: str) -> None:
        assert parse_loadavg(text) is None


class TestMemory:
    def test_the_memory_in_use_is_the_total_minus_the_available(self) -> None:
        assert parse_meminfo(MEMINFO) == pytest.approx((4046344 - 2097152) / 1024)

    def test_a_kernel_without_the_available_line_is_unknown(self) -> None:
        assert parse_meminfo("MemTotal: 100 kB\nMemFree: 20 kB\n") is None

    @pytest.mark.parametrize(
        "text",
        ["", "MemTotal: abc kB\nMemAvailable: 5 kB\n", "MemTotal: 5 kB\nMemAvailable: 9 kB\n"],
    )
    def test_nonsense_is_unknown(self, text: str) -> None:
        assert parse_meminfo(text) is None


class TestReadingTheFiles:
    def test_the_files_of_proc_are_read(self, tmp_path: Path) -> None:
        (tmp_path / "loadavg").write_text("1.25 1 1 1/1 1\n")
        (tmp_path / "meminfo").write_text(MEMINFO)
        stats = read_system_stats(tmp_path)
        assert stats.load_1m == 1.25
        assert stats.memory_used_mb == pytest.approx(1903.0, abs=1.0)

    def test_a_system_without_proc_reports_nothing(self, tmp_path: Path) -> None:
        assert read_system_stats(tmp_path / "missing") == SystemStats(None, None)
