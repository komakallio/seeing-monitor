"""The memory reader, the CPU clock, and the load sampler."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable

import pytest

from seeingmon.perf.load import (
    QUIET_BUSY_PERCENT,
    busy_percent_between,
    parse_proc_stat,
    system_busy_percent,
    wait_for_quiet,
)
from seeingmon.perf.memory import (
    current_rss_bytes,
    parse_proc_status,
    peak_rss_bytes,
    process_cpu_ns,
    read_memory,
)

MB = 1024 * 1024

ALLOCATE = """
import json
from seeingmon.perf.memory import current_rss_bytes, peak_rss_bytes

before = peak_rss_bytes()
block = bytearray(b"\\x01") * (100 * 1024 * 1024)  # allocates the block and writes every byte
after = peak_rss_bytes()
print(json.dumps({"before": before, "after": after, "size": len(block)}))
"""

PROC_STATUS = """\
Name:\tpython
VmPeak:\t  900000 kB
VmHWM:\t  123456 kB
VmRSS:\t  100000 kB
Threads:\t4
"""


class TestProcStatus:
    def test_it_reads_the_peak_and_the_current_size_in_bytes(self) -> None:
        reading = parse_proc_status(PROC_STATUS)
        assert reading is not None
        assert reading.peak_rss_bytes == 123_456 * 1024
        assert reading.rss_bytes == 100_000 * 1024

    def test_a_text_without_the_peak_gives_no_reading(self) -> None:
        assert parse_proc_status("Name:\tpython\nVmRSS:\t  100 kB\n") is None
        assert parse_proc_status("") is None

    def test_a_missing_current_size_stays_unknown(self) -> None:
        reading = parse_proc_status("VmHWM:\t 2048 kB\n")
        assert reading is not None
        assert reading.rss_bytes is None
        assert reading.peak_rss_bytes == 2048 * 1024


class TestMemoryReader:
    def test_the_reader_gives_a_plausible_reading_of_this_process(self) -> None:
        reading = read_memory()
        if reading is None:
            pytest.skip("this platform gives no memory reading")
        assert 5 * MB < reading.peak_rss_bytes < 1024 * 1024 * MB
        if reading.rss_bytes is not None:
            assert 0 < reading.rss_bytes <= reading.peak_rss_bytes

    def test_the_helpers_agree_with_the_reading(self) -> None:
        if read_memory() is None:
            pytest.skip("this platform gives no memory reading")
        assert peak_rss_bytes() is not None
        current = current_rss_bytes()
        assert current is None or current > 0

    def test_a_known_allocation_raises_the_peak_by_a_plausible_margin(self) -> None:
        if read_memory() is None:
            pytest.skip("this platform gives no memory reading")
        # A fresh process, so that nothing that the test run allocated earlier hides the rise.
        completed = subprocess.run(
            [sys.executable, "-c", ALLOCATE],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        figures = json.loads(completed.stdout.strip().splitlines()[-1])
        assert figures["size"] == 100 * MB
        rise = (figures["after"] - figures["before"]) / MB
        # The block is 100 MB. The bounds are generous, because the system may trim a working set
        # or round up to pages and arenas.
        assert 60 < rise < 300


class TestCpuClock:
    def test_the_process_cpu_time_never_goes_back(self) -> None:
        first = process_cpu_ns()
        sum(range(200_000))
        assert process_cpu_ns() >= first >= 0


class TestLoadSampler:
    def test_it_reads_the_first_line_of_proc_stat(self) -> None:
        # user nice system idle iowait irq softirq steal guest guest_nice
        assert parse_proc_stat("cpu  100 10 50 800 40 0 0 0 5 0") == (160, 1000)

    def test_it_accepts_an_old_kernel_with_four_columns(self) -> None:
        assert parse_proc_stat("cpu 10 0 10 80") == (20, 100)

    @pytest.mark.parametrize("line", ["", "cpu0 1 2 3 4 5", "cpu a b c d e", "intr 1 2 3 4 5"])
    def test_it_rejects_a_line_that_is_not_the_total(self, line: str) -> None:
        assert parse_proc_stat(line) is None

    def test_the_busy_share_is_the_change_in_busy_ticks_over_the_change_in_total(self) -> None:
        assert busy_percent_between((100, 1000), (150, 1100)) == pytest.approx(50.0)
        assert busy_percent_between((0, 0), (0, 100)) == 0.0

    def test_no_duration_gives_no_share(self) -> None:
        assert busy_percent_between((10, 100), (10, 100)) is None

    def test_a_counter_that_goes_back_is_clamped(self) -> None:
        assert busy_percent_between((50, 100), (40, 200)) == 0.0

    def test_the_sampler_returns_a_percentage_or_nothing(self) -> None:
        naps: list[float] = []
        value = system_busy_percent(0.01, sleep=naps.append)
        assert naps == [0.01]
        assert value is None or 0.0 <= value <= 100.0


class TestWaitForQuiet:
    """The wait takes samples of the load, and the tests script the samples."""

    @staticmethod
    def sampler(values: list[float | None]) -> tuple[Callable[[float], float | None], list[float]]:
        taken: list[float] = []
        remaining = list(values)

        def sample(seconds: float) -> float | None:
            taken.append(seconds)
            return remaining.pop(0)

        return sample, taken

    def test_a_quiet_machine_ends_the_wait_at_the_first_sample(self) -> None:
        sample, taken = self.sampler([3.0])
        assert wait_for_quiet(60.0, sample=sample) == (3.0, 0.25)
        assert taken == [0.25]

    def test_it_waits_until_the_load_falls_to_the_threshold(self) -> None:
        sample, taken = self.sampler([80.0, 40.0, 15.0, 2.0])
        busy, waited = wait_for_quiet(60.0, threshold_percent=15.0, sample=sample)
        assert busy == 15.0  # the threshold itself counts as quiet
        assert waited == pytest.approx(0.75)
        assert len(taken) == 3

    def test_it_gives_up_at_the_end_of_the_wait_and_returns_the_last_load(self) -> None:
        sample, taken = self.sampler([90.0] * 10)
        busy, waited = wait_for_quiet(1.0, sample_s=0.5, sample=sample)
        assert busy == 90.0
        assert waited == pytest.approx(1.0)
        assert len(taken) == 2

    def test_a_wait_of_zero_takes_one_sample(self) -> None:
        sample, taken = self.sampler([70.0])
        assert wait_for_quiet(0.0, sample=sample) == (70.0, 0.25)
        assert len(taken) == 1

    def test_a_system_that_gives_no_load_ends_the_wait(self) -> None:
        sample, taken = self.sampler([None])
        assert wait_for_quiet(60.0, sample=sample) == (None, 0.25)
        assert len(taken) == 1

    def test_the_threshold_is_the_one_that_the_report_states(self) -> None:
        assert QUIET_BUSY_PERCENT == 15.0
