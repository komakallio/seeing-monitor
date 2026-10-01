"""The benchmark runs, reports microseconds per frame, and takes its clock as a parameter."""

from __future__ import annotations

import json
import math

import pytest

from seeingmon.clock import ClockStatus
from seeingmon.fastpath.benchmark import (
    CaseResult,
    format_report,
    main,
    run_benchmark,
    star_frames,
)
from seeingmon.profile import Profile


class TickingClock:
    """A clock that advances by a fixed step at every reading, so that timings are exact."""

    def __init__(self, step_ns: int = 50_000) -> None:
        self._step_ns = step_ns
        self._now_ns = 0

    def utc_ns(self) -> int:
        return self.monotonic_ns()

    def monotonic_ns(self) -> int:
        self._now_ns += self._step_ns
        return self._now_ns

    def sleep(self, seconds: float) -> None:  # pragma: no cover - the benchmark never sleeps
        raise AssertionError("the benchmark must not sleep")

    def status(self) -> ClockStatus:
        return ClockStatus(synchronized=True, error_bound_ns=0, source="test")


def test_the_benchmark_reports_microseconds_per_frame_for_both_cases(profile: Profile) -> None:
    results = run_benchmark(
        frames=30,
        repeats=2,
        profile=profile,
        clock=TickingClock(),
        close_window_s=2.0,
    )
    assert [r.name for r in results] == ["bin1_128x128_uint16", "bin2_64x64_uint16"]
    bin1, bin2 = results
    assert bin1.shape == (128, 128)
    assert bin2.shape == (64, 64)
    for result in results:
        # Each batch takes one tick of 50 us, so a batch of 30 frames reads 1.67 us a frame.
        assert result.kernel_us == pytest.approx(50.0 / 30.0, rel=1e-6)
        assert result.stack_us == pytest.approx(50.0 / 30.0, rel=1e-6)
        assert result.push_us == pytest.approx(50.0 / 30.0, rel=1e-6)
        assert result.close_ms == pytest.approx(0.05)
        assert result.kernel_best_us <= result.kernel_us


def test_the_default_timer_gives_finite_positive_numbers(profile: Profile) -> None:
    """The default timer has a fine tick on every platform, so a batch never reads as zero.

    The monotonic clock of Windows before Python 3.13 ticks every 15.6 ms, and a batch of 40
    frames takes less than that. The test checks no speed: a CI runner can be fast or slow.
    """
    (bin1, bin2) = run_benchmark(frames=40, repeats=2, profile=profile, close_window_s=3.0)
    for result in (bin1, bin2):
        for value in (
            result.kernel_us,
            result.kernel_best_us,
            result.stack_us,
            result.push_us,
            result.push_best_us,
            result.close_ms,
        ):
            assert math.isfinite(value)
            assert value > 0.0


def test_a_coarse_clock_gives_finite_non_negative_numbers(profile: Profile) -> None:
    """A clock that ticks every 15.6 ms (Windows before Python 3.13) reads a short batch as zero."""

    class CoarseClock(TickingClock):
        def monotonic_ns(self) -> int:
            return super().monotonic_ns() // 15_600_000 * 15_600_000

    results = run_benchmark(
        frames=20, repeats=1, profile=profile, clock=CoarseClock(1_000), close_window_s=2.0
    )
    for result in results:
        for value in (result.kernel_us, result.stack_us, result.push_us, result.close_ms):
            assert math.isfinite(value)
            assert value >= 0.0


def test_the_report_names_every_case(profile: Profile) -> None:
    results = run_benchmark(
        frames=20,
        repeats=1,
        profile=profile,
        clock=TickingClock(),
        close_window_s=2.0,
    )
    text = format_report(results)
    assert "bin1_128x128_uint16" in text
    assert "bin2_64x64_uint16" in text
    assert text.count("us/frame") == 2
    assert "close one window" in text


def test_the_command_prints_the_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--frames", "20", "--repeats", "1", "--close-window-s", "2"]) == 0
    assert "kernel" in capsys.readouterr().out


def test_the_command_prints_json_on_request(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--frames", "20", "--repeats", "1", "--close-window-s", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["mode"] for item in payload] == ["bin1", "bin2"]
    assert set(payload[0]) == set(CaseResult.__dataclass_fields__)


def test_the_star_frames_have_a_star_and_a_black_level(profile: Profile) -> None:
    frames = star_frames((128, 128), 3.5, 2.65, 12)
    assert frames.dtype.name == "uint16"
    assert frames.shape == (64, 128, 128)
    assert int(frames.max()) < 65_536
    assert int(frames.max()) > 5_000
    assert float(frames[0, :10].mean()) == pytest.approx(30 * 16, abs=16)  # the black level
