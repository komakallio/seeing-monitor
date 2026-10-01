"""The timer and its statistics, on synthetic functions and a scripted clock."""

from __future__ import annotations

import math

import numpy as np
import pytest

from seeingmon.perf.timing import NS_PER_S, Timer, TimingStats, autorange, percentile


class ScriptedClock:
    """A clock for a `Timer`: each batch takes the next duration of the script.

    The timer reads the clock twice for each batch, once before the calls and once after. The
    clock returns a start time and then the start plus the scripted duration.
    """

    def __init__(self, durations_ns: list[int]) -> None:
        self._durations = list(durations_ns)
        self._now = 1_000
        self._starting = True

    def __call__(self) -> int:
        if self._starting:
            self._starting = False
            return self._now
        self._starting = True
        self._now += self._durations.pop(0)
        return self._now


class TestPercentile:
    @pytest.mark.parametrize("q", [0, 5, 25, 50, 75, 95, 99, 100])
    def test_it_matches_numpy_on_a_sorted_sample(self, q: float) -> None:
        values = sorted([3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 5.8, 9.7, 9.3])
        assert percentile(values, q) == pytest.approx(float(np.percentile(values, q)))

    def test_one_value_is_every_percentile(self) -> None:
        assert percentile([7.0], 95.0) == 7.0

    def test_an_empty_sequence_and_a_bad_q_are_errors(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            percentile([], 50.0)
        with pytest.raises(ValueError, match="between 0 and 100"):
            percentile([1.0], 101.0)


class TestTimingStats:
    def test_it_summarizes_samples(self) -> None:
        stats = TimingStats.from_samples([5.0, 1.0, 3.0, 2.0, 4.0])
        assert (stats.samples, stats.min, stats.median, stats.max, stats.mean) == (5, 1, 3, 5, 3)
        assert stats.p95 == pytest.approx(4.8)
        assert stats.ordered()

    def test_the_median_of_an_even_count_is_the_middle_pair_average(self) -> None:
        assert TimingStats.from_samples([1.0, 2.0, 3.0, 10.0]).median == 2.5

    def test_scaled_multiplies_every_time_and_keeps_the_count(self) -> None:
        stats = TimingStats.from_samples([1.0, 2.0, 3.0]).scaled(1e3)
        assert (stats.samples, stats.min, stats.median, stats.max) == (3, 1e3, 2e3, 3e3)

    def test_it_survives_a_round_trip_through_a_dict(self) -> None:
        stats = TimingStats.from_samples([0.5, 0.25, 1.0, 0.75])
        assert TimingStats.from_dict(stats.to_dict()) == stats

    def test_it_rejects_no_samples_and_non_finite_values(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            TimingStats.from_samples([])
        with pytest.raises(ValueError, match="finite"):
            TimingStats.from_samples([1.0, math.nan])

    def test_a_malformed_dict_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="incomplete"):
            TimingStats.from_dict({"samples": 3})


class TestTimer:
    def test_it_reports_the_statistics_of_the_batch_times(self) -> None:
        durations = [d * 1_000_000 for d in (4, 1, 3, 2, 10)]  # milliseconds, in nanoseconds
        calls: list[int] = []
        timer = Timer(repeats=5, number=1, warmup=0, clock=ScriptedClock(durations))
        stats = timer.measure(lambda: calls.append(1))
        assert len(calls) == 5
        assert (stats.min, stats.median, stats.max) == pytest.approx((1e-3, 3e-3, 10e-3))
        assert stats.mean == pytest.approx(4e-3)
        assert stats.samples == 5

    def test_a_batch_of_many_calls_reports_the_time_of_one_call(self) -> None:
        timer = Timer(repeats=2, number=10, warmup=0, clock=ScriptedClock([10_000, 30_000]))
        stats = timer.measure(lambda: None)
        assert (stats.min, stats.max) == pytest.approx((1_000 / NS_PER_S, 3_000 / NS_PER_S))

    def test_the_warmup_calls_run_first_and_are_not_timed(self) -> None:
        calls: list[int] = []
        timer = Timer(repeats=3, number=2, warmup=4, clock=ScriptedClock([1, 1, 1]))
        timer.measure(lambda: calls.append(1))
        assert len(calls) == 4 + 3 * 2

    def test_it_times_a_real_function_with_the_default_clock(self) -> None:
        stats = Timer(repeats=3, number=5, warmup=1).measure(lambda: sum(range(1000)))
        assert 0 < stats.min <= stats.median <= stats.p95 <= stats.max < 1.0

    @pytest.mark.parametrize(("repeats", "number", "warmup"), [(0, 1, 0), (1, 0, 0), (1, 1, -1)])
    def test_it_rejects_a_bad_setting(self, repeats: int, number: int, warmup: int) -> None:
        with pytest.raises(ValueError, match="at least"):
            Timer(repeats=repeats, number=number, warmup=warmup)


class TestAutorange:
    def test_it_finds_a_batch_that_is_long_enough(self) -> None:
        ticks = iter(range(0, 10**12, 3_000_000))  # each reading is 3 ms after the last

        def clock() -> int:
            return next(ticks)

        # Every batch takes 3 ms whatever its size, so one call is long enough for a 2 ms target.
        assert autorange(lambda: None, 0.002, clock=clock) == 1

    def test_it_tries_larger_batches_until_one_is_long_enough(self) -> None:
        durations = iter([1_000, 1_000, 1_000, 5_000_000])  # 1, 2, and 5 calls, then 10 calls
        now = 0
        starting = True

        def clock() -> int:
            nonlocal now, starting
            if starting:
                starting = False
                return now
            starting = True
            now += next(durations)
            return now

        assert autorange(lambda: None, 0.001, clock=clock) == 10

    def test_a_clock_that_never_advances_stops_instead_of_looping(self) -> None:
        assert autorange(lambda: None, 1.0, clock=lambda: 0) == 10**6

    def test_it_rejects_a_target_that_is_not_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            autorange(lambda: None, 0.0)
