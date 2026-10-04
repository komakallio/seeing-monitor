"""The focus history: the last 120 values, the spikes, and the best value with its reset."""

from __future__ import annotations

import math

import pytest

from seeingmon.services.core.alignment.focus import (
    HISTORY_LENGTH,
    SPIKE_FACTOR,
    SPIKE_MIN_PREVIOUS,
    SPIKE_WINDOW,
    FocusHistory,
)

NS = 1_000_000_000


def best_of(history: FocusHistory) -> float | None:
    """The best value, read through a function so that a type checker keeps no narrowing."""
    return history.best_px


def fill(history: FocusHistory, values: list[float], first_seq: int = 1) -> list[bool]:
    """Add one value for each frame, half a second apart, and return the spike flags."""
    flags = []
    for offset, value in enumerate(values):
        point = history.add(first_seq + offset, (first_seq + offset) * NS // 2, value, 30)
        assert point is not None
        flags.append(point.spike)
    return flags


class TestTheHistory:
    def test_the_history_keeps_the_last_120_values_with_the_time_of_their_frames(self) -> None:
        assert HISTORY_LENGTH == 120
        history = FocusHistory()
        fill(history, [2.5 + 0.01 * (i % 7) for i in range(300)])
        points = history.snapshot().points
        assert len(points) == 120
        assert [p.seq for p in points] == list(range(181, 301))  # the newest 120 frames
        assert points[0].t_utc_ns == 181 * NS // 2
        assert points[-1].t_utc_ns == 300 * NS // 2

    def test_the_index_counts_every_point_of_the_session_and_never_repeats(self) -> None:
        history = FocusHistory()
        fill(history, [2.5] * 150)
        indexes = [p.index for p in history.snapshot().points]
        assert indexes == list(range(31, 151))

    def test_a_point_says_how_many_stars_it_rests_on(self) -> None:
        history = FocusHistory()
        point = history.add(7, 123, 2.4, 41)
        assert point is not None
        assert (point.index, point.seq, point.t_utc_ns, point.fwhm_px, point.n_stars) == (
            1,
            7,
            123,
            2.4,
            41,
        )

    @pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
    def test_a_value_that_is_not_a_positive_number_is_ignored(self, value: float) -> None:
        history = FocusHistory()
        assert history.add(1, 1, value, 10) is None
        assert history.snapshot().points == ()
        assert best_of(history) is None

    def test_the_snapshot_does_not_change_when_the_history_does(self) -> None:
        history = FocusHistory()
        fill(history, [2.5, 2.4])
        before = history.snapshot()
        fill(history, [2.3], first_seq=3)
        assert len(before.points) == 2
        assert len(history.snapshot().points) == 3

    def test_the_snapshot_finds_the_point_of_a_frame(self) -> None:
        history = FocusHistory()
        fill(history, [2.5, 2.4, 2.3])
        snapshot = history.snapshot()
        point = snapshot.point_for(2)
        assert point is not None
        assert point.fwhm_px == 2.4
        assert snapshot.point_for(99) is None

    def test_the_length_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            FocusHistory(0)


class TestSpikes:
    def test_a_value_above_twice_the_median_of_the_preceding_ten_is_a_spike(self) -> None:
        assert (SPIKE_FACTOR, SPIKE_WINDOW, SPIKE_MIN_PREVIOUS) == (2.0, 10, 3)
        history = FocusHistory()
        flags = fill(history, [2.0] * 12 + [4.1, 2.0])
        assert flags == [False] * 12 + [True, False]

    def test_exactly_twice_the_median_is_not_a_spike(self) -> None:
        history = FocusHistory()
        flags = fill(history, [2.0] * 10 + [4.0])
        assert flags[-1] is False

    def test_the_first_values_are_never_spikes(self) -> None:
        history = FocusHistory()
        flags = fill(history, [2.0, 9.0, 9.0, 9.0])
        assert flags[:3] == [False, False, False]  # fewer than three values precede them
        assert flags[3] is False  # the median of [2, 9, 9] is 9

    def test_the_median_uses_the_preceding_ten_values_and_not_older_ones(self) -> None:
        history = FocusHistory()
        flags = fill(history, [1.0] * 10 + [3.0] * 10 + [5.0])
        # the ten values before the last one are all 3.0, so 5.0 is below 2 x 3.0
        assert flags[-1] is False
        assert flags[10] is True  # the first 3.0 follows ten values of 1.0

    def test_a_lasting_change_is_a_spike_only_until_it_fills_half_the_window(self) -> None:
        history = FocusHistory()
        flags = fill(history, [2.0] * 10 + [5.0] * 8)
        assert flags[10:] == [True] * 5 + [False] * 3  # the sixth value at the new level counts

    def test_the_spike_flag_stays_with_its_point(self) -> None:
        history = FocusHistory()
        fill(history, [2.0] * 5 + [9.0] + [2.0] * 5)
        points = history.snapshot().points
        assert [p.spike for p in points].count(True) == 1
        assert points[5].spike is True


class TestTheBestValue:
    def test_the_best_value_is_the_smallest_of_the_session(self) -> None:
        history = FocusHistory()
        assert best_of(history) is None
        fill(history, [3.0, 2.2, 2.8])
        assert best_of(history) == 2.2
        assert history.snapshot().best_px == 2.2

    def test_a_spike_never_sets_the_best_value_even_when_it_is_the_smallest(self) -> None:
        history = FocusHistory()
        fill(history, [1.0] * 5)
        history.reset_best()  # a refocus: every new value is higher than the old median
        values = [2.5, 2.4, 2.6] + [2.5] * 9
        flags = fill(history, values, first_seq=6)
        assert flags[1] is True  # 2.4 is the smallest of the new values, and it is a spike
        assert flags[-1] is False  # after a while the new level counts
        counted = [value for value, spike in zip(values, flags, strict=True) if not spike]
        best = best_of(history)
        assert best == min(counted)
        assert best is not None
        assert best > 2.4

    def test_a_low_value_is_no_spike_and_sets_the_best_value(self) -> None:
        history = FocusHistory()
        fill(history, [10.0, 10.0, 10.0, 10.0])
        flags = fill(history, [1.0], first_seq=5)  # far below the median, and a spike is high
        assert flags == [False]
        assert best_of(history) == 1.0

    def test_a_reset_forgets_the_best_value_and_keeps_the_history(self) -> None:
        history = FocusHistory()
        fill(history, [2.0, 2.1, 2.2, 2.3])
        assert best_of(history) == 2.0
        history.reset_best()
        assert best_of(history) is None
        assert len(history.snapshot().points) == 4  # the curve before the refocus stays
        fill(history, [2.4], first_seq=5)
        assert best_of(history) == 2.4  # the next value that counts starts the best again
        fill(history, [2.2], first_seq=6)
        assert best_of(history) == 2.2

    def test_a_clear_starts_a_new_session(self) -> None:
        history = FocusHistory()
        fill(history, [2.0, 2.1, 2.2])
        first = history.snapshot()
        history.clear()
        second = history.snapshot()
        assert second.session == first.session + 1
        assert second.points == ()
        assert second.best_px is None
        point = history.add(1, 1, 2.5, 10)
        assert point is not None
        assert point.index == 1  # the index counts again
