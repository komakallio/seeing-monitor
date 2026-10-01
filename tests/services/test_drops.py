"""Drop accounting: each source alone and combined."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.services.acquire.drops import DropAccountant, DropCounters

PERIOD_NS = 11_300_000
PERIOD_S = PERIOD_NS / 1e9


def steady(accountant: DropAccountant, count: int, start_ns: int = 0) -> int:
    """Feed `count` frames one period apart, and return the time of the last."""
    at = start_ns
    for _ in range(count):
        assert accountant.frame_arrived(at, 0) == 0
        at += PERIOD_NS
    return at - PERIOD_NS


def accountant(**options: float) -> DropAccountant:
    instance = DropAccountant(**options)
    instance.reset(PERIOD_S)
    return instance


class TestEachSourceAlone:
    def test_a_steady_stream_loses_nothing(self) -> None:
        drops = accountant()
        steady(drops, 100)
        assert drops.counters == DropCounters(0, 0, 0)
        assert drops.counters.total == 0

    def test_the_driver_counter_alone(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        assert drops.frame_arrived(last + PERIOD_NS, 3) == 3
        assert drops.counters == DropCounters(driver=3, gap=0, overflow=0)

    def test_a_gap_alone(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        # Two frames did not arrive, and nothing counted them: 3 periods from the last frame.
        assert drops.frame_arrived(last + 3 * PERIOD_NS, 0) == 2
        assert drops.counters == DropCounters(driver=0, gap=2, overflow=0)

    def test_queue_overflow_alone(self) -> None:
        drops = accountant()
        steady(drops, 10)
        drops.record_overflow(4)
        drops.record_overflow(1)
        assert drops.counters == DropCounters(driver=0, gap=0, overflow=5)
        assert drops.counters.total == 5


class TestCombined:
    def test_a_gap_that_the_driver_counted_is_one_loss(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        assert drops.frame_arrived(last + 3 * PERIOD_NS, 2) == 2  # not 4
        assert drops.counters == DropCounters(driver=2, gap=0, overflow=0)

    def test_a_gap_larger_than_the_driver_count_adds_the_unexplained_frames(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        assert drops.frame_arrived(last + 4 * PERIOD_NS, 1) == 3
        assert drops.counters == DropCounters(driver=1, gap=2, overflow=0)

    def test_a_driver_count_larger_than_the_gap_stands(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        assert drops.frame_arrived(last + PERIOD_NS, 5) == 5  # no gap, but the SDK lost 5
        assert drops.counters == DropCounters(driver=5, gap=0, overflow=0)

    def test_the_three_sources_sum_in_the_totals(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        lost = drops.frame_arrived(last + 3 * PERIOD_NS, 0)  # gap: 2
        lost += drops.frame_arrived(last + 4 * PERIOD_NS, 7)  # driver: 7
        drops.record_overflow(3)
        assert lost == 9
        assert drops.counters == DropCounters(driver=7, gap=2, overflow=3)
        assert drops.counters.total == 12

    def test_a_late_frame_after_a_counted_loss_is_not_counted_again(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        assert drops.frame_arrived(last + 2 * PERIOD_NS, 1) == 1
        assert drops.frame_arrived(last + 3 * PERIOD_NS, 0) == 0


class TestTheGapRule:
    @pytest.mark.parametrize(
        ("interval_in_periods", "lost"),
        [
            (1.0, 0),
            (1.2, 0),
            (1.49, 0),
            (1.51, 1),
            (2.0, 1),
            (2.49, 1),
            (2.51, 2),
            (3.0, 2),
            (11.0, 10),
        ],
    )
    def test_an_interval_over_one_and_a_half_periods_loses_the_whole_periods_between(
        self, interval_in_periods: float, lost: int
    ) -> None:
        drops = accountant()
        last = steady(drops, 5)
        assert drops.frame_arrived(last + round(interval_in_periods * PERIOD_NS), 0) == lost

    def test_the_factor_is_adjustable(self) -> None:
        drops = accountant(gap_factor=3.0)
        last = steady(drops, 5)
        assert drops.frame_arrived(last + 2 * PERIOD_NS, 0) == 0
        assert drops.frame_arrived(last + 6 * PERIOD_NS, 0) == 3
        with pytest.raises(ValueError, match="exceed 1"):
            DropAccountant(gap_factor=1.0)

    def test_a_backward_or_zero_interval_is_not_a_gap(self) -> None:
        drops = accountant()
        last = steady(drops, 5)
        assert drops.frame_arrived(last, 0) == 0
        assert drops.frame_arrived(last - 10 * PERIOD_NS, 0) == 0

    def test_a_new_stream_has_no_gap_to_the_old_one(self) -> None:
        drops = accountant()
        last = steady(drops, 5)
        drops.reset(PERIOD_S)
        assert drops.frame_arrived(last + 1000 * PERIOD_NS, 0) == 0
        assert drops.counters.gap == 0

    def test_the_period_argument_overrides_the_hint(self) -> None:
        drops = accountant()
        last = steady(drops, 5)
        # The measured period is twice the expected one, so this interval is no gap.
        assert drops.frame_arrived(last + 3 * PERIOD_NS, 0, period_s=2 * PERIOD_S) == 0

    def test_the_totals_survive_a_new_stream(self) -> None:
        drops = accountant()
        last = steady(drops, 5)
        drops.frame_arrived(last + 3 * PERIOD_NS, 0)
        drops.reset(PERIOD_S)
        assert drops.counters.gap == 2


class TestLearningThePeriod:
    def test_without_a_hint_the_period_is_learned_before_a_gap_counts(self) -> None:
        drops = DropAccountant()
        drops.reset(None)
        at = 0
        for _ in range(4):  # three intervals are learned and not judged
            assert drops.frame_arrived(at, 0) == 0
            at += PERIOD_NS
        at += 3 * PERIOD_NS
        assert drops.frame_arrived(at, 0) == 3  # now the median interval is the period

    def test_an_early_gap_before_the_period_is_known_is_not_counted(self) -> None:
        drops = DropAccountant()
        drops.reset(None)
        drops.frame_arrived(0, 0)
        assert drops.frame_arrived(10 * PERIOD_NS, 0) == 0


class TestJitter:
    @given(
        seed=st.integers(0, 10_000),
        missing=st.lists(st.integers(0, 4), min_size=20, max_size=40),
    )
    def test_with_jitter_under_a_fifth_of_a_period_every_loss_is_counted_exactly(
        self, seed: int, missing: list[int]
    ) -> None:
        rng = np.random.default_rng(seed)
        drops = accountant()
        index = 0
        previous_jitter = None
        for gap in missing:
            index += 1 + gap  # `gap` frames vanish before this one
            jitter = round(float(rng.uniform(-0.2, 0.2)) * PERIOD_NS)
            lost = drops.frame_arrived(index * PERIOD_NS + jitter, 0)
            if previous_jitter is not None:  # the first frame has no interval
                assert lost == gap
            previous_jitter = jitter
        assert drops.counters.driver == 0
