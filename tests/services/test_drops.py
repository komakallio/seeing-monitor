"""Drop accounting: each source alone and combined, and late reads."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.services.acquire.drops import DropAccountant, DropCounters

PERIOD_NS = 11_300_000
PERIOD_S = PERIOD_NS / 1e9
CATCH_UP_NS = 100_000  # a buffered frame comes this long after the late one


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

    def test_a_gap_alone_reaches_the_frame_after_it(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        # Two frames did not arrive, and nothing counted them: 3 periods from the last frame.
        assert drops.frame_arrived(last + 3 * PERIOD_NS, 0) == 0  # the next frame decides
        assert drops.counters == DropCounters(driver=0, gap=0, overflow=0)
        assert drops.frame_arrived(last + 4 * PERIOD_NS, 0) == 2  # a regular interval confirms it
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
        assert drops.frame_arrived(last + 4 * PERIOD_NS, 0) == 0  # and nothing more
        assert drops.counters == DropCounters(driver=2, gap=0, overflow=0)

    def test_a_gap_larger_than_the_driver_count_adds_the_unexplained_frames(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        assert drops.frame_arrived(last + 4 * PERIOD_NS, 1) == 1  # the driver counted one
        assert drops.frame_arrived(last + 5 * PERIOD_NS, 0) == 2  # the other two, confirmed
        assert drops.counters == DropCounters(driver=1, gap=2, overflow=0)

    def test_a_driver_count_larger_than_the_gap_stands(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        assert drops.frame_arrived(last + PERIOD_NS, 5) == 5  # no gap, but the SDK lost 5
        assert drops.counters == DropCounters(driver=5, gap=0, overflow=0)

    def test_the_three_sources_sum_in_the_totals(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        lost = drops.frame_arrived(last + 3 * PERIOD_NS, 0)  # gap: 2, confirmed by the next frame
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
        at = last + round(interval_in_periods * PERIOD_NS)
        assert drops.frame_arrived(at, 0) == 0
        assert drops.frame_arrived(at + PERIOD_NS, 0) == lost  # the next regular frame confirms

    def test_the_factor_is_adjustable(self) -> None:
        drops = accountant(gap_factor=3.0)
        last = steady(drops, 5)
        assert drops.frame_arrived(last + 2 * PERIOD_NS, 0) == 0
        assert drops.frame_arrived(last + 6 * PERIOD_NS, 0) == 0  # four periods: three are missing
        assert drops.frame_arrived(last + 7 * PERIOD_NS, 0) == 3
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
        drops.frame_arrived(last + 3 * PERIOD_NS, 0)  # a gap that the next frame would confirm
        drops.reset(PERIOD_S)
        assert drops.frame_arrived(last + 1000 * PERIOD_NS, 0) == 0
        assert drops.frame_arrived(last + 1001 * PERIOD_NS, 0) == 0
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
        drops.frame_arrived(last + 4 * PERIOD_NS, 0)  # confirms the gap
        drops.reset(PERIOD_S)
        assert drops.counters.gap == 2


class TestLateReads:
    def test_a_late_read_that_a_catch_up_follows_loses_nothing(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        late = last + round(2.2 * PERIOD_NS)  # 1.2 periods late, which looks like a lost frame
        assert drops.frame_arrived(late, 0) == 0
        assert drops.frame_arrived(late + CATCH_UP_NS, 0) == 0  # the buffered frame
        assert drops.frame_arrived(late + CATCH_UP_NS + PERIOD_NS, 0) == 0
        assert drops.counters == DropCounters(driver=0, gap=0, overflow=0, late=1)
        assert drops.counters.total == 0

    def test_a_stall_of_several_periods_and_the_burst_after_it_lose_nothing(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        at = last + 5 * PERIOD_NS  # the host was away for four periods
        lost = [drops.frame_arrived(at + step * CATCH_UP_NS, 0) for step in range(4)]  # four frames
        lost.append(drops.frame_arrived(at + 3 * CATCH_UP_NS + PERIOD_NS, 0))
        assert lost == [0] * 5
        assert drops.counters == DropCounters(driver=0, gap=0, overflow=0, late=1)

    def test_the_driver_count_stands_when_the_catch_up_clears_the_rest(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        # The camera held three frames and lost one (the SDK counted it). The host was four periods
        # late, which looks like three lost frames, and the buffered frames follow at once.
        assert drops.frame_arrived(last + 4 * PERIOD_NS, 1) == 1
        assert drops.frame_arrived(last + 4 * PERIOD_NS + CATCH_UP_NS, 0) == 0
        assert drops.counters == DropCounters(driver=1, gap=0, overflow=0, late=1)

    def test_each_late_read_is_judged_by_the_read_that_follows_it(self) -> None:
        drops = accountant()
        at = 0
        drops.frame_arrived(at, 0)
        for _ in range(100):  # three frames in three periods: one late read, two catch-ups
            for step in (2.6, 0.2, 0.2):
                at += round(step * PERIOD_NS)
                assert drops.frame_arrived(at, 0) == 0
        assert drops.counters == DropCounters(driver=0, gap=0, overflow=0, late=100)

    def test_a_read_after_a_regular_interval_confirms_the_gap_before_it(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        late = last + 3 * PERIOD_NS
        assert drops.frame_arrived(late, 0) == 0
        # The next frame follows after 0.51 periods, which is no catch-up, so two frames are lost.
        assert drops.frame_arrived(late + round(0.51 * PERIOD_NS), 0) == 2
        assert drops.counters.late == 0

    def test_a_catch_up_ends_at_half_a_period(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        late = last + 3 * PERIOD_NS
        drops.frame_arrived(late, 0)
        assert drops.frame_arrived(late + round(0.49 * PERIOD_NS), 0) == 0
        assert drops.counters.late == 1

    def test_consecutive_gaps_confirm_one_another(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        # A camera that delivers every second frame: each interval is two periods.
        lost = [drops.frame_arrived(last + 2 * step * PERIOD_NS, 0) for step in range(1, 6)]
        assert lost == [0, 1, 1, 1, 1]  # each read confirms the gap before the previous one
        assert drops.counters.gap == 4

    def test_the_catch_up_counts_for_the_late_read_only(self) -> None:
        drops = accountant()
        last = steady(drops, 10)
        late = last + 3 * PERIOD_NS
        drops.frame_arrived(late, 0)
        drops.frame_arrived(late + CATCH_UP_NS, 0)  # the catch-up
        drops.frame_arrived(late + 2 * CATCH_UP_NS, 0)  # a second buffered frame
        assert drops.counters.late == 1


class TestLearningThePeriod:
    def test_without_a_hint_the_period_is_learned_before_a_gap_counts(self) -> None:
        drops = DropAccountant()
        drops.reset(None)
        at = 0
        for _ in range(4):  # three intervals are learned and not judged
            assert drops.frame_arrived(at, 0) == 0
            at += PERIOD_NS
        at += 3 * PERIOD_NS
        assert drops.frame_arrived(at, 0) == 0  # now the median interval is the period
        assert drops.frame_arrived(at + PERIOD_NS, 0) == 3

    def test_an_early_gap_before_the_period_is_known_is_not_counted(self) -> None:
        drops = DropAccountant()
        drops.reset(None)
        drops.frame_arrived(0, 0)
        assert drops.frame_arrived(10 * PERIOD_NS, 0) == 0
        assert drops.frame_arrived(11 * PERIOD_NS, 0) == 0


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
        charged = []
        for gap in missing:
            index += 1 + gap  # `gap` frames vanish before this one
            jitter = round(float(rng.uniform(-0.2, 0.2)) * PERIOD_NS)
            charged.append(drops.frame_arrived(index * PERIOD_NS + jitter, 0))
        # The first frame has no interval. Each later gap reaches the frame after the gap.
        assert charged[:2] == [0, 0]
        assert charged[2:] == missing[1:-1]
        assert drops.counters.driver == 0
        assert drops.counters.late == 0
