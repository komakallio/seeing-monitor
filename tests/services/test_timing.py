"""The time stamper, against synthetic arrival times with known truth."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.clock import ClockStatus, VirtualClock
from seeingmon.frames import FrameFlag, TimeQuality
from seeingmon.services.acquire import timing
from seeingmon.services.acquire.timing import (
    FrameTime,
    StreamTiming,
    TimeStamper,
    TimingConfig,
)

T0 = 1_800_000_000 * 1_000_000_000  # the true arrival time of frame 0
PERIOD_NS = 11_300_000  # a 128-row ROI in bin1
EXPOSURE_S = 0.002
BOUND_NS = 1_000_000
LATENCY_SIGMA_NS = 5_000_000
OFFSET_NS = PERIOD_NS - round(EXPOSURE_S / 2 * 1e9)  # arrival minus this is mid-exposure


def make(
    *,
    latency_s: float = 0.0,
    status: ClockStatus | None = None,
    **options: Any,
) -> tuple[TimeStamper, VirtualClock]:
    clock = VirtualClock(
        status=status or ClockStatus(synchronized=True, error_bound_ns=BOUND_NS, source="test")
    )
    config = TimingConfig(latency_s=latency_s, latency_sigma_s=LATENCY_SIGMA_NS / 1e9, **options)
    stamper = TimeStamper(clock, config)
    stamper.configure(StreamTiming(exposure_s=EXPOSURE_S, period_s=PERIOD_NS / 1e9))
    return stamper, clock


def truth_utc(k: int, latency_ns: int = 0) -> int:
    """The true mid-exposure time of frame `k` of a clean stream."""
    return T0 + k * PERIOD_NS - OFFSET_NS - latency_ns


def run(
    stamper: TimeStamper,
    count: int,
    *,
    jitter_ns: float = 0.0,
    seed: int = 1,
    first: int = 0,
    shift_ns: int = 0,
) -> list[FrameTime]:
    """Stamp `count` clean frames, `first` onward. Jitter is Gaussian, clipped at 3 sigma."""
    rng = np.random.default_rng(seed)
    results = []
    for k in range(first, first + count):
        noise = 0.0
        if jitter_ns:
            noise = float(np.clip(rng.normal(0, jitter_ns), -3 * jitter_ns, 3 * jitter_ns))
        results.append(stamper.stamp(T0 + k * PERIOD_NS + shift_ns + round(noise)))
    return results


class TestQuality:
    def test_the_line_is_estimated_until_it_is_warm_and_then_fitted(self) -> None:
        stamper, _ = make()
        results = run(stamper, 60, jitter_ns=300_000)
        assert [r.t_quality for r in results[:19]] == [TimeQuality.ESTIMATED] * 19
        assert [r.t_quality for r in results[19:]] == [TimeQuality.FITTED] * 41
        assert stamper.warm
        assert all(r.flags == FrameFlag.NONE for r in results)

    def test_a_restart_goes_back_to_estimated(self) -> None:
        stamper, _ = make()
        run(stamper, 60)
        stamper.reset()
        assert not stamper.warm
        again = run(stamper, 25)
        assert again[0].t_quality is TimeQuality.ESTIMATED
        assert again[-1].t_quality is TimeQuality.FITTED
        assert sum(r.t_quality is TimeQuality.ESTIMATED for r in again) == 19

    def test_a_new_stream_starts_a_new_line(self) -> None:
        stamper, _ = make()
        run(stamper, 60)
        stamper.configure(StreamTiming(exposure_s=0.01, period_s=0.02))
        assert not stamper.warm
        assert stamper.stamp(T0).t_quality is TimeQuality.ESTIMATED


class TestTheTimeItSets:
    def test_a_clean_stream_gets_the_exact_mid_exposure_time(self) -> None:
        stamper, _ = make()
        for k, result in enumerate(run(stamper, 80)):
            assert result.t_utc_ns == truth_utc(k)  # arrival - period + exposure / 2

    def test_the_latency_shifts_every_time(self) -> None:
        stamper, _ = make(latency_s=0.004)
        for k, result in enumerate(run(stamper, 40)):
            assert result.t_utc_ns == truth_utc(k, latency_ns=4_000_000)

    def test_the_fit_removes_arrival_jitter(self) -> None:
        stamper, _ = make()
        jitter_ns = 500_000
        results = run(stamper, 500, jitter_ns=jitter_ns, seed=7)
        errors = np.array([r.t_utc_ns - truth_utc(k) for k, r in enumerate(results)])
        settled = errors[128:]
        rms = float(np.sqrt(np.mean(settled**2)))
        # The expected error at the end of a 128-frame window is 0.18 of the jitter.
        assert rms < 0.25 * jitter_ns  # a quarter of the jitter, so at least 4 times smaller
        assert stamper.outliers == 0
        assert stamper.resets == 0
        assert stamper.jitter_s == pytest.approx(jitter_ns / 1e9, rel=0.35)

    def test_the_fit_matches_a_least_squares_line_of_the_window(self) -> None:
        stamper, _ = make()
        rng = np.random.default_rng(3)
        window: list[tuple[int, int]] = []
        for k in range(400):
            noise = round(float(np.clip(rng.normal(0, 400_000), -1_200_000, 1_200_000)))
            arrival = T0 + k * PERIOD_NS + noise
            result = stamper.stamp(arrival)
            window.append((k, arrival - T0))
            window = window[-128:]
            if k >= 150:
                x = np.array([point[0] for point in window], dtype=float)
                y = np.array([point[1] for point in window], dtype=float)
                slope, intercept = np.polyfit(x, y, 1)
                expected = T0 + round(intercept + slope * k) - OFFSET_NS
                assert abs(result.t_utc_ns - expected) <= 20  # float rounding of the reference

    def test_the_measured_period_matches_the_stream(self) -> None:
        stamper, _ = make()
        assert stamper.period_s is None
        run(stamper, 60)
        assert stamper.period_s == pytest.approx(PERIOD_NS / 1e9, rel=1e-9)

    def test_the_rebase_of_a_long_stream_changes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reference, _ = make()
        expected = run(reference, 400, jitter_ns=300_000, seed=5)
        monkeypatch.setattr(timing, "REBASE_AT", 40)
        rebased, _ = make()
        actual = run(rebased, 400, jitter_ns=300_000, seed=5)
        assert max(abs(a.t_utc_ns - e.t_utc_ns) for a, e in zip(actual, expected, strict=True)) <= 2
        assert [a.t_quality for a in actual] == [e.t_quality for e in expected]


class TestError:
    def test_the_error_is_the_clock_bound_plus_the_latency_sigma_plus_the_fit(self) -> None:
        stamper, _ = make()
        results = run(stamper, 60)
        assert results[-1].t_err_ns == BOUND_NS + LATENCY_SIGMA_NS  # no jitter, no fit error
        assert results[0].t_err_ns == BOUND_NS + LATENCY_SIGMA_NS + 2_000_000  # cold: jitter

    def test_jitter_adds_a_small_fit_error_to_a_warm_frame(self) -> None:
        stamper, _ = make()
        last = run(stamper, 300, jitter_ns=500_000, seed=2)[-1]
        extra = last.t_err_ns - BOUND_NS - LATENCY_SIGMA_NS
        assert 20_000 < extra < 250_000  # about 0.18 of the jitter

    def test_an_unknown_bound_uses_the_configured_default(self) -> None:
        status = ClockStatus(synchronized=True, error_bound_ns=None, source="test")
        stamper, _ = make(status=status, unknown_clock_error_s=0.25)
        last = run(stamper, 60)[-1]
        assert last.t_err_ns == 250_000_000 + LATENCY_SIGMA_NS

    def test_a_clock_that_cannot_tell_is_trusted_but_not_exact(self) -> None:
        status = ClockStatus(synchronized=None, error_bound_ns=None, source="system")
        stamper, _ = make(status=status)
        result = run(stamper, 60)[-1]
        assert result.t_quality is TimeQuality.FITTED
        assert result.flags == FrameFlag.NONE
        assert result.t_err_ns >= 100_000_000

    def test_a_clock_that_is_not_synchronized_makes_the_time_invalid(self) -> None:
        status = ClockStatus(synchronized=False, error_bound_ns=None, source="chrony")
        stamper, _ = make(status=status, invalid_clock_error_s=1000.0)
        for result in run(stamper, 60):
            assert result.t_quality is TimeQuality.INVALID
            assert result.flags == FrameFlag.TIME_INVALID
            assert result.t_err_ns == 1_000_000_000_000


class TestDrops:
    def test_counted_drops_keep_the_line_straight(self) -> None:
        stamper, _ = make()
        k = 0
        results = []
        for step in range(150):
            dropped = 3 if step in (60, 100) else 0
            k += 1 + dropped if step else 0
            results.append((k, stamper.stamp(T0 + k * PERIOD_NS, dropped)))
        for k, result in results[19:]:
            assert result.t_quality is TimeQuality.FITTED
            assert result.t_utc_ns == truth_utc(k)
        assert stamper.outliers == 0
        assert stamper.resets == 0

    def test_drops_that_nobody_counted_restart_the_line_and_it_recovers(self) -> None:
        stamper, _ = make()
        run(stamper, 60)
        gap_k = 62  # two frames vanished (60 and 61), and no source said so
        results = [stamper.stamp(T0 + k * PERIOD_NS) for k in range(gap_k, gap_k + 60)]
        assert stamper.resets == 1
        assert [r.t_quality for r in results[:2]] == [TimeQuality.ESTIMATED] * 2  # the disagreement
        assert results[0].t_err_ns > BOUND_NS + LATENCY_SIGMA_NS + PERIOD_NS  # and it says so
        assert results[-1].t_quality is TimeQuality.FITTED
        assert results[-1].t_utc_ns == truth_utc(gap_k + 59)

    @given(
        drops=st.lists(st.integers(0, 40), min_size=1, max_size=8),
        period_us=st.integers(2_000, 200_000),
    )
    def test_any_pattern_of_counted_drops_keeps_exact_times(
        self, drops: list[int], period_us: int
    ) -> None:
        period_ns = period_us * 1000
        clock = VirtualClock(status=ClockStatus(True, 0, "test"))
        stamper = TimeStamper(clock, TimingConfig())
        stamper.configure(StreamTiming(exposure_s=0.001, period_s=period_ns / 1e9))
        offset_ns = period_ns - 500_000
        k = 0
        first = True
        for gap in drops:
            for index in range(30):
                dropped = gap if index == 0 and not first else 0
                k = 0 if first else k + 1 + dropped
                first = False
                result = stamper.stamp(T0 + k * period_ns, dropped)
                if result.t_quality is TimeQuality.FITTED:
                    assert result.t_utc_ns == T0 + k * period_ns - offset_ns
        assert stamper.resets == 0
        assert stamper.outliers == 0


class TestClockSteps:
    def test_a_step_is_noticed_after_three_frames_and_the_line_is_rebuilt(self) -> None:
        stamper, _ = make()
        run(stamper, 200)
        step_ns = 1_500_000_000
        after = run(stamper, 60, first=200, shift_ns=step_ns)
        assert stamper.resets == 1
        # The first two frames disagree with the old line. They get its prediction, flagged by
        # their quality and by an error that includes the disagreement.
        assert [r.t_quality for r in after[:2]] == [TimeQuality.ESTIMATED] * 2
        assert after[0].t_utc_ns == truth_utc(200)
        assert after[0].t_err_ns >= step_ns
        # The third frame starts a new line from the new clock, and 20 frames later it is warm.
        assert after[2].t_quality is TimeQuality.ESTIMATED
        assert after[2].t_utc_ns == truth_utc(202) + step_ns
        assert [r.t_quality for r in after[2:21]] == [TimeQuality.ESTIMATED] * 19
        assert [r.t_quality for r in after[21:]] == [TimeQuality.FITTED] * 39
        assert after[-1].t_utc_ns == truth_utc(259) + step_ns

    def test_a_step_backward_is_handled_the_same_way(self) -> None:
        stamper, _ = make()
        run(stamper, 100)
        after = run(stamper, 50, first=100, shift_ns=-2_000_000_000)
        assert stamper.resets == 1
        assert after[-1].t_quality is TimeQuality.FITTED
        assert after[-1].t_utc_ns == truth_utc(149) - 2_000_000_000

    def test_synchronization_changing_starts_a_new_line(self) -> None:
        stamper, clock = make(
            status=ClockStatus(synchronized=False, error_bound_ns=None, source="chrony")
        )
        before = run(stamper, 50)
        assert all(r.t_quality is TimeQuality.INVALID for r in before)
        clock.set_status(ClockStatus(synchronized=True, error_bound_ns=2_000_000, source="chrony"))
        stepped = run(stamper, 50, first=50, shift_ns=40_000_000_000)  # chrony stepped the clock
        assert stamper.resets == 1
        assert stepped[0].t_quality is TimeQuality.ESTIMATED
        assert stepped[0].flags == FrameFlag.NONE
        assert stepped[0].t_err_ns == 2_000_000 + LATENCY_SIGMA_NS + 2_000_000
        assert stepped[-1].t_quality is TimeQuality.FITTED
        assert stepped[-1].t_utc_ns == truth_utc(99) + 40_000_000_000

    def test_one_late_frame_is_an_outlier_and_changes_nothing_else(self) -> None:
        stamper, _ = make()
        results = run(stamper, 100)
        late = stamper.stamp(T0 + 100 * PERIOD_NS + 8_000_000)  # 8 ms late
        assert late.t_quality is TimeQuality.ESTIMATED
        assert late.t_utc_ns == truth_utc(100)  # the line, not the arrival
        assert late.t_err_ns >= BOUND_NS + LATENCY_SIGMA_NS + 8_000_000
        after = stamper.stamp(T0 + 101 * PERIOD_NS)
        assert after.t_quality is TimeQuality.FITTED
        assert after.t_utc_ns == truth_utc(101)
        assert stamper.outliers == 1
        assert stamper.resets == 0
        assert len(results) == 100

    def test_a_burst_after_a_stall_is_stamped_from_the_line(self) -> None:
        """The camera buffered four frames during a stall, and they arrive together."""
        stamper, _ = make()
        run(stamper, 100)
        burst_ns = T0 + 104 * PERIOD_NS  # the end of the stall
        stamped = [stamper.stamp(burst_ns + 100_000 * index) for index in range(4)]
        assert [s.t_utc_ns for s in stamped] == [truth_utc(k) for k in range(100, 104)]
        assert all(s.t_quality is TimeQuality.ESTIMATED for s in stamped)
        # Each frame is one period closer to the line, so this is no step.
        assert stamper.outliers == 4
        assert stamper.resets == 0
        recovered = stamper.stamp(T0 + 104 * PERIOD_NS)
        assert recovered.t_quality is TimeQuality.FITTED
        assert recovered.t_utc_ns == truth_utc(104)


class TestSettings:
    def test_the_settings_are_checked(self) -> None:
        with pytest.raises(ValueError, match="warm-up"):
            TimingConfig(window=10, warmup=20)
        with pytest.raises(ValueError, match="step_frames"):
            TimingConfig(step_frames=1)
        with pytest.raises(ValueError, match="positive"):
            StreamTiming(exposure_s=0.0, period_s=0.01)

    def test_the_first_frame_gets_its_arrival_minus_the_offset(self) -> None:
        stamper, _ = make()
        assert math.isclose(stamper.stamp(T0).t_utc_ns, T0 - OFFSET_NS, abs_tol=1)
