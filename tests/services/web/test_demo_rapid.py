"""The readings of the demo for the rapid focus mode: the curve, the taps, and the offer."""

from __future__ import annotations

import math
import statistics

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.frames import Roi
from seeingmon.services.core.alignment.focus import SPIKE_FACTOR
from seeingmon.services.core.alignment.rapid_availability import CoarseFocus, coarse_problem
from seeingmon.services.web.contract import RapidLocatedBy
from seeingmon.services.web.demo_rapid import (
    RAPID_AMPLITUDES,
    RAPID_BEST_ARCSEC,
    RAPID_BLUR_ARCSEC,
    RAPID_JOLT_AFTER,
    RAPID_JOLT_EVERY,
    RAPID_JOLT_FACTOR,
    RAPID_MAX_FWHM_ARCSEC,
    RAPID_SCATTER,
    RAPID_SWEEP_S,
    RapidDemo,
    focuser_offset,
    offer_problem,
    offer_view,
    reading_width_arcsec,
    true_width_arcsec,
)

ROI = Roi(2008, 1347, 128, 128)


def star(width: float, exposure_us: int, gain: int) -> tuple[float, bool]:
    """A star whose peak follows the exposure, and that saturates from 8 ms on."""
    return min(1.0, 0.27 * exposure_us / 2000), exposure_us >= 8000


def ended_of(demo: RapidDemo) -> str | None:
    """The reason of the last end, read through a call so that a type checker does not narrow it."""
    return demo.ended


def make(clock: VirtualClock | None = None) -> tuple[RapidDemo, VirtualClock]:
    clock = clock or VirtualClock(1_800_000_000_000_000_000)
    return RapidDemo(clock, mode="bin1", roi=ROI, scale_arcsec_px=1.91, star=star), clock


# --- The curve --------------------------------------------------------------------------------


class TestTheCurve:
    def test_a_sweep_crosses_focus_twice_and_reaches_the_other_side_between(self) -> None:
        assert focuser_offset(0.0) == 1.0
        assert focuser_offset(RAPID_SWEEP_S / 4) == pytest.approx(0.0, abs=1e-12)
        assert focuser_offset(RAPID_SWEEP_S / 2) == -1.0
        assert focuser_offset(3 * RAPID_SWEEP_S / 4) == pytest.approx(0.0, abs=1e-12)
        assert focuser_offset(RAPID_SWEEP_S) == pytest.approx(RAPID_AMPLITUDES[1])

    def test_each_sweep_is_narrower_than_the_one_before_and_then_the_run_starts_again(self) -> None:
        peaks = [
            max(abs(focuser_offset((k + f / 100) * RAPID_SWEEP_S)) for f in range(0, 100, 5))
            for k in range(len(RAPID_AMPLITUDES) + 1)
        ]
        assert peaks[: len(RAPID_AMPLITUDES)] == sorted(
            peaks[: len(RAPID_AMPLITUDES)], reverse=True
        )
        assert peaks[len(RAPID_AMPLITUDES)] == peaks[0]  # the person loses focus and starts again

    def test_the_width_is_the_width_in_focus_with_the_blur_added_in_quadrature(self) -> None:
        assert true_width_arcsec(RAPID_SWEEP_S / 4) == pytest.approx(RAPID_BEST_ARCSEC)
        assert true_width_arcsec(0.0) == pytest.approx(
            math.hypot(RAPID_BEST_ARCSEC, RAPID_BLUR_ARCSEC)
        )
        widths = [true_width_arcsec(t / 10) for t in range(0, 6 * int(RAPID_SWEEP_S * 10))]
        assert min(widths) >= RAPID_BEST_ARCSEC - 1e-9
        assert max(widths) < 12.0  # a star that the rapid mode is offered for stays in the aperture

    def test_the_width_in_focus_is_the_one_of_the_star_of_the_video(self) -> None:
        assert RAPID_BEST_ARCSEC == 3.0  # 2.99 arcseconds is the 0.6 pixel sigma of the demo video

    def test_a_reading_is_the_same_every_time_and_another_in_another_run(self) -> None:
        assert reading_width_arcsec(1, 17) == reading_width_arcsec(1, 17)
        assert reading_width_arcsec(1, 17) != reading_width_arcsec(2, 17)

    def test_the_readings_scatter_by_the_stated_amount(self) -> None:
        ratios = [
            reading_width_arcsec(1, n) / true_width_arcsec(n * 0.05)
            for n in range(1, RAPID_JOLT_EVERY - 1)
        ]
        assert statistics.mean(ratios) == pytest.approx(1.0, abs=0.01)
        assert statistics.pstdev(ratios) == pytest.approx(RAPID_SCATTER, rel=0.25)

    def test_a_tap_inflates_two_readings_in_a_row_and_never_before_the_first_ones(self) -> None:
        for n in (RAPID_JOLT_EVERY, RAPID_JOLT_EVERY + 1, 5 * RAPID_JOLT_EVERY):
            ratio = reading_width_arcsec(1, n) / true_width_arcsec(n * 0.05)
            assert ratio == pytest.approx(RAPID_JOLT_FACTOR, rel=0.2), n
        for n in (RAPID_JOLT_EVERY - 1, RAPID_JOLT_EVERY + 2):
            ratio = reading_width_arcsec(1, n) / true_width_arcsec(n * 0.05)
            assert ratio < 1.3, n
        assert RAPID_JOLT_AFTER < RAPID_JOLT_EVERY
        assert reading_width_arcsec(1, 0) / true_width_arcsec(0.0) < 1.3  # n = 0 is no tap


# --- A run ------------------------------------------------------------------------------------


class TestARun:
    def test_nothing_runs_before_begin(self) -> None:
        demo, _ = make()
        assert (demo.active, demo.ended, demo.view()) == (False, None, None)

    def test_a_run_makes_twenty_readings_a_second_of_the_clock(self) -> None:
        demo, clock = make()
        demo.begin(2000, 0)
        clock.advance(3.0)
        view = demo.view()
        assert view is not None
        assert (view.available, view.active, view.n_stars) == (True, True, 1)
        readings = view.readings
        assert readings is not None
        assert len(readings.index) == 60
        assert readings.index == list(range(1, 61))
        assert readings.reset is True
        assert readings.session == 1
        assert readings.t_utc_ms[1] - readings.t_utc_ms[0] == 50

    def test_a_reading_holds_four_frames_and_every_tenth_one_five(self) -> None:
        demo, clock = make()
        demo.begin(2000, 0)
        clock.advance(2.0)
        view = demo.view()
        assert view is not None
        assert view.readings is not None
        frames = view.readings.n_frames
        assert [n for i, n in zip(view.readings.index, frames, strict=True) if i % 10 == 0] == [
            5
        ] * 4
        assert set(frames) == {4, 5}

    def test_the_state_describes_the_stream(self) -> None:
        demo, clock = make()
        demo.begin(1500, 40)
        clock.advance(1.0)
        view = demo.view()
        assert view is not None
        assert (view.mode, view.exposure_us, view.gain) == ("bin1", 1500, 40)
        assert view.scale_arcsec_px == 1.91
        assert view.roi is not None
        assert (view.roi.x, view.roi.y, view.roi.width, view.roi.height) == (2008, 1347, 128, 128)
        assert view.since_utc is not None
        assert view.fwhm_arcsec is not None
        assert view.peak_fraction == pytest.approx(0.27 * 1500 / 2000, abs=1e-4)

    def test_the_curve_shows_in_the_readings(self) -> None:
        demo, clock = make()
        demo.begin(2000, 0)
        clock.advance(RAPID_SWEEP_S)  # one sweep of 28 s
        view = demo.view()
        assert view is not None
        assert view.readings is not None
        widths = view.readings.fwhm_arcsec
        assert len(widths) == 560  # 20 a second for 28 s
        assert max(widths[:20]) > 9.0  # the sweep starts far out of focus
        assert min(widths[130:150]) < 3.6  # and crosses focus after 7 seconds
        assert max(widths[270:290]) > 9.0  # the other side of focus, after 14 seconds

    def test_the_taps_are_flagged_by_the_rule_of_core(self) -> None:
        demo, clock = make()
        demo.begin(2000, 0)
        clock.advance(15.0)
        view = demo.view()
        assert view is not None
        readings = view.readings
        assert readings is not None
        tapped = {RAPID_JOLT_EVERY, RAPID_JOLT_EVERY + 1}
        flagged = {i for i, spike in zip(readings.index, readings.spike, strict=True) if spike}
        assert tapped <= flagged
        for index in flagged:  # whatever is flagged satisfies the rule on its preceding readings
            if index <= 3:
                continue
            position = readings.index.index(index)
            preceding = readings.fwhm_arcsec[max(0, position - 10) : position]
            assert readings.fwhm_arcsec[position] > SPIKE_FACTOR * statistics.median(preceding)
        assert view.spike is False  # the newest reading is not a tap

    def test_the_best_value_is_smoothed_and_never_a_tap(self) -> None:
        demo, clock = make()
        demo.begin(2000, 0)
        clock.advance(0.2)
        early = demo.view()
        assert early is not None
        assert early.best_fwhm_arcsec is None
        assert "best_fwhm_arcsec" in early.quality  # it needs ten readings
        clock.advance(RAPID_SWEEP_S)
        view = demo.view()
        assert view is not None
        assert view.readings is not None
        assert view.best_fwhm_arcsec is not None
        assert RAPID_BEST_ARCSEC - 0.2 < view.best_fwhm_arcsec < RAPID_BEST_ARCSEC + 0.6
        assert view.best_fwhm_arcsec > min(view.readings.fwhm_arcsec)  # not the lowest reading

    def test_the_history_keeps_the_last_600_readings(self) -> None:
        demo, clock = make()
        demo.begin(2000, 0)
        clock.advance(60.0)
        view = demo.view()
        assert view is not None
        assert view.readings is not None
        assert len(view.readings.index) == 600
        assert view.readings.index[-1] == 1200

    def test_a_clock_that_jumps_gives_the_readings_that_it_skipped(self) -> None:
        demo, clock = make()
        demo.begin(2000, 0)
        clock.advance(5.0)
        first = demo.view()
        clock.advance(5.0)
        second = demo.view()
        assert first is not None
        assert second is not None
        assert first.readings is not None
        assert second.readings is not None
        assert second.readings.index == list(range(1, 201))

    def test_a_new_exposure_changes_the_readings_that_follow(self) -> None:
        demo, clock = make()
        demo.begin(2000, 0)
        clock.advance(1.0)
        demo.configure(4000, 0)
        clock.advance(1.0)
        view = demo.view()
        assert view is not None
        assert view.readings is not None
        peaks = view.readings.peak_fraction
        assert set(peaks[:20]) == {0.27}
        assert set(peaks[20:]) == {0.54}
        assert (view.exposure_us, view.gain) == (4000, 0)
        assert view.readings.session == 1  # the same run

    def test_a_saturated_star_is_flagged_and_never_sets_the_best_value(self) -> None:
        demo, clock = make()
        demo.begin(8000, 0)
        clock.advance(5.0)
        view = demo.view()
        assert view is not None
        assert view.readings is not None
        assert all(view.readings.saturated)
        assert view.saturated is True
        assert view.best_fwhm_arcsec is None
        assert "saturated" in view.quality

    def test_the_end_keeps_the_reason_and_the_next_run_is_a_new_session(self) -> None:
        demo, clock = make()
        demo.begin(2000, 0)
        clock.advance(1.0)
        demo.end("you stopped rapid focus")
        assert (demo.active, demo.ended, demo.view()) == (False, "you stopped rapid focus", None)
        demo.end("a second end changes nothing")
        assert demo.ended == "you stopped rapid focus"
        clock.advance(10.0)
        demo.begin(2000, 0)
        assert (demo.active, ended_of(demo)) == (True, None)
        clock.advance(0.5)
        view = demo.view()
        assert view is not None
        assert view.readings is not None
        assert view.readings.session == 2
        assert view.readings.index == list(range(1, 11))  # the readings start again

    def test_a_reset_restarts_the_best_value_and_keeps_the_readings(self) -> None:
        demo, clock = make()
        demo.begin(2000, 0)
        clock.advance(8.0)
        before = demo.view()
        assert before is not None
        assert before.best_fwhm_arcsec is not None
        demo.reset_best()
        after = demo.view()
        assert after is not None
        assert after.best_fwhm_arcsec is None
        assert after.readings is not None
        assert before.readings is not None
        assert after.readings.index == before.readings.index

    def test_the_width_for_the_picture_follows_the_curve_without_the_scatter(self) -> None:
        demo, clock = make()
        demo.begin(2000, 0)
        assert demo.width_arcsec() == pytest.approx(true_width_arcsec(0.0))
        clock.advance(RAPID_SWEEP_S / 4)
        assert demo.width_arcsec() == pytest.approx(RAPID_BEST_ARCSEC)


# --- The offer --------------------------------------------------------------------------------


class TestTheOffer:
    @pytest.mark.parametrize("frames", [0, 1, 2, 3, 4])
    def test_the_first_frames_do_not_offer_the_mode_and_count_the_focus_values(
        self, frames: int
    ) -> None:
        view = offer_view(
            frames=frames, coarse_fwhm_arcsec=9.2, located_by="current solution", ended=None
        )
        assert (view.available, view.active) == (False, False)
        assert view.reason == (
            f"the coarse focus is not known: the quick solve has measured {frames} of the 5 "
            "focus values that it needs"
        )
        assert (view.located_by, view.coarse_fwhm_arcsec) == (None, None)
        assert view.max_fwhm_arcsec == RAPID_MAX_FWHM_ARCSEC == 12.0

    def test_the_sentence_is_the_one_of_core(self) -> None:
        assert offer_problem(3) == coarse_problem(CoarseFocus(None, None, 3), 3.82, 12.0)

    @pytest.mark.parametrize("located_by", ["current solution", "last solution"])
    def test_five_frames_offer_it_with_the_coarse_focus(self, located_by: RapidLocatedBy) -> None:
        view = offer_view(frames=5, coarse_fwhm_arcsec=9.2, located_by=located_by, ended=None)
        assert (view.available, view.reason, view.located_by) == (True, None, located_by)
        assert view.coarse_fwhm_arcsec == 9.2
        assert view.readings is None

    def test_the_reason_that_the_last_run_ended_shows_in_the_offer(self) -> None:
        view = offer_view(
            frames=20,
            coarse_fwhm_arcsec=9.2,
            located_by="current solution",
            ended="nobody used it for 2 min",
        )
        assert (view.available, view.ended_reason) == (True, "nobody used it for 2 min")

    def test_no_problem_after_five_frames(self) -> None:
        assert offer_problem(5) is None
        assert offer_problem(500) is None
