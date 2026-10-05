"""The fake core of the demo in the rapid focus mode: the offer, the run, the video, and the end.

The demo core follows its own clock, so the tests use a virtual clock and move it by hand.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TypeVar

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.scheduler.commands import (
    Pause,
    RejectReason,
    Resume,
    StartAlignment,
    StartRapidFocus,
    StopAlignment,
    StopRapidFocus,
)
from seeingmon.services.web.contract import AlignmentFrame, PolarisFrame, RapidFocusView
from seeingmon.services.web.demo import FRAME_PERIOD_S, PLATE_SCALE_ARCSEC_PX, DemoCore
from seeingmon.services.web.demo_activity import RAPID_TIMEOUT_S
from seeingmon.services.web.demo_rapid import RAPID_BEST_ARCSEC, RAPID_SWEEP_S

T = TypeVar("T")


async def collect(source: AsyncIterator[T], count: int, timeout_s: float = 10.0) -> list[T]:
    items: list[T] = []

    async def run() -> None:
        async for item in source:
            items.append(item)
            if len(items) == count:
                return

    await asyncio.wait_for(run(), timeout_s)
    return items


def video(core: DemoCore, count: int, timeout_s: float = 10.0) -> list[PolarisFrame]:
    return asyncio.run(collect(core.polaris_frames(), count, timeout_s))


def alignment(core: DemoCore, count: int, timeout_s: float = 10.0) -> list[AlignmentFrame]:
    return asyncio.run(collect(core.alignment_frames(), count, timeout_s))


def make(period_s: float = FRAME_PERIOD_S) -> tuple[DemoCore, VirtualClock]:
    clock = VirtualClock(1_800_000_000_000_000_000)
    core = DemoCore(clock, period_s=period_s, polaris_period_s=0.0)
    return core, clock


def aligning(period_s: float = FRAME_PERIOD_S) -> tuple[DemoCore, VirtualClock]:
    """A core whose scheduler aligns, with the mode on offer (five frames have gone by)."""
    core, clock = make(period_s)
    assert core.submit(StartAlignment()).accepted
    clock.advance(3.0)
    return core, clock


def rapid_of(core: DemoCore) -> RapidFocusView:
    view = core.alignment_state().rapid_focus
    assert view is not None
    return view


# --- The offer --------------------------------------------------------------------------------


class TestTheOffer:
    def test_the_first_frames_do_not_offer_the_mode_and_the_start_says_why(self) -> None:
        core, clock = make()
        assert core.submit(StartAlignment()).accepted
        clock.advance(1.0)  # two frames of half a second
        view = rapid_of(core)
        assert view.available is False
        assert view.reason is not None
        assert "measured 2 of the 5 focus values" in view.reason
        result = core.rapid_focus_start()
        assert (result.accepted, result.reason) == (False, RejectReason.NOT_AVAILABLE)
        assert result.message == view.reason
        assert core.submitted == [StartAlignment()]  # the refused start is no command

    def test_after_five_frames_the_mode_is_offered_with_the_coarse_focus(self) -> None:
        core, _ = aligning()
        view = rapid_of(core)
        assert (view.available, view.reason, view.active) == (True, None, False)
        assert view.located_by == "current solution"
        assert view.coarse_fwhm_arcsec is not None
        assert 8.0 < view.coarse_fwhm_arcsec < 10.5  # the focus values of the demo, in arcseconds
        assert view.max_fwhm_arcsec == 12.0
        assert view.readings is None

    def test_outside_the_alignment_the_start_is_not_aligning(self) -> None:
        core, _ = make()
        result = core.rapid_focus_start()
        assert (result.accepted, result.reason) == (False, RejectReason.NOT_ALIGNING)
        assert core.alignment_state().active is False

    def test_the_frames_of_the_alignment_carry_the_offer_and_count_the_focus_values(self) -> None:
        core, _ = make(period_s=0.01)
        assert core.submit(StartAlignment()).accepted
        frames = alignment(core, 6)
        offers = [frame.state.rapid_focus for frame in frames]
        assert all(offer is not None for offer in offers)
        flags = [offer.available for offer in offers if offer is not None]
        assert flags == [False, False, False, False, True, True]
        first = offers[0]
        assert first is not None
        assert first.reason is not None
        assert "measured 1 of the 5 focus values" in first.reason


# --- The run ----------------------------------------------------------------------------------


class TestTheRun:
    def test_a_start_runs_the_mode_and_the_state_carries_the_readings(self) -> None:
        core, clock = aligning()
        result = core.rapid_focus_start()
        assert (result.accepted, result.message) == (True, "rapid focus started")
        assert core.submitted[-1] == StartRapidFocus(*core.rapid_center)
        clock.advance(2.0)
        view = rapid_of(core)
        assert (view.available, view.active, view.n_stars) == (True, True, 1)
        assert view.readings is not None
        assert len(view.readings.index) == 40  # 20 a second for 2 seconds
        assert view.readings.reset is True
        assert (view.mode, view.exposure_us, view.gain) == ("bin1", 2000, 0)
        assert view.fwhm_arcsec is not None

    def test_the_status_says_that_the_mode_runs_and_what_comes_next(self) -> None:
        core, clock = aligning()
        core.rapid_focus_start()
        clock.advance(5.0)
        activity = core.status().scheduler.activity
        assert activity is not None
        assert (activity.state, activity.phase) == ("align", "rapid_focus")
        assert activity.label == "Rapid focus on Polaris"
        assert activity.next_label == "Back to the alignment view of the whole frame"
        assert activity.detail is not None
        assert "5 s" in activity.detail
        assert activity.ends_utc_ns is not None
        assert activity.since_utc_ns is not None
        assert activity.ends_utc_ns - activity.since_utc_ns == pytest.approx(
            (RAPID_TIMEOUT_S + 5.0) * 1e9, rel=1e-6
        )

    def test_the_alignment_stream_sends_nothing_while_the_mode_runs(self) -> None:
        core, _ = make(period_s=0.01)
        assert core.submit(StartAlignment()).accepted
        assert len(alignment(core, 5)) == 5
        assert core.rapid_focus_start().accepted
        with pytest.raises(asyncio.TimeoutError):
            alignment(core, 1, timeout_s=0.4)

    def test_a_second_start_keeps_the_run_and_a_new_setting_changes_the_stream(self) -> None:
        core, clock = aligning()
        core.rapid_focus_start()
        clock.advance(1.0)
        again = core.rapid_focus_start()
        assert again.message == "rapid focus already runs, so the idle timer restarted"
        before = rapid_of(core)
        assert before.readings is not None
        clock.advance(1.0)
        assert core.rapid_focus_start(exposure_us=4000, gain=10).accepted
        after = rapid_of(core)
        assert (after.exposure_us, after.gain) == (4000, 10)
        assert after.readings is not None
        assert after.readings.session == before.readings.session  # the same run
        assert after.readings.index[: len(before.readings.index)] == before.readings.index

    def test_a_start_without_settings_keeps_the_settings_that_run(self) -> None:
        core, _ = aligning()
        assert core.rapid_focus_start(exposure_us=3000, gain=20).accepted
        assert core.rapid_focus_start().accepted
        view = rapid_of(core)
        assert (view.exposure_us, view.gain) == (3000, 20)

    def test_the_curve_shows_in_the_readings_and_the_best_value_comes_after_ten(self) -> None:
        core, clock = aligning()
        core.rapid_focus_start()
        clock.advance(0.3)
        assert rapid_of(core).best_fwhm_arcsec is None
        clock.advance(RAPID_SWEEP_S)
        view = rapid_of(core)
        assert view.readings is not None
        assert max(view.readings.fwhm_arcsec) > 9.0
        assert view.best_fwhm_arcsec is not None
        assert RAPID_BEST_ARCSEC - 0.2 < view.best_fwhm_arcsec < RAPID_BEST_ARCSEC + 0.6

    def test_a_reset_of_the_focus_restarts_the_best_value_of_the_mode_too(self) -> None:
        core, clock = aligning()
        core.rapid_focus_start()
        clock.advance(10.0)
        assert rapid_of(core).best_fwhm_arcsec is not None
        core.alignment_reset_focus()
        assert rapid_of(core).best_fwhm_arcsec is None
        assert rapid_of(core).readings is not None


# --- The end ----------------------------------------------------------------------------------


class TestTheEnd:
    def test_a_stop_ends_the_mode_and_the_offer_says_why(self) -> None:
        core, clock = aligning()
        core.rapid_focus_start()
        clock.advance(2.0)
        stopped = core.submit(StopRapidFocus())
        assert (stopped.accepted, stopped.state) == (True, "align")
        view = rapid_of(core)
        assert (view.active, view.readings, view.ended_reason) == (
            False,
            None,
            "you stopped rapid focus",
        )
        assert view.available is True  # it can start again
        activity = core.status().scheduler.activity
        assert activity is not None
        assert activity.phase == "align"

    def test_the_alignment_stream_comes_back_after_a_stop(self) -> None:
        core, _ = make(period_s=0.01)
        assert core.submit(StartAlignment()).accepted
        alignment(core, 5)
        core.rapid_focus_start()
        core.submit(StopRapidFocus())
        frames = alignment(core, 1)
        offer = frames[0].state.rapid_focus
        assert offer is not None
        assert offer.ended_reason == "you stopped rapid focus"

    @pytest.mark.parametrize(
        ("ending", "reason"),
        [(StopAlignment(), "the alignment ended"), (Pause(), "you paused the scheduler")],
    )
    def test_the_end_of_the_alignment_ends_the_mode(
        self, ending: StopAlignment | Pause, reason: str
    ) -> None:
        core, _ = aligning()
        core.rapid_focus_start()
        assert core.submit(ending).accepted
        assert core.rapid_running is False
        assert core.alignment_state().active is False
        core.submit(Resume())  # a paused scheduler first goes back to work
        core.submit(StartAlignment())  # an alignment of its own after the pause or the stop
        view = core.alignment_state().rapid_focus
        assert view is not None
        assert (view.active, view.ended_reason) == (False, reason)

    def test_the_mode_ends_after_the_idle_time_when_nobody_uses_it(self) -> None:
        core, clock = aligning()
        core.rapid_focus_start()
        clock.advance(RAPID_TIMEOUT_S - 1.0)
        assert rapid_of(core).active is True
        clock.advance(2.0)
        view = rapid_of(core)
        assert (view.active, view.ended_reason) == (False, "nobody used it for 2 min")
        assert core.status().scheduler.activity.phase == "align"  # type: ignore[union-attr]

    def test_a_start_restarts_the_idle_timer(self) -> None:
        core, clock = aligning()
        core.rapid_focus_start()
        clock.advance(100.0)
        assert core.rapid_focus_start().accepted
        clock.advance(100.0)  # 200 s after the first start, and 100 s after the second
        assert rapid_of(core).active is True

    def test_a_viewer_of_the_live_view_counts_as_use(self) -> None:
        core, clock = aligning()
        core.rapid_focus_start()
        core.streams_opened += 1  # the page keeps the stream of the live view open
        for _ in range(5):
            clock.advance(60.0)
            assert rapid_of(core).active is True
        core.streams_closed += 1
        clock.advance(RAPID_TIMEOUT_S + 1.0)
        assert rapid_of(core).active is False


# --- The video --------------------------------------------------------------------------------


class TestTheVideo:
    def test_the_video_runs_in_the_alignment_while_the_mode_runs_and_carries_the_readings(
        self,
    ) -> None:
        core, clock = aligning()
        core.rapid_focus_start()
        clock.advance(2.0)
        frames = video(core, 3)
        assert len(frames) == 3
        for frame in frames:
            state = frame.state
            assert state.rapid_focus is not None
            assert (state.rapid_focus.active, state.rapid_focus.n_stars) == (True, 1)
            assert state.rapid_focus.readings is not None
            assert len(state.rapid_focus.readings.index) >= 40
            assert state.live_seeing is None
            assert (
                state.quality["live_seeing"] == "the rapid focus mode makes no rolling seeing value"
            )
            assert (state.mode, state.exposure_us, state.gain) == ("bin1", 2000, 0)
            assert state.star.found is True

    def test_the_video_is_quiet_in_the_alignment_without_the_mode(self) -> None:
        core, _ = aligning()
        with pytest.raises(asyncio.TimeoutError):
            video(core, 1, timeout_s=0.4)

    def test_the_star_of_the_video_follows_the_curve(self) -> None:
        core, clock = aligning()
        core.rapid_focus_start()
        far = video(core, 4)
        clock.advance(RAPID_SWEEP_S / 4)  # the focuser crosses focus
        near = video(core, 4)
        widths_far = [f.state.star.fwhm_arcsec for f in far if f.state.star.fwhm_arcsec]
        widths_near = [f.state.star.fwhm_arcsec for f in near if f.state.star.fwhm_arcsec]
        assert widths_far
        assert widths_near
        assert min(widths_far) > 1.4 * max(widths_near)
        peaks_far = [f.state.star.peak_fraction or 0.0 for f in far]
        peaks_near = [f.state.star.peak_fraction or 0.0 for f in near]
        assert min(peaks_near) > 4 * max(peaks_far)  # the same light in fewer pixels

    def test_a_long_exposure_saturates_the_star_and_the_readings_say_so(self) -> None:
        core, clock = aligning()
        assert core.rapid_focus_start(exposure_us=16000).accepted
        clock.advance(RAPID_SWEEP_S / 4)  # in focus, where the star is brightest
        frame = video(core, 2)[-1]
        view = frame.state.rapid_focus
        assert view is not None
        assert view.saturated is True
        assert "saturated" in view.quality
        assert view.readings is not None
        assert view.readings.saturated[-1] is True
        assert frame.state.star.peak_fraction == pytest.approx(1.0, abs=0.01)
        assert frame.state.exposure_us == 16000

    def test_the_default_exposure_does_not_saturate(self) -> None:
        core, clock = aligning()
        core.rapid_focus_start()
        clock.advance(RAPID_SWEEP_S / 4)
        view = video(core, 2)[-1].state.rapid_focus
        assert view is not None
        assert view.saturated is False
        assert view.peak_fraction is not None
        assert 0.2 < view.peak_fraction < 0.4

    def test_the_video_of_the_normal_fast_stream_has_no_rapid_view(self) -> None:
        core, _ = make()
        frame = video(core, 1)[0]
        assert frame.state.rapid_focus is None
        assert PLATE_SCALE_ARCSEC_PX == 3.82  # the plate scale of the alignment frames
