"""The alignment helper and the rapid focus mode: the offer in the state, and the start command.

The helper judges the offer from what it holds (the focus history, the solutions, the quick solve),
puts it into every state, and makes the command that starts the mode with the center of the ROI, so
that the page sends no position. While the mode runs, the state carries the readings instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.fastpath import FastPathConfig, create_fast_analyzer
from seeingmon.frames import Frame
from seeingmon.profile import load_profile
from seeingmon.services.core.alignment.helper import AlignmentHelper
from seeingmon.services.core.alignment.rapid import RapidFocusHelper
from seeingmon.services.core.alignment.solve import QuickSolution
from seeingmon.services.core.settings import AlignmentSettings
from seeingmon.services.web.contract import MAX_STATE_BYTES, AlignmentState, unpack_frame
from tests.fastpath.helpers import make_frame
from tests.services.core.test_rapid_focus import PERIOD_S, ROI, T0_NS, star_pool
from tests.survey.synth import make_attitude

from .rig import sky_frame

PROFILE = load_profile("asi294mm-gs250")
SETTINGS = AlignmentSettings(histogram_bins=16, min_interval_s=0.0)
T0 = 1_800_000_000 * NS_PER_S


def solution(seq: int = 1, **changes: Any) -> QuickSolution:
    fields: dict[str, Any] = {
        "t_utc_ns": T0,
        "seq": seq,
        "solved": True,
        "x_px": 303.0,
        "y_px": 196.0,
        "roll_deg": 1.5,
        "n_matched": 60,
        "rms_arcsec": 0.7,
        "scale_arcsec_px": 3.82,
        "solver": "tracker",
        "n_detected": 90,
        "focus_fwhm_px": 2.4,
        "n_focus_stars": 30,
    }
    fields.update(changes)
    return QuickSolution(**fields)


class StubSolver:
    def __init__(self) -> None:
        self.result = solution()

    def solve(self, frame: Frame) -> QuickSolution:
        return self.result


class Active:
    def __init__(self, value: bool = True) -> None:
        self.value = value

    def __call__(self) -> bool:
        return self.value


class Rig:
    """An alignment helper with a rapid focus helper, on a virtual clock."""

    def __init__(self, *, with_rapid: bool = True, settings: AlignmentSettings = SETTINGS) -> None:
        self.clock = VirtualClock(T0)
        self.active = Active()
        self.solver = StubSolver()
        self.rapid = (
            RapidFocusHelper(
                profile=PROFILE,
                clock=self.clock,
                kernel_setup=create_fast_analyzer(PROFILE, FastPathConfig(), "t").kernel_setup,
            )
            if with_rapid
            else None
        )
        self.helper = AlignmentHelper(
            settings=settings,
            profile=PROFILE,
            clock=self.clock,
            is_active=self.active,
            solver=self.solver,
            rapid=self.rapid,
        )

    def solve(self, count: int, **changes: Any) -> None:
        """Run `count` quick solves whose solutions carry `changes`, and show the last frame."""
        for seq in range(1, count + 1):
            self.solver.result = solution(seq, **changes)
            self.helper.solve_frame(sky_frame(seq))
        self.helper.process_frame(sky_frame(count))

    def state(self) -> AlignmentState:
        return self.helper.state()


@pytest.fixture
def rig() -> Iterator[Rig]:
    made = Rig()
    yield made
    made.helper.stop()


class TestTheOfferInTheState:
    def test_a_good_focus_and_a_current_solution_offer_the_mode(self, rig: Rig) -> None:
        rig.solve(6)
        view = rig.state().rapid_focus
        assert view is not None
        assert (view.available, view.reason, view.active) == (True, None, False)
        assert view.located_by == "current solution"
        assert view.coarse_fwhm_arcsec == pytest.approx(2.4 * 3.82, abs=0.01)
        assert view.max_fwhm_arcsec == 12.0
        assert view.readings is None  # nothing runs

    def test_the_first_focus_values_do_not_offer_it_yet_and_say_why(self, rig: Rig) -> None:
        rig.solve(3)
        view = rig.state().rapid_focus
        assert view is not None
        assert view.available is False
        assert view.reason is not None
        assert "measured 3 of the 5 focus values" in view.reason

    def test_wide_stars_say_how_wide(self, rig: Rig) -> None:
        rig.solve(6, focus_fwhm_px=5.5)
        view = rig.state().rapid_focus
        assert view is not None
        assert view.available is False
        assert (
            view.reason == "the stars are too wide for the rapid mode: 21 arcsec, the limit is 12"
        )

    def test_the_limit_comes_from_the_settings(self) -> None:
        rig = Rig(settings=SETTINGS.model_copy(update={"rapid_focus_max_fwhm_arcsec": 25.0}))
        try:
            rig.solve(6, focus_fwhm_px=5.5)
            view = rig.state().rapid_focus
            assert view is not None
            assert view.available is True
        finally:
            rig.helper.stop()

    def test_a_failed_solve_keeps_the_offer_through_the_last_good_solution(self, rig: Rig) -> None:
        """The last solution has the attitude of the camera, and Polaris turns with the Earth."""
        attitude = _camera()
        rig.solve(6, attitude=attitude, polaris_colatitude_deg=0.6265)
        rig.solver.result = solution(
            7, solved=False, x_px=None, y_px=None, note="too few stars", focus_fwhm_px=2.4
        )
        rig.helper.solve_frame(sky_frame(7))
        rig.clock.advance(30.0)
        rig.helper.process_frame(sky_frame(7, t_utc_ns=T0 + 30 * NS_PER_S))
        state = rig.state()
        assert state.solved is None  # the solution is not current
        view = state.rapid_focus
        assert view is not None
        assert (view.available, view.located_by) == (True, "last solution")

    def test_without_a_solution_the_brightest_star_offers_it(self, rig: Rig) -> None:
        rig.solve(
            6,
            solved=False,
            x_px=None,
            y_px=None,
            note="no star field",
            brightest_x_px=320.0,
            brightest_y_px=240.0,
            brightest_ratio=9.0,
        )
        view = rig.state().rapid_focus
        assert view is not None
        assert (view.available, view.located_by) == (True, "brightest star")

    def test_without_any_place_for_polaris_the_reason_says_so(self, rig: Rig) -> None:
        rig.solve(6, solved=False, x_px=None, y_px=None, note="no star field")
        view = rig.state().rapid_focus
        assert view is not None
        assert view.available is False
        assert view.reason is not None
        assert view.reason.startswith("Polaris is not located: no solve has found the star field")

    def test_the_state_of_the_live_view_carries_it_too(self, rig: Rig) -> None:
        rig.solve(6)
        payload = rig.helper.process_frame(sky_frame(7))
        state = unpack_frame(payload).state
        assert state.rapid_focus is not None
        assert state.rapid_focus.available is True

    def test_before_the_first_frame_there_is_no_view_of_a_mode_that_does_not_run(
        self, rig: Rig
    ) -> None:
        state = rig.state()
        assert state.rapid_focus is None
        assert state.quality["frame"] == "no frame has arrived yet"

    def test_outside_the_alignment_there_is_no_view(self, rig: Rig) -> None:
        rig.solve(6)
        rig.active.value = False
        assert rig.state() == AlignmentState(active=False)

    def test_a_helper_without_the_mode_has_no_view_and_no_command(self) -> None:
        rig = Rig(with_rapid=False)
        try:
            rig.solve(6)
            assert rig.state().rapid_focus is None
            command, message = rig.helper.rapid_start_command()
            assert command is None
            assert "no rapid focus helper" in message
        finally:
            rig.helper.stop()


class TestTheStartCommand:
    def test_the_command_has_the_center_in_pixels_of_the_fast_readout_mode(self, rig: Rig) -> None:
        rig.solve(6)
        command, message = rig.helper.rapid_start_command()
        assert message == ""
        assert command is not None
        # The solution puts Polaris at bin2 pixel (303, 196), which is bin1 pixel (606.5, 392.5).
        assert (command.center_x_px, command.center_y_px) == pytest.approx((606.5, 392.5))
        assert (command.exposure_us, command.gain) == (None, None)

    def test_the_settings_of_the_caller_go_into_the_command(self, rig: Rig) -> None:
        rig.solve(6)
        command, _ = rig.helper.rapid_start_command(exposure_us=1000, gain=30)
        assert command is not None
        assert (command.exposure_us, command.gain) == (1000, 30)

    def test_the_center_follows_the_solution_that_is_current(self, rig: Rig) -> None:
        rig.solve(6, x_px=500.0, y_px=300.0)
        command, _ = rig.helper.rapid_start_command()
        assert command is not None
        assert (command.center_x_px, command.center_y_px) == pytest.approx((1000.5, 600.5))

    def test_a_mode_that_is_not_offered_gives_the_reason_and_no_command(self, rig: Rig) -> None:
        rig.solve(6, focus_fwhm_px=5.5)
        command, message = rig.helper.rapid_start_command()
        assert command is None
        assert message == "the stars are too wide for the rapid mode: 21 arcsec, the limit is 12"

    def test_the_command_and_the_view_agree_whatever_the_state(self, rig: Rig) -> None:
        for changes in (
            {},
            {"focus_fwhm_px": 5.5},
            {"x_px": 20.0},
            {"solved": False, "x_px": None},
        ):
            rig.helper._end_session()
            rig.solve(6, **changes)
            view = rig.state().rapid_focus
            command, message = rig.helper.rapid_start_command()
            assert view is not None
            assert (command is not None) == view.available, changes
            if not view.available:
                assert message == view.reason

    def test_outside_the_alignment_there_is_nothing_to_start(self, rig: Rig) -> None:
        rig.solve(6)
        rig.active.value = False
        command, message = rig.helper.rapid_start_command()
        assert command is None
        assert "the alignment does not run" in message

    def test_before_a_frame_there_is_nothing_to_start(self, rig: Rig) -> None:
        command, message = rig.helper.rapid_start_command()
        assert command is None
        assert message == "no frame has arrived yet"


class TestWhileTheModeRuns:
    def feed(self, rig: Rig, seconds: float = 3.0, sigma: float = 1.0) -> None:
        assert rig.rapid is not None
        pool = star_pool(sigma, 14_000.0, 64.3)
        count = round(seconds / PERIOD_S)
        for seq in range(count):
            frame = make_frame(
                pool[seq % len(pool)],
                seq=seq,
                t_ns=T0_NS + round(seq * PERIOD_S * NS_PER_S),
                roi=ROI,
            )
            rig.rapid.push(frame)

    def test_the_state_carries_the_readings_and_keeps_the_last_frame(self, rig: Rig) -> None:
        rig.solve(6)
        assert rig.rapid is not None
        rig.rapid.begin_session()
        self.feed(rig)
        state = rig.state()
        assert state.frame is not None  # the last frame of the normal view stays
        view = state.rapid_focus
        assert view is not None
        assert (view.available, view.active, view.reason) == (True, True, None)
        assert view.coarse_fwhm_arcsec is None  # the offer is not judged while it runs
        assert view.readings is not None
        assert len(view.readings.index) > 50
        assert view.fwhm_arcsec == view.readings.fwhm_arcsec[-1]
        assert view.n_stars == 1

    def test_the_state_of_the_live_view_stays_below_the_message_limit(self, rig: Rig) -> None:
        rig.solve(6)
        assert rig.rapid is not None
        rig.rapid.begin_session()
        self.feed(rig, seconds=32.0)
        view = rig.state().rapid_focus
        assert view is not None
        assert view.readings is not None
        assert len(view.readings.index) == 600
        payload = rig.helper.process_frame(sky_frame(9))
        state = unpack_frame(
            payload
        ).state  # the JPEG follows the state, and the limit is the state's
        assert state.rapid_focus is not None
        assert len(state.model_dump_json()) < MAX_STATE_BYTES

    def test_a_start_while_it_runs_keeps_the_star_where_the_last_frame_showed_it(
        self, rig: Rig
    ) -> None:
        rig.solve(6)
        assert rig.rapid is not None
        rig.rapid.begin_session()
        self.feed(rig, seconds=1.0)
        command, _ = rig.helper.rapid_start_command(exposure_us=1000)
        assert command is not None
        assert (command.center_x_px, command.center_y_px) == pytest.approx(
            (ROI.x + 64.3, ROI.y + 63.7), abs=0.2
        )
        assert command.exposure_us == 1000

    def test_after_the_end_the_state_offers_the_mode_again_and_says_why_it_ended(
        self, rig: Rig
    ) -> None:
        rig.solve(6)
        assert rig.rapid is not None
        rig.rapid.begin_session()
        self.feed(rig, seconds=1.0)
        rig.rapid.end_session("the star was not in the window for 450 frames")
        view = rig.state().rapid_focus
        assert view is not None
        assert (view.active, view.available) == (False, True)
        assert view.ended_reason == "the star was not in the window for 450 frames"
        assert view.readings is None

    def test_a_reset_of_the_focus_restarts_the_best_value_of_both(self, rig: Rig) -> None:
        rig.solve(6)
        assert rig.rapid is not None
        rig.rapid.begin_session()
        self.feed(rig, seconds=2.0)
        before = rig.state()
        assert before.focus is not None
        assert before.focus.best_fwhm_px is not None
        assert before.rapid_focus is not None
        assert before.rapid_focus.best_fwhm_arcsec is not None
        rig.helper.reset_focus()
        after = rig.state()
        assert after.rapid_focus is not None
        assert after.rapid_focus.best_fwhm_arcsec is None
        assert after.focus is not None
        assert after.focus.best_fwhm_px is None
        assert len(after.focus.history.index) == 6  # type: ignore[union-attr]
        assert after.rapid_focus.readings is not None
        assert len(after.rapid_focus.readings.index) > 30  # the history stays

    def test_the_end_of_the_alignment_ends_the_mode_in_the_helper_too(self, rig: Rig) -> None:
        rig.solve(6)
        assert rig.rapid is not None
        rig.rapid.begin_session()
        rig.active.value = False
        assert rig.state() == AlignmentState(active=False)
        assert rig.rapid.snapshot().active is False  # nothing is left over for the next alignment


def _camera() -> Any:
    from seeingmon.survey.geometry import ARCSEC_PER_RAD
    from seeingmon.survey.wcs_fit import CameraAttitude, pixel_center

    # A coarse scale (19.1 arcseconds per pixel) brings the orbit of Polaris, 0.63 degree wide,
    # to a radius of 118 pixels, so that Polaris falls inside this small frame.
    return CameraAttitude(
        rotation=make_attitude(0.05, 40.0, -65.0),
        scale_rad_px=19.1 / ARCSEC_PER_RAD,
        parity=1,
        center_px=pixel_center(640, 480),
    )
