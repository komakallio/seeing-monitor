"""The alignment state: the target, the offset, the roll, and the reasons for what is missing."""

from __future__ import annotations

import json
import math

import pytest

from seeingmon.clock import NS_PER_S
from seeingmon.services.core.alignment.solve import QuickSolution
from seeingmon.services.core.alignment.state import (
    FrameSummary,
    Target,
    build_state,
    resolve_target,
    wrap_degrees,
)
from seeingmon.services.core.settings import AlignmentSettings
from seeingmon.services.web.contract import (
    MAX_STATE_BYTES,
    HistogramView,
    SaturationView,
    SkyView,
    decode_alignment_state,
    pack_frame,
)
from seeingmon.survey.geometry import ARCSEC_PER_RAD
from seeingmon.survey.skyview import build_sky_view
from seeingmon.survey.tracker import PointingTracker
from seeingmon.survey.wcs_fit import CameraAttitude, pixel_center
from tests.services.web.helpers import tiny_jpeg
from tests.survey import synth

T0 = 1_800_000_000 * NS_PER_S
SCALE = 3.82
COLATITUDE = 0.6265  # degrees: about the colatitude of Polaris in 2026


def camera(distance_deg: float = 0.9, roll_deg: float = -65.0) -> CameraAttitude:
    """A camera whose pole lies `distance_deg` from the center of a 4144 x 2822 frame."""
    return CameraAttitude(
        rotation=synth.make_attitude(distance_deg, 40.0, roll_deg),
        scale_rad_px=SCALE / ARCSEC_PER_RAD,
        parity=1,
        center_px=pixel_center(4144, 2822),
    )


def frame_summary(**changes: object) -> FrameSummary:
    fields: dict[str, object] = {
        "seq": 7,
        "t_utc_ns": T0,
        "width_px": 4144,
        "height_px": 2822,
        "mode": "bin2",
        "exposure_s": 0.5,
        "gain": 120,
        "plate_scale_arcsec_px": SCALE,
        "histogram": HistogramView(counts=[5, 4, 3, 2, 1], min_dn=0.0, max_dn=65532.0),
        "saturation": SaturationView(fraction=0.0, warning=False),
    }
    fields.update(changes)
    return FrameSummary(**fields)  # type: ignore[arg-type]


def solution(**changes: object) -> QuickSolution:
    fields: dict[str, object] = {
        "t_utc_ns": T0,
        "seq": 7,
        "solved": True,
        "x_px": 2075.0,
        "y_px": 1400.0,
        "roll_deg": 12.0,
        "n_matched": 80,
        "rms_arcsec": 0.8,
        "scale_arcsec_px": SCALE,
        "solver": "tracker",
        "n_detected": 120,
        "focus_fwhm_px": 2.4,
        "n_focus_stars": 40,
        "attitude": camera(),
        "polaris_colatitude_deg": COLATITUDE,
    }
    fields.update(changes)
    return QuickSolution(**fields)  # type: ignore[arg-type]


SETTINGS = AlignmentSettings(target_x_px=2072.0, target_y_px=1411.0, target_roll_deg=10.0)
TARGET = Target(2072.0, 1411.0, 10.0)


class TestWrapping:
    @pytest.mark.parametrize(
        ("angle", "expected"),
        [
            (0.0, 0.0),
            (190.0, -170.0),
            (-190.0, 170.0),
            (180.0, 180.0),
            (-180.0, 180.0),
            (360.0, 0.0),
        ],
    )
    def test_the_angle_lands_in_the_half_open_range(self, angle: float, expected: float) -> None:
        assert wrap_degrees(angle) == pytest.approx(expected)


class TestOffset:
    def test_the_offset_is_the_solved_position_minus_the_target(self) -> None:
        state = build_state(frame_summary(), solution(), TARGET, SETTINGS, best_fwhm_px=2.1)
        assert state.active is True
        assert state.t_utc is not None
        assert state.t_utc.endswith("Z")
        assert state.target is not None
        assert (state.target.x_px, state.target.y_px, state.target.roll_deg) == (
            2072.0,
            1411.0,
            10.0,
        )
        assert state.solved is not None
        assert (state.solved.x_px, state.solved.y_px, state.solved.n_matched) == (
            2075.0,
            1400.0,
            80,
        )
        offset = state.offset
        assert offset is not None
        assert (offset.dx_px, offset.dy_px) == (3.0, -11.0)
        assert offset.distance_px == pytest.approx((3.0**2 + 11.0**2) ** 0.5)
        assert offset.dx_arcsec == pytest.approx(3.0 * SCALE)
        assert offset.dy_arcsec == pytest.approx(-11.0 * SCALE)
        assert offset.distance_arcsec == pytest.approx(offset.distance_px * SCALE)
        assert offset.roll_deg == pytest.approx(2.0)
        assert state.quality == {}

    def test_the_roll_offset_wraps_across_the_discontinuity(self) -> None:
        state = build_state(
            frame_summary(),
            solution(roll_deg=-179.0),
            Target(2072.0, 1411.0, 179.0),
            SETTINGS,
        )
        assert state.offset is not None
        assert state.offset.roll_deg == pytest.approx(2.0)

    def test_an_undefined_roll_leaves_the_roll_offset_out(self) -> None:
        state = build_state(frame_summary(), solution(roll_deg=None), TARGET, SETTINGS)
        assert state.offset is not None
        assert state.offset.roll_deg is None

    def test_a_target_without_a_roll_gives_no_roll_offset(self) -> None:
        state = build_state(frame_summary(), solution(), Target(2072.0, 1411.0), SETTINGS)
        assert state.offset is not None
        assert state.offset.roll_deg is None

    def test_the_plate_scale_of_the_solution_stands_in_for_a_missing_mode_scale(self) -> None:
        state = build_state(
            frame_summary(plate_scale_arcsec_px=None),
            solution(scale_arcsec_px=4.0),
            TARGET,
            SETTINGS,
        )
        assert state.offset is not None
        assert state.offset.dx_arcsec == pytest.approx(12.0)


class TestMissingParts:
    def test_no_solve_yet_says_so_and_leaves_the_offset_out(self) -> None:
        state = build_state(frame_summary(), None, TARGET, SETTINGS)
        assert (state.solved, state.offset, state.focus) == (None, None, None)
        assert state.quality["solved"] == "no solve has finished yet"
        assert state.quality["offset"] == "the offset needs a current solution"
        assert state.quality["focus"]

    def test_an_unsolved_frame_gives_its_note(self) -> None:
        failed = solution(solved=False, x_px=None, y_px=None, note="only 3 stars for a solver")
        state = build_state(frame_summary(), failed, TARGET, SETTINGS)
        assert state.solved is None
        assert state.quality["solved"] == "only 3 stars for a solver"

    def test_a_stale_solution_is_no_longer_current(self) -> None:
        old = solution(t_utc_ns=T0 - 30 * NS_PER_S)
        state = build_state(frame_summary(), old, TARGET, SETTINGS)
        assert state.solved is None
        assert "30 s old" in state.quality["solved"]

    def test_a_solution_a_few_seconds_old_still_counts_and_reports_its_age(self) -> None:
        state = build_state(frame_summary(), solution(t_utc_ns=T0 - 3 * NS_PER_S), TARGET, SETTINGS)
        assert state.solved is not None
        assert state.solved.age_s == pytest.approx(3.0)
        assert state.offset is not None

    def test_a_solution_from_a_later_frame_has_age_zero(self) -> None:
        state = build_state(frame_summary(), solution(t_utc_ns=T0 + 2 * NS_PER_S), TARGET, SETTINGS)
        assert state.solved is not None
        assert state.solved.age_s == 0.0

    def test_without_a_target_the_state_has_no_target_and_no_offset(self) -> None:
        state = build_state(frame_summary(), solution(), None, AlignmentSettings())
        assert (state.target, state.offset) == (None, None)
        assert state.solved is not None
        assert "target" in state.quality

    def test_no_focus_stars_leave_the_focus_out(self) -> None:
        state = build_state(frame_summary(), solution(focus_fwhm_px=None), TARGET, SETTINGS)
        assert state.focus is None
        assert state.quality["focus"] == "no unsaturated stars to measure"

    def test_the_focus_carries_the_best_value_of_the_session(self) -> None:
        state = build_state(frame_summary(), solution(), TARGET, SETTINGS, best_fwhm_px=2.0)
        assert state.focus is not None
        assert (state.focus.fwhm_px, state.focus.best_fwhm_px, state.focus.n_stars) == (
            2.4,
            2.0,
            40,
        )

    def test_the_histogram_and_the_saturation_pass_through(self) -> None:
        saturated = SaturationView(fraction=0.01, warning=True)
        state = build_state(frame_summary(saturation=saturated), None, None, AlignmentSettings())
        assert state.histogram is not None
        assert state.histogram.counts == [5, 4, 3, 2, 1]
        assert state.saturation == saturated

    def test_a_frame_without_a_histogram_says_so(self) -> None:
        state = build_state(frame_summary(histogram=None), None, None, AlignmentSettings())
        assert state.histogram is None
        assert "histogram" in state.quality

    def test_the_state_survives_the_json_of_the_contract(self) -> None:
        from seeingmon.services.web.contract import decode_alignment_state

        state = build_state(frame_summary(), solution(), TARGET, SETTINGS, best_fwhm_px=2.1)
        again = decode_alignment_state(state.model_dump(mode="json"))
        assert again == state


class TestTarget:
    def test_the_configured_target_wins(self) -> None:
        tracker = PointingTracker(synth.reference_profile())
        assert resolve_target(SETTINGS, tracker, T0, "bin2") == TARGET

    def test_without_a_target_or_a_reference_there_is_none(self) -> None:
        tracker = PointingTracker(synth.reference_profile())
        assert resolve_target(AlignmentSettings(), tracker, T0, "bin2") is None
        assert resolve_target(AlignmentSettings(), None, T0, "bin2") is None


class TestSky:
    """The sky view follows the freshness rule of the solved position and ignores the target."""

    def test_a_current_solution_gives_the_sky_view_of_its_attitude(self) -> None:
        state = build_state(frame_summary(), solution(), TARGET, SETTINGS)
        expected = SkyView.from_geometry(build_sky_view(camera(), 4144, 2822, COLATITUDE))
        assert state.sky == expected
        assert "sky" not in state.quality
        assert state.sky is not None
        assert state.sky.pole.in_front
        assert state.sky.pole.roll_deg == pytest.approx(-65.0, abs=0.01)
        assert state.sky.polaris_colatitude_deg == COLATITUDE
        assert state.sky.orbit is not None
        assert state.sky.orbit.fits

    def test_a_new_install_without_a_target_still_has_the_pole_and_the_solved_position(
        self,
    ) -> None:
        state = build_state(frame_summary(), solution(), None, AlignmentSettings())
        assert state.target is None
        assert state.offset is None
        assert state.solved is not None
        assert state.sky is not None
        assert state.sky.pole.in_front

    def test_the_view_measures_from_the_center_of_the_frame_that_it_describes(self) -> None:
        state = build_state(
            frame_summary(width_px=3000, height_px=2000), solution(), None, SETTINGS
        )
        assert state.sky is not None
        pole = camera().pole_pixel()
        assert pole is not None
        assert state.sky.pole.dx_px == pytest.approx(pole[0] - 1499.5, abs=0.01)
        assert state.sky.pole.dy_px == pytest.approx(pole[1] - 999.5, abs=0.01)

    def test_a_colatitude_that_is_not_known_leaves_the_orbit_out(self) -> None:
        state = build_state(
            frame_summary(), solution(polaris_colatitude_deg=None), TARGET, SETTINGS
        )
        assert state.sky is not None
        assert state.sky.orbit is None
        assert state.sky.polaris_colatitude_deg is None
        assert state.sky.pole.in_front

    def test_without_a_solve_the_sky_is_null_and_the_quality_says_why(self) -> None:
        state = build_state(frame_summary(), None, TARGET, SETTINGS)
        assert state.sky is None
        assert state.quality["sky"] == "no solve has finished yet"

    def test_an_unsolved_frame_gives_its_note_for_the_sky_too(self) -> None:
        failed = solution(solved=False, x_px=None, y_px=None, attitude=None, note="only 3 stars")
        state = build_state(frame_summary(), failed, TARGET, SETTINGS)
        assert state.sky is None
        assert state.quality["sky"] == "only 3 stars"

    def test_a_stale_solution_leaves_the_sky_out_like_the_solved_position(self) -> None:
        state = build_state(
            frame_summary(), solution(t_utc_ns=T0 - 30 * NS_PER_S), TARGET, SETTINGS
        )
        assert state.solved is None
        assert state.sky is None
        assert state.quality["sky"] == state.quality["solved"]
        assert "30 s old" in state.quality["sky"]

    def test_the_limit_of_the_age_is_the_same_for_both(self) -> None:
        just_in = solution(t_utc_ns=T0 - round(9.9 * NS_PER_S))
        just_out = solution(t_utc_ns=T0 - round(10.1 * NS_PER_S))
        assert build_state(frame_summary(), just_in, None, AlignmentSettings()).sky is not None
        assert build_state(frame_summary(), just_out, None, AlignmentSettings()).sky is None

    def test_a_solution_without_an_attitude_keeps_the_position_and_says_why(self) -> None:
        state = build_state(frame_summary(), solution(attitude=None), TARGET, SETTINGS)
        assert state.solved is not None
        assert state.sky is None
        assert state.quality["sky"] == "the solution carries no camera attitude"

    def test_a_pole_behind_the_camera_is_a_normal_state(self) -> None:
        state = build_state(frame_summary(), solution(attitude=camera(120.0)), None, SETTINGS)
        assert state.sky is not None
        assert not state.sky.pole.in_front
        assert state.sky.orbit is not None
        assert not state.sky.orbit.fits

    def test_the_sky_survives_the_json_of_the_contract(self) -> None:
        state = build_state(frame_summary(), solution(), TARGET, SETTINGS)
        assert decode_alignment_state(json.loads(state.model_dump_json())) == state

    def test_the_state_travels_with_every_frame_and_stays_far_below_the_limit(self) -> None:
        state = build_state(frame_summary(), solution(), TARGET, SETTINGS, best_fwhm_px=2.1)
        assert state.sky is not None
        assert len(state.sky.model_dump_json()) < 700
        payload = pack_frame(state, tiny_jpeg())
        assert len(payload) < MAX_STATE_BYTES // 16
        assert math.isfinite(state.sky.pole.distance_arcmin or 0.0)
