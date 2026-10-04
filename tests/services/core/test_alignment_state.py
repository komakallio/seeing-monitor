"""The alignment state: the target, the offset, the roll, and the reasons for what is missing."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, utc_ns_to_iso
from seeingmon.scheduler.config import SiteConfig
from seeingmon.services.core.alignment.solve import QuickSolution
from seeingmon.services.core.alignment.state import (
    FrameSummary,
    Target,
    build_state,
    resolve_target,
    ring_from_solution,
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
from seeingmon.survey import apparent
from seeingmon.survey.apparent import earth_rotation_angle
from seeingmon.survey.geometry import ARCSEC_PER_RAD, nearest_rotation
from seeingmon.survey.pointing import PointingSolution, polaris_colatitude_deg
from seeingmon.survey.skyview import build_sky_view, reticle_geometry, zenith_vector
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
        expected = SkyView.from_geometry(
            build_sky_view(camera(), 4144, 2822, COLATITUDE, polaris_xy=(2075.0, 1400.0))
        )
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


# A synthetic round-number site. It is not a real station.
SITE = SiteConfig(latitude_deg=50.0, longitude_deg=10.0)


def altaz_camera(altitude_deg: float, azimuth_deg: float) -> CameraAttitude:
    """A camera that looks at an altitude and an azimuth of the site at the time `T0`."""
    up = zenith_vector(SITE.latitude_deg, SITE.longitude_deg, earth_rotation_angle(T0))
    pole = np.array([0.0, 0.0, 1.0])
    north = pole - up * float(pole @ up)
    north /= np.linalg.norm(north)
    east = np.cross(north, up)
    alt, az = math.radians(altitude_deg), math.radians(azimuth_deg)
    boresight = math.cos(alt) * (math.cos(az) * north + math.sin(az) * east) + math.sin(alt) * up
    down = -(up - boresight * float(up @ boresight))
    down /= np.linalg.norm(down)
    return CameraAttitude(
        rotation=nearest_rotation(np.stack([np.cross(down, boresight), down, boresight])),
        scale_rad_px=SCALE / ARCSEC_PER_RAD,
        parity=1,
        center_px=pixel_center(4144, 2822),
    )


class TestReticle:
    """The first layer is fixed in the picture, and it needs no solution."""

    def test_the_reticle_is_a_circle_around_the_center_with_the_radius_of_the_orbit(self) -> None:
        state = build_state(frame_summary(), None, None, AlignmentSettings())
        expected = reticle_geometry(4144, 2822, SCALE, polaris_colatitude_deg(T0))
        assert expected is not None
        assert state.reticle is not None
        assert (state.reticle.x_px, state.reticle.y_px) == (expected.x_px, expected.y_px)
        assert state.reticle.radius_px == expected.radius_px
        # The apparent colatitude of Polaris moves by 20 arcseconds in a year (aberration), and it
        # was about 0.617 degrees at the time of the test, which is 581.6 pixels at 3.82 arcsec/px.
        assert state.reticle.polaris_colatitude_deg == pytest.approx(0.617, abs=0.003)
        assert state.reticle.radius_px == pytest.approx(581.6, abs=1.0)
        assert (state.reticle.x_px, state.reticle.y_px) == (2071.5, 1410.5)  # the frame center

    def test_without_a_solution_the_state_has_the_reticle_and_no_sky(self) -> None:
        state = build_state(frame_summary(), None, None, AlignmentSettings())
        assert state.reticle is not None
        assert state.sky is None
        assert state.quality["sky"] == "no solve has finished yet"
        assert "reticle" not in state.quality

    def test_the_reticle_holds_still_for_five_different_poles(self) -> None:
        reticles = set()
        poles = set()
        for altitude, azimuth in ((0.0, 0.0), (0.4, 0.1), (-0.3, -0.2), (0.7, 0.3), (-0.1, 0.5)):
            camera = altaz_camera(SITE.latitude_deg + altitude, azimuth)
            state = build_state(
                frame_summary(),
                solution(attitude=camera),
                None,
                AlignmentSettings(),
                site=SITE,
            )
            assert state.reticle is not None
            assert state.sky is not None
            reticles.add(state.reticle.model_dump_json())
            poles.add(state.sky.pole.model_dump_json())
        assert len(reticles) == 1
        assert len(poles) == 5

    def test_the_reticle_follows_the_size_and_the_scale_of_the_frame_and_nothing_else(self) -> None:
        wide = build_state(frame_summary(), None, None, AlignmentSettings()).reticle
        small = build_state(
            frame_summary(width_px=2072, height_px=1411), None, None, AlignmentSettings()
        ).reticle
        coarse = build_state(
            frame_summary(plate_scale_arcsec_px=2 * SCALE), None, None, AlignmentSettings()
        ).reticle
        assert wide is not None
        assert small is not None
        assert coarse is not None
        assert (small.x_px, small.y_px) == (1035.5, 705.0)
        assert small.radius_px == wide.radius_px
        assert coarse.radius_px == pytest.approx(wide.radius_px / 2, abs=0.01)

    def test_the_configured_aim_moves_the_center_of_the_circle(self) -> None:
        settings = AlignmentSettings(aim_x_px=1800.0, aim_y_px=1200.0)
        state = build_state(frame_summary(), solution(), None, settings)
        assert state.reticle is not None
        assert (state.reticle.x_px, state.reticle.y_px) == (1800.0, 1200.0)
        assert state.sky is not None
        assert state.sky.aim is not None
        assert (state.sky.aim.x_px, state.sky.aim.y_px) == (1800.0, 1200.0)

    def test_the_target_does_not_move_the_circle(self) -> None:
        plain = build_state(frame_summary(), solution(), None, AlignmentSettings())
        targeted = build_state(frame_summary(), solution(), TARGET, SETTINGS)
        assert plain.reticle == targeted.reticle

    def test_the_scale_of_the_solution_stands_in_for_a_missing_scale_of_the_mode(self) -> None:
        state = build_state(
            frame_summary(plate_scale_arcsec_px=None),
            solution(scale_arcsec_px=SCALE),
            None,
            SETTINGS,
        )
        assert state.reticle is not None
        state = build_state(frame_summary(plate_scale_arcsec_px=None), None, None, SETTINGS)
        assert state.reticle is None
        assert state.quality["reticle"] == "the plate scale is not known"


class TestAimRingAndAdjustment:
    def test_the_state_carries_the_aim_the_ring_and_the_center_as_default_aim(self) -> None:
        state = build_state(frame_summary(), solution(), None, AlignmentSettings())
        assert state.sky is not None
        assert state.sky.aim is not None
        assert (state.sky.aim.x_px, state.sky.aim.y_px) == (2071.5, 1410.5)
        assert state.sky.aim_ring is not None
        pole = state.sky.pole
        assert pole.x_px is not None
        assert pole.y_px is not None
        # The solution puts Polaris at (2075, 1400): the ring is that pixel plus aim minus pole.
        assert state.sky.aim_ring.x_px == pytest.approx(2075.0 + 2071.5 - pole.x_px, abs=2e-3)
        assert state.sky.aim_ring.y_px == pytest.approx(1400.0 + 1410.5 - pole.y_px, abs=2e-3)

    def test_the_ring_lies_on_the_circle_of_the_reticle(self) -> None:
        for altitude, azimuth in ((0.2, 0.05), (-0.3, 0.1), (0.5, -0.2)):
            camera = altaz_camera(SITE.latitude_deg + altitude, azimuth)
            rho = math.radians(polaris_colatitude_deg(T0))
            x, y, _ = camera.project(np.array([math.sin(rho), 0.0, math.cos(rho)]))
            state = build_state(
                frame_summary(),
                solution(attitude=camera, x_px=float(x[0]), y_px=float(y[0])),
                None,
                AlignmentSettings(),
                site=SITE,
            )
            assert state.reticle is not None
            assert state.sky is not None
            assert state.sky.aim_ring is not None
            distance = math.hypot(
                state.sky.aim_ring.x_px - state.reticle.x_px,
                state.sky.aim_ring.y_px - state.reticle.y_px,
            )
            assert distance == pytest.approx(state.reticle.radius_px, abs=1.0)

    def test_with_a_site_the_state_says_how_to_move_in_altitude_and_azimuth(self) -> None:
        raised = altaz_camera(SITE.latitude_deg + 0.5, 0.0)  # 30 arcminutes too high
        state = build_state(frame_summary(), solution(attitude=raised), None, SETTINGS, site=SITE)
        assert state.sky is not None
        assert state.sky.altitude_arcmin == pytest.approx(-30.0, abs=0.1)  # lower it
        assert state.sky.azimuth_arcmin == pytest.approx(0.0, abs=0.1)
        assert state.sky.axes is not None
        lowered = altaz_camera(SITE.latitude_deg - 0.25, 0.0)  # 15 arcminutes too low
        state = build_state(frame_summary(), solution(attitude=lowered), None, SETTINGS, site=SITE)
        assert state.sky is not None
        assert state.sky.altitude_arcmin == pytest.approx(15.0, abs=0.1)  # raise it

    def test_an_azimuth_error_comes_out_with_the_sign_of_the_turn_to_make(self) -> None:
        east_of_the_pole = altaz_camera(SITE.latitude_deg, 0.2)  # 0.2 degrees of azimuth east
        state = build_state(
            frame_summary(), solution(attitude=east_of_the_pole), None, SETTINGS, site=SITE
        )
        assert state.sky is not None
        assert state.sky.azimuth_arcmin is not None
        assert state.sky.azimuth_arcmin < -5.0  # turn it west
        west_of_the_pole = altaz_camera(SITE.latitude_deg, -0.2)
        state = build_state(
            frame_summary(), solution(attitude=west_of_the_pole), None, SETTINGS, site=SITE
        )
        assert state.sky is not None
        assert state.sky.azimuth_arcmin is not None
        assert state.sky.azimuth_arcmin > 5.0  # turn it east

    def test_without_a_site_there_is_no_move_and_no_arrows(self) -> None:
        state = build_state(frame_summary(), solution(), None, SETTINGS)
        assert state.sky is not None
        assert state.sky.altitude_arcmin is None
        assert state.sky.azimuth_arcmin is None
        assert state.sky.axes is None
        assert state.sky.aim is not None

    def test_a_site_leaves_the_rest_of_the_state_alone(self) -> None:
        without = build_state(frame_summary(), solution(), TARGET, SETTINGS)
        with_site = build_state(frame_summary(), solution(), TARGET, SETTINGS, site=SITE)
        assert with_site.sky is not None
        assert without.sky is not None
        assert with_site.sky.pole == without.sky.pole
        assert with_site.reticle == without.reticle
        assert with_site.offset == without.offset
        assert with_site.quality == without.quality

    def test_the_state_round_trips_through_the_contract_with_everything_set(self) -> None:
        state = build_state(frame_summary(), solution(), TARGET, SETTINGS, site=SITE)
        assert state.reticle is not None
        assert state.sky is not None
        assert state.sky.altitude_arcmin is not None
        from seeingmon.services.web.contract import decode_alignment_state

        again = decode_alignment_state(json.loads(state.model_dump_json()))
        assert again == state
        assert len(pack_frame(state, tiny_jpeg())) < MAX_STATE_BYTES // 16


class TestTiming:
    """The state says which frame each part comes from, and how old the frame is."""

    def test_the_timing_names_the_frame_and_the_frame_of_the_solution(self) -> None:
        state = build_state(
            frame_summary(seq=9, received_ns=T0 + 100_000_000, preview_s=0.25),
            solution(seq=6, t_utc_ns=T0 - 2 * NS_PER_S),
            TARGET,
            SETTINGS,
            now_utc_ns=T0 + 400_000_000,
            solve_elapsed_s=1.2,
            solving=(8, 0.4),
        )
        timing = state.timing
        assert timing is not None
        assert (timing.frame_seq, timing.solution_frame_seq) == (9, 6)
        assert timing.frame_t_utc == state.t_utc
        assert timing.frame_age_s == pytest.approx(0.4)
        assert timing.receive_lag_s == pytest.approx(0.1)
        assert timing.preview_s == 0.25
        assert timing.solve_elapsed_s == 1.2
        assert (timing.solving_frame_seq, timing.solving_s) == (8, 0.4)
        assert state.solved is not None
        assert state.solved.age_s == pytest.approx(2.0)  # the age of the solution, as before

    def test_without_a_solve_the_state_has_the_frame_and_nothing_of_the_solution(self) -> None:
        state = build_state(frame_summary(), None, TARGET, SETTINGS, now_utc_ns=T0)
        timing = state.timing
        assert timing is not None
        assert timing.frame_seq == 7
        assert (timing.solution_frame_seq, timing.solve_elapsed_s) == (None, None)
        assert (timing.solving_frame_seq, timing.solving_s) == (None, None)

    def test_an_unsolved_frame_still_names_the_frame_that_the_solve_ran_on(self) -> None:
        failed = solution(seq=4, solved=False, x_px=None, y_px=None, note="too few stars")
        state = build_state(
            frame_summary(), failed, TARGET, SETTINGS, now_utc_ns=T0, solve_elapsed_s=0.9
        )
        assert state.timing is not None
        assert (state.timing.solution_frame_seq, state.timing.solve_elapsed_s) == (4, 0.9)
        assert state.solved is None

    def test_while_the_first_solve_runs_the_reason_says_so(self) -> None:
        state = build_state(frame_summary(), None, TARGET, SETTINGS, solving=(4, 3.4))
        reason = "the first solve is running (frame 4, 3 s so far)"
        assert state.quality["solved"] == reason
        assert state.quality["sky"] == reason

    def test_the_ages_are_unknown_without_a_clock_reading(self) -> None:
        timing = build_state(frame_summary(), None, None, AlignmentSettings()).timing
        assert timing is not None
        assert (timing.frame_age_s, timing.receive_lag_s) == (None, None)

    def test_a_frame_without_a_valid_time_has_no_ages(self) -> None:
        state = build_state(
            frame_summary(time_valid=False, received_ns=T0),
            None,
            None,
            AlignmentSettings(),
            now_utc_ns=T0 + NS_PER_S,
        )
        assert state.timing is not None
        assert (state.timing.frame_age_s, state.timing.receive_lag_s) == (None, None)

    def test_an_age_is_never_negative(self) -> None:
        state = build_state(
            frame_summary(received_ns=T0 - 5),
            None,
            None,
            AlignmentSettings(),
            now_utc_ns=T0 - 1000,
        )
        assert state.timing is not None
        assert (state.timing.frame_age_s, state.timing.receive_lag_s) == (0.0, 0.0)

    def test_the_timing_survives_the_json_of_the_contract_and_stays_small(self) -> None:
        state = build_state(
            frame_summary(received_ns=T0, preview_s=0.2),
            solution(),
            TARGET,
            SETTINGS,
            now_utc_ns=T0 + 1,
            solve_elapsed_s=0.5,
            solving=(8, 1.0),
        )
        assert state.timing is not None
        assert len(state.timing.model_dump_json()) < 400
        assert decode_alignment_state(json.loads(state.model_dump_json())) == state


def solved_at(
    t_utc_ns: int, attitude: CameraAttitude | None = None, **changes: object
) -> QuickSolution:
    """A solution whose Polaris pixel is where the camera model puts the real Polaris."""
    attitude = camera() if attitude is None else attitude
    epoch = apparent.epoch_from_utc_ns(t_utc_ns)
    x, y, _ = attitude.project(apparent.apparent_vectors_for(apparent.POLARIS, epoch))
    return solution(
        t_utc_ns=t_utc_ns,
        attitude=attitude,
        x_px=float(x[0]),
        y_px=float(y[0]),
        polaris_colatitude_deg=polaris_colatitude_deg(t_utc_ns),
        **changes,
    )


def reference_ring(
    last: QuickSolution, t_utc_ns: int, aim: tuple[float, float] = (2071.5, 1410.5)
) -> tuple[float, float]:
    """The ring that the survey code gives for the time: the Earth-fixed solution, then project."""
    assert last.attitude is not None
    fixed = PointingSolution.from_attitude(
        last.attitude,
        apparent.epoch_from_utc_ns(last.t_utc_ns),
        mode="bin2",
        width_px=4144,
        height_px=2822,
    )
    polaris = fixed.polaris_pixel(t_utc_ns)
    pole = fixed.attitude_at(t_utc_ns).pole_pixel()
    assert polaris is not None
    assert pole is not None
    return polaris[0] + aim[0] - pole[0], polaris[1] + aim[1] - pole[1]


class TestAimRingWithoutASolution:
    """The ring depends on the twist of the picture and the time, so it outlives a lost solution."""

    def test_a_current_solution_gives_the_ring_of_its_sky_view(self) -> None:
        current = solved_at(T0 - 2 * NS_PER_S, seq=4)
        state = build_state(frame_summary(), current, TARGET, SETTINGS)
        assert state.sky is not None
        assert state.sky.aim_ring is not None
        ring = state.aim_ring
        assert ring is not None
        assert (ring.x_px, ring.y_px) == (state.sky.aim_ring.x_px, state.sky.aim_ring.y_px)
        assert ring.source == "current frame"
        assert ring.age_s == pytest.approx(2.0)
        assert ring.solution_frame_seq == 4
        assert "aim_ring" not in state.quality

    def test_without_a_current_solution_the_last_one_gives_the_ring(self) -> None:
        last = solved_at(T0 - 12 * NS_PER_S, seq=3)
        failed = solution(
            seq=9, solved=False, x_px=None, y_px=None, attitude=None, note="too few stars"
        )
        state = build_state(frame_summary(), failed, TARGET, SETTINGS, last_good=last)
        assert state.solved is None
        assert state.sky is None  # the pole and the grid need the pointing of this frame
        assert state.quality["solved"] == "too few stars"
        ring = state.aim_ring
        assert ring is not None
        assert ring.source == "last solution"
        assert ring.age_s == pytest.approx(12.0)
        assert ring.solution_frame_seq == 3
        expected = reference_ring(last, T0)
        assert (ring.x_px, ring.y_px) == pytest.approx(expected, abs=2e-3)
        assert "aim_ring" not in state.quality

    def test_the_ring_from_the_last_solution_is_the_ring_of_the_same_solution_when_current(
        self,
    ) -> None:
        current = solved_at(T0)
        with_solution = build_state(frame_summary(), current, TARGET, SETTINGS)
        without = build_state(frame_summary(), None, TARGET, SETTINGS, last_good=current)
        assert with_solution.aim_ring is not None
        assert without.aim_ring is not None
        assert (without.aim_ring.x_px, without.aim_ring.y_px) == pytest.approx(
            (with_solution.aim_ring.x_px, with_solution.aim_ring.y_px), abs=2e-3
        )

    @pytest.mark.parametrize("hours", [0.25, 1.0, 6.0, 12.0, -3.0])
    def test_the_ring_turns_with_the_earth(self, hours: float) -> None:
        last = solved_at(T0)
        t_frame = T0 + round(hours * 3600 * NS_PER_S)
        state = build_state(frame_summary(t_utc_ns=t_frame), None, TARGET, SETTINGS, last_good=last)
        assert state.aim_ring is not None
        expected = reference_ring(last, t_frame)
        assert (state.aim_ring.x_px, state.aim_ring.y_px) == pytest.approx(expected, abs=2e-3)
        assert state.reticle is not None
        angle = []
        for time_ns in (T0, t_frame):
            ring = ring_from_solution(last, time_ns, 4144, 2822, AlignmentSettings())
            assert ring is not None
            angle.append(math.atan2(ring[1] - state.reticle.y_px, ring[0] - state.reticle.x_px))
        turned = math.degrees(angle[1] - angle[0])
        earth = math.degrees(earth_rotation_angle(t_frame) - earth_rotation_angle(T0))
        assert wrap_degrees(turned - earth) == pytest.approx(0.0, abs=0.01) or wrap_degrees(
            turned + earth
        ) == pytest.approx(0.0, abs=0.01)  # the sense depends on the parity of the picture

    def test_the_ring_stays_on_the_circle_of_the_reticle_through_a_day(self) -> None:
        last = solved_at(T0)
        for hours in range(0, 25, 3):
            t_frame = T0 + hours * 3600 * NS_PER_S
            state = build_state(
                frame_summary(t_utc_ns=t_frame), None, TARGET, SETTINGS, last_good=last
            )
            assert state.aim_ring is not None
            assert state.reticle is not None
            distance = math.hypot(
                state.aim_ring.x_px - state.reticle.x_px, state.aim_ring.y_px - state.reticle.y_px
            )
            assert distance == pytest.approx(state.reticle.radius_px, abs=1.0)

    def test_a_move_of_the_mount_leaves_the_ring_where_it_is(self) -> None:
        """Altitude and azimuth moves translate the picture, and the ring needs the twist only."""
        later = T0 + 1800 * NS_PER_S
        moves = (
            (0.0, 0.0, 0.0),
            (0.4, 0.1, 1.5),  # altitude and azimuth moves in degrees, and the shift that they allow
            (-0.3, -0.2, 2.5),
            (0.7, 0.3, 3.5),
            (-0.1, 0.5, 4.5),
            (1.0, 1.0, 9.0),
        )
        rings = []
        for altitude, azimuth, _ in moves:
            attitude = altaz_camera(SITE.latitude_deg + altitude, azimuth)
            state = build_state(
                frame_summary(t_utc_ns=later),
                None,
                None,
                AlignmentSettings(),
                last_good=solved_at(T0, attitude),
            )
            assert state.aim_ring is not None
            rings.append((state.aim_ring.x_px, state.aim_ring.y_px))
        for ring, (_, _, limit) in zip(rings[1:], moves[1:], strict=True):
            assert math.hypot(ring[0] - rings[0][0], ring[1] - rings[0][1]) < limit  # of 580 px

    def test_the_configured_aim_moves_the_ring_with_the_circle(self) -> None:
        last = solved_at(T0)
        centered = build_state(frame_summary(), None, None, AlignmentSettings(), last_good=last)
        moved = build_state(
            frame_summary(),
            None,
            None,
            AlignmentSettings(aim_x_px=1800.0, aim_y_px=1200.0),
            last_good=last,
        )
        assert centered.aim_ring is not None
        assert moved.aim_ring is not None
        assert moved.aim_ring.x_px - centered.aim_ring.x_px == pytest.approx(
            1800.0 - 2071.5, abs=2e-3
        )
        assert moved.aim_ring.y_px - centered.aim_ring.y_px == pytest.approx(
            1200.0 - 1410.5, abs=2e-3
        )

    def test_a_state_that_has_no_solution_at_all_has_no_ring_and_says_why(self) -> None:
        state = build_state(frame_summary(), None, TARGET, SETTINGS)
        assert state.aim_ring is None
        assert state.last_solution is None
        assert (
            state.quality["aim_ring"] == "no solve has found the star field in this alignment yet"
        )

    def test_a_last_solution_without_an_attitude_gives_no_ring(self) -> None:
        bare = solution(attitude=None)
        state = build_state(frame_summary(), None, TARGET, SETTINGS, last_good=bare)
        assert state.aim_ring is None
        assert state.quality["aim_ring"] == "the last solution carries no camera attitude"
        assert state.last_solution is not None  # its age is still worth showing

    def test_a_pole_behind_the_camera_gives_no_ring(self) -> None:
        behind = solved_at(T0, camera(120.0))
        state = build_state(frame_summary(), None, TARGET, SETTINGS, last_good=behind)
        assert state.aim_ring is None
        assert state.quality["aim_ring"] == "the pole of the last solution lies behind the camera"

    def test_the_polar_grid_and_the_pole_stay_out_of_a_state_without_a_current_solution(
        self,
    ) -> None:
        state = build_state(
            frame_summary(), None, TARGET, SETTINGS, last_good=solved_at(T0 - 60 * NS_PER_S)
        )
        assert (
            state.sky is None
        )  # the camera model, the pole, and the move need the current pointing
        assert state.quality["sky"] == "no solve has finished yet"
        assert state.aim_ring is not None
        assert state.reticle is not None  # the dashed circle needs no solution either

    def test_the_last_solution_says_what_it_was_and_how_old_it_is(self) -> None:
        last = solved_at(T0 - 90 * NS_PER_S, seq=12, solver="astap", n_matched=55, rms_arcsec=0.9)
        state = build_state(frame_summary(), None, TARGET, SETTINGS, last_good=last)
        view = state.last_solution
        assert view is not None
        assert view.frame_seq == 12
        assert view.t_utc == utc_ns_to_iso(T0 - 90 * NS_PER_S)
        assert view.age_s == pytest.approx(90.0)
        assert (view.roll_deg, view.n_matched, view.rms_arcsec, view.solver) == (
            12.0,
            55,
            0.9,
            "astap",
        )
        assert view.polaris_colatitude_deg == round(polaris_colatitude_deg(T0 - 90 * NS_PER_S), 5)

    def test_a_current_solution_is_also_the_last_solution(self) -> None:
        current = solved_at(T0 - NS_PER_S, seq=5)
        state = build_state(frame_summary(), current, TARGET, SETTINGS, last_good=current)
        assert state.last_solution is not None
        assert (state.last_solution.frame_seq, state.last_solution.age_s) == (5, 1.0)
        assert state.aim_ring is not None
        assert state.aim_ring.source == "current frame"  # the sky view wins while it exists

    def test_the_ring_and_the_last_solution_survive_the_json_and_stay_small(self) -> None:
        state = build_state(
            frame_summary(), None, TARGET, SETTINGS, last_good=solved_at(T0 - 5 * NS_PER_S)
        )
        assert state.aim_ring is not None
        assert state.last_solution is not None
        assert len(state.aim_ring.model_dump_json()) < 200
        assert len(state.last_solution.model_dump_json()) < 300
        assert decode_alignment_state(json.loads(state.model_dump_json())) == state
        assert len(pack_frame(state, tiny_jpeg())) < MAX_STATE_BYTES // 16


def test_the_settings_take_the_aim_as_a_pair() -> None:
    assert AlignmentSettings().aim_xy is None
    assert AlignmentSettings(aim_x_px=10.0, aim_y_px=20.0).aim_xy == (10.0, 20.0)
    with pytest.raises(ValueError, match="aim_x_px and aim_y_px together"):
        AlignmentSettings(aim_x_px=10.0)
    with pytest.raises(ValueError, match="aim_x_px and aim_y_px together"):
        AlignmentSettings(aim_y_px=10.0)
