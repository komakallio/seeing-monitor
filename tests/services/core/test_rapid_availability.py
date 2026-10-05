"""The offer of the rapid focus mode: the coarse focus, the place of Polaris, and the center.

The rule has two halves. The coarse focus (the median of the last five focus values of the normal
view, with at least five stars) must be at most 12 arcseconds, and `core` must know where Polaris
is: from the current solution, from a young last solution, or from the brightest star of the quick
solve. Each branch gets a test, and so does the wording of what is missing.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S
from seeingmon.config import load_config
from seeingmon.profile import load_profile
from seeingmon.services.core.alignment.focus import FocusHistory, FocusSnapshot
from seeingmon.services.core.alignment.rapid import RapidAvailability
from seeingmon.services.core.alignment.rapid_availability import (
    MIN_BRIGHTNESS_RATIO,
    MIN_MARGIN_PX,
    CoarseFocus,
    coarse_focus,
    convert_pixel,
    rapid_availability,
)
from seeingmon.services.core.alignment.solve import (
    LONE_STAR_RATIO,
    QuickSolution,
    brightest_star,
)
from seeingmon.services.core.alignment.state import FrameSummary
from seeingmon.services.core.settings import AlignmentSettings
from seeingmon.services.web.contract import SolvedView
from seeingmon.survey.detect import Detections, StarFlag
from seeingmon.survey.tracker import PointingTracker

PROFILE = load_profile("asi294mm-gs250")
T0 = 1_800_000_000 * NS_PER_S
SCALE = PROFILE.plate_scale_arcsec_per_px("bin2")  # 3.82 arcseconds per pixel
SETTINGS = AlignmentSettings()


def frame(**changes: Any) -> FrameSummary:
    fields: dict[str, Any] = {
        "seq": 9,
        "t_utc_ns": T0,
        "width_px": 4144,
        "height_px": 2822,
        "mode": "bin2",
        "exposure_s": 0.5,
        "gain": 120,
        "plate_scale_arcsec_px": SCALE,
    }
    fields.update(changes)
    return FrameSummary(**fields)


def focus_of(fwhm_px: list[float], stars: list[int] | None = None) -> FocusSnapshot:
    history = FocusHistory()
    counts = stars or [30] * len(fwhm_px)
    for index, (value, count) in enumerate(zip(fwhm_px, counts, strict=True)):
        history.add(index + 1, T0 - (len(fwhm_px) - index) * NS_PER_S, value, count)
    return history.snapshot()


GOOD_FOCUS = focus_of([2.4, 2.5, 2.3, 2.4, 2.45])  # about 9.2 arcseconds


def solved(x: float = 2075.0, y: float = 1400.0) -> SolvedView:
    return SolvedView(x_px=x, y_px=y, n_matched=60, rms_arcsec=0.8, age_s=0.4)


def quick(**changes: Any) -> QuickSolution:
    fields: dict[str, Any] = {"t_utc_ns": T0 - 20 * NS_PER_S, "seq": 3, "solved": True}
    fields.update(changes)
    return QuickSolution(**fields)


def judge(**changes: Any) -> RapidAvailability:
    """The availability for a good coarse focus and the current solution, with changes."""
    arguments: dict[str, Any] = {
        "profile": PROFILE,
        "settings": SETTINGS,
        "frame": frame(),
        "scale_arcsec_px": SCALE,
        "focus": GOOD_FOCUS,
        "current": solved(),
        "last_good": None,
        "last_good_position": None,
        "latest": None,
    }
    arguments.update(changes)
    return rapid_availability(**arguments)


# --- The pixel of another readout mode --------------------------------------------------------


class TestConvertPixel:
    def test_a_bin2_pixel_is_two_bin1_pixels_and_its_center_is_half_a_pixel_in(self) -> None:
        assert convert_pixel(PROFILE, 0.0, 0.0, "bin2", "bin1") == pytest.approx((0.5, 0.5))
        assert convert_pixel(PROFILE, 10.0, 20.0, "bin2", "bin1") == pytest.approx((20.5, 40.5))

    def test_the_center_of_the_frame_is_the_center_of_the_frame_in_every_mode(self) -> None:
        bin2 = ((4144 - 1) / 2, (2822 - 1) / 2)
        bin1 = ((8288 - 1) / 2, (5644 - 1) / 2)
        assert convert_pixel(PROFILE, *bin2, "bin2", "bin1") == pytest.approx(bin1)
        assert convert_pixel(PROFILE, *bin1, "bin1", "bin2") == pytest.approx(bin2)

    def test_the_conversion_goes_back_and_the_same_mode_changes_nothing(self) -> None:
        x, y = convert_pixel(PROFILE, 1234.5, 987.25, "bin2", "bin1")
        assert convert_pixel(PROFILE, x, y, "bin1", "bin2") == pytest.approx((1234.5, 987.25))
        assert convert_pixel(PROFILE, 5.5, 6.5, "bin1", "bin1") == (5.5, 6.5)

    def test_it_agrees_with_the_conversion_of_the_pointing_tracker(self) -> None:
        tracker = PointingTracker(PROFILE)
        for x, y in ((0.0, 0.0), (2075.0, 1400.0), (4143.0, 2821.0)):
            assert convert_pixel(PROFILE, x, y, "bin2", "bin1") == pytest.approx(
                tracker.convert(x, y, "bin2", "bin1")
            )

    def test_an_unknown_mode_is_an_error(self) -> None:
        with pytest.raises(Exception, match="unknown readout mode"):
            convert_pixel(PROFILE, 1.0, 1.0, "bin2", "bin9")


# --- The coarse focus -------------------------------------------------------------------------


class TestTheCoarseFocus:
    def test_it_is_the_median_of_the_last_five_values_in_arcseconds(self) -> None:
        focus = focus_of([9.0, 9.0, 2.4, 2.5, 2.3, 2.4, 2.45])  # the first two do not count
        result = coarse_focus(focus, SCALE)
        assert result.count == 5
        assert result.fwhm_arcsec == pytest.approx(2.4 * SCALE)
        assert result.stars == 30.0

    def test_fewer_than_five_values_give_none(self) -> None:
        assert coarse_focus(focus_of([2.4] * 4), SCALE) == CoarseFocus(None, None, 4)
        assert coarse_focus(None, SCALE) == CoarseFocus(None, None, 0)

    def test_an_unknown_plate_scale_gives_none(self) -> None:
        assert coarse_focus(GOOD_FOCUS, None).fwhm_arcsec is None

    def test_a_single_spike_among_the_five_does_not_move_it(self) -> None:
        focus = focus_of([2.4, 2.5, 7.0, 2.4, 2.45])
        assert coarse_focus(focus, SCALE).fwhm_arcsec == pytest.approx(2.45 * SCALE)

    def test_it_is_the_median_of_the_star_counts_too(self) -> None:
        result = coarse_focus(focus_of([2.4] * 5, [3, 9, 5, 4, 20]), SCALE)
        assert result.stars == 5.0


class TestWhatTheCoarseFocusAllows:
    def test_a_good_focus_with_the_current_solution_offers_the_mode(self) -> None:
        result = judge()
        assert result.available is True
        assert result.reason is None
        assert result.located_by == "current solution"
        assert result.coarse_fwhm_arcsec == pytest.approx(2.4 * SCALE, abs=0.01)
        assert result.max_fwhm_arcsec == 12.0
        assert result.center is not None

    def test_stars_that_are_too_wide_say_how_wide_and_what_the_limit_is(self) -> None:
        result = judge(focus=focus_of([5.5] * 5))  # 21 arcseconds
        assert result.available is False
        assert result.reason == (
            "the stars are too wide for the rapid mode: 21 arcsec, the limit is 12"
        )
        assert result.coarse_fwhm_arcsec == pytest.approx(21.01, abs=0.01)
        assert result.center is None
        assert result.located_by is None  # nothing is offered, so nothing is promised

    def test_the_limit_is_a_setting_and_a_star_at_the_limit_counts(self) -> None:
        value = 2.4 * SCALE
        exact = AlignmentSettings(rapid_focus_max_fwhm_arcsec=value + 0.001)
        assert judge(settings=exact).available is True
        below = AlignmentSettings(rapid_focus_max_fwhm_arcsec=value - 0.001)
        result = judge(settings=below)
        assert result.available is False
        assert result.reason is not None
        assert "too wide" in result.reason
        assert result.max_fwhm_arcsec == pytest.approx(value - 0.001)

    def test_a_decimal_in_the_sentence_has_one_place_at_most(self) -> None:
        result = judge(focus=focus_of([3.35] * 5))  # 12.797 arcseconds
        assert (
            result.reason
            == "the stars are too wide for the rapid mode: 12.8 arcsec, the limit is 12"
        )

    def test_too_few_values_say_how_many_there_are(self) -> None:
        result = judge(focus=focus_of([2.4, 2.4, 2.4]))
        assert result.available is False
        assert result.reason == (
            "the coarse focus is not known: the quick solve has measured 3 of the 5 focus values "
            "that it needs"
        )
        assert result.coarse_fwhm_arcsec is None

    def test_no_history_at_all_is_the_same_case(self) -> None:
        result = judge(focus=None)
        assert result.available is False
        assert result.reason is not None
        assert "measured 0 of the 5 focus values" in result.reason

    def test_too_few_stars_say_how_many(self) -> None:
        result = judge(focus=focus_of([2.4] * 5, [3, 4, 3, 2, 3]))
        assert result.available is False
        assert (
            result.reason == "the coarse focus rests on too few stars: 3, and it needs at least 5"
        )
        assert result.coarse_fwhm_arcsec == pytest.approx(2.4 * SCALE, abs=0.01)  # still reported

    def test_five_stars_are_enough(self) -> None:
        assert judge(focus=focus_of([2.4] * 5, [5, 5, 5, 5, 5])).available is True
        assert judge(focus=focus_of([2.4] * 5, [4, 4, 4, 4, 4])).available is False

    def test_an_unknown_plate_scale_says_so(self) -> None:
        result = judge(scale_arcsec_px=None, frame=frame(plate_scale_arcsec_px=None))
        assert result.available is False
        assert result.reason is not None
        assert "plate scale of the frame is not known" in result.reason


# --- Where Polaris is -------------------------------------------------------------------------


class TestTheCurrentSolution:
    def test_the_center_is_the_solved_pixel_in_pixels_of_the_fast_readout_mode(self) -> None:
        result = judge(current=solved(2075.0, 1400.0))
        assert result.center == pytest.approx((4150.5, 2800.5))  # 2 x + 0.5

    def test_a_position_inside_the_margin_counts(self) -> None:
        edge = MIN_MARGIN_PX
        assert judge(current=solved(edge, 1400.0)).available is True
        assert judge(current=solved(4143.0 - edge, 1400.0)).available is True
        assert judge(current=solved(2000.0, edge)).available is True
        assert judge(current=solved(2000.0, 2821.0 - edge)).available is True

    @pytest.mark.parametrize(
        ("x", "y", "edge"),
        [
            (35.0, 1400.0, "35"),
            (4100.0, 1400.0, "43"),
            (2000.0, 10.0, "10"),
            (2000.0, 2800.0, "21"),
        ],
    )
    def test_a_position_near_an_edge_is_not_in_view(self, x: float, y: float, edge: str) -> None:
        result = judge(current=solved(x, y))
        assert result.available is False
        assert result.reason == (
            f"Polaris is not in view: the current solution puts it {edge} bin2 pixels from its "
            "edge, and it must be at least 80 bin2 pixels in"
        )
        assert result.center is None

    def test_a_position_outside_the_frame_says_so(self) -> None:
        result = judge(current=solved(-50.0, 1400.0))
        assert result.reason is not None
        assert "puts it outside the frame" in result.reason

    def test_an_older_solution_does_not_overrule_a_current_one(self) -> None:
        result = judge(
            current=solved(35.0, 1400.0),
            last_good=quick(),
            last_good_position=(2000.0, 1400.0),
            latest=quick(brightest_x_px=2000.0, brightest_y_px=1400.0, brightest_ratio=50.0),
        )
        assert result.available is False
        assert result.reason is not None
        assert "the current solution puts it" in result.reason

    def test_the_margin_follows_the_readout_mode_of_the_frame(self) -> None:
        """80 bin2 pixels are 160 bin1 pixels, the same distance on the sky."""
        bin1 = frame(
            mode="bin1",
            width_px=8288,
            height_px=5644,
            plate_scale_arcsec_px=PROFILE.plate_scale_arcsec_per_px("bin1"),
        )
        scale = PROFILE.plate_scale_arcsec_per_px("bin1")
        near = judge(frame=bin1, scale_arcsec_px=scale, current=solved(120.0, 2800.0))
        assert near.available is False
        assert near.reason is not None
        assert "at least 160 bin1 pixels in" in near.reason
        assert judge(frame=bin1, scale_arcsec_px=scale, current=solved(170.0, 2800.0)).available


class TestTheLastSolution:
    def test_a_young_solution_places_polaris_where_it_falls_now(self) -> None:
        result = judge(current=None, last_good=quick(), last_good_position=(2075.0, 1400.0))
        assert result.available is True
        assert result.located_by == "last solution"
        assert result.center == pytest.approx((4150.5, 2800.5))

    def test_the_age_limit_is_a_setting(self) -> None:
        old = quick(t_utc_ns=T0 - 601 * NS_PER_S)
        result = judge(current=None, last_good=old, last_good_position=(2075.0, 1400.0))
        assert result.available is False
        assert result.reason is not None
        assert result.reason.startswith("Polaris is not located: the last solution is 601 s old")
        assert "(the limit is 600 s)" in result.reason
        young = quick(t_utc_ns=T0 - 599 * NS_PER_S)
        assert judge(current=None, last_good=young, last_good_position=(2075.0, 1400.0)).available
        longer = AlignmentSettings(rapid_focus_max_solution_age_s=1200.0)
        assert judge(
            settings=longer, current=None, last_good=old, last_good_position=(2075.0, 1400.0)
        ).available

    def test_a_solution_that_cannot_place_polaris_is_no_help(self) -> None:
        result = judge(current=None, last_good=quick(), last_good_position=None)
        assert result.available is False
        assert result.reason is not None
        assert "the last solution cannot place Polaris" in result.reason

    def test_a_young_solution_near_the_edge_is_not_in_view_and_the_brightest_star_does_not_help(
        self,
    ) -> None:
        result = judge(
            current=None,
            last_good=quick(),
            last_good_position=(30.0, 1400.0),
            latest=quick(brightest_x_px=2000.0, brightest_y_px=1400.0, brightest_ratio=80.0),
        )
        assert result.available is False
        assert result.reason is not None
        assert "the last solution puts it 30 bin2 pixels from its edge" in result.reason

    def test_an_old_solution_falls_back_to_the_brightest_star(self) -> None:
        old = quick(t_utc_ns=T0 - 900 * NS_PER_S)
        result = judge(
            current=None,
            last_good=old,
            last_good_position=None,
            latest=quick(brightest_x_px=1800.0, brightest_y_px=900.0, brightest_ratio=12.0),
        )
        assert result.available is True
        assert result.located_by == "brightest star"
        assert result.center == pytest.approx((3600.5, 1800.5))


class TestTheBrightestStar:
    def test_without_a_solution_the_brightest_star_is_polaris_when_it_stands_out(self) -> None:
        latest = quick(
            solved=False, brightest_x_px=1800.0, brightest_y_px=900.0, brightest_ratio=7.0
        )
        result = judge(current=None, latest=latest)
        assert result.available is True
        assert result.located_by == "brightest star"
        assert result.center == pytest.approx((3600.5, 1800.5))

    def test_five_times_is_enough_and_less_is_not(self) -> None:
        at = quick(
            brightest_x_px=1800.0, brightest_y_px=900.0, brightest_ratio=MIN_BRIGHTNESS_RATIO
        )
        assert judge(current=None, latest=at).available is True
        below = quick(brightest_x_px=1800.0, brightest_y_px=900.0, brightest_ratio=2.14)
        result = judge(current=None, latest=below)
        assert result.available is False
        assert result.reason == (
            "Polaris is not located: no solve has found the star field in this alignment yet; "
            "the brightest star is only 2.1 times as bright as the next one (it must be at least "
            "5 times)"
        )

    def test_a_lone_star_counts(self) -> None:
        lone = quick(brightest_x_px=1800.0, brightest_y_px=900.0, brightest_ratio=LONE_STAR_RATIO)
        assert judge(current=None, latest=lone).available is True

    def test_no_detection_and_no_solve_say_so(self) -> None:
        none_found = judge(current=None, latest=quick(solved=False))
        assert none_found.reason == (
            "Polaris is not located: no solve has found the star field in this alignment yet; "
            "the quick solve found no star"
        )
        never = judge(current=None, latest=None)
        assert never.reason == (
            "Polaris is not located: no solve has found the star field in this alignment yet; "
            "no solve has finished yet"
        )

    def test_the_brightest_star_near_the_edge_is_not_in_view(self) -> None:
        latest = quick(brightest_x_px=40.0, brightest_y_px=900.0, brightest_ratio=30.0)
        result = judge(current=None, latest=latest)
        assert result.available is False
        assert result.reason is not None
        assert "the brightest star puts it 40 bin2 pixels from its edge" in result.reason


class TestSeveralProblems:
    def test_every_missing_thing_is_named_in_one_reason(self) -> None:
        result = judge(focus=focus_of([5.5] * 5), current=None, latest=None)
        assert result.available is False
        assert result.reason is not None
        parts = result.reason.split("; ")
        assert parts[0] == "the stars are too wide for the rapid mode: 21 arcsec, the limit is 12"
        assert parts[1].startswith("Polaris is not located: ")
        assert result.coarse_fwhm_arcsec == pytest.approx(21.01, abs=0.01)  # the numbers stay

    def test_an_unknown_readout_mode_is_a_problem_and_not_an_error(self) -> None:
        odd = frame(mode="bin3")
        result = judge(frame=odd)
        assert result.available is False
        assert result.reason is not None
        assert "the profile does not describe the readout mode bin3" in result.reason


# --- The brightest detection of a quick solve -------------------------------------------------


def detections(flux: list[float], flags: list[int] | None = None) -> Detections:
    n = len(flux)
    zeros = np.zeros(n)
    return Detections(
        shape=(100, 100),
        x=np.arange(n, dtype=np.float64) * 10.0 + 5.0,
        y=np.arange(n, dtype=np.float64) * 20.0 + 7.0,
        flux=np.array(flux, dtype=np.float64),
        peak=np.ones(n),
        fwhm_px=np.full(n, 2.0),
        x_error_px=zeros.copy(),
        y_error_px=zeros.copy(),
        elongation=np.ones(n),
        trail_length_px=zeros.copy(),
        trail_angle_rad=zeros.copy(),
        flags=np.array(flags if flags is not None else [0] * n, dtype=np.uint16),
        snr=np.full(n, 50.0),
        n_pixels=np.ones(n, dtype=np.int32),
        background_level=0.0,
        background_rms=1.0,
    )


class TestBrightestStar:
    def test_it_gives_the_position_and_the_ratio_to_the_next_star(self) -> None:
        x, y, ratio = brightest_star(detections([100.0, 5000.0, 900.0, 40.0]))
        assert (x, y) == (15.0, 27.0)  # the second detection
        assert ratio == pytest.approx(5000.0 / 900.0)

    def test_the_order_of_the_detections_does_not_matter(self) -> None:
        shuffled = brightest_star(detections([900.0, 40.0, 5000.0, 100.0]))
        assert shuffled[:2] == (25.0, 47.0)
        assert shuffled[2] == pytest.approx(5000.0 / 900.0)

    def test_a_saturated_star_counts_because_polaris_saturates(self) -> None:
        flags = [0, int(StarFlag.SATURATED), 0]
        x, _, ratio = brightest_star(detections([100.0, 5000.0, 900.0], flags))
        assert x == 15.0
        assert ratio is not None

    @pytest.mark.parametrize("flag", [StarFlag.HOT_PIXEL, StarFlag.STREAK])
    def test_a_hot_pixel_and_a_streak_do_not_count(self, flag: StarFlag) -> None:
        x, y, ratio = brightest_star(detections([100.0, 5000.0, 900.0], [0, int(flag), 0]))
        assert (x, y) == (25.0, 47.0)  # the third detection
        assert ratio == pytest.approx(9.0)

    def test_one_star_has_the_ratio_of_a_lone_star(self) -> None:
        assert brightest_star(detections([100.0])) == (5.0, 7.0, LONE_STAR_RATIO)
        only = brightest_star(detections([100.0, 5000.0], [int(StarFlag.HOT_PIXEL), 0]))
        assert only == (15.0, 27.0, LONE_STAR_RATIO)

    def test_a_huge_ratio_is_capped_so_that_it_stays_a_number_that_json_can_carry(self) -> None:
        _, _, ratio = brightest_star(detections([1e9, 1e-3]))
        assert ratio == LONE_STAR_RATIO
        assert math.isfinite(LONE_STAR_RATIO)

    def test_without_a_star_that_counts_the_answer_is_none(self) -> None:
        assert brightest_star(None) == (None, None, None)
        assert brightest_star(detections([])) == (None, None, None)
        assert brightest_star(detections([5.0], [int(StarFlag.HOT_PIXEL)])) == (None, None, None)
        assert brightest_star(detections([0.0, -3.0, float("nan")])) == (None, None, None)


class TestTheSolutionThatCarriesIt:
    def test_the_fields_survive_the_trip_between_the_processes(self) -> None:
        solution = quick(brightest_x_px=1800.5, brightest_y_px=900.25, brightest_ratio=12.5)
        again = QuickSolution.from_dict(solution.to_dict())
        assert (again.brightest_x_px, again.brightest_y_px, again.brightest_ratio) == (
            1800.5,
            900.25,
            12.5,
        )
        assert again == solution

    def test_a_dictionary_from_an_older_worker_has_no_brightest_star(self) -> None:
        data = quick().to_dict()
        for key in ("brightest_x_px", "brightest_y_px", "brightest_ratio"):
            del data[key]
        again = QuickSolution.from_dict(data)
        assert (again.brightest_x_px, again.brightest_y_px, again.brightest_ratio) == (
            None,
            None,
            None,
        )

    def test_the_dictionary_holds_plain_numbers(self) -> None:
        data = quick(
            brightest_x_px=1.0, brightest_y_px=2.0, brightest_ratio=LONE_STAR_RATIO
        ).to_dict()
        assert all(isinstance(data[k], float) for k in ("brightest_x_px", "brightest_ratio"))
        assert dataclasses.is_dataclass(QuickSolution)


class TestTheSettings:
    def test_the_defaults_are_12_arcseconds_and_600_seconds(self) -> None:
        assert AlignmentSettings().rapid_focus_max_fwhm_arcsec == 12.0
        assert AlignmentSettings().rapid_focus_max_solution_age_s == 600.0

    def test_the_default_file_gives_the_same_values_and_the_local_file_changes_them(
        self, tmp_path: Path
    ) -> None:
        defaults = load_config(local_file=tmp_path / "none.toml", env={}).section(
            "alignment", AlignmentSettings
        )
        assert defaults == AlignmentSettings()
        local = tmp_path / "config.toml"
        local.write_text(
            "[alignment]\n"
            "rapid_focus_max_fwhm_arcsec = 15.0\n"
            "rapid_focus_max_solution_age_s = 60\n",
            encoding="utf-8",
        )
        changed = load_config(local_file=local, env={}).section("alignment", AlignmentSettings)
        assert (changed.rapid_focus_max_fwhm_arcsec, changed.rapid_focus_max_solution_age_s) == (
            15.0,
            60.0,
        )

    @pytest.mark.parametrize(
        "name", ["rapid_focus_max_fwhm_arcsec", "rapid_focus_max_solution_age_s"]
    )
    def test_a_limit_must_be_positive(self, name: str) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            AlignmentSettings(**{name: 0.0})
