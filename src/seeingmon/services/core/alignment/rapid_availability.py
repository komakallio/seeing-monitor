"""Whether the rapid focus mode is offered, and where it puts its ROI.

The person turns the focuser and watches the star width. The rapid focus mode serves that, and the
Align page offers it only when it can work: the stars are narrow enough for the aperture of the
fast analysis (the coarse focus is good), and `core` knows where Polaris is, so that it can place a
ROI of 4 arcminutes on it. `rapid_availability` judges both from what the alignment helper already
holds, and it says in words what is missing.

**The coarse focus.** It is the median of the last `COARSE_READINGS` (five) values of the focus
history of the normal view, in arcseconds through the plate scale of the frame, and it must rest
on at least `COARSE_MIN_STARS` (five) stars (the median of the star counts of those values). It
must not exceed `[alignment] rapid_focus_max_fwhm_arcsec` (12). A history of fewer than five values
gives no coarse focus, so the mode is not offered in the first seconds of an alignment.

**Where Polaris is.** The sources, in this order:

1. *The current solution* places Polaris at the pixel that it solved.
2. *The last solution* places Polaris where the Earth's rotation has carried it since: the helper
   turns the solution to the time of the frame (`polaris_from_solution`). It counts when it is
   younger than `[alignment] rapid_focus_max_solution_age_s` (600). It is right while the mount
   has not moved since, which is the case when you focus. The aim ring, which says where Polaris
   would be if the pole sat at the aim, comes from the same solution and is not used: a mount that
   is still being aligned has the pole elsewhere.
3. *The brightest star*, when no solution counts. Polaris is far brighter than any other star of
   the field, so the brightest detection of the quick solve is Polaris when it is at least
   `MIN_BRIGHTNESS_RATIO` (five) times brighter than the second brightest.

A position counts only when it lies at least `MIN_MARGIN_PX` (80, in pixels of the survey readout
mode) inside every edge of the frame, so that the ROI and the edge margin of the fast stream fit
and the position is not at the edge of what the solve can trust. The source that comes first
decides: a current solution that puts Polaris outside the margin is not overruled by an older one.

**The center.** The position is in pixels of the frame of the alignment (bin2). The ROI lives in
the fast readout mode (bin1), so `convert_pixel` converts it through the plate scales, with the
pixel convention of the profile: the center of the first pixel is at 0 in every mode, and a binned
pixel covers `factor` pixels of the finer mode, so bin2 pixel `i` has its center at `2 i + 0.5` in
bin1.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Final

from seeingmon.profile import Profile, ProfileError
from seeingmon.services.core.alignment.focus import FocusSnapshot
from seeingmon.services.core.alignment.rapid import RapidAvailability
from seeingmon.services.core.alignment.solve import QuickSolution
from seeingmon.services.core.alignment.state import FrameSummary
from seeingmon.services.core.settings import AlignmentSettings
from seeingmon.services.web.contract import RapidLocatedBy, SolvedView

COARSE_READINGS = 5
COARSE_MIN_STARS = 5
MIN_MARGIN_PX = 80.0
MIN_BRIGHTNESS_RATIO = 5.0

BY_CURRENT: Final = "current solution"
BY_LAST: Final = "last solution"
BY_BRIGHTEST: Final = "brightest star"


def convert_pixel(
    profile: Profile, x: float, y: float, from_mode: str, to_mode: str
) -> tuple[float, float]:
    """The pixel of another readout mode where the same point of the sky falls.

    The center of the first pixel is 0 in every mode, and a pixel of the coarser mode covers
    `factor` pixels of the finer one, with `factor` the ratio of the plate scales. A pixel `i` of
    the coarser mode therefore has its center at `(i + 0.5) * factor - 0.5` in the finer mode, and
    the inverse holds the other way: bin2 pixel `i` is bin1 pixel `2 i + 0.5`.
    """
    if from_mode == to_mode:
        return x, y
    factor = profile.plate_scale_arcsec_per_px(from_mode) / profile.plate_scale_arcsec_per_px(
        to_mode
    )
    return (x + 0.5) * factor - 0.5, (y + 0.5) * factor - 0.5


@dataclass(frozen=True, slots=True)
class CoarseFocus:
    """The coarse focus: `fwhm_arcsec` and `stars` are medians, `count` is the values that it has.

    The first two are `None` while fewer than `COARSE_READINGS` values exist, or while the plate
    scale is unknown.
    """

    fwhm_arcsec: float | None
    stars: float | None
    count: int


def coarse_focus(focus: FocusSnapshot | None, scale_arcsec_px: float | None) -> CoarseFocus:
    """The median of the last five focus values and of their star counts."""
    points = () if focus is None else focus.points[-COARSE_READINGS:]
    if len(points) < COARSE_READINGS or scale_arcsec_px is None:
        return CoarseFocus(None, None, len(points))
    return CoarseFocus(
        statistics.median(point.fwhm_px for point in points) * scale_arcsec_px,
        float(statistics.median(point.n_stars for point in points)),
        len(points),
    )


def _text(value: float) -> str:
    """A number for a sentence: one decimal at most, and no `.0`."""
    return f"{round(value, 1):g}"


def coarse_problem(
    coarse: CoarseFocus, scale_arcsec_px: float | None, limit_arcsec: float
) -> str | None:
    if coarse.count < COARSE_READINGS:
        return (
            f"the coarse focus is not known: the quick solve has measured {coarse.count} of the "
            f"{COARSE_READINGS} focus values that it needs"
        )
    if scale_arcsec_px is None or coarse.fwhm_arcsec is None or coarse.stars is None:
        return "the coarse focus is not known: the plate scale of the frame is not known"
    if coarse.stars < COARSE_MIN_STARS:
        return (
            f"the coarse focus rests on too few stars: {_text(coarse.stars)}, and it needs at "
            f"least {COARSE_MIN_STARS}"
        )
    if coarse.fwhm_arcsec > limit_arcsec:
        return (
            f"the stars are too wide for the rapid mode: {_text(coarse.fwhm_arcsec)} arcsec, "
            f"the limit is {_text(limit_arcsec)}"
        )
    return None


def rapid_availability(
    *,
    profile: Profile,
    settings: AlignmentSettings,
    frame: FrameSummary,
    scale_arcsec_px: float | None,
    focus: FocusSnapshot | None,
    current: SolvedView | None,
    last_good: QuickSolution | None,
    last_good_position: tuple[float, float] | None,
    latest: QuickSolution | None,
) -> RapidAvailability:
    """Judge whether the mode is offered, and where its ROI goes. See the module text.

    `frame` is the frame that the state describes, with its size and its readout mode, and
    `scale_arcsec_px` is its plate scale. `current` is the current solution as the state shows it,
    `last_good` is the latest solution that found the star field, and `last_good_position` is the
    pixel where Polaris falls at the time of the frame according to it (the caller computes it
    only when `current` is `None`, because it costs a projection). `latest` is the newest quick
    solution, solved or not, whose brightest detection stands in for a solution.
    """
    limit = settings.rapid_focus_max_fwhm_arcsec
    coarse = coarse_focus(focus, scale_arcsec_px)
    problems: list[str] = []
    problem = coarse_problem(coarse, scale_arcsec_px, limit)
    if problem is not None:
        problems.append(problem)

    located_by: RapidLocatedBy | None = None
    position: tuple[float, float] | None = None
    notes: list[str] = []
    if current is not None:
        located_by, position = BY_CURRENT, (current.x_px, current.y_px)
    else:
        age_s = None if last_good is None else (frame.t_utc_ns - last_good.t_utc_ns) / 1e9
        max_age_s = settings.rapid_focus_max_solution_age_s
        if last_good is None:
            notes.append("no solve has found the star field in this alignment yet")
        elif age_s is not None and age_s > max_age_s:
            notes.append(f"the last solution is {age_s:.0f} s old (the limit is {max_age_s:.0f} s)")
        elif last_good_position is None:
            notes.append("the last solution cannot place Polaris")
        else:
            located_by, position = BY_LAST, last_good_position
        if position is None:
            position, located_by, note = _brightest_star(latest)
            if note:
                notes.append(note)
    center: tuple[float, float] | None = None
    if position is None:
        problems.append("Polaris is not located: " + "; ".join(notes))
    else:
        problem = _margin_problem(profile, frame, scale_arcsec_px, position, located_by)
        if problem is not None:
            problems.append(problem)
            located_by = None
        else:
            try:
                center = convert_pixel(
                    profile, position[0], position[1], frame.mode, profile.fast_mode.mode
                )
            except ProfileError:
                problems.append(f"the profile does not describe the readout mode {frame.mode}")
    available = not problems
    return RapidAvailability(
        available=available,
        reason=None if available else "; ".join(problems),
        located_by=located_by if available else None,
        coarse_fwhm_arcsec=None if coarse.fwhm_arcsec is None else round(coarse.fwhm_arcsec, 3),
        max_fwhm_arcsec=limit,
        center=center if available else None,
    )


def _brightest_star(
    latest: QuickSolution | None,
) -> tuple[tuple[float, float] | None, RapidLocatedBy | None, str]:
    """Polaris as the brightest star of the newest quick solve, or the reason that it is not."""
    if latest is None:
        return None, None, "no solve has finished yet"
    ratio = latest.brightest_ratio
    if latest.brightest_x_px is None or latest.brightest_y_px is None or ratio is None:
        return None, None, "the quick solve found no star"
    if ratio < MIN_BRIGHTNESS_RATIO:
        return (
            None,
            None,
            f"the brightest star is only {_text(ratio)} times as bright as the next one (it "
            f"must be at least {_text(MIN_BRIGHTNESS_RATIO)} times)",
        )
    return (latest.brightest_x_px, latest.brightest_y_px), BY_BRIGHTEST, ""


def _margin_problem(
    profile: Profile,
    frame: FrameSummary,
    scale_arcsec_px: float | None,
    position: tuple[float, float],
    located_by: RapidLocatedBy | None,
) -> str | None:
    """The reason that a position is too near the edge of the frame, or `None`."""
    margin = MIN_MARGIN_PX
    try:
        survey_scale = profile.plate_scale_arcsec_per_px(profile.survey_mode.mode)
    except ProfileError:
        survey_scale = None
    if survey_scale is not None and scale_arcsec_px is not None:
        margin = MIN_MARGIN_PX * survey_scale / scale_arcsec_px  # the same distance on the sky
    x, y = position
    edge = min(x, frame.width_px - 1 - x, y, frame.height_px - 1 - y)
    if edge >= margin:
        return None
    source = {
        BY_CURRENT: "the current solution",
        BY_LAST: "the last solution",
        BY_BRIGHTEST: "the brightest star",
    }.get(located_by or "", "the position")
    where = "outside the frame" if edge < 0 else f"{_text(edge)} {frame.mode} pixels from its edge"
    return (
        f"Polaris is not in view: {source} puts it {where}, and it must be at least "
        f"{_text(margin)} {frame.mode} pixels in"
    )
