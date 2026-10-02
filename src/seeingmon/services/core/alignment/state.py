"""The `AlignmentState` that `web` shows next to the live view.

`build_state` merges what the helper knows about one frame (its size, the histogram, the share of
saturated pixels) with the latest quick solve and the target. The shapes come from the contract
that `web` and `core` share (`seeingmon.services.web.contract`). A part that `core` does not know is
`None`, and `quality` says why, so the UI can show a reason instead of a blank.

**The target.** A target in `[alignment]` (a position and a roll, in pixels of the readout mode of
the alignment stream) wins. Without one, the target is the place where the reference solution
predicts Polaris at the time of the frame, with the roll of that solution. Without either, the state
has no target and no offset.

**The offset.** The offset is the solved position minus the target, in pixels and in arcseconds
(through the plate scale of the readout mode), and the roll offset is the solved roll minus the
target roll, wrapped to -180 to 180 degrees. A solution older than `solution_max_age_s` is no longer
current, and the state leaves it out.

**The reticle.** The first layer of the live view is fixed in the picture: a circle centered on
the aim (the center of the frame, unless `[alignment]` names another pixel) with the radius of the
orbit of Polaris in pixels (`seeingmon.survey.skyview.reticle_geometry`). It needs the frame size,
the plate scale, and the colatitude of Polaris at the time of the frame, which follows from the time
alone, so the state carries it without a solution too.

**The sky view.** A current solution that carries a camera attitude also gives the `sky` view, the
layer that is fixed to the stars: the pole, the aim and the offset from it, the aim ring (where
Polaris belongs now), the circle that Polaris follows, and the camera model that projects them.
With a site (`[site]`), the view adds the move in altitude and in azimuth that brings the pole to
the aim. It does not depend on the target, so a new install that has none still shows all of this.
It follows the freshness rule of the solved position, and without a current solution `sky` is
`None` and `quality["sky"]` says why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from seeingmon.clock import NS_PER_S, utc_ns_to_iso
from seeingmon.scheduler.config import SiteConfig
from seeingmon.services.core.alignment.solve import QuickSolution
from seeingmon.services.core.settings import AlignmentSettings
from seeingmon.services.web.contract import (
    AlignmentFrameInfo,
    AlignmentState,
    FocusView,
    HistogramView,
    OffsetView,
    ReticleView,
    SaturationView,
    SkyView,
    SolvedView,
    TargetView,
)
from seeingmon.survey.apparent import earth_rotation_angle
from seeingmon.survey.pointing import polaris_colatitude_deg
from seeingmon.survey.skyview import build_sky_view, reticle_geometry, zenith_vector
from seeingmon.survey.tracker import PointingTracker


@dataclass(frozen=True, slots=True)
class FrameSummary:
    """What the helper measured in one frame."""

    seq: int
    t_utc_ns: int
    width_px: int
    height_px: int
    mode: str
    exposure_s: float | None
    gain: int | None
    plate_scale_arcsec_px: float | None
    histogram: HistogramView | None = None
    saturation: SaturationView | None = None


@dataclass(frozen=True, slots=True)
class Target:
    """Where Polaris belongs, and the roll that the camera should have."""

    x_px: float
    y_px: float
    roll_deg: float | None = None


def wrap_degrees(angle: float) -> float:
    """The angle wrapped into the range from -180 (excluded) to 180 degrees."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def resolve_target(
    settings: AlignmentSettings, tracker: PointingTracker | None, t_utc_ns: int, mode: str
) -> Target | None:
    """The target for a frame: the configured one, else the reference solution's, else `None`."""
    if settings.target_x_px is not None and settings.target_y_px is not None:
        return Target(settings.target_x_px, settings.target_y_px, settings.target_roll_deg)
    reference = None if tracker is None else tracker.reference
    if tracker is None or reference is None:
        return None
    solution = reference.solution
    position = solution.polaris_pixel(t_utc_ns)
    if position is None:
        return None
    x, y = tracker.convert(position[0], position[1], solution.mode, mode)
    return Target(x, y, solution.attitude_at(t_utc_ns).roll_deg())


def build_state(
    frame: FrameSummary,
    solution: QuickSolution | None,
    target: Target | None,
    settings: AlignmentSettings,
    *,
    best_fwhm_px: float | None = None,
    site: SiteConfig | None = None,
) -> AlignmentState:
    """The state that describes `frame`, with the latest solution and the target.

    `site` is the `[site]` of the configuration. With it, the sky view says how to move the camera
    in altitude and in azimuth, and without it the view gives the image directions only.
    """
    quality: dict[str, str] = {}
    info = AlignmentFrameInfo(
        seq=frame.seq,
        width_px=frame.width_px,
        height_px=frame.height_px,
        readout_mode=frame.mode[:64],
        exposure_s=frame.exposure_s,
        gain=frame.gain,
        plate_scale_arcsec_px=frame.plate_scale_arcsec_px,
    )
    target_view = (
        None
        if target is None
        else TargetView(x_px=target.x_px, y_px=target.y_px, roll_deg=target.roll_deg)
    )
    if target is None:
        quality["target"] = "no target is configured and no reference solution exists"

    solved_view, reason = _current_solution(solution, frame, settings)
    offset_view: OffsetView | None = None
    if reason is not None:
        quality["solved"] = reason
    if solved_view is not None and target is not None:
        scale = frame.plate_scale_arcsec_px or (
            None if solution is None else solution.scale_arcsec_px
        )
        offset_view = _offset(solved_view, target, scale)
    elif target is not None:
        quality["offset"] = "the offset needs a current solution"

    reticle_view = _reticle(frame, solution, settings, quality)

    sky_view: SkyView | None = None
    if solved_view is None:
        quality["sky"] = reason or "no current solution"
    elif solution is None or solution.attitude is None:
        quality["sky"] = "the solution carries no camera attitude"
    else:
        polaris = (
            None
            if solution.x_px is None or solution.y_px is None
            else (solution.x_px, solution.y_px)
        )
        zenith = (
            None
            if site is None
            else zenith_vector(
                site.latitude_deg, site.longitude_deg, earth_rotation_angle(solution.t_utc_ns)
            )
        )
        sky_view = SkyView.from_geometry(
            build_sky_view(
                solution.attitude,
                frame.width_px,
                frame.height_px,
                solution.polaris_colatitude_deg,
                polaris_xy=polaris,
                aim_xy=settings.aim_xy,
                zenith=zenith,
            )
        )

    focus_view: FocusView | None = None
    if solution is not None and solution.focus_fwhm_px is not None:
        focus_view = FocusView(
            fwhm_px=solution.focus_fwhm_px,
            best_fwhm_px=best_fwhm_px,
            n_stars=solution.n_focus_stars,
        )
    else:
        quality["focus"] = "no unsaturated stars to measure"

    if frame.histogram is None:
        quality["histogram"] = "the frame has no histogram"
    return AlignmentState(
        active=True,
        t_utc=utc_ns_to_iso(frame.t_utc_ns),
        frame=info,
        target=target_view,
        solved=solved_view,
        offset=offset_view,
        focus=focus_view,
        histogram=frame.histogram,
        saturation=frame.saturation,
        reticle=reticle_view,
        sky=sky_view,
        quality=quality,
    )


def _reticle(
    frame: FrameSummary,
    solution: QuickSolution | None,
    settings: AlignmentSettings,
    quality: dict[str, str],
) -> ReticleView | None:
    """The fixed circle of the reticle. It follows from the frame and the time, not a solution."""
    scale = frame.plate_scale_arcsec_px or (None if solution is None else solution.scale_arcsec_px)
    if scale is None:
        quality["reticle"] = "the plate scale is not known"
        return None
    colatitude = polaris_colatitude_deg(frame.t_utc_ns)
    geometry = reticle_geometry(
        frame.width_px, frame.height_px, scale, colatitude, aim_xy=settings.aim_xy
    )
    if geometry is None:
        quality["reticle"] = "the radius of the orbit of Polaris is not known"
        return None
    return ReticleView(
        x_px=geometry.x_px,
        y_px=geometry.y_px,
        radius_px=geometry.radius_px,
        polaris_colatitude_deg=round(colatitude, 5),
    )


def _current_solution(
    solution: QuickSolution | None, frame: FrameSummary, settings: AlignmentSettings
) -> tuple[SolvedView | None, str | None]:
    """The solved position when it is current, else the reason that it is not."""
    if solution is None:
        return None, "no solve has finished yet"
    if not solution.solved or solution.x_px is None or solution.y_px is None:
        return None, solution.note or "the latest frame could not be solved"
    age_s = max(0.0, (frame.t_utc_ns - solution.t_utc_ns) / NS_PER_S)
    if age_s > settings.solution_max_age_s:
        return None, f"the last solution is {age_s:.0f} s old"
    view = SolvedView(
        x_px=solution.x_px,
        y_px=solution.y_px,
        roll_deg=solution.roll_deg,
        n_matched=solution.n_matched,
        rms_arcsec=solution.rms_arcsec,
        age_s=age_s,
    )
    return view, None


def _offset(solved: SolvedView, target: Target, scale_arcsec_px: float | None) -> OffsetView:
    dx = solved.x_px - target.x_px
    dy = solved.y_px - target.y_px
    distance = math.hypot(dx, dy)
    roll = (
        None
        if solved.roll_deg is None or target.roll_deg is None
        else wrap_degrees(solved.roll_deg - target.roll_deg)
    )
    scale = scale_arcsec_px
    return OffsetView(
        dx_px=dx,
        dy_px=dy,
        distance_px=distance,
        dx_arcsec=None if scale is None else dx * scale,
        dy_arcsec=None if scale is None else dy * scale,
        distance_arcsec=None if scale is None else distance * scale,
        roll_deg=roll,
    )
