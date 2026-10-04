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

**The aim ring without a solution.** The aim ring (where Polaris belongs on the circle of the
reticle) depends on the twist of the picture about the optical axis and on the time, and a move in
altitude or azimuth leaves the twist alone. So when the latest solve fails or is too old, the state
keeps the ring: it turns the last good solution to the time of the frame (the Earth rotates the
picture about the pole) and projects Polaris and the pole through it. `aim_ring.source` says `last
solution`, and `last_solution` says how old that solution is. The pole, the polar grid, and the move
in altitude and azimuth need the pointing of the frame, so they stay out of such a state: `sky` is
`None`.

**The timing.** The frame and the solution are two different frames: the solver takes the newest
frame when it is free, and it needs time. `timing` says which frame each part comes from and how old
the frame is, so that the page can show the lag instead of hiding it (`TimingView`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from seeingmon.clock import NS_PER_S, utc_ns_to_iso
from seeingmon.scheduler.config import SiteConfig
from seeingmon.services.core.alignment.focus import FocusSnapshot
from seeingmon.services.core.alignment.solve import QuickSolution
from seeingmon.services.core.settings import AlignmentSettings
from seeingmon.services.web.contract import (
    AimRingView,
    AlignmentFrameInfo,
    AlignmentState,
    FocusHistoryView,
    FocusView,
    HistogramView,
    LastSolutionView,
    OffsetView,
    ReticleView,
    SaturationView,
    SkyView,
    SolvedView,
    TargetView,
    TimingView,
)
from seeingmon.survey import apparent
from seeingmon.survey.apparent import earth_rotation_angle
from seeingmon.survey.geometry import rot_z
from seeingmon.survey.pointing import polaris_colatitude_deg
from seeingmon.survey.skyview import (
    build_sky_view,
    frame_center,
    reticle_geometry,
    zenith_vector,
)
from seeingmon.survey.tracker import PointingTracker
from seeingmon.survey.wcs_fit import CameraAttitude


@dataclass(frozen=True, slots=True)
class FrameSummary:
    """What the helper measured in one frame.

    `received_ns` is the time of `core` (UTC) when the frame arrived, `preview_s` the time from
    that arrival to the finished preview, and `time_valid` is false when the clock was not
    synchronized when the camera took the frame, so that the frame has no age.
    """

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
    received_ns: int | None = None
    preview_s: float | None = None
    time_valid: bool = True


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
    now_utc_ns: int | None = None,
    solve_elapsed_s: float | None = None,
    solving: tuple[int, float] | None = None,
    last_good: QuickSolution | None = None,
    focus: FocusSnapshot | None = None,
) -> AlignmentState:
    """The state that describes `frame`, with the latest solution and the target.

    `site` is the `[site]` of the configuration. With it, the sky view says how to move the camera
    in altitude and in azimuth, and without it the view gives the image directions only.
    `now_utc_ns` is the time of the state, which gives the age of the frame. `solve_elapsed_s` is
    the time of the latest finished solve, and `solving` is the frame that the solver works on now
    with the seconds that it has worked. `last_good` is the latest solution that found the star
    field, which gives the aim ring while the current solve has none. `focus` is the history of the
    focus values: with it, the focus view carries the best value, the spike flag, and the history
    (`best_fwhm_px` stands in for the best value when there is no history).
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

    solved_view, reason = _current_solution(solution, frame, settings, solving)
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

    aim_ring_view = _aim_ring(frame, solution, solved_view, sky_view, last_good, settings, quality)

    focus_view = _focus_view(frame, solution, best_fwhm_px, focus, quality)

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
        aim_ring=aim_ring_view,
        last_solution=_last_solution_view(frame, last_good),
        timing=_timing(frame, solution, now_utc_ns, solve_elapsed_s, solving),
        quality=quality,
    )


def _arcsec(value_px: float | None, scale_arcsec_px: float | None) -> float | None:
    if value_px is None or scale_arcsec_px is None:
        return None
    return round(value_px * scale_arcsec_px, 4)


def _focus_view(
    frame: FrameSummary,
    solution: QuickSolution | None,
    best_px: float | None,
    snapshot: FocusSnapshot | None,
    quality: dict[str, str],
) -> FocusView | None:
    """The focus view: the value of the latest solve, the best value, and the history.

    The value belongs to the frame of the latest solve. When that frame has no value, the view
    keeps the best value and the history, and `quality["focus"]` says why the value is missing.
    The plate scale of the frame turns pixels into arcseconds.
    """
    value = None if solution is None else solution.focus_fwhm_px
    if value is None:
        quality["focus"] = "no unsaturated stars to measure"
    if snapshot is not None:
        best_px = snapshot.best_px
    if value is None and (snapshot is None or not snapshot.points):
        return None
    scale = frame.plate_scale_arcsec_px or (None if solution is None else solution.scale_arcsec_px)
    point = None
    if value is not None and snapshot is not None and solution is not None:
        point = snapshot.point_for(solution.seq)
    return FocusView(
        fwhm_px=value,
        best_fwhm_px=best_px,
        n_stars=None if value is None or solution is None else solution.n_focus_stars,
        fwhm_arcsec=_arcsec(value, scale),
        best_fwhm_arcsec=_arcsec(best_px, scale),
        spike=False if point is None else point.spike,
        frame_seq=None if value is None or solution is None else solution.seq,
        history=None if snapshot is None or not snapshot.points else _history_view(snapshot, scale),
    )


def _history_view(snapshot: FocusSnapshot, scale_arcsec_px: float | None) -> FocusHistoryView:
    """The points of a snapshot as the parallel lists of the contract."""
    points = snapshot.points
    return FocusHistoryView(
        session=snapshot.session,
        reset=True,
        index=[point.index for point in points],
        seq=[point.seq for point in points],
        t_utc_ms=[point.t_utc_ns // 1_000_000 for point in points],
        fwhm_px=[round(point.fwhm_px, 4) for point in points],
        fwhm_arcsec=[_arcsec(point.fwhm_px, scale_arcsec_px) for point in points],
        n_stars=[point.n_stars for point in points],
        spike=[point.spike for point in points],
    )


def _aim_ring(
    frame: FrameSummary,
    solution: QuickSolution | None,
    solved_view: SolvedView | None,
    sky_view: SkyView | None,
    last_good: QuickSolution | None,
    settings: AlignmentSettings,
    quality: dict[str, str],
) -> AimRingView | None:
    """The aim ring from the current solution, or else from the last good one.

    A current solution gives the ring of its sky view. Without one, the ring comes from the last
    good solution, turned to the time of the frame (`ring_from_solution`).
    """
    ring = None if sky_view is None else sky_view.aim_ring
    if ring is not None and solution is not None and solved_view is not None:
        return AimRingView(
            x_px=ring.x_px,
            y_px=ring.y_px,
            source="current frame",
            age_s=solved_view.age_s,
            solution_frame_seq=solution.seq,
        )
    if last_good is None:
        quality["aim_ring"] = "no solve has found the star field in this alignment yet"
        return None
    if last_good.attitude is None:
        quality["aim_ring"] = "the last solution carries no camera attitude"
        return None
    place = ring_from_solution(last_good, frame.t_utc_ns, frame.width_px, frame.height_px, settings)
    if place is None:
        quality["aim_ring"] = "the pole of the last solution lies behind the camera"
        return None
    return AimRingView(
        x_px=place[0],
        y_px=place[1],
        source="last solution",
        age_s=max(0.0, (frame.t_utc_ns - last_good.t_utc_ns) / NS_PER_S),
        solution_frame_seq=last_good.seq,
    )


def ring_from_solution(
    solution: QuickSolution,
    t_utc_ns: int,
    width_px: int,
    height_px: int,
    settings: AlignmentSettings,
) -> tuple[float, float] | None:
    """Where the aim ring falls at a time, from a solution of an earlier or later frame.

    The camera of a rigid mount is fixed to the Earth, so the solution fixes the attitude at any
    other time: the Earth turns the camera about the pole by the change of the Earth rotation angle
    (`CameraAttitude.rotation @ rot_z(era_then - era_now)`). The ring is the aim plus the vector
    from the pole to Polaris in the picture, as for the sky view. It needs the roll of the picture
    and the time only, so it holds after a move in altitude or azimuth, which translates the
    picture. Returns `None` for a solution without an attitude and when the pole or Polaris lies
    behind the camera.
    """
    attitude = solution.attitude
    if attitude is None:
        return None
    epoch = apparent.epoch_from_utc_ns(t_utc_ns)
    turned = CameraAttitude(
        rotation=attitude.rotation @ rot_z(earth_rotation_angle(solution.t_utc_ns) - epoch.era_rad),
        scale_rad_px=attitude.scale_rad_px,
        parity=attitude.parity,
        center_px=attitude.center_px,
    )
    x, y, front = turned.project(apparent.apparent_vectors_for(apparent.POLARIS, epoch))
    pole = turned.pole_pixel()
    if pole is None or not bool(front[0]):
        return None
    aim = settings.aim_xy or frame_center(width_px, height_px)
    return round(float(x[0]) + aim[0] - pole[0], 3), round(float(y[0]) + aim[1] - pole[1], 3)


def _last_solution_view(
    frame: FrameSummary, last_good: QuickSolution | None
) -> LastSolutionView | None:
    if last_good is None:
        return None
    return LastSolutionView(
        frame_seq=last_good.seq,
        t_utc=utc_ns_to_iso(last_good.t_utc_ns),
        age_s=max(0.0, (frame.t_utc_ns - last_good.t_utc_ns) / NS_PER_S),
        roll_deg=last_good.roll_deg,
        polaris_colatitude_deg=(
            None
            if last_good.polaris_colatitude_deg is None
            else round(last_good.polaris_colatitude_deg, 5)
        ),
        n_matched=last_good.n_matched,
        rms_arcsec=last_good.rms_arcsec,
        solver=last_good.solver[:32],
    )


def _timing(
    frame: FrameSummary,
    solution: QuickSolution | None,
    now_utc_ns: int | None,
    solve_elapsed_s: float | None,
    solving: tuple[int, float] | None,
) -> TimingView:
    """The ages and the frames of one state. An age is `None` when `core` cannot know it."""

    def age_s(t_ns: int | None) -> float | None:
        if t_ns is None or not frame.time_valid:
            return None
        return max(0.0, (t_ns - frame.t_utc_ns) / NS_PER_S)

    return TimingView(
        frame_seq=frame.seq,
        frame_t_utc=utc_ns_to_iso(frame.t_utc_ns),
        frame_age_s=age_s(now_utc_ns),
        receive_lag_s=age_s(frame.received_ns),
        preview_s=frame.preview_s,
        solution_frame_seq=None if solution is None else solution.seq,
        solve_elapsed_s=None if solution is None else solve_elapsed_s,
        solving_frame_seq=None if solving is None else solving[0],
        solving_s=None if solving is None else solving[1],
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
    solution: QuickSolution | None,
    frame: FrameSummary,
    settings: AlignmentSettings,
    solving: tuple[int, float] | None = None,
) -> tuple[SolvedView | None, str | None]:
    """The solved position when it is current, else the reason that it is not."""
    if solution is None:
        if solving is not None:
            return (
                None,
                f"the first solve is running (frame {solving[0]}, {solving[1]:.0f} s so far)",
            )
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
