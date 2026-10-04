"""The pointing reference: find a good solution in the store, and rebuild it for the reference file.

The Pointing card shows the offset of the camera from a reference solution, which the file that
`[survey.pointing] reference_file` names holds (`seeingmon.survey.pointing.ReferenceSolution`).
Nothing in the survey path writes that file. After you align the camera,
`seeingmon pointing set-reference` (`seeingmon.survey.pointing_cli`) takes the newest good
`pointing` record of the store and saves it as the reference.

**Which record.** `newest_solution_record` takes the newest `pointing` record that has a solution
(`solver` is not `none`), at least `min_matched` matched stars, a finite residual, and no
`time_invalid` flag, and that is not older than `max_age_s`. The Earth-fixed attitude of a
reference depends on the time of the record, so a record from a clock that was not synchronized
cannot serve.

**From a record to a solution.** A `pointing` record holds the rotation `R_cirs` of the frame at
the time of the record, the plate scale, the readout mode, the matched stars, and the residual
(see `seeingmon.survey.pointing`). `solution_from_record` rebuilds the `PointingSolution` of the
fit from them:

- The frame size and the principal point are those of the readout mode in the profile, as
  `SurveyPipeline` sets them.
- The record does not hold the parity of the image. The function takes the parity that reproduces
  the position of Polaris and the roll in the record. When both parities fit, because Polaris
  and the pole lie on the middle row of the frame, it takes +1, an image that no mirror flips. The
  offset from a reference does not use the parity.
- The Earth-fixed attitude uses `dut1_s` from `[survey]`, as the pipeline does.

The function checks its own result: the model that it builds must put Polaris where the record says,
to 0.01 pixel. A record that fails the check came from another profile, and the function refuses it.

**Seeding the tracker.** `seed_solution` gives `core` the solution that starts a `PointingTracker`
after a restart. It takes the newest record that `newest_solution_record` accepts within the
validity limit of the tracker (a solution older than that predicts nothing), at least the stars and
at most the residual that the analyzer asks of a solution that updates the tracker, and no record
from the future, which a clock that stepped back would give. The first solve then starts from where
Polaris was, and not from a blind search around the pole.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import numpy as np

from seeingmon.clock import NS_PER_S, utc_ns_to_iso
from seeingmon.profile import ProfileError
from seeingmon.records.survey import PointingRecord
from seeingmon.survey import apparent
from seeingmon.survey.geometry import ARCSEC_PER_RAD, FloatArray, nearest_rotation
from seeingmon.survey.pointing import PointingSolution, ReferenceSolution
from seeingmon.survey.wcs_fit import CameraAttitude, pixel_center

if TYPE_CHECKING:
    from seeingmon.profile import Profile
    from seeingmon.store.db import StoreReader

REFERENCE_FILENAME = "pointing-reference.json"
# The ID of a reference goes into every `pointing` record and from there to the sinks.
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

# How well the rebuilt model has to reproduce the record: Polaris in pixels, and the roll in
# degrees. The values that the pipeline stores come from the same model, so they agree to 1e-9.
POLARIS_TOLERANCE_PX = 0.01
ROLL_TOLERANCE_DEG = 0.01
ROTATION_TOLERANCE = 1e-6
# The most records that one search reads, newest first: about ten days of a survey cadence.
SCAN_LIMIT = 10_000
_END_OF_TIME_NS = 2**63 - 1  # the largest integer that SQLite stores
# A record this far ahead of the clock means that the clock stepped back since the record.
FUTURE_TOLERANCE_NS = 60 * NS_PER_S
_OPTION_HINT = re.compile(r" \(see --[a-z-]+\)")


class PointingReferenceError(Exception):
    """A pointing record cannot become a reference solution. The message says why, in one line."""


class NoSolutionError(PointingReferenceError):
    """The store holds no pointing solution that fits the limits."""


def default_reference_id(t_utc_ns: int) -> str:
    """The default ID of a reference: `reference-` and the UTC time of its solution."""
    stamp = utc_ns_to_iso(t_utc_ns, digits=0).replace("-", "").replace(":", "")
    return f"reference-{stamp}"


def is_valid_reference_id(text: str) -> bool:
    """Whether `text` can be the ID of a reference.

    An ID has 1 to 64 letters, digits, `.`, `_`, or `-`, and it starts with a letter or digit.
    """
    return ID_PATTERN.fullmatch(text) is not None


def format_time(t_utc_ns: int) -> str:
    """A UTC time to the second, such as `2026-10-04T18:38:52Z`."""
    return utc_ns_to_iso(t_utc_ns, digits=0)


def format_number(value: float, digits: int) -> str:
    """A number with `digits` decimals, where a value that rounds to zero has no minus sign."""
    text = f"{value:.{digits}f}"
    return text.removeprefix("-") if float(text) == 0.0 else text


def format_age(seconds: float) -> str:
    """A short age: `40 s`, `12 min`, `3.5 h`, or `2.0 days`."""
    seconds = max(seconds, 0.0)
    if seconds < 90.0:
        return f"{seconds:.0f} s"
    if seconds < 90.0 * 60.0:
        return f"{seconds / 60.0:.0f} min"
    if seconds < 48.0 * 3600.0:
        return f"{seconds / 3600.0:.1f} h"
    return f"{seconds / 86400.0:.1f} days"


def _minutes(minutes: float) -> str:
    return "1 minute" if minutes == 1.0 else f"{minutes:g} minutes"


# --- Finding the record ------------------------------------------------------------------------


def newest_solution_record(
    reader: StoreReader, *, now_utc_ns: int, min_matched: int, max_age_s: float | None
) -> PointingRecord:
    """The newest `pointing` record that can serve as a reference.

    The record has a solution, at least `min_matched` matched stars, a finite residual, and no
    `time_invalid` flag, and its time is within `max_age_s` of `now_utc_ns` (`None` sets no limit).
    The function reads one snapshot of the store, so it is safe while `core` writes. It raises
    `NoSolutionError` with a one-line reason when no record fits.
    """
    max_age_ns = None if max_age_s is None else max_age_s * NS_PER_S  # a float, which can be inf
    start_ns = (
        0 if max_age_ns is None or max_age_ns >= now_utc_ns else now_utc_ns - round(max_age_ns)
    )
    with reader.snapshot() as snapshot:
        rows = snapshot.range("pointing", start_ns, _END_OF_TIME_NS, SCAN_LIMIT, descending=True)
        latest = None if rows else snapshot.latest("pointing")
    if max_age_s is None:
        window, limit = "the store", ""
    else:
        limit = _minutes(max_age_s / 60.0)
        window = f"the last {limit}"
    if not rows:
        if latest is None:
            raise NoSolutionError("the store holds no pointing record")
        age = format_age((now_utc_ns - int(latest.values["t_utc_ns"])) / NS_PER_S)
        raise NoSolutionError(
            f"the newest pointing record is {age} old, and the limit is {limit} (see --max-age-min)"
        )
    untimed = thin = unmeasured = 0
    most = 0
    for row in rows:
        values = row.values
        if (
            values["solver"] == "none"
            or values["attitude"] is None
            or values["plate_scale_arcsec_px"] is None
        ):
            continue
        if "time_invalid" in values["flags"]:
            untimed += 1
            continue
        matched = int(values["n_matched"])
        most = max(most, matched)
        if matched < min_matched:
            thin += 1
            continue
        rms = values["solve_rms_arcsec"]
        if rms is None or not math.isfinite(float(rms)):
            unmeasured += 1
            continue
        return PointingRecord.from_row(values)
    # Name the check that the best record got furthest through.
    if unmeasured:
        raise NoSolutionError(f"no pointing solution of {window} has a finite residual")
    if thin:
        raise NoSolutionError(
            f"no pointing solution of {window} has {min_matched} matched stars: "
            f"the best has {most} (see --min-matched)"
        )
    if untimed:
        raise NoSolutionError(f"every pointing solution of {window} has the time_invalid flag")
    raise NoSolutionError(f"no pointing record of {window} has a solution")


# --- Rebuilding the solution -------------------------------------------------------------------


def _is_rotation(matrix: FloatArray) -> bool:
    return bool(
        np.all(np.isfinite(matrix))
        and np.allclose(matrix @ matrix.T, np.eye(3), atol=ROTATION_TOLERANCE)
        and abs(float(np.linalg.det(matrix)) - 1.0) < ROTATION_TOLERANCE
    )


def _reproduces(model: CameraAttitude, record: PointingRecord, polaris: FloatArray) -> bool:
    """Whether the model puts Polaris and the pole where the record says."""
    if record.polaris_x_px is not None and record.polaris_y_px is not None:
        x, y, front = model.project(polaris)
        if not front[0]:
            return False
        gap = math.hypot(float(x[0]) - record.polaris_x_px, float(y[0]) - record.polaris_y_px)
        if gap > POLARIS_TOLERANCE_PX:
            return False
    if record.roll_deg is not None:
        roll = model.roll_deg()
        if roll is None:
            return False
        if abs((roll - record.roll_deg + 180.0) % 360.0 - 180.0) > ROLL_TOLERANCE_DEG:
            return False
    return True


def solution_from_record(
    record: PointingRecord, profile: Profile, *, dut1_s: float = 0.0
) -> PointingSolution:
    """The `PointingSolution` that the survey fit made, rebuilt from its `pointing` record.

    `profile` gives the frame size and the principal point of the readout mode, and `dut1_s` is the
    `[survey]` setting that the pipeline used. Raises `PointingReferenceError` when the record
    has no solution, when its attitude is not a rotation, when the profile does not know its
    readout mode, or when the model that the function builds does not put Polaris where the record
    says (the record comes from another profile).
    """
    if record.solver == "none" or record.attitude is None or record.plate_scale_arcsec_px is None:
        raise PointingReferenceError("the pointing record has no solution")
    try:
        readout = profile.mode(record.readout_mode)
    except ProfileError as exc:
        raise PointingReferenceError(
            f"the pointing record does not fit the profile: {exc}"
        ) from None
    matrix = np.asarray(record.attitude, dtype=np.float64).reshape(3, 3)
    if not _is_rotation(matrix):
        raise PointingReferenceError("the attitude of the pointing record is not a rotation")
    rotation = nearest_rotation(matrix)
    scale_rad_px = record.plate_scale_arcsec_px / ARCSEC_PER_RAD
    center = pixel_center(readout.width_px, readout.height_px)
    epoch = apparent.epoch_from_utc_ns(record.t_utc_ns, dut1_s)
    polaris = apparent.apparent_vectors_for(apparent.POLARIS, epoch)
    for parity in (1, -1):  # +1 first: it wins when the record cannot tell the two apart
        attitude = CameraAttitude(rotation, scale_rad_px, parity, center)
        if _reproduces(attitude, record, polaris):
            break
    else:
        raise PointingReferenceError(
            "the pointing record does not fit the profile of the configuration: the model that "
            "the profile gives puts Polaris away from the position in the record"
        )
    return PointingSolution.from_attitude(
        attitude,
        epoch,
        mode=record.readout_mode,
        width_px=readout.width_px,
        height_px=readout.height_px,
        n_matched=record.n_matched,
        rms_arcsec=record.solve_rms_arcsec,
        solver=record.solver,
    )


# --- Seeding the tracker -----------------------------------------------------------------------


def seed_solution(
    reader: StoreReader,
    profile: Profile,
    *,
    now_utc_ns: int,
    max_age_s: float,
    min_matched: int,
    max_rms_px: float,
    dut1_s: float = 0.0,
) -> PointingSolution:
    """The newest stored solution that can start a pointing tracker.

    The record is the newest that `newest_solution_record` accepts: solved, timed, finite, with at
    least `min_matched` stars, and not older than `max_age_s` before `now_utc_ns`. Its residual
    must not exceed `max_rms_px`, and its time must not lie more than a minute ahead of the clock.
    Raises `PointingReferenceError` with a one-line reason when no record fits, or when the
    newest one cannot be rebuilt (see `solution_from_record`).
    """
    try:
        record = newest_solution_record(
            reader, now_utc_ns=now_utc_ns, min_matched=min_matched, max_age_s=max_age_s
        )
    except NoSolutionError as error:  # the hints name options of the command, not of `core`
        raise NoSolutionError(_OPTION_HINT.sub("", str(error))) from None
    if record.t_utc_ns > now_utc_ns + FUTURE_TOLERANCE_NS:
        raise PointingReferenceError(
            "the newest pointing solution is from "
            f"{format_time(record.t_utc_ns)}, which is ahead of the clock"
        )
    rms_arcsec, scale = record.solve_rms_arcsec, record.plate_scale_arcsec_px
    if rms_arcsec is None or scale is None:  # the search has ruled this out
        raise PointingReferenceError("the newest pointing solution has no residual")
    rms_px = rms_arcsec / scale
    if rms_px > max_rms_px:
        raise PointingReferenceError(
            f"the newest pointing solution has a residual of {rms_px:.2f} px, "
            f"and the limit is {max_rms_px:g} px"
        )
    return solution_from_record(record, profile, dut1_s=dut1_s)


# --- Describing a reference --------------------------------------------------------------------


def angle_from_pole_deg(solution: PointingSolution) -> float:
    """The angle between the camera center (the boresight) and the celestial pole, in degrees."""
    boresight = solution.boresight_earth_fixed()
    return float(np.degrees(np.arctan2(np.hypot(boresight[0], boresight[1]), boresight[2])))


def summary_rows(
    reference: ReferenceSolution, *, now_utc_ns: int | None = None
) -> list[tuple[str, str]]:
    """The label and the value of each line that describes a reference.

    With `now_utc_ns`, the time of the solution also says how long ago it was.
    """
    solution = reference.solution
    when = format_time(solution.t_utc_ns)
    if now_utc_ns is not None:
        when += f" ({format_age((now_utc_ns - solution.t_utc_ns) / NS_PER_S)} ago)"
    rms = solution.rms_arcsec
    roll = solution.attitude_at(solution.t_utc_ns).roll_deg()
    scale = solution.scale_rad_px * ARCSEC_PER_RAD
    return [
        ("reference ID", reference.reference_id),
        ("solution time", when),
        ("matched stars", str(solution.n_matched)),
        ("residual", "not recorded" if rms is None else f"{rms:.2f} arcsec rms"),
        (
            "roll",
            "undefined (the pole is at the center)"
            if roll is None
            else f"{format_number(roll, 2)} degrees",
        ),
        ("plate scale", f"{scale:.3f} arcsec/px in {solution.mode}"),
        ("center from the pole", f"{angle_from_pole_deg(solution):.3f} degrees"),
    ]
