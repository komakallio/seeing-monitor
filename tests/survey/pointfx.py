"""`pointing` records of a camera that is rigid on the Earth, for the pointing reference tests.

The Earth-fixed attitude of such a camera stays, and the attitude in the frame of date turns with
the Earth rotation angle. `made` builds the record of one frame at a time as the survey pipeline
does (`build_pointing_record`), together with the solution that the fit made, so that a test can
compare what a command rebuilds from the record with what the pipeline had.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seeingmon.clock import NS_PER_S
from seeingmon.records.survey import PointingRecord
from seeingmon.survey import apparent
from seeingmon.survey import pointing as pt
from seeingmon.survey.geometry import ARCMIN_PER_RAD, ARCSEC_PER_RAD, FloatArray, exp_so3
from seeingmon.survey.wcs_fit import CameraAttitude, pixel_center
from tests.survey import synth

T0 = synth.NIGHT_UTC_NS
MINUTE_NS = 60 * NS_PER_S
HOUR_NS = 60 * MINUTE_NS
PROFILE = synth.reference_profile()
BIN2 = PROFILE.mode("bin2")
CENTER = pixel_center(BIN2.width_px, BIN2.height_px)
SCALE_RAD = 3.82 / ARCSEC_PER_RAD
STATION = "private-station-name"


@dataclass(frozen=True)
class Made:
    """A `pointing` record and the solution that the survey fit made for it."""

    record: PointingRecord
    solution: pt.PointingSolution


def mount(polar_distance_deg: float = 0.4) -> FloatArray:
    """The Earth-fixed attitude of a camera that does not move: the roll is 25 degrees."""
    return synth.make_attitude(polar_distance_deg, 40.0, 25.0)


def tilted(rotation_tirs: FloatArray, arcmin: float) -> FloatArray:
    """The mount after a tilt of the camera by `arcmin` about its x axis."""
    return np.asarray(exp_so3([arcmin / ARCMIN_PER_RAD, 0.0, 0.0]) @ rotation_tirs)


def made(
    t_utc_ns: int,
    *,
    rotation_tirs: FloatArray | None = None,
    parity: int = 1,
    solved: bool = True,
    n_matched: int = 300,
    rms_arcsec: float | None = 0.4,
    time_invalid: bool = False,
    reference: pt.ReferenceSolution | None = None,
    solver: str = "astrometry.net",
    mode: str = "bin2",
    station_id: str = STATION,
) -> Made:
    """The record that the survey path writes for a frame, as the pipeline builds it."""
    rotation = mount() if rotation_tirs is None else rotation_tirs
    epoch = apparent.epoch_from_utc_ns(t_utc_ns)
    attitude = CameraAttitude(
        rotation @ apparent.cirs_to_earth_fixed(epoch.era_rad), SCALE_RAD, parity, CENTER
    )
    solution = pt.PointingSolution.from_attitude(
        attitude,
        epoch,
        mode=mode,
        width_px=BIN2.width_px,
        height_px=BIN2.height_px,
        n_matched=n_matched,
        rms_arcsec=rms_arcsec,
        solver=solver,
    )
    record = pt.build_pointing_record(
        station_id=station_id,
        profile_id=PROFILE.id,
        t_utc_ns=t_utc_ns,
        mode=mode,
        solver=solver if solved else "none",
        provenance={"algo": pt.POINTING_ALGORITHM},
        attitude=attitude if solved else None,
        epoch=epoch if solved else None,
        solution=solution if solved else None,
        n_matched=n_matched if solved else 0,
        rms_arcsec=rms_arcsec if solved else None,
        polaris_xy=solution.polaris_pixel(t_utc_ns) if solved else None,
        reference=reference,
        time_invalid=time_invalid,
    )
    return Made(record, solution)
