"""Pointing solutions: an attitude fixed to the Earth, offsets, references, and records.

A camera that is rigid on the Earth has a constant attitude in an Earth-fixed frame. In the
apparent frame (CIRS) that attitude turns about the pole with the Earth rotation angle. A
`PointingSolution` stores the Earth-fixed attitude, so one solution predicts the attitude at
any other time, and it describes how far the mount has moved by comparing two solutions.

- `PointingSolution.attitude_at(t)` gives the camera model at time `t`: `R_cirs(t) = R_earth_fixed
  @ rot_z(-ERA(t))`.
- `PointingSolution.polaris_pixel(t)` projects the apparent place of Polaris through it.
- `offset_between` returns the boresight offset in arcminutes and the roll change in degrees.
- `ReferenceSolution` is a solution with an ID, saved as JSON at commissioning, and later
  solutions flag a move against it.
- `build_pointing_record` makes the `PointingRecord` of a fit, or the record of a failed solve.

**The attitude in the record.** `PointingRecord.attitude` holds nine numbers, row by row: the
rotation matrix `R_cirs` of the frame at the time of the record, with `camera = R @ cirs`. CIRS
is the frame of date (true pole and the celestial intermediate origin). To express the attitude
in ICRS, multiply by the bias-precession-nutation matrix of that time (`R_icrs = R_cirs @ NPB`).
The record's `center_ra_deg` and `center_dec_deg` are already in ICRS, with aberration removed.
The Earth rotation angle uses UT1 = UTC, which can shift the roll of the Earth-fixed attitude by
up to 14 arcsec when UT1 - UTC is 0.9 s. See `seeingmon.survey.apparent`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from seeingmon.records.survey import PointingRecord
from seeingmon.survey import apparent
from seeingmon.survey.apparent import ObservationEpoch
from seeingmon.survey.geometry import (
    ARCMIN_PER_RAD,
    FloatArray,
    angular_separation,
    nearest_rotation,
    rot_z,
)
from seeingmon.survey.wcs_fit import CameraAttitude, FitResult

POINTING_ALGORITHM = "survey-1"
SOLUTION_FORMAT = 1
_Z = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True, slots=True, eq=False)
class PointingSolution:
    """The camera attitude fixed to the Earth, with the readout mode it belongs to.

    `rotation_earth_fixed` maps Earth-fixed unit vectors (TIRS) to camera vectors. `t_utc_ns`
    is the time of the frame that gave the solution. The pixel geometry (`width_px`, `height_px`,
    and the principal point) belongs to `mode`.
    """

    rotation_earth_fixed: FloatArray
    scale_rad_px: float
    parity: int
    mode: str
    width_px: int
    height_px: int
    center_px: tuple[float, float]
    t_utc_ns: int
    dut1_s: float = 0.0
    n_matched: int = 0
    rms_arcsec: float | None = None
    solver: str = ""

    @classmethod
    def from_attitude(
        cls,
        attitude: CameraAttitude,
        epoch: ObservationEpoch,
        *,
        mode: str,
        width_px: int,
        height_px: int,
        n_matched: int = 0,
        rms_arcsec: float | None = None,
        solver: str = "",
    ) -> PointingSolution:
        """The solution for an apparent-frame attitude that a fit found at `epoch`."""
        return cls(
            rotation_earth_fixed=np.asarray(
                attitude.rotation @ rot_z(epoch.era_rad), dtype=np.float64
            ),
            scale_rad_px=attitude.scale_rad_px,
            parity=attitude.parity,
            mode=mode,
            width_px=width_px,
            height_px=height_px,
            center_px=attitude.center_px,
            t_utc_ns=epoch.t_utc_ns,
            dut1_s=epoch.dut1_s,
            n_matched=n_matched,
            rms_arcsec=rms_arcsec,
            solver=solver,
        )

    def attitude_at(self, t_utc_ns: int) -> CameraAttitude:
        """The camera model in the apparent frame at a time. The Earth has turned since."""
        era = apparent.earth_rotation_angle(t_utc_ns, self.dut1_s)
        return CameraAttitude(
            rotation=np.asarray(self.rotation_earth_fixed @ rot_z(-era), dtype=np.float64),
            scale_rad_px=self.scale_rad_px,
            parity=self.parity,
            center_px=self.center_px,
        )

    def boresight_earth_fixed(self) -> FloatArray:
        """The boresight as an Earth-fixed unit vector. It stays constant for a rigid mount."""
        return np.asarray(self.rotation_earth_fixed.T @ _Z, dtype=np.float64)

    def polaris_pixel(self, t_utc_ns: int) -> tuple[float, float] | None:
        """Where the apparent place of Polaris falls at a time, in pixels of `mode`.

        Returns `None` when Polaris lies behind the camera.
        """
        epoch = apparent.epoch_from_utc_ns(t_utc_ns, self.dut1_s)
        vector = apparent.apparent_vectors_for(apparent.POLARIS, epoch)
        x, y, front = self.attitude_at(t_utc_ns).project(vector)
        return (float(x[0]), float(y[0])) if front[0] else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SOLUTION_FORMAT,
            "rotation_earth_fixed": [float(v) for v in self.rotation_earth_fixed.reshape(-1)],
            "scale_rad_px": self.scale_rad_px,
            "parity": self.parity,
            "mode": self.mode,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "center_px": list(self.center_px),
            "t_utc_ns": self.t_utc_ns,
            "dut1_s": self.dut1_s,
            "n_matched": self.n_matched,
            "rms_arcsec": self.rms_arcsec,
            "solver": self.solver,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PointingSolution:
        if data.get("format") != SOLUTION_FORMAT:
            raise ValueError(f"unsupported solution format {data.get('format')!r}")
        matrix = np.array(data["rotation_earth_fixed"], dtype=np.float64).reshape(3, 3)
        return cls(
            rotation_earth_fixed=nearest_rotation(matrix),
            scale_rad_px=float(data["scale_rad_px"]),
            parity=int(data["parity"]),
            mode=str(data["mode"]),
            width_px=int(data["width_px"]),
            height_px=int(data["height_px"]),
            center_px=(float(data["center_px"][0]), float(data["center_px"][1])),
            t_utc_ns=int(data["t_utc_ns"]),
            dut1_s=float(data.get("dut1_s", 0.0)),
            n_matched=int(data.get("n_matched", 0)),
            rms_arcsec=None if data.get("rms_arcsec") is None else float(data["rms_arcsec"]),
            solver=str(data.get("solver", "")),
        )


@dataclass(frozen=True, slots=True)
class PointingOffset:
    """How far a solution lies from a reference: the boresight and the roll."""

    boresight_arcmin: float
    roll_deg: float


def offset_between(solution: PointingSolution, reference: PointingSolution) -> PointingOffset:
    """The offset of `solution` from `reference`, both in the Earth-fixed frame.

    The boresight offset is the angle between the two boresight directions. The roll offset is
    the twist of the camera about its own axis (positive when the sky turns counterclockwise in
    the image), which stays meaningful when the boresight is on the pole.
    """
    separation = float(
        angular_separation(solution.boresight_earth_fixed(), reference.boresight_earth_fixed())
    )
    relative = solution.rotation_earth_fixed @ reference.rotation_earth_fixed.T
    twist = np.arctan2(relative[1, 0] - relative[0, 1], relative[0, 0] + relative[1, 1])
    return PointingOffset(
        boresight_arcmin=separation * ARCMIN_PER_RAD, roll_deg=float(np.degrees(twist))
    )


@dataclass(frozen=True, slots=True, eq=False)
class ReferenceSolution:
    """A solution that commissioning fixed as the reference, with an ID that records name."""

    reference_id: str
    solution: PointingSolution

    def to_json(self) -> str:
        return json.dumps(
            {"reference_id": self.reference_id, "solution": self.solution.to_dict()}, indent=2
        )

    @classmethod
    def from_json(cls, text: str) -> ReferenceSolution:
        data = json.loads(text)
        return cls(str(data["reference_id"]), PointingSolution.from_dict(data["solution"]))


def save_reference(path: str | os.PathLike[str], reference: ReferenceSolution) -> None:
    """Write the reference as JSON. The file appears whole or not at all."""
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(reference.to_json(), encoding="utf-8")
    os.replace(temporary, target)


def load_reference(path: str | os.PathLike[str]) -> ReferenceSolution | None:
    """Read the reference. Returns `None` when the file does not exist."""
    source = Path(path)
    if not source.is_file():
        return None
    return ReferenceSolution.from_json(source.read_text(encoding="utf-8"))


# --- Records -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PointingLimits:
    """Thresholds that turn a fit into flags."""

    few_stars: int = 12  # fewer matched stars flag `few_stars`
    moved_arcmin: float = 5.0  # a larger boresight offset from the reference flags `moved`
    moved_roll_deg: float = 0.5  # so does a larger roll change


def pointing_flags(
    *,
    solved: bool,
    n_matched: int,
    roll_defined: bool,
    offset: PointingOffset | None,
    time_invalid: bool,
    limits: PointingLimits,
) -> list[str]:
    """The documented flag codes for a solution."""
    flags: list[str] = []
    if not solved:
        flags.append("unsolved")
    else:
        if n_matched < limits.few_stars:
            flags.append("few_stars")
        if not roll_defined:
            flags.append("roll_undefined")
        if offset is not None and (
            offset.boresight_arcmin > limits.moved_arcmin
            or abs(offset.roll_deg) > limits.moved_roll_deg
        ):
            flags.append("moved")
    if time_invalid:
        flags.append("time_invalid")
    return flags


def attitude_row_major(attitude: CameraAttitude) -> list[float]:
    """The nine numbers of the rotation matrix, row by row, for the record."""
    return [float(v) for v in attitude.rotation.reshape(-1)]


def fit_summary(fit: FitResult | None) -> dict[str, Any]:
    """The fit numbers that the record needs, with `None` for a failed fit."""
    if fit is None:
        return {"n_matched": 0, "rms_arcsec": None}
    return {"n_matched": fit.n_matched, "rms_arcsec": fit.rms_arcsec}


def build_pointing_record(
    *,
    station_id: str,
    profile_id: str,
    t_utc_ns: int,
    mode: str,
    solver: str,
    provenance: dict[str, str],
    attitude: CameraAttitude | None,
    epoch: ObservationEpoch | None,
    solution: PointingSolution | None,
    n_matched: int = 0,
    rms_arcsec: float | None = None,
    focus_fwhm_px: float | None = None,
    solve_time_s: float | None = None,
    polaris_xy: tuple[float, float] | None = None,
    reference: ReferenceSolution | None = None,
    time_invalid: bool = False,
    limits: PointingLimits | None = None,
) -> PointingRecord:
    """The `pointing` record of one survey frame.

    With `attitude` set, the record carries the geometry. With `attitude` `None`, the record
    has the `unsolved` flag, `null` in every geometry field, and a `quality` entry for each
    that says why. `solution` is the Earth-fixed form of `attitude`, which the offset from the
    reference needs.
    """
    cap = limits or PointingLimits()
    if attitude is None or epoch is None:
        flags = pointing_flags(
            solved=False,
            n_matched=0,
            roll_defined=False,
            offset=None,
            time_invalid=time_invalid,
            limits=cap,
        )
        missing = (
            "center_ra_deg",
            "center_dec_deg",
            "roll_deg",
            "plate_scale_arcsec_px",
            "attitude",
            "offset_arcmin",
            "solve_rms_arcsec",
            "polaris_x_px",
            "polaris_y_px",
        )
        return PointingRecord(
            station_id=station_id,
            t_utc_ns=t_utc_ns,
            profile_id=profile_id,
            provenance=provenance,
            quality=dict.fromkeys(missing, "no pointing solution"),
            n_matched=0,
            focus_fwhm_px=focus_fwhm_px,
            readout_mode=mode,
            solver=solver,
            solve_time_s=solve_time_s,
            reference_id=None if reference is None else reference.reference_id,
            flags=flags,
        )
    ra, dec = attitude.center_icrs(epoch)
    roll = attitude.roll_deg()
    offset = (
        offset_between(solution, reference.solution)
        if solution is not None and reference is not None
        else None
    )
    flags = pointing_flags(
        solved=True,
        n_matched=n_matched,
        roll_defined=roll is not None,
        offset=offset,
        time_invalid=time_invalid,
        limits=cap,
    )
    quality: dict[str, str] = {}
    if roll is None:
        quality["roll_deg"] = "the field center is on the pole"
    if reference is None:
        quality["offset_arcmin"] = "no reference solution"
    if polaris_xy is None:
        quality["polaris_x_px"] = "Polaris is not in front of the camera"
        quality["polaris_y_px"] = "Polaris is not in front of the camera"
    return PointingRecord(
        station_id=station_id,
        t_utc_ns=t_utc_ns,
        profile_id=profile_id,
        provenance=provenance,
        quality=quality or None,
        center_ra_deg=ra,
        center_dec_deg=dec,
        roll_deg=roll,
        plate_scale_arcsec_px=attitude.scale_arcsec_px,
        attitude=attitude_row_major(attitude),
        offset_arcmin=None if offset is None else offset.boresight_arcmin,
        solve_rms_arcsec=rms_arcsec,
        n_matched=n_matched,
        focus_fwhm_px=focus_fwhm_px,
        polaris_x_px=None if polaris_xy is None else polaris_xy[0],
        polaris_y_px=None if polaris_xy is None else polaris_xy[1],
        readout_mode=mode,
        solver=solver,
        solve_time_s=solve_time_s,
        reference_id=None if reference is None else reference.reference_id,
        flags=flags,
    )
