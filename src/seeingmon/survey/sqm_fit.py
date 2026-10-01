"""Fit the offset between the SQM-LE readings and the survey's sky brightness.

The SQM-LE sits on the roof and points about 45 degrees up to the north through a plastic dome.
The survey camera looks at the pole at the altitude of the site's latitude, without a dome. The
two instruments differ in four ways, and the fit separates them:

- **A band offset.** The SQM band (a TSL237 with a Hoya CM-500 filter) is not V, and the
  difference depends on the spectrum of the sky. One number, `offset_mag`, covers its mean.
- **The dome loss.** The dome absorbs light, so the fixed SQM reads fainter than a meter without
  a dome. Only handheld readings outside the dome (the `manual` source) tell the dome loss from
  the band offset. Without them the two cannot be told apart, and the fit reports their sum as
  the offset.
- **The altitude difference.** The sky is brighter nearer to the horizon, roughly in proportion to
  the airmass. The term `k * (X_reference - X_camera)` carries it, where `X` is the airmass
  of Kasten and Young (1989) and `k` is the fitted change of the sky magnitude per airmass.
  Readings at one altitude cannot fix `k`. The fit then drops the term.
- **Noise.** The moon, clouds, and dew spoil some pairs, so the fit clips pairs beyond 3 sigma.

For each reading the model is

    reading - ours = offset + dome * [reading is fixed] + k * (X_reference - X_camera)

`ours` is the survey's V-equivalent sky brightness (`sky_mag_arcsec2_v`) at the time of the
reading. The fit gives `offset_mag`, the number for `[survey.sky] sqm_offset_mag`.

The real fit waits for commissioning: it needs weeks of readings and handheld measurements. The
tests here run it on synthetic readings with a known offset, dome loss, and altitude term.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S
from seeingmon.records.reference import ReferenceRecord
from seeingmon.records.survey import SkyQualityRecord

FloatArray = npt.NDArray[np.float64]
FIXED = "fixed"
MANUAL = "manual"


def airmass(altitude_deg: float) -> float:
    """The relative airmass at an altitude above the horizon (Kasten and Young, 1989)."""
    h = min(max(altitude_deg, 0.0), 90.0)
    return 1.0 / (math.sin(math.radians(h)) + 0.50572 * math.pow(h + 6.07995, -1.6364))


@dataclass(frozen=True, slots=True)
class SqmPair:
    """A reading of the reference meter and the survey's sky brightness at the same time."""

    t_utc_ns: int
    reading_mag: float
    ours_mag: float
    source: str
    altitude_deg: float


def pair_readings(
    references: Sequence[ReferenceRecord],
    sky: Sequence[SkyQualityRecord],
    *,
    max_gap_s: float = 600.0,
    default_altitude_deg: float | None = None,
) -> list[SqmPair]:
    """Pair each reading with the nearest survey sky brightness (V equivalent) in time.

    A reading with no sky record within `max_gap_s`, and a reading with no altitude when
    `default_altitude_deg` is `None`, stay out.
    """
    stamped = sorted(
        (record.t_utc_ns, record.sky_mag_arcsec2_v)
        for record in sky
        if record.sky_mag_arcsec2_v is not None
    )
    times = [item[0] for item in stamped]
    pairs: list[SqmPair] = []
    limit_ns = round(max_gap_s * NS_PER_S)
    for reading in references:
        altitude = (
            reading.altitude_deg if reading.altitude_deg is not None else default_altitude_deg
        )
        if altitude is None or not times:
            continue
        position = bisect.bisect_left(times, reading.t_utc_ns)
        candidates = [i for i in (position - 1, position) if 0 <= i < len(times)]
        best = min(candidates, key=lambda i: abs(times[i] - reading.t_utc_ns))
        if abs(times[best] - reading.t_utc_ns) > limit_ns:
            continue
        ours = stamped[best][1]
        assert ours is not None
        pairs.append(
            SqmPair(reading.t_utc_ns, reading.value_mag_arcsec2, ours, reading.source, altitude)
        )
    return pairs


@dataclass(frozen=True, slots=True)
class SqmFit:
    """The fitted offsets. An error field is the 1-sigma sampling error of its value.

    `offset_mag` is for readings outside the dome at the camera's altitude, and it is the number
    for `sqm_offset_mag`. `dome_loss_mag` and `altitude_mag_per_airmass` are `None` when the
    readings cannot fix them.
    """

    offset_mag: float
    offset_error_mag: float
    dome_loss_mag: float | None
    dome_loss_error_mag: float | None
    altitude_mag_per_airmass: float | None
    altitude_error_mag_per_airmass: float | None
    rms_mag: float
    n_fixed: int
    n_manual: int
    n_used: int


def fit_pairs(
    pairs: Sequence[SqmPair],
    *,
    camera_altitude_deg: float,
    applied_offset_mag: float = 0.0,
    min_pairs: int = 10,
    clip_sigma: float = 3.0,
    min_airmass_spread: float = 0.05,
) -> SqmFit | None:
    """Fit the offset, the dome loss, and the altitude term to pairs of readings.

    `camera_altitude_deg` is the altitude of the survey camera (the latitude of the site, where
    Polaris stands). `applied_offset_mag` is the offset that the survey's sky values already
    include, and the result is the total offset to configure. The function returns `None` with
    fewer than `min_pairs` pairs.
    """
    if len(pairs) < min_pairs:
        return None
    difference = np.array([p.reading_mag - p.ours_mag for p in pairs])
    fixed = np.array([p.source == FIXED for p in pairs])
    delta_x = np.array([airmass(p.altitude_deg) - airmass(camera_altitude_deg) for p in pairs])
    use = np.ones(len(pairs), dtype=bool)
    solution: FloatArray = np.zeros(1)
    covariance: FloatArray = np.zeros((1, 1))
    has_dome = has_altitude = False
    for _ in range(6):
        has_dome = bool(fixed[use].any() and (~fixed[use]).any())
        has_altitude = bool(np.ptp(delta_x[use]) >= min_airmass_spread)
        columns = [np.ones(int(use.sum()))]
        if has_dome:
            columns.append(fixed[use].astype(np.float64))
        if has_altitude:
            columns.append(delta_x[use])
        design = np.column_stack(columns)
        if design.shape[0] <= design.shape[1]:
            return None
        solution, *_ = np.linalg.lstsq(design, difference[use], rcond=None)
        residual = difference[use] - design @ solution
        dof = design.shape[0] - design.shape[1]
        scale2 = float(residual @ residual) / dof
        covariance = scale2 * np.linalg.inv(design.T @ design)
        # Clip against the residual of every pair, with the same columns.
        full = [np.ones(len(pairs))]
        if has_dome:
            full.append(fixed.astype(np.float64))
        if has_altitude:
            full.append(delta_x)
        all_residual = difference - np.column_stack(full) @ solution
        spread = 1.4826 * float(np.median(np.abs(all_residual[use] - np.median(all_residual[use]))))
        keep = np.abs(all_residual) <= clip_sigma * max(spread, 1e-6)
        if np.array_equal(keep, use):
            break
        use = keep
    index = 1
    dome = dome_error = k = k_error = None
    if has_dome:
        dome, dome_error = float(solution[index]), float(np.sqrt(covariance[index, index]))
        index += 1
    if has_altitude:
        k, k_error = float(solution[index]), float(np.sqrt(covariance[index, index]))
    final = [np.ones(int(use.sum()))]
    if has_dome:
        final.append(fixed[use].astype(np.float64))
    if has_altitude:
        final.append(delta_x[use])
    rms = float(np.sqrt(np.mean((difference[use] - np.column_stack(final) @ solution) ** 2)))
    return SqmFit(
        offset_mag=float(solution[0]) + applied_offset_mag,
        offset_error_mag=float(np.sqrt(covariance[0, 0])),
        dome_loss_mag=dome,
        dome_loss_error_mag=dome_error,
        altitude_mag_per_airmass=k,
        altitude_error_mag_per_airmass=k_error,
        rms_mag=rms,
        n_fixed=int(fixed.sum()),
        n_manual=int((~fixed).sum()),
        n_used=int(use.sum()),
    )


def fit_sqm_offset(
    references: Sequence[ReferenceRecord],
    sky: Sequence[SkyQualityRecord],
    *,
    camera_altitude_deg: float,
    default_altitude_deg: float | None = None,
    max_gap_s: float = 600.0,
    applied_offset_mag: float = 0.0,
    min_pairs: int = 10,
) -> SqmFit | None:
    """Fit the offset between `reference` records and the survey's `sky_quality` records.

    The function pairs each reading with the sky record nearest in time (`pair_readings`) and
    fits the pairs (`fit_pairs`). `default_altitude_deg` stands in for a reading that carries no
    altitude, such as the altitude of the SQM-LE in the configuration.
    """
    pairs = pair_readings(
        references, sky, max_gap_s=max_gap_s, default_altitude_deg=default_altitude_deg
    )
    return fit_pairs(
        pairs,
        camera_altitude_deg=camera_altitude_deg,
        applied_offset_mag=applied_offset_mag,
        min_pairs=min_pairs,
    )
