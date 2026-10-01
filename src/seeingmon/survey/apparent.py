"""Apparent places: where catalog stars appear at the time of a frame.

A catalog gives astrometric positions in ICRS at one epoch, with proper motions and
parallaxes. A camera sees the apparent places at the time of the exposure. The difference
matters at the level of the pointing target:

- Annual aberration moves a star by up to 20.5 arcsec during the year, which is 5 pixels in
  bin2. Without the correction it would look like a seasonal pointing drift.
- Precession moves the pole by about 20 arcsec per year, and nutation swings it by up to
  9 arcsec.
- Proper motion and parallax reach a few arcseconds over a decade for the brightest stars.

`apparent_vectors` applies them with `pyerfa`, the ERFA routines that underlie `astropy`.
The result is a set of unit vectors in the Celestial Intermediate Reference System (CIRS),
the frame of the true pole of date. The camera attitude in `seeingmon.survey.wcs_fit` refers
to this frame.

**Frames.**

- ICRS: the catalog frame (barycentric, at the catalog epoch).
- CIRS: geocentric apparent directions, with proper motion, parallax, light deflection by the
  Sun, aberration, and the bias-precession-nutation matrix applied.
- Earth-fixed (TIRS): `cirs_to_earth_fixed(era) @ v_cirs`. A camera that is rigid on the Earth
  has a constant attitude in this frame. The module ignores polar motion (under 0.4 arcsec),
  diurnal aberration (a constant shift of the camera frame, 0.3 arcsec at most), and
  refraction. The Earth rotation angle uses UT1 = UTC + `dut1_s`, with `dut1_s` set to 0 by
  default, so a UT1 - UTC offset of up to 0.9 s appears as a rotation about the pole of at
  most 14 arcsec. A star at the far edge of a 3 degree field then moves by 0.7 arcsec, which
  is 0.2 pixels in bin2. Pass a measured `dut1_s` to remove it.

Nothing here does arithmetic on right ascension and declination. Stars move along the tangent
vectors, and the code works on unit vectors, so a star at the pole is no special case.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import erfa
import numpy as np
import numpy.typing as npt

from seeingmon.survey.geometry import (
    MAS_PER_RAD,
    TWO_PI,
    FloatArray,
    normalize,
    rot_z,
)

NS_PER_DAY = 86_400 * 10**9
UNIX_EPOCH_JD = 2_440_587.5
J2000_JD = 2_451_545.0
JULIAN_YEAR_DAYS = 365.25
SECONDS_PER_DAY = 86_400.0

# The light time for 1 au in Julian years, for the Roemer term of the proper motion interval.
_AU_LIGHT_TIME_YEARS = 499.004782 / SECONDS_PER_DAY / JULIAN_YEAR_DAYS

# The Gaia DR3 reference epoch, which the cap catalog uses for positions and proper motions.
CATALOG_EPOCH_JYEAR = 2016.0

# The rotation rate of the Earth in radians per second of UT1.
EARTH_ROTATION_RATE_RAD_S = TWO_PI * 1.00273781191135448 / SECONDS_PER_DAY


@dataclass(frozen=True, slots=True)
class StarAstrometry:
    """Astrometry of one star: ICRS position, proper motion, and parallax at an epoch."""

    ra_deg: float
    dec_deg: float
    pm_ra_mas_yr: float  # the proper motion in right ascension times cos(dec)
    pm_dec_mas_yr: float
    parallax_mas: float
    epoch_jyear: float


# Polaris (alpha Ursae Minoris, HIP 11767), from the Hipparcos new reduction at J2000.0. Gaia
# DR3 has no good astrometry for it, so the tracker and the catalog builder use this entry.
POLARIS = StarAstrometry(
    ra_deg=15.0 * (2.0 + 31.0 / 60.0 + 49.09456 / 3600.0),
    dec_deg=89.0 + 15.0 / 60.0 + 50.7923 / 3600.0,
    pm_ra_mas_yr=44.48,
    pm_dec_mas_yr=-11.85,
    parallax_mas=7.54,
    epoch_jyear=2000.0,
)


@dataclass(frozen=True, slots=True, eq=False)
class ObservationEpoch:
    """Everything that depends on the time of a frame and not on the star.

    Build one with `epoch_from_utc_ns` and reuse it for every star of a frame.

    `astrom` is the ERFA record of star-independent parameters (Earth position and velocity,
    the Sun's direction, and the bias-precession-nutation matrix). `npb` is the matrix from
    GCRS to CIRS. `era_rad` is the Earth rotation angle at the time, in [0, 2 pi).
    """

    t_utc_ns: int
    dut1_s: float
    era_rad: float
    npb: FloatArray
    astrom: Any


def utc_two_part_jd(t_utc_ns: int) -> tuple[float, float]:
    """The UTC date as a two-part Julian date. The split keeps the day fraction exact.

    Unix time has no leap seconds, so the date is exact except on a day that has one.
    """
    days, remainder_ns = divmod(t_utc_ns, NS_PER_DAY)
    return UNIX_EPOCH_JD + days, remainder_ns / NS_PER_DAY


def epoch_from_utc_ns(t_utc_ns: int, dut1_s: float = 0.0) -> ObservationEpoch:
    """Compute the time-dependent parameters for a UTC time.

    `dut1_s` is UT1 - UTC in seconds (the IERS bulletins publish it). The default of 0 is
    wrong by up to 0.9 s, which matters only for the Earth rotation angle.
    """
    utc1, utc2 = utc_two_part_jd(t_utc_ns)
    with warnings.catch_warnings():
        # ERFA warns about a "dubious year" for dates far after its leap-second table. No
        # leap second has been added since 2017, so the conversion stays exact.
        warnings.simplefilter("ignore", erfa.ErfaWarning)
        tai1, tai2 = erfa.utctai(utc1, utc2)
    tt1, tt2 = erfa.taitt(tai1, tai2)
    astrom, _equation_of_origins = erfa.apci13(tt1, tt2)  # TDB - TT is under 2 ms, so use TT
    era = float(erfa.era00(utc1, utc2 + dut1_s / SECONDS_PER_DAY))
    return ObservationEpoch(
        t_utc_ns=t_utc_ns,
        dut1_s=dut1_s,
        era_rad=era,
        npb=np.array(astrom["bpn"], dtype=np.float64),
        astrom=astrom,
    )


def earth_rotation_angle(t_utc_ns: int, dut1_s: float = 0.0) -> float:
    """The Earth rotation angle in radians, in [0, 2 pi), for a UTC time."""
    utc1, utc2 = utc_two_part_jd(t_utc_ns)
    return float(erfa.era00(utc1, utc2 + dut1_s / SECONDS_PER_DAY))


def cirs_to_earth_fixed(era_rad: float) -> FloatArray:
    """The matrix that turns CIRS vectors into Earth-fixed (TIRS) vectors.

    The Earth turns eastward, so a fixed point on the equator has the CIRS direction
    `[cos(era + lon), sin(era + lon), 0]`, and this matrix maps it to `[cos lon, sin lon, 0]`.
    """
    return rot_z(-era_rad)


def apparent_vectors(
    ra_deg: npt.ArrayLike,
    dec_deg: npt.ArrayLike,
    pm_ra_mas_yr: npt.ArrayLike,
    pm_dec_mas_yr: npt.ArrayLike,
    parallax_mas: npt.ArrayLike,
    epoch: ObservationEpoch,
    *,
    catalog_epoch_jyear: float = CATALOG_EPOCH_JYEAR,
) -> FloatArray:
    """Unit vectors in CIRS for catalog stars at the time of `epoch`.

    The inputs are ICRS positions in degrees at `catalog_epoch_jyear`, proper motions in
    milliarcseconds per year (right ascension times the cosine of declination), and parallaxes
    in milliarcseconds. A negative parallax counts as zero. The steps follow `eraAtciq`:

    1. Move the star with its proper motion and parallax, which gives the barycentric
       direction.
    2. Bend the light in the gravitational field of the Sun.
    3. Add the annual aberration.
    4. Apply the bias-precession-nutation matrix.

    The proper motion moves the star along the east and north vectors, which have no
    singularity at the pole. The result has shape `(N, 3)`, or `(3,)` for scalar input.
    """
    astrom = epoch.astrom
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    mu_ra = np.asarray(pm_ra_mas_yr, dtype=np.float64) / MAS_PER_RAD
    mu_dec = np.asarray(pm_dec_mas_yr, dtype=np.float64) / MAS_PER_RAD
    parallax = np.clip(np.asarray(parallax_mas, dtype=np.float64), 0.0, None) / MAS_PER_RAD

    sin_ra, cos_ra = np.sin(ra), np.cos(ra)
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    direction = np.stack([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec], axis=-1)
    east = np.stack([-sin_ra, cos_ra, np.zeros_like(ra)], axis=-1)
    north = np.stack([-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec], axis=-1)
    velocity = mu_ra[..., None] * east + mu_dec[..., None] * north  # radians per year

    earth = np.asarray(astrom["eb"], dtype=np.float64)  # barycentric position in au
    interval_years = (float(astrom["pmt"]) - (catalog_epoch_jyear - 2000.0)) + (
        direction @ earth
    ) * _AU_LIGHT_TIME_YEARS
    moved = direction + interval_years[..., None] * velocity - parallax[..., None] * earth
    barycentric = normalize(moved)

    deflected = erfa.ldsun(barycentric, astrom["eh"], astrom["em"])
    proper = erfa.ab(deflected, astrom["v"], astrom["em"], astrom["bm1"])
    return np.asarray(erfa.rxp(astrom["bpn"], proper), dtype=np.float64)


def astrometric_from_apparent(vectors_cirs: npt.ArrayLike, epoch: ObservationEpoch) -> FloatArray:
    """Undo the apparent-place steps: CIRS vectors back to ICRS directions without proper motion.

    The function inverts the matrix, the aberration, and the light deflection (the last two
    by fixed-point iteration, which converges to far below a microarcsecond in three steps).
    Use it to express a field center in ICRS. Proper motion and parallax are not part of a
    direction, so they do not appear.
    """
    astrom = epoch.astrom
    cirs = np.asarray(vectors_cirs, dtype=np.float64)
    proper = erfa.trxp(astrom["bpn"], cirs)  # transpose of the matrix applied to the vectors
    natural = np.array(proper)
    for _ in range(4):  # aberration is 1e-4 rad, so each pass gains four orders of magnitude
        shifted = erfa.ab(natural, astrom["v"], astrom["em"], astrom["bm1"])
        natural = normalize(proper - (shifted - natural))
    astrometric = np.array(natural)
    for _ in range(2):
        bent = erfa.ldsun(astrometric, astrom["eh"], astrom["em"])
        astrometric = normalize(natural - (bent - astrometric))
    return np.asarray(astrometric, dtype=np.float64)


def apparent_vectors_for(star: StarAstrometry, epoch: ObservationEpoch) -> FloatArray:
    """Apparent CIRS unit vector of one star, as a `(3,)` array."""
    return apparent_vectors(
        star.ra_deg,
        star.dec_deg,
        star.pm_ra_mas_yr,
        star.pm_dec_mas_yr,
        star.parallax_mas,
        epoch,
        catalog_epoch_jyear=star.epoch_jyear,
    )
