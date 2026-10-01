"""Star fields, the camera pointing, and the rotation of the sky.

A `StarField` is a list of stars with a right ascension, a declination, and a magnitude in the
simulated camera's band. `make_polar_field` builds a synthetic field around the celestial pole,
and the survey lane can pass a real catalog instead, because the simulator accepts any field.

**Pointing.** The camera is fixed to the ground and looks near Polaris. `Pointing` gives the
direction of the optical axis at one reference time, and the roll about that axis. Earth turns
the sky about the celestial pole at the sidereal rate, so a star at angular distance `theta` from
the pole moves `15.041 sin(theta)` arcseconds per second. The projection is gnomonic about the
optical axis, as a real lens of this focal length works, so a solver that fits a TAN world
coordinate system finds no residual.

**Conventions.** The image shows the sky as seen from inside the celestial sphere, with east to
the left when north is up. `roll_deg` is the position angle of north in the image, measured
counter-clockwise from the image's up direction (toward the left). With zero roll, the pole
lies straight above the optical axis in the image. Pixel coordinates put the centre of pixel
`(0, 0)` at `(0, 0)`, with `x` along the columns and `y` down the rows. The optical axis falls
on the centre of the full frame of a readout mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.clock import DEFAULT_START_UTC_NS, NS_PER_S
from seeingmon.drivers.sim.params import ARCSEC_PER_RAD

FloatArray = npt.NDArray[np.float64]

SIDEREAL_RATE_RAD_PER_S = 7.292115855e-5
"""Earth's rotation rate: 15.041 arcseconds per second of time."""

POLARIS_MAG = 2.02
POLARIS_POLE_DISTANCE_DEG = 0.618
POLARIS_RA_DEG = 37.95
POLARIS_DEC_DEG = 90.0 - POLARIS_POLE_DISTANCE_DEG

# Polaris B: V = 8.7, 18.3 arcsec from Polaris at position angle 230 degrees.
POLARIS_B_MAG = 8.7
POLARIS_B_SEPARATION_ARCSEC = 18.3
POLARIS_B_POSITION_ANGLE_DEG = 230.0

# Stars per square degree near the pole to magnitudes 11, 12, and 13: the Gaia counts in the
# design circle of the research notes (233, 510, and 1,085 stars in 10.56 square degrees).
_DENSITY_AT_MAG = {11.0: 233 / 10.56, 12.0: 510 / 10.56, 13.0: 1085 / 10.56}


def _log_density(magnitude: FloatArray) -> FloatArray:
    """`log10` of the cumulative number of stars per square degree brighter than `magnitude`.

    The curve passes through the counts above. Between them, it follows slopes of log10 N per
    magnitude that match the real counts: 0.55 brighter than magnitude 8, 0.45 from 8 to 11, and
    0.28 fainter than magnitude 13.
    """
    log11 = math.log10(_DENSITY_AT_MAG[11.0])
    log13 = math.log10(_DENSITY_AT_MAG[13.0])
    knots_m = np.asarray([0.0, 8.0, 11.0, 12.0, 13.0, 16.0])
    knots_n = np.asarray(
        [
            log11 - 0.45 * 3.0 - 0.55 * 8.0,
            log11 - 0.45 * 3.0,
            log11,
            math.log10(_DENSITY_AT_MAG[12.0]),
            log13,
            log13 + 3 * 0.28,
        ]
    )
    return np.asarray(np.interp(magnitude, knots_m, knots_n), dtype=np.float64)


@dataclass(frozen=True, slots=True, eq=False)
class StarField:
    """Stars as arrays: right ascension and declination in degrees, and a magnitude each."""

    ra_deg: FloatArray
    dec_deg: FloatArray
    mag: FloatArray

    def __post_init__(self) -> None:
        if not (self.ra_deg.shape == self.dec_deg.shape == self.mag.shape) or self.ra_deg.ndim != 1:
            raise ValueError(
                "ra_deg, dec_deg, and mag must be one-dimensional arrays of one length"
            )
        if not (
            np.all(np.isfinite(self.ra_deg))
            and np.all(np.isfinite(self.dec_deg))
            and np.all(np.isfinite(self.mag))
        ):
            raise ValueError("star coordinates and magnitudes must be finite")
        if np.any(np.abs(self.dec_deg) > 90.0):
            raise ValueError("declinations must lie between -90 and +90 degrees")

    def __len__(self) -> int:
        return int(self.ra_deg.shape[0])

    @classmethod
    def from_arrays(
        cls, ra_deg: npt.ArrayLike, dec_deg: npt.ArrayLike, mag: npt.ArrayLike
    ) -> StarField:
        """Copy any array-like inputs into a field."""
        return cls(
            np.array(ra_deg, dtype=np.float64),
            np.array(dec_deg, dtype=np.float64),
            np.array(mag, dtype=np.float64),
        )

    def subset(self, mask: npt.NDArray[np.bool_]) -> StarField:
        """The stars where `mask` is true."""
        return StarField(self.ra_deg[mask].copy(), self.dec_deg[mask].copy(), self.mag[mask].copy())

    def unit_vectors(self) -> FloatArray:
        """The stars as unit vectors in the equatorial frame, with shape `(n, 3)`."""
        ra = np.radians(self.ra_deg)
        dec = np.radians(self.dec_deg)
        return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)


def polaris_field(*, include_polaris_b: bool = True) -> StarField:
    """Polaris, and its companion Polaris B."""
    ra = [POLARIS_RA_DEG]
    dec = [POLARIS_DEC_DEG]
    mag = [POLARIS_MAG]
    if include_polaris_b:
        angle = math.radians(POLARIS_B_POSITION_ANGLE_DEG)
        separation_deg = POLARIS_B_SEPARATION_ARCSEC / 3600.0
        d_dec = separation_deg * math.cos(angle)
        d_ra = separation_deg * math.sin(angle) / math.cos(math.radians(POLARIS_DEC_DEG))
        ra.append(POLARIS_RA_DEG + d_ra)
        dec.append(POLARIS_DEC_DEG + d_dec)
        mag.append(POLARIS_B_MAG)
    return StarField.from_arrays(ra, dec, mag)


def make_polar_field(
    seed: int = 0,
    *,
    cap_radius_deg: float = 15.0,
    mag_limit: float = 13.5,
    include_polaris: bool = True,
    include_polaris_b: bool = True,
) -> StarField:
    """A deterministic synthetic catalog inside a cap around the celestial pole.

    The stars sit at random positions with a uniform density, and their magnitudes follow the
    counts of the research notes: 233, 510, and 1,085 stars to magnitudes 11, 12, and 13 inside
    the 1.83 degree design circle. The counts per square degree extend to `mag_limit`. The
    result is the same for the same `seed`. `include_polaris` adds Polaris (V = 2.02, 0.618
    degrees from the pole), and `include_polaris_b` adds its companion. The real counts rise
    toward the Galactic plane, so a 15 degree cap holds fewer stars here than Gaia lists.
    """
    if not 0.5 <= cap_radius_deg <= 90.0:
        raise ValueError("cap_radius_deg must be between 0.5 and 90")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x57A5]))
    cos_cap = math.cos(math.radians(cap_radius_deg))
    area_deg2 = 2.0 * math.pi * (1.0 - cos_cap) * (180.0 / math.pi) ** 2
    total = float(10.0 ** _log_density(np.asarray(mag_limit)) * area_deg2)
    count = int(rng.poisson(total))
    # Magnitudes by inverse transform of the cumulative counts.
    grid = np.linspace(0.0, mag_limit, 4000)
    cumulative = 10.0 ** _log_density(grid)
    magnitudes = np.interp(rng.random(count) * cumulative[-1], cumulative, grid)
    polar_cos = rng.uniform(cos_cap, 1.0, count)
    dec = 90.0 - np.degrees(np.arccos(polar_cos))
    ra = rng.uniform(0.0, 360.0, count)
    order = np.argsort(magnitudes, kind="stable")
    field = StarField(ra[order], dec[order], magnitudes[order])
    if not include_polaris:
        return field
    parts = [field, polaris_field(include_polaris_b=include_polaris_b)]
    return StarField(
        np.concatenate([p.ra_deg for p in parts]),
        np.concatenate([p.dec_deg for p in parts]),
        np.concatenate([p.mag for p in parts]),
    )


# --- pointing -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pointing:
    """Where the camera looks at a reference time, and how it is rolled.

    `ra_deg` and `dec_deg` give the optical axis at `t_ref_utc_ns`. The sky turns away from
    that position afterward, because the camera stays fixed to the ground. `roll_deg` is the
    position angle of north in the image, counter-clockwise from the up direction. The default
    centres Polaris at the reference time with the pole straight above it. `offset_arcsec` moves
    the optical axis away from the centre of the full frame by `(x, y)` arcseconds, with `y`
    down the rows. A mount never centres a star exactly, and the default places Polaris between
    four pixels of bin1. Set an offset of half a pixel, 0.955 arcsec, to centre it on one.
    """

    ra_deg: float = POLARIS_RA_DEG
    dec_deg: float = POLARIS_DEC_DEG
    roll_deg: float = 0.0
    t_ref_utc_ns: int = DEFAULT_START_UTC_NS
    offset_arcsec: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not -90.0 < self.dec_deg < 90.0:
            raise ValueError("the optical axis must not point at a celestial pole")

    @property
    def pole_distance_deg(self) -> float:
        """The angle from the optical axis to the celestial pole."""
        return 90.0 - self.dec_deg


class SkyProjector:
    """Projects a star field onto the sensor at any time.

    The constructor keeps only the stars that can reach the sensor: those within the field of
    view of the pole's position. `max_field_radius_deg` is the largest angle from the optical
    axis to a sensor corner.
    """

    def __init__(
        self, field: StarField, pointing: Pointing, *, max_field_radius_deg: float = 2.9
    ) -> None:
        self._pointing = pointing
        pole_distance = pointing.pole_distance_deg
        reach = pole_distance + max_field_radius_deg + 0.5
        polar = 90.0 - field.dec_deg
        self._kept = np.nonzero(polar <= reach)[0]
        kept = field.subset(polar <= reach)
        self._field = kept
        self._vectors = kept.unit_vectors()
        ra = math.radians(pointing.ra_deg)
        dec = math.radians(pointing.dec_deg)
        axis = np.asarray(
            [math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)]
        )
        north = np.asarray(
            [-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec)]
        )
        west = np.asarray([math.sin(ra), -math.cos(ra), 0.0])
        roll = math.radians(pointing.roll_deg)
        right = math.cos(roll) * west - math.sin(roll) * north
        up = math.sin(roll) * west + math.cos(roll) * north
        self._basis = np.stack([right, up, axis])  # rows: image right, image up, optical axis
        self._pole = np.asarray([0.0, 0.0, 1.0])

    @property
    def pointing(self) -> Pointing:
        return self._pointing

    @property
    def stars(self) -> StarField:
        """The stars that the projector keeps, in the order that `project` returns them."""
        return self._field

    @property
    def indices(self) -> npt.NDArray[np.intp]:
        """The index of each kept star in the original field."""
        return self._kept

    def _rotation_angle(self, t_utc_ns: int | FloatArray) -> float | FloatArray:
        elapsed = (np.asarray(t_utc_ns) - self._pointing.t_ref_utc_ns) / NS_PER_S
        return SIDEREAL_RATE_RAD_PER_S * elapsed

    def _tangent(self, vectors: FloatArray, t_utc_ns: int) -> tuple[FloatArray, FloatArray]:
        """Image-plane tangent coordinates (right, up) in radians for unit vectors at a time."""
        angle = float(self._rotation_angle(t_utc_ns))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        # Turning every star by -angle about the pole is the same as turning the camera basis
        # by +angle, which is cheaper. The rows of `basis` are the turned image axes.
        rotation = np.asarray([[cos_a, sin_a, 0.0], [-sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
        basis = self._basis @ rotation
        components = vectors @ basis.T
        depth = components[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return components[:, 0] / depth, components[:, 1] / depth

    def _to_pixels(
        self, right: FloatArray, up: FloatArray, pixel_rad: float, width: int, height: int
    ) -> tuple[FloatArray, FloatArray]:
        """Tangent coordinates in radians to pixels, with the optical axis offset applied."""
        offset_x, offset_y = self._pointing.offset_arcsec
        scale = 1.0 / ARCSEC_PER_RAD / pixel_rad
        return (
            (width - 1) / 2.0 + offset_x * scale + right / pixel_rad,
            (height - 1) / 2.0 + offset_y * scale - up / pixel_rad,
        )

    def project(
        self, t_utc_ns: int, pixel_rad: float, width: int, height: int
    ) -> tuple[FloatArray, FloatArray]:
        """Pixel coordinates `(x, y)` of the kept stars at a time, in the full frame of a mode."""
        right, up = self._tangent(self._vectors, t_utc_ns)
        return self._to_pixels(right, up, pixel_rad, width, height)

    def project_stars(
        self,
        indices: npt.NDArray[np.intp],
        t_utc_ns: int,
        pixel_rad: float,
        width: int,
        height: int,
    ) -> tuple[FloatArray, FloatArray]:
        """Pixel coordinates of some of the kept stars (given by position in `stars`)."""
        right, up = self._tangent(self._vectors[indices], t_utc_ns)
        return self._to_pixels(right, up, pixel_rad, width, height)

    def pole_pixel(
        self, t_utc_ns: int, pixel_rad: float, width: int, height: int
    ) -> tuple[float, float]:
        """Pixel coordinates of the celestial pole. They do not depend on the time."""
        right, up = self._tangent(self._pole[None, :], t_utc_ns)
        x, y = self._to_pixels(right, up, pixel_rad, width, height)
        return float(x[0]), float(y[0])

    def roll_deg(self) -> float:
        """The position angle of the direction to the pole, as the architecture defines `roll`."""
        return self._pointing.roll_deg


def sidereal_motion_arcsec_per_s(polar_distance_deg: float) -> float:
    """How fast a star at an angular distance from the pole moves: `15.041 sin(theta)`."""
    return SIDEREAL_RATE_RAD_PER_S * ARCSEC_PER_RAD * math.sin(math.radians(polar_distance_deg))
