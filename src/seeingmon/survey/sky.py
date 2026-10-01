"""The sky brightness of a survey frame, in the camera band and as a V equivalent.

**The measurement.** The frame holds the bias, the dark current, the sky, and the stars. The
dark model (`seeingmon.survey.dark`) gives the first two as one level for the sensor
temperature and the exposure. The function subtracts that level, divides by a flat field, hides
the stars, the saturated pixels, the hot pixels, and the frame edge, and takes a sigma-clipped
median of the pixels that remain. Unresolved stars stay in the sky, as they do for an SQM.

    sky rate = level (counts) * (electrons per count) / exposure     [e-/s per pixel]
             / (plate scale in arcsec per pixel) ** 2                 [e-/s per arcsec^2]

**The magnitude.** With the zero point `ZP` of the frame (the magnitude of a star of one
electron per second), the surface brightness in the camera band is

    sky_mag = ZP - 2.5 log10(sky rate per arcsec^2)

which is the same as the magnitude of the sky in one pixel plus `2.5 log10(scale ** 2)` (2.910 mag
for the 3.82 arcsec pixels of bin2). The zero point comes from stars that shine through the whole
atmosphere, and the sky light comes from inside it, so this magnitude is up to a few tenths too
bright. The fitted SQM offset absorbs the mean of that difference.

**The V equivalent.** The camera band follows Gaia G at a BP-RP color of zero, with the fitted
color term. For a source of color `k`, G is `sky_mag + c k`, and the Gaia DR3 relation

    G - V = -0.02704 + 0.01424 k - 0.2156 k^2 + 0.01426 k^3      (scatter 0.03 mag, -0.5 < k < 5)

gives V. The sky has no single color, so `SkyOptions.bp_rp` sets the one that the conversion
assumes. The conversion adds `sqm_offset_mag`, the offset that `seeingmon.survey.sqm_fit` fits
against the SQM-LE readings. The V equivalent is good to 0.2 to 0.3 mag until that fit exists.

**The flat field.** A unit flat is the default. `ArrayFlat` takes a measured flat and divides it
out. The architecture reserves the flat as a hook for commissioning.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from seeingmon.solvers import fitsio
from seeingmon.survey.geometry import FloatArray

BoolArray = npt.NDArray[np.bool_]

# Gaia DR3 documentation, photometric relations: G - V as a polynomial in BP - RP.
G_MINUS_V = (-0.02704, 0.01424, -0.2156, 0.01426)
G_MINUS_V_RANGE = (-0.5, 5.0)
G_MINUS_V_SCATTER_MAG = 0.03
_SAMPLE_SEED = 0x5C1


class SkyError(Exception):
    """A flat file cannot be used."""


# --- The flat field ------------------------------------------------------------------------


class FlatModel(Protocol):
    """The relative sensitivity across the frame: 1 where the sensor and the optics are at best."""

    @property
    def version(self) -> str:
        """A short name for the provenance of a result."""
        ...

    def at(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """The flat value at sensor pixel positions (the center of the first pixel is (0, 0))."""
        ...

    def image(
        self, shape: tuple[int, int], origin: tuple[int, int] = (0, 0)
    ) -> npt.NDArray[np.float32] | None:
        """The flat for a frame of `shape` whose first pixel sits at `origin` on the sensor.

        A unit flat returns `None`, which the caller reads as "divide by nothing".
        """
        ...


class UnitFlat:
    """No correction: every pixel has the same sensitivity."""

    version = "unit"

    def at(self, x: FloatArray, y: FloatArray) -> FloatArray:
        return np.ones(np.shape(x), dtype=np.float64)

    def image(
        self, shape: tuple[int, int], origin: tuple[int, int] = (0, 0)
    ) -> npt.NDArray[np.float32] | None:
        return None


class ArrayFlat:
    """A measured flat, as an image for the whole sensor. The constructor scales it to median 1."""

    def __init__(self, flat: npt.ArrayLike) -> None:
        array = np.asarray(flat, dtype=np.float32)
        if array.ndim != 2 or not np.all(np.isfinite(array)) or np.any(array <= 0):
            raise SkyError("a flat must be a 2-D image of positive numbers")
        self._flat = np.ascontiguousarray(array / np.float32(np.median(array)))
        self._version = f"flat-{zlib.crc32(self._flat.tobytes()):08x}"

    @property
    def version(self) -> str:
        return self._version

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._flat.shape[0]), int(self._flat.shape[1]))

    def at(self, x: FloatArray, y: FloatArray) -> FloatArray:
        height, width = self._flat.shape
        column = np.clip(np.rint(x).astype(np.intp), 0, width - 1)
        row = np.clip(np.rint(y).astype(np.intp), 0, height - 1)
        return np.asarray(self._flat[row, column], dtype=np.float64)

    def image(
        self, shape: tuple[int, int], origin: tuple[int, int] = (0, 0)
    ) -> npt.NDArray[np.float32] | None:
        x0, y0 = origin
        window = self._flat[y0 : y0 + shape[0], x0 : x0 + shape[1]]
        if window.shape != shape:
            raise SkyError("the frame does not fit inside the flat")
        return window


def load_flat(path: str | Path) -> FlatModel:
    """Read a flat from a NumPy file (`.npy`) or a FITS image. An empty path gives a unit flat."""
    if not str(path):
        return UnitFlat()
    target = Path(path)
    try:
        if target.suffix.lower() == ".npy":
            return ArrayFlat(np.load(target))
        _, image = fitsio.read_image(target)
        return ArrayFlat(image)
    except (OSError, ValueError, fitsio.FitsError) as error:
        raise SkyError(f"cannot read the flat file: {error}") from error


# --- Colors and magnitudes -----------------------------------------------------------------


def g_minus_v(bp_rp: float) -> float:
    """The Gaia DR3 relation between G and V for a color BP-RP in the range -0.5 to 5.0."""
    low, high = G_MINUS_V_RANGE
    color = min(max(bp_rp, low), high)
    a0, a1, a2, a3 = G_MINUS_V
    return a0 + a1 * color + a2 * color**2 + a3 * color**3


def pixel_solid_angle_mag(scale_arcsec_px: float) -> float:
    """The magnitude that a pixel adds to a surface brightness: `2.5 log10(scale ** 2)`."""
    return 2.5 * math.log10(scale_arcsec_px**2)


def sky_magnitude(zero_point_mag: float, rate_e_per_s_arcsec2: float) -> float | None:
    """The surface brightness in magnitudes per square arcsecond, or `None` without a signal."""
    if rate_e_per_s_arcsec2 <= 0.0 or not math.isfinite(rate_e_per_s_arcsec2):
        return None
    return zero_point_mag - 2.5 * math.log10(rate_e_per_s_arcsec2)


def v_equivalent(
    sky_mag_arcsec2: float, *, color_term: float, bp_rp: float, offset_mag: float = 0.0
) -> float:
    """The V-band equivalent of a camera-band surface brightness, for a sky of color `bp_rp`.

    The camera magnitude plus the color term times the color is the Gaia G of that sky. The
    Gaia relation takes G to V, and the offset ties the result to the SQM scale.
    """
    return sky_mag_arcsec2 + color_term * bp_rp - g_minus_v(bp_rp) + offset_mag


# --- The measurement -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SkyOptions:
    """How the sky is measured and converted.

    The sigma-clipped median uses at most `max_samples` pixels, picked at fixed random
    positions. `edge_px` pixels at the frame edge stay out. `bp_rp` is the color that the
    V conversion assumes for the sky, and `sqm_offset_mag` the SQM-fitted offset.
    """

    clip_sigma: float = 3.0
    clip_iterations: int = 4
    max_samples: int = 1_000_000
    edge_px: int = 8
    min_pixels: int = 5_000
    bp_rp: float = 1.0
    sqm_offset_mag: float = 0.0

    def __post_init__(self) -> None:
        if self.clip_sigma <= 0 or self.max_samples < 1000 or self.min_pixels < 100:
            raise ValueError("invalid sky options")


@dataclass(frozen=True, slots=True)
class SkyMeasurement:
    """The sky level of a frame and its signal rate.

    `level_dn` is the sigma-clipped median above the dark level (counts per pixel), `raw_dn` the
    median before the calibration, and `noise_dn` the robust scatter of the pixels. The rates
    are in electrons per second, per pixel and per square arcsecond.
    """

    level_dn: float
    raw_dn: float
    noise_dn: float
    n_pixels: int
    rate_e_per_s_px: float
    rate_e_per_s_arcsec2: float


def measure_sky(
    data: npt.NDArray[np.float32],
    *,
    star_mask: BoolArray | None,
    dark_level_dn: float,
    flat: npt.NDArray[np.float32] | None,
    exposure_s: float,
    e_per_adu: float,
    scale_arcsec_px: float,
    saturation_dn: float,
    bad: BoolArray | None = None,
    options: SkyOptions | None = None,
) -> SkyMeasurement | None:
    """Measure the sky of a frame. Returns `None` when too few pixels remain.

    `data` is in native counts. `star_mask` hides the stars (see `seeingmon.survey.detect.
    star_mask`), `bad` hides other pixels such as hot ones, and the function also hides
    saturated pixels and the frame edge. `dark_level_dn` is the bias plus the dark signal, from
    `DarkModel.level_dn`, and `flat` divides the frame after that level is gone.
    """
    cfg = options or SkyOptions()
    usable = np.ones(data.shape, dtype=np.bool_)
    if cfg.edge_px:
        edge = cfg.edge_px
        usable[:edge, :] = False
        usable[-edge:, :] = False
        usable[:, :edge] = False
        usable[:, -edge:] = False
    usable &= data < np.float32(saturation_dn * 0.98)
    if star_mask is not None:
        usable &= ~star_mask
    if bad is not None:
        usable &= ~bad
    flat_usable = usable.ravel()
    total = flat_usable.size
    if total > cfg.max_samples:
        rng = np.random.default_rng(_SAMPLE_SEED)
        picked = rng.integers(0, total, cfg.max_samples)
        picked = picked[flat_usable[picked]]
    else:
        picked = np.flatnonzero(flat_usable)
    if picked.size < cfg.min_pixels:
        return None
    raw = data.reshape(-1)[picked].astype(np.float32)
    values = raw - np.float32(dark_level_dn)
    if flat is not None:
        values = values / flat.reshape(-1)[picked]
    # Whole counts make the median a step function. A uniform dither of one count between -0.5
    # and +0.5 undoes the rounding, so the median is good to a few hundredths of a count.
    rng = np.random.default_rng(_SAMPLE_SEED + 1)
    values = values + rng.uniform(-0.5, 0.5, values.size).astype(np.float32)
    level, noise = _clipped_median(values, cfg)
    rate_px = level * e_per_adu / exposure_s
    return SkyMeasurement(
        level_dn=level,
        raw_dn=float(np.median(raw)),
        noise_dn=noise,
        n_pixels=int(picked.size),
        rate_e_per_s_px=rate_px,
        rate_e_per_s_arcsec2=rate_px / scale_arcsec_px**2,
    )


def _clipped_median(values: npt.NDArray[np.float32], cfg: SkyOptions) -> tuple[float, float]:
    """The sigma-clipped median and the robust sigma of the values."""
    kept = values
    median = float(np.median(kept))
    sigma = 1.4826 * float(np.median(np.abs(kept - np.float32(median))))
    for _ in range(cfg.clip_iterations):
        if sigma <= 0.0:
            break
        selected = kept[np.abs(kept - np.float32(median)) <= np.float32(cfg.clip_sigma * sigma)]
        if selected.size == kept.size or selected.size < 100:
            kept = selected if selected.size >= 100 else kept
            break
        kept = selected
        median = float(np.median(kept))
        sigma = 1.4826 * float(np.median(np.abs(kept - np.float32(median))))
    return median, sigma
