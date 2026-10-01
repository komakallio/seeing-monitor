"""The photometric zero point: the camera against Gaia G, with a color term.

The zero point is the magnitude of a star that gives one electron per second. For each matched
star the catalog gives the Gaia G magnitude and the BP-RP color, and the photometry gives the
rate `R` in electrons per second. The model is

    G = ZP - 2.5 log10(R) + c * (BP-RP)

so that `ZP` is the zero point for a star of color zero, and `c` is the color term in magnitudes
per magnitude of color. The camera band is not Gaia G (the sensor reaches further into the red
than the Gaia band does), so the fit determines `c`. A narrow range of colors cannot fix `c`.
In that case the fit holds `c` at `color_term_prior` and finds only the zero point.

**Weights.** Each star has the error of its photometry, the error of its catalog magnitude (a
Gaia magnitude is good to a few millimagnitudes, and a Tycho-2 star without a Gaia source has
an estimated G that is good to about 0.08 mag), and a floor for the effects that the photometry
does not model (flat-field residuals, scintillation, and the aperture). The fit adds an
intrinsic scatter to the errors when the stars scatter more than their errors explain.

**Clipping.** A star more than `clip_sigma` of its total error from the fit leaves the sample
(a blend, a variable star, a cloud edge), and the fit repeats until the sample stops changing.

**Sampling error.** The error of the zero point is the square root of the covariance of the
fit, so it falls as the square root of the number of stars. A frame with 100 stars that scatter
by 0.03 mag gives a zero point good to 0.003 mag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.survey.geometry import FloatArray

BoolArray = npt.NDArray[np.bool_]
_MAG_PER_E = 2.5 / np.log(10.0)


@dataclass(frozen=True, slots=True)
class ZeroPointOptions:
    """The limits of the zero-point fit."""

    min_stars: int = 8
    clip_sigma: float = 3.0
    max_iterations: int = 10
    systematic_mag: float = 0.01  # added in quadrature to every star
    gaia_error_mag: float = 0.005
    bright_error_mag: float = 0.02  # the Gaia G of stars brighter than `bright_g_mag`
    bright_g_mag: float = 6.0
    estimated_error_mag: float = 0.08  # a G that the catalog estimated from a Tycho-2 V
    min_color_std: float = 0.25  # a narrower spread of colors holds the color term at its prior
    color_term_prior: float = 0.0

    def __post_init__(self) -> None:
        if self.min_stars < 3 or self.clip_sigma <= 0 or self.max_iterations < 1:
            raise ValueError("invalid zero-point options")


@dataclass(frozen=True, slots=True, eq=False)
class ZeroPointFit:
    """The result of the fit.

    `zero_point_mag` is for BP-RP = 0, and `zero_point_error_mag` is its sampling error
    (1 sigma). `rms_mag` is the scatter of the used stars around the fit, and `intrinsic_mag`
    the extra scatter that the fit added to the errors. `color_fitted` says whether the stars
    fixed the color term. `used` marks the stars of the input that the fit kept.
    """

    zero_point_mag: float
    zero_point_error_mag: float
    color_term: float
    color_term_error: float
    color_fitted: bool
    rms_mag: float
    intrinsic_mag: float
    n_used: int
    n_input: int
    used: BoolArray


def catalog_errors_mag(
    g_mag: FloatArray, estimated: BoolArray, options: ZeroPointOptions
) -> FloatArray:
    """The error of the catalog magnitude of each star, by its kind."""
    error = np.full(g_mag.shape, options.gaia_error_mag)
    error = np.where(g_mag < options.bright_g_mag, options.bright_error_mag, error)
    return np.asarray(np.where(estimated, options.estimated_error_mag, error), dtype=np.float64)


def _weighted_fit(
    y: FloatArray,
    color: FloatArray,
    sigma: FloatArray,
    *,
    fit_color: bool,
    prior: float,
    pivot: float,
) -> tuple[float, float, FloatArray]:
    """Solve for the level at the pivot color and the slope. Returns them and the covariance."""
    weight = 1.0 / sigma**2
    if fit_color:
        design = np.column_stack([np.ones_like(y), color - pivot])
        normal = design.T @ (design * weight[:, None])
        rhs = design.T @ (weight * y)
        covariance = np.linalg.inv(normal)
        solution = covariance @ rhs
        return float(solution[0]), float(solution[1]), covariance
    level = float(np.sum(weight * (y - prior * (color - pivot))) / np.sum(weight))
    variance = 1.0 / float(np.sum(weight))
    return level, prior, np.array([[variance, 0.0], [0.0, 0.0]])


def fit_zero_point(
    g_mag: FloatArray,
    bp_rp: FloatArray,
    rate_e_per_s: FloatArray,
    mag_error: FloatArray,
    catalog_error: FloatArray,
    options: ZeroPointOptions | None = None,
) -> ZeroPointFit | None:
    """Fit the zero point and the color term to matched stars. Returns `None` for too few stars.

    `g_mag` and `bp_rp` come from the catalog (a NaN color is replaced by the median color of
    the sample, so that the star helps the zero point and not the color term), `rate_e_per_s`
    and `mag_error` from the photometry, and `catalog_error` from `catalog_errors_mag`.
    """
    cfg = options or ZeroPointOptions()
    n_input = int(g_mag.size)
    usable = (
        np.isfinite(g_mag)
        & np.isfinite(rate_e_per_s)
        & (rate_e_per_s > 0)
        & np.isfinite(mag_error)
        & (mag_error > 0)
    )
    if int(usable.sum()) < cfg.min_stars:
        return None
    colors = bp_rp.astype(np.float64).copy()
    missing = ~np.isfinite(colors)
    known = ~missing & usable
    fill = float(np.median(colors[known])) if known.any() else 0.0
    color_scatter = float(np.std(colors[known])) if known.sum() > 1 else 0.0
    colors[missing] = fill
    y = g_mag + 2.5 * np.log10(np.where(usable, rate_e_per_s, 1.0))
    base = np.sqrt(mag_error**2 + catalog_error**2 + cfg.systematic_mag**2)

    def total_error(intrinsic: float, slope: float) -> FloatArray:
        # A star without a color carries the error of its stand-in color times the slope.
        extra = np.where(missing, abs(slope) * color_scatter, 0.0)
        return np.asarray(np.sqrt(base**2 + intrinsic**2 + extra**2), dtype=np.float64)

    keep = usable.copy()
    intrinsic = 0.0
    fit_color = True
    pivot = 0.0
    level = 0.0
    slope = cfg.color_term_prior
    covariance = np.zeros((2, 2))
    for _ in range(cfg.max_iterations):
        selected = keep
        if int(selected.sum()) < cfg.min_stars:
            return None
        with_color = selected & ~missing
        spread = float(np.std(colors[with_color])) if with_color.sum() > 1 else 0.0
        fit_color = spread >= cfg.min_color_std
        sigma = total_error(intrinsic, slope)
        pivot = float(np.average(colors[selected], weights=1.0 / sigma[selected] ** 2))
        level, slope, covariance = _weighted_fit(
            y[selected],
            colors[selected],
            sigma[selected],
            fit_color=fit_color,
            prior=cfg.color_term_prior,
            pivot=pivot,
        )
        residual = y - (level + slope * (colors - pivot))
        judged = with_color if with_color.sum() >= 5 else selected
        robust = 1.4826 * float(np.median(np.abs(residual[judged] - np.median(residual[judged]))))
        intrinsic = float(np.sqrt(max(0.0, robust**2 - float(np.median(base[judged] ** 2)))))
        sigma = total_error(intrinsic, slope)
        new_keep = usable & (np.abs(residual) <= cfg.clip_sigma * sigma)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    selected = keep
    n_used = int(selected.sum())
    if n_used < cfg.min_stars:
        return None
    sigma = total_error(intrinsic, slope)
    level, slope, covariance = _weighted_fit(
        y[selected],
        colors[selected],
        sigma[selected],
        fit_color=fit_color,
        prior=cfg.color_term_prior,
        pivot=pivot,
    )
    residual = y[selected] - (level + slope * (colors[selected] - pivot))
    rms = float(np.sqrt(np.mean(residual**2)))
    # The zero point is the value at color zero: a - c * pivot, with the covariance to match.
    zero_point = level - slope * pivot
    variance = covariance[0, 0] + pivot**2 * covariance[1, 1] - 2.0 * pivot * covariance[0, 1]
    return ZeroPointFit(
        zero_point_mag=zero_point,
        zero_point_error_mag=float(np.sqrt(max(variance, 0.0))),
        color_term=slope,
        color_term_error=float(np.sqrt(max(covariance[1, 1], 0.0))),
        color_fitted=fit_color,
        rms_mag=rms,
        intrinsic_mag=intrinsic,
        n_used=n_used,
        n_input=n_input,
        used=keep,
    )
