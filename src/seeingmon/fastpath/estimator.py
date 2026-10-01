"""The seeing estimator: from a window of centroids to the image-motion variance and `r0`.

The estimator works per window, in these steps. `seeingmon.fastpath.models` explains each
correction.

1. **Detrend.** For each axis, the estimator fits a polynomial of order `detrend_order` (2 by
   default) to the centroid over the window and subtracts it. Polaris drifts about 10 arcsec in
   60 s, which is 5 pixels in bin1, and the fit removes that. An outlier test (`outlier_sigma`
   times the robust sigma of the residuals) drops frames that a cosmic ray or a bad frame spoiled.
2. **Variance.** The variance of the residuals is `(1 - rho)` of the variance of the motion plus
   the variance of the centroid noise. `rho` is the share that the fit removes from the
   turbulence, which is 0.3% for the default outer scale. The estimator subtracts the modeled
   noise (photon and pixel noise, from the kernel) and divides by `1 - rho`.
3. **Windowed centroid.** The kernel measures a centroid in a finite aperture, which differs
   slightly from the G-tilt that the formulas describe. `centroid_gain_variance_ratio` is the
   ratio of the two variances (the calibration is in `seeingmon.fastpath.models`), and the
   estimator divides by it.
4. **Outer scale and exposure.** The estimator divides by the outer-scale ratio and by the
   exposure ratio to reach the variance of an instantaneous measurement in Kolmogorov
   turbulence.
5. **Seeing.** It averages the two axes, converts the variance with `K lambda^2 D^(-1/3) r0^(-5/3)`
   (`K` = 0.170, at 500 nm), converts to the zenith with `(cos z)^(3/5)`, and gives the FWHM as
   `0.98 lambda / r0`.

**Cross-check.** The second estimate uses the structure function of the centroid at lags of
`structure_lag_min_s` to `structure_lag_max_s` (40 to 120 ms by default). The difference of two
samples a few frames apart cancels a drift and a vibration that is slower than about 3 Hz. The
estimator takes the mean square difference at each lag, subtracts the noise of both samples,
divides by `2 (1 - kappa)` (the model's normalized covariance at that lag) to get the variance, and
then applies steps 3 to 5. It needs no detrend.

The estimator stores every assumption that it applied (`SeeingEstimate.factors`), so a reader can
undo it. The functions take plain arrays and settings, never a clock, so they run on simulated
and recorded data alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from seeingmon.fastpath import models

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MotionSeries:
    """The centroids of a window on a grid of frame slots.

    `x_px` and `y_px` are in sensor pixels, `period_s` apart, with `NaN` where a slot holds no
    usable frame. `noise_var_x_px2` and `noise_var_y_px2` hold the modeled centroid noise of each
    usable frame, in square pixels (`NaN` when the model is unknown).
    """

    period_s: float
    x_px: FloatArray
    y_px: FloatArray
    noise_var_x_px2: FloatArray
    noise_var_y_px2: FloatArray


@dataclass(frozen=True, slots=True)
class EstimatorSettings:
    """What the estimator needs besides the data.

    `aperture_m` is the diameter of the telescope aperture. `plate_scale_arcsec_per_px` and
    `exposure_s` describe the stream. `outer_scale_m` and `wind_ms` are the assumptions of the
    corrections. `centroid_gain_variance_ratio` is the ratio of the variance of the windowed
    centroid to the variance of the G-tilt. `zenith_angle_deg` is the zenith angle of the star, or
    `None` when it is unknown, and then the estimator does not convert to the zenith.
    """

    aperture_m: float
    plate_scale_arcsec_per_px: float
    exposure_s: float
    outer_scale_m: float = 20.0
    wind_ms: float = 10.0
    detrend_order: int = 2
    outlier_sigma: float = 8.0
    g_tilt_coefficient: float = models.G_TILT_COEFFICIENT
    fwhm_coefficient: float = models.FWHM_COEFFICIENT
    structure_lag_min_s: float = 0.04
    structure_lag_max_s: float = 0.12
    centroid_gain_variance_ratio: float = 1.0
    zenith_angle_deg: float | None = None
    min_samples: int = 100


@dataclass(frozen=True, slots=True)
class AxisResult:
    """The detrended motion of one axis.

    `residual_px` holds the residuals on the slot grid (`NaN` where a slot is missing or an
    outlier), and `cleaned_px` holds the original centroids with `NaN` in the same slots.
    `variance_px2` is the mean square of the residuals, and `noise_var_px2` is the mean modeled
    noise.
    """

    residual_px: FloatArray
    cleaned_px: FloatArray
    samples: int
    outliers: int
    variance_px2: float
    noise_var_px2: float


@dataclass(frozen=True, slots=True)
class Factors:
    """The multipliers that turn the measured variance into the Kolmogorov variance.

    Each is a factor by which the estimator multiplied the variance, so a value above 1 raises it.
    `zenith_r0` is the factor `(cos z)^(3/5)` that the estimator divided `r0` by (1 without a
    zenith angle).
    """

    detrend: float
    centroid_gain: float
    outer_scale: float
    exposure: float
    zenith_r0: float


@dataclass(frozen=True, slots=True)
class SeeingEstimate:
    """The seeing of one window. A value is `None` when the window cannot support it."""

    x: AxisResult | None
    y: AxisResult | None
    rms_x_arcsec: float | None = None
    rms_y_arcsec: float | None = None
    centroid_noise_px: float | None = None
    r0_m: float | None = None
    fwhm_arcsec: float | None = None
    r0_structure_m: float | None = None
    fwhm_structure_arcsec: float | None = None
    structure_variance_arcsec2: float | None = None
    kolmogorov_variance_arcsec2: float | None = None
    factors: Factors | None = None
    quality: dict[str, str] = field(default_factory=dict)


def _polyfit_residuals(
    t: FloatArray, x: FloatArray, mask: npt.NDArray[np.bool_], order: int
) -> FloatArray:
    """The residuals of a polynomial fit to the masked points, at every point."""
    scaled = (t - t[mask].mean()) / max(0.5 * float(np.ptp(t[mask])), 1e-12)
    coefficients = np.polynomial.polynomial.polyfit(scaled[mask], x[mask], order)
    return np.asarray(x - np.polynomial.polynomial.polyval(scaled, coefficients), dtype=np.float64)


def detrend_axis(
    series: FloatArray, noise_var: FloatArray, period_s: float, order: int, outlier_sigma: float
) -> AxisResult | None:
    """Remove the polynomial trend from one axis and return the residuals and their variance."""
    valid = np.isfinite(series)
    if int(valid.sum()) <= order + 2:
        return None
    t = np.arange(len(series), dtype=np.float64) * period_s
    mask = valid.copy()
    residual = np.full(len(series), np.nan)
    for _ in range(2):
        residual = _polyfit_residuals(t, np.where(valid, series, 0.0), mask, order)
        if outlier_sigma <= 0.0:
            break
        sigma = 1.4826 * float(np.median(np.abs(residual[mask])))
        if sigma <= 0.0:
            break
        mask = valid & (np.abs(residual) <= outlier_sigma * sigma)
    residual = np.where(mask, residual, np.nan)
    noise = noise_var[mask]
    noise = noise[np.isfinite(noise)]
    return AxisResult(
        residual_px=residual,
        cleaned_px=np.where(mask, series, np.nan),
        samples=int(mask.sum()),
        outliers=int(valid.sum() - mask.sum()),
        variance_px2=float(np.mean(residual[mask] ** 2)),
        noise_var_px2=float(noise.mean()) if len(noise) else 0.0,
    )


def structure_variance_px2(
    cleaned: FloatArray,
    noise_var_px2: float,
    lags: npt.NDArray[np.intp],
    kappa: FloatArray,
) -> float | None:
    """The variance of the motion from the structure function, in square pixels.

    `cleaned` is the series with `NaN` for missing frames (and outliers). `lags` are in slots, and
    `kappa` is the model's normalized covariance at each lag. The result is the mean over lags of
    `(D(lag) - 2 noise) / (2 (1 - kappa))`.
    """
    estimates: list[float] = []
    for lag, correlation in zip(lags, kappa, strict=True):
        if lag >= len(cleaned):
            continue
        difference = cleaned[lag:] - cleaned[:-lag]
        pairs = difference[np.isfinite(difference)]
        if len(pairs) < 20:
            continue
        structure = float(np.mean(pairs**2)) - 2.0 * noise_var_px2
        estimates.append(structure / (2.0 * (1.0 - float(correlation))))
    return float(np.mean(estimates)) if estimates else None


def _r0_and_seeing(
    variance_arcsec2: float, settings: EstimatorSettings, zenith_r0: float
) -> tuple[float, float]:
    variance_rad2 = variance_arcsec2 / models.ARCSEC_PER_RAD**2
    r0_los = models.r0_from_tilt_variance(
        variance_rad2, settings.aperture_m, coefficient=settings.g_tilt_coefficient
    )
    r0 = r0_los / zenith_r0
    return r0, models.seeing_fwhm_arcsec(r0, coefficient=settings.fwhm_coefficient)


def estimate_seeing(series: MotionSeries, settings: EstimatorSettings) -> SeeingEstimate:
    """Estimate the image motion and the seeing of a window. See the module documentation."""
    quality: dict[str, str] = {}
    x = detrend_axis(
        series.x_px,
        series.noise_var_x_px2,
        series.period_s,
        settings.detrend_order,
        settings.outlier_sigma,
    )
    y = detrend_axis(
        series.y_px,
        series.noise_var_y_px2,
        series.period_s,
        settings.detrend_order,
        settings.outlier_sigma,
    )
    if x is None or y is None or min(x.samples, y.samples) < settings.min_samples:
        for name in ("seeing_fwhm_arcsec", "r0_cm"):
            quality[name] = "too few usable frames"
        return SeeingEstimate(x, y, quality=quality)

    scale = settings.plate_scale_arcsec_per_px
    spectrum = models.tilt_spectrum(settings.aperture_m, settings.outer_scale_m, settings.wind_ms)
    window_s = len(series.x_px) * series.period_s
    detrend_loss = spectrum.detrend_variance_fraction(
        window_s, settings.detrend_order, settings.exposure_s
    )
    exposure_ratio = spectrum.exposure_variance_ratio(settings.exposure_s)
    outer_ratio = models.outer_scale_ratio(settings.aperture_m, settings.outer_scale_m)
    zenith_r0 = (
        1.0
        if settings.zenith_angle_deg is None
        else models.zenith_r0_factor(settings.zenith_angle_deg)
    )
    factors = Factors(
        detrend=1.0 / (1.0 - detrend_loss),
        centroid_gain=1.0 / settings.centroid_gain_variance_ratio,
        outer_scale=1.0 / outer_ratio,
        exposure=1.0 / exposure_ratio,
        zenith_r0=zenith_r0,
    )

    noise_px2 = 0.5 * (x.noise_var_px2 + y.noise_var_px2)
    motion_px2 = [axis.variance_px2 - axis.noise_var_px2 for axis in (x, y)]
    rms = [math.sqrt(max(v, 0.0)) * scale for v in motion_px2]
    result = SeeingEstimate(
        x,
        y,
        rms_x_arcsec=rms[0],
        rms_y_arcsec=rms[1],
        centroid_noise_px=math.sqrt(noise_px2),
        factors=factors,
        quality=quality,
    )
    mean_motion_px2 = 0.5 * sum(motion_px2)
    if not mean_motion_px2 > 0.0:
        quality["seeing_fwhm_arcsec"] = "the motion is below the centroid noise"
        quality["r0_cm"] = quality["seeing_fwhm_arcsec"]
        return result
    chain = factors.detrend * factors.centroid_gain * factors.outer_scale * factors.exposure
    kolmogorov = mean_motion_px2 * scale**2 * chain
    r0, fwhm = _r0_and_seeing(kolmogorov, settings, zenith_r0)
    if settings.zenith_angle_deg is None:
        note = "no zenith angle: the value is for the line of sight"
        quality["seeing_fwhm_arcsec"] = quality["r0_cm"] = note

    # The structure-function cross-check.
    lag_min = max(1, round(settings.structure_lag_min_s / series.period_s))
    lag_max = max(lag_min, round(settings.structure_lag_max_s / series.period_s))
    lags = np.arange(lag_min, lag_max + 1, dtype=np.intp)
    kappa = spectrum.autocorrelation(lags.astype(np.float64) * series.period_s, settings.exposure_s)
    sf = [
        structure_variance_px2(axis.cleaned_px, axis.noise_var_px2, lags, kappa) for axis in (x, y)
    ]
    r0_sf: float | None = None
    fwhm_sf: float | None = None
    sf_variance: float | None = None
    if sf[0] is not None and sf[1] is not None and sf[0] + sf[1] > 0.0:
        sf_chain = factors.centroid_gain * factors.outer_scale * factors.exposure
        sf_variance = 0.5 * (sf[0] + sf[1]) * scale**2 * sf_chain
        r0_sf, fwhm_sf = _r0_and_seeing(sf_variance, settings, zenith_r0)
    else:
        quality["seeing_fwhm_structure_arcsec"] = "too few pairs of frames"
        quality["r0_structure_cm"] = quality["seeing_fwhm_structure_arcsec"]
    return SeeingEstimate(
        x,
        y,
        rms_x_arcsec=rms[0],
        rms_y_arcsec=rms[1],
        centroid_noise_px=math.sqrt(noise_px2),
        r0_m=r0,
        fwhm_arcsec=fwhm,
        r0_structure_m=r0_sf,
        fwhm_structure_arcsec=fwhm_sf,
        structure_variance_arcsec2=sf_variance,
        kolmogorov_variance_arcsec2=kolmogorov,
        factors=factors,
        quality=quality,
    )
