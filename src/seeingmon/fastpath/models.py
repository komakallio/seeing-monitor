"""The theory behind the seeing estimator, and the corrections that it applies.

The estimator turns the one-axis variance of the image motion into the Fried parameter `r0` with

    sigma^2 = K lambda^2 D^(-1/3) r0^(-5/3)        (K = 0.170, Martin 1987, Eq. 7)

for a circular aperture of diameter `D` under Kolmogorov turbulence, and into the seeing with
`FWHM = 0.98 lambda / r0`. The result is achromatic when you scale `r0` as `lambda^(6/5)`, so the
estimator works at 500 nm. Four corrections connect the measured variance to that formula.

**Outer scale.** A finite outer scale `L0` lowers the variance of the G-tilt (the centroid) of a
single aperture. `outer_scale_ratio` integrates the von Karman spectrum, so it needs no
interpolation between tabulated values. For `D` = 50 mm it returns 0.739, 0.793, and 0.848 at
`L0` = 10, 20, and 50 m, the exact values in `docs/research-notes.md`. The estimator divides the
variance by this ratio, so a larger assumed `L0` gives a smaller `r0`.

**Exposure averaging.** A frame of exposure `T` averages the image motion over `T`, which
removes the part of the variance above the frequency `1 / T`. The model is a single frozen layer
that moves at the assumed wind speed `v` (default 10 m/s). Its temporal tilt spectrum follows from
the von Karman phase spectrum, the aperture filter `2 J1(x) / x`, and Taylor's hypothesis: the
tilt along the wind and across the wind have the spectra

    S_L(nu) = (2/v) integral of fx^2 Phi(f) A(f)^2 dfy
    S_T(nu) = (2/v) integral of fy^2 Phi(f) A(f)^2 dfy

where `fx = nu / v`. The model averages the two, because the wind direction is unknown and the
two sensor axes average it out. The exposure keeps the fraction `sinc^2(pi nu T)` of each
frequency, so `exposure_variance_ratio` is the spectrum-weighted mean of that filter. For a
Kolmogorov spectrum the ratio is 0.93, 0.83, and 0.71 at `v T / D` of 1, 2, and 4 (the research
notes give 0.93, 0.83, and 0.72). With the default outer scale it is a little lower, because the
lost high-frequency variance is the same and the total is smaller. The estimator multiplies the
variance by the inverse. The correction needs the wind speed, and at 2 ms it stays small (about
2% in variance at 10 m/s, and between 1% and 7% for winds of 5 to 20 m/s), so a wrong wind speed
costs little. At 10 ms it matters (about 20% in variance), and the assumed wind speed is part of
the result.

**Detrending.** The estimator removes a polynomial trend from each window. The trend also absorbs
a little turbulence at the lowest frequencies. For a window of length `T_w` and a polynomial of
order `p`, the fraction of the spectrum that the fit removes at frequency `nu` is
`sum_k (2k + 1) j_k(pi nu T_w)^2` (the orthonormal Legendre basis, with `j_k` the spherical Bessel
functions). `detrend_variance_fraction` weights it with the same spectrum. For `L0` = 20 m it is
0.3% of the variance, for 100 m it is 0.8%, and for a Kolmogorov spectrum it is 8%.

**Structure function.** `autocorrelation` gives the normalized covariance of the exposure-averaged
tilt at a lag. The structure function at lag `tau` equals `2 C(0) (1 - kappa(tau))`, so the
cross-check estimator divides the measured structure function by `2 (1 - kappa)` to recover the
variance. It ignores drift and slow vibration, because a difference of two samples a few frames
apart cancels both.

**Zenith.** `r0` scales with `(cos z)^(3/5)`, so a measurement at zenith angle `z` converts to the
zenith with `zenith_r0_factor`.

**Noise.** The estimator subtracts the modeled centroid noise, so an error of the model stays in
the motion, in proportion to the share of the noise in the variance. `noise_bias` gives the bias
of `r0` that the measured error of each centroid's model (`NOISE_MODEL_ERROR`) gives at a share,
and the fast path sets the window flag `noisy` from it.

All spectra use a logarithmic grid and the trapezoid rule. The table of spectra is the only
place that needs SciPy (the Bessel function `J1`), and `tilt_spectrum` caches it for each
combination of aperture, outer scale, and wind speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from seeingmon.fastpath import _scipy as sp

FloatArray = npt.NDArray[np.float64]

ARCSEC_PER_RAD = 206264.80624709636
REFERENCE_WAVELENGTH_M = 500e-9
"""The wavelength of the reported `r0` and seeing."""

G_TILT_COEFFICIENT = 0.170
"""The coefficient `K` of the one-axis G-tilt variance (Martin 1987, Eq. 7)."""

FWHM_COEFFICIENT = 0.98
"""The coefficient of the seeing FWHM, `0.98 lambda / r0`."""


# --- seeing and r0 --------------------------------------------------------------------------


def tilt_variance_rad2(
    r0_m: float,
    aperture_m: float,
    *,
    coefficient: float = G_TILT_COEFFICIENT,
    wavelength_m: float = REFERENCE_WAVELENGTH_M,
) -> float:
    """The one-axis variance of the G-tilt for a Kolmogorov spectrum, in rad^2."""
    return float(coefficient * wavelength_m**2 * aperture_m ** (-1.0 / 3.0) * r0_m ** (-5.0 / 3.0))


def r0_from_tilt_variance(
    variance_rad2: float,
    aperture_m: float,
    *,
    coefficient: float = G_TILT_COEFFICIENT,
    wavelength_m: float = REFERENCE_WAVELENGTH_M,
) -> float:
    """The Fried parameter in meters that gives a Kolmogorov one-axis tilt variance in rad^2."""
    if not variance_rad2 > 0.0:
        raise ValueError("the variance must be positive")
    return float(
        (coefficient * wavelength_m**2 * aperture_m ** (-1.0 / 3.0) / variance_rad2) ** 0.6
    )


def seeing_fwhm_arcsec(
    r0_m: float,
    *,
    coefficient: float = FWHM_COEFFICIENT,
    wavelength_m: float = REFERENCE_WAVELENGTH_M,
) -> float:
    """The seeing FWHM in arcseconds, `0.98 lambda / r0`."""
    return coefficient * wavelength_m / r0_m * ARCSEC_PER_RAD


def lambda_over_d_arcsec(wavelength_m: float, aperture_m: float) -> float:
    """The diffraction scale `lambda / D` in arcseconds."""
    return wavelength_m / aperture_m * ARCSEC_PER_RAD


CENTROID_EXCESS = 0.10
"""The fitted coefficient of `windowed_centroid_variance_ratio`."""


def windowed_centroid_variance_ratio(radius_lambda_over_d: float) -> float:
    """The variance of the windowed, recentered centroid over the variance of the G-tilt.

    The G-tilt is the centroid of the whole image, and the formulas of this module describe it. The
    kernel measures the centroid inside a finite aperture that follows the star, and that
    centroid follows the bright core of the image a little more than the whole image does, so its
    variance is a little higher. A fixed aperture would show the opposite (it leaks light and
    responds with a gain below 1), but the aperture that follows the star does not.

    The ratio is `1 + 0.10 / R`, where `R` is the aperture radius in units of `lambda / D`. It
    comes from wave-optics simulations through frozen-flow von Karman screens at `r0` of 5, 10, and
    15 cm, a 50 mm aperture, 2 ms exposures, bin1 sampling, and aperture diameters of 9 to 28
    pixels (`R` of 3.5 to 11). The fit holds to 0.5% over that range (the scatter of the
    simulations) and varies little with `r0`. The excess is 1.6% for a 16-pixel aperture in bin1.
    A radius below 2 is clamped, because the simulations do not reach it.
    """
    return 1.0 + CENTROID_EXCESS / max(radius_lambda_over_d, 2.0)


GAUSSIAN_CENTROID_EXCESS = 0.21
"""The fitted coefficient of `gaussian_centroid_variance_ratio`."""


def gaussian_centroid_variance_ratio(fwhm_lambda_over_d: float) -> float:
    """The variance of the Gaussian-weighted centroid over the variance of the G-tilt.

    A weight that follows the star favors the bright core of the image still more than an aperture
    does, so the weighted centroid moves more like the Zernike tilt (whose variance is 7% above the
    G-tilt's, `docs/research-notes.md`, "Seeing theory") than like the centroid of the whole image.
    The ratio is `1 + 0.21 / W`, where `W` is the FWHM of the weight in units of `lambda / D`. It
    comes from wave-optics simulations of the dark sky through frozen-flow von Karman screens at
    `r0` of 5, 10, and 15 cm, a 50 mm aperture, 2 ms exposures, and bin1 sampling, with two seeds
    each, in which the variance of each frame's centroid against the injected G-tilt read 1.067 for
    a weight of 3 Airy FWHM (`W` = 3.09) and 1.053 for 4 Airy FWHM, both within 0.2% of the fit at
    every `r0`. Below 3 Airy FWHM, the bin1 pixels sample the product of the weight and the
    simulator's sharp image too coarsely: the ratio then depends on the star's position within a
    pixel (1.09 to 1.11 at 2 Airy FWHM, 1.10 to 1.22 at 1.5), so a width below `W` = 2 is clamped
    there, and the fit does not hold below 3 Airy FWHM in bin1. A real image that is wider than
    the simulator's, as the owner's recordings show, changes the ratio, and phase 3 measures it.
    """
    return 1.0 + GAUSSIAN_CENTROID_EXCESS / max(fwhm_lambda_over_d, 2.0)


NOISE_MODEL_ERROR: dict[str, float] = {"aperture": 0.06, "gaussian": 0.04}
"""The relative error of the modeled centroid noise, by centroid: the true noise is `1 + error`
times the model.

The values come from wave-optics simulations of bright skies (`docs/research-notes.md`, "The seeing
in a bright sky"), in which the noise that the sky added to each frame's centroid, against the
injected position, exceeded the model by 3.6 to 5.5% for the aperture at the exposures of the
adaptive loop (7% at 1 ms and 11% at 0.5 ms) and by up to 3.4% for the Gaussian-weighted centroid.
The aperture recenters on its own noisy centroid, which the model of a fixed aperture leaves out.
"""


def noise_model_error(centroid: str) -> float:
    """The relative error of the modeled noise of a centroid: `aperture` or `gaussian`."""
    return NOISE_MODEL_ERROR[centroid]


def noise_bias(noise_share: float, model_error: float) -> float:
    """The size of the bias of `r0` that an error of the noise model gives, as a fraction of `r0`.

    `noise_share` is the modeled noise over the motion variance that the estimator kept, and
    `model_error` the relative error of the model: the true noise is `1 + model_error` times the
    model. The estimate then keeps `model_error * noise_share` of the motion too much, and `r0`
    reads `(1 + model_error * noise_share)^(-3/5)` of the truth. The result is the size of the
    difference from 1: 1 for an infinite share (the motion fell below the noise), and infinite when
    a model that is too high leaves no motion.
    """
    if not noise_share < math.inf:
        return 1.0
    kept = 1.0 + model_error * noise_share
    if not kept > 0.0:
        return math.inf
    return abs(math.pow(kept, -0.6) - 1.0)


def zenith_r0_factor(zenith_angle_deg: float) -> float:
    """The factor `(cos z)^(3/5)` that turns `r0` at the zenith into `r0` along the sight line."""
    if not 0.0 <= zenith_angle_deg < 89.0:
        raise ValueError("the zenith angle must be between 0 and 89 degrees")
    return float(math.cos(math.radians(zenith_angle_deg)) ** 0.6)


# --- outer scale ----------------------------------------------------------------------------


def _tilt_integral(u0: float) -> float:
    """The integral of `(u^2 + u0^2)^(-11/6) 4 J1(u)^2 u` over `u` from 0 to infinity."""

    def integrand(s: float) -> float:
        u = s**3
        return float(
            (u * u + u0 * u0) ** (-11.0 / 6.0) * 4.0 * sp.j1_scalar(u) ** 2 * u * 3 * s * s
        )

    edges = [0.0, 0.05, 0.3, 1.0, 1.7, 2.5, 3.2, 3.9]  # u = s^3 up to about 59
    total = sum(
        sp.quad(integrand, low, high, limit=400, epsrel=1e-10) for low, high in pairwise(edges)
    )
    # Beyond u = 59, 4 J1^2 averages to 4 / (pi u).
    u_start = edges[-1] ** 3
    return float(total + (4.0 / math.pi) * (3.0 / 8.0) * (u_start**2 + u0 * u0) ** (-4.0 / 3.0))


@lru_cache(maxsize=64)
def outer_scale_ratio(aperture_m: float, outer_scale_m: float) -> float:
    """The factor by which a finite outer scale lowers the G-tilt variance of a circular aperture.

    The ratio is the von Karman integral over the Kolmogorov integral. It depends only on
    `D / L0`. It is 1 for an infinite outer scale.
    """
    if aperture_m <= 0.0 or outer_scale_m <= 0.0:
        raise ValueError("the aperture and the outer scale must be positive")
    if math.isinf(outer_scale_m):
        return 1.0
    u0 = round(math.pi * aperture_m / outer_scale_m, 12)
    return _tilt_integral(u0) / _tilt_integral(0.0)


# --- the temporal spectrum of the tilt ------------------------------------------------------


def _aperture_filter(frequency_per_m: FloatArray, aperture_m: float) -> FloatArray:
    """The Fourier transform of a circular aperture's area weight, `2 J1(u) / u`."""
    u = math.pi * aperture_m * frequency_per_m
    out = np.ones_like(u)
    nonzero = u > 1e-9
    out[nonzero] = 2.0 * sp.j1(u[nonzero]) / u[nonzero]
    return out


def detrend_response(nu_hz: FloatArray, window_s: float, order: int) -> FloatArray:
    """The share of the variance at each frequency that a polynomial fit of `order` removes.

    The window has the length `window_s`. The share is `sum_k (2k + 1) j_k(pi nu T)^2` over the
    orthonormal polynomials `k = 0 .. order`, with `j_k` the spherical Bessel functions. It is 1 for
    `nu T` near 0 and falls off fast above `1 / T`.
    """
    x = math.pi * nu_hz * window_s
    out = np.zeros_like(nu_hz)
    for k in range(order + 1):
        out += (2 * k + 1) * sp.spherical_jn(k, x) ** 2
    return out


@dataclass(frozen=True, slots=True, eq=False)
class TiltSpectrum:
    """The temporal spectrum of the one-axis G-tilt of a single frozen layer, as quadrature weights.

    `weight` holds `S(nu) nu dln(nu)` on the grid `nu_hz`, so that the integral of `S(nu) g(nu)`
    over frequency is `sum(weight * g)`. The weights sum to 1. The spectrum averages the tilt
    along and across the wind.
    """

    aperture_m: float
    outer_scale_m: float
    wind_ms: float
    nu_hz: FloatArray
    weight: FloatArray

    def exposure_variance_ratio(self, exposure_s: float) -> float:
        """The share of the variance that an exposure keeps: the weighted mean of `sinc^2`."""
        if exposure_s <= 0.0:
            return 1.0
        return float(np.dot(self.weight, np.sinc(self.nu_hz * exposure_s) ** 2))

    def autocorrelation(self, lags_s: FloatArray, exposure_s: float) -> FloatArray:
        """The normalized covariance of the exposure-averaged tilt at each lag."""
        filtered = self.weight * np.sinc(self.nu_hz * exposure_s) ** 2
        cosines = np.cos(2.0 * math.pi * np.outer(lags_s, self.nu_hz))
        return np.asarray(cosines @ filtered / filtered.sum(), dtype=np.float64)

    def detrend_variance_fraction(self, window_s: float, order: int, exposure_s: float) -> float:
        """The share of the (exposure-averaged) variance that a polynomial fit removes."""
        filtered = self.weight * np.sinc(self.nu_hz * exposure_s) ** 2
        return float(
            np.dot(filtered, detrend_response(self.nu_hz, window_s, order)) / filtered.sum()
        )


@lru_cache(maxsize=8)
def tilt_spectrum(
    aperture_m: float,
    outer_scale_m: float,
    wind_ms: float,
    *,
    points: int = 800,
    cross_points: int = 350,
) -> TiltSpectrum:
    """Build the tilt spectrum of a single frozen layer. The first call for a set of parameters
    takes about 0.1 s, and the result is cached."""
    if aperture_m <= 0.0 or outer_scale_m <= 0.0 or wind_ms <= 0.0:
        raise ValueError("the aperture, the outer scale, and the wind speed must be positive")
    f0 = 0.0 if math.isinf(outer_scale_m) else 1.0 / outer_scale_m
    corner = wind_ms / aperture_m  # the frequency where the aperture filter acts, in Hz
    nu = np.geomspace(1e-9 * corner, 50.0 * corner, points)
    fx = (nu / wind_ms)[:, None]
    fy = np.geomspace(1e-7, 60.0 / aperture_m, cross_points)[None, :]
    f2 = fx * fx + fy * fy
    spectrum = (f2 + f0 * f0) ** (-11.0 / 6.0) * _aperture_filter(np.sqrt(f2), aperture_m) ** 2
    ln_fy = np.log(fy[0])
    along = np.trapezoid(fx * fx * spectrum * fy, ln_fy, axis=1)
    across = np.trapezoid(fy * fy * spectrum * fy, ln_fy, axis=1)
    density = 0.5 * (along + across)  # per hertz, up to a constant
    weight = density * nu
    weight = weight / weight.sum()
    return TiltSpectrum(aperture_m, outer_scale_m, wind_ms, nu, weight)
