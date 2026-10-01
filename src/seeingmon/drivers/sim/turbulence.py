"""Frozen-flow, multi-layer von Karman turbulence for the simulator.

The model gives the phase over the pupil of the telescope at any time. Each layer is a phase
screen that slides across the pupil at its wind velocity (Taylor's frozen flow). The model is a
pure function of the seed and the time, so you can ask for any time in any order.

**How a layer is built.** A layer's phase has two parts that share one von Karman spectrum:

- The *screen part* holds the spatial frequencies from a few cycles per screen length up to the
  Nyquist frequency of the screen grid. An FFT of Gaussian noise builds it as a periodic
  screen. To keep a long run from repeating, the layer replaces its screen every time the
  pupil crosses the screen length. The old and the new screen cross-fade, so the statistics
  stay exact at every instant.
- The *low-frequency part* holds the frequencies below the lowest screen frequency, down to
  scales of many kilometres. It is a sum of a few hundred sinusoids with random phases and
  jittered frequencies. A sinusoid moves exactly with the wind, so this part never repeats.

Image motion comes mostly from scales a few times larger than the aperture, so the low
frequencies matter: with a 50 mm aperture and a pure Kolmogorov spectrum, 20% to 30% of the
image-motion variance comes from scales above a metre. Both parts use the *moment-matched*
weight for each spectral cell. That weight reproduces the integral of the spectrum times the
squared frequency, which is what the gradient (and so the tilt) depends on. It makes the
expected image-motion variance exact to better than 0.1% for any outer scale.

**Conventions.** Phase is in radians at 500 nm, and `r0` is the Fried parameter at 500 nm.
Scale the phase by 500 nm over the wavelength to get the phase at another wavelength. The
pupil plane uses meters, with `x` along the columns and `y` along the rows of the sensor.
Wind direction is the angle, in degrees, that the wind blows toward, from `+x` toward `+y`.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from seeingmon.drivers.sim import _scipy as sp
from seeingmon.drivers.sim.params import ARCSEC_PER_RAD

FloatArray = npt.NDArray[np.float64]
SingleArray = npt.NDArray[np.float32]

REFERENCE_WAVELENGTH_M = 500e-9
"""The wavelength of `r0` and of the stored phase."""

_TWO_PI = 2.0 * math.pi
_HOLE_CELLS = 1  # the screen omits the (2 * 1 + 1)^2 lowest cells, and sinusoids fill them
_LOW_FREQUENCY_LEVELS = 6
_SUBCELLS = 3
_MAX_CACHED_PAIRS = 3


# --- analytic results -----------------------------------------------------------------------


def kolmogorov_psd_constant() -> float:
    """The constant `c` in the phase spectrum `c r0^(-5/3) f^(-11/3)` (about 0.0229).

    With this constant the phase structure function is `6.88 (r / r0)^(5/3)`.
    """
    return (
        math.pow(math.gamma(11 / 6), 2)
        / (2 * math.pow(math.pi, 11 / 3))
        * math.pow(24 / 5 * math.gamma(6 / 5), 5 / 6)
    )


def von_karman_psd(
    frequency_per_m: FloatArray, r0_m: float, outer_scale_m: float = math.inf
) -> FloatArray:
    """The von Karman phase power spectral density in rad^2 per (cycle/m)^2.

    `frequency_per_m` is the spatial frequency in cycles per meter. `r0_m` is the Fried
    parameter at the wavelength of the phase. An infinite `outer_scale_m` gives pure
    Kolmogorov turbulence.
    """
    f0_sq = 0.0 if math.isinf(outer_scale_m) else (1.0 / outer_scale_m) ** 2
    return np.asarray(
        kolmogorov_psd_constant()
        * math.pow(r0_m, -5 / 3)
        * (frequency_per_m**2 + f0_sq) ** (-11 / 6),
        dtype=np.float64,
    )


def _tilt_integral(u0: float) -> float:
    """The integral of `(u^2 + u0^2)^(-11/6) 4 J1(u)^2 u` over `u` from 0 to infinity."""

    def integrand(s: float) -> float:
        u = s**3
        value = math.pow(u * u + u0 * u0, -11 / 6) * 4.0 * sp.j1_scalar(u) ** 2 * u
        return value * 3.0 * s * s

    total = 0.0
    edges = [0.0, 0.05, 0.3, 1.0, 1.7, 2.5, 3.2, 3.9]  # u = s^3 up to about 60
    for low, high in pairwise(edges):
        total += sp.quad(integrand, low, high, limit=400, epsrel=1e-10)
    # Beyond u = 60, 4 J1^2 averages to 4 / (pi u).
    u_start = edges[-1] ** 3
    total += (4 / math.pi) * (3 / 8) * math.pow(u_start**2 + u0 * u0, -4 / 3)
    return total


@lru_cache(maxsize=64)
def _tilt_coefficient(u0: float) -> float:
    return kolmogorov_psd_constant() * math.pow(math.pi, 2 / 3) * _tilt_integral(u0)


def g_tilt_coefficient(aperture_m: float, outer_scale_m: float = math.inf) -> float:
    """The coefficient `K` in the one-axis G-tilt variance `K lambda^2 D^(-1/3) r0^(-5/3)`.

    The G-tilt is the centroid of the image. For pure Kolmogorov turbulence, `K` is 0.1698.
    The result comes from numerical integration of the von Karman spectrum, so a finite
    outer scale lowers it.
    """
    u0 = 0.0 if math.isinf(outer_scale_m) else math.pi * aperture_m / outer_scale_m
    return _tilt_coefficient(round(u0, 12))


def g_tilt_variance_rad2(
    aperture_m: float,
    r0_m: float,
    outer_scale_m: float = math.inf,
    wavelength_m: float = REFERENCE_WAVELENGTH_M,
) -> float:
    """The one-axis variance of the image centroid in rad^2.

    `r0_m` is the Fried parameter at `wavelength_m`. The result does not depend on the
    wavelength when you scale `r0` as `lambda^(6/5)`.
    """
    return (
        g_tilt_coefficient(aperture_m, outer_scale_m)
        * wavelength_m**2
        * math.pow(aperture_m, -1 / 3)
        * math.pow(r0_m, -5 / 3)
    )


def g_tilt_rms_arcsec(
    aperture_m: float, r0_500nm_m: float, outer_scale_m: float = math.inf
) -> float:
    """The one-axis rms image motion in arcseconds, for `r0` at 500 nm."""
    variance = g_tilt_variance_rad2(aperture_m, r0_500nm_m, outer_scale_m)
    return math.sqrt(variance) * ARCSEC_PER_RAD


def outer_scale_ratio(aperture_m: float, outer_scale_m: float) -> float:
    """The factor by which a finite outer scale lowers the G-tilt variance."""
    return g_tilt_coefficient(aperture_m, outer_scale_m) / g_tilt_coefficient(aperture_m)


def seeing_fwhm_arcsec(
    r0_m: float, wavelength_m: float = REFERENCE_WAVELENGTH_M, outer_scale_m: float = math.inf
) -> float:
    """The seeing FWHM in arcseconds, `0.98 lambda / r0`, for Kolmogorov turbulence.

    With a finite outer scale, the result shrinks by the von Karman factor
    `sqrt(1 - 2.183 (r0 / L0)^0.356)`, which holds for `L0 / r0` above 20.
    """
    fwhm = 0.98 * wavelength_m / r0_m * ARCSEC_PER_RAD
    if not math.isinf(outer_scale_m) and outer_scale_m / r0_m > 20.0:
        fwhm *= math.sqrt(max(0.0, 1.0 - 2.183 * math.pow(r0_m / outer_scale_m, 0.356)))
    return fwhm


def r0_at_wavelength(r0_500nm_m: float, wavelength_m: float) -> float:
    """`r0` at another wavelength: it scales as the wavelength to the power 6/5."""
    return r0_500nm_m * math.pow(wavelength_m / REFERENCE_WAVELENGTH_M, 6 / 5)


def r0_at_zenith_angle(r0_zenith_m: float, zenith_angle_deg: float) -> float:
    """`r0` along a line of sight: it scales as `(cos z)^(3/5)`."""
    return r0_zenith_m * math.pow(math.cos(math.radians(zenith_angle_deg)), 3 / 5)


# --- configuration --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Layer:
    """One turbulent layer.

    `cn2_fraction` is the layer's share of the turbulence strength. The model normalizes the
    fractions of all layers to sum to 1. `wind_direction_deg` is the direction the wind blows
    toward, from `+x` (the sensor columns) toward `+y` (the sensor rows).
    """

    cn2_fraction: float = 1.0
    wind_speed_m_s: float = 10.0
    wind_direction_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.cn2_fraction <= 0 or self.wind_speed_m_s < 0:
            raise ValueError(
                "cn2_fraction must be positive and the wind speed must not be negative"
            )

    @property
    def velocity_m_s(self) -> tuple[float, float]:
        angle = math.radians(self.wind_direction_deg)
        return (self.wind_speed_m_s * math.cos(angle), self.wind_speed_m_s * math.sin(angle))


DEFAULT_LAYERS = (
    Layer(0.60, 3.0, 40.0),
    Layer(0.25, 12.0, 110.0),
    Layer(0.15, 25.0, 70.0),
)
"""A ground layer, a mid-level layer, and a fast high layer."""


@dataclass(frozen=True, slots=True)
class TurbulenceConfig:
    """What the turbulence looks like.

    `r0_m` is the Fried parameter at 500 nm at the zenith. `zenith_angle_deg` scales it by
    `(cos z)^(3/5)`. `outer_scale_m` may be `math.inf`. `screen_points` is the size of the
    square screen grid, and the grid spacing is the aperture over 16, so the screen covers
    `screen_points / 16` apertures. A larger screen carries more of the low frequencies in the
    screen part and costs more memory and time. `r0_schedule` lists `(time_s, r0_m)` points
    that the model interpolates linearly. Before the first point and after the last, the
    nearest value holds. `boiling=False` keeps the first screen forever, which gives an exactly
    periodic screen for short experiments.
    """

    r0_m: float = 0.10
    layers: tuple[Layer, ...] = DEFAULT_LAYERS
    outer_scale_m: float = 20.0
    zenith_angle_deg: float = 0.0
    seed: int = 1
    screen_points: int = 512
    r0_schedule: tuple[tuple[float, float], ...] = ()
    boiling: bool = True

    def __post_init__(self) -> None:
        if self.r0_m <= 0 or self.outer_scale_m <= 0:
            raise ValueError("r0_m and outer_scale_m must be positive")
        if not self.layers:
            raise ValueError("at least one layer is required")
        if not 0 <= self.zenith_angle_deg < 89:
            raise ValueError("zenith_angle_deg must be between 0 and 89")
        if self.screen_points < 32 or self.screen_points % 2:
            raise ValueError("screen_points must be an even number of at least 32")
        times = [time for time, _ in self.r0_schedule]
        if times != sorted(times) or any(r0 <= 0 for _, r0 in self.r0_schedule):
            raise ValueError("r0_schedule must have increasing times and positive r0")


# --- grids ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridSpec:
    """A regular grid of points in the pupil plane, in meters, centered on the pupil.

    The first column is at `x0` and the first row at `y0`. Spacing is `dx` along both axes.
    """

    x0: float
    y0: float
    dx: float
    nx: int
    ny: int


@dataclass(frozen=True, slots=True, eq=False)
class PupilGrid:
    """The sampled circular pupil and the weights that turn a phase map into image motion.

    `coverage` holds the fraction of each grid cell inside the circle, which also serves as
    the amplitude of the pupil. `tilt_x` and `tilt_y` are weight maps: the sum of a phase map
    times `tilt_x` is the area-weighted mean of the central-difference phase gradient along
    `x`, in rad/m. The grid has an empty margin cell on every side.
    """

    spec: GridSpec
    aperture_m: float
    coverage: FloatArray
    tilt_x: FloatArray
    tilt_y: FloatArray

    @classmethod
    def circular(cls, aperture_m: float, spacing_m: float) -> PupilGrid:
        """Sample a circular pupil of diameter `aperture_m` at `spacing_m`."""
        if spacing_m <= 0 or spacing_m >= aperture_m / 4:
            raise ValueError("the pupil spacing must be positive and under a quarter aperture")
        n = math.ceil(aperture_m / spacing_m) + 2
        n += n % 2
        centre = (n - 1) / 2
        supersample = 8
        offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
        coords = ((np.arange(n) - centre)[:, None] + offsets[None, :]).reshape(-1) * spacing_m
        xs, ys = np.meshgrid(coords, coords)
        inside = (xs * xs + ys * ys <= (aperture_m / 2) ** 2).astype(np.float64)
        coverage = inside.reshape(n, supersample, n, supersample).mean(axis=(1, 3))
        # Summation by parts turns the mean of the central-difference gradient into a weighted
        # sum of the phase. The weights are the negative gradient of the coverage.
        padded = np.pad(coverage, 1)
        norm = 2.0 * spacing_m * float(coverage.sum())
        tilt_x = -(padded[1:-1, 2:] - padded[1:-1, :-2]) / norm
        tilt_y = -(padded[2:, 1:-1] - padded[:-2, 1:-1]) / norm
        spec = GridSpec(-centre * spacing_m, -centre * spacing_m, spacing_m, n, n)
        return cls(spec, aperture_m, coverage, tilt_x, tilt_y)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.spec.ny, self.spec.nx)

    def coordinates(self) -> tuple[FloatArray, FloatArray]:
        """The `x` and `y` coordinates of the grid columns and rows, in meters."""
        spec = self.spec
        return (
            spec.x0 + spec.dx * np.arange(spec.nx, dtype=np.float64),
            spec.y0 + spec.dx * np.arange(spec.ny, dtype=np.float64),
        )

    def tilt_rad(self, phase_500nm: FloatArray) -> tuple[float, float]:
        """The G-tilt of a phase map, in radians of image motion along `x` and `y`."""
        scale = REFERENCE_WAVELENGTH_M / _TWO_PI
        return (
            scale * float(np.sum(phase_500nm * self.tilt_x)),
            scale * float(np.sum(phase_500nm * self.tilt_y)),
        )


# --- spectral cells -------------------------------------------------------------------------


def _psd_unit(f: FloatArray, f0: float) -> FloatArray:
    """The spectrum for `r0 = 1 m`, in rad^2 per (cycle/m)^2."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(kolmogorov_psd_constant() * (f * f + f0 * f0) ** (-11 / 6))


def _psd_scalar(f: float, f0: float) -> float:
    return kolmogorov_psd_constant() * math.pow(f * f + f0 * f0, -11 / 6)


def _square_moment(half_width: float, f0: float) -> float:
    """The integral of `Phi(f) |f|^2` over the square `|fx|, |fy| <= half_width`, for `r0 = 1`."""

    def radial(theta: float) -> float:
        r_max = half_width / math.cos(theta)
        s_max = r_max ** (1 / 3)

        def g(s: float) -> float:
            r = s**3
            return _psd_scalar(r, f0) * r**3 * 3.0 * s * s

        return sp.quad(g, 0.0, s_max, limit=200, epsrel=1e-10)

    # Eight congruent octants.
    return 8.0 * sp.quad(radial, 0.0, math.pi / 4, limit=100, epsrel=1e-9)


def _cell_moments(
    centers_x: FloatArray, centers_y: FloatArray, half_width: float, f0: float, order: int
) -> FloatArray:
    """The integral of `Phi(f) |f|^2` over squares, by Gauss-Legendre quadrature.

    The squares must not contain the origin, where the Kolmogorov spectrum diverges.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = nodes * half_width
    weights = weights * half_width
    total = np.zeros_like(centers_x, dtype=np.float64)
    with np.errstate(all="ignore"):
        for i in range(order):
            for j in range(order):
                gx = centers_x + nodes[i]
                gy = centers_y + nodes[j]
                f_sq = gx * gx + gy * gy
                total += _psd_unit(np.sqrt(f_sq), f0) * f_sq * weights[i] * weights[j]
    return total


@dataclass(frozen=True, slots=True, eq=False)
class _LowFrequencyPlan:
    """The cells that the sinusoids fill: the square hole below the screen's lowest frequency.

    Each row is one sinusoid: the centre of its frequency cell, the width of the cell (the
    frequency jitters within it), and the cell's moment `integral(Phi |f|^2)` for `r0 = 1`.
    """

    centers: FloatArray  # (K, 2) cycles/m, columns (fx, fy)
    widths: FloatArray  # (K,) cycles/m
    moments: FloatArray  # (K,) moment of the cell for r0 = 1 m


@lru_cache(maxsize=32)
def _low_frequency_plan(delta_f: float, f0: float, levels: int) -> _LowFrequencyPlan:
    """Divide the hole `|f| <= 1.5 delta_f` into cells, with finer cells near the origin."""
    centers: list[tuple[float, float]] = []
    widths: list[float] = []
    moments: list[float] = []
    # The eight cells around the central one, each split into 3 x 3 subcells.
    sub_width = delta_f / _SUBCELLS
    for ix in range(-_HOLE_CELLS, _HOLE_CELLS + 1):
        for iy in range(-_HOLE_CELLS, _HOLE_CELLS + 1):
            if ix == 0 and iy == 0:
                continue
            for sx in range(_SUBCELLS):
                for sy in range(_SUBCELLS):
                    centers.append(
                        (
                            (ix + (sx - (_SUBCELLS - 1) / 2) / _SUBCELLS) * delta_f,
                            (iy + (sy - (_SUBCELLS - 1) / 2) / _SUBCELLS) * delta_f,
                        )
                    )
                    widths.append(sub_width)
    centers_arr = np.asarray(centers, dtype=np.float64)
    moments_arr = _cell_moments(centers_arr[:, 0], centers_arr[:, 1], sub_width / 2, f0, order=6)
    # Scale the ring of outer cells to the exact integral, so that no quadrature error remains.
    exact_outer = _square_moment(_HOLE_CELLS * delta_f + delta_f / 2, f0) - _square_moment(
        delta_f / 2, f0
    )
    moments_arr *= exact_outer / float(moments_arr.sum())
    moments.extend(float(m) for m in moments_arr)
    # The central cell: a hierarchy of 3 x 3 subdivisions, each keeping its outer eight cells.
    half = delta_f / 2
    for _ in range(levels):
        sub = 2 * half / _SUBCELLS  # the width of a subcell
        ring_centers = [
            (ix * sub, iy * sub) for ix in (-1, 0, 1) for iy in (-1, 0, 1) if ix != 0 or iy != 0
        ]
        ring = np.asarray(ring_centers, dtype=np.float64)
        ring_moments = _cell_moments(ring[:, 0], ring[:, 1], sub / 2, f0, order=6)
        exact_ring = _square_moment(half, f0) - _square_moment(sub / 2, f0)
        ring_moments *= exact_ring / float(ring_moments.sum())
        for (cx, cy), moment in zip(ring_centers, ring_moments, strict=True):
            centers.append((cx, cy))
            widths.append(sub)
            moments.append(float(moment))
        half = sub / 2
    # What remains in the central square goes into one sinusoid along each axis. Each carries
    # half of the moment, so the x and y image motion each get exactly half.
    lump = _square_moment(half, f0)
    for cx, cy in ((half, 0.0), (0.0, half)):
        centers.append((cx, cy))
        widths.append(0.0)
        moments.append(lump / 2)
    return _LowFrequencyPlan(
        np.asarray(centers, dtype=np.float64),
        np.asarray(widths, dtype=np.float64),
        np.asarray(moments, dtype=np.float64),
    )


@lru_cache(maxsize=8)
def _screen_amplitudes(n: int, length_m: float, f0: float) -> SingleArray:
    """The amplitude of each screen cell for `r0 = 1`, in the layout of `numpy.fft.ifft2`.

    The cell amplitude is `sqrt(moment / |f|^2)`: the weight that keeps the integral of the
    spectrum times `|f|^2`. The cells of the low-frequency hole are zero.
    """
    delta_f = 1.0 / length_m
    index = np.arange(-n // 2, n // 2)
    fy, fx = np.meshgrid(index * delta_f, index * delta_f, indexing="ij")
    moments = _cell_moments(fx, fy, delta_f / 2, f0, order=3)
    f_sq = fx * fx + fy * fy
    with np.errstate(all="ignore"):
        amplitude = np.sqrt(np.where(f_sq > 0, moments / f_sq, 0.0))
    hole = (np.abs(index)[:, None] <= _HOLE_CELLS) & (np.abs(index)[None, :] <= _HOLE_CELLS)
    amplitude[hole] = 0.0
    return np.ascontiguousarray(np.fft.ifftshift(amplitude), dtype=np.float32)


# --- the model ------------------------------------------------------------------------------


def _keys_weights(frac: FloatArray) -> FloatArray:
    """Catmull-Rom cubic weights for the four samples around a fractional position."""
    t = frac
    w0 = ((-t + 2.0) * t - 1.0) * t / 2.0
    w1 = ((3.0 * t - 5.0) * t * t + 2.0) / 2.0
    w2 = ((-3.0 * t + 4.0) * t + 1.0) * t / 2.0
    w3 = (t - 1.0) * t * t / 2.0
    return np.stack([w0, w1, w2, w3])


def _axis_weights(
    origins: FloatArray, step: float, count: int, spacing: float, size: int
) -> tuple[npt.NDArray[np.intp], SingleArray]:
    """Cubic interpolation along one axis of a periodic grid, for several shifts at once.

    `origins` holds the position of the first sample for each shift. The samples are `count`
    points at `origin + step * i`. The result is the wrapped indices of a patch of the grid, with
    shape `(shifts, patch)`, and a weight array of shape `(shifts, count, patch)` that maps each
    patch to the samples.
    """
    shifts = len(origins)
    u = origins[:, None] / spacing + (step / spacing) * np.arange(count, dtype=np.float64)
    base = np.floor(u)
    frac = u - base
    base_i = base.astype(np.int64)
    low = base_i[:, 0]
    patch = math.floor((count - 1) * step / spacing) + 5
    relative = base_i - low[:, None]
    weights4 = _keys_weights(frac).astype(np.float32)
    weights = np.zeros((shifts, count, patch), dtype=np.float32)
    shift_index = np.arange(shifts)[:, None]
    sample_index = np.arange(count)[None, :]
    for k in range(4):
        weights[shift_index, sample_index, relative + k] = weights4[k]
    indices = (low[:, None] - 1 + np.arange(patch)[None, :]) % size
    return indices, weights


@dataclass(slots=True)
class _LayerState:
    """Per-layer data: the layer's weight and a small cache of its screens."""

    weight: float
    scaled_amplitudes: SingleArray | None = None
    pairs: OrderedDict[int, tuple[SingleArray, SingleArray]] = field(default_factory=OrderedDict)


class TurbulenceModel:
    """Phase over the pupil as a function of time, for a stack of frozen-flow layers.

    Construction is cheap. The model builds screens when a time first needs them. Times are
    seconds since an epoch that the caller picks, and may be negative.
    """

    def __init__(self, config: TurbulenceConfig, aperture_m: float) -> None:
        if aperture_m <= 0:
            raise ValueError("aperture_m must be positive")
        self._config = config
        self._aperture_m = aperture_m
        self._n = config.screen_points
        self._spacing_m = aperture_m / 16.0
        self._length_m = self._n * self._spacing_m
        self._delta_f = 1.0 / self._length_m
        f0 = 0.0 if math.isinf(config.outer_scale_m) else 1.0 / config.outer_scale_m
        self._f0 = f0
        self._r0_observed = r0_at_zenith_angle(config.r0_m, config.zenith_angle_deg)
        total = sum(layer.cn2_fraction for layer in config.layers)
        self._amplitudes = _screen_amplitudes(self._n, self._length_m, f0)
        self._plan = _low_frequency_plan(self._delta_f, f0, _LOW_FREQUENCY_LEVELS)
        weights = [layer.cn2_fraction / total for layer in config.layers]
        self._layers = [_LayerState(weight) for weight in weights]
        velocities = np.asarray([layer.velocity_m_s for layer in config.layers], dtype=np.float64)
        self._vx = velocities[:, 0]
        self._vy = velocities[:, 1]
        self._speed = np.asarray([layer.wind_speed_m_s for layer in config.layers])
        offsets = np.empty((len(config.layers), 2), dtype=np.float64)
        sinusoids = [self._build_sinusoids(i, w, offsets) for i, w in enumerate(weights)]
        self._offset_x = offsets[:, 0]
        self._offset_y = offsets[:, 1]
        # Stack the sinusoids of all layers: the polynomial coefficients add across layers.
        self._lf_amplitude = np.concatenate([item[0] for item in sinusoids])
        self._lf_phase0 = np.concatenate([item[1] for item in sinusoids])
        self._lf_rate = np.concatenate([item[2] for item in sinusoids])
        kx = np.concatenate([item[3] for item in sinusoids])
        ky = np.concatenate([item[4] for item in sinusoids])
        self._lf_basis_c = np.stack([np.ones_like(kx), kx * kx, kx * ky, ky * ky])
        self._lf_basis_s = np.stack([kx, ky, kx**3, kx * kx * ky, kx * ky * ky, ky**3])

    # --- properties ---

    @property
    def config(self) -> TurbulenceConfig:
        return self._config

    @property
    def aperture_m(self) -> float:
        return self._aperture_m

    @property
    def screen_length_m(self) -> float:
        """The side of the periodic screen, in meters."""
        return self._length_m

    def r0_zenith_m(self, t_s: float = 0.0) -> float:
        """`r0` at 500 nm at the zenith, at time `t_s`."""
        schedule = self._config.r0_schedule
        if not schedule:
            return self._config.r0_m
        times = [time for time, _ in schedule]
        values = [r0 for _, r0 in schedule]
        return float(np.interp(t_s, times, values))

    def r0_observed_m(self, t_s: float = 0.0) -> float:
        """`r0` at 500 nm along the line of sight, at time `t_s`."""
        return r0_at_zenith_angle(self.r0_zenith_m(t_s), self._config.zenith_angle_deg)

    def _gain(self, t_s: float) -> float:
        """The factor that scales the stored phase to the scheduled `r0`."""
        if not self._config.r0_schedule:
            return 1.0
        return math.pow(self._config.r0_m / self.r0_zenith_m(t_s), 5 / 6)

    # --- construction ---

    def _build_sinusoids(
        self, index: int, weight: float, offsets: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Draw the low-frequency sinusoids of one layer.

        Returns the amplitude, the starting phase, the phase rate per second of wind, and the
        angular wavenumbers `kx` and `ky` of each sinusoid. It also stores the layer's screen
        offset in `offsets`.
        """
        rng = np.random.default_rng(np.random.SeedSequence([self._config.seed, index, 0]))
        plan = self._plan
        jitter = (rng.random((len(plan.widths), 2)) - 0.5) * plan.widths[:, None]
        freq = plan.centers + jitter
        f_sq = np.einsum("ij,ij->i", freq, freq)
        # The hole holds the same spectrum as the screen. Scale it to this layer.
        scale = weight * math.pow(self._r0_observed, -5 / 3)
        amplitude = np.sqrt(2.0 * scale * plan.moments / f_sq)
        phase0 = rng.random(len(amplitude)) * _TWO_PI
        kx = _TWO_PI * freq[:, 0]
        ky = _TWO_PI * freq[:, 1]
        rate = kx * self._vx[index] + ky * self._vy[index]
        offsets[index] = rng.random(2) * self._length_m
        return amplitude, phase0, rate, kx, ky

    def _screen(self, index: int, slot: int) -> SingleArray:
        """The screen for one slot: the real or the imaginary part of one FFT."""
        state = self._layers[index]
        pair, half = slot >> 1, slot & 1
        cached = state.pairs.get(pair)
        if cached is None:
            n = self._n
            seed = self._config.seed
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, 1, pair + (1 << 40)]))
            coefficients = np.empty((n, n), dtype=np.complex64)
            parts: SingleArray = coefficients.view(np.float32).reshape(n, n, 2)
            rng.standard_normal(dtype=np.float32, out=parts)
            if state.scaled_amplitudes is None:
                scale = math.sqrt(state.weight * math.pow(self._r0_observed, -5 / 3)) * n * n
                state.scaled_amplitudes = (self._amplitudes * np.float32(scale))[:, :, None]
            parts *= state.scaled_amplitudes
            field_c = sp.ifft2(coefficients)
            # The real and imaginary parts are two independent screens. Contiguous copies index
            # much faster than strided views.
            cached = (
                np.ascontiguousarray(field_c.real, dtype=np.float32),
                np.ascontiguousarray(field_c.imag, dtype=np.float32),
            )
            state.pairs[pair] = cached
            while len(state.pairs) > _MAX_CACHED_PAIRS:
                state.pairs.popitem(last=False)
        else:
            state.pairs.move_to_end(pair)
        return cached[half]

    # --- evaluation ---

    def phase(self, t_s: float, grid: GridSpec) -> FloatArray:
        """The total phase on a grid, in radians at 500 nm, at time `t_s`."""
        series = self.phase_series(np.asarray([t_s], dtype=np.float64), grid)
        return np.asarray(series[0], dtype=np.float64)

    def phase_series(self, times_s: FloatArray, grid: GridSpec) -> FloatArray:
        """The total phase on a grid at several times, with shape `(times, ny, nx)`.

        The phase is in radians at 500 nm. One call is much cheaper than one call per time.
        """
        times = np.asarray(times_s, dtype=np.float64)
        xs = grid.x0 + grid.dx * np.arange(grid.nx, dtype=np.float64)
        ys = grid.y0 + grid.dx * np.arange(grid.ny, dtype=np.float64)
        result = self._screen_phase(times, grid)
        result += self._sinusoid_phase(times, xs, ys)
        if self._config.r0_schedule:
            gains = np.asarray([self._gain(float(t)) for t in times])
            result *= gains[:, None, None]
        return result

    def _screen_phase(self, times: FloatArray, grid: GridSpec) -> FloatArray:
        """The screen part, summed over the layers."""
        n = self._n
        layers = len(self._layers)
        count = len(times)
        if self._config.boiling:
            travel = self._speed[:, None] * times[None, :] / self._length_m
            slots = np.floor(travel).astype(np.int64)
            fractions = travel - slots
        else:
            slots = np.zeros((layers, count), dtype=np.int64)
            fractions = np.zeros((layers, count), dtype=np.float64)
        origin_x = grid.x0 + self._offset_x[:, None] - self._vx[:, None] * times[None, :]
        origin_y = grid.y0 + self._offset_y[:, None] - self._vy[:, None] * times[None, :]
        ix, wx = _axis_weights(origin_x.reshape(-1), grid.dx, grid.nx, self._spacing_m, n)
        iy, wy = _axis_weights(origin_y.reshape(-1), grid.dx, grid.ny, self._spacing_m, n)
        patch = np.empty((layers * count, iy.shape[1], ix.shape[1]), dtype=np.float32)
        for layer in range(layers):
            base = layer * count
            for slot in np.unique(slots[layer]):
                selected = np.nonzero(slots[layer] == slot)[0]
                rows = base + selected
                window = (iy[rows][:, :, None], ix[rows][:, None, :])
                first = self._screen(layer, int(slot))[window]
                angles = 0.5 * math.pi * fractions[layer, selected]
                if np.any(angles > 0.0):
                    second = self._screen(layer, int(slot) + 1)[window]
                    cos = np.cos(angles).astype(np.float32)[:, None, None]
                    sin = np.sin(angles).astype(np.float32)[:, None, None]
                    patch[rows] = cos * first + sin * second
                else:
                    patch[rows] = first
        mapped = wy @ patch @ wx.transpose(0, 2, 1)
        return np.asarray(mapped.reshape(layers, count, grid.ny, grid.nx).sum(axis=0), np.float64)

    def _sinusoid_phase(
        self,
        times: FloatArray,
        xs: FloatArray,
        ys: FloatArray,
        window_s: float | None = None,
    ) -> FloatArray:
        """The low-frequency part, expanded to third order about the pupil centre.

        With `window_s`, the result is the mean over a window of that length centred on each time.
        The mean of a sinusoid over a window is its value at the centre times a sinc factor.
        """
        theta = self._lf_phase0[None, :] - np.outer(times, self._lf_rate)
        amplitude = self._lf_amplitude
        if window_s is not None:
            amplitude = amplitude * np.sinc(self._lf_rate * window_s / (2.0 * math.pi))
        c = amplitude * np.cos(theta)
        s = amplitude * np.sin(theta)
        a0, hxx, hxy, hyy = (c @ self._lf_basis_c.T).T
        gx, gy, txxx, txxy, txyy, tyyy = (s @ self._lf_basis_s.T).T
        # Expand `a cos(theta + k.X)` to third order: with c = a cos(theta) and s = a sin(theta),
        # the phase is c - s (k.X) - c (k.X)^2 / 2 + s (k.X)^3 / 6. Collect powers of x.
        y = ys[None, :]
        c0 = a0[:, None] - gy[:, None] * y - 0.5 * hyy[:, None] * y**2 + tyyy[:, None] / 6 * y**3
        c1 = -gx[:, None] - hxy[:, None] * y + 0.5 * txyy[:, None] * y**2
        c2 = -0.5 * hxx[:, None] + 0.5 * txxy[:, None] * y
        c3 = txxx / 6.0
        return np.asarray(
            c0[:, :, None]
            + c1[:, :, None] * xs[None, None, :]
            + c2[:, :, None] * (xs * xs)[None, None, :]
            + c3[:, None, None] * (xs**3)[None, None, :],
            dtype=np.float64,
        )

    def tilt_rad(self, t_s: float, pupil: PupilGrid) -> tuple[float, float]:
        """The instantaneous G-tilt at time `t_s`, in radians of image motion along `x` and `y`."""
        tilt = self.tilt_series_rad(np.asarray([t_s], dtype=np.float64), pupil)
        return float(tilt[0, 0]), float(tilt[0, 1])

    def tilt_series_rad(self, times_s: FloatArray, pupil: PupilGrid) -> FloatArray:
        """The instantaneous G-tilt at several times, with shape `(times, 2)` for `x` and `y`."""
        times = np.asarray(times_s, dtype=np.float64)
        out = np.empty((len(times), 2), dtype=np.float64)
        scale = REFERENCE_WAVELENGTH_M / _TWO_PI
        chunk = 64
        for start in range(0, len(times), chunk):
            phases = self.phase_series(times[start : start + chunk], pupil.spec)
            out[start : start + chunk, 0] = scale * np.einsum("mij,ij->m", phases, pupil.tilt_x)
            out[start : start + chunk, 1] = scale * np.einsum("mij,ij->m", phases, pupil.tilt_y)
        return out

    def exposure_tilt_rad(
        self, t_start_s: float, exposure_s: float, pupil: PupilGrid, n_sub: int
    ) -> tuple[float, float]:
        """The G-tilt averaged over an exposure that starts at `t_start_s`.

        The model samples `n_sub` instants spread evenly over the exposure. The centroid of a
        frame with a constant flux follows this value.
        """
        times = t_start_s + (np.arange(n_sub) + 0.5) * exposure_s / n_sub
        tilt = self.tilt_series_rad(times, pupil).mean(axis=0)
        return float(tilt[0]), float(tilt[1])

    def long_exposure_tilt_rad(
        self, t_start_s: float, exposure_s: float, pupil: PupilGrid
    ) -> tuple[float, float]:
        """The G-tilt averaged over a long exposure, in radians along `x` and `y`.

        The mean of the low-frequency sinusoids over the exposure is exact. The screen part
        averages out over many crossings of the screen, and the model drops it: for an exposure
        of a few seconds, what remains of it is a few milliarcseconds. Use this for exposures
        much longer than a screen crossing (0.1 s or more), where sampling every instant would
        build a new screen at each one.
        """
        xs = pupil.spec.x0 + pupil.spec.dx * np.arange(pupil.spec.nx, dtype=np.float64)
        ys = pupil.spec.y0 + pupil.spec.dx * np.arange(pupil.spec.ny, dtype=np.float64)
        mid = np.asarray([t_start_s + 0.5 * exposure_s], dtype=np.float64)
        phase = self._sinusoid_phase(mid, xs, ys, window_s=exposure_s)[0]
        phase *= self._gain(float(mid[0]))
        return pupil.tilt_rad(phase)

    def suggest_substeps(self, exposure_s: float, minimum: int = 4, maximum: int = 32) -> int:
        """How many instants to sample in an exposure.

        The pupil moves by at most `v T` during an exposure. This returns enough instants that
        the spacing stays under a third of the aperture, within the limits.
        """
        speed = max(layer.wind_speed_m_s for layer in self._config.layers)
        steps = math.ceil(3.0 * speed * exposure_s / self._aperture_m)
        return max(minimum, min(maximum, steps))

    def expected_tilt_variance_rad2(self) -> float:
        """The one-axis variance of the instantaneous G-tilt that this model produces, in rad^2.

        The value comes from the spectral cells that the model actually uses, not from a random
        draw. It differs from `g_tilt_variance_rad2` only by the quadrature error of the cells,
        so it checks the construction without any sampling noise.
        """
        diameter = self._aperture_m
        screen_freq = np.fft.fftfreq(self._n, d=self._spacing_m)
        fy, fx = np.meshgrid(screen_freq, screen_freq, indexing="ij")
        amp_sq = self._amplitudes.astype(np.float64) ** 2
        total = float(np.sum(amp_sq * fx * fx * _aperture_filter(np.hypot(fx, fy), diameter) ** 2))
        plan = self._plan
        f_sq = np.einsum("ij,ij->i", plan.centers, plan.centers)
        sinusoid = np.sum(
            plan.moments
            / f_sq
            * plan.centers[:, 0] ** 2
            * _aperture_filter(np.sqrt(f_sq), diameter) ** 2
        )
        r0_factor = math.pow(self._r0_observed, -5 / 3)
        return REFERENCE_WAVELENGTH_M**2 * r0_factor * (total + float(sinusoid))


def _aperture_filter(frequency_per_m: FloatArray, aperture_m: float) -> FloatArray:
    """The Fourier transform of a circular aperture's area weight, `2 J1(u) / u`."""
    u = math.pi * frequency_per_m * aperture_m
    out = np.ones_like(u)
    nonzero = u > 1e-9
    out[nonzero] = 2.0 * sp.j1(u[nonzero]) / u[nonzero]
    return out
