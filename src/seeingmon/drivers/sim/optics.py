"""The point spread function of a star: wave optics, or a cheap Gaussian mixture.

Both renderers return a *stamp*: the image of one star on a square patch of pixels, scaled to
a total of 1. The star's centre sits on the centre pixel `(fov // 2, fov // 2)` plus the offset
that you pass in. The caller multiplies the stamp by the star's flux and adds it to the frame.

**Wave optics** (`WavePsf`) is the reference. For each of at least four instants in the
exposure, it samples the pupil (a circle of at least 32 points across, with soft edge pixels),
multiplies it by the turbulent phase, and takes the FFT. The FFT grid has four samples per
pixel, and a Laplacian correction brings the box integration of the pixel to 0.5% of the peak.
The pupil spacing follows from the wavelength, so every sample of the focal plane is a quarter
of a pixel wide. The exposure image is the mean of the instants. An optional bandwidth adds two
more wavelengths.

**The Gaussian mixture** (`MixturePsf`) is the cheap mode. It draws a diffraction-limited core
plus a seeing halo as a sum of circular Gaussians, and it shifts the whole thing by the same tilt
that the wave optics would produce. Box integration of a Gaussian has a closed form, so a
stamp costs about a tenth of a millisecond. The Gaussians approximate the Airy pattern (the fit
holds the pixel values to 2% of the peak and the flux to 0.2%) and have no diffraction rings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import numpy.typing as npt

from seeingmon.drivers.sim import _scipy as sp
from seeingmon.drivers.sim.params import ARCSEC_PER_RAD, SimParams
from seeingmon.drivers.sim.turbulence import (
    REFERENCE_WAVELENGTH_M,
    PupilGrid,
    TurbulenceModel,
    r0_at_wavelength,
)

FloatArray = npt.NDArray[np.float64]
SingleArray = npt.NDArray[np.float32]
ComplexSingle = npt.NDArray[np.complex64]

_TWO_PI = 2.0 * math.pi
MIN_PUPIL_POINTS = 32


@dataclass(frozen=True, slots=True)
class PsfConfig:
    """How to render a star.

    `mode` is `wave` or `gaussian`. `oversample` is the number of FFT samples per pixel along
    each axis. `fov_arcsec` is the width of the stamp (the wave optics widens it if the pupil
    would have fewer than 32 points across). `bandwidth_fraction` is the full width of the
    band over the central wavelength: 0 means one wavelength, and a positive value renders
    three wavelengths weighted 1, 2, and 1. An exposure longer than `long_exposure_s` uses the
    Gaussian mixture, because the pupil crosses so much turbulence that wave optics adds nothing.
    """

    mode: Literal["wave", "gaussian"] = "wave"
    oversample: int = 4
    fov_arcsec: float = 122.0
    bandwidth_fraction: float = 0.0
    long_exposure_s: float = 0.1

    def __post_init__(self) -> None:
        if self.mode not in ("wave", "gaussian"):
            raise ValueError(f"unknown PSF mode {self.mode!r}: use 'wave' or 'gaussian'")
        if self.oversample < 2:
            raise ValueError("oversample must be at least 2")
        if self.fov_arcsec <= 0 or not 0.0 <= self.bandwidth_fraction < 1.0:
            raise ValueError("fov_arcsec must be positive and bandwidth_fraction below 1")
        if self.long_exposure_s <= 0:
            raise ValueError("long_exposure_s must be positive")


def wavelengths_of(params: SimParams, config: PsfConfig) -> tuple[FloatArray, FloatArray]:
    """The wavelengths to render and their weights (summing to 1)."""
    if config.bandwidth_fraction == 0.0:
        return np.asarray([params.wavelength_m]), np.asarray([1.0])
    spread = config.bandwidth_fraction * params.wavelength_m
    waves = params.wavelength_m + spread * np.asarray([-0.5, 0.0, 0.5])
    return waves, np.asarray([0.25, 0.5, 0.25])


def stamp_size_px(params: SimParams, config: PsfConfig) -> int:
    """The stamp width in pixels: even, with at least 32 pupil points at the longest wavelength."""
    waves, _ = wavelengths_of(params, config)
    wanted_rad = max(
        config.fov_arcsec / ARCSEC_PER_RAD,
        MIN_PUPIL_POINTS * 1.05 * float(waves.max()) / params.aperture_m,
    )
    size = math.ceil(wanted_rad / params.pixel_rad)
    return size + size % 2


# --- wave optics ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PsfResult:
    """A stamp and the exposure-averaged G-tilt that produced it, in radians along `x` and `y`."""

    stamp: SingleArray
    tilt_x_rad: float
    tilt_y_rad: float


class WavePsf:
    """Wave-optics star images through a `TurbulenceModel`.

    One instance serves one readout mode. It keeps scratch buffers, so do not share it between
    threads.
    """

    def __init__(self, params: SimParams, config: PsfConfig, model: TurbulenceModel) -> None:
        if config.mode != "wave":
            raise ValueError("WavePsf needs a config with mode 'wave'")
        self._params = params
        self._config = config
        self._model = model
        self._oversample = config.oversample
        self._fov_px = stamp_size_px(params, config)
        self._n_fft = self._fov_px * config.oversample
        sample_rad = params.pixel_rad / config.oversample
        self._waves, self._weights = wavelengths_of(params, config)
        self._grids: list[PupilGrid] = []
        self._ramps_x: list[FloatArray] = []
        self._ramps_y: list[FloatArray] = []
        self._checker: list[ComplexSingle] = []
        for wavelength in self._waves:
            spacing = float(wavelength) / (self._n_fft * sample_rad)
            grid = PupilGrid.circular(params.aperture_m, spacing)
            xs, ys = grid.coordinates()
            # A shift of `s` pixels is the linear phase 2 pi s pixel X / lambda.
            scale = _TWO_PI * params.pixel_rad / float(wavelength)
            self._grids.append(grid)
            self._ramps_x.append(scale * xs)
            self._ramps_y.append(scale * ys)
            n = grid.spec.nx
            sign = (-1.0) ** (np.add.outer(np.arange(n), np.arange(n)))
            self._checker.append((sign * grid.coverage).astype(np.complex64))
        self._centre = (config.oversample - 1) / (2.0 * config.oversample)

    @property
    def fov_px(self) -> int:
        """The width of the stamp in pixels."""
        return self._fov_px

    @property
    def pupil_points(self) -> int:
        """The number of pupil samples across the aperture at the central wavelength."""
        centre = len(self._grids) // 2
        return round(self._params.aperture_m / self._grids[centre].spec.dx)

    def render(
        self,
        t_start_s: float,
        exposure_s: float,
        offset_px: tuple[float, float] = (0.0, 0.0),
        n_sub: int | None = None,
    ) -> PsfResult:
        """The image of a star for an exposure that starts at `t_start_s`.

        `offset_px` is the star's offset from the centre pixel, from -0.5 to 0.5 along each
        axis. The turbulent tilt shifts the image further.
        """
        steps = n_sub if n_sub is not None else self._model.suggest_substeps(exposure_s)
        times = t_start_s + (np.arange(steps) + 0.5) * exposure_s / steps
        centre_index = len(self._grids) // 2
        accumulated = np.zeros((self._n_fft, self._n_fft), dtype=np.float32)
        tilt = np.zeros(2, dtype=np.float64)
        for index, grid in enumerate(self._grids):
            phases = self._model.phase_series(times, grid.spec)
            if index == centre_index:
                scale = REFERENCE_WAVELENGTH_M / _TWO_PI
                tilt[0] = scale * float(np.einsum("mij,ij->", phases, grid.tilt_x)) / steps
                tilt[1] = scale * float(np.einsum("mij,ij->", phases, grid.tilt_y)) / steps
            wavelength = float(self._waves[index])
            ramp = self._ramps_x[index][None, :] * (offset_px[0] + self._centre) + self._ramps_y[
                index
            ][:, None] * (offset_px[1] + self._centre)
            chromatic = np.float32(REFERENCE_WAVELENGTH_M / wavelength)
            angles = (phases * chromatic + ramp[None, :, :]).astype(np.float32)
            fields = np.exp(1j * angles) * self._checker[index][None, :, :]
            intensity = np.zeros((self._n_fft, self._n_fft), dtype=np.float32)
            for step in range(steps):
                transformed = sp.padded_fft2(fields[step], self._n_fft)
                intensity += transformed.real**2 + transformed.imag**2
            # Every wavelength carries its own share of the flux.
            accumulated += np.float32(self._weights[index] / float(intensity.sum())) * intensity
        size = self._fov_px
        over = self._oversample
        binned = accumulated.reshape(size, over, size, over).sum(axis=(1, 3))
        binned /= binned.sum()
        # The sum of `over` samples per pixel is a midpoint rule, which overestimates a concave
        # peak. Add its leading error term, a multiple of the Laplacian. The Laplacian sums to
        # zero, so the flux stays 1.
        laplacian = (
            np.roll(binned, 1, 0) + np.roll(binned, -1, 0) + np.roll(binned, 1, 1)
            + np.roll(binned, -1, 1) - 4.0 * binned
        )  # fmt: skip
        binned += laplacian / np.float32(24.0 * over * over)
        return PsfResult(np.asarray(binned, dtype=np.float32), float(tilt[0]), float(tilt[1]))


# --- the Gaussian mixture ------------------------------------------------------------------

# Sigmas of the Gaussians, in units of lambda / D. A flux-weighted non-negative least-squares fit
# to the Airy pattern chooses their weights.
_MIXTURE_SIGMAS = (0.25, 0.4, 0.65, 1.0, 1.7, 3.0, 6.0, 15.0)


def _airy_unit(r: FloatArray) -> FloatArray:
    """The Airy intensity for unit flux, with `r` in units of lambda / D."""
    x = np.pi * r
    safe = np.where(x > 1e-9, x, 1.0)
    value = np.where(x > 1e-9, (2.0 * sp.j1(safe) / safe) ** 2, 1.0)
    return np.asarray((math.pi / 4) * value, dtype=np.float64)


@lru_cache(maxsize=1)
def _airy_mixture() -> tuple[FloatArray, FloatArray]:
    """The sigmas (in lambda / D) and weights of the Gaussians that approximate an Airy pattern."""
    r = np.concatenate([np.linspace(0.0, 6.0, 600, endpoint=False), np.linspace(6.0, 60.0, 900)])
    dr = np.gradient(r)
    sigmas = np.asarray(_MIXTURE_SIGMAS, dtype=np.float64)
    design = np.exp(-(r[:, None] ** 2) / (2 * sigmas[None, :] ** 2)) / (
        2 * math.pi * sigmas[None, :] ** 2
    )
    weight = np.sqrt(2 * math.pi * r * dr + 0.02 * dr)
    matrix = np.vstack([design * weight[:, None], 100.0 * np.ones((1, len(sigmas)))])
    target = np.concatenate([_airy_unit(r) * weight, [100.0]])
    weights = sp.nnls(matrix, target)
    return sigmas, weights / weights.sum()


def strehl_ratio(aperture_m: float, r0_500nm_m: float, wavelength_m: float) -> float:
    """The Strehl ratio of the image without tilt: `exp(-0.134 (D / r0)^(5/3))`.

    The exponent is the phase variance that remains after the piston and the tilt of
    Kolmogorov turbulence (Noll 1976), at the observing wavelength.
    """
    r0 = r0_at_wavelength(r0_500nm_m, wavelength_m)
    return math.exp(-0.134 * math.pow(aperture_m / r0, 5 / 3))


class MixturePsf:
    """Star images as a sum of circular Gaussians that approximate diffraction plus seeing.

    The image of a star at wavelength `lambda` is `S` times the Airy pattern plus `1 - S` times
    a Gaussian halo, where `S` is the Strehl ratio of the image without tilt. The halo has a
    sigma of `0.4 lambda / r0`. `components` returns the Gaussians for the current `r0` as a
    weight array and a sigma array in pixels, and `stamps` renders them.
    """

    def __init__(self, params: SimParams, config: PsfConfig) -> None:
        self._params = params
        self._config = config
        self._fov_px = stamp_size_px(params, config)
        self._waves, self._weights = wavelengths_of(params, config)

    @property
    def fov_px(self) -> int:
        return self._fov_px

    def components(
        self, r0_500nm_m: float, extra_sigma_px: float = 0.0
    ) -> tuple[FloatArray, FloatArray]:
        """The weights and sigmas (in pixels) of the Gaussians for a given `r0` at 500 nm.

        `extra_sigma_px` adds a blur in quadrature to every Gaussian, for example the wander of
        the image during a long exposure.
        """
        params = self._params
        airy_sigma, airy_weight = _airy_mixture()
        all_weights: list[FloatArray] = []
        all_sigmas: list[FloatArray] = []
        for wavelength, share in zip(self._waves, self._weights, strict=True):
            wave = float(wavelength)
            lam_over_d_px = wave / params.aperture_m / params.pixel_rad
            strehl = strehl_ratio(params.aperture_m, r0_500nm_m, wave)
            r0_wave = r0_at_wavelength(r0_500nm_m, wave)
            halo_sigma_px = 0.4 * wave / r0_wave / params.pixel_rad
            all_weights.append(share * strehl * airy_weight)
            all_sigmas.append(airy_sigma * lam_over_d_px)
            all_weights.append(np.asarray([share * (1.0 - strehl)]))
            all_sigmas.append(np.asarray([halo_sigma_px]))
        weights = np.concatenate(all_weights)
        sigmas = np.sqrt(np.concatenate(all_sigmas) ** 2 + extra_sigma_px**2)
        keep = weights > 1e-9
        return weights[keep], sigmas[keep]

    def stamps(
        self,
        offsets_x_px: FloatArray,
        offsets_y_px: FloatArray,
        weights: FloatArray,
        sigmas_px: FloatArray,
        size_px: int | None = None,
    ) -> SingleArray:
        """Render stamps at offsets from the centre pixel, with shape `(stars, size, size)`.

        An offset may be any size, but a star more than a few pixels from the centre loses
        flux off the edge of the stamp. `size_px` picks a smaller stamp for faint stars.
        """
        size = self._fov_px if size_px is None else size_px
        edges = np.arange(size + 1, dtype=np.float64) - size // 2 - 0.5
        scale = 1.0 / (math.sqrt(2.0) * sigmas_px)  # (components,)
        cdf_x = sp.erf((edges[None, None, :] - offsets_x_px[:, None, None]) * scale[None, :, None])
        cdf_y = sp.erf((edges[None, None, :] - offsets_y_px[:, None, None]) * scale[None, :, None])
        mass_x = 0.5 * np.diff(cdf_x, axis=2)  # (stars, components, size)
        mass_y = 0.5 * np.diff(cdf_y, axis=2)
        stamps = np.einsum("c,kci,kcj->kji", weights, mass_x, mass_y, optimize=True)
        return np.asarray(stamps, dtype=np.float32)

    def stamp(
        self,
        offset_px: tuple[float, float],
        r0_500nm_m: float,
        extra_sigma_px: float = 0.0,
    ) -> SingleArray:
        """One stamp for a star at an offset from the centre pixel."""
        weights, sigmas = self.components(r0_500nm_m, extra_sigma_px)
        stamps = self.stamps(
            np.asarray([offset_px[0]]), np.asarray([offset_px[1]]), weights, sigmas
        )
        return np.asarray(stamps[0], dtype=np.float32)
