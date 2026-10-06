"""The detection estimate: the signal-to-noise ratio of Polaris in one fast frame in a bright sky.

The search mode of the fast stream (`docs/architecture.md`, "Scheduler") counts a burst as a
detection when the median signal-to-noise ratio (SNR) of Polaris in its frames reaches a threshold,
and the fast path counts a star as missing below a lower one. Both take the SNR of a matched filter
(`seeingmon.fastpath.matched`). This module predicts that SNR against the Sun's elevation from the
simulator's own photon budget and sky model, so the estimate and a simulated day agree:

- **The star.** Polaris gives `mag0_rate * 10^(-0.4 m) * t` electrons in an exposure `t`. The
  simulator applies no extinction, so the magnitude-0 rate of the profile stands for everything.
- **The exposure.** The longest exposure up to `max_exposure_us` that keeps the sky and the dark
  at or below `target_background_fraction` of the full well, and never shorter than
  `min_exposure_us`.
- **The image.** The simulator's Gaussian-mixture image of Polaris at the `r0` of the line of
  sight, at 8 x 8 positions within a pixel. `blur_fwhm_arcsec` widens it by a Gaussian, for a
  real image that defocus or the color of the optics makes wider than the simulator's.
- **The best weighting.** The highest SNR that any weighting of the pixels reaches, with the
  weights `P / (v + F P)` for the pixel shares `P` of the image: `sqrt(sum((F P)^2 / (v + F P)))`,
  where `F` is the star's flux and `v` the variance of one pixel of sky: the sky and dark
  photons, the read noise, and the rounding of the ADC (`e_per_adu^2 / 12`). It is the physical
  limit, a property of the star, the sky, and the image.
- **The matched filters.** The filters of the fast path: Gaussians of `filter_fwhms_px`, each
  evaluated on the grid of positions a quarter of a pixel apart that the kernel uses, at the grid
  point where the noise-free image gives the highest `S / sqrt(sum(w^2))`. For a star of flux
  `F`, the weighted sum is `S = F sum(w P)`, and the SNR is the kernel's
  `S / sqrt(v sum(w^2) + F sum(w^2 P))`, averaged over the positions within a pixel. A search
  frame keeps the best filter, so the estimate takes the highest of their SNRs.
- **The centroid aperture.** The fast path's soft-edged disk, whose centroid gives the seeing: a
  pixel at distance `r` from the center has the weight `clip(R + 0.5 - r, 0, 1)`, as in
  `seeingmon.fastpath.kernel`. Its SNR is `F f / sqrt(F f + A v)` for the share `f` of the star
  inside it and its area `A`. It describes the noise of the centroids, not the detection.
- **The scintillation.** The flux of a frame scatters by the simulator's log-normal factor with a
  mean of 1. The SNR grows with the flux, so the median frame has the SNR of the median flux,
  which is `exp(-sigma^2 / 2)` times the mean. The search takes the median of a burst, so the
  median frame decides.

The SNRs leave out the noise of the background level (the trimmed mean of the ROI border), as the
kernel does.

`DetectionModel.row` gives one Sun elevation, `crossing_deg` finds the elevation where the matched
SNR of the median frame falls to a threshold, and `crossing_sky_mag_arcsec2` finds the sky where
it does. The section "Polaris in a bright sky" of `docs/research-notes.md` holds the tables, and a
test reproduces them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from seeingmon.drivers.sim.optics import MixturePsf, PsfConfig
from seeingmon.drivers.sim.options import SimOptions
from seeingmon.drivers.sim.params import ARCSEC_PER_RAD, SimParams
from seeingmon.drivers.sim.sky import (
    SYNTHETIC_SITE,
    ScintillationConfig,
    airmass,
    flux_factor,
    sky_brightness_mag_arcsec2,
)
from seeingmon.drivers.sim.stars import POLARIS_MAG
from seeingmon.drivers.sim.turbulence import TurbulenceConfig, r0_at_zenith_angle
from seeingmon.fastpath.config import FastPathConfig
from seeingmon.fastpath.matched import matched_filter
from seeingmon.profile.derived import AIRY_FWHM_FACTOR

FloatArray = npt.NDArray[np.float64]

DETECTION_SNR = 10.0
"""The SNR of a detection, the design value of `scheduler.search.detect_snr`."""

_PHASES = 8  # positions of the star within a pixel, along each axis
_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


@dataclass(frozen=True, slots=True)
class Aperture:
    """A soft-edged aperture on the image of Polaris.

    `radius_px` is the radius at half weight, `area_px` the sum of the weights, and
    `flux_fraction` the share of the star's flux that the weighted sum holds.
    """

    radius_px: float
    area_px: float
    flux_fraction: float


@dataclass(frozen=True, slots=True)
class MatchedResponse:
    """The matched filter on the image of Polaris, at each position of the star within a pixel.

    For position `i`, `signal[i]` is `sum(w P)`, the weighted sum per electron of the star,
    `sum_w2[i]` the sum of the squares of the weights of the grid point that the kernel picks,
    and `sum_w2p[i]` the sum of their squares times the image, `sum(w^2 P)`.
    """

    signal: FloatArray
    sum_w2: FloatArray
    sum_w2p: FloatArray

    def snr(self, star_e: float, pixel_var_e2: float) -> float:
        """The kernel's SNR of a star of `star_e` electrons, averaged over the positions."""
        signal = star_e * self.signal
        variance = pixel_var_e2 * self.sum_w2 + star_e * self.sum_w2p
        return float(np.mean(signal / np.sqrt(variance)))

    @property
    def effective_area_px2(self) -> float:
        """The area of the aperture that would give the same SNR on a sky that dominates the
        noise, if it held the whole star: `sum(w^2) / sum(w P)^2`, averaged over the positions."""
        return float(1.0 / np.mean(self.signal / np.sqrt(self.sum_w2)) ** 2)


@dataclass(frozen=True, slots=True)
class DetectionRow:
    """The estimate at one Sun elevation.

    `exposure_us` is the exposure that the background allows, `background_e` the sky and dark
    electrons in one pixel, and `background_fraction` that over the full well. `star_e` is the
    mean flux of Polaris in the frame, all of it. The SNRs are those of the median frame:
    `snr_best` of the best weighting of the pixels (the physical limit), `snr_matched` of the best
    of the fast path's matched filters (which decides the detection), `snr_centroid` of the fast
    path's centroid aperture, and `snr_filters` of each matched filter, in the order of
    `filter_fwhms_px`.
    """

    sun_elevation_deg: float
    sky_mag_arcsec2: float
    exposure_us: float
    background_e: float
    background_fraction: float
    star_e: float
    snr_best: float
    snr_matched: float
    snr_centroid: float
    snr_filters: tuple[float, ...]


def aperture_weights(radius_px: float, dx: float, dy: float, half: int) -> FloatArray:
    """The weights of an aperture whose center is `(dx, dy)` from the center pixel of a box.

    The box has `2 * half + 1` pixels on a side. The weights are those of the fast path kernel.
    """
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    distance = np.hypot(offsets[None, :] - dx, offsets[:, None] - dy)
    return np.asarray(np.clip(radius_px + 0.5 - distance, 0.0, 1.0), dtype=np.float64)


def snr(flux_e: float, area_px: float, pixel_var_e2: float) -> float:
    """The SNR of an aperture sum: the flux over the root of its photon and pixel noise."""
    return flux_e / math.sqrt(flux_e + area_px * pixel_var_e2)


@dataclass(frozen=True)
class DetectionModel:
    """The photon budget of Polaris in a fast frame, and the sky it sits in.

    `params` is a readout mode of the simulator, normally from `SimParams.from_profile`.
    `aperture_diameter_px` is the fast path's centroid aperture, and `filter_fwhms_px` the FWHMs
    of its matched filters (`None` takes the default `[fastpath] matched_fwhm_airy_widths` times
    the Airy FWHM of the mode). `blur_fwhm_arcsec` widens the simulator's image of the star by a
    Gaussian of that FWHM. `dark_sky_mag_arcsec2` feeds the simulator's sky model, and
    `temperature_c` its dark current. `r0_zenith_m` is the Fried parameter at 500 nm at the
    zenith, and `zenith_angle_deg` the zenith angle of the pole, which set the image of the star
    and the scintillation. `centroid` is the centroid aperture on that image, `matched` the
    response of each matched filter, and `images` the image at each position within a pixel.
    """

    params: SimParams
    gain: int = 0
    max_exposure_us: float = 2000.0
    min_exposure_us: float = 32.0
    target_background_fraction: float = 0.3
    aperture_diameter_px: float = 16.0
    filter_fwhms_px: tuple[float, ...] | None = None
    blur_fwhm_arcsec: float = 0.0
    polaris_mag: float = POLARIS_MAG
    dark_sky_mag_arcsec2: float = 20.5
    temperature_c: float = 19.0
    r0_zenith_m: float = 0.10
    zenith_angle_deg: float = 35.0
    scintillation: ScintillationConfig = field(default_factory=ScintillationConfig)
    centroid: Aperture = field(init=False, repr=False)
    matched: tuple[MatchedResponse, ...] = field(init=False, repr=False)
    images: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 < self.min_exposure_us <= self.max_exposure_us:
            raise ValueError("the exposures must be positive, the shortest first")
        if not 0 < self.target_background_fraction <= 1:
            raise ValueError("target_background_fraction must be between 0 and 1")
        if not self.blur_fwhm_arcsec >= 0:
            raise ValueError("blur_fwhm_arcsec must not be negative")
        if self.filter_fwhms_px is not None and not self.filter_fwhms_px:
            raise ValueError("filter_fwhms_px needs at least one filter, or None")
        stamps, offsets = self._stamps()
        centroid = _aperture(self.aperture_diameter_px / 2.0, stamps, offsets)
        matched = tuple(_matched_response(fwhm, stamps) for fwhm in self.filter_widths_px)
        object.__setattr__(self, "centroid", centroid)
        object.__setattr__(self, "matched", matched)
        object.__setattr__(self, "images", stamps)

    @property
    def airy_fwhm_px(self) -> float:
        """The FWHM of the Airy pattern of the mode, in pixels."""
        params = self.params
        return AIRY_FWHM_FACTOR * params.wavelength_m / params.aperture_m / params.pixel_rad

    @property
    def filter_widths_px(self) -> tuple[float, ...]:
        """The FWHMs of the matched filters, in pixels."""
        if self.filter_fwhms_px is not None:
            return self.filter_fwhms_px
        airy = self.airy_fwhm_px
        return tuple(widths * airy for widths in FastPathConfig().matched_fwhm_airy_widths)

    @property
    def image_area_px2(self) -> float:
        """`1 / sum(P^2)` of the image, averaged over the positions within a pixel: the area of an
        aperture with the same noise in a sky that dominates it, if it held the whole star."""
        return float(np.mean(1.0 / np.sum(self.images**2, axis=(1, 2))))

    @classmethod
    def for_simulator(cls, params: SimParams, **values: Any) -> DetectionModel:
        """A model with the defaults of the simulator: its dark sky, sensor, `r0`, and site.

        `values` set any other field, such as the exposures and the aperture.
        """
        options = SimOptions()
        fields: dict[str, Any] = {
            "dark_sky_mag_arcsec2": options.sky_mag_arcsec2,
            "temperature_c": options.ambient_c + options.sensor_rise_c,
            "r0_zenith_m": TurbulenceConfig().r0_m,
            "zenith_angle_deg": 90.0 - SYNTHETIC_SITE.latitude_deg,
            **values,
        }
        return cls(params, **fields)

    def _stamps(self) -> tuple[FloatArray, FloatArray]:
        """Images of Polaris at `_PHASES` squared positions within a pixel, and the positions."""
        psf = MixturePsf(self.params, PsfConfig(mode="gaussian"))
        r0 = r0_at_zenith_angle(self.r0_zenith_m, self.zenith_angle_deg)
        blur_px = self.blur_fwhm_arcsec / (self.params.pixel_rad * ARCSEC_PER_RAD)
        weights, sigmas = psf.components(r0, blur_px / _FWHM_PER_SIGMA)
        phase = (np.arange(_PHASES) + 0.5) / _PHASES - 0.5
        dx, dy = (a.ravel() for a in np.meshgrid(phase, phase))
        stamps = np.asarray(psf.stamps(dx, dy, weights, sigmas), dtype=np.float64)
        return stamps, np.stack([dx, dy], axis=1)

    def exposure_s(self, background_rate_e_per_s: float) -> float:
        """The exposure that puts the background at the target, within the exposure limits."""
        full_well = self.params.sensor_at(self.gain).full_well_e
        target = self.target_background_fraction * full_well / background_rate_e_per_s
        return max(self.min_exposure_us * 1e-6, min(self.max_exposure_us * 1e-6, target))

    def at_sky(self, sky_mag_arcsec2: float, sun_elevation_deg: float = math.nan) -> DetectionRow:
        """The estimate for a sky of a given surface brightness."""
        params = self.params
        sensor = params.sensor_at(self.gain)
        rate = params.sky_rate_e_per_s_px(sky_mag_arcsec2) + params.dark_rate_e_per_s(
            self.temperature_c
        )
        exposure_s = self.exposure_s(rate)
        background = rate * exposure_s
        star = params.mag0_rate_e_per_s() * math.pow(10.0, -0.4 * self.polaris_mag) * exposure_s
        pixel_var = background + sensor.read_noise_e**2 + sensor.e_per_adu**2 / 12.0
        rms = self.scintillation.index(exposure_s, airmass(self.zenith_angle_deg))
        median = float(flux_factor(rms, 0.0)) * star
        centroid = self.centroid
        image = median * self.images
        best = np.sqrt(np.sum(image**2 / (pixel_var + image), axis=(1, 2)))
        filters = tuple(response.snr(median, pixel_var) for response in self.matched)
        return DetectionRow(
            sun_elevation_deg=sun_elevation_deg,
            sky_mag_arcsec2=sky_mag_arcsec2,
            exposure_us=exposure_s * 1e6,
            background_e=background,
            background_fraction=background / sensor.full_well_e,
            star_e=star,
            snr_best=float(np.mean(best)),
            snr_matched=max(filters),
            snr_centroid=snr(median * centroid.flux_fraction, centroid.area_px, pixel_var),
            snr_filters=filters,
        )

    def row(self, sun_elevation_deg: float) -> DetectionRow:
        """The estimate at a Sun elevation, in the simulator's sky near the pole."""
        sky = float(sky_brightness_mag_arcsec2(self.dark_sky_mag_arcsec2, sun_elevation_deg))
        return self.at_sky(sky, sun_elevation_deg)

    def crossing_deg(
        self,
        threshold: float = DETECTION_SNR,
        *,
        low_deg: float = -18.0,
        high_deg: float = 90.0,
        tolerance_deg: float = 1e-4,
    ) -> float | None:
        """The Sun elevation at which the matched SNR of the median frame falls to `threshold`.

        The SNR falls as the Sun rises, so a bisection finds it. The result is `None` when the
        SNR stays at or above the threshold up to `high_deg`, which means that Polaris stays
        detectable in full daylight.
        """
        if self.row(high_deg).snr_matched >= threshold:
            return None
        if self.row(low_deg).snr_matched < threshold:
            return low_deg
        low, high = low_deg, high_deg
        while high - low > tolerance_deg:
            middle = 0.5 * (low + high)
            if self.row(middle).snr_matched >= threshold:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    def crossing_sky_mag_arcsec2(
        self,
        threshold: float = DETECTION_SNR,
        *,
        brightest: float = -5.0,
        darkest: float = 22.0,
        tolerance: float = 1e-4,
    ) -> float | None:
        """The sky, in mag/arcsec², at which the matched SNR of the median frame falls to
        `threshold`.

        The SNR falls as the sky brightens, which is as its magnitude falls, so a bisection finds
        it. The result is `None` when the SNR stays at or above the threshold even in the
        `brightest` sky, and `darkest` when it is below the threshold already there.
        """
        if self.at_sky(brightest).snr_matched >= threshold:
            return None
        if self.at_sky(darkest).snr_matched < threshold:
            return darkest
        bright, dark = brightest, darkest
        while dark - bright > tolerance:
            middle = 0.5 * (bright + dark)
            if self.at_sky(middle).snr_matched >= threshold:
                dark = middle
            else:
                bright = middle
        return 0.5 * (bright + dark)


def _aperture(radius_px: float, stamps: FloatArray, offsets: FloatArray) -> Aperture:
    """The aperture of a radius, centered on the star, averaged over the star's positions."""
    size = stamps.shape[-1]
    center = size // 2
    half = min(center, math.ceil(radius_px) + 1)
    box = stamps[:, center - half : center + half + 1, center - half : center + half + 1]
    fractions = []
    areas = []
    for stamp, (dx, dy) in zip(box, offsets, strict=True):
        weights = aperture_weights(radius_px, float(dx), float(dy), half)
        fractions.append(float(np.sum(weights * stamp)))
        areas.append(float(np.sum(weights)))
    return Aperture(radius_px, float(np.mean(areas)), float(np.mean(fractions)))


def _matched_response(fwhm_px: float, stamps: FloatArray) -> MatchedResponse:
    """The matched filter of the kernel on each image, at the grid point that it would pick.

    Each image sits on the center pixel of its stamp, and the kernel's grid spans the 3 x 3 pixels
    around it. Without noise, the grid point with the highest `S / sqrt(sum(w^2))` is the one that
    the kernel picks for any flux and sky.
    """
    matched = matched_filter(fwhm_px)
    center = stamps.shape[-1] // 2
    reach = matched.half + 1
    box = stamps[:, center - reach : center + reach + 1, center - reach : center + reach + 1]
    flat = box.reshape(len(box), -1)
    scores = flat @ matched.near_score_t  # (images, grid points)
    best = np.argmax(scores, axis=1)
    sum_w2 = matched.near_sum_w2[best]
    signal = scores[np.arange(len(box)), best] * np.sqrt(sum_w2)
    weights_sq = matched.near_score_t[:, best].T ** 2 * sum_w2[:, None]  # (images, pixels)
    return MatchedResponse(signal, sum_w2, np.sum(weights_sq * flat, axis=1))
