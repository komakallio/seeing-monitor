"""Compose one frame: stars through the atmosphere, the sky, the sensor, and the truth.

`FrameRenderer.render` builds the *mean* electron image of a frame and passes it to the
detector. It handles two cases:

- **Short exposures** (video, up to 0.1 s). Each bright star goes through the wave optics, or
  through the Gaussian mixture in `gaussian` mode, at the instants of its own exposure. The
  rolling shutter shifts those instants by the row of the star times the row time. Faint stars
  share the tilt of the reference star and use the mixture.
- **Long exposures** (snapshots, or video above 0.1 s). Stars trail along the sky rotation, and
  the image of a star is the mixture blurred by the wander of the tilt. The tilt averaged over the
  exposure shifts all stars.

The *reference star* of a frame is the brightest star whose centre falls inside the ROI, or the
brightest one near the ROI when none does. The truth of the frame describes that star.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S
from seeingmon.drivers.sim.detector import Detector, HotPixelConfig, HotPixelMap
from seeingmon.drivers.sim.optics import MixturePsf, PsfConfig, WavePsf
from seeingmon.drivers.sim.params import ARCSEC_PER_RAD, SimParams
from seeingmon.drivers.sim.sky import (
    Clouds,
    ScintillationConfig,
    ScintillationProcess,
    flux_factor,
)
from seeingmon.drivers.sim.stars import SkyProjector
from seeingmon.drivers.sim.truth import LONG_EXPOSURE_S, SimTruth
from seeingmon.drivers.sim.turbulence import g_tilt_variance_rad2
from seeingmon.frames import FrameData, Roi, StreamConfig, StreamKind

FloatArray = npt.NDArray[np.float64]
SingleArray = npt.NDArray[np.float32]

_WAVE_FLUX_THRESHOLD_E = 300.0  # stars fainter than this use the mixture, even in wave mode
_MAX_WAVE_STARS = 4
_MIN_FLUX_E = 0.5  # fainter stars are invisible
_WING_THRESHOLD_E = 0.3  # a stamp reaches out to where the star drops below this per pixel
_MAX_HALF_PX = 256
_SIZE_CLASSES = (6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
_MAX_CORRELATED_STARS = 64  # stars per frame with a time-correlated scintillation
_MAX_TRAIL_SAMPLES = 400
_TRAIL_STEP_PX = 0.5


@dataclass(frozen=True, slots=True)
class Reference:
    """The truth about the reference star of a frame."""

    star_index: int  # in the original field, or -1 when the frame has no star
    t_ref_utc_ns: int
    tilt_x_arcsec: float
    tilt_y_arcsec: float
    star_x_px: float
    star_y_px: float
    catalog_x_px: float
    catalog_y_px: float
    flux_e: float
    scintillation_factor: float
    transparency: float


@dataclass(frozen=True, slots=True)
class RenderedFrame:
    """A frame and what the renderer knows about it."""

    data: FrameData
    reference: Reference
    sky_mag_arcsec2: float
    sun_altitude_deg: float
    sensor_temperature_c: float


def _size_class(half: int) -> int:
    for size in _SIZE_CLASSES:
        if half <= size:
            return size
    return _SIZE_CLASSES[-1]


class FrameRenderer:
    """Renders frames for a simulated camera. One instance serves one driver."""

    def __init__(
        self,
        *,
        truth: SimTruth,
        projector: SkyProjector,
        clouds: Clouds,
        psf: PsfConfig,
        scintillation: ScintillationConfig,
        hot_pixels: HotPixelConfig,
        seed: int,
        epoch_utc_ns: int,
    ) -> None:
        self._truth = truth
        self._projector = projector
        self._clouds = clouds
        self._psf = psf
        self._scintillation = scintillation
        self._hot_config = hot_pixels
        self._seed = seed
        self._epoch_utc_ns = epoch_utc_ns
        self._model = truth.model
        self._wave: dict[str, WavePsf] = {}
        self._mixture: dict[str, MixturePsf] = {}
        self._detectors: dict[tuple[str, int], Detector] = {}
        self._hot: dict[str, HotPixelMap] = {}
        self._processes: dict[int, ScintillationProcess] = {}
        self._airmass = truth.airmass_of_pole()

    # --- helpers ---

    def _turbulence_time(self, t_utc_ns: int) -> float:
        return (t_utc_ns - self._epoch_utc_ns) / NS_PER_S

    def _wave_psf(self, params: SimParams) -> WavePsf:
        if params.mode not in self._wave:
            self._wave[params.mode] = WavePsf(params, self._psf, self._model)
        return self._wave[params.mode]

    def _mixture_psf(self, params: SimParams) -> MixturePsf:
        if params.mode not in self._mixture:
            self._mixture[params.mode] = MixturePsf(params, self._psf)
        return self._mixture[params.mode]

    def _detector(self, params: SimParams) -> Detector:
        key = (params.mode, params.adc_bits)
        if key not in self._detectors:
            self._detectors[key] = Detector(params)
        return self._detectors[key]

    def _hot_map(self, params: SimParams) -> HotPixelMap:
        if params.mode not in self._hot:
            self._hot[params.mode] = HotPixelMap(params, self._hot_config, self._seed)
        return self._hot[params.mode]

    def _correlated_gaussian(self, star: int, t_utc_ns: int) -> float:
        process = self._processes.get(star)
        if process is None:
            process = ScintillationProcess(
                self._seed * 1_000_003 + star, self._scintillation.correlation_time_s
            )
            self._processes[star] = process
        return process.value(self._turbulence_time(t_utc_ns))

    @staticmethod
    def _to_pixels(tilt_arcsec: tuple[float, float], params: SimParams) -> tuple[float, float]:
        scale = 1.0 / (ARCSEC_PER_RAD * params.pixel_rad)
        return tilt_arcsec[0] * scale, tilt_arcsec[1] * scale

    # --- the public entry point ---

    def render(
        self,
        params: SimParams,
        config: StreamConfig,
        roi: Roi,
        t_start_ns: int,
        gain: int,
        rng: np.random.Generator,
    ) -> RenderedFrame:
        """Render the frame whose first-row exposure starts at `t_start_ns`."""
        exposure_s = config.exposure_us * 1e-6
        long = config.kind is StreamKind.SNAPSHOT or exposure_s > LONG_EXPOSURE_S
        t_mid_ns = t_start_ns + round(exposure_s * 0.5 * NS_PER_S)
        truth = self._truth
        sky_mag = float(truth.sky_mag_arcsec2(t_mid_ns))
        temperature = truth.sensor_temperature_c(t_mid_ns)
        signal: SingleArray = np.zeros((roi.height, roi.width), dtype=np.float32)
        if long:
            reference = self._add_long(signal, params, roi, t_start_ns, exposure_s, rng)
        else:
            reference = self._add_short(signal, params, roi, t_start_ns, exposure_s, rng)
        background = params.sky_rate_e_per_s_px(sky_mag) + params.dark_rate_e_per_s(temperature)
        signal += np.float32(background * exposure_s)
        self._hot_map(params).add_to(signal, roi, exposure_s, temperature)
        data = self._detector(params).digitize(
            signal, gain=gain, offset=config.offset, pixel_format=config.pixel_format, rng=rng
        )
        return RenderedFrame(
            data=data,
            reference=reference,
            sky_mag_arcsec2=sky_mag,
            sun_altitude_deg=float(truth.sun_altitude_deg(t_mid_ns)),
            sensor_temperature_c=temperature,
        )

    # --- short exposures ---

    def _add_short(
        self,
        signal: SingleArray,
        params: SimParams,
        roi: Roi,
        t_start_ns: int,
        exposure_s: float,
        rng: np.random.Generator,
    ) -> Reference:
        truth = self._truth
        half_exposure_ns = round(exposure_s * 0.5 * NS_PER_S)
        t_mid_ns = t_start_ns + half_exposure_ns
        transparency = float(truth.transparency(t_mid_ns))
        mixture = self._mixture_psf(params)
        half_fov = mixture.fov_px // 2
        stars = self._projector.stars
        x_all, y_all = self._projector.project(
            t_mid_ns, params.pixel_rad, params.width, params.height
        )
        flux = params.mag0_rate_e_per_s() * np.power(10.0, -0.4 * stars.mag) * exposure_s
        flux = flux * transparency
        near = (
            (x_all > roi.x - half_fov)
            & (x_all < roi.x_end + half_fov)
            & (y_all > roi.y - half_fov)
            & (y_all < roi.y_end + half_fov)
            & (flux > _MIN_FLUX_E)
        )
        candidates = np.nonzero(near)[0]
        candidates = candidates[np.argsort(-flux[candidates], kind="stable")]
        row_ns = round(params.row_time_s * NS_PER_S)

        def start_of(star: int) -> int:
            row = min(max(round(float(y_all[star])) - roi.y, 0), roi.height - 1)
            return t_start_ns + row * row_ns

        if len(candidates) == 0:
            start = t_start_ns + (roi.height // 2) * row_ns
            tilt = truth.tilt_arcsec(start, exposure_s)
            return _no_star(start + half_exposure_ns, tilt, transparency)
        inside = (
            (x_all[candidates] >= roi.x - 0.5)
            & (x_all[candidates] < roi.x_end - 0.5)
            & (y_all[candidates] >= roi.y - 0.5)
            & (y_all[candidates] < roi.y_end - 0.5)
        )
        reference_star = (
            int(candidates[int(np.argmax(inside))]) if inside.any() else int(candidates[0])
        )
        reference_start = start_of(reference_star)
        wave_mode = self._psf.mode == "wave"
        r0 = truth.r0_observed_m(t_mid_ns)
        weights, sigmas = mixture.components(r0)
        rms = self._scintillation.index(exposure_s, self._airmass)
        wave_psf = self._wave_psf(params) if wave_mode else None
        same_grid = (
            wave_psf is not None
            and abs(wave_psf.pupil_spacing_m - truth.pupil.spec.dx) < 1e-12 * truth.pupil.spec.dx
        )

        # Pass 1: the flux of each candidate and whether it needs the wave optics.
        stars_info = []
        for position, star_value in enumerate(candidates):
            star = int(star_value)
            star_start = start_of(star)
            star_mid = star_start + half_exposure_ns
            if rms <= 0.0:
                factor = 1.0
            elif position < _MAX_CORRELATED_STARS:
                factor = float(flux_factor(rms, self._correlated_gaussian(star, star_mid)))
            else:
                factor = float(flux_factor(rms, float(rng.standard_normal())))
            electrons = float(flux[star]) * factor
            use_wave = (
                wave_mode and electrons >= _WAVE_FLUX_THRESHOLD_E and position < _MAX_WAVE_STARS
            )
            stars_info.append((star, star_start, star_mid, factor, electrons, use_wave))

        def render_wave(star: int, start: int, electrons: float) -> tuple[float, float]:
            """Render a star with wave optics and return its truth tilt in arcseconds."""
            assert wave_psf is not None
            x, y = float(x_all[star]), float(y_all[star])
            cx, cy = round(x), round(y)
            result = wave_psf.render(self._turbulence_time(start), exposure_s, (x - cx, y - cy))
            self._add_stamp(signal, roi, result.stamp, cx, cy, electrons)
            if same_grid:  # the wave optics used the truth pupil, so its tilt is the truth tilt
                return (result.tilt_x_rad * ARCSEC_PER_RAD, result.tilt_y_rad * ARCSEC_PER_RAD)
            return truth.tilt_arcsec(start, exposure_s)

        tilts: dict[int, tuple[float, float]] = {}
        reference_info = next(item for item in stars_info if item[0] == reference_star)
        if reference_info[5]:
            tilts[reference_star] = render_wave(reference_star, reference_start, reference_info[4])
        else:
            tilts[reference_star] = truth.tilt_arcsec(reference_start, exposure_s)
        reference_tilt = tilts[reference_star]

        # Pass 2: the other stars. The faint ones share the reference tilt.
        queued: list[tuple[float, float, float]] = []
        reference: Reference | None = None
        for star, star_start, star_mid, factor, electrons, use_wave in stars_info:
            if star == reference_star:
                tilt = reference_tilt
                rendered = use_wave
            elif use_wave:
                tilt = render_wave(star, star_start, electrons)
                rendered = True
            else:
                tilt = reference_tilt
                rendered = False
            tilt_x_px, tilt_y_px = self._to_pixels(tilt, params)
            x, y = float(x_all[star]), float(y_all[star])
            if not rendered:
                queued.append((x + tilt_x_px, y + tilt_y_px, electrons))
            if star == reference_star:
                reference = Reference(
                    star_index=int(self._projector.indices[star]),
                    t_ref_utc_ns=star_mid,
                    tilt_x_arcsec=tilt[0],
                    tilt_y_arcsec=tilt[1],
                    star_x_px=x + tilt_x_px,
                    star_y_px=y + tilt_y_px,
                    catalog_x_px=x,
                    catalog_y_px=y,
                    flux_e=electrons,
                    scintillation_factor=factor,
                    transparency=transparency,
                )
        self._add_mixture_stars(signal, roi, mixture, queued, weights, sigmas, half_fov)
        assert reference is not None
        return reference

    # --- long exposures ---

    def _add_long(
        self,
        signal: SingleArray,
        params: SimParams,
        roi: Roi,
        t_start_ns: int,
        exposure_s: float,
        rng: np.random.Generator,
    ) -> Reference:
        truth = self._truth
        t_mid_ns = t_start_ns + round(exposure_s * 0.5 * NS_PER_S)
        transparency = self._clouds.mean_transparency(t_start_ns, exposure_s)
        mixture = self._mixture_psf(params)
        r0 = truth.r0_observed_m(t_mid_ns)
        tilt = truth.tilt_arcsec(t_start_ns, exposure_s)
        tilt_x_px, tilt_y_px = self._to_pixels(tilt, params)
        wander_rad = math.sqrt(g_tilt_variance_rad2(params.aperture_m, r0, truth.outer_scale_m))
        weights, sigmas = mixture.components(r0, wander_rad / params.pixel_rad)
        stars = self._projector.stars
        t_end_ns = t_start_ns + round(exposure_s * NS_PER_S)
        x0, y0 = self._projector.project(t_start_ns, params.pixel_rad, params.width, params.height)
        x1, y1 = self._projector.project(t_end_ns, params.pixel_rad, params.width, params.height)
        flux = params.mag0_rate_e_per_s() * np.power(10.0, -0.4 * stars.mag) * exposure_s
        flux = flux * transparency
        reach = self._reach(flux, weights, sigmas)
        trail = np.hypot(x1 - x0, y1 - y0)
        margin = reach + trail
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        near = (
            (xm > roi.x - margin)
            & (xm < roi.x_end + margin)
            & (ym > roi.y - margin)
            & (ym < roi.y_end + margin)
            & (flux > _MIN_FLUX_E)
        )
        candidates = np.nonzero(near)[0]
        candidates = candidates[np.argsort(-flux[candidates], kind="stable")]
        if len(candidates) == 0:
            return _no_star(t_mid_ns, tilt, transparency)
        rms = self._scintillation.index(exposure_s, self._airmass)
        factors = (
            np.asarray(flux_factor(rms, rng.standard_normal(len(candidates))))
            if rms > 0
            else np.ones(len(candidates))
        )
        inside = (
            (xm[candidates] >= roi.x - 0.5)
            & (xm[candidates] < roi.x_end - 0.5)
            & (ym[candidates] >= roi.y - 0.5)
            & (ym[candidates] < roi.y_end - 0.5)
        )
        reference_position = int(np.argmax(inside)) if inside.any() else 0
        reference: Reference | None = None
        for position, star_value in enumerate(candidates):
            star = int(star_value)
            electrons = float(flux[star]) * float(factors[position])
            samples = min(
                max(1, math.ceil(float(trail[star]) / _TRAIL_STEP_PX) + 1), _MAX_TRAIL_SAMPLES
            )
            fractions = (np.arange(samples) + 0.5) / samples
            xs = x0[star] + fractions * (x1[star] - x0[star]) + tilt_x_px
            ys = y0[star] + fractions * (y1[star] - y0[star]) + tilt_y_px
            half = min(_size_class(int(reach[star])), _MAX_HALF_PX)
            self._add_trail(signal, roi, mixture, weights, sigmas, xs, ys, electrons, half)
            if position == reference_position:
                row = min(max(round(float(ym[star])) - roi.y, 0), roi.height - 1)
                reference = Reference(
                    star_index=int(self._projector.indices[star]),
                    t_ref_utc_ns=t_mid_ns + round(row * params.row_time_s * NS_PER_S),
                    tilt_x_arcsec=tilt[0],
                    tilt_y_arcsec=tilt[1],
                    star_x_px=float(xm[star]) + tilt_x_px,
                    star_y_px=float(ym[star]) + tilt_y_px,
                    catalog_x_px=float(xm[star]),
                    catalog_y_px=float(ym[star]),
                    flux_e=electrons,
                    scintillation_factor=float(factors[position]),
                    transparency=transparency,
                )
        assert reference is not None
        return reference

    @staticmethod
    def _reach(flux: FloatArray, weights: FloatArray, sigmas: FloatArray) -> FloatArray:
        """How far from its centre a star's image stays above the wing threshold, in pixels."""
        argument = (
            flux[:, None]
            * weights[None, :]
            / (2.0 * math.pi * sigmas[None, :] ** 2 * _WING_THRESHOLD_E)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            radius = np.where(
                argument > 1.0,
                sigmas[None, :] * np.sqrt(2.0 * np.log(np.maximum(argument, 1.0))),
                0.0,
            )
        return np.asarray(np.minimum(radius.max(axis=1) + 2.0, _MAX_HALF_PX), dtype=np.float64)

    # --- stamps ---

    @staticmethod
    def _add_stamp(
        signal: SingleArray, roi: Roi, stamp: SingleArray, cx: int, cy: int, electrons: float
    ) -> None:
        """Add a stamp whose centre pixel `(size // 2, size // 2)` lies at `(cx, cy)`."""
        size = stamp.shape[0]
        half = size // 2
        x_lo, y_lo = cx - half - roi.x, cy - half - roi.y
        sx0, sy0 = max(0, -x_lo), max(0, -y_lo)
        sx1 = min(size, roi.width - x_lo)
        sy1 = min(size, roi.height - y_lo)
        if sx0 >= sx1 or sy0 >= sy1:
            return
        view = signal[y_lo + sy0 : y_lo + sy1, x_lo + sx0 : x_lo + sx1]
        view += np.float32(electrons) * stamp[sy0:sy1, sx0:sx1]

    def _add_mixture_stars(
        self,
        signal: SingleArray,
        roi: Roi,
        mixture: MixturePsf,
        queued: list[tuple[float, float, float]],
        weights: FloatArray,
        sigmas: FloatArray,
        half_cap: int,
    ) -> None:
        """Add stars with the Gaussian mixture. Each tuple holds `x`, `y`, and the electrons."""
        if not queued:
            return
        xs = np.asarray([item[0] for item in queued])
        ys = np.asarray([item[1] for item in queued])
        electrons = np.asarray([item[2] for item in queued])
        reach = self._reach(electrons, weights, sigmas)
        halves = np.asarray([min(_size_class(int(value)), half_cap) for value in reach])
        cxs = np.rint(xs).astype(np.int64)
        cys = np.rint(ys).astype(np.int64)
        for half in np.unique(halves):
            selected = np.nonzero(halves == half)[0]
            stamps = mixture.stamps(
                xs[selected] - cxs[selected],
                ys[selected] - cys[selected],
                weights,
                sigmas,
                size_px=2 * int(half) + 1,
            )
            for row, index in enumerate(selected):
                self._add_stamp(
                    signal,
                    roi,
                    stamps[row],
                    int(cxs[index]),
                    int(cys[index]),
                    float(electrons[index]),
                )

    def _add_trail(
        self,
        signal: SingleArray,
        roi: Roi,
        mixture: MixturePsf,
        weights: FloatArray,
        sigmas: FloatArray,
        xs: FloatArray,
        ys: FloatArray,
        electrons: float,
        half: int,
    ) -> None:
        """Add a star that moved through `xs`, `ys` during the exposure, in equal steps of time."""
        count = len(xs)
        cxs = np.rint(xs).astype(np.int64)
        cys = np.rint(ys).astype(np.int64)
        stamps = mixture.stamps(xs - cxs, ys - cys, weights, sigmas, size_px=2 * half + 1)
        share = electrons / count
        for index in range(count):
            self._add_stamp(signal, roi, stamps[index], int(cxs[index]), int(cys[index]), share)


def _no_star(t_ref_utc_ns: int, tilt: tuple[float, float], transparency: float) -> Reference:
    """The reference of a frame that holds no star."""
    return Reference(
        star_index=-1,
        t_ref_utc_ns=t_ref_utc_ns,
        tilt_x_arcsec=tilt[0],
        tilt_y_arcsec=tilt[1],
        star_x_px=math.nan,
        star_y_px=math.nan,
        catalog_x_px=math.nan,
        catalog_y_px=math.nan,
        flux_e=0.0,
        scintillation_factor=1.0,
        transparency=transparency,
    )
