"""The fast-path analyzer: the `FastAnalyzer` that the scheduler feeds frame by frame.

`FastPathAnalyzer` measures every frame with the kernel, groups the frames into windows, and turns
each finished window into a `SeeingWindowRecord`. Build one with `create_fast_analyzer`:

    fast = config.section("fastpath", FastPathConfig)
    analyzer = create_fast_analyzer(config.profile, fast, config.station_id)
    analyzer.begin_stream(stream)  # after every CameraDriver.configure
    update = analyzer.push(frame)  # update.star is the StarState, update.windows the records
    rows = analyzer.drain_metrics()  # per-frame rows with the `frame` record dtype

**Stream setup.** The first frame of a stream (or `begin_stream`) fixes what the kernel and the
estimator need from the profile: the plate scale, the conversion gain and noise at the frame's
gain, the aperture of the star image, the saturation level of the ADC depth and pixel format, and
the aperture of the telescope. A readout mode that the profile does not know still works, with no
electron units and no noise model.

**Tracking.** The kernel starts each frame at the centroid of the previous one. A lost star
sends the next frame back to the brightest-patch search.

**Search.** `measure` looks for the star in one frame of a search burst with the matched filters
(`seeingmon.fastpath.kernel.search_frame`), within a radius of the position that the scheduler
predicts, and returns the star with the best matched SNR, which decides the detection, and the SNR
of the centroid aperture at that position. The frames reach no window, no metric row, and no live
value. The filters are Gaussians of `[fastpath] matched_fwhm_airy_widths` Airy FWHM of the mode,
so a star in focus and a wider image both meet a filter of about their size. In `push`, the star
is missing when neither the first filter around the centroid nor the centroid aperture reaches
`min_star_snr`.

**Cost.** `push` does the kernel, a few list appends, and the window bookkeeping, and it finishes
in well under a frame period. The work of a window (the fits, the spectrum, the corrections)
runs once per window inside the `push` that closes it. It takes a few milliseconds.

**Context.** `set_context` changes the flags, the heater duty, and the zenith angle that apply to
windows that close afterwards. The zenith angle sets the conversion of `r0` to the zenith.

**Noise.** The kernel models the noise of each centroid with the sky noise that it measures on
the ROI border, and the estimator subtracts it. A window gets `noisy` when the error of that model,
at the window's share of noise in the variance, can bias `r0` by more than `max_noise_bias`
(`seeingmon.fastpath.models.noise_bias`). `[fastpath] centroid` picks the centroid: the aperture's,
the default, or the Gaussian-weighted one, whose noise in a bright sky is a hundredth of the
aperture's (`docs/research-notes.md`, "The seeing in a bright sky").

**Live value.** `push` also keeps the recent frames in a ring, and every `live_every_s` seconds of
frame time it estimates the seeing of the newest `live_span_s` seconds with the estimator of the
windows (`seeingmon.fastpath.live`). `live` holds the newest value as an immutable `LiveSeeing`,
and a thread other than the consumer may read it. A new stream clears it. `live_enabled` switches
the estimate off.

**Thread safety.** One consumer thread calls the analyzer, as `seeingmon.analysis.base` requires.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from seeingmon.analysis.base import NO_STAR, FastContext, FastUpdate, StarState
from seeingmon.fastpath import models
from seeingmon.fastpath.config import FastPathConfig
from seeingmon.fastpath.estimator import EstimatorSettings, MotionSeries, estimate_seeing
from seeingmon.fastpath.kernel import (
    FLAG_EDGE,
    FLAG_HOT_PIXEL,
    FLAG_SATURATED,
    FrameCalibration,
    KernelParams,
    Measurement,
    measure_frame,
    pixel_variance_e2,
    search_frame,
)
from seeingmon.fastpath.live import LiveEstimator, LiveSeeing, LiveStream
from seeingmon.fastpath.scintillation import scintillation_index
from seeingmon.fastpath.spectrum import MotionSpectrum, aliasing_expected, compute_spectrum
from seeingmon.fastpath.windows import ClosedWindow, WindowAssembler, is_partial
from seeingmon.frames import ActiveStream, Frame, Roi
from seeingmon.profile import Profile, ProfileError, derived
from seeingmon.records import SeeingWindowRecord
from seeingmon.records.seeing import SEEING_WINDOW_FLAGS
from seeingmon.records.segments import segment_dtype

_log = logging.getLogger(__name__)

ALGORITHM_REVISION = "fast-2"
"""The algorithm revision that every window record carries in `provenance["algo"]`.

`fast-2` added the sky to the noise model of the centroid and of the scintillation floor, so the
seeing of a window in a bright sky is no longer biased toward bad seeing (`fast-1` read an `r0` of
3.3 cm against 10 cm in the simulator's daylight), and added the `noisy` flag and the option of
the Gaussian-weighted centroid (`[fastpath] centroid`)."""

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
_UINT16_MAX = 65_535
_ANALYSIS_FLAG_MASK = 0x1F  # the bits of `FrameFlag` in the `flags` column of the `frame` record
_KEPT_STREAMS = 4


@dataclass(frozen=True, slots=True)
class _Stream:
    """What the analyzer derives from the profile for one stream."""

    key: tuple[object, ...]
    kernel: KernelParams
    calibration: FrameCalibration
    settings: EstimatorSettings
    plate_scale_arcsec_per_px: float
    pixel_var_e2: float
    e_per_dn: float
    known: bool  # whether the profile describes the readout mode
    saturation_dn: float  # the profile's saturation level in container counts, or `NaN`
    noise_model_error: float  # the relative error of the centroid's noise model, for `noisy`


def _stream_key(
    mode: str, gain: int, exposure_us: int, adc_bits: int, container_bits: int
) -> tuple[object, ...]:
    return (mode, gain, exposure_us, adc_bits, container_bits)


class FastPathAnalyzer:
    """A `FastAnalyzer` with a NumPy kernel, windows of frame time, and the seeing estimator.

    Build it with `create_fast_analyzer`. `station_id` tags the records, and the profile supplies
    the hardware numbers.
    """

    def __init__(
        self,
        profile: Profile,
        config: FastPathConfig | None = None,
        *,
        station_id: str = "unset",
    ) -> None:
        self._profile = profile
        self._config = config or FastPathConfig()
        self._station_id = station_id
        self._assembler = WindowAssembler(self._config.window_s)
        self._context = FastContext()
        self._streams: dict[int, _Stream] = {}
        self._current: _Stream | None = None
        self._stream_id: int | None = None
        self._guess: tuple[float, float] | None = None
        self._rows: list[tuple[Any, ...]] = []
        self._frame_dtype = segment_dtype("frame")
        self.star = NO_STAR
        self.frames_pushed = 0
        self.frames_measured = 0  # frames of `measure`, which reach no window
        self._measure_streams: dict[tuple[object, ...], _Stream] = {}
        self.metrics_dropped = 0
        self._live = LiveEstimator(self._config) if self._config.live_enabled else None
        self.live: LiveSeeing | None = None
        self.live_errors = 0
        aperture_m = profile.optics.aperture_mm * 1e-3
        models.outer_scale_ratio(aperture_m, self._config.outer_scale_m)  # warm the caches
        models.tilt_spectrum(aperture_m, self._config.outer_scale_m, self._config.assumed_wind_ms)

    # --- the FastAnalyzer interface ----------------------------------------------------------

    def begin_stream(self, stream: ActiveStream) -> tuple[SeeingWindowRecord, ...]:
        """Start a stream, and return the windows that the old stream left open (they are partial
        when shorter than the window length)."""
        config = stream.config
        closed = self._assembler.begin_stream(stream.stream_id, stream.frame_period_s)
        records = tuple(self._finalize(window) for window in closed)
        container_bits = config.pixel_format.value
        self._enter(
            stream.stream_id,
            config.mode,
            config.gain,
            config.exposure_us,
            stream.adc_bits,
            container_bits,
        )
        self._guess = None
        self.star = NO_STAR
        return records

    def set_context(self, context: FastContext) -> None:
        """Replace the context that applies to windows closing from now on.

        Raises `ValueError` for a flag that `SEEING_WINDOW_FLAGS` does not declare, so a mistake
        shows at the call and never inside a window that closes later.
        """
        unknown = sorted(set(context.flags) - set(SEEING_WINDOW_FLAGS))
        if unknown:
            raise ValueError(f"unknown window flags: {', '.join(unknown)}")
        self._context = context

    def push(self, frame: Frame) -> FastUpdate:
        """Measure one frame, add it to its window, and return the star and any closed windows."""
        data = frame.data
        container_bits = 8 if data.dtype.itemsize == 1 else 16
        stream = self._current
        key = _stream_key(frame.mode, frame.gain, frame.exposure_us, frame.adc_bits, container_bits)
        if stream is None or frame.stream_id != self._stream_id or key != stream.key:
            stream = self._enter(
                frame.stream_id,
                frame.mode,
                frame.gain,
                frame.exposure_us,
                frame.adc_bits,
                container_bits,
            )
        roi = frame.roi
        measurement = measure_frame(
            data, roi.x, roi.y, stream.kernel, stream.calibration, self._guess
        )
        found = measurement.found
        analysis_flags = measurement.flags
        flux_e = measurement.flux_dn * stream.e_per_dn
        raw_flags = int(frame.flags)
        usable = found and not analysis_flags & (FLAG_EDGE | FLAG_HOT_PIXEL)
        saturated = bool(analysis_flags & FLAG_SATURATED)
        peak = measurement.peak_dn
        pixel_var = pixel_variance_e2(measurement.bg_sigma_dn, stream.calibration)
        self._rows.append(
            (
                frame.t_utc_ns,
                frame.seq,
                min(frame.t_err_ns // 1000, _UINT16_MAX),
                measurement.x,
                measurement.y,
                measurement.width_x,
                measurement.width_y,
                min(int(peak), _UINT16_MAX),
                flux_e,
                measurement.bg_dn,
                (raw_flags & _ANALYSIS_FLAG_MASK) | analysis_flags,
                min(frame.dropped_before, _UINT16_MAX),
            )
        )
        if len(self._rows) > self._config.max_buffered_metrics:
            excess = len(self._rows) - self._config.max_buffered_metrics // 2
            del self._rows[:excess]
            self.metrics_dropped += excess
        closed = self._assembler.add(
            frame.stream_id,
            frame.t_utc_ns,
            frame.dropped_before,
            frame.mode,
            frame.exposure_us,
            frame.gain,
            frame.temperature_c,
            bool(raw_flags & 1) or int(frame.t_quality) == 0,
            usable,
            saturated,
            (
                measurement.x,
                measurement.y,
                measurement.width_x,
                measurement.width_y,
                peak,
                flux_e,
                measurement.bg_dn,
                measurement.noise_var_x,
                measurement.noise_var_y,
            ),
            measurement.snr,
            pixel_var,
        )
        windows = tuple(self._finalize(window) for window in closed) if closed else ()
        live = self._live
        if live is not None and live.add(
            frame.t_utc_ns,
            usable,
            saturated,
            measurement.x,
            measurement.y,
            measurement.width_x,
            measurement.width_y,
            measurement.noise_var_x,
            measurement.noise_var_y,
            peak,
            flux_e,
            frame.dropped_before,
        ):
            self._estimate_live(live, frame, stream)
        if found:
            x, y = measurement.x, measurement.y
            self._guess = (x, y)
            self.star = _star_state(measurement, roi, stream)
        else:
            self._guess = None
            self.star = NO_STAR
        self.frames_pushed += 1
        return FastUpdate(star=self.star, windows=windows)

    def measure(
        self,
        frame: Frame,
        at: tuple[float, float] | None = None,
        radius_px: float | None = None,
    ) -> StarState:
        """Look for the star in one frame without a window, a metric row, or a live value.

        The matched filters look within `radius_px` of `at` (sensor pixels), or over the whole
        frame without one of them, and the best of them gives the star. The stream of `push`, its
        guess, and its star stay as they were. For a readout mode that the profile does not know,
        the kernel measures the frame as `push` does, from `at`.
        """
        data = frame.data
        container_bits = 8 if data.dtype.itemsize == 1 else 16
        key = _stream_key(frame.mode, frame.gain, frame.exposure_us, frame.adc_bits, container_bits)
        stream = self._measure_streams.get(key)
        if stream is None:
            stream = self._build_stream(
                frame.mode, frame.gain, frame.exposure_us, frame.adc_bits, container_bits
            )
            self._measure_streams[key] = stream
            while len(self._measure_streams) > _KEPT_STREAMS:
                del self._measure_streams[next(iter(self._measure_streams))]
        roi = frame.roi
        measurement = search_frame(
            data, roi.x, roi.y, stream.kernel, stream.calibration, at, radius_px
        )
        self.frames_measured += 1
        if not measurement.found:
            return NO_STAR
        return _star_state(measurement, roi, stream)

    def _estimate_live(self, live: LiveEstimator, frame: Frame, stream: _Stream) -> None:
        """Estimate the rolling seeing value. An error never reaches the caller of `push`."""
        try:
            context = self._context
            self.live = live.estimate(
                LiveStream(
                    frame.stream_id,
                    frame.mode,
                    frame.exposure_us,
                    stream.settings,
                    stream.plate_scale_arcsec_per_px,
                    stream.known,
                ),
                context.flags,
                context.zenith_angle_deg,
            )
        except Exception:
            self.live_errors += 1
            if self.live_errors <= 3:
                _log.exception("the rolling seeing value failed")

    def flush(self, reason: str = "end") -> tuple[SeeingWindowRecord, ...]:
        """Close the open window early and return it (with `partial` when it is short)."""
        return tuple(self._finalize(window) for window in self._assembler.flush(reason))

    def drain_metrics(self) -> npt.NDArray[Any] | None:
        """Return the per-frame rows since the last call, or `None` when there are none."""
        if not self._rows:
            return None
        rows = np.array(self._rows, dtype=self._frame_dtype)
        self._rows.clear()
        return rows

    # --- streams -----------------------------------------------------------------------------

    def _enter(
        self,
        stream_id: int,
        mode: str,
        gain: int,
        exposure_us: int,
        adc_bits: int,
        container_bits: int,
    ) -> _Stream:
        """Register the settings of a stream (a new one, or a changed one) and make it current."""
        stream = self._build_stream(mode, gain, exposure_us, adc_bits, container_bits)
        if self._live is not None:
            self._live.reset(None)  # a changed stream or setting starts the ring again
            self.live = None
        if stream_id != self._stream_id:
            self._guess = None
            self.star = NO_STAR
            self._stream_id = stream_id
        self._streams[stream_id] = stream
        while len(self._streams) > _KEPT_STREAMS:  # a window may close after its stream ended
            del self._streams[next(iter(self._streams))]
        self._current = stream
        return stream

    def _build_stream(
        self, mode: str, gain: int, exposure_us: int, adc_bits: int, container_bits: int
    ) -> _Stream:
        config = self._config
        profile = self._profile
        e_per_dn = math.nan
        pixel_var = math.nan
        plate_scale = math.nan
        airy_px = math.nan
        saturation_dn = math.nan
        sat_adc = adc_bits
        try:
            readout = profile.mode(mode)
            if (
                readout.has_high_speed
                and readout.adc_bits_high_speed == adc_bits
                and readout.adc_bits != adc_bits
            ):
                readout = profile.mode(mode, high_speed=True)
            plate_scale = derived.plate_scale_arcsec_per_px(readout, profile.optics)
            airy_px = derived.airy_fwhm_px(readout, profile.optics)
            e_per_adu = derived.e_per_adu(readout, gain)
            read_noise = derived.read_noise_e(readout, gain)
            e_per_dn = e_per_adu * 2.0 ** (readout.adc_bits - container_bits)
            # The level of the scheduler's gate and exposure: an 8-bit container has its full scale.
            saturation_dn = (
                derived.saturation(readout, gain).container_dn if container_bits == 16 else 255.0
            )
            pixel_var = read_noise**2
            if readout.adc_bits > container_bits:  # a coarse container adds quantization noise
                # This is the noise of a signal that the read and photon noise dither across the
                # count edges. Where they don't, as in the halo of the star in the owner's 8-bit
                # videos, the term over-subtracts: docs/recordings-validation.md puts the bias
                # of r0 at about 2.7% for those videos.
                pixel_var += (e_per_dn**2 - e_per_adu**2) / 12.0
        except ProfileError:
            pass
        diameter = config.aperture_diameter_px
        if diameter is None:
            widths = config.aperture_airy_widths * airy_px if airy_px == airy_px else 0.0
            diameter = max(config.aperture_min_px, widths)
        # A real star fills a pixel only when the pixels undersample it.
        spike = config.hot_pixel_ratio if airy_px == airy_px and airy_px >= 1.0 else None
        matched_fwhms: tuple[float, ...] = ()
        centroid_fwhm: float | None = None
        if airy_px == airy_px:
            matched_fwhms = tuple(
                min(max(widths * airy_px, 0.3), 20.0) for widths in config.matched_fwhm_airy_widths
            )
            if config.centroid == "gaussian":
                centroid_fwhm = min(max(config.centroid_fwhm_airy_widths * airy_px, 0.5), 20.0)
        kernel = KernelParams(
            aperture_diameter_px=diameter,
            recenter_iterations=config.recenter_iterations,
            border_px=config.border_px,
            border_step=config.border_step,
            edge_margin_px=config.edge_margin_px,
            min_snr=config.min_star_snr,
            spike_ratio=spike,
            matched_fwhms_px=matched_fwhms,
            centroid_fwhm_px=centroid_fwhm,
        )
        calibration = FrameCalibration.for_container(
            adc_bits=sat_adc,
            container_bits=container_bits,
            saturation_fraction=config.saturation_fraction,
            e_per_dn=e_per_dn,
            pixel_var_e2=pixel_var,
        )
        aperture_m = profile.optics.aperture_mm * 1e-3
        gain_ratio = 1.0
        if config.apply_centroid_gain and plate_scale == plate_scale:
            lambda_over_d = models.lambda_over_d_arcsec(
                profile.optics.wavelength_nm * 1e-9, aperture_m
            )
            if centroid_fwhm is None:
                gain_ratio = models.windowed_centroid_variance_ratio(
                    0.5 * diameter * plate_scale / lambda_over_d
                )
            else:
                gain_ratio = models.gaussian_centroid_variance_ratio(
                    centroid_fwhm * plate_scale / lambda_over_d
                )
        settings = EstimatorSettings(
            aperture_m=aperture_m,
            plate_scale_arcsec_per_px=plate_scale if plate_scale == plate_scale else 1.0,
            exposure_s=exposure_us * 1e-6,
            outer_scale_m=config.outer_scale_m,
            wind_ms=config.assumed_wind_ms,
            detrend_order=config.detrend_order,
            outlier_sigma=config.outlier_sigma,
            g_tilt_coefficient=config.g_tilt_coefficient,
            fwhm_coefficient=config.fwhm_coefficient,
            structure_lag_min_s=config.structure_lag_min_s,
            structure_lag_max_s=config.structure_lag_max_s,
            centroid_gain_variance_ratio=gain_ratio,
            min_samples=config.min_samples,
        )
        return _Stream(
            key=_stream_key(mode, gain, exposure_us, adc_bits, container_bits),
            kernel=kernel,
            calibration=calibration,
            settings=settings,
            plate_scale_arcsec_per_px=plate_scale,
            pixel_var_e2=pixel_var,
            e_per_dn=e_per_dn,
            known=plate_scale == plate_scale,
            saturation_dn=saturation_dn,
            noise_model_error=models.noise_model_error(
                "aperture" if centroid_fwhm is None else "gaussian"
            ),
        )

    def kernel_setup(
        self, mode: str, gain: int, exposure_us: int, adc_bits: int, container_bits: int
    ) -> tuple[KernelParams, FrameCalibration]:
        """The kernel parameters and the calibration that the analyzer uses for a stream.

        Benchmarks and tools use it to run the kernel on frames outside the analyzer.
        """
        stream = self._build_stream(mode, gain, exposure_us, adc_bits, container_bits)
        return stream.kernel, stream.calibration

    # --- windows -----------------------------------------------------------------------------

    def _stream_of(self, window: ClosedWindow) -> _Stream:
        stream = self._streams.get(window.stream_id)
        if stream is not None:
            return stream
        return self._build_stream(
            window.mode, window.gain, window.exposure_us, 14, 16
        )  # a window whose stream we no longer know: the best guess

    def _finalize(self, window: ClosedWindow) -> SeeingWindowRecord:
        config = self._config
        stream = self._stream_of(window)
        context = self._context
        quality: dict[str, str] = {}
        expected = window.n_frames + window.n_dropped
        valid_fraction = window.n_usable / expected if expected else 0.0
        flags = set(context.flags)
        if window.n_dropped > 0.05 * expected:
            flags.add("degraded")
        if is_partial(window, config.window_s):
            flags.add("partial")
        if window.time_invalid:
            flags.add("time_invalid")
        saturated_fraction = window.n_saturated / window.n_frames if window.n_frames else None
        if saturated_fraction is not None and saturated_fraction > config.saturated_window_fraction:
            flags.add("saturated")

        fields: dict[str, Any] = {}
        reason = self._reason_not_analyzed(window, stream, valid_fraction)
        spectrum: MotionSpectrum | None = None
        noise_share: float | None = None
        if reason is None:
            fields, spectrum, noise_share = self._analyze(window, stream, quality)
        else:
            for name in _ANALYSIS_FIELDS:
                quality[name] = reason
        if spectrum is not None and spectrum.lines_hz:
            flags.add("vibration")
        if (
            noise_share is not None
            and models.noise_bias(noise_share, stream.noise_model_error) > config.max_noise_bias
        ):
            flags.add("noisy")

        plate = stream.plate_scale_arcsec_per_px
        usable = window.usable
        fields.update(self._star_statistics(window, usable, plate, stream.saturation_dn, quality))
        zenith = context.zenith_angle_deg
        outer = config.outer_scale_m
        aperture_m = self._profile.optics.aperture_mm * 1e-3
        weight_px = stream.kernel.centroid_fwhm_px
        centroid = "aperture" if weight_px is None else f"gaussian {weight_px:.1f} px"
        provenance = {
            "algo": ALGORITHM_REVISION,
            "assumptions": (
                f"L0={outer:g} m; wind={config.assumed_wind_ms:g} m/s; "
                f"detrend={config.detrend_order}; aperture={stream.kernel.aperture_diameter_px:.1f}"
                f" px; centroid={centroid}; D={aperture_m * 1e3:.0f} mm"
            ),
        }
        record = SeeingWindowRecord(
            station_id=self._station_id,
            t_utc_ns=window.t_start_ns,
            profile_id=self._profile.id,
            provenance=provenance,
            quality=quality or None,
            duration_s=window.duration_s,
            stream_id=window.stream_id,
            readout_mode=window.mode,
            exposure_us=window.exposure_us,
            gain=window.gain,
            n_frames=window.n_frames,
            n_dropped=window.n_dropped,
            valid_fraction=min(valid_fraction, 1.0),
            frame_rate_hz=window.n_frames / window.duration_s if window.n_frames > 1 else None,
            saturated_fraction=saturated_fraction,
            outer_scale_m=None if math.isinf(outer) else outer,
            assumed_wind_ms=config.assumed_wind_ms,
            zenith_angle_deg=zenith,
            heater_duty=context.heater_duty,
            sensor_temperature_c=window.temperature_c,
            flags=sorted(flags),
            **fields,
        )
        return record

    def _reason_not_analyzed(
        self, window: ClosedWindow, stream: _Stream, valid_fraction: float
    ) -> str | None:
        config = self._config
        if not stream.known:
            return "the readout mode is not in the profile"
        if window.period_s <= 0.0 or window.n_usable < config.min_samples:
            return "too few usable frames"
        if window.duration_s < config.min_window_s:
            return "the window is shorter than the minimum"
        if valid_fraction < config.min_valid_fraction:
            return "too few usable frames"
        return None

    def _analyze(
        self, window: ClosedWindow, stream: _Stream, quality: dict[str, str]
    ) -> tuple[dict[str, Any], MotionSpectrum | None, float | None]:
        """The seeing, the scintillation, and the spectrum of a window that has enough frames,
        and the share of the centroid noise in the variance (`None` when the estimate has none,
        infinite when the motion is below the noise)."""
        config = self._config
        context = self._context
        zenith = context.zenith_angle_deg if config.zenith_correction else None
        settings = replace(stream.settings, zenith_angle_deg=zenith)
        series = MotionSeries(
            period_s=window.period_s,
            x_px=window.on_slots(window.x),
            y_px=window.on_slots(window.y),
            noise_var_x_px2=window.on_slots(window.noise_var_x),
            noise_var_y_px2=window.on_slots(window.noise_var_y),
        )
        estimate = estimate_seeing(series, settings)
        quality.update(estimate.quality)
        noise_share: float | None = None
        if estimate.x is not None and estimate.y is not None and estimate.factors is not None:
            noise = 0.5 * (estimate.x.noise_var_px2 + estimate.y.noise_var_px2)
            motion = 0.5 * (estimate.x.variance_px2 + estimate.y.variance_px2) - noise
            noise_share = noise / motion if motion > 0.0 else math.inf
        scale = stream.plate_scale_arcsec_per_px
        fields: dict[str, Any] = {
            "image_motion_rms_x_arcsec": _finite(estimate.rms_x_arcsec),
            "image_motion_rms_y_arcsec": _finite(estimate.rms_y_arcsec),
            "seeing_fwhm_arcsec": _finite(estimate.fwhm_arcsec),
            "r0_cm": _scaled(estimate.r0_m, 100.0),
            "seeing_fwhm_structure_arcsec": _finite(estimate.fwhm_structure_arcsec),
            "r0_structure_cm": _scaled(estimate.r0_structure_m, 100.0),
            "centroid_noise_px": _finite(estimate.centroid_noise_px),
        }
        if estimate.factors is not None:
            fields["exposure_correction_factor"] = estimate.factors.exposure
            fields["outer_scale_correction_factor"] = estimate.factors.outer_scale
            fields["detrend_correction_factor"] = estimate.factors.detrend
            fields["centroid_gain_correction_factor"] = estimate.factors.centroid_gain
        spectrum: MotionSpectrum | None = None
        if estimate.x is not None and estimate.y is not None:
            spectrum = compute_spectrum(
                estimate.x.residual_px * scale,
                estimate.y.residual_px * scale,
                window.period_s,
                segment_s=config.welch_segment_s,
                overlap=config.welch_overlap,
                max_interp_gap=config.max_interp_gap_frames,
                max_interp_fraction=config.max_interp_fraction,
                bins=config.psd_bins,
                threshold=config.vibration_threshold,
                local_bins=config.vibration_local_bins,
                min_line_hz=config.vibration_min_hz,
            )
        if spectrum is not None:
            fields["motion_psd_freq_hz"] = spectrum.bin_freq_hz.tolist()
            fields["motion_psd_x_arcsec2_per_hz"] = spectrum.bin_psd_x.tolist()
            fields["motion_psd_y_arcsec2_per_hz"] = spectrum.bin_psd_y.tolist()
            fields["vibration_lines_hz"] = list(spectrum.lines_hz)
            fields["motion_psd_dof"] = _finite(spectrum.dof)
            if aliasing_expected(spectrum.nyquist_hz, settings.aperture_m, settings.wind_ms):
                note = (
                    f"power above {spectrum.nyquist_hz:.0f} Hz (the Nyquist frequency) folds "
                    "into the spectrum, and lines above it appear at folded frequencies"
                )
                for name in ("motion_psd_freq_hz", "motion_psd_x_arcsec2_per_hz",
                             "motion_psd_y_arcsec2_per_hz", "vibration_lines_hz"):  # fmt: skip
                    quality[name] = note
        else:
            for name in (
                "motion_psd_freq_hz",
                "motion_psd_x_arcsec2_per_hz",
                "motion_psd_y_arcsec2_per_hz",
                "vibration_lines_hz",
                "motion_psd_dof",
            ):
                quality[name] = "too few contiguous frames for a spectrum"
        flux_slots = window.on_slots(window.flux_e)
        # The sky adds to the floor twice: through the photons in the aperture, and through the
        # noise of the background level that each flux subtracts. In the simulator's daylight the
        # two make the aperture's flux scatter by about 15% of its mean, against 1% from the star's
        # own photons.
        measured = window.pixel_var_e2[np.isfinite(window.pixel_var_e2)]
        pixel_var = float(np.mean(measured)) if len(measured) else stream.pixel_var_e2
        scint = scintillation_index(
            flux_slots,
            window.period_s,
            trend_s=config.scintillation_trend_s,
            pixel_var_e2=pixel_var if pixel_var == pixel_var else 0.0,
            area_px2=stream.kernel.area_px2,
            background_e=(
                window.on_slots(window.bg_dn) * stream.e_per_dn
                if stream.e_per_dn == stream.e_per_dn
                else None
            ),
            exclude=window.on_slots(window.saturated.astype(np.float64)) > 0.5,
        )
        if scint is None:
            quality["scintillation_index"] = "too few unsaturated frames"
        elif not math.isnan(stream.e_per_dn):
            fields["scintillation_index"] = scint.index
        else:
            quality["scintillation_index"] = "the electron scale of the readout mode is unknown"
        return fields, spectrum, noise_share

    def _star_statistics(
        self,
        window: ClosedWindow,
        usable: npt.NDArray[np.bool_],
        plate: float,
        saturation_dn: float,
        quality: dict[str, str],
    ) -> dict[str, Any]:
        """Window means of the star's width, peak, flux, and background, and the star's SNR.

        The background as a share of saturation divides the mean background by the profile's
        saturation level, the level that the scheduler's gate and exposure use. The offset of the
        camera counts as background, because the profile does not know it. The SNR is the median
        over the frames with a usable centroid, in the centroid aperture, because it tells the
        noise of the centroids. The matched SNR, which decides whether the star is there, stays
        out of the record.
        """
        out: dict[str, Any] = {}
        if window.n_frames == 0:
            return out
        if usable.any():
            sigma = 0.5 * (window.width_x[usable] + window.width_y[usable])
            if plate == plate:
                out["width_fwhm_arcsec"] = _finite(float(np.mean(sigma)) * _FWHM_PER_SIGMA * plate)
            out["peak_mean_dn"] = _finite(float(np.mean(window.peak_dn[usable])))
            flux = window.flux_e[usable]
            if np.isfinite(flux).all():
                out["flux_mean_e"] = _finite(float(np.mean(flux)))
        snr = window.snr[np.isfinite(window.snr)]
        if len(snr):
            out["star_snr"] = _finite(float(np.median(snr)))
        elif usable.any():
            quality["star_snr"] = "the electron scale of the readout mode is unknown"
        else:
            quality["star_snr"] = "no frame had a usable centroid"
        background = window.bg_dn[np.isfinite(window.bg_dn)]
        if len(background):
            mean = float(np.mean(background))
            out["background_mean_dn"] = _finite(mean)
            if saturation_dn > 0.0:
                out["background_fraction"] = _finite(min(max(mean / saturation_dn, 0.0), 1.0))
            else:
                quality["background_fraction"] = "the saturation level of the mode is unknown"
        return out


_ANALYSIS_FIELDS = (
    "image_motion_rms_x_arcsec",
    "image_motion_rms_y_arcsec",
    "seeing_fwhm_arcsec",
    "r0_cm",
    "seeing_fwhm_structure_arcsec",
    "r0_structure_cm",
    "scintillation_index",
    "centroid_noise_px",
    "motion_psd_freq_hz",
    "motion_psd_x_arcsec2_per_hz",
    "motion_psd_y_arcsec2_per_hz",
    "vibration_lines_hz",
)


def _star_state(measurement: Measurement, roi: Roi, stream: _Stream) -> StarState:
    """The state of a found star: its position, peak, distance to the ROI edge, and both SNRs."""
    x, y = measurement.x, measurement.y
    return StarState(
        found=True,
        x_px=x,
        y_px=y,
        peak_fraction=measurement.peak_dn / stream.calibration.full_scale_dn,
        edge_distance_px=roi.distance_to_edge(x, y),
        snr=_finite(measurement.snr),
        matched_snr=_finite(measurement.matched_snr),
    )


def _finite(value: float | None) -> float | None:
    """The value as a float, or `None` when it is missing or not finite."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _scaled(value: float | None, factor: float) -> float | None:
    return None if value is None else _finite(value * factor)


def create_fast_analyzer(
    profile: Profile,
    config: FastPathConfig | None = None,
    station_id: str = "unset",
) -> FastPathAnalyzer:
    """Build the fast-path analyzer for a profile.

    `config` is the `[fastpath]` section (`config.section("fastpath", FastPathConfig)`), or `None`
    for the defaults. `station_id` tags every record, and the profile supplies the hardware
    numbers. The caller owns the result and feeds it from one thread.
    """
    return FastPathAnalyzer(profile, config, station_id=station_id)
