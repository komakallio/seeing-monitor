"""The simulated camera: a `CameraDriver` that renders frames of a synthetic sky.

`SimDriver` honors the lifecycle of the driver interface (`open`, `configure`, `start`,
`read_frame`, `stop`, `close`), reads the geometry back, and stamps frames the way the
architecture describes. It runs on a `Clock`. With a `VirtualClock`, `read_frame` advances the
clock by the frame period and returns at once, so a simulated night takes seconds. With a
`SystemClock`, it sleeps until each frame is due.

**Timing.** In video mode, the exposure of the first row of frame `k` starts at `t0 + k P`,
where `t0` is the time of `start` and `P` is the frame period: the larger of the exposure and the
readout time (the frame overhead plus the rows times the row time). The frame arrives at the end
of its period. `t_utc_ns` is the middle of the exposure of the first row, so it is exact
(`TimeQuality.EXACT`), and row `r` of the ROI exposes `r` row times later. A snapshot exposes
once, and its period is the exposure plus the readout.

**Faults.** `SimOptions.faults` injects drops, timeouts, slow reads, a disconnect, a stall that
only a recovery clears, and a silent change of geometry. A reader that falls more than
`max_lag_frames` behind loses frames, as a camera without a frame buffer does.

**A cover.** With `SimOptions.cover_file`, the camera is covered while that file exists: a frame
then shows the sensor alone (no star, no sky), as a camera under a lens cap does. The driver asks
the file system at most every `cover_check_s`, so a person can cover and uncover a camera that runs
in another process (`seeingmon dev` prints the file to create and to delete).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from seeingmon.clock import NS_PER_S, Clock, sleep_until_utc_ns
from seeingmon.drivers.base import (
    CameraCaps,
    CameraConfigError,
    CameraDisconnectedError,
    CameraError,
    CameraInfo,
    CameraStateError,
    CameraTimeoutError,
    RecoveryLevel,
)
from seeingmon.drivers.sim.faults import FaultRuntime, GeometryChange
from seeingmon.drivers.sim.optics import PsfConfig, psf_pupil_spacing_m
from seeingmon.drivers.sim.options import SimOptions
from seeingmon.drivers.sim.params import SimParams
from seeingmon.drivers.sim.render import FrameRenderer
from seeingmon.drivers.sim.stars import Pointing, SkyProjector, make_polar_field
from seeingmon.drivers.sim.truth import FrameTruth, SimTruth
from seeingmon.drivers.sim.turbulence import Layer, TurbulenceConfig, TurbulenceModel
from seeingmon.frames import (
    ActiveStream,
    Frame,
    FrameFlag,
    PixelFormat,
    Roi,
    StreamConfig,
    StreamKind,
    TimeQuality,
)

GAIN_RANGE = (0, 570)
EXPOSURE_US_RANGE = (32, 2_000_000_000)
OFFSET_RANGE = (0, 255)


def _modes_from_profile(profile: Any) -> dict[str, SimParams]:
    """Build the readout modes from `None` (the reference hardware), `SimParams`, or a profile."""
    if profile is None:
        return {name: SimParams.reference(name) for name in ("bin1", "bin2")}
    if isinstance(profile, SimParams):
        return {profile.mode: profile}
    if isinstance(profile, Mapping):
        return {str(name): value for name, value in profile.items()}
    return SimParams.modes_from_profile(profile)


class SimDriver:
    """A simulated camera. Build one with `create`, or pass the pieces directly.

    `modes` maps each readout mode name to its parameters. The first mode sets the aperture and
    the optics that all modes share, so give modes of one telescope.
    """

    def __init__(
        self,
        modes: Mapping[str, SimParams],
        clock: Clock,
        options: SimOptions | None = None,
    ) -> None:
        if not modes:
            raise ValueError("at least one readout mode is required")
        self._modes = dict(modes)
        self._clock = clock
        self._options = options or SimOptions()
        first = next(iter(self._modes.values()))
        opts = self._options
        turbulence = opts.turbulence
        if turbulence is None:
            zenith = 90.0 - opts.site.latitude_deg
            turbulence = TurbulenceConfig(seed=opts.seed, zenith_angle_deg=zenith)
        self._model = TurbulenceModel(turbulence, first.aperture_m)
        field = opts.stars if opts.stars is not None else make_polar_field(opts.seed)
        pointing = opts.pointing or Pointing(t_ref_utc_ns=clock.utc_ns())
        self._projector = SkyProjector(field, pointing)
        self._truth = SimTruth(
            model=self._model,
            modes=self._modes,
            field=field,
            projector=self._projector,
            clouds=opts.clouds,
            site=opts.site,
            epoch_utc_ns=opts.epoch_utc_ns,
            dark_sky_mag_arcsec2=opts.sky_mag_arcsec2,
            twilight=opts.twilight,
            scintillation=opts.scintillation,
            ambient_c=opts.ambient_c,
            sensor_rise_c=opts.sensor_rise_c,
            ambient_drift_c_per_hour=opts.ambient_drift_c_per_hour,
            keep_frames=opts.keep_truth_frames,
            pupil_spacing_m=psf_pupil_spacing_m(first, opts.psf),
        )
        self._renderer = FrameRenderer(
            truth=self._truth,
            projector=self._projector,
            clouds=opts.clouds,
            psf=opts.psf,
            scintillation=opts.scintillation,
            hot_pixels=opts.hot_pixels,
            seed=opts.seed,
            epoch_utc_ns=opts.epoch_utc_ns,
        )
        self._faults = FaultRuntime(opts.faults)
        self._cover_path = None if opts.cover_file is None else Path(opts.cover_file)
        self._covered = False
        self._next_cover_check_ns = 0
        self._rng = np.random.default_rng(np.random.SeedSequence([opts.seed, 0xF4A3E]))
        self._opened = False
        self._active: ActiveStream | None = None
        self._params: SimParams | None = None
        self._roi: Roi | None = None
        self._running = False
        self._stream_id = 0
        self._seq = 0
        self._delivered = 0  # frames delivered since open, across streams
        self._dropped_counter = 0
        self._recovered = False
        self._t0_ns = 0
        self._index = 0
        self._period_ns = 1
        self._last_arrival_ns = 0
        self.calls: list[tuple[str, Any]] = []

    # --- extras for tests and tools ---

    @property
    def truth(self) -> SimTruth:
        """The injected truth. See `seeingmon.drivers.sim.truth`."""
        return self._truth

    @property
    def options(self) -> SimOptions:
        return self._options

    @property
    def modes(self) -> dict[str, SimParams]:
        """The readout modes, by name."""
        return dict(self._modes)

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def active(self) -> ActiveStream | None:
        """The stream that the last `configure` returned."""
        return self._active

    # --- CameraDriver ---

    @property
    def name(self) -> str:
        return "sim"

    def open(self) -> CameraInfo:
        self.calls.append(("open", None))
        if self._faults.disconnected:
            raise CameraDisconnectedError("no camera answers")
        self._opened = True
        self._delivered = 0
        self._faults.reset_connection()
        width = max(params.width * params.binning for params in self._modes.values())
        height = max(params.height * params.binning for params in self._modes.values())
        return CameraInfo(
            model="Simulated ZWO ASI294MM",
            driver=self.name,
            sdk_version=None,
            max_width=width,
            max_height=height,
            has_temperature=True,
        )

    def close(self) -> None:
        self.calls.append(("close", None))
        self._running = False
        self._opened = False

    def capabilities(self) -> CameraCaps:
        first = next(iter(self._modes.values()))
        return CameraCaps(
            gain_range=GAIN_RANGE,
            exposure_us_range=EXPOSURE_US_RANGE,
            bins=tuple(sorted({params.binning for params in self._modes.values()})),
            pixel_formats=(PixelFormat.RAW8, PixelFormat.RAW16),
            offset_range=OFFSET_RANGE,
            roi_width_multiple=first.roi_width_multiple,
            roi_height_multiple=first.roi_height_multiple,
        )

    def configure(self, config: StreamConfig) -> ActiveStream:
        self.calls.append(("configure", config))
        if not self._opened:
            raise CameraStateError("configure before open")
        self._running = False
        base = self._modes.get(config.mode)
        if base is None:
            known = ", ".join(sorted(self._modes))
            raise CameraConfigError(f"unknown readout mode {config.mode!r}: use one of {known}")
        if not GAIN_RANGE[0] <= config.gain <= GAIN_RANGE[1]:
            raise CameraConfigError(f"gain {config.gain} is outside {GAIN_RANGE}")
        if not EXPOSURE_US_RANGE[0] <= config.exposure_us <= EXPOSURE_US_RANGE[1]:
            raise CameraConfigError(
                f"exposure {config.exposure_us} us is outside {EXPOSURE_US_RANGE}"
            )
        try:
            params = base.effective(high_speed=config.high_speed)
            roi = params.normalize_roi(config.roi)
        except ValueError as exc:
            raise CameraConfigError(str(exc)) from exc
        change = self._faults.next_geometry_change()
        if change is not None:
            roi = _apply_change(params, roi, change)
        self._params = params
        self._roi = roi
        self._stream_id += 1
        self._seq = 0
        self._dropped_counter = 0
        applied = replace(config, roi=roi, offset=self._offset_of(config, params))
        period_s = self._period_s(applied, params, roi)
        self._active = ActiveStream(
            stream_id=self._stream_id,
            config=applied,
            frame_shape=(roi.height, roi.width),
            adc_bits=params.adc_bits,
            frame_period_s=period_s,
        )
        return self._active

    def start(self) -> None:
        self.calls.append(("start", None))
        if self._active is None or self._params is None or self._roi is None:
            raise CameraStateError("start before configure")
        if self._faults.disconnected:
            raise CameraDisconnectedError("the camera is gone")
        self._running = True
        self._t0_ns = self._clock.utc_ns()
        self._index = 0
        self._dropped_counter = 0
        period_s = self._active.frame_period_s or 0.0
        self._period_ns = max(1, round(period_s * NS_PER_S))

    def read_frame(self, timeout_s: float) -> Frame:
        self.calls.append(("read_frame", timeout_s))
        if not self._running or self._active is None or self._params is None or self._roi is None:
            raise CameraStateError("read_frame while not capturing")
        config = self._active.config
        params = self._params
        fault = self._faults.check(self._delivered)
        if fault.disconnected:
            raise CameraDisconnectedError("the camera was unplugged")
        if fault.timeout:
            self._clock.sleep(timeout_s)
            raise CameraTimeoutError(f"no frame within {timeout_s} s")
        period = self._period_ns
        skipped = fault.lost_frames  # frames that the timeline skips
        index = self._index + skipped
        arrival_ns = self._t0_ns + (index + 1) * period
        now_ns = self._clock.utc_ns()
        late = (now_ns - arrival_ns) // period - self._options.max_lag_frames
        if late > 0:  # the reader fell behind, and the camera lost the frames in between
            index += late
            skipped += late
            arrival_ns += late * period
        lost = skipped
        if self._recovered and self._last_arrival_ns:
            # Frames that the camera would have produced while it was down are a gap.
            lost += max(0, (self._t0_ns - self._last_arrival_ns) // period - 1)
        wait_s = (arrival_ns - now_ns) / NS_PER_S
        if wait_s > timeout_s:
            self._clock.sleep(timeout_s)
            raise CameraTimeoutError(f"no frame within {timeout_s} s")
        if fault.delay_s > 0:
            self._clock.sleep(max(wait_s, 0.0) + fault.delay_s)
        else:
            sleep_until_utc_ns(self._clock, arrival_ns)
        t_arrival_ns = self._clock.utc_ns()
        exposure_s = config.exposure_us * 1e-6
        t_start_ns = self._t0_ns + index * period
        t_utc_ns = t_start_ns + round(exposure_s * 0.5 * NS_PER_S)
        rendered = self._renderer.render(
            params,
            config,
            self._roi,
            t_start_ns,
            config.gain,
            self._rng,
            covered=self._is_covered(),
        )
        flags = FrameFlag.SIMULATED | (FrameFlag.RECOVERED if self._recovered else FrameFlag.NONE)
        frame = Frame(
            data=rendered.data,
            stream_id=self._active.stream_id,
            seq=self._seq,
            t_arrival_ns=t_arrival_ns,
            t_utc_ns=t_utc_ns,
            t_err_ns=self._options.time_error_ns,
            t_quality=TimeQuality.EXACT,
            dropped_before=lost,
            exposure_us=config.exposure_us,
            gain=config.gain,
            mode=config.mode,
            roi=self._roi,
            adc_bits=params.adc_bits,
            temperature_c=rendered.sensor_temperature_c,
            flags=flags,
        )
        reference = rendered.reference
        self._truth.record(
            FrameTruth(
                seq=self._seq,
                stream_id=self._active.stream_id,
                t_utc_ns=t_utc_ns,
                t_ref_utc_ns=reference.t_ref_utc_ns,
                exposure_s=exposure_s,
                mode=config.mode,
                tilt_x_arcsec=reference.tilt_x_arcsec,
                tilt_y_arcsec=reference.tilt_y_arcsec,
                star_index=reference.star_index,
                star_x_px=reference.star_x_px,
                star_y_px=reference.star_y_px,
                catalog_x_px=reference.catalog_x_px,
                catalog_y_px=reference.catalog_y_px,
                flux_e=reference.flux_e,
                scintillation_factor=reference.scintillation_factor,
                transparency=reference.transparency,
                sky_mag_arcsec2=rendered.sky_mag_arcsec2,
                sun_altitude_deg=rendered.sun_altitude_deg,
                sensor_temperature_c=rendered.sensor_temperature_c,
                dropped_before=lost,
            )
        )
        self._recovered = False
        self._last_arrival_ns = t_arrival_ns
        self._dropped_counter += lost
        self._seq += 1
        self._delivered += 1
        self._index = index + 1
        if config.kind is StreamKind.SNAPSHOT:
            self._running = False
        return frame

    def stop(self) -> None:
        self.calls.append(("stop", None))
        self._running = False

    def move_roi(self, x: int, y: int) -> Roi:
        self.calls.append(("move_roi", (x, y)))
        if self._active is None or self._params is None or self._roi is None:
            raise CameraStateError("move_roi before configure")
        params = self._params
        self._roi = Roi(
            min(max(x, 0), params.width - self._roi.width),
            min(max(y, 0), params.height - self._roi.height),
            self._roi.width,
            self._roi.height,
        )
        return self._roi

    def read_temperature_c(self) -> float | None:
        return self._truth.sensor_temperature_c(self._clock.utc_ns())

    def dropped_frames(self) -> int:
        return self._dropped_counter

    def recover(self, level: RecoveryLevel) -> None:
        self.calls.append(("recover", level))
        self._faults.recover(level)
        if self._faults.disconnected:
            raise CameraDisconnectedError("the camera did not come back")
        if self._faults.stalled:
            raise CameraError("the recovery step did not clear the stall")
        self._running = False
        self._recovered = True

    # --- internals ---

    def _is_covered(self) -> bool:
        """Whether the cover file exists. The file system answers at most every `cover_check_s`."""
        path = self._cover_path
        if path is None:
            return False
        now_ns = self._clock.monotonic_ns()
        if now_ns >= self._next_cover_check_ns:
            self._covered = path.exists()
            self._next_cover_check_ns = now_ns + round(self._options.cover_check_s * NS_PER_S)
        return self._covered

    @staticmethod
    def _offset_of(config: StreamConfig, params: SimParams) -> int:
        return params.default_offset if config.offset is None else config.offset

    @staticmethod
    def _period_s(config: StreamConfig, params: SimParams, roi: Roi) -> float:
        exposure_s = config.exposure_us * 1e-6
        readout_s = params.readout_time_s(roi.height)
        if config.kind is StreamKind.SNAPSHOT:
            return exposure_s + readout_s
        return max(exposure_s, readout_s)


def _apply_change(params: SimParams, roi: Roi, change: GeometryChange) -> Roi:
    """Apply a silent geometry change, and keep the ROI inside the frame."""
    width = max(params.roi_width_multiple, roi.width + change.d_width)
    height = max(params.roi_height_multiple, roi.height + change.d_height)
    width = min(width, params.width)
    height = min(height, params.height)
    return Roi(
        min(max(roi.x + change.dx, 0), params.width - width),
        min(max(roi.y + change.dy, 0), params.height - height),
        width,
        height,
    )


def create(
    *,
    profile: Any = None,
    clock: Clock,
    options: SimOptions | Mapping[str, Any] | None = None,
) -> SimDriver:
    """Build a simulated camera. This is the factory that `create_driver("sim", ...)` calls.

    `profile` is a `seeingmon.profile.Profile`, a `SimParams` for one mode, a mapping of mode
    name to `SimParams`, or `None` for the reference hardware (bin1 and bin2). `options` is a
    `SimOptions`, a dictionary for `SimOptions.from_mapping`, or `None` for the defaults.
    """
    if options is None or isinstance(options, SimOptions):
        resolved = options or SimOptions()
    else:
        resolved = SimOptions.from_mapping(options)
    return SimDriver(_modes_from_profile(profile), clock, resolved)


def sim_camera(
    clock: Clock,
    *,
    r0_m: float = 0.10,
    wind_speed_m_s: float = 10.0,
    wind_direction_deg: float = 45.0,
    outer_scale_m: float = math.inf,
    seed: int = 1,
    psf_mode: str = "wave",
    zenith_angle_deg: float = 0.0,
    screen_points: int = 512,
    wind_variability: float | None = None,
    **option_fields: Any,
) -> SimDriver:
    """Build a simulated camera with one turbulent layer of a known strength, for tests.

    The layer has `r0_m` at 500 nm at the zenith (at `zenith_angle_deg`, if you set it), blows at
    `wind_speed_m_s` toward `wind_direction_deg`, and has the outer scale `outer_scale_m`. The
    wind speed fluctuates slowly by `wind_variability` (the `TurbulenceConfig` default, 12%, when
    you leave it out); pass 0 for a constant wind. Other keyword arguments go to `SimOptions`
    (for example `faults=SimFaults(...)` or `stars=...`).
    Read the true image motion from `driver.truth`: `driver.truth.tilt_arcsec(t_start_utc_ns,
    exposure_s)` for any exposure, or `driver.truth.frames` for the frames that you read.
    """
    turbulence = TurbulenceConfig(
        r0_m=r0_m,
        layers=(Layer(1.0, wind_speed_m_s, wind_direction_deg),),
        outer_scale_m=outer_scale_m,
        zenith_angle_deg=zenith_angle_deg,
        seed=seed,
        screen_points=screen_points,
    )
    if wind_variability is not None:
        turbulence = replace(turbulence, wind_variability=wind_variability)
    option_fields.setdefault("psf", PsfConfig(mode=psf_mode))  # type: ignore[arg-type]
    options = SimOptions(seed=seed, turbulence=turbulence, **option_fields)
    return SimDriver(_modes_from_profile(None), clock, options)
