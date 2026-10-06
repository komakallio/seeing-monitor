"""The seeing in a bright sky: the fast path's estimate against the simulator's truth.

A bright sky adds photon noise to every pixel, and so to every centroid. The seeing estimator
subtracts the modeled centroid noise from the variance of the motion, so a reading is right only
while the model is right, and its scatter grows with the share of the noise in the variance. This
module drives the simulated camera at a fixed sky and exposure through the fast path with three
methods, and compares `r0` with the injected truth:

- **a**: the aperture's centroid with the noise model of revision `fast-1`, which counted the read
  noise and the rounding, and not the sky: `w^2 / F + n^2 K' / F^2`, with `F` the aperture sum
  above the median of the border, `n^2` the modeled read and quantization noise, and `K'` the sum
  of the weights times `u^2`. The module computes it from the per-frame metrics of method b.
- **b**: the aperture's centroid with the sky term, the default of the fast path.
- **c**: the Gaussian-weighted centroid (`[fastpath] centroid = "gaussian"`) with its own noise
  model.

Methods b and c run through `FastPathAnalyzer` on the same frames, so they see the same atmosphere
and the same photons. Method a shares the centroids of b. Its estimate is that of b with the old
noise subtracted instead: the estimator's motion variance is the variance of the residuals minus
the mean modeled noise, so the motion of a is that of b plus the noise of b minus the noise of a,
and `r0` follows through the same chain of corrections, as the power `-3/5` of the variance.

**The true noise.** The simulator knows where the star was in every frame (the G-tilt over the
exposure), so the module also measures the noise of each method's centroids directly. Per window
and axis, it fits the centroids with a quadratic in time plus a multiple of the true position,
and takes the variance of what remains. In a dark sky that variance holds the photon noise and the
part of the image motion that a centroid measures differently from the G-tilt, and with the same
seed the atmosphere and that part stay the same, so the difference between a bright case and the
dark case of the same seed, `r0`, and exposure is the noise that the sky adds.
`MethodResult.residual_px2` holds the variance, and the research notes take the difference.

`run_case` measures one case, and `docs/research-notes.md` ("The seeing in a bright sky") holds
the table and the method.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from seeingmon.analysis.base import FastContext
from seeingmon.clock import VirtualClock
from seeingmon.drivers.sim.driver import sim_camera
from seeingmon.fastpath.analyzer import FastPathAnalyzer
from seeingmon.fastpath.config import FastPathConfig
from seeingmon.fastpath.kernel import FLAG_EDGE, FLAG_HOT_PIXEL, FLAG_NO_STAR
from seeingmon.frames import PixelFormat, Roi, StreamConfig
from seeingmon.profile import Profile, derived
from seeingmon.records import SeeingWindowRecord

FloatArray = npt.NDArray[np.float64]

METHODS = ("a", "b", "c")
ZENITH_ANGLE_DEG = 35.0
"""The zenith angle of the pole at the simulator's synthetic site."""

_UNUSABLE = FLAG_NO_STAR | FLAG_EDGE | FLAG_HOT_PIXEL
_MODE = "bin1"


@dataclass(frozen=True, slots=True)
class BiasCase:
    """One sky and exposure of the simulated camera.

    `sky_mag_arcsec2` is the sky near the pole, which stays fixed, and `exposure_us` the fast
    exposure at gain 0. `r0_m` is the injected Fried parameter at 500 nm at the zenith, `seed` the
    seed of the atmosphere and the photons (the same seed gives the same atmosphere at every sky),
    and `windows` the number of windows of `window_s` seconds. The ROI of `roi_px` square pixels
    starts centered on Polaris. `psf_mode` is the simulator's optics: `wave`, its reference, or
    `gaussian`, its cheap mixture.
    """

    sky_mag_arcsec2: float
    exposure_us: int
    r0_m: float = 0.10
    seed: int = 1
    windows: int = 3
    window_s: float = 60.0
    roi_px: int = 128
    psf_mode: str = "wave"


@dataclass(frozen=True, slots=True)
class MethodResult:
    """The windows of one method.

    `r0_cm` holds `r0` of each window in centimeters (`None` when the motion fell below the
    noise), `noise_px2` the mean modeled centroid noise per axis, `motion_px2` the variance of the
    motion per axis that the estimator kept, and `residual_px2` the variance per axis of the
    centroids around the true position (see the module documentation), all in square pixels.
    `flags` holds the flags of each window record. Method a has no records, so its tuple is empty.
    """

    r0_cm: tuple[float | None, ...]
    noise_px2: tuple[float, ...]
    motion_px2: tuple[float, ...]
    residual_px2: tuple[float, ...]
    flags: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class CaseResult:
    """A case: the truth, what the window records of method b carry, and each method."""

    case: BiasCase
    truth_r0_cm: float
    star_snr: tuple[float | None, ...]
    background_fraction: tuple[float | None, ...]
    methods: dict[str, MethodResult] = field(default_factory=dict)

    def r0_ratio(self, method: str) -> tuple[float | None, ...]:
        """`r0` of each window of a method over the truth."""
        return tuple(
            None if value is None else value / self.truth_r0_cm
            for value in self.methods[method].r0_cm
        )

    def noise_share(self, method: str) -> float:
        """The mean modeled noise over the mean motion variance that the estimator kept."""
        result = self.methods[method]
        motion = float(np.mean(result.motion_px2))
        return float(np.mean(result.noise_px2)) / motion if motion > 0.0 else float("inf")


def _plate_scale(profile: Profile) -> float:
    return derived.plate_scale_arcsec_per_px(profile.mode(_MODE), profile.optics)


def _analyzer(profile: Profile, case: BiasCase, centroid: str) -> FastPathAnalyzer:
    config = FastPathConfig(
        window_s=case.window_s, min_window_s=min(5.0, case.window_s), centroid=centroid
    )
    analyzer = FastPathAnalyzer(profile, config, station_id="sim")
    analyzer.set_context(FastContext(zenith_angle_deg=ZENITH_ANGLE_DEG))
    return analyzer


def run_case(profile: Profile, case: BiasCase) -> CaseResult:
    """Drive the simulated camera through the fast path until `case.windows` windows closed."""
    clock = VirtualClock()
    driver = sim_camera(
        clock,
        r0_m=case.r0_m,
        wind_speed_m_s=10.0,
        wind_direction_deg=45.0,
        outer_scale_m=20.0,
        seed=case.seed,
        psf_mode=case.psf_mode,
        zenith_angle_deg=ZENITH_ANGLE_DEG,
        sky_mag_arcsec2=case.sky_mag_arcsec2,
        twilight=False,
    )
    driver.open()
    stars = driver.truth.star_positions(clock.utc_ns(), _MODE)
    brightest = int(np.argmin(stars.mag))
    half = case.roi_px // 2
    x0, y0 = int(stars.x[brightest]) - half, int(stars.y[brightest]) - half
    stream = driver.configure(
        StreamConfig(
            _MODE,
            case.exposure_us,
            0,
            roi=Roi(x0, y0, case.roi_px, case.roi_px),
            pixel_format=PixelFormat.RAW16,
        )
    )
    driver.start()
    analyzers = {
        "b": _analyzer(profile, case, "aperture"),
        "c": _analyzer(profile, case, "gaussian"),
    }
    closed: dict[str, list[SeeingWindowRecord]] = {"b": [], "c": []}
    rows: dict[str, list[npt.NDArray[np.void]]] = {"b": [], "c": []}
    for analyzer in analyzers.values():
        analyzer.begin_stream(stream)
    while min(len(windows) for windows in closed.values()) < case.windows:
        frame = driver.read_frame(1.0)
        for name, analyzer in analyzers.items():
            closed[name] += analyzer.push(frame).windows
    for name, analyzer in analyzers.items():
        drained = analyzer.drain_metrics()
        assert drained is not None
        rows[name].append(drained)
    metrics = {name: np.concatenate(parts) for name, parts in rows.items()}
    truth = driver.truth.frames
    true_xy = np.asarray([(t.star_x_px, t.star_y_px) for t in truth[: len(metrics["b"])]])
    windows = {name: values[: case.windows] for name, values in closed.items()}
    scale = _plate_scale(profile)
    residual = {name: _residuals(metrics[name], true_xy, windows[name]) for name in ("b", "c")}
    methods = {
        "b": _from_records(windows["b"], residual["b"], scale),
        "c": _from_records(windows["c"], residual["c"], scale),
    }
    methods["a"] = _old_model(analyzers["b"], methods["b"], windows["b"], metrics["b"], case)
    return CaseResult(
        case=case,
        truth_r0_cm=driver.truth.r0_zenith_m() * 100.0,
        star_snr=tuple(w.star_snr for w in windows["b"]),
        background_fraction=tuple(w.background_fraction for w in windows["b"]),
        methods={name: methods[name] for name in METHODS},
    )


def _in_window(times: npt.NDArray[np.int64], window: SeeingWindowRecord) -> npt.NDArray[np.bool_]:
    end = window.t_utc_ns + round(window.duration_s * 1e9)
    return np.asarray((times >= window.t_utc_ns) & (times < end), dtype=np.bool_)


def _residuals(
    metrics: npt.NDArray[np.void], true_xy: FloatArray, windows: list[SeeingWindowRecord]
) -> tuple[float, ...]:
    """The variance per axis of the centroids around a quadratic plus a multiple of the truth."""
    times = metrics["t_utc_ns"].astype(np.int64)
    usable = (metrics["flags"].astype(np.int64) & _UNUSABLE) == 0
    measured = np.stack(
        [metrics["cx_px"].astype(np.float64), metrics["cy_px"].astype(np.float64)], axis=1
    )
    out = []
    for window in windows:
        inside = _in_window(times, window) & usable & np.isfinite(measured).all(axis=1)
        t = (times[inside] - window.t_utc_ns) * 1e-9
        variances = []
        for axis in (0, 1):
            design = np.stack([np.ones_like(t), t, t * t, true_xy[inside, axis]], axis=1)
            values = measured[inside, axis]
            coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
            variances.append(float(np.var(values - design @ coefficients)))
        out.append(0.5 * (variances[0] + variances[1]))
    return tuple(out)


def _from_records(
    windows: list[SeeingWindowRecord], residual: tuple[float, ...], scale: float
) -> MethodResult:
    noise = tuple((w.centroid_noise_px or 0.0) ** 2 for w in windows)
    motion = tuple(
        0.5
        * ((w.image_motion_rms_x_arcsec or 0.0) ** 2 + (w.image_motion_rms_y_arcsec or 0.0) ** 2)
        / scale**2
        for w in windows
    )
    flags = tuple(tuple(w.flags) for w in windows)
    return MethodResult(tuple(w.r0_cm for w in windows), noise, motion, residual, flags)


def _old_model(
    analyzer: FastPathAnalyzer,
    b: MethodResult,
    windows: list[SeeingWindowRecord],
    metrics: npt.NDArray[np.void],
    case: BiasCase,
) -> MethodResult:
    """Method a: the windows of b with the noise model of revision `fast-1` subtracted instead."""
    kernel, calibration = analyzer.kernel_setup(_MODE, 0, case.exposure_us, 12, 16)
    times = metrics["t_utc_ns"].astype(np.int64)
    usable = (metrics["flags"].astype(np.int64) & _UNUSABLE) == 0
    flux = metrics["flux_e"].astype(np.float64)
    width_sq = 0.5 * (
        metrics["width_x_px"].astype(np.float64) ** 2
        + metrics["width_y_px"].astype(np.float64) ** 2
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        old = width_sq / flux + calibration.pixel_var_e2 * kernel.second_moment_px4 / flux**2
    r0: list[float | None] = []
    noise: list[float] = []
    motion: list[float] = []
    for window, noise_b, motion_b, r0_b in zip(
        windows, b.noise_px2, b.motion_px2, b.r0_cm, strict=True
    ):
        inside = _in_window(times, window) & usable & np.isfinite(old)
        noise_a = float(np.mean(old[inside]))
        motion_a = motion_b + noise_b - noise_a
        noise.append(noise_a)
        motion.append(motion_a)
        valid = r0_b is not None and motion_b > 0.0 and motion_a > 0.0
        r0.append(r0_b * (motion_a / motion_b) ** -0.6 if valid and r0_b is not None else None)
    return MethodResult(tuple(r0), tuple(noise), tuple(motion), b.residual_px2)


def exposure_for_sky(profile: Profile, sky_mag_arcsec2: float) -> int:
    """The exposure of the adaptive loop for a sky, in whole microseconds: the one that puts the
    background at 0.3 of the full well, at most 2 ms and at least the profile's shortest."""
    from seeingmon.drivers.sim.detection import DetectionModel
    from seeingmon.drivers.sim.params import SimParams

    model = DetectionModel.for_simulator(
        SimParams.from_profile(profile, _MODE),
        min_exposure_us=float(profile.limits.exposure_us_range[0]),
    )
    return round(model.at_sky(sky_mag_arcsec2).exposure_us)
