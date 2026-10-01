"""Frames, series, and simulated turbulence for the fast-path tests.

Every function builds its data from a seed, so a test gives the same result on every run.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S
from seeingmon.frames import Frame, FrameFlag, PixelFormat, Roi, StreamConfig, TimeQuality

FloatArray = npt.NDArray[np.float64]

EPOCH_NS = 1_800_000_000 * NS_PER_S
BIN1_PERIOD_S = 0.0113  # the frame period of a 128-row ROI in bin1: 6.5 ms + 128 x 37.6 us
POLARIS_ELECTRONS_2MS = 14_000.0  # Polaris at 2 ms: 4.6e7 e-/s x 10^(-0.4 x 2.02) x 0.002 s


def box_integrated_gaussian(
    shape: tuple[int, int], x: float, y: float, sigma: float, flux: float = 1.0
) -> FloatArray:
    """A Gaussian star whose pixels hold the exact integral of the Gaussian over the pixel."""
    from seeingmon.fastpath import _scipy

    def edges(length: int, center: float) -> FloatArray:
        grid = np.arange(length + 1) - 0.5
        scaled = np.asarray((grid - center) / (sigma * math.sqrt(2.0)), dtype=np.float64)
        cdf = 0.5 * (1.0 + _scipy.erf(scaled))
        return np.asarray(np.diff(cdf), dtype=np.float64)

    height, width = shape
    return np.asarray(flux * np.outer(edges(height, y), edges(width, x)), dtype=np.float64)


def digitize(
    electrons: FloatArray,
    *,
    e_per_adu: float = 3.5,
    read_noise_e: float = 2.65,
    adc_bits: int = 12,
    container_bits: int = 16,
    offset_adu: float = 30.0,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.uint8] | npt.NDArray[np.uint16]:
    """Turn a mean electron image into camera data, as the detector of the simulator does.

    With `rng=None` the image has no noise, and its counts are rounded to the ADC step. A 16-bit
    container carries the ADC value in the high bits, and an 8-bit container carries the top
    8 bits.
    """
    if rng is not None:
        electrons = rng.poisson(np.maximum(electrons, 0.0)) + rng.normal(
            0.0, read_noise_e, electrons.shape
        )
    adu = np.clip(np.rint(electrons / e_per_adu + offset_adu), 0, (1 << adc_bits) - 1)
    counts = adu.astype(np.uint16)
    if container_bits == 16:
        return np.asarray(counts << (16 - adc_bits), dtype=np.uint16)
    return np.asarray(counts >> (adc_bits - 8), dtype=np.uint8)


def make_frame(
    data: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
    *,
    seq: int = 0,
    stream_id: int = 1,
    t_ns: int | None = None,
    period_s: float = BIN1_PERIOD_S,
    roi: Roi | None = None,
    mode: str = "bin1",
    exposure_us: int = 2000,
    gain: int = 0,
    adc_bits: int = 12,
    dropped_before: int = 0,
    flags: FrameFlag = FrameFlag.SIMULATED,
    t_quality: TimeQuality = TimeQuality.EXACT,
    temperature_c: float | None = 15.0,
) -> Frame:
    """A frame with the given pixels. The time is `seq` frame periods after the epoch."""
    height, width = data.shape
    return Frame(
        data=data,
        stream_id=stream_id,
        seq=seq,
        t_arrival_ns=EPOCH_NS + round(seq * period_s * NS_PER_S),
        t_utc_ns=EPOCH_NS + round(seq * period_s * NS_PER_S) if t_ns is None else t_ns,
        t_err_ns=1_000,
        t_quality=t_quality,
        dropped_before=dropped_before,
        exposure_us=exposure_us,
        gain=gain,
        mode=mode,
        roi=roi or Roi(1000, 2000, width, height),
        adc_bits=adc_bits,
        temperature_c=temperature_c,
        flags=flags,
    )


def stream_config(roi: Roi, **kwargs: object) -> StreamConfig:
    """A bin1 RAW16 stream configuration for a ROI."""
    return StreamConfig("bin1", 2000, 0, roi=roi, pixel_format=PixelFormat.RAW16, **kwargs)  # type: ignore[arg-type]


@cache
def simulated_tilt_arcsec(
    *,
    wind_ms: float = 10.0,
    outer_scale_m: float = 20.0,
    exposure_s: float = 0.002,
    rate_hz: float = 25.0,
    duration_s: float = 80.0,
    seed: int = 7,
    direction_deg: float = 45.0,
    substeps: int = 3,
    screen_points: int = 64,
) -> FloatArray:
    """The exposure-averaged G-tilt of one frozen layer of `r0` = 10 cm, in arcseconds.

    The array has the shape `(frames, 2)` for the sensor `x` and `y`. It comes from the
    wave-optics turbulence model of the simulator (`TurbulenceModel`), with the tilt of a 50 mm
    circular aperture averaged over `substeps` instants of each exposure. The tilt scales as
    `r0^(-5/6)`, so multiply by `(0.10 / r0)^(5/6)` for another `r0`. The result is cached.
    """
    from seeingmon.drivers.sim.params import ARCSEC_PER_RAD
    from seeingmon.drivers.sim.turbulence import (
        Layer,
        PupilGrid,
        TurbulenceConfig,
        TurbulenceModel,
    )

    config = TurbulenceConfig(
        r0_m=0.10,
        layers=(Layer(1.0, wind_ms, direction_deg),),
        outer_scale_m=outer_scale_m,
        seed=seed,
        screen_points=screen_points,
    )
    model = TurbulenceModel(config, 0.05)
    pupil = PupilGrid.circular(0.05, 0.05 / 16)
    frames = round(duration_s * rate_hz)
    starts = np.arange(frames) / rate_hz
    offsets = (np.arange(substeps) + 0.5) / substeps * exposure_s
    times = (starts[:, None] + offsets[None, :]).reshape(-1)
    tilt = model.tilt_series_rad(times, pupil).reshape(frames, substeps, 2).mean(axis=1)
    out = np.asarray(tilt * ARCSEC_PER_RAD, dtype=np.float64)
    out.flags.writeable = False
    return out


def r0_scale(r0_m: float) -> float:
    """The factor that turns the tilt of `r0` = 10 cm into the tilt of another `r0`."""
    return float((0.10 / r0_m) ** (5.0 / 6.0))


def synthetic_tilt_arcsec(
    r0_m: float,
    *,
    rate_hz: float,
    samples: int,
    exposure_s: float = 0.002,
    wind_ms: float = 10.0,
    outer_scale_m: float = 20.0,
    seed: int = 1,
) -> FloatArray:
    """A Gaussian process with exactly the variance and spectrum of the estimator's model.

    The variance is `K lambda^2 D^(-1/3) r0^(-5/3)` times the outer-scale and exposure ratios, and
    the spectrum is the temporal spectrum of the model (`models.tilt_spectrum`) with the exposure
    filter, folded into the Nyquist band by the sampling (aliasing). The generator synthesizes the
    series in the Fourier domain. The result has the shape `(samples, 2)`, one column for each
    sensor axis. The estimator should return `r0_m` from it, so use it to test the estimator's
    mechanics (units, detrending, noise, gaps), and use the simulator for the physics.
    """
    from seeingmon.fastpath import models

    aperture_m = 0.05
    spectrum = models.tilt_spectrum(aperture_m, outer_scale_m, wind_ms)
    variance = (
        models.tilt_variance_rad2(r0_m, aperture_m)
        * models.ARCSEC_PER_RAD**2
        * models.outer_scale_ratio(aperture_m, outer_scale_m)
        * spectrum.exposure_variance_ratio(exposure_s)
    )
    filtered = spectrum.weight * np.sinc(spectrum.nu_hz * exposure_s) ** 2
    filtered = filtered / filtered.sum()
    # Fold every model frequency into the band 0 to Nyquist, and bin the variance on a fine grid.
    bins_count = 1 << 17
    folded = np.abs(((spectrum.nu_hz + rate_hz / 2.0) % rate_hz) - rate_hz / 2.0)
    index = np.minimum(np.rint(folded / (rate_hz / 2.0) * bins_count).astype(int), bins_count)
    power = np.bincount(index, weights=filtered, minlength=bins_count + 1) * variance
    rng = np.random.default_rng(seed)
    length = 2 * bins_count
    columns = []
    for _ in range(2):
        a = rng.normal(size=bins_count + 1) * np.sqrt(power)
        b = rng.normal(size=bins_count + 1) * np.sqrt(power)
        b[0] = b[-1] = 0.0
        columns.append(np.fft.irfft(0.5 * length * (a - 1j * b), n=length)[:samples])
    return np.stack(columns, axis=1)
