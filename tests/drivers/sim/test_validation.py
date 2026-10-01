"""The whole chain: turbulence, wave optics, pixels, noise, and a plain centroid.

Each test reads bin1 frames of Polaris from the simulator, takes the plain centroid of a 41 x 41
window around the star, and subtracts the noise variance of the centroid. The window holds the
whole star: its truncation of the diffraction wings lowers the gain to 0.998 (variance 0.4%
low). The tests use a slow wind (2 m/s, so a 3 ms exposure averages away 0.3% of the variance)
and one frame in each 30 s of simulated time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.clock import VirtualClock
from seeingmon.drivers.sim import ScintillationConfig, polaris_field, sim_camera
from seeingmon.frames import Roi, StreamConfig

FloatArray = npt.NDArray[np.float64]

WINDOW = 41
EXPOSURE_US = 3000
WIND_M_S = 2.0
ROI = Roi(4080, 2758, 128, 128)
PLATE_SCALE = 1.910  # arcsec per pixel in bin1
APERTURE_M = 0.05

# Systematic effects that the test expects, as factors on the variance.
WINDOW_TRUNCATION = 0.996  # gain 0.998 of a 41 x 41 window, squared
EXPOSURE_AVERAGING = 0.997  # a 3 ms exposure at 2 m/s: vT/D = 0.12


@dataclass(frozen=True, slots=True)
class Measurements:
    """Centroids of many frames, in pixels, with their truth and their noise variance."""

    centroid: (
        FloatArray  # window centroid minus the catalog sub-pixel position, x and y alternating
    )
    truth: FloatArray  # the true tilt in pixels, in the same order
    noise: FloatArray  # the noise variance of each centroid, in pixel^2


def measure(r0_m: float, seeds: range, frames_per_seed: int) -> Measurements:
    """Render frames with a known `r0` and measure the centroid of each."""
    centroids: list[float] = []
    truths: list[float] = []
    noises: list[float] = []
    for seed in seeds:
        clock = VirtualClock()
        camera = sim_camera(
            clock,
            r0_m=r0_m,
            wind_speed_m_s=WIND_M_S,
            outer_scale_m=math.inf,
            seed=seed,
            screen_points=128,
            stars=polaris_field(include_polaris_b=False),
            scintillation=ScintillationConfig(enabled=False),
            twilight=False,
        )
        camera.open()
        camera.configure(StreamConfig("bin1", EXPOSURE_US, 0, roi=ROI, offset=30))
        sensor = camera.modes["bin1"].sensor_at(0)
        half = WINDOW // 2
        for _ in range(frames_per_seed):
            camera.stop()
            clock.advance(30.0)
            camera.start()
            frame = camera.read_frame(1.0)
            truth = camera.truth.frames[-1]
            centre_x = round(truth.catalog_x_px) - ROI.x
            centre_y = round(truth.catalog_y_px) - ROI.y
            adu = frame.data.astype(np.float64) / 16.0 - 30.0  # 12 bit, offset 30
            window = adu[
                centre_y - half : centre_y + half + 1, centre_x - half : centre_x + half + 1
            ]
            electrons = window * sensor.e_per_adu
            rows, columns = np.indices(window.shape)
            flux = float(electrons.sum())
            cx = float((columns * electrons).sum()) / flux
            cy = float((rows * electrons).sum()) / flux
            variance = (
                np.maximum(electrons, 0.0) + sensor.read_noise_e**2 + sensor.e_per_adu**2 / 12.0
            )
            noises += [
                float(((columns - cx) ** 2 * variance).sum()) / flux**2,
                float(((rows - cy) ** 2 * variance).sum()) / flux**2,
            ]
            # The window centre is a whole pixel. The true star position has a fraction of a pixel.
            fraction_x = truth.catalog_x_px - ROI.x - centre_x
            fraction_y = truth.catalog_y_px - ROI.y - centre_y
            centroids += [cx - half - fraction_x, cy - half - fraction_y]
            truths += [
                (truth.star_x_px - truth.catalog_x_px),
                (truth.star_y_px - truth.catalog_y_px),
            ]
    return Measurements(np.asarray(centroids), np.asarray(truths), np.asarray(noises))


def theory_variance_px2(r0_m: float) -> float:
    """The one-axis variance in pixels^2: 0.170 lambda^2 D^(-1/3) r0^(-5/3), converted to pixels."""
    variance_rad2 = 0.170 * (500e-9) ** 2 * math.pow(APERTURE_M, -1 / 3) * math.pow(r0_m, -5 / 3)
    return variance_rad2 / (PLATE_SCALE / 206_264.806) ** 2


def test_the_rendered_centroid_follows_the_true_tilt() -> None:
    """Paired comparison: the centroid equals the true tilt of the same frame.

    The truth is the exposure-averaged G-tilt that the turbulence model produced for the frame.
    The residual is photon and read noise, which the test predicts from the pixels. The gain
    error is the noise over the signal spread over the square root of the sample count.
    """
    data = measure(0.08, range(200), 2)  # 800 samples
    truth = data.truth - data.truth.mean()
    centroid = data.centroid - data.centroid.mean()
    gain = float((centroid * truth).sum() / (truth * truth).sum())
    residual = centroid - gain * truth
    gain_error = float(math.sqrt(data.noise.mean()) / (truth.std() * math.sqrt(len(truth))))
    assert gain_error < 0.01
    # The window truncation gives 0.998. The tolerance is three standard errors.
    assert gain == pytest.approx(0.998, abs=3 * gain_error + 0.002), (gain, gain_error)
    # What the truth does not explain is the noise that the test predicts from the pixels.
    assert float(residual.var()) == pytest.approx(float(data.noise.mean()), rel=0.15)
    assert truth.std() > 0.2  # the star really moves: about 0.3 pixel rms


def test_the_rendered_centroid_variance_matches_theory() -> None:
    """The variance of the rendered centroid matches 0.170 lambda^2 D^(-1/3) r0^(-5/3).

    600 frames with two axes give 1,200 samples, so the standard error is about 4%. That cannot
    show 5%, so this fast test only catches gross errors, at three standard errors plus 1%. It
    runs for `r0` of 10 cm. The slow test below covers 5, 10, and 15 cm with 12,000 samples each,
    which gives a standard error of 1.3%.
    """
    r0 = 0.10
    data = measure(r0, range(300), 2)
    signal = float(data.centroid.var(ddof=1) - data.noise.mean())
    expected = theory_variance_px2(r0) * WINDOW_TRUNCATION * EXPOSURE_AVERAGING
    ratio = signal / expected
    standard_error = math.sqrt(2.0 / len(data.centroid))
    assert standard_error < 0.05
    assert abs(ratio - 1.0) < 3 * standard_error + 0.01, (ratio, standard_error)


@pytest.mark.slow
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("r0", [0.05, 0.10, 0.15])
def test_the_rendered_centroid_variance_matches_theory_within_5_percent(r0: float) -> None:
    """The done criterion with 12,000 samples per `r0`: the standard error is about 1.3%."""
    data = measure(r0, range(1000, 4000), 2 * 2)
    signal = float(data.centroid.var(ddof=1) - data.noise.mean())
    expected = theory_variance_px2(r0) * WINDOW_TRUNCATION * EXPOSURE_AVERAGING
    ratio = signal / expected
    # Frames of one seed share the low-frequency sinusoids, so count independent seeds. With
    # 3,000 seeds of four frames, the number of independent samples lies between 6,000 and 24,000.
    standard_error = math.sqrt(2.0 / (len(data.centroid) / 2))
    assert standard_error < 0.025
    assert ratio == pytest.approx(1.0, abs=0.05), (ratio, standard_error)
