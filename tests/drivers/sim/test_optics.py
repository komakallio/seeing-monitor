"""The star renderers: wave optics against the Airy pattern, and the Gaussian mixture."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.drivers.sim import _scipy as sp
from seeingmon.drivers.sim.optics import (
    MixturePsf,
    PsfConfig,
    WavePsf,
    stamp_size_px,
    strehl_ratio,
    wavelengths_of,
)
from seeingmon.drivers.sim.params import SimParams
from seeingmon.drivers.sim.turbulence import Layer, TurbulenceConfig, TurbulenceModel

FloatArray = npt.NDArray[np.float64]

BIN1 = SimParams.reference("bin1")
BIN2 = SimParams.reference("bin2")


def quiet_model(params: SimParams) -> TurbulenceModel:
    """Turbulence so weak that the image is diffraction limited and sits still."""
    config = TurbulenceConfig(r0_m=1e4, layers=(Layer(1.0, 10.0, 0.0),), screen_points=128)
    return TurbulenceModel(config, params.aperture_m)


def airy_stamp(params: SimParams, size: int, offset: tuple[float, float]) -> FloatArray:
    """The exact box-integrated Airy pattern, normalized to the flux inside the stamp."""
    oversample = 32
    sub = (np.arange(oversample) + 0.5) / oversample - 0.5
    pixels = ((np.arange(size) - size // 2)[:, None] + sub[None, :]).reshape(-1)
    xx, yy = np.meshgrid(
        (pixels - offset[0]) * params.pixel_rad, (pixels - offset[1]) * params.pixel_rad
    )
    x = math.pi * params.aperture_m * np.hypot(xx, yy) / params.wavelength_m
    safe = np.where(x > 1e-12, x, 1.0)
    intensity = np.where(x > 1e-12, (2.0 * sp.j1(safe) / safe) ** 2, 1.0)
    stamp = intensity.reshape(size, oversample, size, oversample).sum(axis=(1, 3))
    return np.asarray(stamp / stamp.sum(), dtype=np.float64)


def centroid(stamp: npt.NDArray[np.float32]) -> tuple[float, float]:
    """The plain centroid relative to the centre pixel, as `(x, y)`."""
    rows, columns = np.indices(stamp.shape)
    total = float(stamp.sum())
    centre = stamp.shape[0] // 2
    return (
        float((columns * stamp).sum()) / total - centre,
        float((rows * stamp).sum()) / total - centre,
    )


# --- wave optics --------------------------------------------------------------------------


@pytest.mark.parametrize("params", [BIN1, BIN2], ids=["bin1", "bin2"])
def test_wave_optics_without_turbulence_is_an_airy_pattern(params: SimParams) -> None:
    wave = WavePsf(params, PsfConfig(), quiet_model(params))
    assert wave.pupil_points >= 32  # the brief asks for at least 32 x 32 points
    for offset in ((0.0, 0.0), (0.3, -0.2), (0.5, 0.5)):
        result = wave.render(0.0, 0.002, offset)
        stamp = result.stamp
        assert stamp.shape == (wave.fov_px, wave.fov_px)
        assert float(stamp.sum()) == pytest.approx(1.0, abs=1e-5)
        reference = airy_stamp(params, wave.fov_px, offset)
        # Pixel integration is accurate to 0.5% of the peak. Folded wings add about 1%.
        assert np.abs(stamp - reference).max() < 0.02 * reference.max()
        # The plain centroid of an undersampled image has a pixel-phase bias (bin2 has a gain
        # of 0.7 here), so compare it with the centroid of the exact image.
        assert centroid(stamp) == pytest.approx(centroid(reference.astype(np.float32)), abs=0.01)
        if params is BIN1:  # in bin1 the pixel is smaller than lambda / D, and the bias vanishes
            assert centroid(stamp) == pytest.approx(offset, abs=0.004)
        assert result.tilt_x_rad == pytest.approx(0.0, abs=1e-8)


def test_the_stamp_covers_the_expected_field() -> None:
    assert WavePsf(BIN1, PsfConfig(), quiet_model(BIN1)).fov_px == 64
    assert WavePsf(BIN2, PsfConfig(), quiet_model(BIN2)).fov_px == 32
    # A narrow field widens until the pupil has 32 points across.
    assert stamp_size_px(BIN1, PsfConfig(fov_arcsec=20.0)) >= 32 * BIN1.wavelength_m / (
        BIN1.aperture_m * BIN1.pixel_rad
    )


def test_peak_pixel_matches_the_airy_pattern_in_bin1() -> None:
    """For a sharp star on a pixel centre, the peak pixel holds about 37% of the flux."""
    wave = WavePsf(BIN1, PsfConfig(), quiet_model(BIN1))
    peak = float(wave.render(0.0, 0.002).stamp.max())
    assert peak == pytest.approx(0.37, abs=0.01)


def test_the_wave_centroid_follows_the_true_tilt() -> None:
    """The centroid of a rendered stamp equals the exposure-averaged G-tilt within 0.5%."""
    base = TurbulenceConfig(
        r0_m=0.08,
        layers=(Layer(1.0, 10.0, 30.0),),
        outer_scale_m=math.inf,
        screen_points=128,
    )
    centroids: list[float] = []
    truths: list[float] = []
    for seed in range(40):
        model = TurbulenceModel(replace(base, seed=seed), BIN1.aperture_m)
        result = WavePsf(BIN1, PsfConfig(), model).render(0.5, 0.002)
        measured_x, measured_y = centroid(result.stamp)
        centroids += [measured_x, measured_y]
        truths += [result.tilt_x_rad / BIN1.pixel_rad, result.tilt_y_rad / BIN1.pixel_rad]
    measured_array = np.asarray(centroids)
    truth_array = np.asarray(truths)
    gain = float((measured_array * truth_array).sum() / (truth_array**2).sum())
    residual = measured_array - gain * truth_array
    assert truth_array.std() > 0.15  # the test sees real motion: 0.25 to 0.3 pixel
    # Folded diffraction wings lower the gain of a 64-pixel stamp by about 0.2%.
    assert gain == pytest.approx(1.0, abs=0.005)
    assert residual.std() < 0.005


def test_the_image_is_the_mean_over_the_exposure() -> None:
    """With a fast wind, a long exposure smears the image and lowers the peak.

    One short exposure can fall on a poor instant, so the test compares the mean peak of 12
    seeds, in strong turbulence (`r0` of 2 cm). The means are 0.20 for 0.5 ms and 0.12 for 20 ms,
    a ratio of 0.57 with a standard error of about 0.07.
    """
    short_peaks: list[float] = []
    long_peaks: list[float] = []
    for seed in range(12):
        config = TurbulenceConfig(
            r0_m=0.02,
            layers=(Layer(1.0, 25.0, 0.0),),
            outer_scale_m=20.0,
            screen_points=128,
            seed=seed,
        )
        wave = WavePsf(BIN1, PsfConfig(), TurbulenceModel(config, BIN1.aperture_m))
        short = wave.render(1.0, 0.0005, n_sub=4).stamp
        long = wave.render(1.0, 0.02, n_sub=16).stamp
        assert float(long.sum()) == pytest.approx(1.0, abs=1e-5)
        short_peaks.append(float(short.max()))
        long_peaks.append(float(long.max()))
    assert np.mean(long_peaks) < 0.8 * np.mean(short_peaks)


def test_a_bandwidth_adds_two_wavelengths() -> None:
    config = PsfConfig(bandwidth_fraction=0.2)
    waves, weights = wavelengths_of(BIN1, config)
    assert len(waves) == 3
    assert float(weights.sum()) == pytest.approx(1.0)
    assert waves[2] / waves[0] == pytest.approx(1.1 / 0.9)
    wave = WavePsf(BIN1, config, quiet_model(BIN1))
    stamp = wave.render(0.0, 0.002, (0.2, 0.0)).stamp
    assert float(stamp.sum()) == pytest.approx(1.0, abs=1e-5)
    assert centroid(stamp)[0] == pytest.approx(0.2, abs=0.01)
    mono = WavePsf(BIN1, PsfConfig(), quiet_model(BIN1)).render(0.0, 0.002, (0.2, 0.0)).stamp
    assert not np.allclose(stamp, mono, atol=1e-4)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="mode"):
        PsfConfig(mode="fft")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="oversample"):
        PsfConfig(oversample=1)
    with pytest.raises(ValueError, match="bandwidth"):
        PsfConfig(bandwidth_fraction=1.5)
    with pytest.raises(ValueError, match="wave"):
        WavePsf(BIN1, PsfConfig(mode="gaussian"), quiet_model(BIN1))


# --- the Gaussian mixture -----------------------------------------------------------------


def test_the_mixture_approximates_the_airy_pattern() -> None:
    mixture = MixturePsf(BIN1, PsfConfig(mode="gaussian"))
    for offset in ((0.0, 0.0), (0.3, -0.2), (0.5, 0.5)):
        stamp = mixture.stamp(offset, r0_500nm_m=1e4)
        reference = airy_stamp(BIN1, mixture.fov_px, offset)
        assert float(stamp.sum()) == pytest.approx(0.993, abs=0.004)
        assert np.abs(stamp / stamp.sum() - reference).max() < 0.03 * reference.max()
        assert centroid(stamp)[0] == pytest.approx(offset[0], abs=0.01)


def test_mixture_weights_and_the_seeing_halo() -> None:
    mixture = MixturePsf(BIN1, PsfConfig(mode="gaussian"))
    weights, sigmas = mixture.components(0.10)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-9)
    # A poor r0 moves flux from the core into a wide halo and lowers the peak.
    good = mixture.stamp((0.0, 0.0), r0_500nm_m=0.20)
    poor = mixture.stamp((0.0, 0.0), r0_500nm_m=0.02)
    assert float(poor.max()) < 0.8 * float(good.max())
    # Extra blur (the wander in a long exposure) widens every Gaussian.
    _, blurred = mixture.components(0.10, extra_sigma_px=2.0)
    assert np.all(blurred > sigmas)


def test_strehl_ratio_follows_noll() -> None:
    # D / r0 = 0.5 at 500 nm: exp(-0.134 * 0.5^(5/3)) = 0.96.
    assert strehl_ratio(0.05, 0.10, 500e-9) == pytest.approx(0.9587, abs=0.001)
    assert strehl_ratio(0.05, 0.10, 650e-9) > strehl_ratio(0.05, 0.10, 500e-9)


def test_mixture_stamps_in_bulk_match_single_stamps() -> None:
    mixture = MixturePsf(BIN2, PsfConfig(mode="gaussian"))
    weights, sigmas = mixture.components(0.10)
    offsets_x = np.array([0.0, 0.4, -0.3])
    offsets_y = np.array([0.1, -0.2, 0.5])
    bulk = mixture.stamps(offsets_x, offsets_y, weights, sigmas)
    for index in range(3):
        single = mixture.stamp((float(offsets_x[index]), float(offsets_y[index])), 0.10)
        assert np.allclose(bulk[index], single, atol=1e-7)
    small = mixture.stamps(offsets_x, offsets_y, weights, sigmas, size_px=9)
    assert small.shape == (3, 9, 9)
