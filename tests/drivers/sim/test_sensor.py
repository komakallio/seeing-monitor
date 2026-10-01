"""The sensor and the photometry: saturation, dark current, hot pixels, sky, and clouds."""

from __future__ import annotations

import math

import numpy as np
import pytest

from seeingmon.clock import DEFAULT_START_UTC_NS, NS_PER_S, VirtualClock
from seeingmon.drivers.sim import (
    CloudEvent,
    Clouds,
    HotPixelConfig,
    Pointing,
    ScintillationConfig,
    SimDriver,
    StarField,
    polaris_field,
    sim_camera,
)
from seeingmon.drivers.sim.detector import Detector, HotPixelMap
from seeingmon.drivers.sim.params import SimParams
from seeingmon.frames import PixelFormat, Roi, StreamConfig, StreamKind

BIN1 = SimParams.reference("bin1")
BIN2 = SimParams.reference("bin2")
FULL_WELL_BIN1 = 14_417.0
POLARIS_ROI = Roi(4080, 2758, 128, 128)
NO_STARS = StarField.from_arrays([], [], [])
# Half a pixel in bin1: Polaris then sits on the centre of pixel (4144, 2822) at the start.
ON_A_PIXEL = Pointing(offset_arcsec=(0.955, 0.955))


def quiet_polaris(**kwargs: object) -> SimDriver:
    """A camera that sees only Polaris, with no scintillation, for photometry."""
    defaults: dict[str, object] = {
        "psf_mode": "wave",
        "screen_points": 128,
        "stars": polaris_field(include_polaris_b=False),
        "scintillation": ScintillationConfig(enabled=False),
        "twilight": False,
        "pointing": ON_A_PIXEL,
    }
    defaults.update(kwargs)
    return sim_camera(VirtualClock(), **defaults)  # type: ignore[arg-type]


def peak_electrons(exposure_us: int, seed: int = 1) -> tuple[float, int]:
    """The peak pixel of Polaris at gain 0, in electrons, and the raw ADC value of the peak."""
    camera = quiet_polaris(seed=seed)
    camera.open()
    camera.configure(StreamConfig("bin1", exposure_us, 0, roi=POLARIS_ROI, offset=30))
    camera.start()
    adc = camera.read_frame(1.0).data.astype(np.int64) >> 4
    sensor = BIN1.sensor_at(0)
    return (float(adc.max()) - 30.0) * sensor.e_per_adu, int(adc.max())


def test_polaris_reaches_70_percent_of_the_full_well_in_a_few_milliseconds() -> None:
    """Research notes: the peak pixel reaches 70% of the full well after about 3 ms in bin1.

    The notes estimate 87,000 electrons in 10 ms (good to 30%). The simulator uses the profile
    rate of 4.6e7 electrons per second for magnitude 0, which gives 72,000 in 10 ms, so the
    crossing falls a little later, near 3.8 ms.
    """
    times_ms = np.arange(1.0, 6.01, 0.5)
    peaks = np.array([peak_electrons(round(t * 1000))[0] for t in times_ms])
    fraction = peaks / FULL_WELL_BIN1
    assert fraction[np.searchsorted(times_ms, 3.0)] == pytest.approx(0.55, abs=0.08)
    crossing = float(np.interp(0.7, fraction, times_ms))
    assert 2.5 < crossing < 4.5, crossing
    # The peak grows in proportion to the exposure until the ADC clips it.
    assert peaks[2] / peaks[0] == pytest.approx(2.0, rel=0.1)


def test_polaris_saturates_at_10_ms() -> None:
    peak, adc = peak_electrons(10_000)
    assert adc == 4095  # the 12-bit ADC clips before the 14,417 electron well fills
    assert peak == pytest.approx(4065 * 3.5, rel=0.001)
    # A 2 ms frame keeps the star well inside the range.
    peak_fast, adc_fast = peak_electrons(2000)
    assert adc_fast < 2500
    assert peak_fast == pytest.approx(0.37 * 7.2e6 * 0.002, rel=0.12)


def test_the_star_flux_follows_the_magnitude_and_the_aperture() -> None:
    """Polaris (V = 2.02) gives about 7.2 million electrons per second in the 50 mm aperture."""
    camera = quiet_polaris(seed=3)
    camera.open()
    camera.configure(StreamConfig("bin1", 2000, 0, roi=POLARIS_ROI, offset=30))
    camera.start()
    camera.read_frame(1.0)
    expected = 4.6e7 * 10 ** (-0.4 * 2.02) * 0.002
    assert camera.truth.frames[-1].flux_e == pytest.approx(expected, rel=1e-6)
    # The summed window flux is the star's flux (the wings outside the window are 0.2%).
    camera.stop()
    camera.start()
    frame = camera.read_frame(1.0)
    electrons = (frame.data.astype(np.float64) / 16.0 - 30.0) * 3.5
    assert float(electrons.sum()) == pytest.approx(expected, rel=0.03)


def test_dark_current_follows_the_temperature() -> None:
    """A dark bin1 frame holds the dark current of the profile table, in electrons."""
    for ambient in (15.0, -5.0):
        camera = sim_camera(
            VirtualClock(),
            psf_mode="gaussian",
            screen_points=128,
            stars=NO_STARS,
            twilight=False,
            sky_mag_arcsec2=40.0,
            ambient_c=ambient,
        )
        camera.open()
        camera.configure(
            StreamConfig(
                "bin1", 30_000_000, 0, kind=StreamKind.SNAPSHOT, roi=Roi(0, 0, 256, 256), offset=30
            )
        )
        camera.start()
        frame = camera.read_frame(60.0)
        temperature = frame.temperature_c
        assert temperature == pytest.approx(ambient + 4.0)
        expected = BIN1.dark_rate_e_per_s(float(temperature or 0.0)) * 30.0
        mean_adu = float((frame.data.astype(np.float64) / 16.0).mean()) - 30.0
        assert mean_adu * 3.5 == pytest.approx(expected, abs=0.35), ambient
    # A warmer sensor makes more dark current: about 0.18 electrons per second at 19 degrees.
    assert BIN1.dark_rate_e_per_s(19.0) == pytest.approx(0.179, abs=0.005)
    assert BIN1.dark_rate_e_per_s(13.0) == pytest.approx(BIN1.dark_rate_e_per_s(19.0) / 2, rel=0.05)


def test_hot_pixels_are_a_fixed_map() -> None:
    config = HotPixelConfig(density_per_mpix=200.0, median_rate_e_per_s=20.0, spread=0.5)  # e-/s
    roi = Roi(0, 0, 1024, 1024)
    hot = HotPixelMap(BIN1, config, seed=5)
    assert len(hot) == round(200.0 * BIN1.width * BIN1.height / 1e6)
    again = HotPixelMap(BIN1, config, seed=5)
    assert np.array_equal(hot.x, again.x)
    assert not np.array_equal(hot.x, HotPixelMap(BIN1, config, seed=6).x)
    image = np.zeros((1024, 1024), dtype=np.float32)
    hot.add_to(image, roi, exposure_s=10.0, temperature_c=20.0)
    inside = (hot.x < 1024) & (hot.y < 1024)
    assert int((image > 0).sum()) == int(inside.sum())
    assert float(image.max()) > 100  # a hot pixel stands far above the 2 electron dark current
    colder = np.zeros_like(image)
    hot.add_to(colder, roi, exposure_s=10.0, temperature_c=8.0)
    assert float(colder.sum()) < 0.3 * float(image.sum())
    camera = quiet_polaris(seed=2, hot_pixels=config, stars=NO_STARS)
    camera.open()
    camera.configure(
        StreamConfig("bin1", 10_000_000, 0, kind=StreamKind.SNAPSHOT, roi=Roi(0, 0, 1024, 1024))
    )
    camera.start()
    frame = camera.read_frame(20.0)
    adu = frame.data.astype(np.float64) / 16.0
    # In 10 s, the median hot pixel collects 180 electrons, which is 50 ADU.
    assert int((adu > 30 + 10).sum()) > 0.9 * int(inside.sum())


def test_sky_background_follows_the_sky_brightness() -> None:
    camera = sim_camera(
        VirtualClock(),
        psf_mode="gaussian",
        screen_points=128,
        stars=NO_STARS,
        twilight=False,
        sky_mag_arcsec2=18.0,
        ambient_c=-30.0,  # a cold sensor has no dark current to confuse the sky
    )
    camera.open()
    camera.configure(
        StreamConfig(
            "bin2", 5_000_000, 120, kind=StreamKind.SNAPSHOT, roi=Roi(0, 0, 256, 256), offset=30
        )
    )
    camera.start()
    frame = camera.read_frame(30.0)
    sensor = BIN2.sensor_at(120)
    sky_e = BIN2.sky_rate_e_per_s_px(18.0) * 5.0
    adu = (frame.data.astype(np.float64) / 4.0).mean() - 30.0 * 4  # 14-bit ADC, offset x 4
    assert adu * sensor.e_per_adu == pytest.approx(sky_e, rel=0.02)
    assert sky_e > 100  # a bright sky: 18 mag/arcsec2 in bin2 gives 200 electrons in 5 s


def test_twilight_raises_the_background_over_virtual_time() -> None:
    clock = VirtualClock()
    camera = sim_camera(
        clock,
        psf_mode="gaussian",
        screen_points=128,
        stars=NO_STARS,
        twilight=True,
        sky_mag_arcsec2=20.5,
        ambient_c=-30.0,
    )
    camera.open()
    config = StreamConfig("bin2", 1_000_000, 120, roi=Roi(0, 0, 64, 64), offset=30)
    camera.configure(config)
    levels = []
    sky_mags = []
    for hours in (0.0, 5.0, 6.0, 7.0):
        clock.advance_to_utc_ns(DEFAULT_START_UTC_NS + round(hours * 3600 * NS_PER_S))
        camera.start()
        frame = camera.read_frame(5.0)
        levels.append(float(frame.data.mean()) / 4.0 - 120.0)
        sky_mags.append(camera.truth.frames[-1].sky_mag_arcsec2)
        camera.stop()
    assert sky_mags[0] == pytest.approx(20.5)  # deep night on 1 January at 00:00 UTC
    assert sky_mags[-1] < sky_mags[1] < sky_mags[0] + 1e-9  # the sky brightens toward dawn
    assert levels[-1] > levels[0] + 5


def test_clouds_dim_the_star_in_proportion() -> None:
    start = DEFAULT_START_UTC_NS + 10 * NS_PER_S
    clouds = Clouds(events=(CloudEvent(start, 20.0, 0.5, ramp_s=2.0),))
    clock = VirtualClock()
    camera = sim_camera(
        clock,
        psf_mode="wave",
        screen_points=128,
        stars=polaris_field(include_polaris_b=False),
        scintillation=ScintillationConfig(enabled=False),
        twilight=False,
        clouds=clouds,
        pointing=ON_A_PIXEL,
    )
    camera.open()
    camera.configure(StreamConfig("bin1", 2000, 0, roi=POLARIS_ROI, offset=30))
    fluxes = []
    for offset_s in (0.0, 15.0, 40.0):
        clock.advance_to_utc_ns(DEFAULT_START_UTC_NS + round(offset_s * NS_PER_S))
        camera.start()
        frame = camera.read_frame(1.0)
        camera.stop()
        truth = camera.truth.frames[-1]
        centre_x = round(truth.catalog_x_px) - POLARIS_ROI.x
        centre_y = round(truth.catalog_y_px) - POLARIS_ROI.y
        window = frame.data[centre_y - 20 : centre_y + 21, centre_x - 20 : centre_x + 21]
        electrons = float(((window.astype(np.float64) / 16.0 - 30.0) * 3.5).sum())
        fluxes.append((electrons, truth.transparency))
    assert fluxes[0][1] == 1.0
    assert fluxes[1][1] == pytest.approx(0.5)
    assert fluxes[2][1] == 1.0
    # The window holds 99.6% of the flux, and the noise on its sum is about 2%.
    assert fluxes[1][0] / fluxes[0][0] == pytest.approx(0.5, rel=0.08)
    assert fluxes[2][0] / fluxes[0][0] == pytest.approx(1.0, rel=0.08)


def test_scintillation_changes_the_flux_by_the_expected_rms() -> None:
    camera = sim_camera(
        VirtualClock(),
        psf_mode="gaussian",
        screen_points=128,
        stars=polaris_field(include_polaris_b=False),
        twilight=False,
        seed=7,
    )
    camera.open()
    camera.configure(StreamConfig("bin1", 2000, 0, roi=POLARIS_ROI, offset=30))
    camera.start()
    for _ in range(600):
        camera.read_frame(1.0)
    factors = np.array([f.scintillation_factor for f in camera.truth.frames])
    expected = camera.truth.scintillation_rms(0.002)
    assert factors.mean() == pytest.approx(1.0, abs=0.06)
    assert factors.std() == pytest.approx(expected, rel=0.2)
    # Frames 11 ms apart are nearly independent for a 3 ms correlation time.
    centred = factors - factors.mean()
    assert abs(float(np.mean(centred[:-1] * centred[1:]) / np.mean(centred**2))) < 0.2


def test_detector_handles_the_formats_and_a_dark_input() -> None:
    detector = Detector(BIN2)
    rng = np.random.default_rng(0)
    mean = np.full((64, 64), 5000.0, dtype=np.float32)
    wide = detector.digitize(mean, gain=0, offset=30, pixel_format=PixelFormat.RAW16, rng=rng)
    assert wide.dtype == np.uint16
    assert np.all(wide % 4 == 0)  # 14 bit: the low 2 bits are zero
    assert float((wide.astype(np.float64) / 4).mean()) == pytest.approx(
        5000 / 4.05 + 30 * 4, rel=0.005
    )
    narrow = detector.digitize(
        np.full((8, 8), 1e9, dtype=np.float32),
        gain=0,
        offset=None,
        pixel_format=PixelFormat.RAW8,
        rng=rng,
    )
    assert int(narrow.max()) == 255  # saturated at the top 8 bits of the 14-bit range
    assert detector.saturation_electrons(0) == pytest.approx(16_383 * 4.05, rel=1e-3)
    assert detector.black_level_adu(30) == 120.0
    assert detector.noise_floor_adu(0) == pytest.approx(8.0 / 4.05)
    assert math.isfinite(detector.expected_adu(1000.0, 0, None))
