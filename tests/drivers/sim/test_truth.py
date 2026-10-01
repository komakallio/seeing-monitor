"""The truth object: the recipe that other lanes use to test estimators against injected truth."""

from __future__ import annotations

import math

import numpy as np
import pytest

from seeingmon.clock import DEFAULT_START_UTC_NS, NS_PER_S, VirtualClock
from seeingmon.drivers.sim import (
    Layer,
    Pointing,
    ScintillationConfig,
    SimOptions,
    TurbulenceConfig,
    create,
    polaris_field,
    sim_camera,
)
from seeingmon.drivers.sim.turbulence import g_tilt_rms_arcsec, seeing_fwhm_arcsec
from seeingmon.frames import Roi, StreamConfig

ON_A_PIXEL = Pointing(offset_arcsec=(0.955, 0.955))
ROI = Roi(4080, 2758, 128, 128)
PLATE_SCALE = 1.910


def test_the_recipe_for_a_camera_with_a_known_r0() -> None:
    """The recipe from the module documentation: build a camera, read frames, read the truth."""
    clock = VirtualClock()
    driver = sim_camera(clock, r0_m=0.10, wind_speed_m_s=8.0, outer_scale_m=20.0, seed=3)
    driver.open()
    driver.configure(StreamConfig("bin1", 2000, 0, roi=ROI))
    driver.start()
    frames = [driver.read_frame(timeout_s=1.0) for _ in range(8)]
    truth = driver.truth
    # What the atmosphere is.
    assert truth.r0_zenith_m() == pytest.approx(0.10)
    assert truth.outer_scale_m == 20.0
    assert [(layer.wind_speed_m_s, layer.cn2_fraction) for layer in truth.layers] == [(8.0, 1.0)]
    assert truth.seeing_fwhm_arcsec() == pytest.approx(seeing_fwhm_arcsec(0.10, outer_scale_m=20.0))
    assert truth.kolmogorov_seeing_fwhm_arcsec() == pytest.approx(1.011, abs=0.002)
    assert truth.kolmogorov_image_motion_rms_arcsec() == pytest.approx(0.48, abs=0.01)
    assert truth.image_motion_rms_arcsec() == pytest.approx(0.48 * math.sqrt(0.793), abs=0.01)
    # What happened to each frame you read.
    assert len(truth.frames) == len(frames)
    for frame, record in zip(frames, truth.frames, strict=True):
        assert (record.seq, record.stream_id, record.t_utc_ns) == (
            frame.seq,
            frame.stream_id,
            frame.t_utc_ns,
        )
        assert record.star_index >= 0
    arrays = truth.frame_arrays()
    assert arrays["tilt_x_arcsec"].shape == (8,)
    assert set(arrays) >= {"tilt_x_arcsec", "tilt_y_arcsec", "t_ref_utc_ns", "star_x_px", "flux_e"}


def test_the_truth_tilt_of_a_frame_is_a_pure_function_of_its_time() -> None:
    driver = sim_camera(VirtualClock(), r0_m=0.07, wind_speed_m_s=12.0, seed=9, screen_points=128)
    driver.open()
    driver.configure(StreamConfig("bin1", 2000, 0, roi=ROI))
    driver.start()
    for _ in range(5):
        driver.read_frame(1.0)
    for record in driver.truth.frames:
        start = record.t_ref_utc_ns - round(record.exposure_s * 0.5 * NS_PER_S)
        tilt = driver.truth.tilt_arcsec(start, record.exposure_s)
        assert tilt == pytest.approx((record.tilt_x_arcsec, record.tilt_y_arcsec), abs=1e-9)
    series = driver.truth.tilt_series_arcsec(
        [r.t_ref_utc_ns - 1_000_000 for r in driver.truth.frames], 0.002
    )
    assert series.shape == (5, 2)
    assert np.allclose(series[:, 0], [r.tilt_x_arcsec for r in driver.truth.frames], atol=1e-9)


def test_the_gaussian_centroid_sits_at_the_true_position() -> None:
    """In `gaussian` mode the star is exactly at the catalog position plus the truth tilt."""
    driver = sim_camera(
        VirtualClock(),
        r0_m=0.08,
        wind_speed_m_s=6.0,
        seed=5,
        screen_points=128,
        psf_mode="gaussian",
        stars=polaris_field(include_polaris_b=False),
        scintillation=ScintillationConfig(enabled=False),
        twilight=False,
        pointing=ON_A_PIXEL,
    )
    driver.open()
    driver.configure(StreamConfig("bin1", 3000, 0, roi=ROI, offset=30))
    driver.start()
    differences = []
    for _ in range(60):
        frame = driver.read_frame(1.0)
        record = driver.truth.frames[-1]
        electrons = (frame.data.astype(np.float64) / 16.0 - 30.0) * 3.5
        centre_x = round(record.catalog_x_px) - ROI.x
        centre_y = round(record.catalog_y_px) - ROI.y
        window = electrons[centre_y - 20 : centre_y + 21, centre_x - 20 : centre_x + 21]
        rows, columns = np.indices(window.shape)
        cx = float((columns * window).sum() / window.sum()) - 20 + centre_x + ROI.x
        cy = float((rows * window).sum() / window.sum()) - 20 + centre_y + ROI.y
        differences += [cx - record.star_x_px, cy - record.star_y_px]
    # The centroid noise at 21,000 electrons is about 0.05 pixel.
    assert float(np.mean(differences)) == pytest.approx(0.0, abs=0.02)
    assert float(np.std(differences)) < 0.09
    shifts = [(r.star_x_px - r.catalog_x_px) * PLATE_SCALE for r in driver.truth.frames]
    assert np.std(shifts) > 0.2  # the star moves by arcseconds, so the test is not trivial


def test_the_zenith_angle_follows_the_site_and_scales_r0() -> None:
    driver = create(profile=None, clock=VirtualClock(), options=SimOptions(seed=1))
    # The pole is 35 degrees from the zenith at a latitude of 55 degrees.
    assert driver.truth.zenith_angle_deg == pytest.approx(35.0)
    assert driver.truth.r0_observed_m() == pytest.approx(
        driver.truth.r0_zenith_m() * math.cos(math.radians(35)) ** 0.6
    )
    assert driver.truth.airmass_of_pole() == pytest.approx(1.22, abs=0.01)
    assert driver.truth.site.latitude_deg == 55.0  # a synthetic site


def test_a_scheduled_r0_shows_in_the_truth() -> None:
    config = TurbulenceConfig(
        r0_m=0.10,
        layers=(Layer(1.0, 10.0, 0.0),),
        r0_schedule=((0.0, 0.10), (3600.0, 0.05)),
        screen_points=128,
    )
    driver = create(profile=None, clock=VirtualClock(), options=SimOptions(turbulence=config))
    assert driver.truth.r0_zenith_m(DEFAULT_START_UTC_NS) == pytest.approx(0.10)
    assert driver.truth.r0_zenith_m(DEFAULT_START_UTC_NS + 1800 * NS_PER_S) == pytest.approx(0.075)
    assert driver.truth.image_motion_rms_arcsec(DEFAULT_START_UTC_NS + 3600 * NS_PER_S) == (
        pytest.approx(
            g_tilt_rms_arcsec(
                0.05, driver.truth.r0_observed_m(DEFAULT_START_UTC_NS + 3600 * NS_PER_S), 20.0
            )
        )
    )


def test_star_positions_and_the_pole() -> None:
    driver = sim_camera(VirtualClock(), screen_points=128, pointing=ON_A_PIXEL)
    truth = driver.truth
    stars = truth.star_positions(DEFAULT_START_UTC_NS, "bin1")
    brightest = int(np.argmin(stars.mag))
    assert stars.mag[brightest] == pytest.approx(2.02)
    assert (stars.x[brightest], stars.y[brightest]) == pytest.approx((4144.0, 2822.0), abs=1e-3)
    assert truth.field.mag[stars.index[brightest]] == pytest.approx(2.02)
    pole_x, pole_y = truth.pole_pixel(DEFAULT_START_UTC_NS, "bin1")
    assert (pole_x, 2822.0 - pole_y) == pytest.approx((4144.0, 1165.0), abs=1.0)
    assert truth.pole_pixel(DEFAULT_START_UTC_NS + 3600 * NS_PER_S, "bin1") == pytest.approx(
        (pole_x, pole_y)
    )
    assert truth.roll_deg == 0.0
    later = truth.star_positions(DEFAULT_START_UTC_NS + 600 * NS_PER_S, "bin1")
    assert later.x[brightest] - stars.x[brightest] == pytest.approx(51.0, abs=0.5)


def test_the_driver_keeps_a_bounded_history_when_asked() -> None:
    driver = create(
        profile=None,
        clock=VirtualClock(),
        options=SimOptions(
            keep_truth_frames=3,
            turbulence=TurbulenceConfig(screen_points=128),
            stars=polaris_field(),
        ),
    )
    driver.open()
    driver.configure(StreamConfig("bin1", 2000, 0, roi=Roi(4000, 2700, 128, 128)))
    driver.start()
    for _ in range(7):
        driver.read_frame(1.0)
    assert [r.seq for r in driver.truth.frames] == [4, 5, 6]
    assert driver.truth.frames_recorded == 7
    assert list(driver.truth.frame_arrays(stream_id=1)["seq"]) == [4, 5, 6]


def test_sky_functions_pass_through() -> None:
    driver = sim_camera(VirtualClock(), screen_points=128, sky_mag_arcsec2=21.0)
    truth = driver.truth
    assert float(truth.sun_altitude_deg(DEFAULT_START_UTC_NS)) < -50
    assert float(truth.sky_mag_arcsec2(DEFAULT_START_UTC_NS)) == pytest.approx(21.0)
    assert truth.transparency(DEFAULT_START_UTC_NS) == 1.0
    assert 0.2 < truth.scintillation_rms(0.002) < 0.5
    assert truth.sensor_temperature_c(DEFAULT_START_UTC_NS) == pytest.approx(19.0)
    assert truth.substeps(0.002) >= 4
