"""Long exposures: survey frames with trailing stars, flux calibration, and empty fields."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.clock import DEFAULT_START_UTC_NS, NS_PER_S, VirtualClock
from seeingmon.drivers.sim import (
    Pointing,
    ScintillationConfig,
    SimDriver,
    StarField,
    make_polar_field,
    polaris_field,
    sim_camera,
)
from seeingmon.drivers.sim.params import SimParams
from seeingmon.drivers.sim.stars import (
    POLARIS_DEC_DEG,
    POLARIS_RA_DEG,
    sidereal_motion_arcsec_per_s,
)
from seeingmon.frames import Roi, StreamConfig, StreamKind

BIN2 = SimParams.reference("bin2")
FloatArray = npt.NDArray[np.float64]
NO_STARS = StarField.from_arrays([], [], [])


def survey_camera(stars: StarField, **kwargs: object) -> SimDriver:
    defaults: dict[str, object] = {
        "psf_mode": "wave",
        "screen_points": 128,
        "scintillation": ScintillationConfig(enabled=False),
        "twilight": False,
        "ambient_c": 15.0,
        "stars": stars,
    }
    defaults.update(kwargs)
    return sim_camera(VirtualClock(), **defaults)  # type: ignore[arg-type]


def read_survey(camera: SimDriver, roi: Roi, exposure_s: float, gain: int = 120) -> FloatArray:
    """One snapshot, as ADC values with the 14-bit scaling removed."""
    camera.open()
    camera.configure(
        StreamConfig(
            "bin2", round(exposure_s * 1e6), gain, kind=StreamKind.SNAPSHOT, roi=roi, offset=30
        )
    )
    camera.start()
    frame = camera.read_frame(exposure_s * 2 + 10)
    return np.asarray(frame.data, dtype=np.float64) / 4.0


def aperture_flux(adu: FloatArray, centre: tuple[float, float], radius: float, gain: int) -> float:
    """The electrons in a circular aperture, minus the background from a surrounding annulus."""
    rows, columns = np.indices(adu.shape)
    distance = np.hypot(columns - centre[0], rows - centre[1])
    level = float(np.median(adu[(distance > radius + 8) & (distance <= radius + 20)]))
    return float((adu[distance <= radius] - level).sum()) * BIN2.sensor_at(gain).e_per_adu


def test_a_faint_star_has_the_flux_of_its_magnitude() -> None:
    """A magnitude-11 star gives 4.6e7 x 10^(-4.4) electrons per second: 18,300 in 10 s.

    A brighter star would clip its core in bin2: the Airy core is smaller than a pixel, and the
    full well at gain 120 is 14,400 electrons.
    """
    star = StarField.from_arrays([POLARIS_RA_DEG], [POLARIS_DEC_DEG], [11.0])
    camera = survey_camera(star)
    roi = Roi(1900, 1250, 400, 320)
    adu = read_survey(camera, roi, 10.0)
    truth = camera.truth.frames[-1]
    centre = (truth.star_x_px - roi.x, truth.star_y_px - roi.y)
    assert truth.flux_e == pytest.approx(4.6e7 * 10 ** (-4.4) * 10.0, rel=1e-6)
    assert aperture_flux(adu, centre, 14.0, 120) == pytest.approx(truth.flux_e, rel=0.03)
    # The 10 s of sky rotation moved the star by 0.4 pixel, and the tilt by a fraction of that.
    assert abs(truth.star_x_px - truth.catalog_x_px) < 0.1


def test_stars_trail_along_the_sky_rotation() -> None:
    """A star 2 degrees from the pole trails 15.041 sin(theta) arcsec per second."""
    field = make_polar_field(seed=2, mag_limit=12.0)
    probe = survey_camera(field)
    t_mid = DEFAULT_START_UTC_NS + 30 * NS_PER_S
    positions = probe.truth.star_positions(t_mid, "bin2")
    polar = 90.0 - field.dec_deg[positions.index]
    candidates = np.flatnonzero(
        (np.abs(polar - 2.0) < 0.4)
        & (positions.mag < 11.5)
        & (np.abs(positions.x - 2071.5) < 1900)
        & (np.abs(positions.y - 1410.5) < 1250)
    )
    assert len(candidates) > 0
    star = int(candidates[np.argmin(positions.mag[candidates])])
    margin = 60
    roi = Roi(
        round(float(positions.x[star])) - margin,
        round(float(positions.y[star])) - margin,
        2 * margin,
        2 * margin,
    )
    alone = StarField.from_arrays(
        [field.ra_deg[positions.index[star]]],
        [field.dec_deg[positions.index[star]]],
        [positions.mag[star]],
    )
    exposure = 60.0
    camera = survey_camera(alone)
    image = read_survey(camera, roi, exposure)
    expected_trail_px = sidereal_motion_arcsec_per_s(float(polar[star])) * exposure / 3.820
    level = float(np.median(image))
    noise = 1.4826 * float(np.median(np.abs(image - level)))
    weights = np.clip(image - level, 0.0, None)
    weights[weights < 4 * noise] = 0.0
    rows, columns = np.indices(image.shape)
    total = float(weights.sum())
    cx = float((columns * weights).sum()) / total
    cy = float((rows * weights).sum()) / total
    covariance = (
        np.array(
            [
                [
                    float((weights * (columns - cx) ** 2).sum()),
                    float((weights * (columns - cx) * (rows - cy)).sum()),
                ],
                [
                    float((weights * (columns - cx) * (rows - cy)).sum()),
                    float((weights * (rows - cy) ** 2).sum()),
                ],
            ]
        )
        / total
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # A uniform trail of length L adds L^2 / 12 to the variance along its direction.
    assert float(eigenvalues[1] - eigenvalues[0]) == pytest.approx(
        expected_trail_px**2 / 12.0, rel=0.35
    ), expected_trail_px
    # The trail follows the motion of the star on the sensor.
    start = camera.truth.star_positions(DEFAULT_START_UTC_NS, "bin2")
    end = camera.truth.star_positions(DEFAULT_START_UTC_NS + round(exposure * NS_PER_S), "bin2")
    motion = np.array([end.x[0] - start.x[0], end.y[0] - start.y[0]])
    length = float(np.hypot(*motion))
    assert length == pytest.approx(expected_trail_px, rel=0.02)
    assert abs(float(np.dot(eigenvectors[:, 1], motion / length))) > 0.97


def test_polaris_saturates_and_leaves_a_halo_in_a_survey_frame() -> None:
    camera = survey_camera(polaris_field(include_polaris_b=False))
    roi = Roi(1800, 1300, 500, 400)
    adu = read_survey(camera, roi, 30.0)
    truth = camera.truth.frames[-1]
    assert truth.flux_e == pytest.approx(4.6e7 * 10 ** (-0.4 * 2.02) * 30.0, rel=1e-6)
    assert adu.max() == 16383  # the ADC clips the core
    rows, columns = np.indices(adu.shape)
    distance = np.hypot(columns - (truth.star_x_px - roi.x), rows - (truth.star_y_px - roi.y))
    background = float(np.median(adu))
    # The halo falls with radius but stays above the background at 100 pixels.
    near = float(adu[(distance > 30) & (distance < 40)].mean())
    far = float(adu[(distance > 90) & (distance < 110)].mean())
    assert near > far > background + 1


def test_a_field_without_stars_has_no_reference_star() -> None:
    camera = survey_camera(NO_STARS, psf_mode="gaussian")
    camera.open()
    camera.configure(StreamConfig("bin1", 2000, 0, roi=Roi(1000, 1000, 64, 64)))
    camera.start()
    camera.read_frame(1.0)
    record = camera.truth.frames[-1]
    assert record.star_index == -1
    assert math.isnan(record.star_x_px)
    assert math.isfinite(record.tilt_x_arcsec)  # the atmosphere still moves, with no star to see
    assert record.flux_e == 0.0


def test_alignment_video_runs_with_long_exposures() -> None:
    """Exposures above 0.1 s use the long-exposure model, also in video mode."""
    camera = survey_camera(make_polar_field(seed=2, mag_limit=11.0))
    camera.open()
    active = camera.configure(
        StreamConfig("bin2", 500_000, 120, roi=Roi(1500, 1100, 1000, 700), offset=30)
    )
    assert active.frame_period_s == pytest.approx(0.5)  # the exposure exceeds the readout
    camera.start()
    first = camera.read_frame(2.0)
    second = camera.read_frame(2.0)
    assert second.t_utc_ns - first.t_utc_ns == 500_000_000
    assert camera.truth.frames[-1].star_index >= 0  # Polaris is in the ROI
    assert first.data.shape == (700, 1000)
    assert float(np.median(first.data)) == pytest.approx(float(np.median(second.data)), rel=0.02)


def test_the_pointing_offset_moves_every_star() -> None:
    plain = survey_camera(polaris_field(), pointing=Pointing())
    shifted = survey_camera(polaris_field(), pointing=Pointing(offset_arcsec=(7.64, -3.82)))
    a = plain.truth.star_positions(DEFAULT_START_UTC_NS, "bin2")
    b = shifted.truth.star_positions(DEFAULT_START_UTC_NS, "bin2")
    # 7.64 arcsec is two bin2 pixels to the right, and -3.82 arcsec is one pixel up.
    assert np.allclose(b.x - a.x, 2.0, atol=1e-6)
    assert np.allclose(b.y - a.y, -1.0, atol=1e-6)
