"""The survey path against the simulated camera, from a dark set to a sky quality record.

The `sim` driver renders a survey frame of a known sky: stars at the Gaia magnitudes of a
synthetic catalog, a sky of 20.5 mag/arcsec^2, dark current at the sensor temperature, hot
pixels, and the noise of the sensor. The test

1. records a dark library with `seeingmon.survey.dark_session` against a covered sim camera,
2. renders a survey frame with the same sensor,
3. runs the survey analyzer on it, and
4. compares the sky quality record with what the simulator injected.

The simulator's zero point is the photoelectron rate of a magnitude-0 star in the profile
(4.6e7 e-/s for the 50 mm aperture, so `ZP = 2.5 log10(4.6e7) = 19.157`), and its sky is
`SimOptions.sky_mag_arcsec2`. The simulator has no color term, so the fit must find zero. A
second frame under a simulated cloud with half the transparency checks the transparency.

The synthetic site of the sim (latitude 55, longitude 0) is not a real station.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from seeingmon.analysis import SurveyOutput
from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.drivers.sim import (
    CloudEvent,
    Clouds,
    Pointing,
    SimDriver,
    SimOptions,
    StarField,
)
from seeingmon.drivers.sim.params import SimParams
from seeingmon.frames import Frame, StreamConfig, StreamKind
from seeingmon.profile import Profile
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.survey import apparent
from seeingmon.survey.analyzer import InlineExecutor, SurveyPipelineAnalyzer
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.dark import DarkLibrary
from seeingmon.survey.dark_session import DarkSessionOptions, run_dark_session
from seeingmon.survey.geometry import vector_to_radec
from seeingmon.survey.pipeline import SurveyPipeline
from seeingmon.survey.pointing import PointingSolution
from seeingmon.survey.transparency import MemoryHistory, ZeroPointSample
from seeingmon.survey.wcs_fit import CameraAttitude
from tests.survey import simfx, synth

WIDTH, HEIGHT = 1200, 900
SKY_MAG = 20.5  # the injected sky brightness, mag/arcsec^2 in the camera band
AMBIENT_C = 12.0
SEED = 4
EXPOSURE_S = 30.0
T_START = synth.NIGHT_UTC_NS
T_MID = T_START + round(EXPOSURE_S / 2 * NS_PER_S)
HOT_PER_MPIX = 120.0


@pytest.fixture(scope="module")
def profile() -> Profile:
    return synth.cropped_profile(WIDTH, HEIGHT)


@pytest.fixture(scope="module")
def catalog() -> CapCatalog:
    return synth.synthetic_catalog(cap_radius_deg=8.0, density_scale=1.0, seed=3)


def zero_point_of_sim(profile: Profile) -> float:
    return 2.5 * math.log10(SimParams.from_profile(profile, "bin2").mag0_rate_e_per_s())


def sim_scene(
    profile: Profile, catalog: CapCatalog, t_utc_ns: int, clock: VirtualClock
) -> tuple[StarField, Pointing, CameraAttitude]:
    """The catalog stars at their apparent places for one time, and a pointing at Polaris.

    The sim turns its stars about the pole at the sidereal rate from the reference time of its
    pointing. Stars at their apparent places (CIRS) at the middle of the exposure then land where
    the pipeline predicts them, because the pipeline works in CIRS as well.
    """
    epoch = apparent.epoch_from_utc_ns(t_utc_ns)
    vectors = apparent.apparent_vectors(
        catalog.ra_deg,
        catalog.dec_deg,
        catalog.pm_ra_mas_yr,
        catalog.pm_dec_mas_yr,
        catalog.parallax_mas,
        epoch,
        catalog_epoch_jyear=catalog.epoch_jyear,
    )
    ra, dec = vector_to_radec(vectors)
    field = StarField.from_arrays(ra, dec, catalog.g_mag)
    polaris = int(np.argmin(catalog.g_mag))
    pointing = Pointing(
        ra_deg=float(ra[polaris]),
        dec_deg=float(dec[polaris]),
        roll_deg=25.0,
        t_ref_utc_ns=t_utc_ns,
        offset_arcsec=(0.0, 0.0),
    )
    # The attitude of this pointing in the pipeline's convention: camera = R @ cirs, with x to
    # the right and y down. The sim's rows are (right, up, axis), so y flips.
    ra_r, dec_r, roll = np.radians([pointing.ra_deg, pointing.dec_deg, pointing.roll_deg])
    axis = np.array([np.cos(dec_r) * np.cos(ra_r), np.cos(dec_r) * np.sin(ra_r), np.sin(dec_r)])
    north = np.array([-np.sin(dec_r) * np.cos(ra_r), -np.sin(dec_r) * np.sin(ra_r), np.cos(dec_r)])
    west = np.array([np.sin(ra_r), -np.cos(ra_r), 0.0])
    right = np.cos(roll) * west - np.sin(roll) * north
    up = np.sin(roll) * west + np.cos(roll) * north
    rotation = np.stack([right, -up, axis])
    scale = SimParams.from_profile(profile, "bin2").pixel_rad
    attitude = CameraAttitude(rotation, scale, 1, ((WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0))
    return field, pointing, attitude


def survey_frame(
    profile: Profile,
    catalog: CapCatalog,
    *,
    clouds: Clouds | None = None,
    seed: int = SEED,
) -> tuple[Frame, CameraAttitude, SimDriver]:
    clock = VirtualClock(T_START)
    field, pointing, attitude = sim_scene(profile, catalog, T_MID, clock)
    options = SimOptions(
        seed=seed,
        stars=field,
        pointing=pointing,
        sky_mag_arcsec2=SKY_MAG,
        twilight=False,
        ambient_c=AMBIENT_C,
        clouds=clouds or Clouds(),
        hot_pixels=simfx.covered_options(hot_pixels_per_mpix=HOT_PER_MPIX, seed=SEED).hot_pixels,
    )
    driver = simfx.make_driver(profile, options, clock)
    driver.open()
    driver.configure(StreamConfig("bin2", round(EXPOSURE_S * 1e6), 120, kind=StreamKind.SNAPSHOT))
    driver.start()
    frame = driver.read_frame(timeout_s=EXPOSURE_S + 30.0)
    driver.stop()
    assert frame.t_utc_ns == T_MID
    return frame, attitude, driver


@pytest.fixture(scope="module")
def library(profile: Profile, tmp_path_factory: pytest.TempPathFactory) -> DarkLibrary:
    """A dark set that `seeingmon dark` records from a covered sim camera of the same sensor."""
    directory = tmp_path_factory.mktemp("sim") / "calibration" / "darks"
    library = DarkLibrary(directory)
    clock = VirtualClock(T_START - 3 * 3600 * NS_PER_S)
    covered = simfx.make_driver(
        profile,
        simfx.covered_options(ambient_c=AMBIENT_C, hot_pixels_per_mpix=HOT_PER_MPIX, seed=SEED),
        clock,
    )
    run_dark_session(
        covered,
        library,
        profile,
        clock,
        DarkSessionOptions(wait=False, frames=5, bias_frames=5, exposure_s=EXPOSURE_S),
        say=lambda message: None,
    )
    return library


def analyze(
    profile: Profile,
    catalog: CapCatalog,
    frame: Frame,
    attitude: CameraAttitude,
    library: DarkLibrary,
    *,
    history: MemoryHistory | None = None,
) -> SurveyOutput:
    pipeline = SurveyPipeline(
        station_id="sim-station",
        profile=profile,
        catalog=catalog,
        solvers=[],
        config=SurveyConfig(),
        dark_library=library,
    )
    analyzer = SurveyPipelineAnalyzer(
        profile=profile,
        station_id="sim-station",
        pipeline=pipeline,
        executor=InlineExecutor(),
        history=history,
    )
    epoch = apparent.epoch_from_utc_ns(frame.t_utc_ns)
    analyzer.tracker.update(
        PointingSolution.from_attitude(
            attitude,
            epoch,
            mode="bin2",
            width_px=WIDTH,
            height_px=HEIGHT,
            n_matched=300,
            rms_arcsec=0.05,
            solver="sim-truth",
        )
    )
    analyzer.submit(frame)
    (output,) = analyzer.poll()
    analyzer.close()
    return output


def sky_of(output: SurveyOutput) -> SkyQualityRecord:
    return next(r for r in output.records if isinstance(r, SkyQualityRecord))


def test_the_sim_frame_gives_the_injected_zero_point_and_sky(
    profile: Profile, catalog: CapCatalog, library: DarkLibrary
) -> None:
    """A simulated survey frame with a known sky brightness and a known zero point.

    The frame holds about 85 measurable stars that scatter by 0.014 mag around the fit, so the
    sampling error of the zero point is 0.014 / sqrt(83) = 0.0015 mag, and the color term raises it
    to 0.003 mag. The tolerance of 0.03 mag is ten times that. The error that remains (0.011 mag)
    is a bias, not noise: the zero point refers to the light within 12 pixels, and the simulated
    profile keeps 1.1% of its light beyond that, in the wings of its Airy pattern. The sky
    brightness rests on that zero point and on the median of a million pixels.
    """
    frame, attitude, driver = survey_frame(profile, catalog)
    output = analyze(profile, catalog, frame, attitude, library)
    assert output.solved
    sky = sky_of(output)
    truth_zp = zero_point_of_sim(profile)
    assert sky.zero_point_mag is not None
    assert abs(sky.zero_point_mag - truth_zp) < 0.03
    assert abs(sky.zero_point_mag - truth_zp) < 0.02  # what it achieves: 0.011 mag
    assert sky.n_stars_used > 70
    assert sky.color_term is not None
    assert abs(sky.color_term) < 0.03  # the simulated camera has no color term
    assert sky.sky_mag_arcsec2 is not None
    assert abs(sky.sky_mag_arcsec2 - SKY_MAG) < 0.1  # the camera-band tolerance
    assert abs(sky.sky_mag_arcsec2 - SKY_MAG) < 0.03  # what it achieves: 0.014 mag
    assert float(sky.provenance["ap_corr"]) == pytest.approx(1.0147, abs=0.005)
    # The injected truth of the driver agrees with the inputs of the test.
    assert driver.truth.sky_mag_arcsec2(T_MID) == pytest.approx(SKY_MAG)
    assert frame.temperature_c == pytest.approx(AMBIENT_C + 4.0)
    assert sky.dark_model_version == library.version()
    assert "dark_due" not in sky.flags  # the library holds a set at this temperature
    assert sky.quality is None or "sky_mag_arcsec2" not in sky.quality


def test_a_simulated_cloud_lowers_the_transparency_by_the_injected_factor(
    profile: Profile, catalog: CapCatalog, library: DarkLibrary
) -> None:
    clear, attitude, _ = survey_frame(profile, catalog)
    truth_zp = zero_point_of_sim(profile)
    # The history of a clear season: 30 frames at the simulator's zero point.
    history = MemoryHistory(
        ZeroPointSample(T_START - (i + 1) * 3600 * NS_PER_S, truth_zp, 0.02, 80, 0.0)
        for i in range(30)
    )
    clear_sky = sky_of(analyze(profile, catalog, clear, attitude, library, history=history))
    assert clear_sky.transparency is not None
    assert clear_sky.transparency == pytest.approx(1.0, abs=0.04)
    # A cloud that passes half of the light for the whole exposure.
    cloud = Clouds(
        events=(
            CloudEvent(
                start_utc_ns=T_START - 600 * NS_PER_S,
                duration_s=1200.0,
                transmission=0.5,
                ramp_s=5.0,
            ),
        )
    )
    cloudy, attitude, driver = survey_frame(profile, catalog, clouds=cloud, seed=SEED + 1)
    injected = float(driver.truth.transparency(T_MID))
    assert injected == pytest.approx(0.5, abs=1e-6)
    sky = sky_of(analyze(profile, catalog, cloudy, attitude, library, history=history))
    assert sky.transparency is not None
    assert sky.transparency == pytest.approx(injected, abs=0.03)
    assert "cloud" in sky.flags
