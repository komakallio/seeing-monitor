"""The quick detector of the alignment view, and the lean analysis that uses it."""

from __future__ import annotations

from dataclasses import replace
from math import erf, sqrt

import numpy as np
import pytest

from seeingmon.frames import Frame
from seeingmon.profile import Profile
from seeingmon.solvers.triangles import TriangleSolver
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.centroid import FWHM_PER_SIGMA
from seeingmon.survey.config import DetectConfig, SurveyConfig
from seeingmon.survey.detect import DetectionError, StarFlag
from seeingmon.survey.detect_quick import QuickDetectOptions, block_sum, detect_quick
from seeingmon.survey.geometry import ARCSEC_PER_RAD
from seeingmon.survey.pipeline import SurveyPipeline
from seeingmon.survey.wcs_fit import CameraAttitude
from tests.survey import synth

ADC_BITS = 14
SHIFT = 16 - ADC_BITS
E_PER_ADU = 0.9
SIGMA_PX = 1.0


def sky(width: int = 640, height: int = 480, seed: int = 0) -> np.ndarray:
    """A flat sky of 300 counts with noise, in ADC counts."""
    rng = np.random.default_rng(seed)
    return rng.normal(300.0, 6.0, (height, width))


def add_star(image: np.ndarray, x: float, y: float, flux: float, sigma: float = SIGMA_PX) -> None:
    """Add a Gaussian star of total `flux` counts, integrated over the pixels."""
    half = int(6 * sigma) + 2
    columns = np.arange(int(x) - half, int(x) + half + 1)
    rows = np.arange(int(y) - half, int(y) + half + 1)
    inside_x = (columns >= 0) & (columns < image.shape[1])
    inside_y = (rows >= 0) & (rows < image.shape[0])

    def edge(values: np.ndarray, centre: float) -> np.ndarray:
        cdf = np.array(
            [
                erf((v - centre) / (sigma * sqrt(2.0)))
                for v in np.append(values - 0.5, values[-1] + 0.5)
            ]
        )
        return 0.5 * np.diff(cdf)

    patch = np.outer(edge(rows, y), edge(columns, x)) * flux
    image[np.ix_(rows[inside_y], columns[inside_x])] += patch[np.ix_(inside_y, inside_x)]


def container(image: np.ndarray) -> np.ndarray:
    """A frame as the camera delivers it: the ADC value in the high bits of 16 bits."""
    native = np.clip(np.rint(image), 0, (1 << ADC_BITS) - 1).astype(np.uint16)
    return (native << SHIFT).astype(np.uint16)


STARS = [(100.3, 80.6, 9000.0), (300.7, 120.2, 6000.0), (500.1, 400.9, 12000.0),
         (220.5, 330.4, 4000.0), (40.2, 440.7, 7000.0), (580.6, 60.3, 5000.0)]  # fmt: skip


def scene(seed: int = 0) -> np.ndarray:
    image = sky(seed=seed)
    for x, y, flux in STARS:
        add_star(image, x, y, flux)
    return container(image)


@pytest.mark.parametrize("spare_bits", [0, 2])
def test_the_block_sum_adds_each_block(spare_bits: int) -> None:
    rng = np.random.default_rng(1)
    native = rng.integers(0, 1 << 14, (36, 44)).astype(np.uint16)
    data = (native << spare_bits).astype(np.uint16) if spare_bits else native
    expected = (
        data[:36, :44].reshape(9, 4, 11, 4).sum(axis=(1, 3), dtype=np.float64).astype(np.float32)
    )
    result = block_sum(data, 4, 2 if spare_bits == 2 else 0)
    assert result.dtype == np.float32
    assert result.shape == (9, 11)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_it_finds_the_stars_where_they_are() -> None:
    detections = detect_quick(
        scene(), adc_bits=ADC_BITS, e_per_adu=E_PER_ADU, options=QuickDetectOptions(max_stars=30)
    )
    assert len(detections) >= len(STARS)
    truth = np.array([[x, y] for x, y, _ in STARS])
    found = np.column_stack([detections.x, detections.y])
    for point in truth:
        nearest = np.hypot(*(found - point).T)
        assert nearest.min() < 0.3, point


def test_the_model_fit_measures_the_width() -> None:
    detections = detect_quick(
        scene(), adc_bits=ADC_BITS, e_per_adu=E_PER_ADU, options=QuickDetectOptions(max_stars=30)
    )
    reliable = detections.reliable()
    assert int(reliable.sum()) >= len(STARS) - 1
    assert float(np.median(detections.fwhm_px[reliable])) == pytest.approx(
        FWHM_PER_SIGMA * SIGMA_PX, rel=0.1
    )


def test_the_stars_come_brightest_first() -> None:
    detections = detect_quick(scene(), adc_bits=ADC_BITS, e_per_adu=E_PER_ADU)
    assert np.all(np.diff(detections.flux) <= 0.0)


def test_a_saturated_star_is_flagged_and_not_fitted() -> None:
    image = sky()
    add_star(image, 200.2, 200.4, 2.0e6, sigma=1.5)  # far past the full scale
    add_star(image, 400.3, 300.1, 9000.0)
    detections = detect_quick(container(image), adc_bits=ADC_BITS, e_per_adu=E_PER_ADU)
    flags = detections.flags
    near = np.hypot(detections.x - 200.2, detections.y - 200.4) < 3.0
    assert near.any()
    assert np.all((flags[near] & int(StarFlag.SATURATED)) != 0)
    assert not detections.reliable()[near].any()
    other = np.hypot(detections.x - 400.3, detections.y - 300.1) < 0.5
    assert other.any()
    assert detections.reliable()[other].all()


def test_a_star_at_the_edge_is_flagged() -> None:
    image = sky()
    add_star(image, 3.4, 240.2, 9000.0)
    detections = detect_quick(container(image), adc_bits=ADC_BITS, e_per_adu=E_PER_ADU)
    edge = detections.x < 10.0
    assert edge.any()
    assert np.all((detections.flags[edge] & int(StarFlag.NEAR_EDGE)) != 0)


def test_a_frame_without_signal_is_a_detection_error() -> None:
    flat = np.full((480, 640), 4000, dtype=np.uint16)
    with pytest.raises(DetectionError, match="constant"):
        detect_quick(flat, adc_bits=ADC_BITS, e_per_adu=E_PER_ADU)


def test_a_frame_of_8_bit_counts_is_refused() -> None:
    with pytest.raises(DetectionError, match="16-bit"):
        detect_quick(np.zeros((480, 640), dtype=np.uint8), adc_bits=8, e_per_adu=1.0)


def test_a_frame_that_is_too_small_is_refused() -> None:
    with pytest.raises(DetectionError, match="small"):
        detect_quick(np.zeros((20, 20), dtype=np.uint16), adc_bits=ADC_BITS, e_per_adu=1.0)


@pytest.mark.parametrize("bad", [{"bin_factor": 1}, {"max_stars": 0}, {"threshold_sigma": 0.0}])
def test_the_options_are_checked(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="invalid"):
        QuickDetectOptions(**bad)  # type: ignore[arg-type]


# --- The lean analysis ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def profile() -> Profile:
    return synth.reference_profile()


@pytest.fixture(scope="module")
def catalog() -> CapCatalog:
    return synth.synthetic_catalog(cap_radius_deg=6.0, density_scale=1.0, seed=11)


@pytest.fixture(scope="module")
def scene_frame(profile: Profile, catalog: CapCatalog) -> tuple[Frame, synth.SynthTruth]:
    """A frame of the reference size, 4144 x 2822, taken in 1 s."""
    return synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(1.2, 70.0, 40.0),
        exposure_s=1.0,
        seed=4,
    )


def lean_pipeline(profile: Profile, catalog: CapCatalog) -> SurveyPipeline:
    detect = DetectConfig(
        threshold_sigma=8.0, max_stars=120, refine_stars=80, quick_bin=4, max_saturated_pixels=6
    )
    return SurveyPipeline(
        station_id="test",
        profile=profile,
        catalog=catalog,
        solvers=[TriangleSolver(catalog)],
        config=SurveyConfig(detect=detect),
    )


def test_the_lean_analysis_solves_a_frame_with_no_help_and_builds_no_records(
    profile: Profile, catalog: CapCatalog, scene_frame: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene_frame
    analysis = lean_pipeline(profile, catalog).analyze_quick(frame)
    assert analysis.solved
    assert analysis.records == ()
    assert analysis.solution is not None
    assert analysis.solution.solver == "triangles"
    assert set(analysis.timings) == {"detect", "solve"}
    true_model = CameraAttitude(
        truth.rotation_cirs, truth.scale_arcsec_px / ARCSEC_PER_RAD, truth.parity, truth.center_px
    )
    assert analysis.fit is not None
    gx, gy = np.meshgrid(
        np.linspace(0, truth.width - 1.0, 9), np.linspace(0, truth.height - 1.0, 7)
    )
    x, y, _ = analysis.fit.attitude.project(true_model.unproject(gx.ravel(), gy.ravel()))
    shift = np.hypot(x - gx.ravel(), y - gy.ravel())
    assert float(np.sqrt(np.mean(shift**2))) < 1.0  # pixels


def test_the_next_frame_follows_the_tracker(
    profile: Profile, catalog: CapCatalog, scene_frame: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, _ = scene_frame
    pipeline = lean_pipeline(profile, catalog)
    first = pipeline.analyze_quick(frame)
    assert first.solution is not None
    second = pipeline.analyze_quick(frame, previous=first.solution)
    assert second.solved
    assert second.solution is not None
    assert second.solution.solver == "tracker"


def test_a_frame_of_noise_gives_no_solution(
    profile: Profile, catalog: CapCatalog, scene_frame: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, _ = scene_frame
    rng = np.random.default_rng(9)
    noise = rng.normal(2000.0, 40.0, frame.data.shape).clip(0, 65535).astype(np.uint16)
    analysis = lean_pipeline(profile, catalog).analyze_quick(replace(frame, data=noise))
    assert not analysis.solved
    assert analysis.solution is None


def test_the_quick_bin_is_zero_by_default_and_the_property_says_so(
    profile: Profile, catalog: CapCatalog
) -> None:
    pipeline = SurveyPipeline(station_id="test", profile=profile, catalog=catalog, solvers=[])
    assert pipeline.quick_bin == 0
    assert lean_pipeline(profile, catalog).quick_bin == 4
