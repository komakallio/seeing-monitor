"""The pipeline with the binned search: the pointing, the matched stars, and the zero point.

The binned search runs in frames that have a trail model, which means every frame after the first
solution. The first frame of a run has none, so the pipeline searches it at full resolution.
"""

from __future__ import annotations

import numpy as np
import pytest

from seeingmon.frames import Frame
from seeingmon.profile import Profile
from seeingmon.records.survey import StarListRecord, star_rows
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.config import DetectConfig, SurveyConfig
from seeingmon.survey.detect import StarFlag
from seeingmon.survey.geometry import ARCSEC_PER_RAD, exp_so3
from seeingmon.survey.pipeline import FrameAnalysis
from seeingmon.survey.pointing import PointingSolution
from tests.survey import synth
from tests.survey.test_pipeline import (
    NS,
    pipeline_for,
    pointing_error_px,
    pointing_of,
    sky_of,
    truth_solver,
)

BINNED = SurveyConfig(detect=DetectConfig(coarse_bin=2))
FULL = SurveyConfig(detect=DetectConfig(coarse_bin=1))  # the default is the binned search


@pytest.fixture(scope="module")
def profile() -> Profile:
    return synth.cropped_profile(1200, 800)


@pytest.fixture(scope="module")
def catalog() -> CapCatalog:
    return synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=3.0, seed=1)


@pytest.fixture(scope="module")
def first(profile: Profile, catalog: CapCatalog) -> tuple[Frame, synth.SynthTruth, FrameAnalysis]:
    frame, truth = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        exposure_s=30.0,
        seed=3,
    )
    analysis = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
    assert analysis.solution is not None
    return frame, truth, analysis


@pytest.fixture(scope="module")
def later(
    profile: Profile, catalog: CapCatalog, first: tuple[Frame, synth.SynthTruth, FrameAnalysis]
) -> tuple[Frame, synth.SynthTruth]:
    """The frame of the next survey step: 3 minutes on, the mount where it was."""
    _, truth, _ = first
    return synth.render_frame(
        catalog,
        profile,
        rotation_tirs=truth.rotation_tirs,
        t_utc_ns=truth.t_utc_ns + 180 * NS,
        exposure_s=30.0,
        seed=4,
    )


def test_the_binned_search_gives_the_pointing_and_the_zero_point_of_the_full_search(
    profile: Profile,
    catalog: CapCatalog,
    first: tuple[Frame, synth.SynthTruth, FrameAnalysis],
    later: tuple[Frame, synth.SynthTruth],
) -> None:
    _, _, start = first
    frame, truth = later
    full = pipeline_for(profile, catalog, config=FULL).analyze(frame, previous=start.solution)
    binned = pipeline_for(profile, catalog, config=BINNED).analyze(frame, previous=start.solution)
    assert full.solved
    assert binned.solved
    assert full.detections is not None
    assert binned.detections is not None
    assert not full.detections.has(StarFlag.COARSE).any()
    assert binned.detections.has(StarFlag.COARSE).sum() > 0  # a star with many saturated pixels
    assert pointing_error_px(binned, truth)[1] < 0.05
    assert full.fit is not None
    assert binned.fit is not None
    assert binned.fit.n_matched > 0.9 * full.fit.n_matched
    assert pointing_of(binned).solver == "tracker"
    assert pointing_of(binned).focus_fwhm_px == pytest.approx(0.45 * 2.3548, rel=0.05)
    assert_same_photometry(full, binned, truth, minimum_stars=0.9)


def test_the_brightest_stars_alone_give_the_same_pointing_and_zero_point(
    profile: Profile,
    catalog: CapCatalog,
    first: tuple[Frame, synth.SynthTruth, FrameAnalysis],
    later: tuple[Frame, synth.SynthTruth],
) -> None:
    _, _, start = first
    frame, truth = later
    full = pipeline_for(profile, catalog, config=FULL).analyze(frame, previous=start.solution)
    config = SurveyConfig(detect=DetectConfig(coarse_bin=2, refine_stars=150))
    limited = pipeline_for(profile, catalog, config=config).analyze(frame, previous=start.solution)
    assert limited.solved
    assert limited.detections is not None
    assert limited.detections.has(StarFlag.COARSE).sum() > 150
    assert pointing_error_px(limited, truth)[1] < 0.05
    assert limited.fit is not None
    assert 50 < limited.fit.n_matched < 150  # the reliable stars among the 150 fitted ones
    assert_same_photometry(full, limited, truth, minimum_stars=0.2)
    # The star list holds the coarse stars that no catalog star claims, flag bit included.
    star_list = next(r for r in limited.records if isinstance(r, StarListRecord))
    flags = star_rows(star_list)[:, star_list.columns.index("flags")].astype(int)
    assert ((flags & int(StarFlag.COARSE)) != 0).sum() > 10
    assert ((flags & int(StarFlag.COARSE)) == 0).sum() > 10


def assert_same_photometry(
    full: FrameAnalysis, other: FrameAnalysis, truth: synth.SynthTruth, *, minimum_stars: float
) -> None:
    """The zero point, the cloud fraction, and the stars behind them agree with the full search."""
    zero_full, zero_other = sky_of(full).zero_point_mag, sky_of(other).zero_point_mag
    assert zero_full is not None
    assert zero_other is not None
    assert zero_other == pytest.approx(zero_full, abs=0.01)
    assert zero_other == pytest.approx(truth.zero_point_mag, abs=0.02)
    assert full.cloud_fraction is not None
    assert other.cloud_fraction is not None
    assert abs(other.cloud_fraction - full.cloud_fraction) < 0.05
    assert sky_of(other).n_stars_used > minimum_stars * sky_of(full).n_stars_used


def test_the_first_frame_has_no_trail_model_and_takes_the_full_search(
    profile: Profile,
    catalog: CapCatalog,
    first: tuple[Frame, synth.SynthTruth, FrameAnalysis],
) -> None:
    frame, truth, start = first
    binned = pipeline_for(profile, catalog, [truth_solver(truth, catalog)], config=BINNED).analyze(
        frame
    )
    assert binned.detections is not None
    assert start.detections is not None
    assert not binned.detections.has(StarFlag.COARSE).any()
    assert len(binned.detections) == len(start.detections)
    np.testing.assert_array_equal(binned.detections.x, start.detections.x)
    assert binned.solved
    assert pointing_error_px(binned, truth)[1] < 0.05


def test_the_pointing_is_recovered_within_a_tenth_of_a_pixel_on_a_full_bin2_frame() -> None:
    """The criterion of the full-resolution test, with the binned search, on a frame of 4144 x 2822.

    The tracker starts from a solution that lies a pixel from the truth, as the solution of an
    earlier step lies a little off, and the frame has hot pixels that the mask does not know.
    """
    profile = synth.reference_profile()
    catalog = synth.synthetic_catalog(cap_radius_deg=6.0, density_scale=1.0, seed=11)
    frame, truth = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        exposure_s=30.0,
        n_hot_pixels=200,
        seed=12,
    )
    assert frame.data.shape == (2822, 4144)
    earlier = PointingSolution(
        rotation_earth_fixed=exp_so3([2e-5, -1.5e-5, 5e-6]) @ truth.rotation_tirs,
        scale_rad_px=truth.scale_arcsec_px / ARCSEC_PER_RAD,
        parity=truth.parity,
        mode=truth.mode,
        width_px=truth.width,
        height_px=truth.height,
        center_px=truth.center_px,
        t_utc_ns=truth.t_utc_ns - 180 * NS,
    )
    analysis = pipeline_for(profile, catalog, config=BINNED).analyze(frame, previous=earlier)
    assert analysis.solved
    assert pointing_of(analysis).solver == "tracker"
    rms, worst = pointing_error_px(analysis, truth)
    assert worst < 0.1  # pixels, anywhere in the field
    assert rms < 0.05
    assert analysis.detections is not None
    assert analysis.detections.has(StarFlag.COARSE).sum() > 100
    assert pointing_of(analysis).n_matched > 700
    assert sum(analysis.timings.values()) < 60.0  # seconds, on any CI machine
