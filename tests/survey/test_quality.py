"""The sky quality of a survey frame: the zero point, the sky, clouds, and the dark library.

The frames come from the synthetic renderer, which draws catalog stars with a known zero point
and color term, a sky of a known brightness, the dark current at a known temperature, and the
noise of the sensor. The pipeline sees the frame as a camera would deliver it and has to give
the injected values back.

**The sampling error behind the tolerances.** The renderer's frame holds about 170 stars that
are bright enough to measure and not saturated (G between 11 and 13). They scatter by about
0.017 mag around the model (photon noise, and 1% of flux noise that stands for scintillation), so
the sampling error of the zero point is 0.017 / sqrt(170) = 0.0013 mag, and the color term
doubles it. The tolerance of 0.03 mag is ten times that. The sky brightness rests on the zero
point and on a median of a million pixels, and its tolerance is 0.1 mag in the camera band.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S
from seeingmon.frames import Frame, Roi
from seeingmon.profile import Profile
from seeingmon.records.survey import SkyQualityRecord, SurveyFrameRecord
from seeingmon.solvers.base import PlateSolver
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.config import SkyConfig, SurveyConfig
from seeingmon.survey.dark import DarkLibrary
from seeingmon.survey.pipeline import FrameAnalysis, SurveyPipeline
from seeingmon.survey.quality import FieldStars, missing_count_reason
from seeingmon.survey.sky import ArrayFlat, FlatModel, g_minus_v
from seeingmon.survey.transparency import ZeroPointReference
from tests.survey import synth

ZP_TRUE = 19.30
COLOR_TERM = -0.08
SKY_E_PER_S_PX = 4.2
DARK_E_PER_S_PX = 0.7
BIAS_DN = 120.0
TEMPERATURE_C = 18.0
EXPOSURE_S = 30.0
WIDTH, HEIGHT = 1600, 1200
E_PER_ADU = 0.88  # bin2 at gain 120
SCALE = 3.82


@pytest.fixture(scope="module")
def profile() -> Profile:
    return synth.cropped_profile(WIDTH, HEIGHT)


@pytest.fixture(scope="module")
def catalog() -> CapCatalog:
    return synth.synthetic_catalog(cap_radius_deg=8.0, density_scale=1.0, seed=3)


def render(
    profile: Profile, catalog: CapCatalog, *, seed: int = 5, **kwargs: object
) -> tuple[Frame, synth.SynthTruth]:
    params: dict[str, object] = {
        "zero_point_mag": ZP_TRUE,
        "color_term": COLOR_TERM,
        "star_scatter": 0.01,
        "sky_e_per_s_px": SKY_E_PER_S_PX,
        "dark_e_per_s_px": DARK_E_PER_S_PX,
        "offset_dn": BIAS_DN,
        "temperature_c": TEMPERATURE_C,
        "seed": seed,
    }
    params.update(kwargs)
    return synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        **params,  # type: ignore[arg-type]
    )


def dark_library(tmp_path: Path, *, temperature_c: float = TEMPERATURE_C) -> DarkLibrary:
    """A library with one dark set that matches the renderer's bias and dark current."""
    library = DarkLibrary(tmp_path / "calibration" / "darks")
    master = np.full((HEIGHT, WIDTH), round(BIAS_DN + DARK_E_PER_S_PX * EXPOSURE_S / E_PER_ADU))
    library.add_set(
        master.astype(np.uint16),
        mode="bin2",
        gain=120,
        exposure_s=EXPOSURE_S,
        temperature_c=temperature_c,
        temperature_spread_c=0.3,
        t_utc_ns=synth.NIGHT_UTC_NS - 3600 * NS_PER_S,
        n_frames=9,
        n_bias_frames=9,
        bias_dn=BIAS_DN,
        read_noise_dn=2.1,
        adc_bits=14,
        dark_dn=BIAS_DN + DARK_E_PER_S_PX * EXPOSURE_S / E_PER_ADU,
    )
    return library


def analyze(
    profile: Profile,
    catalog: CapCatalog,
    frame: Frame,
    truth: synth.SynthTruth,
    *,
    library: DarkLibrary | None = None,
    flat: FlatModel | None = None,
    reference: ZeroPointReference | None = None,
    config: SurveyConfig | None = None,
    solve: bool = True,
    sky_quality: bool | None = None,
) -> FrameAnalysis:
    solvers: list[PlateSolver] = (
        [synth.QueueSolver([synth.truth_solve_result(truth, catalog)])] if solve else []
    )
    pipeline = SurveyPipeline(
        station_id="test-station",
        profile=profile,
        catalog=catalog,
        solvers=solvers,
        config=config or SurveyConfig(),
        dark_library=library,
        flat=flat,
    )
    return pipeline.analyze(frame, zp_reference=reference, sky_quality=sky_quality)


def sky_record(analysis: FrameAnalysis) -> SkyQualityRecord:
    return next(r for r in analysis.records if isinstance(r, SkyQualityRecord))


def true_sky_mag(zero_point: float = ZP_TRUE) -> float:
    """The camera-band surface brightness of the injected sky."""
    return zero_point - 2.5 * math.log10(SKY_E_PER_S_PX / SCALE**2)


@pytest.fixture(scope="module")
def clear(profile: Profile, catalog: CapCatalog) -> tuple[Frame, synth.SynthTruth]:
    return render(profile, catalog)


@pytest.fixture(scope="module")
def clear_analysis(
    profile: Profile,
    catalog: CapCatalog,
    clear: tuple[Frame, synth.SynthTruth],
    tmp_path_factory: pytest.TempPathFactory,
) -> FrameAnalysis:
    library = dark_library(tmp_path_factory.mktemp("clear"))
    return analyze(profile, catalog, *clear, library=library)


# --- The zero point and the sky ------------------------------------------------------------


def test_a_clear_frame_gives_the_injected_zero_point_color_term_and_sky(
    clear_analysis: FrameAnalysis,
) -> None:
    record = sky_record(clear_analysis)
    assert record.zero_point_mag is not None
    assert record.color_term is not None
    assert record.zero_point_rms_mag is not None
    # The done criterion is 0.03 mag. The sampling error is about 0.002 mag (see the module text).
    assert abs(record.zero_point_mag - ZP_TRUE) < 0.03
    assert abs(record.zero_point_mag - ZP_TRUE) < 0.01  # what the fit actually achieves
    assert record.color_term == pytest.approx(COLOR_TERM, abs=0.02)
    assert record.zero_point_rms_mag < 0.03
    assert record.n_stars_used > 100
    assert record.sky_mag_arcsec2 is not None
    assert abs(record.sky_mag_arcsec2 - true_sky_mag()) < 0.1  # the camera-band tolerance
    assert abs(record.sky_mag_arcsec2 - true_sky_mag()) < 0.03  # what it achieves
    assert record.sky_rate_e_per_s_arcsec2 == pytest.approx(SKY_E_PER_S_PX / SCALE**2, rel=0.01)
    assert record.quality is not None
    assert "zero_point_mag" not in record.quality
    assert "sky_mag_arcsec2" not in record.quality


def test_the_v_equivalent_follows_the_color_term_the_gaia_relation_and_the_offset(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth], tmp_path: Path
) -> None:
    config = SurveyConfig(sky=SkyConfig(bp_rp=0.8, sqm_offset_mag=0.12))
    analysis = analyze(profile, catalog, *clear, library=dark_library(tmp_path), config=config)
    record = sky_record(analysis)
    assert record.sky_mag_arcsec2 is not None
    assert record.color_term is not None
    assert record.sky_mag_arcsec2_v is not None
    expected = record.sky_mag_arcsec2 + record.color_term * 0.8 - g_minus_v(0.8) + 0.12
    assert record.sky_mag_arcsec2_v == pytest.approx(expected, abs=1e-9)


def test_the_record_names_its_provenance(clear_analysis: FrameAnalysis) -> None:
    record = sky_record(clear_analysis)
    assert record.provenance["algo"] == "sky-1"
    assert record.provenance["flat"] == "unit"
    assert record.provenance["dark"].startswith("darks-")
    assert record.dark_model_version == record.provenance["dark"]
    assert len(record.provenance["catalog"]) == 8
    assert record.t_utc_ns == clear_analysis.records[0].t_utc_ns
    assert record.station_id == "test-station"


def test_without_a_dark_library_the_sky_is_missing_and_a_dark_is_due(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth]
) -> None:
    record = sky_record(analyze(profile, catalog, *clear))
    assert record.zero_point_mag is not None  # the stars do not need a dark model
    assert abs(record.zero_point_mag - ZP_TRUE) < 0.03
    assert record.sky_mag_arcsec2 is None
    assert record.sky_mag_arcsec2_v is None
    assert record.sky_rate_e_per_s_arcsec2 is None
    assert record.dark_model_version is None
    assert record.quality is not None
    assert "no dark model" in record.quality["sky_mag_arcsec2"]
    assert "dark_due" in record.flags


def test_the_dark_due_flag_follows_the_temperature_of_the_library(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth], tmp_path: Path
) -> None:
    covered = sky_record(
        analyze(profile, catalog, *clear, library=dark_library(tmp_path / "a", temperature_c=19.5))
    )
    assert "dark_due" not in covered.flags  # a set 1.5 C away covers the frame
    far = sky_record(
        analyze(profile, catalog, *clear, library=dark_library(tmp_path / "b", temperature_c=6.0))
    )
    assert "dark_due" in far.flags
    assert far.sky_mag_arcsec2 is not None  # the model extrapolates, and the flag warns


def test_a_wrong_dark_level_shows_in_the_sky_and_not_in_the_zero_point(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    """Why the dark model matters: dark current that the library misses reads as sky."""
    frame, truth = render(profile, catalog, dark_e_per_s_px=2.0)  # warmer than the library thinks
    record = sky_record(analyze(profile, catalog, frame, truth, library=dark_library(tmp_path)))
    assert record.zero_point_mag is not None
    assert abs(record.zero_point_mag - ZP_TRUE) < 0.03
    assert record.sky_mag_arcsec2 is not None
    # The extra 1.3 e-/s per pixel adds to the 4.2 of the sky, so the sky looks brighter by
    # 2.5 log10(5.5 / 4.2) = 0.29 mag.
    assert record.sky_mag_arcsec2 == pytest.approx(true_sky_mag() - 0.29, abs=0.03)


@pytest.mark.parametrize("exposure_s", [1.0, 2.0, 5.0])
def test_the_dark_level_scales_to_a_frame_shorter_than_the_dark_set(
    profile: Profile, catalog: CapCatalog, tmp_path: Path, exposure_s: float
) -> None:
    """The adaptive long exposure of twilight takes 1 to 30 s, and the dark set takes 30 s.

    The dark model is the bias plus the dark rate times the exposure, so a 2 s frame gets the bias
    and a fifteenth of the dark of the set. A model that did not scale would subtract the dark of
    30 s (24 counts at gain 120), far more than the 11 counts of sky and dark of a 2 s frame, and
    the frame would read darker than its dark level.
    """
    frame, truth = render(profile, catalog, exposure_s=exposure_s, seed=21)
    library = dark_library(tmp_path)
    model = library.model("bin2", 120, prior_doubling_c=6.0)
    assert model is not None
    dark_s = DARK_E_PER_S_PX / E_PER_ADU  # the dark rate of the set, in counts per second
    # One count of rounding in the master of the set, spread over its 30 s.
    assert model.level_dn(TEMPERATURE_C, exposure_s) == pytest.approx(
        BIAS_DN + dark_s * exposure_s, abs=exposure_s / EXPOSURE_S
    )
    record = sky_record(analyze(profile, catalog, frame, truth, library=library))
    assert record.sky_mag_arcsec2 is not None
    assert record.sky_rate_e_per_s_arcsec2 is not None
    # The frame's own zero point calibrates the sky, and it scatters by up to 0.02 mag at these
    # exposures, which hold fewer stars with enough signal. The run gave errors of 0.023, 0.030,
    # and 0.013 mag at 1, 2, and 5 s.
    assert record.sky_mag_arcsec2 == pytest.approx(true_sky_mag(), abs=0.05)


# --- Transparency and clouds ---------------------------------------------------------------


def reference_at(zero_point: float = ZP_TRUE) -> ZeroPointReference:
    return ZeroPointReference(zero_point, 80, 20, 60.0, 0.9)


def test_a_clear_frame_has_a_transparency_of_one(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth], tmp_path: Path
) -> None:
    record = sky_record(
        analyze(profile, catalog, *clear, library=dark_library(tmp_path), reference=reference_at())
    )
    assert record.transparency is not None
    assert record.transparency == pytest.approx(1.0, abs=0.03)
    assert "cloud" not in record.flags
    assert record.provenance["zp_ref"] == "19.3000"
    assert record.provenance["zp_ref_n"] == "80"
    assert record.quality is None or "transparency" not in record.quality


def test_an_injected_cloud_lowers_the_transparency_to_the_injected_value(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    """A uniform cloud that passes half of the light: the transparency is 0.5 within 0.03."""
    frame, truth = render(profile, catalog, transmission=0.5, seed=6)
    record = sky_record(
        analyze(
            profile, catalog, frame, truth, library=dark_library(tmp_path), reference=reference_at()
        )
    )
    assert record.transparency is not None
    assert record.transparency == pytest.approx(0.5, abs=0.03)
    assert record.zero_point_mag is not None
    assert record.zero_point_mag == pytest.approx(ZP_TRUE + 2.5 * math.log10(0.5), abs=0.03)
    assert "cloud" in record.flags  # below the transparency flag of 0.6


def test_a_thick_cloud_raises_the_cloud_fraction_and_lowers_the_limit(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    clear_frame, clear_truth = render(profile, catalog, seed=7)
    cloud_frame, cloud_truth = render(profile, catalog, transmission=0.01, seed=8)
    library = dark_library(tmp_path)
    clear_record = sky_record(
        analyze(
            profile, catalog, clear_frame, clear_truth, library=library, reference=reference_at()
        )
    )
    cloudy = analyze(
        profile, catalog, cloud_frame, cloud_truth, library=library, reference=reference_at()
    )
    cloud_record = sky_record(cloudy)
    assert clear_record.cloud_fraction is not None
    assert clear_record.cloud_fraction < 0.1
    assert cloud_record.cloud_fraction is not None
    assert cloud_record.cloud_fraction > 0.3  # half of the expected stars fall below the noise
    assert cloudy.cloud_fraction == cloud_record.cloud_fraction  # the same number in the output
    assert "cloud" in cloud_record.flags
    assert cloud_record.transparency is not None
    assert cloud_record.transparency == pytest.approx(0.01, rel=0.1)
    # A clear sky detects more than half of the catalog stars down to G = 13, so its limit comes
    # from the noise. The cloud (5 mag of extinction) brings the limit into the catalog's range.
    assert clear_record.limiting_mag is not None
    assert clear_record.limiting_mag > 16.0
    assert clear_record.quality is not None
    assert "predicted from the noise" in clear_record.quality["limiting_mag"]
    assert cloud_record.limiting_mag is not None
    assert 10.0 < cloud_record.limiting_mag < 12.5
    assert cloud_record.quality is None or "limiting_mag" not in cloud_record.quality
    assert cloud_record.limiting_mag < clear_record.limiting_mag - 4.0


def test_the_record_keeps_the_counts_behind_the_cloud_fraction(
    profile: Profile, catalog: CapCatalog, clear_analysis: FrameAnalysis, tmp_path: Path
) -> None:
    """The expected stars and the found ones give the fraction, and `n_detected` counts more."""
    frame, truth = render(profile, catalog, transmission=0.01, seed=8)
    library = dark_library(tmp_path)
    cloudy = sky_record(
        analyze(profile, catalog, frame, truth, library=library, reference=reference_at())
    )
    counts: dict[str, tuple[int, int]] = {}
    for name, record in (("clear", sky_record(clear_analysis)), ("cloudy", cloudy)):
        expected, found = record.n_expected, record.n_expected_found
        assert expected is not None
        assert found is not None
        assert 0 <= found <= expected
        assert record.cloud_fraction == pytest.approx(1.0 - found / expected, abs=1e-12)
        assert record.quality is None or "n_expected" not in record.quality
        counts[name] = (expected, found)
    assert counts["clear"][1] == counts["clear"][0]  # a clear sky shows every expected star
    assert counts["cloudy"][1] < counts["cloudy"][0]
    (survey_frame,) = [r for r in clear_analysis.records if isinstance(r, SurveyFrameRecord)]
    assert survey_frame.n_detected is not None
    assert survey_frame.n_detected > counts["clear"][0]  # every detection, not the expected stars


def test_too_few_expected_stars_keep_their_counts_and_say_so(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth], tmp_path: Path
) -> None:
    far_off = 15.0  # a reference so low that it expects hardly a star
    record = sky_record(
        analyze(
            profile,
            catalog,
            *clear,
            library=dark_library(tmp_path),
            reference=reference_at(far_off),
        )
    )
    assert record.cloud_fraction is None
    assert record.n_expected is not None
    assert record.n_expected < 8  # [survey.cloud] min_expected
    assert record.n_expected_found is not None
    assert record.quality is not None
    assert record.quality["cloud_fraction"] == f"too few expected stars ({record.n_expected})"
    assert "n_expected" not in record.quality


def test_a_frame_without_a_pointing_has_no_counts_and_says_why(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth], tmp_path: Path
) -> None:
    record = sky_record(
        analyze(profile, catalog, *clear, library=dark_library(tmp_path), solve=False)
    )
    assert record.n_expected is None
    assert record.n_expected_found is None
    assert record.cloud_fraction is None
    assert record.quality is not None
    for name in ("n_expected", "n_expected_found", "cloud_fraction"):
        assert record.quality[name] == "no pointing solution"


def test_without_a_clear_sky_signal_the_counts_are_missing_and_say_why(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth], tmp_path: Path
) -> None:
    """Without a reference zero point or a photometric prior, no star has an expected signal."""
    data = profile.model_dump(mode="python")
    data["photometry"] = None
    bare = Profile.model_validate(data)
    analysis = analyze(bare, catalog, *clear, library=dark_library(tmp_path))
    assert analysis.solved
    record = sky_record(analysis)
    assert record.n_expected is None
    assert record.n_expected_found is None
    assert record.cloud_fraction is None
    assert record.quality is not None
    reason = "no clear-sky signal: no reference zero point and no photometric prior"
    for name in ("n_expected", "n_expected_found", "cloud_fraction"):
        assert record.quality[name] == reason


def test_a_field_without_catalog_stars_has_no_counts_and_says_why() -> None:
    nothing = np.zeros(0)
    empty = FieldStars(
        rows=np.zeros(0, dtype=np.int64),
        g_mag=nothing,
        x=nothing,
        y=nothing,
        found=np.zeros(0, dtype=np.bool_),
    )
    assert missing_count_reason(empty, None) == "no catalog star lies in the frame"
    assert missing_count_reason(None, None) == "no pointing solution"
    one = replace(empty, rows=np.array([7]), g_mag=np.array([11.0]))
    assert missing_count_reason(one, 0) is None  # counts of zero are counts
    assert missing_count_reason(one, None) is not None


def test_a_frame_too_cloudy_for_a_zero_point_still_gives_the_clouds_and_the_sky(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    frame, truth = render(profile, catalog, transmission=0.003, seed=12)
    record = sky_record(
        analyze(
            profile, catalog, frame, truth, library=dark_library(tmp_path), reference=reference_at()
        )
    )
    assert record.zero_point_mag is None
    assert record.transparency is None
    assert record.n_stars_used == 0
    assert record.cloud_fraction is not None
    assert record.cloud_fraction > 0.7
    assert "cloud" in record.flags
    assert record.quality is not None
    assert "measurable stars" in record.quality["zero_point_mag"]
    assert "no zero point" in record.quality["transparency"]
    assert record.sky_mag_arcsec2 is not None  # the reference zero point calibrates the sky
    assert "reference zero point" in record.quality["sky_mag_arcsec2"]


def test_without_a_history_there_is_no_transparency_and_a_reason(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth], tmp_path: Path
) -> None:
    record = sky_record(analyze(profile, catalog, *clear, library=dark_library(tmp_path)))
    assert record.transparency is None
    assert record.quality is not None
    assert "no reference yet" in record.quality["transparency"]
    assert record.zero_point_mag is not None  # everything else stays


# --- A frame without a pointing solution --------------------------------------------------


def test_an_unsolved_frame_has_a_sky_rate_but_no_zero_point(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth], tmp_path: Path
) -> None:
    analysis = analyze(profile, catalog, *clear, library=dark_library(tmp_path), solve=False)
    record = sky_record(analysis)
    assert not analysis.solved
    assert record.zero_point_mag is None
    assert record.n_stars_used == 0
    assert record.sky_rate_e_per_s_arcsec2 == pytest.approx(SKY_E_PER_S_PX / SCALE**2, rel=0.01)
    assert record.sky_mag_arcsec2 is None
    assert record.quality is not None
    assert "no pointing solution" in record.quality["zero_point_mag"]
    assert "no zero point" in record.quality["sky_mag_arcsec2"]
    assert record.limiting_mag is None
    assert record.cloud_fraction is None
    assert analysis.epoch_stars is not None
    assert len(analysis.epoch_stars) == 0


def test_an_unsolved_frame_is_calibrated_with_the_reference_zero_point_when_there_is_one(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth], tmp_path: Path
) -> None:
    record = sky_record(
        analyze(
            profile,
            catalog,
            *clear,
            library=dark_library(tmp_path),
            reference=reference_at(),
            solve=False,
        )
    )
    assert record.zero_point_mag is None
    assert record.sky_mag_arcsec2 is not None
    assert abs(record.sky_mag_arcsec2 - true_sky_mag()) < 0.1
    assert record.quality is not None
    assert "reference zero point" in record.quality["sky_mag_arcsec2"]


# --- A provisional zero point --------------------------------------------------------------


def provisional_at(zero_point: float = ZP_TRUE) -> ZeroPointReference:
    """The stand-in that `provisional_zero_point` gives: six frames of the last six hours."""
    return ZeroPointReference(zero_point, 6, 1, 0.25, 0.5, provisional=True)


def test_an_unsolved_frame_is_calibrated_with_a_provisional_zero_point_when_that_is_all_there_is(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth], tmp_path: Path
) -> None:
    record = sky_record(
        analyze(
            profile,
            catalog,
            *clear,
            library=dark_library(tmp_path),
            reference=provisional_at(19.10),
            solve=False,
        )
    )
    assert record.zero_point_mag is None  # the stand-in is no measurement of this frame
    assert record.n_stars_used == 0
    assert record.sky_mag_arcsec2 is not None
    assert abs(record.sky_mag_arcsec2 - true_sky_mag(19.10)) < 0.03  # the sky follows the stand-in
    assert record.quality is not None
    assert record.quality["sky_mag_arcsec2"] == (
        "calibrated with a provisional zero point (the median of 6 frames of the last 6 h), "
        "because the frame has none"
    )
    assert record.transparency is None
    assert record.quality["transparency"] == "no zero point"
    assert record.provenance["zp_ref"] == "19.1000"
    assert record.provenance["zp_ref_n"] == "6"
    assert record.provenance["zp_ref_provisional"] == "true"


def test_a_provisional_zero_point_sets_no_transparency_for_a_frame_with_its_own_zero_point(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth], tmp_path: Path
) -> None:
    record = sky_record(
        analyze(
            profile,
            catalog,
            *clear,
            library=dark_library(tmp_path),
            reference=provisional_at(),
        )
    )
    assert record.zero_point_mag is not None
    assert abs(record.zero_point_mag - ZP_TRUE) < 0.03
    assert record.transparency is None  # a median of recent frames is no clear-sky level
    assert record.quality is not None
    assert "no reference yet" in record.quality["transparency"]
    assert "cloud" not in record.flags
    assert record.sky_mag_arcsec2 is not None  # the frame's own zero point calibrates its sky
    assert abs(record.sky_mag_arcsec2 - true_sky_mag()) < 0.03
    assert "sky_mag_arcsec2" not in record.quality
    assert record.provenance["zp_ref_provisional"] == "true"


def test_a_strict_reference_leaves_the_provisional_mark_out(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth], tmp_path: Path
) -> None:
    record = sky_record(
        analyze(
            profile,
            catalog,
            *clear,
            library=dark_library(tmp_path),
            reference=reference_at(),
            solve=False,
        )
    )
    assert record.provenance["zp_ref"] == "19.3000"
    assert "zp_ref_provisional" not in record.provenance
    assert record.quality is not None
    assert "provisional" not in record.quality["sky_mag_arcsec2"]


def test_a_provisional_zero_point_does_not_set_the_expected_signal_of_the_cloud_fraction(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth]
) -> None:
    """The cloud fraction keeps the profile's prior, as it does while no reference exists."""
    far_off = 15.0  # a zero point so low that a reference of it expects no star at all
    prior = analyze(profile, catalog, *clear, sky_quality=False)
    provisional = analyze(
        profile, catalog, *clear, reference=provisional_at(far_off), sky_quality=False
    )
    strict = analyze(profile, catalog, *clear, reference=reference_at(far_off), sky_quality=False)
    assert prior.cloud_fraction is not None
    assert provisional.cloud_fraction == prior.cloud_fraction
    assert strict.cloud_fraction is None  # too few expected stars: the reference does count


# --- The nightly summary ------------------------------------------------------------------


def test_the_stars_of_a_frame_carry_their_offsets_and_magnitudes(
    clear_analysis: FrameAnalysis, catalog: CapCatalog
) -> None:
    stars = clear_analysis.epoch_stars
    assert stars is not None
    assert len(stars) > 100
    g = catalog.g_mag[stars.cat_row]
    assert float(np.sqrt(np.mean((stars.mag - g) ** 2))) < 0.04  # the Gaia G of each star
    assert float(np.median(stars.mag - g)) == pytest.approx(0.0, abs=0.01)
    assert float(np.sqrt(np.mean(stars.dx_arcsec**2))) < 0.5  # arcseconds: a tenth of a pixel
    assert float(np.sqrt(np.mean(stars.dy_arcsec**2))) < 0.5
    assert len(set(stars.cat_row.tolist())) == len(stars)


def test_a_cloudy_frame_adds_nothing_to_the_nightly_summary(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    frame, truth = render(profile, catalog, transmission=0.3, seed=9)
    analysis = analyze(
        profile, catalog, frame, truth, library=dark_library(tmp_path), reference=reference_at()
    )
    assert "cloud" in sky_record(analysis).flags
    assert analysis.epoch_stars is not None
    assert len(analysis.epoch_stars) == 0


# --- A window of the sensor, the flat, and the hot pixels ---------------------------------


def test_a_window_of_the_sensor_gives_the_same_zero_point_and_sky(
    profile: Profile, catalog: CapCatalog, clear: tuple[Frame, synth.SynthTruth], tmp_path: Path
) -> None:
    frame, truth = clear
    window = Roi(300, 200, 1000, 700)
    cropped = replace(
        frame,
        data=np.ascontiguousarray(frame.data[200:900, 300:1300], dtype=np.uint16),
        roi=window,
    )
    record = sky_record(analyze(profile, catalog, cropped, truth, library=dark_library(tmp_path)))
    assert record.zero_point_mag is not None
    assert abs(record.zero_point_mag - ZP_TRUE) < 0.03
    assert record.sky_mag_arcsec2 is not None
    assert abs(record.sky_mag_arcsec2 - true_sky_mag()) < 0.1


def vignette(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    radius2 = ((x - (WIDTH - 1) / 2.0) ** 2 + (y - (HEIGHT - 1) / 2.0) ** 2) / (
        (WIDTH / 2.0) ** 2 + (HEIGHT / 2.0) ** 2
    )
    return 1.0 - 0.25 * radius2


def test_a_flat_field_removes_the_vignetting_from_the_zero_point_scatter(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    frame, truth = render(profile, catalog, transmission=vignette, seed=10)
    library = dark_library(tmp_path)
    plain = sky_record(analyze(profile, catalog, frame, truth, library=library))
    columns, rows = np.meshgrid(np.arange(WIDTH, dtype=float), np.arange(HEIGHT, dtype=float))
    flat: FlatModel = ArrayFlat(vignette(columns, rows))
    corrected = sky_record(analyze(profile, catalog, frame, truth, library=library, flat=flat))
    assert plain.zero_point_rms_mag is not None
    assert corrected.zero_point_rms_mag is not None
    assert plain.zero_point_rms_mag > 0.05  # stars in the corners are 25% dimmer than at the middle
    assert corrected.zero_point_rms_mag < 0.03
    assert corrected.provenance["flat"].startswith("flat-")
    assert plain.provenance["flat"] == "unit"


def test_hot_pixels_in_the_dark_library_stay_out_of_the_detections(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    frame, truth = render(profile, catalog, n_hot_pixels=300, hot_e_per_s=40.0, seed=11)
    master = np.full(
        (HEIGHT, WIDTH), round(BIAS_DN + DARK_E_PER_S_PX * EXPOSURE_S / E_PER_ADU), dtype=np.uint16
    )
    master[truth.hot_pixels] += 1000  # the dark library knows the hot pixels
    library = DarkLibrary(tmp_path / "calibration" / "darks")
    library.add_set(
        master,
        mode="bin2",
        gain=120,
        exposure_s=EXPOSURE_S,
        temperature_c=TEMPERATURE_C,
        temperature_spread_c=0.2,
        t_utc_ns=synth.NIGHT_UTC_NS - 3600 * NS_PER_S,
        n_frames=9,
        n_bias_frames=9,
        bias_dn=BIAS_DN,
        read_noise_dn=2.1,
        adc_bits=14,
        dark_dn=BIAS_DN + DARK_E_PER_S_PX * EXPOSURE_S / E_PER_ADU,
    )
    known = analyze(profile, catalog, frame, truth, library=library)
    unknown = analyze(profile, catalog, frame, truth, library=dark_library(tmp_path / "other"))
    assert known.detections is not None
    assert unknown.detections is not None
    assert len(known.detections) < len(unknown.detections)  # the masked hot pixels are not stars
    record = sky_record(known)
    assert record.zero_point_mag is not None
    assert abs(record.zero_point_mag - ZP_TRUE) < 0.03


def test_a_short_exposure_gets_no_sky_quality_and_costs_nothing_for_it(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    """The alignment helper analyzes frames of a second or less: it must not pay for the sky.

    The limit (`[survey.sky] min_exposure_s`, 1 s) is the shortest adaptive long exposure of
    twilight, so a frame of 0.5 s, such as an alignment frame, gets no sky quality.
    """
    frame, truth = render(profile, catalog, exposure_s=0.5, seed=13)
    library = dark_library(tmp_path)
    short = analyze(profile, catalog, frame, truth, library=library)
    assert [r.record_type for r in short.records] == ["survey_frame", "pointing", "star_list"]
    assert short.quality is None
    assert short.epoch_stars is not None
    assert len(short.epoch_stars) == 0
    assert "quality" not in short.timings
    assert short.solved  # the pointing and the cloud fraction are unaffected
    # A caller can force the step, or lower the limit in the configuration.
    forced = analyze(profile, catalog, frame, truth, library=library, sky_quality=True)
    assert "sky_quality" in [r.record_type for r in forced.records]
    assert "quality" in forced.timings
    config = SurveyConfig(sky=SkyConfig(min_exposure_s=0.25))
    lowered = analyze(profile, catalog, frame, truth, library=library, config=config)
    assert "sky_quality" in [r.record_type for r in lowered.records]
    # And a long frame can be switched off.
    clear_frame, clear_truth = render(profile, catalog, seed=14)
    off = analyze(profile, catalog, clear_frame, clear_truth, library=library, sky_quality=False)
    assert "sky_quality" not in [r.record_type for r in off.records]


def test_the_sky_quality_step_is_a_small_part_of_the_frame_time(
    clear_analysis: FrameAnalysis,
) -> None:
    timings: Callable[[str], float] = clear_analysis.timings.__getitem__
    assert timings("quality") >= 0.0
    assert set(clear_analysis.timings) == {"detect", "solve", "match", "quality", "records"}
