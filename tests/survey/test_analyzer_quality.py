"""The analyzer with the sky quality: the history, the nightly summary, and the dark library."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from seeingmon.analysis import SurveyOutput
from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.frames import Frame, Roi, TimeQuality
from seeingmon.profile import Profile
from seeingmon.records.survey import (
    SkyQualityRecord,
    StarEpochRecord,
    SurveyFrameRecord,
    star_rows,
)
from seeingmon.store.layout import DataLayout
from seeingmon.survey.analyzer import (
    InlineExecutor,
    SurveyPipelineAnalyzer,
    create_survey_analyzer,
)
from seeingmon.survey.catalog import CapCatalog, write_catalog
from seeingmon.survey.config import SurveyConfig, TransparencyConfig
from seeingmon.survey.dark import DarkLibrary
from seeingmon.survey.pipeline import FrameAnalysis, SurveyPipeline
from seeingmon.survey.star_epoch import FrameStars
from seeingmon.survey.transparency import (
    MemoryHistory,
    ZeroPointReference,
    ZeroPointSample,
)
from tests.survey import synth

EVENING = iso_to_utc_ns("2026-10-01T19:00:00Z")
STEP_NS = 180 * NS_PER_S


@pytest.fixture(scope="module")
def profile() -> Profile:
    return synth.cropped_profile(1600, 1200)


@pytest.fixture(scope="module")
def catalog() -> CapCatalog:
    return synth.synthetic_catalog(cap_radius_deg=8.0, density_scale=1.0, seed=3)


class ScriptedQuality(SurveyPipeline):
    """A pipeline that returns a scripted zero point and stars for each frame."""

    def __init__(self, profile: Profile, catalog: CapCatalog) -> None:
        self._profile = profile
        self._catalog = catalog
        self.references: list[ZeroPointReference | None] = []
        self.zero_point = 19.3
        self.stars_for: dict[int, FrameStars] = {}

    def analyze(
        self,
        frame: Frame,
        *,
        previous: object = None,
        reference: object = None,
        index: int = 0,
        zp_reference: ZeroPointReference | None = None,
        sky_quality: bool | None = None,
    ) -> FrameAnalysis:
        self.references.append(zp_reference)
        survey = SurveyFrameRecord(
            station_id="test",
            t_utc_ns=frame.t_utc_ns,
            profile_id=self._profile.id,
            provenance={"algo": "scripted"},
            exposure_s=30.0,
            gain=120,
            readout_mode="bin2",
        )
        sky = SkyQualityRecord(
            station_id="test",
            t_utc_ns=frame.t_utc_ns,
            profile_id=self._profile.id,
            provenance={"algo": "scripted"},
            zero_point_mag=self.zero_point,
            zero_point_rms_mag=0.02,
            n_stars_used=60,
            cloud_fraction=0.0,
        )
        return FrameAnalysis(
            records=(survey, sky),
            solved=True,
            cloud_fraction=0.0,
            epoch_stars=self.stars_for.get(index, FrameStars.empty()),
        )


def small_frame(t_utc_ns: int) -> Frame:
    return Frame(
        data=np.zeros((4, 4), dtype=np.uint16),
        stream_id=1,
        seq=0,
        t_arrival_ns=t_utc_ns,
        t_utc_ns=t_utc_ns,
        t_err_ns=0,
        t_quality=TimeQuality.EXACT,
        dropped_before=0,
        exposure_us=30_000_000,
        gain=120,
        mode="bin2",
        roi=Roi(0, 0, 4, 4),
        adc_bits=14,
    )


def scripted(
    profile: Profile, catalog: CapCatalog, **options: object
) -> tuple[SurveyPipelineAnalyzer, ScriptedQuality]:
    pipeline = ScriptedQuality(profile, catalog)
    analyzer = SurveyPipelineAnalyzer(
        profile=profile,
        station_id="test",
        pipeline=pipeline,
        executor=InlineExecutor(),
        **options,  # type: ignore[arg-type]
    )
    return analyzer, pipeline


# --- The real pipeline: submit and poll give the sky quality ---------------------------------


def test_submit_and_poll_give_the_sky_quality_and_the_cloud_fraction(
    profile: Profile, catalog: CapCatalog
) -> None:
    frame, truth = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        zero_point_mag=19.3,
        color_term=-0.08,
        star_scatter=0.01,
        seed=5,
    )
    pipeline = SurveyPipeline(
        station_id="test",
        profile=profile,
        catalog=catalog,
        solvers=[synth.QueueSolver([synth.truth_solve_result(truth, catalog)])],
    )
    analyzer = SurveyPipelineAnalyzer(
        profile=profile, station_id="test", pipeline=pipeline, executor=InlineExecutor()
    )
    analyzer.submit(frame)
    (output,) = analyzer.poll()
    assert [r.record_type for r in output.records] == [
        "survey_frame",
        "sky_quality",
        "pointing",
        "star_list",
    ]
    sky = next(r for r in output.records if isinstance(r, SkyQualityRecord))
    assert sky.zero_point_mag is not None
    assert abs(sky.zero_point_mag - 19.3) < 0.03
    assert output.solved
    assert output.cloud_fraction == sky.cloud_fraction  # the same number in both places
    assert output.cloud_fraction is not None
    assert output.cloud_fraction < 0.1
    analyzer.close()


# --- The history and the reference -----------------------------------------------------------


def test_the_analyzer_feeds_its_history_and_the_reference_appears_after_enough_frames(
    profile: Profile, catalog: CapCatalog
) -> None:
    analyzer, pipeline = scripted(profile, catalog)
    for index in range(24):
        analyzer.submit(small_frame(EVENING + index * STEP_NS))
        assert len(analyzer.poll()) == 1
    # The first frame finds no history. The 19 that follow get the provisional zero point of the
    # frames before them, and the 21st is the first to get the reference.
    assert pipeline.references[0] is None
    for index in range(1, 20):
        stand_in = pipeline.references[index]
        assert stand_in is not None
        assert stand_in.provisional
        assert stand_in.n_samples == index
        assert stand_in.zero_point_mag == pytest.approx(19.3)
    first = pipeline.references[20]
    assert first is not None
    assert not first.provisional
    reference = pipeline.references[-1]
    assert reference is not None
    assert not reference.provisional
    assert reference.zero_point_mag == pytest.approx(19.3)
    assert reference.n_samples >= 20
    samples = analyzer.history.zero_points(0, 2**62)
    assert len(samples) == 24
    assert samples[0].zero_point_mag == 19.3
    assert samples[0].n_stars == 60
    analyzer.close()


def test_a_seeded_history_gives_the_reference_to_the_first_frame(
    profile: Profile, catalog: CapCatalog
) -> None:
    history = MemoryHistory(
        ZeroPointSample(EVENING - (i + 1) * 3600 * NS_PER_S, 19.2, 0.02, 80, 0.0) for i in range(30)
    )
    analyzer, pipeline = scripted(profile, catalog, history=history)
    analyzer.submit(small_frame(EVENING))
    assert len(analyzer.poll()) == 1
    reference = pipeline.references[0]
    assert reference is not None
    assert reference.zero_point_mag == pytest.approx(19.2)
    assert reference.n_samples == 30
    assert analyzer.history is history
    # The analyzer reads a history that you pass and leaves the writing to you.
    assert len(history) == 30
    analyzer.close()


def test_a_history_that_only_reads_works_as_well(profile: Profile, catalog: CapCatalog) -> None:
    class StoreReader:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def zero_points(self, since_utc_ns: int, until_utc_ns: int) -> list[ZeroPointSample]:
            self.calls.append((since_utc_ns, until_utc_ns))
            return [
                ZeroPointSample(since_utc_ns + i * 1000, 19.1, 0.02, 70, 0.0) for i in range(25)
            ]

    reader = StoreReader()
    analyzer, pipeline = scripted(profile, catalog, history=reader)
    analyzer.submit(small_frame(EVENING))
    analyzer.poll()
    assert reader.calls == [(EVENING - 365 * 86400 * NS_PER_S, EVENING)]  # a year of history
    assert pipeline.references[0] is not None
    assert pipeline.references[0].zero_point_mag == pytest.approx(19.1)
    analyzer.close()


# --- The provisional zero point ---------------------------------------------------------------


class RecordingHistory:
    """A history that answers from a list of samples and records the questions."""

    def __init__(self, samples: list[ZeroPointSample]) -> None:
        self.calls: list[tuple[int, int]] = []
        self._memory = MemoryHistory(samples)

    def zero_points(self, since_utc_ns: int, until_utc_ns: int) -> tuple[ZeroPointSample, ...]:
        self.calls.append((since_utc_ns, until_utc_ns))
        return self._memory.zero_points(since_utc_ns, until_utc_ns)


def zero_points_before(t_utc_ns: int, values: list[float]) -> list[ZeroPointSample]:
    """Good samples, 50 minutes apart, the newest 50 minutes before `t_utc_ns`."""
    return [
        ZeroPointSample(t_utc_ns - (i + 1) * 3000 * NS_PER_S, value, 0.03, 60, 0.0)
        for i, value in enumerate(values)
    ]


SIX_GOOD_FRAMES = [18.70, 18.82, 18.95, 18.74, 18.88, 18.78]  # their median is 18.80


def test_a_short_history_gives_the_worker_the_provisional_zero_point(
    profile: Profile, catalog: CapCatalog
) -> None:
    history = RecordingHistory(zero_points_before(EVENING, SIX_GOOD_FRAMES))
    analyzer, pipeline = scripted(profile, catalog, history=history)
    analyzer.submit(small_frame(EVENING))
    assert len(analyzer.poll()) == 1
    (reference,) = pipeline.references
    assert reference is not None
    assert reference.provisional
    assert reference.n_samples == 6
    assert reference.zero_point_mag == pytest.approx(18.80)  # the median, not the 95th percentile
    assert reference.quantile == 0.5
    assert reference.window_days == pytest.approx(0.25)
    # The analyzer asks for the year of the reference first, then for the last 6 hours, and last for
    # the count of the year that the message of the transparency names.
    assert history.calls == [
        (EVENING - 365 * 86400 * NS_PER_S, EVENING),
        (EVENING - 6 * 3600 * NS_PER_S, EVENING),
        (EVENING - 365 * 86400 * NS_PER_S, EVENING),
    ]
    analyzer.close()


def test_the_reference_wins_over_the_provisional_zero_point(
    profile: Profile, catalog: CapCatalog
) -> None:
    values = [19.0 + i / 100.0 for i in range(25)]
    history = RecordingHistory(zero_points_before(EVENING, values))
    analyzer, pipeline = scripted(profile, catalog, history=history)
    analyzer.submit(small_frame(EVENING))
    assert len(analyzer.poll()) == 1
    (reference,) = pipeline.references
    assert reference is not None
    assert not reference.provisional
    assert reference.n_samples == 25
    assert reference.zero_point_mag == pytest.approx(float(np.quantile(values, 0.95)))
    assert reference.zero_point_mag > float(np.median(values)) + 0.05  # not the stand-in
    assert history.calls == [(EVENING - 365 * 86400 * NS_PER_S, EVENING)]  # no second question
    analyzer.close()


def test_zero_points_older_than_the_fallback_window_give_no_provisional_zero_point(
    profile: Profile, catalog: CapCatalog
) -> None:
    old = [
        ZeroPointSample(EVENING - (7 + i) * 3600 * NS_PER_S, 18.8, 0.03, 60, 0.0) for i in range(5)
    ]
    analyzer, pipeline = scripted(profile, catalog, history=MemoryHistory(old))
    analyzer.submit(small_frame(EVENING))
    assert len(analyzer.poll()) == 1
    assert pipeline.references == [None]
    analyzer.close()
    longer = SurveyConfig(transparency=TransparencyConfig(fallback_hours=12.0))
    analyzer, pipeline = scripted(profile, catalog, history=MemoryHistory(old), config=longer)
    analyzer.submit(small_frame(EVENING))
    assert len(analyzer.poll()) == 1
    (reference,) = pipeline.references
    assert reference is not None
    assert reference.n_samples == 5
    assert reference.window_days == pytest.approx(0.5)
    analyzer.close()


def test_a_fallback_of_zero_hours_leaves_a_short_history_without_a_reference(
    profile: Profile, catalog: CapCatalog
) -> None:
    history = RecordingHistory(zero_points_before(EVENING, SIX_GOOD_FRAMES))
    off = SurveyConfig(transparency=TransparencyConfig(fallback_hours=0.0))
    analyzer, pipeline = scripted(profile, catalog, history=history, config=off)
    analyzer.submit(small_frame(EVENING))
    assert len(analyzer.poll()) == 1
    assert pipeline.references == [None]
    assert history.calls == [(EVENING - 365 * 86400 * NS_PER_S, EVENING)]  # the window is not read
    analyzer.close()


def test_a_frame_without_a_pointing_solution_keeps_its_sky_brightness_while_the_history_is_short(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    """The history holds six good zero points, and the pointing solution is lost."""
    write_catalog(tmp_path / "cap.smcat", catalog)
    layout = DataLayout(tmp_path / "data")
    DarkLibrary.from_layout(layout).add_set(
        np.full((1200, 1600), 40, dtype=np.uint16),
        mode="bin2",
        gain=120,
        exposure_s=30.0,
        temperature_c=15.0,
        temperature_spread_c=0.1,
        t_utc_ns=synth.NIGHT_UTC_NS - 3600 * NS_PER_S,
        n_frames=9,
        n_bias_frames=9,
        bias_dn=40.0,
        read_noise_dn=2.1,
        adc_bits=14,
        dark_dn=40.0,
    )
    frame, _ = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        zero_point_mag=19.3,
        sky_e_per_s_px=3.0,
        seed=8,
    )
    history = MemoryHistory(zero_points_before(frame.t_utc_ns, SIX_GOOD_FRAMES))
    config = SurveyConfig(catalog_path=str(tmp_path / "cap.smcat"), solvers=())
    analyzer = create_survey_analyzer(
        profile=profile,
        station_id="test",
        config=config,
        executor=InlineExecutor(),
        layout=layout,
        history=history,
    )
    analyzer.submit(frame)
    (output,) = analyzer.poll()
    sky = next(r for r in output.records if isinstance(r, SkyQualityRecord))
    assert not output.solved  # no solver and no tracker: the pointing is lost
    assert sky.zero_point_mag is None
    assert sky.sky_rate_e_per_s_arcsec2 is not None
    assert sky.sky_mag_arcsec2 is not None
    expected = 18.80 - 2.5 * math.log10(sky.sky_rate_e_per_s_arcsec2)  # the median of the six
    assert sky.sky_mag_arcsec2 == pytest.approx(expected, abs=1e-6)
    assert sky.quality is not None
    assert "provisional zero point" in sky.quality["sky_mag_arcsec2"]
    assert "the median of 6 frames of the last 6 h" in sky.quality["sky_mag_arcsec2"]
    assert sky.transparency is None
    assert sky.provenance["zp_ref_provisional"] == "true"
    analyzer.close()


# --- The nightly summary ---------------------------------------------------------------------


def test_the_next_night_closes_the_last_one_and_flush_closes_the_open_night(
    profile: Profile, catalog: CapCatalog
) -> None:
    analyzer, pipeline = scripted(profile, catalog)
    tonight = [EVENING + i * STEP_NS for i in range(4)]
    tomorrow = [iso_to_utc_ns("2026-10-02T19:00:00Z") + i * STEP_NS for i in range(3)]
    for index in range(len(tonight + tomorrow)):
        offset = 0.2 if index < 4 else -0.3
        pipeline.stars_for[index] = FrameStars.build(
            [10, 11], [offset, 0.0], [0.1, -0.1], [11.0 + 0.01 * index, 12.0]
        )
    outputs: list[SurveyOutput] = []
    for t in tonight + tomorrow:
        analyzer.submit(small_frame(t))
        outputs.extend(analyzer.poll())
    epochs = [r for output in outputs for r in output.records if isinstance(r, StarEpochRecord)]
    assert len(epochs) == 1  # only the first night is closed so far
    first = epochs[0]
    assert (first.night, first.n_frames, first.n_stars, first.t_utc_ns) == (
        "2026-10-01",
        4,
        2,
        tonight[0],
    )
    # The record rides on the output of the first frame of the next night.
    carrier = next(o for o in outputs if any(isinstance(r, StarEpochRecord) for r in o.records))
    assert carrier.t_utc_ns == tomorrow[0]
    table = star_rows(first)
    np.testing.assert_allclose(table[:, 1], [0.2, 0.0], atol=1e-6)  # the mean offset of the night
    np.testing.assert_allclose(table[:, 3], [11.015, 12.0], atol=1e-4)
    assert first.provenance["algo"] == "epoch-1"
    assert len(first.provenance["catalog"]) == 8
    (last,) = analyzer.flush_night()
    assert isinstance(last, StarEpochRecord)
    assert (last.night, last.n_frames) == ("2026-10-02", 3)
    np.testing.assert_allclose(star_rows(last)[:, 1], [-0.3, 0.0], atol=1e-6)
    assert analyzer.flush_night() == ()  # nothing is open now
    analyzer.close()


def test_the_analyzer_still_works_when_the_catalog_cannot_be_read(
    profile: Profile, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = SurveyConfig(catalog_path=str(tmp_path / "missing.smcat"), solvers=())
    with caplog.at_level("WARNING", logger="seeingmon.survey"):
        analyzer = create_survey_analyzer(
            profile=profile, station_id="test", config=config, executor=InlineExecutor()
        )
    assert "no nightly star summary" in caplog.text
    assert analyzer.flush_night() == ()
    analyzer.close()


# --- The dark library from the data layout ---------------------------------------------------


def test_the_data_layout_supplies_the_dark_library(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    write_catalog(tmp_path / "cap.smcat", catalog)
    layout = DataLayout(tmp_path / "data")
    library = DarkLibrary.from_layout(layout)
    library.add_set(
        np.full((1200, 1600), 40, dtype=np.uint16),
        mode="bin2",
        gain=120,
        exposure_s=30.0,
        temperature_c=15.0,
        temperature_spread_c=0.1,
        t_utc_ns=synth.NIGHT_UTC_NS - 3600 * NS_PER_S,
        n_frames=9,
        n_bias_frames=9,
        bias_dn=40.0,
        read_noise_dn=2.1,
        adc_bits=14,
        dark_dn=40.0,
    )
    frame, _ = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        zero_point_mag=19.3,
        sky_e_per_s_px=3.0,
        seed=8,
    )
    config = SurveyConfig(catalog_path=str(tmp_path / "cap.smcat"), solvers=())
    with_layout = create_survey_analyzer(
        profile=profile,
        station_id="test",
        config=config,
        executor=InlineExecutor(),
        layout=layout,
    )
    without = create_survey_analyzer(
        profile=profile, station_id="test", config=config, executor=InlineExecutor()
    )
    for analyzer in (with_layout, without):
        analyzer.submit(frame)
    sky_with = next(r for r in with_layout.poll()[0].records if isinstance(r, SkyQualityRecord))
    sky_without = next(r for r in without.poll()[0].records if isinstance(r, SkyQualityRecord))
    assert sky_with.dark_model_version == library.version()
    assert "dark_due" not in sky_with.flags  # the set is at the frame's temperature
    assert sky_without.dark_model_version is None
    assert "dark_due" in sky_without.flags
    for analyzer in (with_layout, without):
        analyzer.close()


def test_a_configured_calibration_folder_wins_over_the_layout(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    from seeingmon.survey.analyzer import with_calibration

    layout = DataLayout(tmp_path / "data")
    config = SurveyConfig(calibration_dir=str(tmp_path / "mine"))
    assert with_calibration(config, layout).calibration_dir == str(tmp_path / "mine")
    assert with_calibration(SurveyConfig(), layout).calibration_dir == str(
        tmp_path / "data" / "calibration"
    )
    assert with_calibration(SurveyConfig(), None).calibration_dir == ""
