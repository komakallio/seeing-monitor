"""`CoreApp` and the darkness of the sky: the events from real survey records, and a restart.

The records come from the survey pipeline itself. The synthetic renderer of the survey tests draws
a clear and a cloudy frame of a dark sky, with a dark library that matches the renderer, and the
pipeline measures them (`tests/survey/test_quality.py` checks those values against the truth). A
survey analyzer in the rig hands the records out again for the long frames of the scheduler, with
the time of each frame, so `core` sees real `sky_quality` records at the cadence of a night.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.analysis import SurveyOutput
from seeingmon.clock import NS_PER_S
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.scheduler.ephemeris import sun_elevation_deg
from seeingmon.services.core.darkness import CLEAR_VERDICT_EVENT, DARK_EVENT, SkyDarkness
from seeingmon.survey.pipeline import FrameAnalysis
from seeingmon.testing import FakeSurveyAnalyzer
from tests.survey import synth
from tests.survey.test_quality import HEIGHT, WIDTH, analyze, dark_library, render

from .rig import NIGHT, CoreRig, build_rig

# Three solved frames make the sky dark, and two more give the verdict: five survey steps.
DARKNESS = "[survey.darkness]\nframes = 3\nverdict_frames = 2\n"
LONG_US = 1_000_000  # the scheduler's long survey frame is 30 s, the brightness frame 1 ms
CLEAR_THRESHOLD = 0.3  # [scheduler.cloud] clear_threshold


class MeasuredSurvey(FakeSurveyAnalyzer):
    """Hands out measured records for the long frames, in the order of `script`, at their times."""

    def __init__(self, script: list[FrameAnalysis]) -> None:
        super().__init__(station_id="test", profile_id="test")
        self.script = list(script)
        self.times: list[int] = []

    def _output(self, item: Any) -> SurveyOutput:
        frame = item.frame
        if frame.exposure_us < LONG_US or not self.script:
            return super()._output(item)
        analysis = self.script.pop(0)
        update = {"t_utc_ns": frame.t_utc_ns, "station_id": "test"}
        records = tuple(record.model_copy(update=update) for record in analysis.records)
        self.times.append(frame.t_utc_ns)
        return SurveyOutput(frame.t_utc_ns, records, analysis.solved, analysis.cloud_fraction)


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> dict[str, FrameAnalysis]:
    profile = synth.cropped_profile(WIDTH, HEIGHT)
    catalog = synth.synthetic_catalog(cap_radius_deg=8.0, density_scale=1.0, seed=3)
    library = dark_library(tmp_path_factory.mktemp("darks"))
    found: dict[str, FrameAnalysis] = {}
    for name, transmission, seed in (("clear", 1.0, 5), ("cloudy", 0.003, 12)):
        frame, truth = render(profile, catalog, transmission=transmission, seed=seed)
        found[name] = analyze(profile, catalog, frame, truth, library=library)
    return found


def sky_of(analysis: FrameAnalysis) -> SkyQualityRecord:
    return next(r for r in analysis.records if isinstance(r, SkyQualityRecord))


def run_long_frames(rig: CoreRig, survey: MeasuredSurvey, count: int) -> None:
    """Step the scheduler until the analyzer has handed out `count` long frames."""
    rig.app.start()
    for _ in range(60):
        rig.run_for(60.0)
        if len(survey.times) >= count:
            return
    raise AssertionError(f"the analyzer handed out {len(survey.times)} long frames of {count}")


class TestTheEventsFromRealRecords:
    def test_core_writes_sky_dark_and_the_verdict_from_the_survey_records(
        self, tmp_path: Path, measured: dict[str, FrameAnalysis]
    ) -> None:
        clear, cloudy = measured["clear"], measured["cloudy"]
        clear_sky = sky_of(clear)
        assert clear_sky.sky_mag_arcsec2 is not None  # the dark library gives the frame a sky
        assert clear_sky.cloud_fraction == 0.0
        cloudy_fraction = sky_of(cloudy).cloud_fraction
        assert cloudy_fraction is not None
        assert cloudy_fraction > CLEAR_THRESHOLD  # a frame that is not clear
        survey = MeasuredSurvey([clear, clear, clear, cloudy, clear])
        rig = build_rig(tmp_path, config_extra=DARKNESS, parts={"survey": survey})
        try:
            assert isinstance(rig.app.darkness, SkyDarkness)
            run_long_frames(rig, survey, 5)
            (dark,) = rig.events(DARK_EVENT)
            assert dark.t_utc_ns == survey.times[2]  # the third solved frame
            assert dark.detail["sky_mag_arcsec2"] == pytest.approx(
                clear_sky.sky_mag_arcsec2, abs=1e-3
            )
            assert dark.detail["slope_mag_per_hour"] == pytest.approx(0.0, abs=1e-9)
            assert dark.detail["frames"] == 3
            sun = sun_elevation_deg(dark.t_utc_ns, 55.0, 0.0)  # the site of the rig
            assert dark.detail["sun_elevation_deg"] == pytest.approx(sun, abs=0.01)
            assert dark.provenance["source"] == "local"
            (verdict,) = rig.events(CLEAR_VERDICT_EVENT)
            assert verdict.t_utc_ns == survey.times[4]
            assert verdict.detail["clear_share"] == 0.5  # the cloudy frame and a clear one
            assert verdict.detail["frames"] == 2
            assert verdict.detail["transparency_median"] is None  # no reference zero point yet
            stored = rig.records("sky_quality")
            assert [r.t_utc_ns for r in stored] == survey.times  # the records reached the store
            assert stored[0].n_expected == clear_sky.n_expected  # type: ignore[attr-defined]
        finally:
            rig.app.stop()

    def test_a_restart_in_the_same_night_writes_neither_event_again(
        self, tmp_path: Path, measured: dict[str, FrameAnalysis]
    ) -> None:
        clear, cloudy = measured["clear"], measured["cloudy"]
        first = MeasuredSurvey([clear, clear, clear, cloudy])
        rig = build_rig(tmp_path, config_extra=DARKNESS, parts={"survey": first})
        try:
            run_long_frames(rig, first, 4)
            assert len(rig.events(DARK_EVENT)) == 1  # and the verdict has one frame of two
        finally:
            rig.app.stop()
        again = rig.clock.utc_ns() + 600 * NS_PER_S
        assert again < NIGHT + 12 * 3600 * NS_PER_S  # before the split hour: the same night
        second = MeasuredSurvey([clear] * 5)
        rig = build_rig(
            tmp_path, start_utc_ns=again, config_extra=DARKNESS, parts={"survey": second}
        )
        try:
            assert rig.app.darkness.watch.dark_utc_ns == first.times[2]
            run_long_frames(rig, second, 5)
            assert len(rig.events(DARK_EVENT)) == 1  # no second `sky.dark`
            (verdict,) = rig.events(CLEAR_VERDICT_EVENT)
            assert verdict.t_utc_ns == second.times[0]  # the frame before the restart counts
            assert verdict.detail["clear_share"] == 0.5
        finally:
            rig.app.stop()
