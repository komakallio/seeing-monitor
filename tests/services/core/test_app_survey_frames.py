"""`CoreApp` and the survey frames: the wiring, the references in the store, and the shutdown."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("PIL", reason="the previews need Pillow")
pytest.importorskip("astropy", reason="the FITS files need astropy")

from seeingmon.clock import ScaledClock
from seeingmon.records import Record, SurveyFrameRecord
from seeingmon.services.core.survey_frames import SurveyFrames
from seeingmon.store.retention import DiskUsage
from seeingmon.survey import framefile
from seeingmon.testing import FakeSurveyAnalyzer

from .rig import NIGHT, CoreRig, build_rig

GB = 1_000_000_000


class NightFake(FakeSurveyAnalyzer):
    """A fake with a nightly summary, so that `CoreApp` wraps it a second time."""

    def flush_night(self) -> Sequence[Record]:
        return []


def survey_frames(rig: CoreRig) -> list[SurveyFrameRecord]:
    found = rig.records("survey_frame")
    assert all(isinstance(record, SurveyFrameRecord) for record in found)
    return found  # type: ignore[return-value]


def run_until_stored(rig: CoreRig, count: int) -> None:
    """Step the scheduler until `count` survey frames are in the store."""
    rig.app.start()
    for _ in range(60):
        rig.run_for(60.0)
        if len(survey_frames(rig)) >= count:
            return
    raise AssertionError(f"the store holds {len(survey_frames(rig))} survey frames of {count}")


class TestTheWiring:
    def test_the_frames_wrap_the_survey_analyzer(self, tmp_path: Path) -> None:
        rig = build_rig(tmp_path)
        try:
            assert isinstance(rig.app.frames, SurveyFrames)
            assert rig.app.survey is rig.app.frames  # the fake has no nightly summary
            assert rig.app.frames.pending() == 0
            assert rig.app.frames.submitted is rig.survey.submitted  # the analyzer is behind it
        finally:
            rig.app.stop()

    def test_a_nightly_summary_wraps_the_frames_and_keeps_its_nights(self, tmp_path: Path) -> None:
        analyzer = NightFake(station_id="test", profile_id="test")
        rig = build_rig(tmp_path, parts={"survey": analyzer})
        try:
            assert rig.app.nightly is not None
            assert rig.app.survey is rig.app.nightly
            assert isinstance(rig.app.frames, SurveyFrames)
            assert rig.app.nightly.flush() == 0  # the nights still reach the analyzer
        finally:
            rig.app.stop()

    def test_the_frames_can_be_switched_off(self, tmp_path: Path) -> None:
        rig = build_rig(tmp_path, config_extra="[services.core.survey_frames]\nenabled = false\n")
        try:
            assert rig.app.frames is None
            assert rig.app.survey is rig.survey
        finally:
            rig.app.stop()

    def test_the_settings_come_from_the_configuration(self, tmp_path: Path) -> None:
        extra = "[services.core.survey_frames]\nkeep_every = 3\nram_frames = 2\njpeg_quality = 60\n"
        rig = build_rig(tmp_path, config_extra=extra)
        try:
            settings = rig.app.settings.survey_frames
            assert (settings.keep_every, settings.ram_frames, settings.jpeg_quality) == (3, 2, 60)
        finally:
            rig.app.stop()


class TestASteppedRun:
    def test_the_files_reach_the_disk_and_the_references_reach_the_store(
        self, tmp_path: Path
    ) -> None:
        rig = build_rig(tmp_path, start_utc_ns=NIGHT)
        try:
            run_until_stored(rig, 4)
            layout = rig.app.storage.layout  # type: ignore[union-attr]
            frames = survey_frames(rig)
            long_frames = [r for r in frames if r.exposure_s >= 5.0]
            short_frames = [r for r in frames if r.exposure_s < 5.0]
            assert long_frames
            assert short_frames
            assert all(r.image_ref is None for r in short_frames)
            first = long_frames[0]
            assert first.image_ref is not None
            assert first.image_ref.startswith("survey/")  # the first long frame is kept
            assert first.image_ref.endswith(".fits")
            for record in long_frames[1:]:
                assert record.image_ref is not None
                assert record.image_ref.startswith("previews/")
            for record in long_frames:
                assert record.image_ref is not None
                assert layout.resolve(record.image_ref).is_file()
            back = framefile.read_frame_fits(layout.resolve(first.image_ref))
            assert back.header["KEPT"] == "every_10"
            assert back.header["STATION"] == "test"
            assert back.header["EXPTIME"] == pytest.approx(first.exposure_s)
            stats = rig.app.frames.stats  # type: ignore[union-attr]
            assert stats.fits_files == 1
            assert stats.previews == len(long_frames)
            assert stats.failures == 0
        finally:
            rig.app.stop()

    def test_the_stop_writes_the_files_that_wait(self, tmp_path: Path) -> None:
        rig = build_rig(tmp_path, start_utc_ns=NIGHT)
        rig.app.start()
        frames = rig.app.frames
        assert frames is not None
        for _ in range(100_000):  # step the scheduler by hand: no tick, so nothing drains
            rig.app.scheduler.step()
            if frames.queued:
                break
        else:
            raise AssertionError("the scheduler produced no survey frame")
        layout = rig.app.storage.layout  # type: ignore[union-attr]
        assert not list(layout.survey_dir.rglob("*.fits"))
        rig.app.stop("a test")
        assert len(list(layout.survey_dir.rglob("*.fits"))) == 1
        assert list(layout.previews_dir.rglob("*.jpg"))
        assert not list(layout.root.rglob("*.tmp"))

    def test_a_disk_that_is_nearly_full_keeps_the_preview_and_skips_the_fits(
        self, tmp_path: Path
    ) -> None:
        def nearly_full(path: Path) -> DiskUsage:
            return DiskUsage(total_bytes=100 * GB, free_bytes=GB // 2)

        rig = build_rig(tmp_path, start_utc_ns=NIGHT, parts={"disk_usage": nearly_full})
        try:
            run_until_stored(rig, 4)
            layout = rig.app.storage.layout  # type: ignore[union-attr]
            long_frames = [r for r in survey_frames(rig) if r.exposure_s >= 5.0]
            assert long_frames
            assert not list(layout.survey_dir.rglob("*.fits"))
            assert all(r.image_ref and r.image_ref.startswith("previews/") for r in long_frames)
            assert rig.app.frames.stats.skipped_low_space == 1  # type: ignore[union-attr]
        finally:
            rig.app.stop()


class TestTheWriterThread:
    def test_the_thread_writes_while_core_runs_and_ends_with_it(self, tmp_path: Path) -> None:
        clock = ScaledClock(
            start_utc_ns=NIGHT, origin_real_ns=__import__("time").time_ns(), speed=40.0
        )
        rig = build_rig(tmp_path, threads=True, clock=clock)
        outcome: list[int] = []
        thread = threading.Thread(target=lambda: outcome.append(rig.app.run()))
        thread.start()
        waited = threading.Event()
        try:
            layout = rig.app.storage.layout  # type: ignore[union-attr]
            for _ in range(600):
                if any(layout.survey_dir.rglob("*.fits")) and rig.app.frames.queued == 0:  # type: ignore[union-attr]
                    break
                waited.wait(0.1)
            assert list(layout.survey_dir.rglob("*.fits"))
            assert "core-frames" in rig.app._threads
            refs = [r.image_ref for r in survey_frames(rig) if r.image_ref]
            assert refs
            assert all(layout.resolve(ref).is_file() for ref in refs)
        finally:
            rig.app.request_stop("a test")
            thread.join(60.0)
        assert not thread.is_alive()
        assert outcome == [0]
        assert not rig.app._threads["core-frames"].is_alive()
