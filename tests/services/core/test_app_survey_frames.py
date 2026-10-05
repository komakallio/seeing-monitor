"""`CoreApp` and the survey frames: the wiring, the references in the store, and the shutdown."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("PIL", reason="the previews need Pillow")
pytest.importorskip("astropy", reason="the FITS files need astropy")

from seeingmon.clock import NS_PER_S, ScaledClock
from seeingmon.frames import Frame
from seeingmon.records import Record, SurveyFrameRecord
from seeingmon.services.core.alignment.preview import make_preview
from seeingmon.services.core.survey_frames import SurveyFrames
from seeingmon.services.web.contract import unpack_frame
from seeingmon.store.retention import DiskUsage
from seeingmon.survey import framefile
from seeingmon.testing import FakeSurveyAnalyzer

from . import previewfx
from .rig import NIGHT, SMALL_BIN2, CoreRig, build_rig

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


class TestThePreviewCalibration:
    SHAPE = (SMALL_BIN2[1], SMALL_BIN2[0])  # (rows, columns) of the simulated camera in bin2

    def rig(self, tmp_path: Path, *, files: bool) -> CoreRig:
        """A rig whose configuration names a flat, and whose data directory holds a dark set."""
        extra = ""
        if files:
            flat = previewfx.write_flat(tmp_path / "flat.npy", previewfx.sensitivity(self.SHAPE))
            extra = f'[survey]\nflat_file = "{flat.as_posix()}"\n'
        rig = build_rig(tmp_path, config_extra=extra)
        if files:
            previewfx.add_dark_set(rig.app.dark_library, self.SHAPE)
        return rig

    def frame(self) -> Frame:
        return previewfx.make_survey_frame(
            previewfx.sensitivity(self.SHAPE), t_utc_ns=NIGHT, exposure_s=30.0
        )

    def written_preview(self, rig: CoreRig, frame: Frame) -> bytes:
        """Run the frame through the writer of the app, and read the preview that it wrote."""
        frames = rig.app.frames
        assert frames is not None
        frames.submit(frame)
        frames.poll()
        frames.drain()
        layout = rig.app.storage.layout  # type: ignore[union-attr]
        return layout.preview_path(frame.t_utc_ns, kind="survey").read_bytes()

    def test_one_calibrator_from_the_configuration_serves_the_previews_and_the_live_view(
        self, tmp_path: Path
    ) -> None:
        rig = self.rig(tmp_path, files=True)
        try:
            frame = self.frame()
            step = rig.app.preview_calibrator.for_frame(frame)
            assert step is not None  # the flat file and the dark library reached the calibrator
            settings = rig.app.settings.survey_frames
            expected = make_preview(
                frame.data,
                max_pixels=settings.preview_max_pixels,
                quality=settings.jpeg_quality,
                calibration=step,
            ).jpeg
            assert self.written_preview(rig, frame) == expected
            live = rig.app.alignment_settings
            live_expected = make_preview(
                frame.data,
                max_pixels=live.max_preview_pixels,
                quality=live.jpeg_quality,
                calibration=step,
            ).jpeg
            assert unpack_frame(rig.app.alignment.process_frame(frame)).jpeg == live_expected
        finally:
            rig.app.stop()

    def test_a_flat_that_the_owner_activates_reaches_the_next_preview_and_the_live_view(
        self, tmp_path: Path
    ) -> None:
        rig = self.rig(tmp_path, files=True)
        try:

            def at(seconds: int) -> Frame:  # the same sky a minute later, for a new file name
                return previewfx.make_survey_frame(
                    previewfx.sensitivity(self.SHAPE),
                    t_utc_ns=NIGHT + seconds * NS_PER_S,
                    exposure_s=30.0,
                )

            first = at(0)
            before = self.written_preview(rig, first)
            live_before = unpack_frame(rig.app.alignment.process_frame(first)).jpeg
            # A flat of ones takes nothing out, so what it makes differs from what the file's does.
            entry = rig.app.flat_library.add(
                np.ones(self.SHAPE, dtype=np.float32), {"t_utc_ns": NIGHT, "mode": "bin2"}
            )
            assert self.written_preview(rig, at(60)) == before  # a pending flat changes nothing
            # The reader of the RPC checks the flat against the full frame of the profile, and this
            # rig takes small frames, so the library activates the flat as the reader would.
            rig.app.flat_library.activate(entry.version, now_utc_ns=NIGHT)
            later = at(120)
            after = self.written_preview(rig, later)
            live_after = unpack_frame(rig.app.alignment.process_frame(later)).jpeg
            assert after != before
            assert live_after != live_before
            step = rig.app.preview_calibrator.for_frame(later)
            assert getattr(step, "flat_version", None) == entry.version
        finally:
            rig.app.stop()

    def test_without_a_flat_and_a_dark_set_the_previews_stay_as_they_were(
        self, tmp_path: Path
    ) -> None:
        rig = self.rig(tmp_path, files=False)
        try:
            frame = self.frame()
            assert rig.app.preview_calibrator.for_frame(frame) is None
            settings = rig.app.settings.survey_frames
            plain = make_preview(
                frame.data, max_pixels=settings.preview_max_pixels, quality=settings.jpeg_quality
            ).jpeg
            assert self.written_preview(rig, frame) == plain
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
            ok = False
            for _ in range(100):  # a record has its reference while the writer writes the file
                refs = [r.image_ref for r in survey_frames(rig) if r.image_ref]
                ok = bool(refs) and all(layout.resolve(ref).is_file() for ref in refs)
                if ok:
                    break
                waited.wait(0.1)
            assert ok
        finally:
            rig.app.request_stop("a test")
            thread.join(60.0)
        assert not thread.is_alive()
        assert outcome == [0]
        assert not rig.app._threads["core-frames"].is_alive()
