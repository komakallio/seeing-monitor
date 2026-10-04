"""The survey frames on disk: the ring in RAM, the previews, the FITS files, and the references."""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pytest

pytest.importorskip("PIL", reason="the previews need Pillow")
pytest.importorskip("astropy", reason="the FITS files need astropy")

from PIL import Image

from seeingmon.analysis import SurveyOutput
from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.frames import Frame
from seeingmon.profile import Profile, load_profile
from seeingmon.records import SkyQualityRecord, SurveyFrameRecord
from seeingmon.services.core import survey_frames
from seeingmon.services.core.alignment.calibration import PreviewCalibrator
from seeingmon.services.core.alignment.preview import make_preview
from seeingmon.services.core.settings import SurveyFrameSettings
from seeingmon.services.core.survey_frames import SurveyFrames
from seeingmon.store.layout import DataLayout
from seeingmon.survey import framefile
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.dark import DarkLibrary
from seeingmon.survey.sky import SkyError
from seeingmon.testing import FakeSurveyAnalyzer
from tests.scheduler.helpers import make_frame

from . import previewfx

START_NS = 1_790_000_000 * NS_PER_S  # 2026-09-21T14:13:20Z
STEP_NS = 180 * NS_PER_S
LONG_US = 30_000_000
SHORT_US = 1_000


@pytest.fixture(scope="module")
def profile() -> Profile:
    return load_profile("asi294mm-gs250")


def make_survey_frame(
    step: int,
    *,
    exposure_us: int = LONG_US,
    shape: tuple[int, int] = (120, 160),
    offset_ns: int = 0,
) -> Frame:
    """A 14-bit frame in the high bits of a 16-bit container, with a bright pixel."""
    rng = np.random.default_rng(step + 1)
    native = rng.normal(900.0, 12.0, shape).clip(0, 16383).astype(np.uint16)
    native[shape[0] // 3, shape[1] // 3] = 12000
    return make_frame(
        (native << 2).astype(np.uint16),
        mode="bin2",
        gain=120 if exposure_us >= 5_000_000 else 0,
        exposure_us=exposure_us,
        adc_bits=14,
        t_utc_ns=START_NS + step * STEP_NS + offset_ns,
        seq=step,
    )


class BrightFake(FakeSurveyAnalyzer):
    """A fake whose results report a sky background, in native counts."""

    background_dn: float = 100.0

    def _output(self, item: Any) -> SurveyOutput:
        output = super()._output(item)
        (record,) = output.records
        update = {"background_dn": self.background_dn}
        return dataclasses.replace(output, records=(record.model_copy(update=update),))


class SkyFake(BrightFake):
    """A fake whose results also carry a `sky_quality` record with a transparency."""

    transparency: float | None = 0.97

    def _output(self, item: Any) -> SurveyOutput:
        output = super()._output(item)
        sky = SkyQualityRecord(
            station_id="st",
            t_utc_ns=output.t_utc_ns,
            profile_id="p",
            provenance={"algo": "fake"},
            n_stars_used=0,
            transparency=self.transparency,
        )
        return dataclasses.replace(output, records=(*output.records, sky))


@dataclass
class Built:
    frames: SurveyFrames
    analyzer: FakeSurveyAnalyzer
    layout: DataLayout
    clock: VirtualClock
    events: list[tuple[str, str, str, Mapping[str, Any] | None]] = field(default_factory=list)

    def run(self, frame: Frame) -> SurveyOutput:
        """Submit a frame, collect its result, and write its files, as one scheduler step."""
        self.frames.submit(frame)
        (output,) = self.frames.poll()
        self.frames.drain()
        return output

    def reference(self, output: SurveyOutput) -> str | None:
        (record,) = output.records
        assert isinstance(record, SurveyFrameRecord)
        return record.image_ref

    def tree(self) -> list[str]:
        """Every file under the survey and preview folders, as references."""
        return sorted(
            self.layout.relative(p)
            for folder in (self.layout.survey_dir, self.layout.previews_dir)
            for p in folder.rglob("*")
            if p.is_file()
        )


def build(
    tmp_path: Path,
    profile: Profile | None,
    *,
    gate: Callable[[], bool] | None = None,
    analyzer: FakeSurveyAnalyzer | None = None,
    calibrator: PreviewCalibrator | None = None,
    **settings: Any,
) -> Built:
    layout = DataLayout(tmp_path / "data")
    layout.create()
    clock = VirtualClock(START_NS)
    chosen = analyzer or FakeSurveyAnalyzer(station_id="st", profile_id="p")
    built = Built(
        SurveyFrames(
            chosen,
            layout=layout,
            profile=profile,
            station_id="st",
            clock=clock,
            settings=SurveyFrameSettings(**settings),
            long_min_exposure_s=5.0,
            capture_allowed=gate,
            on_event=lambda *args: built.events.append(args),
            calibrator=calibrator,
        ),
        chosen,
        layout,
        clock,
    )
    return built


class TestTheFilesOfAFrame:
    def test_the_first_long_frame_gets_a_preview_and_a_fits_file(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile)
        frame = make_survey_frame(0)
        built.frames.submit(frame)
        (output,) = built.frames.poll()
        assert built.tree() == []  # submit and poll write nothing: the writer thread does
        assert built.frames.queued == 1
        assert built.frames.drain() == 1
        fits_path = built.layout.survey_path(frame.t_utc_ns)
        preview_path = built.layout.preview_path(frame.t_utc_ns, kind="survey")
        assert built.tree() == sorted(
            [built.layout.relative(fits_path), built.layout.relative(preview_path)]
        )
        assert built.reference(output) == built.layout.relative(fits_path)

    def test_the_names_follow_the_frame_time(self, tmp_path: Path, profile: Profile) -> None:
        built = build(tmp_path, profile)
        output = built.run(make_survey_frame(0))
        assert built.reference(output) == "survey/2026/09/21/20260921T141320.000Z.fits"
        assert "previews/2026/09/21/survey-20260921T141320.000Z.jpg" in built.tree()

    def test_a_later_long_frame_has_a_preview_and_the_reference_names_it(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile)
        built.run(make_survey_frame(0))
        output = built.run(make_survey_frame(1))
        assert built.reference(output) == "previews/2026/09/21/survey-20260921T141620.000Z.jpg"
        assert not built.layout.survey_path(START_NS + STEP_NS).exists()

    def test_a_short_frame_gets_no_file_and_no_reference(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile)
        output = built.run(make_survey_frame(0, exposure_us=SHORT_US))
        assert built.reference(output) is None
        assert built.tree() == []
        assert built.frames.queued == 0

    def test_an_event_frame_keeps_its_frame_under_the_event_kind(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile, keep_every=1000)
        built.run(make_survey_frame(0))  # the first long frame, kept by the rule
        built.analyzer.solved = False  # the pointing is lost from here on
        output = built.run(make_survey_frame(1))
        assert built.reference(output) == "survey/2026/09/21/20260921T141620.000Z.fits"
        assert "previews/2026/09/21/event-20260921T141620.000Z.jpg" in built.tree()
        back = framefile.read_frame_fits(built.layout.survey_path(START_NS + STEP_NS))
        assert back.header["KEPT"] == "event:unsolved"
        assert built.frames.stats.event_frames == 1

    def test_a_short_frame_of_a_bright_sky_is_an_event_frame(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        analyzer = BrightFake(station_id="st", profile_id="p")
        built = build(tmp_path, profile, analyzer=analyzer)
        analyzer.background_dn = 100.0  # a dark sky: the short frame gets nothing
        assert built.reference(built.run(make_survey_frame(0, exposure_us=SHORT_US))) is None
        analyzer.background_dn = 9000.0  # more than half of the 16,383 counts of the ADC
        frame = make_survey_frame(1, exposure_us=SHORT_US)
        output = built.run(frame)
        assert built.reference(output) == "survey/2026/09/21/20260921T141620.000Z.fits"
        assert "previews/2026/09/21/event-20260921T141620.000Z.jpg" in built.tree()
        back = framefile.read_frame_fits(built.layout.survey_path(frame.t_utc_ns))
        assert back.header["KEPT"] == "event:bright_sky"
        assert back.header["EXPTIME"] == 0.001

    def test_only_the_survey_frame_record_changes(self, tmp_path: Path, profile: Profile) -> None:
        built = build(tmp_path, profile)
        frame = make_survey_frame(0)
        plain = FakeSurveyAnalyzer(station_id="st", profile_id="p")
        plain.submit(frame)
        (expected,) = plain.poll()[0].records
        output = built.run(frame)
        (record,) = output.records
        assert isinstance(record, SurveyFrameRecord)
        assert record.model_dump(exclude={"image_ref"}) == expected.model_dump(
            exclude={"image_ref"}
        )
        assert output.t_utc_ns == frame.t_utc_ns
        assert output.solved is True

    def test_the_reference_passes_the_validation_of_the_record(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile)
        output = built.run(make_survey_frame(0))
        (record,) = output.records
        assert isinstance(record, SurveyFrameRecord)
        SurveyFrameRecord.model_validate(record.model_dump())  # a relative path, forward slashes
        assert built.layout.resolve(record.image_ref or "").is_file()

    def test_the_preview_is_a_jpeg_of_at_most_a_megapixel(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile)
        frame = make_survey_frame(0, shape=(1500, 2000))  # 3 megapixels
        built.run(frame)
        path = built.layout.preview_path(frame.t_utc_ns, kind="survey")
        with Image.open(path) as image:
            assert image.format == "JPEG"
            assert image.mode == "L"
            assert 100_000 < image.width * image.height <= 1_000_000
            assert image.width / image.height == pytest.approx(2000 / 1500, abs=0.01)
            pixels = np.asarray(image)
        assert pixels.max() > 200  # the star shows
        assert np.median(pixels) < 60  # on a dark sky

    def test_the_fits_file_holds_the_native_counts_and_the_header(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile)
        frame = make_survey_frame(0)
        built.run(frame)
        back = framefile.read_frame_fits(built.layout.survey_path(frame.t_utc_ns))
        assert back.compressed is True
        np.testing.assert_array_equal(back.pixels, frame.data >> 2)
        assert back.header["EXPTIME"] == 30.0
        assert back.header["GAIN"] == 120
        assert back.header["KEPT"] == "every_10"
        assert back.header["STATION"] == "st"
        assert back.header["PIXSCALE"] == pytest.approx(3.82, abs=0.01)

    def test_the_frame_is_left_as_it_was(self, tmp_path: Path, profile: Profile) -> None:
        built = build(tmp_path, profile)
        frame = make_survey_frame(0)
        frame.data.flags.writeable = False
        before = frame.data.copy()
        built.run(frame)
        np.testing.assert_array_equal(frame.data, before)

    def test_a_file_that_is_there_is_not_a_temporary_one(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile)
        built.run(make_survey_frame(0))
        names = [p.name for p in built.layout.root.rglob("*") if p.is_file()]
        assert names
        assert not any(name.startswith(".") or name.endswith(".tmp") for name in names)

    def test_a_fits_file_can_be_left_uncompressed(self, tmp_path: Path, profile: Profile) -> None:
        built = build(tmp_path, profile, fits_compression="none")
        frame = make_survey_frame(0)
        built.run(frame)
        back = framefile.read_frame_fits(built.layout.survey_path(frame.t_utc_ns))
        assert back.compressed is False
        assert built.frames.stats.fits_compressed == 0
        np.testing.assert_array_equal(back.pixels, frame.data >> 2)


class TestTheHeaderOfAFitsFile:
    def test_the_header_carries_the_cloud_fraction_and_the_transparency_of_the_result(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        analyzer = SkyFake(station_id="st", profile_id="p", cloud_fraction=0.04)
        analyzer.transparency = 0.9712
        built = build(tmp_path, profile, analyzer=analyzer)
        built.run(make_survey_frame(0))
        header = framefile.read_frame_header(built.layout.survey_path(START_NS))
        assert header["CLOUDFRC"] == 0.04
        assert header["TRANSP"] == 0.9712
        meta = framefile.parse_frame_header(header)
        assert (meta.cloud_fraction, meta.transparency) == (0.04, 0.9712)
        assert meta.kept == ("every_10",)

    def test_a_result_without_a_transparency_or_a_cloud_fraction_leaves_the_cards_out(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        analyzer = SkyFake(station_id="st", profile_id="p", cloud_fraction=None)
        analyzer.transparency = None  # no reference zero point yet
        built = build(tmp_path, profile, analyzer=analyzer)
        built.run(make_survey_frame(0))
        header = framefile.read_frame_header(built.layout.survey_path(START_NS))
        assert "CLOUDFRC" not in header
        assert "TRANSP" not in header
        assert header["KEPT"] == "every_10"

    def test_an_event_frame_carries_the_values_that_made_it_one(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        analyzer = SkyFake(station_id="st", profile_id="p", cloud_fraction=0.0)
        built = build(tmp_path, profile, analyzer=analyzer, keep_every=1000)
        built.run(make_survey_frame(0))
        analyzer.cloud_fraction = 0.8  # clouds from the next frame on
        analyzer.transparency = 0.4
        built.run(make_survey_frame(1))
        header = framefile.read_frame_header(built.layout.survey_path(START_NS + STEP_NS))
        assert header["KEPT"] == "event:cloud"
        assert header["CLOUDFRC"] == 0.8
        assert header["TRANSP"] == 0.4


class TestTheRing:
    def test_the_ring_holds_the_newest_frames_and_never_copies_them(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile, ram_frames=3)
        frames = [make_survey_frame(step) for step in range(5)]
        for frame in frames:
            built.frames.submit(frame)
        held = built.frames.recent_frames()
        assert [f.t_utc_ns for f in held] == [f.t_utc_ns for f in reversed(frames[2:])]
        assert all(a is b for a, b in zip(held, reversed(frames[2:]), strict=True))
        assert built.frames.ram_frames == 3

    def test_the_ring_holds_two_frames_by_default(self, tmp_path: Path, profile: Profile) -> None:
        built = build(tmp_path, profile)
        frames = [make_survey_frame(step) for step in range(3)]
        for frame in frames:
            built.frames.submit(frame)
        held = built.frames.recent_frames()
        assert [id(f) for f in held] == [id(frames[-1]), id(frames[-2])]
        assert built.frames.ram_frames == 2

    def test_the_size_of_the_ring_is_a_setting(self, tmp_path: Path, profile: Profile) -> None:
        built = build(tmp_path, profile, ram_frames=2)
        for step in range(3):
            built.frames.submit(make_survey_frame(step))
        assert [f.seq for f in built.frames.recent_frames()] == [2, 1]

    def test_a_frame_that_leaves_before_its_result_comes_gets_no_files(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        analyzer = FakeSurveyAnalyzer(station_id="st", profile_id="p", polls_until_ready=2)
        built = build(tmp_path, profile, analyzer=analyzer, ram_frames=3)
        frames = [make_survey_frame(step) for step in range(4)]
        for frame in frames:
            built.frames.submit(frame)
        assert built.frames.stats.dropped == 1  # the fourth frame pushed the first one out
        assert built.frames.poll() == ()
        assert built.frames.poll() == ()
        outputs = built.frames.poll()
        assert [o.t_utc_ns for o in outputs] == [f.t_utc_ns for f in frames]
        refs = [built.reference(output) for output in outputs]
        assert refs[0] is None
        assert all(ref is not None for ref in refs[1:])
        built.frames.drain()
        assert not built.layout.survey_path(frames[0].t_utc_ns).exists()
        assert built.layout.survey_path(frames[1].t_utc_ns).exists()  # the first one that counted

    def test_a_frame_that_leaves_before_its_files_are_written_is_counted_lost(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile, ram_frames=1)
        first = make_survey_frame(0)
        built.frames.submit(first)
        built.frames.poll()  # the job waits for the writer
        built.frames.submit(make_survey_frame(1))  # and the ring lets the first frame go
        built.frames.drain()
        assert built.frames.stats.lost == 1
        assert not built.layout.survey_path(first.t_utc_ns).exists()
        assert built.frames.stats.dropped == 0  # its result had come

    def test_a_result_without_a_frame_passes_through(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        analyzer = FakeSurveyAnalyzer(station_id="st", profile_id="p")
        built = build(tmp_path, profile, analyzer=analyzer)
        analyzer.submit(make_survey_frame(0))  # the analyzer gets a frame that the ring never saw
        (output,) = built.frames.poll()
        assert built.reference(output) is None
        assert built.frames.queued == 0


class TestTheWriter:
    def test_a_failed_write_leaves_no_file_and_one_event(
        self, tmp_path: Path, profile: Profile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing(handle: BinaryIO, *args: object, **kwargs: object) -> bool:
            handle.write(b"half of a file")
            raise OSError("no space left on the device")

        monkeypatch.setattr(survey_frames, "write_frame_fits", failing)
        built = build(tmp_path, profile, keep_every=1)
        for step in range(3):
            built.run(make_survey_frame(step))
        assert built.frames.stats.failures == 3
        assert built.frames.stats.last_error == "OSError: no space left on the device"
        assert built.frames.stats.fits_files == 0
        assert all(name.endswith(".jpg") for name in built.tree())  # the previews went on
        assert len(built.tree()) == 3
        assert not list(built.layout.root.rglob("*.tmp"))
        assert len(built.events) == 1  # three failures in nine minutes make one event
        level, kind, _, detail = built.events[0]
        assert (level, kind) == ("warning", "survey_images.write_failed")
        assert detail == {"t_utc_ns": START_NS, "kind": "survey"}
        built.clock.sleep(900.0)  # after the quiet time, the next failure is reported again
        built.run(make_survey_frame(10))
        assert len(built.events) == 2

    def test_a_writer_that_works_after_a_failure_goes_on(
        self, tmp_path: Path, profile: Profile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = framefile.write_frame_fits
        calls = []

        def flaky(handle: BinaryIO, *args: Any, **kwargs: Any) -> bool:
            calls.append(1)
            if len(calls) == 1:
                raise OSError("a bad sector")
            return real(handle, *args, **kwargs)

        monkeypatch.setattr(survey_frames, "write_frame_fits", flaky)
        built = build(tmp_path, profile, keep_every=1)
        built.run(make_survey_frame(0))
        built.run(make_survey_frame(1))
        assert built.frames.stats.failures == 1
        assert built.frames.stats.fits_files == 1
        assert built.layout.survey_path(START_NS + STEP_NS).is_file()

    def test_without_pillow_the_previews_stop_and_the_fits_files_go_on(
        self, tmp_path: Path, profile: Profile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_pillow(*args: object, **kwargs: object) -> None:
            raise ImportError("No module named 'PIL'")

        monkeypatch.setattr(survey_frames, "make_preview", no_pillow)
        built = build(tmp_path, profile, keep_every=1)
        for step in range(3):
            built.run(make_survey_frame(step))
        assert built.frames.stats.failures == 1  # the first try; then the writer stops trying
        assert built.frames.stats.fits_files == 3
        assert built.frames.stats.previews == 0

    def test_a_decision_that_fails_leaves_the_result_as_it_was(
        self, tmp_path: Path, profile: Profile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built = build(tmp_path, profile)

        def broken(*args: object, **kwargs: object) -> None:
            raise RuntimeError("a bug in the policy")

        monkeypatch.setattr(built.frames._policy, "decide", broken)
        frame = make_survey_frame(0)
        built.frames.submit(frame)
        (output,) = built.frames.poll()
        assert built.reference(output) is None
        assert output.t_utc_ns == frame.t_utc_ns
        assert built.frames.queued == 0

    def test_a_gate_that_closes_after_the_decision_stops_the_fits_file(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        state = {"open": True}
        built = build(tmp_path, profile, gate=lambda: state["open"])
        frame = make_survey_frame(0)
        built.frames.submit(frame)
        (output,) = built.frames.poll()
        state["open"] = False  # the free space fell while the job waited
        built.frames.drain()
        assert built.reference(output) == built.layout.relative(
            built.layout.survey_path(frame.t_utc_ns)
        )
        assert not built.layout.survey_path(frame.t_utc_ns).exists()
        assert built.layout.preview_path(frame.t_utc_ns, kind="survey").is_file()
        assert built.frames.stats.skipped_low_space == 1

    def test_a_closed_gate_at_the_decision_names_the_preview(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile, gate=lambda: False)
        output = built.run(make_survey_frame(0))
        assert built.reference(output) == "previews/2026/09/21/survey-20260921T141320.000Z.jpg"
        assert built.frames.stats.skipped_low_space == 1
        assert not built.layout.survey_path(START_NS).exists()

    def test_the_writer_thread_writes_until_it_is_told_to_finish(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile, keep_every=1, ram_frames=3)
        thread = threading.Thread(target=built.frames.run, daemon=True)
        thread.start()
        for step in range(3):
            built.frames.submit(make_survey_frame(step))
            built.frames.poll()
        built.frames.finish()
        thread.join(30.0)
        assert not thread.is_alive()
        assert built.frames.stats.fits_files == 3
        assert built.frames.stats.previews == 3
        assert built.frames.queued == 0

    def test_finish_before_run_still_writes_what_is_queued(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile)
        built.frames.submit(make_survey_frame(0))
        built.frames.poll()
        built.frames.finish()
        built.frames.run()  # returns after the queue is empty
        assert built.frames.stats.fits_files == 1

    def test_the_stats_count_what_the_writer_did(self, tmp_path: Path, profile: Profile) -> None:
        built = build(tmp_path, profile, keep_every=2)
        for step in range(4):
            built.run(make_survey_frame(step))
            built.run(make_survey_frame(step, exposure_us=SHORT_US, offset_ns=-20 * NS_PER_S))
        stats = built.frames.stats
        assert (stats.frames, stats.results) == (8, 8)
        assert (stats.previews, stats.fits_files) == (4, 2)  # the shorts get nothing
        assert stats.bytes_written > 0
        assert stats.failures == stats.dropped == stats.lost == 0


class TestTheCalibratedPreview:
    SHAPE = (120, 160)

    def calibrator(self, tmp_path: Path, profile: Profile, **parts: Any) -> PreviewCalibrator:
        """A calibrator for the sensor of `previewfx`: its flat and its dark library."""
        flat = previewfx.write_flat(tmp_path / "flat.npy", previewfx.sensitivity(self.SHAPE))
        library = DarkLibrary(tmp_path / "calibration" / "darks")
        previewfx.add_dark_set(library, self.SHAPE)
        config = SurveyConfig(flat_file=str(flat), calibration_dir=str(tmp_path / "calibration"))
        return PreviewCalibrator(config, profile, **parts)

    def frame(self, step: int = 0) -> Frame:
        return previewfx.make_survey_frame(
            previewfx.sensitivity(self.SHAPE), seq=step, t_utc_ns=START_NS + step * STEP_NS
        )

    def preview_bytes(self, built: Built, frame: Frame) -> bytes:
        return built.layout.preview_path(frame.t_utc_ns, kind="survey").read_bytes()

    def test_the_preview_shows_the_calibrated_frame_and_the_file_the_raw_one(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        calibrator = self.calibrator(tmp_path, profile)
        built = build(tmp_path / "out", profile, calibrator=calibrator)
        frame = self.frame()
        built.run(frame)
        settings = SurveyFrameSettings()
        expected = make_preview(
            frame.data,
            max_pixels=settings.preview_max_pixels,
            quality=settings.jpeg_quality,
            calibration=calibrator.for_frame(frame),
        )
        plain = make_preview(
            frame.data, max_pixels=settings.preview_max_pixels, quality=settings.jpeg_quality
        )
        assert self.preview_bytes(built, frame) == expected.jpeg
        assert expected.jpeg != plain.jpeg
        back = framefile.read_frame_fits(built.layout.survey_path(frame.t_utc_ns))
        assert np.array_equal(back.pixels, frame.data >> 2)  # the FITS file keeps the raw counts

    def test_a_calibration_that_fails_leaves_the_preview_as_it_was(
        self, tmp_path: Path, profile: Profile, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken() -> Any:
            raise SkyError("the flat is broken")

        calibrator = self.calibrator(tmp_path, profile, flat_provider=broken)
        built = build(tmp_path / "out", profile, calibrator=calibrator)
        frames = [self.frame(step) for step in range(3)]
        with caplog.at_level(logging.WARNING, logger="seeingmon.services.core.alignment"):
            for frame in frames:
                built.run(frame)
        settings = SurveyFrameSettings()
        for frame in frames:
            plain = make_preview(
                frame.data, max_pixels=settings.preview_max_pixels, quality=settings.jpeg_quality
            )
            assert self.preview_bytes(built, frame) == plain.jpeg
        assert built.frames.stats.failures == 0
        assert built.frames.stats.previews == 3
        assert built.events == []  # a calibration that fails is no failed write
        assert len([r for r in caplog.records if "not calibrated" in r.getMessage()]) == 1

    def test_a_calibrator_with_no_flat_and_no_dark_set_leaves_the_old_preview(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        built = build(tmp_path, profile, calibrator=PreviewCalibrator(SurveyConfig(), profile))
        frame = self.frame()
        built.run(frame)
        settings = SurveyFrameSettings()
        plain = make_preview(
            frame.data, max_pixels=settings.preview_max_pixels, quality=settings.jpeg_quality
        )
        assert self.preview_bytes(built, frame) == plain.jpeg


class TestTheAnalyzerInside:
    def test_what_the_analyzer_has_beyond_the_interface_is_reachable(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        analyzer = FakeSurveyAnalyzer(station_id="st", profile_id="p")
        built = build(tmp_path, profile, analyzer=analyzer)
        assert built.frames.submitted is analyzer.submitted  # not part of the interface
        with pytest.raises(AttributeError):
            built.frames.nothing_like_this  # noqa: B018
        with pytest.raises(AttributeError):
            built.frames._private  # noqa: B018

    def test_pending_is_the_pending_of_the_analyzer(self, tmp_path: Path, profile: Profile) -> None:
        analyzer = FakeSurveyAnalyzer(station_id="st", profile_id="p", polls_until_ready=1)
        built = build(tmp_path, profile, analyzer=analyzer)
        built.frames.submit(make_survey_frame(0))
        assert built.frames.pending() == 1
        built.frames.poll()
        built.frames.poll()
        assert built.frames.pending() == 0
