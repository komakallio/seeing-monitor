"""Retention covers the files of the survey frames: the age limits, the thinning, and the gate.

The tests write the files with the real writer, so the names and the folders are the ones that
`core` produces. They then set the time of the last change, as the clock of a real station would,
and run the retention task that `core` runs every hour.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

pytest.importorskip("PIL", reason="the previews need Pillow")
pytest.importorskip("astropy", reason="the FITS files need astropy")

from seeingmon.clock import NS_PER_S, VirtualClock, iso_to_utc_ns
from seeingmon.services.core.settings import SurveyFrameSettings
from seeingmon.services.core.survey_frames import SurveyFrames
from seeingmon.store.config import GB, RetentionConfig
from seeingmon.store.events import EventEmitter
from seeingmon.store.layout import DataLayout
from seeingmon.store.retention import DAY_NS, RetentionManager
from seeingmon.testing import FakeSurveyAnalyzer
from tests.store.disk import FakeDisk

from .test_survey_frames import LONG_US, make_survey_frame

NOW = iso_to_utc_ns("2026-10-20T12:00:00Z")


class Station:
    """The writer of the survey frames and the retention task, on one data directory."""

    def __init__(self, tmp_path: Path, **retention: float) -> None:
        self.layout = DataLayout(tmp_path / "data")
        self.layout.create()
        self.clock = VirtualClock(NOW)
        self.events: list[object] = []
        self.disk = FakeDisk(self.layout.root, 10**15)
        settings = {"min_free_gb": 0.0, "resume_margin_gb": 0.0, **retention}
        self.retention = RetentionManager(
            self.layout,
            RetentionConfig(**settings),
            self.clock,
            EventEmitter(
                self.events.append, self.clock, station_id="st", profile_id="p", provenance={}
            ),
            disk_usage=self.disk,
        )
        self.frames = SurveyFrames(
            FakeSurveyAnalyzer(station_id="st", profile_id="p"),
            layout=self.layout,
            profile=None,
            station_id="st",
            clock=self.clock,
            settings=SurveyFrameSettings(keep_every=1),  # every long frame is a FITS file
            long_min_exposure_s=5.0,
            capture_allowed=self.retention.capture_allowed,
        )

    def write(self, t_utc_ns: int, *, age_s: float | None = None) -> list[Path]:
        """Write the files of the long frame of that time. They last changed `age_s` ago."""
        frame = make_survey_frame(0, exposure_us=LONG_US, shape=(30, 40))
        frame = dataclasses.replace(frame, t_utc_ns=t_utc_ns)  # the same pixels at another time
        self.frames.submit(frame)
        self.frames.poll()
        self.frames.drain()
        paths = [
            self.layout.survey_path(t_utc_ns),
            self.layout.preview_path(t_utc_ns, kind="survey"),
        ]
        stamp = t_utc_ns + 60 * NS_PER_S if age_s is None else NOW - round(age_s * NS_PER_S)
        for path in paths:
            if path.exists():
                os.utime(path, ns=(stamp, stamp))
        return paths

    def files(self, folder: Path) -> list[str]:
        return sorted(p.name for p in folder.rglob("*") if p.is_file())


@pytest.fixture
def station(tmp_path: Path) -> Station:
    return Station(tmp_path)


def night_frames(day: int) -> list[int]:
    """Three frames of the night that starts at 12:00 UTC `day` days before NOW: 22:00 the evening
    before it ends, midnight (the middle of the night), and 02:00."""
    start = NOW - day * DAY_NS  # 12:00 UTC of that day: the night begins
    return [
        start + 10 * 3600 * NS_PER_S,
        start + 12 * 3600 * NS_PER_S,
        start + 14 * 3600 * NS_PER_S,
    ]


class TestTheAgeLimits:
    def test_the_files_are_where_retention_looks_and_a_pass_keeps_the_new_ones(
        self, station: Station
    ) -> None:
        for t in night_frames(1):
            station.write(t)
        report = station.retention.run_once()
        assert report.deletions == ()
        assert len(station.files(station.layout.survey_dir)) == 3
        assert len(station.files(station.layout.previews_dir)) == 3

    def test_previews_go_after_seven_days_and_the_fits_files_stay_but_one_a_night(
        self, station: Station
    ) -> None:
        recent = [t for day in (1, 3, 6) for t in night_frames(day)]
        old_nights = {day: night_frames(day) for day in (9, 12, 20)}
        for t in recent + [t for night in old_nights.values() for t in night]:
            station.write(t)
        report = station.retention.run_once()
        assert {(d.tier, d.reason) for d in report.deletions} == {
            ("previews", "expired"),
            ("survey", "expired"),
        }
        # Every frame of the last seven days stays, as a FITS file and a preview.
        for t in recent:
            assert station.layout.survey_path(t).is_file()
            assert station.layout.preview_path(t, kind="survey").is_file()
        # An older night keeps the frame nearest to the middle of the night, and no preview.
        for night in old_nights.values():
            evening, midnight, morning = night
            assert station.layout.survey_path(midnight).is_file()
            assert not station.layout.survey_path(evening).exists()
            assert not station.layout.survey_path(morning).exists()
            assert not any(station.layout.preview_path(t, kind="survey").exists() for t in night)

    def test_a_night_older_than_sixty_seven_days_goes_altogether(self, station: Station) -> None:
        old = night_frames(70)
        for t in old:
            station.write(t)
        station.retention.run_once()
        assert station.files(station.layout.survey_dir) == []

    def test_a_pass_removes_the_temporary_name_of_a_crashed_write(self, station: Station) -> None:
        station.write(NOW - 3600 * NS_PER_S)
        leftover = station.layout.survey_dir / "2026/10/20" / ".20261020T110000.000Z.fits.77.0.tmp"
        leftover.write_bytes(b"half a frame")
        stamp = NOW - 2 * 3600 * NS_PER_S
        os.utime(leftover, ns=(stamp, stamp))
        report = station.retention.run_once()
        assert report.temp_files_removed == 1
        assert not leftover.exists()


class TestThePressure:
    def test_under_the_quota_the_previews_shrink_before_the_survey_frames(
        self, station: Station
    ) -> None:
        for t in night_frames(1) + night_frames(2):
            station.write(t)
        previews = sum(p.stat().st_size for p in station.layout.previews_dir.rglob("*.jpg"))
        survey = sum(p.stat().st_size for p in station.layout.survey_dir.rglob("*.fits"))
        total = previews + survey
        # A partition so small that the data directory may use a quarter of it: the quota is
        # a little under the data that the writer made, so the pass must delete the oldest previews.
        station.disk.total = 4 * (total - previews // 2)
        report = station.retention.run_once()
        early = [d for d in report.deletions if d.early]
        assert [d.tier for d in early] == ["previews"]
        assert early[0].reason == "total_quota"
        assert len(station.files(station.layout.survey_dir)) == 6  # no FITS file went
        assert len(station.files(station.layout.previews_dir)) < 6


class TestTheCaptureGate:
    def test_the_fits_files_stop_below_the_free_space_limit_and_resume_after_the_margin(
        self, tmp_path: Path
    ) -> None:
        station = Station(tmp_path, min_free_gb=1.0, resume_margin_gb=0.5)
        station.disk.total = 100 * GB

        def free(gigabytes: float) -> None:
            station.disk.other_used = 100 * GB - round(gigabytes * GB)

        free(0.5)  # below the limit of 1 GB
        stopped = station.write(NOW - 600 * NS_PER_S)
        assert [p.suffix for p in stopped if p.exists()] == [".jpg"]  # the preview went on
        free(1.2)  # above the limit, but inside the margin
        held = station.write(NOW - 500 * NS_PER_S)
        assert [p.suffix for p in held if p.exists()] == [".jpg"]
        free(1.6)  # the margin is met
        resumed = station.write(NOW - 400 * NS_PER_S)
        assert sorted(p.suffix for p in resumed if p.exists()) == [".fits", ".jpg"]
