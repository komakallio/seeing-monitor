"""The read timeout of a snapshot follows the snapshot model of the profile.

A single exposure on the real camera takes much longer than the line model of a video stream says:
in bin2, a 1 ms exposure took 0.29 s for the 20 arcminute ROI and 0.48 to 0.53 s for the full
frame, where the video model gives 7 ms and 53 ms. The scheduler waits twice the frame period that
the driver reports, plus a margin, so the period of a snapshot stream has to hold the snapshot
model. Otherwise the first survey exposure after a start timed out, as it did on a Raspberry Pi 4.

These tests run the scheduler on the production `asi` driver and the fake SDK, which takes the
single-exposure time from the same numbers as the profile, and they read the timeout that the
scheduler passes to each read.
"""

from __future__ import annotations

from typing import Any

import pytest

from seeingmon.clock import VirtualClock, iso_to_utc_ns
from seeingmon.drivers.asi import AsiDriver
from seeingmon.frames import ActiveStream, Frame, StreamConfig, StreamKind
from seeingmon.hardware.asi.fake import FakeAsiSdk
from seeingmon.scheduler import Scheduler, SchedulerConfig
from seeingmon.testing import (
    FakeFastAnalyzer,
    FakePointingProvider,
    FakeSurveyAnalyzer,
    ListRecordWriter,
)
from tests.scheduler.scenario import PROFILE, SITE

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")  # the sun is far below the horizon at the site
SLOWEST_FULL_FRAME_S = 0.532  # the slowest of three runs of 1 ms on a Raspberry Pi 4
SLOWEST_WATCH_ROI_S = 0.296


class RecordingDriver(AsiDriver):
    """The production driver, plus a record of each read: the stream and the timeout."""

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.last_stream: ActiveStream | None = None
        self.reads: list[tuple[StreamConfig, float, float]] = []

    def configure(self, config: StreamConfig) -> ActiveStream:
        self.last_stream = super().configure(config)
        return self.last_stream

    def read_frame(self, timeout_s: float) -> Frame:
        stream = self.last_stream
        assert stream is not None
        assert stream.frame_period_s is not None
        self.reads.append((stream.config, stream.frame_period_s, timeout_s))
        return super().read_frame(timeout_s)


def build(*, solved: bool) -> tuple[Scheduler, RecordingDriver]:
    """A scheduler at night. With `solved`, it knows where Polaris is and starts a fast stream."""
    clock = VirtualClock(NIGHT)
    driver = RecordingDriver(api=FakeAsiSdk(clock), profile=PROFILE, clock=clock)
    config = SchedulerConfig()
    writer = ListRecordWriter()
    pointing = FakePointingProvider(
        {"bin1": (4144.0, 2822.0), "bin2": (2072.0, 1411.0)} if solved else None
    )
    scheduler = Scheduler(
        driver=driver,
        fast=FakeFastAnalyzer(
            station_id="test", profile_id=PROFILE.id, window_s=config.fast.analysis_window_s
        ),
        survey=FakeSurveyAnalyzer(station_id="test", profile_id=PROFILE.id),
        pointing=pointing,
        records=writer,
        metrics=writer,
        clock=clock,
        profile=PROFILE,
        station_id="test",
        config=config,
        site=SITE,
    )
    return scheduler, driver


def step_until(scheduler: Scheduler, driver: RecordingDriver, reads: int) -> None:
    for _ in range(100):
        if len(driver.reads) >= reads:
            return
        scheduler.step()
    raise AssertionError(f"the scheduler made {len(driver.reads)} reads in 100 steps")


def test_the_watch_and_the_survey_wait_for_what_the_camera_needs() -> None:
    """Without a solution, the scheduler takes the brightness frame, and then the survey step: a
    short exposure and a long one, both of the full frame."""
    scheduler, driver = build(solved=False)
    step_until(scheduler, driver, 4)
    scheduler.close()
    # The first survey step also takes a frame of 32 us of the region of the watch, which measures
    # the black level of the 1 ms frame for the long exposure.
    (watch, short, black, long) = driver.reads[:4]
    assert black[0].exposure_us == 32
    assert black[0].roi == watch[0].roi

    watch_config, watch_period_s, watch_timeout_s = watch
    assert watch_config.kind is StreamKind.SNAPSHOT
    assert watch_config.exposure_us == 1000
    assert watch_config.roi is not None
    assert (watch_config.roi.width, watch_config.roi.height) == (312, 314)  # 20 arcminutes
    assert watch_period_s == pytest.approx(0.001 + 0.27 + 314 * 75e-6)  # the snapshot model
    assert watch_timeout_s == pytest.approx(2 * watch_period_s + 0.5)
    assert watch_timeout_s > 1.0
    assert watch_timeout_s - SLOWEST_WATCH_ROI_S > 0.5  # the old model left 0.2 s

    short_config, short_period_s, short_timeout_s = short
    assert short_config.kind is StreamKind.SNAPSHOT
    assert short_config.exposure_us == 1000
    assert short_config.roi is not None
    assert (short_config.roi.width, short_config.roi.height) == (4144, 2822)  # the full frame
    assert short_period_s == pytest.approx(0.001 + 0.27 + 2822 * 75e-6)
    assert short_timeout_s == pytest.approx(2 * short_period_s + 0.5)
    assert short_timeout_s > 1.0  # the video model gave 0.61 s, and the frame needed 0.53 s
    assert short_timeout_s - SLOWEST_FULL_FRAME_S > 0.9  # the old model left 0.08 s

    # The first long frame of an episode of `auto` takes `[survey.twilight] min_exposure_s` (1 s),
    # because the 1 ms frame cannot tell a longer one (`seeingmon.scheduler.exposure`).
    long_config, long_period_s, long_timeout_s = long
    assert long_config.exposure_us == 1_000_000
    assert long_period_s == pytest.approx(1.0 + 0.27 + 2822 * 75e-6)
    assert long_timeout_s == pytest.approx(2 * long_period_s + 0.5)
    assert long_timeout_s > 2 * (1.0 + SLOWEST_FULL_FRAME_S)


def test_the_fast_stream_keeps_its_short_wait() -> None:
    """The snapshot model must not lengthen the hang detection of the fast stream: a video stream
    keeps the line model, and a read waits twice its 12 ms period plus 0.5 s."""
    scheduler, driver = build(solved=True)
    step_until(scheduler, driver, 4)
    scheduler.close()
    video = [read for read in driver.reads if read[0].kind is StreamKind.VIDEO]
    assert video, "no fast stream started"
    config, period_s, timeout_s = video[0]
    assert (config.mode, config.exposure_us) == ("bin1", 2000)
    assert period_s == pytest.approx(PROFILE.frame_period_s("bin1", 128, 2000))
    assert timeout_s == pytest.approx(2 * period_s + 0.5)
    assert timeout_s < 0.6


def test_a_profile_without_a_snapshot_model_still_waits_longer_than_the_camera_took() -> None:
    """A mode that nobody measured takes the video row time and an overhead of 0.3 s, which is
    above the 0.29 s that the reference camera needed for the watch ROI."""
    bare = PROFILE.model_copy(
        update={
            "readout_modes": tuple(
                mode.model_copy(update={"snapshot_overhead_s": None, "snapshot_row_time_us": None})
                for mode in PROFILE.readout_modes
            )
        }
    )
    clock = VirtualClock(NIGHT)
    driver = RecordingDriver(api=FakeAsiSdk(clock), profile=bare, clock=clock)
    writer = ListRecordWriter()
    scheduler = Scheduler(
        driver=driver,
        fast=FakeFastAnalyzer(station_id="test", profile_id=bare.id, window_s=120.0),
        survey=FakeSurveyAnalyzer(station_id="test", profile_id=bare.id),
        pointing=FakePointingProvider(),
        records=writer,
        metrics=writer,
        clock=clock,
        profile=bare,
        station_id="test",
        site=SITE,
    )
    step_until(scheduler, driver, 1)
    scheduler.close()
    _, period_s, timeout_s = driver.reads[0]  # the watch ROI
    assert period_s > SLOWEST_WATCH_ROI_S
    assert timeout_s == pytest.approx(2 * period_s + 0.5)
