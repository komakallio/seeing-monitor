"""An end-to-end evening with the `sim` driver: dusk, a cloud, and a stall that a reopen clears.

The scheduler runs on a virtual clock against the simulated camera, which renders stars, turbulence,
the twilight sky, clouds, and injected faults. The analysis stays with the scripted fakes, and the
pointing provider reads the simulator's truth. The scenario uses a sensor with one eighth of the
real width and height, and a short fast period, so that the evening takes seconds of real time.

The test skips when the simulator is not available, which includes a machine without SciPy.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, Clock, VirtualClock, iso_to_utc_ns
from seeingmon.drivers.base import RecoveryLevel
from seeingmon.frames import ActiveStream, Frame, StreamConfig
from seeingmon.profile import Profile, parse_profile
from seeingmon.records import EventRecord, SeeingWindowRecord
from seeingmon.scheduler import Scheduler, SchedulerConfig, SiteConfig
from seeingmon.scheduler.config import CloudConfig, FastConfig, LoopConfig, SurveyConfig
from seeingmon.scheduler.ephemeris import next_sun_crossing_utc_ns
from seeingmon.testing import FakeFastAnalyzer, FakeSurveyAnalyzer, ListRecordWriter
from tests.profile.builders import reference_data

try:
    import seeingmon.drivers.sim as sim
except ImportError:  # the simulator needs SciPy, which belongs to the `fast` extra
    pytest.skip("the simulator is not available", allow_module_level=True)
if not hasattr(sim, "create"):  # the simulator lands in another lane
    pytest.skip("the sim driver has no factory yet", allow_module_level=True)

# The synthetic site of the simulator and of the scheduler: 55 degrees north on the prime meridian.
SITE = SiteConfig(latitude_deg=55.0, longitude_deg=0.0)
START = iso_to_utc_ns("2026-01-01T16:10:00Z")  # the Sun is 1.5 degrees below the horizon


def small_profile() -> Profile:
    """The reference camera with one eighth of the width and height. The optics stay the same."""
    data = reference_data()
    data["id"] = "asi294mm-gs250-small"
    for mode in data["readout_modes"]:
        mode["width_px"] = mode["width_px"] // 8
        mode["height_px"] = mode["height_px"] // 8
    return parse_profile(data)


class LoggingSim(sim.SimDriver):
    """The simulator, with the time of every `configure` call."""

    def __init__(self, profile: Profile, clock: Clock, options: Any) -> None:
        super().__init__(sim.SimParams.modes_from_profile(profile), clock, options)
        self.configure_log: list[tuple[int, StreamConfig]] = []

    def configure(self, config: StreamConfig) -> ActiveStream:
        self.configure_log.append((self.clock.utc_ns(), config))
        return super().configure(config)


class SimPointing:
    """The Polaris position from the simulator's truth: its brightest star, projected to pixels."""

    def __init__(self, truth: Any) -> None:
        self._truth = truth

    def polaris_position(self, t_utc_ns: int, mode: str) -> tuple[float, float] | None:
        stars = self._truth.star_positions(t_utc_ns, mode)
        brightest = int(np.argmin(stars.mag))
        return float(stars.x[brightest]), float(stars.y[brightest])


class SimSurvey(FakeSurveyAnalyzer):
    """A survey analysis that reads the cloud fraction from the simulator's transparency."""

    def __init__(self, truth: Any, profile_id: str) -> None:
        super().__init__(station_id="test", profile_id=profile_id)
        self._truth = truth

    def submit(self, frame: Frame) -> None:
        long_exposure = frame.exposure_us >= 1_000_000
        transparency = float(self._truth.transparency(frame.t_utc_ns))
        self.cloud_fraction = 1.0 - transparency if long_exposure else None
        self.solved = long_exposure and transparency > 0.15
        super().submit(frame)


@dataclass(frozen=True)
class Evening:
    scheduler: Scheduler
    driver: LoggingSim
    writer: ListRecordWriter
    fast: FakeFastAnalyzer
    dusk_ns: int

    def events(self, kind: str) -> list[EventRecord]:
        found = [e for e in self.writer.of_type("event") if isinstance(e, EventRecord)]
        return [event for event in found if event.kind == kind]

    def windows(self) -> list[SeeingWindowRecord]:
        found = self.writer.of_type("seeing_window")
        return [w for w in found if isinstance(w, SeeingWindowRecord)]

    def fast_starts(self) -> list[tuple[float, StreamConfig]]:
        """The seconds after dusk and the settings of each configure of the fast stream."""
        return [
            ((t - self.dusk_ns) / NS_PER_S, config)
            for t, config in self.driver.configure_log
            if config.mode == "bin1" and config.exposure_us == 2000
        ]


@pytest.fixture(scope="module")
def evening() -> Evening:
    profile = small_profile()
    clock = VirtualClock(START)
    dusk = next_sun_crossing_utc_ns(
        START, SITE.latitude_deg, SITE.longitude_deg, -4.0, rising=False
    )
    assert dusk is not None
    # A cloud of 40 seconds comes 60 seconds after the scheduler starts `auto`. A stall hits the
    # camera after 250 delivered frames, which is in the third fast period.
    options = sim.SimOptions(
        seed=3,
        epoch_utc_ns=START,
        clouds=sim.Clouds(
            events=(
                sim.CloudEvent(start_utc_ns=dusk + 60 * NS_PER_S, duration_s=40, transmission=0.1),
            )
        ),
        faults=sim.SimFaults(stall_at_frame=250, stall_clears_at=RecoveryLevel.REOPEN),
    )
    driver = LoggingSim(profile, clock, options)
    writer = ListRecordWriter()
    fast = FakeFastAnalyzer(station_id="test", profile_id=profile.id, window_s=0.25)
    config = SchedulerConfig(
        fast=FastConfig(window_s=0.5, analysis_window_s=0.25),
        survey=SurveyConfig(cadence_s=20.0, long_exposure_s=2.0),
        cloud=CloudConfig(fast_window_s=0.25, survey_cadence_s=12.0),
        loop=LoopConfig(max_sleep_s=5.0),
    )
    scheduler = Scheduler(
        driver=driver,
        fast=fast,
        survey=SimSurvey(driver.truth, profile.id),
        pointing=SimPointing(driver.truth),
        records=writer,
        metrics=writer,
        clock=clock,
        profile=profile,
        station_id="test",
        config=config,
        site=SITE,
        escalate=lambda level: None,
    )
    scheduler.run_until(dusk + 150 * NS_PER_S)
    scheduler.close()
    return Evening(scheduler, driver, writer, fast, dusk)


def test_the_scheduler_enters_auto_when_the_sun_is_four_degrees_down(evening: Evening) -> None:
    changes = evening.events("scheduler.state_change")
    assert [(e.detail or {})["to"] for e in changes] == ["auto"]
    assert evening.dusk_ns <= changes[0].t_utc_ns <= evening.dusk_ns + 61 * NS_PER_S


def test_the_windows_are_fast_bin1_windows_of_the_simulated_star(evening: Evening) -> None:
    windows = evening.windows()
    assert len(windows) >= 8
    for window in windows:
        assert (window.readout_mode, window.exposure_us, window.gain) == ("bin1", 2000, 0)
        assert "twilight" in window.flags  # the Sun is above -18 degrees all evening
        assert window.zenith_angle_deg is not None
        assert 34.0 <= window.zenith_angle_deg <= 36.0
    full = [w for w in windows if w.n_frames > 15]
    assert full
    for window in full:
        assert window.frame_rate_hz == pytest.approx(88.4, rel=0.03)  # 11.3 ms a frame


def test_only_the_window_after_the_stall_reports_dropped_frames(evening: Evening) -> None:
    """The simulator counts the frames that the stall cost, and the first frame after it says so."""
    dropped = [w for w in evening.windows() if w.n_dropped > 0]
    assert len(dropped) == 1
    assert "degraded" in dropped[0].flags  # more than 5% of the expected frames
    assert dropped[0].n_dropped > 1000  # about 14 seconds at 88 frames a second
    assert evening.scheduler.status().counters.dropped == dropped[0].n_dropped


def test_every_roi_is_centered_on_the_star_that_the_simulator_rendered(evening: Evening) -> None:
    fast = evening.fast_starts()
    assert len(fast) >= 6
    truth = evening.driver.truth
    for seconds, config in fast:
        roi = config.roi
        assert roi is not None
        assert (roi.width, roi.height) == (128, 128)  # 4.1 arcmin through the profile's helpers
        stars = truth.star_positions(evening.dusk_ns + round(seconds * NS_PER_S), "bin1")
        brightest = int(np.argmin(stars.mag))
        x, y = float(stars.x[brightest]), float(stars.y[brightest])
        assert roi.distance_to_edge(x, y) >= 60  # within a few pixels of the center


def test_the_fast_periods_keep_the_cadence_and_the_cloud_shortens_it(evening: Evening) -> None:
    starts = [seconds for seconds, _ in evening.fast_starts()]
    gaps = [round(later - earlier, 1) for earlier, later in itertools.pairwise(starts)]
    assert gaps.count(20.0) >= 1  # the normal cadence
    assert gaps.count(12.0) >= 1  # the cadence under cloud


def test_the_cloud_starts_the_response_and_the_clear_sky_ends_it(evening: Evening) -> None:
    clouds = evening.events("scheduler.cloud")
    assert [(e.detail or {})["active"] for e in clouds] == [True, False]
    assert any("cloud" in window.flags for window in evening.windows())


def test_the_stall_is_cleared_by_a_reopen_after_two_restarts_of_the_capture(
    evening: Evening,
) -> None:
    """The stall outlasts a restart of the capture, and a reopen clears it."""
    levels = [level for name, level in evening.driver.calls if name == "recover"]
    restart, reopen = RecoveryLevel.RESTART_CAPTURE, RecoveryLevel.REOPEN
    assert levels == [restart, restart, reopen]  # two attempts for each step of the ladder
    assert len(evening.events("scheduler.fault")) == 3
    assert evening.scheduler.status().degraded is False
    assert evening.events("scheduler.degraded") == []


def test_no_frame_is_lost_in_the_whole_evening(evening: Evening) -> None:
    assert sum(w.n_frames for w in evening.windows()) == evening.fast.frames_pushed
    rows = sum(len(batch) for _, batch in evening.writer.metrics)
    assert rows == evening.fast.frames_pushed
