"""An end-to-end evening with the `sim` driver: dusk, a cloud, and a stall that a reopen clears.

The scheduler runs on a virtual clock against the simulated camera, which renders stars, turbulence,
the twilight sky, clouds, and injected faults. The analysis stays with the scripted fakes, and the
pointing provider reads the simulator's truth. The scenario uses a sensor with one eighth of the
real width and height, a short fast period, and short search bursts, so that the evening takes
seconds of real time. The simulator's twilight sky is dark enough for `auto` from the start, and
two search bursts find Polaris in the first two periods.

The test skips when the simulator is not available, which includes a machine without SciPy.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, VirtualClock, iso_to_utc_ns
from seeingmon.drivers.base import RecoveryLevel
from seeingmon.frames import StreamConfig
from seeingmon.records import EventRecord, SeeingWindowRecord
from seeingmon.scheduler import Scheduler, SchedulerConfig
from seeingmon.scheduler.config import (
    CloudConfig,
    FastConfig,
    LoopConfig,
    SearchConfig,
    SurveyConfig,
)
from seeingmon.scheduler.ephemeris import next_sun_crossing_utc_ns
from seeingmon.survey.config import TwilightConfig
from seeingmon.testing import FakeFastAnalyzer, ListRecordWriter

try:
    import seeingmon.drivers.sim as sim
except ImportError:  # the simulator needs SciPy, which belongs to the `fast` extra
    pytest.skip("the simulator is not available", allow_module_level=True)
if not hasattr(sim, "create"):  # the simulator lands in another lane
    pytest.skip("the sim driver has no factory yet", allow_module_level=True)

from tests.scheduler.simworld import SITE, LoggingSim, SimPointing, SimSurvey, small_profile

START = iso_to_utc_ns("2026-01-01T16:10:00Z")  # the Sun is 3.9 degrees below the horizon


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

    def starts(self, purpose: str) -> list[tuple[float, StreamConfig]]:
        """The seconds after dusk and the settings of each stream of a purpose."""
        return [
            ((t - self.dusk_ns) / NS_PER_S, config)
            for index, (t, config) in enumerate(self.driver.configure_log)
            if self.driver.purposes.get(index) == purpose
        ]

    def fast_starts(self) -> list[tuple[float, StreamConfig]]:
        """The seconds after dusk and the settings of each fast stream (measure)."""
        return self.starts("fast")

    def period_starts(self) -> list[float]:
        """The seconds after dusk at which each period began: its first burst or its fast stream."""
        starts: list[float] = []
        previous: str | None = None
        for index, (t, _) in enumerate(self.driver.configure_log):
            purpose = self.driver.purposes.get(index)
            if purpose in ("fast", "search") and previous != "search":
                starts.append((t - self.dusk_ns) / NS_PER_S)
            previous = purpose
        return starts


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
        # A burst of 10 frames of 12 ms at the start of each period of 0.5 s.
        search=SearchConfig(burst_frames=10, interval_s=1.0),
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
        # The long frame keeps its 2 s, so that every step reports the clouds: in this bright dusk
        # the adaptive long exposure would skip it (`test_survey_exposure.py` tests that).
        twilight=TwilightConfig(min_exposure_s=2.0),
    )
    driver.scheduler = scheduler
    scheduler.run_until(dusk + 150 * NS_PER_S)
    scheduler.close()
    return Evening(scheduler, driver, writer, fast, dusk)


def test_the_scheduler_enters_auto_at_the_first_brightness_frame(evening: Evening) -> None:
    """The Sun gates nothing, and the simulator's sky at -3.9 degrees is far from saturating."""
    changes = evening.events("scheduler.state_change")
    assert [(e.detail or {})["to"] for e in changes] == ["auto"]
    assert changes[0].t_utc_ns - START < NS_PER_S
    assert evening.scheduler.status().background_fraction is not None


def test_two_bursts_find_polaris_and_measure_starts(evening: Evening) -> None:
    visible = evening.events("polaris.visible")[0]
    detail = visible.detail or {}
    assert -4.5 < detail["sun_elevation_deg"] < -3.5
    assert detail["offset_px"] < 2.0  # the bursts found the star where the truth puts it
    bursts = evening.starts("search")
    assert bursts[1][0] - bursts[0][0] == pytest.approx(20.0, abs=0.1)  # one in each period
    first_fast = evening.fast_starts()[0][0]
    assert first_fast == pytest.approx((visible.t_utc_ns - evening.dusk_ns) / NS_PER_S, abs=0.01)
    # The frames of the bursts reached `measure` and no window.
    counters = evening.scheduler.status().counters
    assert counters.search_frames == 10 * len(bursts) == evening.fast.frames_measured


def test_the_stall_ends_measure_and_the_search_finds_the_star_again(evening: Evening) -> None:
    hidden = evening.events("polaris.hidden")
    assert [(e.detail or {})["reason"] for e in hidden] == ["fault", "shutdown"]
    (fault,) = [e for e in evening.events("scheduler.fault") if e.t_utc_ns == hidden[0].t_utc_ns]
    assert (fault.detail or {})["where"] == "reading a fast frame"
    assert len(evening.events("polaris.visible")) == 2


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
        assert window.frame_rate_hz == pytest.approx(82.1, rel=0.03)  # 12.2 ms a frame


def test_the_frames_that_the_stall_cost_are_counted_and_no_window_claims_them(
    evening: Evening,
) -> None:
    """The simulator counts the frames that the stall cost, and the first frame after it says so.

    The fault ended measure, so that frame belongs to the first search burst after the reopen,
    which makes no window.
    """
    assert evening.scheduler.status().counters.dropped > 1000  # about 14 s at 88 frames a second
    assert [w for w in evening.windows() if w.n_dropped > 0] == []


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
    starts = evening.period_starts()
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
