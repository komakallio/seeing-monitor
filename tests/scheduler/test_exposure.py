"""The adaptive exposure of the fast stream, by rule and on the simulator's sky.

`seeingmon.scheduler.exposure` holds the rule: the exposure that puts the sky background at
`[scheduler.fast] target_background_fraction` (0.3) of saturation, from the background of the last
window or burst, between the profile's shortest exposure and `exposure_us`. The scheduler runs here
against the `sim` driver, which renders the sky, with the fake fast analysis, which reports the
mean of the frame medians as the background of a window:

- **A constant bright sky** of 4.4 mag/arcsec^2, where the detection estimate puts the crossing of
  an SNR of 10 (`docs/research-notes.md`, "Polaris in a bright sky"). It checks the first exposure,
  which the brightness frame gives, and the exposure where the loop settles.
- **A simulated dawn** from a Sun of -2.5 to +4 degrees, where the simulator's sky brightens by
  about one magnitude per degree up to the horizon. The longest exposure is the 50 ms of `seeingmon
  dev`, so that the loop has room to follow.
- **An episode of `auto`** under a constant sky of 6 mag/arcsec^2, with a pause and with a camera
  fault. The first burst after an entry into `auto` takes its exposure from the brightness frame,
  and a fault inside `auto` keeps the background of the last window.
- **The daylight gate** under a sunny sky of 0.8 mag/arcsec^2, which clips the 1 ms brightness
  frame while the fast stream at 32 us sees 18% of saturation, and under a sky that brightens past
  50% at 32 us. The bursts decide the gate there, as the brightness frame cannot.

The tests skip when the simulator is not available, which includes a machine without SciPy.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import pytest

from seeingmon.clock import NS_PER_S, VirtualClock, iso_to_utc_ns
from seeingmon.frames import StreamConfig
from seeingmon.records import EventRecord, SeeingWindowRecord
from seeingmon.scheduler import (
    Command,
    Pause,
    QueueSweep,
    Resume,
    Scheduler,
    SchedulerConfig,
)
from seeingmon.scheduler.config import FastConfig, LoopConfig, SearchConfig, SurveyConfig
from seeingmon.scheduler.ephemeris import next_sun_crossing_utc_ns
from seeingmon.scheduler.exposure import adapted_exposure_us, background_fraction
from seeingmon.testing import FakeFastAnalyzer, ListRecordWriter

TARGET = 0.3


class TestTheRule:
    def test_the_exposure_scales_the_background_to_the_target(self) -> None:
        assert (
            adapted_exposure_us(2000, 0.6, target=TARGET, shortest_us=32, longest_us=2000) == 1000
        )
        assert adapted_exposure_us(500, 0.1, target=TARGET, shortest_us=32, longest_us=2000) == 1500

    def test_the_exposure_stays_between_the_shortest_and_the_longest(self) -> None:
        assert (
            adapted_exposure_us(2000, 100.0, target=TARGET, shortest_us=32, longest_us=2000) == 32
        )
        assert (
            adapted_exposure_us(2000, 0.1, target=TARGET, shortest_us=32, longest_us=2000) == 2000
        )

    @pytest.mark.parametrize("fraction", [0.0, -0.01, math.nan])
    def test_a_dark_sky_or_a_missing_value_gives_the_longest(self, fraction: float) -> None:
        exposure = adapted_exposure_us(
            2000, fraction, target=TARGET, shortest_us=32, longest_us=2000
        )
        assert exposure == 2000

    def test_the_limits_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="shortest_us"):
            adapted_exposure_us(2000, 0.3, target=TARGET, shortest_us=3000, longest_us=2000)

    def test_the_background_counts_above_the_offset(self) -> None:
        assert background_fraction(1480.0, 65_520.0, 480.0) == pytest.approx(1000.0 / 65_520.0)
        assert background_fraction(1480.0, 65_520.0) == pytest.approx(1480.0 / 65_520.0)
        assert background_fraction(400.0, 65_520.0, 480.0) == 0.0
        with pytest.raises(ValueError, match="saturation_dn"):
            background_fraction(100.0, 0.0)


pytest.importorskip("seeingmon.drivers.sim", reason="the simulator needs the fast extra")

from seeingmon.drivers.sim import PsfConfig, SimFaults, SimOptions, SimParams  # noqa: E402
from seeingmon.drivers.sim.detector import Detector  # noqa: E402
from tests.scheduler.simworld import (  # noqa: E402
    SITE,
    LoggingSim,
    SimPointing,
    SimSurvey,
    small_profile,
)

PROFILE = small_profile()
PARAMS = SimParams.from_profile(PROFILE, "bin1")
FULL_WELL_E = PARAMS.sensor_at(0).full_well_e
SATURATION_DN = PROFILE.saturation("bin1", 0).container_dn
# The simulator's offset in a bin1 frame, as a share of saturation: the loop counts it as sky.
OFFSET_FRACTION = (
    Detector(PARAMS).black_level_adu(None) * 2 ** (16 - PARAMS.adc_bits) / SATURATION_DN
)
SENSOR_C = 19.0  # the simulator's sensor: 15 degrees of air and a rise of 4


def best_exposure_us(sky_mag_arcsec2: float, target: float = TARGET) -> float:
    """The exposure that puts the sky and the dark of a bin1 pixel at `target` of the full well."""
    rate = PARAMS.sky_rate_e_per_s_px(sky_mag_arcsec2) + PARAMS.dark_rate_e_per_s(SENSOR_C)
    return target * FULL_WELL_E / rate * 1e6


@dataclass(frozen=True)
class Run:
    scheduler: Scheduler
    driver: LoggingSim
    writer: ListRecordWriter
    start_utc_ns: int

    def windows(self) -> list[SeeingWindowRecord]:
        found = self.writer.of_type("seeing_window")
        return [w for w in found if isinstance(w, SeeingWindowRecord)]

    def streams(self, purpose: str) -> list[tuple[int, int, StreamConfig]]:
        return self.driver.streams(purpose)

    def events(self, kind: str) -> list[EventRecord]:
        found = [e for e in self.writer.of_type("event") if isinstance(e, EventRecord)]
        return [event for event in found if event.kind == kind]

    def sun_at(self, t_utc_ns: int) -> float:
        return float(self.driver.truth.sun_altitude_deg(t_utc_ns))

    def seconds(self, t_utc_ns: int) -> float:
        """The seconds from the start of the run."""
        return (t_utc_ns - self.start_utc_ns) / NS_PER_S


def run(
    start_utc_ns: int,
    seconds: float,
    config: SchedulerConfig,
    *,
    sky: float | None = None,
    sky_at: Callable[[int], float] | None = None,
    faults: SimFaults | None = None,
    pause_s: tuple[float, float] | None = None,
    commands: Sequence[tuple[float, Command]] = (),
) -> Run:
    """Run the scheduler against the simulator for `seconds` from `start_utc_ns`.

    `sky` is a constant sky in mag/arcsec^2, and `sky_at` a sky that changes: the sky in
    mag/arcsec^2 at a UTC time in nanoseconds. Without either, the sky follows the Sun. `faults`
    are the camera's, and `pause_s` gives the seconds from the start at which `Pause` and `Resume`
    come. `commands` gives other commands, each with the seconds from the start at which it comes.
    """
    clock = VirtualClock(start_utc_ns)
    options = SimOptions(seed=3, epoch_utc_ns=start_utc_ns, psf=PsfConfig(mode="gaussian"))
    if sky is not None:
        options = replace(options, sky_mag_arcsec2=sky, twilight=False)
    if faults is not None:
        options = replace(options, faults=faults)
    driver = LoggingSim(PROFILE, clock, options)
    writer = ListRecordWriter()
    scheduler = Scheduler(
        driver=driver,
        fast=FakeFastAnalyzer(
            station_id="test",
            profile_id=PROFILE.id,
            window_s=config.fast.analysis_window_s,
            saturation_dn=SATURATION_DN,
        ),
        survey=SimSurvey(driver.truth, PROFILE.id),
        pointing=SimPointing(driver.truth),
        records=writer,
        metrics=writer,
        clock=clock,
        profile=PROFILE,
        station_id="test",
        config=config,
        site=SITE,
        escalate=lambda level: None,
    )
    driver.scheduler = scheduler
    script = list(commands)
    if pause_s is not None:
        pause, resume = pause_s
        script += [(pause, Pause()), (resume, Resume())]
    with pytest.MonkeyPatch.context() as patch:
        if sky_at is not None:
            patch.setattr(driver.truth, "sky_mag_arcsec2", sky_at)
        for at_s, command in sorted(script, key=lambda item: item[0]):
            scheduler.run_until(start_utc_ns + round(at_s * NS_PER_S))
            assert scheduler.submit(command).accepted
        scheduler.run_until(start_utc_ns + round(seconds * NS_PER_S))
    scheduler.close()
    return Run(scheduler, driver, writer, start_utc_ns)


def short_cycles(
    *, exposure_us: int = 2000, target: float = TARGET, cadence_s: float
) -> SchedulerConfig:
    """Fast periods of 1 s (two windows), short survey frames, and two bursts in a period."""
    return SchedulerConfig(
        fast=FastConfig(
            exposure_us=exposure_us,
            window_s=1.0,
            analysis_window_s=0.5,
            target_background_fraction=target,
            min_slack_fast_s=0.0,  # the cadence leaves a slack that these runs keep idle
        ),
        survey=SurveyConfig(cadence_s=cadence_s, long_exposure_s=2.0),
        search=SearchConfig(burst_frames=10, interval_s=0.25),
        loop=LoopConfig(max_sleep_s=5.0),
    )


BRIGHT_SKY = 4.4  # mag/arcsec^2, a constant sky: the simulator's at a Sun of +8.9 degrees
NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")  # the Sun limits nothing, so the search runs


@pytest.fixture(scope="module")
def bright() -> Run:
    """Ten cycles of 20 s under a constant bright sky, with the default longest exposure."""
    return run(NIGHT, 200.0, short_cycles(cadence_s=20.0), sky=BRIGHT_SKY)


class TestABrightSky:
    def test_the_first_burst_takes_its_exposure_from_the_brightness_frame(
        self, bright: Run
    ) -> None:
        """The brightness frame (1 ms, bin2) says how bright the sky is before any fast frame.

        Its offset counts as sky, about 4% of its signal in this sky, so the first exposure comes
        out that much short of the best one (1.47 ms).
        """
        best = best_exposure_us(BRIGHT_SKY)
        assert best == pytest.approx(1474.0, abs=5.0)  # the detection estimate's 1.48 ms
        _, _, first = bright.streams("search")[0]
        assert first.exposure_us == pytest.approx(best, rel=0.06)
        assert first.exposure_us < best

    def test_the_windows_settle_at_the_target(self, bright: Run) -> None:
        """From the second period on, each period takes the background of the last window.

        The loop counts the offset (0.7% of saturation) as sky, so the sky itself settles that much
        below the target, and the exposure about 2% short of the best one.
        """
        windows = bright.windows()
        later = [w for w in windows if w.t_utc_ns > NIGHT + 30 * NS_PER_S]
        assert len(later) >= 10
        settled = best_exposure_us(BRIGHT_SKY, TARGET - OFFSET_FRACTION)
        for window in later:
            assert window.exposure_us == pytest.approx(settled, rel=0.01)
            assert window.background_fraction == pytest.approx(TARGET, abs=0.005)
            assert window.background_mean_dn is not None
            assert window.star_snr is not None
            assert window.star_snr > 10.0  # Polaris shows in this sky

    def test_every_window_has_the_one_exposure_of_its_stream(self, bright: Run) -> None:
        exposures = {
            stream_id: config.exposure_us for _, stream_id, config in bright.streams("fast")
        }
        windows = bright.windows()
        assert windows
        for window in windows:
            assert window.exposure_us == exposures[window.stream_id]

    def test_without_a_target_every_period_and_burst_takes_the_longest_exposure(self) -> None:
        fixed = run(NIGHT, 60.0, short_cycles(target=0.0, cadence_s=20.0), sky=BRIGHT_SKY)
        streams = fixed.streams("search") + fixed.streams("fast")
        assert streams
        assert {config.exposure_us for _, _, config in streams} == {2000}


# The dawn of the March equinox at the synthetic site: the Sun passes -2.5 degrees at 05:48 UTC
# and +4 degrees 45 minutes later.
DAWN = next_sun_crossing_utc_ns(
    iso_to_utc_ns("2026-03-21T00:00:00Z"), SITE.latitude_deg, SITE.longitude_deg, -2.5, rising=True
)
DEV_LONGEST_US = 50_000  # the longest fast exposure of `seeingmon dev`
DAWN_CADENCE_S = 120.0


@pytest.fixture(scope="module")
def dawn() -> Run:
    assert DAWN is not None
    return run(DAWN, 45 * 60.0, short_cycles(exposure_us=DEV_LONGEST_US, cadence_s=DAWN_CADENCE_S))


class TestADawn:
    def test_the_exposure_follows_the_twilight(self, dawn: Run) -> None:
        """The exposure starts at the longest and falls as the sky brightens, never back up."""
        windows = dawn.windows()
        assert len(windows) >= 40  # two windows in each of the 22 periods, all of them measuring
        assert dawn.sun_at(windows[-1].t_utc_ns) > 3.5
        assert windows[0].exposure_us == DEV_LONGEST_US  # the best exposure is 64 ms
        assert windows[-1].exposure_us < 5000  # 3.9 ms at +3 degrees, 3.3 ms at +4
        for earlier, later in itertools.pairwise(windows):
            # The sky brightens by at least 5% in a cycle here, and the background of a window is a
            # mean over hundreds of frames, so its noise cannot lengthen the next exposure.
            assert later.exposure_us <= earlier.exposure_us

    def test_the_sky_background_stays_near_the_target_far_from_saturation(self, dawn: Run) -> None:
        """A period takes the background of the last window, which is one cycle (120 s) old.

        The sky only brightens, so no window sits below the target, less the noise of its mean
        (far below 0.005). At the steepest twilight the sky brightens by 0.3 mag in a cycle, so a
        window can sit 30% above the target: 0.391 in this run, the value that
        `docs/visibility-brief.md` cites. The loop aims at the sky and not at the star. The 50 ms
        limit of this run lets Polaris clip, which the 2 ms of `exposure_us` prevents on a real
        night, and `TestPolarisAtDusk` (`tests/services/e2e/test_night.py`) checks the star there.
        """
        followed = [w for w in dawn.windows() if w.exposure_us < DEV_LONGEST_US]
        assert len(followed) >= 30
        for window in followed:
            assert window.background_fraction is not None
            assert TARGET - 0.005 < window.background_fraction < 0.40
        assert max(w.background_fraction or 0.0 for w in dawn.windows()) < 0.40
        changes = dawn.events("scheduler.state_change")
        assert [(e.detail or {})["to"] for e in changes] == ["auto"]  # the stream never stopped

    def test_the_exposure_changes_only_where_a_period_starts(self, dawn: Run) -> None:
        """A period is one stream, and each stream has one exposure, so its windows share it."""
        exposures = {stream_id: config.exposure_us for _, stream_id, config in dawn.streams("fast")}
        by_stream: dict[int, set[int]] = {}
        for window in dawn.windows():
            by_stream.setdefault(window.stream_id, set()).add(window.exposure_us)
        assert len(by_stream) >= 20
        for stream_id, seen in by_stream.items():
            assert seen == {exposures[stream_id]}


# A constant sky of 6 mag/arcsec^2, with the longest exposure of `seeingmon dev`. The brightness
# frame and the windows ask for different exposures here: both count the offset as sky, and the
# offset adds about 18% to the sky of the 1 ms bin2 brightness frame, but 0.7% of saturation to a
# window. So the first burst of an episode takes 5.4 ms, and the loop settles at 6.3 ms (the best
# exposure is 6.4 ms), which tells the source of an exposure apart.
DIM_SKY = 6.0


def dim_cycles() -> SchedulerConfig:
    return short_cycles(exposure_us=DEV_LONGEST_US, cadence_s=20.0)


def entries_into_auto(result: Run) -> list[int]:
    changes = result.events("scheduler.state_change")
    return [e.t_utc_ns for e in changes if (e.detail or {}).get("to") == "auto"]


def fast_streams(result: Run) -> list[tuple[int, int, StreamConfig]]:
    """The search bursts and the fast periods, in order."""
    both = result.streams("search") + result.streams("fast")
    return sorted(both, key=lambda stream: (stream[0], stream[1]))


@pytest.fixture(scope="module")
def paused() -> Run:
    """Cycles of 20 s in `auto`, a pause from 30 to 40 s, and `auto` again after the resume."""
    return run(NIGHT, 60.0, dim_cycles(), sky=DIM_SKY, pause_s=(30.0, 40.0))


@pytest.fixture(scope="module")
def faulted() -> Run:
    """A camera timeout at the 120th frame, in the second cycle, and the recovery in `auto`."""
    faults = SimFaults(scripted_timeouts=frozenset({120}))
    return run(NIGHT, 60.0, dim_cycles(), sky=DIM_SKY, faults=faults)


class TestAnEpisodeOfAuto:
    def test_an_entry_into_auto_takes_the_first_exposure_from_the_brightness_frame(
        self, paused: Run
    ) -> None:
        """The sky may have changed while the scheduler was out of `auto`, so after the resume
        the new brightness frame sets the first exposure, and not the last window."""
        entries = entries_into_auto(paused)
        assert len(entries) == 2  # the start, and the resume through `safe`
        first = paused.streams("search")[0][2].exposure_us
        settled = [w for w in paused.windows() if w.t_utc_ns < entries[1]][-1].exposure_us
        assert settled > first * 1.1  # the two sources differ (see `DIM_SKY`)
        resumed = next(config for t, _, config in fast_streams(paused) if t >= entries[1])
        # One sky gives one brightness frame, up to the noise of its median, far below 0.5%.
        assert resumed.exposure_us == pytest.approx(first, rel=0.005)

    def test_a_fault_inside_auto_keeps_the_background_of_the_last_window(
        self, faulted: Run
    ) -> None:
        """A fault ends measure but not the episode, so the next burst scales the last window."""
        (fault,) = faulted.events("scheduler.fault")
        assert len(entries_into_auto(faulted)) == 1  # the fault keeps `auto`
        before = [w for w in faulted.windows() if w.t_utc_ns < fault.t_utc_ns]
        assert len(before) >= 2  # a whole period, so the loop had settled
        last = before[-1]
        assert last.background_fraction is not None
        kept = adapted_exposure_us(
            last.exposure_us,
            last.background_fraction,
            target=TARGET,
            shortest_us=PROFILE.limits.exposure_us_range[0],
            longest_us=DEV_LONGEST_US,
        )
        after = next(config for t, _, config in fast_streams(faulted) if t > fault.t_utc_ns)
        assert after.exposure_us == kept
        first = faulted.streams("search")[0][2].exposure_us
        assert after.exposure_us > first * 1.1  # not the brightness frame's (see `DIM_SKY`)


# --- The daylight gate judges the fast stream ------------------------------------------------

SHORTEST_US = PROFILE.limits.exposure_us_range[0]
# A sunny sky that the fast stream can take: 18% of saturation at 32 us, where the best exposure is
# 53 us. It clips the 1 ms bin2 brightness frame, which gives only a lower bound of 3.3%.
SUNNY_SKY = 0.8
GATE_LIMIT = 0.5  # [scheduler.daylight] saturation_limit
GATE_CADENCE_S = 20.0
# The brightening sky: the sunny sky for a minute, and then 1 mag brighter every 150 s.
RAMP_START_S = 60.0
RAMP_S_PER_MAG = 150.0


def fast_fraction(sky_mag_arcsec2: float, exposure_us: float = SHORTEST_US) -> float:
    """The sky and the dark of a bin1 pixel at gain 0, as a share of the full well."""
    rate = PARAMS.sky_rate_e_per_s_px(sky_mag_arcsec2) + PARAMS.dark_rate_e_per_s(SENSOR_C)
    return rate * exposure_us * 1e-6 / FULL_WELL_E


def brightening_sky(t_utc_ns: int) -> float:
    """The sky of `brightening`, in mag/arcsec^2."""
    seconds = (t_utc_ns - NIGHT) / NS_PER_S
    return SUNNY_SKY - max(0.0, seconds - RAMP_START_S) / RAMP_S_PER_MAG


def bright_watch_times(result: Run) -> list[int]:
    """The times of the watch frames at the profile's shortest exposure."""
    return [t for t, _, config in result.streams("watch") if config.exposure_us == SHORTEST_US]


def state_changes(result: Run) -> list[tuple[int, str, str]]:
    """The time, the state before, and the state after, of each change of state."""
    return [
        (e.t_utc_ns, (e.detail or {})["from"], (e.detail or {})["to"])
        for e in result.events("scheduler.state_change")
    ]


@pytest.fixture(scope="module")
def sunny() -> Run:
    """Fifteen cycles of 20 s under a constant sunny sky."""
    return run(NIGHT, 300.0, short_cycles(cadence_s=GATE_CADENCE_S), sky=SUNNY_SKY)


@pytest.fixture(scope="module")
def brightening() -> Run:
    """Cycles of 20 s under a sky that brightens past what the fast stream can take."""
    return run(NIGHT, 400.0, short_cycles(cadence_s=GATE_CADENCE_S), sky_at=brightening_sky)


class TestTheGateJudgesTheFastStream:
    """In `auto` the daylight gate judges the background of the bursts and windows, scaled to the
    profile's shortest exposure, as well as the 1 ms frame of each survey step
    (`seeingmon.scheduler.gates`). The 1 ms frame clips in these skies, so only the fast stream
    measures them, and a survey step takes a watch frame of 32 us only when it does not."""

    def test_a_sunny_sky_where_the_fast_stream_is_fine_stays_in_auto(self, sunny: Run) -> None:
        assert fast_fraction(SUNNY_SKY) == pytest.approx(0.18, abs=0.01)
        assert [(before, after) for _, before, after in state_changes(sunny)] == [("safe", "auto")]
        # In `safe`, the clipped 1 ms watch frame was followed by one of 32 us, which opened the
        # gate. In `auto`, the bursts decide, so no survey step takes such a frame.
        first_burst = sunny.streams("search")[0][0]
        bright = bright_watch_times(sunny)
        assert len(bright) == 1
        assert bright[0] < first_burst
        steps = [t for t, _, config in sunny.streams("survey") if config.exposure_us == 1000]
        assert len([t for t in steps if t > first_burst]) >= 14  # one in each cycle of 20 s
        # The bursts settle at the best exposure, and the gate judges what they measure. The
        # offset of their frames, 0.7% of saturation, scales down with the exposure to 0.4%.
        status = sunny.scheduler.status()
        assert status.background_fraction == pytest.approx(fast_fraction(SUNNY_SKY), abs=0.01)
        assert status.counters.survey_long_skips >= 13

    def test_a_burst_beyond_the_limit_at_the_shortest_exposure_ends_auto(
        self, brightening: Run
    ) -> None:
        """The bursts shorten their exposure to 32 us as the sky brightens, and the first gate
        check after a burst at 32 us reads more than 50% sends the scheduler to `safe`."""
        changes = state_changes(brightening)
        assert [(before, after) for _, before, after in changes] == [
            ("safe", "auto"),
            ("auto", "safe"),
        ]
        stopped = changes[1][0]
        crossed_s = next(
            s / 10.0
            for s in range(4000)
            if fast_fraction(brightening_sky(NIGHT + s * NS_PER_S // 10)) >= GATE_LIMIT
        )
        assert crossed_s == pytest.approx(227.0, abs=1.0)
        # A cycle (20 s) at most passes between the crossing and the next gate check.
        assert 0.0 < brightening.seconds(stopped) - crossed_s < GATE_CADENCE_S
        before = [config for t, _, config in fast_streams(brightening) if t < stopped]
        assert before[-1].exposure_us == SHORTEST_US
        first_burst = brightening.streams("search")[0][0]
        assert not [t for t in bright_watch_times(brightening) if first_burst <= t <= stopped]

    def test_after_commissioning_the_first_burst_takes_the_shortest_exposure(self) -> None:
        """A sweep in a sunny sky, and `auto` again: the fast background of the episode is gone.

        The last brightness frame is the clipped 1 ms frame of a survey step, a lower bound of
        3.3%, which would ask for 291 us, more than 5 times the best exposure. The first burst
        therefore takes the shortest exposure, and the next one scales its background.
        """
        sweep = QueueSweep(exposure_us=(2000,), gain=(0,), roi_arcmin=(4.1,), window_s=1.0)
        result = run(
            NIGHT,
            160.0,
            short_cycles(cadence_s=GATE_CADENCE_S),
            sky=SUNNY_SKY,
            commands=((90.0, sweep),),
        )
        changes = state_changes(result)
        returned = next(t for t, before, _ in changes if before == "commission")
        assert changes[-1][1:] == ("commission", "auto")
        after = [config.exposure_us for t, _, config in fast_streams(result) if t >= returned]
        assert after[0] == SHORTEST_US
        best = TARGET / fast_fraction(SUNNY_SKY) * SHORTEST_US
        assert best == pytest.approx(53.5, abs=0.5)
        # The offset counts as sky, so the exposure falls about 4% short of the best one.
        assert after[1] == pytest.approx(best, rel=0.06)
