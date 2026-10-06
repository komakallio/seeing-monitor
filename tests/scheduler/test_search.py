"""The search and measure modes of the fast stream, by day and by night.

In `auto` with a pointing solution, the fast stream searches for Polaris in short bursts, and two
detecting bursts in a row switch it to measure, the fast stream with seeing windows. A star that
stays missing in measure returns the stream to search. While the Sun is above the search limit
(`[scheduler.search] max_sun_elevation_deg`: no limit by default, and 12 degrees in the
scenarios), only a probe burst every 10 minutes looks for Polaris. The Sun's elevation gates
nothing else.

The scenarios run on the virtual clock at the synthetic site of `tests.scheduler.scenario`. A
burst there takes 3 frames of 2 s, and the SNR of Polaris follows the centroid aperture's formula
of the detection estimate, which the scenarios use so that a hazy sky can hide Polaris: Polaris is
detectable (an SNR of 10) while the sky at 1 ms in bin2 stays below 0.29 of saturation, and the
fast analysis loses it (an SNR of 6) above 0.80. `pole_sky` gives the sky of the simulator near
the pole, scaled to a daylight value:

- 0.21 (the simulator's daylight) keeps Polaris visible all day.
- 0.85 (a hazy day) hides it: measure loses it at +9.6 degrees, and the search finds it again
  below +3.4 degrees. The fast stream at 32 us would see 3% of saturation, far below the limit
  of the gate, so the scheduler stays in `auto` all day, and a start in `safe` enters `auto` at
  the first brightness frame.

On June 21 at the synthetic site, the Sun passes +12 degrees at 05:10 and 18:54 UTC, and it
reaches +58 degrees at noon. The star of the scenario drifts off the sensor after 13 hours, so a
day runs as a morning and an evening.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace

import pytest

from seeingmon.analysis import StarState
from seeingmon.clock import NS_PER_S, ClockStatus, VirtualClock, iso_to_utc_ns
from seeingmon.frames import Frame
from seeingmon.scheduler import Command, Pause, QueueSweep, Resume, State
from seeingmon.scheduler import activity as words
from seeingmon.scheduler.config import SearchConfig
from seeingmon.scheduler.ephemeris import next_sun_crossing_utc_ns
from seeingmon.scheduler.status import SchedulerStatus
from tests.scheduler.scenario import BLINDING_LIGHT, SITE, TEST_CONFIG, World, pole_sky

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")  # the Sun is 40 degrees down
JUNE = iso_to_utc_ns("2026-06-21T02:00:00Z")  # the Sun is 7.5 degrees down, and rising
JUNE_NOON = iso_to_utc_ns("2026-06-21T12:00:00Z")  # the Sun is 58 degrees up
JUNE_AFTERNOON = iso_to_utc_ns("2026-06-21T16:00:00Z")  # the Sun is 33 degrees up, and sinking
LIMIT_DEG = TEST_CONFIG.search.max_sun_elevation_deg
HAZY = pole_sky(0.85)
INTERVAL_S = TEST_CONFIG.search.interval_s
PROBE_S = TEST_CONFIG.search.probe_interval_s
CYCLE_S = TEST_CONFIG.survey.cadence_s


def crossing(world: World, elevation: float, *, rising: bool) -> float:
    """When the Sun passes an elevation after the start, in seconds since the start."""
    found = next_sun_crossing_utc_ns(
        world.start_utc_ns,
        SITE.latitude_deg,
        SITE.longitude_deg,
        elevation,
        rising=rising,
        horizon_days=1.0,
    )
    assert found is not None
    return world.seconds(found)


def unsynchronized(world: World) -> None:
    assert isinstance(world.clock, VirtualClock)
    world.clock.set_status(ClockStatus(synchronized=False, error_bound_ns=None, source="test"))


def with_search(**values: float) -> World:
    """A June day with the simulator's daylight sky, and search settings of the test."""
    search = TEST_CONFIG.search.model_copy(update=values)
    config = TEST_CONFIG.model_copy(update={"search": search})
    return World(start_utc_ns=JUNE_NOON - 6 * 3600 * NS_PER_S, config=config, sky=pole_sky())


def send(command: Command) -> Callable[[World], None]:
    """A scripted action that hands a command to the scheduler, which accepts it."""

    def action(world: World) -> None:
        assert world.scheduler.submit(command).accepted

    return action


def gaps(times: list[float]) -> list[float]:
    return [round(later - earlier, 3) for earlier, later in itertools.pairwise(times)]


@dataclass(frozen=True)
class Day:
    world: World
    sample: SchedulerStatus  # the status at a time of the day that the fixture names


def run_day(world: World, until_s: float, sample_s: float) -> Day:
    taken: list[SchedulerStatus] = []
    world.at(sample_s, lambda w: taken.append(w.scheduler.status()))
    world.run_until(until_s)
    world.close()
    return Day(world, taken[0])


@pytest.fixture(scope="module")
def clear_day() -> Day:
    """June 21 from 02:00 to 14:00 UTC, in the simulator's daylight sky: Polaris shows all day.

    The sample is the status at noon.
    """
    world = World(start_utc_ns=JUNE, sky=pole_sky())
    return run_day(world, 12 * 3600, world.seconds(JUNE_NOON))


@pytest.fixture(scope="module")
def hazy_morning() -> Day:
    """June 21 from 02:00 to 08:00 UTC, in a hazy sky. The sample is the status at 07:00."""
    world = World(start_utc_ns=JUNE, sky=HAZY)
    return run_day(world, 6 * 3600, 5 * 3600)


@pytest.fixture(scope="module")
def hazy_evening() -> Day:
    """June 21 from 16:00 to 20:30 UTC, in a hazy sky. The sample is the status at 17:00."""
    world = World(start_utc_ns=JUNE_AFTERNOON, sky=HAZY)
    return run_day(world, 4.5 * 3600, 3600)


class TestADayWithAVisiblePolaris:
    def test_the_stream_measures_from_the_night_through_noon(self, clear_day: Day) -> None:
        world = clear_day.world
        assert world.states_visited() == ["safe", "auto"]  # the Sun gates nothing
        (visible,) = world.events("polaris.visible")
        assert world.seconds(visible.t_utc_ns) == pytest.approx(21.0, abs=0.1)
        assert (visible.detail or {})["sun_elevation_deg"] < -6.0
        # Measure ends only when the run ends.
        assert [(e.detail or {})["reason"] for e in world.events("polaris.hidden")] == ["shutdown"]

    def test_windows_come_all_day_with_twilight_then_daylight(self, clear_day: Day) -> None:
        """The Sun rises from -7.5 degrees: `twilight` up to the horizon, `daylight` above it."""
        world = clear_day.world
        windows = world.windows()
        hours = {int(world.seconds(w.t_utc_ns) // 3600) for w in windows}
        assert hours == set(range(12))  # every hour of the run
        sunrise = world.t(crossing(world, 0.0, rising=True))
        margin = 120 * NS_PER_S  # a window spans 60 s, and the context refreshes within it
        before = [w for w in windows if w.t_utc_ns < sunrise - margin]
        after = [w for w in windows if w.t_utc_ns > sunrise + margin]
        assert before
        assert after
        assert all("twilight" in w.flags and "daylight" not in w.flags for w in before)
        assert all("daylight" in w.flags and "twilight" not in w.flags for w in after)
        noon = [w for w in windows if abs(w.t_utc_ns - JUNE_NOON) < 600 * NS_PER_S]
        assert noon
        assert all("daylight" in w.flags for w in noon)

    def test_survey_results_carry_twilight_then_daylight_too(self, clear_day: Day) -> None:
        world = clear_day.world
        sunrise = world.t(crossing(world, 0.0, rising=True))
        qualities = world.records("sky_quality")
        before = [r for r in qualities if r.t_utc_ns < sunrise - 60 * NS_PER_S]
        after = [r for r in qualities if r.t_utc_ns > sunrise + 60 * NS_PER_S]
        assert before
        assert after
        assert all("twilight" in r.flags for r in before)  # type: ignore[attr-defined]
        assert all("daylight" in r.flags for r in after)  # type: ignore[attr-defined]
        assert not any("twilight" in r.flags for r in after)  # type: ignore[attr-defined]

    def test_a_measuring_stream_runs_no_burst_and_no_probe(self, clear_day: Day) -> None:
        world = clear_day.world
        counters = world.scheduler.status().counters
        assert (counters.search_bursts, counters.probe_bursts) == (2, 0)
        assert world.events("polaris.search_limit_low") == []

    def test_the_status_says_that_the_stream_measures(self, clear_day: Day) -> None:
        status = clear_day.sample
        assert status.sun_elevation_deg is not None
        assert status.sun_elevation_deg > 55.0
        assert status.search is not None
        assert status.search.mode == "measure"
        assert status.search.since_utc_ns is not None
        assert abs(status.search.since_utc_ns - clear_day.world.t(21.0)) < NS_PER_S // 10
        assert status.search.next_burst_utc_ns is None


class TestADayWithoutAVisiblePolaris:
    def test_measure_loses_the_star_in_the_brightening_sky(self, hazy_morning: Day) -> None:
        world = hazy_morning.world
        hidden = world.events("polaris.hidden")
        # The run ends while the stream searches, so the close ends no measure.
        assert [(e.detail or {})["reason"] for e in hidden] == ["star_missing"]
        sun = (hidden[0].detail or {})["sun_elevation_deg"]
        assert 9.0 < sun < 10.0  # the SNR falls below 6 at about +9.6 degrees
        assert world.states_visited() == ["safe", "auto"]

    def test_the_search_runs_at_its_interval_below_the_limit(self, hazy_morning: Day) -> None:
        world = hazy_morning.world
        lost = world.hidden_times()[0]
        rise = crossing(world, LIMIT_DEG, rising=True)
        morning = [t for t in world.burst_starts() if lost < t < rise]
        assert len(morning) >= 8
        assert min(gaps(morning)) == pytest.approx(INTERVAL_S, abs=0.01)
        assert max(gaps(morning)) < CYCLE_S  # a burst in every period

    @pytest.mark.parametrize("part", ["hazy_morning", "hazy_evening"])
    def test_above_the_limit_only_a_probe_burst_runs_every_10_minutes(
        self, part: str, request: pytest.FixtureRequest
    ) -> None:
        world = request.getfixturevalue(part).world
        end = world.seconds(world.clock.utc_ns())
        if part == "hazy_morning":
            start = crossing(world, LIMIT_DEG, rising=True)
        else:
            start, end = 0.0, crossing(world, LIMIT_DEG, rising=False)
        probes = [t for t in world.burst_starts() if start + 1.0 < t < end]
        assert len(probes) >= 15
        # A probe waits for the next search period when its time falls in a survey step.
        assert all(PROBE_S - 0.01 <= gap <= PROBE_S + 60.01 for gap in gaps(probes))
        assert world.events("polaris.search_limit_low") == []  # no probe found Polaris

    def test_the_status_and_the_activity_name_the_next_probe(self, hazy_morning: Day) -> None:
        status = hazy_morning.sample
        assert status.search is not None
        assert status.search.mode == "search"
        assert status.search.probe is True
        assert status.search.next_burst_utc_ns is not None
        wait_s = (status.search.next_burst_utc_ns - status.t_utc_ns) / NS_PER_S
        assert 0.0 <= wait_s <= PROBE_S + 60.0

    def test_the_search_finds_polaris_again_in_the_evening(self, hazy_evening: Day) -> None:
        world = hazy_evening.world
        fall = crossing(world, LIMIT_DEG, rising=False)
        evening = [t for t in world.burst_starts() if t > fall + 60.0]
        assert min(gaps(evening)) == pytest.approx(INTERVAL_S, abs=0.01)
        (visible,) = world.events("polaris.visible")
        detail = visible.detail or {}
        assert 2.9 < detail["sun_elevation_deg"] < 3.45  # an SNR of 10 below about +3.4 degrees
        assert detail["probe"] is False
        assert world.fast_starts()[0] == pytest.approx(world.seconds(visible.t_utc_ns), abs=0.1)
        counters = world.scheduler.status().counters
        above = [t for t in world.burst_starts() if t < fall]
        assert counters.probe_bursts == len(above)
        status = hazy_evening.sample
        assert status.search is not None
        assert (status.search.mode, status.search.probe) == ("search", True)


class TestAProbeAboveTheLimit:
    def test_a_probe_that_finds_polaris_starts_measure_and_warns(self) -> None:
        """With a limit of 5 degrees at 06:00 in June, the first burst is a probe."""
        world = with_search(max_sun_elevation_deg=5.0)
        world.run_until(300)
        assert world.burst_starts() == [pytest.approx(0.0, abs=0.1), pytest.approx(15.0, abs=0.1)]
        (visible,) = world.events("polaris.visible")
        assert world.seconds(visible.t_utc_ns) == pytest.approx(21.0, abs=0.1)
        assert (visible.detail or {})["probe"] is True
        (warning,) = world.events("polaris.search_limit_low")
        assert warning.level == "warning"
        assert warning.t_utc_ns == visible.t_utc_ns
        detail = warning.detail or {}
        assert detail["max_sun_elevation_deg"] == 5.0
        assert detail["sun_elevation_deg"] > 15.0
        assert detail["snr"] >= TEST_CONFIG.search.detect_snr
        # The warning follows the visible event, as the measure that it explains.
        kinds = [e.kind for e in world.events() if e.kind.startswith("polaris.")]
        assert kinds == ["polaris.visible", "polaris.search_limit_low"]
        assert world.scheduler.status().counters.probe_bursts == 2
        assert world.fast_starts()[0] == pytest.approx(21.0, abs=0.1)
        world.close()

    def test_a_probe_needs_its_confirmation_and_otherwise_waits_for_the_next_probe(self) -> None:
        """The star hides during the confirming burst, so the next probe comes 10 minutes later."""
        world = with_search(max_sun_elevation_deg=5.0)
        world.hide_star(14, 30)
        world.run_until(900)
        assert world.burst_starts() == [
            pytest.approx(0.0, abs=0.1),
            pytest.approx(15.0, abs=0.1),
            pytest.approx(600.0, abs=0.1),
            pytest.approx(615.0, abs=0.1),
        ]
        assert world.visible_times() == [pytest.approx(621.0, abs=0.1)]
        assert len(world.events("polaris.search_limit_low")) == 1
        world.close()

    def test_the_activity_of_a_search_period_names_the_probe(self) -> None:
        """At 30 s the probe at 0 s is done, and the next one comes after the period ends."""
        world = with_search(max_sun_elevation_deg=5.0)
        world.hide_star(0, 10_000)
        world.run_until(30)
        status = world.scheduler.status()
        activity = status.activity
        assert activity is not None
        assert (activity.phase, activity.label) == ("search", words.PROBE_LABEL)
        assert activity.detail == words.probe_detail(5.0, PROBE_S)
        survey = TEST_CONFIG.survey
        assert activity.next_label == words.survey_step_label(
            survey.short_exposure_s, survey.long_exposure_s
        )
        assert activity.ends_utc_ns is not None
        assert activity.next_utc_ns == activity.ends_utc_ns  # the period ends before the probe
        assert status.search is not None
        assert (status.search.mode, status.search.probe) == ("search", True)
        assert status.search.next_burst_utc_ns is not None
        assert status.search.next_burst_utc_ns >= world.t(PROBE_S)
        world.close()

    def test_the_probe_timer_outlives_a_stop_in_safe(self) -> None:
        """A floodlight sends the scheduler to `safe` at the survey step of 120 s, for 3 minutes.

        `auto` returns at about 300 s, and the next probe still waits for 600 s, 10 minutes after
        the first one, instead of running at the return.
        """
        world = with_search(max_sun_elevation_deg=5.0)
        world.hide_star(0, 10_000)
        world.light(100, 250, BLINDING_LIGHT)
        world.run_until(900)
        assert world.states_visited() == ["safe", "auto", "safe", "auto"]
        returned_at = world.state_changes()[2][0]
        assert 250 < returned_at < 320
        bursts = world.burst_starts()
        assert bursts[0] == pytest.approx(0.0, abs=0.1)
        # The probe waits for a search period when its time falls in a survey step.
        assert PROBE_S - 0.01 <= bursts[1] <= PROBE_S + 60.01
        assert world.scheduler.status().counters.probe_bursts == len(bursts)
        world.close()

    def test_a_limit_of_90_degrees_or_more_lets_the_search_run_all_day(self) -> None:
        world = with_search(max_sun_elevation_deg=90.0)
        world.hide_star(0, 10_000)  # nothing to find, so the search keeps going
        world.run_until(600)
        bursts = world.burst_starts()
        assert len(bursts) >= 20
        assert min(gaps(bursts)) == pytest.approx(INTERVAL_S, abs=0.01)
        assert world.scheduler.status().counters.probe_bursts == 0
        world.close()


class TestACloudGapAtNight:
    def test_measure_ends_the_search_waits_out_the_cloud_and_measure_returns(self) -> None:
        """A thick cloud (transparency 0.01) hides Polaris from 1000 to 1300 s."""
        world = World(start_utc_ns=NIGHT)
        world.cloud(1000, 1300, 0.99)
        world.run_until(1800)
        kinds = [(world.seconds(e.t_utc_ns), e.kind) for e in world.events()]
        polaris = [(t, kind) for t, kind in kinds if kind.startswith("polaris.")]
        assert [kind for _, kind in polaris] == [
            "polaris.visible",
            "polaris.hidden",
            "polaris.visible",
        ]
        hidden = world.events("polaris.hidden")[0]
        assert world.seconds(hidden.t_utc_ns) == pytest.approx(1020.0, abs=2.5)
        assert (hidden.detail or {})["reason"] == "star_missing"
        back = world.visible_times()[1]
        # The first survey step after 1300 ends the cloud response, and the cycle runs at its
        # normal cadence again, so Polaris is back within a cycle and two bursts.
        assert 1300 < back < 1300 + CYCLE_S + 30
        # Between them the stream searched, and the fast stream did not run.
        assert not [t for t in world.fast_starts() if 1021 < t < back - 0.1]
        searching = [t for t in world.burst_starts() if 1021 < t < back]
        assert len(searching) >= 8
        assert (
            world.events("scheduler.solve_requested") == []
        )  # the cloud says nothing of the mount
        assert world.fast_starts()[-1] > back - 0.1
        world.close()


class TestTheMeasuredGate:
    def test_a_move_to_safe_ends_measure_with_hidden(self) -> None:
        """A floodlight saturates the short survey frame at the end of the period at 1080."""
        world = World(start_utc_ns=NIGHT)
        # Three frames before the survey step: too few to lose the star.
        world.light(1195, 1500, BLINDING_LIGHT)
        world.run_until(1300)
        assert world.states_visited() == ["safe", "auto", "safe"]
        (hidden,) = world.events("polaris.hidden")
        detail = hidden.detail or {}
        assert (detail["reason"], detail["state"], detail["state_reason"]) == (
            "state_change",
            "safe",
            "bright_sky",
        )
        assert world.seconds(hidden.t_utc_ns) == pytest.approx(1200.0, abs=3.0)
        assert detail["sun_elevation_deg"] < -18
        world.close()

    def test_a_move_to_safe_ends_measure_in_the_same_step(self) -> None:
        """The step that leaves `auto` ends measure, before any later step could run."""
        world = World(start_utc_ns=NIGHT)
        world.light(1195, 1500, BLINDING_LIGHT)
        world.run_until(1190)
        scheduler = world.scheduler
        while scheduler.state is State.AUTO:
            scheduler.step()
        assert scheduler.state is State.SAFE
        assert scheduler.status().search is None
        polaris = [e.kind for e in world.events() if e.kind.startswith("polaris.")]
        assert polaris[-1] == "polaris.hidden"
        world.close()

    def test_a_task_at_a_boundary_ends_measure_in_the_same_step(self) -> None:
        """A sweep queued at 170 s, while the cycle waits at its boundary, takes `commission`."""
        world = World(start_utc_ns=NIGHT)
        world.run_until(170)
        scheduler = world.scheduler
        sweep = QueueSweep(exposure_us=(2000,), gain=(0,), window_s=1.0)
        assert scheduler.submit(sweep).accepted
        while scheduler.state is State.AUTO:
            scheduler.step()
        assert scheduler.state is State.COMMISSION
        hidden = world.events("polaris.hidden")
        assert len(hidden) == 1
        detail = hidden[0].detail or {}
        assert (detail["reason"], detail["state"]) == ("state_change", "commission")
        assert world.seconds(hidden[0].t_utc_ns) == pytest.approx(170.0, abs=1.0)
        world.close()

    def test_a_saturated_sky_runs_no_burst(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.light(0, 2000, BLINDING_LIGHT)
        world.run_until(1800)
        assert world.states_visited() == ["safe"]
        assert world.burst_starts() == []
        assert world.scheduler.status().search is None
        world.close()


class TestNoSolutionNoSite:
    def test_without_a_solution_no_burst_runs_until_a_survey_step_solves(self) -> None:
        world = World(start_utc_ns=NIGHT, solved_at_start=False)
        world.no_solution(0, 400)
        world.run_until(600)
        solved_at = max(t for t, e in self.survey_longs(world) if t < 500) + 30.0
        assert world.burst_starts()
        assert min(world.burst_starts()) == pytest.approx(solved_at, abs=1.5)
        assert world.events("scheduler.solve_requested")
        world.close()

    @staticmethod
    def survey_longs(world: World) -> list[tuple[float, int]]:
        return [
            (world.seconds(c.t_utc_ns), c.config.exposure_us)
            for c in world.configures(mode="bin2", video=False)
            if c.config.roi is None and c.config.exposure_us == 30_000_000
        ]

    def test_a_lost_solution_stops_the_search_and_asks_for_a_solve(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.hide_star(0, 2000)  # the stream keeps searching
        world.at(200, lambda w: w.pointing.clear())
        world.no_solution(150, 2000)
        world.run_until(600)
        (first,) = world.events("scheduler.solve_requested")[:1]
        assert 200 <= world.seconds(first.t_utc_ns) <= 216  # at the next burst of the period
        assert not [t for t in world.burst_starts() if t > 216]
        world.close()

    def test_a_lost_solution_ends_measure_and_a_new_one_searches_again(self) -> None:
        """Measure runs from 21 s. The solution goes away at 200 s, and no solve works until 700 s.

        The fast period of 180 s runs to its end, and the boundary at 360 s finds no solution, so
        measure ends there with `polaris.hidden`. After a survey step solves, the stream searches,
        and two bursts confirm Polaris again.
        """
        world = World(start_utc_ns=NIGHT)
        world.at(200, lambda w: w.pointing.clear())
        world.no_solution(150, 700)
        world.run_until(1500)
        kinds = [e.kind for e in world.events() if e.kind.startswith("polaris.")]
        assert kinds == ["polaris.visible", "polaris.hidden", "polaris.visible"]
        (hidden,) = world.events("polaris.hidden")
        assert (hidden.detail or {})["reason"] == "no_solution"
        lost = world.seconds(hidden.t_utc_ns)
        assert 300 < lost < 400  # at the first boundary without a solution
        assert world.events("scheduler.solve_requested")
        back = world.visible_times()[1]
        assert back > 700
        searching = [t for t in world.burst_starts() if lost < t < back]
        assert len(searching) == TEST_CONFIG.search.confirm_bursts
        assert min(searching) > 700  # no burst runs without a solution
        assert not [t for t in world.fast_starts() if lost < t < back - 0.1]
        world.close()

    def test_without_a_site_the_search_has_no_limit(self) -> None:
        """At noon in June with a hazy sky: the Sun is unknown, so bursts run every 15 s."""
        world = World(start_utc_ns=JUNE_NOON, site=None, sky=HAZY)
        world.run_until(600)
        bursts = world.burst_starts()
        assert len(bursts) >= 20
        assert min(gaps(bursts)) == pytest.approx(INTERVAL_S, abs=0.01)
        status = world.scheduler.status()
        assert status.counters.probe_bursts == 0
        assert status.search is not None
        assert (status.search.mode, status.search.probe) == ("search", False)
        assert status.sun_elevation_deg is None
        world.close()

    def test_an_unsynchronized_clock_lets_the_search_run_and_the_event_has_no_sun(self) -> None:
        """At noon in June, Polaris shows. Without a clock, no probe and no warning."""
        world = World(start_utc_ns=JUNE_NOON, sky=pole_sky())
        unsynchronized(world)
        world.run_until(300)
        (visible,) = world.events("polaris.visible")
        detail = visible.detail or {}
        assert detail["sun_elevation_deg"] is None
        assert detail["probe"] is False
        assert world.events("polaris.search_limit_low") == []
        assert world.scheduler.status().counters.probe_bursts == 0
        world.close()

    def test_a_synchronized_clock_at_the_same_noon_probes_and_warns(self) -> None:
        """The same noon with a good clock: the Sun is above the limit, so the bursts are probes."""
        world = World(start_utc_ns=JUNE_NOON, sky=pole_sky())
        world.run_until(300)
        (visible,) = world.events("polaris.visible")
        assert (visible.detail or {})["probe"] is True
        assert (visible.detail or {})["sun_elevation_deg"] == pytest.approx(58.4, abs=0.2)
        assert len(world.events("polaris.search_limit_low")) == 1
        world.close()


class TestTheBursts:
    def test_burst_frames_write_no_seeing_window(self) -> None:
        """A night without the star: the bursts read frames, and nothing reaches a window."""
        world = World(start_utc_ns=NIGHT)
        world.hide_star(0, 10_000)
        world.run_until(1800)
        counters = world.scheduler.status().counters
        assert counters.search_bursts >= 50
        assert counters.search_frames == 3 * counters.search_bursts
        assert world.fast.frames_measured == counters.search_frames
        assert world.fast.frames_pushed == 0
        assert world.windows() == []
        assert world.writer.metrics == []
        assert world.events("polaris.visible") == []

    @pytest.mark.parametrize(
        ("hidden", "visible_s"),
        [
            # One of 3 frames without the star: the median (111) still detects, the minimum not.
            ((17.5, 19.5), 21.0),
            # Two of 3 frames: the median is 0, though the mean (37) and the maximum would
            # detect. The bursts at 30 and 45 s confirm Polaris instead.
            ((17.5, 21.5), 51.0),
        ],
    )
    def test_a_burst_detects_by_the_median_snr_of_its_frames(
        self, hidden: tuple[float, float], visible_s: float
    ) -> None:
        """The second burst reads frames at 16, 18, and 20 s, which arrive at 17, 19, and 21 s.

        The star hides around the middle frame, or the last two. At night a frame with the star
        has an SNR of about 111.
        """
        world = World(start_utc_ns=NIGHT)
        world.hide_star(*hidden)
        world.run_until(60)
        assert world.burst_starts()[:2] == [
            pytest.approx(0.0, abs=0.1),
            pytest.approx(15.0, abs=0.1),
        ]
        assert world.visible_times() == [pytest.approx(visible_s, abs=0.1)]
        world.close()

    def test_a_star_beyond_the_radius_is_no_detection(self) -> None:
        """A bump of 10 px at the start moves the star off the prediction, inside the ROI.

        With a radius of 5 px no burst detects, until the survey step at 120 s solves the new
        place. With the default radius of 20 px, the same bursts detect at once.
        """
        narrow = SearchConfig(burst_frames=3, radius_px=5.0)
        world = World(start_utc_ns=NIGHT, config=TEST_CONFIG.model_copy(update={"search": narrow}))
        world.jolt(0, 10.0)
        world.run_until(100)
        status = world.scheduler.status()
        assert status.search is not None
        assert (status.search.mode, status.search.detections) == ("search", 0)
        assert world.events("polaris.visible") == []
        assert len(world.burst_starts()) >= 6
        world.run_until(400)
        assert world.visible_times()
        assert world.visible_times()[0] > 120
        world.close()
        wide = World(start_utc_ns=NIGHT)
        wide.jolt(0, 10.0)
        wide.run_until(100)
        assert wide.visible_times() == [pytest.approx(21.0, abs=0.1)]
        wide.close()

    def test_a_fault_resets_the_detections(self) -> None:
        """The first burst detects. The camera fails at the second, so Polaris needs two more."""
        world = World(start_utc_ns=NIGHT)
        world.camera_fault(15, 17)
        world.run_until(120)
        assert world.scheduler.status().counters.faults >= 1
        after = [t for t in world.burst_starts() if t > 17]
        assert world.visible_times() == [pytest.approx(after[1] + 6.0, abs=0.1)]
        world.close()

    def test_a_return_to_auto_resets_the_detections(self) -> None:
        """The first burst detects. A pause at 8 s and a resume at 10 s leave `auto` and return.

        Back in `auto`, Polaris needs two detecting bursts again.
        """
        world = World(start_utc_ns=NIGHT)
        world.at(8, send(Pause()))
        world.at(10, send(Resume()))
        world.run_until(200)
        assert world.states_visited()[:5] == ["safe", "auto", "paused", "safe", "auto"]
        returned_at = world.state_changes()[3][0]
        after = [t for t in world.burst_starts() if t >= returned_at]
        assert after[0] == pytest.approx(returned_at, abs=1.0)
        assert world.visible_times() == [pytest.approx(after[1] + 6.0, abs=0.1)]
        world.close()

    def test_a_burst_reads_its_frames_on_the_roi_of_the_prediction(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.run_until(30)
        bursts = world.configures(purpose="search")
        assert len(bursts) == 2
        for call in bursts:
            roi = call.config.roi
            assert roi is not None
            x, y = world.star_position(call.t_utc_ns)
            assert roi.distance_to_edge(x, y) >= 14  # centered to within a pixel or two
            fast = TEST_CONFIG.fast
            assert (call.config.exposure_us, call.config.gain) == (fast.exposure_us, fast.gain)
        world.close()

    def test_the_status_and_the_activity_name_the_next_burst(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.run_until(8)  # the first burst ended at 6 s, and the next one starts at 15 s
        status = world.scheduler.status()
        assert status.search is not None
        assert (status.search.mode, status.search.probe, status.search.detections) == (
            "search",
            False,
            1,
        )
        assert status.search.next_burst_utc_ns is not None
        assert abs(status.search.next_burst_utc_ns - world.t(15.0)) < NS_PER_S // 10
        activity = status.activity
        assert activity is not None
        assert (activity.phase, activity.label) == ("search", words.SEARCH_LABEL)
        assert activity.next_label == words.burst_label(TEST_CONFIG.search.burst_frames)
        assert activity.next_utc_ns == status.search.next_burst_utc_ns
        assert activity.ends_utc_ns is not None
        assert (activity.ends_utc_ns - activity.since_utc_ns) / NS_PER_S == pytest.approx(120.0)
        assert activity.detail is not None
        assert "1 of 2 detections in a row" in activity.detail
        world.close()

    def test_a_burst_that_cannot_finish_before_the_period_ends_waits(self) -> None:
        """Bursts every 58 s: at 0 and 58 s. The one at 116 s would end at 122 s, after the period.

        It waits for the next period, so the survey step keeps its cadence.
        """
        search = SearchConfig(burst_frames=3, interval_s=58.0)
        world = World(start_utc_ns=NIGHT, config=TEST_CONFIG.model_copy(update={"search": search}))
        world.hide_star(0, 10_000)
        world.run_until(170)
        assert world.burst_starts() == [pytest.approx(0.0, abs=0.1), pytest.approx(58.0, abs=0.1)]
        shorts = [
            world.seconds(c.t_utc_ns)
            for c in world.configures(purpose="survey")
            if c.config.exposure_us == 1000
        ]
        assert shorts == [pytest.approx(120.0, abs=0.1)]
        world.run_until(400)
        assert world.burst_starts()[2] == pytest.approx(CYCLE_S, abs=0.1)  # the next period
        world.close()


class TestTheStatisticOfABurst:
    """A burst decides by the matched SNR of its frames, and it looks within its radius.

    The fake analysis gives each star one SNR, so these tests replace the two SNRs of the stars
    that `measure` returns: the centroid aperture's `snr` and the matched filter's `matched_snr`.
    """

    @staticmethod
    def night_with_snrs(
        monkeypatch: pytest.MonkeyPatch, snr: float, matched_snr: float
    ) -> tuple[World, list[tuple[tuple[float, float] | None, float | None]]]:
        """A night whose search frames report these SNRs, and the place and radius of each."""
        world = World(start_utc_ns=NIGHT)
        calls: list[tuple[tuple[float, float] | None, float | None]] = []
        measure = world.fast.measure

        def replaced(
            frame: Frame, at: tuple[float, float] | None = None, radius_px: float | None = None
        ) -> StarState:
            calls.append((at, radius_px))
            star = measure(frame, at, radius_px)
            return replace(star, snr=snr, matched_snr=matched_snr) if star.found else star

        monkeypatch.setattr(world.fast, "measure", replaced)
        return world, calls

    def test_a_high_matched_snr_detects_where_the_aperture_would_not(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The matched SNR of 30 detects, though the aperture's 4 is below `detect_snr`."""
        world, _ = self.night_with_snrs(monkeypatch, snr=4.0, matched_snr=30.0)
        world.run_until(60)
        assert world.visible_times() == [pytest.approx(21.0, abs=0.1)]  # the second burst
        world.close()

    def test_a_low_matched_snr_detects_nothing_where_the_aperture_would(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The matched SNR of 4 detects nothing, though the aperture's 30 is above `detect_snr`."""
        world, _ = self.night_with_snrs(monkeypatch, snr=30.0, matched_snr=4.0)
        world.run_until(100)
        assert len(world.burst_starts()) >= 6
        assert world.events("polaris.visible") == []
        status = world.scheduler.status()
        assert status.search is not None
        assert status.search.detections == 0
        world.close()

    def test_a_frame_of_a_burst_is_searched_within_the_radius_of_the_prediction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        world, calls = self.night_with_snrs(monkeypatch, snr=30.0, matched_snr=30.0)
        world.run_until(8)  # the first burst
        assert len(calls) == TEST_CONFIG.search.burst_frames
        start = world.configures(purpose="search")[0].t_utc_ns
        predicted = world.star_position(start)
        for at, radius_px in calls:
            assert radius_px == TEST_CONFIG.search.radius_px
            assert at is not None
            # 0.5 px: the star drifts 0.087 px a second, and the burst keeps its first prediction.
            assert at == pytest.approx(predicted, abs=0.5)
        world.close()
