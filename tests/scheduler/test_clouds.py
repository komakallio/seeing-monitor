"""The response to clouds: shorter fast periods, a faster survey cadence, and the `cloud` flag.

The survey analysis reports a cloud fraction for each long exposure. At or above 0.5 the cloud
response starts, and at or below 0.3 it ends. While it applies, the fast period is 60 seconds
(normally 120), the survey cadence is 100 seconds (normally 180), and windows carry `cloud`.
"""

from __future__ import annotations

import itertools

import pytest

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from tests.scheduler.scenario import World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")
CLOUD_START, CLOUD_END = 1500.0, 3000.0
END_S = 5000


def fast_starts(world: World) -> list[float]:
    return [world.seconds(call.t_utc_ns) for call in world.configures(mode="bin1", video=True)]


@pytest.fixture(scope="module")
def cloudy_world() -> World:
    world = World(start_utc_ns=NIGHT)
    world.cloud(CLOUD_START, CLOUD_END, 0.8)
    world.run_until(END_S)
    world.close()
    return world


def event_times(world: World, kind: str) -> list[float]:
    return [world.seconds(event.t_utc_ns) for event in world.events(kind)]


def test_the_cloud_response_starts_with_the_first_survey_result_under_cloud(
    cloudy_world: World,
) -> None:
    started, ended = (
        event for event in cloudy_world.events("scheduler.cloud")
    )  # exactly two: on and off
    assert (started.detail or {})["active"] is True
    assert (ended.detail or {})["active"] is False
    # A long exposure that ends inside the cloud reports it. The result comes at once.
    assert CLOUD_START < cloudy_world.seconds(started.t_utc_ns) < CLOUD_START + 180 + 40
    # The first long exposure after the cloud leaves reports a clear sky.
    assert CLOUD_END < cloudy_world.seconds(ended.t_utc_ns) < CLOUD_END + 180 + 40


def test_the_fast_period_shortens_to_one_analysis_window_under_cloud(cloudy_world: World) -> None:
    on, off = event_times(cloudy_world, "scheduler.cloud")
    windows = [w for w in cloudy_world.windows() if on + 5 < cloudy_world.seconds(w.t_utc_ns) < off]
    assert len(windows) >= 10
    # One window of 60 seconds in each period, and a new stream for each period.
    assert {w.n_frames for w in windows} == {30}
    assert len({w.stream_id for w in windows}) == len(windows)


def test_windows_under_cloud_carry_the_flag_and_other_windows_do_not(cloudy_world: World) -> None:
    on, off = event_times(cloudy_world, "scheduler.cloud")
    for window in cloudy_world.windows():
        start = cloudy_world.seconds(window.t_utc_ns)
        end = start + window.duration_s
        if on + 5 <= start and end <= off:
            assert "cloud" in window.flags, start
        elif end <= on - 5 or start >= off + 5:
            assert "cloud" not in window.flags, start
    flagged = [w for w in cloudy_world.windows() if "cloud" in w.flags]
    assert 10 <= len(flagged) < len(cloudy_world.windows())


def test_the_survey_runs_on_the_shorter_cadence_while_the_cloud_lasts(cloudy_world: World) -> None:
    on, off = event_times(cloudy_world, "scheduler.cloud")
    starts = [t for t in fast_starts(cloudy_world) if on + 1 < t < off]
    assert len(starts) >= 10
    gaps = [later - earlier for earlier, later in itertools.pairwise(starts)]
    assert all(gap == pytest.approx(100.0, abs=0.05) for gap in gaps)


def test_the_normal_cycle_returns_after_the_cloud(cloudy_world: World) -> None:
    _, off = event_times(cloudy_world, "scheduler.cloud")
    starts = [t for t in fast_starts(cloudy_world) if t > off + 1]
    assert len(starts) >= 8
    gaps = [later - earlier for earlier, later in itertools.pairwise(starts)]
    assert all(gap == pytest.approx(180.0, abs=0.05) for gap in gaps)
    # The last window closed when the test ended the run, so it holds only part of a period.
    after = [
        w
        for w in cloudy_world.windows()
        if off + 200 < cloudy_world.seconds(w.t_utc_ns) < END_S - 120
    ]
    assert after
    assert all(w.flags == [] and w.n_frames == 30 for w in after)


def test_a_change_of_cadence_is_not_an_overrun(cloudy_world: World) -> None:
    assert cloudy_world.scheduler.status().counters.cadence_overruns == 0


def test_survey_results_under_cloud_carry_the_flag_as_well(cloudy_world: World) -> None:
    qualities = cloudy_world.records("sky_quality")
    assert qualities
    for record in qualities:
        cloudy = CLOUD_START <= cloudy_world.seconds(record.t_utc_ns) < CLOUD_END
        flags = record.flags  # type: ignore[attr-defined]
        assert ("cloud" in flags) == cloudy


def test_the_status_follows_the_cloud(cloudy_world: World) -> None:
    status = cloudy_world.scheduler.status()
    assert status.cloud is False
    assert status.cloud_fraction == 0.0
    assert cloudy_world.scheduler.state.value == "auto"


def test_a_fraction_between_the_thresholds_does_not_start_the_response() -> None:
    world = World(start_utc_ns=NIGHT)
    world.cloud(0, 3000, 0.4)  # between 0.3 and 0.5
    world.run_until(1500)
    assert world.events("scheduler.cloud") == []
    assert all("cloud" not in w.flags for w in world.windows())
    world.close()


def test_the_response_holds_while_the_fraction_stays_between_the_thresholds() -> None:
    """Hysteresis: after the response starts at 0.8, a fraction of 0.4 does not end it."""
    world = World(start_utc_ns=NIGHT)
    world.cloud(0, 1000, 0.8)
    world.cloud(1000, 3000, 0.4)
    world.run_until(2500)
    assert [(e.detail or {})["active"] for e in world.events("scheduler.cloud")] == [True]
    world.close()


def test_the_survey_exposures_in_cloud_are_still_taken(cloudy_world: World) -> None:
    """The scheduler lowers the cadence under cloud, so it takes more surveys, not fewer."""
    on, off = event_times(cloudy_world, "scheduler.cloud")
    long_frames = [
        f
        for f in cloudy_world.survey.submitted
        if f.exposure_us >= 1_000_000 and on <= (f.t_utc_ns - NIGHT) / NS_PER_S < off
    ]
    assert len(long_frames) >= 14  # one per 100 seconds in 1,500 seconds
