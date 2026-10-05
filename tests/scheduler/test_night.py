"""A whole night in virtual time: 12 hours, with everything that can happen to it.

The scenario starts at 14:30 UTC on a winter day at a synthetic site (55 degrees north on the
prime meridian), in daylight. The scheduler watches the sky, enters `auto` at dusk when the fast
stream at its shortest exposure would see less than 35% of saturation, searches for Polaris until
it shows, and then the script disturbs the night, one thing at a time:

| Time (hours) | What happens |
|---|---|
| 3.0 | A sweep is queued. It runs at the next cycle boundary. |
| 4.0 | The operator pauses the scheduler, and resumes it 5 minutes later. |
| 5.0 | The star disappears for a minute, which ends a fast period early. No solve follows. |
| 7.0 | Clouds cover 80% of the field for 30 minutes. |
| 8.0 | The operator starts the alignment helper, and stops it 10 minutes later. |
| 9.0 | The camera times out for 10 seconds. |
| 10.0 | The mount is bumped, and the star lands near the edge of the ROI. |

Every disturbance but the clouds and the bump ends measure with `polaris.hidden`, and the search
that follows finds Polaris again with `polaris.visible`. The test checks the records and the flags
that the night produces. It runs in a few seconds of real time, because the clock is virtual.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from seeingmon.clock import NS_PER_S
from seeingmon.drivers.base import RecoveryLevel
from seeingmon.scheduler import (
    Command,
    Pause,
    QueueSweep,
    Resume,
    StartAlignment,
    StopAlignment,
)
from seeingmon.scheduler.ephemeris import next_sun_crossing_utc_ns
from seeingmon.scheduler.events import EVENT_KINDS
from tests.scheduler.scenario import SITE, START, TEST_CONFIG, World

HOUR = 3600.0
NIGHT_LENGTH_S = 12 * HOUR

SWEEP_AT = 3.0 * HOUR
PAUSE_AT, RESUME_AT = 4.0 * HOUR, 4.0 * HOUR + 300
HIDE_FROM, HIDE_TO = 5.0 * HOUR + 10, 5.0 * HOUR + 70
CLOUD_FROM, CLOUD_TO = 7.0 * HOUR, 7.5 * HOUR
ALIGN_FROM, ALIGN_TO = 8.0 * HOUR, 8.0 * HOUR + 600
FAULT_FROM, FAULT_TO = 9.0 * HOUR, 9.0 * HOUR + 10
JOLT_AT = 10.0 * HOUR + 20


@dataclass(frozen=True)
class Night:
    world: World
    real_seconds: float


def sender(command: Command) -> Callable[[World], None]:
    """An action for `World.at` that submits a command."""

    def send(world: World) -> None:
        world.scheduler.submit(command)

    return send


@pytest.fixture(scope="module")
def night() -> Night:
    world = World()
    world.cloud(CLOUD_FROM, CLOUD_TO, 0.8)
    world.hide_star(HIDE_FROM, HIDE_TO)
    world.camera_fault(FAULT_FROM, FAULT_TO)
    world.jolt(JOLT_AT, 13)
    sweep = QueueSweep(exposure_us=(2000, 5000), gain=(0,), roi_arcmin=(4.1,), window_s=2.0)
    world.at(SWEEP_AT, sender(sweep))
    world.at(PAUSE_AT, sender(Pause()))
    world.at(RESUME_AT, sender(Resume()))
    world.at(ALIGN_FROM, sender(StartAlignment(exposure_s=10.0)))
    world.at(ALIGN_TO, sender(StopAlignment()))
    started = time.perf_counter()
    world.run_until(NIGHT_LENGTH_S)
    elapsed = time.perf_counter() - started
    world.close()
    return Night(world, elapsed)


def crossing(elevation: float, *, rising: bool) -> float:
    found = next_sun_crossing_utc_ns(
        START, SITE.latitude_deg, SITE.longitude_deg, elevation, rising=rising, horizon_days=1.0
    )
    assert found is not None
    return (found - START) / NS_PER_S


def disturbed(world: World, t: float, margin: float = 400.0) -> bool:
    """Whether a time lies close to something that the script did to the night.

    The first period that measures at dusk counts too, because the search hands it over in its
    middle.
    """
    dusk = world.visible_times()[0]
    spans = [
        (dusk, dusk),
        (SWEEP_AT, SWEEP_AT + 400),
        (PAUSE_AT, RESUME_AT),
        (HIDE_FROM, HIDE_TO + 200),
        (CLOUD_FROM, CLOUD_TO + 200),
        (ALIGN_FROM, ALIGN_TO),
        (FAULT_FROM, FAULT_TO + 100),
        (JOLT_AT, JOLT_AT + 200),
    ]
    return any(start - margin <= t <= end + margin for start, end in spans)


class TestTheShapeOfTheNight:
    def test_the_night_takes_seconds_of_real_time(self, night: Night) -> None:
        # Twelve hours of virtual time. The bound is loose, because CI machines differ.
        assert night.real_seconds < 120

    def test_the_states_follow_the_script(self, night: Night) -> None:
        assert night.world.states_visited() == [
            "safe",  # daylight
            "auto",  # dusk
            "commission",  # the sweep, at a cycle boundary
            "auto",
            "paused",
            "safe",  # a resume goes through safe
            "auto",
            "align",
            "safe",  # a stop goes through safe
            "auto",
        ]
        assert night.world.scheduler.state.value == "auto"

    def test_every_change_of_state_has_an_event_with_a_reason(self, night: Night) -> None:
        world = night.world
        events = world.events("scheduler.state_change")
        assert len(events) == world.scheduler.status().counters.transitions == 9
        for event in events:
            detail = event.detail or {}
            assert detail["from"] != detail["to"]
            assert detail["reason"]
        reasons = [(e.detail or {})["reason"] for e in events]
        assert reasons == [
            "the sky is dark enough",
            "a task is queued",
            "commissioning is done",
            "pause command",
            "resume command",
            "the sky is dark enough",
            "alignment started",
            "alignment stopped",
            "the sky is dark enough",
        ]

    def test_the_changes_come_at_the_scripted_times(self, night: Night) -> None:
        world = night.world
        times = [t for t, _, _ in world.state_changes()]
        dusk = crossing(-0.420, rising=False)
        # The change comes with the first brightness frame that shows a fast background below the
        # resume level (35%).
        assert dusk <= times[0] <= dusk + 61
        # The boundary of the cycle that was running, which may be the moment of the command.
        assert SWEEP_AT <= times[1] < SWEEP_AT + 200
        # The test injects a command between steps, so a survey exposure in progress holds it.
        assert PAUSE_AT <= times[3] <= PAUSE_AT + 31.0
        assert times[4] == pytest.approx(RESUME_AT, abs=1.0)
        assert times[6] == pytest.approx(ALIGN_FROM, abs=1.0)
        assert times[7] == pytest.approx(ALIGN_TO, abs=1.0)

    def test_the_night_ends_healthy(self, night: Night) -> None:
        status = night.world.scheduler.status()
        assert (status.state, status.degraded, status.queued_tasks) == ("auto", False, 0)
        assert status.fault.failures == 0
        assert status.counters.cadence_overruns == 0


class TestDaylightThenDusk:
    def test_the_sky_is_watched_once_a_minute_until_dusk(self, night: Night) -> None:
        world = night.world
        dusk = world.state_changes()[0][0]
        watches = [
            world.seconds(c.t_utc_ns)
            for c in world.configures(mode="bin2", video=False)
            if c.config.exposure_us == 1000
            and c.config.roi is not None
            and world.seconds(c.t_utc_ns) <= dusk
        ]
        gaps = [later - earlier for earlier, later in itertools.pairwise(watches)]
        assert all(gap == pytest.approx(60.0, abs=0.01) for gap in gaps)
        assert len(watches) == pytest.approx(dusk / 60 + 1, abs=1)
        assert world.configures(mode="bin1")[0].t_utc_ns >= world.t(
            dusk
        )  # no fast frame in daylight

    def test_windows_in_twilight_carry_the_flag_and_those_after_it_do_not(
        self, night: Night
    ) -> None:
        world = night.world
        t18 = crossing(-18.0, rising=False)
        windows = world.windows()
        first = windows[0]
        assert "twilight" in first.flags
        for window in windows:
            start = world.seconds(window.t_utc_ns)
            if start + window.duration_s <= t18 - 60:
                assert "twilight" in window.flags, start
            elif start >= t18 + 60:
                assert "twilight" not in window.flags, start

    def test_the_survey_results_in_twilight_are_flagged_too(self, night: Night) -> None:
        world = night.world
        t18 = crossing(-18.0, rising=False)
        for record in world.records("sky_quality"):
            t = world.seconds(record.t_utc_ns)
            flags = record.flags  # type: ignore[attr-defined]
            if t <= t18 - 60:
                assert "twilight" in flags
            elif t >= t18 + 60 and not CLOUD_FROM <= t <= CLOUD_TO + 200:
                assert "twilight" not in flags


class TestTheCadence:
    def test_undisturbed_cycles_keep_the_cadence_exactly(self, night: Night) -> None:
        world = night.world
        starts = world.period_starts()
        checked = 0
        for earlier, later in itertools.pairwise(starts):
            if disturbed(world, earlier) or disturbed(world, later):
                continue
            assert later - earlier == pytest.approx(180.0, abs=0.05), (earlier, later)
            checked += 1
        assert checked > 120

    def test_each_cycle_has_two_full_windows_outside_the_disturbances(self, night: Night) -> None:
        world = night.world
        calm = [
            w
            for w in world.windows()
            if not disturbed(world, world.seconds(w.t_utc_ns), margin=200)
            and world.seconds(w.t_utc_ns) < NIGHT_LENGTH_S - 200
        ]
        assert len(calm) > 300
        for window in calm:
            assert window.n_frames == 30
            assert "partial" not in window.flags
            assert "degraded" not in window.flags

    def test_every_survey_step_is_a_short_exposure_and_then_a_long_one(self, night: Night) -> None:
        world = night.world
        exposures = [
            c.config.exposure_us
            for c in world.configures(mode="bin2", video=False)
            if c.config.roi is None
        ]
        assert exposures[0] == 1000
        # Pairs, except for a step that the bright sky or a fault cut short.
        assert exposures.count(1000) >= exposures.count(30_000_000)
        assert exposures.count(1000) - exposures.count(30_000_000) <= 2


class TestTheDisturbances:
    def test_the_sweep_ran_once_at_a_boundary_and_its_result_is_pinned(self, night: Night) -> None:
        world = night.world
        (result,) = world.results
        assert (result.kind, result.status, result.pinned) == ("sweep", "ok", True)
        assert [c["status"] for c in result.data["cells"]] == ["ok", "ok"]
        (event,) = world.events("scheduler.sweep_result")
        assert (event.detail or {})["pinned"] is True
        # Its windows are not in the seeing series.
        assert {w.exposure_us for w in world.windows()} == {2_000_000}

    def test_the_pause_made_no_call_to_the_camera(self, night: Night) -> None:
        world = night.world
        paused_calls = [
            call
            for call in world.camera.configure_log
            if world.t(PAUSE_AT + 1) < call.t_utc_ns < world.t(RESUME_AT - 1)
        ]
        assert paused_calls == []
        assert not [
            w for w in world.windows() if PAUSE_AT + 5 < world.seconds(w.t_utc_ns) < RESUME_AT - 5
        ]

    def test_the_missing_star_ended_a_period_early_and_requested_no_solve(
        self, night: Night
    ) -> None:
        world = night.world
        assert world.events("scheduler.solve_requested") == []
        # The tenth frame without the star, 20 seconds after it went, ends the period early.
        ends = [
            world.seconds(w.t_utc_ns) + w.duration_s
            for w in world.windows()
            if HIDE_FROM < world.seconds(w.t_utc_ns) + w.duration_s < HIDE_TO
        ]
        assert ends == [pytest.approx(HIDE_FROM + 20.0, abs=2.5)]
        # The survey step of the cycle follows at once, and the cycle keeps its slot.
        shorts = [
            world.seconds(c.t_utc_ns)
            for c in world.configures(mode="bin2", video=False)
            if c.config.roi is None and c.config.exposure_us == 1000
        ]
        (early,) = [t for t in shorts if HIDE_FROM < t < HIDE_TO]
        assert early == pytest.approx(ends[0], abs=2.5)
        periods = world.period_starts()
        before = max(t for t in periods if t < HIDE_FROM)
        after = min(t for t in periods if t > HIDE_FROM)
        assert after - before == pytest.approx(180.0, abs=0.05)

    def test_the_cloud_shortened_the_windows_and_the_cadence_and_set_the_flag(
        self, night: Night
    ) -> None:
        world = night.world
        on, off = (world.seconds(e.t_utc_ns) for e in world.events("scheduler.cloud"))
        assert CLOUD_FROM < on < CLOUD_FROM + 200
        assert CLOUD_TO < off < CLOUD_TO + 200
        windows = [w for w in world.windows() if on + 5 < world.seconds(w.t_utc_ns) < off - 70]
        assert len(windows) > 10
        assert all("cloud" in w.flags for w in windows)
        starts = [t for t in world.period_starts() if on + 1 < t < off]
        assert [round(b - a) for a, b in itertools.pairwise(starts)] == [100] * (len(starts) - 1)
        # A cloud of 0.8 in a dark sky leaves Polaris at an SNR of 40, so measure goes on.
        assert not [t for t in world.hidden_times() if CLOUD_FROM < t < CLOUD_TO]
        calm = [w for w in world.windows() if world.seconds(w.t_utc_ns) < CLOUD_FROM - 200]
        assert all("cloud" not in w.flags for w in calm)

    def test_every_end_of_measure_writes_hidden_and_every_return_writes_visible(
        self, night: Night
    ) -> None:
        world = night.world
        hidden = world.events("polaris.hidden")
        found = [((e.detail or {})["reason"], (e.detail or {}).get("state")) for e in hidden]
        assert found == [
            ("state_change", "commission"),  # the sweep
            ("state_change", "paused"),
            ("star_missing", None),
            ("state_change", "align"),
            ("fault", None),
            ("shutdown", None),  # the close at the end of the night
        ]
        times = world.hidden_times()
        assert SWEEP_AT <= times[0] < SWEEP_AT + 200  # a cycle boundary, which may be the moment
        assert PAUSE_AT <= times[1] <= PAUSE_AT + 31.0  # after the exposure in progress
        assert times[2] == pytest.approx(HIDE_FROM + 20.0, abs=2.5)
        assert times[3] == pytest.approx(ALIGN_FROM, abs=1.0)
        assert FAULT_FROM < times[4] < FAULT_TO
        # Polaris showed at dusk, and after each of the first five ends of measure.
        visible = world.visible_times()
        assert len(visible) == 6
        for end, back in zip(times[:5], visible[1:], strict=True):
            assert end < back < end + 1000
        for event in hidden:  # the Sun's elevation at the time, to 0.01 degree
            sun = (event.detail or {})["sun_elevation_deg"]
            assert sun == pytest.approx(world.sun_elevation(event.t_utc_ns), abs=0.05)

    def test_polaris_shows_at_dusk_where_its_snr_reaches_the_threshold(self, night: Night) -> None:
        """The scenario's Polaris reaches an SNR of 10 at -3.49 degrees, and two bursts in a row
        confirm it within a period and a half of search."""
        world = night.world
        first = world.events("polaris.visible")[0]
        detail = first.detail or {}
        assert -4.1 < detail["sun_elevation_deg"] < -3.45
        assert detail["snr"] >= TEST_CONFIG.search.detect_snr
        assert detail["probe"] is False
        assert world.events("polaris.search_limit_low") == []
        # Search bursts ran from the move to auto until then, and no fast stream.
        auto_at = world.state_changes()[0][0]
        bursts = [t for t in world.burst_starts() if t < world.seconds(first.t_utc_ns)]
        assert bursts[0] == pytest.approx(auto_at, abs=1.0)
        assert not [t for t in world.fast_starts() if t < world.seconds(first.t_utc_ns) - 0.1]

    def test_the_alignment_streamed_frames_and_ended_in_safe(self, night: Night) -> None:
        world = night.world
        assert len(world.align_frames) == pytest.approx(60, abs=2)  # 600 s of 10 s exposures
        assert {f.exposure_us for f in world.align_frames} == {10_000_000}

    def test_the_short_outage_ran_the_first_steps_of_the_ladder_and_cleared(
        self, night: Night
    ) -> None:
        world = night.world
        assert [(e.detail or {})["step"] for e in world.events("scheduler.recovery_step")] == [
            "restart_capture",
            "restart_capture",
        ]
        assert [level for _, level in world.camera.calls_named("recover")] == [
            RecoveryLevel.RESTART_CAPTURE
        ] * 2
        assert len(world.events("scheduler.recovered")) == 1
        assert world.events("scheduler.degraded") == []
        assert world.escalations == []

    def test_the_bump_ended_one_window_early_and_recentered_the_roi(self, night: Night) -> None:
        world = night.world
        (event,) = world.events("scheduler.roi_recentered")
        assert JOLT_AT < world.seconds(event.t_utc_ns) < JOLT_AT + 5
        partial = [
            w
            for w in world.windows()
            if "partial" in w.flags and abs(world.seconds(w.t_utc_ns) + w.duration_s - JOLT_AT) < 10
        ]
        assert len(partial) == 1

    def test_the_partial_windows_all_have_a_cause(self, night: Night) -> None:
        """No window is cut short without a reason: every `partial` one is next to an event."""
        world = night.world
        # The first period that measures at dusk starts in the middle of a search period, and its
        # last window ends with the period.
        dusk = world.visible_times()[0]
        causes = [dusk, PAUSE_AT, HIDE_FROM, ALIGN_FROM, FAULT_FROM, JOLT_AT, SWEEP_AT]
        for window in world.windows():
            if "partial" in window.flags:
                end = world.seconds(window.t_utc_ns) + window.duration_s
                assert any(-5 <= end - cause <= 400 for cause in causes), end

    def test_the_commands_wrote_events_and_none_was_refused(self, night: Night) -> None:
        events = night.world.events("scheduler.command")
        assert [(e.detail or {})["command"] for e in events] == [
            "QueueSweep",
            "Pause",
            "Resume",
            "StartAlignment",
            "StopAlignment",
        ]
        assert all((e.detail or {})["accepted"] for e in events)


class TestTheRecords:
    def test_no_frame_is_lost_between_the_camera_and_the_store(self, night: Night) -> None:
        world = night.world
        in_windows = sum(w.n_frames for w in world.windows())
        in_sweep = sum(int(cell["n_frames"]) for r in world.results for cell in r.data["cells"])
        assert in_windows + in_sweep == world.fast.frames_pushed
        rows = sum(len(batch) for _, batch in world.writer.metrics)
        assert rows == world.fast.frames_pushed

    def test_windows_never_overlap_and_arrive_in_order(self, night: Night) -> None:
        windows = night.world.windows()
        ends = [w.t_utc_ns + round(w.duration_s * NS_PER_S) for w in windows]
        starts = [w.t_utc_ns for w in windows]
        assert starts == sorted(starts)
        for end, next_start in zip(ends, starts[1:], strict=False):
            assert end <= next_start + NS_PER_S  # to within a frame

    def test_every_stream_has_one_id_and_one_set_of_settings(self, night: Night) -> None:
        by_stream: dict[int, set[tuple[str, int, int]]] = {}
        for window in night.world.windows():
            by_stream.setdefault(window.stream_id, set()).add(
                (window.readout_mode, window.exposure_us, window.gain)
            )
        assert all(len(settings) == 1 for settings in by_stream.values())

    def test_every_event_kind_is_declared(self, night: Night) -> None:
        kinds = {e.kind for e in night.world.events()}
        assert kinds <= set(EVENT_KINDS)
        assert len(kinds) >= 10  # the night touched the main ones

    def test_event_keys_are_unique(self, night: Night) -> None:
        keys = [e.record_key for e in night.world.events()]
        assert len(keys) == len(set(keys))

    def test_the_records_validate_and_round_trip(self, night: Night) -> None:
        for record in night.world.writer.records:
            assert type(record).from_row(record.to_row(), strict=True) == record

    def test_the_numbers_of_the_status_match_the_records(self, night: Night) -> None:
        world = night.world
        counters = world.scheduler.status().counters
        assert counters.windows == len(world.windows())
        assert counters.survey_results == len(world.records("survey_frame"))
        assert counters.tasks_run == 1
        assert counters.faults == 2
        assert counters.roi_recenters == 1
        assert counters.solves_requested == 0  # the night starts with a solution
        assert counters.dropped == 0
