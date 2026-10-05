"""The clock's synchronization, and the list of events that the scheduler writes.

A Raspberry Pi 4 has no real-time clock, so after a boot the UTC time can be days off until
chrony synchronizes it. While the clock is not synchronized, the Sun's elevation and the zenith
angle mean nothing. The scheduler then relies on the measured sky, sets no `twilight` flag, and
marks windows and survey results `time_invalid`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from seeingmon.clock import ClockStatus, VirtualClock, iso_to_utc_ns
from seeingmon.records.system import EVENT_KIND_PATTERN
from seeingmon.scheduler.events import EVENT_KINDS
from tests.scheduler.scenario import World

TWILIGHT = iso_to_utc_ns("2026-01-01T17:00:00Z")  # the Sun is about 10 degrees below the horizon


def set_synchronized(world: World, synchronized: bool | None) -> None:
    assert isinstance(world.clock, VirtualClock)
    world.clock.set_status(
        ClockStatus(synchronized=synchronized, error_bound_ns=None, source="test")
    )


@pytest.fixture(scope="module")
def drift() -> World:
    """In twilight, the clock loses synchronization from 600 to 1800 seconds."""
    world = World(start_utc_ns=TWILIGHT)
    world.at(600, lambda w: set_synchronized(w, False))
    world.at(1800, lambda w: set_synchronized(w, True))
    world.run_until(3000)
    world.close()
    return world


def window_times(world: World) -> list[tuple[float, float, tuple[str, ...], float | None]]:
    return [
        (
            world.seconds(w.t_utc_ns),
            world.seconds(w.t_utc_ns) + w.duration_s,
            tuple(w.flags),
            w.zenith_angle_deg,
        )
        for w in world.windows()
    ]


class TestAnUnsynchronizedClock:
    def test_the_scheduler_notices_within_seconds_and_writes_an_event_each_way(
        self, drift: World
    ) -> None:
        lost, regained = (
            drift.events("scheduler.clock_unsynchronized"),
            drift.events("scheduler.clock_synchronized"),
        )
        assert len(lost) == len(regained) == 1
        assert drift.seconds(lost[0].t_utc_ns) == pytest.approx(600.0, abs=8.0)
        assert drift.seconds(regained[0].t_utc_ns) == pytest.approx(1800.0, abs=8.0)
        assert lost[0].level == "warning"
        assert regained[0].level == "info"
        assert (lost[0].detail or {})["synchronized"] is False

    def test_windows_in_the_gap_carry_time_invalid_and_not_twilight(self, drift: World) -> None:
        inside = [t for t in window_times(drift) if t[0] >= 620 and t[1] <= 1790]
        assert len(inside) >= 6
        for _, _, flags, _ in inside:
            assert "time_invalid" in flags
            assert "twilight" not in flags  # the Sun is unknown, so the flag is off

    def test_windows_outside_the_gap_are_flagged_by_the_sun_as_usual(self, drift: World) -> None:
        before = [t for t in window_times(drift) if t[1] <= 590]
        after = [t for t in window_times(drift) if t[0] >= 1820]
        assert before
        assert after
        for _, _, flags, _ in (*before, *after):
            assert "time_invalid" not in flags
            assert "twilight" in flags

    def test_the_zenith_angle_is_unknown_while_the_clock_is_not_synchronized(
        self, drift: World
    ) -> None:
        inside = [t for t in window_times(drift) if t[0] >= 620 and t[1] <= 1790]
        assert all(zenith is None for _, _, _, zenith in inside)
        outside = [t for t in window_times(drift) if t[1] <= 590 or t[0] >= 1820]
        assert all(zenith is not None for _, _, _, zenith in outside)

    def test_survey_results_carry_the_flag_too(self, drift: World) -> None:
        for record in drift.records("sky_quality"):
            t = drift.seconds(record.t_utc_ns)
            flags = record.flags  # type: ignore[attr-defined]
            if 620 <= t <= 1790:
                assert "time_invalid" in flags
                assert "twilight" not in flags
            elif t <= 590 or t >= 1820:
                assert "time_invalid" not in flags

    def test_the_cycle_does_not_stop_and_nothing_else_changes(self, drift: World) -> None:
        assert drift.states_visited() == ["safe", "auto"]
        assert drift.scheduler.status().counters.cadence_overruns == 0

    def test_the_status_hides_the_sun_during_the_gap(self) -> None:
        world = World(start_utc_ns=TWILIGHT)
        world.run_until(100)
        assert world.scheduler.status().sun_elevation_deg is not None
        set_synchronized(world, False)
        world.run_for(30)
        status = world.scheduler.status()
        assert (status.sun_elevation_deg, status.twilight) == (None, False)
        world.close()


class TestTheClockTheSchedulerTrusts:
    def test_a_clock_that_cannot_tell_is_trusted(self) -> None:
        """A `SystemClock` without a probe reports `None`, and the scheduler takes it as it is."""
        world = World(start_utc_ns=TWILIGHT)
        set_synchronized(world, None)
        world.run_until(300)
        assert world.events("scheduler.clock_unsynchronized") == []
        assert all("time_invalid" not in w.flags for w in world.windows())
        world.close()

    def test_without_synchronization_the_daylight_gate_relies_on_the_measured_sky(self) -> None:
        """In daylight the fast stream could not measure even at 32 us, so `safe` holds."""
        world = World()  # 14:30 UTC, with the Sun up
        set_synchronized(world, False)
        world.run_until(1200)
        assert world.scheduler.state.value == "safe"
        world.close()

    def test_a_dark_sky_and_a_clock_that_is_off_still_start_the_cycle(self) -> None:
        """Without the Sun, a dark sky is enough, and the windows are flagged `time_invalid`."""
        world = World(start_utc_ns=iso_to_utc_ns("2026-01-01T22:00:00Z"))
        set_synchronized(world, False)
        world.run_until(400)
        assert world.states_visited() == ["safe", "auto"]
        assert world.windows()
        assert all("time_invalid" in w.flags for w in world.windows())
        world.close()


class TestTheEventList:
    @staticmethod
    def source_kinds() -> set[str]:
        src = Path(__file__).resolve().parents[2] / "src" / "seeingmon"
        package = src / "scheduler"
        sources = [s for s in package.glob("*.py") if s.name != "events.py"]
        # `core` writes the kinds of the darkness of the sky from the survey results.
        sources.append(src / "services" / "core" / "darkness.py")
        kinds: set[str] = set()
        for source in sources:
            # The scheduler's own kinds, and the kinds of the sky, the camera, and Polaris.
            text = source.read_text(encoding="utf-8")
            kinds |= set(re.findall(r'"((?:scheduler|pointing|sky|polaris)\.[a-z_]+)"', text))
        return kinds

    def test_every_kind_in_the_source_is_in_the_list(self) -> None:
        missing = self.source_kinds() - set(EVENT_KINDS)
        assert missing == set()

    def test_every_listed_kind_is_written_by_the_source_or_names_a_task_result(self) -> None:
        written = self.source_kinds()
        results = {
            "scheduler.sweep_result",
            "scheduler.burst_result",
            "scheduler.replay_result",
            "scheduler.dark_result",
            "scheduler.flat_result",
        }  # built from the kind of the task
        # written through `emit_event` by the dark and flat handlers
        handlers = {"scheduler.dark_phase", "scheduler.flat_phase"}
        assert set(EVENT_KINDS) - written == results | handlers

    def test_the_list_names_the_events_of_the_visibility_of_polaris(self) -> None:
        polaris = {kind for kind in EVENT_KINDS if kind.startswith("polaris.")}
        assert polaris == {"polaris.visible", "polaris.hidden", "polaris.search_limit_low"}
        assert polaris <= self.source_kinds()

    def test_the_kinds_and_their_descriptions_are_well_formed(self) -> None:
        for kind, description in EVENT_KINDS.items():
            assert re.fullmatch(EVENT_KIND_PATTERN, kind), kind
            assert description.endswith(".")
            assert len(description) < 100
