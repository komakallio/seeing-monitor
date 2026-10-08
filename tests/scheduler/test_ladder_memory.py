"""The memory of the ladder: a reboot that an earlier run asked for still counts after a boot.

Without a camera, the ladder climbs to the reboot step about ten minutes after each start. A timer
in memory forgot the request at the boot, and the Pi rebooted again and again (October 8, 2026).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.scheduler.config import FaultConfig, LadderConfig
from seeingmon.scheduler.faults import FaultCause, FaultTracker
from seeingmon.scheduler.ladder_memory import FileLadderMemory
from seeingmon.scheduler.levels import EscalationLevel
from tests.scheduler.scenario import TEST_CONFIG, World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")
LOST_AT = 1000.0
DESTRUCTIVE = (EscalationLevel.REBOOT, EscalationLevel.POWER_CYCLE)


class Memory:
    """A `LadderMemory` in a list, for a scenario."""

    def __init__(self, last: int | None = None) -> None:
        self.last = last
        self.writes: list[int] = []

    def read(self) -> int | None:
        return self.last

    def write(self, utc_ns: int) -> None:
        self.last = utc_ns
        self.writes.append(utc_ns)


def destructive(world: World) -> list[int]:
    return [t for t, level in world.escalations if level in DESTRUCTIVE]


class TestTheFile:
    def test_a_time_comes_back_as_it_went_in(self, tmp_path: Path) -> None:
        memory = FileLadderMemory(tmp_path / "state" / "ladder.json")
        assert memory.read() is None
        memory.write(1_790_000_000 * NS_PER_S)
        assert memory.read() == 1_790_000_000 * NS_PER_S
        memory.write(1_791_000_000 * NS_PER_S)
        assert memory.read() == 1_791_000_000 * NS_PER_S
        assert [p.name for p in memory.path.parent.iterdir()] == ["ladder.json"]

    @pytest.mark.parametrize(
        "text", ["", "not json", "[]", '{"other": 1}', '{"last_destructive_utc_ns": "x"}']
    )
    def test_a_file_that_is_not_ours_says_nothing(self, tmp_path: Path, text: str) -> None:
        path = tmp_path / "ladder.json"
        path.write_text(text, encoding="utf-8")
        assert FileLadderMemory(path).read() is None


class TestTheTracker:
    @staticmethod
    def tracker(age_s: float | None) -> FaultTracker:
        ladder = LadderConfig(attempts_per_level=1, destructive_interval_s=1000.0)
        return FaultTracker(FaultConfig(), ladder, lambda: age_s)

    @staticmethod
    def top_step(tracker: FaultTracker) -> object:
        plan = None
        for k in range(12):
            plan = tracker.failure(k * NS_PER_S, can_escalate=True, cause=FaultCause.DISCONNECTED)
        assert plan is not None
        return plan.step

    def test_a_recent_request_of_an_earlier_run_holds_the_ladder_below_the_reboot(self) -> None:
        assert self.top_step(self.tracker(10.0)) is EscalationLevel.RESTART_ACQUIRE

    def test_an_old_request_lets_the_ladder_reach_the_reboot(self) -> None:
        assert self.top_step(self.tracker(5000.0)) in DESTRUCTIVE

    def test_no_record_lets_the_ladder_reach_the_reboot(self) -> None:
        assert self.top_step(self.tracker(None)) in DESTRUCTIVE


class TestTheScheduler:
    @staticmethod
    def lost(memory: Memory | None) -> World:
        world = World(start_utc_ns=NIGHT, ladder_memory=memory)
        world.camera_gone(LOST_AT, None)
        world.run_until(LOST_AT + 3 * 3600)
        return world

    def test_a_record_from_before_the_boot_stops_the_reboots(self) -> None:
        memory = Memory(last=NIGHT - 120 * NS_PER_S)
        world = self.lost(memory)
        assert destructive(world) == []
        assert memory.writes == []
        world.close()

    def test_the_reboot_comes_when_the_record_is_older_than_the_limit(self) -> None:
        interval = TEST_CONFIG.ladder.destructive_interval_s
        memory = Memory(last=NIGHT - round((interval + 600) * NS_PER_S))
        world = self.lost(memory)
        sent = destructive(world)
        assert sent
        assert memory.writes[0] == sent[0]  # the record is written when the step is asked for
        world.close()

    def test_a_request_is_written_before_it_is_sent(self) -> None:
        memory = Memory()
        world = self.lost(memory)
        assert destructive(world)
        assert len(memory.writes) == len(destructive(world))
        world.close()

    def test_a_clock_that_is_not_synchronized_counts_the_last_request_as_just_made(self) -> None:
        memory = Memory(last=NIGHT - 100_000 * NS_PER_S)
        world = World(start_utc_ns=NIGHT, ladder_memory=memory)
        scheduler = world.scheduler
        assert scheduler._remembered_destructive_age_s() == pytest.approx(100_000.0)
        scheduler._clock_ok = False
        assert scheduler._remembered_destructive_age_s() == 0.0
        scheduler._clock_ok = True
        memory.last = NIGHT + 50 * NS_PER_S  # a record from the future: the clock is behind it
        assert scheduler._remembered_destructive_age_s() == 0.0
        world.close()
