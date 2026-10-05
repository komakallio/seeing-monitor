"""The warning `pointing.moved`: once when a solve finds the camera off its reference.

Every solve after a move keeps the `moved` flag until the owner saves a new reference, so the
event must come on the change from not moved to moved, and not with every survey frame. In these
scenarios a survey step starts every 180 seconds, 120 seconds into each cycle, and the time of its
long frame is the middle of the exposure: 495, 675, 855 seconds, and so on.
"""

from __future__ import annotations

import pytest

from seeingmon.clock import iso_to_utc_ns
from seeingmon.records import PointingRecord
from tests.scheduler.scenario import World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")


def moved_times(world: World) -> list[float]:
    return [world.seconds(e.t_utc_ns) for e in world.events("pointing.moved")]


def flagged(world: World, flag: str) -> list[PointingRecord]:
    return [
        r for r in world.records("pointing") if isinstance(r, PointingRecord) and flag in r.flags
    ]


def test_a_move_writes_one_warning_however_many_solves_keep_the_flag() -> None:
    world = World(start_utc_ns=NIGHT)
    world.pointing_flags(400, 1000, "moved")
    world.run_until(1100)
    assert len(flagged(world, "moved")) == 3  # the solves of 495, 675, and 855
    (event,) = world.events("pointing.moved")
    assert event.level == "warning"
    # The result of the long frame from 480 to 510 arrives at the next poll.
    assert 510 <= world.seconds(event.t_utc_ns) < 520
    detail = event.detail or {}
    assert detail["solver"] == "fake"
    assert detail["n_matched"] == 20
    assert world.seconds(detail["frame_t_utc_ns"]) == pytest.approx(495.0, abs=0.1)
    world.close()


def test_a_solve_that_cannot_tell_keeps_the_state_and_a_clean_solve_ends_it() -> None:
    world = World(start_utc_ns=NIGHT)
    world.pointing_flags(400, 700, "moved")  # the solves of 495 and 675
    world.pointing_flags(700, 1000, "few_stars")  # 855: too few stars to tell
    world.no_solution(1000, 1200)  # 1035: no solve at all
    world.pointing_flags(1200, 1400, "moved")  # 1215 and 1395: still moved, so no new event
    world.pointing_flags(1500, 1700, "moved", "few_stars")  # 1575: a thin solve cannot tell
    # 1755 solves clean, so the mount is back (or the reference was saved again). The move of
    # 2115 is a new one.
    world.pointing_flags(2000, 2300, "moved")
    world.run_until(2400)
    times = moved_times(world)
    assert len(times) == 2
    assert 510 <= times[0] < 520
    assert 2130 <= times[1] < 2140
    world.close()


def test_a_frame_without_a_valid_time_does_not_count_as_a_move() -> None:
    """Without the time, the Earth-fixed attitude turns with the error of the clock."""
    world = World(start_utc_ns=NIGHT)
    world.pointing_flags(400, 1000, "moved", "time_invalid")
    world.pointing_flags(1000, 1300, "moved")
    world.run_until(1400)
    (time,) = moved_times(world)
    assert 1050 <= time < 1060  # the frame of 1035, the first with the flag and a valid time
    world.close()


def test_without_a_move_no_warning_comes() -> None:
    world = World(start_utc_ns=NIGHT)
    world.run_until(1100)
    assert flagged(world, "moved") == []
    assert world.events("pointing.moved") == []
    world.close()
