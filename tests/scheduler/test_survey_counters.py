"""The counter of unsolved survey results: only a frame that a solver could try can fail.

Each survey step takes two frames. The 1 ms frame shows Polaris alone by design, so no solver can
use it. The pipeline gives it an unsolved pointing record, and the scheduler drops that record. The
long frame always gets one, unsolved when the solve fails.
"""

from __future__ import annotations

from seeingmon.clock import iso_to_utc_ns
from seeingmon.records import PointingRecord
from tests.scheduler.scenario import World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")


def pointing_records(world: World) -> list[PointingRecord]:
    return [r for r in world.records("pointing") if isinstance(r, PointingRecord)]


def test_the_short_frame_of_a_step_is_a_result_but_never_an_unsolved_one() -> None:
    world = World(start_utc_ns=NIGHT, solved_at_start=False)
    world.no_solution(0, 4000)
    world.run_until(700)
    counters = world.scheduler.status().counters
    pointing = pointing_records(world)
    assert len(pointing) >= 4  # several steps ran, and each one left its 30 s frame unsolved
    assert all("unsolved" in record.flags for record in pointing)
    assert counters.survey_unsolved == len(pointing)  # the long frames, and not the 1 ms frames
    assert counters.survey_results >= 2 * counters.survey_unsolved  # each step has two results
    world.close()


def test_a_sky_that_solves_leaves_the_counter_at_zero() -> None:
    world = World(start_utc_ns=NIGHT)
    world.run_until(1500)
    counters = world.scheduler.status().counters
    pointing = pointing_records(world)
    assert counters.survey_results >= 6  # the 1 ms and the 30 s frame of several steps
    assert pointing
    assert not any("unsolved" in record.flags for record in pointing)
    assert counters.survey_unsolved == 0
    world.close()


def test_a_solve_that_fails_among_good_ones_counts_once_for_its_step() -> None:
    world = World(start_utc_ns=NIGHT)
    world.no_solution(500, 800)  # the tracker has a solution, and the field does not solve
    world.run_until(1500)
    counters = world.scheduler.status().counters
    unsolved = [r for r in pointing_records(world) if "unsolved" in r.flags]
    assert unsolved
    assert len(unsolved) < len(pointing_records(world))  # the steps outside the gap solved
    assert counters.survey_unsolved == len(unsolved)
    assert counters.survey_results > 2 * counters.survey_unsolved
    world.close()
