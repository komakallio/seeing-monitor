"""The dark library of `FakeCoreClient`: the status of the library and the scripted dark task.

The task follows a `VirtualClock`, so each test moves time with `advance` and reads the library.
"""

from __future__ import annotations

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.scheduler.commands import (
    Pause,
    QueueDark,
    RejectReason,
    Resume,
    StartAlignment,
    StopAlignment,
)
from seeingmon.services.web.contract import DarkModelView
from seeingmon.services.web.core_client import CoreUnavailableError, FakeCoreClient
from seeingmon.services.web.fake_dark import DarkScript
from tests.services.web.helpers import dark_set

SCRIPT = DarkScript(queued_s=5.0, bias_s=4.0, cover_s=3.0, dark_s=6.0, build_s=1.0)


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock(1_790_000_000_000_000_000)


@pytest.fixture
def core(clock: VirtualClock) -> FakeCoreClient:
    return FakeCoreClient(clock=clock, dark_script=SCRIPT, state="auto")


# --- The library -----------------------------------------------------------------------------


def test_an_empty_library_is_due_and_has_an_idle_task(core: FakeCoreClient) -> None:
    library = core.dark_library()
    assert library.sets == []
    assert library.model is None
    assert library.status.due is True
    assert "no dark set" in library.status.reason
    assert library.task.state == "idle"
    assert library.task.phase is None
    assert (library.mode, library.gain, library.exposure_s) == ("bin2", 120, 30.0)


def test_the_status_is_due_when_no_set_lies_near_the_sensor_temperature(
    core: FakeCoreClient,
) -> None:
    core.dark.sensor_temperature_c = 12.3
    core.dark.sets = [dark_set("near-4", 4.1, 10.0), dark_set("near-20", 20.0, 30.0)]
    status = core.dark_library().status
    assert status.due is True
    assert status.gap_c == 7.7
    assert status.nearest_name == "near-20"
    assert "7.7 C away" in status.reason
    assert status.newest_age_days == 10.0


def test_the_status_is_up_to_date_when_a_recent_set_lies_near(core: FakeCoreClient) -> None:
    core.dark.sensor_temperature_c = 12.3
    core.dark.sets = [dark_set("near-12", 11.0, 10.0)]
    status = core.dark_library().status
    assert status.due is False
    assert status.gap_c == 1.3
    assert "10 days old" in status.reason


def test_the_status_is_due_when_the_newest_set_is_too_old(core: FakeCoreClient) -> None:
    core.dark.sensor_temperature_c = 12.3
    core.dark.sets = [dark_set("old", 12.0, 200.0)]
    status = core.dark_library().status
    assert status.due is True
    assert "200 days old" in status.reason


def test_a_library_without_a_temperature_is_not_called_due(core: FakeCoreClient) -> None:
    core.dark.sensor_temperature_c = None
    core.dark.sets = [dark_set("x", 12.0, 3.0)]
    library = core.dark_library()
    assert library.sensor_temperature_c is None
    assert library.status.due is False
    assert "no temperature" in library.status.reason


def test_the_library_lists_the_newest_set_first_and_carries_the_model(
    core: FakeCoreClient,
) -> None:
    core.dark.sets = [dark_set("newer", 12.0, 1.0), dark_set("older", 8.0, 9.0)]
    core.dark.model = DarkModelView(
        reference_c=20.0, rate_ref_e_per_s=0.12, doubling_c=6.0, doubling_fitted=True, n_sets=2
    )
    library = core.dark_library()
    assert [item.name for item in library.sets] == ["newer", "older"]
    assert library.model is not None
    assert library.model.doubling_fitted is True


def test_a_failing_core_fails_the_dark_call_too(core: FakeCoreClient) -> None:
    core.fail_with = CoreUnavailableError("gone")
    with pytest.raises(CoreUnavailableError):
        core.dark_library()


# --- The task --------------------------------------------------------------------------------


def test_a_dark_command_queues_a_task_that_waits_for_its_turn(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    result = core.submit(QueueDark(frames=6, bias_frames=4, label="night one"))
    assert result.accepted
    assert result.task_id == 1
    assert core.status().scheduler.queued_tasks == 1
    task = core.dark_library().task
    assert (task.state, task.task_id, task.phase) == ("queued", 1, None)
    assert (task.frames, task.bias_frames, task.exposure_s) == (6, 4, 30.0)
    assert (task.wait_for_cover, task.pause_after) == (True, True)
    clock.advance(4.9)
    assert core.dark_library().task.state == "queued"


def test_the_task_runs_through_its_phases_and_adds_a_set(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.dark.sensor_temperature_c = 12.3
    core.dark.sets = [dark_set("old", 4.0, 10.0)]
    core.submit(QueueDark(frames=6, bias_frames=4))
    clock.advance(5.0)  # the task starts
    assert core.status().scheduler.state == "commission"
    task = core.dark_library().task
    assert (task.state, task.phase, task.step, task.steps) == ("running", "bias", 1, 4)
    assert task.message == "Bias frame 1 of 4."
    clock.advance(3.0)  # 3 s of 4: the last bias frame
    assert core.dark_library().task.step == 4
    clock.advance(1.0)  # the wait for the cover starts, and the camera is not covered
    task = core.dark_library().task
    assert (task.phase, task.covered) == ("cover", False)
    assert task.level_dn is not None
    assert task.level_dn > 1000
    assert "above the expected level" in task.reason
    clock.advance(2.5)  # the last third: the camera is covered
    task = core.dark_library().task
    assert (task.phase, task.covered, task.reason) == ("cover", True, "")
    clock.advance(0.5)  # the dark frames start
    task = core.dark_library().task
    assert (task.phase, task.step, task.steps, task.covered) == ("dark", 1, 6, True)
    clock.advance(5.5)
    assert core.dark_library().task.step == 6
    clock.advance(0.5)  # the build starts
    assert core.dark_library().task.phase == "build"
    clock.advance(1.0)
    library = core.dark_library()
    task = library.task
    assert (task.state, task.phase) == ("ok", None)
    assert task.set_name is not None
    assert task.set_name.startswith("dark-")
    assert task.set_name in task.summary
    assert task.finished_utc is not None
    assert library.sets[0].name == task.set_name  # the newest set is first
    assert library.sets[0].temperature_c == 12.3
    assert library.sets[0].n_frames == 6
    assert len(library.sets) == 2
    assert library.status.due is False  # the new set closes the gap


def test_the_scheduler_pauses_when_the_task_asks_for_it_and_resume_works(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(QueueDark())
    clock.advance(30)
    scheduler = core.status().scheduler
    assert scheduler.state == "paused"
    assert "covered" in scheduler.state_reason
    assert core.submit(Resume()).accepted
    assert core.status().scheduler.state == "safe"


def test_the_scheduler_goes_to_safe_when_the_task_does_not_ask_for_a_pause(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(QueueDark(pause_after=False))
    clock.advance(30)
    assert core.dark_library().task.state == "ok"
    assert core.status().scheduler.state == "safe"


def test_a_task_that_does_not_wait_for_the_cover_fails_at_the_first_dark_frame(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(QueueDark(wait_for_cover=False))
    clock.advance(5.0 + 3.9)
    assert core.dark_library().task.phase == "bias"
    clock.advance(0.2)
    library = core.dark_library()
    assert library.task.state == "failed"
    assert library.task.set_name is None
    assert "not covered" in library.task.summary
    assert library.sets == []  # a failed task adds nothing
    assert core.status().scheduler.state == "paused"  # the camera may still be uncovered


def test_a_pause_aborts_a_running_task_and_adds_no_set(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(QueueDark())
    clock.advance(10)
    assert core.dark_library().task.state == "running"
    assert core.submit(Pause()).accepted
    library = core.dark_library()
    assert library.task.state == "aborted"
    assert library.sets == []
    assert "paused" in library.task.summary
    clock.advance(60)
    assert core.dark_library().task.state == "aborted"  # nothing moves it on
    assert core.status().scheduler.state == "paused"


def test_a_pause_aborts_a_task_that_still_waits(core: FakeCoreClient, clock: VirtualClock) -> None:
    core.submit(QueueDark())
    assert core.submit(Pause()).accepted
    assert core.dark_library().task.state == "aborted"
    assert core.status().scheduler.queued_tasks == 0


def test_a_second_task_is_refused_as_busy_until_the_first_one_ends(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    assert core.submit(QueueDark()).accepted
    busy = core.submit(QueueDark())
    assert not busy.accepted
    assert busy.reason is RejectReason.BUSY
    clock.advance(60)
    assert core.dark_library().task.state == "ok"
    second = core.submit(QueueDark())  # the first one has ended, so a new one may start
    assert second.accepted
    assert second.task_id == 2


@pytest.mark.parametrize(
    "command",
    [
        QueueDark(frames=2),
        QueueDark(frames=61),
        QueueDark(bias_frames=2),
        QueueDark(bias_frames=61),
        QueueDark(exposure_s=0.0),
        QueueDark(exposure_s=-1.0),
        QueueDark(exposure_s=601.0),
        QueueDark(wait_for_cover_timeout_s=0.0),
        QueueDark(wait_for_cover_timeout_s=-5.0),
        QueueDark(wait_for_cover_timeout_s=7201.0),
        QueueDark(label="x" * 81),
    ],
)
def test_a_dark_command_out_of_range_is_invalid(core: FakeCoreClient, command: QueueDark) -> None:
    result = core.submit(command)
    assert not result.accepted
    assert result.reason is RejectReason.INVALID
    assert core.dark_library().task.state == "idle"


def test_a_paused_scheduler_holds_the_task_until_it_resumes(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(Pause())
    answer = core.submit(QueueDark())
    assert answer.accepted
    assert answer.message == "The scheduler is paused. The dark session starts after you resume it."
    clock.advance(600)
    held = core.dark_library().task
    assert held.state == "queued"
    assert held.message == "The scheduler is paused. The dark session starts after you resume it."
    core.submit(Resume())
    clock.advance(4.9)
    assert core.dark_library().task.state == "queued"  # the queue time counts from the resume
    clock.advance(0.2)
    assert core.dark_library().task.state == "running"


def test_the_task_uses_the_configured_values_where_the_command_leaves_a_field_out(
    core: FakeCoreClient,
) -> None:
    core.dark.exposure_s = 45.0
    core.submit(QueueDark())
    task = core.dark_library().task
    assert task.exposure_s == 45.0
    assert task.frames == 9
    assert task.bias_frames == 9


def test_the_model_counts_the_new_set(core: FakeCoreClient, clock: VirtualClock) -> None:
    core.dark.model = DarkModelView(
        reference_c=20.0, rate_ref_e_per_s=0.12, doubling_c=6.0, doubling_fitted=True, n_sets=0
    )
    core.dark.sensor_temperature_c = 14.0
    core.submit(QueueDark())
    clock.advance(60)
    library = core.dark_library()
    assert library.model is not None
    assert library.model.n_sets == 1
    expected = 0.12 * 2 ** ((14.0 - 20.0) / 6.0)
    assert library.sets[0].rate_e_per_s == pytest.approx(expected, rel=1e-3)


def test_a_wait_for_the_cover_that_runs_out_before_the_camera_is_covered_fails(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(
        QueueDark(wait_for_cover_timeout_s=1.5)
    )  # the camera is covered after 2 s of the wait
    clock.advance(SCRIPT.queued_s + SCRIPT.bias_s + 1.4)
    assert core.dark_library().task.state == "running"
    clock.advance(0.2)
    library = core.dark_library()
    assert library.task.state == "failed"
    assert "not covered within 1.5 seconds" in library.task.summary
    assert library.sets == []


def test_a_wait_for_the_cover_that_is_long_enough_does_not_fail(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(QueueDark(wait_for_cover_timeout_s=600.0))
    clock.advance(60)
    assert core.dark_library().task.state == "ok"


def test_a_running_alignment_holds_the_task_until_it_ends(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(StartAlignment())
    answer = core.submit(QueueDark())
    assert answer.accepted
    assert answer.message == "The alignment helper runs. The dark session starts after it ends."
    clock.advance(600)
    held = core.dark_library().task
    assert held.state == "queued"
    assert held.message == "The alignment helper runs. The dark session starts after it ends."
    core.submit(StopAlignment())  # the scheduler goes back to safe
    clock.advance(SCRIPT.queued_s + 0.1)
    assert core.dark_library().task.state == "running"
    assert core.dark_library().task.message.startswith("Bias frame")
