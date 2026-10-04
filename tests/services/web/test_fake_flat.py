"""The flat library of `FakeCoreClient`: the library, the scripted flat task, and the decisions.

The task follows a `VirtualClock`, so each test moves time with `advance` and reads the library.
The last tests hold the sentences of the fake against the ones of the real code.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.scheduler.commands import (
    CancelTask,
    Pause,
    QueueDark,
    QueueFlat,
    RejectReason,
    Resume,
    StartAlignment,
    StopAlignment,
)
from seeingmon.services.web import fake_flat as fake
from seeingmon.services.web.contract import FlatTiltView
from seeingmon.services.web.core_client import CoreUnavailableError, FakeCoreClient
from seeingmon.services.web.fake_dark import DarkScript
from seeingmon.services.web.fake_flat import FlatLook, FlatScript
from tests.services.web.helpers import dark_set

# Setup 1 s, the search 2 s, the frames 8 s, and the build 1 s: the task runs 12 s after 5 s.
SCRIPT = FlatScript(queued_s=5.0, setup_s=1.0, exposure_s=2.0, capture_s=8.0, build_s=1.0)
VERSION = re.compile(r"^flat-[0-9a-f]{8}$")


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock(1_790_000_000_000_000_000)


def make_core(clock: VirtualClock, script: FlatScript = SCRIPT) -> FakeCoreClient:
    core = FakeCoreClient(clock=clock, flat_script=script, state="auto")
    core.dark.sets = [dark_set("set-a", 12.0, 5.0)]  # the bias comes from the dark library
    return core


@pytest.fixture
def core(clock: VirtualClock) -> FakeCoreClient:
    return make_core(clock)


def run_session(
    core: FakeCoreClient, clock: VirtualClock, command: QueueFlat | None = None
) -> None:
    """Queue a session and let it end."""
    assert core.submit(command or QueueFlat(frames=16)).accepted
    clock.advance(30.0)
    assert core.flat_library().task.state in ("ok", "failed", "aborted")


# --- The library -----------------------------------------------------------------------------


def test_an_empty_library_has_an_idle_task_and_no_blocker(core: FakeCoreClient) -> None:
    library = core.flat_library()
    assert (library.mode, library.gain) == ("bin2", 120)
    assert (library.flats, library.session, library.blocker) == ([], None, None)
    assert (library.active_version, library.pending_version) == (None, None)
    assert (library.flat_file_pinned, library.library_overrides) == (False, False)
    assert library.task.state == "idle"


def test_without_a_dark_set_a_session_is_blocked(clock: VirtualClock) -> None:
    core = FakeCoreClient(clock=clock, flat_script=SCRIPT)
    assert core.flat_library().blocker == fake.DARK_FIRST
    answer = core.submit(QueueFlat())
    assert (answer.accepted, answer.reason) == (False, RejectReason.INVALID)
    assert answer.message == fake.DARK_FIRST
    core.dark.sets = [dark_set("set-a", 12.0, 5.0)]
    assert core.flat_library().blocker is None
    assert core.submit(QueueFlat()).accepted


def test_a_seeded_flat_carries_the_numbers_of_a_report(core: FakeCoreClient) -> None:
    seeded = core.flat.seed(age_days=12.0, active=True)
    assert VERSION.match(seeded.version)
    assert (seeded.state, seeded.active, seeded.pending) == ("approved", True, False)
    assert seeded.age_days == pytest.approx(12.0, abs=0.01)
    assert (seeded.mode, seeded.gain, seeded.width_px, seeded.height_px) == (
        "bin2",
        120,
        4144,
        2822,
    )
    assert seeded.corner_percent == -9.6
    assert seeded.vignetting[-1].corner is True
    assert seeded.vignetting[-1].change_percent == seeded.corner_percent
    changes = [point.change_percent or 0.0 for point in seeded.vignetting]
    assert changes == sorted(changes, reverse=True)  # the loss grows with the radius
    assert seeded.shadows == len(seeded.shadow_items) == 3
    assert seeded.shadow_min_depth_percent == 1.1
    assert (seeded.second_set, seeded.optics_tilt, seeded.agreement) == (False, None, None)
    assert seeded.has_image is True


def test_the_library_lists_the_newest_flat_first_and_names_the_pending_one(
    core: FakeCoreClient,
) -> None:
    core.flat.seed(age_days=40.0, active=True)
    pending = core.flat.seed(age_days=1.0, state="pending")
    core.flat.seed(age_days=12.0)
    library = core.flat_library()
    ages = [item.age_days for item in library.flats]
    assert ages == sorted(ages)
    assert library.flats[0].version == pending.version
    assert library.pending_version == pending.version
    assert library.active_version == library.flats[-1].version


def test_a_pinned_flat_file_shows_that_the_library_wins(core: FakeCoreClient) -> None:
    core.flat.flat_file_pinned = True
    core.flat.seed(age_days=3.0, active=True)
    library = core.flat_library()
    assert (library.flat_file_pinned, library.library_overrides) == (True, True)


def test_the_library_keeps_the_newest_ten_and_the_flat_in_use(core: FakeCoreClient) -> None:
    kept = core.flat.seed(age_days=300.0, active=True)
    for age in range(1, 13):
        core.flat.seed(age_days=float(age))
    versions = [item.version for item in core.flat_library().flats]
    assert len(versions) == 11  # the ten newest, and the active flat of 300 days
    assert kept.version in versions


def test_a_failing_core_fails_the_flat_calls_too(core: FakeCoreClient) -> None:
    core.fail_with = CoreUnavailableError("gone")
    for call in (
        core.flat_library,
        lambda: core.flat_activate("flat-1a2b3c4d"),
        lambda: core.flat_delete("flat-1a2b3c4d"),
        lambda: core.flat_image("flat-1a2b3c4d"),
    ):
        with pytest.raises(CoreUnavailableError):
            call()


# --- The commands ----------------------------------------------------------------------------


def test_a_flat_command_queues_a_task_that_waits_for_its_turn(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    result = core.submit(QueueFlat(frames=16, target_fraction=0.4, pause_after=False))
    assert result.accepted
    assert result.task_id == 1
    assert core.status().scheduler.queued_tasks == 1
    task = core.flat_library().task
    assert (task.state, task.task_id, task.phase) == ("queued", 1, None)
    assert (task.frames, task.target_fraction, task.set_number) == (16, 0.4, 1)
    assert task.pause_after is False
    clock.advance(4.9)
    assert core.flat_library().task.state == "queued"


@pytest.mark.parametrize(
    ("command", "text"),
    [
        (QueueFlat(frames=7), "frames must be between 8 and 64"),
        (QueueFlat(frames=65), "frames must be between 8 and 64"),
        (QueueFlat(target_fraction=0.29), "target_fraction must be between 0.3 and 0.7"),
        (QueueFlat(target_fraction=0.71), "target_fraction must be between 0.3 and 0.7"),
        (QueueFlat(set_number=3), "set_number must be 1 or 2"),
    ],
)
def test_a_flat_command_out_of_range_is_invalid(
    core: FakeCoreClient, command: QueueFlat, text: str
) -> None:
    result = core.submit(command)
    assert (result.accepted, result.reason, result.message) == (False, RejectReason.INVALID, text)
    assert core.flat_library().task.state == "idle"


def test_a_second_set_without_a_first_one_is_invalid(core: FakeCoreClient) -> None:
    result = core.submit(QueueFlat(set_number=2))
    assert (result.accepted, result.reason) == (False, RejectReason.INVALID)
    assert result.message == fake.NO_FIRST_SET


def test_a_second_task_is_busy_until_the_first_one_ends(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    assert core.submit(QueueFlat()).accepted
    busy = core.submit(QueueFlat())
    assert (busy.accepted, busy.reason) == (False, RejectReason.BUSY)
    clock.advance(30.0)
    assert core.flat_library().task.state == "ok"
    second = core.submit(QueueFlat(set_number=2))  # the first one has ended
    assert second.accepted
    assert second.task_id == 2


def test_a_paused_scheduler_holds_the_task_until_it_resumes(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(Pause())
    answer = core.submit(QueueFlat())
    assert answer.accepted
    assert answer.message == fake.HOLD_MESSAGES["paused"]
    clock.advance(600)
    held = core.flat_library().task
    assert (held.state, held.message) == ("queued", fake.HOLD_MESSAGES["paused"])
    core.submit(Resume())
    clock.advance(4.9)
    assert core.flat_library().task.state == "queued"  # the queue time counts from the resume
    clock.advance(0.2)
    assert core.flat_library().task.state == "running"


def test_a_running_alignment_holds_the_task_until_it_ends(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(StartAlignment())
    answer = core.submit(QueueFlat())
    assert answer.message == fake.HOLD_MESSAGES["align"]
    clock.advance(600)
    assert core.flat_library().task.state == "queued"
    core.submit(StopAlignment())
    clock.advance(SCRIPT.queued_s + 0.1)
    assert core.flat_library().task.state == "running"


def test_a_dark_task_and_a_flat_task_are_counted_as_two_queued_tasks(
    clock: VirtualClock,
) -> None:
    core = FakeCoreClient(
        clock=clock, dark_script=DarkScript(queued_s=3600.0), flat_script=SCRIPT, state="auto"
    )
    core.dark.sets = [dark_set("set-a", 12.0, 5.0)]
    core.submit(QueueDark())
    core.submit(QueueFlat())
    assert core.status().scheduler.queued_tasks == 2


# --- The task --------------------------------------------------------------------------------


def test_the_task_runs_through_its_phases_and_adds_a_pending_flat(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(QueueFlat(frames=16, target_fraction=0.5))
    clock.advance(5.0)  # the task starts
    assert core.status().scheduler.state == "commission"
    task = core.flat_library().task
    assert (task.state, task.phase, task.step, task.steps) == ("running", "setup", 0, 1)
    assert task.message == "Setting up the camera and the library."
    clock.advance(0.6)
    assert core.flat_library().task.step == 1  # the camera is ready
    clock.advance(0.5)  # the search starts: a first try that misses the target
    task = core.flat_library().task
    assert (task.phase, task.step, task.steps) == ("exposure", 1, 8)
    assert task.exposure_s == pytest.approx(0.02)
    assert task.level_fraction == pytest.approx(0.256)
    assert task.message.startswith("Try 1 of 8: 20 ms gives 26 % of full scale")
    clock.advance(1.0)  # the second try lands on the target
    task = core.flat_library().task
    assert (task.phase, task.step) == ("exposure", 2)
    assert task.level_fraction == pytest.approx(0.5, abs=0.01)
    assert task.exposure_s == pytest.approx(0.5 / 12.8)
    clock.advance(1.0)  # the frames start
    task = core.flat_library().task
    assert (task.phase, task.step, task.steps) == ("capture", 1, 16)
    assert task.exposure_s == pytest.approx(0.5 / 12.8)
    assert task.level_fraction == pytest.approx(0.5, abs=0.02)
    assert task.message.startswith("Frame 1 of 16: ")
    clock.advance(7.5)
    assert core.flat_library().task.step == 16
    clock.advance(0.5)  # the build starts
    assert core.flat_library().task.phase == "build"
    clock.advance(1.0)
    library = core.flat_library()
    task = library.task
    assert (task.state, task.phase) == ("ok", None)
    assert task.version is not None
    assert task.version in task.summary
    assert task.summary.startswith(f"Made the flat {task.version} from 16 frames of 39.1 ms.")
    assert "The corners get 10 % less light than the center." in task.summary
    assert task.summary.endswith("It waits on the Flat page for you to use it or discard it.")
    assert task.finished_utc is not None
    (flat,) = library.flats
    assert flat.version == task.version
    assert (flat.state, flat.pending, flat.active) == ("pending", True, False)
    assert library.pending_version == flat.version
    assert (flat.frames_taken, flat.exposure_s, flat.target_fraction) == (16, 0.039062, 0.5)
    assert flat.warnings == [fake.SOURCE_WARNING]  # one set cannot tell the source from the optics
    assert library.session is not None
    assert library.session.version == flat.version
    assert (library.session.frames, library.session.exposure_s) == (16, 0.039062)


def test_the_session_expires_24_hours_after_the_first_set(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    run_session(core, clock)
    session = core.flat_library().session
    assert session is not None
    first = core.flat_library().flats[0]
    assert session.t_utc == first.t_utc
    assert session.expires_utc.endswith("Z")
    assert session.expires_utc > session.t_utc


def test_the_scheduler_pauses_after_the_task_and_resume_works(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    run_session(core, clock)
    scheduler = core.status().scheduler
    assert scheduler.state == "paused"
    assert "light source" in scheduler.state_reason
    assert core.submit(Resume()).accepted
    assert core.status().scheduler.state == "safe"


def test_the_scheduler_goes_to_safe_when_the_task_does_not_ask_for_a_pause(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    run_session(core, clock, QueueFlat(frames=16, pause_after=False))
    assert core.flat_library().task.state == "ok"
    assert core.status().scheduler.state == "safe"


def test_the_warnings_of_a_script_show_from_the_middle_of_the_frames_on(
    clock: VirtualClock,
) -> None:
    note = "The light drifts: a frame is 3.4 % above the median level."
    core = make_core(clock, replace(SCRIPT, warnings=(note,)))
    core.submit(QueueFlat(frames=16))
    clock.advance(5.0 + 1.0 + 2.0 + 3.0)  # a third of the frames
    assert core.flat_library().task.warnings == []
    clock.advance(2.0)  # past the middle
    assert core.flat_library().task.warnings == [note]
    clock.advance(1.0 + 2.0)  # the build keeps the note
    assert core.flat_library().task.warnings == [note]
    clock.advance(1.0)
    (flat,) = core.flat_library().flats
    assert flat.warnings[0] == note  # the report keeps it


@pytest.mark.parametrize(
    ("outcome", "start"),
    [("dim", "Not enough light:"), ("bright", "Too much light:")],
)
def test_a_light_that_is_too_dim_or_too_bright_fails_the_task(
    clock: VirtualClock, outcome: str, start: str
) -> None:
    core = make_core(clock, replace(SCRIPT, outcome=outcome))
    core.submit(QueueFlat())
    clock.advance(5.0 + 1.0 + 1.9)
    task = core.flat_library().task
    assert (task.state, task.phase) == ("running", "exposure")
    assert task.step == 3  # the search runs out of exposures
    clock.advance(0.2)
    library = core.flat_library()
    assert library.task.state == "failed"
    assert library.task.summary.startswith(start)
    assert library.task.summary.endswith("The library is unchanged.")
    assert (library.flats, library.session, library.task.version) == ([], None, None)
    assert core.status().scheduler.state == "paused"  # the light may still cover the camera


def test_a_pause_aborts_a_running_task_and_adds_no_flat(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(QueueFlat())
    clock.advance(9.0)
    assert core.flat_library().task.state == "running"
    assert core.submit(Pause()).accepted
    library = core.flat_library()
    assert library.task.state == "aborted"
    assert library.task.summary == fake.ABORTED_SUMMARY
    assert (library.flats, library.session) == ([], None)
    clock.advance(60)
    assert core.flat_library().task.state == "aborted"  # nothing moves it on
    assert core.status().scheduler.state == "paused"


def test_a_pause_leaves_a_task_that_waits_in_the_queue(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(QueueFlat())
    assert core.submit(Pause()).accepted
    assert core.flat_library().task.state == "queued"  # as in the real scheduler
    assert core.status().scheduler.queued_tasks == 1


# --- The second set --------------------------------------------------------------------------


def test_a_second_set_makes_one_flat_from_both_and_ends_the_session(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    run_session(core, clock)
    first = core.flat_library().flats[0]
    core.submit(Resume())
    run_session(core, clock, QueueFlat(frames=16, set_number=2))
    library = core.flat_library()
    (flat,) = library.flats  # the flat of the first set gave way
    assert flat.version != first.version
    assert library.session is None
    assert (flat.second_set, flat.source_turned, flat.pending) == (True, True, True)
    assert flat.frames_taken == 32
    assert [item.number for item in flat.sets] == [1, 2]
    assert flat.optics_tilt is not None
    assert flat.source_tilt is not None
    assert flat.agreement is not None
    assert flat.agreement.plane is not None
    assert fake.SOURCE_WARNING not in flat.warnings  # two sets separate the source
    assert library.task.summary.startswith(f"Made the flat {flat.version} from two sets of frames.")
    assert library.task.set_number == 2


def test_the_tilt_of_two_sets_adds_up_to_the_tilt_of_the_optics_and_the_source(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.flat.look = FlatLook(tilt=(-0.40, 0.30), source_tilt=(-0.22, 0.11))
    run_session(core, clock)
    core.submit(Resume())
    run_session(core, clock, QueueFlat(frames=16, set_number=2))
    (flat,) = core.flat_library().flats
    first, second = (item.tilt for item in flat.sets)
    assert first == FlatTiltView(width_percent=-0.62, height_percent=0.41)
    assert second == FlatTiltView(width_percent=-0.18, height_percent=0.19)  # the source turned
    assert flat.tilt.width_percent == pytest.approx(-0.40)  # the combined flat keeps the optics


# --- The cancel ------------------------------------------------------------------------------


def test_a_cancel_removes_a_task_that_waits(core: FakeCoreClient) -> None:
    core.submit(Pause())
    queued = core.submit(QueueFlat())
    answer = core.submit(CancelTask(kind="flat"))
    assert answer.accepted
    assert answer.task_id == queued.task_id
    task = core.flat_library().task
    assert (task.state, task.summary) == ("aborted", fake.CANCELLED_SUMMARY)
    assert core.status().scheduler.queued_tasks == 0
    assert core.submit(QueueFlat()).accepted  # the slot is free again


def test_a_cancel_stops_a_running_task_and_the_scheduler_pauses(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.submit(QueueFlat())
    clock.advance(9.0)
    answer = core.submit(CancelTask(kind="flat"))
    assert answer.accepted
    assert "stops at its next check" in answer.message
    library = core.flat_library()
    assert (library.task.state, library.task.summary) == ("aborted", fake.ABORTED_SUMMARY)
    assert library.flats == []
    assert core.status().scheduler.state == "paused"


def test_a_cancel_without_a_task_is_a_rejection(core: FakeCoreClient) -> None:
    answer = core.submit(CancelTask(kind="flat"))
    assert (answer.accepted, answer.reason) == (False, RejectReason.NO_TASK)
    other = core.submit(CancelTask(kind="burst"))
    assert (other.accepted, other.reason) == (False, RejectReason.NO_TASK)
    unknown = core.submit(CancelTask(kind="coffee"))
    assert (unknown.accepted, unknown.reason) == (False, RejectReason.INVALID)
    assert core.submit(CancelTask()).reason is RejectReason.INVALID


def test_a_cancel_of_the_dark_kind_stops_the_dark_task_only(core: FakeCoreClient) -> None:
    core.submit(QueueDark())
    core.submit(QueueFlat())
    assert core.submit(CancelTask(kind="dark")).accepted
    assert core.dark.task().state == "aborted"
    assert core.flat_library().task.state == "queued"


# --- The decisions ---------------------------------------------------------------------------


def test_activating_the_pending_flat_makes_it_the_one_in_use(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    old = core.flat.seed(age_days=30.0, active=True)
    run_session(core, clock)
    pending = core.flat_library().task.version
    assert pending is not None
    answer = core.flat_activate(pending)
    assert (answer.ok, answer.reason, answer.version) == (True, None, pending)
    assert (
        answer.message
        == f"The flat {pending} is in use. The survey divides by it from its next frame."
    )
    library = core.flat_library()
    assert (library.active_version, library.pending_version) == (pending, None)
    assert library.session is None  # the decision ends the session
    by_version = {item.version: item for item in library.flats}
    assert (by_version[pending].state, by_version[pending].active) == ("approved", True)
    assert by_version[pending].activated_utc is not None
    assert by_version[old.version].active is False


def test_a_pinned_flat_file_is_named_in_the_answer_of_an_activation(
    core: FakeCoreClient,
) -> None:
    core.flat.flat_file_pinned = True
    seeded = core.flat.seed(age_days=3.0, state="pending")
    answer = core.flat_activate(seeded.version)
    assert answer.message.endswith("It replaces the flat of the setting flat_file.")


def test_discarding_the_pending_flat_removes_it_and_ends_the_session(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    run_session(core, clock)
    version = core.flat_library().flats[0].version
    answer = core.flat_delete(version)
    assert (answer.ok, answer.reason) == (True, None)
    assert answer.message == f"The flat {version} is deleted."
    library = core.flat_library()
    assert (library.flats, library.session) == ([], None)
    assert core.flat_image(version) is None


def test_the_flat_in_use_cannot_be_deleted(core: FakeCoreClient) -> None:
    seeded = core.flat.seed(age_days=3.0, active=True)
    answer = core.flat_delete(seeded.version)
    assert (answer.ok, answer.reason, answer.message) == (False, "active", fake.ACTIVE_MESSAGE)
    assert [item.version for item in core.flat_library().flats] == [seeded.version]


def test_an_unknown_flat_is_unknown_for_every_decision(core: FakeCoreClient) -> None:
    for action in (core.flat_activate, core.flat_delete):
        answer = action("flat-00000000")
        assert (answer.ok, answer.reason, answer.message) == (
            False,
            "unknown",
            fake.UNKNOWN_MESSAGE,
        )


def test_no_decision_can_come_while_a_session_waits_or_runs(
    core: FakeCoreClient, clock: VirtualClock
) -> None:
    seeded = core.flat.seed(age_days=3.0, state="pending")
    core.submit(QueueFlat())
    for action in (core.flat_activate, core.flat_delete):
        answer = action(seeded.version)
        assert (answer.ok, answer.reason, answer.message) == (False, "busy", fake.BUSY_MESSAGE)
    clock.advance(30.0)
    assert core.flat_activate(seeded.version).ok is True  # the session has ended


# --- The image -------------------------------------------------------------------------------


def test_a_flat_has_a_jpeg_preview_that_depends_on_its_numbers(core: FakeCoreClient) -> None:
    first = core.flat.seed(age_days=3.0)
    second = core.flat.seed(age_days=2.0, look=FlatLook(corner_percent=-5.0, shadows=()))
    one, two = core.flat_image(first.version), core.flat_image(second.version)
    assert one is not None
    assert one.startswith(b"\xff\xd8\xff")
    assert one != two
    assert core.flat_image(first.version) == one  # the same bytes each time
    assert len(one) < 600_000
    assert core.flat_image("flat-00000000") is None
