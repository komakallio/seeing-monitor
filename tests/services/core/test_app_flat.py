"""A flat task in `core`: the command, the RPC, the pause after it, and the library it leaves.

The tests build `CoreApp` on a scripted camera and a `VirtualClock` (see `rig`), send the command
through the RPC method `submit` as `web` does, and read the library through the RPC method
`flat_library`. The camera looks at a steady light panel through a lens on a sensor of 640 x 480
pixels, and the dark library holds two sets of bin2 at gain 120.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.frames import FrameData, Roi, StreamConfig
from seeingmon.profile import Profile
from seeingmon.scheduler import (
    CancelTask,
    Command,
    CommandResult,
    Pause,
    QueueFlat,
    RejectReason,
    Resume,
    State,
)
from seeingmon.services.simsky import write_small_profile
from seeingmon.services.web.contract import (
    METHOD_FLAT_ACTIVATE,
    METHOD_FLAT_DELETE,
    METHOD_FLAT_IMAGE,
    METHOD_FLAT_LIBRARY,
    FlatActionView,
    FlatLibraryView,
    decode_flat_action,
    decode_flat_image,
    decode_flat_library,
    decode_result,
    encode_command,
)
from seeingmon.survey.flat_library import active_flat
from seeingmon.survey.flat_session import DARK_FIRST, NO_FIRST_SET
from seeingmon.testing import FakeCameraDriver
from tests.survey import flatfx as fx

from .rig import NIGHT, CoreRig, build_rig, events_of

SHAPE = (480, 640)  # the bin2 sensor of the small profile: rows, columns
FRAMES = 8
RATE_DN_PER_S = 204_800.0  # the level above the bias in the middle, per second of exposure
BIAS_DN = 131.0
GRADIENT = (-0.0064, 0.0068)  # the tilt that the source adds, as in the data of the owner


class PanelDriver(FakeCameraDriver):
    """A camera in front of a steady panel. `hook` runs before each read, with the reads so far."""

    def __init__(self, clock: Clock) -> None:
        super().__init__(
            clock,
            full_frames={"bin1": (SHAPE[1] * 2, SHAPE[0] * 2), "bin2": (SHAPE[1], SHAPE[0])},
            temperature_c=25.0,
            frame_factory=self._frame,
        )
        self.truth = fx.lens_flat(fx.OWNER_LENS, SHAPE, edge_artifact_x=14.0)
        self.bias = fx.bias_pattern(SHAPE, level=BIAS_DN)
        self.gradient = GRADIENT
        self.reads = 0
        self.hook: Callable[[int], None] | None = None

    def _frame(self, config: StreamConfig, roi: Roi, seq: int) -> FrameData:
        if (roi.height, roi.width) != SHAPE:  # a window of the fast path: plain sky
            return np.full((roi.height, roi.width), 500 << 2, dtype=np.uint16)
        rng = np.random.default_rng([11, self.reads, config.exposure_us])
        level = RATE_DN_PER_S * config.exposure_us / 1e6
        pattern = fx.source_pattern(SHAPE, self.gradient)
        return fx.panel_frame(self.truth, pattern, level=level, bias=self.bias, rng=rng)

    def read_frame(self, timeout_s: float) -> Any:
        if self.hook is not None:
            self.hook(self.reads)
        self.reads += 1
        return super().read_frame(timeout_s)


def make_rig(tmp_path: Path, *, darks: bool = True, start_utc_ns: int = NIGHT) -> CoreRig:
    def driver(clock: Clock, profile: Profile) -> PanelDriver:
        return PanelDriver(clock)

    built = build_rig(
        tmp_path,
        start_utc_ns=start_utc_ns,
        profile=write_small_profile(tmp_path),
        driver_factory=driver,
    )
    if darks:
        fx.make_library(Path(built.app.survey_config.calibration_dir))
    return built


def panel(rig: CoreRig) -> PanelDriver:
    driver = rig.app.parts.driver
    assert isinstance(driver, PanelDriver)
    return driver


@pytest.fixture
def rig(tmp_path: Path) -> CoreRig:
    built = make_rig(tmp_path)
    built.app.start()
    return built


def call(rig: CoreRig, method: str, params: dict[str, Any] | None = None) -> Any:
    return rig.app.rpc.handlers()[method](params or {})


def send(rig: CoreRig, command: Command) -> CommandResult:
    return decode_result(call(rig, "submit", {"command": encode_command(command)}))


def library(rig: CoreRig) -> FlatLibraryView:
    return decode_flat_library(call(rig, METHOD_FLAT_LIBRARY))


def run_until(rig: CoreRig, done: Callable[[], bool], *, limit_s: float = 3600.0) -> None:
    waited = 0.0
    while not done():
        rig.run_for(2.0)
        waited += 2.0
        assert waited < limit_s, "the condition never held"


def paused(rig: CoreRig) -> bool:
    return rig.app.scheduler.state is State.PAUSED


def ended(rig: CoreRig) -> bool:
    return library(rig).task.state in ("ok", "failed", "aborted")


def take(rig: CoreRig, command: QueueFlat | None = None) -> FlatLibraryView:
    """Queue a flat task, and run the scheduler until the task ends."""
    result = send(rig, command or QueueFlat(frames=FRAMES))
    assert result.accepted, result.message
    run_until(rig, lambda: ended(rig))
    return library(rig)


def resume(rig: CoreRig) -> None:
    assert send(rig, Resume()).accepted
    run_until(rig, lambda: rig.app.scheduler.state is State.AUTO)


class TestTheRpc:
    def test_flat_library_answers_with_the_view_of_an_empty_library(self, rig: CoreRig) -> None:
        view = library(rig)
        assert view.flats == []
        assert (view.mode, view.gain) == ("bin2", 120)
        assert view.task.state == "idle"
        assert view.blocker is None
        assert view.session is None
        assert view.sensor_temperature_c is None  # no frame has come yet
        assert (view.flat_file_pinned, view.library_overrides) == (False, False)

    def test_the_methods_are_among_the_handlers_of_the_rpc(self, rig: CoreRig) -> None:
        handlers = rig.app.rpc.handlers()
        for method in (
            METHOD_FLAT_LIBRARY,
            METHOD_FLAT_ACTIVATE,
            METHOD_FLAT_DELETE,
            METHOD_FLAT_IMAGE,
        ):
            assert method in handlers

    def test_a_task_that_the_scheduler_accepted_is_queued_in_the_view(self, rig: CoreRig) -> None:
        result = send(rig, QueueFlat(frames=FRAMES, target_fraction=0.4, pause_after=False))
        assert result.accepted
        view = library(rig)
        assert (view.task.state, view.task.task_id) == ("queued", result.task_id)
        assert (view.task.frames, view.task.target_fraction) == (FRAMES, 0.4)
        assert view.task.pause_after is False

    def test_a_task_queued_while_the_scheduler_is_paused_waits_for_the_resume(
        self, rig: CoreRig
    ) -> None:
        assert send(rig, Pause()).accepted
        send(rig, QueueFlat(frames=FRAMES))
        rig.run_for(60.0)
        view = library(rig)
        assert view.task.state == "queued"  # nothing runs while the scheduler is paused
        assert "paused" in view.task.message
        assert "resume" in view.task.message
        assert send(rig, Resume()).accepted
        run_until(rig, lambda: ended(rig))
        assert library(rig).task.state == "ok"

    def test_a_command_that_the_scheduler_rejected_leaves_the_view_alone(
        self, rig: CoreRig
    ) -> None:
        send(rig, QueueFlat(frames=FRAMES))
        second = send(rig, QueueFlat(frames=16))
        assert (second.accepted, second.reason) == (False, RejectReason.BUSY)
        assert library(rig).task.frames == FRAMES  # still the first task's settings
        bad = send(rig, QueueFlat(frames=7))
        assert (bad.accepted, bad.reason) == (False, RejectReason.INVALID)

    def test_a_flat_without_a_dark_set_is_refused_before_the_scheduler_sees_it(
        self, tmp_path: Path
    ) -> None:
        made = make_rig(tmp_path, darks=False)
        made.app.start()
        assert library(made).blocker == DARK_FIRST
        answer = send(made, QueueFlat(frames=FRAMES))
        assert (answer.accepted, answer.reason) == (False, RejectReason.INVALID)
        assert answer.message == DARK_FIRST
        assert made.app.scheduler.status().queued_tasks == 0
        assert library(made).task.state == "idle"

    def test_a_second_set_without_a_first_one_is_refused(self, rig: CoreRig) -> None:
        answer = send(rig, QueueFlat(frames=FRAMES, set_number=2))
        assert (answer.accepted, answer.reason) == (False, RejectReason.INVALID)
        assert answer.message == NO_FIRST_SET


class TestTheFlow:
    def test_the_task_adds_a_pending_flat_and_the_scheduler_pauses_after_it(
        self, rig: CoreRig
    ) -> None:
        view = take(rig)
        assert view.task.state == "ok"
        assert view.task.summary.startswith("Made the flat flat-")
        (flat,) = view.flats
        assert view.task.version == flat.version
        assert (flat.pending, flat.active, flat.state) == (True, False, "pending")
        assert view.pending_version == flat.version
        assert view.session is not None  # the first set waits for a second one
        assert view.session.version == flat.version
        assert view.sensor_temperature_c == pytest.approx(25.0)
        assert (flat.mode, flat.gain, flat.second_set) == ("bin2", 120, False)
        assert (flat.frames_taken, flat.frames_used) == (FRAMES, FRAMES)
        assert flat.has_image is True
        status = rig.app.scheduler.status()
        assert status.state == "paused"
        assert "flat session" in status.state_reason
        assert "light source" in status.state_reason

    def test_nothing_changes_for_the_survey_before_the_flat_is_activated(
        self, rig: CoreRig
    ) -> None:
        take(rig)
        assert active_flat(rig.app.survey_config).version == "unit"
        assert rig.app.flat_library.active_version() is None

    def test_the_events_tell_the_phases_and_the_result(self, rig: CoreRig) -> None:
        take(rig)
        run_until(rig, lambda: paused(rig))
        rig.run_for(2.0)
        records = rig.records("event")
        phases = [e.detail["phase"] for e in events_of(records, "scheduler.flat_phase")]
        assert phases == ["setup", "exposure", "capture", "build"]
        (result,) = events_of(records, "scheduler.flat_result")
        assert result.detail["status"] == "ok"
        assert result.detail["data"]["remove_light"] is True
        assert result.detail["data"]["second_set"] is False
        assert result.detail["artifacts"][0].startswith("calibration/flats/flat-")
        change = [
            e for e in events_of(records, "scheduler.state_change") if e.detail["to"] == "paused"
        ]
        assert "flat session" in change[0].detail["reason"]

    def test_resume_brings_the_system_back_to_auto(self, rig: CoreRig) -> None:
        take(rig)
        run_until(rig, lambda: paused(rig))
        resume(rig)
        assert library(rig).task.state == "ok"  # the result stays in view

    def test_the_image_of_the_flat_is_a_jpeg(self, rig: CoreRig) -> None:
        version = take(rig).flats[0].version
        answer = call(rig, METHOD_FLAT_IMAGE, {"version": version})
        assert answer["found"] is True
        jpeg = decode_flat_image(answer)
        assert jpeg is not None
        assert base64.b64decode(answer["jpeg"]) == jpeg
        assert jpeg.startswith(b"\xff\xd8\xff")

    def test_activating_the_flat_changes_what_the_survey_divides_by(self, rig: CoreRig) -> None:
        version = take(rig).flats[0].version
        answer = decode_flat_action(call(rig, METHOD_FLAT_ACTIVATE, {"version": version}))
        assert answer.ok is True
        assert version in answer.message
        view = library(rig)
        assert (view.active_version, view.pending_version) == (version, None)
        assert view.session is None  # the decision ended the session
        assert (view.flats[0].state, view.flats[0].active) == ("approved", True)
        # The call that the previews and the pipeline make follows the pointer without a restart.
        assert active_flat(rig.app.survey_config).version == version

    def test_a_flat_that_is_not_in_use_can_be_deleted(self, rig: CoreRig) -> None:
        version = take(rig).flats[0].version
        answer = decode_flat_action(call(rig, METHOD_FLAT_DELETE, {"version": version}))
        assert answer.ok is True
        view = library(rig)
        assert (view.flats, view.session) == ([], None)
        assert decode_flat_image(call(rig, METHOD_FLAT_IMAGE, {"version": version})) is None

    def test_the_flat_in_use_cannot_be_deleted(self, rig: CoreRig) -> None:
        version = take(rig).flats[0].version
        call(rig, METHOD_FLAT_ACTIVATE, {"version": version})
        answer = decode_flat_action(call(rig, METHOD_FLAT_DELETE, {"version": version}))
        assert (answer.ok, answer.reason) == (False, "active")
        assert [item.version for item in library(rig).flats] == [version]

    def test_a_name_that_is_no_version_never_reaches_a_file(self, rig: CoreRig) -> None:
        for name in ("../current", "flat-1", "", "FLAT-12345678", 7, None):
            for method in (METHOD_FLAT_ACTIVATE, METHOD_FLAT_DELETE):
                answer: FlatActionView = decode_flat_action(call(rig, method, {"version": name}))
                assert (answer.ok, answer.reason) == (False, "unknown")
            assert decode_flat_image(call(rig, METHOD_FLAT_IMAGE, {"version": name})) is None

    def test_no_decision_can_come_while_a_session_waits_or_runs(self, rig: CoreRig) -> None:
        version = take(rig).flats[0].version
        assert send(rig, QueueFlat(frames=FRAMES, set_number=2)).accepted
        for method in (METHOD_FLAT_ACTIVATE, METHOD_FLAT_DELETE):
            answer = decode_flat_action(call(rig, method, {"version": version}))
            assert (answer.ok, answer.reason) == (False, "busy")

    def test_a_second_set_makes_one_flat_from_both(self, rig: CoreRig) -> None:
        first = take(rig)
        assert first.task.state == "ok"
        panel(rig).gradient = (-GRADIENT[0], -GRADIENT[1])  # the source turned by 180 degrees
        # The scheduler is paused after the first set, and the second set starts when it resumes.
        assert send(rig, QueueFlat(frames=FRAMES, set_number=2)).accepted
        assert library(rig).task.state == "queued"
        assert send(rig, Resume()).accepted
        run_until(rig, lambda: ended(rig))
        view = library(rig)
        assert (view.task.state, view.task.set_number) == ("ok", 2)
        (flat,) = view.flats  # the flat of the first set alone gave way
        assert flat.version != first.flats[0].version
        assert (flat.second_set, flat.source_turned) == (True, True)
        assert flat.optics_tilt is not None
        assert flat.source_tilt is not None
        assert flat.agreement is not None
        assert len(flat.sets) == 2
        assert view.session is None
        assert view.task.summary.startswith(f"Made the flat {flat.version} from two sets")

    def test_a_pause_in_the_middle_of_the_frames_aborts_the_task(self, tmp_path: Path) -> None:
        made = make_rig(tmp_path)
        made.app.start()

        def pause_at_the_fourth_frame(reads: int) -> None:
            task = library(made).task
            if task.phase == "capture" and task.step == 3:
                panel(made).hook = None
                assert made.app.scheduler.submit(Pause()).accepted

        panel(made).hook = pause_at_the_fourth_frame
        send(made, QueueFlat(frames=FRAMES))
        run_until(made, lambda: ended(made))
        view = library(made)
        assert view.task.state == "aborted"
        assert view.task.summary == (
            "The flat session stopped before it added a flat, and the library is unchanged."
        )
        assert (view.flats, view.session) == ([], None)
        assert not made.app.flat_library.session.directory.exists()  # the partial frames are gone
        assert made.app.scheduler.state is State.PAUSED  # the pause that the owner pressed
        assert [r.status for r in made.app.scheduler.results()] == ["aborted"]


class TestAFailure:
    def test_a_light_that_is_too_dim_fails_the_task_and_leaves_the_library_alone(
        self, tmp_path: Path
    ) -> None:
        made = make_rig(tmp_path)
        made.app.start()
        driver = panel(made)
        driver.truth = driver.truth * np.float32(0.001)  # almost no light reaches the sensor
        view = take(made)
        assert view.task.state == "failed"
        assert view.task.summary.startswith("Not enough light:")
        assert view.task.summary.endswith("The library is unchanged.")
        assert (view.flats, view.session) == ([], None)
        (result,) = events_of(made.records("event"), "scheduler.flat_result")
        assert result.level == "warning"
        run_until(made, lambda: paused(made))  # the light source may still cover the camera


class TestTheCancel:
    def test_a_queued_task_is_removed_and_a_new_one_may_follow(self, rig: CoreRig) -> None:
        send(rig, Pause())
        queued = send(rig, QueueFlat(frames=FRAMES))
        assert queued.accepted
        answer = send(rig, CancelTask(kind="flat"))
        assert answer.accepted
        assert answer.task_id == queued.task_id
        view = library(rig)
        assert view.task.state == "aborted"
        assert view.task.summary == "The flat session was cancelled before it started."
        assert rig.app.scheduler.status().queued_tasks == 0
        assert send(rig, QueueFlat(frames=FRAMES)).accepted

    def test_nothing_to_cancel_is_a_rejection(self, rig: CoreRig) -> None:
        answer = send(rig, CancelTask(kind="flat"))
        assert (answer.accepted, answer.reason) == (False, RejectReason.NO_TASK)

    def test_a_running_task_stops_and_the_scheduler_still_pauses(self, tmp_path: Path) -> None:
        made = make_rig(tmp_path)
        made.app.start()
        answers: list[CommandResult] = []

        def cancel_at_the_third_frame(reads: int) -> None:
            task = library(made).task
            if task.phase == "capture" and task.step == 2 and not answers:
                answers.append(made.app.scheduler.submit(CancelTask(kind="flat")))

        panel(made).hook = cancel_at_the_third_frame
        send(made, QueueFlat(frames=FRAMES))
        run_until(made, lambda: ended(made))
        assert answers[0].accepted
        assert "stops at its next check" in answers[0].message
        assert library(made).task.state == "aborted"
        run_until(made, lambda: paused(made))
        view = library(made)
        assert (view.flats, view.session) == ([], None)
        assert not made.app.flat_library.session.directory.exists()  # the partial frames are gone


class TestTheSweep:
    def test_the_frames_of_a_first_set_that_nobody_used_go_after_24_hours(
        self, rig: CoreRig
    ) -> None:
        take(rig)
        folder = rig.app.flat_library.session.directory
        rig.run_for(700.0)  # the sweep runs every 10 minutes, and the first set is young
        assert (folder / "set1.ser").is_file()
        rig.clock.advance(25 * 3600)
        rig.run_for(700.0)
        assert not folder.exists()
        view = library(rig)
        assert view.session is None
        assert [item.state for item in view.flats] == ["pending"]  # the flat itself stays

    def test_a_session_that_captures_keeps_its_folder(self, tmp_path: Path) -> None:
        made = make_rig(tmp_path)
        made.app.start()
        seen: list[bool] = []

        def sweep_in_the_middle(reads: int) -> None:
            task = library(made).task
            if task.phase == "capture" and task.step == 3 and not seen:
                made.clock.advance(25 * 3600)  # the session looks expired to the sweep
                seen.append(made.app.flat_library.session.sweep_idle(made.clock.utc_ns()))

        panel(made).hook = sweep_in_the_middle
        view = take(made)
        assert seen == [False]  # the sweep found the folder in use and did nothing
        assert view.task.state == "ok"


class TestAfterARestart:
    def test_the_library_and_the_active_flat_survive_a_restart_of_core(
        self, tmp_path: Path
    ) -> None:
        first = make_rig(tmp_path)
        first.app.start()
        version = take(first).flats[0].version
        call(first, METHOD_FLAT_ACTIVATE, {"version": version})
        first.app.stop()

        second = make_rig(tmp_path, darks=False)
        second.app.start()
        try:
            view = library(second)
            assert [item.version for item in view.flats] == [version]
            assert view.active_version == version
            assert view.task.state == "idle"  # the session belongs to the process that ran it
            assert active_flat(second.app.survey_config).version == version
        finally:
            second.app.stop()

    def test_a_first_set_that_expired_while_core_was_down_is_gone_at_the_start(
        self, tmp_path: Path
    ) -> None:
        first = make_rig(tmp_path)
        first.app.start()
        take(first)
        folder = first.app.flat_library.session.directory
        assert (folder / "set1.ser").is_file()
        first.app.stop()
        second = make_rig(tmp_path, darks=False, start_utc_ns=NIGHT + 25 * 3600 * NS_PER_S)
        second.app.start()
        try:
            assert not folder.exists()
            view = library(second)
            assert view.session is None
            assert [item.state for item in view.flats] == ["pending"]
        finally:
            second.app.stop()


def test_the_view_has_no_path_in_it(rig: CoreRig, tmp_path: Path) -> None:
    take(rig)
    text = json.dumps(call(rig, METHOD_FLAT_LIBRARY))
    assert str(tmp_path) not in text
    assert tmp_path.as_posix() not in text
