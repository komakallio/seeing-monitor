"""A dark task in `core`: the command, the RPC, the pause after it, and the library it leaves.

The tests build `CoreApp` on a simulated camera and a `VirtualClock` (see `rig`), send the command
through the RPC method `submit` as `web` does, and read the library through the RPC method
`dark_library`. The camera is covered, or it has a cover that goes on while the task waits.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from seeingmon.clock import Clock
from seeingmon.drivers.base import CameraDriver
from seeingmon.drivers.sim import SimOptions
from seeingmon.frames import Frame
from seeingmon.profile import Profile
from seeingmon.scheduler import (
    Command,
    CommandResult,
    Pause,
    QueueDark,
    RejectReason,
    Resume,
    State,
)
from seeingmon.services.simsky import write_small_profile
from seeingmon.services.web.contract import (
    METHOD_DARK_LIBRARY,
    DarkLibraryView,
    decode_dark_library,
    decode_result,
    encode_command,
)
from tests.survey import simfx

from .rig import CoreRig, build_rig, events_of

DARK_SETTINGS = """
[survey.dark]
frames = 3
bias_frames = 3
poll_s = 1.0
stable_polls = 2
wait_timeout_s = 120.0
"""


def covered_sim(clock: Clock, profile: Profile) -> CameraDriver:
    options = simfx.covered_options(ambient_c=12.0, hot_pixels_per_mpix=200.0, seed=3)
    return simfx.make_driver(profile, options, clock)


class CoverGoesOn(simfx.CoverableDriver):
    """A camera with a cover that goes on after some reads, and a hook that runs at each read."""

    def __init__(self, uncovered: Any, covered: Any, *, cover_after_reads: int) -> None:
        super().__init__(uncovered, covered, cover_after_reads=cover_after_reads)
        self.hook: Callable[[int], None] | None = None

    def read_frame(self, timeout_s: float) -> Frame:
        if self.hook is not None:
            self.hook(self.reads)
        return super().read_frame(timeout_s)


def uncovered_then_covered(
    cover_after_reads: int, holder: dict[str, CoverGoesOn]
) -> Callable[[Clock, Profile], CameraDriver]:
    def make(clock: Clock, profile: Profile) -> CameraDriver:
        uncovered = simfx.make_driver(profile, SimOptions(seed=5), clock)
        cover = simfx.make_driver(
            profile, simfx.covered_options(ambient_c=12.0, hot_pixels_per_mpix=200.0, seed=5), clock
        )
        driver = CoverGoesOn(uncovered, cover, cover_after_reads=cover_after_reads)
        holder["driver"] = driver
        return driver

    return make


def make_rig(
    tmp_path: Path, factory: Callable[[Clock, Profile], CameraDriver] = covered_sim
) -> CoreRig:
    profile = write_small_profile(tmp_path)
    return build_rig(tmp_path, profile=profile, config_extra=DARK_SETTINGS, driver_factory=factory)


@pytest.fixture
def rig(tmp_path: Path) -> CoreRig:
    built = make_rig(tmp_path)
    built.app.start()
    return built


def call(rig: CoreRig, method: str, params: dict[str, Any] | None = None) -> Any:
    return rig.app.rpc.handlers()[method](params or {})


def send(rig: CoreRig, command: Command) -> CommandResult:
    return decode_result(call(rig, "submit", {"command": encode_command(command)}))


def library(rig: CoreRig) -> DarkLibraryView:
    return decode_dark_library(call(rig, METHOD_DARK_LIBRARY))


def run_until(rig: CoreRig, done: Callable[[], bool], *, limit_s: float = 3600.0) -> None:
    waited = 0.0
    while not done():
        rig.run_for(5.0)
        waited += 5.0
        assert waited < limit_s, "the condition never held"


def paused(rig: CoreRig) -> bool:
    return rig.app.scheduler.state is State.PAUSED


class TestTheRpc:
    def test_dark_library_answers_with_the_view_of_an_empty_library(self, rig: CoreRig) -> None:
        view = library(rig)
        assert view.sets == []
        assert view.model is None
        assert view.status.due is True
        assert view.status.reason == "the library holds no dark set"
        assert (view.mode, view.gain, view.exposure_s) == ("bin2", 120, 30.0)
        assert view.task.state == "idle"
        assert view.sensor_temperature_c is None  # no frame has come yet

    def test_the_method_is_among_the_handlers_of_the_rpc(self, rig: CoreRig) -> None:
        assert METHOD_DARK_LIBRARY in rig.app.rpc.handlers()

    def test_a_task_that_the_scheduler_accepted_is_queued_in_the_view(self, rig: CoreRig) -> None:
        result = send(rig, QueueDark(wait_for_cover=False))
        assert result.accepted
        view = library(rig)
        assert (view.task.state, view.task.task_id) == ("queued", result.task_id)
        assert (view.task.frames, view.task.bias_frames, view.task.exposure_s) == (3, 3, 30.0)

    def test_a_task_queued_while_the_scheduler_is_paused_says_that_it_waits_for_the_resume(
        self, rig: CoreRig
    ) -> None:
        assert send(rig, Pause()).accepted
        send(rig, QueueDark(wait_for_cover=False))
        rig.run_for(60.0)
        view = library(rig)
        assert view.task.state == "queued"  # nothing runs while the scheduler is paused
        assert "paused" in view.task.message
        assert "resume" in view.task.message
        assert send(rig, Resume()).accepted
        run_until(rig, lambda: library(rig).task.state == "ok")

    def test_a_command_that_the_scheduler_rejected_leaves_the_view_alone(
        self, rig: CoreRig
    ) -> None:
        send(rig, QueueDark(wait_for_cover=False))
        second = send(rig, QueueDark(frames=5))
        assert (second.accepted, second.reason) == (False, RejectReason.BUSY)
        assert library(rig).task.frames == 3  # still the first task's settings
        bad = send(rig, QueueDark(frames=2))
        assert (bad.accepted, bad.reason) == (False, RejectReason.INVALID)


class TestTheFlow:
    def test_the_task_adds_a_set_and_the_scheduler_pauses_after_it(self, rig: CoreRig) -> None:
        before = rig.app.health.build()
        send(rig, QueueDark(wait_for_cover=False))
        run_until(rig, lambda: paused(rig))
        view = library(rig)
        assert view.task.state == "ok"
        assert view.task.summary.startswith("Added dark-")
        (item,) = view.sets
        assert view.task.set_name == item.name
        assert view.status.due is False
        assert view.model is not None
        assert view.sensor_temperature_c == pytest.approx(16.0)
        status = rig.app.scheduler.status()
        assert status.state == "paused"
        assert "dark session" in status.state_reason
        assert "covered" in status.state_reason
        after = rig.app.health.build()
        assert before.dark_due is True
        assert after.dark_due is False  # the library covers the temperature now

    def test_the_events_tell_the_phases_and_the_result(self, rig: CoreRig) -> None:
        send(rig, QueueDark(wait_for_cover=False))
        run_until(rig, lambda: paused(rig))
        rig.run_for(2.0)
        records = rig.records("event")
        phases = [e.detail["phase"] for e in events_of(records, "scheduler.dark_phase")]
        assert phases == ["bias", "dark", "build"]
        (result,) = events_of(records, "scheduler.dark_result")
        assert result.detail["status"] == "ok"
        assert result.detail["data"]["remove_cover"] is True
        assert result.detail["artifacts"][0].startswith("calibration/darks/dark-")
        change = [
            e for e in events_of(records, "scheduler.state_change") if e.detail["to"] == "paused"
        ]
        assert "dark session" in change[0].detail["reason"]

    def test_resume_brings_the_system_back_to_auto(self, rig: CoreRig) -> None:
        send(rig, QueueDark(wait_for_cover=False))
        run_until(rig, lambda: paused(rig))
        resumed = send(rig, Resume())
        assert resumed.accepted
        run_until(rig, lambda: rig.app.scheduler.state is State.AUTO)
        assert library(rig).task.state == "ok"  # the result stays in view

    def test_the_wait_for_the_cover_shows_in_the_view_and_ends_when_the_cover_is_on(
        self, tmp_path: Path
    ) -> None:
        holder: dict[str, CoverGoesOn] = {}
        rig = make_rig(tmp_path, uncovered_then_covered(3 + 3, holder))
        rig.app.start()
        seen: list[tuple[str | None, bool | None]] = []

        def watch(reads: int) -> None:
            task = library(rig).task
            seen.append((task.phase, task.covered))

        holder["driver"].hook = watch
        send(rig, QueueDark())
        run_until(rig, lambda: paused(rig))
        assert ("cover", False) in seen  # a test frame that saw the sky
        assert ("cover", True) in seen  # and then one that did not
        assert seen[0][0] == "bias"
        assert seen[-1][0] == "dark"
        assert library(rig).task.state == "ok"
        assert len(library(rig).sets) == 1

    def test_a_camera_that_is_not_covered_fails_a_task_that_does_not_wait(
        self, tmp_path: Path
    ) -> None:
        holder: dict[str, CoverGoesOn] = {}
        rig = make_rig(tmp_path, uncovered_then_covered(10**6, holder))
        rig.app.start()
        send(rig, QueueDark(wait_for_cover=False))
        run_until(rig, lambda: paused(rig))
        view = library(rig)
        assert view.task.state == "failed"
        assert view.task.summary.startswith("Dark frame 1 of 3 is not dark:")
        assert view.sets == []
        assert view.status.due is True  # the library stays as it was
        (result,) = events_of(rig.records("event"), "scheduler.dark_result")
        assert result.level == "warning"

    def test_a_pause_in_the_middle_of_the_dark_frames_aborts_the_task(self, tmp_path: Path) -> None:
        holder: dict[str, CoverGoesOn] = {}
        rig = make_rig(tmp_path, uncovered_then_covered(0, holder))  # covered from the start
        rig.app.start()

        def pause_at_the_second_dark_frame(reads: int) -> None:
            if reads == 3 + 1:
                assert rig.app.scheduler.submit(Pause()).accepted

        holder["driver"].hook = pause_at_the_second_dark_frame
        send(rig, QueueDark(wait_for_cover=False))
        run_until(rig, lambda: library(rig).task.state == "aborted")
        view = library(rig)
        assert "another command took the camera" in view.task.summary
        assert view.sets == []
        assert rig.app.scheduler.state is State.PAUSED  # the pause that the owner pressed
        results = rig.app.scheduler.results()
        assert [r.status for r in results] == ["aborted"]


class TestAfterARestart:
    def test_the_library_survives_a_restart_of_core(self, tmp_path: Path) -> None:
        first = make_rig(tmp_path)
        first.app.start()
        send(first, QueueDark(wait_for_cover=False))
        run_until(first, lambda: paused(first))
        (made,) = library(first).sets
        first.app.stop()

        second = make_rig(tmp_path)
        second.app.start()
        try:
            view = library(second)
            assert [item.name for item in view.sets] == [made.name]
            assert view.task.state == "idle"  # the session belongs to the process that ran it
            assert view.model is not None
            assert view.model.n_sets == 1
        finally:
            second.app.stop()

    def test_the_status_of_the_library_follows_the_temperature_of_the_scheduler(
        self, rig: CoreRig
    ) -> None:
        send(rig, QueueDark(wait_for_cover=False))
        run_until(rig, lambda: paused(rig))
        send(rig, Resume())
        run_until(rig, lambda: rig.app.scheduler.state is State.AUTO)
        assert library(rig).status.due is False
