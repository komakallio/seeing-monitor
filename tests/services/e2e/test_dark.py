"""A dark session on real processes: the owner covers the camera, starts it, and uncovers it again.

`acquire` (with the `sim` camera), `core`, and `web` run as subprocesses with a scaled clock. The
test plays the owner. It puts the cover on the simulated camera with a file (`System.cover`), it
queues the dark task through the RPC of `core` as `web` does, it watches the `dark_library` view
that the Dark page shows, and it presses Resume when the session is done. The tests check what the
owner sees (the progress, the new set, the paused scheduler), and what the system records (the
events, the health flag, and the next survey frame).

The tests share one running system and go in order, because a dark session takes half a minute
of real time. Each test begins with `ensure_auto`, which presses Resume when the scheduler waits
in `paused`. The module carries the `slow` marker, so run it with `--slow`.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Iterator
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("scipy", reason="the fast path needs the fast extra")
pytest.importorskip("fastapi", reason="web needs the web extra")

from seeingmon.clock import iso_to_utc_ns
from seeingmon.scheduler import Command, CommandResult, Pause, QueueDark, RejectReason, Resume
from seeingmon.services.web.contract import (
    METHOD_DARK_LIBRARY,
    DarkLibraryView,
    DarkTaskView,
    decode_dark_library,
    decode_result,
    decode_status,
    encode_command,
)

from .system import System, clean_environment

pytestmark = pytest.mark.slow

WINDOW_S = 8.0
CYCLE_S = 20.0
DARK_EXPOSURE_S = 10.0  # the exposure of a survey frame, and of a dark frame
FINISHED = ("ok", "failed", "aborted")


@pytest.fixture(scope="module")
def system(tmp_path_factory: pytest.TempPathFactory) -> Iterator[System]:
    built = System(
        tmp_path_factory.mktemp("dark"),
        speed=2.0,
        window_s=WINDOW_S,
        core_overrides={
            "scheduler": {
                "fast": {"window_s": WINDOW_S, "exposure_us": 50_000},
                "survey": {"cadence_s": CYCLE_S, "long_exposure_s": DARK_EXPOSURE_S},
                "cloud": {"fast_window_s": WINDOW_S, "survey_cadence_s": CYCLE_S},
            },
            "survey": {
                "dark": {
                    "exposure_s": DARK_EXPOSURE_S,
                    "frames": 3,
                    "bias_frames": 3,
                    "poll_s": 2.0,
                    "test_exposure_s": 0.5,
                    "wait_timeout_s": 600.0,
                }
            },
        },
    )
    built.start_all()
    yield built
    built.stop_all()


def send(system: System, command: Command) -> CommandResult:
    """Send a command through the RPC of `core`, as `web` does."""
    return decode_result(system.core_call("submit", {"command": encode_command(command)}))


def library(system: System) -> DarkLibraryView:
    return decode_dark_library(system.core_call(METHOD_DARK_LIBRARY))


def scheduler_state(system: System) -> str:
    return decode_status(system.core_call("status")).scheduler.state


def wait_for_pause(system: System) -> None:
    """The view says that the task is over a moment before the scheduler pauses."""
    system.wait_for(lambda: scheduler_state(system) == "paused", "the scheduler to pause")


def ensure_auto(system: System) -> None:
    """Press Resume when the scheduler waits in `paused`, and wait until it runs again."""
    if scheduler_state(system) == "paused":
        assert send(system, Resume()).accepted
    system.wait_for(lambda: scheduler_state(system) == "auto", "the scheduler to run in auto")


def watch(
    system: System,
    what: str,
    done: Callable[[DarkTaskView], bool],
    each: Callable[[DarkTaskView], None] | None = None,
) -> DarkTaskView:
    """Look at the task as the Dark page does until `done` holds. `each` sees every look."""
    latest: list[DarkTaskView] = []

    def look() -> bool:
        task = library(system).task
        latest.append(task)
        if each is not None:
            each(task)
        return done(task)

    system.wait_for(look, what)
    return latest[-1]


def finished(task: DarkTaskView) -> bool:
    return task.state in FINISHED


def newest_health(system: System) -> Any:
    return max(system.records("health"), key=lambda record: record.t_utc_ns)


class TestTheFlow:
    def test_the_owner_covers_the_camera_and_the_task_adds_a_set(self, system: System) -> None:
        ensure_auto(system)
        system.uncover()
        before = library(system)
        assert before.sets == []
        assert before.status.due is True
        assert before.task.state == "idle"
        system.wait_for(lambda: bool(system.records("health")), "the first health record")
        assert min(system.records("health"), key=lambda r: r.t_utc_ns).dark_due is True

        queued = send(system, QueueDark())
        assert queued.accepted
        first = library(system).task
        assert first.task_id == queued.task_id  # the view follows the new task at once
        assert first.state in ("queued", "running")
        seen: list[tuple[str, str | None, bool | None]] = []
        again: list[CommandResult] = []

        def look(task: DarkTaskView) -> None:
            seen.append((task.state, task.phase, task.covered))
            if not again:
                again.append(send(system, QueueDark()))  # the owner presses Start a second time
            if task.phase == "cover" and task.covered is False:
                system.cover()  # the camera sees the sky, so the owner puts the cover on it

        task = watch(system, "the dark task to finish", finished, look)

        assert task.state == "ok", task.summary
        assert (again[0].accepted, again[0].reason) == (False, RejectReason.BUSY)
        phases = [phase for _, phase, _ in seen if phase is not None]
        assert phases[0] in ("bias", "cover")  # a look may miss the short bias phase
        assert "cover" in phases
        assert phases.index("cover") < phases.index("dark")
        assert (task.frames, task.bias_frames, task.exposure_s) == (3, 3, DARK_EXPOSURE_S)

        view = library(system)
        (added,) = view.sets
        assert task.set_name == added.name
        assert view.status.due is False
        assert view.model is not None
        assert view.task.summary.startswith("Added dark-")

    def test_the_system_records_what_happened(self, system: System) -> None:
        system.wait_for(
            lambda: len(system.events("scheduler.dark_result")) == 1, "the result event of the task"
        )
        phases = [e.detail["phase"] for e in system.events("scheduler.dark_phase")]
        assert phases == ["bias", "cover", "dark", "build"]
        (result,) = system.events("scheduler.dark_result")
        assert result.detail["status"] == "ok"
        assert result.detail["data"]["remove_cover"] is True
        assert result.detail["artifacts"][0].startswith("calibration/darks/dark-")

    def test_the_scheduler_waits_in_pause_for_the_owner_to_uncover_the_camera(
        self, system: System
    ) -> None:
        wait_for_pause(system)
        scheduler = decode_status(system.core_call("status")).scheduler
        assert "dark session" in scheduler.state_reason
        assert "covered" in scheduler.state_reason
        system.wait_for(
            lambda: (
                (system.get_or_none("status") or {}).get("scheduler", {}).get("state") == "paused"
            ),
            "the web status to show the pause",
        )

    def test_the_health_record_stops_asking_for_a_dark_set(self, system: System) -> None:
        system.wait_for(lambda: newest_health(system).dark_due is False, "dark_due to clear")

    def test_resume_brings_back_the_survey_with_the_new_set(self, system: System) -> None:
        task = library(system).task
        assert task.finished_utc is not None
        finished_ns = iso_to_utc_ns(task.finished_utc)
        system.uncover()
        assert send(system, Resume()).accepted
        system.wait_for(lambda: scheduler_state(system) == "auto", "the scheduler to run again")

        def served() -> bool:
            return any(
                r.t_utc_ns > finished_ns and r.provenance.get("dark", "none") != "none"
                for r in system.records("sky_quality")
            )

        system.wait_for(served, "a survey frame that the new set serves")
        newest = max(
            (r for r in system.records("sky_quality") if r.t_utc_ns > finished_ns),
            key=lambda r: r.t_utc_ns,
        )
        assert "dark_due" not in newest.flags


class TestAfterARestartOfCore:
    def test_the_library_stays_and_the_session_is_forgotten(self, system: System) -> None:
        ensure_auto(system)
        (added,) = library(system).sets
        system.kill("core")
        system.restart("core")

        view = library(system)
        assert [item.name for item in view.sets] == [added.name]
        assert view.model is not None
        assert view.status.due is False
        assert view.task.state == "idle"  # the session belongs to the process that ran it


class TestADarkTaskThatFails:
    def test_a_camera_that_is_not_covered_fails_a_task_that_does_not_wait(
        self, system: System
    ) -> None:
        ensure_auto(system)
        system.uncover()
        before = [item.name for item in library(system).sets]

        assert send(system, QueueDark(wait_for_cover=False)).accepted
        task = watch(system, "the dark task to fail", finished)

        assert task.state == "failed"
        assert task.summary.startswith("Dark frame 1 of 3 is not dark:")
        assert [item.name for item in library(system).sets] == before
        system.wait_for(
            lambda: len(system.events("scheduler.dark_result")) == 2, "the result event of the task"
        )
        assert system.events("scheduler.dark_result")[-1].level == "warning"
        wait_for_pause(system)  # the owner looks at the camera first


class TestAStopInTheMiddle:
    def test_a_pause_during_the_dark_frames_aborts_the_task_and_adds_no_set(
        self, system: System
    ) -> None:
        ensure_auto(system)
        system.cover()
        before = [item.name for item in library(system).sets]

        assert send(system, QueueDark(wait_for_cover=False)).accepted
        watch(system, "the first dark frame", lambda t: t.phase == "dark" and t.step >= 1)
        assert send(system, Pause()).accepted
        task = watch(system, "the dark task to stop", finished)

        assert task.state == "aborted"
        assert "another command took the camera" in task.summary
        assert [item.name for item in library(system).sets] == before
        wait_for_pause(system)  # the pause that the owner pressed
        system.uncover()
        ensure_auto(system)


class TestTheCommandLine:
    def test_seeingmon_dark_queues_the_session_in_core_and_shows_it(self, system: System) -> None:
        ensure_auto(system)
        system.cover()  # the owner covered the camera before typing the command
        sets_before = len(library(system).sets)
        env = {**clean_environment(), "SEEINGMON_SERVICES__CONNECTION_KEY": system.plan.key}
        command = [
            sys.executable,
            "-m",
            "seeingmon",
            "dark",
            "--no-wait",
            "--address",
            system.plan.core_endpoint,
            "--local-config",
            str(system.directory / "no-owner-settings.toml"),
        ]
        done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=240.0)

        assert done.returncode == 0, done.stdout + done.stderr
        assert "Recording 3 dark frames of 10 s." in done.stdout  # a message that core reported
        assert ": ok. Added dark-" in done.stdout
        assert "The scheduler waits in pause. Uncover the camera" in done.stdout
        assert len(library(system).sets) == sets_before + 1
        wait_for_pause(system)
        system.uncover()
        ensure_auto(system)
