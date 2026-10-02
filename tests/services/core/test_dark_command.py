"""`seeingmon dark` through core: it queues the session, shows the progress, and prints the result.

A command that goes through `core` connects to a running `CoreApp` over the real IPC layer, as in
`test_commands`. A thread steps the scheduler on a virtual clock, and the camera is a simulated one
that is covered, or not. The progress display (`CoreCommandClient.follow_dark`) has its own tests
against a scripted RPC, because a session on a virtual clock ends before the first look.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("scipy", reason="the fast path needs the fast extra")

from seeingmon.cli import main
from seeingmon.clock import Clock, VirtualClock
from seeingmon.drivers.base import CameraDriver
from seeingmon.profile import Profile
from seeingmon.services.core.commissioning import client as client_module
from seeingmon.services.core.commissioning.client import CoreCommandClient, CoreCommandError
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.errors import IpcClosedError
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.simsky import write_small_profile
from seeingmon.services.web.contract import (
    METHOD_DARK_LIBRARY,
    DarkLibraryView,
    DarkStatusView,
    DarkTaskView,
)
from tests.services.addresses import unique_address

from ..conftest import wait_until
from .rig import CoreRig, build_rig, local_config_text
from .test_app_dark import DARK_SETTINGS, covered_sim
from .test_commands import KEY_TEXT, KEY_VARIABLE, Served, Stepper


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("SEEINGMON_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    monkeypatch.setenv(KEY_VARIABLE, KEY_TEXT)


def serve(
    tmp_path: Path, factory: Callable[[Clock, Profile], CameraDriver] = covered_sim
) -> tuple[Served, Callable[[], None]]:
    rig: CoreRig = build_rig(
        tmp_path,
        key=ConnectionKey.from_text(KEY_TEXT),
        profile=write_small_profile(tmp_path),
        config_extra=DARK_SETTINGS,
        driver_factory=factory,
    )
    rig.app.start()
    stepper = Stepper(rig)
    stepper.start()
    served = Served(
        rig, stepper, str(rig.app.bound_endpoint), tmp_path / "local.toml", tmp_path / "data"
    )

    def stop() -> None:
        stepper.stop_event.set()
        stepper.join(30.0)
        rig.app.stop()
        assert stepper.error is None

    return served, stop


@pytest.fixture
def served(tmp_path: Path) -> Iterator[Served]:
    built, stop = serve(tmp_path)
    yield built
    stop()


def uncovered_sim(clock: Clock, profile: Profile) -> CameraDriver:
    from seeingmon.drivers.sim import SimOptions
    from tests.survey import simfx

    return simfx.make_driver(profile, SimOptions(seed=5), clock)


def run(
    capsys: pytest.CaptureFixture[str], served: Served, *arguments: str
) -> tuple[int, str, str]:
    code = main(
        ["dark", *arguments, "--address", served.address, "--local-config", str(served.config)]
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def sets(served: Served) -> list[str]:
    return [item.name for item in served.rig.app.dark_library.sets()]


class TestThroughCore:
    def test_the_session_runs_in_core_and_the_result_is_printed(
        self, served: Served, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = run(capsys, served, "--no-wait")
        assert code == 0, err
        assert "following task 1" in out
        assert "dark 1: ok. Added dark-" in out
        assert "file: calibration/darks/dark-" in out.replace("\\", "/")
        assert "The scheduler waits in pause. Uncover the camera" in out
        assert len(sets(served)) == 1
        # The result is there a moment before the scheduler pauses.
        assert wait_until(lambda: served.rig.app.scheduler.status().state == "paused", 5.0)

    def test_the_numbers_of_the_command_line_reach_the_session(
        self, served: Served, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = run(
            capsys, served, "--no-wait", "--frames", "4", "--bias-frames", "3", "--exposure-s", "10"
        )
        assert code == 0, err
        (item,) = served.rig.app.dark_library.sets()
        assert (item.n_frames, item.n_bias_frames, item.exposure_s) == (4, 3, 10.0)

    def test_a_camera_that_is_not_covered_fails_the_command_with_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        served, stop = serve(tmp_path, uncovered_sim)
        try:
            code, out, _ = run(capsys, served, "--no-wait")
        finally:
            stop()
        assert code == 1
        assert "dark 1: failed. Dark frame 1 of 3 is not dark:" in out
        assert "The scheduler waits in pause" in out  # the camera may still be covered
        assert sets(served) == []

    def test_numbers_that_the_scheduler_refuses_exit_with_two(
        self, served: Served, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = run(capsys, served, "--frames", "2")
        assert code == 2
        assert "core rejected the command" in out
        assert sets(served) == []

    def test_detach_queues_the_session_and_returns(
        self, served: Served, capsys: pytest.CaptureFixture[str]
    ) -> None:
        served.stepper.held.set()  # the scheduler stands still, so the task stays queued
        time.sleep(0.05)  # a step in progress ends
        try:
            code, out, err = run(capsys, served, "--detach")
            assert (code, err) == (0, "")
            assert "following" not in out
            assert served.rig.app.dark_state.snapshot().state == "queued"
            # A second session is refused while the first one waits.
            code, out, _ = run(capsys, served)
            assert code == 2
            assert "core rejected the command" in out
        finally:
            served.stepper.held.clear()

    def test_ctrl_c_stops_the_display_and_leaves_the_session_in_core(
        self, served: Served, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def interrupted(*args: Any, **kwargs: Any) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(CoreCommandClient, "follow_dark", interrupted)
        code, out, _ = run(capsys, served, "--no-wait")
        assert code == 1
        assert "stopped following; the session goes on in core" in out


class TestTheOptionsOfEachRun:
    @pytest.mark.parametrize(
        "option",
        [["--mode", "bin2"], ["--gain", "120"], ["--driver", "sim"], ["--wait-timeout", "5"]],
    )
    def test_an_option_that_core_decides_needs_standalone(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], option: list[str]
    ) -> None:
        config = tmp_path / "local.toml"
        config.write_text(local_config_text(tmp_path / "data"), encoding="utf-8")
        assert main(["dark", *option, "--local-config", str(config)]) == 2
        error = capsys.readouterr().err
        assert f"{option[0]} needs --standalone" in error

    def test_the_library_option_needs_standalone_too(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = tmp_path / "local.toml"
        config.write_text(local_config_text(tmp_path / "data"), encoding="utf-8")
        assert main(["dark", "--library", str(tmp_path), "--local-config", str(config)]) == 2
        assert "--library needs --standalone" in capsys.readouterr().err

    def test_the_options_of_the_run_through_core_do_not_combine_with_standalone(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["dark", "--standalone", "--detach"]) == 2
        assert "belong to the run through core" in capsys.readouterr().err

    def test_nobody_listens_so_the_message_says_what_to_do(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = tmp_path / "local.toml"
        config.write_text(local_config_text(tmp_path / "data"), encoding="utf-8")
        code = main(["dark", "--address", unique_address("core"), "--local-config", str(config)])
        error = capsys.readouterr().err
        assert code == 1
        assert "cannot reach core" in error
        assert "--standalone" in error


# --- The progress display ----------------------------------------------------------------------


def view(task: DarkTaskView) -> dict[str, Any]:
    library = DarkLibraryView(
        mode="bin2",
        gain=120,
        exposure_s=30.0,
        status=DarkStatusView(due=True, reason="none", tolerance_c=3.0, max_age_days=183.0),
        task=task,
    )
    return library.model_dump(mode="json")


class ScriptedRpc:
    """The RPC client of `core` as a script: one view of the task for each look, then the result."""

    def __init__(self, views: list[DarkTaskView], result: Mapping[str, Any] | None) -> None:
        self.views = views
        self.result = result
        self.looks = 0
        self.fail = False

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        if self.fail:
            raise IpcClosedError("core went away")
        if method == METHOD_DARK_LIBRARY:
            task = self.views[min(self.looks, len(self.views) - 1)]
            self.looks += 1
            return view(task)
        assert method == "results"
        done = self.result is not None and self.looks >= len(self.views)
        return {"results": [self.result] if done else []}

    def close(self, reason: str = "") -> None:
        return None


def scripted_client(
    monkeypatch: pytest.MonkeyPatch, rpc: ScriptedRpc, clock: VirtualClock
) -> CoreCommandClient:
    monkeypatch.setattr(client_module, "connect_rpc", lambda *args, **kwargs: (rpc, {}))
    key = ConnectionKey.from_text(KEY_TEXT)
    return CoreCommandClient(Endpoint.loopback(0), key, clock=clock)


def running(task_id: int, message: str, phase: str = "bias") -> DarkTaskView:
    return DarkTaskView(state="running", task_id=task_id, phase=phase, message=message)


class TestFollowing:
    def test_each_new_message_of_the_task_shows_once_and_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        views = [
            DarkTaskView(state="queued", task_id=7, message="The session waits."),
            running(7, "Taking 3 bias frames."),
            running(7, "Taking 3 bias frames."),  # the same message again
            running(6, "Another task.", "dark"),  # not this task
            DarkTaskView(state="running", task_id=7),  # no message
            running(7, "Dark frame 1 of 3.", "dark"),
            DarkTaskView(state="ok", task_id=7),
        ]
        result = {"task_id": 7, "kind": "dark", "status": "ok"}
        clock = VirtualClock()
        client = scripted_client(monkeypatch, ScriptedRpc(views, result), clock)
        shown: list[str] = []
        outcome = client.follow_dark(7, show=shown.append, poll_s=2.0)
        assert shown == ["The session waits.", "Taking 3 bias frames.", "Dark frame 1 of 3."]
        assert outcome.result == result
        assert outcome.waited_s == pytest.approx(2.0 * (len(views) - 1))

    def test_a_result_that_is_already_there_ends_the_wait_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = {"task_id": 3, "kind": "dark", "status": "failed"}
        rpc = ScriptedRpc([DarkTaskView(state="failed", task_id=3)], result)
        client = scripted_client(monkeypatch, rpc, VirtualClock())
        outcome = client.follow_dark(3, show=lambda line: None)
        assert outcome.result == result
        assert outcome.waited_s == 0.0

    def test_the_wait_gives_up_after_the_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rpc = ScriptedRpc([running(4, "Dark frame 1 of 3.", "dark")], None)
        client = scripted_client(monkeypatch, rpc, VirtualClock())
        outcome = client.follow_dark(4, show=lambda line: None, timeout_s=5.0, poll_s=2.0)
        assert outcome.result is None
        assert outcome.waited_s >= 5.0

    def test_a_connection_that_goes_away_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rpc = ScriptedRpc([running(4, "Dark frame 1 of 3.", "dark")], None)
        client = scripted_client(monkeypatch, rpc, VirtualClock())
        rpc.fail = True
        with pytest.raises(CoreCommandError, match="did not give the dark library"):
            client.follow_dark(4, show=lambda line: None)

    def test_the_library_view_comes_back_as_a_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rpc = ScriptedRpc([running(1, "Taking 3 bias frames.")], None)
        client = scripted_client(monkeypatch, rpc, VirtualClock())
        library = client.dark_library()
        assert (library.mode, library.task.phase) == ("bin2", "bias")
