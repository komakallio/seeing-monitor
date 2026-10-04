"""The RPC of `core`, called by the real client of the web lane: status, commands, live view."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("PIL.Image", reason="the preview needs Pillow")

from seeingmon.scheduler.commands import (
    Pause,
    QueueBurst,
    QueueReplay,
    QueueSweep,
    RejectReason,
    Resume,
    StartAlignment,
    StopAlignment,
)
from seeingmon.services.ipc.errors import RpcInvalidParamsError, RpcMethodNotFoundError
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.rpc import connect_rpc
from seeingmon.services.web.contract import AlignmentFrame, AlignmentState
from seeingmon.services.web.core_client import CoreUnavailableError, RpcCoreClient

from ..conftest import native_endpoint, wait_until
from .rig import CoreRig, build_rig, sky_frame

KEY = ConnectionKey.from_text("a-test-key-of-more-than-32-characters")


@pytest.fixture
def served(short_dir: Path, tmp_path: Path) -> Iterator[tuple[CoreRig, RpcCoreClient]]:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "known.ser").write_bytes(b"not a real recording")
    rig = build_rig(
        tmp_path,
        key=KEY,
        endpoint=native_endpoint(short_dir, "core"),
        config_extra=f'[replay]\nrecordings_dir = "{recordings.as_posix()}"\n',
    )
    rig.app.start()
    client = RpcCoreClient(rig.app.bound_endpoint, KEY, retry_interval_s=0.01)  # type: ignore[arg-type]
    yield rig, client
    client.close()
    rig.app.stop()


class TestTheMethods:
    def test_ping_names_the_process_instance(self, served: tuple[CoreRig, RpcCoreClient]) -> None:
        rig, client = served
        assert client.ping() == rig.app.instance

    def test_status_carries_the_scheduler_status_of_the_contract(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        rig.app.scheduler.step()
        status = client.status()
        assert status.instance == rig.app.instance
        assert status.scheduler.state in ("safe", "auto")
        assert status.scheduler.stream is None or status.scheduler.stream.purpose
        assert status.scheduler.counters["frames"] >= 0

    def test_status_says_what_the_scheduler_does(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        rig.app.scheduler.step()
        scheduler = client.status().scheduler
        activity = scheduler.activity
        assert activity is not None
        assert activity.state == scheduler.state
        assert activity.label
        assert activity.since_utc_ns <= scheduler.t_utc_ns

    def test_the_alignment_state_is_inactive_outside_alignment(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        _, client = served
        assert client.alignment_state() == AlignmentState(active=False)

    def test_the_focus_reset_reaches_the_helper_and_keeps_the_history(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        focus = rig.app.alignment._focus
        focus.add(1, 1, 2.0, 30)
        focus.add(2, 2, 2.2, 30)
        before = focus.snapshot().best_px
        client.alignment_reset_focus()
        after = focus.snapshot()
        assert before == 2.0
        assert after.best_px is None
        assert len(after.points) == 2

    def test_an_unknown_method_is_a_protocol_error_for_the_client(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, _ = served
        raw, _ = connect_rpc(rig.app.bound_endpoint, KEY, {"role": "cli"})  # type: ignore[arg-type]
        try:
            with pytest.raises(RpcMethodNotFoundError):
                raw.call("no_such_method")
        finally:
            raw.close()

    def test_the_wrong_key_is_refused_without_a_message_from_core(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, _ = served
        bad = RpcCoreClient(
            rig.app.bound_endpoint,  # type: ignore[arg-type]
            ConnectionKey.from_text("another-key-of-more-than-32-characters"),
            retry_interval_s=0.01,
        )
        with pytest.raises(CoreUnavailableError, match="refused the connection key"):
            bad.status()


class TestCommands:
    def test_a_pause_is_accepted_and_the_scheduler_follows(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        result = client.submit(Pause())
        assert result.accepted
        rig.app.scheduler.step()
        assert rig.app.scheduler.state.value == "paused"
        again = client.submit(Pause())
        assert not again.accepted  # a rejection is a normal answer, and not an error
        assert again.reason is RejectReason.ALREADY_PAUSED
        assert client.submit(Resume()).accepted

    def test_a_queued_task_gets_an_id(self, served: tuple[CoreRig, RpcCoreClient]) -> None:
        _, client = served
        result = client.submit(QueueBurst(duration_s=1.0, label="x"))
        assert result.accepted
        assert result.task_id == 1
        assert client.submit(QueueSweep(window_s=1.0)).task_id == 2

    def test_a_malformed_command_is_an_invalid_params_error(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, _ = served
        raw, _ = connect_rpc(rig.app.bound_endpoint, KEY, {"role": "cli"})  # type: ignore[arg-type]
        try:
            with pytest.raises(RpcInvalidParamsError):
                raw.call("submit", {"command": {"type": "reboot_the_machine"}})
        finally:
            raw.close()

    def test_a_replay_of_a_recording_that_exists_is_queued(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        _, client = served
        assert client.submit(QueueReplay(source="known.ser", speed=0)).accepted

    @pytest.mark.parametrize(
        ("command", "fragment"),
        [
            (QueueReplay(source="../known.ser"), "without a directory part"),
            (QueueReplay(source="missing.ser"), "no recording has that name"),
            (QueueReplay(source="known.ser", options={"path": "x"}), "'path'"),
        ],
    )
    def test_a_replay_that_cannot_run_is_refused_before_it_is_queued(
        self, served: tuple[CoreRig, RpcCoreClient], command: QueueReplay, fragment: str
    ) -> None:
        rig, client = served
        result = client.submit(command)
        assert not result.accepted
        assert result.reason is RejectReason.INVALID
        assert fragment in result.message
        assert rig.app.scheduler.status().queued_tasks == 0
        rig.app.scheduler.step()  # the scheduler flushes its events
        rig.app.tick()
        kinds = [e.kind for e in rig.events()]
        assert "core.command_refused" in kinds

    def test_the_results_method_lists_what_the_scheduler_finished(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        assert client.submit(QueueBurst(duration_s=1.0)).accepted
        rig.run_for(240.0)
        raw, _ = connect_rpc(rig.app.bound_endpoint, KEY, {"role": "cli"})  # type: ignore[arg-type]
        try:
            answer: Any = raw.call("results")
        finally:
            raw.close()
        assert answer["instance"] == rig.app.instance
        (burst,) = [r for r in answer["results"] if r["kind"] == "burst"]
        assert burst["task_id"] == 1
        assert burst["status"] == "ok"
        assert burst["pinned"] is True
        assert burst["artifacts"][0].startswith("bursts/")


class TestTheLiveView:
    def test_the_state_follows_the_alignment_of_the_scheduler(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        assert client.submit(StartAlignment(exposure_s=0.5)).accepted
        for _ in range(20):
            rig.app.scheduler.step()
            if rig.app.scheduler.state.value == "align":
                break
        assert rig.app.scheduler.state.value == "align"
        assert client.alignment_state().active is True
        assert client.alignment_state().quality["frame"]  # no frame yet
        assert client.submit(StopAlignment()).accepted

    def test_a_viewer_gets_the_frames_that_the_helper_encodes(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        assert client.submit(StartAlignment(exposure_s=0.5)).accepted
        for _ in range(20):
            rig.app.scheduler.step()
        frames: list[AlignmentFrame] = []

        async def watch() -> None:
            async for frame in client.alignment_frames():
                frames.append(frame)
                if len(frames) == 2:
                    return

        async def main() -> None:
            task = asyncio.create_task(watch())
            for seq in (1, 2, 3, 4):  # the helper encodes while the viewer waits
                await asyncio.to_thread(rig.app.alignment.process_frame, sky_frame(seq))
                await asyncio.sleep(0.2)
                if task.done():
                    break
            await asyncio.wait_for(task, 20.0)

        asyncio.run(main())
        assert len(frames) == 2
        assert frames[0].jpeg.startswith(b"\xff\xd8\xff")
        assert frames[0].state.frame is not None
        assert frames[0].state.frame.width_px == 640

        def viewer_left() -> bool:
            rig.app.alignment.housekeeping()  # reads the close of the connection
            return rig.app.alignment.viewers == 0

        assert wait_until(viewer_left)

    def test_the_web_component_is_ok_while_a_web_client_is_connected(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        client.ping()  # the client connects with the role `web`
        rig.app.tick()
        health = rig.records("health")[-1]
        assert health.components["web"] == "ok"  # type: ignore[attr-defined]
        client.close()
        assert wait_until(lambda: not rig.app.rpc.web_connected)
