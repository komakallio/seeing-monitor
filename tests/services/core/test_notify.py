"""The messages of `core` to systemd: ready, heartbeat, status, and stopping.

A test gives `core` a notifier that hands each datagram to a list, so no socket is involved and the
tests run on every platform. The datagrams are the ones that `sd_notify(3)` defines.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.clock import ClockStatus, ScaledClock
from seeingmon.frames import Frame
from seeingmon.scheduler.commands import Pause
from seeingmon.services.ipc.errors import IpcError
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.notify import SystemdNotifier
from seeingmon.services.web.core_client import RpcCoreClient
from seeingmon.testing import FakeCameraDriver

from ..conftest import wait_until
from .rig import NIGHT, SMALL_BIN2, CoreRig, build_rig

KEY = ConnectionKey.from_text("a-test-key-of-more-than-32-characters")
HEARTBEAT_EVERY_10_S = 20_000_000  # WATCHDOG_USEC: systemd advises a heartbeat at half of it


class Notes:
    """What systemd would hear. `on_ready` runs inside the datagram that says `READY=1`."""

    def __init__(self, watchdog_usec: int | None = None) -> None:
        self.messages: list[str] = []
        self.on_ready: Callable[[], None] | None = None
        self._lock = threading.Lock()
        env = {} if watchdog_usec is None else {"WATCHDOG_USEC": str(watchdog_usec)}
        self.notifier = SystemdNotifier(env=env, send=self._send)

    def _send(self, payload: bytes) -> None:
        text = payload.decode("utf-8")
        with self._lock:
            self.messages.append(text)
        if "READY=1" in text.split("\n") and self.on_ready is not None:
            self.on_ready()

    def lines(self) -> list[str]:
        with self._lock:
            return [line for message in self.messages for line in message.split("\n")]

    def count(self, line: str) -> int:
        return self.lines().count(line)

    def statuses(self) -> list[str]:
        return [line.removeprefix("STATUS=") for line in self.lines() if line.startswith("STATUS=")]


def rig_with(tmp_path: Path, notes: Notes, extra: str = "", **options: Any) -> CoreRig:
    parts = {"notifier": notes.notifier, **options.pop("parts", {})}
    return build_rig(tmp_path, key=KEY, parts=parts, config_extra=extra, **options)


class TestReady:
    def test_ready_comes_when_the_rpc_serves_and_the_store_is_open(self, tmp_path: Path) -> None:
        notes = Notes()
        rig = rig_with(tmp_path, notes)
        seen: dict[str, Any] = {}

        def when_ready() -> None:
            client = RpcCoreClient(rig.app.bound_endpoint, KEY, retry_interval_s=0.01)  # type: ignore[arg-type]
            try:
                seen["instance"] = client.ping()  # the RPC answers already
            finally:
                client.close()
            seen["events"] = [e.kind for e in rig.events()]  # the store takes writes

        notes.on_ready = when_ready
        assert notes.lines() == []  # nothing is sent while core is built
        rig.app.start()
        try:
            assert notes.lines()[0] == "READY=1"
            assert notes.statuses()[0].startswith("listening")
            assert seen == {"instance": rig.app.instance, "events": ["core.started"]}
        finally:
            rig.app.stop()

    def test_a_start_that_fails_sends_no_ready(self, tmp_path: Path) -> None:
        first_notes, second_notes = Notes(), Notes()
        (tmp_path / "one").mkdir()
        (tmp_path / "two").mkdir()
        first = rig_with(tmp_path / "one", first_notes)
        first.app.start()
        try:
            second = rig_with(
                tmp_path / "two", second_notes, endpoint=first.app.bound_endpoint
            )  # the same address: the second core cannot listen
            with pytest.raises(IpcError):
                second.app.start()
            second.app.stop()
            assert "READY=1" not in second_notes.lines()
            assert "READY=1" in first_notes.lines()
        finally:
            first.app.stop()


class TestHeartbeat:
    def test_there_is_no_heartbeat_unless_systemd_asks_for_one(self, tmp_path: Path) -> None:
        notes = Notes()  # no WATCHDOG_USEC
        rig = rig_with(tmp_path, notes)
        rig.app.start()
        try:
            rig.run_for(60.0)
            assert notes.count("WATCHDOG=1") == 0
        finally:
            rig.app.stop()

    def test_the_heartbeat_comes_at_half_of_the_watchdog_interval(self, tmp_path: Path) -> None:
        notes = Notes(watchdog_usec=60_000_000)  # WatchdogSec=60s, so a heartbeat every 30 s
        rig = rig_with(tmp_path, notes)
        rig.app.start()
        try:
            rig.run_for(100.0)
            # At 0, about 30, about 60, and about 90 s. A scheduler step can take seconds of
            # virtual time (a survey exposure), and the tick that follows is late by that much.
            assert 3 <= notes.count("WATCHDOG=1") <= 4
        finally:
            rig.app.stop()

    def test_a_stuck_scheduler_gets_no_heartbeat_and_a_recovered_one_does(
        self, tmp_path: Path
    ) -> None:
        notes = Notes(watchdog_usec=HEARTBEAT_EVERY_10_S)
        rig = rig_with(tmp_path, notes, "[services.core]\nscheduler_stall_s = 30.0\n")
        rig.app.start()
        try:
            rig.run_for(5.0)
            before = notes.count("WATCHDOG=1")
            assert before >= 1
            rig.clock.advance(100.0)  # the scheduler thread makes no step in all that time
            rig.app.tick()
            assert notes.count("WATCHDOG=1") == before
            rig.run_for(15.0)  # it works again
            assert notes.count("WATCHDOG=1") > before
        finally:
            rig.app.stop()

    def test_a_scheduler_in_a_long_driver_call_still_counts_as_alive(self, tmp_path: Path) -> None:
        notes = Notes(watchdog_usec=HEARTBEAT_EVERY_10_S)
        rig = rig_with(tmp_path, notes, "[services.core]\nscheduler_stall_s = 30.0\n")
        rig.app.start()
        try:
            rig.run_for(5.0)
            rig.app.liveness.expect(200.0)  # what `InfoDriver` does when a call begins
            before = notes.count("WATCHDOG=1")
            rig.clock.advance(100.0)
            rig.app.tick()
            assert notes.count("WATCHDOG=1") == before + 1  # within the bound of the call
            rig.clock.advance(150.0)  # the call outlasts its bound
            rig.app.tick()
            assert notes.count("WATCHDOG=1") == before + 1
        finally:
            rig.app.stop()


class TestStatus:
    def test_the_status_goes_out_when_it_changes_and_not_otherwise(self, tmp_path: Path) -> None:
        notes = Notes()
        rig = rig_with(tmp_path, notes)
        rig.app.start()
        try:
            rig.run_for(30.0)
            statuses = notes.statuses()
            assert statuses[0].startswith("listening")  # the message of READY=1
            assert statuses[-1] == rig.app.status_text()
            assert all(a != b for a, b in pairwise(statuses))
            sent = len(statuses)
            rig.run_for(30.0)  # nothing changes
            assert len(notes.statuses()) == sent
        finally:
            rig.app.stop()

    def test_the_status_names_what_is_wrong(self, tmp_path: Path) -> None:
        notes = Notes()
        rig = rig_with(tmp_path, notes)
        rig.app.start()
        try:
            rig.run_for(5.0)
            rig.app.scheduler.submit(Pause())
            rig.run_for(3.0)
            assert notes.statuses()[-1] == "paused"
            rig.remote.connected = False
            rig.run_for(3.0)
            assert notes.statuses()[-1] == "paused, acquire is not connected"
            rig.clock.set_status(ClockStatus(False, 86_400_000_000_000, "test"))
            rig.run_for(3.0)
            assert notes.statuses()[-1] == (
                "paused, acquire is not connected, the clock is not synchronized"
            )
        finally:
            rig.app.stop()


class TestStopping:
    def test_stopping_is_the_last_message_and_it_comes_once(self, tmp_path: Path) -> None:
        notes = Notes(watchdog_usec=HEARTBEAT_EVERY_10_S)
        rig = rig_with(tmp_path, notes)
        rig.app.start()
        rig.run_for(5.0)
        rig.app.stop("a test")
        rig.app.stop()
        assert notes.lines()[-1] == "STOPPING=1"
        assert notes.count("STOPPING=1") == 1


class HoldingCamera(FakeCameraDriver):
    """A fake camera whose reads block while `hold` is set: the scheduler thread hangs inside."""

    def __init__(self, clock: ScaledClock) -> None:
        super().__init__(clock, full_frames={"bin1": (8288, 5644), "bin2": SMALL_BIN2})
        self.hold = threading.Event()
        self.holding = threading.Event()
        self.release = threading.Event()

    def read_frame(self, timeout_s: float) -> Frame:
        if self.hold.is_set():
            self.holding.set()
            self.release.wait(120.0)
        return super().read_frame(timeout_s)


class TestTheThreadsOfTheWatchdog:
    def test_the_heartbeat_stops_while_the_scheduler_thread_hangs_and_resumes_after(
        self, tmp_path: Path
    ) -> None:
        clock = ScaledClock(start_utc_ns=NIGHT, origin_real_ns=time.time_ns(), speed=10.0)
        camera = HoldingCamera(clock)
        notes = Notes(watchdog_usec=2_000_000)  # a heartbeat each second of clock time
        rig = rig_with(
            tmp_path,
            notes,
            "[services.core]\nscheduler_stall_s = 3.0\ndriver_call_limit_s = 3.0\n",
            threads=True,
            clock=clock,
            parts={"driver": camera},
        )
        rig.app.start()
        try:
            assert wait_until(lambda: notes.count("WATCHDOG=1") >= 3, 30.0)
            lines = notes.lines()
            # A status line may come before READY=1 when the threads run late, so the test checks
            # that the service is ready before its first heartbeat, and not that READY=1 comes
            # first.
            assert "READY=1" in lines
            assert lines.index("READY=1") < lines.index("WATCHDOG=1")
            camera.hold.set()  # the next read never returns
            assert wait_until(camera.holding.is_set, 30.0)
            assert wait_until(lambda: not rig.app.scheduler_alive(), 30.0)
            time.sleep(0.3)  # a heartbeat that was in flight lands
            stuck = notes.count("WATCHDOG=1")
            time.sleep(0.6)  # more than five more intervals pass, and the other threads run on
            assert notes.count("WATCHDOG=1") == stuck
            camera.release.set()
            assert wait_until(lambda: notes.count("WATCHDOG=1") > stuck, 30.0)
        finally:
            camera.release.set()
            rig.app.stop()
        assert notes.lines()[-1] == "STOPPING=1"
