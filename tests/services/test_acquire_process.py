"""`seeingmon acquire` as a real subprocess, and a `RemoteCameraDriver` that streams from it.

These tests start `python -m seeingmon acquire` with the fake driver and a scaled clock, and
they connect over the real address. They show what no in-process test can: that a process that
dies leaves the client with a clear error, that a restarted process starts clean, and that every
frame is accounted for across the restart.
"""

from __future__ import annotations

import itertools
import os
import secrets
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from seeingmon.drivers.base import (
    CameraConfigError,
    CameraDisconnectedError,
    CameraStateError,
)
from seeingmon.frames import Frame, FrameFlag, TimeQuality
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.remote import RemoteCameraDriver

from . import no_pickle
from .process import AcquireProcess
from .rig import FAST, SMALL

StartProcess = Callable[..., AcquireProcess]

SLOW_CONSUMER = {"ACQUIRE__QUEUE_DEPTH": "8", "STREAM_WINDOW_MESSAGES": "4"}


@pytest.fixture
def start_process(short_dir: Path) -> Iterator[StartProcess]:
    processes: list[AcquireProcess] = []
    names = itertools.count()

    def start(**options: Any) -> AcquireProcess:
        wait = options.pop("wait", True)
        key_text = options.pop("key_text", secrets.token_urlsafe(24))
        process = AcquireProcess(
            short_dir, key_text, name=options.pop("name", f"p{next(names)}"), **options
        )
        processes.append(process)
        process.start(wait=wait)
        return process

    yield start
    for process in processes:
        process.close()


def read_frames(driver: RemoteCameraDriver, count: int, pause_s: float = 0.0) -> list[Frame]:
    frames = []
    for _ in range(count):
        frames.append(driver.read_frame(15.0))
        if pause_s:
            time.sleep(pause_s)
    return frames


def read_until_disconnected(driver: RemoteCameraDriver) -> list[Frame]:
    """Read until the driver raises `CameraDisconnectedError`. Returns the frames read before."""
    frames: list[Frame] = []
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            frames.append(driver.read_frame(10.0))
        except CameraDisconnectedError:
            return frames
    raise AssertionError("the driver never reported the disconnect")


def read_forever(driver: RemoteCameraDriver) -> None:
    """Read frames until the driver raises. A driver that never raises ends the test by timeout."""
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        driver.read_frame(10.0)


def run_into_the_hang(driver: RemoteCameraDriver) -> None:
    """Make the calls that reach the hung driver call, whichever it is."""
    driver.configure(FAST)
    driver.start()
    driver.read_frame(30.0)


def assert_every_frame_is_accounted_for(frames: list[Frame]) -> None:
    """The sequence number is the delivered count before a frame plus the drops reported so far."""
    reported = 0
    for position, frame in enumerate(frames):
        reported += frame.dropped_before
        assert frame.seq == position + reported, (position, frame.seq, reported)


def test_frames_stream_from_a_real_process(start_process: StartProcess) -> None:
    process = start_process()
    driver = process.driver()
    info = driver.open()
    assert info.driver == "fake"
    active = driver.configure(FAST)
    driver.start()
    frames = read_frames(driver, 40)
    assert [f.seq for f in frames] == list(range(40))
    assert all(f.stream_id == active.stream_id for f in frames)
    assert all(f.t_quality is TimeQuality.EXACT for f in frames)  # the fake stamps exactly
    assert all(f.data.shape == (128, 128) for f in frames)
    health = driver.health()
    assert health["state"] == "streaming"
    assert health["driver"] == "fake"
    assert health["client_connected"] is True
    driver.stop()
    driver.close()
    assert process.running


def test_a_slow_consumer_makes_the_queue_drop_and_the_drops_are_counted(
    start_process: StartProcess,
) -> None:
    process = start_process(env=SLOW_CONSUMER)
    driver = process.driver()
    driver.open()
    driver.configure(FAST)
    driver.start()
    frames = read_frames(driver, 60, pause_s=0.02)  # slower than the camera
    assert_every_frame_is_accounted_for(frames)
    reported = sum(f.dropped_before for f in frames)
    health = driver.health()
    assert reported > 0
    assert health["dropped_queue"] >= reported
    assert health["flow_stalls"] > 0
    assert health["queue_peak_frames"] <= 8


def test_kill_and_restart_recovers_and_accounts_for_every_frame(
    start_process: StartProcess,
) -> None:
    process = start_process(env=SLOW_CONSUMER)
    driver = process.driver()
    driver.open()
    first = driver.configure(FAST)
    driver.start()

    # Stream with a consumer that is slower than the camera, so that the queue drops frames.
    before = read_frames(driver, 50, pause_s=0.02)
    first_instance = driver.instance
    assert sum(f.dropped_before for f in before) > 0
    assert driver.health()["dropped_queue"] >= sum(f.dropped_before for f in before)

    # Kill the process while it streams, as a crash or a hang that its watchdog ended does.
    process.kill()
    in_flight = read_until_disconnected(driver)  # frames that arrived before the kill come first
    assert_every_frame_is_accounted_for(before + in_flight)
    with pytest.raises(CameraDisconnectedError):
        driver.configure(FAST)  # a restart is a disconnect, and nothing pretends otherwise
    with pytest.raises(CameraDisconnectedError):
        driver.read_frame(0.5)
    assert not driver.connected

    # Restart it. The caller reopens and reconfigures, and the stream starts clean.
    process.start()
    info = driver.open()
    assert info.driver == "fake"
    assert driver.instance != first_instance
    assert driver.connections == 2
    second = driver.configure(FAST)
    assert second.stream_id == first.stream_id  # a new process counts from the start again
    driver.start()
    after = read_frames(driver, 40)
    assert after[0].seq == 0  # nothing of the old process arrives
    assert_every_frame_is_accounted_for(after)
    assert driver.health()["state"] == "streaming"


def test_a_process_that_restarts_while_the_client_waits_for_a_frame(
    start_process: StartProcess,
) -> None:
    process = start_process()
    driver = process.driver()
    driver.open()
    driver.configure(SMALL)
    driver.start()
    read_frames(driver, 5)
    process.kill()
    with pytest.raises(CameraDisconnectedError):
        read_forever(driver)
    process.start()
    driver.open()
    driver.configure(SMALL)
    driver.start()
    assert driver.read_frame(15.0).seq == 0


def test_the_wrong_key_is_refused_and_the_process_keeps_serving(
    start_process: StartProcess,
) -> None:
    process = start_process()
    stranger = RemoteCameraDriver(
        process.endpoint, ConnectionKey.from_text(secrets.token_urlsafe(24)), connect_timeout_s=5.0
    )
    with pytest.raises(CameraConfigError, match="connection key"):
        stranger.open()
    good = process.driver()
    assert good.open().driver == "fake"
    good.configure(SMALL)
    good.start()
    assert good.read_frame(15.0).seq == 0


def test_a_process_without_a_key_stops_and_says_why(start_process: StartProcess) -> None:
    process = start_process(with_key=False, wait=False)
    assert process.wait(60.0) == 1
    assert "no connection key" in process.log_text()


def test_a_second_process_cannot_take_the_address(start_process: StartProcess) -> None:
    first = start_process()
    rival = AcquireProcess(first.directory, first.key_text, name="rival")
    rival.endpoint = first.endpoint
    try:
        rival.start(wait=False)
        assert rival.wait(60.0) == 1
        assert "cannot listen" in rival.log_text() or "already listens" in rival.log_text()
        assert first.driver().open().driver == "fake"  # the first process is not disturbed
    finally:
        rival.close()


@pytest.mark.parametrize("hang_in", ["configure", "read_frame"])
def test_a_hung_driver_call_ends_the_process_so_that_systemd_can_restart_it(
    start_process: StartProcess, hang_in: str
) -> None:
    process = start_process(
        env={
            "ACQUIRE__DRIVER_OPTIONS__HANG_IN": hang_in,
            "ACQUIRE__CALL_TIMEOUTS__CONFIGURE_S": "1",
            "ACQUIRE__CALL_TIMEOUTS__READ_GRACE_S": "0.5",
        }
    )
    driver = process.driver()
    driver.open()
    with pytest.raises(CameraDisconnectedError):
        run_into_the_hang(driver)
    code = process.wait(30.0)
    assert code == 70  # HANG_EXIT_CODE: the guard ended the process
    assert "ran for" in process.log_text()  # the guard's report names the call


def test_a_stop_signal_ends_the_process_cleanly(start_process: StartProcess) -> None:
    process = start_process()
    driver = process.driver()
    driver.open()
    driver.configure(SMALL)
    driver.start()
    read_frames(driver, 3)
    if sys.platform == "win32":
        try:
            code = process.stop(timeout_s=15.0)
        except (OSError, subprocess.TimeoutExpired):
            pytest.skip("this environment cannot send a console break to the process")
    else:
        code = process.stop()
    assert code == 0
    with pytest.raises(CameraDisconnectedError):
        read_forever(driver)
    if sys.platform != "win32":
        assert not os.path.exists(str(process.endpoint))  # the socket file is gone


def test_no_pickle_crosses_the_boundary_in_either_process(start_process: StartProcess) -> None:
    no_pickle.CALLS.clear()
    restore = no_pickle.install()
    try:
        process = start_process(no_pickle=True)
        driver = process.driver()
        driver.open()
        with pytest.raises(CameraStateError):
            driver.start()  # an error crosses the boundary as JSON
        driver.configure(SMALL)
        driver.start()
        frames = read_frames(driver, 20)
        assert [f.seq for f in frames] == list(range(20))
        assert driver.health()["state"] == "streaming"
        driver.stop()
        driver.close()
        process.restart()  # a reconnect also runs the handshake again
        again = process.driver()
        again.open()
        again.configure(SMALL)
        again.start()
        assert again.read_frame(15.0).seq == 0
        assert process.running
        assert process.pickle_calls() == []
        assert "pickle must not cross" not in process.log_text()
    finally:
        restore()
    assert no_pickle.CALLS == []


@pytest.mark.slow
def test_the_simulated_camera_streams_through_a_real_process(start_process: StartProcess) -> None:
    """The simulator needs a few seconds to import and to build its turbulence, so this is slow."""
    process = start_process(env={"ACQUIRE__DRIVER": "sim"})
    driver = process.driver()
    info = driver.open()
    assert (info.driver, info.model) == ("sim", "Simulated ZWO ASI294MM")
    active = driver.configure(FAST)
    driver.start()
    frames = read_frames(driver, 60)
    assert all(f.stream_id == active.stream_id for f in frames)
    assert all(f.t_quality is TimeQuality.EXACT for f in frames)  # the simulator knows the truth
    assert all(f.flags & FrameFlag.SIMULATED for f in frames)
    assert frames[0].data.shape == (128, 128)
    assert frames[-1].t_utc_ns > frames[0].t_utc_ns
    assert driver.health()["driver"] == "sim"
