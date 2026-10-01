"""Fixtures of the services tests: addresses, keys, and a way to wait for conditions."""

from __future__ import annotations

import os
import secrets
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from seeingmon.services.ipc.endpoint import FAMILY_PIPE, FAMILY_UNIX, PIPE_PREFIX, Endpoint
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.server import ChannelHandler, IpcServer
from seeingmon.services.ipc.wire import Wire


def native_endpoint(directory: Path, name: str = "s") -> Endpoint:
    """An endpoint of the platform's own family. Pass another `name` for a second one."""
    if os.name == "nt":
        return Endpoint(f"{PIPE_PREFIX}seeingmon-test-{uuid.uuid4().hex[:12]}", FAMILY_PIPE)
    return Endpoint(str(directory / f"{name}.sock"), FAMILY_UNIX)


@pytest.fixture
def short_dir() -> Iterator[Path]:
    """A directory with a short path, because a Unix socket path holds at most 107 bytes."""
    path = Path(tempfile.mkdtemp(prefix="smon-"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(params=["native", "loopback"])
def endpoint(request: pytest.FixtureRequest, short_dir: Path) -> Endpoint:
    """A fresh endpoint. Each test runs on the native family and on the loopback socket path."""
    if request.param == "loopback":
        return Endpoint.loopback(0)
    return native_endpoint(short_dir)


@pytest.fixture
def native(short_dir: Path) -> Endpoint:
    """A fresh endpoint of the native family only."""
    return native_endpoint(short_dir)


@pytest.fixture
def key() -> ConnectionKey:
    """A random connection key, different for every test."""
    return ConnectionKey.from_text(secrets.token_urlsafe(24))


@pytest.fixture
def other_key() -> ConnectionKey:
    """A second random key, for the tests that use the wrong one."""
    return ConnectionKey.from_text(secrets.token_urlsafe(24))


def wait_until(
    condition: Callable[[], bool], timeout_s: float = 10.0, interval_s: float = 0.01
) -> bool:
    """Poll until the condition holds or the time runs out. Returns the last result."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval_s)
    return condition()


@pytest.fixture
def servers() -> Iterator[list[IpcServer]]:
    """The servers that a test started. They stop when the test ends."""
    started: list[IpcServer] = []
    yield started
    for server in started:
        server.stop()


@pytest.fixture
def start_server(
    endpoint: Endpoint, key: ConnectionKey, servers: list[IpcServer]
) -> Callable[..., IpcServer]:
    """Start an `IpcServer` with the given channels at this test's endpoint."""

    def start(channels: Mapping[str, ChannelHandler] | None = None, **options: Any) -> IpcServer:
        server = IpcServer(
            endpoint,
            key,
            channels or {},
            handshake_timeout_s=options.pop("handshake_timeout_s", 2.0),
            **options,
        )
        server.start()
        servers.append(server)
        return server

    return start


def drain(wire: Wire) -> None:
    """Receive until the wire raises, which ends the test that expects it. Never returns."""
    while True:
        wire.recv(0.2)
