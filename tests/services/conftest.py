"""Fixtures of the services tests: addresses, keys, and a way to wait for conditions."""

from __future__ import annotations

import os
import secrets
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from seeingmon.services.ipc.endpoint import FAMILY_PIPE, FAMILY_UNIX, PIPE_PREFIX, Endpoint
from seeingmon.services.ipc.keys import ConnectionKey


def native_endpoint(directory: Path) -> Endpoint:
    """An endpoint of the platform's own family, unique to this call."""
    if os.name == "nt":
        return Endpoint(f"{PIPE_PREFIX}seeingmon-test-{uuid.uuid4().hex[:12]}", FAMILY_PIPE)
    return Endpoint(str(directory / "s.sock"), FAMILY_UNIX)


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
