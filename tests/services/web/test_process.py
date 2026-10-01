"""The real process: `seeingmon web --demo` in a child, stopped with a signal.

The tests of the runner stop the server through `request_stop`. This one starts the command as a
person (or systemd) does, waits for the URL, makes a request, sends a termination signal, and checks
the exit code and the error output. A clean stop returns 0, and a signal never leaves a traceback.
"""

from __future__ import annotations

import json
import queue
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from typing import IO

import pytest

from tests.services.web.server import API, fetch

TIMEOUT_S = 90


def read_line(stream: IO[str], seconds: float) -> str | None:
    """Read one line without blocking forever. Pipes cannot be polled on Windows."""
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: lines.put(stream.readline()), daemon=True).start()
    try:
        return lines.get(timeout=seconds)
    except queue.Empty:
        return None


@pytest.fixture
def demo_process() -> Iterator[subprocess.Popen[str]]:
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP  # allows CTRL_BREAK_EVENT
    else:
        flags = 0
    process = subprocess.Popen(
        [sys.executable, "-m", "seeingmon", "web", "--demo", "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=flags,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=30)


def stop_signals() -> list[str]:
    return ["CTRL_BREAK_EVENT"] if sys.platform == "win32" else ["SIGTERM", "SIGINT"]


@pytest.mark.parametrize("name", stop_signals())
def test_the_demo_process_prints_its_url_serves_and_stops_cleanly_on_a_signal(
    demo_process: subprocess.Popen[str], name: str
) -> None:
    assert demo_process.stdout is not None
    url = read_line(demo_process.stdout, TIMEOUT_S)
    assert url is not None, "the demo printed no URL"
    assert url.startswith("http://127.0.0.1:")
    assert url.endswith("/\n")
    port = int(url.removeprefix("http://127.0.0.1:").removesuffix("/\n"))
    status, _headers, body = fetch("127.0.0.1", port, f"{API}/status")
    assert status == 200
    assert json.loads(body)["demo"] is True
    demo_process.send_signal(getattr(signal, name))
    assert demo_process.wait(timeout=TIMEOUT_S) == 0
    assert demo_process.stdout.read() == ""  # the URL was the only line
    assert demo_process.stderr is not None
    errors = demo_process.stderr.read()
    assert "Traceback" not in errors
    assert "Error" not in errors
