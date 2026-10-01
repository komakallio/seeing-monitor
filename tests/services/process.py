"""Run `seeingmon acquire` as a real subprocess, for the end-to-end tests.

`AcquireProcess` starts `python -m seeingmon acquire` with the `fake` driver, a key and a clock
that come from the environment, and an address that is unique to the test. It waits until the
process answers, and it can kill, stop, and restart it. The log of the process goes to a file
that the test prints when it fails.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Any

from seeingmon.clock import DEFAULT_START_UTC_NS
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.errors import IpcError
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.rpc import connect_rpc
from seeingmon.services.remote import RemoteCameraDriver

from .conftest import native_endpoint

REPO_ROOT = Path(__file__).resolve().parents[2]

# Forbid pickle in the child. A call is recorded in the file that SMON_TEST_PICKLE_LOG names,
# even when a thread swallows the error.
NO_PICKLE_BOOTSTRAP = """\
import sys
from tests.services import no_pickle
no_pickle.install()
from seeingmon.cli import main
sys.exit(main(sys.argv[1:]))
"""


class AcquireProcess:
    """One `seeingmon acquire` subprocess with a stable address, so that it can restart."""

    def __init__(
        self,
        directory: Path,
        key_text: str,
        *,
        env: dict[str, str] | None = None,
        no_pickle: bool = False,
        speed: float = 1.0,
        name: str = "acquire",
        with_key: bool = True,
    ) -> None:
        self.directory = directory
        self.key_text = key_text
        self.key = ConnectionKey.from_text(key_text)
        self.endpoint: Endpoint = native_endpoint(directory, name)
        self.log_path = directory / f"{name}.log"
        self.pickle_log = directory / f"{name}.pickle.log"
        self._no_pickle = no_pickle
        self._env = self._build_env(env or {}, speed, with_key)
        self._log: IO[bytes] | None = None
        self.popen: subprocess.Popen[bytes] | None = None
        self.starts = 0
        self.drivers: list[RemoteCameraDriver] = []

    def _build_env(self, extra: dict[str, str], speed: float, with_key: bool) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("SEEINGMON_")}
        env.pop("CREDENTIALS_DIRECTORY", None)
        env.pop("NOTIFY_SOCKET", None)
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part
        )
        env["PYTHONUNBUFFERED"] = "1"
        env["SMON_TEST_PICKLE_LOG"] = str(self.pickle_log)
        origin_ns = time.time_ns()
        settings = {
            "ACQUIRE__DRIVER": "fake",
            "ACQUIRE__GAP_FACTOR": "1000",
            "ACQUIRE__RAISE_PRIORITY": "false",
            "ACQUIRE__WATCHDOG_TICK_S": "0.1",
            "CLOCK__KIND": "scaled",
            "CLOCK__SPEED": str(speed),
            "CLOCK__START_UTC_NS": str(DEFAULT_START_UTC_NS),
            "CLOCK__ORIGIN_REAL_NS": str(origin_ns),
            "HANDSHAKE_TIMEOUT_S": "3",
        }
        if with_key:
            settings["CONNECTION_KEY"] = self.key_text
        for name, value in settings.items():
            env[f"SEEINGMON_SERVICES__{name}"] = value
        for name, value in extra.items():
            env[f"SEEINGMON_SERVICES__{name}"] = value
        return env

    def _command(self) -> list[str]:
        arguments = [
            "acquire",
            "--address",
            str(self.endpoint),
            "--local-config",
            str(self.directory / "no-local-config.toml"),
        ]
        if self._no_pickle:
            return [sys.executable, "-c", NO_PICKLE_BOOTSTRAP, *arguments]
        return [sys.executable, "-m", "seeingmon", *arguments]

    # --- Running ---------------------------------------------------------------------------

    def start(self, *, wait: bool = True, timeout_s: float = 60.0) -> None:
        """Start the process, and by default wait until it answers."""
        self.close_log()
        self._log = self.log_path.open("ab")
        self._log.write(f"--- start {self.starts + 1} ---\n".encode())
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.popen = subprocess.Popen(
            self._command(),
            env=self._env,
            cwd=self.directory,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        self.starts += 1
        if wait:
            self.wait_ready(timeout_s)

    def wait_ready(self, timeout_s: float = 60.0) -> None:
        """Wait until the process answers a `ping`. Fails at once if the process exits."""
        assert self.popen is not None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            code = self.popen.poll()
            if code is not None:
                raise RuntimeError(f"acquire exited with code {code}:\n{self.log_text()}")
            try:
                client, _ = connect_rpc(
                    self.endpoint, self.key, connect_timeout_s=0.3, default_timeout_s=5.0
                )
            except IpcError:
                time.sleep(0.1)
                continue
            try:
                client.call("ping")
            finally:
                client.close()
            return
        raise RuntimeError(f"acquire did not answer within {timeout_s} s:\n{self.log_text()}")

    @property
    def running(self) -> bool:
        return self.popen is not None and self.popen.poll() is None

    def kill(self) -> None:
        """End the process at once, as a crash or `SIGKILL` does."""
        if self.popen is not None and self.popen.poll() is None:
            self.popen.kill()
        self.wait(10.0)

    def stop(self, timeout_s: float = 20.0) -> int:
        """Ask the process to stop, as systemd does, and return its exit code."""
        assert self.popen is not None
        if self.popen.poll() is None:
            if sys.platform == "win32":
                os.kill(self.popen.pid, signal.CTRL_BREAK_EVENT)
            else:
                self.popen.send_signal(signal.SIGTERM)
        return self.wait(timeout_s)

    def wait(self, timeout_s: float) -> int:
        assert self.popen is not None
        return self.popen.wait(timeout_s)

    def restart(self, *, hard: bool = True) -> None:
        """Kill the process (or stop it), and start it again on the same address."""
        if hard:
            self.kill()
        else:
            self.stop()
        self.start()

    # --- Clients ---------------------------------------------------------------------------

    def driver(self, **options: Any) -> RemoteCameraDriver:
        """A remote driver for this process. It closes when the process object closes."""
        options.setdefault("connect_timeout_s", 30.0)
        options.setdefault("rpc_timeout_s", 20.0)
        driver = RemoteCameraDriver(self.endpoint, self.key, **options)
        self.drivers.append(driver)
        return driver

    # --- The log ---------------------------------------------------------------------------

    def pickle_calls(self) -> list[str]:
        """The pickle calls that the child recorded. Empty means that none ran."""
        try:
            return self.pickle_log.read_text(encoding="utf-8").split()
        except OSError:
            return []

    def log_text(self) -> str:
        if self._log is not None:
            self._log.flush()
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(no log)"

    def close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def close(self) -> None:
        for driver in self.drivers:
            driver.close()
        if self.popen is not None and self.popen.poll() is None:
            self.popen.kill()
            self.popen.wait(10.0)
        print(self.log_text())  # shown by pytest when the test fails
        self.close_log()
