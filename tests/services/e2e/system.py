"""The whole system as real processes: `acquire`, `core`, and `web` on a simulated sky.

`System` builds the plan of `seeingmon dev` (`seeingmon.services.dev.build_plan`) in a folder of the
test, with a scaled clock that all three processes share, and it starts, kills, and restarts the
children. A killed process is a hard kill (`Popen.kill`), as a crash or the out-of-memory killer
does it, and a restart starts the same command with the same settings, so the process takes the
same address and finds the same data folder and the same simulated timeline.

The tests read the results from the store of `core` through a read-only connection, they ask
`web` over HTTP on the loopback interface, as a browser does, and they call the RPC of `core`, as
`web` and the command line do.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from seeingmon.records import Record
from seeingmon.services.dev import (
    Child,
    DevOptions,
    DevPlan,
    build_plan,
    wait_for_web,
    wait_until_ready,
)
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.rpc import connect_rpc
from seeingmon.services.web.contract import RPC_CHANNEL
from seeingmon.store.db import StoreReader

from ..conftest import wait_until
from ..core.rig import read_all

READY_TIMEOUT_S = 120.0


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def clean_environment() -> dict[str, str]:
    """The environment of the test, without the settings of the person who runs it."""
    return {k: v for k, v in os.environ.items() if not k.startswith("SEEINGMON_")}


class System:
    """`acquire`, `core`, and `web` as children of the test."""

    def __init__(
        self, directory: Path, *, with_web: bool = True, owner_settings: str = "", **options: Any
    ) -> None:
        """`owner_settings` is the text of a local configuration, for a plan that reads one (the
        real sky takes its site and its survey settings from it). A plan of the simulated sky reads
        no owner settings, and the default is a file that does not exist."""
        self.directory = directory
        self.with_web = with_web
        local = directory / "no-owner-settings.toml"
        if owner_settings:
            local = directory / "owner-settings.toml"
            local.write_text(owner_settings, encoding="utf-8")
        self.plan: DevPlan = build_plan(
            DevOptions(port=free_port(), **options),
            directory=directory / "run",
            local_file=local,
            env=clean_environment(),
        )
        self.children: dict[str, Child] = {
            spec.name: Child(spec) for spec in self.plan.children if with_web or spec.name != "web"
        }
        self.starts: dict[str, int] = dict.fromkeys(self.children, 0)

    # --- Starting and stopping -------------------------------------------------------------

    def start(self, name: str) -> None:
        """Start one child and wait until it answers (web: until it accepts a connection)."""
        child = self.children[name]
        child.start()
        self.starts[name] += 1
        if name == "web":
            if not wait_for_web(self.plan, child, READY_TIMEOUT_S):
                raise AssertionError(f"web did not start:\n{child.log_tail(40)}")
        else:
            wait_until_ready(self.plan, self.children, READY_TIMEOUT_S, (name,))

    def start_all(self) -> None:
        for name in ("acquire", "core", "web"):  # core connects to acquire, so acquire comes first
            if name in self.children:
                self.start(name)

    def kill(self, name: str) -> None:
        """Kill a child the hard way, and wait until it is gone."""
        child = self.children[name]
        assert child.popen is not None
        child.popen.kill()
        child.popen.wait(30.0)
        child.stop()  # closes the log file of the dead child

    def restart(self, name: str) -> None:
        """Start a child again after `kill`, with the same command and settings."""
        spec = self.children[name].spec
        self.children[name] = Child(spec)
        self.start(name)

    def stop_all(self) -> None:
        for name in ("web", "core", "acquire"):
            if name in self.children:
                self.children[name].stop()

    def alive(self) -> dict[str, bool]:
        return {name: child.running for name, child in self.children.items()}

    def logs(self) -> str:
        """The tail of every log, for the message of a failed test."""
        return "\n".join(f"--- {n}\n{c.log_tail(25)}" for n, c in self.children.items())

    # --- The cover of the simulated camera -------------------------------------------------

    def cover(self) -> None:
        """Put the cover on the simulated camera, as the owner does with a hand on a real one."""
        assert self.plan.cover_file is not None
        self.plan.cover_file.touch()

    def uncover(self) -> None:
        assert self.plan.cover_file is not None
        self.plan.cover_file.unlink(missing_ok=True)

    # --- Looking at the results ------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        """The data folder of the run: the folder that the options name, or the one in the run."""
        return self.plan.options.data_dir or self.plan.directory / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db" / "results.sqlite"

    def records(self, record_type: str) -> list[Any]:
        """Every stored record of a type, read through a fresh read-only connection."""
        with StoreReader.open(self.db_path) as store:
            found: list[Record] = read_all(store, record_type)
        return list(found)

    def events(self, kind: str | None = None) -> list[Any]:
        return [e for e in self.records("event") if kind is None or e.kind == kind]

    def wait_for(
        self, condition: Callable[[], bool], what: str, *, timeout_s: float = 240.0
    ) -> None:
        """Poll until the condition holds. The message names what never happened, with the logs."""
        if not wait_until(condition, timeout_s, interval_s=0.5):
            raise AssertionError(f"timed out waiting for {what}\n{self.logs()}")

    # --- Asking core -----------------------------------------------------------------------

    def core_call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Call a method of the RPC of `core` on a fresh connection, and return its result."""
        client, _ = connect_rpc(
            Endpoint.parse(self.plan.core_endpoint),
            ConnectionKey.from_text(self.plan.key),
            {"role": "cli"},
            channel=RPC_CHANNEL,
        )
        try:
            return client.call(method, params)
        finally:
            client.close()

    # --- Asking web ------------------------------------------------------------------------

    def get(self, path: str, *, timeout_s: float = 10.0) -> Any:
        """GET `/api/v1/<path>` and decode the JSON. Raises `URLError` when web is not there."""
        url = f"http://127.0.0.1:{self.plan.port}/api/v1/{path}"
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        authorized: bool = True,
        timeout_s: float = 10.0,
    ) -> tuple[int, Any]:
        """POST JSON to `/api/v1/<path>`. Returns the HTTP status and the decoded JSON answer.

        The request carries the API token of the run, as the UI does, unless `authorized` is
        false. A status of 400 or more comes back as a value, and not as an exception.
        """
        url = f"http://127.0.0.1:{self.plan.port}/api/v1/{path}"
        headers = {"Content-Type": "application/json"}
        if authorized:
            assert self.plan.token is not None, "the run has no generated API token"
            headers["Authorization"] = f"Bearer {self.plan.token}"
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return int(response.status), json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return int(error.code), json.loads(error.read().decode("utf-8") or "null")

    def get_or_none(self, path: str) -> Any:
        try:
            return self.get(path)
        except (urllib.error.URLError, OSError, ValueError):
            return None
