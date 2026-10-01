"""The remote power-cycle hook: the last step of the recovery ladder.

A ZWO camera on a Pi can stall until someone cuts its power, and the Pi cannot switch the power
of one USB port. The last resort is a hard power cycle of the whole Pi, through its PoE switch port
or a smart plug on its injector. The route is undecided, so `PowerCycle` runs whatever the
configuration says:

- `command`: an argument list that the system runs without a shell (`argv`). Write the list, never a
  shell string, so a value cannot inject a command.
- `http`: a request with a method, a URL, headers, and a body.

**No secrets in the repository.** The address, the URL, and any token live in the local
configuration or in the environment. Any `${NAME}` in an argument, the URL, a header value, or the
body expands from the environment when the cycle runs, so the configuration file can name a
variable instead of holding the secret. A variable that is not set fails the request before
anything is sent. The events and the results never carry a command, a URL, a header, or an output.
They carry the route, the outcome, and counts.

**Rate limits.** A runaway loop must not cycle the Pi again and again. `min_interval_s` sets the
shortest time between two attempts, and `max_per_day` caps the attempts in any 24 hours. A failed
attempt counts too. The cycle kills the process that asked for it, so the attempt is written to
`state_file` before the command runs, and the next process reads it. Without a `state_file`, the
limits live in memory only and a reboot forgets them, so set one in production. If the clock is not
synchronized (a Pi 4 has no real-time clock), the limits cannot be trusted, and the hook refuses
while an earlier attempt is on record.

**Dry run.** With `dry_run`, the hook checks the configuration and the environment, reports what it
would do, and sends nothing. A dry run never counts against the limits.

**Events.** The hook reports through `on_event` (see `seeingmon.hardware.events`):
`power.cycle_requested` before the attempt, then `power.cycle_done` or `power.cycle_failed`
(you may never see the first if the cycle works), `power.cycle_blocked`, `power.cycle_dry_run`, and
`power.cycle_unavailable`.

**Not verified.** The route is an open decision of the owner (blocker B4). The hook is tested with a
fake command runner, a local HTTP server, and a real subprocess.
"""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse

from pydantic import Field, model_validator

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.config import SectionModel
from seeingmon.hardware.events import EventCallback, HardwareEvent, emit

_log = logging.getLogger(__name__)

DAY_NS = 24 * 3600 * NS_PER_S
MAX_REASON_CHARS = 200
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PowerCycleError(Exception):
    """The request cannot run: a variable is missing, or the route is not configured."""


class PowerOutcome(StrEnum):
    DONE = "done"  # the command exited with 0, or the server answered with a success status
    FAILED = "failed"
    DRY_RUN = "dry_run"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"  # no route is configured


@dataclass(frozen=True, slots=True)
class PowerResult:
    """What `PowerCycle.request` did. `detail` names the cause and never holds a secret."""

    outcome: PowerOutcome
    detail: str = ""


class CommandRoute(SectionModel):
    """The `command` route: an argument list. The first item is the program."""

    argv: list[str] = Field(default_factory=list)


class HttpRoute(SectionModel):
    """The `http` route. `verify_tls` applies to an `https` URL."""

    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = None
    verify_tls: bool = True


class PowerConfig(SectionModel):
    """The `[power]` section. The hook does nothing unless you choose a route."""

    route: Literal["none", "command", "http"] = "none"
    dry_run: bool = False
    min_interval_s: float = Field(default=3600.0, ge=0.0)
    max_per_day: int = Field(default=3, ge=1)
    timeout_s: float = Field(default=30.0, gt=0.0)
    state_file: str | None = None
    command: CommandRoute = Field(default_factory=CommandRoute)
    http: HttpRoute = Field(default_factory=HttpRoute)

    @model_validator(mode="after")
    def _check_route(self) -> PowerConfig:
        if self.route == "command" and not self.command.argv:
            raise ValueError("the command route needs a non-empty argv")
        if self.route == "http":
            url = self.http.url
            if urlparse(url).scheme not in ("http", "https") and not url.startswith("${"):
                raise ValueError("the http route needs a url that starts with http:// or https://")
        return self


# --- Transports ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int | None  # `None` after a timeout
    timed_out: bool = False


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str], timeout_s: float) -> CommandResult:
        """Run the program with its arguments, without a shell."""
        ...


class SubprocessRunner:
    """Run a command with `subprocess`. It reads no output and passes no input."""

    def run(self, argv: Sequence[str], timeout_s: float) -> CommandResult:
        try:
            completed = subprocess.run(
                list(argv),
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(None, timed_out=True)
        return CommandResult(completed.returncode)


@dataclass(frozen=True, slots=True)
class HttpResult:
    status: int | None  # `None` when no response arrived
    error: str = ""  # the class of the failure, with no address


class HttpSender(Protocol):
    def send(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
        verify_tls: bool,
    ) -> HttpResult:
        """Send one request and return the status, or the failure class."""
        ...


class UrllibSender:
    """Send a request with `urllib`, which needs no dependency."""

    def send(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
        verify_tls: bool,
    ) -> HttpResult:
        request = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
        context = None
        if not verify_tls:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen(request, timeout=timeout_s, context=context) as response:
                return HttpResult(response.status)
        except urllib.error.HTTPError as error:
            return HttpResult(error.code)
        except (urllib.error.URLError, OSError, ValueError) as error:
            reason = error.reason if isinstance(error, urllib.error.URLError) else error
            return HttpResult(None, type(reason).__name__)


# --- The hook ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CommandAction:
    argv: list[str]


@dataclass(frozen=True, slots=True)
class _HttpAction:
    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None


def expand(text: str, env: Mapping[str, str]) -> str:
    """Replace each `${NAME}` with the environment variable of that name.

    Raises `PowerCycleError` for a variable that is not set. The message names the variable and
    never shows a value.
    """

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in env:
            raise PowerCycleError(f"the environment variable {name} is not set")
        return env[name]

    return _VARIABLE.sub(substitute, text)


class PowerCycle:
    """Request a power cycle through the configured route, within the rate limits.

    Args:
        config: The `[power]` section.
        clock: The time source for the limits.
        runner: Runs the command route. The default uses `subprocess`.
        sender: Sends the http route. The default uses `urllib`.
        on_event: Receives the events that the module documentation lists.
        env: The environment that `${NAME}` expands from. The default is `os.environ`.
    """

    def __init__(
        self,
        config: PowerConfig,
        *,
        clock: Clock,
        runner: CommandRunner | None = None,
        sender: HttpSender | None = None,
        on_event: EventCallback | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._cfg = config
        self._clock = clock
        self._runner = runner or SubprocessRunner()
        self._sender = sender or UrllibSender()
        self._on_event = on_event
        self._env = os.environ if env is None else env
        self._state_file = None if config.state_file is None else Path(config.state_file)
        self._lock = threading.Lock()  # two requests at once must not both pass the limits
        self._attempts: list[int] = self._load_attempts()

    # --- Events ---

    def _emit(self, level: str, kind: str, message: str, **detail: object) -> None:
        event = HardwareEvent(level, kind, message, self._clock.utc_ns(), detail or None)
        emit(self._on_event, event)

    # --- The attempt log ---

    def _load_attempts(self) -> list[int]:
        if self._state_file is None or not self._state_file.is_file():
            return []
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            values = data["attempts_utc_ns"]
            if not isinstance(values, list) or not all(
                isinstance(item, int) and not isinstance(item, bool) for item in values
            ):
                raise ValueError("not a list of integers")
            return sorted(values)
        except (OSError, ValueError, KeyError, TypeError):
            _log.warning("the power-cycle state file is unreadable, so the limits start empty")
            self._emit(
                "warning",
                "power.state_unreadable",
                "The power-cycle state file could not be read, so the limits start empty.",
            )
            return []

    def _save_attempts(self) -> None:
        if self._state_file is None:
            return
        temporary = self._state_file.with_name(self._state_file.name + ".tmp")
        try:
            temporary.write_text(json.dumps({"attempts_utc_ns": self._attempts}), encoding="utf-8")
            os.replace(temporary, self._state_file)
        except OSError:
            _log.warning("the power-cycle state file could not be written")
            self._emit(
                "warning",
                "power.state_unwritable",
                "The power-cycle state file could not be written, so a reboot forgets the limits.",
            )

    def attempts_today(self) -> int:
        """The attempts in the last 24 hours, by the clock."""
        now = self._clock.utc_ns()
        return sum(1 for moment in self._attempts if 0 <= now - moment < DAY_NS)

    def _blocked_reason(self, now_ns: int) -> str | None:
        """Why the limits forbid an attempt now, or `None` when they allow one."""
        recent = [moment for moment in self._attempts if now_ns - moment < DAY_NS]
        if self._clock.status().synchronized is False and self._attempts:
            return "clock_not_synchronized"
        if any(moment > now_ns for moment in self._attempts):
            return "clock_behind_the_last_attempt"  # the clock stepped back, so nothing is provable
        if recent and now_ns - max(recent) < round(self._cfg.min_interval_s * NS_PER_S):
            return "minimum_interval"
        if len(recent) >= self._cfg.max_per_day:
            return "daily_maximum"
        return None

    # --- The request ---

    def request(self, reason: str) -> PowerResult:
        """Ask for a power cycle. `reason` is a short, secret-free sentence for the event.

        The method returns a `PowerResult` and never raises for a failed route. With a working
        route, the cycle may kill the process before this method returns.
        """
        with self._lock:
            return self._request(reason[:MAX_REASON_CHARS])

    def _request(self, reason: str) -> PowerResult:
        route = self._cfg.route
        if route == "none":
            self._emit(
                "info",
                "power.cycle_unavailable",
                "No power-cycle route is configured.",
                reason=reason,
            )
            return PowerResult(PowerOutcome.UNAVAILABLE, "no route is configured")
        try:
            action = self._prepare()
        except PowerCycleError as error:
            self._emit(
                "error",
                "power.cycle_failed",
                "The power-cycle request is not valid.",
                route=route,
                error=str(error),
                reason=reason,
            )
            return PowerResult(PowerOutcome.FAILED, str(error))
        if self._cfg.dry_run:
            self._emit(
                "info",
                "power.cycle_dry_run",
                "A dry run: the power cycle would run now.",
                route=route,
                reason=reason,
            )
            return PowerResult(PowerOutcome.DRY_RUN)
        now = self._clock.utc_ns()
        blocked = self._blocked_reason(now)
        if blocked is not None:
            self._emit(
                "warning",
                "power.cycle_blocked",
                "A rate limit blocked the power cycle.",
                route=route,
                limit=blocked,
                reason=reason,
            )
            return PowerResult(PowerOutcome.RATE_LIMITED, blocked)
        self._attempts = [moment for moment in self._attempts if now - moment < DAY_NS]
        self._attempts.append(now)
        self._save_attempts()  # before the attempt: the cycle can kill this process
        self._emit(
            "warning",
            "power.cycle_requested",
            "The system requests a power cycle.",
            route=route,
            reason=reason,
            attempts_today=len(self._attempts),
        )
        return self._execute(action, route, reason)

    def _prepare(self) -> _CommandAction | _HttpAction:
        """Expand the configured route. Raises `PowerCycleError` for a missing variable or a URL
        that does not start with http:// or https:// after the expansion."""
        if self._cfg.route == "command":
            return _CommandAction([expand(item, self._env) for item in self._cfg.command.argv])
        http = self._cfg.http
        url = expand(http.url, self._env)
        if urlparse(url).scheme not in ("http", "https"):
            raise PowerCycleError("the url must start with http:// or https://")
        headers = {name: expand(value, self._env) for name, value in http.headers.items()}
        body = None if http.body is None else expand(http.body, self._env).encode("utf-8")
        return _HttpAction(http.method, url, headers, body)

    def _execute(
        self, action: _CommandAction | _HttpAction, route: str, reason: str
    ) -> PowerResult:
        try:
            if isinstance(action, _CommandAction):
                ran = self._runner.run(action.argv, self._cfg.timeout_s)
                ok = ran.returncode == 0
                cause = "timeout" if ran.timed_out else f"exit code {ran.returncode}"
                status = ran.returncode
            else:
                sent = self._sender.send(
                    action.method,
                    action.url,
                    action.headers,
                    action.body,
                    self._cfg.timeout_s,
                    self._cfg.http.verify_tls,
                )
                ok = sent.status is not None and 200 <= sent.status < 300
                cause = sent.error or f"status {sent.status}"
                status = sent.status
        except OSError as error:  # the program is missing, or it cannot start
            ok, cause, status = False, type(error).__name__, None
        if ok:
            self._emit(
                "warning",
                "power.cycle_done",
                "The power-cycle request succeeded.",
                route=route,
                status=status,
            )
            return PowerResult(PowerOutcome.DONE)
        self._emit(
            "error",
            "power.cycle_failed",
            "The power-cycle request failed.",
            route=route,
            cause=cause,
            reason=reason,
        )
        return PowerResult(PowerOutcome.FAILED, cause)
