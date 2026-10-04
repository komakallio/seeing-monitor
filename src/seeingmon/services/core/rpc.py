"""What `core` serves: the RPC methods and the live-view stream.

`core` listens at one address and serves two channels (see `seeingmon.services.web.contract`, which
`web` owns and `core` implements):

- **`rpc`**, an `RpcService` with the methods `ping`, `status`, `submit`, `alignment_state`,
  `alignment_reset_focus`, and `dark_library`. Every method answers at once, so all of them run
  inline on the connection thread and no worker is needed. `alignment_reset_focus` restarts the
  best focus value of the helper and answers `{"reset": true}`. `dark_library` answers with the
  `DarkLibraryView` as JSON: the sets of the dark library, whether it is due, the model, the sensor
  temperature, and the progress of the latest dark session. `core` adds one method that the
  contract does not name: `results` answers with the latest commissioning results (the `detail` of
  each result), so that `seeingmon burst --wait` can show the outcome of its task. A client that
  does not know the method never calls it.
- **`alignment`**, a `StreamService`. The helper takes each client as a `StreamSender` and pushes
  the frames of the live view (see `seeingmon.services.core.alignment.helper`).

**Commands.** `submit` decodes the command (a malformed command raises a `CodecError`, which the
connection layer turns into an `InvalidParams` error) and hands it to `Scheduler.submit`, which
answers at once. A replay command is checked before it reaches the scheduler: its source must be a
recording name without a directory part that exists, and its options must be on the list that the
replay accepts. A command that fails the check is a normal answer with `accepted` false, and
`core` writes an event for it, as the scheduler does for every command that it sees. The owner of
the RPC can ask to hear about each command that the scheduler accepted (`on_accepted`), which is how
`core` learns that a dark task is queued.

**Roles.** A client names itself in the hello parameters (`role` is `web` or `cli`). The health
record counts the `web` component as `ok` while a client with that role is connected.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from seeingmon.scheduler.commands import (
    Command,
    CommandResult,
    QueueReplay,
    RejectReason,
)
from seeingmon.scheduler.commission import CommissionResult
from seeingmon.scheduler.status import SchedulerStatus
from seeingmon.services.core.events import EventWriter
from seeingmon.services.ipc.codec import as_mapping
from seeingmon.services.ipc.rpc import RpcService
from seeingmon.services.ipc.stream import StreamSender, StreamService, StreamWindow
from seeingmon.services.web.contract import (
    METHOD_ALIGNMENT_RESET_FOCUS,
    METHOD_ALIGNMENT_STATE,
    METHOD_DARK_LIBRARY,
    METHOD_PING,
    METHOD_STATUS,
    METHOD_SUBMIT,
    AlignmentState,
    DarkLibraryView,
    decode_command,
    encode_result,
    encode_status,
)

METHOD_RESULTS = "results"
MAX_RESULTS = 32
LIVE_VIEW_MAX_MESSAGE_BYTES = 32 * 1024 * 1024

_log = logging.getLogger(__name__)


class SchedulerPort(Protocol):
    """The part of the scheduler that the RPC uses. `Scheduler` fits."""

    def status(self) -> SchedulerStatus: ...

    def submit(self, command: Command) -> CommandResult: ...

    def results(self) -> tuple[CommissionResult, ...]: ...


class AlignmentPort(Protocol):
    """The part of the alignment helper that the RPC uses. `AlignmentHelper` fits."""

    def state(self) -> AlignmentState: ...

    def reset_focus(self) -> None: ...

    def attach(self, sender: StreamSender, params: Mapping[str, Any] | None = None) -> None: ...


class CoreRpc:
    """The methods of the `rpc` channel, and the builders of the two services."""

    def __init__(
        self,
        *,
        instance: str,
        scheduler: SchedulerPort,
        alignment: AlignmentPort,
        check_replay: Callable[[QueueReplay], str | None] | None = None,
        writer: EventWriter | None = None,
        dark_library: Callable[[], DarkLibraryView] | None = None,
        on_accepted: Callable[[Command, CommandResult], None] | None = None,
    ) -> None:
        self.instance = instance
        self._scheduler = scheduler
        self._alignment = alignment
        self._check_replay = check_replay
        self._writer = writer
        self._dark_library = dark_library
        self._on_accepted = on_accepted
        self._service: RpcService | None = None
        self.submitted = 0
        self.refused = 0

    # --- The methods -----------------------------------------------------------------------

    def handlers(self) -> dict[str, Callable[[Mapping[str, Any]], Any]]:
        methods: dict[str, Callable[[Mapping[str, Any]], Any]] = {
            METHOD_PING: self._ping,
            METHOD_STATUS: self._status,
            METHOD_SUBMIT: self._submit,
            METHOD_ALIGNMENT_STATE: self._alignment_state,
            METHOD_ALIGNMENT_RESET_FOCUS: self._alignment_reset_focus,
            METHOD_RESULTS: self._results,
        }
        if self._dark_library is not None:
            methods[METHOD_DARK_LIBRARY] = self._answer_dark_library
        return methods

    def _ping(self, params: Mapping[str, Any]) -> Any:
        return {"instance": self.instance}

    def _status(self, params: Mapping[str, Any]) -> Any:
        return encode_status(self._scheduler.status(), self.instance)

    def _alignment_state(self, params: Mapping[str, Any]) -> Any:
        return self._alignment.state().model_dump(mode="json")

    def _alignment_reset_focus(self, params: Mapping[str, Any]) -> Any:
        self._alignment.reset_focus()
        return {"reset": True}

    def _answer_dark_library(self, params: Mapping[str, Any]) -> Any:
        assert self._dark_library is not None
        return self._dark_library().model_dump(mode="json")

    def _results(self, params: Mapping[str, Any]) -> Any:
        recent = self._scheduler.results()[-MAX_RESULTS:]
        return {"instance": self.instance, "results": [r.to_detail() for r in recent]}

    def _submit(self, params: Mapping[str, Any]) -> Any:
        command = decode_command(as_mapping(params, "params").get("command"))
        problem = self._problem_with(command)
        if problem is not None:
            self.refused += 1
            state = self._scheduler.status().state
            if self._writer is not None:
                self._writer.emit(
                    "warning",
                    "core.command_refused",
                    f"Refused {type(command).__name__}: {problem}",
                    {"command": type(command).__name__},
                )
            return encode_result(
                CommandResult(
                    accepted=False, message=problem, state=state, reason=RejectReason.INVALID
                )
            )
        self.submitted += 1
        result = self._scheduler.submit(command)
        if result.accepted and self._on_accepted is not None:
            try:
                self._on_accepted(command, result)
            except Exception:
                _log.exception("the listener of the accepted commands failed")
        return encode_result(result)

    def _problem_with(self, command: Command) -> str | None:
        if isinstance(command, QueueReplay) and self._check_replay is not None:
            return self._check_replay(command)
        return None

    # --- The services ----------------------------------------------------------------------

    @property
    def web_connected(self) -> bool:
        """Whether a client with the role `web` is connected."""
        service = self._service
        return service is not None and any(
            c.params.get("role") == "web" and not c.closed for c in service.connections
        )

    def rpc_service(self, *, max_connections: int, max_message_bytes: int) -> RpcService:
        """The `RpcService` of the `rpc` channel. All methods run inline."""
        self._service = RpcService(
            self.handlers(),
            workers=1,
            worker_name="core-rpc",
            inline=set(self.handlers()),
            max_connections=max_connections,
            max_message_bytes=max_message_bytes,
        )
        return self._service

    def stream_service(self, window: StreamWindow) -> StreamService:
        """The `StreamService` of the `alignment` channel."""
        return StreamService(
            self._alignment.attach,
            max_window=window,
            max_message_bytes=LIVE_VIEW_MAX_MESSAGE_BYTES,
            name="core-alignment",
        )


__all__ = ["METHOD_RESULTS", "AlignmentPort", "CoreRpc", "SchedulerPort"]
