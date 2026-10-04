"""What `core` serves: the RPC methods and the live-view streams.

`core` listens at one address and serves three channels (see `seeingmon.services.web.contract`,
which `web` owns and `core` implements):

- **`rpc`**, an `RpcService` with the methods `ping`, `status`, `submit`, `alignment_state`,
  `alignment_reset_focus`, `dark_library`, `live_seeing`, `flat_library`, `flat_activate`,
  `flat_delete`, and `flat_image`. Every method answers at once, so all of them run inline on the
  connection thread and no worker is needed. `alignment_reset_focus` restarts the best focus value
  of the helper and answers `{"reset": true}`. `dark_library` answers with the `DarkLibraryView` as
  JSON: the sets of the dark library, whether it is due, the model, the sensor temperature, and the
  progress of the latest dark session. `live_seeing` answers with the `LiveSeeingView`, the rolling
  seeing value of the fast stream, or with `null` while `core` has none. `flat_library` answers
  with the `FlatLibraryView`: the flats with the numbers of their reports, the flat in use, the
  flat that waits for a decision, and the progress of the latest flat session. `flat_activate` and
  `flat_delete` change the library, and `flat_image` answers with the preview of a flat. `core`
  adds one method that the contract does not name: `results` answers with the latest commissioning
  results (the `detail` of each result), so that `seeingmon burst --wait` can show the outcome of
  its task. A client that does not know the method never calls it.
- **`alignment`**, a `StreamService`. The helper takes each client as a `StreamSender` and pushes
  the frames of the live view (see `seeingmon.services.core.alignment.helper`).
- **`polaris`**, a `StreamService`. `PolarisStream` takes each client as a `StreamSender` and pushes
  the frames of the live video of Polaris (see `seeingmon.services.core.live`).

**Commands.** `submit` decodes the command (a malformed command raises a `CodecError`, which the
connection layer turns into an `InvalidParams` error) and hands it to `Scheduler.submit`, which
answers at once. A replay command is checked before it reaches the scheduler: its source must be a
recording name without a directory part that exists, and its options must be on the list that the
replay accepts. A flat command is checked too: a flat needs a dark set for its bias, and a second
set needs the first set of a session. A command that fails the check is a normal answer with
`accepted` false, and `core` writes an event for it, as the scheduler does for every command that
it sees. The owner of the RPC can ask to hear about each command that the scheduler accepted
(`on_accepted`), which is how `core` learns that a dark or flat task is queued, or that a flat task
was cancelled.

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
    QueueFlat,
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
    METHOD_FLAT_ACTIVATE,
    METHOD_FLAT_DELETE,
    METHOD_FLAT_IMAGE,
    METHOD_FLAT_LIBRARY,
    METHOD_LIVE_SEEING,
    METHOD_PING,
    METHOD_STATUS,
    METHOD_SUBMIT,
    AlignmentState,
    DarkLibraryView,
    FlatActionView,
    FlatLibraryView,
    LiveSeeingView,
    decode_command,
    encode_flat_image,
    encode_result,
    encode_status,
)
from seeingmon.survey.flat_library import UNKNOWN_FLAT, is_version

METHOD_RESULTS = "results"
MAX_RESULTS = 32
LIVE_VIEW_MAX_MESSAGE_BYTES = 32 * 1024 * 1024
POLARIS_MAX_MESSAGE_BYTES = 1024 * 1024

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


class PolarisPort(Protocol):
    """The part of the video of Polaris that the server uses. `PolarisStream` fits."""

    def attach(self, sender: StreamSender, params: Mapping[str, Any] | None = None) -> None: ...


class FlatPort(Protocol):
    """The part of the flat library that the RPC uses. `FlatLibraryReader` fits."""

    def view(self) -> FlatLibraryView: ...

    def activate(self, version: str) -> FlatActionView: ...

    def delete(self, version: str) -> FlatActionView: ...

    def image(self, version: str) -> bytes | None: ...

    def check(self, command: QueueFlat) -> str | None: ...


class CoreRpc:
    """The methods of the `rpc` channel, and the builders of the stream services."""

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
        live_seeing: Callable[[], LiveSeeingView | None] | None = None,
        polaris: PolarisPort | None = None,
        flat: FlatPort | None = None,
    ) -> None:
        self.instance = instance
        self._scheduler = scheduler
        self._alignment = alignment
        self._live_seeing = live_seeing
        self._polaris = polaris
        self._check_replay = check_replay
        self._writer = writer
        self._dark_library = dark_library
        self._on_accepted = on_accepted
        self._flat = flat
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
        if self._live_seeing is not None:
            methods[METHOD_LIVE_SEEING] = self._answer_live_seeing
        if self._flat is not None:
            methods[METHOD_FLAT_LIBRARY] = self._answer_flat_library
            methods[METHOD_FLAT_ACTIVATE] = self._flat_activate
            methods[METHOD_FLAT_DELETE] = self._flat_delete
            methods[METHOD_FLAT_IMAGE] = self._flat_image
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

    def _answer_live_seeing(self, params: Mapping[str, Any]) -> Any:
        assert self._live_seeing is not None
        live = self._live_seeing()
        return None if live is None else live.model_dump(mode="json")

    def _answer_flat_library(self, params: Mapping[str, Any]) -> Any:
        assert self._flat is not None
        return self._flat.view().model_dump(mode="json")

    @staticmethod
    def _version_of(params: Mapping[str, Any]) -> str | None:
        """The flat version of the parameters, or `None` when it is not one. No path follows."""
        version = params.get("version")
        return version if is_version(version) else None

    def _flat_activate(self, params: Mapping[str, Any]) -> Any:
        assert self._flat is not None
        version = self._version_of(params)
        if version is None:
            return _unknown_flat()
        return self._flat.activate(version).model_dump(mode="json")

    def _flat_delete(self, params: Mapping[str, Any]) -> Any:
        assert self._flat is not None
        version = self._version_of(params)
        if version is None:
            return _unknown_flat()
        return self._flat.delete(version).model_dump(mode="json")

    def _flat_image(self, params: Mapping[str, Any]) -> Any:
        assert self._flat is not None
        version = self._version_of(params)
        return encode_flat_image(None if version is None else self._flat.image(version))

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
        if isinstance(command, QueueFlat) and self._flat is not None:
            return self._flat.check(command)
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

    def polaris_service(self, window: StreamWindow) -> StreamService:
        """The `StreamService` of the `polaris` channel. Raises `ValueError` without a stream."""
        if self._polaris is None:
            raise ValueError("this CoreRpc has no video of Polaris to serve")
        return StreamService(
            self._polaris.attach,
            max_window=window,
            max_message_bytes=POLARIS_MAX_MESSAGE_BYTES,
            name="core-polaris",
        )


def _unknown_flat() -> dict[str, Any]:
    """The answer for a name that is no flat version, which no file can follow from."""
    return FlatActionView(ok=False, reason="unknown", message=UNKNOWN_FLAT).model_dump(mode="json")


__all__ = [
    "METHOD_RESULTS",
    "AlignmentPort",
    "CoreRpc",
    "FlatPort",
    "PolarisPort",
    "SchedulerPort",
]
