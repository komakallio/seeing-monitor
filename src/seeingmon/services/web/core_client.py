"""The seam between the web process and `core`: the `CoreClient` protocol and its two clients.

The API needs four things from `core`: the status of the scheduler, a way to submit a scheduler
command, the state of the alignment helper, and the live-view frames. `CoreClient` names them.

- `RpcCoreClient` is the production client. It connects to `core` over the local connection layer
  and speaks the methods that `seeingmon.services.web.contract` documents. It connects when the
  first call needs it, and it reconnects after `core` restarts. A failed connection is
  remembered: for `retry_interval_s`, a call fails at once with `CoreUnavailableError`, and the
  next call then tries again. That try waits only `probe_timeout_s` for `core`, not the whole
  `connect_timeout_s`, so a page that polls a `core` that is down pays a short wait once in a
  while and not the long one on every refresh.
- `FakeCoreClient` answers in memory, for tests and for the demo. It follows the rules of the real
  scheduler for the commands, and it streams the frames of a source that you give it.

The calls `status`, `submit`, and `alignment_state` block, so call them from a thread (FastAPI runs
a plain `def` endpoint in its thread pool). `alignment_frames` is an async iterator.

**Errors.** A call raises `CoreUnavailableError` when `core` cannot be reached or does not answer in
time, and `CoreProtocolError` when `core` answers something that this client cannot use. Neither
message carries text from `core`, because that text could hold a private path.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from seeingmon.clock import Clock, SystemClock
from seeingmon.scheduler.commands import (
    TASK_KINDS,
    Command,
    CommandResult,
    Pause,
    QueueBurst,
    QueueReplay,
    QueueSweep,
    RejectReason,
    Resume,
    StartAlignment,
    StopAlignment,
)
from seeingmon.services.config import ServicesConfig
from seeingmon.services.ipc.codec import CodecError
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.errors import (
    IpcAuthError,
    IpcClosedError,
    IpcError,
    RpcError,
    RpcInvalidParamsError,
    RpcMethodNotFoundError,
    RpcTimeoutError,
)
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.rpc import RpcClient, connect_rpc
from seeingmon.services.ipc.stream import StreamKind, StreamReceiver, StreamWindow, connect_stream
from seeingmon.services.web.config import CoreLinkSettings
from seeingmon.services.web.contract import (
    ALIGNMENT_CHANNEL,
    METHOD_ALIGNMENT_STATE,
    METHOD_PING,
    METHOD_STATUS,
    METHOD_SUBMIT,
    RPC_CHANNEL,
    AlignmentFrame,
    AlignmentState,
    CoreStatus,
    SchedulerView,
    StreamView,
    decode_alignment_state,
    decode_result,
    decode_status,
    encode_command,
    unpack_frame,
)

MIB = 1024 * 1024
FrameSource = Callable[[], AsyncIterator[AlignmentFrame]]

_log = logging.getLogger(__name__)


class CoreError(Exception):
    """Base class for the failures of a call to `core`."""


class CoreUnavailableError(CoreError):
    """`core` cannot be reached, or it did not answer in time. A retry may work."""


class CoreProtocolError(CoreError):
    """`core` answered something that this client cannot use, so the two versions may differ."""


@runtime_checkable
class CoreClient(Protocol):
    """What the API needs from `core`."""

    def status(self) -> CoreStatus:
        """The status of the scheduler. Raises `CoreError`."""
        ...

    def submit(self, command: Command) -> CommandResult:
        """Hand a command to the scheduler. A rejection is a result, not an error."""
        ...

    def alignment_state(self) -> AlignmentState:
        """The state of the alignment helper, with `active` false outside alignment."""
        ...

    def alignment_frames(self) -> AsyncIterator[AlignmentFrame]:
        """The live-view frames, newest last. Raises `CoreError` when the stream breaks."""
        ...

    def close(self) -> None:
        """Release the connections. Safe to call twice."""
        ...


# --- The production client -------------------------------------------------------------------


class RpcCoreClient:
    """A `CoreClient` that talks to the `core` process over the connection layer."""

    def __init__(
        self,
        endpoint: Endpoint,
        key: ConnectionKey,
        *,
        connect_timeout_s: float = 2.0,
        handshake_timeout_s: float = 5.0,
        rpc_timeout_s: float = 5.0,
        submit_timeout_s: float = 10.0,
        retry_interval_s: float = 3.0,
        probe_timeout_s: float = 0.25,
        max_rpc_bytes: int = 1 * MIB,
        max_frame_bytes: int = 32 * MIB,
        window: StreamWindow | None = None,
        poll_s: float = 1.0,
        clock: Clock | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._key = key
        self._connect_timeout_s = connect_timeout_s
        self._handshake_timeout_s = handshake_timeout_s
        self._rpc_timeout_s = rpc_timeout_s
        self._submit_timeout_s = submit_timeout_s
        self._retry_interval_ns = round(retry_interval_s * 1e9)
        self._probe_timeout_s = probe_timeout_s
        self._max_rpc_bytes = max_rpc_bytes
        self._max_frame_bytes = max_frame_bytes
        self._window = window or StreamWindow(messages=4, bytes=2 * max_frame_bytes)
        self._poll_s = poll_s
        self._clock = SystemClock() if clock is None else clock
        self._lock = threading.RLock()
        self._rpc: RpcClient | None = None
        self._retry_after_ns = 0
        self._failed = False  # the last connection attempt failed, or the link broke
        self._last_failure = "core has not answered yet"
        self.connections = 0

    @classmethod
    def from_config(
        cls,
        services: ServicesConfig,
        link: CoreLinkSettings,
        *,
        env: Mapping[str, str] | None = None,
        clock: Clock | None = None,
    ) -> RpcCoreClient:
        """Build the client from `[services]` (the address and the key) and `[web.core]`."""
        return cls(
            services.endpoint("core", env=env),
            services.load_key(env),
            connect_timeout_s=link.connect_timeout_s,
            handshake_timeout_s=services.handshake_timeout_s,
            rpc_timeout_s=link.rpc_timeout_s,
            submit_timeout_s=link.submit_timeout_s,
            retry_interval_s=link.retry_interval_s,
            probe_timeout_s=link.probe_timeout_s,
            max_rpc_bytes=services.max_rpc_bytes,
            max_frame_bytes=min(services.max_frame_bytes, 32 * MIB),
            clock=clock,
        )

    # --- Connection ------------------------------------------------------------------------

    def _fail(self, message: str) -> CoreUnavailableError:
        with self._lock:
            self._last_failure = message
            self._failed = True
            self._retry_after_ns = self._clock.monotonic_ns() + self._retry_interval_ns
        return CoreUnavailableError(message)

    def _connected(self) -> RpcClient:
        with self._lock:
            client = self._rpc
            if client is not None and not client.closed:
                return client
            self._rpc = None
            if self._clock.monotonic_ns() < self._retry_after_ns:
                raise CoreUnavailableError(self._last_failure)
            # The first try waits for `core` as long as `connect_timeout_s` (it may be starting).
            # A try after a failure is a probe: `core` is probably still down, and a call that
            # waits for it holds up a page.
            wait_s = (
                min(self._probe_timeout_s, self._connect_timeout_s)
                if self._failed
                else self._connect_timeout_s
            )
            try:
                client, _ = connect_rpc(
                    self._endpoint,
                    self._key,
                    {"role": "web"},
                    channel=RPC_CHANNEL,
                    connect_timeout_s=wait_s,
                    handshake_timeout_s=self._handshake_timeout_s,
                    default_timeout_s=self._rpc_timeout_s,
                    max_message_bytes=self._max_rpc_bytes,
                    name="web-rpc",
                )
            except IpcAuthError:
                raise self._fail("core refused the connection key") from None
            except IpcError:
                raise self._fail("core does not answer") from None
            self._rpc = client
            self._failed = False
            self.connections += 1
            return client

    def _drop(self, client: RpcClient) -> None:
        with self._lock:
            if self._rpc is client:
                self._rpc = None
        client.close("the web process dropped the connection")

    def _call(self, method: str, params: Mapping[str, Any] | None, timeout_s: float) -> Any:
        client = self._connected()
        try:
            return client.call(method, params, timeout_s=timeout_s)
        except IpcClosedError:
            self._drop(client)
            raise self._fail("the connection to core closed") from None
        except RpcTimeoutError:
            raise CoreUnavailableError(f"core did not answer {method} in time") from None
        except RpcMethodNotFoundError:
            raise CoreProtocolError(f"core does not serve {method}: the versions differ") from None
        except (RpcInvalidParamsError, RpcError, ValueError, TypeError) as error:
            _log.warning("core failed %s: %s", method, error)
            raise CoreProtocolError(f"core could not handle {method}") from None
        except IpcError:
            raise CoreUnavailableError("the connection to core failed") from None

    # --- CoreClient ------------------------------------------------------------------------

    def ping(self) -> str:
        """The identity of the `core` process. The ID changes when it restarts."""
        answer = self._call(METHOD_PING, None, self._rpc_timeout_s)
        instance = answer.get("instance") if isinstance(answer, Mapping) else None
        if not isinstance(instance, str):
            raise CoreProtocolError("core sent an unreadable ping answer")
        return instance

    def status(self) -> CoreStatus:
        """The status of the scheduler, from the `status` method."""
        answer = self._call(METHOD_STATUS, None, self._rpc_timeout_s)
        try:
            return decode_status(answer)
        except CodecError as error:
            _log.warning("core sent an unreadable status: %s", error)
            raise CoreProtocolError("core sent an unreadable status") from None

    def submit(self, command: Command) -> CommandResult:
        """Send a command with the `submit` method. A rejection comes back as a result."""
        answer = self._call(
            METHOD_SUBMIT, {"command": encode_command(command)}, self._submit_timeout_s
        )
        try:
            return decode_result(answer)
        except CodecError as error:
            _log.warning("core sent an unreadable command result: %s", error)
            raise CoreProtocolError("core sent an unreadable command result") from None

    def alignment_state(self) -> AlignmentState:
        """The state of the alignment helper, from the `alignment_state` method."""
        answer = self._call(METHOD_ALIGNMENT_STATE, None, self._rpc_timeout_s)
        try:
            return decode_alignment_state(answer)
        except CodecError as error:
            _log.warning("core sent an unreadable alignment state: %s", error)
            raise CoreProtocolError("core sent an unreadable alignment state") from None

    def _open_stream(self) -> StreamReceiver:
        try:
            receiver, _ = connect_stream(
                self._endpoint,
                self._key,
                {"role": "web"},
                channel=ALIGNMENT_CHANNEL,
                window=self._window,
                connect_timeout_s=self._connect_timeout_s,
                handshake_timeout_s=self._handshake_timeout_s,
                max_message_bytes=self._max_frame_bytes,
                name="web-alignment",
            )
        except IpcAuthError:
            raise CoreUnavailableError("core refused the connection key") from None
        except IpcError:
            raise CoreUnavailableError("core does not answer") from None
        return receiver

    async def alignment_frames(self) -> AsyncIterator[AlignmentFrame]:
        """Open the `alignment` stream and yield its frames until the caller stops iterating.

        The generator closes the stream when the caller stops, even when a task cancels it.
        """
        receiver = await asyncio.to_thread(self._open_stream)
        try:
            while True:
                try:
                    message = await asyncio.to_thread(receiver.recv, self._poll_s)
                except IpcClosedError:
                    raise CoreUnavailableError("the alignment stream of core closed") from None
                if message is None or message.kind is not StreamKind.DATA:
                    continue
                try:
                    yield unpack_frame(message.payload)
                except CodecError as error:
                    _log.warning("core sent an unreadable alignment frame: %s", error)
                    raise CoreProtocolError("core sent an unreadable alignment frame") from None
        finally:
            # Closing waits for the reader thread, so it must not block the event loop.
            threading.Thread(target=receiver.close, name="web-alignment-close", daemon=True).start()

    def close(self) -> None:
        """Close the connection to `core`. A later call connects again."""
        with self._lock:
            client, self._rpc = self._rpc, None
        if client is not None:
            client.close("the web process closed the connection")


# --- The client for tests and the demo -------------------------------------------------------


class FakeCoreClient:
    """A `CoreClient` that answers in memory.

    The scheduler rules for the commands match `Scheduler.submit`: alignment needs a scheduler that
    is not paused, a stop needs an alignment, a pause needs a running scheduler, a resume needs a
    pause, and the commissioning queue holds `max_queued` tasks. Every command that the client
    receives lands in `submitted`, in order. Set `fail_with` to make every call raise that error,
    and set it back to `None` to recover. Pass `frames` to give the live view a source: a function
    that returns an async iterator of `AlignmentFrame`.
    """

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        frames: FrameSource | None = None,
        state: str = "auto",
        instance: str = "fake-core",
        max_queued: int = 8,
        alignment_state: AlignmentState | None = None,
    ) -> None:
        self._clock = SystemClock() if clock is None else clock
        self._frames = frames
        self._instance = instance
        self._max_queued = max_queued
        self._lock = threading.Lock()
        self._state = state
        self._since_ns = self._clock.utc_ns()
        self._queued = 0
        self._next_task_id = 1
        self._accepted = 0
        self._rejected = 0
        self._alignment = alignment_state
        self.degraded = False
        self.fail_with: CoreError | None = None
        self.submitted: list[Command] = []
        self.status_calls = 0
        self.streams_opened = 0
        self.streams_closed = 0
        self.closed = False

    @property
    def state(self) -> str:
        """The state of the fake scheduler."""
        with self._lock:
            return self._state

    def set_alignment_state(self, state: AlignmentState | None) -> None:
        """Set the state that `alignment_state` answers while the fake scheduler aligns."""
        with self._lock:
            self._alignment = state

    def _check(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with

    def _transition(self, state: str) -> None:
        self._state = state
        self._since_ns = self._clock.utc_ns()

    def status(self) -> CoreStatus:
        self._check()
        with self._lock:
            self.status_calls += 1
            now_ns = self._clock.utc_ns()
            return CoreStatus(
                instance=self._instance,
                scheduler=SchedulerView(
                    t_utc_ns=now_ns,
                    state=self._state,
                    state_reason="a fake transition",
                    state_since_utc_ns=self._since_ns,
                    last_transition_utc_ns=self._since_ns,
                    degraded=self.degraded,
                    stream=StreamView(
                        stream_id=1, purpose="fast", mode="bin1", exposure_us=2000, gain=0
                    ),
                    queued_tasks=self._queued,
                    counters={
                        "commands_accepted": self._accepted,
                        "commands_rejected": self._rejected,
                    },
                ),
            )

    def _result(self, accepted: bool, message: str, **fields: Any) -> CommandResult:
        if accepted:
            self._accepted += 1
        else:
            self._rejected += 1
        return CommandResult(accepted=accepted, message=message, state=self._state, **fields)

    def _reject(self, reason: RejectReason, message: str) -> CommandResult:
        return self._result(False, message, reason=reason)

    def submit(self, command: Command) -> CommandResult:
        self._check()
        with self._lock:
            self.submitted.append(command)
            return self._apply(command)

    def _apply(self, command: Command) -> CommandResult:
        if isinstance(command, StartAlignment):
            if self._state == "paused":
                return self._reject(RejectReason.PAUSED, "the scheduler is paused; resume it first")
            if self.degraded:
                return self._reject(RejectReason.DEGRADED, "the camera has failed repeatedly")
            if self._state == "align":
                return self._result(True, "alignment already runs, so the idle timer restarted")
            self._transition("align")
            return self._result(True, "alignment started")
        if isinstance(command, StopAlignment):
            if self._state != "align":
                return self._reject(RejectReason.NOT_ALIGNING, "no alignment runs")
            self._transition("safe")
            return self._result(True, "alignment stopped")
        if isinstance(command, Pause):
            if self._state == "paused":
                return self._reject(RejectReason.ALREADY_PAUSED, "the scheduler is already paused")
            self._transition("paused")
            return self._result(True, "the scheduler paused, and nothing runs until you resume it")
        if isinstance(command, Resume):
            if self._state != "paused":
                return self._reject(RejectReason.NOT_PAUSED, "the scheduler is not paused")
            self._transition("safe")
            return self._result(True, "the scheduler resumed in safe and checks the sky")
        if isinstance(command, QueueBurst | QueueSweep | QueueReplay):
            if self._queued >= self._max_queued:
                return self._reject(RejectReason.QUEUE_FULL, "the commissioning queue is full")
            task_id = self._next_task_id
            self._next_task_id += 1
            self._queued += 1
            return self._result(
                True,
                f"the {TASK_KINDS[type(command)]} is queued and runs at the next cycle boundary",
                task_id=task_id,
            )
        return self._reject(RejectReason.INVALID, "unknown command")

    def alignment_state(self) -> AlignmentState:
        self._check()
        with self._lock:
            if self._state != "align":
                return AlignmentState(active=False)
            return self._alignment or AlignmentState(active=True)

    async def alignment_frames(self) -> AsyncIterator[AlignmentFrame]:
        self._check()
        with self._lock:
            self.streams_opened += 1
        try:
            if self._frames is None:
                await asyncio.Event().wait()  # an open stream that sends nothing
            else:
                async for frame in self._frames():
                    self._check()
                    yield frame
        finally:
            with self._lock:
                self.streams_closed += 1

    def close(self) -> None:
        self.closed = True


__all__ = [
    "CoreClient",
    "CoreError",
    "CoreProtocolError",
    "CoreUnavailableError",
    "FakeCoreClient",
    "FrameSource",
    "RpcCoreClient",
]
