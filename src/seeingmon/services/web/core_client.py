"""The seam between the web process and `core`: the `CoreClient` protocol and its two clients.

The API needs these things from `core`: the status of the scheduler, a way to submit a scheduler
command, the start of the rapid focus mode (`core` knows where Polaris is), the state of the
alignment helper, the dark library with the progress of the dark task, the flat library with the
progress of the flat task (and a way to use a flat, to delete one, and to read its preview), the
rolling seeing value, and the frames of the two live views (the alignment helper and the video
of Polaris). `CoreClient` names them.

- `RpcCoreClient` is the production client. It connects to `core` over the local connection layer
  and speaks the methods that `seeingmon.services.web.contract` documents. It connects when the
  first call needs it, and it reconnects after `core` restarts. A failed connection is
  remembered: for `retry_interval_s`, a call fails at once with `CoreUnavailableError`, and the
  next call then tries again. That try waits only `probe_timeout_s` for `core`, not the whole
  `connect_timeout_s`, so a page that polls a `core` that is down pays a short wait once in a
  while and not the long one on every refresh.
- `FakeCoreClient` answers in memory, for tests and for the demo. It follows the rules of the real
  scheduler for the commands, and it streams the frames of a source that you give it.

The calls `status`, `submit`, `rapid_focus_start`, `alignment_state`, `dark_library`,
`live_seeing`, and the four flat calls block, so call them from a thread (FastAPI runs a plain
`def` endpoint in its thread pool). `alignment_frames` and `polaris_frames` are async iterators.

**Errors.** A call raises `CoreUnavailableError` when `core` cannot be reached or does not answer in
time, and `CoreProtocolError` when `core` answers something that this client cannot use. Neither
message carries text from `core`, because that text could hold a private path.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, Protocol, TypeVar, runtime_checkable

from seeingmon.clock import Clock, SystemClock
from seeingmon.scheduler.commands import (
    TASK_KINDS,
    CancelTask,
    Command,
    CommandResult,
    Pause,
    QueueBurst,
    QueueDark,
    QueueFlat,
    QueueReplay,
    QueueSweep,
    RejectReason,
    Resume,
    StartAlignment,
    StartRapidFocus,
    StopAlignment,
    StopRapidFocus,
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
    METHOD_ALIGNMENT_RESET_FOCUS,
    METHOD_ALIGNMENT_STATE,
    METHOD_DARK_LIBRARY,
    METHOD_FLAT_ACTIVATE,
    METHOD_FLAT_DELETE,
    METHOD_FLAT_IMAGE,
    METHOD_FLAT_LIBRARY,
    METHOD_LIVE_SEEING,
    METHOD_PING,
    METHOD_RAPID_FOCUS_START,
    METHOD_STATUS,
    METHOD_SUBMIT,
    POLARIS_CHANNEL,
    RPC_CHANNEL,
    ActivityView,
    AlignmentFrame,
    AlignmentState,
    CoreStatus,
    DarkLibraryView,
    FaultView,
    FlatActionView,
    FlatLibraryView,
    LiveSeeingView,
    PolarisFrame,
    SchedulerView,
    StreamView,
    decode_alignment_state,
    decode_dark_library,
    decode_flat_action,
    decode_flat_image,
    decode_flat_library,
    decode_live_seeing,
    decode_result,
    decode_status,
    encode_command,
    encode_rapid_focus_params,
    unpack_frame,
    unpack_polaris_frame,
)
from seeingmon.services.web.fake_dark import DarkScript, DarkSimulator
from seeingmon.services.web.fake_flat import FlatScript, FlatSimulator

MIB = 1024 * 1024
# A frame of the video of Polaris is a few kilobytes, and the stream sends 20 of them a second, so
# the window holds eight of them: the event loop may stall for 400 ms before `core` skips a frame.
POLARIS_WINDOW = StreamWindow(messages=8, bytes=4 * MIB)
FrameSource = Callable[[], AsyncIterator[AlignmentFrame]]
PolarisSource = Callable[[], AsyncIterator[PolarisFrame]]
F = TypeVar("F")

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

    def rapid_focus_start(
        self, exposure_us: int | None = None, gain: int | None = None
    ) -> CommandResult:
        """Start the rapid focus mode, or keep it alive, with the center that `core` chooses.

        A mode that `core` does not offer is a rejected result (`not_aligning` or
        `not_available`), not an error.
        """
        ...

    def alignment_state(self) -> AlignmentState:
        """The state of the alignment helper, with `active` false outside alignment."""
        ...

    def alignment_reset_focus(self) -> None:
        """Restart the best focus value of the alignment. Raises `CoreError`."""
        ...

    def dark_library(self) -> DarkLibraryView:
        """The dark library, its status, and the latest dark task. Raises `CoreError`."""
        ...

    def live_seeing(self) -> LiveSeeingView | None:
        """The rolling seeing value of the fast stream, or `None` while `core` has none."""
        ...

    def flat_library(self) -> FlatLibraryView:
        """The flat library, the flat in use, and the latest flat task. Raises `CoreError`."""
        ...

    def flat_activate(self, version: str) -> FlatActionView:
        """Make a flat the one that the survey uses. A refusal is a result with `ok` false."""
        ...

    def flat_delete(self, version: str) -> FlatActionView:
        """Delete a flat that is not in use. A refusal is a result with `ok` false."""
        ...

    def flat_image(self, version: str) -> bytes | None:
        """The JPEG preview of a flat, or `None` when `core` has none. Raises `CoreError`."""
        ...

    def alignment_frames(self) -> AsyncIterator[AlignmentFrame]:
        """The live-view frames, newest last. Raises `CoreError` when the stream breaks."""
        ...

    def polaris_frames(self) -> AsyncIterator[PolarisFrame]:
        """The frames of the video of Polaris, newest last. Raises `CoreError` when it breaks."""
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
        polaris_window: StreamWindow | None = None,
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
        self._polaris_window = polaris_window or POLARIS_WINDOW
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

    def rapid_focus_start(
        self, exposure_us: int | None = None, gain: int | None = None
    ) -> CommandResult:
        """Start the rapid focus mode with the `rapid_focus_start` method."""
        answer = self._call(
            METHOD_RAPID_FOCUS_START,
            encode_rapid_focus_params(exposure_us, gain),
            self._submit_timeout_s,
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

    def alignment_reset_focus(self) -> None:
        """Restart the best focus value, with the `alignment_reset_focus` method."""
        answer = self._call(METHOD_ALIGNMENT_RESET_FOCUS, None, self._rpc_timeout_s)
        if not isinstance(answer, Mapping) or answer.get("reset") is not True:
            raise CoreProtocolError("core sent an unreadable answer to the focus reset")

    def dark_library(self) -> DarkLibraryView:
        """The dark library and the dark task, from the `dark_library` method."""
        answer = self._call(METHOD_DARK_LIBRARY, None, self._rpc_timeout_s)
        try:
            return decode_dark_library(answer)
        except CodecError as error:
            _log.warning("core sent an unreadable dark library: %s", error)
            raise CoreProtocolError("core sent an unreadable dark library") from None

    def live_seeing(self) -> LiveSeeingView | None:
        """The rolling seeing value, from the `live_seeing` method. `None` means no value."""
        answer = self._call(METHOD_LIVE_SEEING, None, self._rpc_timeout_s)
        try:
            return decode_live_seeing(answer)
        except CodecError as error:
            _log.warning("core sent an unreadable live seeing value: %s", error)
            raise CoreProtocolError("core sent an unreadable live seeing value") from None

    def flat_library(self) -> FlatLibraryView:
        """The flat library and the flat task, from the `flat_library` method."""
        answer = self._call(METHOD_FLAT_LIBRARY, None, self._rpc_timeout_s)
        try:
            return decode_flat_library(answer)
        except CodecError as error:
            _log.warning("core sent an unreadable flat library: %s", error)
            raise CoreProtocolError("core sent an unreadable flat library") from None

    def _flat_action(self, method: str, version: str) -> FlatActionView:
        answer = self._call(method, {"version": version}, self._rpc_timeout_s)
        try:
            return decode_flat_action(answer)
        except CodecError as error:
            _log.warning("core sent an unreadable flat answer: %s", error)
            raise CoreProtocolError("core sent an unreadable flat answer") from None

    def flat_activate(self, version: str) -> FlatActionView:
        """Activate a flat, with the `flat_activate` method."""
        return self._flat_action(METHOD_FLAT_ACTIVATE, version)

    def flat_delete(self, version: str) -> FlatActionView:
        """Delete a flat, with the `flat_delete` method."""
        return self._flat_action(METHOD_FLAT_DELETE, version)

    def flat_image(self, version: str) -> bytes | None:
        """The preview of a flat, from the `flat_image` method."""
        answer = self._call(METHOD_FLAT_IMAGE, {"version": version}, self._rpc_timeout_s)
        try:
            return decode_flat_image(answer)
        except CodecError as error:
            _log.warning("core sent an unreadable flat image: %s", error)
            raise CoreProtocolError("core sent an unreadable flat image") from None

    def _open_stream(self, channel: str, window: StreamWindow, name: str) -> StreamReceiver:
        try:
            receiver, _ = connect_stream(
                self._endpoint,
                self._key,
                {"role": "web"},
                channel=channel,
                window=window,
                connect_timeout_s=self._connect_timeout_s,
                handshake_timeout_s=self._handshake_timeout_s,
                max_message_bytes=self._max_frame_bytes,
                name=f"web-{name}",
            )
        except IpcAuthError:
            raise CoreUnavailableError("core refused the connection key") from None
        except IpcError:
            raise CoreUnavailableError("core does not answer") from None
        return receiver

    async def _frames(
        self,
        channel: str,
        window: StreamWindow,
        name: str,
        noun: str,
        unpack: Callable[[memoryview], F],
    ) -> AsyncIterator[F]:
        """Open a stream and yield its frames until the caller stops iterating.

        The generator closes the stream when the caller stops, even when a task cancels it.
        """
        receiver = await asyncio.to_thread(self._open_stream, channel, window, name)
        try:
            while True:
                try:
                    message = await asyncio.to_thread(receiver.recv, self._poll_s)
                except IpcClosedError:
                    raise CoreUnavailableError(f"the {noun} stream of core closed") from None
                if message is None or message.kind is not StreamKind.DATA:
                    continue
                try:
                    yield unpack(message.payload)
                except CodecError as error:
                    _log.warning("core sent an unreadable %s frame: %s", noun, error)
                    raise CoreProtocolError(f"core sent an unreadable {noun} frame") from None
        finally:
            # Closing waits for the reader thread, so it must not block the event loop.
            threading.Thread(target=receiver.close, name=f"web-{name}-close", daemon=True).start()

    def alignment_frames(self) -> AsyncIterator[AlignmentFrame]:
        """Open the `alignment` stream and yield its frames until the caller stops iterating."""
        return self._frames(ALIGNMENT_CHANNEL, self._window, "alignment", "alignment", unpack_frame)

    def polaris_frames(self) -> AsyncIterator[PolarisFrame]:
        """Open the `polaris` stream and yield its frames until the caller stops iterating."""
        return self._frames(
            POLARIS_CHANNEL, self._polaris_window, "polaris", "Polaris", unpack_polaris_frame
        )

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
    that returns an async iterator of `AlignmentFrame`. Pass `polaris` for the video of Polaris in
    the same way (`PolarisFrame`), and set `live` to the rolling seeing value that `live_seeing`
    answers with.

    `focus_resets` counts the calls of `alignment_reset_focus`.

    The rapid focus mode follows the rules of the scheduler: a start and a stop need an alignment,
    and `rapid_running` says whether the mode runs. `rapid_focus_start` also needs the offer of
    `core`, which you set with `rapid_offer` (`None` offers the mode, and a text is the reason that
    it is not offered) and `rapid_center` (the pixel of Polaris in pixels of the fast mode). The
    calls land in `rapid_starts` as `(exposure_us, gain)`.

    The dark library lives in `dark`, a `DarkSimulator`: set its `sets`, `model`, and
    `sensor_temperature_c`, and pass `dark_script` to set how long each part of a dark task lasts.
    `QueueDark` starts a scripted task that follows the clock (see `fake_dark`). The flat library
    lives in `flat`, a `FlatSimulator`, and `flat_script` scripts its task (see `fake_flat`). A
    flat session needs a dark set, as in the real `core`. `CancelTask` works for the kinds `dark`
    and `flat`.

    The status answers with the `activity` that you set, which is `None` at first. A subclass can
    compute the activity and the fault from the clock instead (`_activity_view` and `_fault_view`).
    """

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        frames: FrameSource | None = None,
        polaris: PolarisSource | None = None,
        state: str = "auto",
        instance: str = "fake-core",
        max_queued: int = 8,
        alignment_state: AlignmentState | None = None,
        dark_script: DarkScript | None = None,
        flat_script: FlatScript | None = None,
    ) -> None:
        self._clock = SystemClock() if clock is None else clock
        self.dark = DarkSimulator(self._clock, script=dark_script)
        self.flat = FlatSimulator(
            self._clock, script=flat_script, dark_ready=lambda: bool(self.dark.sets)
        )
        self._frames = frames
        self._polaris = polaris
        self.live: LiveSeeingView | None = None
        self._instance = instance
        self._max_queued = max_queued
        self._lock = threading.Lock()
        self._state = state
        self._reason = "a fake transition"
        self._since_ns = self._clock.utc_ns()
        self._queued = 0
        self._next_task_id = 1
        self._accepted = 0
        self._rejected = 0
        self._alignment = alignment_state
        self.degraded = False
        self.activity: ActivityView | None = None
        self.fail_with: CoreError | None = None
        self.submitted: list[Command] = []
        self.status_calls = 0
        self.focus_resets = 0
        self.rapid_offer: str | None = None
        self.rapid_center = (1036.0, 705.5)
        self.rapid_running = False
        self.rapid_starts: list[tuple[int | None, int | None]] = []
        self.streams_opened = 0
        self.streams_closed = 0
        self.polaris_streams_opened = 0
        self.polaris_streams_closed = 0
        self.live_seeing_calls = 0
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

    def _transition(self, state: str, reason: str = "a fake transition") -> None:
        if state != "align":
            self.rapid_running = False
        self._state = state
        self._reason = reason
        self._since_ns = self._clock.utc_ns()

    def _settle_dark(self) -> None:
        """Let the dark task and the flat task follow the clock, and apply what they do."""
        for change in self.dark.settle(state=self._state):
            self._transition(change.state, change.reason)
        for change in self.flat.settle(state=self._state):
            self._transition(change.state, change.reason)

    def _activity_view(self, now_ns: int) -> ActivityView | None:
        """The activity that `status` answers with. A subclass can compute it from the clock."""
        return self.activity

    def _fault_view(self, now_ns: int) -> FaultView:
        """The fault that `status` answers with. A subclass can script an episode."""
        return FaultView()

    def status(self) -> CoreStatus:
        self._check()
        with self._lock:
            self._settle_dark()
            self.status_calls += 1
            now_ns = self._clock.utc_ns()
            return CoreStatus(
                instance=self._instance,
                scheduler=SchedulerView(
                    t_utc_ns=now_ns,
                    state=self._state,
                    state_reason=self._reason,
                    state_since_utc_ns=self._since_ns,
                    last_transition_utc_ns=self._since_ns,
                    degraded=self.degraded,
                    stream=StreamView(
                        stream_id=1, purpose="fast", mode="bin1", exposure_us=2000, gain=0
                    ),
                    fault=self._fault_view(now_ns),
                    queued_tasks=(
                        self._queued
                        + (1 if self.dark.queued else 0)
                        + (1 if self.flat.queued else 0)
                    ),
                    counters={
                        "commands_accepted": self._accepted,
                        "commands_rejected": self._rejected,
                    },
                    activity=self._activity_view(now_ns),
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
            self._settle_dark()
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
        if isinstance(command, StartRapidFocus):
            if self._state != "align":
                return self._reject(RejectReason.NOT_ALIGNING, "no alignment runs")
            if self.degraded:
                return self._reject(RejectReason.DEGRADED, "the camera has failed repeatedly")
            again, self.rapid_running = self.rapid_running, True
            return self._result(
                True,
                "rapid focus already runs, so the idle timer restarted"
                if again
                else "rapid focus started",
            )
        if isinstance(command, StopRapidFocus):
            if self._state != "align":
                return self._reject(RejectReason.NOT_ALIGNING, "no alignment runs")
            was, self.rapid_running = self.rapid_running, False
            return self._result(
                True,
                "rapid focus stopped" if was else "rapid focus does not run, so nothing stops",
            )
        if isinstance(command, Pause):
            if self._state == "paused":
                return self._reject(RejectReason.ALREADY_PAUSED, "the scheduler is already paused")
            self.dark.abort()
            self.flat.abort()
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
        if isinstance(command, QueueDark):
            accepted, reason, message = self.dark.submit(command, self._next_task_id, self._state)
            if not accepted:
                assert reason is not None
                return self._reject(reason, message)
            task_id = self._next_task_id
            self._next_task_id += 1
            return self._result(True, message, task_id=task_id)
        if isinstance(command, QueueFlat):
            accepted, reason, message = self.flat.submit(command, self._next_task_id, self._state)
            if not accepted:
                assert reason is not None
                return self._reject(reason, message)
            task_id = self._next_task_id
            self._next_task_id += 1
            return self._result(True, message, task_id=task_id)
        if isinstance(command, CancelTask):
            return self._cancel(command)
        return self._reject(RejectReason.INVALID, "unknown command")

    def _cancel(self, command: CancelTask) -> CommandResult:
        """Cancel the task of a kind. The fake tracks the kinds `dark` and `flat` only."""
        kind = command.kind
        if kind not in TASK_KINDS.values():
            return self._reject(RejectReason.INVALID, f"{kind!r} is not a kind of task")
        if kind == "flat":
            outcome = self.flat.cancel(self._state)
            if not outcome.accepted:
                assert outcome.reason is not None
                return self._reject(outcome.reason, outcome.message)
            for change in outcome.changes:
                self._transition(change.state, change.reason)
            return self._result(True, outcome.message, task_id=outcome.task_id)
        if kind == "dark" and self.dark.abort():
            return self._result(True, "the dark task is removed or stops at its next check")
        return self._reject(RejectReason.NO_TASK, f"no {kind} task waits or runs")

    def dark_library(self) -> DarkLibraryView:
        self._check()
        with self._lock:
            self._settle_dark()
            return self.dark.library()

    def flat_library(self) -> FlatLibraryView:
        self._check()
        with self._lock:
            self._settle_dark()
            return self.flat.library()

    def flat_activate(self, version: str) -> FlatActionView:
        self._check()
        with self._lock:
            self._settle_dark()
            return self.flat.activate(version)

    def flat_delete(self, version: str) -> FlatActionView:
        self._check()
        with self._lock:
            self._settle_dark()
            return self.flat.delete(version)

    def flat_image(self, version: str) -> bytes | None:
        self._check()
        with self._lock:
            return self.flat.image(version)

    def rapid_focus_start(
        self, exposure_us: int | None = None, gain: int | None = None
    ) -> CommandResult:
        self._check()
        with self._lock:
            self.rapid_starts.append((exposure_us, gain))
            self._settle_dark()
            if self._state != "align":
                return self._reject(RejectReason.NOT_ALIGNING, "no alignment runs")
            if self.rapid_offer is not None and not self.rapid_running:
                return self._reject(RejectReason.NOT_AVAILABLE, self.rapid_offer)
            command = StartRapidFocus(*self.rapid_center, exposure_us, gain)
            self.submitted.append(command)
            return self._apply(command)

    def alignment_state(self) -> AlignmentState:
        self._check()
        with self._lock:
            if self._state != "align":
                return AlignmentState(active=False)
            return self._alignment or AlignmentState(active=True)

    def alignment_reset_focus(self) -> None:
        self._check()
        with self._lock:
            self.focus_resets += 1

    def live_seeing(self) -> LiveSeeingView | None:
        self._check()
        with self._lock:
            self.live_seeing_calls += 1
            return self.live

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

    async def polaris_frames(self) -> AsyncIterator[PolarisFrame]:
        self._check()
        with self._lock:
            self.polaris_streams_opened += 1
        try:
            if self._polaris is None:
                await asyncio.Event().wait()  # an open stream that sends nothing
            else:
                async for frame in self._polaris():
                    self._check()
                    yield frame
        finally:
            with self._lock:
                self.polaris_streams_closed += 1

    def close(self) -> None:
        self.closed = True


__all__ = [
    "CoreClient",
    "CoreError",
    "CoreProtocolError",
    "CoreUnavailableError",
    "FakeCoreClient",
    "FrameSource",
    "PolarisSource",
    "RpcCoreClient",
]
