"""`RemoteCameraDriver`: a `CameraDriver` that talks to the `acquire` process.

`core` works against the `CameraDriver` interface. In a test the scheduler gets a fake or the
simulator, and in production it gets this class, so the same scheduler code runs in both. Each
method becomes a call to `acquire` (see `seeingmon.services.acquire.service`), and frames arrive
on the stream channel.

**Connection.** `open` connects to `acquire`, authenticates with the connection key, and opens
two connections that form one session: the RPC connection and the frame stream. While `acquire`
is not up (it restarts, or it has not started yet), `open` retries for `connect_timeout_s`. The
other methods connect to nothing: they work on the session that `open` made.

**Errors.** A call that `acquire` answers with a camera error raises that class
(`CameraStateError`, `CameraConfigError`, and so on). A call that gets no answer in time raises
`CameraTimeoutError`, and a call on a connection that is gone raises `CameraDisconnectedError`.

**A restart is a disconnect.** When `acquire` exits (a crash, a hang that its watchdog ended,
or `systemctl restart`), the driver sees the connection close. Frames that already arrived are
delivered first, and then every call raises `CameraDisconnectedError`, until you call `open`
again. A new `acquire` process knows nothing of the old stream, so the caller reopens and
reconfigures through the usual ladder, and the driver never pretends that the stream survived.

**Stale frames.** `configure`, `start`, and `stop` each return the capture epoch that `acquire`
started, and every message on the stream carries the epoch of its frame. `read_frame` discards a
message of another epoch, and a frame of another stream, so the frames of an old stream never
reach the caller after a new `configure`, even when they were already on the wire.
"""

from __future__ import annotations

import enum
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.drivers.base import (
    CameraCaps,
    CameraConfigError,
    CameraDisconnectedError,
    CameraError,
    CameraInfo,
    CameraStateError,
    CameraTimeoutError,
    RecoveryLevel,
)
from seeingmon.frames import (
    ActiveStream,
    Frame,
    FrameDecodeError,
    Roi,
    StreamConfig,
    StreamKind,
    decode_frame,
)
from seeingmon.services.acquire.events import EventBatch, decode_batch
from seeingmon.services.config import ServicesConfig
from seeingmon.services.ipc.codec import (
    CodecError,
    as_mapping,
    decode_active_stream,
    decode_camera_caps,
    decode_camera_info,
    decode_exception,
    decode_json,
    decode_roi,
    encode_stream_config,
    get_int,
    get_str,
)
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.errors import (
    IpcAuthError,
    IpcClosedError,
    IpcError,
    RpcError,
    RpcTimeoutError,
)
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.rpc import RpcClient, connect_rpc
from seeingmon.services.ipc.stream import (
    StreamKind as WireKind,
)
from seeingmon.services.ipc.stream import (
    StreamReceiver,
    StreamWindow,
    connect_stream,
)

RPC_CHANNEL = "rpc"
FRAMES_CHANNEL = "frames"
MAX_TAG = 2**32 - 1

_log = logging.getLogger(__name__)
_REAL_CLOCK = SystemClock()


class _State(enum.Enum):
    NEW = "new"  # nothing connected: before `open`, and after `close`
    CONNECTED = "connected"
    LOST = "lost"  # the connection ended, and only `open` starts a new session


@dataclass(slots=True)
class _Link:
    """The two connections of a session."""

    rpc: RpcClient
    frames: StreamReceiver
    instance: str

    def close(self, reason: str) -> None:
        self.frames.close(reason)
        self.rpc.close(reason)


class RemoteCameraDriver:
    """A `CameraDriver` for the camera that the `acquire` process owns."""

    def __init__(
        self,
        endpoint: Endpoint,
        key: ConnectionKey,
        *,
        connect_timeout_s: float = 5.0,
        handshake_timeout_s: float = 5.0,
        rpc_timeout_s: float = 30.0,
        slow_call_timeout_s: float = 90.0,
        recover_timeout_s: float = 200.0,
        window: StreamWindow | None = None,
        max_rpc_bytes: int = 1024 * 1024,
        max_frame_bytes: int = 128 * 1024 * 1024,
        clock: Clock | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._key = key
        self._connect_timeout_s = connect_timeout_s
        self._handshake_timeout_s = handshake_timeout_s
        self._rpc_timeout_s = rpc_timeout_s
        self._slow_call_timeout_s = max(slow_call_timeout_s, rpc_timeout_s)
        self._recover_timeout_s = max(recover_timeout_s, rpc_timeout_s)
        self._window = window or StreamWindow()
        self._max_rpc_bytes = max_rpc_bytes
        self._max_frame_bytes = max_frame_bytes
        self._clock = _REAL_CLOCK if clock is None else clock
        self._lock = threading.RLock()
        self._open_lock = threading.Lock()
        self._state = _State.NEW
        self._link: _Link | None = None
        self._active: ActiveStream | None = None
        self._capturing = False
        self._epoch = -1
        self.frames_received = 0
        self.stale_discarded = 0
        self.events_received = 0
        self.connections = 0

    @classmethod
    def from_config(
        cls,
        settings: ServicesConfig,
        *,
        env: Mapping[str, str] | None = None,
        clock: Clock | None = None,
    ) -> RemoteCameraDriver:
        """Build the driver from the `[services]` section: its address, key, and limits."""
        return cls(
            settings.endpoint("acquire", env=env),
            settings.load_key(env),
            connect_timeout_s=settings.connect_timeout_s,
            handshake_timeout_s=settings.handshake_timeout_s,
            rpc_timeout_s=settings.rpc_timeout_s,
            window=StreamWindow(settings.stream_window_messages, settings.stream_window_bytes),
            max_rpc_bytes=settings.max_rpc_bytes,
            max_frame_bytes=settings.max_frame_bytes,
            clock=clock,
        )

    # --- State -----------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The backend name: `remote`. `CameraInfo.driver` names the driver in `acquire`."""
        return "remote"

    @property
    def connected(self) -> bool:
        """Whether a session is open and its connections are alive."""
        with self._lock:
            link = self._link
            return self._state is _State.CONNECTED and link is not None and not link.rpc.closed

    @property
    def instance(self) -> str | None:
        """The identity of the `acquire` process of this session. A restart changes it."""
        with self._lock:
            return None if self._link is None else self._link.instance

    def _mark_lost(self, link: _Link, reason: str) -> None:
        with self._lock:
            if self._link is link and self._state is _State.CONNECTED:
                self._state = _State.LOST
                _log.warning("the connection to acquire ended: %s", reason)

    def _require(self) -> _Link:
        """The session for a call. Raises the error that the caller would see without one."""
        with self._lock:
            link, state = self._link, self._state
        if state is _State.NEW or link is None:
            raise CameraStateError("the camera is not open: call open first")
        if state is _State.LOST or link.rpc.closed:
            self._mark_lost(link, link.rpc.close_reason)
            raise CameraDisconnectedError(
                "the connection to acquire was lost: call open to start a new session"
            )
        return link

    def _call(
        self, method: str, params: Mapping[str, Any] | None = None, *, slow: bool = False
    ) -> Any:
        link = self._require()
        timeout_s = self._slow_call_timeout_s if slow else self._rpc_timeout_s
        try:
            return link.rpc.call(method, params, timeout_s=timeout_s)
        except IpcClosedError as error:
            self._mark_lost(link, str(error))
            raise CameraDisconnectedError(f"acquire went away during {method}: {error}") from None
        except RpcTimeoutError as error:
            raise CameraTimeoutError(f"acquire did not answer {method} in time: {error}") from None
        except RpcError as error:  # includes the exceptions that this side does not know
            raise CameraError(f"acquire failed during {method}: {error}") from None
        except IpcError as error:
            raise CameraError(
                f"the connection to acquire failed during {method}: {error}"
            ) from None

    # --- Connecting ------------------------------------------------------------------------

    def _connect(self) -> _Link:
        try:
            rpc, hello = connect_rpc(
                self._endpoint,
                self._key,
                {"role": "core"},
                channel=RPC_CHANNEL,
                connect_timeout_s=self._connect_timeout_s,
                handshake_timeout_s=self._handshake_timeout_s,
                default_timeout_s=self._rpc_timeout_s,
                max_message_bytes=self._max_rpc_bytes,
                clock=self._clock,
                name="remote-rpc",
            )
        except IpcAuthError as error:
            raise CameraConfigError(f"acquire refused the connection key: {error}") from None
        except IpcError as error:
            raise CameraDisconnectedError(f"cannot reach acquire: {error}") from None
        try:
            session = get_str(hello, "session", "hello")
            instance = get_str(hello, "instance", "hello")
            frames, reply = connect_stream(
                self._endpoint,
                self._key,
                {"session": session},
                channel=FRAMES_CHANNEL,
                window=self._window,
                connect_timeout_s=self._connect_timeout_s,
                handshake_timeout_s=self._handshake_timeout_s,
                max_message_bytes=self._max_frame_bytes,
                clock=self._clock,
                name="remote-frames",
            )
        except (IpcError, CodecError) as error:
            rpc.close("the frame stream did not connect")
            raise CameraDisconnectedError(f"cannot open the frame stream: {error}") from None
        if reply.get("instance") != instance:
            frames.close("acquire restarted during the connection")
            rpc.close("acquire restarted during the connection")
            raise CameraDisconnectedError("acquire restarted while the driver connected")
        return _Link(rpc, frames, instance)

    # --- CameraDriver ----------------------------------------------------------------------

    def open(self) -> CameraInfo:
        """Connect to `acquire` if needed, and open the camera there.

        Raises `CameraDisconnectedError` when `acquire` does not answer within
        `connect_timeout_s`, and `CameraConfigError` when it refuses the connection key. Any
        error that the camera raises while it opens comes back as it is.
        """
        with self._open_lock:
            with self._lock:
                link, state = self._link, self._state
            healthy = (
                state is _State.CONNECTED
                and link is not None
                and not link.rpc.closed
                and not link.frames.closed
            )
            if not healthy:
                if link is not None:
                    link.close("the driver opens a new session")
                new_link = self._connect()
                with self._lock:
                    self._link = new_link
                    self._state = _State.CONNECTED
                    self._active = None
                    self._capturing = False
                    self._epoch = -1
                    self.connections += 1
            try:
                return decode_camera_info(self._call("open", slow=True))
            except CodecError as error:
                raise CameraError(f"acquire sent unreadable camera info: {error}") from None

    def close(self) -> None:
        """Close the camera in `acquire` and end the session. Safe to call twice."""
        with self._lock:
            link, state = self._link, self._state
            self._link = None
            self._state = _State.NEW
            self._active = None
            self._capturing = False
        if link is None:
            return
        if state is _State.CONNECTED and not link.rpc.closed:
            try:
                link.rpc.call("close", timeout_s=self._rpc_timeout_s)
            except (IpcError, CameraError):
                _log.debug("acquire did not confirm the close", exc_info=True)
        link.close("the driver closed")

    def capabilities(self) -> CameraCaps:
        """The limits that the camera reports."""
        try:
            return decode_camera_caps(self._call("capabilities"))
        except CodecError as error:
            raise CameraError(f"acquire sent unreadable capabilities: {error}") from None

    def configure(self, config: StreamConfig) -> ActiveStream:
        """Configure the stream in `acquire`. Frames of every earlier stream become stale."""
        with self._lock:
            self._capturing = False
            self._active = None
        result = as_mapping(
            self._call("configure", {"config": encode_stream_config(config)}, slow=True),
            "configure answer",
        )
        try:
            active = decode_active_stream(result.get("stream"))
            epoch = get_int(result, "epoch", "configure answer")
        except CodecError as error:
            raise CameraError(f"acquire sent an unreadable stream: {error}") from None
        with self._lock:
            self._active = active
            self._epoch = epoch
        return active

    def start(self) -> None:
        """Start the capture. Frames flow to `read_frame`."""
        epoch = self._epoch_of("start", self._call("start", slow=True))
        with self._lock:
            self._epoch = epoch
            self._capturing = True

    def stop(self) -> None:
        """Stop the capture. Frames that are still on the wire are discarded.

        Safe to call when nothing runs, and before `open`.
        """
        with self._lock:
            self._capturing = False
            if self._state is _State.NEW:
                return
        epoch = self._epoch_of("stop", self._call("stop", slow=True))
        with self._lock:
            self._epoch = epoch

    @staticmethod
    def _epoch_of(method: str, answer: Any) -> int:
        try:
            return get_int(as_mapping(answer, f"{method} answer"), "epoch", f"{method} answer")
        except CodecError as error:
            raise CameraError(f"acquire sent an unreadable answer to {method}: {error}") from None

    def move_roi(self, x: int, y: int) -> Roi:
        """Move the ROI of the running stream. Returns the ROI that the camera applied."""
        try:
            return decode_roi(self._call("move_roi", {"x": x, "y": y}))
        except CodecError as error:
            raise CameraError(f"acquire sent an unreadable ROI: {error}") from None

    def read_temperature_c(self) -> float | None:
        """The sensor temperature, or `None` when the camera has no sensor."""
        value = self._call("read_temperature_c")
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CameraError("acquire sent an unreadable temperature")
        return float(value)

    def dropped_frames(self) -> int:
        """The driver's own drop counter in `acquire`."""
        value = self._call("dropped_frames")
        if isinstance(value, bool) or not isinstance(value, int):
            raise CameraError("acquire sent an unreadable drop count")
        return value

    def recover(self, level: RecoveryLevel) -> None:
        """Perform one recovery step in the driver of `acquire`.

        A recovery starts a new capture epoch: the frames of before are history, and the first
        frame after it carries `FrameFlag.RECOVERED`.
        """
        epoch = self._epoch_of("recover", self._call("recover", {"level": int(level)}, slow=True))
        with self._lock:
            self._epoch = epoch

    # --- Frames ----------------------------------------------------------------------------

    def read_frame(self, timeout_s: float) -> Frame:
        """Block for the next frame of the current stream, up to `timeout_s`.

        Raises `CameraTimeoutError` when the time passes, `CameraDisconnectedError` when
        `acquire` is gone, `CameraStateError` when no capture runs, and any camera error that the
        driver in `acquire` raised while it read. Messages of an old stream or epoch are
        discarded.
        """
        started_ns = self._clock.monotonic_ns()
        while True:
            with self._lock:
                link, state = self._link, self._state
                active, capturing, epoch = self._active, self._capturing, self._epoch
            if state is _State.NEW or link is None:
                raise CameraStateError("the camera is not open: call open first")
            if active is None or not capturing:
                if state is _State.LOST:
                    raise CameraDisconnectedError("the connection to acquire was lost")
                raise CameraStateError("read_frame while not capturing")
            remaining_s = timeout_s - (self._clock.monotonic_ns() - started_ns) / NS_PER_S
            if remaining_s <= 0:
                raise CameraTimeoutError(f"no frame within {timeout_s} s")
            try:
                message = link.frames.recv(remaining_s)
            except IpcClosedError as error:
                self._mark_lost(link, str(error))
                raise CameraDisconnectedError(f"the frame stream ended: {error}") from None
            if message is None:
                raise CameraTimeoutError(f"no frame within {timeout_s} s")
            if message.tag != epoch & MAX_TAG:
                self.stale_discarded += 1
                continue
            if message.kind is WireKind.EVENT:
                self._handle_event(message.payload, active)
                continue
            try:
                frame = decode_frame(message.payload)
            except FrameDecodeError as error:
                raise CameraError(f"acquire sent an unreadable frame: {error}") from None
            if frame.stream_id != active.stream_id:
                self.stale_discarded += 1
                continue
            self.frames_received += 1
            if active.config.kind is StreamKind.SNAPSHOT:
                with self._lock:
                    if self._epoch == epoch:
                        self._capturing = False  # one exposure per `start`
            return frame

    def _handle_event(self, payload: memoryview, active: ActiveStream) -> None:
        """Raise the camera error that an event carries, unless it belongs to an old stream."""
        try:
            data = as_mapping(decode_json(payload), "event")
            stream_id = get_int(data, "stream_id", "event")
            error = decode_exception(data.get("error"))
        except CodecError:
            _log.warning("acquire sent an event that the driver cannot read")
            return
        if stream_id != active.stream_id:
            self.stale_discarded += 1
            return
        self.events_received += 1
        if isinstance(error, CameraStateError):
            with self._lock:
                self._capturing = False  # the driver in acquire stopped, as the error says
        raise error

    # --- Beyond the driver interface -------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """The health summary of `acquire` (see `AcquireHealth`). `core` folds it into its own."""
        return dict(as_mapping(self._call("health"), "health"))

    def events(self, after: int = 0) -> EventBatch:
        """The hardware events that the driver in `acquire` reported after number `after`.

        Pass the `last` of the previous batch to get only new events. `lost` counts the events
        that the log dropped before you asked. The call works with or without a running stream.
        """
        try:
            return decode_batch(self._call("events", {"after": after}))
        except CodecError as error:
            raise CameraError(f"acquire sent unreadable events: {error}") from None

    def request_restart(self, reason: str = "requested by core") -> None:
        """Ask `acquire` to exit, so that its supervisor starts a new process.

        The process answers and then exits with `EXIT_RESTART_REQUESTED`. A connection that
        closes before the answer arrives counts as success, because `acquire` is going away
        either way. The driver then reports a disconnect, and `open` reconnects to the new process.
        """
        try:
            self._call("restart", {"reason": reason[:200]})
        except CameraDisconnectedError:
            return

    def ping(self) -> str:
        """The identity of the `acquire` process. Answers even while a slow call runs."""
        return get_str(as_mapping(self._call("ping"), "ping answer"), "instance", "ping answer")
