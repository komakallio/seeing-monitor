"""A typed request-response layer on one connection.

A client calls a method by name with JSON parameters and gets a JSON result back, or an
exception of the right class. Many threads can call at once on one connection, and each call
has its own timeout. A call that times out leaves the connection usable, and a late answer is
dropped.

**Messages.** All messages are UTF-8 JSON:

    request   {"v": 1, "id": 7, "method": "configure", "params": {...}}
    answer    {"v": 1, "id": 7, "ok": true, "result": ...}
    error     {"v": 1, "id": 7, "ok": false, "error": {"type": "CameraStateError", "message": "x"}}

**Errors.** The service encodes an exception that a handler raises with
`seeingmon.services.ipc.codec.encode_exception`, and the client raises the registered class
for it. A class that the client does not know arrives as `RemoteError`. A call fails with
`IpcClosedError` when the connection goes away, and with `RpcTimeoutError` when the answer
takes too long.

**The service.** `RpcService` is the server side. Handlers take the parameters (a dict) and
return a JSON value. `workers` threads run the handlers in the order of arrival. A method named
in `inline` runs on the reading thread of the connection instead, which suits a quick call that
must answer even while a slow handler runs. A service accepts `max_connections` clients at a
time, and a newer connection replaces the oldest.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import queue
import threading
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

from seeingmon.clock import Clock
from seeingmon.services.ipc.client import connect_channel
from seeingmon.services.ipc.codec import (
    DEFAULT_ERRORS,
    CodecError,
    ErrorRegistry,
    as_mapping,
    decode_exception,
    decode_json,
    encode_exception,
    encode_json,
    get_int,
    get_str,
)
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.errors import (
    IpcClosedError,
    IpcError,
    IpcProtocolError,
    RpcError,
    RpcInvalidParamsError,
    RpcMethodNotFoundError,
    RpcTimeoutError,
)
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.server import Accepted
from seeingmon.services.ipc.wire import Wire

RPC_VERSION = 1
RPC_CHANNEL = "rpc"
MAX_METHOD_CHARS = 100
MAX_QUEUED_REQUESTS = 256
DEFAULT_RPC_BYTES = 1024 * 1024

Handler = Callable[[Mapping[str, Any]], Any]

_log = logging.getLogger(__name__)


# --- The client ----------------------------------------------------------------------------


@dataclass(slots=True)
class _Pending:
    event: threading.Event
    result: Any = None
    error: Exception | None = None


class RpcClient:
    """Call methods on a service. Safe to use from many threads at once."""

    def __init__(
        self,
        wire: Wire,
        *,
        registry: ErrorRegistry = DEFAULT_ERRORS,
        default_timeout_s: float = 30.0,
        name: str = "rpc",
    ) -> None:
        self._wire = wire
        self._registry = registry
        self._default_timeout_s = default_timeout_s
        self._lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._ids = itertools.count(1)
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, name=f"{name}-reader", daemon=True)
        self._reader.start()

    @property
    def closed(self) -> bool:
        """Whether the connection is closed. Every later call raises `IpcClosedError`."""
        return self._closed or self._wire.closed

    @property
    def close_reason(self) -> str:
        """Why the connection closed, in words."""
        return self._wire.reason

    def call(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Call `method` and return its result.

        Raises the exception that the handler raised (a registered class, or `RemoteError`),
        `RpcTimeoutError` after `timeout_s` (default: the client's default), and
        `IpcClosedError` when the connection is or goes away.
        """
        timeout = self._default_timeout_s if timeout_s is None else timeout_s
        if timeout <= 0:
            raise ValueError("timeout_s must be positive")
        pending = _Pending(threading.Event())
        with self._lock:
            if self._closed or self._wire.closed:
                raise IpcClosedError(self._wire.reason)
            request_id = next(self._ids)
            self._pending[request_id] = pending
        try:
            message = encode_json(
                {
                    "v": RPC_VERSION,
                    "id": request_id,
                    "method": method,
                    "params": None if params is None else dict(params),
                }
            )
            self._wire.send(message)
        except BaseException:
            self._forget(request_id)
            raise
        if not pending.event.wait(timeout):
            self._forget(request_id)
            raise RpcTimeoutError(f"{method} did not answer within {timeout:g} s")
        if pending.error is not None:
            raise pending.error
        return pending.result

    def _forget(self, request_id: int) -> None:
        with self._lock:
            self._pending.pop(request_id, None)

    def close(self, reason: str = "the client closed the connection") -> None:
        """Close the connection. Calls that wait for an answer fail with `IpcClosedError`."""
        self._wire.close(reason)
        self._fail_all()
        if self._reader is not threading.current_thread():
            self._reader.join(2.0)

    def __enter__(self) -> RpcClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _fail_all(self) -> None:
        with self._lock:
            self._closed = True
            pending, self._pending = list(self._pending.values()), {}
        for item in pending:
            item.error = IpcClosedError(self._wire.reason)
            item.event.set()

    def _read_loop(self) -> None:
        try:
            while True:
                raw = self._wire.recv(None)
                if raw is not None:
                    self._dispatch(raw)
        except IpcClosedError:
            pass
        except IpcError as error:
            _log.warning("%s: closing after a bad message: %s", self._wire.name, error)
            self._wire.close("the server sent a message that the client cannot read")
        except Exception:
            _log.exception("%s: the reader failed", self._wire.name)
            self._wire.close("the reader failed")
        finally:
            self._fail_all()

    def _dispatch(self, raw: bytes) -> None:
        data = as_mapping(decode_json(raw), "answer")
        request_id = data.get("id")
        if request_id is None:
            error = data.get("error")
            detail = error.get("message") if isinstance(error, Mapping) else None
            raise IpcProtocolError(f"the server rejected the connection: {detail}")
        request_id = get_int(data, "id", "answer")
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            return  # the call timed out, and this answer came late
        if data.get("ok") is True:
            pending.result = data.get("result")
        elif data.get("ok") is False:
            try:
                pending.error = decode_exception(data.get("error"), self._registry)
            except CodecError as error:
                pending.error = RpcError(f"the server sent an unreadable error: {error}")
        else:
            pending.error = RpcError("the server sent an answer without ok")
        pending.event.set()


def connect_rpc(
    endpoint: Endpoint,
    key: ConnectionKey,
    params: Mapping[str, Any] | None = None,
    *,
    channel: str = RPC_CHANNEL,
    connect_timeout_s: float = 5.0,
    handshake_timeout_s: float = 5.0,
    default_timeout_s: float = 30.0,
    max_message_bytes: int = DEFAULT_RPC_BYTES,
    registry: ErrorRegistry = DEFAULT_ERRORS,
    clock: Clock | None = None,
    name: str = "rpc",
) -> tuple[RpcClient, Mapping[str, Any]]:
    """Connect to an `RpcService` and return the client and the service's hello reply."""
    wire, reply = connect_channel(
        endpoint,
        key,
        channel,
        params,
        connect_timeout_s=connect_timeout_s,
        handshake_timeout_s=handshake_timeout_s,
        max_message_bytes=max_message_bytes,
        clock=clock,
    )
    wire.name = name
    return RpcClient(wire, registry=registry, default_timeout_s=default_timeout_s, name=name), reply


# --- The service ---------------------------------------------------------------------------


class RpcConnection:
    """One client of an `RpcService`.

    `params` holds the hello parameters that the client sent, and `context` is free for the
    application, for example to remember a session.
    """

    def __init__(self, service: RpcService, wire: Wire, params: Mapping[str, Any], number: int):
        self._service = service
        self.wire = wire
        self.params = params
        self.number = number
        self.context: dict[str, Any] = {}
        self._finished = False

    @property
    def closed(self) -> bool:
        """Whether the connection is closed."""
        return self.wire.closed

    def close(self, reason: str = "the service closed the connection") -> None:
        """Close the connection. The service then calls `on_disconnect` once."""
        self.wire.close(reason)

    def _send(self, message: Mapping[str, Any]) -> None:
        with contextlib.suppress(IpcClosedError):  # the client may leave before the answer is ready
            self.wire.send(encode_json(message))


class RpcService:
    """The server side of the request-response layer. Register it as a channel of an `IpcServer`."""

    def __init__(
        self,
        handlers: Mapping[str, Handler],
        *,
        registry: ErrorRegistry = DEFAULT_ERRORS,
        workers: int = 1,
        worker_name: str = "rpc-worker",
        inline: Collection[str] = (),
        max_connections: int = 1,
        max_message_bytes: int = DEFAULT_RPC_BYTES,
        on_connect: Callable[[RpcConnection], Mapping[str, Any] | None] | None = None,
        on_disconnect: Callable[[RpcConnection], None] | None = None,
    ) -> None:
        self._handlers = dict(handlers)
        self._registry = registry
        self._worker_count = workers
        self._worker_name = worker_name
        self._inline = frozenset(inline)
        self._max_connections = max_connections
        self._max_message_bytes = max_message_bytes
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._lock = threading.Lock()
        self._connections: dict[int, RpcConnection] = {}
        self._numbers = itertools.count(1)
        self._jobs: queue.Queue[Callable[[], None]] = queue.Queue(MAX_QUEUED_REQUESTS)
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()

    @property
    def connections(self) -> list[RpcConnection]:
        """The clients that are connected now."""
        with self._lock:
            return list(self._connections.values())

    @property
    def worker_threads(self) -> list[threading.Thread]:
        """The threads that run the handlers."""
        return list(self._threads)

    def start(self) -> None:
        """Start the worker threads."""
        if self._threads:
            raise RuntimeError("the service already started")
        for index in range(self._worker_count):
            name = self._worker_name if self._worker_count == 1 else f"{self._worker_name}-{index}"
            thread = threading.Thread(target=self._work, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        """Close every connection and stop the workers."""
        self._stopping.set()
        for connection in self.connections:
            connection.close("the service stopped")
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(2.0)
        self._threads = []

    def submit(self, job: Callable[[], None]) -> None:
        """Queue a job for a worker thread, after the requests that are already queued.

        An application uses this to run a cleanup on the thread that owns a resource.
        """
        self._jobs.put(job)

    # --- ChannelHandler ---

    def accept(self, wire: Wire, params: Mapping[str, Any]) -> Accepted:
        """Take a connection that asked for this channel. Called by the `IpcServer`."""
        if self._stopping.is_set():
            raise IpcProtocolError("the service is stopping")
        wire.max_message_bytes = self._max_message_bytes
        connection = RpcConnection(self, wire, params, next(self._numbers))
        extra: Mapping[str, Any] | None = None
        if self._on_connect is not None:
            extra = self._on_connect(connection)  # may refuse by raising; nothing registered yet
        replaced: list[RpcConnection] = []
        with self._lock:
            while len(self._connections) >= self._max_connections:
                oldest = min(self._connections)
                replaced.append(self._connections.pop(oldest))
            self._connections[connection.number] = connection
        for old in replaced:
            old.close("a newer connection replaced this one")
        reply = {"connection": connection.number, **(extra or {})}

        def activate() -> None:
            threading.Thread(
                target=self._read_loop,
                args=(connection,),
                name=f"{self._worker_name}-reader-{connection.number}",
                daemon=True,
            ).start()

        def abort() -> None:
            connection.close("the client left during the hello")
            self._finish(connection)

        return Accepted(reply=reply, activate=activate, abort=abort)

    # --- Internals ---

    def _finish(self, connection: RpcConnection) -> None:
        with self._lock:
            if connection._finished:
                return
            connection._finished = True
            self._connections.pop(connection.number, None)
        if self._on_disconnect is not None:
            try:
                self._on_disconnect(connection)
            except Exception:
                _log.exception("%s: on_disconnect failed", self._worker_name)

    def _read_loop(self, connection: RpcConnection) -> None:
        try:
            while True:
                raw = connection.wire.recv(None)
                if raw is not None and not self._handle(connection, raw):
                    break
        except IpcClosedError:
            pass
        except IpcProtocolError as error:
            _log.warning("%s: closing a connection: %s", self._worker_name, error)
        except Exception:
            _log.exception("%s: a reader failed", self._worker_name)
        finally:
            connection.close()
            self._finish(connection)

    def _handle(self, connection: RpcConnection, raw: bytes) -> bool:
        """Decode one request and run it. Returns `False` when the connection must close."""
        try:
            data = as_mapping(decode_json(raw), "request")
            if get_int(data, "v", "request") != RPC_VERSION:
                raise CodecError("request.v is not a version that this service speaks")
            request_id = get_int(data, "id", "request")
            method = get_str(data, "method", "request")
            if len(method) > MAX_METHOD_CHARS:
                raise CodecError("request.method is too long")
            raw_params = data.get("params")
            params = {} if raw_params is None else as_mapping(raw_params, "request.params")
        except CodecError as error:
            connection._send(
                {
                    "v": RPC_VERSION,
                    "id": None,
                    "ok": False,
                    "error": {"type": "InvalidParams", "message": str(error)},
                }
            )
            return False  # the peer does not speak the protocol, so end the session

        def job() -> None:
            self._run(connection, request_id, method, params)

        if method in self._inline:
            job()
            return True
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            connection._send(
                {
                    "v": RPC_VERSION,
                    "id": request_id,
                    "ok": False,
                    "error": {"type": "InternalError", "message": "the service is busy"},
                }
            )
        return True

    def _run(
        self, connection: RpcConnection, request_id: int, method: str, params: Mapping[str, Any]
    ) -> None:
        handler = self._handlers.get(method)
        try:
            if handler is None:
                raise RpcMethodNotFoundError(f"no method named {method!r}")
            try:
                result = handler(params)
            except CodecError as error:
                raise RpcInvalidParamsError(str(error)) from error
            body = encode_json({"v": RPC_VERSION, "id": request_id, "ok": True, "result": result})
            if len(body) > connection.wire.max_message_bytes:
                raise RpcError("the result exceeds the message limit")
        except Exception as error:
            if self._registry.name_of(error) is None and not isinstance(error, RpcError):
                _log.exception("%s: %s failed", self._worker_name, method)
            connection._send(
                {
                    "v": RPC_VERSION,
                    "id": request_id,
                    "ok": False,
                    "error": encode_exception(error, self._registry),
                }
            )
            return
        with contextlib.suppress(IpcClosedError):  # the client may leave before the answer is ready
            connection.wire.send(body)

    def _work(self) -> None:
        while not self._stopping.is_set():
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                job()
            except Exception:
                _log.exception("%s: a job failed", self._worker_name)
