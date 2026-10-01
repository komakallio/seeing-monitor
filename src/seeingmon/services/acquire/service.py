"""The `acquire` service: one camera driver, a capture thread, and a stream to `core`.

`acquire` owns the camera. It runs four threads:

- The **capture thread** calls `driver.read_frame` in a loop, at raised priority where the
  platform allows it. It stamps each frame (see `seeingmon.services.acquire.timing`), counts the
  frames that were lost (see `seeingmon.services.acquire.drops`), and puts the frame on a bounded
  queue. It analyzes nothing, and it never waits for the network.
- The **control thread** is the worker of the RPC service. It serves the calls that mirror
  `CameraDriver`: `open`, `close`, `capabilities`, `configure`, `start`, `stop`, `move_roi`,
  `read_temperature_c`, `dropped_frames`, and `recover`. Two more calls answer at once on the
  connection's reading thread: `ping` and `health`.
- The **sender thread** takes frames from the queue and sends them over the stream channel, as
  far as the receiver's window allows. A full queue drops its oldest frame and counts it.
- The **watchdog thread** checks the guard around the driver calls, checks that the other
  threads live, sends the heartbeat to systemd (only when the process runs under it), and logs a
  health summary.

**Hangs.** Every driver call runs under a deadline. The vendor SDK is a closed binary, and a call
can block for good after a USB fault. When a call outlives its deadline, the guard calls its
`on_hang` hook, which ends the process in production, and systemd starts a new one. The guard is
a `seeingmon.hardware.asi.watchdog.CallWatchdog`.

**One client.** `core` is the only client. It opens two connections that share a session: the
RPC connection, and the frame stream, which presents the session from the RPC hello. A new RPC
connection replaces the old session: `acquire` stops the capture, discards the queue, and keeps
the camera open. When the RPC connection closes, `acquire` stops the capture too, so a camera
that nobody reads does not stream.

**Epochs.** Every `configure`, `start`, `stop`, `close`, and `recover` starts a new capture
epoch. The sender tags each message with the epoch of the frame, and `core` discards a message
whose epoch is not the one that the last call returned. So no frame of an old stream reaches the
scheduler as a frame of the new one, even when it was already on the wire.

**Threads and the driver.** A driver that is not thread-safe is called by one thread at a time
(see `seeingmon.services.acquire.gate`). A driver that is safe, such as `asi`, is called from
both threads at once, so a `stop` can interrupt a long exposure.
"""

from __future__ import annotations

import contextlib
import logging
import os
import secrets
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import Any, Protocol

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.drivers.base import (
    CameraDriver,
    CameraError,
    CameraStateError,
    CameraTimeoutError,
)
from seeingmon.frames import (
    ActiveStream,
    Frame,
    FrameFlag,
    StreamConfig,
    StreamKind,
    TimeQuality,
    encode_frame,
)
from seeingmon.services.acquire.drops import DropAccountant
from seeingmon.services.acquire.gate import DriverGate
from seeingmon.services.acquire.health import AcquireHealth
from seeingmon.services.acquire.notify import SystemdNotifier
from seeingmon.services.acquire.priority import raise_current_thread_priority
from seeingmon.services.acquire.queue import FrameQueue, QueueItem
from seeingmon.services.acquire.timing import StreamTiming, TimeStamper, TimingConfig
from seeingmon.services.config import AcquireSettings, ServicesConfig
from seeingmon.services.ipc.codec import (
    decode_recovery_level,
    decode_stream_config,
    encode_active_stream,
    encode_camera_caps,
    encode_camera_info,
    encode_exception,
    encode_json,
    encode_roi,
    get_int,
)
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.errors import IpcClosedError, IpcProtocolError
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.rpc import RpcConnection, RpcService
from seeingmon.services.ipc.server import IpcServer
from seeingmon.services.ipc.stream import (
    StreamKind as WireKind,
)
from seeingmon.services.ipc.stream import (
    StreamSender,
    StreamService,
    StreamWindow,
)

RPC_CHANNEL = "rpc"
FRAMES_CHANNEL = "frames"
EXIT_THREAD_DIED = 71
INSTANCE_BYTES = 8
MAX_TAG = 2**32 - 1
STALL_PERIODS = 5
STALL_FLOOR_S = 3.0

_log = logging.getLogger(__name__)
_REAL_CLOCK = SystemClock()


class CallGuard(Protocol):
    """A deadline around a driver call. `seeingmon.hardware.asi.watchdog.CallWatchdog` fits."""

    def guard(self, name: str, timeout_s: float) -> AbstractContextManager[None]:
        """Run a block under a deadline of `timeout_s` seconds."""
        ...

    def check(self) -> object:
        """Report each call that is past its deadline."""
        ...


def default_guard(clock: Clock) -> CallGuard:
    """The production guard: a `CallWatchdog` that ends the process when a call hangs.

    The watchdog thread of the service calls `check`, so the guard needs no thread of its own.
    """
    from seeingmon.hardware.asi.watchdog import CallWatchdog, exit_process_on_hang

    return CallWatchdog(clock, exit_process_on_hang)


def exit_on_fatal(reason: str) -> None:
    """End the process at once, as the hang handler does, so that systemd starts a new one."""
    print(f"acquire: fatal: {reason}", file=sys.stderr, flush=True)
    os._exit(EXIT_THREAD_DIED)


@dataclass(slots=True)
class _Session:
    id: str
    connection: RpcConnection
    sender: StreamSender | None = None


@dataclass(slots=True)
class _Counters:
    frames_captured: int = 0
    frames_sent: int = 0
    flow_stalls: int = 0
    read_timeouts: int = 0
    read_errors: int = 0
    internal_errors: int = 0
    last_error: str | None = None
    last_frame_mono_ns: int | None = None


def timing_config(settings: AcquireSettings) -> TimingConfig:
    """The stamper settings that the `[services.acquire]` section names."""
    return TimingConfig(
        window=settings.fit_window,
        warmup=settings.fit_warmup,
        latency_s=settings.latency_s,
        latency_sigma_s=settings.latency_sigma_s,
        arrival_jitter_s=settings.arrival_jitter_s,
        unknown_clock_error_s=settings.unknown_clock_error_s,
        invalid_clock_error_s=settings.invalid_clock_error_s,
        outlier_sigmas=settings.outlier_sigmas,
        outlier_floor_s=settings.outlier_floor_s,
        step_frames=settings.step_frames,
    )


class AcquireService:
    """The `acquire` process. Call `run`, or `start` and `stop` from a test."""

    def __init__(
        self,
        driver: CameraDriver,
        clock: Clock,
        endpoint: Endpoint,
        key: ConnectionKey,
        settings: ServicesConfig,
        *,
        guard: CallGuard | None = None,
        notifier: SystemdNotifier | None = None,
        priority_hook: Callable[[], str] | None = None,
        on_fatal: Callable[[str], None] | None = None,
    ) -> None:
        self._driver = driver
        self._clock = clock
        self._settings = settings
        self._cfg = settings.acquire
        self._guard = guard if guard is not None else default_guard(clock)
        self._notifier = notifier if notifier is not None else SystemdNotifier()
        if priority_hook is not None:
            self._priority_hook = priority_hook
        elif self._cfg.raise_priority:
            self._priority_hook = raise_current_thread_priority
        else:
            self._priority_hook = lambda: "disabled"
        self._on_fatal = on_fatal or exit_on_fatal
        self.instance = secrets.token_hex(INSTANCE_BYTES)
        self._started_ns = clock.monotonic_ns()

        self._gate: DriverGate | None = None if self._driver_is_thread_safe() else DriverGate()
        self._queue = FrameQueue(self._cfg.queue_depth, self._cfg.queue_max_bytes)
        self._drops = DropAccountant(gap_factor=self._cfg.gap_factor)
        self._stamper = TimeStamper(clock, timing_config(self._cfg))
        self._counters = _Counters()

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._capturing_event = threading.Event()
        self._opened = False
        self._info: Any = None
        self._active: ActiveStream | None = None
        self._capturing = False
        self._epoch = 0
        self._disruptions = 0
        self._session: _Session | None = None
        self._recovered_pending = False
        self._timing_stream_id: int | None = None
        self._error_repeats: dict[str, int] = {}
        self._priority = "not set"
        self._started = False
        self._stopped = False
        self._fatal_reason: str | None = None
        self._exit_reason = "stopped"
        self._frame_rate_hz = 0.0
        self._capture_since_ns = 0
        self._threads: dict[str, threading.Thread] = {}

        self._rpc = RpcService(
            self._handlers(),
            workers=1,
            worker_name="acquire-control",
            inline={"ping", "health"},
            max_connections=1,
            max_message_bytes=settings.max_rpc_bytes,
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
        )
        self._stream = StreamService(
            self._on_sender,
            max_window=StreamWindow(settings.stream_window_messages, settings.stream_window_bytes),
            max_message_bytes=settings.max_frame_bytes,
            validate=self._validate_stream,
            name="acquire-frames",
        )
        self._server = IpcServer(
            endpoint,
            key,
            {RPC_CHANNEL: self._rpc, FRAMES_CHANNEL: self._stream},
            handshake_timeout_s=settings.handshake_timeout_s,
            name="acquire",
        )

    # --- Setup -----------------------------------------------------------------------------

    def _driver_is_thread_safe(self) -> bool:
        mode = self._cfg.driver_threads
        if mode != "auto":
            return mode == "concurrent"
        return bool(getattr(self._driver, "thread_safe", False)) or self._driver.name == "asi"

    @property
    def endpoint(self) -> Endpoint:
        """Where the service listens. For a TCP endpoint with a free port, the real port."""
        return self._server.endpoint

    @property
    def exit_reason(self) -> str:
        """Why the service stopped, in words."""
        return self._fatal_reason or self._exit_reason

    def start(self) -> Endpoint:
        """Bind the address, and start the threads. Returns the address that clients use."""
        if self._started:
            raise RuntimeError("the service already started")
        self._started = True
        self._rpc.start()
        endpoint = self._server.start()
        for name, target in (
            ("acquire-capture", self._capture_loop),
            ("acquire-sender", self._sender_loop),
            ("acquire-watchdog", self._watchdog_loop),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._threads[name] = thread
            thread.start()
        self._notifier.ready(f"listening, driver {self._driver.name}")
        _log.info(
            "acquire %s listens at %s (driver %s)", self.instance, endpoint, self._driver.name
        )
        return endpoint

    def request_stop(self, reason: str = "stopped") -> None:
        """Ask the service to stop. Safe to call from a signal handler or any thread."""
        self._exit_reason = reason
        self._stop.set()

    def wait(self, timeout_s: float | None = None) -> bool:
        """Wait until a stop was requested. Returns whether it was."""
        return self._stop.wait(timeout_s)

    def run(self) -> int:
        """Start, wait for a stop request, and stop. Returns the exit code of the process."""
        self.start()
        try:
            while not self._stop.wait(0.5):
                pass
        finally:
            self.stop()
        return 0 if self._fatal_reason is None else EXIT_THREAD_DIED

    def stop(self) -> None:
        """Stop the threads, close the connections, and release the camera. Safe to call twice."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        self._stop.set()
        self._notifier.stopping()
        self._set_capturing(False)
        self._server.stop()
        with self._lock:
            session = self._session
            self._session = None
        if session is not None and session.sender is not None:
            session.sender.close()
        self._rpc.stop()
        for thread in self._threads.values():
            if thread is not threading.current_thread():
                thread.join(2.0)
        self._notifier.close()
        capture = self._threads.get("acquire-capture")
        if capture is None or not capture.is_alive():
            self._release_driver()
        else:
            _log.warning("the capture thread is still in a driver call, so the driver stays open")

    def _release_driver(self) -> None:
        """Stop the capture and close the camera, as far as the driver lets us."""
        for name, call in (("stop", self._driver.stop), ("close", self._driver.close)):
            try:
                with self._guard.guard(name, self._timeout_for(name)):
                    call()
            except Exception:
                _log.warning("the driver did not %s cleanly", name, exc_info=True)

    # --- Calls into the driver -------------------------------------------------------------

    def _timeout_for(self, name: str) -> float:
        timeouts = self._cfg.call_timeouts
        return {
            "open": timeouts.open_s,
            "close": timeouts.close_s,
            "capabilities": timeouts.capabilities_s,
            "configure": timeouts.configure_s,
            "start": timeouts.start_s,
            "stop": timeouts.stop_s,
            "move_roi": timeouts.move_roi_s,
            "read_temperature_c": timeouts.temperature_s,
            "dropped_frames": timeouts.dropped_s,
            "recover": timeouts.recover_s,
        }[name]

    @contextlib.contextmanager
    def _exclusive(self) -> Iterator[None]:
        """Hold the driver for a control call, unless the driver is thread-safe."""
        if self._gate is None:
            yield
        else:
            with self._gate.control():
                yield

    @contextlib.contextmanager
    def _disruptive(self) -> Iterator[None]:
        """Hold the driver for a call that can stop or restart the capture.

        The capture thread starts no new read while such a call runs, and a read that is in
        flight belongs to history, because the call starts a new epoch. A driver that stops its
        capture during `recover` raises `CameraStateError` from the read in flight, and that
        error must not end the capture.
        """
        with self._exclusive():
            with self._lock:
                self._disruptions += 1
                self._epoch += 1
            try:
                yield
            finally:
                with self._lock:
                    self._disruptions -= 1

    def _call(self, name: str, call: Callable[..., Any], *args: Any) -> Any:
        """Run one driver call under its deadline."""
        with self._guard.guard(name, self._timeout_for(name)):
            return call(*args)

    def _set_capturing(self, value: bool, epoch: int | None = None) -> None:
        """Set the capture flag. With `epoch`, only if no control call has intervened."""
        with self._lock:
            if epoch is not None and self._epoch != epoch:
                return
            self._capturing = value
            if value:
                self._capturing_event.set()
            else:
                self._capturing_event.clear()

    def _is_current(self, epoch: int) -> bool:
        """Whether the stream is still capturing in this epoch."""
        with self._lock:
            return self._capturing and self._epoch == epoch

    def _new_epoch(self) -> int:
        with self._lock:
            self._epoch += 1
            return self._epoch

    # --- The RPC methods -------------------------------------------------------------------

    def _handlers(self) -> dict[str, Callable[[Mapping[str, Any]], Any]]:
        return {
            "open": self._h_open,
            "close": self._h_close,
            "capabilities": self._h_capabilities,
            "configure": self._h_configure,
            "start": self._h_start,
            "stop": self._h_stop,
            "move_roi": self._h_move_roi,
            "read_temperature_c": self._h_temperature,
            "dropped_frames": self._h_dropped,
            "recover": self._h_recover,
            "ping": self._h_ping,
            "health": self._h_health,
        }

    def _h_ping(self, params: Mapping[str, Any]) -> Any:
        return {"instance": self.instance, "driver": self._driver.name}

    def _h_health(self, params: Mapping[str, Any]) -> Any:
        return self.health().to_json()

    def _h_open(self, params: Mapping[str, Any]) -> Any:
        with self._exclusive():
            if self._opened:  # a second `open`, from a new session, finds the camera open
                return encode_camera_info(self._info)
            info = self._call("open", self._driver.open)
            with self._lock:
                self._opened = True
                self._info = info
        return encode_camera_info(info)

    def _h_close(self, params: Mapping[str, Any]) -> Any:
        with self._disruptive():
            self._set_capturing(False)
            self._queue.clear()
            with self._lock:
                self._active = None
            try:
                self._call("close", self._driver.close)
            finally:
                with self._lock:
                    self._opened = False
                    self._info = None
        return None

    def _h_capabilities(self, params: Mapping[str, Any]) -> Any:
        with self._exclusive():
            return encode_camera_caps(self._call("capabilities", self._driver.capabilities))

    def _stream_timing(self, active: ActiveStream) -> StreamTiming:
        exposure_s = active.config.exposure_us / 1e6
        period_s = active.frame_period_s or max(exposure_s, self._cfg.default_frame_period_s)
        return StreamTiming(exposure_s=exposure_s, period_s=max(period_s, exposure_s))

    def _h_configure(self, params: Mapping[str, Any]) -> Any:
        config: StreamConfig = decode_stream_config(params.get("config"))
        with self._disruptive():
            self._set_capturing(False)
            self._queue.clear()
            with self._lock:
                self._active = None
            active = self._call("configure", self._driver.configure, config)
            self._stamper.configure(self._stream_timing(active))
            self._drops.reset(active.frame_period_s)
            self._timing_stream_id = active.stream_id
            with self._lock:
                self._active = active
            epoch = self._new_epoch()
        return {"stream": encode_active_stream(active), "epoch": epoch}

    def _h_start(self, params: Mapping[str, Any]) -> Any:
        with self._disruptive():
            with self._lock:
                active = self._active
            if active is None:
                raise CameraStateError("start before configure")
            self._call("start", self._driver.start)
            self._stamper.reset()
            self._drops.reset(active.frame_period_s)
            self._queue.clear()
            epoch = self._new_epoch()
            self._capture_since_ns = self._clock.monotonic_ns()
            self._set_capturing(True, epoch)
        return {"epoch": epoch}

    def _h_stop(self, params: Mapping[str, Any]) -> Any:
        with self._disruptive():
            self._set_capturing(False)
            try:
                self._call("stop", self._driver.stop)
            finally:
                self._queue.clear()
            epoch = self._new_epoch()
        return {"epoch": epoch}

    def _h_move_roi(self, params: Mapping[str, Any]) -> Any:
        x = get_int(params, "x", "move_roi")
        y = get_int(params, "y", "move_roi")
        with self._exclusive():
            return encode_roi(self._call("move_roi", self._driver.move_roi, x, y))

    def _h_temperature(self, params: Mapping[str, Any]) -> Any:
        with self._exclusive():
            return self._call("read_temperature_c", self._driver.read_temperature_c)

    def _h_dropped(self, params: Mapping[str, Any]) -> Any:
        with self._exclusive():
            return int(self._call("dropped_frames", self._driver.dropped_frames))

    def _h_recover(self, params: Mapping[str, Any]) -> Any:
        level = decode_recovery_level(params.get("level"))
        with self._disruptive():
            self._call("recover", self._driver.recover, level)
            with self._lock:
                self._recovered_pending = True
            # A recovery is a discontinuity, like a restart of the stream: the timing and the
            # drop counts start again, and the frames of before are history. The new epoch also
            # makes the capture thread ignore what a read in flight reports while the driver
            # stops and restarts its capture.
            self._stamper.reset()
            active = self._active
            self._drops.reset(None if active is None else active.frame_period_s)
            self._queue.clear()
            epoch = self._new_epoch()
        return {"epoch": epoch}

    # --- Sessions --------------------------------------------------------------------------

    def _on_connect(self, connection: RpcConnection) -> Mapping[str, Any]:
        """A client connected: its session replaces the old one."""
        session = _Session(secrets.token_hex(INSTANCE_BYTES), connection)
        with self._lock:
            previous, self._session = self._session, session
        if previous is not None and previous.sender is not None:
            previous.sender.close("a new client took over")
        # The cleanup runs on the control thread, after the calls that are already queued and
        # before any call of the new client, because the client can send only after the reply.
        self._rpc.submit(lambda: self._end_stream_for_new_client("a new client connected"))
        return {"session": session.id, "instance": self.instance, "driver": self._driver.name}

    def _on_disconnect(self, connection: RpcConnection) -> None:
        """A client left. If it was the current one, nobody reads the camera any more."""
        with self._lock:
            session = self._session
            if session is None or session.connection is not connection:
                return  # an old connection, which a newer one replaced
            self._session = None
        if session.sender is not None:
            session.sender.close("the client closed the control connection")
        self._rpc.submit(lambda: self._end_stream_for_new_client("the client left"))

    def _end_stream_for_new_client(self, reason: str) -> None:
        """Stop the capture and discard the queue. The camera stays open."""
        with self._disruptive():
            was_capturing = self._capturing
            self._set_capturing(False)
            self._queue.clear()
            with self._lock:
                self._active = None
            if was_capturing:
                try:
                    self._call("stop", self._driver.stop)
                except Exception:
                    _log.warning("the driver did not stop after %s", reason, exc_info=True)
        _log.info("capture ended: %s", reason)

    def _validate_stream(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            session = self._session
        if session is None or params.get("session") != session.id:
            raise IpcProtocolError("the frame stream does not belong to an open session")
        return {"instance": self.instance}

    def _on_sender(self, sender: StreamSender, params: Mapping[str, Any]) -> None:
        with self._lock:
            session = self._session
            if session is None or params.get("session") != session.id:
                stale = True
                previous = None
            else:
                stale = False
                previous, session.sender = session.sender, sender
        if stale:
            sender.close("the session ended")
        elif previous is not None:
            previous.close("a new stream replaced this one")

    # --- The capture thread ----------------------------------------------------------------

    def _read_timeout(self, stream: ActiveStream) -> float:
        exposure_s = stream.config.exposure_us / 1e6
        period_s = max(stream.frame_period_s or self._cfg.default_frame_period_s, exposure_s)
        return period_s * self._cfg.read_timeout_factor + self._cfg.read_timeout_margin_s

    def _capture_loop(self) -> None:
        try:
            self._priority = self._priority_hook()
            _log.info("capture thread priority: %s", self._priority)
        except Exception:
            _log.warning("could not raise the capture thread priority", exc_info=True)
        while not self._stop.is_set():
            try:
                self._capture_once()
            except Exception:
                self._counters.internal_errors += 1
                _log.exception("the capture loop failed")
                self._stop.wait(self._cfg.error_backoff_s or 0.05)

    def _capture_once(self) -> None:
        if not self._capturing_event.wait(0.1):
            return
        with self._lock:
            paused = self._disruptions > 0
            stream, epoch = self._active, self._epoch
            if not paused and (not self._capturing or stream is None):
                self._capturing_event.clear()
                return
        if paused or stream is None:  # a call that can stop the capture runs: wait for it
            self._stop.wait(0.005)
            return
        timeout_s = self._read_timeout(stream)
        frame: Frame | None = None
        error: CameraError | None = None
        read = self._gate.read(self._stop.is_set) if self._gate else contextlib.nullcontext(True)
        with read as admitted:
            if not admitted:
                return
            if not self._is_current(epoch):
                return  # a control call came in while this thread waited for the gate
            try:
                grace_s = self._cfg.call_timeouts.read_grace_s
                with self._guard.guard("read_frame", timeout_s + grace_s):
                    frame = self._driver.read_frame(timeout_s)
                seen_utc_ns = self._clock.utc_ns()
                seen_mono_ns = self._clock.monotonic_ns()
            except CameraError as caught:
                error = caught
            except Exception as unexpected:  # a driver bug must not kill the capture thread
                self._counters.internal_errors += 1
                _log.exception("the driver raised an unexpected exception")
                error = CameraError(f"unexpected {type(unexpected).__name__}: {unexpected}")
        if error is not None:
            self._on_read_error(error, stream, epoch)
        elif frame is not None:
            self._on_frame(frame, seen_utc_ns, seen_mono_ns, stream, epoch)

    def _on_read_error(self, error: CameraError, stream: ActiveStream, epoch: int) -> None:
        with self._lock:
            if self._epoch != epoch:
                return  # a control call ended the stream, and the error belongs to history
        if isinstance(error, CameraTimeoutError):
            self._counters.read_timeouts += 1
        else:
            self._counters.read_errors += 1
        self._counters.last_error = f"{type(error).__name__}: {error}"
        if isinstance(error, CameraStateError):
            self._set_capturing(False, epoch)  # the driver stopped, so wait for the next `start`
        self._post_error(error, stream, epoch)
        self._stop.wait(self._cfg.error_backoff_s)

    def _post_error(self, error: CameraError, stream: ActiveStream, epoch: int) -> None:
        """Tell the client, in order with the frames. A repeated error takes one place."""
        key = f"{type(error).__name__}:{error}"
        repeats = self._error_repeats.get(key, 0) + 1
        self._error_repeats = {key: repeats}
        payload = encode_json(
            {
                "kind": "error",
                "stream_id": stream.stream_id,
                "repeat": repeats,
                "error": encode_exception(error),
            }
        )
        self._queue.put_event(epoch, payload, key)

    def _keeps_driver_time(self, frame: Frame) -> bool:
        source = self._cfg.time_source
        if source == "driver":
            return True
        return source == "auto" and (
            frame.t_quality is TimeQuality.EXACT or bool(frame.flags & FrameFlag.REPLAYED)
        )

    def _on_frame(
        self,
        frame: Frame,
        seen_utc_ns: int,
        seen_mono_ns: int,
        stream: ActiveStream,
        epoch: int,
    ) -> None:
        with self._lock:
            if not self._is_current(epoch):
                return  # the frame belongs to a stream that a control call ended
            recovered = self._recovered_pending
            self._recovered_pending = False
        self._error_repeats = {}
        if frame.stream_id != self._timing_stream_id:  # the driver changed the stream itself
            self._timing_stream_id = frame.stream_id
            self._stamper.reset()
            self._drops.reset(stream.frame_period_s)
        arrival_ns = frame.t_arrival_ns if frame.t_arrival_ns > 0 else seen_utc_ns
        lost = self._drops.frame_arrived(
            seen_mono_ns, frame.dropped_before, period_s=self._stamper.period_s
        )
        flags = frame.flags | (FrameFlag.RECOVERED if recovered else FrameFlag.NONE)
        if self._keeps_driver_time(frame):
            stamped = replace(frame, dropped_before=lost, flags=flags)
        else:
            frame_time = self._stamper.stamp(arrival_ns, lost)
            stamped = replace(
                frame,
                t_arrival_ns=arrival_ns,
                t_utc_ns=frame_time.t_utc_ns,
                t_err_ns=frame_time.t_err_ns,
                t_quality=frame_time.t_quality,
                dropped_before=lost,
                flags=flags | frame_time.flags,
            )
        overflow = self._queue.put_frame(stamped, epoch)
        if overflow:
            self._drops.record_overflow(overflow)
        self._counters.frames_captured += 1
        self._counters.last_frame_mono_ns = seen_mono_ns
        if stream.config.kind is StreamKind.SNAPSHOT:
            self._set_capturing(False, epoch)  # one exposure per `start`

    # --- The sender thread -----------------------------------------------------------------

    def _current_sender(self) -> StreamSender | None:
        with self._lock:
            session = self._session
            return None if session is None else session.sender

    def _drop_sender(self, sender: StreamSender) -> None:
        with self._lock:
            session = self._session
            if session is not None and session.sender is sender:
                session.sender = None
        sender.close()

    def _sender_loop(self) -> None:
        while not self._stop.is_set():
            sender = self._current_sender()
            if sender is None:
                self._stop.wait(0.05)
                continue
            try:
                self._send_once(sender)
            except (IpcClosedError, IpcProtocolError) as error:
                _log.info("the frame stream ended: %s", error)
                self._drop_sender(sender)
            except Exception:
                self._counters.internal_errors += 1
                _log.exception("the sender failed")
                self._drop_sender(sender)
                self._stop.wait(0.05)

    def _send_once(self, sender: StreamSender) -> None:
        item = self._queue.peek(0.05)
        if item is None:
            sender.pump(0.0)  # notice a receiver that left, and read the acknowledgements
            return
        if not sender.has_credit(item.nbytes):
            self._counters.flow_stalls += 1
            sender.pump(0.02)  # the receiver is slow, so the queue absorbs and drops
            return
        if not self._queue.pop(item):
            return  # a drop took the head first
        self._send_item(sender, item)

    def _send_item(self, sender: StreamSender, item: QueueItem) -> None:
        tag = item.epoch & MAX_TAG
        if item.frame is not None:
            sender.send(encode_frame(item.frame), tag=tag)
            self._counters.frames_sent += 1
        elif item.event is not None:
            sender.send(item.event, tag=tag, kind=WireKind.EVENT)

    # --- The watchdog thread ---------------------------------------------------------------

    def _threads_alive(self) -> bool:
        return all(thread.is_alive() for thread in self._threads.values()) and all(
            thread.is_alive() for thread in self._rpc.worker_threads
        )

    def _fatal(self, reason: str) -> None:
        self._fatal_reason = reason
        _log.critical("acquire cannot continue: %s", reason)
        self._on_fatal(reason)
        self._stop.set()

    def _watchdog_loop(self) -> None:
        cfg = self._cfg
        interval_s = self._notifier.watchdog_interval_s or cfg.heartbeat_interval_s
        next_heartbeat_ns = 0
        next_log_ns = _REAL_CLOCK.monotonic_ns() + round(cfg.health_log_interval_s * NS_PER_S)
        previous_frames = 0
        previous_ns = self._clock.monotonic_ns()
        while not self._stop.wait(cfg.watchdog_tick_s):
            try:
                self._guard.check()
                healthy = self._threads_alive()
                if not healthy and not self._stop.is_set():
                    self._fatal("a service thread died")
                    return
                now_ns = self._clock.monotonic_ns()
                if now_ns > previous_ns:
                    frames = self._counters.frames_captured
                    rate = (frames - previous_frames) / ((now_ns - previous_ns) / NS_PER_S)
                    self._frame_rate_hz = 0.7 * self._frame_rate_hz + 0.3 * rate
                    previous_frames, previous_ns = frames, now_ns
                real_ns = _REAL_CLOCK.monotonic_ns()
                if real_ns >= next_heartbeat_ns:
                    next_heartbeat_ns = real_ns + round(interval_s * NS_PER_S)
                    health = self.health()
                    self._notifier.watchdog()
                    self._notifier.status(health.summary())
                if real_ns >= next_log_ns:
                    next_log_ns = real_ns + round(cfg.health_log_interval_s * NS_PER_S)
                    _log.info("health: %s", self.health().summary())
            except Exception:
                self._counters.internal_errors += 1
                _log.exception("the watchdog check failed")

    # --- Health ----------------------------------------------------------------------------

    def health(self) -> AcquireHealth:
        """A snapshot of the process. Safe to call from any thread."""
        with self._lock:
            opened, capturing, active = self._opened, self._capturing, self._active
            session = self._session
        status = self._clock.status()
        counters = self._counters
        drops = self._drops.counters
        now_ns = self._clock.monotonic_ns()
        state = "closed"
        if self._stop.is_set():
            state = "stopping"
        elif not self._started:
            state = "starting"
        elif capturing:
            period_s = (
                active.frame_period_s if active else None
            ) or self._cfg.default_frame_period_s
            limit_ns = round(max(STALL_FLOOR_S, STALL_PERIODS * period_s) * NS_PER_S)
            reference_ns = max(self._capture_since_ns, counters.last_frame_mono_ns or 0)
            state = "stalled" if now_ns - reference_ns > limit_ns else "streaming"
        elif opened:
            state = "ready"
        return AcquireHealth(
            state=state,
            instance=self.instance,
            driver=self._driver.name,
            uptime_s=(now_ns - self._started_ns) / NS_PER_S,
            opened=opened,
            capturing=capturing,
            stream_id=None if active is None else active.stream_id,
            frames_captured=counters.frames_captured,
            frames_sent=counters.frames_sent,
            frame_rate_hz=round(self._frame_rate_hz, 2),
            dropped_driver=drops.driver,
            dropped_gap=drops.gap,
            dropped_queue=drops.overflow,
            queue_frames=len(self._queue),
            queue_bytes=self._queue.nbytes,
            queue_peak_frames=self._queue.stats.peak_frames,
            flow_stalls=counters.flow_stalls,
            read_timeouts=counters.read_timeouts,
            read_errors=counters.read_errors,
            internal_errors=counters.internal_errors,
            last_error=counters.last_error,
            time_resets=self._stamper.resets,
            time_outliers=self._stamper.outliers,
            clock_synchronized=status.synchronized,
            clock_error_bound_ns=status.error_bound_ns,
            client_connected=session is not None,
            stream_connected=session is not None and session.sender is not None,
            threads_alive=self._threads_alive() if self._started else True,
            priority=self._priority,
        )


__all__ = [
    "EXIT_THREAD_DIED",
    "AcquireService",
    "CallGuard",
    "default_guard",
    "exit_on_fatal",
    "timing_config",
]
