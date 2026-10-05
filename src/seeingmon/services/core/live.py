"""The live video of Polaris in `core`: the decorator of the fast analyzer, and the stream.

The scheduler calls `FastAnalyzer.push` for every frame of the fast stream, about 80 times a second.
`LiveFastAnalyzer` wraps the analyzer that the scheduler gets, and after each `push` it calls
`PolarisStream.offer(frame, update)`, so the scheduler stays as it is. Only frames that the fast
analyzer receives can reach the video: the frames of the survey, the alignment helper, and the
bursts never pass through it.

**The scheduler thread.** `offer` runs on the scheduler thread for every frame, so it stays cheap:

- With no viewer it returns after one attribute check (`_active`).
- With a viewer it thins the stream by the time of the frames (`max_fps`, 20 by default). The
  thinning keeps the average rate even: the next frame is due one interval after the last due time,
  and not one interval after the last kept frame, so frames that come every 12 ms still give 20 a
  second. A new stream or a step back in time resyncs it. A kept frame costs one copy of the ROI
  (the camera may reuse the buffer) and a few attributes, and the call puts them into one slot
  under a lock and wakes the encoder. A slot that the encoder has not taken yet gives way to the
  next frame, so nothing queues.
- It never blocks and never raises into the scheduler. An error switches the video off for
  `disable_s` seconds and logs one line with the traceback.

**The rapid focus mode.** The scheduler hands the frames of that mode to the rapid focus helper,
which measures them and offers each one here with `rapid=True` (see
`seeingmon.services.core.alignment.rapid`). They take the same path as the frames of the fast
stream, so the page shows one video for both. The thread of the stream asks `rapid` for the state of
the mode when it renders such a frame, and the state carries it (`PolarisState.rapid_focus`). A
frame of the mode carries no rolling seeing value.

**The stream thread.** A thread takes the newest slot and renders it (`PolarisRenderer`: the
stretch, the width, the PNG, and the state), and sends `pack_polaris_frame(state, image)` to every
open stream. A stream with no room in its window skips the frame, because a slow consumer must never
make `core` buffer. The thread also reads the acknowledgements, drops the streams that closed, and
turns the video on and off with the viewers. `attach` takes the `StreamSender` of a new client (see
`seeingmon.services.core.rpc`).

**For tests.** `process` does the work of one loop turn on the calling thread, so a test needs no
thread and no waiting. `start` launches the thread for a real run.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any

import numpy.typing as npt

from seeingmon.analysis.base import FastAnalyzer, FastContext, FastUpdate
from seeingmon.clock import NS_PER_S, Clock
from seeingmon.frames import ActiveStream, Frame
from seeingmon.records import SeeingWindowRecord
from seeingmon.services.core.polaris import FrameSlot, PolarisRenderer
from seeingmon.services.core.settings import PolarisSettings
from seeingmon.services.ipc.errors import IpcError
from seeingmon.services.ipc.stream import StreamSender
from seeingmon.services.web.contract import LiveSeeingView, RapidFocusView, pack_polaris_frame

_log = logging.getLogger(__name__)

WAKE_S = 0.5  # the longest that the stream thread waits before it looks around


def live_view(live: Any) -> LiveSeeingView | None:
    """The rolling seeing value as the contract describes it. `live` is a `LiveSeeing` or `None`."""
    if live is None:
        return None
    fields = dataclasses.asdict(live)
    fields["flags"] = list(fields["flags"])
    fields["quality"] = dict(fields["quality"])
    return LiveSeeingView.model_validate(fields)


class PolarisStream:
    """Thin, stretch, and encode the fast stream for the clients of the `polaris` channel."""

    def __init__(
        self,
        settings: PolarisSettings,
        *,
        clock: Clock,
        renderer: PolarisRenderer,
        live: Callable[[], LiveSeeingView | None] | None = None,
        rapid: Callable[[], RapidFocusView | None] | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._renderer = renderer
        self._live = live
        self._rapid = rapid
        self._interval_ns = max(1, round(NS_PER_S / settings.max_fps))

        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._senders: list[StreamSender] = []
        self._slot: FrameSlot | None = None
        # The scheduler thread reads `_active` for every frame without the lock.
        self._active = False
        self._watched = False
        self._failed_until_ns: int | None = None
        # The thinning state belongs to the scheduler thread.
        self._stream_id = -1
        self._last_seen_ns = 0
        self._next_due_ns = 0
        self._count = 0

        self.frames_kept = 0
        self.frames_replaced = 0
        self.frames_encoded = 0
        self.frames_sent = 0
        self.frames_skipped = 0
        self.offer_errors = 0
        self.encode_errors = 0
        self.last_encode_s = 0.0
        self.bytes_sent = 0

    # --- The scheduler's side --------------------------------------------------------------

    def offer(self, frame: Frame, update: FastUpdate, rapid: bool = False) -> None:
        """Take a frame of the fast stream. Called on the scheduler thread, returns at once.

        `rapid` says that the frame comes from the rapid focus mode.
        """
        if not self._active:
            return
        try:
            self._keep(frame, update, rapid)
        except Exception:
            self._fail()

    def _keep(self, frame: Frame, update: FastUpdate, rapid: bool) -> None:
        t_ns = frame.t_utc_ns
        self._count += 1 + frame.dropped_before
        previous = self._last_seen_ns
        self._last_seen_ns = t_ns
        resync = frame.stream_id != self._stream_id or t_ns < previous
        if not resync and t_ns < self._next_due_ns:
            return
        self._stream_id = frame.stream_id
        if resync:
            self._next_due_ns = t_ns + self._interval_ns
        else:
            due = self._next_due_ns + self._interval_ns
            self._next_due_ns = due if due > t_ns else t_ns + self._interval_ns
        slot = FrameSlot(
            frame.data.copy(),
            frame.stream_id,
            t_ns,
            frame.mode,
            frame.exposure_us,
            frame.gain,
            frame.adc_bits,
            frame.roi,
            update.star,
            self._count,
            rapid,
        )
        with self._wake:
            if self._slot is not None:
                self.frames_replaced += 1  # the encoder did not take the last one in time
            self._slot = slot
            self._wake.notify()
        self.frames_kept += 1

    def _fail(self) -> None:
        """Switch the video off for a while after an error, and say so once."""
        with self._lock:
            self.offer_errors += 1
            self._active = False
            self._failed_until_ns = self._clock.monotonic_ns() + round(
                self._settings.disable_s * NS_PER_S
            )
        _log.exception(
            "the video of Polaris failed on the scheduler thread, so it stays off for %g s",
            self._settings.disable_s,
        )

    # --- The streams -----------------------------------------------------------------------

    def attach(self, sender: StreamSender, params: Mapping[str, Any] | None = None) -> None:
        """Take a client of the video. Called when the client connects."""
        with self._lock:
            self._senders.append(sender)
            self._watched = True
            self._active = self._failed_until_ns is None
            self._stream_id = -1  # keep the next frame at once, so that the client sees one soon
            self._wake.notify_all()

    @property
    def viewers(self) -> int:
        """The number of open streams."""
        with self._lock:
            return sum(1 for sender in self._senders if not sender.closed)

    @property
    def active(self) -> bool:
        """Whether `offer` keeps frames now: somebody watches, and the video is not switched off."""
        return self._active

    # --- The work of one loop turn ---------------------------------------------------------

    def current_live(self) -> LiveSeeingView | None:
        """The newest rolling seeing value, or `None` when `core` has none."""
        return None if self._live is None else self._live()

    def process(self, slot: FrameSlot) -> bytes:
        """Render a frame and send it to the open streams. Returns the payload that went out."""
        started = self._clock.monotonic_ns()
        if slot.rapid:  # the rapid focus mode: its state, and no rolling seeing value
            frame = self._renderer.render(
                slot, None, None if self._rapid is None else self._rapid()
            )
        else:
            frame = self._renderer.render(slot, self.current_live())
        payload = pack_polaris_frame(frame.state, frame.image)
        self.frames_encoded += 1
        self._publish(payload)
        self.last_encode_s = (self._clock.monotonic_ns() - started) / NS_PER_S
        return payload

    def _publish(self, payload: bytes) -> None:
        with self._lock:
            senders = list(self._senders)
        gone: list[StreamSender] = []
        for sender in senders:
            try:
                sender.pump(0.0)  # reads the acknowledgements, and notices a client that left
                if sender.closed:
                    gone.append(sender)
                elif sender.has_credit(len(payload)):
                    sender.send(payload)
                    self.frames_sent += 1
                    self.bytes_sent += len(payload)
                else:
                    self.frames_skipped += 1  # a slow client: skip the frame, never queue it
            except IpcError:
                gone.append(sender)
        if gone:
            with self._lock:
                self._senders = [s for s in self._senders if s not in gone]

    def _reap(self) -> None:
        """Read the acknowledgements of every stream, and drop the streams that closed."""
        with self._lock:
            senders = list(self._senders)
        gone: list[StreamSender] = []
        for sender in senders:
            try:
                sender.pump(0.0)  # a client that left shows here as a closed connection
            except IpcError:
                gone.append(sender)
            if sender.closed and sender not in gone:
                gone.append(sender)
        if gone:
            with self._lock:
                self._senders = [s for s in self._senders if s not in gone]

    def housekeeping(self) -> None:
        """Drop closed streams, and turn the video on or off with the viewers."""
        self._reap()
        with self._lock:
            watched = bool(self._senders)
            failed = self._failed_until_ns
            if failed is not None and self._clock.monotonic_ns() >= failed:
                self._failed_until_ns = failed = None
            self._watched = watched
            self._active = watched and failed is None
            if not watched:
                self._slot = None

    # --- The thread ------------------------------------------------------------------------

    def start(self) -> None:
        """Start the stream thread."""
        if self._thread is not None:
            raise RuntimeError("the video of Polaris already runs")
        self._thread = threading.Thread(target=self._loop, name="polaris-encode", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 10.0) -> None:
        """Stop the thread and close the streams. Safe to call twice."""
        self._stop.set()
        with self._wake:
            self._wake.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout_s)
        with self._lock:
            senders, self._senders = self._senders, []
            self._active = False
            self._watched = False
            self._slot = None
        for sender in senders:
            sender.close("core is shutting down")

    def _take(self) -> FrameSlot | None:
        """Take the newest slot, waiting up to `WAKE_S` for one."""
        with self._wake:
            if self._slot is None and not self._stop.is_set():
                self._wake.wait(WAKE_S)
            slot, self._slot = self._slot, None
            return slot

    def _loop(self) -> None:
        while not self._stop.is_set():
            slot = self._take()
            if slot is not None:
                try:
                    self.process(slot)
                except Exception:
                    self.encode_errors += 1
                    if self.encode_errors <= 3 or self.encode_errors % 1000 == 0:
                        _log.exception("a frame of the video of Polaris could not be encoded")
            try:
                self.housekeeping()
            except Exception:
                _log.exception("the housekeeping of the video of Polaris failed")


class LiveFastAnalyzer:
    """A `FastAnalyzer` that hands each frame to the video of Polaris after the analysis.

    It forwards every method to the analyzer that it wraps, and every other attribute too, so it
    stands in for the analyzer wherever the scheduler or a test reaches for one. `live` is the
    rolling seeing value of the wrapped analyzer, or `None` when it keeps none.
    """

    def __init__(self, inner: FastAnalyzer, stream: PolarisStream) -> None:
        self._inner = inner
        self._stream = stream

    @property
    def inner(self) -> FastAnalyzer:
        """The analyzer that this one wraps."""
        return self._inner

    @property
    def live(self) -> Any:
        """The rolling seeing value of the wrapped analyzer (a `LiveSeeing`), or `None`."""
        return getattr(self._inner, "live", None)

    def begin_stream(self, stream: ActiveStream) -> tuple[SeeingWindowRecord, ...]:
        return self._inner.begin_stream(stream)

    def set_context(self, context: FastContext) -> None:
        self._inner.set_context(context)

    def push(self, frame: Frame) -> FastUpdate:
        update = self._inner.push(frame)
        self._stream.offer(frame, update)
        return update

    def flush(self, reason: str = "end") -> tuple[SeeingWindowRecord, ...]:
        return self._inner.flush(reason)

    def drain_metrics(self) -> npt.NDArray[Any] | None:
        return self._inner.drain_metrics()

    def __getattr__(self, name: str) -> Any:
        # Python calls this only for a name that the decorator does not define. The guard keeps a
        # half-built object (for example while it is copied) from calling itself without end.
        if name in ("_inner", "_stream"):
            raise AttributeError(name)
        return getattr(self._inner, name)
