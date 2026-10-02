"""The alignment helper: frames in from the scheduler, JPEGs and state out to `web`.

**Intake.** In the `align` state the scheduler calls `sink(frame)` on its own thread for every
frame. `sink` keeps the newest frame in two slots (one for the encoder, one for the solver) and
returns at once. A frame that is still in a slot when the next one arrives is replaced and counted
as dropped, so nothing queues: the live view always shows the newest frame that the machine could
handle, and the latency stays at one exposure plus one encode.

**The encoder thread** takes the newest frame, makes the preview and the measures of the frame
(`seeingmon.services.core.alignment.preview`), builds the `AlignmentState` with the latest solve,
and sends `pack_frame(state, jpeg)` to every open live-view stream. A stream with no room in its
window skips the frame, because a slow consumer must never make `core` buffer. While a stream is
open, the thread tells the scheduler now and then that someone watches (`touch`), so the idle timer
does not end `align`.

**The solver thread** takes the newest frame about once a second (`solve_interval_s`, measured from
the start of the previous solve) and runs the quick solve on it. The result joins the next state.

**For tests.** `process_frame` and `solve_frame` do the work of one loop turn on the calling thread,
so a test needs no threads and no waiting. `start` launches the two threads for a real run.

**Streams.** `attach` takes the `StreamSender` of a new live-view client (see
`seeingmon.services.core.rpc`). The encoder thread owns the sender from then on: it sends and
reads the acknowledgements, and it drops the sender when the client goes away.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.frames import Frame
from seeingmon.profile import Profile
from seeingmon.services.core.alignment.preview import (
    frame_saturation_dn,
    histogram_counts,
    make_preview,
    saturated_fraction,
)
from seeingmon.services.core.alignment.solve import QuickSolution
from seeingmon.services.core.alignment.state import (
    FrameSummary,
    build_state,
    resolve_target,
)
from seeingmon.services.core.settings import AlignmentSettings
from seeingmon.services.ipc.errors import IpcError
from seeingmon.services.ipc.stream import StreamSender
from seeingmon.services.web.contract import (
    AlignmentState,
    HistogramView,
    SaturationView,
    pack_frame,
)
from seeingmon.survey.tracker import PointingTracker

_log = logging.getLogger(__name__)

WAKE_S = 0.5  # the longest that a worker thread waits before it looks around


class Solver(Protocol):
    """`QuickSolver` fits."""

    def solve(self, frame: Frame) -> QuickSolution: ...


class AlignmentHelper:
    """The live view and the quick solve of the alignment state. See the module text."""

    def __init__(
        self,
        *,
        settings: AlignmentSettings,
        profile: Profile,
        clock: Clock,
        is_active: Callable[[], bool],
        solver: Solver | None = None,
        tracker: PointingTracker | None = None,
        touch: Callable[[], None] | None = None,
    ) -> None:
        self._settings = settings
        self._profile = profile
        self._clock = clock
        self._is_active = is_active
        self._solver = solver
        self._tracker = tracker
        self._touch = touch

        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._encode_slot: Frame | None = None
        self._solve_slot: Frame | None = None
        self._summary: FrameSummary | None = None
        self._solution: QuickSolution | None = None
        self._best_fwhm: float | None = None
        self._senders: list[StreamSender] = []
        self._session = False
        self._last_publish_ns: int | None = None
        self._last_solve_ns: int | None = None
        self._next_touch_ns = 0

        self.frames_received = 0
        self.frames_dropped = 0
        self.frames_encoded = 0
        self.frames_sent = 0
        self.frames_skipped = 0
        self.solves = 0
        self.solve_failures = 0
        self.encode_errors = 0
        self.last_encode_s = 0.0
        self.last_solve_s = 0.0

    # --- The scheduler's side --------------------------------------------------------------

    def sink(self, frame: Frame) -> None:
        """Take a frame of the alignment stream. Called on the scheduler thread, returns at once."""
        with self._wake:
            self.frames_received += 1
            if self._encode_slot is not None:
                self.frames_dropped += 1  # the encoder did not take the last one in time
            self._encode_slot = frame
            self._solve_slot = frame
            self._wake.notify_all()

    # --- The streams -----------------------------------------------------------------------

    def attach(self, sender: StreamSender, params: Mapping[str, Any] | None = None) -> None:
        """Take a live-view client. Called when the client connects."""
        with self._lock:
            self._senders.append(sender)
            self._next_touch_ns = 0  # tell the scheduler at once that someone watches
            self._wake.notify_all()

    @property
    def viewers(self) -> int:
        """The number of open live-view streams."""
        with self._lock:
            return sum(1 for sender in self._senders if not sender.closed)

    # --- The state -------------------------------------------------------------------------

    def state(self) -> AlignmentState:
        """The state for `alignment_state`: `active` is false outside `align`."""
        if not self._is_active():
            self._end_session()
            return AlignmentState(active=False)
        with self._lock:
            summary, solution, best = self._summary, self._solution, self._best_fwhm
        if summary is None:
            return AlignmentState(active=True, quality={"frame": "no frame has arrived yet"})
        return self._build(summary, solution, best)

    def _build(
        self, summary: FrameSummary, solution: QuickSolution | None, best: float | None
    ) -> AlignmentState:
        target = resolve_target(self._settings, self._tracker, summary.t_utc_ns, summary.mode)
        state = build_state(summary, solution, target, self._settings, best_fwhm_px=best)
        if self._solver is not None:
            return state
        reason = "the quick solve is not available: no catalog is configured"
        replaced = {key: reason for key in ("solved", "sky") if key in state.quality}
        if not replaced:
            return state
        return state.model_copy(update={"quality": {**state.quality, **replaced}})

    def _end_session(self) -> None:
        with self._lock:
            if not self._session and self._summary is None:
                return
            self._session = False
            self._summary = None
            self._solution = None
            self._best_fwhm = None
            self._encode_slot = None
            self._solve_slot = None

    # --- The work of one loop turn ---------------------------------------------------------

    def summarize(self, frame: Frame) -> FrameSummary:
        """Measure a frame: its size, the histogram, and the share of saturated pixels."""
        readout = self._profile.mode(frame.mode)
        saturation = self._profile.saturation(frame.mode, frame.gain)
        level = frame_saturation_dn(frame, saturation.native_dn)
        bins = self._settings.histogram_bins
        fraction = saturated_fraction(frame.data, self._settings.saturation_level * level)
        return FrameSummary(
            seq=frame.seq,
            t_utc_ns=frame.t_utc_ns,
            width_px=frame.roi.width,
            height_px=frame.roi.height,
            mode=frame.mode,
            exposure_s=frame.exposure_us / 1e6 if frame.exposure_us > 0 else None,
            gain=frame.gain,
            plate_scale_arcsec_px=self._profile.plate_scale_arcsec_per_px(readout),
            histogram=HistogramView(
                counts=histogram_counts(frame.data, bins, level), min_dn=0.0, max_dn=level
            ),
            saturation=SaturationView(
                fraction=fraction, warning=fraction > self._settings.saturation_warn_fraction
            ),
        )

    def process_frame(self, frame: Frame) -> bytes:
        """Encode a frame, build its state, and send both to the open streams.

        Returns the payload that went out (or would go out, with no stream open).
        """
        started = self._clock.monotonic_ns()
        summary = self.summarize(frame)
        preview = make_preview(
            frame.data,
            max_pixels=self._settings.max_preview_pixels,
            quality=self._settings.jpeg_quality,
        )
        with self._lock:
            self._summary = summary
            self._session = True
            solution, best = self._solution, self._best_fwhm
        payload = pack_frame(self._build(summary, solution, best), preview.jpeg)
        self.frames_encoded += 1
        self._publish(payload)
        self.last_encode_s = (self._clock.monotonic_ns() - started) / NS_PER_S
        return payload

    def solve_frame(self, frame: Frame) -> QuickSolution | None:
        """Run the quick solve on a frame, and keep the result for the next states."""
        if self._solver is None:
            return None
        started = self._clock.monotonic_ns()
        solution = self._solver.solve(frame)
        with self._lock:
            self._solution = solution
            if solution.focus_fwhm_px is not None and (
                self._best_fwhm is None or solution.focus_fwhm_px < self._best_fwhm
            ):
                self._best_fwhm = solution.focus_fwhm_px
        if solution.solved:
            self.solves += 1
        else:
            self.solve_failures += 1
        self.last_solve_s = (self._clock.monotonic_ns() - started) / NS_PER_S
        return solution

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
        """Drop closed streams, touch the scheduler while someone watches, and end a session."""
        self._reap()
        with self._lock:
            watched = bool(self._senders)
        if not self._is_active():
            self._end_session()
            return
        now = self._clock.monotonic_ns()
        if watched and self._touch is not None and now >= self._next_touch_ns:
            self._next_touch_ns = now + round(self._settings.touch_interval_s * NS_PER_S)
            self._touch()

    # --- Threads ---------------------------------------------------------------------------

    def start(self) -> None:
        """Start the encoder and solver threads."""
        if self._threads:
            raise RuntimeError("the helper already runs")
        for name, target in (
            ("align-encode", self._encode_loop),
            ("align-solve", self._solve_loop),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._threads.append(thread)
            thread.start()

    def stop(self, timeout_s: float = 10.0) -> None:
        """Stop the threads and close the streams. Safe to call twice."""
        self._stop.set()
        with self._wake:
            self._wake.notify_all()
        for thread in self._threads:
            thread.join(timeout_s)
        self._threads.clear()
        with self._lock:
            senders, self._senders = self._senders, []
        for sender in senders:
            sender.close("core is shutting down")

    def _take_for_encoding(self) -> Frame | None:
        """Take the newest frame for the encoder, waiting up to `WAKE_S` for one."""
        with self._wake:
            if self._encode_slot is None and not self._stop.is_set():
                self._wake.wait(WAKE_S)
            frame, self._encode_slot = self._encode_slot, None
            return frame

    def _take_for_solving(self) -> Frame | None:
        """Take the newest frame for the solver, waiting up to `WAKE_S` for one."""
        with self._wake:
            if self._solve_slot is None and not self._stop.is_set():
                self._wake.wait(WAKE_S)
            frame, self._solve_slot = self._solve_slot, None
            return frame

    def _encode_loop(self) -> None:
        interval_ns = round(self._settings.min_interval_s * NS_PER_S)
        while not self._stop.is_set():
            frame = self._take_for_encoding()
            if frame is not None:
                now = self._clock.monotonic_ns()
                last = self._last_publish_ns
                if last is not None and now - last < interval_ns:
                    with self._wake:  # too soon: keep the frame, and look again shortly
                        if self._encode_slot is None:
                            self._encode_slot = frame
                        else:
                            self.frames_dropped += 1
                    self._stop.wait(min(self._settings.min_interval_s, WAKE_S))
                    continue
                try:
                    self.process_frame(frame)
                    self._last_publish_ns = now
                except Exception:
                    self.encode_errors += 1
                    _log.exception("the live-view frame %d could not be encoded", frame.seq)
            try:
                self.housekeeping()
            except Exception:
                _log.exception("the housekeeping of the alignment helper failed")

    def _solve_loop(self) -> None:
        interval_ns = round(self._settings.solve_interval_s * NS_PER_S)
        while not self._stop.is_set():
            if self._solver is None:
                self._stop.wait(WAKE_S)
                continue
            last = self._last_solve_ns
            if last is not None and self._clock.monotonic_ns() - last < interval_ns:
                self._stop.wait(0.05)
                continue
            frame = self._take_for_solving()
            if frame is None or not self._is_active():
                continue
            self._last_solve_ns = self._clock.monotonic_ns()
            try:
                self.solve_frame(frame)
            except Exception:
                self.solve_failures += 1
                _log.exception("the quick solve of frame %d failed", frame.seq)
