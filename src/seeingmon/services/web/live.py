"""The live-view hubs: one frame stream from `core`, shared by every viewer.

`core` pushes the frames of a live view over a channel: the alignment frames over `alignment`, and
the video of Polaris over `polaris`. A hub opens its stream while somebody watches, keeps the
newest frame, and hands it to every viewer. A viewer is a WebSocket client (`subscribe`) or an HTTP
client that polls for the newest frame (`touch` and `latest`). A slow viewer never makes anything
buffer: each viewer takes the newest frame when it is ready and skips the frames in between.

`FrameHub` holds the logic, and the two hubs only say where their frames come from: `AlignmentHub`
reads `CoreClient.alignment_frames()`, and `PolarisHub` reads `CoreClient.polaris_frames()` and
stamps its sequence number into the state of each frame. The hubs are independent. Each one has its
own viewers, its own pump, and its own idle time.

**Life of the stream.** The first viewer starts the pump task, which reads the frames of the
source. When the stream breaks (`core` restarts, or the connection drops), the pump tells the
viewers once and reconnects after `retry_s`. When no viewer has shown interest for `idle_s`,
`check_idle` stops the pump and closes the stream to `core`. A ticker task calls `check_idle` every
`tick_s`, and a test can call it directly with a `VirtualClock`.

**The focus history.** `core` sends the whole focus history (the last 120 values, in parallel
lists) in the state of every alignment frame. A viewer needs it once and then only the new points,
so each WebSocket client of the alignment view has a `HistoryCursor` that rewrites the history of a
state before the server sends it: the first message of a viewer holds the whole history (`reset` is
`true`), and the next ones only the points that this viewer lacks (`reset` is `false`). A viewer
that skips a frame still gets every point, because the cursor follows what the viewer received and
not what the hub saw.

The readings of the rapid focus mode follow the same rule. `core` sends all of them (the last
600) in the state of every frame of the video of Polaris while the mode runs, and the cursor of
each Polaris WebSocket client cuts them down to the readings that this client lacks. The poll
route of the video leaves them out of its header.

**Event loop.** A hub belongs to one event loop. It creates its tasks in the loop that runs the
first call, and it starts again if a later call comes from another loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.services.web.contract import AlignmentFrame, PolarisFrame
from seeingmon.services.web.core_client import (
    CoreClient,
    CoreProtocolError,
    CoreUnavailableError,
)

_log = logging.getLogger(__name__)

ERROR_UNAVAILABLE = "core_unavailable"
ERROR_PROTOCOL = "core_error"
ERROR_INTERNAL = "internal_error"

Sleep = Callable[[float], Awaitable[None]]
F = TypeVar("F")


# The lists of `FocusHistoryView` that hold one entry for each point, and where it sits in a state.
HISTORY_LISTS = ("index", "seq", "t_utc_ms", "fwhm_px", "fwhm_arcsec", "n_stars", "spike")
HISTORY_PATH = ("focus", "history")
# The same for the readings of the rapid focus mode (`RapidReadingsView`).
RAPID_LISTS = (
    "index",
    "t_utc_ms",
    "fwhm_arcsec",
    "peak_fraction",
    "n_frames",
    "spike",
    "saturated",
)
RAPID_PATH = ("rapid_focus", "readings")


class HistoryCursor:
    """What one viewer has received of a history. See the module text.

    The history is the focus history or the readings of the rapid focus mode. `path` says
    where it sits in a state, and `lists` names its lists that hold one entry for each point.
    """

    def __init__(
        self, path: tuple[str, ...] = HISTORY_PATH, lists: tuple[str, ...] = HISTORY_LISTS
    ) -> None:
        self._path = path
        self._lists = lists
        self._session: int | None = None
        self._last = 0

    def delta(self, state: dict[str, Any]) -> None:
        """Cut the history of a state (as JSON) down to the points that this viewer lacks.

        The history stays whole, with `reset` true, for a viewer that has seen nothing, for a new
        session, and for a history that runs backward (`core` restarted). A state without the
        history stays as it is.
        """
        history: Any = state
        for key in self._path:
            history = history.get(key) if isinstance(history, dict) else None
        if not isinstance(history, dict):
            return
        index: list[int] = history.get("index") or []
        fresh = (
            self._session is None
            or history.get("session") != self._session
            or (bool(index) and index[-1] < self._last)
        )
        if fresh:
            history["reset"] = True
            self._session = history.get("session")
            self._last = index[-1] if index else 0
            return
        keep = [position for position, number in enumerate(index) if number > self._last]
        for name in self._lists:
            history[name] = [history[name][position] for position in keep]
        history["reset"] = False
        if keep:
            self._last = index[keep[-1]]


@dataclass(frozen=True, slots=True)
class HubFrame(Generic[F]):
    """A frame with the hub's own sequence number, which grows by one for each frame."""

    seq: int
    frame: F


@dataclass(frozen=True, slots=True)
class Update(Generic[F]):
    """What a viewer finds when it wakes: the newest unseen frame, and the state of the stream.

    `error` is the code of the current stream problem, or `None` while the stream works.
    `error_changed` is true when the problem began or ended since the viewer last looked.
    """

    frame: HubFrame[F] | None
    error: str | None
    error_changed: bool


class Subscription(Generic[F]):
    """One viewer of a hub. Get it from `FrameHub.subscribe`."""

    def __init__(self, hub: FrameHub[F]) -> None:
        self._hub = hub
        self._wake = asyncio.Event()
        self._seen_seq = 0
        self._seen_error_seq = hub.error_seq

    def _has_news(self) -> bool:
        latest = self._hub.latest
        return (latest is not None and latest.seq > self._seen_seq) or (
            self._hub.error_seq != self._seen_error_seq
        )

    def wake(self) -> None:
        self._wake.set()

    def newest(self) -> HubFrame[F] | None:
        """The newest frame that this viewer has not seen, and mark it as seen."""
        latest = self._hub.latest
        if latest is None or latest.seq <= self._seen_seq:
            return None
        self._seen_seq = latest.seq
        return latest

    def _update(self) -> Update[F]:
        changed = self._hub.error_seq != self._seen_error_seq
        self._seen_error_seq = self._hub.error_seq
        return Update(self.newest(), self._hub.error, changed)

    async def next_update(self, timeout_s: float) -> Update[F] | None:
        """Wait for a new frame or a change of the stream state. Returns `None` on a timeout."""
        if not self._has_news():
            self._wake.clear()
            if not self._has_news():
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout_s)
                except TimeoutError:
                    return None
        return self._update()


class FrameHub(Generic[F]):
    """Share one stream of frames between many viewers.

    `source` returns the async iterator of the frames of a new stream to `core`, and `name` names
    the hub in the names of its tasks and in the log. A subclass can change a frame before the hub
    keeps it (`stamp`).
    """

    def __init__(
        self,
        source: Callable[[], AsyncIterator[F]],
        clock: Clock,
        *,
        name: str,
        idle_s: float = 10.0,
        retry_s: float = 1.0,
        tick_s: float = 1.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._source = source
        self._clock = clock
        self.name = name
        self._idle_ns = round(idle_s * NS_PER_S)
        self._retry_s = retry_s
        self._tick_s = tick_s
        self.sleep = sleep
        self._latest: HubFrame[F] | None = None
        self._seq = 0
        self._error: str | None = None
        self._error_seq = 0
        self._subscriptions: set[Subscription[F]] = set()
        self._interest_ns = clock.monotonic_ns()
        self._pump: asyncio.Task[None] | None = None
        self._ticker: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.frames_received = 0
        self.streams_started = 0

    def stamp(self, seq: int, frame: F) -> F:
        """The frame that the hub keeps under the sequence number `seq`. The default keeps it."""
        return frame

    # --- State -----------------------------------------------------------------------------

    @property
    def latest(self) -> HubFrame[F] | None:
        """The newest frame, or `None` when no frame has arrived since the pump started."""
        return self._latest

    @property
    def error(self) -> str | None:
        """The code of the current stream problem, or `None` while the stream works."""
        return self._error

    @property
    def error_seq(self) -> int:
        return self._error_seq

    @property
    def viewers(self) -> int:
        """The number of WebSocket viewers."""
        return len(self._subscriptions)

    @property
    def running(self) -> bool:
        """Whether the pump task runs."""
        return self._pump is not None and not self._pump.done()

    def touch(self) -> None:
        """Say that somebody wants frames now. A polling client calls it with each request."""
        self._interest_ns = self._clock.monotonic_ns()
        self.ensure_running()

    # --- Viewers ---------------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def subscribe(self) -> AsyncIterator[Subscription[F]]:
        """Register a viewer for the length of the block, and start the stream if it is not up."""
        subscription = Subscription(self)
        self._subscriptions.add(subscription)
        self.touch()
        try:
            yield subscription
        finally:
            self._subscriptions.discard(subscription)
            self._interest_ns = self._clock.monotonic_ns()

    # --- The pump --------------------------------------------------------------------------

    def ensure_running(self) -> None:
        """Start the pump and the idle ticker in the running loop, unless they already run."""
        loop = asyncio.get_running_loop()
        if self._pump is not None and (self._pump.done() or self._loop is not loop):
            self._abandon()
        if self._pump is None:
            self._loop = loop
            self._pump = loop.create_task(self._run(), name=f"{self.name}-pump")
            self._ticker = loop.create_task(self._tick(), name=f"{self.name}-idle")

    def _abandon(self) -> None:
        for task in (self._pump, self._ticker):
            if task is not None and not task.done():
                with contextlib.suppress(RuntimeError):  # the loop of the task may be closed
                    task.cancel()
        self._pump = None
        self._ticker = None
        self._loop = None

    def _wanted(self) -> bool:
        if self._subscriptions:
            return True
        return self._clock.monotonic_ns() - self._interest_ns < self._idle_ns

    def check_idle(self) -> bool:
        """Stop the pump when nobody is interested. Returns whether it stopped."""
        if not self.running or self._wanted():
            return False
        self._abandon()
        self._latest = None
        return True

    async def _tick(self) -> None:
        while True:
            await self.sleep(self._tick_s)
            if self.check_idle():
                return

    def _publish(self, frame: F) -> None:
        self._seq += 1
        self.frames_received += 1
        self._latest = HubFrame(self._seq, self.stamp(self._seq, frame))
        self._set_error(None)
        for subscription in self._subscriptions:
            subscription.wake()

    def _set_error(self, code: str | None) -> None:
        if code == self._error:
            return
        self._error = code
        self._error_seq += 1
        for subscription in self._subscriptions:
            subscription.wake()

    async def _run(self) -> None:
        while True:
            self.streams_started += 1
            try:
                async for frame in self._source():
                    self._publish(frame)
                self._set_error(ERROR_UNAVAILABLE)  # a stream that ends cleanly is gone as well
            except CoreUnavailableError:
                self._set_error(ERROR_UNAVAILABLE)
            except CoreProtocolError:
                self._set_error(ERROR_PROTOCOL)
            except Exception:
                _log.exception("the %s stream failed", self.name)
                self._set_error(ERROR_INTERNAL)
            await self.sleep(self._retry_s)

    async def close(self) -> None:
        """Stop the pump and the ticker, and wait for them. Safe to call twice."""
        tasks = [task for task in (self._pump, self._ticker) if task is not None]
        self._abandon()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


class AlignmentHub(FrameHub[AlignmentFrame]):
    """Share one alignment stream between many viewers."""

    def __init__(
        self,
        core: CoreClient,
        clock: Clock,
        *,
        idle_s: float = 10.0,
        retry_s: float = 1.0,
        tick_s: float = 1.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        super().__init__(
            lambda: core.alignment_frames(),
            clock,
            name="alignment",
            idle_s=idle_s,
            retry_s=retry_s,
            tick_s=tick_s,
            sleep=sleep,
        )


class PolarisHub(FrameHub[PolarisFrame]):
    """Share one stream of the video of Polaris between many viewers.

    The hub numbers the frames that it receives and puts the number into the `seq` of each state,
    because `core` does not know it. The number grows by one for each frame, and it starts again at
    1 when the web process restarts.
    """

    def __init__(
        self,
        core: CoreClient,
        clock: Clock,
        *,
        idle_s: float = 10.0,
        retry_s: float = 1.0,
        tick_s: float = 1.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        super().__init__(
            lambda: core.polaris_frames(),
            clock,
            name="polaris",
            idle_s=idle_s,
            retry_s=retry_s,
            tick_s=tick_s,
            sleep=sleep,
        )

    def stamp(self, seq: int, frame: PolarisFrame) -> PolarisFrame:
        return replace(frame, state=frame.state.model_copy(update={"seq": seq}))
