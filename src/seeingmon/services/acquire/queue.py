"""The bounded queue between the capture thread and the sender thread.

The capture thread must never wait for the network, so the queue is bounded, and a full queue
drops its oldest frame. The bound is a number of frames (`depth`) and a number of bytes
(`max_bytes`), because a survey frame is thousands of times larger than a fast frame.

**Counting what drops.** A dropped frame is not silent: the queue adds the frames that were lost
before it, plus the frame itself, to `dropped_before` of the next frame in the queue. So the
count of the frames lost before any delivered frame stays exact, however many frames the queue
dropped in a row. The newest frame never drops, so there is always a next frame to carry the
count.

**Events.** The queue also carries small events, such as a camera error, in order with the
frames. The frame bounds do not apply to them, and a full queue never drops one. At most
`max_events` wait at a time, and the oldest event goes first beyond that.

**Threads.** One lock covers the queue. The sender thread takes the head in two steps: `peek`,
and `pop` when it can send. The capture thread may drop that frame in between, and then `pop`
returns `False`, and the sender looks again.

**Batches.** A sender that puts several frames in one message uses `peek_batch` and `pop_items`
in the same two steps. `peek_batch` holds a short run of frames back until a flush time that the
sender names (`due_ns`), so a stream of 100 frames a second costs the sender a wake-up for each
batch and not for each frame. While it holds the run back, `put_frame` does not wake it, unless
the run is full. A put wakes a sender that waits for the first item, so a frame that arrives in a
quiet queue goes out at once.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.frames import FRAME_HEADER_SIZE, Frame

_REAL_CLOCK = SystemClock()


@dataclass(frozen=True, slots=True, eq=False)
class QueueItem:
    """A frame or an event, with the capture epoch it belongs to and its size in bytes."""

    epoch: int
    nbytes: int
    frame: Frame | None = None
    event: bytes | None = None
    key: str | None = None


@dataclass(slots=True)
class QueueStats:
    """Counters of a `FrameQueue`."""

    frames_in: int = 0
    frames_out: int = 0
    frames_dropped: int = 0
    frames_flushed: int = 0
    events_in: int = 0
    events_dropped: int = 0
    peak_frames: int = 0


class FrameQueue:
    """A queue of frames and events that drops its oldest frame when it is full."""

    def __init__(self, depth: int, max_bytes: int, *, max_events: int = 32) -> None:
        if depth < 1 or max_bytes < 1 or max_events < 1:
            raise ValueError("depth, max_bytes, and max_events must be at least 1")
        self._depth = depth
        self._max_bytes = max_bytes
        self._max_events = max_events
        self._cond = threading.Condition()
        self._items: deque[QueueItem] = deque()
        self._frames = 0
        self._events = 0
        self._bytes = 0
        self._idle_waiters = 0  # threads that wait for the first item: a put wakes them
        self._wake_at_frames = 0  # a put that brings the queue to this many frames wakes a waiter
        self.stats = QueueStats()

    def __len__(self) -> int:
        """The number of frames in the queue."""
        with self._cond:
            return self._frames

    @property
    def nbytes(self) -> int:
        """The bytes of the frames in the queue."""
        with self._cond:
            return self._bytes

    @property
    def events(self) -> int:
        """The number of events in the queue."""
        with self._cond:
            return self._events

    def put_frame(self, frame: Frame, epoch: int) -> int:
        """Add a frame, and drop the oldest frames while the queue is over its bounds.

        Returns the number of frames dropped by this call.
        """
        item = QueueItem(epoch, frame.data.nbytes + FRAME_HEADER_SIZE, frame=frame)
        dropped = 0
        with self._cond:
            self._items.append(item)
            self._frames += 1
            self._bytes += item.nbytes
            self.stats.frames_in += 1
            while self._frames > 1 and (
                self._frames > self._depth or self._bytes > self._max_bytes
            ):
                self._drop_oldest_frame()
                dropped += 1
            self.stats.frames_dropped += dropped
            self.stats.peak_frames = max(self.stats.peak_frames, self._frames)
            if self._idle_waiters or (
                self._wake_at_frames and self._frames >= self._wake_at_frames
            ):
                self._cond.notify_all()
        return dropped

    def _drop_oldest_frame(self) -> None:
        """Remove the oldest frame, and hand its count to the next frame. The lock is held."""
        victim = next(index for index, item in enumerate(self._items) if item.frame is not None)
        removed = self._items[victim]
        del self._items[victim]
        assert removed.frame is not None
        self._frames -= 1
        self._bytes -= removed.nbytes
        carry = removed.frame.dropped_before + 1
        for index in range(victim, len(self._items)):
            heir = self._items[index]
            if heir.frame is not None:
                moved = replace(heir.frame, dropped_before=heir.frame.dropped_before + carry)
                self._items[index] = replace(heir, frame=moved)
                return

    def put_event(self, epoch: int, payload: bytes, key: str | None = None) -> None:
        """Add an event. An event with the same `key` at the tail of the queue is replaced.

        The replacement lets a repeating condition, such as a camera that stays unplugged, take
        one place in the queue instead of one for every attempt.
        """
        item = QueueItem(epoch, len(payload), event=payload, key=key)
        with self._cond:
            tail = self._items[-1] if self._items else None
            if (
                key is not None
                and tail is not None
                and tail.event is not None
                and tail.key == key
                and tail.epoch == epoch
            ):
                self._items[-1] = item
            else:
                self._items.append(item)
                self._events += 1
                if self._events > self._max_events:
                    oldest = next(
                        index
                        for index, queued in enumerate(self._items)
                        if queued.event is not None
                    )
                    del self._items[oldest]
                    self._events -= 1
                    self.stats.events_dropped += 1
            self.stats.events_in += 1
            self._cond.notify_all()

    def peek(self, timeout_s: float, *, clock: Clock | None = None) -> QueueItem | None:
        """The item at the head, or `None` when the queue stays empty for `timeout_s`."""
        clock = _REAL_CLOCK if clock is None else clock
        started_ns = clock.monotonic_ns()
        with self._cond:
            while not self._items:
                remaining_s = timeout_s - (clock.monotonic_ns() - started_ns) / NS_PER_S
                if remaining_s <= 0:
                    return None
                self._idle_waiters += 1
                try:
                    self._cond.wait(remaining_s)
                finally:
                    self._idle_waiters -= 1
            return self._items[0]

    def peek_batch(
        self,
        timeout_s: float,
        *,
        max_frames: int,
        max_bytes: int,
        due_ns: int = 0,
        clock: Clock | None = None,
    ) -> tuple[QueueItem, ...]:
        """The items for the next message, or `()` when the queue stays empty for `timeout_s`.

        The items are one event, or a run of frames of one epoch from the head: at most
        `max_frames` frames, and at most `max_bytes` bytes (one frame may exceed that alone). The
        run is final when it has reached a limit, or when something else follows it (an event, or
        a frame of another epoch). The call returns a final run at once, and it holds any other
        run back until `due_ns` on `clock`, so that more frames can join it. With `due_ns=0` no
        run waits.

        The call asks `put_frame` to wake it only for a run that fills up, or for the first item
        of an empty queue after `due_ns`. Like `peek`, it removes nothing: pass the result to
        `pop_items`.
        """
        clock = _REAL_CLOCK if clock is None else clock
        give_up_ns = clock.monotonic_ns() + round(timeout_s * NS_PER_S)
        with self._cond:
            try:
                while True:
                    now_ns = clock.monotonic_ns()
                    if self._items:
                        run, final = self._head_run(max_frames, max_bytes)
                        if final or now_ns >= due_ns:
                            return run
                        self._wake_at_frames = max_frames
                        self._cond.wait((due_ns - now_ns) / NS_PER_S)
                        continue
                    remaining_s = (give_up_ns - now_ns) / NS_PER_S
                    if remaining_s <= 0:
                        return ()
                    if now_ns < due_ns:
                        # A stream in flow: sleep to the flush time, and wake for no arrival.
                        self._cond.wait(min(remaining_s, (due_ns - now_ns) / NS_PER_S))
                        continue
                    self._idle_waiters += 1
                    try:
                        self._cond.wait(remaining_s)
                    finally:
                        self._idle_waiters -= 1
            finally:
                self._wake_at_frames = 0

    def _head_run(self, max_frames: int, max_bytes: int) -> tuple[tuple[QueueItem, ...], bool]:
        """The items at the head that go in one message, and whether the run is final."""
        head = self._items[0]
        if head.frame is None:
            return (head,), True  # an event goes alone
        run: list[QueueItem] = []
        total = 0
        for item in self._items:
            if item.frame is None or item.epoch != head.epoch:
                return tuple(run), True  # something else follows, so the run cannot grow
            if run and (len(run) >= max_frames or total + item.nbytes > max_bytes):
                return tuple(run), True
            run.append(item)
            total += item.nbytes
        return tuple(run), len(run) >= max_frames or total >= max_bytes

    def pop_items(self, items: Sequence[QueueItem]) -> bool:
        """Remove `items` from the head if they are all still there, in order.

        Returns `False` when a drop replaced one of them meanwhile, and removes nothing then.
        """
        with self._cond:
            if len(self._items) < len(items):
                return False
            if any(self._items[index] is not item for index, item in enumerate(items)):
                return False
            for item in items:
                self._items.popleft()
                if item.frame is not None:
                    self._frames -= 1
                    self._bytes -= item.nbytes
                    self.stats.frames_out += 1
                else:
                    self._events -= 1
            return True

    def pop(self, item: QueueItem) -> bool:
        """Remove the head if it is `item`. Returns `False` when a drop replaced it meanwhile."""
        with self._cond:
            if not self._items or self._items[0] is not item:
                return False
            self._items.popleft()
            if item.frame is not None:
                self._frames -= 1
                self._bytes -= item.nbytes
                self.stats.frames_out += 1
            else:
                self._events -= 1
            return True

    def clear(self) -> int:
        """Discard everything, as a new stream does. Returns the frames discarded.

        A flush is a decision, not a loss, so the frames do not count as dropped.
        """
        with self._cond:
            flushed = self._frames
            self._items.clear()
            self._frames = self._events = self._bytes = 0
            self.stats.frames_flushed += flushed
            return flushed
