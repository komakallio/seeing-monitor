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
"""

from __future__ import annotations

import threading
from collections import deque
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
                self._cond.wait(remaining_s)
            return self._items[0]

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
