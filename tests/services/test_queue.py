"""The bounded frame queue: overflow drops, the carried count, and events."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from seeingmon.frames import FRAME_HEADER_SIZE, Frame, Roi, TimeQuality
from seeingmon.services.acquire.queue import FrameQueue, QueueItem


def frame(seq: int, *, dropped: int = 0, size: int = 8) -> Frame:
    return Frame(
        data=np.zeros((2, size), dtype=np.uint16),
        stream_id=1,
        seq=seq,
        t_arrival_ns=seq,
        t_utc_ns=seq,
        t_err_ns=0,
        t_quality=TimeQuality.FITTED,
        dropped_before=dropped,
        exposure_us=1000,
        gain=1,
        mode="bin1",
        roi=Roi(0, 0, size, 2),
        adc_bits=14,
    )


def take_all(queue: FrameQueue) -> list[QueueItem]:
    items = []
    while (item := queue.peek(0.0)) is not None:
        assert queue.pop(item)
        items.append(item)
    return items


def seqs_and_drops(items: list[QueueItem]) -> list[tuple[int, int]]:
    return [(i.frame.seq, i.frame.dropped_before) for i in items if i.frame is not None]


class TestOrder:
    def test_frames_come_out_in_order_with_their_epoch(self) -> None:
        queue = FrameQueue(10, 1 << 20)
        for seq in range(5):
            assert queue.put_frame(frame(seq), epoch=seq % 2) == 0
        items = take_all(queue)
        assert [i.frame.seq for i in items if i.frame] == [0, 1, 2, 3, 4]
        assert [i.epoch for i in items] == [0, 1, 0, 1, 0]
        assert len(queue) == 0
        assert queue.stats.frames_in == queue.stats.frames_out == 5

    def test_peek_does_not_remove_and_pop_checks_identity(self) -> None:
        queue = FrameQueue(10, 1 << 20)
        queue.put_frame(frame(0), 0)
        queue.put_frame(frame(1), 0)
        head = queue.peek(0.0)
        assert head is not None
        assert queue.peek(0.0) is head
        other = queue.peek(0.0)
        assert other is head
        assert not queue.pop(QueueItem(0, 0))  # not the head
        assert queue.pop(head)
        assert not queue.pop(head)  # already gone
        assert len(queue) == 1

    def test_peek_waits_for_an_item_and_times_out_when_there_is_none(self) -> None:
        queue = FrameQueue(10, 1 << 20)
        started = time.monotonic()
        assert queue.peek(0.2) is None
        assert time.monotonic() - started >= 0.1
        timer = threading.Timer(0.15, lambda: queue.put_frame(frame(7), 0))
        timer.start()
        try:
            item = queue.peek(5.0)
        finally:
            timer.join()
        assert item is not None
        assert item.frame is not None
        assert item.frame.seq == 7


class TestOverflow:
    def test_a_full_queue_drops_its_oldest_frame_and_counts_it_on_the_next(self) -> None:
        queue = FrameQueue(3, 1 << 20)
        dropped = [queue.put_frame(frame(seq), 0) for seq in range(5)]
        assert dropped == [0, 0, 0, 1, 1]
        assert seqs_and_drops(take_all(queue)) == [(2, 2), (3, 0), (4, 0)]
        assert queue.stats.frames_dropped == 2

    def test_the_count_accumulates_across_many_drops(self) -> None:
        queue = FrameQueue(2, 1 << 20)
        for seq in range(10):
            queue.put_frame(frame(seq), 0)
        # Frames 0 to 7 were dropped. Frame 8 follows them, and frame 9 follows frame 8.
        assert seqs_and_drops(take_all(queue)) == [(8, 8), (9, 0)]

    def test_a_dropped_frame_passes_on_what_the_driver_and_the_gap_counted_before_it(self) -> None:
        queue = FrameQueue(2, 1 << 20)
        queue.put_frame(frame(0, dropped=2), 0)  # two frames were lost before frame 0
        queue.put_frame(frame(1, dropped=1), 0)
        queue.put_frame(frame(2), 0)  # frame 0 drops: 2 before it, and itself
        assert seqs_and_drops(take_all(queue)) == [(1, 1 + 2 + 1), (2, 0)]

    def test_the_identity_holds_for_every_delivered_frame(self) -> None:
        """seq equals the delivered count before the frame plus the drops reported up to it."""
        queue = FrameQueue(4, 1 << 20)
        delivered = 0
        reported = 0
        rng = np.random.default_rng(0)
        produced = 0
        for _ in range(300):
            for _ in range(int(rng.integers(1, 9))):
                queue.put_frame(frame(produced), 0)
                produced += 1
            for _ in range(int(rng.integers(0, 4))):
                item = queue.peek(0.0)
                if item is None or item.frame is None:
                    break
                assert queue.pop(item)
                reported += item.frame.dropped_before
                assert item.frame.seq == delivered + reported
                delivered += 1

    def test_the_byte_bound_applies_too(self) -> None:
        one = 16 * 2 + FRAME_HEADER_SIZE  # a 2 x 8 frame of 16-bit pixels
        queue = FrameQueue(100, 3 * one)
        for seq in range(5):
            queue.put_frame(frame(seq), 0)
        assert len(queue) == 3
        assert queue.nbytes == 3 * one
        assert seqs_and_drops(take_all(queue)) == [(2, 2), (3, 0), (4, 0)]

    def test_a_frame_larger_than_the_byte_bound_still_goes_in_alone(self) -> None:
        queue = FrameQueue(100, 10)
        assert queue.put_frame(frame(0), 0) == 0
        assert len(queue) == 1
        assert queue.put_frame(frame(1), 0) == 1
        assert seqs_and_drops(take_all(queue)) == [(1, 1)]

    def test_the_sender_can_lose_the_head_between_peek_and_pop(self) -> None:
        queue = FrameQueue(2, 1 << 20)
        queue.put_frame(frame(0), 0)
        queue.put_frame(frame(1), 0)
        head = queue.peek(0.0)
        assert head is not None
        queue.put_frame(frame(2), 0)  # drops frame 0
        assert not queue.pop(head)
        new_head = queue.peek(0.0)
        assert new_head is not None
        assert new_head.frame is not None
        assert (new_head.frame.seq, new_head.frame.dropped_before) == (1, 1)
        assert queue.pop(new_head)

    def test_the_peak_depth_is_recorded(self) -> None:
        queue = FrameQueue(3, 1 << 20)
        for seq in range(6):
            queue.put_frame(frame(seq), 0)
        assert queue.stats.peak_frames == 3

    def test_the_bounds_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            FrameQueue(0, 1)


class TestEvents:
    def test_events_keep_their_place_among_the_frames(self) -> None:
        queue = FrameQueue(10, 1 << 20)
        queue.put_frame(frame(0), 1)
        queue.put_event(1, b'{"e":1}')
        queue.put_frame(frame(1), 1)
        items = take_all(queue)
        assert [i.frame.seq if i.frame else i.event for i in items] == [0, b'{"e":1}', 1]

    def test_overflow_drops_frames_and_never_events(self) -> None:
        queue = FrameQueue(2, 1 << 20)
        queue.put_frame(frame(0), 1)
        queue.put_event(1, b"E")
        queue.put_frame(frame(1), 1)
        queue.put_frame(frame(2), 1)  # frame 0 drops, and the event stays
        items = take_all(queue)
        assert [i.frame.seq if i.frame else i.event for i in items] == [b"E", 1, 2]
        assert seqs_and_drops(items) == [(1, 1), (2, 0)]

    def test_an_event_with_the_same_key_at_the_tail_is_replaced(self) -> None:
        queue = FrameQueue(10, 1 << 20)
        queue.put_event(1, b"first", key="unplugged")
        queue.put_event(1, b"second", key="unplugged")
        queue.put_event(2, b"third", key="unplugged")  # another epoch is another event
        queue.put_event(2, b"other", key="stalled")
        assert [i.event for i in take_all(queue)] == [b"second", b"third", b"other"]

    def test_an_event_after_a_frame_is_not_merged_with_one_before_it(self) -> None:
        queue = FrameQueue(10, 1 << 20)
        queue.put_event(1, b"a", key="k")
        queue.put_frame(frame(0), 1)
        queue.put_event(1, b"b", key="k")
        assert len(take_all(queue)) == 3

    def test_too_many_events_drop_the_oldest_event(self) -> None:
        queue = FrameQueue(10, 1 << 20, max_events=3)
        for number in range(6):
            queue.put_event(1, f"event {number}".encode())
        assert [i.event for i in take_all(queue)] == [b"event 3", b"event 4", b"event 5"]
        assert queue.stats.events_dropped == 3


class TestClear:
    def test_a_flush_discards_everything_and_is_not_a_drop(self) -> None:
        queue = FrameQueue(10, 1 << 20)
        for seq in range(4):
            queue.put_frame(frame(seq), 0)
        queue.put_event(0, b"x")
        assert queue.clear() == 4
        assert len(queue) == 0
        assert queue.events == 0
        assert queue.nbytes == 0
        assert queue.peek(0.0) is None
        assert queue.stats.frames_flushed == 4
        assert queue.stats.frames_dropped == 0
        queue.put_frame(frame(9), 0)
        assert seqs_and_drops(take_all(queue)) == [(9, 0)]


def test_a_producer_and_a_consumer_on_two_threads_lose_no_count() -> None:
    queue = FrameQueue(8, 1 << 20)
    total = 3000
    received: list[tuple[int, int]] = []
    done = threading.Event()

    def consume() -> None:
        while not done.is_set() or len(queue):
            item = queue.peek(0.05)
            if item is not None and item.frame is not None and queue.pop(item):
                received.append((item.frame.seq, item.frame.dropped_before))
                if len(received) % 7 == 0:
                    time.sleep(0.0005)  # a consumer that is sometimes slow

    consumer = threading.Thread(target=consume)
    consumer.start()
    for seq in range(total):
        queue.put_frame(frame(seq), 0)
    done.set()
    consumer.join(30.0)
    assert not consumer.is_alive()
    reported = 0
    for position, (seq, dropped) in enumerate(received):
        reported += dropped
        assert seq == position + reported
    assert len(received) + reported == total
    assert queue.stats.frames_dropped == total - len(received)
