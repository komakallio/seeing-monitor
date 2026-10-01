"""The batch calls of the frame queue: `peek_batch`, `pop_items`, and when `put_frame` wakes."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np

from seeingmon.clock import SystemClock
from seeingmon.frames import FRAME_HEADER_SIZE, Frame, Roi, TimeQuality
from seeingmon.services.acquire.queue import FrameQueue, QueueItem

NS = 1_000_000_000
FRAME_BYTES = 2 * 8 * 2 + FRAME_HEADER_SIZE  # a frame of the helper below, as the queue counts it
CLOCK = SystemClock()


def frame(seq: int, *, dropped: int = 0, width: int = 8) -> Frame:
    return Frame(
        data=np.zeros((2, width), dtype=np.uint16),
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
        roi=Roi(0, 0, width, 2),
        adc_bits=14,
    )


def seqs(items: tuple[QueueItem, ...]) -> list[int]:
    return [item.frame.seq for item in items if item.frame is not None]


def after(seconds: float) -> int:
    return CLOCK.monotonic_ns() + round(seconds * NS)


def run_in_thread(call: Callable[[], object]) -> tuple[threading.Thread, list[object]]:
    result: list[object] = []
    thread = threading.Thread(target=lambda: result.append(call()), daemon=True)
    thread.start()
    return thread, result


class TestRuns:
    def test_a_run_that_has_reached_its_limit_is_final_and_goes_at_once(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        for seq in range(5):
            queue.put_frame(frame(seq), 0)
        started = CLOCK.monotonic_ns()
        run = queue.peek_batch(1.0, max_frames=3, max_bytes=1 << 20, due_ns=after(30.0))
        assert CLOCK.monotonic_ns() - started < 5 * NS
        assert seqs(run) == [0, 1, 2]

    def test_a_run_that_stays_under_its_limits_waits_for_the_flush_time(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        queue.put_frame(frame(0), 0)
        due_ns = after(0.2)
        run = queue.peek_batch(5.0, max_frames=8, max_bytes=1 << 20, due_ns=due_ns)
        assert CLOCK.monotonic_ns() >= due_ns
        assert seqs(run) == [0]

    def test_frames_that_arrive_while_the_run_waits_join_it(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        queue.put_frame(frame(0), 0)
        thread, result = run_in_thread(
            lambda: queue.peek_batch(5.0, max_frames=8, max_bytes=1 << 20, due_ns=after(0.4))
        )
        for seq in (1, 2):
            time.sleep(0.02)
            queue.put_frame(frame(seq), 0)
        thread.join(10.0)
        assert not thread.is_alive()
        assert seqs(result[0]) == [0, 1, 2]  # type: ignore[arg-type]

    def test_the_frame_that_fills_the_run_wakes_the_waiting_sender(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        queue.put_frame(frame(0), 0)
        started = CLOCK.monotonic_ns()
        due_ns = after(30.0)  # far away: only the third frame can end the wait
        thread, result = run_in_thread(
            lambda: queue.peek_batch(60.0, max_frames=3, max_bytes=1 << 20, due_ns=due_ns)
        )
        time.sleep(0.1)
        queue.put_frame(frame(1), 0)
        time.sleep(0.1)
        assert thread.is_alive()  # two frames do not fill the run, and nobody woke the thread
        queue.put_frame(frame(2), 0)
        thread.join(10.0)
        assert not thread.is_alive()
        assert CLOCK.monotonic_ns() < due_ns
        assert CLOCK.monotonic_ns() - started < 25 * NS
        assert seqs(result[0]) == [0, 1, 2]  # type: ignore[arg-type]

    def test_a_frame_of_another_epoch_ends_the_run(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        queue.put_frame(frame(0), 1)
        queue.put_frame(frame(1), 1)
        queue.put_frame(frame(2), 2)
        run = queue.peek_batch(1.0, max_frames=8, max_bytes=1 << 20, due_ns=after(30.0))
        assert seqs(run) == [0, 1]
        assert {item.epoch for item in run} == {1}

    def test_an_event_ends_the_run_and_goes_alone(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        queue.put_frame(frame(0), 0)
        queue.put_event(0, b"{}")
        queue.put_frame(frame(1), 0)
        run = queue.peek_batch(1.0, max_frames=8, max_bytes=1 << 20, due_ns=after(30.0))
        assert seqs(run) == [0]
        assert queue.pop_items(run)
        event = queue.peek_batch(1.0, max_frames=8, max_bytes=1 << 20, due_ns=after(30.0))
        assert len(event) == 1
        assert event[0].event == b"{}"
        assert queue.pop_items(event)
        assert seqs(queue.peek_batch(0.0, max_frames=8, max_bytes=1 << 20)) == [1]

    def test_the_byte_limit_ends_a_run_and_one_large_frame_goes_at_once(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        for seq in range(5):
            queue.put_frame(frame(seq), 0)
        run = queue.peek_batch(1.0, max_frames=8, max_bytes=2 * FRAME_BYTES + 1, due_ns=after(30.0))
        assert seqs(run) == [0, 1]
        big = FrameQueue(50, 1 << 20)
        big.put_frame(frame(0, width=512), 0)
        started = CLOCK.monotonic_ns()
        run = big.peek_batch(1.0, max_frames=8, max_bytes=100, due_ns=after(30.0))
        assert seqs(run) == [0]
        assert CLOCK.monotonic_ns() - started < 5 * NS

    def test_without_a_flush_time_nothing_waits(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        queue.put_frame(frame(0), 0)
        started = CLOCK.monotonic_ns()
        assert seqs(queue.peek_batch(1.0, max_frames=8, max_bytes=1 << 20)) == [0]
        assert CLOCK.monotonic_ns() - started < 5 * NS


class TestWaiting:
    def test_an_empty_queue_returns_nothing_when_the_time_is_up(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        started = CLOCK.monotonic_ns()
        assert queue.peek_batch(0.05, max_frames=8, max_bytes=1 << 20) == ()
        assert CLOCK.monotonic_ns() - started >= round(0.05 * NS)

    def test_a_frame_that_arrives_in_a_quiet_queue_wakes_the_sender_at_once(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        thread, result = run_in_thread(
            lambda: queue.peek_batch(30.0, max_frames=8, max_bytes=1 << 20)
        )
        time.sleep(0.05)
        assert thread.is_alive()
        queue.put_frame(frame(0), 0)
        thread.join(10.0)
        assert not thread.is_alive()
        assert seqs(result[0]) == [0]  # type: ignore[arg-type]

    def test_a_frame_after_the_flush_time_goes_out_at_once_too(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        thread, result = run_in_thread(
            lambda: queue.peek_batch(30.0, max_frames=8, max_bytes=1 << 20, due_ns=after(0.1))
        )
        time.sleep(0.4)  # the flush time passes with an empty queue, and the sender waits quietly
        assert thread.is_alive()
        queue.put_frame(frame(0), 0)
        thread.join(10.0)
        assert not thread.is_alive()
        assert seqs(result[0]) == [0]  # type: ignore[arg-type]

    def test_a_put_wakes_nobody_while_a_run_waits_for_its_flush_time(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        wakes: list[int] = []
        notify_all = queue._cond.notify_all

        def counting() -> None:
            wakes.append(len(queue))
            notify_all()

        queue._cond.notify_all = counting  # type: ignore[method-assign]
        queue.put_frame(frame(0), 0)
        wakes.clear()
        thread, result = run_in_thread(
            lambda: queue.peek_batch(60.0, max_frames=4, max_bytes=1 << 20, due_ns=after(30.0))
        )
        time.sleep(0.1)
        for seq in (1, 2):
            queue.put_frame(frame(seq), 0)
        assert wakes == []  # the sender sleeps until its flush time, whatever arrives
        queue.put_frame(frame(3), 0)  # the run is full now
        thread.join(10.0)
        assert not thread.is_alive()
        assert wakes == [4]
        assert seqs(result[0]) == [0, 1, 2, 3]  # type: ignore[arg-type]

    def test_a_put_wakes_a_sender_that_waits_for_the_first_frame(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        wakes: list[int] = []
        notify_all = queue._cond.notify_all

        def counting() -> None:
            wakes.append(len(queue))
            notify_all()

        queue._cond.notify_all = counting  # type: ignore[method-assign]
        thread, result = run_in_thread(
            lambda: queue.peek_batch(60.0, max_frames=4, max_bytes=1 << 20)
        )
        time.sleep(0.1)
        queue.put_frame(frame(0), 0)
        thread.join(10.0)
        assert not thread.is_alive()
        assert wakes == [1]
        assert seqs(result[0]) == [0]  # type: ignore[arg-type]

    def test_an_event_wakes_a_sender_that_holds_a_run_back(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        queue.put_frame(frame(0), 0)
        thread, result = run_in_thread(
            lambda: queue.peek_batch(60.0, max_frames=8, max_bytes=1 << 20, due_ns=after(30.0))
        )
        time.sleep(0.1)
        assert thread.is_alive()
        queue.put_event(0, b"{}")
        thread.join(10.0)
        assert not thread.is_alive()
        assert seqs(result[0]) == [0]  # type: ignore[arg-type]

    def test_a_blocked_peek_still_wakes_for_every_frame(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        thread, result = run_in_thread(lambda: queue.peek(30.0))
        time.sleep(0.05)
        queue.put_frame(frame(7), 0)
        thread.join(10.0)
        assert not thread.is_alive()
        item = result[0]
        assert isinstance(item, QueueItem)
        assert item.frame is not None
        assert item.frame.seq == 7


class TestPopItems:
    def test_it_removes_the_items_and_counts_them(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        for seq in range(4):
            queue.put_frame(frame(seq), 0)
        run = queue.peek_batch(1.0, max_frames=3, max_bytes=1 << 20)
        assert queue.pop_items(run)
        assert len(queue) == 1
        assert queue.nbytes == FRAME_BYTES
        assert queue.stats.frames_out == 3
        assert seqs(queue.peek_batch(0.0, max_frames=8, max_bytes=1 << 20)) == [3]

    def test_it_removes_nothing_when_a_drop_replaced_a_frame(self) -> None:
        queue = FrameQueue(3, 1 << 20)
        for seq in range(3):
            queue.put_frame(frame(seq), 0)
        run = queue.peek_batch(1.0, max_frames=8, max_bytes=1 << 20)
        assert seqs(run) == [0, 1, 2]
        assert queue.put_frame(frame(3), 0) == 1  # the oldest frame drops, and frame 1 carries it
        assert not queue.pop_items(run)
        assert len(queue) == 3
        again = queue.peek_batch(1.0, max_frames=8, max_bytes=1 << 20)
        assert [(i.frame.seq, i.frame.dropped_before) for i in again if i.frame] == [
            (1, 1),
            (2, 0),
            (3, 0),
        ]
        assert queue.pop_items(again)
        assert len(queue) == 0

    def test_it_removes_an_event_and_counts_it(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        queue.put_event(0, b"{}")
        run = queue.peek_batch(1.0, max_frames=8, max_bytes=1 << 20)
        assert queue.pop_items(run)
        assert queue.events == 0

    def test_it_refuses_more_items_than_the_queue_holds(self) -> None:
        queue = FrameQueue(50, 1 << 20)
        queue.put_frame(frame(0), 0)
        run = queue.peek_batch(1.0, max_frames=8, max_bytes=1 << 20)
        assert queue.pop_items(run)
        assert not queue.pop_items(run)
