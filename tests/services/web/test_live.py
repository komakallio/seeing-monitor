"""The live-view hubs: one stream from `core`, shared by many viewers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import Any

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.services.web.contract import AlignmentFrame, PolarisFrame
from seeingmon.services.web.core_client import (
    CoreProtocolError,
    CoreUnavailableError,
    FakeCoreClient,
)
from seeingmon.services.web.live import AlignmentHub, PolarisHub
from tests.services.web.helpers import alignment_state, polaris_state, tiny_jpeg, tiny_png


def frame(seq: int) -> AlignmentFrame:
    return AlignmentFrame(alignment_state(seq), tiny_jpeg(seq % 200))


class Source:
    """A frame source that the test feeds. Put a frame, an exception, or `None` (a clean end)."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.opened = 0
        self.closed = 0

    def __call__(self) -> AsyncIterator[Any]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Any]:
        self.opened += 1
        try:
            while True:
                item = await self.queue.get()
                if isinstance(item, BaseException):
                    raise item
                if item is None:
                    return
                yield item
        finally:
            self.closed += 1


async def fast_sleep(seconds: float) -> None:
    """Stands in for `asyncio.sleep`: it yields to the loop and does not wait."""
    await asyncio.sleep(0)


async def until(condition: Callable[[], bool], tries: int = 2000) -> None:
    """Let the event loop run until the condition holds. It never waits on the wall clock."""
    for _ in range(tries):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("the condition never held")


def make_hub(source: Source, clock: VirtualClock | None = None, **options: Any) -> AlignmentHub:
    settings: dict[str, Any] = {"idle_s": 10.0, "retry_s": 0.0, "tick_s": 0.0, "sleep": fast_sleep}
    settings.update(options)
    core = FakeCoreClient(frames=source)
    return AlignmentHub(core, clock or VirtualClock(), **settings)


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def running(hub: AlignmentHub) -> bool:
    """The `running` flag, read through a call so that a type checker does not narrow it."""
    return hub.running


def test_the_first_viewer_starts_the_stream_and_gets_the_frames() -> None:
    async def main() -> None:
        source = Source()
        hub = make_hub(source)
        assert not running(hub)
        async with hub.subscribe() as viewer:
            assert running(hub)
            source.queue.put_nowait(frame(1))
            update = await viewer.next_update(5)
            assert update is not None
            assert update.frame is not None
            assert update.frame.frame == frame(1)
            assert update.frame.seq == 1
            assert update.error is None
            assert update.error_changed is False
        await hub.close()

    run(main())


def test_a_slow_viewer_gets_the_newest_frame_and_skips_the_rest() -> None:
    async def main() -> None:
        source = Source()
        hub = make_hub(source)
        async with hub.subscribe() as viewer:
            for seq in (1, 2, 3):
                source.queue.put_nowait(frame(seq))
            await until(lambda: hub.frames_received == 3)
            update = await viewer.next_update(5)
            assert update is not None
            assert update.frame is not None
            assert update.frame.seq == 3
            assert update.frame.frame == frame(3)
            assert viewer.newest() is None  # nothing newer has arrived
        await hub.close()

    run(main())


def test_every_viewer_gets_the_frame_and_the_stream_is_opened_once() -> None:
    async def main() -> None:
        source = Source()
        hub = make_hub(source)
        async with hub.subscribe() as first, hub.subscribe() as second:
            assert hub.viewers == 2
            source.queue.put_nowait(frame(7))
            a, b = await asyncio.gather(first.next_update(5), second.next_update(5))
            assert a is not None
            assert b is not None
            assert a.frame == b.frame
        assert source.opened == 1
        assert hub.streams_started == 1
        await hub.close()

    run(main())


def test_a_viewer_that_waits_without_a_frame_gets_none_after_the_timeout() -> None:
    async def main() -> None:
        hub = make_hub(Source())
        async with hub.subscribe() as viewer:
            assert await viewer.next_update(0.01) is None
        await hub.close()

    run(main())


def test_a_late_viewer_sees_the_newest_frame_at_once() -> None:
    async def main() -> None:
        source = Source()
        hub = make_hub(source)
        async with hub.subscribe():
            source.queue.put_nowait(frame(4))
            await until(lambda: hub.latest is not None)
            async with hub.subscribe() as late:
                update = await late.next_update(5)
                assert update is not None
                assert update.frame is not None
                assert update.frame.seq == 1  # the hub numbers the frames that it receives
        await hub.close()

    run(main())


def test_a_broken_stream_is_reported_once_and_the_pump_reconnects() -> None:
    async def main() -> None:
        source = Source()
        hub = make_hub(source)
        async with hub.subscribe() as viewer:
            source.queue.put_nowait(frame(1))
            first = await viewer.next_update(5)
            assert first is not None
            assert first.error is None
            source.queue.put_nowait(CoreUnavailableError("gone"))
            update = await viewer.next_update(5)
            assert update is not None
            assert update.error == "core_unavailable"
            assert update.error_changed is True
            await until(lambda: source.opened == 2)  # the pump opened a new stream
            source.queue.put_nowait(CoreUnavailableError("still gone"))
            await until(lambda: source.opened == 3)
            assert hub.error == "core_unavailable"
            source.queue.put_nowait(frame(2))
            recovered = await viewer.next_update(5)
            assert recovered is not None
            assert recovered.error is None
            assert recovered.error_changed is True
            assert recovered.frame is not None
            assert recovered.frame.frame == frame(2)
        await hub.close()

    run(main())


def test_the_same_error_twice_in_a_row_is_one_change() -> None:
    async def main() -> None:
        source = Source()
        hub = make_hub(source)
        async with hub.subscribe() as viewer:
            source.queue.put_nowait(CoreUnavailableError("a"))
            await until(lambda: hub.error_seq == 1)
            source.queue.put_nowait(CoreUnavailableError("b"))
            await until(lambda: source.opened == 3)
            assert hub.error_seq == 1
            update = await viewer.next_update(5)
            assert update is not None
            assert update.error_changed
            assert await viewer.next_update(0.01) is None
        await hub.close()

    run(main())


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (CoreProtocolError("bad"), "core_error"),
        (RuntimeError("oops"), "internal_error"),
    ],
)
def test_other_failures_have_their_own_codes(raised: Exception, code: str) -> None:
    async def main() -> None:
        source = Source()
        hub = make_hub(source)
        async with hub.subscribe():
            source.queue.put_nowait(raised)
            await until(lambda: hub.error == code)
        await hub.close()

    run(main())


def test_a_stream_that_ends_cleanly_counts_as_gone_and_is_reopened() -> None:
    async def main() -> None:
        source = Source()
        hub = make_hub(source)
        async with hub.subscribe():
            source.queue.put_nowait(None)
            await until(lambda: hub.error == "core_unavailable")
            await until(lambda: source.opened == 2)
        await hub.close()

    run(main())


def test_the_stream_stops_when_nobody_has_shown_interest_for_the_idle_time() -> None:
    async def main() -> None:
        clock = VirtualClock()
        source = Source()
        hub = make_hub(source, clock, idle_s=10.0)
        async with hub.subscribe():
            source.queue.put_nowait(frame(1))
            await until(lambda: hub.latest is not None)
        assert running(hub)
        assert hub.check_idle() is False  # the last viewer has just left
        clock.advance(9.9)
        assert hub.check_idle() is False
        clock.advance(0.2)
        assert hub.check_idle() is True
        assert not running(hub)
        assert hub.latest is None  # a stale frame is not kept
        await until(lambda: source.closed == 1)  # the stream to core closed
        await hub.close()

    run(main())


def test_a_viewer_keeps_the_stream_open_however_long_it_stays() -> None:
    async def main() -> None:
        clock = VirtualClock()
        hub = make_hub(Source(), clock, idle_s=10.0)
        async with hub.subscribe():
            clock.advance(1000)
            assert hub.check_idle() is False
            assert running(hub)
        await hub.close()

    run(main())


def test_a_poll_keeps_the_stream_open_for_the_idle_time() -> None:
    async def main() -> None:
        clock = VirtualClock()
        source = Source()
        hub = make_hub(source, clock, idle_s=10.0)
        hub.touch()
        assert running(hub)
        clock.advance(8)
        hub.touch()  # the next poll
        clock.advance(8)
        assert hub.check_idle() is False
        clock.advance(3)
        assert hub.check_idle() is True
        await hub.close()

    run(main())


def test_the_idle_ticker_stops_the_stream_without_a_call_from_outside() -> None:
    async def main() -> None:
        clock = VirtualClock()
        source = Source()
        hub = make_hub(source, clock, idle_s=10.0)
        hub.touch()
        await until(lambda: source.opened == 1)
        clock.advance(11)
        await until(lambda: not running(hub))  # the ticker found the stream idle
        await until(lambda: source.closed == 1)
        await hub.close()

    run(main())


def test_a_new_viewer_after_the_stream_stopped_starts_it_again() -> None:
    async def main() -> None:
        clock = VirtualClock()
        source = Source()
        hub = make_hub(source, clock, idle_s=1.0)
        async with hub.subscribe():
            await until(lambda: source.opened == 1)
        clock.advance(5)
        assert hub.check_idle() is True
        async with hub.subscribe() as viewer:
            await until(lambda: source.opened == 2)
            source.queue.put_nowait(frame(9))
            update = await viewer.next_update(5)
            assert update is not None
            assert update.frame is not None
        await hub.close()

    run(main())


def test_the_hub_starts_again_when_it_is_used_from_another_event_loop() -> None:
    clock = VirtualClock()
    source = Source()
    core = FakeCoreClient(frames=source)
    hub = AlignmentHub(core, clock, idle_s=10.0, retry_s=0.0, tick_s=0.0, sleep=fast_sleep)

    async def first() -> None:
        hub.touch()
        await until(lambda: hub.streams_started == 1)

    async def second() -> None:
        hub.touch()
        await until(lambda: hub.streams_started == 2)
        await hub.close()

    run(first())
    run(second())


def test_close_stops_the_pump_and_the_ticker_and_can_repeat() -> None:
    async def main() -> None:
        source = Source()
        hub = make_hub(source)
        hub.touch()
        await until(lambda: source.opened == 1)
        await hub.close()
        assert not running(hub)
        await until(lambda: source.closed == 1)
        await hub.close()

    run(main())


def test_the_hub_counts_the_frames_that_it_receives() -> None:
    async def main() -> None:
        source = Source()
        hub = make_hub(source)
        async with hub.subscribe():
            for seq in range(5):
                source.queue.put_nowait(frame(seq))
            await until(lambda: hub.frames_received == 5)
            assert hub.latest is not None
            assert hub.latest.seq == 5
        await hub.close()

    run(main())


# --- The hub of the video of Polaris ---------------------------------------------------------


def polaris_frame(seq: int = 0) -> PolarisFrame:
    """A frame as `core` sends it: the state carries `seq` 0, because the hub numbers the frames."""
    return PolarisFrame(polaris_state(seq=0), tiny_png(seq % 200))


def make_polaris_hub(
    source: Source, clock: VirtualClock | None = None, **options: Any
) -> PolarisHub:
    settings: dict[str, Any] = {"idle_s": 10.0, "retry_s": 0.0, "tick_s": 0.0, "sleep": fast_sleep}
    settings.update(options)
    core = FakeCoreClient(polaris=source)
    return PolarisHub(core, clock or VirtualClock(), **settings)


def received(hub: PolarisHub, count: int) -> bool:
    return hub.frames_received == count


def test_the_polaris_hub_stamps_its_sequence_number_into_the_state_of_each_frame() -> None:
    async def main() -> None:
        source = Source()
        hub = make_polaris_hub(source)
        async with hub.subscribe() as viewer:
            for index in range(3):
                source.queue.put_nowait(polaris_frame(index))
            await until(lambda: hub.frames_received == 3)
            update = await viewer.next_update(5)
            assert update is not None
            assert update.frame is not None
            assert update.frame.seq == 3
            assert update.frame.frame.state.seq == 3
            assert update.frame.frame.state == polaris_state(seq=3)
            assert update.frame.frame.image == tiny_png(2)
        await hub.close()

    run(main())


def test_the_polaris_hub_numbers_the_frames_one_by_one_and_keeps_the_rest_of_the_state() -> None:
    async def main() -> None:
        source = Source()
        hub = make_polaris_hub(source)
        async with hub.subscribe():
            seen = []
            for count in range(1, 4):
                source.queue.put_nowait(polaris_frame(count))
                await until(partial(received, hub, count))
                assert hub.latest is not None
                seen.append(hub.latest.frame.state.seq)
                assert hub.latest.frame.state.model_copy(update={"seq": 0}) == polaris_state(0)
            assert seen == [1, 2, 3]
        await hub.close()

    run(main())


def test_the_two_hubs_are_independent() -> None:
    async def main() -> None:
        aligned, polaris = Source(), Source()
        clock = VirtualClock()
        core = FakeCoreClient(frames=aligned, polaris=polaris)
        options: dict[str, Any] = {"retry_s": 0.0, "tick_s": 0.0, "sleep": fast_sleep}
        alignment_hub = AlignmentHub(core, clock, **options)
        polaris_hub = PolarisHub(core, clock, **options)
        async with alignment_hub.subscribe() as viewer:
            assert alignment_hub.viewers == 1
            assert polaris_hub.viewers == 0
            assert not polaris_hub.running  # a viewer of one view does not open the other stream
            await until(lambda: aligned.opened == 1)
            assert polaris.opened == 0
            async with polaris_hub.subscribe() as other:
                await until(lambda: polaris.opened == 1)
                aligned.queue.put_nowait(frame(1))
                polaris.queue.put_nowait(polaris_frame(1))
                one, two = await asyncio.gather(viewer.next_update(5), other.next_update(5))
                assert one is not None
                assert two is not None
                assert one.frame is not None
                assert two.frame is not None
                assert one.frame.frame == frame(1)
                assert two.frame.frame.state.seq == 1
        await alignment_hub.close()
        await polaris_hub.close()
        assert (aligned.closed, polaris.closed) == (1, 1)

    run(main())


def test_a_broken_polaris_stream_is_reported_with_the_same_codes_and_reconnects() -> None:
    async def main() -> None:
        source = Source()
        hub = make_polaris_hub(source)
        async with hub.subscribe() as viewer:
            source.queue.put_nowait(CoreUnavailableError("gone"))
            update = await viewer.next_update(5)
            assert update is not None
            assert (update.error, update.error_changed) == ("core_unavailable", True)
            await until(lambda: source.opened == 2)
            source.queue.put_nowait(polaris_frame(1))
            recovered = await viewer.next_update(5)
            assert recovered is not None
            assert recovered.error is None
            assert recovered.frame is not None
            assert recovered.frame.frame.state.seq == 1
        await hub.close()

    run(main())


def test_the_polaris_hub_stops_its_stream_when_nobody_is_interested() -> None:
    async def main() -> None:
        clock = VirtualClock()
        source = Source()
        hub = make_polaris_hub(source, clock, idle_s=10.0)
        hub.touch()
        await until(lambda: source.opened == 1)
        assert hub.name == "polaris"
        clock.advance(11)
        assert hub.check_idle() is True
        await until(lambda: source.closed == 1)
        await hub.close()

    run(main())
