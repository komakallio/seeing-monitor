"""Batches of frames on the stream of `acquire`: what a client gets, and what stays exact."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from seeingmon.clock import DEFAULT_START_UTC_NS, VirtualClock
from seeingmon.drivers.base import CameraTimeoutError
from seeingmon.frames import StreamConfig, decode_frames
from seeingmon.services.acquire.service import MIN_FRAMES_TO_HOLD
from seeingmon.services.ipc.codec import encode_stream_config
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.rpc import RpcClient, connect_rpc
from seeingmon.services.ipc.stream import StreamKind, StreamReceiver, StreamWindow, connect_stream
from seeingmon.services.remote import RemoteCameraDriver

from .conftest import wait_until
from .rig import FAST, SMALL, ParkingFake, Rig, make_rig

RigFactory = Callable[..., Rig]


@pytest.fixture
def build(native: Endpoint, key: ConnectionKey) -> Iterator[RigFactory]:
    rigs: list[Rig] = []

    def factory(**options: Any) -> Rig:
        rig = make_rig(options.pop("endpoint", native), options.pop("key", key), **options)
        rigs.append(rig)
        return rig

    yield factory
    for rig in rigs:
        rig.close()


def parked_rig(build: RigFactory, frames: int, **options: Any) -> Rig:
    """A rig on a virtual clock whose fake makes `frames` frames at once and then parks."""
    clock = VirtualClock(start_utc_ns=DEFAULT_START_UTC_NS)
    acquire = {"queue_depth": 1000, **options.pop("acquire", {})}
    return build(
        clock=clock,
        fake_class=ParkingFake,
        fake_options={"park_after": frames},
        acquire=acquire,
        **options,
    )


@dataclass
class Raw:
    """A client that speaks the protocol by hand, so that a test sees the messages as they are."""

    rig: Rig
    rpc: RpcClient
    session: str
    receivers: list[StreamReceiver]

    @classmethod
    def start(cls, rig: Rig, config: StreamConfig = SMALL) -> Raw:
        rpc, hello = connect_rpc(rig.endpoint, rig.key, {"role": "core"}, default_timeout_s=20.0)
        raw = cls(rig, rpc, str(hello["session"]), [])
        rpc.call("open")
        rpc.call("configure", {"config": encode_stream_config(config)})
        rpc.call("start")
        return raw

    def stream(
        self, ask: Any = 8, window: StreamWindow | None = None
    ) -> tuple[StreamReceiver, dict[str, Any]]:
        params: dict[str, Any] = {"session": self.session}
        if ask is not None:
            params["batch_frames"] = ask
        receiver, reply = connect_stream(
            self.rig.endpoint, self.rig.key, params, channel="frames", window=window
        )
        self.receivers.append(receiver)
        return receiver, dict(reply)

    def close(self) -> None:
        for receiver in self.receivers:
            receiver.close()
        self.rpc.close()


@pytest.fixture
def raws() -> Iterator[list[Raw]]:
    made: list[Raw] = []
    yield made
    for raw in made:
        raw.close()


def messages_of_frames(receiver: StreamReceiver, frames: int) -> list[list[int]]:
    """The sequence numbers of the frames in each message, until `frames` frames have come."""
    sizes: list[list[int]] = []
    total = 0
    while total < frames:
        message = receiver.recv(20.0)
        assert message is not None, "the stream stopped before all the frames came"
        assert message.kind is StreamKind.DATA
        seqs = [frame.seq for frame in decode_frames(message.payload)]
        sizes.append(seqs)
        total += len(seqs)
    return sizes


def read_until_it_raises(driver: RemoteCameraDriver, seqs: list[int]) -> None:
    """Read frames into `seqs` until the driver raises. Never returns."""
    while True:
        seqs.append(driver.read_frame(10.0).seq)


class TestWhatTheClientGets:
    @pytest.mark.parametrize(
        ("allowed", "ask", "granted"),
        [
            (16, 100, 16),
            (16, 4, 4),
            (4, 100, 4),
            (16, None, 1),
            (16, 0, 1),
            (16, -5, 1),
            (16, True, 1),
            (16, "8", 1),
            (1, 8, 1),
        ],
    )
    def test_the_reply_grants_what_the_client_asks_and_the_service_allows(
        self,
        build: RigFactory,
        raws: list[Raw],
        allowed: int,
        ask: Any,
        granted: int,
    ) -> None:
        rig = parked_rig(build, 5, services={"stream_batch_frames": allowed})
        raw = Raw.start(rig)
        raws.append(raw)
        _, reply = raw.stream(ask)
        assert reply["batch_frames"] == granted

    def test_a_client_that_asks_gets_runs_of_frames_up_to_its_number(
        self, build: RigFactory, raws: list[Raw]
    ) -> None:
        rig = parked_rig(build, 100)
        raw = Raw.start(rig)
        raws.append(raw)
        assert isinstance(rig.fake, ParkingFake)
        assert rig.fake.parked.wait(30.0)  # all 100 frames wait in the queue
        receiver, _ = raw.stream(ask=8)
        runs = messages_of_frames(receiver, 100)
        assert [len(run) for run in runs] == [8] * 12 + [4]
        assert [seq for run in runs for seq in run] == list(range(100))

    def test_a_client_that_does_not_ask_gets_one_frame_in_each_message(
        self, build: RigFactory, raws: list[Raw]
    ) -> None:
        rig = parked_rig(build, 30)
        raw = Raw.start(rig)
        raws.append(raw)
        assert isinstance(rig.fake, ParkingFake)
        assert rig.fake.parked.wait(30.0)
        receiver, _ = raw.stream(ask=None)
        runs = messages_of_frames(receiver, 30)
        assert [len(run) for run in runs] == [1] * 30
        assert [run[0] for run in runs] == list(range(30))

    def test_the_byte_limit_ends_a_run(self, build: RigFactory, raws: list[Raw]) -> None:
        frame_bytes = 16 * 16 * 2 + 96
        rig = parked_rig(build, 100, acquire={"batch_bytes": 3 * frame_bytes + 1})
        raw = Raw.start(rig)
        raws.append(raw)
        assert isinstance(rig.fake, ParkingFake)
        assert rig.fake.parked.wait(30.0)
        receiver, _ = raw.stream(ask=50)
        runs = messages_of_frames(receiver, 100)
        assert [len(run) for run in runs] == [3] * 33 + [1]

    def test_a_frame_larger_than_the_byte_limit_goes_alone_and_at_once(
        self, build: RigFactory, raws: list[Raw]
    ) -> None:
        rig = parked_rig(build, 20, acquire={"batch_bytes": 1024})
        raw = Raw.start(rig, FAST)
        raws.append(raw)
        assert isinstance(rig.fake, ParkingFake)
        assert rig.fake.parked.wait(30.0)
        receiver, _ = raw.stream(ask=8)
        runs = messages_of_frames(receiver, 20)
        assert [len(run) for run in runs] == [1] * 20

    def test_the_window_counts_messages_and_every_frame_still_arrives_in_order(
        self, build: RigFactory, raws: list[Raw]
    ) -> None:
        rig = parked_rig(build, 100)
        raw = Raw.start(rig)
        raws.append(raw)
        assert isinstance(rig.fake, ParkingFake)
        assert rig.fake.parked.wait(30.0)
        receiver, _ = raw.stream(ask=8, window=StreamWindow(2, 1 << 20))
        assert wait_until(lambda: receiver.pending == 2)  # two messages fill the window
        assert wait_until(lambda: rig.service.health().flow_stalls > 0)
        runs = messages_of_frames(receiver, 100)
        assert [seq for run in runs for seq in run] == list(range(100))
        assert all(len(run) <= 8 for run in runs)


class TestWhatStaysExact:
    def test_a_slow_reader_makes_the_queue_drop_and_the_counts_still_add_up(
        self, build: RigFactory
    ) -> None:
        clock = VirtualClock(start_utc_ns=DEFAULT_START_UTC_NS)
        rig = build(
            clock=clock,
            fake_class=ParkingFake,
            fake_options={"park_after": 200},
            acquire={"queue_depth": 4},
            services={"stream_window_messages": 2},
        )
        assert isinstance(rig.fake, ParkingFake)
        driver = rig.driver()
        driver.open()
        driver.configure(FAST)
        driver.start()
        assert rig.fake.parked.wait(30.0)  # 200 frames, and nobody read
        frames = [driver.read_frame(10.0)]
        while frames[-1].seq < 199:
            frames.append(driver.read_frame(10.0))
        reported = 0
        for position, frame in enumerate(frames):
            reported += frame.dropped_before
            assert frame.seq == position + reported  # the drops explain every missing number
        assert len(frames) + reported == 200
        health = driver.health()
        assert health["dropped_queue"] == reported
        assert health["queue_peak_frames"] <= 4

    def test_an_error_keeps_its_place_between_the_frames(self, build: RigFactory) -> None:
        rig = build()
        driver = rig.driver()
        driver.open()
        driver.configure(FAST)
        driver.start()
        seqs = [driver.read_frame(10.0).seq for _ in range(5)]
        rig.fake.fail_reads(CameraTimeoutError("scripted"))
        with pytest.raises(CameraTimeoutError, match="scripted"):
            read_until_it_raises(driver, seqs)
        seqs.extend(driver.read_frame(10.0).seq for _ in range(5))
        assert seqs == list(range(len(seqs)))  # the frames before the error all came first

    def test_the_frames_of_an_old_stream_that_a_message_carried_are_discarded(
        self, build: RigFactory
    ) -> None:
        rig = build(acquire={"batch_delay_s": 0.2})
        driver = rig.driver()
        driver.open()
        driver.configure(FAST)
        driver.start()

        def leftovers() -> int:
            link = driver._link
            assert link is not None
            return len(link.ready)

        for _ in range(500):  # a message holds several frames once the rate is known
            driver.read_frame(10.0)
            if leftovers():
                break
        assert leftovers(), "no message held more than one frame"
        active = driver.configure(FAST)
        driver.start()
        frame = driver.read_frame(10.0)
        assert frame.stream_id == active.stream_id
        assert driver.stale_discarded >= 1


class TestWhenToHold:
    def test_a_stream_that_delivers_fewer_than_one_and_a_half_frames_in_the_delay_goes_at_once(
        self, build: RigFactory
    ) -> None:
        service = build().service
        delay_s = 0.05
        service._frame_rate_hz = (MIN_FRAMES_TO_HOLD - 0.1) / delay_s
        assert service._next_flush_ns() == 0

    def test_a_fast_stream_holds_its_frames_for_the_delay(self, build: RigFactory) -> None:
        service = build().service
        service._frame_rate_hz = (MIN_FRAMES_TO_HOLD + 0.1) / service._cfg.batch_delay_s
        assert service._next_flush_ns() > 0

    def test_no_delay_means_no_holding(self, build: RigFactory) -> None:
        service = build(acquire={"batch_delay_s": 0.0}).service
        service._frame_rate_hz = 1000.0
        assert service._next_flush_ns() == 0
