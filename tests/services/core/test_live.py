"""The live video of Polaris in `core`: the decorator, the thinning, and the streams."""

from __future__ import annotations

import copy
import dataclasses
import io
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from seeingmon.analysis.base import FastAnalyzer, FastContext, FastUpdate, StarState
from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.frames import (
    ActiveStream,
    Frame,
    FrameFlag,
    PixelFormat,
    Roi,
    StreamConfig,
    TimeQuality,
)
from seeingmon.services.core.live import LiveFastAnalyzer, PolarisStream, live_view
from seeingmon.services.core.polaris import FrameSlot, PolarisRenderer
from seeingmon.services.core.settings import PolarisSettings
from seeingmon.services.ipc.errors import IpcClosedError
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.server import IpcServer
from seeingmon.services.ipc.stream import (
    StreamKind,
    StreamReceiver,
    StreamSender,
    StreamService,
    StreamWindow,
    connect_stream,
)
from seeingmon.services.web.contract import (
    PNG_MAGIC,
    POLARIS_CHANNEL,
    LiveSeeingView,
    PolarisFrame,
    RapidFocusView,
    unpack_polaris_frame,
)
from seeingmon.testing import FakeFastAnalyzer
from tests.services.conftest import wait_until
from tests.services.core.test_polaris import star_frame
from tests.services.web.helpers import live_seeing_view

T0 = 1_790_000_000 * NS_PER_S
PERIOD_NS = 12_180_000  # the camera runs at 82 frames per second
FOUND = StarState(True, 2072.0, 1411.0, 0.28, 60.0)
UPDATE = FastUpdate(star=FOUND)


def fast_frame(
    index: int,
    *,
    stream_id: int = 1,
    t_utc_ns: int | None = None,
    dropped_before: int = 0,
    data: npt.NDArray[np.uint16] | None = None,
    roi: Roi | None = None,
    mode: str = "bin1",
    exposure_us: int = 2000,
) -> Frame:
    pixels = data if data is not None else np.full((128, 128), 500, dtype=np.uint16)
    when = T0 + index * PERIOD_NS if t_utc_ns is None else t_utc_ns
    return Frame(
        data=pixels,
        stream_id=stream_id,
        seq=index,
        t_arrival_ns=when,
        t_utc_ns=when,
        t_err_ns=1000,
        t_quality=TimeQuality.EXACT,
        dropped_before=dropped_before,
        exposure_us=exposure_us,
        gain=0,
        mode=mode,
        roi=roi or Roi(2008, 1347, pixels.shape[1], pixels.shape[0]),
        adc_bits=12,
        flags=FrameFlag.SIMULATED,
    )


class FakeSender:
    """A `StreamSender` that needs no connection: it keeps the payloads that it is given."""

    def __init__(self, credit: bool = True) -> None:
        self.payloads: list[bytes] = []
        self.closed = False
        self.credit = credit

    def pump(self, timeout_s: float = 0.0) -> None:
        return None

    def has_credit(self, nbytes: int) -> bool:
        return self.credit

    def send(self, payload: bytes | bytearray | memoryview) -> int:
        self.payloads.append(bytes(payload))
        return len(self.payloads)

    def close(self, reason: str = "") -> None:
        self.closed = True


class BreakingSender(FakeSender):
    def send(self, payload: bytes | bytearray | memoryview) -> int:
        raise IpcClosedError("the client left")


class FailingRenderer(PolarisRenderer):
    """A renderer that fails for the first frames that it gets."""

    def __init__(self, failures: int) -> None:
        super().__init__(scale_for=lambda mode: 1.91)
        self.failures = failures

    def render(
        self,
        slot: FrameSlot,
        live: LiveSeeingView | None = None,
        rapid: RapidFocusView | None = None,
    ) -> PolarisFrame:
        if self.failures > 0:
            self.failures -= 1
            raise ValueError("this frame cannot be rendered")
        return super().render(slot, live, rapid)


def make_stream(
    *,
    clock: VirtualClock | None = None,
    live: Callable[[], LiveSeeingView | None] | None = None,
    settings: PolarisSettings | None = None,
    renderer: PolarisRenderer | None = None,
) -> PolarisStream:
    chosen = settings or PolarisSettings()
    return PolarisStream(
        chosen,
        clock=clock or VirtualClock(T0),
        renderer=renderer or PolarisRenderer(chosen, scale_for=lambda mode: 1.91),
        live=live,
    )


def watched(stream: PolarisStream, sender: FakeSender | None = None) -> FakeSender:
    """Attach a client to the stream, so that it keeps frames."""
    client = sender or FakeSender()
    stream.attach(cast(StreamSender, client))
    return client


def feed(
    stream: PolarisStream, count: int, *, period_ns: int = PERIOD_NS, start: int = 0, **options: Any
) -> int:
    """Offer `count` frames that are `period_ns` apart. Returns the number of kept frames."""
    before = stream.frames_kept
    for index in range(start, start + count):
        stream.offer(fast_frame(index, t_utc_ns=T0 + index * period_ns, **options), UPDATE)
    return stream.frames_kept - before


def active(stream: PolarisStream) -> bool:
    """The `active` flag, read through a call so that a type checker does not narrow it."""
    return stream.active


def kept_times(stream: PolarisStream, count: int) -> list[int]:
    """Offer `count` frames, and return the times of the frames that the stream kept."""
    times: list[int] = []
    for index in range(count):
        before = stream.frames_kept
        stream.offer(fast_frame(index), UPDATE)
        if stream.frames_kept > before:
            assert stream._slot is not None
            times.append(stream._slot.t_utc_ns)
    return times


# --- The decorator ---------------------------------------------------------------------------


class TestDecorator:
    def test_the_wrapper_is_a_fast_analyzer(self) -> None:
        wrapper = LiveFastAnalyzer(FakeFastAnalyzer(), make_stream())
        assert isinstance(wrapper, FastAnalyzer)

    def test_every_method_goes_to_the_analyzer_and_its_answer_comes_back(self) -> None:
        inner = FakeFastAnalyzer(window_s=1.0)
        wrapper = LiveFastAnalyzer(inner, make_stream())
        active = ActiveStream(
            stream_id=1,
            config=StreamConfig("bin1", 2000, 0, pixel_format=PixelFormat.RAW16),
            frame_shape=(128, 128),
            adc_bits=12,
            frame_period_s=0.0122,
        )
        assert wrapper.begin_stream(active) == ()
        wrapper.set_context(FastContext(flags=frozenset({"cloud"})))
        data = star_frame()
        update = None
        index = 0
        for index in range(100):  # 1.2 s of frames: a window of 1 s closes
            update = wrapper.push(fast_frame(index, data=data))
            if update.windows:
                break
        assert update is not None
        assert len(update.windows) == 1
        assert "cloud" in update.windows[0].flags  # the context reached the analyzer
        assert update.star.found is True
        assert inner.frames_pushed == index + 1
        assert wrapper.drain_metrics() is not None
        assert wrapper.drain_metrics() is None  # the analyzer cleared its rows
        assert len(wrapper.flush("end")) == 1  # the open window closes early

    def test_the_stream_gets_each_frame_after_the_analysis_with_the_update(self) -> None:
        offered: list[tuple[Frame, FastUpdate]] = []
        inner = FakeFastAnalyzer()

        class Recorder:
            def offer(self, frame: Frame, update: FastUpdate) -> None:
                assert inner.frames_pushed == len(offered) + 1  # the analysis came first
                offered.append((frame, update))

        wrapper = LiveFastAnalyzer(inner, cast(PolarisStream, Recorder()))
        frames = [fast_frame(index, data=star_frame()) for index in range(3)]
        updates = [wrapper.push(frame) for frame in frames]
        assert [pair[0] for pair in offered] == frames
        assert all(pair[1] is update for pair, update in zip(offered, updates, strict=True))

    def test_a_frame_of_a_search_burst_stays_out_of_the_video(self) -> None:
        offered: list[Frame] = []
        inner = FakeFastAnalyzer()

        class Recorder:
            def offer(self, frame: Frame, update: FastUpdate) -> None:
                offered.append(frame)

        wrapper = LiveFastAnalyzer(inner, cast(PolarisStream, Recorder()))
        star = wrapper.measure(fast_frame(0, data=star_frame()), None)
        assert star.found
        assert inner.frames_measured == 1
        assert offered == []  # the video sees no burst frame
        assert inner.frames_pushed == 0

    def test_other_attributes_of_the_analyzer_stay_reachable(self) -> None:
        inner = FakeFastAnalyzer()
        wrapper = LiveFastAnalyzer(inner, make_stream())
        wrapper.push(fast_frame(0, data=star_frame()))
        assert wrapper.frames_pushed == 1
        assert wrapper.star is inner.star
        assert wrapper.inner is inner
        with pytest.raises(AttributeError):
            _ = wrapper.no_such_attribute

    def test_live_is_the_value_of_the_analyzer_or_none(self) -> None:
        inner = FakeFastAnalyzer()
        wrapper = LiveFastAnalyzer(inner, make_stream())
        assert wrapper.live is None  # this analyzer keeps no rolling value
        setattr(inner, "live", "a value")  # noqa: B010 - the fake has no such attribute
        assert wrapper.live == "a value"

    def test_a_half_built_wrapper_does_not_call_itself_without_end(self) -> None:
        wrapper = LiveFastAnalyzer(FakeFastAnalyzer(), make_stream())
        assert copy.copy(wrapper).inner is wrapper.inner
        bare = LiveFastAnalyzer.__new__(LiveFastAnalyzer)
        with pytest.raises(AttributeError):
            _ = bare.anything


# --- What offer does on the scheduler thread -------------------------------------------------


class Untouchable:
    """A frame that fails the test when `offer` reads anything from it."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"offer read {name} of a frame while nobody watched")


class TestOffer:
    def test_with_no_viewer_a_frame_is_not_touched(self) -> None:
        stream = make_stream()
        assert active(stream) is False
        stream.offer(cast(Frame, Untouchable()), cast(FastUpdate, Untouchable()))
        assert (stream.frames_kept, stream.offer_errors) == (0, 0)

    def test_the_first_frame_after_a_viewer_attaches_is_kept_at_once(self) -> None:
        stream = make_stream()
        watched(stream)
        assert active(stream) is True
        assert feed(stream, 1) == 1

    def test_the_stream_is_thinned_to_the_rate_by_frame_time(self) -> None:
        stream = make_stream()
        watched(stream)
        kept = feed(stream, round(5 * 82.1))  # five seconds of frames at 82.1 frames a second
        assert kept in (100, 101)  # 20 frames a second

    @pytest.mark.parametrize("fps", [40.0, 82.1, 100.0, 250.0])
    def test_the_thinning_keeps_the_average_rate_for_any_camera_rate(self, fps: float) -> None:
        stream = make_stream()
        watched(stream)
        kept = feed(stream, round(10 * fps), period_ns=round(NS_PER_S / fps))
        assert kept == pytest.approx(200, abs=2)

    def test_a_slow_camera_gives_every_frame(self) -> None:
        stream = make_stream()
        watched(stream)
        assert feed(stream, 30, period_ns=NS_PER_S // 5) == 30  # 5 frames a second

    def test_the_rate_follows_the_setting(self) -> None:
        stream = make_stream(settings=PolarisSettings(max_fps=5.0))
        watched(stream)
        assert feed(stream, round(10 * 82.1)) == pytest.approx(50, abs=2)

    def test_the_kept_frames_are_spread_evenly(self) -> None:
        stream = make_stream()
        watched(stream)
        gaps = np.diff(kept_times(stream, 900)) / NS_PER_S
        assert gaps.min() >= 0.048  # four frames of 12.2 ms
        assert gaps.max() <= 0.062  # five frames
        assert gaps.mean() == pytest.approx(0.05, abs=0.002)

    def test_lost_frames_do_not_change_the_cadence_and_they_count_toward_the_rate(self) -> None:
        stream = make_stream()
        watched(stream)
        for index in range(0, 500, 2):  # the camera lost every second frame
            stream.offer(fast_frame(index, dropped_before=1), UPDATE)
        assert stream.frames_kept == pytest.approx(500 * 0.01218 * 20, abs=2)
        assert stream._count == 500  # 250 frames arrived, and 250 were lost

    def test_a_new_stream_is_kept_at_once(self) -> None:
        stream = make_stream()
        watched(stream)
        feed(stream, 3)
        assert stream.frames_kept == 1
        stream.offer(fast_frame(3, stream_id=2), UPDATE)
        assert stream.frames_kept == 2
        assert stream._slot is not None
        assert stream._slot.stream_id == 2

    def test_a_step_back_in_time_does_not_freeze_the_video(self) -> None:
        stream = make_stream()
        watched(stream)
        feed(stream, 100)
        before = stream.frames_kept
        hour_ns = 3600 * NS_PER_S
        for index in range(100, 400):  # the clock stepped back by an hour
            stream.offer(fast_frame(index, t_utc_ns=T0 - hour_ns + index * PERIOD_NS), UPDATE)
        assert stream.frames_kept - before >= 70  # 300 frames of 12 ms, 20 frames a second

    def test_a_long_gap_gives_one_frame_and_no_burst_afterward(self) -> None:
        stream = make_stream()
        watched(stream)
        feed(stream, 50)
        before = stream.frames_kept
        feed(stream, 50, start=50 + 5000)  # the camera paused for a minute
        assert stream.frames_kept - before in (12, 13)  # 50 frames of 12 ms, thinned

    def test_a_kept_frame_is_a_copy_of_the_pixels(self) -> None:
        stream = make_stream()
        watched(stream)
        data = star_frame()
        original = data.copy()
        stream.offer(fast_frame(0, data=data), UPDATE)
        data[:] = 0  # the camera reuses its buffer
        slot = stream._slot
        assert slot is not None
        assert np.array_equal(slot.data, original)
        assert not np.shares_memory(slot.data, data)

    def test_a_read_only_frame_is_copied_too(self) -> None:
        stream = make_stream()
        watched(stream)
        data = star_frame()
        data.flags.writeable = False
        stream.offer(fast_frame(0, data=data), UPDATE)
        assert stream._slot is not None
        assert np.array_equal(stream._slot.data, data)

    def test_the_slot_holds_what_the_renderer_needs(self) -> None:
        stream = make_stream()
        watched(stream)
        roi = Roi(100, 200, 128, 128)
        stream.offer(fast_frame(7, data=star_frame(), roi=roi), UPDATE)
        slot = stream._slot
        assert slot is not None
        assert (slot.stream_id, slot.mode, slot.exposure_us, slot.gain) == (1, "bin1", 2000, 0)
        assert (slot.adc_bits, slot.roi, slot.star) == (12, roi, FOUND)
        assert slot.t_utc_ns == T0 + 7 * PERIOD_NS

    def test_a_slot_that_nobody_took_gives_way_to_the_next_frame(self) -> None:
        stream = make_stream()
        watched(stream)
        feed(stream, 30)  # about eight frames are kept, and nobody takes them
        assert stream.frames_kept in (7, 8)
        assert stream.frames_replaced == stream.frames_kept - 1
        assert stream._slot is not None
        assert stream._slot.t_utc_ns >= T0 + 25 * PERIOD_NS  # the newest one

    def test_the_video_stops_when_the_last_viewer_goes(self) -> None:
        stream = make_stream()
        client = watched(stream)
        feed(stream, 20)
        assert active(stream) is True
        client.closed = True
        stream.housekeeping()
        assert active(stream) is False
        assert stream.viewers == 0
        assert stream._slot is None  # the stale frame is dropped
        stream.offer(cast(Frame, Untouchable()), cast(FastUpdate, Untouchable()))  # not read


class TestFailure:
    def test_an_error_switches_the_video_off_and_never_reaches_the_scheduler(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        stream = make_stream(settings=PolarisSettings(disable_s=30.0))
        watched(stream)

        class Broken:
            """A frame whose pixels cannot be copied."""

            stream_id = 1
            t_utc_ns = T0
            dropped_before = 0

            @property
            def data(self) -> Any:
                raise MemoryError("no room for a copy")

        with caplog.at_level(logging.ERROR, logger="seeingmon.services.core.live"):
            stream.offer(cast(Frame, Broken()), UPDATE)
            stream.offer(cast(Frame, Broken()), UPDATE)
        assert stream.offer_errors == 1  # the second call found the video off
        assert active(stream) is False
        assert len(caplog.records) == 1
        assert "stays off for 30 s" in caplog.records[0].getMessage()
        assert caplog.records[0].exc_info is not None  # the log holds the traceback

    def test_the_video_comes_back_after_the_pause_if_somebody_still_watches(self) -> None:
        clock = VirtualClock(T0)
        stream = make_stream(clock=clock, settings=PolarisSettings(disable_s=30.0))
        watched(stream)
        stream._fail()
        stream.housekeeping()
        assert active(stream) is False
        clock.advance(29.0)
        stream.housekeeping()
        assert active(stream) is False
        clock.advance(2.0)
        stream.housekeeping()
        assert active(stream) is True
        assert feed(stream, 5) >= 1

    def test_a_new_viewer_during_the_pause_does_not_switch_the_video_on(self) -> None:
        clock = VirtualClock(T0)
        stream = make_stream(clock=clock, settings=PolarisSettings(disable_s=30.0))
        watched(stream)
        stream._fail()
        watched(stream)
        assert active(stream) is False
        clock.advance(31.0)
        stream.housekeeping()
        assert active(stream) is True


# --- The work of one loop turn ---------------------------------------------------------------


def one_slot(stream: PolarisStream, index: int = 0) -> FrameSlot:
    """Offer the frame with this index (use multiples of 5, so that the thinning keeps it)."""
    stream.offer(fast_frame(index, data=star_frame(seed=index)), UPDATE)
    slot = stream._slot
    assert slot is not None
    assert slot.t_utc_ns == T0 + index * PERIOD_NS
    return slot


class TestProcess:
    def test_the_payload_holds_a_png_and_the_state_of_the_same_frame(self) -> None:
        stream = make_stream(live=live_seeing_view)
        client = watched(stream)
        slot = one_slot(stream)
        payload = stream.process(slot)
        assert client.payloads == [payload]
        decoded = unpack_polaris_frame(payload)
        assert decoded.state.seq == 0
        assert decoded.state.t_utc_ns == slot.t_utc_ns
        assert decoded.image.startswith(PNG_MAGIC)
        assert Image.open(io.BytesIO(decoded.image)).size == (128, 128)
        assert decoded.state.star.found is True
        assert decoded.state.live_seeing == live_seeing_view()
        assert stream.frames_encoded == 1
        assert stream.frames_sent == 1
        assert stream.bytes_sent == len(payload)
        assert stream.last_encode_s >= 0.0

    def test_every_client_gets_the_frame(self) -> None:
        stream = make_stream()
        clients = [watched(stream), watched(stream), watched(stream)]
        stream.process(one_slot(stream))
        assert [len(client.payloads) for client in clients] == [1, 1, 1]
        assert stream.frames_sent == 3

    def test_a_client_with_no_room_skips_the_frame_and_the_others_still_get_it(self) -> None:
        stream = make_stream()
        slow, quick = watched(stream, FakeSender(credit=False)), watched(stream)
        stream.process(one_slot(stream))
        assert (len(slow.payloads), len(quick.payloads)) == (0, 1)
        assert stream.frames_skipped == 1
        slow.credit = True
        stream.process(one_slot(stream, 5))
        assert len(slow.payloads) == 1  # the newest frame, with no backlog

    def test_a_closed_client_is_dropped(self) -> None:
        stream = make_stream()
        gone = watched(stream)
        stay = watched(stream)
        gone.closed = True
        stream.process(one_slot(stream))
        assert gone.payloads == []
        assert len(stay.payloads) == 1
        assert stream.viewers == 1

    def test_a_client_that_fails_to_send_is_dropped_and_the_stream_goes_on(self) -> None:
        stream = make_stream()
        watched(stream, BreakingSender())
        stay = watched(stream)
        stream.process(one_slot(stream))
        assert len(stay.payloads) == 1
        assert len(stream._senders) == 1
        assert stream._senders[0] is cast(StreamSender, stay)

    def test_the_rolling_value_is_read_for_each_frame_and_none_gives_null(self) -> None:
        values: list[LiveSeeingView | None] = [None, live_seeing_view(seeing_fwhm_arcsec=1.7)]
        stream = make_stream(live=lambda: values.pop(0))
        client = watched(stream)
        stream.process(one_slot(stream, 0))
        stream.process(one_slot(stream, 5))
        first, second = (unpack_polaris_frame(payload).state for payload in client.payloads)
        assert first.live_seeing is None
        assert second.live_seeing is not None
        assert second.live_seeing.seeing_fwhm_arcsec == 1.7

    def test_the_rolling_value_of_the_analyzer_becomes_the_view_of_the_contract(self) -> None:
        @dataclasses.dataclass(frozen=True)
        class Live:
            t_utc_ns: int = T0
            span_s: float = 10.0
            n_frames: int = 820
            n_usable: int = 815
            valid_fraction: float = 0.99
            seeing_fwhm_arcsec: float | None = 1.5
            seeing_fwhm_structure_arcsec: float | None = None
            r0_cm: float | None = 6.5
            r0_structure_cm: float | None = None
            image_motion_rms_x_arcsec: float | None = 0.6
            image_motion_rms_y_arcsec: float | None = 0.5
            width_fwhm_arcsec: float | None = 2.8
            stream_id: int = 3
            readout_mode: str = "bin1"
            exposure_us: int = 2000
            flags: tuple[str, ...] = ("cloud",)
            quality: dict[str, str] = dataclasses.field(
                default_factory=lambda: {"seeing_fwhm_structure_arcsec": "too few pairs"}
            )

        view = live_view(Live())
        assert view is not None
        assert view.flags == ["cloud"]
        assert view.quality == {"seeing_fwhm_structure_arcsec": "too few pairs"}
        assert view.seeing_fwhm_arcsec == 1.5
        assert view.r0_structure_cm is None
        assert live_view(None) is None


# --- Streams to the clients ------------------------------------------------------------------


class Collector:
    def __init__(self, stream: PolarisStream) -> None:
        self.stream = stream
        self.senders: list[StreamSender] = []

    def on_sender(self, sender: StreamSender, params: Any) -> None:
        self.senders.append(sender)
        self.stream.attach(sender, params)


@pytest.fixture
def receivers() -> Iterator[list[StreamReceiver]]:
    opened: list[StreamReceiver] = []
    yield opened
    for receiver in opened:
        receiver.close()


@pytest.fixture
def streams() -> Iterator[list[PolarisStream]]:
    made: list[PolarisStream] = []
    yield made
    for stream in made:
        stream.stop()


@pytest.fixture
def viewer(
    start_server: Callable[..., IpcServer],
    key: ConnectionKey,
    receivers: list[StreamReceiver],
    streams: list[PolarisStream],
) -> Callable[..., tuple[PolarisStream, StreamReceiver]]:
    def open_viewer(
        window: StreamWindow | None = None, **options: Any
    ) -> tuple[PolarisStream, StreamReceiver]:
        stream = make_stream(**options)
        streams.append(stream)
        collector = Collector(stream)
        server = start_server({POLARIS_CHANNEL: StreamService(collector.on_sender)})
        receiver, _ = connect_stream(
            server.endpoint, key, {"role": "web"}, channel=POLARIS_CHANNEL, window=window
        )
        receivers.append(receiver)
        assert wait_until(lambda: stream.viewers == 1)
        return stream, receiver

    return open_viewer


Viewer = Callable[..., tuple[PolarisStream, StreamReceiver]]


class TestStreams:
    def test_a_client_gets_every_frame_in_order_over_the_connection(self, viewer: Viewer) -> None:
        stream, receiver = viewer()
        for count in range(3):
            stream.process(one_slot(stream, count * 5))
            message = receiver.recv(10.0)
            assert message is not None
            assert message.kind is StreamKind.DATA
            decoded = unpack_polaris_frame(message.payload)
            assert decoded.state.t_utc_ns == T0 + count * 5 * PERIOD_NS
            assert decoded.image.startswith(PNG_MAGIC)
        assert (stream.frames_sent, stream.frames_skipped) == (3, 0)

    def test_a_slow_client_makes_the_stream_skip_frames_and_never_queue_them(
        self, viewer: Viewer
    ) -> None:
        stream, receiver = viewer(StreamWindow(messages=1, bytes=64 * 1024 * 1024))
        for count in range(5):
            stream.process(one_slot(stream, count * 5))  # the receiver reads nothing meanwhile
        assert (stream.frames_sent, stream.frames_skipped) == (1, 4)
        first = receiver.recv(10.0)  # taking the message returns the credit
        assert first is not None
        assert wait_until(lambda: stream._senders[0].has_credit(10_000))
        stream.process(one_slot(stream, 30))
        second = receiver.recv(10.0)
        assert second is not None
        assert unpack_polaris_frame(second.payload).state.t_utc_ns == T0 + 30 * PERIOD_NS

    def test_a_client_that_leaves_is_dropped_and_the_video_stops(self, viewer: Viewer) -> None:
        stream, receiver = viewer()
        assert active(stream) is True
        receiver.close()

        def gone() -> bool:
            stream.housekeeping()
            return stream.viewers == 0

        assert wait_until(gone)
        assert active(stream) is False

    def test_the_thread_encodes_the_newest_slot_and_stops_cleanly(self, viewer: Viewer) -> None:
        stream, receiver = viewer()
        stream.start()
        feed(stream, 40)  # about eight frames are kept, and the thread takes the newest each time
        message = receiver.recv(10.0)
        assert message is not None
        assert unpack_polaris_frame(message.payload).image.startswith(PNG_MAGIC)
        assert wait_until(lambda: stream.frames_encoded >= 1)
        stream.stop()
        stream.stop()  # safe to call twice
        assert active(stream) is False
        assert stream.viewers == 0  # the stop closed the streams

    def test_the_thread_counts_a_frame_that_cannot_be_rendered_and_goes_on(
        self, viewer: Viewer, caplog: pytest.LogCaptureFixture
    ) -> None:
        stream, receiver = viewer(renderer=FailingRenderer(failures=1))
        with caplog.at_level(logging.ERROR, logger="seeingmon.services.core.live"):
            stream.start()
            stream.offer(fast_frame(0, data=star_frame()), UPDATE)
            assert wait_until(lambda: stream.encode_errors == 1)
            stream.offer(fast_frame(5, data=star_frame()), UPDATE)
            message = receiver.recv(10.0)
        assert message is not None
        assert unpack_polaris_frame(message.payload).state.t_utc_ns == T0 + 5 * PERIOD_NS
        assert any("could not be encoded" in record.getMessage() for record in caplog.records)
        assert stream.frames_encoded == 1


def test_a_second_start_is_refused() -> None:
    stream = make_stream()
    stream.start()
    try:
        with pytest.raises(RuntimeError, match="already runs"):
            stream.start()
    finally:
        stream.stop()


# --- The cost on the scheduler thread --------------------------------------------------------


def mean_cost_us(stream: PolarisStream, frames: list[Frame], rounds: int = 5) -> float:
    """The best mean time of one `offer` over several rounds, in microseconds."""
    best = float("inf")
    for _ in range(rounds):
        started = time.perf_counter()
        for frame in frames:
            stream.offer(frame, UPDATE)
        best = min(best, (time.perf_counter() - started) / len(frames))
    return best * 1e6


def test_the_scheduler_thread_pays_almost_nothing_without_a_viewer() -> None:
    stream = make_stream()
    frames = [fast_frame(i, data=star_frame()) for i in range(2000)]
    assert (
        mean_cost_us(stream, frames) < 20.0
    )  # about 0.2 microseconds; the bound allows a busy host


def test_the_scheduler_thread_pays_a_few_microseconds_with_a_viewer() -> None:
    stream = make_stream()
    watched(stream)
    data = star_frame()
    frames = [fast_frame(i, data=data) for i in range(2000)]
    assert (
        mean_cost_us(stream, frames) < 200.0
    )  # about 5 microseconds; the bound allows a busy host
