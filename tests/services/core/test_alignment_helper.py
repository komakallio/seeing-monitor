"""The alignment helper: frames in, JPEGs and states out, streams that never queue."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.frames import Frame
from seeingmon.profile import Profile, load_profile
from seeingmon.services.core.alignment.helper import AlignmentHelper
from seeingmon.services.core.alignment.solve import QuickSolution
from seeingmon.services.core.settings import AlignmentSettings
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
from seeingmon.services.web.contract import unpack_frame
from tests.scheduler.helpers import make_frame

from ..conftest import wait_until
from .rig import sky_frame

PIL = pytest.importorskip("PIL.Image", reason="the preview needs Pillow")

T0 = 1_800_000_000 * NS_PER_S
SETTINGS = AlignmentSettings(
    target_x_px=300.0, target_y_px=200.0, target_roll_deg=0.0, histogram_bins=16
)


@pytest.fixture(scope="module")
def profile() -> Profile:
    return load_profile("asi294mm-gs250")


def solution(**changes: Any) -> QuickSolution:
    fields: dict[str, Any] = {
        "t_utc_ns": T0,
        "seq": 1,
        "solved": True,
        "x_px": 303.0,
        "y_px": 196.0,
        "roll_deg": 1.5,
        "n_matched": 60,
        "rms_arcsec": 0.7,
        "scale_arcsec_px": 3.82,
        "solver": "tracker",
        "n_detected": 90,
        "focus_fwhm_px": 2.5,
        "n_focus_stars": 30,
    }
    fields.update(changes)
    return QuickSolution(**fields)


class StubSolver:
    def __init__(self, result: QuickSolution | None = None) -> None:
        self.result = result or solution()
        self.frames: list[int] = []

    def solve(self, frame: Frame) -> QuickSolution:
        self.frames.append(frame.seq)
        return self.result


class Active:
    def __init__(self, value: bool = True) -> None:
        self.value = value

    def __call__(self) -> bool:
        return self.value


Build = Callable[..., AlignmentHelper]


@pytest.fixture
def build(profile: Profile) -> Iterator[Build]:
    helpers: list[AlignmentHelper] = []

    def make(**parts: Any) -> AlignmentHelper:
        clock = parts.pop("clock", VirtualClock(T0))
        parts.setdefault("settings", SETTINGS)
        parts.setdefault("is_active", Active())
        helper = AlignmentHelper(profile=profile, clock=clock, **parts)
        helpers.append(helper)
        return helper

    yield make
    for helper in helpers:
        helper.stop()


class TestOneFrame:
    def test_the_payload_holds_a_jpeg_and_the_state_of_the_same_frame(self, build: Build) -> None:
        helper = build(solver=StubSolver())
        frame = sky_frame(seq=7)
        helper.solve_frame(frame)
        payload = helper.process_frame(frame)
        decoded = unpack_frame(payload)
        assert PIL.open(io.BytesIO(decoded.jpeg)).size == (640, 480)
        state = decoded.state
        assert state.active is True
        assert state.frame is not None
        assert (state.frame.seq, state.frame.width_px, state.frame.height_px) == (7, 640, 480)
        assert state.frame.readout_mode == "bin2"
        assert state.frame.exposure_s == 0.5
        assert state.frame.plate_scale_arcsec_px == pytest.approx(3.82, abs=0.01)
        assert state.offset is not None
        assert (state.offset.dx_px, state.offset.dy_px) == (3.0, -4.0)
        assert state.offset.distance_px == 5.0
        assert state.offset.roll_deg == 1.5
        assert state.focus is not None
        assert (state.focus.fwhm_px, state.focus.best_fwhm_px) == (2.5, 2.5)
        assert state.histogram is not None
        assert len(state.histogram.counts) == 16
        assert state.histogram.max_dn == pytest.approx(65532.0)
        assert state.saturation is not None
        assert state.saturation.warning is False

    def test_a_saturated_frame_raises_the_warning(self, build: Build) -> None:
        helper = build()
        state = unpack_frame(helper.process_frame(sky_frame(saturate=0.05))).state
        assert state.saturation is not None
        assert state.saturation.fraction == pytest.approx(0.05, abs=0.002)
        assert state.saturation.warning is True
        assert state.histogram is not None
        assert state.histogram.counts[-1] > 0  # the saturated pixels sit in the top bin

    def test_without_a_solver_the_state_says_so(self, build: Build) -> None:
        helper = build()
        state = unpack_frame(helper.process_frame(sky_frame())).state
        assert state.solved is None
        assert "not available" in state.quality["solved"]
        assert helper.solve_frame(sky_frame()) is None

    def test_the_best_focus_of_the_session_is_the_smallest(self, build: Build) -> None:
        solver = StubSolver()
        helper = build(solver=solver)
        for seq, fwhm in enumerate([3.0, 2.2, 2.8], start=1):
            solver.result = solution(seq=seq, focus_fwhm_px=fwhm)
            helper.solve_frame(sky_frame(seq))
        state = unpack_frame(helper.process_frame(sky_frame(4))).state
        assert state.focus is not None
        assert (state.focus.fwhm_px, state.focus.best_fwhm_px) == (2.8, 2.2)
        assert (helper.solves, helper.solve_failures) == (3, 0)

    def test_a_failed_solve_is_counted_and_reported(self, build: Build) -> None:
        solver = StubSolver(solution(solved=False, x_px=None, y_px=None, note="too few stars"))
        helper = build(solver=solver)
        helper.solve_frame(sky_frame())
        state = unpack_frame(helper.process_frame(sky_frame())).state
        assert state.solved is None
        assert state.quality["solved"] == "too few stars"
        assert helper.solve_failures == 1


class TestTheState:
    def test_outside_alignment_the_state_is_inactive(self, build: Build) -> None:
        active = Active(False)
        helper = build(is_active=active)
        assert helper.state().active is False

    def test_a_session_without_frames_says_so(self, build: Build) -> None:
        helper = build()
        state = helper.state()
        assert state.active is True
        assert state.frame is None
        assert state.quality == {"frame": "no frame has arrived yet"}

    def test_the_state_follows_the_latest_solve_between_frames(self, build: Build) -> None:
        solver = StubSolver()
        helper = build(solver=solver)
        helper.process_frame(sky_frame())
        assert helper.state().solved is None
        helper.solve_frame(sky_frame())
        state = helper.state()
        assert state.solved is not None
        assert state.offset is not None

    def test_ending_the_alignment_forgets_the_session(self, build: Build) -> None:
        active = Active(True)
        solver = StubSolver()
        helper = build(is_active=active, solver=solver)
        helper.solve_frame(sky_frame())
        helper.process_frame(sky_frame())
        assert helper.state().focus is not None
        active.value = False
        assert helper.state().active is False
        active.value = True
        assert helper.state().frame is None  # a new session starts with nothing
        helper.process_frame(sky_frame())
        assert helper.state().focus is None  # and with no best focus


class TestIntake:
    def test_a_frame_that_the_encoder_has_not_taken_is_replaced_and_counted(
        self, build: Build
    ) -> None:
        helper = build()
        for seq in range(1, 6):
            helper.sink(sky_frame(seq))
        assert (helper.frames_received, helper.frames_dropped) == (5, 4)

    def test_the_sink_returns_at_once_whatever_the_load(self, build: Build) -> None:
        helper = build()
        big = make_frame(np.zeros((2822, 4144), dtype=np.uint16), t_utc_ns=T0)
        helper.sink(big)  # the full bin2 frame: the sink only stores a reference
        assert helper.frames_encoded == 0


# --- Streams -----------------------------------------------------------------------------------


def read_until_closed(receiver: StreamReceiver) -> None:
    """Receive until the stream closes, which raises `IpcClosedError`."""
    for _ in range(50):
        receiver.recv(0.2)


class Collector:
    """Takes the senders that the stream service hands out."""

    def __init__(self, helper: AlignmentHelper) -> None:
        self.helper = helper
        self.senders: list[StreamSender] = []

    def on_sender(self, sender: StreamSender, params: Any) -> None:
        self.senders.append(sender)
        self.helper.attach(sender, params)


@pytest.fixture
def receivers() -> Iterator[list[StreamReceiver]]:
    opened: list[StreamReceiver] = []
    yield opened
    for receiver in opened:
        receiver.close()


@pytest.fixture
def viewer(
    build: Build,
    start_server: Callable[..., IpcServer],
    key: ConnectionKey,
    receivers: list[StreamReceiver],
) -> Callable[..., tuple[AlignmentHelper, StreamReceiver]]:
    def open_viewer(
        window: StreamWindow | None = None, **parts: Any
    ) -> tuple[AlignmentHelper, StreamReceiver]:
        helper = build(**parts)
        collector = Collector(helper)
        server = start_server({"alignment": StreamService(collector.on_sender)})
        receiver, _ = connect_stream(
            server.endpoint, key, {"role": "web"}, channel="alignment", window=window
        )
        receivers.append(receiver)
        assert wait_until(lambda: helper.viewers == 1)
        return helper, receiver

    return open_viewer


class TestStreams:
    def test_a_viewer_gets_every_encoded_frame_in_order(
        self, viewer: Callable[..., tuple[AlignmentHelper, StreamReceiver]]
    ) -> None:
        helper, receiver = viewer(solver=StubSolver())
        for seq in (1, 2, 3):
            helper.process_frame(sky_frame(seq))
            message = receiver.recv(10.0)
            assert message is not None
            assert message.kind is StreamKind.DATA
            decoded = unpack_frame(message.payload)
            assert decoded.state.frame is not None
            assert decoded.state.frame.seq == seq
        assert (helper.frames_sent, helper.frames_skipped) == (3, 0)

    def test_a_slow_viewer_makes_the_helper_skip_frames_and_never_queue_them(
        self, viewer: Callable[..., tuple[AlignmentHelper, StreamReceiver]]
    ) -> None:
        helper, receiver = viewer(StreamWindow(messages=1, bytes=64 * 1024 * 1024))
        for seq in range(1, 6):
            helper.process_frame(sky_frame(seq))  # the receiver reads nothing meanwhile
        assert (helper.frames_sent, helper.frames_skipped) == (1, 4)
        first = receiver.recv(10.0)  # taking the message returns the credit
        assert first is not None
        assert wait_until(lambda: helper._senders[0].has_credit(1000))
        helper.process_frame(sky_frame(6))
        second = receiver.recv(10.0)
        assert second is not None
        decoded = unpack_frame(second.payload)
        assert decoded.state.frame is not None
        assert decoded.state.frame.seq == 6  # the newest, not one of the skipped frames

    def test_a_viewer_that_leaves_is_dropped(
        self,
        viewer: Callable[..., tuple[AlignmentHelper, StreamReceiver]],
        receivers: list[StreamReceiver],
    ) -> None:
        helper, receiver = viewer()
        receiver.close()

        def gone() -> bool:
            helper.housekeeping()
            return helper.viewers == 0

        assert wait_until(gone)
        helper.process_frame(sky_frame())  # with no viewer, the frame goes nowhere
        assert helper.frames_sent == 0

    def test_a_watched_alignment_touches_the_scheduler_now_and_then(
        self, viewer: Callable[..., tuple[AlignmentHelper, StreamReceiver]]
    ) -> None:
        clock = VirtualClock(T0)
        touches: list[int] = []
        helper, _ = viewer(clock=clock, touch=lambda: touches.append(clock.monotonic_ns()))
        helper.housekeeping()
        helper.housekeeping()  # at once again: the interval has not passed
        assert len(touches) == 1
        clock.advance(SETTINGS.touch_interval_s + 1)
        helper.housekeeping()
        assert len(touches) == 2

    def test_nobody_watching_means_no_touch(self, build: Build) -> None:
        touches: list[int] = []
        helper = build(touch=lambda: touches.append(1))
        helper.housekeeping()
        assert touches == []

    def test_an_inactive_alignment_does_not_touch_the_scheduler(
        self, viewer: Callable[..., tuple[AlignmentHelper, StreamReceiver]]
    ) -> None:
        touches: list[int] = []
        helper, _ = viewer(is_active=Active(False), touch=lambda: touches.append(1))
        helper.housekeeping()
        assert touches == []


# --- Threads -----------------------------------------------------------------------------------


class TestThreads:
    def test_the_threads_encode_and_solve_the_newest_frame(self, build: Build) -> None:
        solver = StubSolver()
        helper = build(solver=solver)
        helper.start()
        helper.sink(sky_frame(1))
        assert wait_until(lambda: helper.frames_encoded == 1 and helper.solves == 1)
        assert solver.frames == [1]
        assert helper.state().frame is not None

    def test_the_solver_starts_a_solve_only_after_the_interval(self, build: Build) -> None:
        solver = StubSolver()
        clock = VirtualClock(T0)
        helper = build(solver=solver, clock=clock)
        helper.start()
        helper.sink(sky_frame(1))
        assert wait_until(lambda: len(solver.frames) == 1)
        helper.sink(sky_frame(2))
        assert wait_until(lambda: helper.frames_encoded >= 1)
        assert solver.frames == [1]  # the second frame waits: the clock did not move
        clock.advance(SETTINGS.solve_interval_s + 0.5)
        assert wait_until(lambda: solver.frames == [1, 2])

    def test_the_encoder_keeps_the_pace_of_the_minimum_interval(self, build: Build) -> None:
        clock = VirtualClock(T0)
        helper = build(clock=clock, settings=AlignmentSettings(min_interval_s=0.5))
        helper.start()
        helper.sink(sky_frame(1))
        assert wait_until(lambda: helper.frames_encoded == 1)
        helper.sink(sky_frame(2))
        helper.sink(sky_frame(3))  # replaces the second one, which waits for its turn
        assert helper.frames_encoded == 1
        clock.advance(1.0)
        assert wait_until(lambda: helper.frames_encoded == 2)
        decoded = unpack_frame(helper.process_frame(sky_frame(9)))
        assert decoded.state.frame is not None

    def test_a_frame_that_cannot_be_encoded_is_counted_and_the_thread_goes_on(
        self, build: Build
    ) -> None:
        helper = build(settings=AlignmentSettings(min_interval_s=0.0))
        helper.start()
        helper.sink(make_frame(np.zeros((4, 4), dtype=np.uint16), mode="no-such-mode"))
        assert wait_until(lambda: helper.encode_errors == 1)
        helper.sink(sky_frame(2))
        assert wait_until(lambda: helper.frames_encoded == 1)

    def test_stopping_twice_is_safe_and_closes_the_streams(
        self, viewer: Callable[..., tuple[AlignmentHelper, StreamReceiver]]
    ) -> None:
        helper, receiver = viewer()
        helper.start()
        helper.stop()
        helper.stop()
        assert helper.viewers == 0
        with pytest.raises(IpcClosedError):
            read_until_closed(receiver)
