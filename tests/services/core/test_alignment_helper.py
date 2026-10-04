"""The alignment helper: frames in, JPEGs and states out, streams that never queue."""

from __future__ import annotations

import dataclasses
import io
import logging
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, VirtualClock, utc_ns_to_iso
from seeingmon.frames import Frame, FrameFlag
from seeingmon.profile import Profile, load_profile
from seeingmon.scheduler.config import SiteConfig
from seeingmon.services.core.alignment.calibration import PreviewCalibrator
from seeingmon.services.core.alignment.helper import AlignmentHelper
from seeingmon.services.core.alignment.preview import make_preview
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
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.dark import DarkLibrary
from seeingmon.survey.geometry import ARCSEC_PER_RAD
from seeingmon.survey.sky import SkyError
from seeingmon.survey.wcs_fit import CameraAttitude, pixel_center
from tests.scheduler.helpers import make_frame
from tests.survey.synth import make_attitude

from ..conftest import wait_until
from . import previewfx
from .rig import sky_frame

PIL = pytest.importorskip("PIL.Image", reason="the preview needs Pillow")

T0 = 1_800_000_000 * NS_PER_S
SETTINGS = AlignmentSettings(
    target_x_px=300.0, target_y_px=200.0, target_roll_deg=0.0, histogram_bins=16
)
SPACED = SETTINGS.model_copy(update={"solve_interval_s": 1.0})
FAST = SETTINGS.model_copy(update={"min_interval_s": 0.0})  # no pause between two previews


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


def camera(width: int = 640, height: int = 480, distance_deg: float = 0.2) -> CameraAttitude:
    """A camera whose pole lies `distance_deg` from the center of a `width` x `height` frame."""
    return CameraAttitude(
        rotation=make_attitude(distance_deg, 40.0, -65.0),
        scale_rad_px=3.82 / ARCSEC_PER_RAD,
        parity=1,
        center_px=pixel_center(width, height),
    )


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
        assert state.sky is None
        assert "not available" in state.quality["sky"]
        assert helper.solve_frame(sky_frame()) is None

    def test_the_state_carries_the_sky_of_the_latest_solution(self, build: Build) -> None:
        solver = StubSolver(solution(attitude=camera(), polaris_colatitude_deg=0.6265))
        helper = build(solver=solver)
        frame = sky_frame(seq=3)
        helper.solve_frame(frame)
        state = unpack_frame(helper.process_frame(frame)).state
        assert state.sky is not None
        assert state.sky.pole.in_front
        assert state.sky.polaris_colatitude_deg == 0.6265
        assert state.sky.orbit is not None
        pole = camera().pole_pixel()
        assert pole is not None
        assert state.sky.pole.dx_px == pytest.approx(
            pole[0] - 319.5, abs=0.01
        )  # the frame is 640 x 480
        assert state.sky.pole.dy_px == pytest.approx(pole[1] - 239.5, abs=0.01)
        assert "sky" not in state.quality

    def test_the_state_carries_the_reticle_with_and_without_a_solution(self, build: Build) -> None:
        helper = build()  # no solver, so there is never a solution
        state = unpack_frame(helper.process_frame(sky_frame())).state
        assert state.sky is None
        assert state.reticle is not None
        assert (state.reticle.x_px, state.reticle.y_px) == (319.5, 239.5)  # the frame is 640 x 480
        assert state.reticle.radius_px == pytest.approx(580.0, abs=15.0)  # about 0.62 degrees

    def test_a_site_gives_the_state_the_move_in_altitude_and_azimuth(self, build: Build) -> None:
        site = SiteConfig(latitude_deg=50.0, longitude_deg=10.0)  # a synthetic site
        solver = StubSolver(solution(attitude=camera(), polaris_colatitude_deg=0.6265))
        with_site = build(solver=solver, site=site)
        frame = sky_frame(seq=3)
        with_site.solve_frame(frame)
        state = unpack_frame(with_site.process_frame(frame)).state
        assert state.sky is not None
        assert state.sky.altitude_arcmin is not None
        assert state.sky.azimuth_arcmin is not None
        assert state.sky.axes is not None
        without_site = build(solver=StubSolver(solution(attitude=camera())))
        without_site.solve_frame(frame)
        plain = unpack_frame(without_site.process_frame(frame)).state
        assert plain.sky is not None
        assert plain.sky.altitude_arcmin is None
        assert plain.sky.axes is None

    def test_a_helper_without_a_target_still_has_the_sky_and_the_solved_position(
        self, build: Build
    ) -> None:
        solver = StubSolver(solution(attitude=camera(), polaris_colatitude_deg=0.6265))
        helper = build(solver=solver, settings=AlignmentSettings(histogram_bins=16))
        helper.solve_frame(sky_frame())
        state = unpack_frame(helper.process_frame(sky_frame())).state
        assert state.target is None
        assert state.offset is None
        assert state.solved is not None
        assert state.sky is not None

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


class TestTheCalibratedLiveView:
    def calibrator(self, tmp_path: Path, profile: Profile, **parts: Any) -> PreviewCalibrator:
        """A calibrator for the sensor of `previewfx`: its flat and its dark library."""
        flat = previewfx.write_flat(tmp_path / "flat.npy", previewfx.sensitivity())
        library = DarkLibrary(tmp_path / "calibration" / "darks")
        previewfx.add_dark_set(library)
        config = SurveyConfig(flat_file=str(flat), calibration_dir=str(tmp_path / "calibration"))
        return PreviewCalibrator(config, profile, **parts)

    def plain(self, frame: Frame) -> bytes:
        return make_preview(
            frame.data,
            max_pixels=SETTINGS.max_preview_pixels,
            quality=SETTINGS.jpeg_quality,
        ).jpeg

    def test_the_jpeg_comes_from_the_calibrated_frame(
        self, build: Build, tmp_path: Path, profile: Profile
    ) -> None:
        calibrator = self.calibrator(tmp_path, profile)
        helper = build(calibrator=calibrator)
        frame = previewfx.make_survey_frame(exposure_s=0.5)  # the exposure of the alignment view
        jpeg = unpack_frame(helper.process_frame(frame)).jpeg
        calibrated = make_preview(
            frame.data,
            max_pixels=SETTINGS.max_preview_pixels,
            quality=SETTINGS.jpeg_quality,
            calibration=calibrator.for_frame(frame),
        )
        assert jpeg == calibrated.jpeg
        assert jpeg != self.plain(frame)

    def test_the_measures_still_come_from_the_raw_frame(
        self, build: Build, tmp_path: Path, profile: Profile
    ) -> None:
        frame = previewfx.make_survey_frame(exposure_s=0.5)
        calibrated = unpack_frame(
            build(calibrator=self.calibrator(tmp_path, profile)).process_frame(frame)
        ).state
        plain = unpack_frame(build().process_frame(frame)).state
        assert calibrated.histogram == plain.histogram
        assert calibrated.saturation == plain.saturation

    def test_a_calibration_that_fails_leaves_the_live_view_as_it_was(
        self, build: Build, tmp_path: Path, profile: Profile
    ) -> None:
        def broken() -> Any:
            raise SkyError("the flat is broken")

        helper = build(calibrator=self.calibrator(tmp_path, profile, flat_provider=broken))
        frame = previewfx.make_survey_frame(exposure_s=0.5)
        for _ in range(3):
            assert unpack_frame(helper.process_frame(frame)).jpeg == self.plain(frame)
        assert helper.frames_encoded == 3
        assert helper.encode_errors == 0

    def test_a_helper_without_a_calibrator_makes_the_old_jpeg(self, build: Build) -> None:
        frame = previewfx.make_survey_frame(exposure_s=0.5)
        assert unpack_frame(build().process_frame(frame)).jpeg == self.plain(frame)


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
        helper = build(solver=solver, clock=clock, settings=SPACED)
        helper.start()
        helper.sink(sky_frame(1))
        assert wait_until(lambda: len(solver.frames) == 1)
        helper.sink(sky_frame(2))
        assert wait_until(lambda: helper.frames_encoded >= 1)
        assert solver.frames == [1]  # the second frame waits: the clock did not move
        clock.advance(SPACED.solve_interval_s + 0.5)
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


# --- Decoupling --------------------------------------------------------------------------------


class BlockingSolver:
    """A solver that waits for the test, so that a solve stays open as long as the test wants."""

    def __init__(self) -> None:
        self.frames: list[int] = []
        self.started = threading.Semaphore(0)  # released when a solve starts
        self.go = threading.Semaphore(0)  # the test releases one solve at a time

    def solve(self, frame: Frame) -> QuickSolution:
        self.frames.append(frame.seq)
        self.started.release()
        assert self.go.acquire(timeout=30.0)
        return solution(seq=frame.seq, t_utc_ns=frame.t_utc_ns)


def newest_encoded(helper: AlignmentHelper, seq: int) -> bool:
    state = helper.state()
    return state.frame is not None and state.frame.seq == seq


class TestTheSolverNeverHoldsTheViewBack:
    def test_the_preview_of_each_frame_comes_out_while_a_solve_is_open(self, build: Build) -> None:
        solver = BlockingSolver()
        helper = build(solver=solver, settings=FAST)
        helper.start()
        helper.sink(sky_frame(1))
        assert solver.started.acquire(timeout=10.0)  # the solver works on frame 1, and stays there
        for seq in range(2, 8):
            helper.sink(sky_frame(seq))
            assert wait_until(lambda seq=seq: newest_encoded(helper, seq))  # type: ignore[misc]
        assert solver.frames == [1]  # not one frame went to the solver meanwhile
        timing = helper.state().timing
        assert timing is not None
        assert (timing.frame_seq, timing.solving_frame_seq) == (7, 1)
        assert timing.solution_frame_seq is None  # no solve has finished
        solver.go.release()
        solver.go.release()

    def test_the_next_solve_starts_on_the_newest_frame_and_the_older_ones_are_dropped(
        self, build: Build
    ) -> None:
        solver = BlockingSolver()
        helper = build(solver=solver, settings=FAST)
        helper.start()
        helper.sink(sky_frame(1))
        assert solver.started.acquire(timeout=10.0)
        for seq in range(2, 8):
            helper.sink(sky_frame(seq))
        assert helper.frames_unsolved == 5  # frames 2 to 6 were replaced in the solver slot
        solver.go.release()  # frame 1 finishes: nothing interrupted it
        assert solver.started.acquire(timeout=10.0)
        assert solver.frames == [1, 7]
        solver.go.release()
        assert wait_until(lambda: helper.solves == 2)
        state = helper.state()
        assert state.timing is not None
        assert state.timing.solution_frame_seq == 7

    def test_a_fast_solver_follows_every_frame(self, build: Build) -> None:
        solver = StubSolver()
        helper = build(solver=solver, settings=FAST)  # the interval is zero by default
        helper.start()
        for seq in range(1, 6):
            helper.sink(sky_frame(seq))
            assert wait_until(lambda seq=seq: helper.solves == seq)  # type: ignore[misc]
        assert solver.frames == [1, 2, 3, 4, 5]

    def test_without_a_solver_the_solver_slot_stays_empty(self, build: Build) -> None:
        helper = build()
        for seq in range(1, 5):
            helper.sink(sky_frame(seq))
        assert helper.frames_unsolved == 0

    def test_a_solve_that_ends_after_the_alignment_does_not_show_in_the_next_session(
        self, build: Build
    ) -> None:
        active = Active(True)
        helpers: list[AlignmentHelper] = []

        class EndsTheAlignment:
            def solve(self, frame: Frame) -> QuickSolution:
                active.value = False  # the alignment ends while the solve runs
                helpers[0].state()  # and the next request of the page sees it
                return solution(seq=frame.seq)

        helper = build(is_active=active, solver=EndsTheAlignment())
        helpers.append(helper)
        helper.process_frame(sky_frame(1))
        helper.solve_frame(sky_frame(1))
        assert (helper.solves, helper.solve_failures) == (0, 0)
        active.value = True
        helper.process_frame(sky_frame(2))
        state = helper.state()
        assert state.solved is None  # the late solution is gone
        assert state.focus is None
        assert state.timing is not None
        assert state.timing.solution_frame_seq is None


class TestTheTimingOfTheState:
    def test_the_state_says_how_old_the_frame_is_and_where_the_time_went(
        self, build: Build
    ) -> None:
        clock = VirtualClock(T0)
        helper = build(clock=clock)
        frame = sky_frame(7, t_utc_ns=T0)
        received = T0 + round(0.15 * NS_PER_S)  # the frame reached core 0.15 s after its capture
        clock.advance(0.35)  # and the preview was ready 0.2 s later
        timing = unpack_frame(helper.process_frame(frame, received)).state.timing
        assert timing is not None
        assert timing.frame_seq == 7
        assert timing.frame_t_utc == utc_ns_to_iso(T0)
        assert timing.frame_age_s == pytest.approx(0.35)
        assert timing.receive_lag_s == pytest.approx(0.15)
        assert timing.preview_s == pytest.approx(0.2)
        assert (timing.solution_frame_seq, timing.solve_elapsed_s) == (None, None)
        assert (timing.solving_frame_seq, timing.solving_s) == (None, None)

    def test_the_age_of_the_frame_grows_in_the_state_that_a_request_reads(
        self, build: Build
    ) -> None:
        clock = VirtualClock(T0)
        helper = build(clock=clock)
        helper.process_frame(sky_frame(1, t_utc_ns=T0))
        clock.advance(4.0)  # no new frame for 4 seconds: the page can see the stall
        timing = helper.state().timing
        assert timing is not None
        assert timing.frame_age_s == pytest.approx(4.0)

    def test_a_frame_that_core_did_not_time_has_no_arrival_figures(self, build: Build) -> None:
        helper = build()
        timing = unpack_frame(helper.process_frame(sky_frame(1, t_utc_ns=T0))).state.timing
        assert timing is not None
        assert timing.receive_lag_s is None
        assert timing.frame_age_s is not None
        assert timing.preview_s == 0.0

    def test_the_state_names_the_frame_of_the_solution_and_the_time_of_the_solve(
        self, build: Build
    ) -> None:
        clock = VirtualClock(T0)

        class Slow:
            def solve(self, frame: Frame) -> QuickSolution:
                clock.advance(2.0)
                return solution(seq=frame.seq, t_utc_ns=frame.t_utc_ns)

        helper = build(clock=clock, solver=Slow())
        helper.solve_frame(sky_frame(5, t_utc_ns=T0))
        later = sky_frame(9, t_utc_ns=T0 + 4 * NS_PER_S)
        timing = unpack_frame(helper.process_frame(later)).state.timing
        assert timing is not None
        assert (timing.frame_seq, timing.solution_frame_seq) == (9, 5)
        assert timing.solve_elapsed_s == pytest.approx(2.0)
        assert helper.last_solve_s == pytest.approx(2.0)

    def test_a_solve_in_progress_shows_its_frame_and_its_age(self, build: Build) -> None:
        clock = VirtualClock(T0)
        solver = BlockingSolver()
        helper = build(clock=clock, solver=solver, settings=FAST)
        helper.start()
        helper.sink(sky_frame(1, t_utc_ns=T0))
        assert solver.started.acquire(timeout=10.0)
        assert wait_until(lambda: newest_encoded(helper, 1))
        clock.advance(1.5)
        state = helper.state()
        assert state.timing is not None
        assert (state.timing.solving_frame_seq, state.timing.solving_s) == (1, 1.5)
        assert state.quality["solved"] == "the first solve is running (frame 1, 2 s so far)"
        solver.go.release()
        assert wait_until(lambda: helper.solves == 1)
        finished = helper.state().timing
        assert finished is not None
        assert (finished.solving_frame_seq, finished.solving_s) == (None, None)

    def test_a_frame_without_a_valid_time_has_no_age(self, build: Build) -> None:
        helper = build()
        frame = dataclasses.replace(sky_frame(1, t_utc_ns=T0), flags=FrameFlag.TIME_INVALID)
        timing = unpack_frame(helper.process_frame(frame, T0)).state.timing
        assert timing is not None
        assert (timing.frame_age_s, timing.receive_lag_s) == (None, None)
        assert timing.frame_seq == 1  # the frame is still named

    def test_a_clock_that_runs_behind_the_frame_gives_an_age_of_zero(self, build: Build) -> None:
        helper = build(clock=VirtualClock(T0))
        frame = sky_frame(1, t_utc_ns=T0 + 5 * NS_PER_S)  # the frame claims a time in the future
        timing = unpack_frame(helper.process_frame(frame)).state.timing
        assert timing is not None
        assert timing.frame_age_s == 0.0


class TestTheLastSolution:
    def test_the_ring_survives_the_solves_that_fail(self, build: Build) -> None:
        solver = StubSolver(solution(attitude=camera(), polaris_colatitude_deg=0.6265, seq=1))
        helper = build(solver=solver)
        helper.solve_frame(sky_frame(1, t_utc_ns=T0))
        current = unpack_frame(helper.process_frame(sky_frame(1, t_utc_ns=T0))).state
        assert current.aim_ring is not None
        assert current.aim_ring.source == "current frame"
        solver.result = solution(
            seq=2, solved=False, x_px=None, y_px=None, attitude=None, note="too few stars"
        )
        helper.solve_frame(sky_frame(2, t_utc_ns=T0 + NS_PER_S))
        later = sky_frame(3, t_utc_ns=T0 + 30 * NS_PER_S)  # the solver found nothing for 30 s
        lost = unpack_frame(helper.process_frame(later)).state
        assert lost.solved is None
        assert lost.sky is None
        assert lost.aim_ring is not None
        assert lost.aim_ring.source == "last solution"
        assert lost.aim_ring.age_s == pytest.approx(30.0)
        assert lost.aim_ring.solution_frame_seq == 1
        assert lost.last_solution is not None
        assert (lost.last_solution.frame_seq, lost.last_solution.age_s) == (1, 30.0)
        assert lost.timing is not None
        assert lost.timing.solution_frame_seq == 2  # the latest solve failed on frame 2
        assert lost.reticle is not None

    def test_a_new_solution_takes_the_ring_back_to_the_current_frame(self, build: Build) -> None:
        solver = StubSolver(solution(attitude=camera(), seq=1))
        helper = build(solver=solver)
        helper.solve_frame(sky_frame(1, t_utc_ns=T0))
        solver.result = solution(seq=2, solved=False, x_px=None, y_px=None, attitude=None)
        helper.solve_frame(sky_frame(2, t_utc_ns=T0 + NS_PER_S))
        solver.result = solution(
            seq=3, attitude=camera(distance_deg=0.4), t_utc_ns=T0 + 2 * NS_PER_S
        )
        helper.solve_frame(sky_frame(3, t_utc_ns=T0 + 2 * NS_PER_S))
        state = unpack_frame(helper.process_frame(sky_frame(3, t_utc_ns=T0 + 2 * NS_PER_S))).state
        assert state.aim_ring is not None
        assert state.aim_ring.source == "current frame"
        assert state.last_solution is not None
        assert state.last_solution.frame_seq == 3

    def test_a_solution_without_an_attitude_is_not_kept(self, build: Build) -> None:
        helper = build(solver=StubSolver(solution(attitude=None)))
        helper.solve_frame(sky_frame(1, t_utc_ns=T0))
        state = unpack_frame(helper.process_frame(sky_frame(2, t_utc_ns=T0))).state
        assert state.last_solution is None
        assert state.aim_ring is None
        assert "aim_ring" in state.quality

    def test_the_end_of_the_alignment_forgets_the_last_solution(self, build: Build) -> None:
        active = Active(True)
        helper = build(is_active=active, solver=StubSolver(solution(attitude=camera(), seq=1)))
        helper.solve_frame(sky_frame(1, t_utc_ns=T0))
        helper.process_frame(sky_frame(1, t_utc_ns=T0))
        assert helper.state().last_solution is not None
        active.value = False
        assert helper.state().active is False
        active.value = True
        helper.process_frame(sky_frame(2, t_utc_ns=T0))
        state = helper.state()
        assert state.last_solution is None  # a new alignment starts with no solution
        assert state.aim_ring is None

    def test_without_a_solver_the_ring_has_the_same_reason_as_the_rest(self, build: Build) -> None:
        state = unpack_frame(build().process_frame(sky_frame(1))).state
        reason = "the quick solve is not available: no catalog is configured"
        assert state.quality["aim_ring"] == reason
        assert state.quality["solved"] == reason


class TestTheFocusHistory:
    def feed(
        self,
        helper: AlignmentHelper,
        solver: StubSolver,
        values: list[float | None],
        first_seq: int = 1,
    ) -> None:
        """Solve one frame for each value, half a second apart."""
        for offset, value in enumerate(values):
            seq = first_seq + offset
            t_ns = T0 + offset * NS_PER_S // 2
            solver.result = solution(seq=seq, t_utc_ns=t_ns, focus_fwhm_px=value)
            helper.solve_frame(sky_frame(seq, t_utc_ns=t_ns))

    def test_each_solve_adds_a_point_with_the_time_of_its_frame(self, build: Build) -> None:
        solver = StubSolver()
        helper = build(solver=solver)
        self.feed(helper, solver, [2.4, 2.5, 2.3])
        state = unpack_frame(helper.process_frame(sky_frame(4, t_utc_ns=T0 + 2 * NS_PER_S))).state
        assert state.focus is not None
        history = state.focus.history
        assert history is not None
        assert history.index == [1, 2, 3]
        assert history.seq == [1, 2, 3]
        assert history.t_utc_ms == [T0 // 1_000_000 + 500 * n for n in range(3)]
        assert history.fwhm_px == [2.4, 2.5, 2.3]
        assert (state.focus.fwhm_px, state.focus.best_fwhm_px) == (2.3, 2.3)

    def test_the_state_carries_the_arcseconds_of_the_plate_scale_of_the_frame(
        self, build: Build
    ) -> None:
        solver = StubSolver()
        helper = build(solver=solver)
        self.feed(helper, solver, [2.5])
        state = unpack_frame(helper.process_frame(sky_frame(1, t_utc_ns=T0))).state
        assert state.focus is not None
        assert state.focus.fwhm_arcsec == pytest.approx(2.5 * 3.82, abs=0.01)
        assert state.focus.best_fwhm_arcsec == pytest.approx(2.5 * 3.82, abs=0.01)

    def test_the_history_is_the_same_for_a_page_that_reloads(self, build: Build) -> None:
        solver = StubSolver()
        helper = build(solver=solver)
        self.feed(helper, solver, [2.4, 2.5, 2.3])
        helper.process_frame(sky_frame(3, t_utc_ns=T0))
        first = helper.state().focus
        second = helper.state().focus
        assert first is not None
        assert first == second
        assert first.history is not None
        assert first.history.reset is True  # a state always holds the whole history

    def test_a_spike_is_flagged_and_does_not_set_the_best_value(self, build: Build) -> None:
        solver = StubSolver()
        helper = build(solver=solver)
        self.feed(helper, solver, [2.4, 2.5, 2.4, 2.5, 7.0])
        state = unpack_frame(helper.process_frame(sky_frame(5, t_utc_ns=T0))).state
        assert state.focus is not None
        assert state.focus.spike is True
        assert state.focus.best_fwhm_px == 2.4
        assert state.focus.history is not None
        assert state.focus.history.spike == [False, False, False, False, True]

    def test_a_solve_without_a_value_adds_no_point_and_keeps_the_history(
        self, build: Build
    ) -> None:
        solver = StubSolver()
        helper = build(solver=solver)
        self.feed(helper, solver, [2.4, 2.5, None])
        state = unpack_frame(helper.process_frame(sky_frame(3, t_utc_ns=T0))).state
        assert state.focus is not None
        assert state.focus.fwhm_px is None
        assert state.focus.best_fwhm_px == 2.4
        assert state.focus.history is not None
        assert state.focus.history.index == [1, 2]

    def test_the_history_holds_120_values_at_most(self, build: Build) -> None:
        solver = StubSolver()
        helper = build(solver=solver)
        self.feed(helper, solver, [2.4 + 0.001 * (n % 5) for n in range(130)])
        state = unpack_frame(helper.process_frame(sky_frame(130, t_utc_ns=T0))).state
        assert state.focus is not None
        assert state.focus.history is not None
        assert state.focus.history.index == list(range(11, 131))

    def test_the_reset_restarts_the_best_value_and_keeps_the_history(self, build: Build) -> None:
        solver = StubSolver()
        helper = build(solver=solver)
        self.feed(helper, solver, [2.0, 2.4, 2.5])
        helper.reset_focus()
        state = unpack_frame(helper.process_frame(sky_frame(3, t_utc_ns=T0))).state
        assert state.focus is not None
        assert state.focus.best_fwhm_px is None
        assert state.focus.history is not None
        assert state.focus.history.index == [1, 2, 3]
        self.feed(helper, solver, [2.6], first_seq=4)
        later = unpack_frame(helper.process_frame(sky_frame(4, t_utc_ns=T0))).state
        assert later.focus is not None
        assert later.focus.best_fwhm_px == 2.6  # the next value that counts starts it again

    def test_a_new_alignment_starts_a_new_history(self, build: Build) -> None:
        active = Active(True)
        solver = StubSolver()
        helper = build(is_active=active, solver=solver)
        self.feed(helper, solver, [2.4, 2.5])
        helper.process_frame(sky_frame(2, t_utc_ns=T0))
        first = helper.state().focus
        assert first is not None
        assert first.history is not None
        active.value = False
        assert helper.state().active is False
        active.value = True
        helper.process_frame(sky_frame(3, t_utc_ns=T0))
        fresh = helper.state().focus
        assert fresh is None  # no value yet, and no history, and no best value
        self.feed(helper, solver, [2.8], first_seq=3)
        again = helper.state().focus
        assert again is not None
        assert again.history is not None
        assert again.history.session == first.history.session + 1  # a reader can tell
        assert again.history.index == [1]
        assert again.best_fwhm_px == 2.8

    def test_a_late_solve_does_not_add_a_point_to_the_next_session(self, build: Build) -> None:
        active = Active(True)
        helpers: list[AlignmentHelper] = []

        class EndsTheAlignment:
            def solve(self, frame: Frame) -> QuickSolution:
                active.value = False
                helpers[0].state()
                return solution(seq=frame.seq, focus_fwhm_px=2.4)

        helper = build(is_active=active, solver=EndsTheAlignment())
        helpers.append(helper)
        helper.process_frame(sky_frame(1))
        helper.solve_frame(sky_frame(1))
        active.value = True
        helper.process_frame(sky_frame(2))
        assert helper.state().focus is None


class Lifecycle(StubSolver):
    """A solver that has `release` and `close`, and counts the calls."""

    def __init__(self, release_error: bool = False, prepare_error: bool = False) -> None:
        super().__init__()
        self.prepared = 0
        self.released = 0
        self.closed = 0
        self._release_error = release_error
        self._prepare_error = prepare_error

    def prepare(self) -> None:
        self.prepared += 1
        if self._prepare_error:
            raise RuntimeError("cannot prepare")

    def release(self) -> None:
        self.released += 1
        if self._release_error:
            raise RuntimeError("cannot release")

    def close(self) -> None:
        self.closed += 1


class TestTheLifeOfTheSolver:
    def test_the_end_of_the_alignment_releases_the_solver_once(self, build: Build) -> None:
        active = Active(True)
        solver = Lifecycle()
        helper = build(is_active=active, solver=solver)
        helper.process_frame(sky_frame(1))
        helper.solve_frame(sky_frame(1))
        assert solver.released == 0  # the alignment runs
        active.value = False
        assert helper.state().active is False
        assert helper.state().active is False
        helper.housekeeping()
        assert solver.released == 1  # once, not for every look at the state

    def test_the_solver_is_told_once_when_the_alignment_starts(self, build: Build) -> None:
        solver = Lifecycle()
        helper = build(solver=solver)
        helper.housekeeping()
        helper.housekeeping()
        assert solver.prepared == 1  # a worker loads while the first frames arrive

    def test_a_new_alignment_prepares_the_solver_again(self, build: Build) -> None:
        active = Active(True)
        solver = Lifecycle()
        helper = build(is_active=active, solver=solver)
        helper.housekeeping()
        active.value = False
        helper.housekeeping()  # the alignment ended
        active.value = True
        helper.housekeeping()
        assert (solver.prepared, solver.released) == (2, 1)

    def test_an_alignment_that_ends_before_a_frame_arrived_still_releases_the_solver(
        self, build: Build
    ) -> None:
        active = Active(True)
        solver = Lifecycle()
        helper = build(is_active=active, solver=solver)
        helper.housekeeping()  # prepared, and no frame yet
        active.value = False
        helper.housekeeping()
        assert (solver.prepared, solver.released) == (1, 1)

    def test_a_solver_that_fails_to_prepare_does_not_stop_the_helper(
        self, build: Build, caplog: pytest.LogCaptureFixture
    ) -> None:
        solver = Lifecycle(prepare_error=True)
        helper = build(solver=solver)
        with caplog.at_level(logging.ERROR):
            helper.housekeeping()
            helper.housekeeping()
        assert "failed to prepare" in caplog.text
        assert solver.prepared == 1  # it does not try again for the same alignment

    def test_a_helper_without_a_solver_prepares_nothing(self, build: Build) -> None:
        helper = build()
        helper.housekeeping()  # no solver, and nothing to call

    def test_the_stop_closes_the_solver(self, build: Build) -> None:
        solver = Lifecycle()
        helper = build(solver=solver)
        helper.start()
        helper.stop()
        assert solver.closed == 1

    def test_a_solver_without_the_two_methods_is_fine(self, build: Build) -> None:
        active = Active(True)
        helper = build(is_active=active, solver=StubSolver())
        helper.process_frame(sky_frame(1))
        active.value = False
        assert helper.state().active is False
        helper.stop()

    def test_a_solver_that_fails_to_release_does_not_stop_the_helper(
        self, build: Build, caplog: pytest.LogCaptureFixture
    ) -> None:
        active = Active(True)
        helper = build(is_active=active, solver=Lifecycle(release_error=True))
        helper.process_frame(sky_frame(1))
        active.value = False
        with caplog.at_level(logging.ERROR):
            assert helper.state().active is False
        assert "failed to release" in caplog.text


class TestTheLog:
    def test_the_log_says_when_the_solve_starts_to_fail_and_when_it_works_again(
        self, build: Build, caplog: pytest.LogCaptureFixture
    ) -> None:
        solver = StubSolver(solution(solved=False, x_px=None, y_px=None, note="too few stars"))
        helper = build(solver=solver)
        with caplog.at_level(logging.INFO, logger="seeingmon.services.core.alignment.helper"):
            for seq in (1, 2, 3):
                helper.solve_frame(sky_frame(seq))
            solver.result = solution(seq=4)
            helper.solve_frame(sky_frame(4))
            helper.solve_frame(sky_frame(5))
        messages = [record.getMessage() for record in caplog.records]
        assert len(messages) == 2  # one line for each change, not one for each frame
        assert "fails" in messages[0]
        assert "too few stars" in messages[0]
        assert "works" in messages[1]
