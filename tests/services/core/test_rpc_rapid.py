"""The rapid focus mode through `core`: the RPC method, the scheduler, and the video.

The rig is a whole `CoreApp` on a virtual clock with a camera that shows a star in the ROI of the
fast readout mode. The alignment helper gets the quick solves that a solver of the test makes, so
the offer of the mode follows the numbers that the test sets. These tests prove the wiring of the
parts that the other tests prove one by one: the helper, the scheduler, the measurement, the
state of the alignment, the video of Polaris, and the RPC.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("PIL.Image", reason="the preview needs Pillow")

from seeingmon.clock import Clock
from seeingmon.drivers.base import CameraDriver
from seeingmon.fastpath import FastPathConfig, create_fast_analyzer
from seeingmon.frames import FrameData, Roi, StreamConfig
from seeingmon.profile import Profile, load_profile
from seeingmon.scheduler.commands import (
    Pause,
    RejectReason,
    StartAlignment,
    StartRapidFocus,
    StopAlignment,
    StopRapidFocus,
)
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.stream import StreamKind, connect_stream
from seeingmon.services.web.contract import (
    PNG_MAGIC,
    POLARIS_CHANNEL,
    unpack_polaris_frame,
)
from seeingmon.services.web.core_client import RpcCoreClient
from seeingmon.testing import FakeCameraDriver
from tests.services.core.test_alignment_rapid import StubSolver, solution
from tests.services.core.test_rapid_focus import FLUX_E, POOL, STAR_X, star_pool

from ..conftest import native_endpoint, wait_until
from .rig import NIGHT, SMALL_BIN2, CoreRig, build_rig, sky_frame

KEY = ConnectionKey.from_text("a-test-key-of-more-than-32-characters")
PROFILE = load_profile("asi294mm-gs250")
SIGMA_PX = 1.0  # a star of 4.7 arcseconds in the fast readout mode


def star_frames(config: StreamConfig, roi: Roi, seq: int) -> FrameData:
    """The pixels of the camera: Polaris in a ROI of 128 pixels, and flat sky everywhere else."""
    wide = np.dtype(config.pixel_format.dtype).itemsize == 2  # RAW16, else RAW8
    if (roi.width, roi.height) == (128, 128):
        frame = star_pool(SIGMA_PX, FLUX_E, STAR_X)[seq % POOL]
        return frame if wide else np.asarray(frame >> 8, dtype=np.uint8)
    if wide:
        return np.full((roi.height, roi.width), 100 << (16 - 12), dtype=np.uint16)
    return np.full((roi.height, roi.width), 100, dtype=np.uint8)


CAMERAS: list[FakeCameraDriver] = []  # the camera that the last rig built


def star_camera(clock: Clock, profile: Profile) -> CameraDriver:
    camera = FakeCameraDriver(
        clock,
        full_frames={"bin1": (8288, 5644), "bin2": SMALL_BIN2},
        adc_bits=12,
        frame_factory=star_frames,
    )
    CAMERAS[:] = [camera]
    return camera


@pytest.fixture
def served(short_dir: Path, tmp_path: Path) -> Iterator[tuple[CoreRig, RpcCoreClient]]:
    solver = StubSolver()
    rig = build_rig(
        tmp_path,
        key=KEY,
        endpoint=native_endpoint(short_dir, "core"),
        parts={
            "quick_solver": solver,
            "fast": create_fast_analyzer(PROFILE, FastPathConfig(), "test"),
        },
        driver_factory=star_camera,
    )
    rig.extra["solver"] = solver
    rig.extra["camera"] = CAMERAS[0]
    assert rig.app.profile.id == PROFILE.id
    rig.app.start()
    client = RpcCoreClient(rig.app.bound_endpoint, KEY, retry_interval_s=0.01)  # type: ignore[arg-type]
    yield rig, client
    client.close()
    rig.app.stop()


def configures(rig: CoreRig) -> list[StreamConfig]:
    """The stream configurations that the camera received, in order."""
    return [c for name, c in rig.extra["camera"].calls if name == "configure"]


def align(rig: CoreRig, client: RpcCoreClient) -> None:
    """Start the alignment, and let the scheduler enter it."""
    assert client.submit(StartAlignment()).accepted
    rig.run_for(1.0)
    assert rig.app.scheduler.state.value == "align"


def solve(rig: CoreRig, count: int = 6, **changes: Any) -> None:
    """Give the helper `count` quick solves with these fields, and let it show the last frame."""
    solver = rig.extra["solver"]
    for seq in range(1, count + 1):
        solver.result = solution(seq, t_utc_ns=NIGHT, **changes)
        rig.app.alignment.solve_frame(sky_frame(seq))
    rig.app.alignment.process_frame(sky_frame(count))


def run_mode(rig: CoreRig, client: RpcCoreClient, seconds: float = 3.0) -> None:
    """Align, give the solves that offer the mode, start it, and run it for `seconds`."""
    align(rig, client)
    solve(rig)
    result = client.rapid_focus_start()
    assert result.accepted, result.message
    rig.run_for(seconds)


# --- The refusals of core --------------------------------------------------------------------


class TestTheOfferThroughTheMethod:
    def test_outside_the_alignment_the_start_is_refused_as_not_aligning(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        result = client.rapid_focus_start()
        assert (result.accepted, result.reason) == (False, RejectReason.NOT_ALIGNING)
        assert result.message == "the alignment does not run, and rapid focus belongs to it"
        assert rig.app.scheduler.state.value != "align"

    def test_before_the_first_frame_the_start_is_not_available(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        align(rig, client)
        result = client.rapid_focus_start()
        assert (result.accepted, result.reason) == (False, RejectReason.NOT_AVAILABLE)
        assert result.message == "no frame has arrived yet"

    def test_wide_stars_refuse_the_start_with_the_numbers(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        align(rig, client)
        solve(rig, focus_fwhm_px=5.5)
        result = client.rapid_focus_start()
        assert (result.accepted, result.reason) == (False, RejectReason.NOT_AVAILABLE)
        assert (
            result.message
            == "the stars are too wide for the rapid mode: 21 arcsec, the limit is 12"
        )
        # The state says the same, so a page can show the reason before the person asks.
        view = client.alignment_state().rapid_focus
        assert view is not None
        assert (view.available, view.reason) == (False, result.message)

    def test_a_refusal_is_a_normal_answer_and_an_event(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        align(rig, client)
        assert not client.rapid_focus_start().accepted
        rig.app.scheduler.step()
        rig.app.tick()
        refused = rig.events("core.command_refused")
        assert [e.message for e in refused] == ["Refused StartRapidFocus: no frame has arrived yet"]
        assert rig.app.rpc.refused == 1

    def test_the_offer_is_in_the_state_before_the_start(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        align(rig, client)
        solve(rig)
        view = client.alignment_state().rapid_focus
        assert view is not None
        assert (view.available, view.located_by, view.active) == (True, "current solution", False)


# --- The mode ------------------------------------------------------------------------------


class TestTheModeThroughTheScheduler:
    def test_the_start_switches_the_camera_to_a_small_roi_of_the_fast_mode(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        run_mode(rig, client, seconds=1.0)
        configs = configures(rig)
        fast_mode = rig.app.profile.fast_mode.mode
        assert configs[-1].mode == fast_mode
        roi = configs[-1].roi
        assert roi is not None
        assert (roi.width, roi.height) == (128, 128)
        # The solved pixel of Polaris, (303, 196) in bin2, is (606.5, 392.5) in the fast mode.
        assert roi.x <= 606.5 <= roi.x + roi.width
        assert roi.y <= 392.5 <= roi.y + roi.height
        stream = rig.app.scheduler.status().stream
        assert stream is not None
        assert stream.purpose == "rapid_focus"

    def test_the_status_says_what_the_scheduler_does(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        run_mode(rig, client, seconds=1.0)
        activity = client.status().scheduler.activity
        assert activity is not None
        assert (activity.state, activity.phase) == ("align", "rapid_focus")
        assert activity.label == "Rapid focus on Polaris"
        assert activity.next_label == "Back to the alignment view of the whole frame"

    def test_the_helper_measures_twenty_readings_a_second(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        run_mode(rig, client, seconds=3.0)
        snapshot = rig.app.rapid.snapshot()
        assert snapshot.active
        assert 50 <= len(snapshot.columns) <= 61  # 20 a second for 3 seconds
        assert rig.app.rapid.frames > 200
        widths = snapshot.columns.fwhm_arcsec
        expected = 2.3548 * np.sqrt(SIGMA_PX**2 + 1 / 12) * 1.91
        assert float(np.median(widths)) == pytest.approx(expected, rel=0.05)
        assert snapshot.star_found

    def test_the_state_of_the_alignment_carries_the_readings_while_the_mode_runs(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        run_mode(rig, client, seconds=2.0)
        view = client.alignment_state().rapid_focus
        assert view is not None
        assert (view.available, view.active, view.n_stars) == (True, True, 1)
        assert view.mode == rig.app.profile.fast_mode.mode
        assert view.scale_arcsec_px == pytest.approx(1.91, abs=0.01)
        assert view.readings is not None
        assert view.readings.reset is True
        assert len(view.readings.index) > 30
        assert view.fwhm_arcsec is not None
        assert 4.0 < view.fwhm_arcsec < 5.5

    def test_the_alignment_helper_gets_no_frame_while_the_mode_runs(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        align(rig, client)
        solve(rig)
        received = rig.app.alignment.frames_received
        assert client.rapid_focus_start().accepted
        rig.run_for(2.0)
        assert rig.app.alignment.frames_received == received

    def test_a_second_start_keeps_the_mode_alive_at_the_place_of_the_star(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        run_mode(rig, client, seconds=1.0)
        before = len(configures(rig))
        again = client.rapid_focus_start()
        assert again.accepted
        assert again.message == "rapid focus already runs, so the idle timer restarted"
        rig.run_for(0.5)
        assert len(configures(rig)) == before

    def test_a_new_exposure_reconfigures_the_stream(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        run_mode(rig, client, seconds=1.0)
        result = client.rapid_focus_start(exposure_us=1500, gain=10)
        assert result.accepted
        rig.run_for(1.0)
        configs = configures(rig)
        assert (configs[-1].exposure_us, configs[-1].gain) == (1500, 10)
        view = client.alignment_state().rapid_focus
        assert view is not None
        assert (view.exposure_us, view.gain) == (1500, 10)

    def test_a_stop_returns_to_the_alignment_view_and_says_so(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        run_mode(rig, client, seconds=1.0)
        assert client.submit(StopRapidFocus()).accepted
        rig.run_for(1.0)
        stream = rig.app.scheduler.status().stream
        assert stream is not None
        assert stream.purpose == "align"
        assert stream.mode == rig.app.profile.survey_mode.mode
        view = client.alignment_state().rapid_focus
        assert view is not None
        assert (view.active, view.readings) == (False, None)
        assert view.ended_reason
        assert not rig.app.rapid.active
        activity = client.status().scheduler.activity
        assert activity is not None
        assert (activity.state, activity.phase) == ("align", "align")

    @pytest.mark.parametrize("ending", [StopAlignment(), Pause()])
    def test_the_end_of_the_alignment_ends_the_mode(
        self, served: tuple[CoreRig, RpcCoreClient], ending: StopAlignment | Pause
    ) -> None:
        rig, client = served
        run_mode(rig, client, seconds=1.0)
        assert client.submit(ending).accepted
        rig.run_for(1.0)
        assert not rig.app.rapid.active
        assert client.alignment_state().active is False

    def test_the_command_with_a_center_goes_through_submit_too(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        align(rig, client)
        result = client.submit(StartRapidFocus(1200.0, 900.0))
        assert result.accepted
        rig.run_for(1.0)
        assert rig.app.rapid.active
        far = client.submit(StartRapidFocus(-5.0, 900.0))  # outside the fast frame
        assert (far.accepted, far.reason) == (False, RejectReason.INVALID)


# --- The video -----------------------------------------------------------------------------


class TestTheVideo:
    def test_the_video_carries_the_readings_with_the_frames_of_the_mode(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        run_mode(rig, client, seconds=0.5)
        receiver, _reply = connect_stream(
            rig.app.bound_endpoint,  # type: ignore[arg-type]
            KEY,
            {"role": "web"},
            channel=POLARIS_CHANNEL,
        )
        try:
            assert wait_until(lambda: rig.app.polaris.viewers == 1)
            rig.run_for(3.0)
            slot = rig.app.polaris._slot  # no thread runs in this rig: the test does its work
            assert slot is not None
            assert slot.rapid is True
            rig.app.polaris.process(slot)
            message = receiver.recv(10.0)
            assert message is not None
            assert message.kind is StreamKind.DATA
            frame = unpack_polaris_frame(message.payload)
        finally:
            receiver.close()
        assert frame.image.startswith(PNG_MAGIC)
        assert (frame.state.roi.width, frame.state.roi.height) == (128, 128)
        assert frame.state.mode == rig.app.profile.fast_mode.mode
        view = frame.state.rapid_focus
        assert view is not None
        assert (view.active, view.n_stars) == (True, 1)
        assert view.readings is not None
        assert len(view.readings.index) > 40
        assert frame.state.star.found
        assert frame.state.live_seeing is None  # the mode makes no rolling seeing value
        assert frame.state.quality["live_seeing"] == (
            "the rapid focus mode makes no rolling seeing value"
        )

    def test_a_frame_of_the_normal_fast_stream_has_no_rapid_view(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, _ = served
        assert rig.app.rapid_focus() is None  # no session has run
        view_before = rig.app.rapid.snapshot()
        assert not view_before.active
