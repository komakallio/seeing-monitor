"""Rapid focus: the alignment session streams a ROI of the fast readout mode around Polaris.

`StartRapidFocus` switches the alignment from its normal view (bin2, the whole frame, 0.5 s) to the
fast readout mode (bin1, a ROI of 128 x 128 pixels, 2 ms, about 88 frames a second in the tests).
The scheduler hands those frames to the focus consumer and never to the alignment consumer or the
fast analyzer. The mode ends on `StopRapidFocus`, after its own idle time, when the star stays out
of the window, and with the alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, VirtualClock, iso_to_utc_ns
from seeingmon.drivers.base import CameraTimeoutError
from seeingmon.frames import FrameData, PixelFormat, Roi, StreamConfig, StreamKind
from seeingmon.profile.models import ModeSelection
from seeingmon.scheduler import (
    Command,
    CommandResult,
    Pause,
    RejectReason,
    Scheduler,
    SchedulerConfig,
    StartAlignment,
    StartRapidFocus,
    StopAlignment,
    StopRapidFocus,
)
from seeingmon.scheduler import activity as words
from seeingmon.scheduler.config import AlignConfig, FastConfig, LoopConfig
from seeingmon.testing import (
    FakeCameraDriver,
    FakeFastAnalyzer,
    FakeFocusSink,
    FakePointingProvider,
    FakeSurveyAnalyzer,
    ListRecordWriter,
)
from tests.scheduler.scenario import PROFILE, World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")
ALIGN_AT = 5.0

# The fast stream of the tests takes the real exposure, and the ROI of 4.1 arcminutes (128 pixels).
CONFIG = SchedulerConfig(
    fast=FastConfig(exposure_us=2000, roi_arcmin=4.1, missing_star_frames=100),
    loop=LoopConfig(max_sleep_s=5.0),
)


def submit_at(world: World, seconds: float, command: Command) -> list[CommandResult]:
    """Submit a command at a time, and return a list that will hold the result."""
    holder: list[CommandResult] = []
    world.at(seconds, lambda w: holder.append(w.scheduler.submit(command)))
    return holder


def rapid_at(world: World, seconds: float, **fields: int) -> StartRapidFocus:
    """The command that centers rapid focus on the star as it truly is at `seconds`."""
    x, y = world.star_position(world.t(seconds))
    return StartRapidFocus(x, y, **fields)


def with_align(**fields: float) -> SchedulerConfig:
    """The configuration of the tests with other settings of `[scheduler.align]`."""
    return CONFIG.model_copy(update={"align": AlignConfig(**fields)})


def rapid_world(
    config: SchedulerConfig = CONFIG, *, rapid_at_s: float | None = 20.0
) -> tuple[World, list[CommandResult]]:
    """A night with the alignment from 5 s on, and rapid focus from `rapid_at_s` on."""
    world = World(start_utc_ns=NIGHT, config=config)
    submit_at(world, ALIGN_AT, StartAlignment())
    started: list[CommandResult] = []
    if rapid_at_s is not None:
        started = submit_at(world, rapid_at_s, rapid_at(world, rapid_at_s))
    return world, started


def rapid_configures(world: World) -> list[StreamConfig]:
    """The configurations of the rapid stream: the fast readout mode, after the alignment began."""
    return [
        call.config
        for call in world.configures(mode="bin1", video=True)
        if world.seconds(call.t_utc_ns) > ALIGN_AT
    ]


def align_configures(world: World) -> list[tuple[float, StreamConfig]]:
    return [
        (world.seconds(call.t_utc_ns), call.config)
        for call in world.configures(mode="bin2", video=True)
    ]


def centered(roi: Roi, x: float, y: float, within_px: float = 1.0) -> bool:
    """Whether the ROI is centered on the point, to `within_px` pixels."""
    return (
        abs(roi.x + roi.width / 2 - x) <= within_px and abs(roi.y + roi.height / 2 - y) <= within_px
    )


def purpose(world: World) -> str:
    """What the camera stream serves now, as the status says."""
    stream = world.scheduler.status().stream
    assert stream is not None
    return stream.purpose


# --- Starting and stopping the mode -----------------------------------------------------------


@dataclass(frozen=True)
class Run:
    world: World
    started: list[CommandResult]
    stopped: list[CommandResult]


@pytest.fixture(scope="module")
def run() -> Run:
    """The alignment starts at 5 s, rapid focus at 20 s, and its stop comes at 60 s."""
    world, started = rapid_world()
    stopped = submit_at(world, 60.0, StopRapidFocus())
    world.run_until(100)
    world.close()
    return Run(world, started, stopped)


class TestStartAndStop:
    def test_the_command_is_accepted_and_the_state_stays_align(self, run: Run) -> None:
        (result,) = run.started
        assert result.accepted
        assert (result.state, result.message) == ("align", "rapid focus started")
        assert run.world.states_visited() == ["safe", "auto", "align"]

    def test_the_camera_streams_the_fast_readout_mode_on_a_roi_around_the_center(
        self, run: Run
    ) -> None:
        (config,) = rapid_configures(run.world)
        x, y = run.world.star_position(run.world.t(20))
        assert (config.mode, config.kind, config.pixel_format) == (
            "bin1",
            StreamKind.VIDEO,
            PixelFormat.RAW16,
        )
        assert (config.exposure_us, config.gain) == (2000, 0)  # the fast stream's settings
        assert config.roi is not None
        assert (config.roi.width, config.roi.height) == (128, 128)
        assert centered(config.roi, x, y)

    def test_the_frames_go_to_the_focus_consumer_and_to_nobody_else(self, run: Run) -> None:
        world = run.world
        frames = world.focus.frames
        assert 3200 < len(frames) < 3800  # 40 s at about 88 frames a second
        assert len({f.stream_id for f in frames}) == 1
        assert {(f.mode, f.exposure_us, f.gain) for f in frames} == {("bin1", 2000, 0)}
        assert {(f.roi.width, f.roi.height) for f in frames} == {(128, 128)}
        first_s, last_s = (world.seconds(frames[i].t_utc_ns) for i in (0, -1))
        assert 20.0 <= first_s <= 21.0
        assert last_s <= 60.5
        # No alignment frame arrived while the mode ran, and no frame reached the fast analyzer.
        during = [f for f in world.align_frames if first_s < world.seconds(f.t_utc_ns) < last_s]
        assert during == []
        assert world.fast.frames_pushed == sum(w.n_frames for w in world.windows())
        assert all(f.mode != "bin1" for f in world.survey.submitted)

    def test_the_session_starts_once_and_ends_with_the_stop(self, run: Run) -> None:
        assert run.world.focus.begun == 1
        assert run.world.focus.ended == ["you stopped rapid focus"]
        (result,) = run.stopped
        assert (result.accepted, result.state) == (True, "align")
        assert "alignment view returns" in result.message

    def test_the_alignment_view_comes_back_after_the_stop(self, run: Run) -> None:
        world = run.world
        configs = align_configures(world)
        assert [(round(t), c.exposure_us, c.gain, c.roi) for t, c in configs] == [
            (5, 500_000, 120, None),
            (60, 500_000, 120, None),
        ]
        later = [f for f in world.align_frames if world.seconds(f.t_utc_ns) > 61]
        assert len(later) >= 70  # 40 s of frames of 0.5 s
        assert world.scheduler.state.value == "align"

    def test_every_command_and_the_start_and_the_end_wrote_an_event(self, run: Run) -> None:
        world = run.world
        commands = [(e.detail or {})["command"] for e in world.events("scheduler.command")]
        assert commands == ["StartAlignment", "StartRapidFocus", "StopRapidFocus"]
        started, ended = world.events("scheduler.rapid_focus")
        detail = started.detail or {}
        assert (detail["phase"], detail["mode"], detail["exposure_us"]) == ("started", "bin1", 2000)
        assert (detail["roi"]["width"], detail["roi"]["height"]) == (128, 128)
        assert (ended.detail or {}) == {"phase": "ended", "reason": "you stopped rapid focus"}
        assert ended.message == "Rapid focus ended: you stopped rapid focus."

    def test_the_stream_has_its_own_purpose_in_the_status(self) -> None:
        world, _ = rapid_world()
        world.run_until(30)
        status = world.scheduler.status()
        assert status.stream is not None
        assert (status.stream.purpose, status.stream.mode) == ("rapid_focus", "bin1")
        assert status.stream.roi is not None
        assert status.state == "align"
        world.close()

    def test_the_read_timeout_follows_the_fast_stream_and_not_the_alignment(self) -> None:
        world, _ = rapid_world()
        world.run_until(40)
        calls = world.camera.calls
        start = max(  # the last configuration of the fast readout mode is the one of rapid focus
            i
            for i, (name, value) in enumerate(calls)
            if name == "configure" and getattr(value, "mode", "") == "bin1"
        )
        timeouts = [float(value) for name, value in calls[start:] if name == "read_frame"]
        assert len(timeouts) > 1000
        # The frame period is 11.3 ms: two periods and the margin of 0.5 s. The alignment waits
        # 2.5 s for its frames of 0.5 s.
        assert max(timeouts) == min(timeouts) == pytest.approx(0.5226, abs=0.001)
        world.close()

    def test_a_second_session_starts_after_the_first_ended(self) -> None:
        world, _ = rapid_world()
        submit_at(world, 40.0, StopRapidFocus())
        again = submit_at(world, 50.0, rapid_at(world, 50.0))
        world.run_until(70)
        assert again[0].accepted
        assert world.focus.begun == 2
        assert len(rapid_configures(world)) == 2
        assert world.focus.ended == ["you stopped rapid focus"]
        world.close()


# --- What the scheduler refuses ---------------------------------------------------------------


class TestRejections:
    @pytest.mark.parametrize("setup", ["auto", "paused"])
    def test_the_mode_belongs_to_the_alignment(self, setup: str) -> None:
        world = World(start_utc_ns=NIGHT, config=CONFIG)
        if setup == "paused":
            submit_at(world, 10, Pause())
        refused = submit_at(world, 20, rapid_at(world, 20))
        stopped = submit_at(world, 21, StopRapidFocus())
        world.run_until(30)
        assert [r.reason for r in (*refused, *stopped)] == [RejectReason.NOT_ALIGNING] * 2
        assert world.scheduler.state.value == setup
        assert world.focus.begun == 0
        world.close()

    def test_the_mode_is_refused_in_safe_too(self) -> None:
        world = World(config=CONFIG)  # daylight: the scheduler stays in safe
        refused = submit_at(world, 20, StartRapidFocus(4144.0, 2822.0))
        world.run_until(30)
        assert world.scheduler.state.value == "safe"
        assert refused[0].reason is RejectReason.NOT_ALIGNING
        world.close()

    def test_a_stop_while_the_alignment_runs_without_the_mode_does_nothing(self) -> None:
        world, _ = rapid_world(rapid_at_s=None)
        stopped = submit_at(world, 20, StopRapidFocus())
        world.run_until(40)
        (result,) = stopped
        assert result.accepted
        assert "nothing to stop" in result.message
        assert len(align_configures(world)) == 1  # the alignment stream never restarted
        assert world.focus.ended == []
        world.close()

    @pytest.mark.parametrize(
        "fields",
        [
            {"exposure_us": 1},  # shorter than the camera's shortest exposure
            {"exposure_us": 3_000_000_000},
            {"gain": 600},
            {"gain": -1},
        ],
    )
    def test_settings_outside_the_profile_are_refused(self, fields: dict[str, int]) -> None:
        world, _ = rapid_world(rapid_at_s=None)
        refused = submit_at(world, 20, rapid_at(world, 20, **fields))
        world.run_until(30)
        assert refused[0].reason is RejectReason.INVALID
        assert world.focus.begun == 0
        assert rapid_configures(world) == []
        world.close()

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (float("nan"), 2822.0),
            (4144.0, float("inf")),
            (-1.0, 2822.0),
            (4144.0, -0.5),
            (8288.0, 2822.0),  # the frame is 8288 pixels wide, so the last pixel is 8287
            (4144.0, 5644.0),
        ],
    )
    def test_a_center_outside_the_frame_is_refused(self, x: float, y: float) -> None:
        world, _ = rapid_world(rapid_at_s=None)
        refused = submit_at(world, 20, StartRapidFocus(x, y))
        world.run_until(30)
        (result,) = refused
        assert result.reason is RejectReason.INVALID
        assert "8288 x 5644" in result.message
        assert world.focus.begun == 0
        world.close()

    def test_without_a_consumer_of_the_frames_the_mode_is_refused(self) -> None:
        world, _ = rapid_world(rapid_at_s=None)
        world.scheduler._focus_sink = None  # a scheduler that was built without a focus sink
        refused = submit_at(world, 20, rapid_at(world, 20))
        world.run_until(30)
        assert refused[0].reason is RejectReason.NO_HANDLER
        world.close()

    def test_a_camera_fault_that_is_being_recovered_refuses_the_mode(self) -> None:
        world, _ = rapid_world(rapid_at_s=None)
        world.camera_fault(30, 50)
        probes: list[tuple[str, CommandResult]] = []

        def probe(w: World) -> None:
            activity = w.scheduler.status().activity
            assert activity is not None
            probes.append((activity.phase, w.scheduler.submit(rapid_at(w, 38))))

        world.at(38, probe)
        world.run_until(60)
        ((phase, result),) = probes
        assert phase == "camera_fault"  # the scheduler waits for its next recovery step
        assert result.reason is RejectReason.CAMERA_FAULT
        assert "recovers" in result.message
        assert world.focus.begun == 0
        world.close()

    def test_a_refusal_writes_a_warning_event(self) -> None:
        world = World(start_utc_ns=NIGHT, config=CONFIG)
        submit_at(world, 20, rapid_at(world, 20))
        world.run_until(30)
        event = world.events("scheduler.command")[-1]
        assert (event.level, (event.detail or {})["reason"]) == ("warning", "not_aligning")
        world.close()


# --- A start while the mode runs --------------------------------------------------------------


class TestAlreadyRunning:
    def test_a_start_with_the_same_settings_restarts_the_timer_and_leaves_the_stream_alone(
        self,
    ) -> None:
        world, _ = rapid_world()
        again = submit_at(world, 100.0, rapid_at(world, 100.0))
        world.run_until(130)
        (result,) = again
        assert result.accepted
        assert "already runs" in result.message
        assert len(rapid_configures(world)) == 1
        assert world.focus.begun == 1
        idle_s = world.scheduler.status().alignment_idle_s
        assert idle_s is not None
        assert idle_s < 40
        world.close()

    def test_a_start_with_another_exposure_restarts_the_stream_in_the_same_session(self) -> None:
        world, _ = rapid_world()
        submit_at(world, 40.0, rapid_at(world, 40.0, exposure_us=1000, gain=30))
        world.run_until(60)
        first, second = rapid_configures(world)
        assert (first.exposure_us, first.gain) == (2000, 0)
        assert (second.exposure_us, second.gain) == (1000, 30)
        assert world.focus.begun == 1  # one session
        assert world.focus.ended == []
        assert {f.exposure_us for f in world.focus.frames} == {1000, 2000}
        activity = world.scheduler.status().activity
        assert activity is not None
        assert activity.since_utc_ns == pytest.approx(world.t(20), abs=NS_PER_S)  # the first start
        world.close()

    def test_a_start_without_settings_keeps_the_settings_that_run(self) -> None:
        world, _ = rapid_world()
        submit_at(world, 40.0, rapid_at(world, 40.0, exposure_us=1000, gain=30))
        again = submit_at(world, 80.0, rapid_at(world, 80.0))
        world.run_until(100)
        (result,) = again
        assert result.accepted
        assert "already runs" in result.message
        configs = rapid_configures(world)
        assert len(configs) == 2  # the keep-alive did not restart the stream
        assert (configs[-1].exposure_us, configs[-1].gain) == (1000, 30)
        assert {f.exposure_us for f in world.focus.frames if world.seconds(f.t_utc_ns) > 82} == {
            1000
        }
        world.close()

    def test_a_start_with_one_setting_changes_only_that_one(self) -> None:
        world, _ = rapid_world()
        submit_at(world, 40.0, rapid_at(world, 40.0, exposure_us=1000, gain=30))
        x, y = world.star_position(world.t(80.0))
        submit_at(world, 80.0, StartRapidFocus(x, y, gain=60))
        world.run_until(100)
        configs = rapid_configures(world)
        assert [(c.exposure_us, c.gain) for c in configs] == [(2000, 0), (1000, 30), (1000, 60)]
        world.close()

    def test_a_start_of_the_alignment_does_not_end_the_mode(self) -> None:
        world, _ = rapid_world()
        submit_at(world, 40.0, StartAlignment())
        world.run_until(60)
        assert purpose(world) == "rapid_focus"
        assert world.focus.ended == []
        world.close()


# --- The timers -------------------------------------------------------------------------------


class TestTheIdleTimers:
    def test_the_mode_returns_to_the_alignment_view_after_its_own_idle_time(self) -> None:
        world, _ = rapid_world(with_align(rapid_focus_idle_timeout_s=30.0))
        world.run_until(20 + 30 - 5)
        assert purpose(world) == "rapid_focus"
        world.run_until(20 + 30 + 5)
        assert (world.scheduler.state.value, purpose(world)) == ("align", "align")
        ended = world.events("scheduler.rapid_focus")[-1]
        assert world.seconds(ended.t_utc_ns) == pytest.approx(20 + 30, abs=1.0)
        assert world.focus.ended == ["nobody used it for 30 s"]
        assert len(rapid_configures(world)) == 1
        assert [f for f in world.align_frames if world.seconds(f.t_utc_ns) > 51]  # frames are back
        world.close()

    def test_touching_the_alignment_keeps_the_mode_alive(self) -> None:
        world, _ = rapid_world(with_align(rapid_focus_idle_timeout_s=30.0))
        for seconds in (40, 60, 80, 100):  # each touch comes within 30 s of the one before
            world.at(seconds, lambda w: w.scheduler.touch_alignment())
        world.run_until(125)
        assert purpose(world) == "rapid_focus"
        world.run_until(100 + 30 + 5)  # the last touch at 100 s, and then 30 s without use
        assert purpose(world) == "align"
        assert world.focus.ended == ["nobody used it for 30 s"]
        world.close()

    def test_the_alignment_still_ends_after_its_own_idle_time(self) -> None:
        world, _ = rapid_world(with_align(idle_timeout_s=60.0, rapid_focus_idle_timeout_s=1000.0))
        world.at(40, lambda w: w.scheduler.touch_alignment())  # the alignment ends at 100 s
        world.run_until(80)
        assert world.scheduler.state.value == "align"
        world.run_until(110)
        # The alignment ends in `safe`, and the check of the sky sends the scheduler on to `auto`.
        assert world.states_visited() == ["safe", "auto", "align", "safe", "auto"]
        assert world.focus.ended == ["the alignment ended"]
        assert world.scheduler.status().alignment_idle_s is None
        change = world.events("scheduler.state_change")[2]
        assert (change.detail or {})["reason"] == "alignment idle timeout"
        world.close()

    def test_a_stop_of_the_alignment_ends_the_mode_with_it(self) -> None:
        world, _ = rapid_world()
        submit_at(world, 50.0, StopAlignment())
        world.run_until(80)
        assert world.focus.ended == ["the alignment ended"]
        assert world.scheduler.state.value in ("safe", "auto")
        assert purpose(world) != "rapid_focus"
        frames_at_stop = len(world.focus.frames)
        world.run_until(120)
        assert len(world.focus.frames) == frames_at_stop
        world.close()

    def test_a_pause_ends_the_mode_with_the_alignment(self) -> None:
        world, _ = rapid_world()
        submit_at(world, 50.0, Pause())
        world.run_until(80)
        assert world.focus.ended == ["you paused the scheduler"]
        assert world.scheduler.state.value == "paused"
        frames_at_pause = len(world.focus.frames)
        world.run_until(120)
        assert len(world.focus.frames) == frames_at_pause
        world.close()


# --- The ROI and the star ---------------------------------------------------------------------


class TestFollowingTheStar:
    def test_a_star_near_the_edge_moves_the_roi_onto_it(self) -> None:
        world, _ = rapid_world()
        world.jolt(40, 50.0)  # the mount is bumped: the star lands 14 pixels from the ROI edge
        world.run_until(60)
        ((_, arguments),) = world.camera.calls_named("move_roi")
        assert isinstance(arguments, tuple)
        x, y = arguments
        star_x, star_y = world.star_position(world.t(45))
        assert (x + 64, y + 64) == pytest.approx((star_x, star_y), abs=2.0)  # centered on it
        (event,) = world.events("scheduler.roi_recentered")
        detail = event.detail or {}
        assert detail["edge_distance_px"] < detail["margin_px"] == 16.0
        assert detail["to"]["x"] - detail["from"]["x"] == pytest.approx(50.0, abs=3.0)
        status = world.scheduler.status()
        assert status.counters.roi_recenters == 1
        assert len(rapid_configures(world)) == 1  # the ROI moved, and the stream stayed
        assert len({f.stream_id for f in world.focus.frames}) == 1
        assert status.stream is not None
        assert status.stream.roi is not None
        assert status.stream.roi.x == x
        assert world.focus.ended == []
        world.close()

    def test_a_second_bump_inside_the_cooldown_waits_for_it(self) -> None:
        world, _ = rapid_world()
        world.jolt(40, 50.0)
        world.jolt(41, 50.0)  # one second later, inside the 5 s cooldown
        world.run_until(60)
        times = [world.seconds(e.t_utc_ns) for e in world.events("scheduler.roi_recentered")]
        assert len(times) == 2
        assert times[1] - times[0] >= 5.0
        world.close()

    def test_a_restart_of_the_stream_after_a_fault_keeps_the_star_in_view(self) -> None:
        world, _ = rapid_world()
        world.jolt(40, 50.0)
        world.camera_fault(50, 52)
        world.run_until(80)
        configs = rapid_configures(world)
        assert world.events("scheduler.fault")
        assert len(configs) == 2
        star_x, star_y = world.star_position(world.t(70))
        first, second = configs
        assert first.roi is not None
        assert not centered(first.roi, star_x, star_y, 10.0)
        # The moved ROI. The star drifted 2 pixels since it moved the ROI.
        assert second.roi is not None
        assert centered(second.roi, star_x, star_y, 4.0)
        assert (world.focus.begun, world.focus.ended) == (1, [])  # one session throughout
        assert world.seconds(world.focus.frames[-1].t_utc_ns) > 75  # the mode went on
        world.close()

    def test_a_star_that_the_roi_cannot_center_is_reported_once(self) -> None:
        world = World(start_utc_ns=NIGHT, config=CONFIG)
        # The mount is bumped while the alignment runs: the star sits 2 pixels from the left edge
        # of the sensor, and the scheduler never sees it in the fast stream.
        world.jolt(10, 2.0 - 4144.0 - 0.087 * 10)
        submit_at(world, ALIGN_AT, StartAlignment())
        submit_at(world, 20, rapid_at(world, 20))
        world.run_until(30)
        (event,) = world.events("scheduler.roi_at_limit")
        assert (event.detail or {})["roi"]["x"] == 0  # the ROI sits against the sensor edge
        world.run_until(80)
        assert len(world.events("scheduler.roi_at_limit")) == 1  # said once
        assert world.scheduler.status().counters.roi_recenters == 0
        assert world.focus.ended == []  # the mode goes on with the star near the edge
        world.close()

    def test_a_star_that_stays_out_of_the_window_ends_the_mode(self) -> None:
        world, _ = rapid_world()
        world.hide_star(40, 300)
        world.run_until(60)
        assert world.focus.ended == ["the star was not in the window for 100 frames"]
        ended = [
            e for e in world.events("scheduler.rapid_focus") if (e.detail or {})["phase"] == "ended"
        ]
        assert world.seconds(ended[0].t_utc_ns) == pytest.approx(40 + 100 * 0.0113, abs=0.5)
        assert (world.scheduler.state.value, purpose(world)) == ("align", "align")
        assert [c.exposure_us for _, c in align_configures(world)] == [500_000, 500_000]
        world.close()

    def test_a_short_loss_of_the_star_does_nothing(self) -> None:
        world, _ = rapid_world()
        world.hide_star(40, 40.8)  # about 70 frames, and the limit is 100
        world.run_until(60)
        assert world.focus.ended == []
        assert purpose(world) == "rapid_focus"
        world.close()


class TestAFaultAndAFailingConsumer:
    def test_the_mode_resumes_after_a_recovered_fault(self) -> None:
        world, _ = rapid_world()
        world.camera_fault(40, 43)
        world.run_until(80)
        assert world.events("scheduler.fault")
        assert world.scheduler.state.value == "align"
        assert purpose(world) == "rapid_focus"
        assert len(rapid_configures(world)) >= 2  # the stream restarted after the recovery step
        assert (world.focus.begun, world.focus.ended) == (1, [])
        world.close()

    def test_a_read_that_times_out_in_the_mode_counts_as_a_camera_fault(self) -> None:
        world, _ = rapid_world()
        world.run_until(30)
        world.camera.fail_reads(CameraTimeoutError("no frame"))
        world.run_until(40)
        assert world.scheduler.status().counters.faults == 1
        assert world.focus.begun == 1
        world.close()

    def test_a_consumer_that_raises_is_reported_once_and_never_stops_the_camera(self) -> None:
        world, _ = rapid_world()
        world.at(30, lambda w: setattr(w.focus, "failing", True))
        world.run_until(30.8)  # about 70 frames: fewer than the 100 that make a star lost
        assert purpose(world) == "rapid_focus"
        (event,) = world.events("scheduler.focus_sink_failed")
        assert event.level == "warning"
        assert "RuntimeError" in event.message
        assert world.scheduler.status().counters.faults == 0
        # A consumer that gives no star is a missing star, and the mode ends as it does then.
        world.run_until(34)
        assert len(world.events("scheduler.focus_sink_failed")) == 1  # said once
        assert world.focus.ended == ["the star was not in the window for 100 frames"]
        assert world.scheduler.status().counters.faults == 0
        world.close()


# --- The activity -----------------------------------------------------------------------------


class TestTheActivity:
    def test_the_status_says_that_rapid_focus_runs_and_what_comes_next(self) -> None:
        world, _ = rapid_world()
        world.run_until(50)
        activity = world.scheduler.status().activity
        assert activity is not None
        assert (activity.state, activity.phase) == ("align", "rapid_focus")
        assert activity.label == "Rapid focus on Polaris"
        assert activity.reason == "you started rapid focus"
        assert activity.next_label == words.RAPID_FOCUS_NEXT_LABEL
        assert activity.cadence_s is None
        # The time is the time since the start, and the end is the idle time of the mode.
        assert activity.since_utc_ns == pytest.approx(world.t(20), abs=NS_PER_S)
        assert activity.ends_utc_ns is not None
        assert activity.next_utc_ns == activity.ends_utc_ns
        assert activity.ends_utc_ns == pytest.approx(world.t(20 + 120), abs=NS_PER_S)
        assert activity.detail is not None
        assert activity.detail.startswith("Rapid focus has run for 30 s;")
        assert "returns after 2 min without use" in activity.detail
        world.close()

    def test_touching_moves_the_end_and_the_idle_time(self) -> None:
        world, _ = rapid_world()
        world.run_until(100)
        before = world.scheduler.status().activity
        world.scheduler.touch_alignment()
        after = world.scheduler.status().activity
        assert before is not None
        assert after is not None
        assert before.ends_utc_ns is not None
        assert after.ends_utc_ns is not None
        assert after.ends_utc_ns - before.ends_utc_ns == pytest.approx(80 * NS_PER_S, abs=NS_PER_S)
        assert after.detail is not None
        assert after.detail.endswith("the idle time is 0 s")
        world.close()

    def test_after_the_stop_the_alignment_activity_returns(self) -> None:
        world, _ = rapid_world()
        submit_at(world, 40.0, StopRapidFocus())
        world.run_until(50)
        activity = world.scheduler.status().activity
        assert activity is not None
        assert (activity.phase, activity.label) == ("align", words.ALIGN_LABEL)
        assert activity.reason == "you started the alignment"
        world.close()

    def test_the_words(self) -> None:
        assert words.rapid_focus_detail(80.0, 3.0, 120.0) == (
            "Rapid focus has run for 1 min 20 s; the alignment view returns after 2 min without "
            "use, and the idle time is 3 s"
        )


# --- The geometry of the ROI ------------------------------------------------------------------


def lone_star(config: StreamConfig, roi: Roi, seq: int) -> FrameData:
    """A frame with one bright pixel at the middle of the ROI."""
    data = np.full((roi.height, roi.width), 200, dtype=np.uint16)
    data[roi.height // 2, roi.width // 2] = 6000
    return data


def scheduler_for(fast_mode: str) -> tuple[Scheduler, FakeCameraDriver]:
    """A scheduler on a profile whose fast mode is `fast_mode`, with a camera that shows a star."""
    profile = PROFILE.model_copy(
        update={"fast_mode": ModeSelection(mode=fast_mode, pixel_format=PixelFormat.RAW16)}
    )
    clock = VirtualClock(NIGHT)
    camera = FakeCameraDriver(clock, frame_factory=lone_star)
    writer = ListRecordWriter()
    scheduler = Scheduler(
        driver=camera,
        fast=FakeFastAnalyzer(window_s=CONFIG.fast.analysis_window_s),
        survey=FakeSurveyAnalyzer(),
        pointing=FakePointingProvider(),
        records=writer,
        metrics=writer,
        clock=clock,
        profile=profile,
        station_id="test",
        config=CONFIG,
        focus_sink=FakeFocusSink(),
    )
    return scheduler, camera


def last_configure(camera: FakeCameraDriver) -> StreamConfig:
    config = [value for name, value in camera.calls if name == "configure"][-1]
    assert isinstance(config, StreamConfig)
    return config


class TestTheRoiInBothReadoutModes:
    @pytest.mark.parametrize(
        ("mode", "size", "center"),
        [
            ("bin1", (128, 128), (3000.0, 2000.0)),
            ("bin2", (64, 64), (1500.0, 1000.0)),
        ],
    )
    def test_the_roi_covers_four_arcminutes_around_the_center_in_pixels_of_the_fast_mode(
        self, mode: str, size: tuple[int, int], center: tuple[float, float]
    ) -> None:
        scheduler, camera = scheduler_for(mode)
        scheduler.submit(StartAlignment())
        for _ in range(3):
            scheduler.step()
        assert scheduler.submit(StartRapidFocus(*center)).accepted
        for _ in range(5):
            scheduler.step()
        config = last_configure(camera)
        assert config.mode == mode
        roi = config.roi
        assert roi is not None
        assert (roi.width, roi.height) == size
        assert centered(roi, *center)
        assert (roi.width % 8, roi.height % 2) == (0, 0)  # the rules of the camera
        status = scheduler.status()
        assert status.stream is not None
        assert (status.stream.mode, status.stream.roi) == (mode, roi)
        scheduler.close()

    def test_a_center_at_the_corner_gives_a_roi_inside_the_frame(self) -> None:
        scheduler, camera = scheduler_for("bin1")
        scheduler.submit(StartAlignment())
        scheduler.step()
        assert scheduler.submit(StartRapidFocus(0.0, 0.0)).accepted
        scheduler.step()
        scheduler.step()
        assert last_configure(camera).roi == Roi(0, 0, 128, 128)
        scheduler.close()

    def test_the_frames_reach_the_focus_sink_with_the_roi_of_the_stream(self) -> None:
        scheduler, _ = scheduler_for("bin2")
        scheduler.submit(StartAlignment())
        scheduler.step()
        scheduler.submit(StartRapidFocus(1500.0, 1000.0))
        for _ in range(8):
            scheduler.step()
        sink = scheduler._focus_sink
        assert isinstance(sink, FakeFocusSink)
        assert sink.frames
        first = sink.frames[0]
        assert first.mode == "bin2"
        assert (first.roi.width, first.roi.height) == (64, 64)
        scheduler.close()
