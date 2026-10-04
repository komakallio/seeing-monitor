"""The flat handler: the session on the lent camera, its result, and the state that the RPC reads.

The handler runs against a stand-in for the scheduler's context that wraps a scripted panel camera
on a `VirtualClock` (`tests.survey.panelfx`), so a session takes no real time. The lens is the one
of the owner on a sensor of 256 x 176 pixels, and the dark library holds two sets of bin2 at
gain 120.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from seeingmon.clock import NS_PER_S, VirtualClock, iso_to_utc_ns, utc_ns_to_iso
from seeingmon.drivers.base import CameraConfigError, CameraTimeoutError
from seeingmon.frames import ActiveStream, Frame, StreamConfig
from seeingmon.profile import Profile
from seeingmon.scheduler.commands import QueueBurst, QueueFlat
from seeingmon.scheduler.commission import CommissionTask, FastWindowSample
from seeingmon.scheduler.config import SchedulerConfig
from seeingmon.services.core.commissioning.dark import ContextCamera
from seeingmon.services.core.commissioning.flat import (
    PHASE_EVENT,
    FlatHandler,
    FlatLibraryReader,
    FlatTaskState,
    describe_session,
    flat_view,
)
from seeingmon.services.web.contract import (
    MAX_FLAT_JPEG_BYTES,
    FlatTaskView,
    decode_flat_library,
)
from seeingmon.store.layout import DataLayout
from seeingmon.survey import flat_make as fm
from seeingmon.survey.config import DarkConfig, FlatConfig, SurveyConfig
from seeingmon.survey.dark import DarkLibrary
from seeingmon.survey.flat_library import APPROVED, PENDING, FlatEntry, FlatLibrary
from seeingmon.survey.flat_session import DARK_FIRST, NO_FIRST_SET
from tests.survey import flatfx as fx
from tests.survey.panelfx import PanelCamera

SHAPE = (176, 256)
PROFILE: Profile = fx.scaled_profile(256, 176)
START = iso_to_utc_ns("2026-10-04T20:00:00Z")
MAKE = fm.MakeOptions(bin_factor=2, high_pass_px=10.0, edge_margin_px=6.0)
FRAMES = 16


def truth() -> fx.FloatImage:
    return fx.lens_flat(
        fx.OWNER_LENS, SHAPE, scale_down=fx.REFERENCE_SHAPE[1] / SHAPE[1], edge_artifact_x=3.0
    )


class PanelContext:
    """The context of a handler on a scripted panel camera that the test controls."""

    def __init__(self, clock: VirtualClock, camera: PanelCamera, profile: Profile) -> None:
        self._clock = clock
        self._profile = profile
        self.camera = camera
        self.reads = 0
        self.stops_after: int | None = None
        self.on_read: Callable[[int], None] | None = None
        self.events: list[tuple[str, str, str, Mapping[str, Any] | None]] = []
        self.configure_error: Exception | None = None

    @property
    def clock(self) -> VirtualClock:
        return self._clock

    @property
    def profile(self) -> Profile:
        return self._profile

    @property
    def config(self) -> SchedulerConfig:
        return SchedulerConfig()

    def should_stop(self) -> bool:
        return self.stops_after is not None and self.reads >= self.stops_after

    def emit_event(
        self, level: str, kind: str, message: str, detail: Mapping[str, Any] | None = None
    ) -> None:
        self.events.append((level, kind, message, detail))

    def fast_stream_config(self, **options: Any) -> StreamConfig | None:
        return None

    def configure(self, config: StreamConfig) -> ActiveStream:
        if self.configure_error is not None:
            raise self.configure_error
        self.camera.configure(config)
        return self.camera.active_stream(config, SHAPE)

    def start(self) -> None:
        self.camera.start()

    def stop(self) -> None:
        self.camera.stop()

    def read_frame(self, timeout_s: float | None = None) -> Frame:
        if self.on_read is not None:
            self.on_read(self.reads)
        frame = self.camera.read_frame(timeout_s)
        self.reads += 1
        return frame

    def run_fast_window(self, config: StreamConfig, duration_s: float) -> FastWindowSample:
        raise NotImplementedError("a flat session runs no fast window")


class Rig:
    def __init__(self, tmp_path: Path, *, with_darks: bool = True, **camera: Any) -> None:
        self.clock = VirtualClock(START)
        self.layout = DataLayout(tmp_path / "data")
        self.layout.create()
        calibration = self.layout.root / "calibration"
        self.flats = FlatLibrary(calibration / "flats")
        self.darks: DarkLibrary = (
            fx.make_library(calibration) if with_darks else DarkLibrary(calibration / "darks")
        )
        self.survey = SurveyConfig(dark=DarkConfig(mode="bin2", gain=120), flat=FlatConfig())
        self.state = FlatTaskState(self.clock)
        self.camera = PanelCamera(self.clock, truth(), **camera)
        self.handler = self.make_handler()

    def make_handler(self, **options: Any) -> FlatHandler:
        return FlatHandler(
            flats=self.flats,
            darks=self.darks,
            layout=self.layout,
            profile=PROFILE,
            clock=self.clock,
            survey=self.survey,
            state=self.state,
            make_options=MAKE,
            **options,
        )

    def context(self) -> PanelContext:
        return PanelContext(self.clock, self.camera, PROFILE)

    def run(self, command: QueueFlat, context: PanelContext | None = None, task_id: int = 1) -> Any:
        task = CommissionTask(task_id, "flat", command, START)
        return self.handler.run(task, context or self.context())

    def reader(self, temperature: float | None = 25.0, **changes: Any) -> FlatLibraryReader:
        parts: dict[str, Any] = {
            "flats": self.flats,
            "darks": self.darks,
            "profile": PROFILE,
            "clock": self.clock,
            "survey": self.survey,
            "state": self.state,
            "temperature_c": lambda: temperature,
        }
        parts.update(changes)
        return FlatLibraryReader(**parts)


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


SMALL = QueueFlat(frames=FRAMES)


# --- A session that works ------------------------------------------------------------------------


class TestASessionThatWorks:
    def test_a_steady_light_gives_an_ok_result_with_a_pending_flat(self, rig: Rig) -> None:
        result = rig.run(SMALL)
        (entry,) = rig.flats.entries()
        assert (result.status, result.kind, result.task_id) == ("ok", "flat", 1)
        assert result.pinned is True
        assert entry.pending is True
        assert result.summary.startswith(f"Made the flat {entry.version} from 16 frames of ")
        assert result.summary.endswith("It waits on the Flat page for you to use it or discard it.")
        assert "The corners get 10 % less light than the center." in result.summary
        assert rig.flats.active_version() is None

    def test_the_data_names_the_flat_and_the_numbers_of_the_session(self, rig: Rig) -> None:
        result = rig.run(SMALL)
        (entry,) = rig.flats.entries()
        data = result.data
        assert data["version"] == entry.version
        assert (data["set_number"], data["second_set"]) == (1, False)
        assert data["frames_taken"] == data["frames_used"] == FRAMES
        assert data["exposure_s"] == pytest.approx(0.039, rel=0.05)
        assert data["level_fraction"] == pytest.approx(0.5, abs=0.03)
        assert data["temperature_c"] == 25.0
        assert -12.0 < data["corner_percent"] < -8.0
        assert data["remove_light"] is True
        assert isinstance(data["warnings"], list)
        json.dumps(dict(data))  # the result goes to an event record, so it is plain JSON

    def test_the_artifact_is_the_file_of_the_flat_under_the_data_directory(self, rig: Rig) -> None:
        result = rig.run(SMALL)
        (entry,) = rig.flats.entries()
        assert result.artifacts == (f"calibration/flats/{entry.version}.npy",)
        assert rig.layout.resolve(result.artifacts[0]).is_file()

    def test_every_frame_is_a_snapshot_that_the_handler_starts_and_stops(self, rig: Rig) -> None:
        context = rig.context()
        rig.run(SMALL, context)
        assert rig.camera.starts == rig.camera.stops == context.reads
        assert context.reads == 2 + FRAMES  # two tries of the search, and the frames

    def test_the_fields_of_the_command_win_over_the_defaults(self, rig: Rig) -> None:
        result = rig.run(QueueFlat(frames=12, target_fraction=0.35))
        assert result.data["frames_taken"] == 12
        assert result.data["level_fraction"] == pytest.approx(0.35, abs=0.03)

    def test_the_survey_settings_choose_the_readout_mode_and_the_gain(self, rig: Rig) -> None:
        rig.run(SMALL)
        configure = rig.camera.configures[0]
        assert (configure.mode, configure.gain) == ("bin2", 120)

    def test_a_second_set_ends_the_session_and_says_so(self, rig: Rig) -> None:
        rig.run(SMALL)
        again = rig.run(QueueFlat(frames=FRAMES, set_number=2), task_id=2)
        assert again.status == "ok"
        assert again.data["second_set"] is True
        assert again.data["set_number"] == 2
        assert again.summary.startswith("Made the flat flat-")
        assert "from two sets of frames" in again.summary
        assert len(rig.flats.entries()) == 1


# --- The state ------------------------------------------------------------------------------------


class TestTheState:
    def test_the_state_follows_the_session_and_ends_ok(self, rig: Rig) -> None:
        seen: dict[int, FlatTaskView] = {}
        context = rig.context()

        def watch(index: int) -> None:
            seen.setdefault(index, rig.state.snapshot())

        context.on_read = watch
        result = rig.run(SMALL, context)
        first = seen[0]  # before the first frame of the search
        assert (first.state, first.phase) == ("running", "setup")
        assert first.task_id == 1
        search = seen[1]  # after the first try: the level is known
        assert (search.phase, search.step, search.steps) == ("exposure", 1, 8)
        assert search.level_fraction == pytest.approx(0.26, abs=0.03)
        assert search.exposure_s == pytest.approx(0.02)
        assert search.target_fraction == 0.5
        capturing = seen[2 + 4]  # the fifth frame of the set is about to come
        assert (capturing.phase, capturing.step, capturing.steps) == ("capture", 4, FRAMES)
        assert capturing.message == "Frame 4 of 16: 50 % of full scale."
        assert capturing.exposure_s == pytest.approx(0.039, rel=0.05)
        assert capturing.level_fraction == pytest.approx(0.5, abs=0.03)
        view = rig.state.snapshot()
        assert (view.state, view.summary) == ("ok", result.summary)
        assert view.version == rig.flats.entries()[0].version
        assert view.phase is None
        assert view.finished_utc is not None

    def test_the_view_has_the_settings_of_the_command(self, rig: Rig) -> None:
        context = rig.context()
        seen: list[FlatTaskView] = []
        context.on_read = lambda index: seen.append(rig.state.snapshot()) if index == 0 else None
        rig.run(QueueFlat(frames=FRAMES, target_fraction=0.4, pause_after=False), context)
        view = seen[0]
        assert (view.frames, view.target_fraction, view.pause_after) == (FRAMES, 0.4, False)
        assert view.set_number == 1
        assert view.started_utc == utc_ns_to_iso(START, digits=0)

    def test_the_warnings_of_the_session_show_while_it_runs(self, tmp_path: Path) -> None:
        def one_bad_frame(index: int, exposure_s: float) -> float:
            return 204_800.0 * exposure_s * (1.3 if index == 8 else 1.0)

        made = Rig(tmp_path, light=one_bad_frame)
        context = made.context()
        seen: list[FlatTaskView] = []
        context.on_read = lambda index: seen.append(made.state.snapshot())
        made.run(SMALL, context)
        assert not seen[5].warnings
        assert any("The light drifts" in w for w in seen[-1].warnings)

    def test_a_queued_task_shows_the_settings_that_will_apply(self, rig: Rig) -> None:
        rig.state.queued(7, QueueFlat(frames=24, target_fraction=0.4, pause_after=False))
        view = rig.state.snapshot()
        assert (view.state, view.task_id) == ("queued", 7)
        assert (view.frames, view.target_fraction, view.pause_after) == (24, 0.4, False)
        assert view.started_utc is None
        assert view.message

    @pytest.mark.parametrize(
        ("scheduler_state", "words"),
        [
            ("auto", "waits for the scheduler to start it"),
            ("paused", "paused. The flat session starts after you resume it"),
            ("align", "alignment helper runs. The flat session starts after it ends"),
        ],
    )
    def test_a_queued_task_says_what_it_waits_for(
        self, rig: Rig, scheduler_state: str, words: str
    ) -> None:
        rig.state.queued(7, QueueFlat(), scheduler_state)
        assert words in rig.state.snapshot().message

    def test_a_task_that_started_before_the_call_that_queued_it_stays_running(
        self, rig: Rig
    ) -> None:
        command = QueueFlat()
        rig.state.started(3, command)
        rig.state.queued(3, command)  # the scheduler was quicker than `core`
        assert rig.state.snapshot().state == "running"
        rig.state.queued(4, command)
        assert (rig.state.snapshot().state, rig.state.snapshot().task_id) == ("queued", 4)

    def test_a_cancel_ends_a_queued_task_and_leaves_a_running_one_to_its_handler(
        self, rig: Rig
    ) -> None:
        rig.state.queued(5, QueueFlat())
        rig.state.cancelled(6)  # another task: nothing happens
        assert rig.state.snapshot().state == "queued"
        rig.state.cancelled(5)
        view = rig.state.snapshot()
        assert view.state == "aborted"
        assert view.summary == "The flat session was cancelled before it started."
        rig.state.started(8, QueueFlat())
        rig.state.cancelled(8)  # the handler ends it
        assert rig.state.snapshot().state == "running"

    def test_a_new_task_clears_what_the_last_one_left(self, rig: Rig) -> None:
        rig.state.queued(1, QueueFlat())
        rig.state.started(1, QueueFlat())
        rig.state.finished("ok", "Made a flat.", "flat-1a2b3c4d")
        assert rig.state.snapshot().version == "flat-1a2b3c4d"
        rig.state.queued(2, QueueFlat())
        view = rig.state.snapshot()
        assert (view.state, view.summary, view.version, view.finished_utc) == (
            "queued",
            "",
            None,
            None,
        )

    def test_the_state_starts_idle(self, rig: Rig) -> None:
        view = rig.state.snapshot()
        assert (view.state, view.task_id, view.summary) == ("idle", None, "")


# --- What goes wrong ------------------------------------------------------------------------------


class TestWhatGoesWrong:
    def test_without_a_dark_set_the_task_fails_with_the_words_of_the_brief(
        self, tmp_path: Path
    ) -> None:
        made = Rig(tmp_path, with_darks=False)
        result = made.run(SMALL)
        assert result.status == "failed"
        assert result.summary == "Record a dark set first (Dark page). The library is unchanged."
        assert made.camera.exposures_s == []
        view = made.state.snapshot()
        assert (view.state, view.summary) == ("failed", result.summary)
        assert made.flats.entries() == []

    def test_a_light_that_is_too_weak_is_a_failed_task_with_the_numbers(
        self, tmp_path: Path
    ) -> None:
        made = Rig(tmp_path, rate_dn_per_s=1_000.0)
        result = made.run(SMALL)
        assert result.status == "failed"
        assert result.summary == (
            "Not enough light: the frame reaches 6 % of full scale at the longest exposure of "
            "1 s. Use a brighter source, or hold it closer to the lens. The library is unchanged."
        )
        assert result.pinned is False

    def test_a_light_that_is_too_strong_is_a_failed_task(self, tmp_path: Path) -> None:
        made = Rig(tmp_path, rate_dn_per_s=5e10)
        result = made.run(SMALL)
        assert result.status == "failed"
        assert result.summary.startswith("Too much light: the frame saturates")

    def test_a_command_that_takes_the_camera_aborts_the_task(self, rig: Rig) -> None:
        context = rig.context()
        context.stops_after = 2 + 4  # in the middle of the frames
        result = rig.run(SMALL, context)
        assert result.status == "aborted"
        assert result.summary == (
            "The flat session stopped before it added a flat, and the library is unchanged."
        )
        assert rig.flats.entries() == []
        assert result.pinned is False
        view = rig.state.snapshot()
        assert (view.state, view.summary) == ("aborted", result.summary)
        assert not rig.flats.session.directory.exists()  # the partial frames are gone

    def test_a_camera_error_goes_on_to_the_scheduler_and_the_task_is_failed(self, rig: Rig) -> None:
        rig.camera.fail_at = 2 + 3
        with pytest.raises(CameraTimeoutError):
            rig.run(SMALL)
        view = rig.state.snapshot()
        assert view.state == "failed"
        assert view.summary == "The camera failed: no frame came."
        assert rig.flats.entries() == []

    def test_a_camera_that_refuses_the_settings_fails_the_task_without_a_recovery(
        self, rig: Rig
    ) -> None:
        context = rig.context()
        context.configure_error = CameraConfigError("the gain is out of range")
        result = rig.run(SMALL, context)
        assert result.status == "failed"
        assert result.summary == "The camera refused the flat settings: the gain is out of range."

    def test_an_unexpected_error_still_leaves_the_state_failed(self, rig: Rig) -> None:
        context = rig.context()
        context.on_read = lambda index: (
            (_ for _ in ()).throw(RuntimeError("a bug")) if index == 3 else None
        )
        with pytest.raises(RuntimeError):
            rig.run(SMALL, context)
        assert rig.state.snapshot().state == "failed"

    def test_a_write_error_is_a_failed_task_without_a_path(
        self, rig: Rig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(*args: object, **kwargs: object) -> None:
            raise PermissionError("denied for the folder of the library")

        monkeypatch.setattr(rig.flats, "add", broken)
        result = rig.run(SMALL)
        assert result.status == "failed"
        assert (
            result.summary
            == "The flat could not be stored (PermissionError). The library is unchanged."
        )
        assert str(rig.layout.root) not in result.summary

    def test_a_disk_without_room_fails_the_task_before_the_camera_moves(
        self, tmp_path: Path
    ) -> None:
        made = Rig(tmp_path)
        made.handler = made.make_handler(free_bytes=lambda path: 10_000, reserve_bytes=1_000_000)
        result = made.run(SMALL)
        assert result.status == "failed"
        assert result.summary.startswith("There is not enough free disk space for the frames")
        assert made.camera.exposures_s == []

    def test_a_storage_that_stopped_raw_capture_fails_the_task(self, tmp_path: Path) -> None:
        made = Rig(tmp_path)
        made.handler = made.make_handler(capture_allowed=lambda: False)
        result = made.run(SMALL)
        assert result.status == "failed"
        assert "free disk space fell below the limit" in result.summary

    def test_settings_that_the_session_refuses_fail_the_task(self, rig: Rig) -> None:
        result = rig.run(QueueFlat(frames=2))
        assert result.status == "failed"
        assert result.summary.startswith("The flat settings are not valid:")
        assert rig.state.snapshot().state == "failed"

    def test_a_second_set_without_a_first_one_fails_with_the_sentence_of_the_check(
        self, rig: Rig
    ) -> None:
        result = rig.run(QueueFlat(frames=FRAMES, set_number=2))
        assert result.status == "failed"
        assert result.summary.startswith(NO_FIRST_SET)

    def test_a_task_of_another_kind_is_refused(self, rig: Rig) -> None:
        task = CommissionTask(1, "flat", QueueBurst(), START)
        result = rig.handler.run(task, rig.context())
        assert (result.status, result.summary) == ("failed", "The task holds no flat command.")


# --- The events and the results -------------------------------------------------------------------


class TestTheEvents:
    def test_one_event_for_each_phase_and_none_for_each_frame(self, rig: Rig) -> None:
        context = rig.context()
        rig.run(SMALL, context)
        kinds = {kind for _, kind, _, _ in context.events}
        assert kinds == {PHASE_EVENT}
        phases = [detail["phase"] for _, _, _, detail in context.events if detail]
        assert phases == ["setup", "exposure", "capture", "build"]
        first = context.events[0]
        assert first[0] == "info"
        assert first[2] == "Setting up the camera and the library."
        assert first[3] == {"task_id": 1, "phase": "setup", "steps": 1}

    def test_the_names_of_the_files_are_not_in_the_results_as_paths(self, rig: Rig) -> None:
        result = rig.run(SMALL)
        text = json.dumps({"summary": result.summary, "data": dict(result.data)})
        assert str(rig.layout.root) not in text
        assert "\\" not in text


# --- The view for the RPC -------------------------------------------------------------------------


class TestTheLibraryView:
    def test_an_empty_library_has_no_flat_and_no_blocker_when_a_dark_set_exists(
        self, rig: Rig
    ) -> None:
        view = rig.reader(temperature=None).view()
        assert view.flats == []
        assert (view.mode, view.gain) == ("bin2", 120)
        assert view.sensor_temperature_c is None
        assert (view.active_version, view.pending_version) == (None, None)
        assert (view.flat_file_pinned, view.library_overrides) == (False, False)
        assert view.blocker is None
        assert view.session is None
        assert view.task.state == "idle"

    def test_without_a_dark_set_the_blocker_says_what_to_do(self, tmp_path: Path) -> None:
        made = Rig(tmp_path, with_darks=False)
        assert made.reader().view().blocker == DARK_FIRST

    def test_a_flat_shows_the_numbers_of_its_report(self, rig: Rig) -> None:
        rig.run(SMALL)
        view = rig.reader().view()
        (flat,) = view.flats
        assert flat.state == "pending"
        assert (flat.pending, flat.active) == (True, False)
        assert view.pending_version == flat.version
        assert (flat.mode, flat.gain, flat.width_px, flat.height_px) == ("bin2", 120, 256, 176)
        assert flat.sensor_temperature_c == 25.0
        assert flat.exposure_s == pytest.approx(0.039, rel=0.05)
        assert flat.frames_taken == flat.frames_used == FRAMES
        assert -12.0 < (flat.corner_percent or 0.0) < -8.0
        assert [p.corner for p in flat.vignetting] == [False] * 5 + [True]
        assert flat.shadows == len(flat.shadow_items)
        assert flat.has_image is True
        assert flat.second_set is False
        assert flat.optics_tilt is None
        assert flat.bias_source == "library"
        assert flat.age_days < 0.01
        assert flat.t_utc.endswith("Z")
        assert flat.activated_utc is None
        assert any("second set with the source turned" in w for w in flat.warnings)

    def test_a_flat_of_two_sets_has_the_split_and_the_agreement(self, rig: Rig) -> None:
        rig.run(SMALL)
        rig.run(QueueFlat(frames=FRAMES, set_number=2), task_id=2)
        (flat,) = rig.reader().view().flats
        assert flat.second_set is True
        assert flat.source_turned is True
        assert flat.optics_tilt is not None
        assert flat.source_tilt is not None
        assert flat.agreement is not None
        assert [s.number for s in flat.sets] == [1, 2]

    def test_the_first_set_of_a_session_is_in_the_view_until_the_session_ends(
        self, rig: Rig
    ) -> None:
        rig.run(SMALL)
        view = rig.reader().view()
        assert view.session is not None
        assert view.session.version == view.flats[0].version
        assert view.session.frames == FRAMES
        assert view.session.t_utc == "2026-10-04T20:00:00Z"
        assert view.session.expires_utc == "2026-10-05T20:00:00Z"
        rig.run(QueueFlat(frames=FRAMES, set_number=2), task_id=2)
        assert rig.reader().view().session is None

    def test_a_session_that_expired_is_gone_from_the_view(self, rig: Rig) -> None:
        rig.run(SMALL)
        rig.clock.advance(24 * 3600 + 60)
        assert rig.reader().view().session is None

    def test_flat_file_and_the_library_show_in_the_view(self, tmp_path: Path) -> None:
        made = Rig(tmp_path)
        made.survey = made.survey.model_copy(
            update={
                "flat_file": "pinned.npy",
                "calibration_dir": str(made.layout.root / "calibration"),
            }
        )
        made.run(SMALL)
        reader = made.reader()
        before = reader.view()
        assert (before.flat_file_pinned, before.library_overrides) == (True, False)
        assert reader.activate(before.flats[0].version).ok
        after = reader.view()
        assert (after.flat_file_pinned, after.library_overrides) == (True, True)
        assert after.active_version == after.flats[0].version

    def test_the_view_is_what_the_contract_decodes(self, rig: Rig) -> None:
        rig.run(SMALL)
        rig.state.queued(9, QueueFlat())
        raw = rig.reader().view().model_dump(mode="json")
        decoded = decode_flat_library(json.loads(json.dumps(raw)))
        assert len(decoded.flats) == 1
        assert decoded.task.state == "queued"
        assert decoded.flats[0].vignetting[-1].corner is True

    def test_a_report_of_an_older_shape_still_shows(self) -> None:
        entry = FlatEntry(
            version="flat-1a2b3c4d",
            t_utc_ns=START,
            state=PENDING,
            report={"schema": 1, "t_utc_ns": START, "version": "flat-1a2b3c4d", "state": PENDING},
        )
        view = flat_view(entry, active=False, now_ns=START + NS_PER_S, has_image=False)
        assert view.version == "flat-1a2b3c4d"
        assert view.t_utc == "2026-10-04T20:00:00Z"
        assert (view.shadows, view.frames_used, view.has_image) == (0, 0, False)
        assert view.vignetting == []
        assert view.corner_percent is None
        assert view.warnings == []

    def test_a_malformed_value_in_a_report_does_not_break_the_view(self) -> None:
        entry = FlatEntry(
            version="flat-1a2b3c4d",
            t_utc_ns=START,
            state=APPROVED,
            report={
                "schema": 1,
                "t_utc_ns": START,
                "version": "flat-1a2b3c4d",
                "state": APPROVED,
                "gain": True,
                "noise_percent": "a lot",
                "sets": [7, {"number": "one", "used": 3, "dropped": {"low": 2}}],
                "vignetting": {"points": ["x", {"radius_deg": 1.0, "change_percent": None}]},
            },
        )
        view = flat_view(entry, active=True, now_ns=START, has_image=True)
        assert view.gain == 0
        assert view.noise_percent is None
        assert [(s.number, s.used, s.dropped) for s in view.sets] == [(0, 3, {"low": 2})]
        assert len(view.vignetting) == 1
        assert view.active is True


# --- The actions ----------------------------------------------------------------------------------


class TestTheActions:
    def test_activating_a_flat_puts_it_in_use_and_ends_its_session(self, rig: Rig) -> None:
        rig.run(SMALL)
        reader = rig.reader()
        (flat,) = reader.view().flats
        answer = reader.activate(flat.version)
        assert (answer.ok, answer.reason, answer.version) == (True, None, flat.version)
        assert answer.message == (
            f"The flat {flat.version} is in use. The survey divides by it from its next frame."
        )
        assert rig.flats.active_version() == flat.version
        view = reader.view()
        assert (view.active_version, view.pending_version) == (flat.version, None)
        assert view.flats[0].active is True
        assert view.flats[0].state == "approved"
        assert view.flats[0].activated_utc is not None
        assert view.session is None  # the first set has no use any more
        assert not rig.flats.session.directory.exists()

    def test_the_answer_says_that_the_flat_replaces_the_flat_file(self, tmp_path: Path) -> None:
        made = Rig(tmp_path)
        made.survey = made.survey.model_copy(update={"flat_file": "pinned.npy"})
        made.run(SMALL)
        reader = made.reader()
        answer = reader.activate(reader.view().flats[0].version)
        assert answer.message.endswith("It replaces the flat of the setting flat_file.")

    def test_an_unknown_flat_is_refused_with_its_reason(self, rig: Rig) -> None:
        rig.run(SMALL)
        answer = rig.reader().activate("flat-00000000")
        assert (answer.ok, answer.reason) == (False, "unknown")
        assert answer.message == "There is no flat with that name."

    def test_a_flat_of_another_size_is_refused(self, rig: Rig) -> None:
        entry = rig.flats.add(
            fx.lens_flat(fx.OWNER_LENS, (40, 60), scale_down=60.0), {"t_utc_ns": START}
        )
        answer = rig.reader().activate(entry.version)
        assert (answer.ok, answer.reason) == (False, "invalid")
        assert "60 x 40 pixels" in answer.message

    def test_a_session_that_is_queued_or_running_blocks_both_actions(self, rig: Rig) -> None:
        rig.run(SMALL)
        reader = rig.reader()
        version = reader.view().flats[0].version
        for state in ("queued", "running"):
            rig.state.queued(5, QueueFlat()) if state == "queued" else rig.state.started(
                5, QueueFlat()
            )
            for answer in (reader.activate(version), reader.delete(version)):
                assert (answer.ok, answer.reason) == (False, "busy")
                assert answer.message == (
                    "A flat session is queued or running. Wait until it ends, or stop it."
                )
        assert rig.flats.get(version) is not None
        assert rig.flats.active_version() is None

    def test_deleting_a_pending_flat_removes_it_and_ends_its_session(self, rig: Rig) -> None:
        rig.run(SMALL)
        reader = rig.reader()
        version = reader.view().flats[0].version
        answer = reader.delete(version)
        assert (answer.ok, answer.message) == (True, f"The flat {version} is deleted.")
        view = reader.view()
        assert view.flats == []
        assert view.pending_version is None
        assert view.session is None
        assert rig.flats.directory.exists()
        assert not any(rig.flats.directory.glob("flat-*"))

    def test_the_flat_in_use_cannot_be_deleted(self, rig: Rig) -> None:
        rig.run(SMALL)
        reader = rig.reader()
        version = reader.view().flats[0].version
        reader.activate(version)
        answer = reader.delete(version)
        assert (answer.ok, answer.reason) == (False, "active")
        assert answer.message == "The flat is in use. Activate another flat first."
        assert rig.flats.get(version) is not None

    def test_deleting_an_unknown_flat_is_refused(self, rig: Rig) -> None:
        answer = rig.reader().delete("flat-00000000")
        assert (answer.ok, answer.reason) == (False, "unknown")

    def test_a_flat_that_is_not_the_one_of_the_session_leaves_the_session_alone(
        self, rig: Rig
    ) -> None:
        rig.run(SMALL)
        other = rig.flats.add(
            fx.lens_flat(fx.OWNER_LENS, SHAPE, scale_down=16.0, seed=9), {"t_utc_ns": START - 5}
        )
        reader = rig.reader()
        assert reader.delete(other.version).ok
        assert reader.view().session is not None

    def test_the_image_of_a_flat_is_its_jpeg(self, rig: Rig) -> None:
        rig.run(SMALL)
        reader = rig.reader()
        jpeg = reader.image(reader.view().flats[0].version)
        assert jpeg is not None
        assert jpeg.startswith(b"\xff\xd8\xff")

    def test_a_flat_without_an_image_or_with_an_image_that_is_too_large_has_none(
        self, rig: Rig
    ) -> None:
        reader = rig.reader()
        assert reader.image("flat-00000000") is None
        assert reader.image("not a version") is None
        entry = rig.flats.add(
            fx.lens_flat(fx.OWNER_LENS, SHAPE, scale_down=16.0, seed=4),
            {"t_utc_ns": START},
            preview=b"\xff\xd8\xff" + b"0" * (MAX_FLAT_JPEG_BYTES + 1),
        )
        assert reader.image(entry.version) is None


# --- The check of a command -----------------------------------------------------------------------


class TestTheCheckOfACommand:
    def test_a_first_set_with_a_dark_library_passes(self, rig: Rig) -> None:
        assert rig.reader().check(QueueFlat()) is None

    def test_without_a_dark_set_the_check_says_what_to_do(self, tmp_path: Path) -> None:
        made = Rig(tmp_path, with_darks=False)
        assert made.reader().check(QueueFlat()) == "Record a dark set first (Dark page)."

    def test_a_second_set_needs_a_first_set_that_is_still_there(self, rig: Rig) -> None:
        reader = rig.reader()
        assert reader.check(QueueFlat(set_number=2)) == NO_FIRST_SET
        rig.run(SMALL)
        assert reader.check(QueueFlat(set_number=2)) is None
        rig.clock.advance(24 * 3600 + 60)
        assert reader.check(QueueFlat(set_number=2)) == NO_FIRST_SET


def test_the_camera_adapter_of_the_dark_session_serves_the_flat_session(rig: Rig) -> None:
    context = rig.context()
    camera = ContextCamera(context)
    camera.configure(StreamConfig("bin2", 40_000, 120))
    frame = camera.take(0.04)
    assert frame.shape == SHAPE
    assert (rig.camera.starts, rig.camera.stops) == (1, 1)


def test_the_summary_of_a_session_says_what_to_do_next(rig: Rig) -> None:
    result = rig.run(SMALL)
    (entry,) = rig.flats.entries()
    assert describe_session.__doc__ is not None
    assert entry.version in result.summary
