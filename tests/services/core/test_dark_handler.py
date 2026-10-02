"""The dark handler: the session on the lent camera, its result, and the state that the RPC reads.

The handler runs against a stand-in for the scheduler's context that wraps a simulated camera on a
`VirtualClock`, so a 30 s exposure and a wait for the cover take no real time. The camera is
covered (no stars, no sky) or has a cover that goes on after some frames.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, VirtualClock, utc_ns_to_iso
from seeingmon.drivers.base import (
    CameraConfigError,
    CameraDriver,
    CameraTimeoutError,
)
from seeingmon.drivers.sim import SimDriver, SimOptions
from seeingmon.frames import ActiveStream, Frame, StreamConfig
from seeingmon.profile import Profile
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.scheduler.commands import QueueBurst, QueueDark
from seeingmon.scheduler.commission import CommissionTask, FastWindowSample
from seeingmon.scheduler.config import SchedulerConfig
from seeingmon.services.core.commissioning import dark as dark_module
from seeingmon.services.core.commissioning.dark import (
    PHASE_EVENT,
    DarkHandler,
    DarkLibraryReader,
    DarkTaskState,
)
from seeingmon.services.web.contract import MAX_LIST_ITEMS, DarkTaskView, decode_dark_library
from seeingmon.store.layout import DataLayout
from seeingmon.survey.analyzer import InlineExecutor, create_survey_analyzer
from seeingmon.survey.catalog import write_catalog
from seeingmon.survey.config import DarkConfig, SurveyConfig
from seeingmon.survey.dark import DarkLibrary
from tests.survey import simfx, synth

PROFILE = synth.cropped_profile(512, 384)
START = 1_800_000_000 * NS_PER_S
SMALL = DarkConfig(frames=3, bias_frames=3, poll_s=1.0, stable_polls=2, wait_timeout_s=120.0)


class SimContext:
    """The context of a handler on a simulated camera that the test controls."""

    def __init__(self, clock: VirtualClock, driver: CameraDriver, profile: Profile) -> None:
        self._clock = clock
        self._profile = profile
        self.driver = driver
        driver.open()
        self.reads = 0
        self.stops_after: int | None = None
        self.on_read: Callable[[int], None] | None = None
        self.fail_read_at: int | None = None
        self.events: list[tuple[str, str, str, Mapping[str, Any] | None]] = []
        self.configures: list[StreamConfig] = []
        self.starts = self.stop_calls = 0

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
        self.configures.append(config)
        return self.driver.configure(config)

    def start(self) -> None:
        self.starts += 1
        self.driver.start()

    def stop(self) -> None:
        self.stop_calls += 1
        self.driver.stop()

    def read_frame(self, timeout_s: float | None = None) -> Frame:
        if self.fail_read_at is not None and self.reads == self.fail_read_at:
            raise CameraTimeoutError("no frame came")
        if self.on_read is not None:
            self.on_read(self.reads)
        frame = self.driver.read_frame(timeout_s or 60.0)
        self.reads += 1
        return frame

    def run_fast_window(self, config: StreamConfig, duration_s: float) -> FastWindowSample:
        raise NotImplementedError("a dark session runs no fast window")


def covered(clock: VirtualClock, *, seed: int = 3, profile: Profile = PROFILE) -> SimDriver:
    options = simfx.covered_options(ambient_c=12.0, hot_pixels_per_mpix=200.0, seed=seed)
    return simfx.make_driver(profile, options, clock)


class Rig:
    def __init__(
        self, tmp_path: Path, config: DarkConfig = SMALL, profile: Profile = PROFILE
    ) -> None:
        self.clock = VirtualClock(START)
        self.profile = profile
        self.layout = DataLayout(tmp_path / "data")
        self.layout.create()
        self.library = DarkLibrary(self.layout.root / "calibration" / "darks")
        self.config = config
        self.state = DarkTaskState(self.clock, config)
        self.handler = DarkHandler(
            library=self.library,
            layout=self.layout,
            profile=profile,
            clock=self.clock,
            config=config,
            state=self.state,
        )

    def context(self, driver: CameraDriver | None = None) -> SimContext:
        return SimContext(
            self.clock, driver or covered(self.clock, profile=self.profile), self.profile
        )

    def run(self, command: QueueDark, context: SimContext, task_id: int = 1) -> Any:
        task = CommissionTask(task_id, "dark", command, START)
        return self.handler.run(task, context)


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


class TestASetThatWorks:
    def test_a_covered_camera_gives_an_ok_result_with_the_set(self, rig: Rig) -> None:
        context = rig.context()
        result = rig.run(QueueDark(wait_for_cover=False), context)
        (dark_set,) = rig.library.sets()
        assert (result.status, result.kind, result.task_id) == ("ok", "dark", 1)
        assert result.pinned is True
        assert result.summary.startswith(f"Added {dark_set.name} to the dark library:")
        assert result.summary.endswith(".")
        assert dark_set.n_frames == 3
        assert dark_set.n_bias_frames == 3
        assert dark_set.exposure_s == 30.0
        assert dark_set.temperature_c == pytest.approx(16.0)  # 12 C and 4 C of self-heating

    def test_the_data_names_the_set_the_rate_and_the_model(self, rig: Rig) -> None:
        result = rig.run(QueueDark(wait_for_cover=False), rig.context())
        data = result.data
        (dark_set,) = rig.library.sets()
        e_per_adu = PROFILE.e_per_adu("bin2", 120)
        assert data["set_name"] == dark_set.name
        assert data["rate_e_per_s"] == pytest.approx(dark_set.rate_dn_per_s * e_per_adu)
        assert data["temperature_c"] == pytest.approx(dark_set.temperature_c)
        assert data["hot_pixels"] == dark_set.n_hot_pixels
        assert (data["frames"], data["bias_frames"], data["exposure_s"]) == (3, 3, 30.0)
        assert data["remove_cover"] is True
        assert data["due"] is False
        assert data["due_reason"] == "a recent set covers the temperature"
        assert data["n_sets"] == 1
        model = data["model"]
        assert isinstance(model, dict)
        assert model["n_sets"] == 1
        assert model["doubling_fitted"] is False
        json.dumps(dict(data))  # the result goes to an event record, so it is plain JSON

    def test_the_artifact_is_the_file_of_the_set_under_the_data_directory(self, rig: Rig) -> None:
        result = rig.run(QueueDark(wait_for_cover=False), rig.context())
        (dark_set,) = rig.library.sets()
        assert result.artifacts == (f"calibration/darks/{dark_set.name}",)
        assert rig.layout.resolve(result.artifacts[0]).is_file()

    def test_the_fields_of_the_command_win_over_the_defaults(self, rig: Rig) -> None:
        context = rig.context()
        rig.run(QueueDark(exposure_s=10.0, frames=4, bias_frames=5, wait_for_cover=False), context)
        (dark_set,) = rig.library.sets()
        assert (dark_set.exposure_s, dark_set.n_frames, dark_set.n_bias_frames) == (10.0, 4, 5)
        assert [c.exposure_us for c in context.configures][-1] == 10_000_000

    def test_the_defaults_come_from_the_survey_dark_section(self, tmp_path: Path) -> None:
        rig = Rig(tmp_path, SMALL.model_copy(update={"frames": 5, "exposure_s": 20.0, "gain": 100}))
        rig.run(QueueDark(wait_for_cover=False), rig.context())
        (dark_set,) = rig.library.sets()
        assert (dark_set.n_frames, dark_set.exposure_s, dark_set.gain) == (5, 20.0, 100)

    def test_every_frame_is_a_snapshot_that_the_handler_starts_and_stops(self, rig: Rig) -> None:
        context = rig.context()
        rig.run(QueueDark(wait_for_cover=False), context)
        assert context.starts == context.stop_calls == 3 + 3
        assert context.reads == 6
        assert len(context.configures) == 2  # the bias, and the dark frames

    def test_a_second_set_adds_to_the_library(self, rig: Rig) -> None:
        rig.run(QueueDark(wait_for_cover=False), rig.context())
        again = rig.run(QueueDark(wait_for_cover=False), rig.context(covered(rig.clock, seed=4)), 2)
        assert again.status == "ok"
        assert again.data["n_sets"] == 2
        assert len(rig.library.sets()) == 2


class TestTheWaitForTheCover:
    def make_context(self, rig: Rig, *, cover_after: int) -> SimContext:
        uncovered = simfx.make_driver(PROFILE, SimOptions(seed=5), rig.clock)
        cover = covered(rig.clock, seed=5)
        driver = simfx.CoverableDriver(uncovered, cover, cover_after_reads=cover_after)
        return rig.context(driver)

    def test_the_task_waits_until_the_camera_is_dark_and_goes_on(self, rig: Rig) -> None:
        context = self.make_context(rig, cover_after=3 + 2)
        result = rig.run(QueueDark(), context)
        assert result.status == "ok"
        assert float(result.data["waited_s"]) > 0
        assert len(rig.library.sets()) == 1

    def test_the_state_shows_the_wait_and_then_the_frames(self, rig: Rig) -> None:
        context = self.make_context(rig, cover_after=3 + 2)
        seen: dict[int, DarkTaskView] = {}

        def watch(index: int) -> None:
            seen.setdefault(index, rig.state.snapshot())

        context.on_read = watch
        rig.run(QueueDark(), context)
        before_cover = seen[3]  # the first test frame comes after the three bias frames
        assert (before_cover.state, before_cover.phase) == ("running", "cover")
        assert before_cover.message == "Cover the camera now. Waiting for a dark frame."
        lit = seen[4]  # after the first test frame, which saw the sky
        assert (lit.phase, lit.covered) == ("cover", False)
        assert lit.reason  # the check says why: the level is above the bias
        assert lit.level_dn is not None
        dark_frame = seen[5 + 2 + 1]  # inside the dark phase
        assert dark_frame.phase == "dark"
        assert dark_frame.covered is True
        assert (dark_frame.steps, dark_frame.step) == (3, 1)

    def test_a_camera_that_stays_uncovered_fails_the_task_with_a_plain_reason(
        self, rig: Rig
    ) -> None:
        uncovered = simfx.make_driver(PROFILE, SimOptions(seed=5), rig.clock)
        config = SMALL.model_copy(update={"wait_timeout_s": 20.0})
        small = Rig(Path(rig.layout.root).parent / "other", config)
        result = small.run(QueueDark(), small.context(uncovered))
        assert result.status == "failed"
        assert result.summary.startswith("The camera was not dark after 20 s")
        assert "cover it" in result.summary
        assert result.summary.endswith("The library is unchanged.")
        assert small.library.sets() == ()
        view = small.state.snapshot()
        assert (view.state, view.summary) == ("failed", result.summary)
        assert view.set_name is None
        assert view.finished_utc is not None


class TestWithoutTheWait:
    def test_the_first_frame_that_is_not_dark_fails_the_task(self, rig: Rig) -> None:
        uncovered = simfx.make_driver(PROFILE, SimOptions(seed=5), rig.clock)
        result = rig.run(QueueDark(wait_for_cover=False), rig.context(uncovered))
        assert result.status == "failed"
        assert result.summary.startswith("Dark frame 1 of 3 is not dark:")
        assert result.summary.endswith("The library is unchanged.")
        assert rig.library.sets() == ()

    def test_a_camera_without_a_temperature_fails_with_a_plain_reason(self, rig: Rig) -> None:
        class NoTemperature:
            def __init__(self, inner: SimDriver) -> None:
                self._inner = inner

            def __getattr__(self, name: str) -> object:
                return getattr(self._inner, name)

            def read_frame(self, timeout_s: float) -> Frame:
                return replace(self._inner.read_frame(timeout_s), temperature_c=None)

        driver: Any = NoTemperature(covered(rig.clock))
        result = rig.run(QueueDark(wait_for_cover=False), rig.context(driver))
        assert result.status == "failed"
        assert "the camera reports no sensor temperature" in result.summary.lower()
        assert rig.library.sets() == ()

    def test_settings_that_the_session_refuses_fail_the_task(self, tmp_path: Path) -> None:
        rig = Rig(tmp_path, SMALL.model_copy(update={"mode": "bin9"}))
        result = rig.run(QueueDark(wait_for_cover=False), rig.context())
        assert result.status == "failed"
        assert result.summary.startswith("The dark settings are not valid:")
        assert "bin9" in result.summary
        assert rig.state.snapshot().state == "failed"

    def test_a_camera_that_refuses_the_settings_fails_the_task_without_a_recovery(
        self, rig: Rig
    ) -> None:
        context = rig.context()

        def refuse(config: StreamConfig) -> ActiveStream:
            raise CameraConfigError("the exposure is out of range")

        context.configure = refuse  # type: ignore[method-assign]
        result = rig.run(QueueDark(wait_for_cover=False), context)
        assert result.status == "failed"
        assert (
            result.summary == "The camera refused the dark settings: the exposure is out of range."
        )


class TestStoppingAndFaults:
    def test_a_command_that_takes_the_camera_aborts_the_task(self, rig: Rig) -> None:
        context = rig.context()
        context.stops_after = 3 + 1  # after the bias frames and the first dark frame
        result = rig.run(QueueDark(wait_for_cover=False), context)
        assert result.status == "aborted"
        assert "another command took the camera" in result.summary
        assert "the library is unchanged" in result.summary
        assert rig.library.sets() == ()
        view = rig.state.snapshot()
        assert (view.state, view.summary) == ("aborted", result.summary)

    def test_a_camera_error_goes_on_to_the_scheduler_and_the_task_is_failed(self, rig: Rig) -> None:
        context = rig.context()
        context.fail_read_at = 4
        with pytest.raises(CameraTimeoutError):
            rig.run(QueueDark(wait_for_cover=False), context)
        view = rig.state.snapshot()
        assert view.state == "failed"
        assert view.summary == "The camera failed: no frame came."
        assert rig.library.sets() == ()

    def test_an_unexpected_error_still_leaves_the_state_failed(self, rig: Rig) -> None:
        context = rig.context()
        context.on_read = lambda index: (
            (_ for _ in ()).throw(RuntimeError("a bug")) if index == 2 else None
        )
        with pytest.raises(RuntimeError):
            rig.run(QueueDark(wait_for_cover=False), context)
        assert rig.state.snapshot().state == "failed"

    def test_a_write_error_is_a_failed_task_without_a_path(
        self, rig: Rig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(*args: object, **kwargs: object) -> None:
            raise PermissionError("denied for the folder of the library")

        monkeypatch.setattr(rig.library, "add_set", broken)
        result = rig.run(QueueDark(wait_for_cover=False), rig.context())
        assert result.status == "failed"
        assert result.summary == "The dark set could not be written (PermissionError)."
        assert str(rig.layout.root) not in result.summary

    def test_a_task_of_another_kind_is_refused(self, rig: Rig) -> None:
        task = CommissionTask(1, "dark", QueueBurst(), START)
        result = rig.handler.run(task, rig.context())
        assert (result.status, result.summary) == ("failed", "The task holds no dark command.")


class TestTheEvents:
    def test_one_event_for_each_phase_and_none_for_each_frame(self, rig: Rig) -> None:
        context = rig.context()
        rig.run(QueueDark(wait_for_cover=False), context)
        kinds = [kind for _, kind, _, _ in context.events]
        assert kinds == [PHASE_EVENT] * 3
        phases = [detail["phase"] for _, _, _, detail in context.events if detail]
        assert phases == ["bias", "dark", "build"]

    def test_the_wait_adds_a_phase(self, rig: Rig) -> None:
        uncovered = simfx.make_driver(PROFILE, SimOptions(seed=5), rig.clock)
        driver = simfx.CoverableDriver(uncovered, covered(rig.clock, seed=5), cover_after_reads=5)
        context = rig.context(driver)
        rig.run(QueueDark(), context)
        phases = [detail["phase"] for _, _, _, detail in context.events if detail]
        assert phases == ["bias", "cover", "dark", "build"]
        first = context.events[0]
        assert first[0] == "info"
        assert first[2] == "Taking 3 bias frames at the shortest exposure."
        assert first[3] == {"task_id": 1, "phase": "bias", "steps": 3}


class TestTheState:
    def state(self) -> DarkTaskState:
        return DarkTaskState(VirtualClock(START), SMALL)

    def test_it_starts_idle(self) -> None:
        view = self.state().snapshot()
        assert view.state == "idle"
        assert view.task_id is None
        assert view.summary == ""

    def test_a_queued_task_shows_the_settings_that_will_apply(self) -> None:
        state = self.state()
        state.queued(7, QueueDark(frames=5, wait_for_cover=False, pause_after=False))
        view = state.snapshot()
        assert (view.state, view.task_id) == ("queued", 7)
        assert (view.frames, view.bias_frames, view.exposure_s) == (5, 3, 30.0)
        assert (view.wait_for_cover, view.pause_after) == (False, False)
        assert view.started_utc is None
        assert view.message

    def test_a_task_that_started_before_the_call_that_queued_it_stays_running(self) -> None:
        state = self.state()
        command = QueueDark()
        state.started(3, command, None)
        state.queued(3, command)  # the scheduler was quicker than `core`
        assert state.snapshot().state == "running"
        state.queued(4, command)  # a new task replaces the old state
        assert (state.snapshot().state, state.snapshot().task_id) == ("queued", 4)

    def test_a_new_task_clears_what_the_last_one_left(self) -> None:
        state = self.state()
        state.queued(1, QueueDark())
        state.started(1, QueueDark(), None)
        state.finished("ok", "Added a set.", "dark-set.fits")
        assert state.snapshot().set_name == "dark-set.fits"
        state.queued(2, QueueDark())
        view = state.snapshot()
        assert (view.state, view.summary, view.set_name, view.finished_utc) == (
            "queued",
            "",
            None,
            None,
        )

    def test_a_started_task_has_its_time_and_the_finished_one_clears_the_phase(self) -> None:
        clock = VirtualClock(START)
        state = DarkTaskState(clock, SMALL)
        state.started(1, QueueDark(), None)
        assert state.snapshot().started_utc == utc_ns_to_iso(START, digits=0)
        clock.step_utc_ns(60 * NS_PER_S)
        state.finished("failed", "It failed.")
        view = state.snapshot()
        assert view.phase is None
        assert view.finished_utc == utc_ns_to_iso(START + 60 * NS_PER_S, digits=0)
        assert (view.state, view.summary) == ("failed", "It failed.")


class TestTheLibraryView:
    def reader(self, rig: Rig, temperature: float | None = 16.0) -> DarkLibraryReader:
        return DarkLibraryReader(
            library=rig.library,
            profile=PROFILE,
            clock=rig.clock,
            config=rig.config,
            state=rig.state,
            temperature_c=lambda: temperature,
        )

    def test_an_empty_library_is_due(self, rig: Rig) -> None:
        view = self.reader(rig).view()
        assert view.sets == []
        assert view.model is None
        assert view.status.due is True
        assert view.status.reason == "the library holds no dark set"
        assert (view.mode, view.gain, view.exposure_s) == ("bin2", 120, 30.0)
        assert view.sensor_temperature_c == 16.0
        assert view.task.state == "idle"

    def test_the_view_describes_the_sets_newest_first_in_electrons(self, rig: Rig) -> None:
        rig.run(QueueDark(wait_for_cover=False), rig.context())
        rig.clock.step_utc_ns(2 * 86_400 * NS_PER_S)
        rig.run(QueueDark(wait_for_cover=False), rig.context(covered(rig.clock, seed=4)), 2)
        rig.clock.step_utc_ns(3_600 * NS_PER_S)
        view = self.reader(rig).view()
        older, newer = rig.library.sets()
        assert [s.name for s in view.sets] == [newer.name, older.name]
        item = view.sets[0]
        e_per_adu = PROFILE.e_per_adu("bin2", 120)
        assert item.rate_e_per_s == pytest.approx(newer.rate_dn_per_s * e_per_adu)
        assert item.t_utc == utc_ns_to_iso(newer.t_utc_ns, digits=0)
        assert item.age_days == pytest.approx(newer.age_s(rig.clock.utc_ns()) / 86_400)
        assert 0.04 < item.age_days < 0.2
        assert 2.0 < view.sets[1].age_days < 2.3
        assert (item.n_frames, item.n_bias_frames, item.hot_pixels) == (3, 3, newer.n_hot_pixels)
        assert item.temperature_c == pytest.approx(newer.temperature_c)

    def test_the_status_and_the_model_follow_the_survey_settings(self, rig: Rig) -> None:
        rig.run(QueueDark(wait_for_cover=False), rig.context())
        view = self.reader(rig, temperature=16.0).view()
        (dark_set,) = rig.library.sets()
        assert view.status.due is False
        assert view.status.reason == "a recent set covers the temperature"
        assert view.status.nearest_name == dark_set.name
        assert view.status.gap_c == pytest.approx(0.0, abs=0.05)
        assert (view.status.tolerance_c, view.status.max_age_days) == (3.0, 183.0)
        assert view.status.newest_age_days is not None
        model = view.model
        assert model is not None
        assert (model.n_sets, model.doubling_fitted, model.reference_c) == (1, False, 20.0)
        assert model.rate_ref_e_per_s > 0
        far = self.reader(rig, temperature=30.0).view()
        assert far.status.due is True
        assert "no recent set within 3.0 C of 30.0 C" in far.status.reason
        none = self.reader(rig, temperature=None).view()
        assert none.sensor_temperature_c is None
        assert none.status.reason == "the camera reports no temperature"

    def test_the_list_holds_at_most_the_limit_of_the_contract(
        self, rig: Rig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for index in range(5):
            rig.library.add_set(
                np.full((8, 8), 100, dtype=np.uint16),
                mode="bin2",
                gain=120,
                exposure_s=30.0,
                temperature_c=10.0 + index,
                temperature_spread_c=0.1,
                t_utc_ns=START - (index + 1) * 3_600 * NS_PER_S,
                n_frames=9,
                n_bias_frames=9,
                bias_dn=100.0,
                read_noise_dn=2.0,
                adc_bits=14,
            )
        monkeypatch.setattr(dark_module, "MAX_LIST_ITEMS", 3)
        view = self.reader(rig).view()
        assert len(view.sets) == 3
        assert [s.temperature_c for s in view.sets] == [10.0, 11.0, 12.0]  # the newest three
        assert MAX_LIST_ITEMS == 256

    def test_the_task_of_the_state_is_in_the_view(self, rig: Rig) -> None:
        rig.state.queued(9, QueueDark())
        assert self.reader(rig).view().task.state == "queued"

    def test_the_view_is_what_the_contract_decodes(self, rig: Rig) -> None:
        rig.run(QueueDark(wait_for_cover=False), rig.context())
        raw = self.reader(rig).view().model_dump(mode="json")
        decoded = decode_dark_library(json.loads(json.dumps(raw)))
        assert decoded.status.due is False
        assert len(decoded.sets) == 1
        assert decoded.task.state == "ok"
        assert decoded.task.set_name == decoded.sets[0].name


def test_the_names_of_the_files_are_not_in_the_results_as_paths(rig: Rig) -> None:
    result = rig.run(QueueDark(wait_for_cover=False), rig.context())
    text = json.dumps({"summary": result.summary, "data": dict(result.data)})
    assert str(rig.layout.root) not in text
    assert "\\" not in text


def test_a_state_is_reused_across_the_tasks_of_a_process(rig: Rig) -> None:
    rig.run(QueueDark(wait_for_cover=False), rig.context())
    assert rig.state.snapshot().state == "ok"
    bad = SMALL.model_copy(update={"mode": "bin9"})
    rig.handler = DarkHandler(
        library=rig.library,
        layout=rig.layout,
        profile=PROFILE,
        clock=rig.clock,
        config=bad,
        state=rig.state,
    )
    rig.run(QueueDark(wait_for_cover=False), rig.context(), 2)
    view = rig.state.snapshot()
    assert (view.state, view.task_id) == ("failed", 2)


class TestTheNextSurveyFrame:
    """The survey analysis reads the folder of the library for every frame, so no restart is due."""

    def test_a_set_that_the_task_adds_serves_the_very_next_frame(self, tmp_path: Path) -> None:
        profile = synth.cropped_profile(1600, 1200)
        catalog = synth.synthetic_catalog(cap_radius_deg=8.0, density_scale=1.0, seed=3)
        write_catalog(tmp_path / "cap.smcat", catalog)
        rig = Rig(tmp_path, SMALL, profile)
        analyzer = create_survey_analyzer(
            profile=profile,
            station_id="test",
            config=SurveyConfig(catalog_path=str(tmp_path / "cap.smcat"), solvers=()),
            executor=InlineExecutor(),
            layout=rig.layout,  # the library of the analyzer is the one of the handler
        )
        frame, _ = synth.render_frame(
            catalog,
            profile,
            rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
            zero_point_mag=19.3,
            sky_e_per_s_px=3.0,
            seed=8,
        )

        def sky_quality() -> SkyQualityRecord:
            (output,) = analyzer.poll()
            return next(r for r in output.records if isinstance(r, SkyQualityRecord))

        analyzer.submit(frame)
        before = sky_quality()
        result = rig.run(QueueDark(wait_for_cover=False), rig.context())
        assert result.status == "ok"
        analyzer.submit(
            replace(
                frame,
                seq=frame.seq + 1,
                t_arrival_ns=frame.t_arrival_ns + 180 * NS_PER_S,
                t_utc_ns=frame.t_utc_ns + 180 * NS_PER_S,
            )
        )
        after = sky_quality()
        analyzer.close()
        assert before.dark_model_version is None
        assert "dark_due" in before.flags
        assert after.dark_model_version == rig.library.version()
        assert "dark_due" not in after.flags
