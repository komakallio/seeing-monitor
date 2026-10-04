"""The flat session: the search for the exposure, the frames, the combination, and the second set.

The camera is a scripted panel (`tests.survey.panelfx`) on a `VirtualClock`: the level in the middle
of its frames is the rate of the panel times the exposure, so a test knows what the search must
find. The lens is the one of the owner on a sensor of 256 x 176 pixels, and the dark library holds
two sets of bin2 at gain 120 (a bias of 128 counts at 20.2 C and 134.2 at 30 C).
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seeingmon.clock import VirtualClock, iso_to_utc_ns
from seeingmon.drivers.base import CameraTimeoutError
from seeingmon.profile import Profile
from seeingmon.recordings.ser import SerFile
from seeingmon.survey import flat_make as fm
from seeingmon.survey import flat_session as fs
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.dark import DarkLibrary
from seeingmon.survey.flat_library import FlatLibrary
from tests.survey import flatfx as fx
from tests.survey.panelfx import PanelCamera

SHAPE = (176, 256)
START = iso_to_utc_ns("2026-10-04T20:00:00Z")
PROFILE: Profile = fx.scaled_profile(256, 176)
MAKE = fm.MakeOptions(bin_factor=2, high_pass_px=10.0, edge_margin_px=6.0)
FRAMES = 16
FULL_SCALE = 16383.0


@lru_cache(maxsize=1)
def truth() -> fx.FloatImage:
    return fx.lens_flat(
        fx.OWNER_LENS, SHAPE, scale_down=fx.REFERENCE_SHAPE[1] / SHAPE[1], edge_artifact_x=3.0
    )


@lru_cache(maxsize=1)
def dim_corners() -> fx.FloatImage:
    """A panel that lights the corners with 40% of the light of the middle."""
    height, width = SHAPE
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = np.hypot(x - (width - 1) / 2, y - (height - 1) / 2) / np.hypot(width / 2, height / 2)
    return np.asarray(1.0 - 0.7 * radius**2, dtype=np.float32)


class Counting:
    """A clock that counts how often the session reads the monotonic time."""

    def __init__(self, inner: VirtualClock) -> None:
        self.inner = inner
        self.reads = 0

    def utc_ns(self) -> int:
        return self.inner.utc_ns()

    def monotonic_ns(self) -> int:
        self.reads += 1
        return self.inner.monotonic_ns()

    def sleep(self, seconds: float) -> None:
        self.inner.sleep(seconds)

    def status(self) -> Any:
        return self.inner.status()


@dataclass
class Rig:
    clock: VirtualClock
    flats: FlatLibrary
    darks: DarkLibrary
    camera: PanelCamera
    progress: list[fs.FlatProgress]
    said: list[str]
    stop: Callable[[], bool]
    options: fs.FlatSessionOptions

    def run(self, **changes: Any) -> fs.FlatSessionResult:
        options = replace(self.options, **{k: v for k, v in changes.items() if k in OPTION_KEYS})
        extra = {k: v for k, v in changes.items() if k not in OPTION_KEYS}
        return fs.record_flat(
            self.camera,
            self.flats,
            self.darks,
            PROFILE,
            extra.pop("clock", self.clock),
            options,
            say=self.said.append,
            progress=self.progress.append,
            should_stop=self.stop,
            **extra,
        )

    def ser_files(self) -> list[str]:
        folder = self.flats.session.directory
        return sorted(p.name for p in folder.glob("*.ser")) if folder.exists() else []


OPTION_KEYS = {
    "frames",
    "target_fraction",
    "set_number",
    "start_exposure_s",
    "max_exposure_s",
    "max_iterations",
    "level_tolerance",
    "min_level_fraction",
    "drift_percent",
    "gain",
    "mode",
}


def make_rig(tmp_path: Path, *, library: bool = True, **camera: Any) -> Rig:
    clock = VirtualClock(START)
    truth_image = camera.pop("truth", None)
    return Rig(
        clock=clock,
        flats=FlatLibrary(tmp_path / "calibration" / "flats"),
        darks=(
            fx.make_library(tmp_path / "calibration")
            if library
            else DarkLibrary(tmp_path / "empty" / "darks")
        ),
        camera=PanelCamera(clock, truth() if truth_image is None else truth_image, **camera),
        progress=[],
        said=[],
        stop=lambda: False,
        options=fs.FlatSessionOptions(frames=FRAMES, make=MAKE),
    )


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return make_rig(tmp_path)


# --- The search ---------------------------------------------------------------------------------


class TestTheSearch:
    def test_a_steady_light_is_found_in_two_tries(self, rig: Rig) -> None:
        result = rig.run()
        # the start is 20 ms and gives 25 to 26%, so the second try is about twice as long
        assert rig.camera.exposures_s[:2] == pytest.approx([0.02, 0.039], rel=0.05)
        assert result.exposure_s == pytest.approx(0.039, rel=0.05)
        assert result.level_fraction == pytest.approx(0.5, abs=0.02)
        assert len(rig.camera.exposures_s) == 2 + FRAMES
        assert result.frames_taken == FRAMES

    def test_the_exposure_follows_the_target(self, tmp_path: Path) -> None:
        for target, expected_s in ((0.3, 0.024), (0.7, 0.056)):
            made = make_rig(tmp_path / f"t{target}")
            result = made.run(target_fraction=target)
            assert result.exposure_s == pytest.approx(expected_s, rel=0.06)
            assert result.level_fraction == pytest.approx(target, rel=0.08)

    def test_a_dim_light_needs_a_long_exposure_and_a_bright_one_a_short_exposure(
        self, tmp_path: Path
    ) -> None:
        dim = make_rig(tmp_path / "dim", rate_dn_per_s=20_000.0).run()
        assert dim.exposure_s == pytest.approx(0.41, rel=0.08)
        bright = make_rig(tmp_path / "bright", rate_dn_per_s=5_000_000.0).run()
        assert bright.exposure_s == pytest.approx(0.00164, rel=0.1)
        assert bright.level_fraction == pytest.approx(0.5, abs=0.05)

    def test_a_saturated_try_does_not_fool_the_search(self, tmp_path: Path) -> None:
        made = make_rig(tmp_path, rate_dn_per_s=5_000_000.0)
        made.run()
        tries = [p.exposure_s for p in made.progress if p.exposure_s and p.phase == "exposure"]
        assert tries[0] == pytest.approx(0.02)
        assert all(a > b for a, b in itertools.pairwise(tries))  # shorter every time
        assert len(tries) <= 4  # the clipped frame cuts the exposure by 8 and the next one lands

    def test_the_search_reports_every_try_in_words(self, rig: Rig) -> None:
        rig.run()
        tries = [p for p in rig.progress if p.phase == "exposure"]
        assert [p.step for p in tries] == [1, 2]
        assert tries[0].steps == 8
        assert re.fullmatch(
            r"Try 1 of 8: 20 ms gives 2[56] % of full scale \(target 50 %\)\.", tries[0].message
        )
        assert tries[0].exposure_s == pytest.approx(0.02)
        assert tries[0].level_fraction == pytest.approx(0.25, abs=0.02)
        assert tries[1].level_fraction == pytest.approx(0.5, abs=0.03)

    def test_the_tries_use_snapshots_of_the_survey_mode_and_gain(self, rig: Rig) -> None:
        rig.run()
        configure = rig.camera.configures[0]
        assert (configure.mode, configure.gain, configure.kind.value) == ("bin2", 120, "snapshot")
        assert configure.roi is None
        assert configure.pixel_format.value == 16
        assert rig.camera.configures[-1].exposure_us == pytest.approx(39_000, rel=0.05)

    def test_a_light_that_is_too_weak_ends_the_session_with_numbers(self, tmp_path: Path) -> None:
        made = make_rig(tmp_path, rate_dn_per_s=1_000.0)
        with pytest.raises(fs.FlatSessionError) as error:
            made.run()
        assert str(error.value) == (
            "Not enough light: the frame reaches 6 % of full scale at the longest exposure of "
            "1 s. Use a brighter source, or hold it closer to the lens."
        )
        assert made.camera.exposures_s[-1] == pytest.approx(1.0)
        assert made.ser_files() == []

    def test_the_longest_exposure_is_a_setting(self, tmp_path: Path) -> None:
        made = make_rig(tmp_path, rate_dn_per_s=1_000.0)
        with pytest.raises(fs.FlatSessionError, match="longest exposure of 500 ms"):
            made.run(max_exposure_s=0.5)
        assert max(made.camera.exposures_s) == pytest.approx(0.5)

    def test_a_camera_in_the_dark_says_that_there_is_not_enough_light(self, tmp_path: Path) -> None:
        made = make_rig(tmp_path, rate_dn_per_s=0.0)
        with pytest.raises(fs.FlatSessionError, match=r"Not enough light: .* 0 % of full scale"):
            made.run()

    def test_a_light_that_saturates_the_shortest_exposure_is_too_much(self, tmp_path: Path) -> None:
        made = make_rig(tmp_path, rate_dn_per_s=5e10)
        with pytest.raises(fs.FlatSessionError) as error:
            made.run()
        assert str(error.value) == (
            "Too much light: the frame saturates at the shortest exposure of 32 µs. "
            "Dim the source, or put a layer of cloth between it and the lens."
        )
        assert min(made.camera.exposures_s) == pytest.approx(32e-6)

    def test_a_level_that_the_exposure_cannot_move_is_taken_with_a_note(
        self, tmp_path: Path
    ) -> None:
        made = make_rig(tmp_path, light=lambda index, exposure_s: 0.4 * FULL_SCALE)
        result = made.run()
        assert result.level_fraction == pytest.approx(0.4, abs=0.02)
        assert any("not the target of 50 %" in note for note in result.warnings)
        assert len([p for p in made.progress if p.phase == "exposure"]) == 8

    def test_a_light_that_jumps_about_is_too_unsteady(self, tmp_path: Path) -> None:
        def jumping(index: int, exposure_s: float) -> float:
            return 1.5 * FULL_SCALE if index % 2 == 0 else 0.01 * FULL_SCALE

        made = make_rig(tmp_path, light=jumping)
        with pytest.raises(fs.FlatSessionError, match="too unsteady to find an exposure"):
            made.run()

    def test_a_bright_light_that_does_not_saturate_is_too_much_at_the_shortest_exposure(
        self, tmp_path: Path
    ) -> None:
        made = make_rig(tmp_path, light=lambda index, exposure_s: 0.86 * FULL_SCALE)
        # 86% above the bias is over the 80% that `make_flat` takes, and no exposure can help
        with pytest.raises(fs.FlatSessionError) as error:
            made.run(max_iterations=20)
        assert str(error.value).startswith("Too much light: the frame reaches 8")
        assert "at the shortest exposure of 32 µs" in str(error.value)

    def test_the_profile_limits_the_exposure(self, tmp_path: Path) -> None:
        low_us, high_us = PROFILE.limits.exposure_us_range
        assert low_us == 32
        made = make_rig(tmp_path, rate_dn_per_s=0.001)
        with pytest.raises(fs.FlatSessionError, match="longest exposure of 2000 s"):
            made.run(max_exposure_s=1e6)  # a setting above the limit of the camera
        assert max(made.camera.exposures_s) * 1e6 == high_us


# --- The frames and the flat


class TestTheFlat:
    def test_the_flat_is_a_pending_flat_of_the_library_with_its_files(self, rig: Rig) -> None:
        result = rig.run()
        (entry,) = rig.flats.entries()
        assert entry == result.entry
        assert entry.pending is True
        assert rig.flats.active_version() is None  # nothing changes for the survey yet
        assert rig.flats.load(entry.version).version == entry.version
        jpeg = rig.flats.jpeg(entry.version)
        assert jpeg is not None
        assert jpeg.startswith(b"\xff\xd8\xff")

    def test_the_report_has_the_numbers_of_the_session_and_the_lens(self, rig: Rig) -> None:
        result = rig.run()
        report = dict(result.entry.report)
        assert (report["mode"], report["gain"]) == ("bin2", 120)
        assert (report["width_px"], report["height_px"]) == (256, 176)
        assert report["exposure_s"] == pytest.approx(0.039, rel=0.05)
        assert report["sensor_temperature_c"] == 25.0
        assert report["target_fraction"] == 0.5
        assert report["second_set"] is False
        assert report["bias"]["source"] == "library"
        assert report["t_utc"].startswith("2026-10-04T20:")
        (first,) = report["sets"]
        assert (first["frames"], first["used"]) == (FRAMES, FRAMES)
        assert first["level_fraction"] == pytest.approx(0.5, abs=0.03)
        corner = report["vignetting"]["corner_percent"]
        assert -12.0 < corner < -8.0  # the lens of the owner
        json.dumps(report, allow_nan=False)

    def test_the_bias_comes_from_the_library_at_the_temperature_of_the_frames(
        self, tmp_path: Path
    ) -> None:
        made = make_rig(tmp_path, temperature_c=25.0)
        result = made.run()
        note = result.make.bias.note
        assert "interpolated to 25.0 C" in note
        assert result.make.bias.level == pytest.approx(131.1, abs=0.1)

    def test_a_camera_without_a_temperature_takes_the_mean_of_the_sets(
        self, tmp_path: Path
    ) -> None:
        made = make_rig(tmp_path, temperature_c=None)
        result = made.run()
        assert result.temperature_c is None
        assert result.entry.report["sensor_temperature_c"] is None
        assert "the mean of the sets" in result.make.bias.note

    def test_the_frames_of_the_set_stay_in_the_session_folder_for_a_second_set(
        self, rig: Rig
    ) -> None:
        result = rig.run()
        assert rig.ser_files() == ["set1.ser"]
        with SerFile(rig.flats.session.ser_path(1)) as recording:
            assert recording.frame_count == FRAMES
            assert recording.pixel_depth == 14  # native counts of the ADC
            frame = recording.frame(0)
            assert frame.shape == SHAPE
            assert int(frame.max()) <= 16383
            middle = float(np.median(frame[44:132, 64:192]))
            assert middle == pytest.approx(131.0 + 0.5 * FULL_SCALE, rel=0.05)
            assert recording.timestamps_utc_ns() is not None
        session = rig.flats.session.load(rig.clock.utc_ns())
        assert session is not None
        assert session.version == result.entry.version
        assert (session.frames, session.mode, session.gain) == (FRAMES, "bin2", 120)
        assert session.exposure_s == pytest.approx(0.039, rel=0.05)

    def test_the_session_says_what_it_did_in_order(self, rig: Rig) -> None:
        rig.run()
        phases = [p.phase for p in rig.progress]
        assert phases[0] == "setup"
        assert phases[-1] == "done"
        order = [
            phase for index, phase in enumerate(phases) if index == 0 or phases[index - 1] != phase
        ]
        assert order == ["setup", "exposure", "capture", "build", "done"]
        capture = [p for p in rig.progress if p.phase == "capture"]
        assert [p.step for p in capture] == list(range(FRAMES + 1))
        assert capture[1].message == "Frame 1 of 16: 50 % of full scale."
        assert capture[-1].step == capture[-1].steps == FRAMES
        assert rig.progress[-1].message.startswith("Added flat-")
        assert rig.said[0] == "Setting up the camera and the library."

    def test_a_steady_light_gives_no_warning_but_the_one_about_the_tilt_of_one_set(
        self, rig: Rig
    ) -> None:
        result = rig.run()
        assert len(result.warnings) == 1
        assert "second set with the source turned by 180 degrees" in result.warnings[0]
        assert "--" not in result.warnings[0]
        assert all(not p.warnings for p in rig.progress)

    def test_the_summary_data_name_no_path(self, rig: Rig, tmp_path: Path) -> None:
        result = rig.run()
        text = json.dumps(dict(result.entry.report)) + " ".join(rig.said)
        assert str(tmp_path) not in text
        assert "\\\\" not in text


# --- Watching the light ---------------------------------------------------------------------------


class TestWatching:
    def test_a_light_that_drifts_gives_a_warning_and_the_flat_drops_the_frames(
        self, tmp_path: Path
    ) -> None:
        def drifting(index: int, exposure_s: float) -> float:
            steps = max(index - 2, 0)  # the search takes two frames
            return 204_800.0 * exposure_s * (1.0 + 0.012 * steps)

        made = make_rig(tmp_path, light=drifting)
        result = made.run()
        assert any("The light drifts" in note for note in result.warnings)
        (first,) = result.entry.report["sets"]
        assert first["used"] < first["frames"]  # the flat drops what drifted more than 5%
        assert "flicker" in first["dropped"]
        drift_seen = [p for p in made.progress if any("drifts" in w for w in p.warnings)]
        assert drift_seen  # the page could show it while the frames came

    def test_the_note_says_how_far_the_frame_is_from_the_median(self, tmp_path: Path) -> None:
        def one_bad_frame(index: int, exposure_s: float) -> float:
            return 204_800.0 * exposure_s * (1.3 if index == 8 else 1.0)

        made = make_rig(tmp_path, light=one_bad_frame)
        result = made.run()
        note = next(n for n in result.warnings if "drifts" in n)
        assert note.startswith("The light drifts: a frame is 3")
        assert note.endswith("% above the median level.")

    def test_a_light_that_saturates_the_frames_is_too_bright(self, tmp_path: Path) -> None:
        def brighter(index: int, exposure_s: float) -> float:
            return 204_800.0 * exposure_s * (1.0 if index < 2 else 3.0)

        made = make_rig(tmp_path, light=brighter)
        with pytest.raises(fs.FlatSessionError) as error:
            made.run()
        assert str(error.value).startswith("The flat could not be made. No frame of the frames")
        notes = made.progress[-1].warnings
        assert any(note.startswith("Too bright: the frame saturates") for note in notes)
        assert any("outside 20 % to 80 %" in note for note in notes)
        assert made.ser_files() == []  # the frames of a set that failed are gone
        assert made.flats.entries() == []

    def test_a_panel_that_does_not_light_the_corners_is_named(self, tmp_path: Path) -> None:
        made = make_rig(tmp_path, truth=dim_corners())
        result = made.run()
        note = next(n for n in result.warnings if "may not cover" in n)
        assert note.startswith("The light may not cover the whole lens: the corners get 4")
        assert note.endswith("% of the light of the middle.")


# --- The checks before the frames


class TestTheChecksBeforeTheFrames:
    def test_without_a_dark_model_the_session_says_what_to_do_before_it_touches_the_camera(
        self, tmp_path: Path
    ) -> None:
        made = make_rig(tmp_path, library=False)
        with pytest.raises(fs.FlatSessionError) as error:
            made.run()
        assert str(error.value) == "Record a dark set first (Dark page)."
        assert made.camera.configures == []
        assert made.camera.exposures_s == []

    def test_a_library_of_another_gain_has_no_model_for_this_one(self, tmp_path: Path) -> None:
        made = make_rig(tmp_path)
        with pytest.raises(fs.FlatSessionError, match="Record a dark set first"):
            made.run(gain=300)
        assert made.camera.exposures_s == []

    def test_a_readout_mode_that_the_profile_lacks_is_a_plain_error(self, rig: Rig) -> None:
        with pytest.raises(fs.FlatSessionError, match="The flat settings are not valid"):
            rig.run(mode="bin9")

    def test_a_disk_that_cannot_hold_the_frames_is_refused_with_the_numbers(self, rig: Rig) -> None:
        with pytest.raises(fs.FlatSessionError, match="not enough free disk space for the frames"):
            rig.run(free_bytes=lambda path: 1_000_000)
        assert rig.camera.exposures_s == []
        assert rig.ser_files() == []

    def test_memory_that_cannot_hold_the_combination_is_refused_before_the_frames(
        self, rig: Rig
    ) -> None:
        needed = SHAPE[0] * SHAPE[1] * fs.MEMORY_BYTES_PER_PIXEL
        with pytest.raises(fs.FlatSessionError, match="not enough free memory") as error:
            rig.run(available_memory=lambda: needed - 1)
        assert "the combination needs about" in str(error.value)
        assert "restart core" in str(error.value)
        assert rig.camera.exposures_s == []
        assert rig.ser_files() == []
        rig.run(available_memory=lambda: needed)  # exactly enough is enough

    def test_a_system_that_does_not_know_its_free_memory_is_not_checked(self, rig: Rig) -> None:
        rig.run(available_memory=lambda: None)

    def test_the_memory_sentence_names_gigabytes_with_one_digit(self, tmp_path: Path) -> None:
        # the frames of the survey mode (4144 x 2822) need about 0.6 GB, and 0.3 GB are free
        geometry = fm.Geometry((2822, 4144), 3.82, 14)
        with pytest.raises(fs.FlatSessionError) as error:
            fs._check_memory(geometry, lambda: 300_000_000)
        assert str(error.value) == (
            "There is not enough free memory to combine the frames: the combination needs about "
            "0.6 GB, and 0.3 GB are free. Stop other programs, or restart core, and try again."
        )

    def test_the_free_memory_of_this_machine_is_a_number_or_unknown(self) -> None:
        value = fs.memory_available_bytes()
        assert value is None or value > 0

    def test_the_reserve_counts_and_a_big_disk_is_fine(self, rig: Rig) -> None:
        need = FRAMES * (SHAPE[0] * SHAPE[1] * 2 + 8) + 4096
        with pytest.raises(fs.FlatSessionError, match="not enough free disk space"):
            rig.run(free_bytes=lambda path: need + 999, reserve_bytes=1_000)
        rig.run(free_bytes=lambda path: need + 1_000, reserve_bytes=1_000)

    def test_the_numbers_of_the_sentence_are_gigabytes_with_one_digit(self, tmp_path: Path) -> None:
        # a full frame of bin2 is 23.4 MB, so 64 frames need 1.5 GB
        flats = FlatLibrary(tmp_path / "flats")
        options = fs.FlatSessionOptions(frames=64)
        geometry = fm.Geometry((2822, 4144), 3.82, 14)
        with pytest.raises(fs.FlatSessionError) as error:
            fs._check_disk(
                flats, options, geometry, True, lambda path: 2_000_000_000, 1_000_000_000
            )
        assert str(error.value) == (
            "There is not enough free disk space for the frames: the set needs about 1.5 GB, "
            "and 1.0 GB are free beyond the reserve."
        )

    def test_a_storage_that_has_stopped_raw_capture_ends_the_session_and_drops_the_frames(
        self, rig: Rig
    ) -> None:
        calls = {"n": 0}

        def allowed() -> bool:
            calls["n"] += 1
            return calls["n"] <= 4

        with pytest.raises(fs.FlatSessionError, match="free disk space fell below the limit"):
            rig.run(capture_allowed=allowed)
        assert rig.ser_files() == []
        assert rig.flats.entries() == []

    def test_a_camera_that_gives_another_frame_size_is_named(self, rig: Rig) -> None:
        rig.camera.shape_override = (100, 200)
        with pytest.raises(fs.FlatSessionError, match="frames of 200 x 100 pixels"):
            rig.run()
        assert rig.ser_files() == []


# --- Stopping and faults


class TestStoppingAndFaults:
    def test_a_stop_before_the_first_frame_changes_nothing(self, rig: Rig) -> None:
        rig.stop = lambda: True
        with pytest.raises(fs.FlatAborted):
            rig.run()
        assert rig.camera.exposures_s == []
        assert rig.flats.entries() == []
        assert rig.ser_files() == []

    def test_a_stop_during_the_frames_deletes_the_partial_set(self, rig: Rig) -> None:
        rig.stop = lambda: len(rig.camera.exposures_s) >= 2 + 5
        with pytest.raises(fs.FlatAborted):
            rig.run()
        assert len(rig.camera.exposures_s) == 7
        assert rig.ser_files() == []
        assert rig.flats.entries() == []
        assert not rig.flats.session.directory.exists()

    def test_a_stop_during_the_combination_ends_it_and_keeps_nothing(self, rig: Rig) -> None:
        rig.stop = lambda: len(rig.camera.exposures_s) >= 2 + FRAMES
        with pytest.raises(fs.FlatAborted):
            rig.run()
        assert rig.ser_files() == []
        assert rig.flats.entries() == []

    def test_a_camera_error_goes_on_to_the_caller_and_the_frames_are_deleted(
        self, rig: Rig
    ) -> None:
        rig.camera.fail_at = 2 + 3
        with pytest.raises(CameraTimeoutError):
            rig.run()
        assert rig.ser_files() == []
        assert rig.flats.entries() == []

    def test_the_clock_is_read_for_every_frame_that_the_combination_reads(self, rig: Rig) -> None:
        counting = Counting(rig.clock)
        beats: list[int] = []

        def watch(progress: fs.FlatProgress) -> None:
            rig.progress.append(progress)
            beats.append(counting.reads)

        rig.camera.clock = rig.clock
        fs.record_flat(
            rig.camera,
            rig.flats,
            rig.darks,
            PROFILE,
            counting,
            rig.options,
            say=rig.said.append,
            progress=watch,
        )
        start = next(i for i, p in enumerate(rig.progress) if p.phase == "build")
        end = next(i for i, p in enumerate(rig.progress) if p.phase == "done")
        # `make_flat` reads each frame three times, and every read tells the clock
        assert beats[end] - beats[start] >= 3 * FRAMES


# --- The second set


class TestTheSecondSet:
    def first_set(self, rig: Rig) -> fs.FlatSessionResult:
        rig.camera.pattern = fx.source_pattern(SHAPE, fx.OWNER_SOURCE_GRADIENT)
        return rig.run()

    def turn_the_source(self, rig: Rig) -> None:
        gx, gy = fx.OWNER_SOURCE_GRADIENT
        rig.camera.pattern = fx.source_pattern(SHAPE, (-gx, -gy))

    def test_two_sets_make_one_flat_that_replaces_the_flat_of_the_first_set(self, rig: Rig) -> None:
        first = self.first_set(rig)
        self.turn_the_source(rig)
        rig.camera.seed += 1
        second = rig.run(set_number=2)
        assert second.entry.version != first.entry.version
        assert [e.version for e in rig.flats.entries()] == [second.entry.version]
        report = second.entry.report
        assert report["second_set"] is True
        assert report["source_turned"] is True
        assert [s["number"] for s in report["sets"]] == [1, 2]
        assert report["split"] is not None
        assert report["agreement"] is not None
        assert second.entry.pending is True

    def test_the_gradient_of_the_source_is_found_and_the_tilt_of_the_optics_stays(
        self, rig: Rig
    ) -> None:
        self.first_set(rig)
        self.turn_the_source(rig)
        rig.camera.seed += 1
        report = rig.run(set_number=2).entry.report
        optics, source = report["split"]["optics"], report["split"]["source"]
        assert optics["width_percent"] == pytest.approx(0.56, abs=0.15)
        assert source["width_percent"] == pytest.approx(-0.64, abs=0.15)

    def test_the_second_set_ends_the_session_and_deletes_every_frame(self, rig: Rig) -> None:
        self.first_set(rig)
        assert rig.ser_files() == ["set1.ser"]
        self.turn_the_source(rig)
        rig.run(set_number=2)
        assert rig.ser_files() == []
        assert not rig.flats.session.directory.exists()
        assert rig.flats.session.load(rig.clock.utc_ns()) is None

    def test_the_second_set_starts_its_search_where_the_first_one_ended(self, rig: Rig) -> None:
        first = self.first_set(rig)
        before = len(rig.camera.exposures_s)
        self.turn_the_source(rig)
        rig.run(set_number=2)
        assert rig.camera.exposures_s[before] == pytest.approx(first.exposure_s)
        tries = [p for p in rig.progress if p.phase == "exposure"]
        assert len(tries) == 2 + 1  # the first set took two tries, and the second one try

    def test_the_report_keeps_the_notes_of_both_sets(self, tmp_path: Path) -> None:
        def one_bad_frame(index: int, exposure_s: float) -> float:
            return 204_800.0 * exposure_s * (1.3 if index == 8 else 1.0)

        made = make_rig(tmp_path, light=one_bad_frame)
        made.run()
        made.camera.light = None
        second = made.run(set_number=2)
        assert any("drifts" in note for note in second.entry.report["warnings"])
        assert len(second.entry.report["warnings"]) == len(set(second.entry.report["warnings"]))

    def test_a_second_set_without_a_first_one_is_refused_in_words(self, rig: Rig) -> None:
        with pytest.raises(fs.FlatSessionError) as error:
            rig.run(set_number=2)
        assert str(error.value) == (
            "There is no first set to combine with. Take the first set, and take the second "
            "within 24 hours."
        )
        assert rig.camera.exposures_s == []

    def test_a_first_set_older_than_24_hours_is_gone(self, rig: Rig) -> None:
        rig.run()
        rig.clock.advance(24 * 3600 + 1)
        with pytest.raises(fs.FlatSessionError, match="no first set to combine with"):
            rig.run(set_number=2)
        assert rig.ser_files() == []  # the expired frames went with the check

    def test_a_first_set_within_24_hours_is_still_there(self, rig: Rig) -> None:
        rig.run()
        rig.clock.advance(23 * 3600)
        assert rig.run(set_number=2).entry.report["second_set"] is True

    def test_a_second_set_that_fails_leaves_the_first_set_for_another_try(self, rig: Rig) -> None:
        first = rig.run()
        rig.camera.fail_at = len(rig.camera.exposures_s) + 1 + 4
        with pytest.raises(CameraTimeoutError):
            rig.run(set_number=2)
        assert rig.ser_files() == ["set1.ser"]
        assert [e.version for e in rig.flats.entries()] == [first.entry.version]
        rig.camera.fail_at = None
        again = rig.run(set_number=2)
        assert again.entry.report["second_set"] is True

    def test_a_second_set_that_is_stopped_leaves_the_first_set_too(self, rig: Rig) -> None:
        first = rig.run()
        base = len(rig.camera.exposures_s)
        rig.stop = lambda: len(rig.camera.exposures_s) >= base + 1 + 3
        with pytest.raises(fs.FlatAborted):
            rig.run(set_number=2)
        assert rig.ser_files() == ["set1.ser"]
        assert [e.version for e in rig.flats.entries()] == [first.entry.version]

    def test_a_second_set_needs_the_camera_settings_of_the_first(self, tmp_path: Path) -> None:
        made = make_rig(tmp_path)
        darks = fx.make_library(tmp_path / "calibration", gain=90)  # a second gain in the library
        assert darks.model("bin2", 90) is not None
        made.run()
        with pytest.raises(fs.FlatSessionError, match="other camera settings"):
            made.run(set_number=2, gain=90)

    def test_a_new_first_set_replaces_the_session_and_keeps_both_flats(self, rig: Rig) -> None:
        first = rig.run()
        rig.camera.seed += 7
        second = rig.run()
        versions = {e.version for e in rig.flats.entries()}
        assert versions == {first.entry.version, second.entry.version}
        session = rig.flats.session.load(rig.clock.utc_ns())
        assert session is not None
        assert session.version == second.entry.version  # only the new flat can take a second set
        assert rig.ser_files() == ["set1.ser"]

    def test_the_disk_check_counts_the_frames_that_the_new_session_replaces(self, rig: Rig) -> None:
        rig.run()
        need = FRAMES * (SHAPE[0] * SHAPE[1] * 2 + 8) + 4096
        old = rig.flats.session.ser_path(1).stat().st_size
        assert old > 0
        # nothing is free beyond the frames that the old session holds
        rig.camera.seed += 1
        rig.run(free_bytes=lambda path: need - old)
        # a second set cannot count on them, because it needs the first set
        rig.camera.seed += 1
        with pytest.raises(fs.FlatSessionError, match="not enough free disk space"):
            rig.run(set_number=2, free_bytes=lambda path: need - 1)


# --- The pieces


class TestThePieces:
    def test_exposures_read_in_the_unit_that_fits(self) -> None:
        assert fs.format_exposure(1.0) == "1 s"
        assert fs.format_exposure(1.5) == "1.5 s"
        assert fs.format_exposure(0.04) == "40 ms"
        assert fs.format_exposure(0.00164) == "1.64 ms"
        assert fs.format_exposure(32e-6) == "32 µs"

    def test_a_frame_is_measured_in_the_middle_above_the_bias(self, rig: Rig) -> None:
        frame = rig.camera.take(0.04)
        level = fs.measure_frame(frame, full_scale=FULL_SCALE, bias_dn=131.0)
        assert level.fraction == pytest.approx(0.5, abs=0.02)
        assert level.raw_fraction == pytest.approx(0.5 + 131.0 / FULL_SCALE, abs=0.02)
        assert level.saturated_fraction == 0.0
        assert level.corner_ratio is not None
        assert 0.85 < level.corner_ratio < 0.95  # the corners of the lens lose 10%
        assert level.temperature_c == 25.0
        assert level.saturated is False

    def test_a_dark_frame_has_no_corner_ratio(self, rig: Rig) -> None:
        rig.camera.rate = 0.0
        level = fs.measure_frame(rig.camera.take(0.04), full_scale=FULL_SCALE, bias_dn=131.0)
        assert level.corner_ratio is None or abs(level.fraction) < 0.01

    def test_the_options_come_from_the_two_sections_of_the_survey_configuration(self) -> None:
        survey = SurveyConfig.model_validate(
            {
                "dark": {"mode": "bin2", "gain": 100, "doubling_c": 5.5},
                "flat": {"start_exposure_s": 0.5, "max_exposure_s": 3.0, "max_iterations": 5},
            }
        )
        options = fs.FlatSessionOptions.from_config(
            survey, frames=24, target_fraction=0.4, set_number=2
        )
        assert (options.mode, options.gain, options.doubling_c) == ("bin2", 100, 5.5)
        assert (options.start_exposure_s, options.max_exposure_s, options.max_iterations) == (
            0.5,
            3.0,
            5,
        )
        assert (options.frames, options.target_fraction, options.set_number) == (24, 0.4, 2)

    @pytest.mark.parametrize(
        "changes",
        [
            {"frames": 2},
            {"target_fraction": 0.0},
            {"target_fraction": 1.0},
            {"set_number": 3},
            {"start_exposure_s": 0.0},
            {"max_exposure_s": -1.0},
            {"max_iterations": 0},
            {"level_tolerance": 0.0},
            {"min_level_fraction": 1.0},
        ],
    )
    def test_the_options_refuse_nonsense(self, changes: dict[str, Any]) -> None:
        with pytest.raises(ValueError):  # noqa: PT011 (the messages differ for each option)
            fs.FlatSessionOptions(**changes)

    def test_the_default_options_follow_the_defaults_of_the_configuration(self) -> None:
        options = fs.FlatSessionOptions.from_config(
            SurveyConfig(), frames=32, target_fraction=0.5, set_number=1
        )
        defaults = fs.FlatSessionOptions()
        assert replace(options, make=None) == defaults
