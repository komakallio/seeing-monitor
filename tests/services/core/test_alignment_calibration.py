"""The calibration of the previews: the dark level, the flat field, and the hot pixels.

The tests use a synthetic sky (`previewfx`): a `cos^4` vignetting, one dust shadow, and one hot
pixel, with a flat file and a dark library that describe the same sensor. The sky has no noise, so
a calibrated image can be asked to be flat to a fraction of a percent.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("scipy", reason="the dark library needs the fast extra")

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.frames import Frame
from seeingmon.profile import Profile, load_profile
from seeingmon.services.core.alignment import calibration
from seeingmon.services.core.alignment.calibration import FileFlatProvider, PreviewCalibrator
from seeingmon.services.core.alignment.preview import (
    block_mean,
    encode_jpeg,
    make_preview,
    shrink_factor,
    stretch_asinh,
)
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.dark import DarkLibrary
from seeingmon.survey.sky import SkyError, UnitFlat, load_flat

from .previewfx import (
    DUST,
    HOT,
    SHAPE,
    SKY_DN,
    ListedFlat,
    add_dark_set,
    make_survey_frame,
    sensitivity,
    window_of,
    write_flat,
)

PIL = pytest.importorskip("PIL.Image", reason="the preview needs Pillow")

PIXELS = 1_000  # the pixel limit that shrinks the 96 x 128 test frame by 4
FACTOR = 4
WEAK = (20, 20)  # a second, weaker hot pixel
LOGGER = calibration.__name__


@pytest.fixture(scope="module")
def profile() -> Profile:
    return load_profile("asi294mm-gs250")


@dataclass
class Sensor:
    """The files of one sensor: the flat, and the dark library with a set at 20 C."""

    config: SurveyConfig
    library: DarkLibrary
    flat_path: Path

    def calibrator(self, profile: Profile, **parts: Any) -> PreviewCalibrator:
        return PreviewCalibrator(self.config, profile, **parts)

    def without_flat(self) -> SurveyConfig:
        return self.config.model_copy(update={"flat_file": ""})


@pytest.fixture
def sensor(tmp_path: Path) -> Sensor:
    flat_path = write_flat(tmp_path / "flat.npy", sensitivity())
    calibration_dir = tmp_path / "calibration"
    library = DarkLibrary(calibration_dir / "darks")
    add_dark_set(library)
    config = SurveyConfig(flat_file=str(flat_path), calibration_dir=str(calibration_dir))
    return Sensor(config, library, flat_path)


def shrunk_and_calibrated(
    frame: Frame, calibrator: PreviewCalibrator, max_pixels: int = PIXELS
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """The shrunk image of a frame as `make_preview` makes it, before and after the calibration."""
    height, width = frame.data.shape
    factor = shrink_factor(height, width, max_pixels)
    raw = block_mean(frame.data, factor)
    step = calibrator.for_frame(frame)
    assert step is not None
    return raw, step(frame.data, raw.copy(), factor)


def corner_to_center(image: np.ndarray[Any, Any]) -> float:
    """The mean of the four corner patches over the mean of the central patch (3 x 3 blocks)."""
    rows, columns = image.shape
    corners = [image[:3, :3], image[:3, -3:], image[-3:, :3], image[-3:, -3:]]
    center = image[rows // 2 - 1 : rows // 2 + 2, columns // 2 - 1 : columns // 2 + 2]
    return float(np.mean([patch.mean() for patch in corners]) / center.mean())


def block_of(row: float, column: float) -> tuple[int, int]:
    return int(row) // FACTOR, int(column) // FACTOR


def patch(image: np.ndarray[Any, Any], block: tuple[int, int]) -> float:
    row, column = block
    return float(image[row - 1 : row + 2, column - 1 : column + 2].mean())


class TestTheCalibratedImage:
    def test_the_corners_match_the_center(self, sensor: Sensor, profile: Profile) -> None:
        raw, calibrated = shrunk_and_calibrated(make_survey_frame(), sensor.calibrator(profile))
        assert abs(corner_to_center(raw) - 1.0) > 0.20  # the vignetting darkens the corners
        assert abs(corner_to_center(calibrated) - 1.0) < 0.03

    def test_the_dust_shadow_is_gone(self, sensor: Sensor, profile: Profile) -> None:
        dusty = make_survey_frame()
        clean = make_survey_frame(sensitivity(dust=None))
        raw, calibrated = shrunk_and_calibrated(dusty, sensor.calibrator(profile))
        shadow = block_of(DUST[0], DUST[1])
        cost = 1.0 - patch(raw, shadow) / patch(block_mean(clean.data, FACTOR), shadow)
        assert cost > 0.05  # the shadow takes more than 5% of the raw patch
        assert abs(patch(calibrated, shadow) / np.median(calibrated) - 1.0) < 0.01

    def test_the_hot_pixel_is_replaced(self, sensor: Sensor, profile: Profile) -> None:
        raw, calibrated = shrunk_and_calibrated(make_survey_frame(), sensor.calibrator(profile))
        block = block_of(*HOT)
        assert raw[block] > 1.2 * np.median(raw)  # in the raw image it passes for a star
        assert abs(calibrated[block] / np.median(calibrated) - 1.0) < 0.01

    def test_the_whole_sky_is_flat(self, sensor: Sensor, profile: Profile) -> None:
        raw, calibrated = shrunk_and_calibrated(make_survey_frame(), sensor.calibrator(profile))
        assert (raw.max() - raw.min()) / np.median(raw) > 0.4
        assert (calibrated.max() - calibrated.min()) / np.median(calibrated) < 0.01

    def test_the_sky_level_is_the_light_above_the_dark_level(
        self, sensor: Sensor, profile: Profile
    ) -> None:
        _, calibrated = shrunk_and_calibrated(make_survey_frame(), sensor.calibrator(profile))
        # The flat has a median of 1, so the sky has the level of the pixels at the median flat.
        expected = SKY_DN * 4 * np.median(sensitivity())  # in frame units
        assert np.median(calibrated) == pytest.approx(expected, rel=0.002)

    def test_the_stretched_preview_comes_from_the_calibrated_image(
        self, sensor: Sensor, profile: Profile
    ) -> None:
        frame = make_survey_frame()
        calibrator = sensor.calibrator(profile)
        calibrated = make_preview(
            frame.data, max_pixels=PIXELS, quality=80, calibration=calibrator.for_frame(frame)
        )
        _, image = shrunk_and_calibrated(frame, calibrator)
        assert calibrated.jpeg == encode_jpeg(stretch_asinh(image), 80)
        plain = make_preview(frame.data, max_pixels=PIXELS, quality=80)
        assert calibrated.jpeg != plain.jpeg
        assert (calibrated.width_px, calibrated.height_px) == (plain.width_px, plain.height_px)
        assert PIL.open(io.BytesIO(calibrated.jpeg)).size == (32, 24)

    def test_the_frame_itself_stays_as_it_is(self, sensor: Sensor, profile: Profile) -> None:
        frame = make_survey_frame()
        before = frame.data.copy()
        shrunk_and_calibrated(frame, sensor.calibrator(profile))
        assert np.array_equal(frame.data, before)

    def test_a_window_of_the_sensor_gets_its_part_of_the_flat(
        self, sensor: Sensor, profile: Profile
    ) -> None:
        whole = make_survey_frame()
        window = window_of(whole, x=16, y=8, width=96, height=72)
        _, from_whole = shrunk_and_calibrated(whole, sensor.calibrator(profile))
        raw, from_window = shrunk_and_calibrated(window, sensor.calibrator(profile))
        assert (from_window.max() - from_window.min()) / np.median(from_window) < 0.01
        assert abs(raw.max() / raw.min() - 1.0) > 0.15  # the raw window still shows the lens
        assert np.median(from_window) == pytest.approx(np.median(from_whole), rel=0.002)
        hot = block_of(HOT[0] - 8, HOT[1] - 16)  # the hot pixel sits at another place in the window
        assert abs(from_window[hot] / np.median(from_window) - 1.0) < 0.01

    def test_an_8_bit_frame_calibrates_too(self, sensor: Sensor, profile: Profile) -> None:
        # The frame holds the top 8 bits of the 14, so it needs a bright sky and a bright pixel.
        frame = make_survey_frame(sky_dn=6_000.0, hot={HOT: 6_000.0}, dtype=np.uint8)
        assert frame.data.dtype == np.uint8
        raw, calibrated = shrunk_and_calibrated(frame, sensor.calibrator(profile))
        assert abs(corner_to_center(raw) - 1.0) > 0.20
        assert abs(corner_to_center(calibrated) - 1.0) < 0.03
        block = block_of(*HOT)
        assert raw[block] > 1.04 * np.median(raw)
        assert abs(calibrated[block] / np.median(calibrated) - 1.0) < 0.03

    def test_a_provider_takes_the_place_of_the_flat_file(
        self, sensor: Sensor, profile: Profile
    ) -> None:
        no_dust = ListedFlat(sensitivity(dust=None), version="a flat without the dust")
        calibrator = sensor.calibrator(profile, flat_provider=lambda: no_dust)
        _, calibrated = shrunk_and_calibrated(make_survey_frame(), calibrator)
        assert abs(corner_to_center(calibrated) - 1.0) < 0.03
        shadow = patch(calibrated, block_of(DUST[0], DUST[1])) / np.median(calibrated)
        assert shadow < 0.95  # the provider's flat has no dust, so the shadow stays


class TestHotPixels:
    def test_without_a_flat_only_the_hot_pixels_change(
        self, sensor: Sensor, profile: Profile
    ) -> None:
        calibrator = PreviewCalibrator(sensor.without_flat(), profile)
        raw, calibrated = shrunk_and_calibrated(make_survey_frame(), calibrator)
        changed = np.argwhere(calibrated != raw)
        assert changed.tolist() == [list(block_of(*HOT))]
        assert abs(calibrated[block_of(*HOT)] / np.median(calibrated) - 1.0) < 0.15

    def test_a_short_exposure_replaces_only_the_hottest_pixels(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        library = DarkLibrary(tmp_path / "darks")
        add_dark_set(library, hot={HOT: 3000.0, WEAK: 40.0})
        config = SurveyConfig(calibration_dir=str(tmp_path))
        calibrator = PreviewCalibrator(config, profile)
        # Both pixels carry 800 counts in the frames, to see whether the calibrator takes them.
        hot = {HOT: 800.0, WEAK: 800.0}
        for exposure_s, replaced in ((30.0, {HOT, WEAK}), (0.5, {HOT})):
            frame = make_survey_frame(exposure_s=exposure_s, hot=hot)
            raw = block_mean(frame.data, FACTOR)
            step = calibrator.for_frame(frame)
            assert step is not None
            calibrated = step(frame.data, raw.copy(), FACTOR)
            for pixel in (HOT, WEAK):
                changed = bool(calibrated[block_of(*pixel)] != raw[block_of(*pixel)])
                assert changed is (pixel in replaced), (exposure_s, pixel)

    def test_a_hot_pixel_counts_for_the_temperature_of_the_frame(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        library = DarkLibrary(tmp_path / "darks")
        add_dark_set(library, hot={WEAK: 6.0}, noise_dn=0.0)  # 6 counts at 20 C
        calibrator = PreviewCalibrator(SurveyConfig(calibration_dir=str(tmp_path)), profile)
        for temperature_c, replaced in ((8.0, False), (20.0, False), (32.0, True)):
            frame = make_survey_frame(temperature_c=temperature_c, hot={WEAK: 800.0})
            raw = block_mean(frame.data, FACTOR)
            step = calibrator.for_frame(frame)
            assert step is not None
            calibrated = step(frame.data, raw.copy(), FACTOR)
            changed = bool(calibrated[block_of(*WEAK)] != raw[block_of(*WEAK)])
            assert changed is replaced, temperature_c  # two doublings make 24 counts of 6


class TestWhatPassesUnchanged:
    def test_a_frame_of_another_readout_mode(self, sensor: Sensor, profile: Profile) -> None:
        assert sensor.calibrator(profile).for_frame(make_survey_frame(mode="bin1")) is None

    def test_a_frame_without_a_sensor_temperature(self, sensor: Sensor, profile: Profile) -> None:
        assert sensor.calibrator(profile).for_frame(make_survey_frame(temperature_c=None)) is None

    def test_a_gain_that_the_library_has_no_set_for(self, sensor: Sensor, profile: Profile) -> None:
        assert sensor.calibrator(profile).for_frame(make_survey_frame(gain=300)) is None

    def test_a_flat_without_a_dark_model_stays_unused_and_says_why_once(
        self, tmp_path: Path, profile: Profile, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The bias would turn into the inverse of the flat, so the flat waits for a dark set.
        config = SurveyConfig(
            flat_file=str(write_flat(tmp_path / "flat.npy", sensitivity())),
            calibration_dir=str(tmp_path / "calibration"),
        )
        calibrator = PreviewCalibrator(config, profile)
        frame = make_survey_frame()
        with caplog.at_level(logging.INFO, logger=LOGGER):
            assert calibrator.for_frame(frame) is None  # the folder of the library does not exist
            library = DarkLibrary(tmp_path / "calibration" / "darks")
            assert calibrator.for_frame(frame) is None  # the folder is empty
        (note,) = [r for r in caplog.records if r.name == LOGGER]
        assert note.levelno == logging.INFO
        assert "no set for the readout mode bin2 at gain 120" in note.getMessage()
        add_dark_set(library)  # a new set takes effect with no restart
        raw, calibrated = shrunk_and_calibrated(frame, calibrator)
        assert abs(corner_to_center(raw) - corner_to_center(calibrated)) > 0.2

    def test_no_flat_and_no_dark_model_leave_the_preview_as_it_was(
        self, tmp_path: Path, profile: Profile, caplog: pytest.LogCaptureFixture
    ) -> None:
        calibrator = PreviewCalibrator(SurveyConfig(), profile)
        frame = make_survey_frame()
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert calibrator.for_frame(frame) is None
        assert [r for r in caplog.records if r.name == LOGGER] == []  # nothing to say
        old = make_preview(frame.data, max_pixels=PIXELS, quality=80)
        new = make_preview(
            frame.data, max_pixels=PIXELS, quality=80, calibration=calibrator.for_frame(frame)
        )
        assert new.jpeg == old.jpeg
        assert old.jpeg == encode_jpeg(stretch_asinh(block_mean(frame.data, FACTOR)), 80)

    def test_a_unit_flat_and_no_hot_pixel_change_nothing(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        library = DarkLibrary(tmp_path / "darks")
        add_dark_set(library, hot={HOT: 0.0})  # a master dark with no hot pixel
        calibrator = PreviewCalibrator(SurveyConfig(calibration_dir=str(tmp_path)), profile)
        assert calibrator.for_frame(make_survey_frame()) is None


class TestFailures:
    def warnings(self, caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
        return [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == LOGGER]

    def test_a_failing_provider_falls_back_and_logs_once(
        self, sensor: Sensor, profile: Profile, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken() -> Any:
            raise SkyError("the flat is broken")

        calibrator = sensor.calibrator(profile, flat_provider=broken)
        frame = make_survey_frame()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            steps = [calibrator.for_frame(frame) for _ in range(5)]
        assert steps == [None] * 5
        (record,) = self.warnings(caplog)
        assert "the flat is broken" in record.getMessage()
        assert "not calibrated" in record.getMessage()
        plain = make_preview(frame.data, max_pixels=PIXELS, quality=80)
        fallback = make_preview(
            frame.data, max_pixels=PIXELS, quality=80, calibration=calibrator.for_frame(frame)
        )
        assert fallback.jpeg == plain.jpeg

    def test_the_warning_comes_again_after_an_hour(
        self, sensor: Sensor, profile: Profile, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken() -> Any:
            raise SkyError("the flat is broken")

        clock = VirtualClock(1_800_000_000 * NS_PER_S)
        calibrator = sensor.calibrator(profile, flat_provider=broken, clock=clock)
        frame = make_survey_frame()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            calibrator.for_frame(frame)
            clock.sleep(3000.0)
            calibrator.for_frame(frame)
            assert len(self.warnings(caplog)) == 1
            clock.sleep(700.0)
            calibrator.for_frame(frame)
            assert len(self.warnings(caplog)) == 2

    def test_a_missing_flat_file_falls_back_and_logs_once(
        self, sensor: Sensor, profile: Profile, caplog: pytest.LogCaptureFixture
    ) -> None:
        os.remove(sensor.flat_path)
        calibrator = sensor.calibrator(profile)
        frame = make_survey_frame()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert [calibrator.for_frame(frame) for _ in range(3)] == [None] * 3
        (record,) = self.warnings(caplog)
        assert "cannot read the flat file" in record.getMessage()

    def test_a_flat_that_does_not_cover_the_frame_logs_once(
        self, tmp_path: Path, sensor: Sensor, profile: Profile, caplog: pytest.LogCaptureFixture
    ) -> None:
        write_flat(sensor.flat_path, sensitivity((64, 64)))
        calibrator = sensor.calibrator(profile)
        frame = make_survey_frame()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert [calibrator.for_frame(frame) for _ in range(3)] == [None] * 3
        (record,) = self.warnings(caplog)
        assert "does not fit inside the flat" in record.getMessage()

    def test_a_step_that_fails_gives_the_image_back_as_it_was(
        self, sensor: Sensor, profile: Profile, caplog: pytest.LogCaptureFixture
    ) -> None:
        frame = make_survey_frame()
        step = sensor.calibrator(profile).for_frame(frame)
        assert step is not None
        image = block_mean(frame.data, FACTOR)
        wrong = image[:-1].copy()  # a shape that does not belong to the frame
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            results = [step(frame.data, wrong, FACTOR) for _ in range(3)]
        assert all(result is wrong for result in results)
        assert np.array_equal(wrong, image[:-1])
        (record,) = self.warnings(caplog)
        assert "does not belong to the frame" in record.getMessage()

    def test_a_hot_pixel_table_that_cannot_be_read_falls_back_and_logs_once(
        self, sensor: Sensor, profile: Profile, caplog: pytest.LogCaptureFixture
    ) -> None:
        (path,) = sensor.library.directory.glob("dark-*.fits")
        data = path.read_bytes()
        path.write_bytes(data[: len(data) - 5000])  # cut the table off
        calibrator = sensor.calibrator(profile)
        frame = make_survey_frame()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert [calibrator.for_frame(frame) for _ in range(3)] == [None] * 3
        assert len(self.warnings(caplog)) == 1


class TestTheFlatFile:
    def test_an_empty_path_gives_a_unit_flat(self) -> None:
        assert isinstance(FileFlatProvider("")(), UnitFlat)

    def test_the_flat_reloads_when_the_file_changes(
        self, sensor: Sensor, profile: Profile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loads: list[str] = []

        def counting(path: str | Path) -> Any:
            loads.append(str(path))
            return load_flat(path)

        monkeypatch.setattr(f"{LOGGER}.load_flat", counting)
        calibrator = sensor.calibrator(profile)
        frame = make_survey_frame()
        _, first = shrunk_and_calibrated(frame, calibrator)
        _, again = shrunk_and_calibrated(frame, calibrator)
        assert len(loads) == 1  # the file did not change, so nothing read it again
        assert np.array_equal(first, again)
        assert abs(corner_to_center(first) - 1.0) < 0.03
        # A new flat with no vignetting replaces the file, with a later modification time.
        stamp = sensor.flat_path.stat().st_mtime_ns + 5 * NS_PER_S
        write_flat(sensor.flat_path, np.ones(SHAPE, dtype=np.float32))
        os.utime(sensor.flat_path, ns=(stamp, stamp))
        _, changed = shrunk_and_calibrated(frame, calibrator)
        assert len(loads) == 2
        assert abs(corner_to_center(changed) - 1.0) > 0.20  # the flat no longer takes it out
        # The same file again: the calibrator reads nothing.
        shrunk_and_calibrated(frame, calibrator)
        assert len(loads) == 2

    def test_a_file_that_cannot_be_read_is_not_read_again_until_it_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "flat.npy"
        path.write_bytes(b"this is not a NumPy file")
        loads: list[int] = []

        def counting(name: str | Path) -> Any:
            loads.append(1)
            return load_flat(name)

        monkeypatch.setattr(f"{LOGGER}.load_flat", counting)
        provider = FileFlatProvider(path)
        for _ in range(3):
            with pytest.raises(SkyError, match="cannot read the flat file"):
                provider()
        assert len(loads) == 1
        write_flat(path, sensitivity())
        stamp = path.stat().st_mtime_ns + 5 * NS_PER_S
        os.utime(path, ns=(stamp, stamp))
        assert provider().version.startswith("flat-")
        assert len(loads) == 2


class TestThreads:
    def test_threads_share_the_caches_and_agree(
        self, sensor: Sensor, profile: Profile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame = make_survey_frame()
        _, expected = shrunk_and_calibrated(frame, sensor.calibrator(profile))
        loads: list[int] = []
        shrinks: list[int] = []
        real_shrink = calibration._inverse_of_blocks

        def counting_load(path: str | Path) -> Any:
            loads.append(1)
            return load_flat(path)

        def counting_shrink(window: Any, factor: int) -> Any:
            shrinks.append(factor)
            return real_shrink(window, factor)

        monkeypatch.setattr(f"{LOGGER}.load_flat", counting_load)
        monkeypatch.setattr(f"{LOGGER}._inverse_of_blocks", counting_shrink)
        calibrator = sensor.calibrator(profile)
        barrier = threading.Barrier(8)
        results: list[np.ndarray[Any, Any]] = []
        errors: list[BaseException] = []

        def work() -> None:
            try:
                barrier.wait()
                for _ in range(5):
                    step = calibrator.for_frame(frame)
                    assert step is not None
                    results.append(step(frame.data, block_mean(frame.data, FACTOR), FACTOR))
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30.0)
        assert not errors
        assert len(results) == 40
        assert all(np.array_equal(result, expected) for result in results)
        assert loads == [1]  # one thread read the flat, and the others waited for it
        assert shrinks == [FACTOR]  # and one shrank it
