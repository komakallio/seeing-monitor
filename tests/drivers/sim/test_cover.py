"""The cover of the simulated camera: a file that covers it while it exists.

`acquire` runs in its own process, so a person (or a test) cannot call the driver to put a lens cap
on it. With `cover_file`, the camera is covered while the file exists: a frame shows the sensor
alone, and a dark session accepts it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.drivers.sim import SimDriver, SimOptions
from seeingmon.drivers.sim.params import SimParams
from seeingmon.frames import Frame, StreamConfig, StreamKind
from seeingmon.services.acquire.factory import create_camera_driver
from seeingmon.survey.dark import DarkError, DarkLibrary, check_dark_frame
from seeingmon.survey.dark_session import DarkSessionOptions, run_dark_session
from seeingmon.survey.rawdata import native_u16
from tests.survey import simfx, synth

PROFILE = synth.cropped_profile(512, 384)
SIM_PARAMS = SimParams.from_profile(PROFILE, "bin2")
E_PER_ADU = PROFILE.e_per_adu("bin2", 120)
BIAS_DN = 120.0  # the offset of the sim, in the native counts of the sensor
SENSOR_C = 19.0  # the ambient 15 C of the options, and 4 C of self-heating
FAST = DarkSessionOptions(wait=False, frames=3, bias_frames=3)


def sim(clock: VirtualClock, cover: Path, **options: Any) -> SimDriver:
    driver = simfx.make_driver(PROFILE, SimOptions(seed=5, cover_file=str(cover), **options), clock)
    driver.open()
    return driver


def take(driver: SimDriver, exposure_s: float = 1.0) -> Frame:
    driver.configure(StreamConfig("bin2", round(exposure_s * 1e6), 120, kind=StreamKind.SNAPSHOT))
    driver.start()
    try:
        return driver.read_frame(timeout_s=exposure_s + 30.0)
    finally:
        driver.stop()


def is_dark(frame: Frame, exposure_s: float) -> bool:
    """The check of a dark session: the level above the bias, the noise, and no stars."""
    check = check_dark_frame(
        native_u16(frame),
        bias_dn=BIAS_DN,
        exposure_s=exposure_s,
        e_per_adu=E_PER_ADU,
        read_noise_e=PROFILE.read_noise_e("bin2", 120),
        expected_rate_e_per_s=SIM_PARAMS.dark_rate_e_per_s(SENSOR_C),
    )
    return check.ok


class TestTheFile:
    def test_without_the_file_the_camera_sees_the_sky(self, tmp_path: Path) -> None:
        driver = sim(VirtualClock(), tmp_path / "cover")
        assert not is_dark(take(driver, 30.0), 30.0)

    def test_with_the_file_a_frame_shows_the_sensor_alone(self, tmp_path: Path) -> None:
        cover = tmp_path / "cover"
        clock = VirtualClock()
        driver = sim(clock, cover)
        assert not is_dark(take(driver, 30.0), 30.0)
        cover.touch()
        clock.sleep(1.0)  # the next look at the file
        assert is_dark(take(driver, 30.0), 30.0)

    def test_a_covered_frame_has_no_star_in_the_truth(self, tmp_path: Path) -> None:
        cover = tmp_path / "cover"
        cover.touch()
        driver = sim(VirtualClock(), cover)
        take(driver)
        assert driver.truth.frames[-1].star_index == -1

    def test_removing_the_file_uncovers_the_camera_again(self, tmp_path: Path) -> None:
        cover = tmp_path / "cover"
        clock = VirtualClock()
        driver = sim(clock, cover)
        cover.touch()
        assert is_dark(take(driver, 30.0), 30.0)
        cover.unlink()
        clock.sleep(1.0)
        assert not is_dark(take(driver, 30.0), 30.0)

    def test_the_file_is_looked_at_only_every_so_many_seconds(self, tmp_path: Path) -> None:
        cover = tmp_path / "cover"
        clock = VirtualClock()
        driver = sim(clock, cover, cover_check_s=10.0)
        dark = [is_dark(take(driver, 1.0), 1.0)]  # the first read looks at the file: absent
        cover.touch()
        dark.extend(is_dark(take(driver, 1.0), 1.0) for _ in range(12))
        # The next look is ten clock seconds after the first, so the first reads still see the sky.
        assert dark[0] is False
        assert dark[1] is False
        assert dark[-1] is True
        switched = dark.index(True)
        assert 5 <= switched <= 12
        assert all(dark[switched:])

    def test_without_the_option_nothing_changes(self) -> None:
        clock = VirtualClock()
        driver = simfx.make_driver(PROFILE, SimOptions(seed=5), clock)
        driver.open()
        assert not is_dark(take(driver, 30.0), 30.0)
        assert driver.options.cover_file is None


class TestADarkSessionOnIt:
    def test_a_session_with_the_file_in_place_makes_a_set(self, tmp_path: Path) -> None:
        cover = tmp_path / "cover"
        cover.touch()
        clock = VirtualClock()
        driver = sim(clock, cover)
        driver.close()
        library = DarkLibrary(tmp_path / "darks")
        run_dark_session(driver, library, PROFILE, clock, FAST, say=lambda _: None)
        (dark_set,) = library.sets()
        assert dark_set.temperature_c == pytest.approx(SENSOR_C)

    def test_a_session_without_the_file_is_refused(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        driver = sim(clock, tmp_path / "cover")
        driver.close()
        with pytest.raises(DarkError, match="not dark"):
            run_dark_session(
                driver, DarkLibrary(tmp_path / "darks"), PROFILE, clock, FAST, say=lambda _: None
            )

    def test_a_session_that_waits_ends_when_the_file_appears(self, tmp_path: Path) -> None:
        cover = tmp_path / "cover"
        clock = VirtualClock()
        driver = sim(clock, cover)
        driver.close()
        options = DarkSessionOptions(frames=3, bias_frames=3, poll_s=1.0, stable_polls=2)
        sleeps: list[float] = []
        real_sleep = clock.sleep

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            if len(sleeps) == 3:
                cover.touch()  # the owner puts the cover on after the third wait
            real_sleep(seconds)

        clock.sleep = sleep  # type: ignore[method-assign]
        library = DarkLibrary(tmp_path / "darks")
        result = run_dark_session(driver, library, PROFILE, clock, options, say=lambda _: None)
        assert result.waited_s > 3.0
        assert len(library.sets()) == 1


class TestTheOptions:
    def test_the_options_come_from_a_table(self) -> None:
        options = SimOptions.from_mapping({"cover_file": "cover-here", "cover_check_s": 2.0})
        assert (options.cover_file, options.cover_check_s) == ("cover-here", 2.0)

    def test_the_defaults_have_no_file_and_check_twice_a_second(self) -> None:
        options = SimOptions()
        assert options.cover_file is None
        assert options.cover_check_s == 0.5

    def test_nonsense_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cover_check_s"):
            SimOptions(cover_check_s=-1.0)
        with pytest.raises(ValueError, match="cover_file"):
            SimOptions(cover_file="")

    def test_acquire_passes_the_option_to_the_sim_driver(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        plain = create_camera_driver(
            "sim", profile=PROFILE, clock=clock, options={"cover_file": str(tmp_path / "c")}
        )
        assert isinstance(plain, SimDriver)
        assert plain.options.cover_file == str(tmp_path / "c")
        pointed = create_camera_driver(
            "sim",
            profile=PROFILE,
            clock=clock,
            options={
                "cover_file": str(tmp_path / "c"),
                "polaris": "real",
                "pointing": {"t_ref_utc_ns": 1_800_000_000_000_000_000},
            },
        )
        assert isinstance(pointed, SimDriver)
        assert pointed.options.cover_file == str(tmp_path / "c")
