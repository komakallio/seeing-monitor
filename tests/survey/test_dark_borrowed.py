"""The dark session on a borrowed camera: progress, stopping, and the options of the config.

The camera that `core` lends has to be configured and read, and it must stay open. These tests lend
a sim camera through a small `DarkCamera` that counts what the session does with it, and they check
what the progress callback hears and what a stop leaves behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.drivers.base import CameraDriver
from seeingmon.drivers.sim import SimDriver, SimOptions
from seeingmon.frames import Frame, StreamConfig, StreamKind
from seeingmon.survey.config import DarkConfig
from seeingmon.survey.dark import DarkError, DarkLibrary
from seeingmon.survey.dark_session import (
    DarkAborted,
    DarkCamera,
    DarkProgress,
    DarkSessionOptions,
    DriverCamera,
    record_dark_set,
    run_dark_session,
)
from tests.survey import simfx, synth

PROFILE = synth.cropped_profile(512, 384)
FAST = DarkSessionOptions(wait=False, frames=3, bias_frames=3)


class Lent:
    """A camera on loan: it counts the calls, and it opens and closes nothing."""

    def __init__(self, driver: SimDriver) -> None:
        self._driver = driver
        self.configures: list[StreamConfig] = []
        self.takes: list[float] = []
        self.temperature_reads = 0

    def configure(self, config: StreamConfig) -> None:
        self.configures.append(config)
        self._driver.configure(config)

    def take(self, exposure_s: float) -> Frame:
        self.takes.append(exposure_s)
        self._driver.start()
        try:
            return self._driver.read_frame(timeout_s=exposure_s + 30.0)
        finally:
            self._driver.stop()

    def read_temperature_c(self) -> float | None:
        self.temperature_reads += 1
        return self._driver.read_temperature_c()


def opened_covered(clock: VirtualClock, *, seed: int = 3) -> SimDriver:
    driver = simfx.make_driver(PROFILE, simfx.covered_options(ambient_c=12.0, seed=seed), clock)
    driver.open()
    return driver


def run(
    library: DarkLibrary,
    camera: DarkCamera,
    clock: VirtualClock,
    options: DarkSessionOptions = FAST,
    **keywords: object,
) -> tuple[list[DarkProgress], list[str]]:
    heard: list[DarkProgress] = []
    lines: list[str] = []
    record_dark_set(
        camera,
        library,
        PROFILE,
        clock,
        options,
        say=lines.append,
        progress=heard.append,
        **keywords,  # type: ignore[arg-type]
    )
    return heard, lines


class TestABorrowedCamera:
    def test_the_session_opens_and_closes_nothing(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        driver = opened_covered(clock)
        library = DarkLibrary(tmp_path / "darks")
        run(library, Lent(driver), clock)
        names = [name for name, _ in driver.calls]
        assert names.count("open") == 1  # the test's own call, and not the session's
        assert "close" not in names
        assert len(library.sets()) == 1

    def test_it_configures_a_snapshot_for_each_phase(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        camera = Lent(opened_covered(clock))
        run(DarkLibrary(tmp_path / "darks"), camera, clock)
        shortest_us = PROFILE.limits.exposure_us_range[0]
        assert [c.exposure_us for c in camera.configures] == [shortest_us, 30_000_000]
        assert all(c.kind is StreamKind.SNAPSHOT for c in camera.configures)
        assert (camera.configures[0].mode, camera.configures[0].gain) == ("bin2", 120)
        assert camera.takes == [shortest_us / 1e6] * 3 + [30.0] * 3

    def test_the_result_is_the_one_of_the_command_line_path(self, tmp_path: Path) -> None:
        clock_a, clock_b = VirtualClock(), VirtualClock()
        borrowed = opened_covered(clock_a)
        result = record_dark_set(
            Lent(borrowed), DarkLibrary(tmp_path / "a"), PROFILE, clock_a, FAST, say=lambda _: None
        )
        driver = simfx.make_driver(PROFILE, simfx.covered_options(ambient_c=12.0, seed=3), clock_b)
        direct = run_dark_session(
            driver, DarkLibrary(tmp_path / "b"), PROFILE, clock_b, FAST, say=lambda _: None
        )
        assert result.dark_set.rate_dn_per_s == pytest.approx(direct.dark_set.rate_dn_per_s)
        assert result.dark_set.temperature_c == pytest.approx(direct.dark_set.temperature_c)
        assert result.dark_set.n_hot_pixels == direct.dark_set.n_hot_pixels

    def test_a_camera_without_a_temperature_fails_with_a_plain_reason(self, tmp_path: Path) -> None:
        clock = VirtualClock()

        class NoTemperature(Lent):
            def take(self, exposure_s: float) -> Frame:
                from dataclasses import replace

                return replace(super().take(exposure_s), temperature_c=None)

            def read_temperature_c(self) -> float | None:
                return None

        library = DarkLibrary(tmp_path / "darks")
        with pytest.raises(DarkError, match="no sensor temperature"):
            run(library, NoTemperature(opened_covered(clock)), clock)
        assert library.sets() == ()


class TestProgress:
    def test_the_phases_come_in_order_and_each_step_counts_up(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        heard, _ = run(DarkLibrary(tmp_path / "darks"), Lent(opened_covered(clock)), clock)
        phases = [p.phase for p in heard]
        assert phases[0] == "bias"
        assert phases[-1] == "done"
        order = ["bias", "dark", "build", "done"]  # no cover phase without the wait
        assert [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]] == order
        for phase in ("bias", "dark"):
            steps = [p.step for p in heard if p.phase == phase]
            assert steps[0] == 0
            assert steps == sorted(steps)
            assert steps[-1] == 3
            assert {p.steps for p in heard if p.phase == phase} == {3}
        assert [p.step for p in heard if p.phase == "build"] == [0, 1]

    def test_a_frame_of_the_dark_phase_carries_its_check(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        heard, _ = run(DarkLibrary(tmp_path / "darks"), Lent(opened_covered(clock)), clock)
        dark = [p for p in heard if p.phase == "dark"]
        assert dark[0].check is None  # the phase begins before the first frame
        assert all(p.check is not None and p.check.ok for p in dark[1:])
        assert dark[1].message == "Dark frame 1 of 3."

    def test_every_message_is_a_sentence(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        heard, _ = run(DarkLibrary(tmp_path / "darks"), Lent(opened_covered(clock)), clock)
        assert all(p.message.endswith(".") and p.message[0].isupper() for p in heard)

    def test_the_cover_phase_reports_each_test_frame_and_its_reason(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        uncovered = simfx.make_driver(PROFILE, SimOptions(seed=5), clock)
        cover = simfx.make_driver(PROFILE, simfx.covered_options(ambient_c=15.0, seed=5), clock)
        wrapper = simfx.CoverableDriver(uncovered, cover, cover_after_reads=3 + 2)
        wrapper.open()
        options = DarkSessionOptions(frames=3, bias_frames=3, poll_s=1.0, stable_polls=2)
        heard, _ = run(
            DarkLibrary(tmp_path / "darks"),
            DriverCamera(wrapper),
            clock,
            options,
        )
        cover_steps = [p for p in heard if p.phase == "cover"]
        assert cover_steps[0].message == "Cover the camera now. Waiting for a dark frame."
        polls = cover_steps[1:]
        lit = [p for p in polls if p.check is not None and not p.check.ok]
        dark = [p for p in polls if p.check is not None and p.check.ok]
        assert len(lit) == 2  # two lit frames,
        assert len(dark) == 2  # and then two dark frames in a row
        assert all("is not dark yet" in p.message for p in lit)
        assert all(p.step == 0 for p in lit)
        assert all(p.check is not None and p.check.reason for p in lit)
        assert [p.step for p in dark] == [1, 2]
        assert dark[-1].message == "The camera is dark."
        assert {p.steps for p in cover_steps} == {2}


class TestStopping:
    def test_a_stop_before_the_first_frame_takes_no_frame(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        camera = Lent(opened_covered(clock))
        library = DarkLibrary(tmp_path / "darks")
        with pytest.raises(DarkAborted):
            run(library, camera, clock, should_stop=lambda: True)
        assert camera.takes == []
        assert library.sets() == ()

    def test_a_stop_between_two_dark_frames_leaves_the_library_alone(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        camera = Lent(opened_covered(clock))
        library = DarkLibrary(tmp_path / "darks")
        with pytest.raises(DarkAborted):
            run(library, camera, clock, should_stop=lambda: len(camera.takes) >= 3 + 2)
        assert camera.takes[3:] == [30.0, 30.0]  # two dark frames came before the stop
        assert library.sets() == ()
        assert not (tmp_path / "darks").exists()

    def test_the_question_is_asked_before_every_frame(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        camera = Lent(opened_covered(clock))
        asked: list[int] = []

        def should_stop() -> bool:
            asked.append(len(camera.takes))
            return False

        run(DarkLibrary(tmp_path / "darks"), camera, clock, should_stop=should_stop)
        # Before each of the six frames, and once more before the set is written.
        assert asked == [0, 1, 2, 3, 4, 5, 6]

    def test_a_stop_during_the_wait_for_the_cover_ends_the_wait(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        uncovered = simfx.make_driver(PROFILE, SimOptions(seed=5), clock)
        uncovered.open()
        library = DarkLibrary(tmp_path / "darks")
        options = DarkSessionOptions(frames=3, bias_frames=3, poll_s=5.0, wait_timeout_s=600.0)
        started = clock.monotonic_ns()
        with pytest.raises(DarkAborted):
            run(
                library,
                Lent(uncovered),
                clock,
                options,
                should_stop=lambda: (clock.monotonic_ns() - started) / 1e9 > 100.0,
            )
        waited_s = (clock.monotonic_ns() - started) / 1e9
        assert waited_s < 130.0  # the wait looked every quarter of a second, not once in five
        assert library.sets() == ()

    def test_the_wait_sleeps_in_one_piece_when_nobody_can_stop_it(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        sleeps: list[float] = []
        real_sleep = clock.sleep

        def recording(seconds: float) -> None:
            sleeps.append(seconds)
            real_sleep(seconds)

        clock.sleep = recording  # type: ignore[method-assign]
        uncovered = simfx.make_driver(PROFILE, SimOptions(seed=5), clock)
        cover = simfx.make_driver(PROFILE, simfx.covered_options(ambient_c=15.0, seed=5), clock)
        wrapper = simfx.CoverableDriver(uncovered, cover, cover_after_reads=3 + 1)
        wrapper.open()
        options = DarkSessionOptions(frames=3, bias_frames=3, poll_s=5.0, stable_polls=1)
        run_dark_session(
            wrapper,
            DarkLibrary(tmp_path / "darks"),
            PROFILE,
            clock,
            options,
            say=lambda _: None,
        )
        assert 5.0 in sleeps  # a whole poll interval, as `seeingmon dark` has always waited
        assert 0.25 not in sleeps  # and no quarter-second slices

    def test_the_command_line_path_closes_the_driver_after_a_failure(self, tmp_path: Path) -> None:
        clock = VirtualClock()
        driver: CameraDriver = simfx.make_driver(PROFILE, SimOptions(seed=5), clock)  # uncovered
        with pytest.raises(DarkError, match="not dark"):
            run_dark_session(
                driver,
                DarkLibrary(tmp_path / "darks"),
                PROFILE,
                clock,
                DarkSessionOptions(wait=False, frames=3, bias_frames=3),
                say=lambda _: None,
            )
        assert [name for name, _ in driver.calls][-1] == "close"  # type: ignore[attr-defined]


class TestTheOptionsOfTheConfig:
    def test_the_defaults_of_the_config_are_the_defaults_of_the_options(self) -> None:
        assert DarkSessionOptions.from_config(DarkConfig()) == DarkSessionOptions()

    def test_a_value_that_you_pass_replaces_the_configured_one(self) -> None:
        cfg = DarkConfig(frames=12, exposure_s=20.0, poll_s=2.0)
        options = DarkSessionOptions.from_config(
            cfg, mode="bin1", gain=60, exposure_s=10.0, frames=4, wait=False, wait_timeout_s=30.0
        )
        assert (options.mode, options.gain, options.exposure_s) == ("bin1", 60, 10.0)
        assert (options.frames, options.bias_frames, options.poll_s) == (4, 9, 2.0)
        assert (options.wait, options.wait_timeout_s) == (False, 30.0)
        kept = DarkSessionOptions.from_config(cfg)
        assert (kept.frames, kept.exposure_s, kept.wait) == (12, 20.0, True)

    def test_an_empty_mode_takes_the_configured_one(self) -> None:
        assert DarkSessionOptions.from_config(DarkConfig(mode="bin1"), mode="").mode == "bin1"

    def test_the_check_limits_follow_the_config(self) -> None:
        cfg = DarkConfig(rate_factor=2.0, min_rate_e_per_s=0.7, noise_factor=1.2)
        check = DarkSessionOptions.from_config(cfg).check
        assert (check.rate_factor, check.min_rate_e_per_s, check.noise_factor) == (2.0, 0.7, 1.2)

    def test_a_value_that_the_options_refuse_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            DarkSessionOptions.from_config(DarkConfig(), frames=2)
