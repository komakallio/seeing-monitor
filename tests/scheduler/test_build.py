"""Building the scheduler from the configuration, and running it in real time."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from seeingmon.clock import NS_PER_S, ScaledClock, iso_to_utc_ns
from seeingmon.config import ConfigError, load_config
from seeingmon.profile import ProfileError
from seeingmon.records import EventRecord
from seeingmon.scheduler import Scheduler, SchedulerConfig, build_scheduler
from seeingmon.scheduler.config import FastConfig, SurveyConfig, WatchConfig
from seeingmon.survey.config import TwilightConfig
from tests.scheduler.scenario import PROFILE, TEST_CONFIG, World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")


def collaborators(world: World) -> dict[str, Any]:
    """The keyword arguments that `Scheduler` and `build_scheduler` take besides the settings."""
    return {
        "driver": world.camera,
        "fast": world.fast,
        "survey": world.survey,
        "pointing": world.pointing,
        "records": world.writer,
        "metrics": world.writer,
        "clock": world.clock,
    }


def events(world: World) -> list[EventRecord]:
    return world.events()


class TestBuildFromTheConfiguration:
    @staticmethod
    def write_local(tmp_path: Path, text: str) -> Path:
        path = tmp_path / "config.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_it_reads_the_station_the_site_the_profile_and_the_scheduler_table(
        self, tmp_path: Path
    ) -> None:
        local = self.write_local(
            tmp_path,
            'station_id = "synthetic-station"\n'
            "[site]\nlatitude_deg = 55.0\nlongitude_deg = 0.0\n"  # a synthetic site
            "[scheduler.fast]\nwindow_s = 60.0\n"
            "[scheduler.survey]\ncadence_s = 90.0\n",
        )
        config = load_config(local_file=local, env={})
        world = World(start_utc_ns=NIGHT)
        scheduler = build_scheduler(config, **collaborators(world))
        scheduler.step()
        scheduler.close()
        (start,) = world.events("scheduler.start")
        assert start.station_id == "synthetic-station"
        assert (start.detail or {}) == {"site_configured": True, "profile": "asi294mm-gs250"}
        assert scheduler.config.fast.window_s == 60.0
        assert scheduler.config.survey.cadence_s == 90.0
        assert scheduler.config.fast.analysis_window_s == 60.0  # the other keys keep the defaults
        assert scheduler.status().sun_elevation_deg is not None  # the site gives the Sun

    def test_without_a_site_table_the_scheduler_runs_without_the_sun(self, tmp_path: Path) -> None:
        local = self.write_local(tmp_path, 'station_id = "synthetic-station"\n')
        world = World(start_utc_ns=NIGHT)
        scheduler = build_scheduler(load_config(local_file=local, env={}), **collaborators(world))
        scheduler.step()
        kinds = [e.kind for e in events(world)]
        # The sky is dark, so the first brightness frame also sends the scheduler to `auto`.
        assert kinds == ["scheduler.start", "scheduler.no_site", "scheduler.state_change"]
        assert scheduler.status().sun_elevation_deg is None
        scheduler.close()

    def test_environment_variables_override_the_scheduler_table(self, tmp_path: Path) -> None:
        local = self.write_local(tmp_path, 'station_id = "synthetic-station"\n')
        env = {"SEEINGMON_SCHEDULER__FAST__WINDOW_S": "60"}
        config = load_config(local_file=local, env=env)
        world = World(start_utc_ns=NIGHT)
        scheduler = build_scheduler(config, **collaborators(world))
        assert scheduler.config.fast.window_s == 60.0
        scheduler.close()

    @pytest.mark.parametrize(("twilight", "first_long_s"), [("", 1.0), ("8.0", 8.0)])
    def test_it_passes_the_twilight_table_of_the_survey_to_the_scheduler(
        self, tmp_path: Path, twilight: str, first_long_s: float
    ) -> None:
        """`[survey.twilight] min_exposure_s` sets the first long survey frame after a start.

        The scheduler table takes the slow stream of the scenario (`TEST_CONFIG`), so that the run
        takes little time.
        """
        text = (
            'station_id = "synthetic-station"\n'
            "[scheduler.fast]\nexposure_us = 2000000\nroi_arcmin = 1.0\n"
            "roi_edge_margin_px = 4.0\nmissing_star_frames = 10\n"
            "target_background_fraction = 0.0\n"
            "[scheduler.search]\nburst_frames = 3\n"
            "[scheduler.loop]\nmax_sleep_s = 5.0\n"
        )
        if twilight:
            text += f"[survey.twilight]\nmin_exposure_s = {twilight}\n"
        local = self.write_local(tmp_path, text)
        world = World(start_utc_ns=NIGHT)
        scheduler = build_scheduler(load_config(local_file=local, env={}), **collaborators(world))
        assert scheduler.config.fast == TEST_CONFIG.fast
        scheduler.run_until(NIGHT + 400 * NS_PER_S)
        scheduler.close()
        longs = [f.exposure_us for f in world.survey.submitted if f.exposure_us >= 1_000_000]
        assert longs
        assert longs[0] == round(first_long_s * 1e6)

    def test_a_bad_scheduler_table_is_a_configuration_error_that_names_the_section(
        self, tmp_path: Path
    ) -> None:
        local = self.write_local(tmp_path, "[scheduler.fast]\nwindow_s = -5\n")
        config = load_config(local_file=local, env={})
        with pytest.raises(ConfigError, match=r"\[scheduler\]"):
            build_scheduler(config, **collaborators(World(start_utc_ns=NIGHT)))

    def test_a_bad_site_table_is_a_configuration_error_that_names_the_section(
        self, tmp_path: Path
    ) -> None:
        local = self.write_local(tmp_path, "[site]\nlatitude_deg = 95.0\nlongitude_deg = 0.0\n")
        config = load_config(local_file=local, env={})
        with pytest.raises(ConfigError, match=r"\[site\]"):
            build_scheduler(config, **collaborators(World(start_utc_ns=NIGHT)))


class TestProfileLimits:
    @pytest.mark.parametrize(
        ("config", "key"),
        [
            (SchedulerConfig(fast=FastConfig(exposure_us=5)), "scheduler.fast.exposure_us"),
            (SchedulerConfig(fast=FastConfig(gain=9999)), "scheduler.fast.gain"),
            (
                SchedulerConfig(survey=SurveyConfig(long_exposure_s=5000.0)),
                "scheduler.survey.long_exposure_s",
            ),
            (SchedulerConfig(survey=SurveyConfig(short_gain=700)), "scheduler.survey.short_gain"),
            (
                SchedulerConfig(watch=WatchConfig(bright_exposure_us=5)),
                "scheduler.watch.bright_exposure_us",
            ),
        ],
        ids=["fast-exposure", "fast-gain", "long-exposure", "short-gain", "bright-exposure"],
    )
    def test_a_value_outside_the_profile_fails_at_construction_and_names_the_key(
        self, config: SchedulerConfig, key: str
    ) -> None:
        world = World(start_utc_ns=NIGHT)
        with pytest.raises(ValueError, match="outside the profile") as excinfo:
            Scheduler(profile=PROFILE, station_id="test", config=config, **collaborators(world))
        assert key.rsplit(".", 1)[-1] in str(excinfo.value)

    def test_a_shortest_long_exposure_outside_the_profile_fails_and_names_the_key(self) -> None:
        twilight = TwilightConfig(min_exposure_s=5000.0)
        world = World(start_utc_ns=NIGHT)
        with pytest.raises(ValueError, match="outside the profile") as excinfo:
            Scheduler(profile=PROFILE, station_id="test", twilight=twilight, **collaborators(world))
        assert "survey.twilight.min_exposure_s" in str(excinfo.value)

    def test_the_default_configuration_fits_the_reference_profile(self) -> None:
        world = World(start_utc_ns=NIGHT)
        scheduler = Scheduler(profile=PROFILE, station_id="test", **collaborators(world))
        assert scheduler.config == SchedulerConfig()

    def test_a_high_speed_fast_stream_needs_high_speed_values_in_the_profile(self) -> None:
        no_high_speed = {
            "adc_bits_high_speed": None,
            "row_time_us_high_speed": None,
            "frame_overhead_ms_high_speed": None,
        }
        profile = PROFILE.model_copy(
            update={
                "readout_modes": [m.model_copy(update=no_high_speed) for m in PROFILE.readout_modes]
            }
        )
        config = SchedulerConfig(fast=FastConfig(high_speed=True))
        world = World(start_utc_ns=NIGHT)
        with pytest.raises(ProfileError, match="no high-speed variant"):
            Scheduler(profile=profile, station_id="test", config=config, **collaborators(world))
        # Without the key, the same profile is fine.
        Scheduler(profile=profile, station_id="test", **collaborators(World(start_utc_ns=NIGHT)))


class TestRealTime:
    def test_the_loop_runs_on_a_scaled_clock_in_a_thread_and_stops_on_the_event(self) -> None:
        """A real thread, real sleeps, and a clock that runs 600 times faster than real time.

        In a few seconds of real time the scheduler watches the sky, enters `auto`, and runs
        several cycles. Then the stop event ends the loop, and the camera closes.
        """
        clock = ScaledClock(start_utc_ns=NIGHT, origin_real_ns=time.time_ns(), speed=600.0)
        world = World(start_utc_ns=NIGHT, clock=clock)
        stop = threading.Event()
        errors: list[BaseException] = []

        def loop() -> None:
            try:
                world.scheduler.run(stop)
            except BaseException as error:  # the thread must report anything that escapes
                errors.append(error)

        thread = threading.Thread(target=loop)
        started = time.perf_counter()
        thread.start()
        while time.perf_counter() < started + 30.0 and len(world.windows()) < 4:
            time.sleep(0.05)
        stop.set()
        thread.join(timeout=30)
        elapsed_real_s = time.perf_counter() - started
        assert not thread.is_alive()
        assert errors == []
        assert len(world.windows()) >= 4
        assert world.camera.calls_named("close")
        assert sum(w.n_frames for w in world.windows()) == world.fast.frames_pushed
        # The loop never runs faster than the clock allows.
        elapsed_virtual_s = (clock.utc_ns() - NIGHT) / NS_PER_S
        assert elapsed_virtual_s <= 600.0 * elapsed_real_s * 1.2
        assert world.scheduler.status().counters.fast_periods >= 1
