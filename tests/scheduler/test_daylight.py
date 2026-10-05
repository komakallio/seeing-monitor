"""Daylight, dusk, twilight, and the measured sky: the gate that keeps the camera safe.

Every scenario runs on a virtual clock at a synthetic site, 55 degrees north on the prime
meridian. The scenario starts at 14:30 UTC on a winter day: the Sun is 6 degrees up, it sets at
about 15:55, and astronomical night begins at about 18:10.

The Sun's elevation gates nothing. The gate judges the background that the fast stream would
have at its shortest exposure (32 us). The default sky of the scenario clips the 1 ms brightness
frame (0.9 of its saturation) until the Sun is 2.49 degrees down, and a second watch frame of
32 us then reads the sky. The fast stream at 32 us would see 35% of saturation, where `auto`
starts, with the Sun 0.42 degrees down, and 50%, where `auto` stops, with the Sun 0.10 degrees
down. Polaris becomes detectable at about 3.5 degrees down. A floodlight adds to the sky: one of
1 (the 1 ms frame at its saturation) is 3.7% for the fast stream at 32 us, and one of 20 is 74%.
"""

from __future__ import annotations

import itertools

import pytest

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.frames import StreamKind
from seeingmon.scheduler.ephemeris import next_sun_crossing_utc_ns
from tests.scheduler.scenario import BLINDING_LIGHT, SITE, START, World

DARK = iso_to_utc_ns("2026-01-01T22:00:00Z")  # the Sun is 40 degrees down
# The default sky saturates the 1 ms brightness frame (0.9 of saturation with the offset of 200
# counts) down to this elevation, and Polaris reaches an SNR of 10 below about -3.49 degrees.
CLIP_DEG = -2.485
# The fast stream at 32 us would see 35% of saturation at this elevation, so a scheduler in `safe`
# resumes there, and 50% at the next, where a scheduler in `auto` stops. The watch frame of 32 us
# reads it, with the offset counted as sky.
RESUME_DEG = -0.420
STOP_DEG = -0.103
# A floodlight that clips the 1 ms frame, but that the fast stream can take: 18% of saturation at
# 32 us. `BLINDING_LIGHT` gives 74%.
SUNNY_LIGHT = 5.0


def watch_times(world: World, exposure_us: int = 1000) -> list[float]:
    """The times of the brightness frames: snapshots of the central ROI in bin2, at 1 ms or at the
    bright exposure that follows a clipped one."""
    return [
        world.seconds(call.t_utc_ns)
        for call in world.configures(mode="bin2", video=False)
        if call.config.exposure_us == exposure_us and call.config.roi is not None
    ]


def crossing(elevation: float, *, rising: bool, after: int = START) -> int:
    found = next_sun_crossing_utc_ns(
        after, SITE.latitude_deg, SITE.longitude_deg, elevation, rising=rising, horizon_days=1.0
    )
    assert found is not None
    return found


class TestDaylight:
    def test_the_scheduler_starts_in_safe_and_watches_the_sky_once_a_minute(self) -> None:
        world = World()
        world.run_until(3600)
        assert world.scheduler.state.value == "safe"
        times = watch_times(world)
        assert times[0] == pytest.approx(0.0, abs=0.01)  # the first frame comes at once
        gaps = [later - earlier for earlier, later in itertools.pairwise(times)]
        # The watch interval is 60 seconds, and a frame takes a few milliseconds of that.
        assert all(gap == pytest.approx(60.0, abs=0.01) for gap in gaps)
        assert len(times) == 60
        world.close()

    def test_nothing_but_the_watch_runs_in_daylight(self) -> None:
        world = World()
        world.run_until(3600)
        assert world.configures(mode="bin1") == []  # no fast stream
        assert world.windows() == []
        assert world.records("survey_frame") == []
        status = world.scheduler.status()
        assert status.state == "safe"
        assert status.stream is not None
        assert status.stream.purpose == "watch"
        # Each 1 ms frame clips, so a frame of 32 us follows it at once. It says that the fast
        # stream would see 74% of saturation in the daylight of the scenario, too much to measure.
        assert status.counters.watch_frames == 120
        assert watch_times(world, 32) == pytest.approx(watch_times(world), abs=1.0)
        assert status.background_fraction == pytest.approx(0.744, abs=0.001)
        assert world.state_changes() == []
        world.close()

    def test_the_watch_uses_the_survey_readout_mode_at_the_configured_exposure(self) -> None:
        world = World()
        world.run_until(130)
        calls = [c for c in world.configures() if c.config.exposure_us == 1000]
        assert len(calls) == 3
        for call in calls:
            config = call.config
            assert (config.mode, config.gain, config.kind) == ("bin2", 0, StreamKind.SNAPSHOT)
            assert config.roi is not None
            assert 300 <= config.roi.width <= 330  # 20 arcmin at 3.82 arcsec per pixel
        world.close()

    def test_the_start_is_one_event_and_the_missing_site_is_not(self) -> None:
        world = World()
        world.run_for(10)
        assert [e.kind for e in world.events()] == ["scheduler.start"]
        assert (world.events("scheduler.start")[0].detail or {})["site_configured"] is True
        world.close()

    def test_without_a_site_the_scheduler_says_so_and_relies_on_the_measured_sky(self) -> None:
        world = World(site=None)
        world.run_for(10)
        assert [e.kind for e in world.events()] == ["scheduler.start", "scheduler.no_site"]
        world.run_until(3000)  # a bright sky keeps it in safe, with no Sun elevation to help
        assert world.scheduler.state.value == "safe"
        status = world.scheduler.status()
        assert status.sun_elevation_deg is None
        assert status.twilight is False
        world.close()


class TestDusk:
    def test_the_scheduler_enters_auto_where_the_fast_stream_could_measure(self) -> None:
        """The gate opens where the fast stream at 32 us would see less than 35% of saturation.

        The 1 ms frame still clips there, two degrees above the Sun of its clip, and the watch
        frame of 32 us that follows it reads the sky.
        """
        world = World()
        clip = crossing(CLIP_DEG, rising=False)
        dusk = crossing(RESUME_DEG, rising=False)
        world.run_until(world.seconds(dusk) + 300)
        assert world.states_visited() == ["safe", "auto"]
        ((changed_at, _, _),) = world.state_changes()
        # The change happens at the first brightness frame after the sky falls below the level.
        assert world.seconds(dusk) <= changed_at <= world.seconds(dusk) + 61
        assert changed_at < world.seconds(clip) - 600  # the 1 ms frame still clips
        assert world.sun_elevation(world.t(changed_at)) > -0.7
        event = world.events("scheduler.state_change")[0]
        assert event.detail == {"from": "safe", "to": "auto", "reason": "the sky is dark enough"}
        world.close()

    def test_the_search_follows_the_change_at_once_and_measure_waits_for_polaris(self) -> None:
        world = World()
        clip = crossing(CLIP_DEG, rising=False)
        world.run_until(world.seconds(crossing(-4.5, rising=False)))
        ((changed_at, _, _),) = world.state_changes()
        bursts = world.burst_starts()
        assert bursts[0] == pytest.approx(changed_at, abs=1.0)
        assert bursts[0] < world.seconds(clip)  # the search starts while the 1 ms frame clips
        # Polaris shows in the bursts once its SNR reaches 10, at about -3.49 degrees, and the
        # second detection in a row starts measure, within a period and a half after that.
        (visible,) = world.events("polaris.visible")
        sun = (visible.detail or {})["sun_elevation_deg"]
        assert -4.1 < sun < -3.45
        (first_fast,) = world.fast_starts()[:1]
        assert first_fast == pytest.approx(world.seconds(visible.t_utc_ns), abs=0.1)
        assert all(t < first_fast for t in bursts[:2])
        world.close()

    def test_a_saturated_sky_holds_the_scheduler_in_safe_without_a_burst(self) -> None:
        """A floodlight that the fast stream cannot take even at 32 us keeps the gate shut."""
        world = World(start_utc_ns=DARK)
        world.light(0, 3600, BLINDING_LIGHT)
        world.run_until(3000)
        assert world.scheduler.state.value == "safe"
        assert world.state_changes() == []
        assert world.configures(mode="bin1") == []  # no burst, no fast stream
        assert world.scheduler.status().counters.search_bursts == 0
        world.run_until(3600 + 120)  # the light goes out, and the next frame lets auto begin
        assert world.states_visited() == ["safe", "auto"]
        ((changed_at, _, _),) = world.state_changes()
        assert 3600 <= changed_at <= 3600 + 61
        world.close()

    def test_a_bright_sky_that_the_fast_stream_could_still_measure_lets_auto_start(self) -> None:
        """The gate reads the background that the fast stream would have at 32 us.

        A floodlight at 50% of saturation in the 1 ms bin2 frame is 1.9% of saturation in a bin1
        frame of 32 us, far below the limit, so `auto` starts and the search runs. A gate on the
        frame's own share would wait for 35%.
        """
        world = World(start_utc_ns=DARK)
        world.light(0, 10_000, 0.5)
        world.run_until(600)
        assert world.states_visited() == ["safe", "auto"]
        status = world.scheduler.status()
        assert status.background_fraction == pytest.approx(0.019, abs=0.002)
        assert world.burst_starts()
        world.close()

    def test_a_clipped_brightness_frame_does_not_hold_safe(self) -> None:
        """A floodlight that clips the 1 ms frame (5 times its saturation) at night, from the start.

        The watch frame of 32 us that follows the clipped one reads 18% for the fast stream, so
        `auto` starts at the first watch. The clip of the 1 ms frame no longer decides.
        """
        world = World(start_utc_ns=DARK)
        world.light(0, 10_000, SUNNY_LIGHT)
        world.run_until(600)
        assert world.states_visited() == ["safe", "auto"]
        ((changed_at, _, _),) = world.state_changes()
        assert changed_at == pytest.approx(0.0, abs=1.0)
        assert watch_times(world, 32)[0] == pytest.approx(0.0, abs=1.0)
        world.close()


class TestASunnySky:
    """A bright sky that the fast stream can still take: the 1 ms frame clips, and `auto` holds."""

    def test_a_sunny_sky_where_the_fast_stream_is_fine_stays_in_auto(self) -> None:
        """A floodlight of 5 times the saturation of the 1 ms frame comes on in `auto`.

        Each survey step finds its 1 ms frame clipped, and the scenario's slow fast stream gives
        no background that decides, so a watch frame of 32 us follows the 1 ms frame. It reads 18%
        for the fast stream at 32 us, far below the limit of 50%, so the scheduler stays.
        """
        world = World(start_utc_ns=DARK)
        world.light(300, 10_000, SUNNY_LIGHT)
        world.run_until(1800)
        assert world.states_visited() == ["safe", "auto"]
        status = world.scheduler.status()
        assert status.background_fraction == pytest.approx(0.185, abs=0.01)
        steps_in_light = [t for t in watch_times(world, 32) if t > 300]
        assert len(steps_in_light) >= 7  # one for each survey step of the 1,500 s in the light
        assert world.burst_starts() or world.fast_starts()
        world.close()


class TestAutoStopsWhenTheSkyBrightens:
    def test_a_floodlight_forces_safe_and_skips_the_long_exposure(self) -> None:
        """A sky where even 32 us would pass 50% of saturation: the fast stream would see 74%."""
        world = World(start_utc_ns=DARK)
        world.light(1000, 2500, BLINDING_LIGHT)
        world.run_until(2400)
        # Auto stops at the next survey step: the short exposure sees the bright sky.
        assert world.states_visited() == ["safe", "auto", "safe"]
        changed_at = world.state_changes()[1][0]
        assert 1000 < changed_at < 1000 + 180 + 5
        event = world.events("scheduler.state_change")[1]
        assert (event.detail or {})["reason"] == "bright_sky"
        # The survey frame that found the bright sky was the short one. No long one followed.
        last = world.survey.submitted[-1]
        assert last.exposure_us == 1000
        assert world.survey.submitted[-1].t_utc_ns / NS_PER_S - START / NS_PER_S > 0
        assert world.scheduler.status().counters.survey_frames % 2 == 1  # a short without a long
        # The 1 ms frame clipped, so a watch frame of 32 us measured the sky before the decision.
        (checked_at,) = [t for t in watch_times(world, 32) if 1000 < t < changed_at + 1]
        assert checked_at == pytest.approx(changed_at, abs=1.0)
        world.close()

    def test_auto_resumes_when_the_light_goes_out(self) -> None:
        world = World(start_utc_ns=DARK)
        world.light(1000, 2500, BLINDING_LIGHT)
        world.run_until(3000)
        assert world.states_visited() == ["safe", "auto", "safe", "auto"]
        resumed_at = world.state_changes()[2][0]
        assert 2500 < resumed_at <= 2500 + 61
        world.close()

    def test_the_watch_runs_a_minute_after_the_scheduler_entered_safe(self) -> None:
        world = World(start_utc_ns=DARK)
        world.light(1000, 2500, BLINDING_LIGHT)
        world.run_until(2400)
        changed_at = world.state_changes()[1][0]
        later = [t for t in watch_times(world) if t > changed_at]
        assert later[0] == pytest.approx(changed_at + 60.0, abs=1.0)
        world.close()

    def test_the_brightening_sky_stops_auto_at_the_next_survey_step(self) -> None:
        """At dawn the sky passes 50% for the fast stream at 32 us, and a survey step sees it.

        Before that, the brightening sky hides Polaris (an SNR below 6 at about -2.6 degrees), so
        measure ends with `polaris.hidden`, and the search finds nothing more. The 1 ms frame
        clips from -2.49 degrees on, and the watch frame of 32 us that follows it decides.
        """
        start = iso_to_utc_ns("2026-01-02T07:00:00Z")  # the Sun is 6 degrees down
        world = World(start_utc_ns=start)
        dawn_limit = crossing(STOP_DEG, rising=True, after=start)
        world.run_until(world.seconds(dawn_limit) + 600)
        assert world.states_visited() == ["safe", "auto", "safe"]
        stopped_at = world.state_changes()[1][0]
        # The short survey frame at the end of each period checks the sky. The missing star ended
        # a period early and moved its survey step ahead, so the next check can come a cadence
        # and a period after it.
        assert world.seconds(dawn_limit) <= stopped_at <= world.seconds(dawn_limit) + 180 + 120 + 5
        assert (world.events("scheduler.state_change")[1].detail or {})["reason"] == "bright_sky"
        (hidden,) = world.events("polaris.hidden")
        assert (hidden.detail or {})["reason"] == "star_missing"
        assert -2.8 < (hidden.detail or {})["sun_elevation_deg"] < -2.4
        world.close()


@pytest.fixture(scope="module")
def dusk_world() -> World:
    """Two and a half hours from 17:00 UTC: the end of twilight, and then a dark sky."""
    start = iso_to_utc_ns("2026-01-01T17:00:00Z")  # the Sun is about 10 degrees down
    world = World(start_utc_ns=start)
    world.run_until(2 * 3600 + 1800)
    world.close()
    return world


class TestTwilight:
    def test_windows_in_twilight_carry_the_flag_and_later_windows_do_not(
        self, dusk_world: World
    ) -> None:
        world = dusk_world
        night = crossing(-18.0, rising=False, after=world.start_utc_ns)
        windows = world.windows()
        assert len(windows) > 60
        for window in windows:
            end_ns = window.t_utc_ns + round(window.duration_s * NS_PER_S)
            if end_ns <= night - 60 * NS_PER_S:
                assert "twilight" in window.flags
            elif window.t_utc_ns >= night + 60 * NS_PER_S:
                assert window.flags == []
        flagged = sum("twilight" in w.flags for w in windows)
        assert 0 < flagged < len(windows)

    def test_survey_results_carry_the_twilight_flag_too(self, dusk_world: World) -> None:
        world = dusk_world
        night = crossing(-18.0, rising=False, after=world.start_utc_ns)
        qualities = world.records("sky_quality")
        assert len(qualities) > 20
        for record in qualities:
            flags = record.flags  # type: ignore[attr-defined]
            if record.t_utc_ns <= night - 60 * NS_PER_S:
                assert "twilight" in flags
            elif record.t_utc_ns >= night + 60 * NS_PER_S:
                assert "twilight" not in flags

    def test_the_status_reports_the_sun_and_the_twilight_state(self, dusk_world: World) -> None:
        status = dusk_world.scheduler.status()
        assert status.twilight is False  # it is dark by now
        assert status.sun_elevation_deg is not None
        assert status.sun_elevation_deg < -18.0
        assert status.state == "auto"

    def test_windows_carry_the_zenith_angle_of_polaris_from_the_scheduler(
        self, dusk_world: World
    ) -> None:
        """The scheduler supplies the zenith angle, as the fast analysis cannot know it."""
        angles = [w.zenith_angle_deg for w in dusk_world.windows()]
        assert all(angle is not None for angle in angles)
        # Polaris is 35 degrees from the zenith at 55 degrees north, give or take its circle.
        assert all(34.0 <= angle <= 36.0 for angle in angles if angle is not None)
        assert min(a for a in angles if a is not None) < max(a for a in angles if a is not None)

    def test_windows_carry_no_other_flags_in_clear_weather(self, dusk_world: World) -> None:
        flags = {flag for window in dusk_world.windows() for flag in window.flags}
        assert flags <= {"twilight"}
