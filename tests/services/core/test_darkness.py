"""The events `sky.dark` and `sky.clear_verdict`: the rule, simulated nights, restarts, the wrapper.

The simulated nights drive `DarknessWatch` with the sky of the simulator's own model
(`seeingmon.drivers.sim.sky`) at the survey cadence across whole nights at the synthetic site, with
the simulator's clouds. A night of real survey frames through the pipeline would take minutes, so
`observe` stands in for the pipeline with a small model of what a long frame measures:

- **The sky brightness** is the model's sky plus a measurement noise of 0.03 mag, a few times the
  zero point's sampling error (the research notes give a floor of 0.01 mag). The simulator's
  clouds dim the stars and not the sky, and the frame's own zero point carries the cloud, so a
  solved frame under a cloud of transparency `T` reads the sky brighter by `-2.5 log10(T)`.
- **The stars.** A clear dark frame detects stars down to G 15, and a brighter sky raises that
  limit by half its brightening (a frame limited by the sky's noise). The expected stars are the
  catalog stars brighter than G 12 that a clear sky shows at an SNR of 20, 1.5 mag above the limit
  of detection: 60 of them in a dark sky, with star counts that rise by 0.4 dex a magnitude. A
  cloud moves the limit by `2.5 log10(T)`, each expected star brighter than the limit is found,
  and the count of found stars is a binomial draw. The frame solves with 8 found stars, and it
  has a cloud fraction with 8 expected stars (`[survey.cloud] min_expected`).

With this noise the slope of a fit over 5 frames (12 minutes) scatters by 0.19 mag an hour, while
the twilight sky of the model changes by 1.4 mag an hour as the Sun passes -18 degrees in January.
Across 200 seeds, `sky.dark` came with the Sun between -18.8 and -20.5 degrees on a winter night,
and within 14 minutes of the Sun's lowest point on a midsummer night. The tests run 20 seeds.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seeingmon.analysis import SurveyOutput
from seeingmon.clock import NS_PER_S, VirtualClock, iso_to_utc_ns
from seeingmon.drivers.sim.sky import (
    SYNTHETIC_SITE,
    CloudEvent,
    Clouds,
    sky_brightness_mag_arcsec2,
    sun_altitude_deg,
)
from seeingmon.frames import Frame
from seeingmon.records import EventRecord, Record, SurveyFrameRecord
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.scheduler.config import SiteConfig
from seeingmon.scheduler.ephemeris import sun_elevation_deg
from seeingmon.services.core.darkness import (
    CLEAR_VERDICT_EVENT,
    DARK_EVENT,
    DarknessWatch,
    SkyDarkness,
    SkyEvent,
    SkyFrame,
    fit_line,
)
from seeingmon.services.core.events import EventWriter
from seeingmon.store.db import Store
from seeingmon.survey.config import DarknessConfig
from tests.scheduler.helpers import make_frame
from tests.services.core.rig import read_all

NIGHT_START = iso_to_utc_ns("2026-01-10T12:00:00Z")  # the first moment of the night 2026-01-10
EVENING = iso_to_utc_ns("2026-01-10T19:00:00Z")
CADENCE_S = 180.0  # [scheduler.survey] cadence_s
SETTINGS = DarknessConfig()  # 0.3 mag an hour, 5 frames, 5 verdict frames
CLEAR_THRESHOLD = 0.3  # [scheduler.cloud] clear_threshold
SITE = SiteConfig(
    latitude_deg=SYNTHETIC_SITE.latitude_deg, longitude_deg=SYNTHETIC_SITE.longitude_deg
)
SEEDS = range(20)


def at(minutes: float) -> int:
    """A time in the evening of the night 2026-01-10."""
    return EVENING + round(minutes * 60 * NS_PER_S)


def sky_frame(
    minutes: float,
    sky: float | None = 20.5,
    *,
    solved: bool = True,
    cloud: float | None = 0.0,
    transparency: float | None = None,
    time_valid: bool = True,
) -> SkyFrame:
    return SkyFrame(at(minutes), solved, sky, cloud, transparency, time_valid)


def watch(
    settings: DarknessConfig = SETTINGS, *, site: SiteConfig | None = None, split: float = 12.0
) -> DarknessWatch:
    return DarknessWatch(settings, clear_threshold=CLEAR_THRESHOLD, split_utc_hour=split, site=site)


def feed(target: DarknessWatch, frames: list[SkyFrame]) -> list[SkyEvent]:
    events: list[SkyEvent] = []
    for frame in frames:
        events += target.update(frame)
    return events


def kinds(events: list[SkyEvent]) -> list[str]:
    return [event.kind for event in events]


def dark_run(count: int = 5, *, start: float = 0.0, slope_per_hour: float = 0.0) -> list[SkyFrame]:
    """`count` solved frames at the survey cadence along a line, from `start` minutes."""
    return [
        sky_frame(start + 3.0 * k, 20.5 + slope_per_hour * (start + 3.0 * k) / 60.0)
        for k in range(count)
    ]


# --- The rule -------------------------------------------------------------------------------


class TestTheLine:
    def test_the_fit_gives_the_slope_per_hour_and_the_value_at_the_newest_point(self) -> None:
        points = [(at(3.0 * k), 18.0 + 0.5 * 3.0 * k / 60.0) for k in range(5)]
        slope, level = fit_line(points)
        assert slope == pytest.approx(0.5, abs=1e-9)
        assert level == pytest.approx(18.0 + 0.5 * 12.0 / 60.0, abs=1e-9)


class TestSkyDark:
    def test_a_flat_run_of_solved_frames_makes_the_sky_dark_once(self) -> None:
        target = watch()
        events = feed(target, dark_run(4))
        assert events == []  # the fit needs 5 frames
        (event,) = feed(target, [sky_frame(12.0)])
        assert event.kind == DARK_EVENT
        assert event.t_utc_ns == at(12.0)  # the frame that completed the run
        assert event.detail == {
            "sky_mag_arcsec2": 20.5,
            "slope_mag_per_hour": 0.0,
            "frames": 5,
            "sun_elevation_deg": None,  # no site
        }
        assert target.dark_utc_ns == at(12.0)
        assert kinds(feed(target, dark_run(3, start=15.0))) == []  # once a night

    def test_the_limit_is_a_change_per_hour_and_strict(self) -> None:
        assert kinds(feed(watch(), dark_run(slope_per_hour=0.29))) == [DARK_EVENT]
        assert kinds(feed(watch(), dark_run(slope_per_hour=-0.29))) == [DARK_EVENT]  # dawn too
        assert feed(watch(), dark_run(slope_per_hour=0.31)) == []
        assert feed(watch(), dark_run(slope_per_hour=1.4)) == []  # the twilight at -18 degrees
        assert feed(watch(), dark_run(slope_per_hour=-0.31)) == []
        assert feed(watch(), dark_run(slope_per_hour=-1.4)) == []  # a brightening dawn sky

    def test_the_detail_holds_the_line_at_the_newest_frame(self) -> None:
        (event,) = feed(watch(), dark_run(slope_per_hour=0.2))
        assert event.detail["slope_mag_per_hour"] == pytest.approx(0.2, abs=1e-3)
        assert event.detail["sky_mag_arcsec2"] == pytest.approx(20.5 + 0.2 * 12.0 / 60.0, abs=1e-3)

    @pytest.mark.parametrize(
        "spoiler",
        [
            sky_frame(12.0, solved=False),  # clouds, for example
            sky_frame(12.0, None),  # no dark model, or no zero point
        ],
        ids=["unsolved", "no-sky"],
    )
    def test_a_frame_that_did_not_solve_or_has_no_sky_starts_the_run_again(
        self, spoiler: SkyFrame
    ) -> None:
        target = watch()
        assert feed(target, [*dark_run(4), spoiler, *dark_run(4, start=15.0)]) == []
        assert kinds(feed(target, [sky_frame(27.0)])) == [DARK_EVENT]

    def test_a_gap_longer_than_the_limit_starts_the_run_again(self) -> None:
        assert SETTINGS.max_gap_s == 600.0
        target = watch()
        # 10 minutes and 6 seconds after the second frame: the scheduler took no long frames.
        assert feed(target, [*dark_run(2), *dark_run(4, start=13.1)]) == []
        assert kinds(feed(target, [sky_frame(25.1)])) == [DARK_EVENT]  # five frames after it

    def test_a_gap_at_the_limit_keeps_the_run(self) -> None:
        # Exactly 10 minutes, as when the scheduler skips two survey steps of 180 s and a bit.
        assert kinds(feed(watch(), [*dark_run(2), *dark_run(3, start=13.0)])) == [DARK_EVENT]

    def test_a_clock_that_goes_back_starts_the_run_again(self) -> None:
        target = watch()
        assert feed(target, [*dark_run(4, start=30.0), *dark_run(4)]) == []
        assert kinds(feed(target, [sky_frame(12.0)])) == [DARK_EVENT]

    def test_the_suns_elevation_needs_a_site_and_a_synchronized_clock(self) -> None:
        (event,) = feed(watch(site=SITE), dark_run())
        expected = sun_elevation_deg(at(12.0), SITE.latitude_deg, SITE.longitude_deg)
        assert event.detail["sun_elevation_deg"] == pytest.approx(expected, abs=0.01)
        assert event.detail["sun_elevation_deg"] < -18.0  # 19:12 UTC in January at 55 degrees
        frames = [*dark_run(4), sky_frame(12.0, time_valid=False)]
        (event,) = feed(watch(site=SITE), frames)
        assert event.detail["sun_elevation_deg"] is None


class TestTheVerdict:
    def test_the_verdict_counts_the_frames_with_a_cloud_fraction_solved_or_not(self) -> None:
        target = watch()
        assert kinds(feed(target, dark_run())) == [DARK_EVENT]
        after = [
            sky_frame(15.0, cloud=0.1, transparency=0.9),
            sky_frame(18.0, None, solved=False, cloud=0.95),  # a thick cloud: no solve
            sky_frame(21.0, cloud=None),  # too few expected stars: no count
            sky_frame(24.0, cloud=0.3, transparency=0.7),  # at the threshold: clear
            sky_frame(27.0, cloud=0.31, transparency=0.5),
        ]
        assert feed(target, after) == []  # four frames with a cloud fraction so far
        (event,) = feed(target, [sky_frame(30.0, cloud=0.0, transparency=1.0)])
        assert event.kind == CLEAR_VERDICT_EVENT
        assert event.t_utc_ns == at(30.0)
        assert event.detail == {
            "clear_share": 0.6,  # 0.1, 0.3, and 0.0 of five
            "frames": 5,
            "clear_threshold": CLEAR_THRESHOLD,
            "transparency_median": 0.8,  # of 0.9, 0.7, 0.5, and 1.0
        }
        assert target.verdict_given
        assert feed(target, [sky_frame(33.0)]) == []  # once a night

    def test_frames_without_a_transparency_give_no_median(self) -> None:
        target = watch(DarknessConfig(verdict_frames=2))
        events = feed(target, [*dark_run(), sky_frame(15.0), sky_frame(18.0, cloud=0.8)])
        assert kinds(events) == [DARK_EVENT, CLEAR_VERDICT_EVENT]
        assert events[1].detail["clear_share"] == 0.5
        assert events[1].detail["transparency_median"] is None

    def test_the_frame_of_sky_dark_is_not_a_verdict_frame(self) -> None:
        target = watch(DarknessConfig(verdict_frames=1))
        events = feed(target, dark_run())
        assert kinds(events) == [DARK_EVENT]
        assert kinds(feed(target, [sky_frame(15.0, cloud=0.9)])) == [CLEAR_VERDICT_EVENT]

    def test_a_frame_at_or_before_sky_dark_is_no_verdict_frame(self) -> None:
        target = watch(DarknessConfig(verdict_frames=1))
        assert kinds(feed(target, dark_run())) == [DARK_EVENT]  # at 12 minutes
        # The clock stepped back: these frames come after `sky.dark` but are not later than it.
        assert feed(target, [sky_frame(12.0, cloud=0.9), sky_frame(9.0, cloud=0.9)]) == []
        (verdict,) = feed(target, [sky_frame(15.0, cloud=0.0)])
        assert verdict.kind == CLEAR_VERDICT_EVENT
        assert verdict.detail["clear_share"] == 1.0  # the cloudy frames did not count


class TestTheNights:
    def test_each_event_comes_once_a_night_and_the_split_hour_starts_the_next_night(self) -> None:
        target = watch(DarknessConfig(verdict_frames=1))
        first = feed(target, [*dark_run(), sky_frame(15.0), *dark_run(3, start=18.0)])
        assert kinds(first) == [DARK_EVENT, CLEAR_VERDICT_EVENT]
        next_night = 24 * 60.0  # the evening of 2026-01-11
        second = feed(target, [*dark_run(start=next_night), sky_frame(next_night + 15.0)])
        assert kinds(second) == [DARK_EVENT, CLEAR_VERDICT_EVENT]
        assert target.night == "2026-01-11"

    def test_the_split_hour_ends_a_run(self) -> None:
        target = watch(split=19.2)  # 19:12 UTC: an hour that the test may put in the dark
        assert feed(target, dark_run(5)) == []  # the fifth frame, at 19:12, starts a night
        assert kinds(feed(target, dark_run(4, start=15.0))) == [DARK_EVENT]


# --- Simulated nights -----------------------------------------------------------------------

DARK_SKY_MAG = 20.5  # the simulator's default dark sky
SKY_NOISE_MAG = 0.03
DARK_LIMIT_G = 15.0
EXPECTED_LIMIT_G = 12.0  # [survey.cloud] mag_limit
SNR_MARGIN_MAG = 2.5 * math.log10(20.0 / 5.0)  # expected at an SNR of 20, detected at 5 sigma
COUNT_SLOPE = 0.4  # dex per magnitude
EXPECTED_IN_THE_DARK = 60
SOLVE_STARS = 8
EXPOSURE_S = 30.0


@dataclass(frozen=True, slots=True)
class Observed:
    """An event of a simulated night, with the Sun's elevation of the simulator at its time."""

    kind: str
    t_utc_ns: int
    sun_deg: float
    detail: dict[str, Any]


def observe(t_utc_ns: int, clouds: Clouds, rng: np.random.Generator) -> SkyFrame:
    """What a long survey frame measures at a time, after the model in the module text."""
    sun = float(sun_altitude_deg(SYNTHETIC_SITE, t_utc_ns))
    sky = float(sky_brightness_mag_arcsec2(DARK_SKY_MAG, sun))
    start = t_utc_ns - round(EXPOSURE_S / 2 * NS_PER_S)
    transparency = clouds.mean_transparency(start, EXPOSURE_S)
    extinction = 2.5 * math.log10(max(transparency, 1e-9))
    clear_limit = DARK_LIMIT_G - 0.5 * (DARK_SKY_MAG - sky)
    expected_limit = min(EXPECTED_LIMIT_G, clear_limit - SNR_MARGIN_MAG)
    n_expected = round(EXPECTED_IN_THE_DARK * 10 ** (COUNT_SLOPE * (expected_limit - 12.0)))
    share = min(1.0, 10 ** (COUNT_SLOPE * (clear_limit + extinction - expected_limit)))
    found = int(rng.binomial(n_expected, share))
    noise = float(rng.normal(0.0, SKY_NOISE_MAG))
    solved = found >= SOLVE_STARS
    return SkyFrame(
        t_utc_ns=t_utc_ns,
        solved=solved,
        # A frame that does not solve takes the reference zero point, which knows no cloud.
        sky_mag_arcsec2=sky + noise + (extinction if solved else 0.0),
        cloud_fraction=1.0 - found / n_expected if n_expected >= 8 else None,
        transparency=transparency if solved else None,
    )


def frame_times(start: str, hours: float) -> list[int]:
    """The times of the long survey frames at the survey cadence."""
    first = iso_to_utc_ns(start)
    return [first + round(k * CADENCE_S * NS_PER_S) for k in range(round(hours * 3600 / CADENCE_S))]


def run_night(times: list[int], *, seed: int, clouds: Clouds | None = None) -> list[Observed]:
    """Run the survey frames at `times` through a watch, and return its events."""
    rng = np.random.default_rng(seed)
    target = watch(site=SITE)
    observed: list[Observed] = []
    for t in times:
        for event in target.update(observe(t, clouds or Clouds(), rng)):
            sun = float(sun_altitude_deg(SYNTHETIC_SITE, event.t_utc_ns))
            observed.append(Observed(event.kind, event.t_utc_ns, sun, event.detail))
    return observed


def simulate(
    start: str, hours: float, *, seed: int, clouds: Clouds | None = None
) -> list[Observed]:
    """Run the survey frames of a night through a watch, and return its events."""
    return run_night(frame_times(start, hours), seed=seed, clouds=clouds)


def lowest_sun(start: str, hours: float) -> tuple[int, float]:
    """The time and the elevation of the Sun's lowest point, to the minute."""
    first = iso_to_utc_ns(start)
    times = first + np.arange(round(hours * 60)) * 60 * NS_PER_S
    elevation = np.asarray(sun_altitude_deg(SYNTHETIC_SITE, times))
    index = int(np.argmin(elevation))
    return int(times[index]), float(elevation[index])


WINTER = "2026-01-10T15:00:00Z"  # sunset at the synthetic site is near 15:40 UTC
SUMMER = "2026-06-21T18:00:00Z"
DAWN = "2026-01-11T06:20:00Z"  # the morning twilight of the night 2026-01-10
NIGHT_HOURS = 12.0
CADENCE_NS = round(CADENCE_S * NS_PER_S)
# An overcast: 7.5 mag of extinction leaves 1.6% of the expected stars, so no frame solves.
OVERCAST = 0.001


class TestSimulatedNights:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_clear_winter_night_gets_dark_after_the_twilight_and_a_clear_verdict(
        self, seed: int
    ) -> None:
        dark, verdict = simulate(WINTER, NIGHT_HOURS, seed=seed)
        assert dark.kind == DARK_EVENT
        # The twilight sky changes by 1.4 mag an hour down to -18 degrees, and the run of 5
        # frames needs 3 or 4 frames below it (200 seeds: -18.8 to -20.5 degrees).
        assert -21.5 < dark.sun_deg < -18.0
        assert dark.detail["sun_elevation_deg"] == pytest.approx(dark.sun_deg, abs=0.1)
        assert dark.detail["sky_mag_arcsec2"] == pytest.approx(DARK_SKY_MAG, abs=0.1)  # 4 sigma
        assert abs(dark.detail["slope_mag_per_hour"]) < SETTINGS.max_slope_mag_per_hour
        assert verdict.kind == CLEAR_VERDICT_EVENT
        assert verdict.t_utc_ns == dark.t_utc_ns + 5 * CADENCE_NS  # every frame has a fraction
        assert verdict.detail["clear_share"] == 1.0
        assert verdict.detail["transparency_median"] == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_summer_night_with_the_sun_above_minus_18_degrees_still_gets_dark(
        self, seed: int
    ) -> None:
        lowest_utc_ns, lowest_deg = lowest_sun(SUMMER, NIGHT_HOURS)
        assert -12.0 < lowest_deg < -11.0  # 90 - 55 - 23.44: the Sun never reaches -18 degrees
        dark, verdict = simulate(SUMMER, NIGHT_HOURS, seed=seed)
        assert dark.kind == DARK_EVENT
        # The sky stops changing near the Sun's lowest point (200 seeds: within 14 minutes).
        assert abs(dark.t_utc_ns - lowest_utc_ns) < 20 * 60 * NS_PER_S
        assert dark.sun_deg == pytest.approx(lowest_deg, abs=0.2)
        true_sky = float(sky_brightness_mag_arcsec2(DARK_SKY_MAG, dark.sun_deg))
        assert dark.detail["sky_mag_arcsec2"] == pytest.approx(true_sky, abs=0.1)
        assert verdict.kind == CLEAR_VERDICT_EVENT
        assert verdict.detail["clear_share"] == 1.0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_thick_clouds_after_dark_give_a_cloudy_verdict(self, seed: int) -> None:
        clear_dark, _ = simulate(WINTER, NIGHT_HOURS, seed=seed)
        # An overcast that keeps every frame from solving, from a minute after `sky.dark`.
        overcast = CloudEvent(
            clear_dark.t_utc_ns + 60 * NS_PER_S, duration_s=7200.0, transmission=OVERCAST
        )
        dark, verdict = simulate(WINTER, NIGHT_HOURS, seed=seed, clouds=Clouds(events=(overcast,)))
        assert dark.t_utc_ns == clear_dark.t_utc_ns
        assert verdict.kind == CLEAR_VERDICT_EVENT
        assert verdict.t_utc_ns == dark.t_utc_ns + 5 * CADENCE_NS  # the verdict does not wait
        assert verdict.detail["clear_share"] == 0.0
        assert verdict.detail["transparency_median"] is None  # no frame solved

    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_cloud_over_two_of_the_five_verdict_frames_gives_three_fifths(
        self, seed: int
    ) -> None:
        clear_dark, _ = simulate(WINTER, NIGHT_HOURS, seed=seed)
        # From 1.5 to 3.5 cadences after `sky.dark`: it covers the second and the third frame.
        cloud = CloudEvent(
            clear_dark.t_utc_ns + round(1.5 * CADENCE_NS),
            duration_s=2.0 * CADENCE_S,
            transmission=0.02,  # the bright stars still show, so the frames solve
        )
        _, verdict = simulate(WINTER, NIGHT_HOURS, seed=seed, clouds=Clouds(events=(cloud,)))
        assert verdict.detail["clear_share"] == 0.6
        assert verdict.detail["transparency_median"] == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hours_without_frames_never_make_a_dawn_sky_dark(self, seed: int) -> None:
        # Two dark frames after midnight, then none until the morning twilight: the scheduler was
        # paused, or the camera degraded. A line across the gap would barely slope.
        times = [*frame_times("2026-01-11T01:00:00Z", 0.1), *frame_times(DAWN, 3.0)]
        assert len(times) == 2 + 60
        dawn_sun = float(sun_altitude_deg(SYNTHETIC_SITE, iso_to_utc_ns(DAWN)))
        assert -18.0 < dawn_sun < -12.0  # the twilight sky brightens by 1.4 mag an hour or more
        assert run_night(times, seed=seed) == []

    @pytest.mark.parametrize("seed", SEEDS)
    def test_clouds_at_dusk_hold_sky_dark_back_until_the_sky_clears(self, seed: int) -> None:
        start = iso_to_utc_ns("2026-01-10T17:00:00Z")  # the Sun near -12 degrees
        overcast = CloudEvent(start, duration_s=3 * 3600.0, transmission=OVERCAST)
        dark, verdict = simulate(WINTER, NIGHT_HOURS, seed=seed, clouds=Clouds(events=(overcast,)))
        end = start + 3 * 3600 * NS_PER_S
        # The run needs 5 solved frames after the cloud, and the sky is dark by then.
        assert end + 4 * CADENCE_NS <= dark.t_utc_ns <= end + 8 * CADENCE_NS
        assert verdict.detail["clear_share"] == 1.0


# --- A restart ------------------------------------------------------------------------------


def event_record(kind: str, t_utc_ns: int) -> EventRecord:
    return EventRecord(
        station_id="test",
        t_utc_ns=t_utc_ns,
        profile_id="test",
        provenance={"source": "local"},
        level="info",
        kind=kind,
        message="An event of an earlier run of core.",
    )


def sky_record(t_utc_ns: int, cloud: float | None, transparency: float | None = None) -> Record:
    return SkyQualityRecord(
        station_id="test",
        t_utc_ns=t_utc_ns,
        profile_id="test",
        provenance={"algo": "test"},
        cloud_fraction=cloud,
        transparency=transparency,
        n_stars_used=0,
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "db.sqlite") as opened:
        yield opened


class TestARestart:
    def test_a_restart_after_sky_dark_writes_no_second_one_and_keeps_the_verdict_frames(
        self, store: Store
    ) -> None:
        store.write(event_record(DARK_EVENT, at(12.0)))
        store.write(event_record("scheduler.cloud", at(12.0)))  # moved by 1 ns in the store
        store.write(sky_record(at(12.0), 0.0))  # the frame of `sky.dark` itself
        store.write(sky_record(at(15.0), 0.9, 0.2))
        store.write(sky_record(at(18.0), None))
        store.write(sky_record(at(21.0), 0.1, 1.0))
        target = watch(DarknessConfig(verdict_frames=3))
        target.restore(store, at(22.0))
        assert target.dark_utc_ns == at(12.0)
        assert not target.verdict_given
        events = feed(target, [*dark_run(5, start=24.0)])
        assert kinds(events) == [CLEAR_VERDICT_EVENT]  # no second `sky.dark`
        assert events[0].t_utc_ns == at(24.0)
        assert events[0].detail["clear_share"] == pytest.approx(2 / 3, abs=1e-3)  # 0.1 and 0.0
        assert events[0].detail["transparency_median"] == pytest.approx(0.6)  # of 0.2 and 1.0

    def test_a_restart_after_the_verdict_writes_nothing_more_in_this_night(
        self, store: Store
    ) -> None:
        store.write(event_record(DARK_EVENT, at(12.0)))
        store.write(event_record(CLEAR_VERDICT_EVENT, at(27.0)))
        target = watch(DarknessConfig(verdict_frames=1))
        target.restore(store, at(40.0))
        assert target.verdict_given
        assert feed(target, dark_run(8, start=45.0)) == []
        next_night = 24 * 60.0
        assert kinds(feed(target, dark_run(start=next_night))) == [DARK_EVENT]

    def test_the_events_of_the_last_night_do_not_count(self, store: Store) -> None:
        store.write(event_record(DARK_EVENT, at(-24 * 60.0)))  # the night 2026-01-09
        target = watch()
        target.restore(store, at(0.0))
        assert target.dark_utc_ns is None
        assert kinds(feed(target, dark_run())) == [DARK_EVENT]

    def test_a_night_with_many_events_is_read_to_its_end(self, store: Store) -> None:
        for k in range(1200):  # more than one read of the store returns
            store.write(event_record("scheduler.cloud", NIGHT_START + k * NS_PER_S))
        store.write(event_record(DARK_EVENT, at(12.0)))
        target = watch()
        target.restore(store, at(20.0))
        assert target.dark_utc_ns == at(12.0)


# --- The wrapper ----------------------------------------------------------------------------


def survey_output(frame: SkyFrame, *, quality: bool = True) -> SurveyOutput:
    records: list[Record] = [
        SurveyFrameRecord(
            station_id="test",
            t_utc_ns=frame.t_utc_ns,
            profile_id="test",
            provenance={"algo": "test"},
            exposure_s=30.0,
            gain=120,
            readout_mode="bin2",
        )
    ]
    if quality:
        records.append(
            SkyQualityRecord(
                station_id="test",
                t_utc_ns=frame.t_utc_ns,
                profile_id="test",
                provenance={"algo": "test"},
                sky_mag_arcsec2=frame.sky_mag_arcsec2,
                cloud_fraction=frame.cloud_fraction,
                n_stars_used=20,
                flags=[] if frame.time_valid else ["time_invalid"],
            )
        )
    return SurveyOutput(frame.t_utc_ns, tuple(records), frame.solved, frame.cloud_fraction)


class Emitted:
    """An `EmitEvent` that keeps the kinds."""

    def __init__(self) -> None:
        self.kinds: list[str] = []

    def __call__(
        self,
        level: str,
        kind: str,
        message: str,
        detail: Mapping[str, Any] | None = None,
        *,
        t_utc_ns: int | None = None,
    ) -> None:
        self.kinds.append(kind)


class ScriptedSurvey:
    """A survey analyzer that returns the scripted results, one for each `poll`."""

    def __init__(self, outputs: list[SurveyOutput]) -> None:
        self.outputs = outputs
        self.submitted: list[Frame] = []
        self.tracker = "the tracker"

    def submit(self, frame: Frame) -> None:
        self.submitted.append(frame)

    def poll(self) -> tuple[SurveyOutput, ...]:
        return (self.outputs.pop(0),) if self.outputs else ()

    def pending(self) -> int:
        return len(self.outputs)


class TestTheWrapper:
    def test_the_results_pass_unchanged_and_the_events_reach_the_store(self, store: Store) -> None:
        frames = [*dark_run(), sky_frame(15.0, cloud=0.5)]
        outputs = [survey_output(f) for f in frames]
        clock = VirtualClock(at(20.0))
        writer = EventWriter(store.write, station_id="test", profile_id="test", clock=clock)
        survey = SkyDarkness(
            ScriptedSurvey(list(outputs)),
            watch(DarknessConfig(verdict_frames=1), site=SITE),
            emit=writer.emit,
        )
        polled = [output for _ in frames for output in survey.poll()]
        assert polled == outputs
        assert survey.written == 2
        stored = read_all(store, "event")
        assert [(e.kind, e.t_utc_ns) for e in stored] == [  # type: ignore[attr-defined]
            (DARK_EVENT, at(12.0)),
            (CLEAR_VERDICT_EVENT, at(15.0)),
        ]
        dark, verdict = stored
        assert dark.provenance["source"] == "local"
        assert dark.detail["sun_elevation_deg"] < -18.0  # type: ignore[attr-defined]
        assert verdict.detail["clear_share"] == 0.0  # type: ignore[attr-defined]

    def test_a_record_with_the_time_invalid_flag_gives_no_suns_elevation(
        self, store: Store
    ) -> None:
        frames = [*dark_run(4), sky_frame(12.0, time_valid=False)]
        writer = EventWriter(
            store.write, station_id="test", profile_id="test", clock=VirtualClock(at(20.0))
        )
        survey = SkyDarkness(
            ScriptedSurvey([survey_output(f) for f in frames]), watch(site=SITE), emit=writer.emit
        )
        for _ in frames:
            survey.poll()
        (dark,) = read_all(store, "event")
        assert dark.kind == DARK_EVENT  # type: ignore[attr-defined]
        assert dark.detail["sun_elevation_deg"] is None  # type: ignore[attr-defined]
        assert dark.detail["frames"] == 5  # type: ignore[attr-defined]

    def test_a_short_frame_without_sky_quality_is_no_frame_of_the_run(self) -> None:
        frames = dark_run()
        outputs = [survey_output(f) for f in frames[:4]]
        outputs.append(survey_output(sky_frame(10.0, solved=False), quality=False))  # 1 ms frame
        outputs.append(survey_output(frames[4]))
        emitted = Emitted()
        survey = SkyDarkness(ScriptedSurvey(outputs), watch(), emit=emitted)
        for _ in range(6):
            survey.poll()
        assert emitted.kinds == [DARK_EVENT]

    def test_a_failing_watch_never_reaches_the_scheduler(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class Broken(DarknessWatch):
            def update(self, frame: SkyFrame) -> list[SkyEvent]:
                raise RuntimeError("a bug")

        output = survey_output(sky_frame(0.0))
        emitted = Emitted()
        broken = Broken(SETTINGS, clear_threshold=CLEAR_THRESHOLD)
        survey = SkyDarkness(ScriptedSurvey([output]), broken, emit=emitted)
        assert survey.poll() == (output,)  # the records of the frame still reach the store
        assert "the darkness watch failed" in caplog.text
        assert emitted.kinds == []

    def test_the_rest_of_the_analyzer_stays_reachable(self) -> None:
        inner = ScriptedSurvey([])
        survey = SkyDarkness(inner, watch(), emit=Emitted())
        frame = make_frame(np.zeros((4, 4), dtype=np.uint16), mode="bin2", t_utc_ns=at(0.0))
        survey.submit(frame)
        assert inner.submitted == [frame]
        assert survey.pending() == 0
        assert survey.tracker == "the tracker"
