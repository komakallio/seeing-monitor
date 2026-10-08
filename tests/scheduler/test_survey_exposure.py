"""The adaptive long survey exposure and the skip in daylight (`[survey.twilight]`).

`SurveyExposure` holds the rule. The tests below drive it with frames of a model sky first, and then
run the scheduler in the scenario world (`tests.scheduler.scenario`), whose long survey frames
render the sky at their own exposure, with `twilight=TwilightConfig()`.

The model sky uses the numbers of the reference camera in bin2: one count of sky in the 1 ms frame
at gain 0 gives 4.6 counts in the long frame at gain 120 for each millisecond of the long exposure
(the ratio of the conversion gains, 4.05 over 0.88 electrons per count), a 16-bit container that
holds 14 bits (a step of 4 counts), and a saturation level of 65,532 counts. The black level is
480 counts, the simulator's offset.
"""

from __future__ import annotations

import itertools
import math

import pytest

from seeingmon.clock import ClockStatus, VirtualClock, iso_to_utc_ns
from seeingmon.frames import Frame
from seeingmon.scheduler.exposure import (
    EPISODE_TOLERANCE_COUNTS,
    LONG_STEP_FACTOR,
    PLAN_BRIGHT,
    PLAN_DISABLED,
    PLAN_FIRST,
    PLAN_PREVIOUS,
    PLAN_UNCHANGED,
    REFRESH_COUNTS,
    SKIP_MARGIN_COUNTS,
    LongFrame,
    ShortFrame,
    SurveyExposure,
)
from seeingmon.scheduler.gates import median_dn
from seeingmon.survey.config import TwilightConfig
from tests.scheduler.scenario import OFFSET_DN, World, pole_sky

TARGET = 0.3
SHORTEST_US = 1_000_000
LONGEST_US = 30_000_000
SATURATION_DN = 65_532.0
STEP_DN = 4.0
BLACK_DN = 480.0
SHORT_US = 1000
# The counts of sky in the long frame for each count in the 1 ms frame, per microsecond.
LONG_PER_SHORT_US = (4.05 / 0.88) / SHORT_US
CLIP = 0.9


def controller() -> SurveyExposure:
    return SurveyExposure(
        target=TARGET,
        shortest_us=SHORTEST_US,
        longest_us=LONGEST_US,
        saturation_dn=SATURATION_DN,
    )


def short_frame(sky_dn: float) -> ShortFrame:
    """The 1 ms frame of a sky of `sky_dn` counts, whose median the ADC rounds to a whole step."""
    median = min(SATURATION_DN, STEP_DN * round((BLACK_DN + sky_dn) / STEP_DN))
    return ShortFrame(
        median_dn=median,
        clipped=median >= CLIP * SATURATION_DN,
        long_dn_per_dn_us=LONG_PER_SHORT_US,
        step_dn=STEP_DN,
    )


def long_frame(sky_dn: float, exposure_us: int) -> LongFrame:
    """The long frame of the same sky (counts in the 1 ms frame) at `exposure_us`."""
    level = min(SATURATION_DN, BLACK_DN + sky_dn * LONG_PER_SHORT_US * exposure_us)
    return LongFrame(
        exposure_us=exposure_us,
        median_dn=level,
        saturation_dn=SATURATION_DN,
        clipped=level >= CLIP * SATURATION_DN,
    )


def sky_for(fraction_at_1_s: float) -> float:
    """The sky of the 1 ms frame, in counts, that puts a long frame of 1 s at this background."""
    return fraction_at_1_s * SATURATION_DN / (LONG_PER_SHORT_US * SHORTEST_US)


def step(control: SurveyExposure, sky_dn: float) -> int | None:
    """One survey step: the plan from the 1 ms frame, and the long frame if one is taken."""
    short = short_frame(sky_dn)
    plan = control.plan(short)
    if plan.exposure_us is not None:
        control.update(short, long_frame(sky_dn, plan.exposure_us))
    return plan.exposure_us


DARK_SKY = sky_for(0.0004)  # a long frame of 30 s holds 1.2% of saturation of sky


class TestTheModelSky:
    def test_one_count_of_the_1_ms_frame_is_about_the_target_in_1_s(self) -> None:
        """The 1 ms frame cannot place the long exposure between 1 and 30 s."""
        fraction = LONG_PER_SHORT_US * SHORTEST_US / SATURATION_DN
        assert fraction * STEP_DN == pytest.approx(0.281, abs=0.001)  # one count of the ADC


class TestADarkSky:
    def test_the_long_frames_ramp_up_by_4_times_a_step_and_stay_at_the_longest(self) -> None:
        control = controller()
        exposures = [step(control, DARK_SKY) for _ in range(6)]
        assert exposures == [1_000_000, 4_000_000, 16_000_000, 30_000_000, 30_000_000, 30_000_000]

    def test_a_long_frame_that_did_not_clip_gives_the_exposure_of_the_target_at_once(self) -> None:
        control = controller()
        assert control.jump_us(short_frame(DARK_SKY)) is None  # no long frame yet
        control.plan(short_frame(DARK_SKY))
        control.update(short_frame(DARK_SKY), long_frame(DARK_SKY, SHORTEST_US))
        assert control.jump_us(short_frame(DARK_SKY)) == LONGEST_US

    def test_a_long_frame_that_clipped_gives_no_jump(self) -> None:
        control = controller()
        sky = sky_for(30.0)
        control.plan(short_frame(sky))
        control.update(short_frame(sky), long_frame(sky, SHORTEST_US))
        assert control.jump_us(short_frame(sky)) is None

    def test_the_first_plan_says_why(self) -> None:
        control = controller()
        plan = control.plan(short_frame(DARK_SKY))
        assert (plan.exposure_us, plan.reason) == (SHORTEST_US, PLAN_FIRST)
        control.update(short_frame(DARK_SKY), long_frame(DARK_SKY, SHORTEST_US))
        assert control.plan(short_frame(DARK_SKY)).reason == PLAN_PREVIOUS

    def test_a_long_frame_that_does_not_clip_measures_the_black_level(self) -> None:
        """The 1 ms frame holds `b + s`, the long one `b + k s`: two equations, two unknowns.

        The median of the 1 ms frame rounds to a whole step of 4 counts, so the black level comes
        out within half a step.
        """
        control = controller()
        assert control.black_dn is None
        step(control, DARK_SKY)
        assert control.black_measured
        assert control.black_dn == pytest.approx(BLACK_DN, abs=STEP_DN / 2)


class TestABrightSky:
    def test_the_1_ms_frame_skips_the_long_exposure_once_the_black_level_is_known(
        self,
    ) -> None:
        """A sky that stands 16 counts of the ADC above the black level, and a little more."""
        control = controller()
        step(control, DARK_SKY)  # the night measures the black level
        margin_dn = SKIP_MARGIN_COUNTS * STEP_DN
        bright = margin_dn + sky_for(0.6)  # beyond the margin by twice the target at 1 s
        plan = control.plan(short_frame(bright))
        assert (plan.exposure_us, plan.reason) == (None, PLAN_BRIGHT)
        # A sky just inside the margin takes the long frame: the margin covers a black level that
        # the frames misjudged by a few counts.
        assert control.plan(short_frame(margin_dn)).exposure_us is not None

    def test_a_sky_that_the_1_ms_frame_proves_caps_the_long_exposure(self) -> None:
        """The Moon rises after a dark hour: the last long frame took 30 s.

        The 1 ms frame shows one count of the ADC of sky beyond the margin, which puts a long frame
        of 1 s at least at 0.28 of saturation. The next long frame takes at most 1.07 s, and not
        the 30 s of the last one.
        """
        control = controller()
        for _ in range(4):
            step(control, DARK_SKY)
        moon = (SKIP_MARGIN_COUNTS + 1) * STEP_DN
        plan = control.plan(short_frame(moon))
        assert plan.reason == PLAN_PREVIOUS
        proven = STEP_DN * LONG_PER_SHORT_US * SHORTEST_US / SATURATION_DN
        assert proven == pytest.approx(0.281, abs=0.001)
        # The black level that the night measured is 0.006 counts off, from the sky of 30 s.
        assert plan.exposure_us == pytest.approx(SHORTEST_US * TARGET / proven, rel=0.01)

    def test_without_the_black_level_a_saturated_long_frame_skips_while_the_sky_stays(
        self,
    ) -> None:
        """A start by day: the first long frame saturates, and the 1 ms frame alone cannot tell.

        The skip holds while the 1 ms frame shows the same sky, and any darkening takes a long
        frame again, so the skip cannot outlast the daylight.
        """
        control = controller()
        day = sky_for(2000.0)  # a long frame of 1 s would hold 2,000 times its saturation
        assert step(control, day) == SHORTEST_US  # it saturates
        assert not control.black_measured
        plan = control.plan(short_frame(day))
        assert (plan.exposure_us, plan.reason) == (None, PLAN_UNCHANGED)
        # The saturated frame bounds the black level 8 counts below the 1 ms frame, so a 1 ms frame
        # brighter by far more than the margin skips by itself.
        brighter = control.plan(short_frame(day * 1.1))
        assert (brighter.exposure_us, brighter.reason) == (None, PLAN_BRIGHT)
        darker = control.plan(short_frame(day * 0.9))
        assert (darker.exposure_us, darker.reason) == (SHORTEST_US, PLAN_PREVIOUS)

    def test_a_long_frame_beyond_the_guard_at_the_shortest_exposure_skips_the_next(
        self,
    ) -> None:
        """At 1 s the sky reads 0.85, beyond the 80% that the saturation guard of the pipeline
        accepts, and the 1 ms frame cannot resolve the sky: the next step skips."""
        control = controller()
        sky = sky_for(0.85)
        step(control, sky)
        assert control.plan(short_frame(sky)).reason == PLAN_UNCHANGED

    def test_a_long_frame_between_the_target_and_the_guard_still_serves(self) -> None:
        """At 1 s the sky reads 0.5: above the target, and the frame still serves.

        Near the target the 1 ms frame resolves no darkening, so a skip would hold for steps
        after the sky allowed the long frame again.
        """
        control = controller()
        sky = sky_for(0.5)
        step(control, sky)
        plan = control.plan(short_frame(sky))
        assert (plan.exposure_us, plan.reason) == (SHORTEST_US, PLAN_PREVIOUS)

    def test_a_clipped_long_frame_lowers_the_exposure_by_the_full_factor(self) -> None:
        """Its level is only a lower bound, so the ratio to the target says too little."""
        control = controller()
        for _ in range(4):
            step(control, DARK_SKY)  # 1, 4, 16, and 30 s
        dawn = sky_for(0.2)  # 6 times the saturation at 30 s
        short = short_frame(dawn)
        control.update(short, long_frame(dawn, LONGEST_US))
        plan = control.plan(short_frame(dawn * 0.99))  # a little darker, so no skip
        assert plan.exposure_us == round(LONGEST_US / LONG_STEP_FACTOR)


class TestTheTwilight:
    @pytest.mark.parametrize("best_s", [2.0, 7.0, 25.0])
    def test_the_background_settles_at_the_target(self, best_s: float) -> None:
        """A constant twilight sky where `best_s` puts the sky at the target.

        The black level counts as sky in the long frame (0.7% of saturation), so the background
        settles that much below the target, and the exposure 2.5% short of the best one.
        """
        sky = sky_for(TARGET / best_s)
        control = controller()
        exposures = [step(control, sky) for _ in range(8)]
        settled = exposures[-1]
        assert settled is not None
        background = long_frame(sky, settled).fraction
        assert background == pytest.approx(TARGET, abs=0.001)
        assert settled / 1e6 == pytest.approx(
            best_s * (1 - BLACK_DN / SATURATION_DN / TARGET), rel=0.01
        )
        steps = [
            later / earlier for earlier, later in itertools.pairwise(exposures) if earlier and later
        ]
        assert all(1 / LONG_STEP_FACTOR <= ratio <= LONG_STEP_FACTOR for ratio in steps)

    def test_a_dusk_never_saturates_a_long_frame_once_the_black_level_is_known(self) -> None:
        """The sky darkens by 1.5 times a step (0.44 mag), as at the steepest twilight.

        The night before measured the black level. The steps skip while the 1 ms frame shows the
        sky beyond its margin, then take the shortest long exposure, and scale up from there.
        """
        control = controller()
        step(control, DARK_SKY)
        control.reset()  # a new evening
        sky = sky_for(30.0)  # 30 times the saturation in 1 s
        exposures: list[int | None] = []
        backgrounds: list[float] = []
        while sky > DARK_SKY:
            exposure = step(control, sky)
            exposures.append(exposure)
            if exposure is not None:
                backgrounds.append(long_frame(sky, exposure).fraction)
            sky /= 1.5
        assert exposures[0] is None
        first = next(index for index, value in enumerate(exposures) if value is not None)
        assert exposures[first] == SHORTEST_US
        # The margin lets the first long frames come while the sky still passes the target at
        # 1 s: up to 16 counts of the ADC, or 4.5 times the saturation.
        assert max(backgrounds) <= 1.0
        assert sum(fraction >= CLIP for fraction in backgrounds) <= 4
        assert exposures[-1] == LONGEST_US


class TestAnEpisode:
    def test_a_pause_at_night_keeps_the_long_exposure(self) -> None:
        control = controller()
        for _ in range(4):
            step(control, DARK_SKY)
        control.reset()
        plan = control.plan(short_frame(DARK_SKY))
        assert (plan.exposure_us, plan.reason) == (LONGEST_US, PLAN_PREVIOUS)

    def test_another_sky_starts_afresh(self) -> None:
        """A 1 ms frame more than 2 counts of the ADC from the last one shows another sky."""
        control = controller()
        for _ in range(4):
            step(control, DARK_SKY)
        control.reset()
        other = DARK_SKY + (EPISODE_TOLERANCE_COUNTS + 1) * STEP_DN
        plan = control.plan(short_frame(other))
        assert (plan.exposure_us, plan.reason) == (SHORTEST_US, PLAN_FIRST)

    def test_a_change_within_the_tolerance_is_the_same_sky(self) -> None:
        control = controller()
        for _ in range(4):
            step(control, DARK_SKY)
        control.reset()
        same = DARK_SKY + EPISODE_TOLERANCE_COUNTS * STEP_DN
        assert control.plan(short_frame(same)).reason == PLAN_PREVIOUS


class TestTheBlackLevel:
    def test_two_short_frames_of_one_sky_measure_the_black_level(self) -> None:
        """A start by day: the 1 ms frame and a frame of 32 us of the same sky."""
        control = controller()
        sky = 3000.0  # counts of sky in the 1 ms frame, by day
        control.measure_black(BLACK_DN + sky, 1000, BLACK_DN + sky * 32 / 1000, 32)
        assert control.black_measured
        assert control.black_dn == pytest.approx(BLACK_DN, abs=1e-9)
        # With the black level known, the 1 ms frame of the day skips the long exposure at once.
        assert control.plan(short_frame(sky)).reason == PLAN_BRIGHT

    def test_a_skip_that_rests_on_a_few_counts_measures_the_black_level_again(self) -> None:
        """Until a pair measured it, and where the 1 ms frame skips with less than 32 counts of the
        ADC of sky. Far beyond, as by day, and below the skip, as at night, the level stands."""
        control = controller()
        assert control.needs_black(short_frame(DARK_SKY))  # not measured yet
        step(control, DARK_SKY)
        assert not control.needs_black(short_frame(DARK_SKY))  # a night: no skip
        near = (SKIP_MARGIN_COUNTS + 4) * STEP_DN  # skips by about 4 counts of the ADC
        assert control.plan(short_frame(near)).reason == PLAN_BRIGHT
        assert control.needs_black(short_frame(near))
        far = (REFRESH_COUNTS + 1) * STEP_DN
        assert control.plan(short_frame(far)).reason == PLAN_BRIGHT
        assert not control.needs_black(short_frame(far))
        assert not control.needs_black(short_frame(2 * SATURATION_DN))  # a clipped 1 ms frame

    def test_the_second_frame_must_be_the_shorter(self) -> None:
        with pytest.raises(ValueError, match="shorter"):
            controller().measure_black(500.0, 1000, 500.0, 1000)

    def test_a_saturated_long_frame_bounds_the_black_level_from_above(self) -> None:
        control = controller()
        sky = sky_for(5.0)
        short = short_frame(sky)
        control.update(short, long_frame(sky, SHORTEST_US))
        assert not control.black_measured
        assert control.black_dn is not None
        assert BLACK_DN <= control.black_dn < short.median_dn

    def test_a_saturated_1_ms_frame_leaves_the_black_level_as_it_was(self) -> None:
        control = controller()
        step(control, DARK_SKY)
        black = control.black_dn
        sky = 2 * SATURATION_DN  # the 1 ms frame clips, so its median is no level at all
        short = short_frame(sky)
        assert short.clipped
        control.update(short, long_frame(sky, SHORTEST_US))
        assert control.black_dn == black

    def test_the_black_level_never_rises_above_a_1_ms_frame(self) -> None:
        control = controller()
        control.plan(short_frame(0.0))
        assert control.black_dn == pytest.approx(BLACK_DN, abs=STEP_DN / 2)


class TestTheLimits:
    def test_no_room_to_adapt_keeps_the_longest_and_never_skips(self) -> None:
        control = SurveyExposure(
            target=TARGET,
            shortest_us=LONGEST_US,
            longest_us=LONGEST_US,
            saturation_dn=SATURATION_DN,
        )
        plan = control.plan(short_frame(sky_for(1000.0)))
        assert (plan.exposure_us, plan.reason) == (LONGEST_US, PLAN_DISABLED)

    @pytest.mark.parametrize(
        "arguments",
        [
            {"target": 0.0},
            {"target": 1.0},
            {"shortest_us": 0},
            {"shortest_us": LONGEST_US + 1},
            {"saturation_dn": 0.0},
            {"saturation_dn": math.nan},
        ],
    )
    def test_the_settings_must_make_sense(self, arguments: dict[str, float]) -> None:
        values: dict[str, float] = {
            "target": TARGET,
            "shortest_us": SHORTEST_US,
            "longest_us": LONGEST_US,
            "saturation_dn": SATURATION_DN,
        }
        values.update(arguments)
        with pytest.raises(ValueError, match="must"):
            SurveyExposure(
                target=values["target"],
                shortest_us=int(values["shortest_us"]),
                longest_us=int(values["longest_us"]),
                saturation_dn=values["saturation_dn"],
            )


# --- In the scheduler -------------------------------------------------------------------------

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")  # the Sun is 40 degrees down
# Before the dawn of January 2: the Sun is 19.1 degrees down. The gate closes 9,086 s later, when
# the fast stream at 32 us would see 50% of saturation (the Sun at -0.10 degrees).
BEFORE_DAWN = iso_to_utc_ns("2026-01-02T06:00:00Z")
GATE_CLOSES_S = 9086.0
JUNE_NOON = iso_to_utc_ns("2026-06-21T12:00:00Z")  # the Sun is 58 degrees up
ADAPTIVE = TwilightConfig()
SUNNY_LIGHT = 5.0  # clips the 1 ms frame, but the fast stream at 32 us would see only 18%


def long_frames(world: World) -> list[Frame]:
    """The long survey frames that the scheduler handed to the analysis, in order."""
    return [frame for frame in world.survey.submitted if frame.exposure_us >= SHORTEST_US]


def background(frame: Frame) -> float:
    """The median of a frame as a share of the saturation of the bin2 frames, offset included."""
    return median_dn(frame) / SATURATION_DN


def steps(world: World, start: float = 0.0) -> list[tuple[float, int | None]]:
    """Each survey step after `start`: its time, and its long exposure, or `None` for a skip."""
    found: list[tuple[float, int | None]] = []
    for frame in world.survey.submitted:
        t = world.seconds(frame.t_utc_ns)
        if t < start:
            continue
        if frame.exposure_us < SHORTEST_US:
            found.append((t, None))
        elif found:
            found[-1] = (found[-1][0], frame.exposure_us)
    return found


def set_synchronized(world: World, synchronized: bool) -> None:
    assert isinstance(world.clock, VirtualClock)
    world.clock.set_status(
        ClockStatus(synchronized=synchronized, error_bound_ns=None, source="test")
    )


@pytest.fixture(scope="module")
def floodlit() -> World:
    """A dark hour, then a floodlight that clips the 1 ms frame from 3600 s to 5400 s."""
    world = World(start_utc_ns=NIGHT, twilight=ADAPTIVE)
    world.light(3600, 5400, SUNNY_LIGHT)
    world.run_until(7200)
    world.close()
    return world


class TestTheSchedulerInTheDark:
    def test_the_first_step_probes_with_1_s_and_takes_the_long_frame_at_once(
        self, floodlit: World
    ) -> None:
        """No climb of 4 times a step: the 1 s frame measures, and 30 s follows in the same step."""
        exposures = [frame.exposure_us for frame in long_frames(floodlit)]
        assert exposures[:5] == [1_000_000, 30_000_000, 30_000_000, 30_000_000, 30_000_000]
        dark = [e for t, e in steps(floodlit) if t < 3600]
        assert set(dark) == {30_000_000}
        assert floodlit.scheduler.status().counters.survey_probes >= 1

    def test_the_activity_names_the_long_exposure_of_the_step(self) -> None:
        """The status after each step of the loop, through the first three survey steps."""
        world = World(start_utc_ns=NIGHT, twilight=ADAPTIVE)
        labels: list[str] = []
        while world.seconds(world.clock.utc_ns()) < 3 * 180:
            world.scheduler.step()
            activity = world.scheduler.status().activity
            if (
                activity is not None
                and activity.phase == "survey_long"
                and (not labels or labels[-1] != activity.label)
            ):
                labels.append(activity.label)
        world.close()
        assert labels == ["Survey step: the 1 s frame", "Survey step: the 30 s frame"]


class TestASunnyFloodlight:
    def test_the_steps_skip_their_long_exposure_and_keep_the_1_ms_frame(
        self, floodlit: World
    ) -> None:
        lit = [(t, e) for t, e in steps(floodlit) if 3600 + 200 < t < 5400]
        assert len(lit) >= 8
        assert all(exposure is None for _, exposure in lit)
        assert floodlit.scheduler.status().counters.survey_long_skips >= len(lit)

    def test_the_fast_stream_measures_on_in_the_light(self, floodlit: World) -> None:
        """The light clips the 1 ms frame, but the gate judges the fast stream: no `safe`.

        The slow fast stream of the scenario reports no background that decides, so a watch frame
        of 32 us measures what the fast stream would see. `TestTheGateJudgesTheFastStream`
        (`test_exposure.py`) shows the bursts deciding on the simulator.
        """
        assert floodlit.states_visited() == ["safe", "auto"]

    def test_the_long_frames_return_at_their_exposure_of_the_dark(self, floodlit: World) -> None:
        """After the light, the 1 ms frame shows the sky of the last long frame again."""
        after = [e for t, e in steps(floodlit) if t > 5400 + 200]
        assert after
        assert set(after) == {30_000_000}


@pytest.fixture(scope="module")
def dawn() -> World:
    """From the Sun at -19.1 degrees until the gate closes in the morning, and 15 min more."""
    world = World(start_utc_ns=BEFORE_DAWN, twilight=ADAPTIVE)
    world.run_until(GATE_CLOSES_S + 900)
    world.close()
    return world


class TestADawn:
    def test_the_long_exposure_falls_as_the_sky_brightens_and_never_clips(
        self, dawn: World
    ) -> None:
        frames = long_frames(dawn)
        later = frames[4:]  # after the ramp from the start
        assert len(later) >= 10
        for earlier, frame in itertools.pairwise(later):
            assert frame.exposure_us <= earlier.exposure_us
        # The last one passes the saturation guard (80%), and the 1 ms frame shows a brighter sky
        # after it, so the steps skip from then on. The sky brightens 1.55 times a step here, so
        # that frame does not clip, and every frame before it serves.
        assert later[-1].exposure_us == SHORTEST_US
        assert 0.8 <= background(later[-1]) < CLIP
        assert max(background(frame) for frame in frames[:-1]) < 0.8

    def test_the_steps_skip_the_long_exposure_before_the_gate_closes(self, dawn: World) -> None:
        """Fix 3: a skipped step keeps its 1 ms frame, which the daylight gate reads."""
        ((stopped_at, _, _),) = [c for c in dawn.state_changes() if c[2] == "safe"]
        skipped = [t for t, e in steps(dawn) if e is None]
        assert skipped
        assert skipped[0] < stopped_at - 600
        # Every step until the gate closed took its 1 ms frame, the last one included.
        assert max(t for t, _ in steps(dawn)) == pytest.approx(stopped_at, abs=2.0)


class TestWithoutTheSun:
    """The long exposure rests on the measured background, so neither a site nor a synchronized
    clock changes it."""

    def test_without_a_site_the_steps_skip_and_return_as_with_one(self, floodlit: World) -> None:
        world = World(start_utc_ns=NIGHT, twilight=ADAPTIVE, site=None)
        world.light(3600, 5400, SUNNY_LIGHT)
        world.run_until(7200)
        world.close()
        assert [e for _, e in steps(world)] == [e for _, e in steps(floodlit)]

    def test_with_a_clock_that_is_not_synchronized_too(self, floodlit: World) -> None:
        world = World(start_utc_ns=NIGHT, twilight=ADAPTIVE)
        world.at(0, lambda w: set_synchronized(w, False))
        world.light(3600, 5400, SUNNY_LIGHT)
        world.run_until(7200)
        world.close()
        assert [e for _, e in steps(world)] == [e for _, e in steps(floodlit)]


class TestADayInAuto:
    """The simulator's sky near the pole by day: the fast stream measures, and the 1 ms frame does
    not clip (0.21 of its saturation)."""

    @pytest.fixture(scope="class")
    @staticmethod
    def day() -> World:
        world = World(start_utc_ns=JUNE_NOON, sky=pole_sky(), twilight=ADAPTIVE)
        world.run_until(2 * 3600)
        world.close()
        return world

    def test_the_first_step_measures_the_black_level_with_a_frame_of_32_us(
        self, day: World
    ) -> None:
        """The pair of the 1 ms frame and a frame of 32 us of the same region gives the offset of
        the scenario's frames, 200 counts, within a count of the ADC (4 counts here)."""
        black = [
            call
            for call in day.configures(mode="bin2", video=False)
            if call.config.exposure_us == 32
        ]
        assert len(black) == 1  # once: the level stays
        exposure = day.scheduler._long  # the controller of the long exposure
        assert exposure.black_measured
        assert exposure.black_dn == pytest.approx(OFFSET_DN, abs=4.0)

    def test_every_step_skips_its_long_exposure_and_keeps_the_1_ms_frame(self, day: World) -> None:
        """Fix 3: the 1 ms frame keeps the gate working, and no long frame saturates by day."""
        all_steps = steps(day)
        assert len(all_steps) >= 35  # a step every 180 s for two hours
        assert all(exposure is None for _, exposure in all_steps)
        assert long_frames(day) == []
        status = day.scheduler.status()
        assert status.counters.survey_long_skips == len(all_steps)
        assert status.state == "auto"
