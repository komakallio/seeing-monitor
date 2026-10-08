"""The adaptive exposures: the ones that keep the sky background of a frame at a target.

Two streams adapt: the fast stream, and the long frame of each survey step.

**The fast stream.** A 2 ms bin1 frame at gain 0 saturates its sky background while the Sun is
still a few degrees up, long before the sky hides Polaris (see `docs/research-notes.md`, "Polaris
in a bright sky"). So the fast stream shortens its exposure as the sky brightens. Before each fast
period and each search burst, the scheduler takes the sky background of the previous window or
burst as a share of saturation, and it picks the exposure that puts the background at
`[scheduler.fast] target_background_fraction`. The background grows in proportion to the exposure,
so the rule is one ratio, clamped between the profile's shortest exposure and
`[scheduler.fast] exposure_us`. In a dark sky the rule asks for more than the longest exposure, so
the stream runs at `exposure_us` all night.

**Between windows, never within one.** The exposure changes only where a stream starts. A fast
period holds whole analysis windows on one stream, so the scheduler adapts at the start of each
period, and every window has one exposure. The fast analyzer derives the exposure correction of
the seeing for each stream (`exposure_correction_factor` in the window record).

**The offset.** The background of a frame includes the camera's offset (the black level). The
profile does not know the offset, because the camera reports it at run time, so the scheduler
counts it as sky, as the daylight gate does (`seeingmon.scheduler.gates.read_sky`). With the
simulator's offset that is 0.7% of saturation in a bin1 frame, so the background settles that much
below the target, and the exposure about 2% short of it.

**The long survey frame.** A long frame of 30 s at gain 120 saturates its sky in twilight, long
before the sky hides its stars. `SurveyExposure` picks the long exposure of each survey step that
puts the background at `[survey.twilight] target_background_fraction` (0.3) of saturation, between
`[survey.twilight] min_exposure_s` (1 s) and `[scheduler.survey] long_exposure_s` (30 s). It rests
on the measured background alone, not on the Sun, so it works without a site and with a clock that
is not synchronized. The long frame of the step before gives the exposure, by the same ratio as the
fast stream, changing by at most 4 times a step. A frame that clipped gives only a lower bound,
so the exposure falls by the full 4 times. In a dark sky the rule asks for more than
`long_exposure_s`, so the long frame stays there.

**The 1 ms frame.** The first long frame after a start has no long frame before it, and neither
has the first of an episode of `auto` whose 1 ms frame shows another sky than the last long frame
did (by more than 2 counts of the ADC). The 1 ms frame of the step (gain 0) speaks for it. Its sky
converts to the long frame through the exposures and the conversion gains: at the defaults, one
count of the ADC of sky in the 1 ms frame is 28% of saturation in a long frame of 1 s, about the
target. The median of the 1 ms frame resolves one count, and its level holds the camera's black
level (about 120 counts in bin2 with the simulator's offset), so the 1 ms frame cannot place the
exposure between 1 and 30 s. It shows only a sky that is far too bright for the shortest long
exposure. Such a first long frame therefore takes `min_exposure_s` unless the 1 ms frame resolves
its sky, and the long frames that follow scale up by at most 4 times a step (1, 4, 16, and 30 s in
a dark sky). A pause at night keeps the long exposure, because the 1 ms frame shows the same sky
before and after. Whenever the 1 ms frame shows a sky beyond the margin below, that sky also caps
the long exposure, as when the Moon rises after a dark hour.

**The black level.** To read the sky of the 1 ms frame, `SurveyExposure` learns the black level
from the frames:

- The black level is never above the median of a 1 ms frame, because the sky is never negative.
- The first survey step after a start, whose 1 ms frame does not clip, takes a second short frame
  at `[scheduler.watch] bright_exposure_us` (32 us) of the central region of the watch. The two
  frames see the same sky through two exposures, with the same black level, so the pair measures
  it, whatever the sky (`SurveyExposure.measure_black`). A step whose skip rests on a few counts
  (less than 32 counts of the ADC of sky) takes the pair again (`SurveyExposure.needs_black`).
- A step whose long frame does not clip measures it again: the long frame and the 1 ms frame see
  the same sky through known exposures and gains, and the same black level, because the camera's
  offset is a number of counts that does not depend on the gain. That holds for the simulator. A
  long frame that clips bounds it from above.

**The skip in daylight.** When even `min_exposure_s` would pass the target, the step skips its long
exposure and keeps its 1 ms frame, which the daylight gate reads. Two rules decide it:

- The 1 ms frame shows a sky above the black level that would put the long frame at
  `min_exposure_s` above the target. Only the sky beyond a margin of 4 counts of the ADC counts,
  because the median of each frame of a pair rounds to a count. At the defaults the long frames
  therefore start where the 1 ms frame shows about 5 counts of sky, which puts a long frame of 1 s
  at 1.4 times its saturation, so a dusk costs a frame or two that clip.
- The long frame of the step before was, or would be at `min_exposure_s`, beyond what the
  saturation guard of the pipeline accepts (`[survey.twilight] max_background_fraction`, 80% of
  saturation), and the 1 ms frame has not darkened since. Any darkening takes a long frame again,
  so this rule never holds a skip once the sky falls. A long frame at `min_exposure_s` between the
  target and the guard still serves, and near the target the 1 ms frame cannot resolve the
  darkening that would end a skip, so such a frame does not skip the next.

Until the pair of short frames measures the black level, which a 1 ms frame that clips postpones,
only the second rule skips. A sky that darkens after a start by day then costs a long frame at
`min_exposure_s` in each step whose 1 ms frame darkened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def background_fraction(
    background_dn: float, saturation_dn: float, offset_dn: float = 0.0
) -> float:
    """The sky background of a frame as a share of saturation: the level above the offset.

    `background_dn` and `offset_dn` are in the counts that the frame carries, and `saturation_dn`
    is the saturation level of its readout mode and gain in the same counts (the profile's level
    ignores the offset). A level at or below the offset gives 0.
    """
    if not saturation_dn > 0.0:
        raise ValueError("saturation_dn must be positive")
    return max(background_dn - offset_dn, 0.0) / saturation_dn


def adapted_exposure_us(
    exposure_us: float,
    fraction: float,
    *,
    target: float,
    shortest_us: int,
    longest_us: int,
) -> int:
    """The exposure that moves a background of `fraction` at `exposure_us` to `target`.

    The result lies between `shortest_us` and `longest_us`. A background of 0, which a dark sky
    above the offset gives, or a value that is not a number, gives `longest_us`.
    """
    if shortest_us > longest_us:
        raise ValueError("shortest_us must not exceed longest_us")
    if not (fraction > 0.0 and math.isfinite(fraction)):
        return longest_us
    wanted = exposure_us * target / fraction
    if not math.isfinite(wanted):
        return longest_us
    return int(min(longest_us, max(shortest_us, round(wanted))))


# The longest exposure divided by the shortest that one step of the long survey frame may take.
LONG_STEP_FACTOR = 4.0
# The sky of a 1 ms frame must stand this many counts of the ADC above the black level before the
# frame alone skips the long exposure: the median of each frame of a pair rounds to a count.
SKIP_MARGIN_COUNTS = 4
# A step whose 1 ms frame skips the long exposure with less sky than this many counts of the ADC
# above the black level measures the black level again, because its decision rests on a few counts.
REFRESH_COUNTS = 32
# At the start of an episode of `auto`, a 1 ms frame within this many counts of the ADC of the one
# of the last long frame shows the same sky, so the last long exposure carries on.
EPISODE_TOLERANCE_COUNTS = 2

# Why `SurveyExposure.plan` chose its exposure, or skipped.
PLAN_BRIGHT = "bright"  # the 1 ms frame shows a sky too bright for the shortest long exposure
PLAN_UNCHANGED = "unchanged"  # the last long frame was too bright, and the sky has not darkened
PLAN_PREVIOUS = "previous"  # scaled from the long frame of the step before
PLAN_FIRST = "first"  # the first long frame of the episode: the shortest, or the 1 ms frame's
PLAN_DISABLED = "disabled"  # no room to adapt: the shortest long exposure is the longest


@dataclass(frozen=True, slots=True)
class ShortFrame:
    """What the 1 ms frame of a survey step tells the long exposure.

    `median_dn` is the median of the frame in the counts of its container, and `clipped` says that
    it reached the clip level. `long_dn_per_dn_us` converts its sky to the long frame: the counts
    that one count of sky here gives in the long frame for each microsecond of the long exposure.
    `step_dn` is one count of the ADC in the counts of the container, which is how finely the
    median resolves the sky.
    """

    median_dn: float
    clipped: bool
    long_dn_per_dn_us: float
    step_dn: float = 1.0


@dataclass(frozen=True, slots=True)
class LongFrame:
    """What the long frame of a survey step showed: its exposure, and its median against saturation.

    The median is in the counts of the container of the 1 ms frame of the same step, which uses the
    same readout mode and pixel format.
    """

    exposure_us: int
    median_dn: float
    saturation_dn: float
    clipped: bool

    @property
    def fraction(self) -> float:
        """The background as a share of saturation, with the black level counted as sky."""
        return self.median_dn / self.saturation_dn


@dataclass(frozen=True, slots=True)
class LongPlan:
    """The long exposure of a survey step: `exposure_us`, or `None` to skip it, and why."""

    exposure_us: int | None
    reason: str


class SurveyExposure:
    """The adaptive long exposure of the survey step. See the module text.

    Call `plan` with the 1 ms frame of each step, and `update` with both frames when the long
    frame was taken. Call `reset` at the start of an episode of `auto`: the sky may have changed
    while the scheduler was out of `auto`, so the first step of the episode keeps the long frame of
    the last step only when its 1 ms frame shows the same sky, within `EPISODE_TOLERANCE_COUNTS`
    counts of the ADC. A pause at night then keeps the long exposure, and a new evening starts
    afresh. The black level stays, because it belongs to the camera.
    """

    def __init__(
        self,
        *,
        target: float,
        shortest_us: int,
        longest_us: int,
        saturation_dn: float,
        usable_fraction: float = 0.8,
    ) -> None:
        if not 0.0 < target < 1.0:
            raise ValueError("target must lie between 0 and 1")
        if not target <= usable_fraction <= 1.0:
            raise ValueError("usable_fraction must lie between the target and 1")
        if not 0 < shortest_us <= longest_us:
            raise ValueError("shortest_us must be positive and not above longest_us")
        if not saturation_dn > 0.0:
            raise ValueError("saturation_dn must be positive")
        self._target = target
        self._shortest_us = shortest_us
        self._longest_us = longest_us
        self._saturation_dn = saturation_dn
        self._usable = usable_fraction
        self._black_dn: float | None = None
        self._black_measured = False
        self._previous: tuple[ShortFrame, LongFrame] | None = None
        self._episode_start = False

    @property
    def black_dn(self) -> float | None:
        """The black level of the 1 ms frame that the frames allow, at most, or `None`."""
        return self._black_dn

    @property
    def black_measured(self) -> bool:
        """Whether a pair of frames measured the black level, and did not only bound it."""
        return self._black_measured

    @property
    def adaptive(self) -> bool:
        """Whether the long exposure has room to adapt: the shortest is below the longest."""
        return self._shortest_us < self._longest_us

    def measure_black(
        self, short_dn: float, short_us: float, bright_dn: float, bright_us: float
    ) -> None:
        """Measure the black level from the 1 ms frame and a shorter one of the same sky.

        Both frames take one readout mode, gain, and region, so they hold the same black level `b`
        and a sky in proportion to their exposures: `short = b + s` and `bright = b + r s`, with
        `r` the ratio of the exposures. Any sky gives `b`, also a dark one, as long as neither
        frame clipped. The median of each rounds to a whole count of the ADC, so `b` comes out
        within about one count.
        """
        ratio = bright_us / short_us
        if not 0.0 < ratio < 1.0:
            raise ValueError("bright_us must be shorter than short_us")
        black = (bright_dn - ratio * short_dn) / (1.0 - ratio)
        self._black_dn = min(black, bright_dn, short_dn)
        self._black_measured = True

    def reset(self) -> None:
        """Start an episode: the next step keeps the last long frame only for the same sky."""
        self._episode_start = True

    def next_estimate_us(self) -> int:
        """The exposure of the next long frame if the sky stays as it was, for the status."""
        if self._previous is None:
            return self._shortest_us
        return self._from_previous(self._previous[1])

    def needs_black(self, short: ShortFrame) -> bool:
        """Whether the step should measure the black level with a pair of short frames.

        It should until a pair has measured it, and again where a skip rests on a few counts: the
        1 ms frame shows a sky that skips the long exposure, but less than `REFRESH_COUNTS` counts
        of the ADC of it. A drift of the black level then cannot hold the skip into the night. A
        1 ms frame that clipped gives no level.
        """
        if not self.adaptive or short.clipped:
            return False
        black = self._black_dn
        if not self._black_measured or black is None:
            return True
        sky = short.median_dn - black
        return sky < REFRESH_COUNTS * short.step_dn and self._proves_bright(short, black)

    def _proves_bright(self, short: ShortFrame, black: float) -> bool:
        """Whether the sky beyond the margin puts a long frame at the shortest above the target."""
        at_least = max(0.0, short.median_dn - black - SKIP_MARGIN_COUNTS * short.step_dn)
        return self._long_fraction(short, at_least, self._shortest_us) > self._target

    def plan(self, short: ShortFrame) -> LongPlan:
        """The long exposure of a step whose 1 ms frame is `short`, or a skip."""
        self._note_black(short.median_dn)
        if self._episode_start:
            self._episode_start = False
            previous = self._previous
            tolerance = EPISODE_TOLERANCE_COUNTS * short.step_dn
            if previous is not None and abs(short.median_dn - previous[0].median_dn) > tolerance:
                self._previous = None
        if self._shortest_us >= self._longest_us:
            return LongPlan(self._longest_us, PLAN_DISABLED)
        black = self._black_dn if self._black_dn is not None else short.median_dn
        margin = SKIP_MARGIN_COUNTS * short.step_dn
        if self._proven_fraction(short) > self._target:
            return LongPlan(None, PLAN_BRIGHT)
        # The sky that the 1 ms frame proves caps the exposure, as when the Moon rises at night.
        cap = self._cap_us(short)
        previous = self._previous
        if previous is not None:
            before, long = previous
            at_shortest = long.fraction * self._shortest_us / long.exposure_us
            if at_shortest >= self._usable and short.median_dn >= before.median_dn:
                return LongPlan(None, PLAN_UNCHANGED)
            return LongPlan(min(cap, self._from_previous(long)), PLAN_PREVIOUS)
        if not self._black_measured:  # the 1 ms frame cannot tell its sky from the black level
            return LongPlan(self._shortest_us, PLAN_FIRST)
        at_most = max(0.0, short.median_dn - black + margin)
        fraction = self._long_fraction(short, at_most, self._shortest_us)
        exposure = adapted_exposure_us(
            self._shortest_us,
            fraction,
            target=self._target,
            shortest_us=self._shortest_us,
            longest_us=self._longest_us,
        )
        return LongPlan(exposure, PLAN_FIRST)

    def _proven_fraction(self, short: ShortFrame) -> float:
        """The background that the proven sky of the 1 ms frame gives a frame at the shortest."""
        black = self._black_dn if self._black_dn is not None else short.median_dn
        margin = SKIP_MARGIN_COUNTS * short.step_dn
        at_least = max(0.0, short.median_dn - black - margin)
        return self._long_fraction(short, at_least, self._shortest_us)

    def _cap_us(self, short: ShortFrame) -> int:
        """The longest exposure that the sky proven by the 1 ms frame allows."""
        proven = self._proven_fraction(short)
        if proven > 0.0:
            return max(self._shortest_us, round(self._shortest_us * self._target / proven))
        return self._longest_us

    def jump_us(self, short: ShortFrame) -> int | None:
        """The exposure that the long frame just taken asks for, with no limit on the step.

        The first long frame after a start is the shortest one, and it measures the sky: the
        background grows in proportion to the exposure, so a frame that did not clip gives the
        exposure of the target at once. The scheduler takes the real long frame in the same step,
        instead of climbing by `LONG_STEP_FACTOR` once a step. The sky that the 1 ms frame proves
        still caps the result. `None` when there is no long frame, or when it clipped, which leaves
        the steps to their usual rule.
        """
        if self._previous is None:
            return None
        long = self._previous[1]
        if long.clipped:
            return None
        wanted = adapted_exposure_us(
            long.exposure_us,
            long.fraction,
            target=self._target,
            shortest_us=self._shortest_us,
            longest_us=self._longest_us,
        )
        return min(self._cap_us(short), wanted)

    def update(self, short: ShortFrame, long: LongFrame) -> None:
        """Take the long frame of a step and its 1 ms frame: the next step scales from them.

        The pair also measures the black level, or bounds it when the long frame clipped. The two
        frames hold the same black level `b` and the same sky, which gives `k` counts in the long
        frame for each count in the 1 ms frame: `short = b + s` and `long = b + k s`.
        """
        self._note_black(short.median_dn)
        self._previous = (short, long)
        k = long.exposure_us * short.long_dn_per_dn_us
        if short.clipped or k <= 1.0:  # a clipped 1 ms frame, or a long frame with no more sky
            return
        black = (k * short.median_dn - long.median_dn) / (k - 1.0)
        black = min(black, short.median_dn)
        if long.clipped:  # the true level is higher, so the black level is lower than this
            self._note_black(black)
        else:
            self._black_dn = black
            self._black_measured = True

    def _note_black(self, upper_dn: float) -> None:
        if self._black_dn is None or upper_dn < self._black_dn:
            self._black_dn = upper_dn

    def _long_fraction(self, short: ShortFrame, sky_dn: float, exposure_us: int) -> float:
        """The background that `sky_dn` counts of sky in the 1 ms frame give a long frame."""
        return sky_dn * short.long_dn_per_dn_us * exposure_us / self._saturation_dn

    def _from_previous(self, long: LongFrame) -> int:
        """The exposure that scales the background of `long` to the target, in one step."""
        if long.clipped:
            wanted = long.exposure_us / LONG_STEP_FACTOR
        else:
            wanted = adapted_exposure_us(
                long.exposure_us,
                long.fraction,
                target=self._target,
                shortest_us=1,
                longest_us=max(self._longest_us, long.exposure_us) * round(LONG_STEP_FACTOR),
            )
        step_low = long.exposure_us / LONG_STEP_FACTOR
        step_high = long.exposure_us * LONG_STEP_FACTOR
        wanted = min(max(wanted, step_low), step_high)
        return int(min(self._longest_us, max(self._shortest_us, round(wanted))))
