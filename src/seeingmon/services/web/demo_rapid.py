"""The rapid focus mode of the demo: a person who turns the focuser through focus.

While the fake `core` of the demo plays the mode, it serves what the real `core` serves: the
readings of the star width in arcseconds, 20 a second, in the state, and the video of the star in
the same messages. This module makes the readings. `demo` makes the video from the same numbers.

**The curve.** The person sweeps the focuser through focus and back (`RAPID_SWEEP_S` seconds for a
pass in each direction), and narrows the sweep each time (`RAPID_AMPLITUDES`), so the width shows a
V that gets narrower. After the narrowest sweep the person loses focus and starts again. The width
is the width of the star in focus (`RAPID_BEST_ARCSEC`, the width that the video of the demo shows)
with the blur of the defocus added in quadrature, which gives the hyperbola of a real focus run.

**The readings.** Each reading scatters by `RAPID_SCATTER`. Every `RAPID_JOLT_EVERY` readings a tap
on the telescope inflates two readings in a row by `RAPID_JOLT_FACTOR`, and the spike rule of
`core` flags them (the demo uses the history of `core`, so the spike flag, the best value, and the
numbering come out as they do there). A reading holds four frames, and every tenth one holds five,
because the camera makes 82 frames a second. A long exposure or a high gain makes the star
saturate, and the readings then carry the flag `saturated`, as a real run does. The numbers are
made up. They show a plausible curve, and they measure nothing.

**The offer.** `offer_view` builds the state of the mode when it does not run: whether it is
offered, what is missing, the coarse focus, and why the last run ended. The first alignment frames
do not offer the mode, because the coarse focus needs five focus values, as in `core`.
"""

from __future__ import annotations

import math
import random
import threading
from collections.abc import Callable

from seeingmon.clock import NS_PER_S, Clock, utc_ns_to_iso
from seeingmon.frames import Roi
from seeingmon.services.core.alignment.rapid import (
    READING_INTERVAL_S,
    RapidAvailability,
    RapidColumns,
    RapidFacts,
    RapidHistory,
    RapidSnapshot,
    build_view,
)
from seeingmon.services.core.alignment.rapid_availability import (
    COARSE_READINGS,
    CoarseFocus,
    coarse_problem,
)
from seeingmon.services.core.settings import AlignmentSettings
from seeingmon.services.web.contract import RapidFocusView, RapidLocatedBy

RAPID_BEST_ARCSEC = 3.0  # the width of the star in focus
RAPID_BLUR_ARCSEC = 11.0  # the blur that one unit of defocus adds, in quadrature
RAPID_SWEEP_S = 28.0
RAPID_AMPLITUDES = (1.0, 0.7, 0.45, 0.28, 0.16, 0.1)  # the sweeps, in units of defocus
RAPID_SCATTER = 0.05
RAPID_JOLT_EVERY = 211
RAPID_JOLT_AFTER = 30  # the readings before the first tap
RAPID_JOLT_FACTOR = 2.8
RAPID_FRAMES = 4  # the frames of a reading; every tenth reading has one more
RAPID_MAX_FWHM_ARCSEC = AlignmentSettings().rapid_focus_max_fwhm_arcsec

# What a star looks like for a width and a setting of the camera: the share of the full scale that
# its brightest pixel reaches, and whether the pixel saturates. `demo` knows the star.
StarFor = Callable[[float, int, int], tuple[float, bool]]


def focuser_offset(t_s: float) -> float:
    """The place of the focuser relative to focus at `t_s` seconds into a run, in units of defocus.

    A pass goes from one side of focus to the other and back, so it crosses focus twice. The
    amplitude of the pass falls from sweep to sweep and then starts again.
    """
    sweeps, phase = divmod(t_s / RAPID_SWEEP_S, 1.0)
    amplitude = RAPID_AMPLITUDES[int(sweeps) % len(RAPID_AMPLITUDES)]
    return amplitude * (4.0 * abs(phase - 0.5) - 1.0)


def true_width_arcsec(t_s: float) -> float:
    """The width of the star at `t_s` seconds into a run, without the scatter of a reading."""
    return math.hypot(RAPID_BEST_ARCSEC, RAPID_BLUR_ARCSEC * focuser_offset(t_s))


def reading_width_arcsec(session: int, number: int) -> float:
    """The width that reading `number` of run `session` shows: the star, its scatter, and a tap."""
    rng = random.Random(session * 1_000_003 + number)
    width = true_width_arcsec(number * READING_INTERVAL_S) * (
        1.0 + RAPID_SCATTER * rng.gauss(0.0, 1.0)
    )
    if number > RAPID_JOLT_AFTER and number % RAPID_JOLT_EVERY in (0, 1):
        width *= RAPID_JOLT_FACTOR
    return width


class RapidDemo:
    """The runs of the rapid focus mode in the fake `core`. Safe to call from any thread.

    `begin` starts a run and `end` ends it. While a run is active, `view` makes the readings up to
    now, at 20 a second of the clock, and returns the state of the mode. `star` says how bright
    the star is for a width and a setting, and `roi` and `scale_arcsec_px` describe the stream.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        mode: str,
        roi: Roi,
        scale_arcsec_px: float,
        star: StarFor,
    ) -> None:
        self._clock = clock
        self._facts = (mode, roi, scale_arcsec_px)
        self._star = star
        self._lock = threading.Lock()
        self._history = RapidHistory()
        self._active = False
        self._ended: str | None = None
        self._since_utc: str | None = None
        self._start_mono = 0
        self._start_utc_ns = 0
        self._exposure_us = 0
        self._gain = 0
        self._made = 0  # the readings of the run so far

    @property
    def active(self) -> bool:
        """Whether a run is active."""
        return self._active

    @property
    def ended(self) -> str | None:
        """Why the last run ended, or `None` while a run is active and before the first one."""
        return self._ended

    @property
    def settings(self) -> tuple[int, int]:
        """The exposure and the gain of the run (of the last run, when none is active)."""
        return self._exposure_us, self._gain

    def begin(self, exposure_us: int, gain: int) -> None:
        """Start a run: no readings yet, a new session number, and the time of the clock."""
        with self._lock:
            self._history.clear()
            self._active = True
            self._ended = None
            self._start_mono = self._clock.monotonic_ns()
            self._start_utc_ns = self._clock.utc_ns()
            self._since_utc = utc_ns_to_iso(self._start_utc_ns, digits=3)
            self._exposure_us, self._gain = exposure_us, gain
            self._made = 0

    def configure(self, exposure_us: int, gain: int) -> None:
        """Change the exposure and the gain inside the run. The readings go on."""
        with self._lock:
            self._catch_up()  # the readings so far belong to the old settings
            self._exposure_us, self._gain = exposure_us, gain

    def end(self, reason: str) -> None:
        """End the run and keep the reason. Ending a run that does not run changes nothing."""
        with self._lock:
            if self._active:
                self._active = False
                self._ended = reason

    def reset_best(self) -> None:
        """Restart the best value. The readings stay."""
        with self._lock:
            self._history.reset_best()

    def elapsed_s(self) -> float:
        """The seconds since the run began."""
        return max(0.0, (self._clock.monotonic_ns() - self._start_mono) / NS_PER_S)

    def width_arcsec(self) -> float:
        """The width of the star now, for the picture: the curve without the scatter."""
        return true_width_arcsec(self.elapsed_s())

    def view(self) -> RapidFocusView | None:
        """The state of the mode with the readings up to now, or `None` while no run is active."""
        with self._lock:
            if not self._active:
                return None
            self._catch_up()
            mode, roi, scale = self._facts
            snapshot = RapidSnapshot(
                session=self._history.session,
                active=True,
                since_utc=self._since_utc,
                ended=None,
                columns=self._history.columns(),
                best_arcsec=self._history.best_arcsec,
                star_found=True,
                facts=RapidFacts(mode, self._exposure_us, self._gain, roi, scale),
            )
        return build_view(snapshot)

    def _catch_up(self) -> None:
        """Make the readings up to the clock. The caller holds the lock."""
        if not self._active:
            return
        last = int(self.elapsed_s() / READING_INTERVAL_S)
        session = self._history.session
        while self._made < last:
            self._made += 1
            number = self._made
            width = reading_width_arcsec(session, number)
            peak, saturated = self._star(
                true_width_arcsec(number * READING_INTERVAL_S), self._exposure_us, self._gain
            )
            self._history.add(
                self._start_utc_ns + number * round(READING_INTERVAL_S * NS_PER_S),
                width,
                peak,
                RAPID_FRAMES + (1 if number % 10 == 0 else 0),
                saturated,
            )


def offer_problem(frames: int) -> str | None:
    """Why the mode is not offered after `frames` alignment frames, or `None` when it is.

    The demo has stars of one width that the limit lets through, so the only thing that can be
    missing is the coarse focus: it needs `COARSE_READINGS` focus values.
    """
    coarse = CoarseFocus(None, None, min(frames, COARSE_READINGS - 1))
    if frames >= COARSE_READINGS:
        return None
    return coarse_problem(coarse, 3.82, RAPID_MAX_FWHM_ARCSEC)


def offer_view(
    *,
    frames: int,
    coarse_fwhm_arcsec: float,
    located_by: RapidLocatedBy,
    ended: str | None,
) -> RapidFocusView:
    """The state of the mode while it does not run, `frames` frames into the alignment.

    `coarse_fwhm_arcsec` is the median of the last five focus values, `located_by` says what
    places Polaris, and `ended` is the reason that the last run ended, or `None`.
    """
    reason = offer_problem(frames)
    availability = RapidAvailability(
        available=reason is None,
        reason=reason,
        located_by=located_by if reason is None else None,
        coarse_fwhm_arcsec=None if frames < COARSE_READINGS else round(coarse_fwhm_arcsec, 3),
        max_fwhm_arcsec=RAPID_MAX_FWHM_ARCSEC,
        center=None,
    )
    snapshot = RapidSnapshot(
        session=0,
        active=False,
        since_utc=None,
        ended=ended,
        columns=RapidColumns([], [], [], [], [], [], []),
        best_arcsec=None,
        star_found=False,
        facts=None,
    )
    return build_view(snapshot, availability)
