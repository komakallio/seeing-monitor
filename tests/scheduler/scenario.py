"""A scripted world for the scheduler tests: sky light, clouds, a star, faults, and commands.

`World` wires a `Scheduler` to the fakes in `seeingmon.testing`, on a `VirtualClock`. A script
describes the world as functions of time, and the frame factory of the fake camera renders it:

- **Sky light.** The background follows the Sun's elevation at a synthetic site (latitude 55
  degrees north, longitude 0, which is nobody's real site), through a brightness curve. The
  default curve (`saturating_sky`) is too bright by day even for the fast stream at its shortest
  exposure, so the scheduler waits in `safe` and enters `auto` at about -0.4 degrees, while the
  1 ms brightness frame still clips. `pole_sky` is the simulator's sky near the pole, which never
  saturates it, for a day in `auto`. `light` adds a floodlight on top.
- **Polaris.** The fake fast analysis takes the SNR of the star from the truth of the world: the
  centroid aperture's formula of the detection estimate in `docs/research-notes.md` ("Polaris in
  a bright sky") for a 2 ms bin1 frame, with the sky of the curve and the transparency of the
  clouds. Polaris is detectable (an SNR of 10) while the sky at 1 ms in bin2 stays below about
  0.29 of saturation. The real fast path decides by the matched SNR, which is about five times
  higher in a bright sky, and the scenarios keep the lower SNR so that a sky in `auto` can hide
  Polaris.
- **Clouds.** `cloud` sets the cloud fraction that the survey analysis reports and dims the star.
  A cloud of 0.8 leaves Polaris at an SNR of about 40 in a dark sky, so it stays detectable.
- **The star.** Polaris sits near the middle of the bin1 sensor and drifts 0.087 pixels a second.
  `hide_star` removes it, and `jolt` shifts it, as a bumped mount would.
- **Faults.** `camera_fault` makes every read time out. A recovery step of a given level, or an
  escalation, clears it.
- **A lost camera.** `camera_gone` makes every read time out, as a camera does that nobody unplugged
  yet, and the recovery steps fail as they do on a camera that is gone: the restart of the capture
  fails with an SDK error, the reopen finds no camera, and the USB reset finds no device. Nothing
  but the end of the window brings it back.
- **Commands.** `at` runs any action at a time, such as `world.scheduler.submit(Pause())`.

The scenario uses a small bin2 frame and a slow, small fast stream (one frame in 2 s on a
32-pixel ROI), because the point is the scheduling and not the pixels. That keeps a 12 hour
night to a few seconds of real time. The slow stream stands for the real one of 2 ms, so the
world cannot render a shorter exposure of it, and the adaptive exposure is off
(`target_background_fraction = 0`). `test_exposure.py` tests the adaptive exposure on the
simulator's sky. The long survey frame keeps its 30 s too (`SCENARIO_TWILIGHT`), so that the
timing of every scenario stays that of a dark sky. Pass `twilight=TwilightConfig()` for the
adaptive long exposure and the skip in daylight (`test_survey_exposure.py`).
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from seeingmon.analysis import FastContext, SurveyOutput
from seeingmon.clock import NS_PER_S, Clock, VirtualClock, iso_to_utc_ns
from seeingmon.drivers.base import (
    CameraDisconnectedError,
    CameraError,
    CameraInfo,
    CameraTimeoutError,
    RecoveryLevel,
)
from seeingmon.drivers.sim.sky import DAYLIGHT_SKY_MAG_ARCSEC2, sky_brightness_mag_arcsec2
from seeingmon.drivers.sim.stars import POLARIS_MAG
from seeingmon.frames import (
    ActiveStream,
    Frame,
    FrameData,
    FrameFlag,
    Roi,
    StreamConfig,
    StreamKind,
    TimeQuality,
)
from seeingmon.profile import load_profile
from seeingmon.records import (
    EventRecord,
    PointingRecord,
    Record,
    SeeingWindowRecord,
    SkyQualityRecord,
)
from seeingmon.scheduler import CommissionResult, Scheduler, SchedulerConfig, SiteConfig
from seeingmon.scheduler.config import FastConfig, LoopConfig, SearchConfig, SurveyConfig
from seeingmon.scheduler.ephemeris import sun_elevation_deg
from seeingmon.scheduler.levels import EscalationLevel
from seeingmon.survey.config import TwilightConfig
from seeingmon.testing import (
    FakeCameraDriver,
    FakeFastAnalyzer,
    FakeFocusSink,
    FakePointingProvider,
    FakeSurveyAnalyzer,
    ListRecordWriter,
)

PROFILE = load_profile("asi294mm-gs250")

# A synthetic site, 55 degrees north on the prime meridian. It is nobody's real site.
SITE = SiteConfig(latitude_deg=55.0, longitude_deg=0.0)

# 14:30 UTC on 2026-01-01: the Sun is 6 degrees up, sets at about 15:55, and astronomical night
# (the Sun at -18 degrees) begins at about 18:10.
START = iso_to_utc_ns("2026-01-01T14:30:00Z")

STAR_BIN1 = (4144.0, 2822.0)  # Polaris on the bin1 sensor at the start
STAR_DRIFT_PX_PER_S = (0.087, 0.0)  # 10 arcsec in 60 s, at 1.91 arcsec per bin1 pixel
STAR_PEAK_DN = 5000
OFFSET_DN = 200  # the bias level of the frames
SATURATION_DN = 65_532  # the bin2 saturation level in 16-bit counts, from the profile
NIGHT_SKY_FRACTION = 2e-8  # the brightness of a dark sky at 1 ms, as a share of saturation
REAL_FAST_EXPOSURE_MS = 2.0
MAX_REAL_EXPOSURE_US = 100_000  # a longer fast exposure is the slow stream of the test
SMALL_BIN2 = (640, 480)  # the full bin2 frame of the fake camera, much smaller than the real one

# The SNR of Polaris in a fast frame: the inputs of the detection estimate in
# `docs/research-notes.md` ("Polaris in a bright sky"), for bin1 at gain 0, through the fast path's
# centroid aperture, which holds 97% of the star on 201 square pixels. It is the estimate's lower
# column, not the matched SNR that the real fast path decides by: the matched SNR stays above 10
# up to a sky that clips the brightness frame, and the scenarios need skies in `auto` that hide
# Polaris. The fake analysis reports this SNR as its matched SNR.
APERTURE_FRACTION = 0.97
APERTURE_PX2 = 201.0
POLARIS_E_PER_MS = PROFILE.star_electron_rate_e_per_s(POLARIS_MAG) * 1e-3
BIN1_PIXEL_VAR_E2 = PROFILE.read_noise_e("bin1", 0) ** 2 + PROFILE.e_per_adu("bin1", 0) ** 2 / 12
BIN2_FULL_WELL_E = PROFILE.saturation("bin2", 0).full_well_e
BIN1_PER_BIN2_AREA = (PROFILE.mode("bin1").pixel_size_um / PROFILE.mode("bin2").pixel_size_um) ** 2
MIN_STAR_SNR = 6.0  # the fast path's `min_star_snr`: a weaker star counts as missing

TEST_CONFIG = SchedulerConfig(
    fast=FastConfig(
        exposure_us=2_000_000,
        roi_arcmin=1.0,  # 32 pixels in bin1
        roi_edge_margin_px=4.0,
        missing_star_frames=10,
        target_background_fraction=0.0,  # the slow stream keeps its exposure (see above)
    ),
    # A burst of 3 frames of 2 s takes 6 s, which fits the interval of 15 s. The default has no
    # Sun limit, and the scenarios set one, so that they cover the probe bursts above it.
    search=SearchConfig(burst_frames=3, max_sun_elevation_deg=12.0),
    loop=LoopConfig(max_sleep_s=5.0),
    survey=SurveyConfig(long_exposure_s=30.0),  # the scenarios plan around the old 30 s
)

# A sky of 20 times the saturation of the 1 ms frame: the fast stream at 32 us would see 74% of
# saturation, so nothing can be measured. The default sky stops there by day, and a floodlight of
# this level forces `safe`. A floodlight of 1 clips the 1 ms frame, but the fast stream at 32 us
# would see only 3.7%, so it measures on.
BLINDING_LIGHT = 20.0

# The shortest adaptive long exposure is the longest one (30 s), so the long frame never adapts and
# never skips.
SCENARIO_TWILIGHT = TwilightConfig(min_exposure_s=30.0)

SkyCurve = Callable[[float], float]
"""The sky at 1 ms and gain 0 in bin2, as a share of saturation, against the Sun's elevation."""


def saturating_sky(elevation: float) -> float:
    """The default sky: 0.5 at -3 degrees, and 3 times brighter for each degree that the Sun rises.

    It stops at `BLINDING_LIGHT`, a daylight that the fast stream cannot take even at its shortest
    exposure, so by day the scheduler waits in `safe`. The 1 ms brightness frame clips (0.9 of its
    saturation) above about -2.5 degrees. The fast stream at 32 us would see 35% of saturation at
    about -0.4 degrees, where `auto` starts, and 50% at about -0.1 degrees, where it stops. Polaris
    becomes detectable below about -3.5 degrees.
    """
    return min(BLINDING_LIGHT, 0.5 * 10 ** (0.493 * (elevation + 3.0)))


def pole_sky(daylight: float = 0.21) -> SkyCurve:
    """The simulator's sky near the pole, at `daylight` of saturation with the Sun at +10 degrees.

    The curve follows `seeingmon.drivers.sim.sky` (6 mag/arcsec^2 at 0 degrees, and the daylight
    sky from +10 degrees up), scaled so that the daylight sky reads `daylight` at 1 ms in bin2. The
    default, 0.21, is the simulator's daylight sky. Polaris is detectable in a sky below about
    0.29, so the default shows it all day.
    """

    def curve(elevation: float) -> float:
        mag = float(sky_brightness_mag_arcsec2(20.5, elevation))
        return daylight * 10 ** (-0.4 * (mag - DAYLIGHT_SKY_MAG_ARCSEC2))

    return curve


@dataclass(frozen=True, slots=True)
class ConfigureCall:
    """One call of `configure` on the scenario camera.

    `purpose` is what the scheduler configured the stream for (`fast`, `search`, `survey`,
    `watch`, `align`, `rapid_focus`, or `commission`), noted when the stream starts. It is `None`
    for a stream that never started.
    """

    t_utc_ns: int
    config: StreamConfig
    purpose: str | None = None


class ScenarioCamera(FakeCameraDriver):
    """The fake camera, plus scripted faults and a log of every `configure`.

    While the clock is not synchronized, a frame carries `TimeQuality.INVALID` and
    `FrameFlag.TIME_INVALID`, as `acquire` marks it.
    """

    def __init__(self, clock: Clock, world: World) -> None:
        super().__init__(
            clock,
            full_frames={"bin1": (8288, 5644), "bin2": SMALL_BIN2},
            frame_factory=world.render,
        )
        self.world = world
        self.configure_log: list[ConfigureCall] = []
        self.fault_windows: list[list[int | None]] = []  # [start, end], with None for open ended
        self.gone_windows: list[list[int | None]] = []  # the camera is not there
        self.fixed_by: int | None = None  # the lowest ladder step that clears a fault

    def add_fault(self, start_utc_ns: int, end_utc_ns: int | None) -> None:
        self.fault_windows.append([start_utc_ns, end_utc_ns])

    def clear_faults(self) -> None:
        now = self._clock.utc_ns()
        for window in self.fault_windows:
            if window[1] is None or window[1] > now:
                window[1] = now

    def fault_active(self) -> bool:
        now = self._clock.utc_ns()
        return any(
            start <= now and (end is None or now < end)
            for start, end in ((w[0], w[1]) for w in self.fault_windows if w[0] is not None)
        )

    def add_gone(self, start_utc_ns: int, end_utc_ns: int | None) -> None:
        self.gone_windows.append([start_utc_ns, end_utc_ns])

    def gone_active(self) -> bool:
        now = self._clock.utc_ns()
        return any(
            start is not None and start <= now and (end is None or now < end)
            for start, end in ((w[0], w[1]) for w in self.gone_windows)
        )

    def open(self) -> CameraInfo:
        if self.gone_active():
            self.calls.append(("open", None))
            raise CameraDisconnectedError("no ASI camera is connected")
        return super().open()

    def configure(self, config: StreamConfig) -> ActiveStream:
        self.configure_log.append(ConfigureCall(self._clock.utc_ns(), config))
        return super().configure(config)

    def start(self) -> None:
        super().start()
        stream = self.world.scheduler.stream
        last = self.configure_log[-1]
        if stream is not None and last.purpose is None:
            self.configure_log[-1] = replace(last, purpose=stream.purpose)

    def read_frame(self, timeout_s: float) -> Frame:
        if self.fault_active() or self.gone_active():
            self.calls.append(("read_frame", timeout_s))
            self._clock.sleep(timeout_s)
            raise CameraTimeoutError("scripted fault")
        frame = super().read_frame(timeout_s)
        if self._clock.status().synchronized is False:  # `acquire` marks such a frame
            frame = replace(
                frame,
                t_quality=TimeQuality.INVALID,
                flags=frame.flags | FrameFlag.TIME_INVALID,
            )
        return frame

    def recover(self, level: RecoveryLevel) -> None:
        if self.gone_active():
            self.calls.append(("recover", level))
            if level is RecoveryLevel.RESTART_CAPTURE:
                raise CameraError(
                    "ASISetControlValue failed with GENERAL_ERROR (16): the SDK reports a "
                    "general error"
                )
            if level is RecoveryLevel.REOPEN:
                raise CameraDisconnectedError("no ASI camera is connected")
            raise CameraError("no USB device matches the camera's vendor ID")
        super().recover(level)
        if self.fixed_by is not None and int(level) >= self.fixed_by:
            self.clear_faults()

    def calls_named(self, name: str) -> list[tuple[str, object]]:
        return [call for call in self.calls if call[0] == name]


class ScenarioSurvey(FakeSurveyAnalyzer):
    """The fake survey analysis, driven by the script.

    A long exposure reports the cloud fraction of the script, and it solves unless clouds cover the
    field or the script forbids it. It always gets a pointing record, unsolved when it does not
    solve. A short exposure cannot tell the cloud fraction, and it does not solve. Its pointing
    record is unsolved, as the 1 ms frame of a real survey step gets one, and the scheduler drops
    it. A solved result gives the pointing provider the true position, and its pointing record
    carries the flags that the script sets for its time (`World.pointing_flags`).
    """

    def __init__(self, world: World, polls_until_ready: int = 0) -> None:
        super().__init__(
            station_id="test", profile_id=PROFILE.id, polls_until_ready=polls_until_ready
        )
        self._world = world

    def submit(self, frame: Frame) -> None:
        long_exposure = frame.exposure_us >= 1_000_000
        fraction = self._world.cloud_fraction(frame.t_utc_ns)
        self.cloud_fraction = fraction if long_exposure else None
        self.solved = long_exposure and fraction < 0.9 and self._world.can_solve(frame.t_utc_ns)
        super().submit(frame)

    def poll(self) -> tuple[SurveyOutput, ...]:
        outputs = []
        for output in super().poll():
            if output.solved:
                self._world.solve(output.t_utc_ns)
            if output.cloud_fraction is not None:  # a long frame also yields two more records
                quality = SkyQualityRecord(
                    station_id="test",
                    t_utc_ns=output.t_utc_ns,
                    profile_id=PROFILE.id,
                    provenance={"algo": "fake"},
                    n_stars_used=0,
                    cloud_fraction=output.cloud_fraction,
                )
                pointing = PointingRecord(
                    station_id="test",
                    t_utc_ns=output.t_utc_ns,
                    profile_id=PROFILE.id,
                    provenance={"algo": "fake"},
                    n_matched=20 if output.solved else 0,
                    readout_mode="bin2",
                    solver="fake" if output.solved else "none",
                    flags=(
                        self._world.pointing_flags_at(output.t_utc_ns)
                        if output.solved
                        else ["unsolved"]
                    ),
                )
                output = replace(output, records=(*output.records, quality, pointing))
            else:  # the real pipeline gives the 1 ms frame an unsolved record too
                brief = PointingRecord(
                    station_id="test",
                    t_utc_ns=output.t_utc_ns,
                    profile_id=PROFILE.id,
                    provenance={"algo": "fake"},
                    n_matched=0,
                    readout_mode="bin2",
                    solver="none",
                    flags=["unsolved"],
                )
                output = replace(output, records=(*output.records, brief))
            outputs.append(output)
        return tuple(outputs)


def _interval_value(
    intervals: list[tuple[int, int, float]], t_utc_ns: int, default: float = 0.0
) -> float:
    return max(
        (value for start, end, value in intervals if start <= t_utc_ns < end), default=default
    )


class World:
    """The scheduler, the fakes, the clock, and the script. Build one per test."""

    def __init__(
        self,
        *,
        start_utc_ns: int = START,
        config: SchedulerConfig | None = None,
        site: SiteConfig | None = SITE,
        solved_at_start: bool = True,
        escalate: bool = True,
        survey_polls: int = 0,
        clock: Clock | None = None,
        context_provider: Callable[[int], FastContext] | None = None,
        sky: SkyCurve = saturating_sky,
        twilight: TwilightConfig = SCENARIO_TWILIGHT,
    ) -> None:
        self.start_utc_ns = start_utc_ns
        self.clock: Clock = clock or VirtualClock(start_utc_ns)
        self.writer = ListRecordWriter()
        self.camera = ScenarioCamera(self.clock, self)
        self.config = config or TEST_CONFIG
        self.sky = sky
        self.fast = FakeFastAnalyzer(
            station_id="test",
            profile_id=PROFILE.id,
            window_s=self.config.fast.analysis_window_s,
            min_snr=MIN_STAR_SNR,
            snr_model=self.frame_snr,
        )
        self.survey = ScenarioSurvey(self, survey_polls)
        self.pointing = FakePointingProvider()
        self.escalations: list[tuple[int, EscalationLevel]] = []
        self.align_frames: list[Frame] = []
        self.focus = FakeFocusSink()  # the consumer of the rapid focus frames
        self.results: list[CommissionResult] = []
        self._lights: list[tuple[int, int, float]] = []
        self._clouds: list[tuple[int, int, float]] = []
        self._hidden: list[tuple[int, int, float]] = []
        self._unsolvable: list[tuple[int, int, float]] = []
        self._pointing_flags: list[tuple[int, int, tuple[str, ...]]] = []
        self._jolts: list[tuple[int, float, float]] = []
        self._actions: list[tuple[int, int, Callable[[World], None]]] = []
        self._action_count = 0
        self._elevation_cache: dict[int, float] = {}
        self._gain_cache: dict[tuple[str, int], float] = {}
        if solved_at_start:
            self.solve(start_utc_ns)
        self.scheduler = Scheduler(
            driver=self.camera,
            fast=self.fast,
            survey=self.survey,
            pointing=self.pointing,
            records=self.writer,
            metrics=self.writer,
            clock=self.clock,
            profile=PROFILE,
            station_id="test",
            config=self.config,
            site=site,
            escalate=self._escalate if escalate else None,
            alignment_sink=self.align_frames.append,
            focus_sink=self.focus,
            result_sink=self.results.append,
            context_provider=context_provider,
            twilight=twilight,
        )

    # --- Time ---

    def t(self, seconds: float) -> int:
        """The UTC time `seconds` after the start of the scenario."""
        return self.start_utc_ns + round(seconds * NS_PER_S)

    def seconds(self, t_utc_ns: int) -> float:
        """Seconds since the start of the scenario."""
        return (t_utc_ns - self.start_utc_ns) / NS_PER_S

    # --- The script ---

    def light(self, start: float, end: float, fraction: float) -> None:
        """Add sky light between two times, as a share of saturation at 1 ms and gain 0."""
        self._lights.append((self.t(start), self.t(end), fraction))

    def cloud(self, start: float, end: float, fraction: float) -> None:
        """Clouds cover `fraction` of the field between two times."""
        self._clouds.append((self.t(start), self.t(end), fraction))

    def hide_star(self, start: float, end: float) -> None:
        self._hidden.append((self.t(start), self.t(end), 1.0))

    def no_solution(self, start: float, end: float) -> None:
        """The survey analysis cannot solve between two times."""
        self._unsolvable.append((self.t(start), self.t(end), 1.0))

    def pointing_flags(self, start: float, end: float, *flags: str) -> None:
        """The solved pointing records of the frames between two times carry these flags."""
        self._pointing_flags.append((self.t(start), self.t(end), flags))

    def jolt(self, at: float, dx_px: float, dy_px: float = 0.0) -> None:
        """Shift the true star position, in bin1 pixels, as a bumped mount would."""
        self._jolts.append((self.t(at), dx_px, dy_px))

    def camera_fault(
        self, start: float, end: float | None = None, *, fixed_by: int | None = None
    ) -> None:
        """Every read times out from `start` to `end`, or until a ladder step `fixed_by` runs."""
        self.camera.add_fault(self.t(start), None if end is None else self.t(end))
        if fixed_by is not None:
            self.camera.fixed_by = fixed_by

    def camera_gone(self, start: float, end: float | None = None) -> None:
        """The camera is not there from `start` to `end`, or for good (see the module text)."""
        self.camera.add_gone(self.t(start), None if end is None else self.t(end))

    def at(self, seconds: float, action: Callable[[World], None]) -> None:
        """Run `action(world)` at the first step boundary at or after this time."""
        self._action_count += 1
        heapq.heappush(self._actions, (self.t(seconds), self._action_count, action))

    # --- The truth ---

    def sun_elevation(self, t_utc_ns: int) -> float:
        """The Sun's elevation, cached per 10 seconds, because it changes slowly."""
        bucket = t_utc_ns // (10 * NS_PER_S)
        elevation = self._elevation_cache.get(bucket)
        if elevation is None:
            elevation = sun_elevation_deg(t_utc_ns, SITE.latitude_deg, SITE.longitude_deg)
            self._elevation_cache[bucket] = elevation
        return elevation

    def sky_fraction_at_1ms(self, t_utc_ns: int) -> float:
        """The sky background at 1 ms and gain 0 in bin2, as a share of saturation.

        The sky curve of the world gives it, with the brightness of a dark sky as its floor, and
        a floodlight adds to it. It may pass 1: the frame clips, and a shorter exposure, such as
        the watch frame of 32 us, still reads it.
        """
        base = max(NIGHT_SKY_FRACTION, self.sky(self.sun_elevation(t_utc_ns)))
        return base + _interval_value(self._lights, t_utc_ns)

    def snr(self, t_utc_ns: int, exposure_ms: float = REAL_FAST_EXPOSURE_MS) -> float:
        """The SNR of Polaris in a bin1 frame at gain 0, from the truth of the world.

        It is the centroid aperture's formula of the detection estimate: the star's electrons in
        the aperture over the root of their photon noise and of the aperture area times the
        variance of a pixel (the sky, the read noise, and the rounding of the ADC). Clouds dim the
        star by their transparency. A hidden star has an SNR of 0.
        """
        if not self.star_visible(t_utc_ns):
            return 0.0
        star = POLARIS_E_PER_MS * exposure_ms * APERTURE_FRACTION * self.transparency(t_utc_ns)
        if star <= 0.0:
            return 0.0
        sky = self.sky_fraction_at_1ms(t_utc_ns) * BIN2_FULL_WELL_E * BIN1_PER_BIN2_AREA
        variance = sky * exposure_ms + BIN1_PIXEL_VAR_E2
        return star / math.sqrt(star + APERTURE_PX2 * variance)

    def frame_snr(self, frame: Frame) -> float:
        """The SNR model of the fake fast analysis: the SNR of the world at the frame's time."""
        return self.snr(frame.t_utc_ns, self._real_exposure_ms(frame.exposure_us, frame.mode))

    @staticmethod
    def _real_exposure_ms(exposure_us: int, mode: str) -> float:
        """The exposure that a frame stands for: the slow test stream stands for the real one."""
        if exposure_us > MAX_REAL_EXPOSURE_US and mode == "bin1":
            return REAL_FAST_EXPOSURE_MS
        return exposure_us / 1000.0

    def cloud_fraction(self, t_utc_ns: int) -> float:
        return _interval_value(self._clouds, t_utc_ns)

    def transparency(self, t_utc_ns: int) -> float:
        return max(0.0, 1.0 - self.cloud_fraction(t_utc_ns))

    def star_visible(self, t_utc_ns: int) -> bool:
        return _interval_value(self._hidden, t_utc_ns) == 0.0

    def can_solve(self, t_utc_ns: int) -> bool:
        return _interval_value(self._unsolvable, t_utc_ns) == 0.0

    def pointing_flags_at(self, t_utc_ns: int) -> list[str]:
        """The flags that the script gives a solved pointing record of this time."""
        return sorted(
            {
                flag
                for start, end, flags in self._pointing_flags
                if start <= t_utc_ns < end
                for flag in flags
            }
        )

    def star_position(self, t_utc_ns: int, mode: str = "bin1") -> tuple[float, float]:
        """Where the star truly is, in sensor pixels of `mode`."""
        elapsed = (t_utc_ns - self.start_utc_ns) / NS_PER_S
        x = STAR_BIN1[0] + STAR_DRIFT_PX_PER_S[0] * elapsed
        y = STAR_BIN1[1] + STAR_DRIFT_PX_PER_S[1] * elapsed
        for when, dx, dy in self._jolts:
            if t_utc_ns >= when:
                x, y = x + dx, y + dy
        scale = 1.0 if mode == "bin1" else 0.5
        return x * scale, y * scale

    def solve(self, t_utc_ns: int) -> None:
        """Give the pointing provider the true position at this time, as a solved survey does."""
        for mode, scale in (("bin1", 1.0), ("bin2", 0.5)):
            x, y = self.star_position(t_utc_ns, mode)
            self.pointing.set_solution(
                mode,
                x,
                y,
                t0_utc_ns=t_utc_ns,
                drift_px_per_s=(STAR_DRIFT_PX_PER_S[0] * scale, STAR_DRIFT_PX_PER_S[1] * scale),
            )

    def _sky_scale(self, mode: str, gain: int) -> float:
        """How much brighter the sky looks in this mode and gain than in bin2 at gain 0."""
        key = (mode, gain)
        scale = self._gain_cache.get(key)
        if scale is None:
            area = (PROFILE.mode(mode).pixel_size_um / PROFILE.mode("bin2").pixel_size_um) ** 2
            conversion = PROFILE.e_per_adu("bin2", 0) / PROFILE.e_per_adu(mode, gain)
            scale = area * conversion
            self._gain_cache[key] = scale
        return scale

    def render(self, config: StreamConfig, roi: Roi, seq: int) -> FrameData:
        """The frame factory of the fake camera."""
        t_ns = self.clock.utc_ns()
        gain_factor = self._sky_scale(config.mode, config.gain)
        # The fast stream of the test configuration runs 1,000 times slower than the real one. The
        # sky then uses the real exposure of the fast stream, so that the frames look as they would.
        # A sweep or a burst asks for a real exposure, and it gets that one.
        slow_test_stream = config.exposure_us > MAX_REAL_EXPOSURE_US
        exposure_ms = (
            REAL_FAST_EXPOSURE_MS
            if slow_test_stream and config.kind is StreamKind.VIDEO and config.mode == "bin1"
            else config.exposure_us / 1000.0
        )
        sky_dn = self.sky_fraction_at_1ms(t_ns) * SATURATION_DN * exposure_ms * gain_factor
        level = int(min(65_535, OFFSET_DN + sky_dn))
        data = np.full((roi.height, roi.width), level, dtype=np.uint16)
        if self.star_visible(t_ns):
            x, y = self.star_position(t_ns, config.mode)
            if roi.contains(x, y):
                peak = level + round(STAR_PEAK_DN * self.transparency(t_ns))
                data[int(y) - roi.y, int(x) - roi.x] = min(65_535, peak)
        return data

    # --- Running ---

    def _escalate(self, level: EscalationLevel) -> None:
        self.escalations.append((self.clock.utc_ns(), level))
        if self.camera.fixed_by is not None and int(level) >= self.camera.fixed_by:
            self.camera.clear_faults()

    def _fire_due(self) -> None:
        while self._actions and self._actions[0][0] <= self.clock.utc_ns():
            _, _, action = heapq.heappop(self._actions)
            action(self)

    def run_until(self, seconds: float) -> None:
        """Run the scheduler until `seconds` after the start, firing scripted actions on time."""
        target = self.t(seconds)
        while self.clock.utc_ns() < target:
            self._fire_due()
            next_stop = target
            if self._actions:
                next_stop = min(next_stop, self._actions[0][0])
            if next_stop > self.clock.utc_ns():
                self.scheduler.run_until(next_stop)
        self._fire_due()

    def run_for(self, seconds: float) -> None:
        self.run_until(self.seconds(self.clock.utc_ns()) + seconds)

    def close(self) -> None:
        self.scheduler.close()

    # --- What happened ---

    def records(self, record_type: str) -> list[Record]:
        return self.writer.of_type(record_type)

    def events(self, kind: str | None = None) -> list[EventRecord]:
        events = [event for event in self.writer.of_type("event") if isinstance(event, EventRecord)]
        return [event for event in events if kind is None or event.kind == kind]

    def windows(self) -> list[SeeingWindowRecord]:
        return [
            w for w in self.writer.of_type("seeing_window") if isinstance(w, SeeingWindowRecord)
        ]

    def state_changes(self) -> list[tuple[float, str, str]]:
        """Each change of state as (seconds since the start, from, to)."""
        return [
            (self.seconds(e.t_utc_ns), (e.detail or {})["from"], (e.detail or {})["to"])
            for e in self.events("scheduler.state_change")
        ]

    def states_visited(self) -> list[str]:
        changes = self.state_changes()
        return ["safe", *(to for _, _, to in changes)]

    def configures(
        self,
        *,
        mode: str | None = None,
        video: bool | None = None,
        purpose: str | None = None,
    ) -> list[ConfigureCall]:
        calls = self.camera.configure_log
        if mode is not None:
            calls = [call for call in calls if call.config.mode == mode]
        if video is not None:
            calls = [call for call in calls if (call.config.kind.value == "video") == video]
        if purpose is not None:
            calls = [call for call in calls if call.purpose == purpose]
        return calls

    def fast_starts(self) -> list[float]:
        """When each fast stream (measure) began, in seconds since the start. No burst counts."""
        return [self.seconds(call.t_utc_ns) for call in self.configures(purpose="fast")]

    def burst_starts(self) -> list[float]:
        """When each search burst began, in seconds since the start."""
        return [self.seconds(call.t_utc_ns) for call in self.configures(purpose="search")]

    def period_starts(self) -> list[float]:
        """When each period of the cycle began: its first burst, or its fast stream.

        Within a period, bursts may follow each other, and a fast stream may follow them. Any
        other stream before a burst or a fast stream, or a fast stream, ends a period. Two search
        periods without a survey step between them (a skipped step) read as one.
        """
        starts: list[float] = []
        previous: str | None = None
        for call in self.camera.configure_log:
            if call.purpose in ("fast", "search") and previous != "search":
                starts.append(self.seconds(call.t_utc_ns))
            previous = call.purpose
        return starts

    def visible_times(self) -> list[float]:
        """When each `polaris.visible` came, in seconds since the start."""
        return [self.seconds(e.t_utc_ns) for e in self.events("polaris.visible")]

    def hidden_times(self) -> list[float]:
        """When each `polaris.hidden` came, in seconds since the start."""
        return [self.seconds(e.t_utc_ns) for e in self.events("polaris.hidden")]
