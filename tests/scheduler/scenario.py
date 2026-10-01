"""A scripted world for the scheduler tests: sky light, clouds, a star, faults, and commands.

`World` wires a `Scheduler` to the fakes in `seeingmon.testing`, on a `VirtualClock`. A script
describes the world as functions of time, and the frame factory of the fake camera renders it:

- **Sky light.** The background follows the Sun's elevation at a synthetic site (latitude 55
  degrees north, longitude 0, which is nobody's real site), through a smooth brightness curve that
  crosses the daylight gate at about -3 degrees. `light` adds a floodlight on top.
- **Clouds.** `cloud` sets the cloud fraction that the survey analysis reports and dims the star.
- **The star.** Polaris sits near the middle of the bin1 sensor and drifts 0.087 pixels a second.
  `hide_star` removes it, and `jolt` shifts it, as a bumped mount would.
- **Faults.** `camera_fault` makes every read time out. A recovery step of a given level, or an
  escalation, clears it.
- **Commands.** `at` runs any action at a time, such as `world.scheduler.submit(Pause())`.

The scenario uses a small bin2 frame and a slow, small fast stream (one frame in 2 s on a
32-pixel ROI), because the point is the scheduling and not the pixels. That keeps a 12 hour
night to a few seconds of real time.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from seeingmon.analysis import SurveyOutput
from seeingmon.clock import NS_PER_S, Clock, VirtualClock, iso_to_utc_ns
from seeingmon.drivers.base import CameraTimeoutError, RecoveryLevel
from seeingmon.frames import ActiveStream, Frame, FrameData, Roi, StreamConfig, StreamKind
from seeingmon.profile import load_profile
from seeingmon.records import EventRecord, Record, SeeingWindowRecord, SkyQualityRecord
from seeingmon.scheduler import CommissionResult, Scheduler, SchedulerConfig, SiteConfig
from seeingmon.scheduler.config import FastConfig, LoopConfig
from seeingmon.scheduler.ephemeris import sun_elevation_deg
from seeingmon.scheduler.levels import EscalationLevel
from seeingmon.testing import (
    FakeCameraDriver,
    FakeFastAnalyzer,
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

TEST_CONFIG = SchedulerConfig(
    fast=FastConfig(
        exposure_us=2_000_000,
        roi_arcmin=1.0,  # 32 pixels in bin1
        roi_edge_margin_px=4.0,
        missing_star_frames=10,
    ),
    loop=LoopConfig(max_sleep_s=5.0),
)


@dataclass(frozen=True, slots=True)
class ConfigureCall:
    """One call of `configure` on the scenario camera."""

    t_utc_ns: int
    config: StreamConfig


class ScenarioCamera(FakeCameraDriver):
    """The fake camera, plus scripted faults and a log of every `configure`."""

    def __init__(self, clock: Clock, world: World) -> None:
        super().__init__(
            clock,
            full_frames={"bin1": (8288, 5644), "bin2": SMALL_BIN2},
            frame_factory=world.render,
        )
        self.world = world
        self.configure_log: list[ConfigureCall] = []
        self.fault_windows: list[list[int | None]] = []  # [start, end], with None for open ended
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

    def configure(self, config: StreamConfig) -> ActiveStream:
        self.configure_log.append(ConfigureCall(self._clock.utc_ns(), config))
        return super().configure(config)

    def read_frame(self, timeout_s: float) -> Frame:
        if self.fault_active():
            self.calls.append(("read_frame", timeout_s))
            self._clock.sleep(timeout_s)
            raise CameraTimeoutError("scripted fault")
        return super().read_frame(timeout_s)

    def recover(self, level: RecoveryLevel) -> None:
        super().recover(level)
        if self.fixed_by is not None and int(level) >= self.fixed_by:
            self.clear_faults()

    def calls_named(self, name: str) -> list[tuple[str, object]]:
        return [call for call in self.calls if call[0] == name]


class ScenarioSurvey(FakeSurveyAnalyzer):
    """The fake survey analysis, driven by the script.

    A long exposure reports the cloud fraction of the script, and it solves unless clouds cover the
    field or the script forbids it. A short exposure cannot tell the cloud fraction, and it does
    not solve. A solved result gives the pointing provider the true position.
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
            if output.cloud_fraction is not None:  # a long frame also yields a sky quality record
                quality = SkyQualityRecord(
                    station_id="test",
                    t_utc_ns=output.t_utc_ns,
                    profile_id=PROFILE.id,
                    provenance={"algo": "fake"},
                    n_stars_used=0,
                    cloud_fraction=output.cloud_fraction,
                )
                output = replace(output, records=(*output.records, quality))
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
    ) -> None:
        self.start_utc_ns = start_utc_ns
        self.clock: Clock = clock or VirtualClock(start_utc_ns)
        self.writer = ListRecordWriter()
        self.camera = ScenarioCamera(self.clock, self)
        self.config = config or TEST_CONFIG
        self.fast = FakeFastAnalyzer(
            station_id="test", profile_id=PROFILE.id, window_s=self.config.fast.analysis_window_s
        )
        self.survey = ScenarioSurvey(self, survey_polls)
        self.pointing = FakePointingProvider()
        self.escalations: list[tuple[int, EscalationLevel]] = []
        self.align_frames: list[Frame] = []
        self.results: list[CommissionResult] = []
        self._lights: list[tuple[int, int, float]] = []
        self._clouds: list[tuple[int, int, float]] = []
        self._hidden: list[tuple[int, int, float]] = []
        self._unsolvable: list[tuple[int, int, float]] = []
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
            result_sink=self.results.append,
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

        It falls by a factor of 3 for each degree that the Sun sinks, from 0.5 at -3 degrees to the
        brightness of a dark sky at -18 degrees.
        """
        elevation = self.sun_elevation(t_utc_ns)
        base = max(NIGHT_SKY_FRACTION, 0.5 * 10 ** (0.493 * (elevation + 3.0)))
        return min(1.0, base + _interval_value(self._lights, t_utc_ns))

    def cloud_fraction(self, t_utc_ns: int) -> float:
        return _interval_value(self._clouds, t_utc_ns)

    def transparency(self, t_utc_ns: int) -> float:
        return max(0.0, 1.0 - self.cloud_fraction(t_utc_ns))

    def star_visible(self, t_utc_ns: int) -> bool:
        return _interval_value(self._hidden, t_utc_ns) == 0.0

    def can_solve(self, t_utc_ns: int) -> bool:
        return _interval_value(self._unsolvable, t_utc_ns) == 0.0

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
        self, *, mode: str | None = None, video: bool | None = None
    ) -> list[ConfigureCall]:
        calls = self.camera.configure_log
        if mode is not None:
            calls = [call for call in calls if call.config.mode == mode]
        if video is not None:
            calls = [call for call in calls if (call.config.kind.value == "video") == video]
        return calls
