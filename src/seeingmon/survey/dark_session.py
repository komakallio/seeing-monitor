"""Record a dark set with the camera: the flow behind `seeingmon dark` and the dark task of `core`.

The camera has no lens cap, so you cover it. The session

1. takes bias frames at the shortest exposure, which give the bias level and the read noise,
2. waits until a short test frame is dark, which means that you have covered the camera
   (skip this with `wait=False`),
3. records dark frames at the survey exposure, and checks each one,
4. builds the master dark (the per-pixel median), finds the hot pixels, and adds the set to the
   `DarkLibrary`, and
5. reports the dark rate, the dark model, and whether the library still needs a set.

**Two ways to run it.** `run_dark_session` is the command-line path. It takes a driver, opens it,
and closes it when the session ends, so no other process may hold the camera. `record_dark_set` is
the same flow on a borrowed camera: any `DarkCamera` that can configure itself and take one frame.
It opens and closes nothing, so the scheduler of `core` can lend its camera for the session (see
`seeingmon.services.core.commissioning.dark`).

**Progress and stopping.** The flow names its phase (`DarkProgress`: `bias`, `cover`, `dark`,
`build`, and `done`) to a `progress` callback, after each frame and at each change of phase.
`should_stop` is asked before each frame and during the wait for the cover. When it answers true,
the session raises `DarkAborted`, and the library stays as it was.

The session takes the camera, the clock, and the library as arguments, so a test runs it against
the `sim` driver on a `VirtualClock` with no real waiting. It never writes a path, a host, or a
serial number to the output.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.drivers.base import CameraDriver, CameraInfo
from seeingmon.frames import Frame, StreamConfig, StreamKind
from seeingmon.profile import Profile
from seeingmon.survey.dark import (
    DEFAULT_DOUBLING_C,
    DarkCheck,
    DarkCheckOptions,
    DarkError,
    DarkLibrary,
    DarkModel,
    DarkSet,
    DarkStatus,
    check_dark_frame,
    dark_status,
    master_median,
)
from seeingmon.survey.rawdata import clipped_mean, native_u16, robust_level

if TYPE_CHECKING:
    from seeingmon.survey.config import DarkConfig

log = logging.getLogger("seeingmon.survey")

_READ_MARGIN_S = 30.0  # a read waits this long beyond the exposure
_FALLBACK_TEMPERATURE_C = 20.0  # for the check of a frame that carries no temperature
_STOP_CHECK_S = 0.25  # the wait for the cover looks at `should_stop` this often

DarkPhase = Literal["bias", "cover", "dark", "build", "done"]


@dataclass(frozen=True, slots=True)
class DarkSessionOptions:
    """What the session records, and how it decides that the camera is covered.

    `exposure_s` should equal the survey exposure, because the dark rate is the master dark
    above the bias divided by it. `test_exposure_s` is the exposure of the frames that the wait
    checks, and the wait needs `stable_polls` dark frames in a row, `poll_s` apart.
    """

    mode: str = "bin2"
    gain: int = 120
    exposure_s: float = 30.0
    frames: int = 9
    bias_frames: int = 9
    offset: int | None = None
    wait: bool = True
    test_exposure_s: float = 1.0
    poll_s: float = 5.0
    stable_polls: int = 2
    wait_timeout_s: float = 1800.0
    max_temperature_spread_c: float = 2.0
    check: DarkCheckOptions = field(default_factory=DarkCheckOptions)
    hot_sigma: float = 6.0
    hot_min_excess_dn: float = 3.0
    prior_doubling_c: float = DEFAULT_DOUBLING_C
    tolerance_c: float = 3.0
    max_age_days: float = 183.0

    def __post_init__(self) -> None:
        if self.frames < 3 or self.bias_frames < 3:
            raise ValueError("a set needs at least 3 dark frames and 3 bias frames")
        if self.exposure_s <= 0 or self.test_exposure_s <= 0 or self.poll_s < 0:
            raise ValueError("the exposures must be positive")
        if self.stable_polls < 1 or self.wait_timeout_s <= 0:
            raise ValueError("stable_polls must be at least 1 and the wait limit positive")

    @classmethod
    def from_config(
        cls,
        cfg: DarkConfig,
        *,
        mode: str | None = None,
        gain: int | None = None,
        exposure_s: float | None = None,
        frames: int | None = None,
        bias_frames: int | None = None,
        wait: bool = True,
        wait_timeout_s: float | None = None,
    ) -> DarkSessionOptions:
        """The options of `[survey.dark]`, with any value that you pass in its place.

        A value left at `None` (and an empty `mode`) takes the configured one. Raises
        `ValueError` for a value that the options refuse.
        """
        return cls(
            mode=mode or cfg.mode,
            gain=cfg.gain if gain is None else gain,
            exposure_s=cfg.exposure_s if exposure_s is None else exposure_s,
            frames=cfg.frames if frames is None else frames,
            bias_frames=cfg.bias_frames if bias_frames is None else bias_frames,
            wait=wait,
            test_exposure_s=cfg.test_exposure_s,
            poll_s=cfg.poll_s,
            stable_polls=cfg.stable_polls,
            wait_timeout_s=cfg.wait_timeout_s if wait_timeout_s is None else wait_timeout_s,
            max_temperature_spread_c=cfg.max_temperature_spread_c,
            check=DarkCheckOptions(
                rate_factor=cfg.rate_factor,
                min_rate_e_per_s=cfg.min_rate_e_per_s,
                noise_factor=cfg.noise_factor,
                max_tail_fraction=cfg.max_tail_fraction,
            ),
            hot_sigma=cfg.hot_sigma,
            hot_min_excess_dn=cfg.hot_min_excess_dn,
            prior_doubling_c=cfg.doubling_c,
            tolerance_c=cfg.temperature_tolerance_c,
            max_age_days=cfg.max_age_days,
        )


@dataclass(frozen=True, slots=True)
class DarkSessionResult:
    """What a session produced. `waited_s` is the time spent waiting for the cover."""

    dark_set: DarkSet
    check: DarkCheck
    waited_s: float
    status: DarkStatus
    model: DarkModel | None
    n_sets: int


@dataclass(frozen=True, slots=True)
class DarkProgress:
    """Where a session stands. `step` of `steps` counts what the phase has done so far.

    In `bias` and `dark` the steps are the frames. In `cover` the step counts the dark test frames
    in a row, and the steps are the number that the wait needs. In `build` and `done` the session
    has one step. `check` is the latest check of a frame in `cover` and `dark`, and `None` before
    the first one.
    """

    phase: DarkPhase
    step: int
    steps: int
    message: str
    check: DarkCheck | None = None


class DarkAborted(Exception):  # noqa: N818 (a stop on purpose, not an error)
    """`should_stop` ended the session before it added a set. The library stays as it was."""


class DarkCamera(Protocol):
    """What the session needs from a camera: no more than this.

    `take` returns one frame of a snapshot at the configured settings. `read_temperature_c`
    answers when a frame carries no sensor temperature, and it returns `None` when the camera
    has none to give.
    """

    def configure(self, config: StreamConfig) -> None: ...

    def take(self, exposure_s: float) -> Frame: ...

    def read_temperature_c(self) -> float | None: ...


class DriverCamera:
    """A driver as a `DarkCamera`, for the command line. It opens and closes the driver itself."""

    def __init__(self, driver: CameraDriver) -> None:
        self._driver = driver

    def open(self) -> CameraInfo:
        return self._driver.open()

    def close(self) -> None:
        self._driver.close()

    def configure(self, config: StreamConfig) -> None:
        self._driver.configure(config)

    def take(self, exposure_s: float) -> Frame:
        self._driver.start()
        try:
            return self._driver.read_frame(timeout_s=exposure_s + _READ_MARGIN_S)
        finally:
            self._driver.stop()

    def read_temperature_c(self) -> float | None:
        return self._driver.read_temperature_c()


@dataclass(slots=True)
class _Setup:
    """The numbers of the readout setting that the checks share."""

    profile: Profile
    e_per_adu: float
    read_noise_e: float
    sdk_bin: int
    model: DarkModel | None
    bias_dn: float = 0.0

    def expected_rate_e_per_s(self, temperature_c: float) -> float:
        """The dark rate to expect: from the library when it has a model, else from the prior."""
        if self.model is not None:
            return self.model.rate_dn_per_s(temperature_c) * self.e_per_adu
        return self.profile.dark_current_e_per_s_per_px(temperature_c) * self.sdk_bin**2

    def check(
        self, frame: Frame, exposure_s: float, options: DarkCheckOptions, temperature_c: float
    ) -> DarkCheck:
        return check_dark_frame(
            native_u16(frame),
            bias_dn=self.bias_dn,
            exposure_s=exposure_s,
            e_per_adu=self.e_per_adu,
            read_noise_e=self.read_noise_e,
            expected_rate_e_per_s=self.expected_rate_e_per_s(temperature_c),
            options=options,
        )


class _Recorder:
    """One session: the camera, the clock, and the three ways to hear from the caller."""

    def __init__(
        self,
        camera: DarkCamera,
        cfg: DarkSessionOptions,
        clock: Clock,
        setup: _Setup,
        say: Callable[[str], None],
        progress: Callable[[DarkProgress], None] | None,
        should_stop: Callable[[], bool] | None,
    ) -> None:
        self.camera = camera
        self.cfg = cfg
        self.clock = clock
        self.setup = setup
        self.say = say
        self._progress = progress
        self._should_stop = should_stop

    def report(
        self,
        phase: DarkPhase,
        step: int,
        steps: int,
        message: str,
        check: DarkCheck | None = None,
    ) -> None:
        if self._progress is not None:
            self._progress(DarkProgress(phase, step, steps, message, check))

    def stop_requested(self) -> None:
        """Raise `DarkAborted` when the caller asked the session to end."""
        if self._should_stop is not None and self._should_stop():
            raise DarkAborted("the dark session was stopped before it added a set")

    def configure(self, exposure_s: float) -> None:
        self.camera.configure(
            StreamConfig(
                self.cfg.mode,
                max(1, round(exposure_s * 1e6)),
                self.cfg.gain,
                kind=StreamKind.SNAPSHOT,
                offset=self.cfg.offset,
            )
        )

    def take(self, exposure_s: float) -> Frame:
        self.stop_requested()
        return self.camera.take(exposure_s)

    def temperature_of(self, frame: Frame) -> float:
        if frame.temperature_c is not None:
            return frame.temperature_c
        reading = self.camera.read_temperature_c()
        return _FALLBACK_TEMPERATURE_C if reading is None else reading

    def mean_temperature(self, frames: list[Frame]) -> tuple[float, float]:
        """The mean sensor temperature of the frames and the spread (maximum minus minimum)."""
        values = [frame.temperature_c for frame in frames if frame.temperature_c is not None]
        if not values:
            reading = self.camera.read_temperature_c()
            if reading is None:
                raise DarkError(
                    "the camera reports no sensor temperature, and the dark model needs one"
                )
            values = [reading]
        return float(np.mean(values)), float(max(values) - min(values))

    def record_bias(self, min_exposure_s: float) -> tuple[float, float]:
        """Take the bias frames. Returns the bias level and the read noise, both in counts."""
        count = self.cfg.bias_frames
        message = f"Taking {count} bias frames at the shortest exposure."
        self.say(message)
        self.report("bias", 0, count, message)
        self.configure(min_exposure_s)
        frames: list[npt.NDArray[np.uint16]] = []
        for index in range(count):
            frames.append(native_u16(self.take(min_exposure_s)))
            self.report("bias", index + 1, count, f"Bias frame {index + 1} of {count}.")
        level = float(np.mean([clipped_mean(frame) for frame in frames]))
        difference = frames[0].astype(np.float32) - frames[1].astype(np.float32)
        read_noise = robust_level(difference)[1] / float(np.sqrt(2.0))
        message = f"The bias is {level:.1f} counts, and the read noise {read_noise:.2f} counts."
        self.say(message)
        self.report("bias", count, count, message)
        return level, read_noise

    def wait_for_cover(self) -> float:
        """Take short frames until `stable_polls` in a row are dark. Returns the time waited."""
        cfg = self.cfg
        message = "Cover the camera now. Waiting for a dark frame."
        self.say(message)
        self.report("cover", 0, cfg.stable_polls, message)
        self.configure(cfg.test_exposure_s)
        started_ns = self.clock.monotonic_ns()
        stable = 0
        last_reason = "no frame yet"
        while True:
            waited_s = (self.clock.monotonic_ns() - started_ns) / NS_PER_S
            if waited_s > cfg.wait_timeout_s:
                raise DarkError(
                    f"the camera was not dark after {cfg.wait_timeout_s:.0f} s ({last_reason}); "
                    "cover it, and check the cover for gaps"
                )
            frame = self.take(cfg.test_exposure_s)
            check = self.setup.check(
                frame, cfg.test_exposure_s, cfg.check, self.temperature_of(frame)
            )
            if check.ok:
                stable += 1
                if stable >= cfg.stable_polls:
                    self.say("The camera is dark.")
                    self.report("cover", stable, cfg.stable_polls, "The camera is dark.", check)
                    return (self.clock.monotonic_ns() - started_ns) / NS_PER_S
                self.report(
                    "cover",
                    stable,
                    cfg.stable_polls,
                    f"The test frame is dark ({stable} of {cfg.stable_polls}).",
                    check,
                )
            else:
                stable = 0
                last_reason = check.reason
                self.report(
                    "cover",
                    0,
                    cfg.stable_polls,
                    f"The camera is not dark yet: {check.reason}.",
                    check,
                )
            self.sleep(cfg.poll_s)

    def sleep(self, seconds: float) -> None:
        """Sleep on the clock. With `should_stop`, look at it every quarter of a second."""
        if self._should_stop is None:
            self.clock.sleep(seconds)
            return
        remaining = seconds
        while remaining > 0:
            step = min(remaining, _STOP_CHECK_S)
            self.clock.sleep(step)
            remaining -= step
            self.stop_requested()


def record_dark_set(
    camera: DarkCamera,
    library: DarkLibrary,
    profile: Profile,
    clock: Clock,
    options: DarkSessionOptions | None = None,
    *,
    say: Callable[[str], None] = print,
    progress: Callable[[DarkProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> DarkSessionResult:
    """Record a dark set on a camera, and add it to the library. See the module documentation.

    The camera is open already, and it stays open. Raises `DarkError` when the camera never
    becomes dark, when a recorded frame is not dark, or when the camera gives no temperature,
    `DarkAborted` when `should_stop` answers true, and `seeingmon.drivers.CameraError` for a
    camera fault. The library stays as it was after any of these.
    """
    cfg = options or DarkSessionOptions()
    e_per_adu = profile.e_per_adu(cfg.mode, cfg.gain)
    setup = _Setup(
        profile=profile,
        e_per_adu=e_per_adu,
        read_noise_e=profile.read_noise_e(cfg.mode, cfg.gain),
        sdk_bin=profile.mode(cfg.mode).sdk_bin,
        model=library.model(cfg.mode, cfg.gain, prior_doubling_c=cfg.prior_doubling_c),
    )
    recorder = _Recorder(camera, cfg, clock, setup, say, progress, should_stop)
    bias_dn, read_noise_dn = recorder.record_bias(profile.limits.exposure_us_range[0] / 1e6)
    setup.bias_dn = bias_dn
    waited_s = recorder.wait_for_cover() if cfg.wait else 0.0
    message = f"Recording {cfg.frames} dark frames of {cfg.exposure_s:g} s."
    say(message)
    recorder.report("dark", 0, cfg.frames, message)
    recorder.configure(cfg.exposure_s)
    frames: list[Frame] = []
    stack: list[npt.NDArray[np.uint16]] = []
    levels: list[float] = []
    last_check: DarkCheck | None = None
    for index in range(cfg.frames):
        frame = recorder.take(cfg.exposure_s)
        last_check = setup.check(frame, cfg.exposure_s, cfg.check, recorder.temperature_of(frame))
        if not last_check.ok:
            raise DarkError(
                f"dark frame {index + 1} of {cfg.frames} is not dark: {last_check.reason}"
            )
        frames.append(frame)
        stack.append(native_u16(frame))
        levels.append(last_check.level_dn)
        recorder.report(
            "dark", index + 1, cfg.frames, f"Dark frame {index + 1} of {cfg.frames}.", last_check
        )
    assert last_check is not None
    temperature_c, spread_c = recorder.mean_temperature(frames)
    if spread_c > cfg.max_temperature_spread_c:
        say(
            f"Warning: the sensor temperature changed by {spread_c:.1f} C during the set, "
            "so the set describes the mean."
        )
    recorder.report("build", 0, 1, "Building the master dark and adding the set to the library.")
    recorder.stop_requested()
    dark_set = library.add_set(
        master_median(stack),
        mode=cfg.mode,
        gain=cfg.gain,
        exposure_s=cfg.exposure_s,
        temperature_c=temperature_c,
        temperature_spread_c=spread_c,
        t_utc_ns=round(float(np.mean([frame.t_utc_ns for frame in frames]))),
        n_frames=len(frames),
        n_bias_frames=cfg.bias_frames,
        bias_dn=bias_dn,
        read_noise_dn=read_noise_dn,
        adc_bits=frames[0].adc_bits,
        dark_dn=float(np.mean(levels)),
        hot_sigma=cfg.hot_sigma,
        hot_min_excess_dn=cfg.hot_min_excess_dn,
    )
    model = library.model(cfg.mode, cfg.gain, prior_doubling_c=cfg.prior_doubling_c)
    status = dark_status(
        library,
        temperature_c,
        clock.utc_ns(),
        mode=cfg.mode,
        gain=cfg.gain,
        tolerance_c=cfg.tolerance_c,
        max_age_days=cfg.max_age_days,
    )
    n_sets = len(library.sets())
    recorder.report("build", 1, 1, f"Added {dark_set.name} to the library.")
    say(
        f"The master dark at {temperature_c:.1f} C rises "
        f"{dark_set.rate_dn_per_s * e_per_adu:.3f} e-/s per pixel above the bias, and "
        f"{dark_set.n_hot_pixels} hot pixels stand out."
    )
    say(f"Added {dark_set.name} to the library, which now holds {n_sets} sets.")
    if model is not None:
        how = "fitted" if model.doubling_fitted else "assumed"
        say(f"The dark current doubles every {model.doubling_c:.1f} C ({how}).")
    say(
        f"The library is due for another set: {status.reason}."
        if status.due
        else "The library covers this temperature."
    )
    recorder.report("done", 1, 1, "The dark session is done.", last_check)
    return DarkSessionResult(dark_set, last_check, waited_s, status, model, n_sets)


def run_dark_session(
    driver: CameraDriver,
    library: DarkLibrary,
    profile: Profile,
    clock: Clock,
    options: DarkSessionOptions | None = None,
    *,
    say: Callable[[str], None] = print,
) -> DarkSessionResult:
    """Record a dark set with a driver that this call opens and closes. For `seeingmon dark`.

    Raises `DarkError` when the camera never becomes dark, when a recorded frame is not dark, or
    when the camera gives no temperature, and `seeingmon.drivers.CameraError` for a camera fault.
    The library stays as it was after any of these, and the driver is closed in every case.
    """
    camera = DriverCamera(driver)
    info = camera.open()
    log.debug("dark session on a %s camera", info.driver)
    try:
        return record_dark_set(camera, library, profile, clock, options, say=say)
    finally:
        camera.close()


__all__ = [
    "DarkAborted",
    "DarkCamera",
    "DarkPhase",
    "DarkProgress",
    "DarkSessionOptions",
    "DarkSessionResult",
    "DriverCamera",
    "record_dark_set",
    "run_dark_session",
]
