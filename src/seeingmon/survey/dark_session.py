"""Record a dark set with the camera: the flow behind `seeingmon dark`.

The camera has no lens cap, so you cover it. The session

1. takes bias frames at the shortest exposure, which give the bias level and the read noise,
2. waits until a short test frame is dark, which means that you have covered the camera
   (skip this with `wait=False`),
3. records dark frames at the survey exposure, and checks each one,
4. builds the master dark (the per-pixel median), finds the hot pixels, and adds the set to the
   `DarkLibrary`, and
5. reports the dark rate, the dark model, and whether the library still needs a set.

The session takes the driver, the clock, and the library as arguments, so a test runs it against
the `sim` driver on a `VirtualClock` with no real waiting. It closes the driver when it ends,
and it never writes a path, a host, or a serial number to the output.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.drivers.base import CameraDriver
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

log = logging.getLogger("seeingmon.survey")

_READ_MARGIN_S = 30.0  # a read waits this long beyond the exposure
_FALLBACK_TEMPERATURE_C = 20.0  # for the check of a frame that carries no temperature


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


def _take(driver: CameraDriver, exposure_s: float) -> Frame:
    driver.start()
    try:
        return driver.read_frame(timeout_s=exposure_s + _READ_MARGIN_S)
    finally:
        driver.stop()


def _configure(driver: CameraDriver, options: DarkSessionOptions, exposure_s: float) -> None:
    driver.configure(
        StreamConfig(
            options.mode,
            max(1, round(exposure_s * 1e6)),
            options.gain,
            kind=StreamKind.SNAPSHOT,
            offset=options.offset,
        )
    )


def _temperature_of(frame: Frame, driver: CameraDriver) -> float:
    if frame.temperature_c is not None:
        return frame.temperature_c
    reading = driver.read_temperature_c()
    return _FALLBACK_TEMPERATURE_C if reading is None else reading


def _mean_temperature(frames: list[Frame], driver: CameraDriver) -> tuple[float, float]:
    """The mean sensor temperature of the frames and the spread (maximum minus minimum)."""
    values = [frame.temperature_c for frame in frames if frame.temperature_c is not None]
    if not values:
        reading = driver.read_temperature_c()
        if reading is None:
            raise DarkError(
                "the camera reports no sensor temperature, and the dark model needs one"
            )
        values = [reading]
    return float(np.mean(values)), float(max(values) - min(values))


def _record_bias(
    driver: CameraDriver, cfg: DarkSessionOptions, min_exposure_s: float, say: Callable[[str], None]
) -> tuple[float, float]:
    """Take the bias frames. Returns the bias level and the read noise, both in counts."""
    say(f"Taking {cfg.bias_frames} bias frames at the shortest exposure.")
    _configure(driver, cfg, min_exposure_s)
    frames = [native_u16(_take(driver, min_exposure_s)) for _ in range(cfg.bias_frames)]
    level = float(np.mean([clipped_mean(frame) for frame in frames]))
    difference = frames[0].astype(np.float32) - frames[1].astype(np.float32)
    read_noise = robust_level(difference)[1] / float(np.sqrt(2.0))
    say(f"The bias is {level:.1f} counts, and the read noise {read_noise:.2f} counts.")
    return level, read_noise


def _wait_for_cover(
    driver: CameraDriver,
    cfg: DarkSessionOptions,
    clock: Clock,
    say: Callable[[str], None],
    setup: _Setup,
) -> float:
    """Take short frames until `stable_polls` in a row are dark. Returns the time waited."""
    say("Cover the camera now. Waiting for a dark frame.")
    _configure(driver, cfg, cfg.test_exposure_s)
    started_ns = clock.monotonic_ns()
    stable = 0
    last_reason = "no frame yet"
    while True:
        waited_s = (clock.monotonic_ns() - started_ns) / NS_PER_S
        if waited_s > cfg.wait_timeout_s:
            raise DarkError(
                f"the camera was not dark after {cfg.wait_timeout_s:.0f} s ({last_reason}); "
                "cover it, and check the cover for gaps"
            )
        frame = _take(driver, cfg.test_exposure_s)
        check = setup.check(frame, cfg.test_exposure_s, cfg.check, _temperature_of(frame, driver))
        if check.ok:
            stable += 1
            if stable >= cfg.stable_polls:
                say("The camera is dark.")
                return (clock.monotonic_ns() - started_ns) / NS_PER_S
        else:
            stable = 0
            last_reason = check.reason
        clock.sleep(cfg.poll_s)


def run_dark_session(
    driver: CameraDriver,
    library: DarkLibrary,
    profile: Profile,
    clock: Clock,
    options: DarkSessionOptions | None = None,
    *,
    say: Callable[[str], None] = print,
) -> DarkSessionResult:
    """Record a dark set and add it to the library. See the module documentation.

    Raises `DarkError` when the camera never becomes dark, when a recorded frame is not dark, or
    when the camera gives no temperature, and `seeingmon.drivers.CameraError` for a camera fault.
    The library stays as it was after any of these.
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
    info = driver.open()
    log.debug("dark session on a %s camera", info.driver)
    try:
        bias_dn, read_noise_dn = _record_bias(
            driver, cfg, profile.limits.exposure_us_range[0] / 1e6, say
        )
        setup = replace(setup, bias_dn=bias_dn)
        waited_s = _wait_for_cover(driver, cfg, clock, say, setup) if cfg.wait else 0.0
        say(f"Recording {cfg.frames} dark frames of {cfg.exposure_s:g} s.")
        _configure(driver, cfg, cfg.exposure_s)
        frames: list[Frame] = []
        stack: list[npt.NDArray[np.uint16]] = []
        levels: list[float] = []
        last_check: DarkCheck | None = None
        for index in range(cfg.frames):
            frame = _take(driver, cfg.exposure_s)
            last_check = setup.check(
                frame, cfg.exposure_s, cfg.check, _temperature_of(frame, driver)
            )
            if not last_check.ok:
                raise DarkError(
                    f"dark frame {index + 1} of {cfg.frames} is not dark: {last_check.reason}"
                )
            frames.append(frame)
            stack.append(native_u16(frame))
            levels.append(last_check.level_dn)
        assert last_check is not None
        temperature_c, spread_c = _mean_temperature(frames, driver)
        if spread_c > cfg.max_temperature_spread_c:
            say(
                f"Warning: the sensor temperature changed by {spread_c:.1f} C during the set, "
                "so the set describes the mean."
            )
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
    finally:
        driver.close()
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
    return DarkSessionResult(dark_set, last_check, waited_s, status, model, n_sets)
