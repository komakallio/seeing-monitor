"""Frame times from arrival times: a sliding linear fit, and an honest error bound.

The camera SDK returns no frame timestamp, so the driver stamps each frame with the real-time
clock when the read returns (`Frame.t_arrival_ns`). That stamp carries the jitter of the
operating system: scheduling, USB, and the read itself. The frames themselves come at a steady
rate, so the `TimeStamper` fits a line through arrival time against frame number over a sliding
window of recent frames and reads the frame time from the line. The jitter averages out, and the
error of the fitted time is a small fraction of the jitter.

**The convention.** The simulator, the fake driver, and the `asi` driver all place the middle of
the exposure of the first row at `arrival - period + exposure / 2`, where `period` is the frame
period of the stream. The `TimeStamper` uses the same offset, plus the configured `latency_s`: the
delay that commissioning measures with a light pulse. The latency is the one number that shifts
every time of one installation.

**Quality.** The stamper reports what it knows:

- `FITTED`: the window holds at least `warmup` frames, and the frame agrees with the line.
- `ESTIMATED`: the line is not warm yet, or this frame disagrees with it. A disagreeing frame (an
  outlier, such as a late delivery after a stall) gets the time that the line predicts, and its
  error grows by the disagreement.
- `INVALID` with `FrameFlag.TIME_INVALID`: the clock says that it is not synchronized.

**Counting frames.** The line needs the camera's frame number, which counts the frames that a
drop removed. `stamp` takes `dropped_before`, the frames lost since the previous frame, and adds
them to the count. A wrong count shows up as a disagreement of a whole number of frame periods.

**Steps.** When `step_frames` frames in a row disagree with the line, the clock stepped (an NTP
step, or the first synchronization after a boot) or the count was wrong. The stamper starts a new
line from the current frame and reports `ESTIMATED` until it is warm again. It also starts a new
line when `Clock.status().synchronized` changes, because the clock likely stepped then.

**Error.** `t_err_ns` is the 1-sigma error. It is the clock's error bound (from
`Clock.status()`, or `unknown_clock_error_s` when the clock does not say) plus the latency
uncertainty plus the error of the fit. The architecture states the same sum.

The arithmetic keeps exact integer sums, so a stream of any length loses no precision.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.frames import FrameFlag, TimeQuality

REBASE_AT = 1 << 22  # frame numbers beyond this move to a new origin, to keep the integers small
_SIGMA_WEIGHT = 1 / 16  # the weight of one frame in the running estimate of the jitter


@dataclass(frozen=True, slots=True)
class TimingConfig:
    """The settings of a `TimeStamper`. Times are in seconds."""

    window: int = 128
    warmup: int = 20
    latency_s: float = 0.0
    latency_sigma_s: float = 0.005
    arrival_jitter_s: float = 0.002
    unknown_clock_error_s: float = 0.1
    invalid_clock_error_s: float = 86_400.0
    outlier_sigmas: float = 6.0
    outlier_floor_s: float = 0.001
    step_frames: int = 3

    def __post_init__(self) -> None:
        if self.warmup < 4 or self.window < self.warmup:
            raise ValueError("the window must hold at least the warm-up, and the warm-up 4 frames")
        if self.step_frames < 2:
            raise ValueError("step_frames must be at least 2")


@dataclass(frozen=True, slots=True)
class StreamTiming:
    """What the stamper needs to know about the stream: the exposure and the frame period."""

    exposure_s: float
    period_s: float

    def __post_init__(self) -> None:
        if self.exposure_s <= 0 or self.period_s <= 0:
            raise ValueError("the exposure and the period must be positive")


@dataclass(frozen=True, slots=True)
class FrameTime:
    """The time of a frame, as `Frame` carries it."""

    t_utc_ns: int
    t_err_ns: int
    t_quality: TimeQuality
    flags: FrameFlag


class TimeStamper:
    """Turn the arrival times of one stream into frame times. Use one thread."""

    def __init__(self, clock: Clock, config: TimingConfig | None = None) -> None:
        self._clock = clock
        self._config = config or TimingConfig()
        self._offset_ns = 0
        self._period_hint_ns = 0.0
        self._synchronized: bool | None = None
        self._seen_status = False
        self.resets = 0
        self.outliers = 0
        self._clear()

    # --- Setup ---

    def configure(self, timing: StreamTiming) -> None:
        """Start a new stream: forget the line, and set the offset from the exposure and period."""
        half_exposure_s = timing.exposure_s / 2
        self._offset_ns = round(
            (timing.period_s - half_exposure_s + self._config.latency_s) * NS_PER_S
        )
        self._period_hint_ns = timing.period_s * NS_PER_S
        self._clear()

    def reset(self) -> None:
        """Forget the line, as a restart of the process does. The offset stays."""
        self._clear()

    def _clear(self) -> None:
        self._k = -1  # the frame number of the last frame, counting the dropped ones
        self._has_baseline = False
        self._k_base = 0  # the frame number of the origin of the line
        self._a0 = 0  # the arrival time of the origin of the line
        self._points: deque[tuple[int, int]] = deque()
        self._sx = self._sy = self._sxx = self._sxy = 0
        self._sigma2: float | None = None  # variance of the arrival jitter, in ns squared
        self._outlier_run = 0
        self._last_residual_ns = 0.0

    # --- State ---

    @property
    def warm(self) -> bool:
        """Whether the line is warm: it has at least `warmup` frames."""
        return len(self._points) >= self._config.warmup

    @property
    def period_s(self) -> float | None:
        """The frame period that the line measures, or `None` before it is warm."""
        if not self.warm:
            return None
        return self._solve()[1] / NS_PER_S

    @property
    def jitter_s(self) -> float | None:
        """The root-mean-square arrival jitter that the stamper estimates, once it is warm."""
        return None if self._sigma2 is None else math.sqrt(self._sigma2) / NS_PER_S

    # --- The fit ---

    def _solve(self) -> tuple[float, float]:
        """The intercept and the slope of arrival against frame number, in ns."""
        n = len(self._points)
        denominator = n * self._sxx - self._sx * self._sx
        if n < 2 or denominator == 0:
            return float(self._sy) / max(n, 1), self._period_hint_ns
        slope = (n * self._sxy - self._sx * self._sy) / denominator
        intercept = (self._sy - slope * self._sx) / n
        return intercept, slope

    def _add(self, x: int, y: int) -> None:
        self._points.append((x, y))
        self._sx += x
        self._sy += y
        self._sxx += x * x
        self._sxy += x * y
        if len(self._points) > self._config.window:
            old_x, old_y = self._points.popleft()
            self._sx -= old_x
            self._sy -= old_y
            self._sxx -= old_x * old_x
            self._sxy -= old_x * old_y

    def _rebase(self) -> None:
        """Move the origin to the first point, so the integers stay small on a long stream."""
        x0, y0 = self._points[0]
        shifted = [(x - x0, y - y0) for x, y in self._points]
        self._points.clear()
        self._sx = self._sy = self._sxx = self._sxy = 0
        for x, y in shifted:
            self._points.append((x, y))
            self._sx += x
            self._sy += y
            self._sxx += x * x
            self._sxy += x * y
        self._k_base += x0
        self._a0 += y0

    def _start_baseline(self, arrival_ns: int) -> None:
        """Start a new line with the current frame as its origin."""
        self._points.clear()
        self._sx = self._sy = self._sxx = self._sxy = 0
        self._sigma2 = None
        self._outlier_run = 0
        self._k_base = self._k
        self._a0 = arrival_ns
        self._has_baseline = True

    def _restart(self) -> None:
        """Give up the line, because the clock stepped or the frame count broke."""
        self.resets += 1
        self._has_baseline = False

    def _threshold_ns(self) -> float:
        floor = self._config.outlier_floor_s * NS_PER_S
        if self._sigma2 is None:
            return floor
        return max(self._config.outlier_sigmas * math.sqrt(self._sigma2), floor)

    def _agreement_ns(self, period_ns: float) -> float:
        """How closely the residuals of a step agree: the jitter, and at most half a period."""
        floor = self._config.outlier_floor_s * NS_PER_S
        jitter_ns = 0.0 if self._sigma2 is None else 4 * math.sqrt(self._sigma2)
        return min(max(jitter_ns, floor), 0.5 * period_ns)

    def _init_sigma(self) -> None:
        """Estimate the jitter from the residuals of the window, when the line first warms up."""
        intercept, slope = self._solve()
        n = len(self._points)
        total = sum((y - (intercept + slope * x)) ** 2 for x, y in self._points)
        self._sigma2 = total / max(n - 2, 1)

    def _fit_error_ns(self, x: int) -> float:
        """The standard error of the line at frame number `x`, in ns."""
        if self._sigma2 is None:
            return 0.0
        n = len(self._points)
        mean_x = self._sx / n
        spread = self._sxx - self._sx * self._sx / n
        leverage = 1 / n + ((x - mean_x) ** 2 / spread if spread > 0 else 0.0)
        return math.sqrt(self._sigma2 * leverage)

    # --- Stamping ---

    def stamp(self, t_arrival_ns: int, dropped_before: int = 0) -> FrameTime:
        """Return the time of the next frame.

        `t_arrival_ns` is the arrival time that the driver stamped, and `dropped_before` is the
        number of frames that the camera or the driver lost since the previous frame.
        """
        config = self._config
        status = self._clock.status()
        if self._seen_status and status.synchronized != self._synchronized:
            self._restart()  # the clock probably stepped when it changed its mind
        self._synchronized = status.synchronized
        self._seen_status = True

        self._k = 0 if self._k < 0 else self._k + 1 + dropped_before
        if not self._has_baseline:
            self._start_baseline(t_arrival_ns)
        residual_ns = 0.0
        outlier = False
        if self.warm:
            intercept, slope = self._solve()
            x = self._k - self._k_base
            residual_ns = (t_arrival_ns - self._a0) - (intercept + slope * x)
            if abs(residual_ns) > self._threshold_ns():
                # A step or a wrong count disagrees by the same amount frame after frame. A burst
                # after a stall does not: each frame is one period closer to the line.
                if self._outlier_run and abs(residual_ns - self._last_residual_ns) <= (
                    self._agreement_ns(slope)
                ):
                    self._outlier_run += 1
                else:
                    self._outlier_run = 1
                self._last_residual_ns = residual_ns
                if self._outlier_run >= config.step_frames:
                    self._restart()
                    self._start_baseline(t_arrival_ns)
                    residual_ns = 0.0
                else:
                    outlier = True
                    self.outliers += 1
            else:
                self._outlier_run = 0
        if outlier:
            intercept, slope = self._solve()
            fitted_ns = self._a0 + round(intercept + slope * (self._k - self._k_base))
            extra_err_ns = abs(residual_ns)
            quality = TimeQuality.ESTIMATED
        else:
            was_warm = self.warm
            self._add(self._k - self._k_base, t_arrival_ns - self._a0)
            if self._k - self._k_base > REBASE_AT:
                self._rebase()
            if self.warm:
                if was_warm and self._sigma2 is not None:
                    self._sigma2 += _SIGMA_WEIGHT * (residual_ns * residual_ns - self._sigma2)
                else:
                    self._init_sigma()
                x = self._k - self._k_base
                intercept, slope = self._solve()
                fitted_ns = self._a0 + round(intercept + slope * x)
                extra_err_ns = self._fit_error_ns(x)
                quality = TimeQuality.FITTED
            else:
                fitted_ns = t_arrival_ns
                extra_err_ns = config.arrival_jitter_s * NS_PER_S
                quality = TimeQuality.ESTIMATED
        t_utc_ns = fitted_ns - self._offset_ns
        if status.synchronized is False:
            return FrameTime(
                t_utc_ns,
                round(config.invalid_clock_error_s * NS_PER_S),
                TimeQuality.INVALID,
                FrameFlag.TIME_INVALID,
            )
        bound_ns = (
            round(config.unknown_clock_error_s * NS_PER_S)
            if status.error_bound_ns is None
            else status.error_bound_ns
        )
        t_err_ns = bound_ns + round(config.latency_sigma_s * NS_PER_S) + round(extra_err_ns)
        return FrameTime(t_utc_ns, t_err_ns, quality, FrameFlag.NONE)
