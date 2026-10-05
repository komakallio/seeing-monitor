"""The readings of the rapid focus mode: the star width 20 times a second, and its best value.

The person focuses by hand and watches a number fall. The focus value of the normal alignment view
comes from the bin2 frame twice a second, which is too slow and too coarse for that. In the rapid
focus mode the scheduler streams the fast readout mode on a ROI around Polaris (128 by 128 pixels in
bin1, 1.91 arcseconds per pixel, about 82 frames a second), and `RapidFocusHelper` measures every
frame.

**The frame.** The kernel of the fast path (`seeingmon.fastpath.kernel.measure_frame`) gives the
centroid, the second-moment widths, and the peak of each frame in about 20 to 100 microseconds on a
development machine, with the aperture and the saturation level that the fast analyzer uses for the
stream. The helper hands the frame to the video of Polaris with the star that it found, so the video
and the star of each frame come from the code of the Now page. It never gives a frame to the fast
analyzer, so no frame reaches a stored window.

**The background of the width.** The kernel takes the background from the median of the border of
the ROI. The pixels are whole counts, and the noise of the camera at gain 0 (0.76 counts) dithers
them only partly, so the median of the border can differ from the mean level of the sky by up to
half a count. A half count is 1.7 electrons in each of the 200 pixels of the aperture, and the
second moment weights them by the squared distance from the star, so the width of a star in focus
(about 1 pixel sigma) can read up to 30% off: too wide when the median sits below the mean, and too
narrow when it sits above. The centroid does not care. The helper therefore corrects the second
moments for the difference between the mean of the border (clipped at five times the noise of a
pixel, so that a hot pixel does not count) and the median that the kernel used, with the sums of
the aperture that the kernel keeps. In the tests, where the median sits below the mean, the
corrected width of a Gaussian star comes back within 1% of its true width from 0.6 to 2.2 pixels
sigma, where the kernel alone reads 2 to 30% too wide. The stored windows and the per-frame width
of the video still use the median.

**The reading.** A reading covers an interval of 50 milliseconds of frame time, on a grid that
starts with the first frame of the session, which makes 20 readings a second at any frame rate.
Its width is the median over the usable frames of the interval (a star that the kernel found, that
is not at the ROI edge and not an isolated bright pixel) of the mean of the two second-moment
widths, times 2.355, through the plate scale of the readout mode. Its peak is the brightest pixel
of the interval as a share of the full scale of the ADC, and `saturated` says that a frame of the
interval reached the saturation level, where a star reads too narrow. An interval with no usable
frame gives no reading.

**Spikes.** Touching the telescope or a gust inflates the width for a moment. A reading above twice
the median of the preceding ten readings has the flag `spike`, with the rule of the focus history of
the normal view (`seeingmon.services.core.alignment.focus`): the preceding readings include earlier
spikes, and the first three readings of a session are never spikes.

**The best value.** The lowest single reading is a poor best value: each reading scatters by a few
percent, and the lowest of hundreds of them sits far below what the star shows at rest, so the
current value never reaches it. The best value is the lowest *smoothed* reading instead: the median
of the last 20 readings (one second) that are neither spikes nor saturated, once ten of them exist.
A saturated star reads too narrow, so it never sets the best value. `reset_best` restarts it, and
the history stays.

**The history.** The helper keeps the last 600 readings (30 seconds) in parallel columns, so that a
reading costs a few appends, and `snapshot` copies the columns for the contract. A session starts
with `begin_session`, which clears the history and numbers the session, and ends with `end_session`,
which keeps the reason for the page.

**Threads.** The scheduler thread calls `push`, and `begin_session` and `end_session` may run on the
thread of a command. The scheduler thread owns the state of the frame loop (the tracked position and
the open interval), and `begin_session` asks it to start fresh by counting an epoch. A lock guards
the history and the facts that the other threads read. `snapshot` is safe to call from any thread.
"""

from __future__ import annotations

import math
import statistics
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from seeingmon.analysis.base import NO_STAR, FastUpdate, StarState
from seeingmon.clock import NS_PER_S, Clock, utc_ns_to_iso
from seeingmon.fastpath.kernel import (
    FLAG_EDGE,
    FLAG_HOT_PIXEL,
    FLAG_SATURATED,
    FrameCalibration,
    KernelParams,
    Measurement,
    _border_indices,
    measure_frame,
)
from seeingmon.frames import Frame, FrameData, Roi
from seeingmon.profile import Profile, ProfileError
from seeingmon.services.core.alignment.focus import (
    SPIKE_FACTOR,
    SPIKE_MIN_PREVIOUS,
    SPIKE_WINDOW,
)
from seeingmon.services.core.polaris import FWHM_PER_SIGMA
from seeingmon.services.web.contract import (
    MAX_RAPID_READINGS,
    RapidFocusView,
    RapidLocatedBy,
    RapidReadingsView,
    RoiView,
)

READING_INTERVAL_S = 0.05
HISTORY_LENGTH = MAX_RAPID_READINGS
BEST_SPAN = 20
BEST_MIN = 10
UNUSABLE_FLAGS = FLAG_EDGE | FLAG_HOT_PIXEL

SATURATED_NOTE = (
    "the star reaches the saturation level, so its width reads too small: shorten the exposure or "
    "lower the gain"
)

KernelSetup = Callable[[str, int, int, int, int], tuple[KernelParams, FrameCalibration]]


class FrameOffer(Protocol):
    """What the helper needs from the video of Polaris. `PolarisStream` fits."""

    def offer(self, frame: Frame, update: FastUpdate, rapid: bool = False) -> None: ...


@dataclass(frozen=True, slots=True)
class RapidAvailability:
    """Whether the mode is offered now, and where it would put its ROI.

    `reason` says in words what is missing, and it is `None` when the mode is offered.
    `located_by` says what tells `core` where Polaris is. `coarse_fwhm_arcsec` is the coarse focus,
    and `max_fwhm_arcsec` is the limit that it has to meet. `center` is the center of the ROI in
    pixels of the fast readout mode, and it is `None` while Polaris is not located.
    """

    available: bool
    reason: str | None
    located_by: RapidLocatedBy | None
    coarse_fwhm_arcsec: float | None
    max_fwhm_arcsec: float
    center: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class RapidReading:
    """One reading. `index` counts the readings of the session from 1 and never repeats."""

    index: int
    t_utc_ns: int
    fwhm_arcsec: float
    peak_fraction: float
    n_frames: int
    spike: bool
    saturated: bool


@dataclass(frozen=True, slots=True)
class RapidColumns:
    """The history as the parallel lists of `RapidReadingsView`, oldest first."""

    index: list[int]
    t_utc_ms: list[int]
    fwhm_arcsec: list[float]
    peak_fraction: list[float]
    n_frames: list[int]
    spike: list[bool]
    saturated: list[bool]

    def __len__(self) -> int:
        return len(self.index)


class RapidHistory:
    """The last readings, the spike flag, and the best value. Not thread-safe: hold a lock."""

    def __init__(self, length: int = HISTORY_LENGTH) -> None:
        if length < 1:
            raise ValueError("the history holds at least one reading")
        self._index: deque[int] = deque(maxlen=length)
        self._t_utc_ms: deque[int] = deque(maxlen=length)
        self._fwhm: deque[float] = deque(maxlen=length)
        self._peak: deque[float] = deque(maxlen=length)
        self._n_frames: deque[int] = deque(maxlen=length)
        self._spike: deque[bool] = deque(maxlen=length)
        self._saturated: deque[bool] = deque(maxlen=length)
        self._preceding: deque[float] = deque(maxlen=SPIKE_WINDOW)
        self._smooth: deque[float] = deque(maxlen=BEST_SPAN)
        self._best: float | None = None
        self._next_index = 1
        self._session = 0

    @property
    def session(self) -> int:
        """The number of the session. It grows each time that `clear` starts a new one."""
        return self._session

    @property
    def best_arcsec(self) -> float | None:
        """The best value of the session or since `reset_best`, or `None`."""
        return self._best

    def add(
        self,
        t_utc_ns: int,
        fwhm_arcsec: float,
        peak_fraction: float,
        n_frames: int,
        saturated: bool,
    ) -> RapidReading | None:
        """Add a reading and return it. A width that is not a positive number is ignored."""
        if not math.isfinite(fwhm_arcsec) or fwhm_arcsec <= 0.0:
            return None
        spike = len(self._preceding) >= SPIKE_MIN_PREVIOUS and fwhm_arcsec > SPIKE_FACTOR * (
            statistics.median(self._preceding)
        )
        reading = RapidReading(
            self._next_index,
            t_utc_ns,
            round(fwhm_arcsec, 3),
            round(peak_fraction, 4),
            n_frames,
            spike,
            saturated,
        )
        self._next_index += 1
        self._index.append(reading.index)
        self._t_utc_ms.append(t_utc_ns // 1_000_000)
        self._fwhm.append(reading.fwhm_arcsec)
        self._peak.append(reading.peak_fraction)
        self._n_frames.append(n_frames)
        self._spike.append(spike)
        self._saturated.append(saturated)
        self._preceding.append(fwhm_arcsec)
        if not spike and not saturated:
            self._smooth.append(fwhm_arcsec)
            if len(self._smooth) >= BEST_MIN:
                value = statistics.median(self._smooth)
                if self._best is None or value < self._best:
                    self._best = round(value, 3)
        return reading

    def reset_best(self) -> None:
        """Forget the best value. Later readings set the next one, and the history stays."""
        self._best = None
        self._smooth.clear()

    def clear(self) -> None:
        """Start a new session: no readings, no best value, and a new session number."""
        for column in (
            self._index,
            self._t_utc_ms,
            self._fwhm,
            self._peak,
            self._n_frames,
            self._spike,
            self._saturated,
            self._preceding,
            self._smooth,
        ):
            column.clear()
        self._best = None
        self._next_index = 1
        self._session += 1

    def columns(self) -> RapidColumns:
        """A copy of the columns."""
        return RapidColumns(
            list(self._index),
            list(self._t_utc_ms),
            list(self._fwhm),
            list(self._peak),
            list(self._n_frames),
            list(self._spike),
            list(self._saturated),
        )


@dataclass(frozen=True, slots=True)
class RapidFacts:
    """What the helper knows about the stream of the newest reading."""

    mode: str
    exposure_us: int
    gain: int
    roi: Roi
    scale_arcsec_px: float | None


@dataclass(frozen=True, slots=True)
class RapidSnapshot:
    """The state of the mode at one moment, for the views.

    `since_utc` is the start of the session as ISO 8601, `ended` is the reason that the last session
    ended on its own (it is `None` while a session runs and before the first one), `star_found` says
    whether the newest frame showed the star, and `facts` describe the stream of the newest reading.
    """

    session: int
    active: bool
    since_utc: str | None
    ended: str | None
    columns: RapidColumns
    best_arcsec: float | None
    star_found: bool
    facts: RapidFacts | None


@dataclass(frozen=True, slots=True)
class _Setup:
    """What the loop of the frames derives from the stream: the kernel and the plate scale."""

    key: tuple[str, int, int, int, int]
    params: KernelParams
    calibration: FrameCalibration
    scale_arcsec_px: float | None
    area_px2: float  # the sum of the weights of the aperture
    second_moment_px4: float  # the sum of the weights times the squared distance along one axis
    clip_dn: float | None  # five times the noise of a pixel in container counts, if it is known


class RapidFocusHelper:
    """The `FocusSink` of `core`: measure the frames of the rapid focus mode. See the module text.

    `polaris` is the video of Polaris, which gets every frame with its star. `kernel_setup` gives
    the kernel parameters and the calibration of a stream, as `FastPathAnalyzer.kernel_setup` does,
    so that the aperture and the saturation level follow the stored windows. Without it the helper
    uses the default aperture and the full scale of the container. A readout mode that the profile
    does not describe gives no reading, because the width needs the plate scale.
    """

    def __init__(
        self,
        *,
        profile: Profile,
        clock: Clock,
        polaris: FrameOffer | None = None,
        kernel_setup: KernelSetup | None = None,
        interval_s: float = READING_INTERVAL_S,
        length: int = HISTORY_LENGTH,
    ) -> None:
        if not 0.0 < interval_s < 10.0:
            raise ValueError("the interval of a reading is between 0 and 10 seconds")
        self._profile = profile
        self._clock = clock
        self._polaris = polaris
        self._kernel_setup = kernel_setup
        self._interval_ns = max(1, round(interval_s * NS_PER_S))

        self._lock = threading.Lock()
        self._history = RapidHistory(length)
        self._active = False
        self._since_utc: str | None = None
        self._ended: str | None = None
        self._facts: RapidFacts | None = None
        self._epoch = 0  # counts the sessions that began, for the loop of the frames
        # The state of the frame loop belongs to the scheduler thread.
        self._seen_epoch = 0
        self._setup: _Setup | None = None
        self._scales: dict[str, float | None] = {}
        self._guess: tuple[float, float] | None = None
        self._star_found = False
        self._t0_ns: int | None = None
        self._slot = 0
        self._widths: list[float] = []
        self._peak = 0.0
        self._saturated_frames = 0
        self._first_ns = 0
        self._last_ns = 0
        self.frames = 0
        self.readings = 0

    # --- The session (the FocusSink) --------------------------------------------------------

    def begin_session(self) -> None:
        """Start a session: clear the history, and number it. The frame loop starts fresh."""
        with self._lock:
            self._history.clear()
            self._active = True
            self._ended = None
            self._facts = None
            self._since_utc = utc_ns_to_iso(self._clock.utc_ns(), digits=3)
            self._epoch += 1

    def end_session(self, reason: str) -> None:
        """End the session. The readings stay for a last look, and `reason` stays for the page."""
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._ended = reason

    @property
    def active(self) -> bool:
        """Whether a session runs."""
        return self._active

    def reset_best(self) -> None:
        """Restart the best value, for example after a refocus. The readings stay."""
        with self._lock:
            self._history.reset_best()

    def last_star(self) -> tuple[float, float] | None:
        """Where the newest frame showed the star, in sensor pixels, or `None`."""
        return self._guess

    # --- The frames (the scheduler thread) --------------------------------------------------

    def push(self, frame: Frame) -> StarState | None:
        """Measure a frame, offer it to the video, and return the star. Returns at once."""
        if not self._active:
            return None
        if self._epoch != self._seen_epoch:
            self._start_loop()
        setup = self._setup_for(frame)
        roi = frame.roi
        measured = measure_frame(
            frame.data, roi.x, roi.y, setup.params, setup.calibration, self._guess
        )
        self.frames += 1
        if measured.found:
            self._guess = (measured.x, measured.y)
            star = StarState(
                True,
                measured.x,
                measured.y,
                measured.peak_dn / setup.calibration.full_scale_dn,
                roi.distance_to_edge(measured.x, measured.y),
            )
            self._star_found = True
        else:
            self._guess = None
            star = NO_STAR
            self._star_found = False
        t_ns = frame.t_utc_ns
        t0 = self._t0_ns
        if t0 is None or t_ns < t0:  # the first frame, or the time stepped back
            self._t0_ns = t_ns
            self._slot = 0
            self._open_interval()
        else:
            slot = (t_ns - t0) // self._interval_ns
            if slot != self._slot:
                self._close_interval(frame, setup)
                self._slot = slot
        if measured.found and not measured.flags & UNUSABLE_FLAGS:
            if not self._widths:
                self._first_ns = t_ns
            self._widths.append(_width_px(frame.data, measured, setup))
            self._last_ns = t_ns
            peak_fraction = star.peak_fraction or 0.0
            if peak_fraction > self._peak:
                self._peak = peak_fraction
            if measured.flags & FLAG_SATURATED:
                self._saturated_frames += 1
        polaris = self._polaris
        if polaris is not None:
            polaris.offer(frame, FastUpdate(star=star), rapid=True)
        return star

    def _start_loop(self) -> None:
        """A new session began: forget the position and the open interval."""
        self._seen_epoch = self._epoch
        self._guess = None
        self._star_found = False
        self._t0_ns = None
        self._slot = 0
        self._open_interval()

    def _open_interval(self) -> None:
        self._widths = []
        self._peak = 0.0
        self._saturated_frames = 0

    def _setup_for(self, frame: Frame) -> _Setup:
        bits = 8 * frame.data.dtype.itemsize
        key = (frame.mode, frame.gain, frame.exposure_us, frame.adc_bits, bits)
        setup = self._setup
        if setup is not None and setup.key == key:
            return setup
        params: KernelParams
        calibration: FrameCalibration
        if self._kernel_setup is not None:
            params, calibration = self._kernel_setup(*key)
        else:
            params = KernelParams()
            calibration = FrameCalibration.for_container(
                adc_bits=frame.adc_bits, container_bits=bits
            )
        if frame.mode not in self._scales:
            try:
                self._scales[frame.mode] = self._profile.plate_scale_arcsec_per_px(frame.mode)
            except ProfileError:
                self._scales[frame.mode] = None
        noise_e = math.sqrt(calibration.pixel_var_e2)  # NaN for a mode that the profile lacks
        clip_dn = 5.0 * noise_e / calibration.e_per_dn
        setup = _Setup(
            key,
            params,
            calibration,
            self._scales[frame.mode],
            params.area_px2,
            params.second_moment_px4,
            clip_dn if math.isfinite(clip_dn) and clip_dn > 0.0 else None,
        )
        self._setup = setup
        return setup

    def _close_interval(self, frame: Frame, setup: _Setup) -> None:
        """The interval ended: make its reading from the usable frames that it holds."""
        widths = self._widths
        if widths and setup.scale_arcsec_px is not None:
            width_px = statistics.median(widths) if len(widths) > 1 else widths[0]
            fwhm = FWHM_PER_SIGMA * width_px * setup.scale_arcsec_px
            middle = (self._first_ns + self._last_ns) // 2
            roi = frame.roi
            facts = RapidFacts(
                frame.mode,
                frame.exposure_us,
                frame.gain,
                roi,
                setup.scale_arcsec_px,
            )
            with self._lock:
                if self._active and self._history.add(
                    middle, fwhm, self._peak, len(widths), self._saturated_frames > 0
                ):
                    self._facts = facts
                    self.readings += 1
        self._open_interval()

    # --- The state (any thread) -------------------------------------------------------------

    def snapshot(self) -> RapidSnapshot:
        """The state of the mode now. The lists are copies, so the caller may keep them."""
        with self._lock:
            return RapidSnapshot(
                session=self._history.session,
                active=self._active,
                since_utc=self._since_utc,
                ended=self._ended,
                columns=self._history.columns(),
                best_arcsec=self._history.best_arcsec,
                star_found=self._star_found,
                facts=self._facts,
            )


def _width_px(data: FrameData, measured: Measurement, setup: _Setup) -> float:
    """The mean of the two second-moment widths of a measured star, in pixels, background corrected.

    The kernel subtracted the median of the border of the ROI. With `delta` the mean of that border
    above the median, the sums of the aperture change by `delta` times its area (the flux) and by
    `delta` times its second moment (the sums of the squared distances), which gives the variances
    that the correct background would have given. The aperture sits on the star, so the offset of
    the centroid from its center, which the variance subtracts, is a few hundredths of a pixel and
    stays out of the correction. A correction that would take most of the flux, or a mode without
    a noise model, leaves the widths of the kernel as they are.
    """
    var_x = measured.width_x * measured.width_x
    var_y = measured.width_y * measured.width_y
    clip = setup.clip_dn
    flux = measured.flux_dn
    if clip is not None and flux > 0.0:
        height, width = data.shape
        indices, _, _ = _border_indices(
            height, width, setup.params.border_px, setup.params.border_step
        )
        ring = data.reshape(-1)[indices] - measured.bg_dn  # a new array of floats
        np.maximum(ring, -clip, out=ring)  # the ufuncs cost a few microseconds, `np.clip` more
        np.minimum(ring, clip, out=ring)
        delta = float(np.add.reduce(ring)) / len(ring)
        remaining = flux - delta * setup.area_px2
        if remaining > 0.5 * flux:
            shift = delta * setup.second_moment_px4
            var_x = max((var_x * flux - shift) / remaining, 0.0)
            var_y = max((var_y * flux - shift) / remaining, 0.0)
    return 0.5 * (math.sqrt(var_x) + math.sqrt(var_y))


def build_view(
    snapshot: RapidSnapshot, availability: RapidAvailability | None = None
) -> RapidFocusView:
    """The state of the mode as the contract describes it.

    While a session runs, the view carries the readings, and `available` is `true`, because a start
    request then keeps the mode alive. Otherwise it carries the `availability`, and the reason that
    the last session ended on its own. A session that has ended shows no readings.
    """
    if not snapshot.active:
        info = availability
        return RapidFocusView(
            available=False if info is None else info.available,
            reason=None if info is None else info.reason,
            located_by=None if info is None else info.located_by,
            coarse_fwhm_arcsec=None if info is None else info.coarse_fwhm_arcsec,
            max_fwhm_arcsec=None if info is None else info.max_fwhm_arcsec,
            active=False,
            ended_reason=snapshot.ended,
        )
    columns = snapshot.columns
    quality: dict[str, str] = {}
    facts = snapshot.facts
    latest = len(columns) - 1
    fwhm = columns.fwhm_arcsec[latest] if latest >= 0 else None
    saturated = bool(columns.saturated[latest]) if latest >= 0 else False
    if latest < 0:
        quality["fwhm_arcsec"] = "no reading yet: the star has not shown in the frames"
    if snapshot.best_arcsec is None:
        quality["best_fwhm_arcsec"] = (
            "the best value needs ten readings that are not spikes and not saturated"
        )
    if saturated:
        quality["saturated"] = SATURATED_NOTE
    roi = None if facts is None else facts.roi
    return RapidFocusView(
        available=True,
        active=True,
        since_utc=snapshot.since_utc,
        mode=None if facts is None else facts.mode,
        exposure_us=None if facts is None else facts.exposure_us,
        gain=None if facts is None else facts.gain,
        roi=None if roi is None else RoiView(x=roi.x, y=roi.y, width=roi.width, height=roi.height),
        scale_arcsec_px=None if facts is None else facts.scale_arcsec_px,
        n_stars=1 if snapshot.star_found else 0,
        fwhm_arcsec=fwhm,
        best_fwhm_arcsec=snapshot.best_arcsec,
        peak_fraction=columns.peak_fraction[latest] if latest >= 0 else None,
        spike=bool(columns.spike[latest]) if latest >= 0 else False,
        saturated=saturated,
        readings=readings_view(snapshot.session, columns),
        quality=quality,
    )


def readings_view(session: int, columns: RapidColumns) -> RapidReadingsView:
    """The columns as the contract describes them, with `reset` true.

    The lists come from the helper and have one length, so the view skips the validation of
    every item, which costs more than copying the lists.
    """
    fields: dict[str, Any] = {
        "session": session,
        "reset": True,
        "index": columns.index,
        "t_utc_ms": columns.t_utc_ms,
        "fwhm_arcsec": columns.fwhm_arcsec,
        "peak_fraction": columns.peak_fraction,
        "n_frames": columns.n_frames,
        "spike": columns.spike,
        "saturated": columns.saturated,
    }
    return RapidReadingsView.model_construct(**fields)
