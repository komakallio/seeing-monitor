"""Record a flat: the flow behind the flat session of the web UI.

You cover the front of the guide scope with a uniform light source (an LED panel, or a white cloth
over the aperture under daylight), and the session does the rest on the camera that the scheduler
lends (see `seeingmon.services.core.commissioning.flat`):

1. **Check.** The dark library must hold a model for the readout mode and the gain, because the bias
   of the flat comes from it. A second set needs the first set of the same session. The disk must
   hold the frames.
2. **Find the exposure.** The session takes a frame, measures the median of the middle of the frame
   above the bias (as a fraction of the full scale), and scales the exposure toward the target. It
   stops within `level_tolerance` of the target, or after `max_iterations` frames. The exposure
   stays between the shortest exposure of the profile and `max_exposure_s`. A light that is too
   weak at the longest exposure, and one that saturates the frame at the shortest, end the session
   with a plain sentence.
3. **Take the frames.** The frames of the set go into a SER file in the session folder of the flat
   library, in native counts, and the session watches each frame: a level that drifts, saturated
   pixels, a light that does not cover the corners, and a level outside what `make_flat` accepts.
4. **Combine.** `make_flat` (`seeingmon.survey.flat_make`) turns the frames into the flat, with the
   bias level from the dark library at the mean sensor temperature of the frames. A second set (the
   source turned by 180 degrees) combines with the first set that the session kept.
5. **Store.** The flat joins the library as a pending flat, with its report and its preview
   (`seeingmon.survey.flat_library`). The first set stays in the session folder for a second
   set. The second set ends the session.

The flow takes the camera, the clock, and the libraries as arguments, so a test runs it against a
scripted camera on a `VirtualClock`. It checks `should_stop` before each frame and before each frame
that the combination reads, and it reads the clock for each of them, so the watchdog of `core` sees
a scheduler that works. It never writes a path, a host, or a serial number to a message.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from seeingmon.clock import Clock
from seeingmon.frames import Frame, StreamConfig, StreamKind
from seeingmon.profile import Profile, ProfileError
from seeingmon.recordings.ser import SerError, SerWriter
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.dark import DarkLibrary
from seeingmon.survey.flat_files import FlatFileError
from seeingmon.survey.flat_library import (
    FlatEntry,
    FlatInfo,
    FlatLibrary,
    FlatLibraryError,
    FlatSession,
    build_report,
    render_preview,
)
from seeingmon.survey.flat_make import (
    BiasInput,
    FlatError,
    Frame2D,
    Geometry,
    MakeOptions,
    MakeResult,
    SerSource,
    library_bias,
    make_flat,
)
from seeingmon.survey.rawdata import native_u16
from seeingmon.survey.sky import SkyError

log = logging.getLogger("seeingmon.survey")

FlatPhase = Literal["setup", "exposure", "capture", "build", "done"]

DARK_FIRST = "Record a dark set first (Dark page)."
NO_FIRST_SET = (
    "There is no first set to combine with. Take the first set, and take the second within "
    "24 hours."
)
SATURATED_FRACTION = 0.98  # of full scale, from where a pixel counts as saturated
MAX_SATURATED_SHARE = 0.001  # more saturated pixels than this warn, as `make_flat` does
MIN_RAW_FRACTION = 0.20  # `make_flat` drops a frame whose mean level lies outside 20 to 80%
MAX_RAW_FRACTION = 0.80
MIN_CORNER_RATIO = 0.5  # corners with less than this share of the light of the middle warn
MIN_FRACTION = 0.002  # a level below this share does not tell how much light there is
MAX_STEP = 50.0  # the largest change of the exposure in one step of the search
SAMPLE_STRIDE = 4  # the median of the middle looks at every fourth pixel in each direction
READ_MARGIN_S = 30.0  # a frame read waits this long beyond the exposure (the camera adapter)
# The combination holds a few images of the frame in memory, and one frame at a time. A run on
# frames of 4144 x 2822 pixels peaked at 421 MB of arrays (36 bytes for each pixel), whatever the
# number of frames, so a check asks for a third more.
MEMORY_BYTES_PER_PIXEL = 48


class FlatSessionError(Exception):
    """The session cannot go on. The message is one plain sentence for the page."""


class FlatAborted(Exception):  # noqa: N818 (a stop on purpose, not an error)
    """`should_stop` ended the session. The library stays as it was."""


class FlatCamera(Protocol):
    """What the session needs from a camera: configure it, and take one frame."""

    def configure(self, config: StreamConfig) -> None: ...

    def take(self, exposure_s: float) -> Frame: ...


# --- The settings -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FlatSessionOptions:
    """What the session records, and how it finds the exposure.

    `target_fraction` is the level to aim at, as a fraction of the full scale. `make` replaces the
    options of `make_flat` (the full scale always comes from the profile). `doubling_c` is the
    prior of the dark model, as in `[survey.dark]`.
    """

    mode: str = "bin2"
    gain: int = 120
    frames: int = 32
    target_fraction: float = 0.5
    set_number: int = 1
    start_exposure_s: float = 0.02
    max_exposure_s: float = 1.0
    max_iterations: int = 8
    level_tolerance: float = 0.1
    min_level_fraction: float = 0.25
    drift_percent: float = 3.0
    doubling_c: float = 6.0
    offset: int | None = None
    make: MakeOptions | None = None

    def __post_init__(self) -> None:
        if self.frames < 3:
            raise ValueError("a flat set needs at least 3 frames")
        if not 0.0 < self.target_fraction < 1.0:
            raise ValueError("the target level is a fraction of the full scale, between 0 and 1")
        if self.set_number not in (1, 2):
            raise ValueError("a session has the sets 1 and 2")
        if self.start_exposure_s <= 0 or self.max_exposure_s <= 0 or self.max_iterations < 1:
            raise ValueError("the exposures must be positive, and the search needs a frame")
        if not 0.0 < self.level_tolerance < 1.0 or not 0.0 <= self.min_level_fraction < 1.0:
            raise ValueError("the tolerance and the lowest level are fractions")

    @classmethod
    def from_config(
        cls,
        survey: SurveyConfig,
        *,
        frames: int,
        target_fraction: float,
        set_number: int,
        make: MakeOptions | None = None,
    ) -> FlatSessionOptions:
        """The options of `[survey.flat]` and `[survey.dark]`, with the values of the command.

        Raises `ValueError` for a value that the options refuse.
        """
        flat = survey.flat
        return cls(
            mode=survey.dark.mode,
            gain=survey.dark.gain,
            frames=frames,
            target_fraction=target_fraction,
            set_number=set_number,
            start_exposure_s=flat.start_exposure_s,
            max_exposure_s=flat.max_exposure_s,
            max_iterations=flat.max_iterations,
            level_tolerance=flat.level_tolerance,
            min_level_fraction=flat.min_level_fraction,
            drift_percent=flat.drift_percent,
            doubling_c=survey.dark.doubling_c,
            make=make,
        )


@dataclass(frozen=True, slots=True)
class FlatProgress:
    """Where a session stands. `step` of `steps` counts what the phase has done so far.

    In `exposure`, a step is one frame of the search. In `capture`, a step is one frame of the set.
    In `setup`, `build`, and `done` the session has one step. `exposure_s` is the exposure in use,
    and `level_fraction` is the latest level above the bias, as a fraction of the full scale.
    `saturated_fraction` is the share of saturated pixels of the latest frame. `warnings` hold
    every note of the session so far.
    """

    phase: FlatPhase
    step: int
    steps: int
    message: str
    exposure_s: float | None = None
    level_fraction: float | None = None
    saturated_fraction: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FlatSessionResult:
    """What a session produced.

    `exposure_s` and `level_fraction` describe the set that this session took. `make` is the result
    of `make_flat`, and `entry` the flat that went into the library.
    """

    entry: FlatEntry
    make: MakeResult
    set_number: int
    exposure_s: float
    level_fraction: float
    frames_taken: int
    temperature_c: float | None
    warnings: tuple[str, ...]


def format_exposure(seconds: float) -> str:
    """An exposure in words: `1.5 s`, `40 ms`, or `32 us`."""
    if seconds >= 100.0:
        return f"{seconds:.0f} s"
    if seconds >= 1.0:
        return f"{seconds:.3g} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3g} ms"
    return f"{seconds * 1e6:.3g} µs"


def _percent(fraction: float) -> str:
    return f"{100.0 * fraction:.0f} %"


def _gigabytes(count: int) -> str:
    return f"{count / 1e9:.1f} GB"


# --- Measuring a frame ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameLevel:
    """What the session measures of one frame, in fractions of the full scale.

    `fraction` is the median of the middle of the frame above the bias, and `raw_fraction` is the
    same median with the bias in it (which is what `make_flat` tests). `corner_ratio` is the mean of
    the four corners above the bias over the middle above the bias, or `None` when the middle holds
    no light. `saturated_fraction` is the share of pixels at or above 98% of the full scale.
    """

    fraction: float
    raw_fraction: float
    saturated_fraction: float
    corner_ratio: float | None
    temperature_c: float | None

    @property
    def saturated(self) -> bool:
        """Whether the frame has more saturated pixels than `make_flat` allows."""
        return self.saturated_fraction > MAX_SATURATED_SHARE or self.raw_fraction >= 0.995


def measure_frame(frame: Frame, *, full_scale: float, bias_dn: float) -> FrameLevel:
    """The level of a frame: the median of its middle half in each direction, above the bias."""
    return measure_native(
        native_u16(frame),
        full_scale=full_scale,
        bias_dn=bias_dn,
        temperature_c=frame.temperature_c,
    )


def measure_native(
    native: npt.NDArray[np.uint16],
    *,
    full_scale: float,
    bias_dn: float,
    temperature_c: float | None = None,
) -> FrameLevel:
    """The level of a frame that is in native counts already. See `measure_frame`."""
    height, width = native.shape
    middle = native[
        height // 4 : 3 * height // 4 : SAMPLE_STRIDE, width // 4 : 3 * width // 4 : SAMPLE_STRIDE
    ]
    raw = float(np.median(middle))
    saturated = float(np.count_nonzero(native >= SATURATED_FRACTION * full_scale)) / native.size
    rows, columns = max(2, height // 10), max(2, width // 10)
    corners = np.concatenate(
        [
            native[:rows, :columns].ravel(),
            native[:rows, -columns:].ravel(),
            native[-rows:, :columns].ravel(),
            native[-rows:, -columns:].ravel(),
        ]
    )
    above = raw - bias_dn
    ratio = (float(corners.mean()) - bias_dn) / above if above > 0 else None
    return FrameLevel(
        fraction=above / full_scale,
        raw_fraction=raw / full_scale,
        saturated_fraction=saturated,
        corner_ratio=ratio,
        temperature_c=temperature_c,
    )


class _Watch:
    """The notes of a session, one for each kind, and the latest wording of each."""

    def __init__(self) -> None:
        self._notes: dict[str, str] = {}

    def note(self, key: str, text: str) -> None:
        self._notes[key] = text

    def texts(self) -> tuple[str, ...]:
        return tuple(self._notes.values())

    def check_frame(self, level: FrameLevel) -> None:
        if level.saturated:
            self.note(
                "saturated",
                f"Too bright: the frame saturates ({100 * level.saturated_fraction:.1f} % of the "
                "pixels). Dim the light.",
            )
        if level.corner_ratio is not None and level.corner_ratio < MIN_CORNER_RATIO:
            self.note(
                "corners",
                "The light may not cover the whole lens: the corners get "
                f"{_percent(max(level.corner_ratio, 0.0))} of the light of the middle.",
            )
        if not MIN_RAW_FRACTION <= level.raw_fraction <= MAX_RAW_FRACTION:
            self.note(
                "band",
                f"The level is {_percent(level.raw_fraction)} of full scale, outside "
                f"{_percent(MIN_RAW_FRACTION)} to {_percent(MAX_RAW_FRACTION)}, so the flat "
                "does not count this frame.",
            )


# --- The search for the exposure ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Setup:
    """The numbers of the readout setting that every step shares."""

    options: FlatSessionOptions
    profile: Profile
    geometry: Geometry
    full_scale: float
    min_exposure_us: int
    max_exposure_us: int
    bias_of: Callable[[float | None], float]


@dataclass(frozen=True, slots=True)
class ExposureChoice:
    """The exposure that the search settled on, and the frame that measured it."""

    exposure_us: int
    level: FrameLevel
    tries: int
    note: str | None = None


class _Session:
    """One run of the flow: the camera, the callbacks, and the notes."""

    def __init__(
        self,
        camera: FlatCamera,
        setup: _Setup,
        clock: Clock,
        say: Callable[[str], None],
        progress: Callable[[FlatProgress], None] | None,
        should_stop: Callable[[], bool] | None,
    ) -> None:
        self.camera = camera
        self.setup = setup
        self.clock = clock
        self.say = say
        self._progress = progress
        self._should_stop = should_stop
        self.watch = _Watch()

    def report(
        self,
        phase: FlatPhase,
        step: int,
        steps: int,
        message: str,
        *,
        exposure_s: float | None = None,
        level: FrameLevel | None = None,
    ) -> None:
        self.say(message)
        if self._progress is not None:
            self._progress(
                FlatProgress(
                    phase,
                    step,
                    steps,
                    message,
                    exposure_s=exposure_s,
                    level_fraction=None if level is None else level.fraction,
                    saturated_fraction=None if level is None else level.saturated_fraction,
                    warnings=self.watch.texts(),
                )
            )

    def stop_requested(self) -> None:
        """Beat for the watchdog, and raise `FlatAborted` when the caller asked to end."""
        self.clock.monotonic_ns()
        if self._should_stop is not None and self._should_stop():
            raise FlatAborted("the flat session was stopped before it added a flat")

    def stream(self, exposure_us: int) -> StreamConfig:
        options = self.setup.options
        return StreamConfig(
            options.mode,
            exposure_us,
            options.gain,
            kind=StreamKind.SNAPSHOT,
            offset=options.offset,
        )

    def take(self, exposure_us: int) -> Frame:
        self.stop_requested()
        return self.camera.take(exposure_us / 1e6)

    def measure(self, frame: Frame) -> FrameLevel:
        setup = self.setup
        return measure_frame(
            frame,
            full_scale=setup.full_scale,
            bias_dn=setup.bias_of(frame.temperature_c),
        )

    def find_exposure(self, start_s: float) -> ExposureChoice:
        """Scale the exposure from `start_s` toward the target, until the level is close enough."""
        setup = self.setup
        options = setup.options
        low, high = setup.min_exposure_us, setup.max_exposure_us
        next_us = min(max(round(start_s * 1e6), low), high)
        target = options.target_fraction
        measured_us, level, tries = next_us, None, 0
        for tries in range(1, options.max_iterations + 1):
            measured_us = next_us
            self.camera.configure(self.stream(measured_us))
            level = self.measure(self.take(measured_us))
            self.report(
                "exposure",
                tries,
                options.max_iterations,
                f"Try {tries} of {options.max_iterations}: {format_exposure(measured_us / 1e6)} "
                f"gives {_percent(level.fraction)} of full scale (target {_percent(target)}).",
                exposure_s=measured_us / 1e6,
                level=level,
            )
            if (
                not level.saturated
                and abs(level.fraction - target) <= options.level_tolerance * target
            ):
                return ExposureChoice(measured_us, level, tries)
            ratio = target / max(level.fraction, MIN_FRACTION)
            if level.saturated:  # the level that we measured is a lower bound
                ratio = min(ratio, 0.125 if level.raw_fraction >= 0.995 else 0.5)
            ratio = min(max(ratio, 1.0 / MAX_STEP), MAX_STEP)
            next_us = min(max(round(measured_us * ratio), low), high)
            if next_us == measured_us:  # the search sits at a limit of the exposure
                break
        assert level is not None
        return self._accept_or_fail(ExposureChoice(measured_us, level, tries))

    def _accept_or_fail(self, choice: ExposureChoice) -> ExposureChoice:
        """Judge the last try of a search that did not land inside the tolerance."""
        setup = self.setup
        options = setup.options
        level, exposure_us = choice.level, choice.exposure_us
        at_low = exposure_us <= setup.min_exposure_us
        at_high = exposure_us >= setup.max_exposure_us
        too_bright = level.saturated or level.raw_fraction > MAX_RAW_FRACTION
        if too_bright and at_low:
            what = (
                "saturates"
                if level.saturated
                else f"reaches {_percent(level.raw_fraction)} of full scale"
            )
            raise FlatSessionError(
                f"Too much light: the frame {what} at the shortest exposure of "
                f"{format_exposure(exposure_us / 1e6)}. Dim the source, or put a layer of cloth "
                "between it and the lens."
            )
        if level.fraction < options.min_level_fraction and at_high:
            raise FlatSessionError(
                f"Not enough light: the frame reaches {_percent(max(level.fraction, 0.0))} of full "
                f"scale at the longest exposure of {format_exposure(exposure_us / 1e6)}. Use a "
                "brighter source, or hold it closer to the lens."
            )
        if too_bright or level.fraction < options.min_level_fraction:
            raise FlatSessionError(
                f"The light is too unsteady to find an exposure: after {choice.tries} tries the "
                f"level is {_percent(max(level.fraction, 0.0))} of full scale, and the target is "
                f"{_percent(options.target_fraction)}. Check that the light is steady and covers "
                "the whole lens."
            )
        note = (
            f"The level is {_percent(level.fraction)} of full scale, not the target of "
            f"{_percent(options.target_fraction)}."
        )
        return ExposureChoice(exposure_us, level, choice.tries, note)

    # --- The frames ---------------------------------------------------------------------------

    def capture(
        self,
        writer: SerWriter,
        exposure_us: int,
        *,
        capture_allowed: Callable[[], bool] | None,
    ) -> tuple[list[FrameLevel], list[float]]:
        """Take the frames of the set into the SER writer. Returns the levels and temperatures."""
        options = self.setup.options
        shape = self.setup.geometry.shape
        self.camera.configure(self.stream(exposure_us))
        levels: list[FrameLevel] = []
        temperatures: list[float] = []
        for index in range(options.frames):
            if capture_allowed is not None and not capture_allowed():
                raise FlatSessionError(
                    "Raw capture stopped because the free disk space fell below the limit."
                )
            frame = self.take(exposure_us)
            if frame.shape != shape:
                raise FlatSessionError(
                    f"The camera gave frames of {frame.shape[1]} x {frame.shape[0]} pixels, and "
                    f"the survey mode has {shape[1]} x {shape[0]}. Take the flat with no region "
                    "of interest."
                )
            native = native_u16(frame)
            level = measure_native(
                native,
                full_scale=self.setup.full_scale,
                bias_dn=self.setup.bias_of(frame.temperature_c),
                temperature_c=frame.temperature_c,
            )
            writer.write_frame(native, frame.t_utc_ns)
            levels.append(level)
            if frame.temperature_c is not None:
                temperatures.append(frame.temperature_c)
            self.watch.check_frame(level)
            self._check_drift(levels)
            self.report(
                "capture",
                index + 1,
                options.frames,
                f"Frame {index + 1} of {options.frames}: {_percent(level.fraction)} of full scale.",
                exposure_s=exposure_us / 1e6,
                level=level,
            )
        return levels, temperatures

    def _check_drift(self, levels: list[FrameLevel]) -> None:
        if len(levels) < 3:
            return
        fractions = [level.fraction for level in levels]
        middle = float(np.median(fractions))
        if middle <= 0:
            return
        worst = max(fractions, key=lambda value: abs(value - middle))
        change = (worst - middle) / middle
        if abs(change) * 100.0 > self.setup.options.drift_percent:
            self.watch.note(
                "drift",
                f"The light drifts: a frame is {abs(change) * 100:.1f} % "
                f"{'above' if change > 0 else 'below'} the median level.",
            )


# --- Keeping the frames of a session ------------------------------------------------------------


class _Source:
    """A SER file as a source of frames for `make_flat`, with the checks of a long job.

    Each frame that the combination reads first tells the clock (the watchdog of `core` watches
    it) and asks whether to stop. The source also states the sensor temperature and the gain,
    which a SER file cannot carry.
    """

    def __init__(
        self,
        path: Path,
        *,
        temperature_c: float | None,
        gain: int,
        tick: Callable[[], None],
    ) -> None:
        self._inner = SerSource(path)
        self._temperature_c = temperature_c
        self._gain = gain
        self._tick = tick

    @property
    def count(self) -> int:
        return self._inner.count

    @property
    def shape(self) -> tuple[int, int]:
        return self._inner.shape

    @property
    def declared_bits(self) -> int | None:
        return self._inner.declared_bits

    @property
    def temperature_c(self) -> float | None:
        return self._temperature_c

    @property
    def gain(self) -> int | None:
        return self._gain

    def frame(self, index: int) -> Frame2D:
        self._tick()
        return self._inner.frame(index)

    def close(self) -> None:
        self._inner.close()


@dataclass(frozen=True, slots=True)
class _TakenSet:
    """The set that the session just took: its file, and what it measured."""

    ser_path: Path
    exposure_s: float
    level_fraction: float
    temperature_c: float | None
    frames: int
    warnings: tuple[str, ...]


def disk_free_bytes(path: Path) -> int:
    """The free bytes of the partition that holds `path`, which need not exist yet."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def memory_available_bytes() -> int | None:
    """The memory that a program may use without swapping (`MemAvailable` of Linux), or `None`.

    The answer is `None` where the system gives no such number, and then the session does not check.
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


# --- The session --------------------------------------------------------------------------------


def record_flat(
    camera: FlatCamera,
    flats: FlatLibrary,
    darks: DarkLibrary,
    profile: Profile,
    clock: Clock,
    options: FlatSessionOptions,
    *,
    say: Callable[[str], None] = print,
    progress: Callable[[FlatProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    free_bytes: Callable[[Path], int] = disk_free_bytes,
    reserve_bytes: int = 0,
    capture_allowed: Callable[[], bool] | None = None,
    available_memory: Callable[[], int | None] = memory_available_bytes,
) -> FlatSessionResult:
    """Record one set of a flat, and add the flat to the library. See the module documentation.

    The camera is open already, and it stays open. `free_bytes` tells the free space of the
    partition that holds a path, and the session keeps `reserve_bytes` free besides the frames.
    `capture_allowed` is the gate of the storage (false when the free space is low), asked before
    each frame. `available_memory` tells the memory that is free (`None`: it is not known), and the
    session refuses to start when the combination would not fit. Raises `FlatSessionError` with one
    plain sentence when the session cannot go on, `FlatAborted` when `should_stop` answers true,
    and `seeingmon.drivers.CameraError` for a camera fault. The library keeps its flats in every
    case, and the frames of a set that did not finish are deleted.
    """
    try:
        readout = profile.mode(options.mode)
        saturation = profile.saturation(options.mode, options.gain)
    except ProfileError as error:
        raise FlatSessionError(f"The flat settings are not valid: {error}.") from None
    geometry = Geometry(
        shape=(readout.height_px, readout.width_px),
        scale_arcsec_px=profile.plate_scale_arcsec_per_px(readout),
        adc_bits=readout.adc_bits,
    )
    full_scale = float(saturation.native_dn)
    low_us, high_us = profile.limits.exposure_us_range
    model = darks.model(options.mode, options.gain, prior_doubling_c=options.doubling_c)
    if model is None:
        raise FlatSessionError(DARK_FIRST)
    mean_bias = float(np.mean([bias for _, bias in model.bias_points]))

    def bias_of(temperature_c: float | None) -> float:
        return mean_bias if temperature_c is None else float(model.bias_dn(temperature_c))

    setup = _Setup(
        options=options,
        profile=profile,
        geometry=geometry,
        full_scale=full_scale,
        min_exposure_us=max(1, low_us),
        max_exposure_us=max(low_us, min(high_us, round(options.max_exposure_s * 1e6))),
        bias_of=bias_of,
    )
    session = _Session(camera, setup, clock, say, progress, should_stop)
    session.report("setup", 0, 1, "Setting up the camera and the library.")
    flats.session.sweep(clock.utc_ns())
    first = _first_set(flats, options, geometry, clock.utc_ns())
    _check_disk(flats, options, geometry, first is None, free_bytes, reserve_bytes)
    _check_memory(geometry, available_memory)
    session.report("setup", 1, 1, "The camera and the library are ready.")

    choice = session.find_exposure(options.start_exposure_s if first is None else first.exposure_s)
    if choice.note is not None:
        session.watch.note("target", choice.note)
    if first is None:
        flats.session.clear()  # a new session replaces the one that waits for its second set
    ser_path = flats.session.ser_path(options.set_number)
    finished = False
    try:
        levels, temperatures = _take_set(
            session, ser_path, choice, geometry, readout.adc_bits, capture_allowed
        )
        taken = _TakenSet(
            ser_path=ser_path,
            exposure_s=choice.exposure_us / 1e6,
            level_fraction=float(np.median([level.fraction for level in levels])),
            temperature_c=_mean(temperatures),
            frames=len(levels),
            warnings=session.watch.texts(),
        )
        made, entry = _combine(session, flats, darks, first, taken, geometry, full_scale, clock)
        finished = True
    finally:
        if not finished:
            _drop_unfinished(flats, options.set_number, ser_path)
    return FlatSessionResult(
        entry=entry,
        make=made,
        set_number=options.set_number,
        exposure_s=taken.exposure_s,
        level_fraction=taken.level_fraction,
        frames_taken=taken.frames,
        temperature_c=taken.temperature_c,
        warnings=tuple(str(text) for text in entry.report.get("warnings", ())),
    )


def _first_set(
    flats: FlatLibrary, options: FlatSessionOptions, geometry: Geometry, now_ns: int
) -> FlatSession | None:
    """The first set that a second set combines with, or `None` for the first set itself."""
    if options.set_number == 1:
        return None
    first = flats.session.load(now_ns)
    if first is None:
        raise FlatSessionError(NO_FIRST_SET)
    same_camera = (first.mode, first.gain) == (options.mode, options.gain)
    if not same_camera or (first.height_px, first.width_px) != geometry.shape:
        raise FlatSessionError(
            "The first set has other camera settings than this session. Take the first set again."
        )
    return first


def _check_disk(
    flats: FlatLibrary,
    options: FlatSessionOptions,
    geometry: Geometry,
    new_session: bool,
    free_bytes: Callable[[Path], int],
    reserve_bytes: int,
) -> None:
    """Refuse a session whose frames the disk cannot hold, with the numbers in the sentence."""
    height, width = geometry.shape
    needed = options.frames * (height * width * 2 + 8) + 4096
    free = free_bytes(flats.session.directory)
    if new_session:  # the frames of the session that this one replaces go first
        old = flats.session.ser_path(1)
        free += old.stat().st_size if old.is_file() else 0
    if free - needed < reserve_bytes:
        raise FlatSessionError(
            "There is not enough free disk space for the frames: the set needs about "
            f"{_gigabytes(needed)}, and {_gigabytes(max(free - reserve_bytes, 0))} are free "
            "beyond the reserve."
        )


def _check_memory(geometry: Geometry, available_memory: Callable[[], int | None]) -> None:
    """Refuse a session whose combination the free memory cannot hold, before the frames come."""
    free = available_memory()
    if free is None:
        return
    height, width = geometry.shape
    needed = height * width * MEMORY_BYTES_PER_PIXEL
    if free < needed:
        raise FlatSessionError(
            "There is not enough free memory to combine the frames: the combination needs about "
            f"{_gigabytes(needed)}, and {_gigabytes(free)} are free. Stop other programs, or "
            "restart core, and try again."
        )


def _take_set(
    session: _Session,
    ser_path: Path,
    choice: ExposureChoice,
    geometry: Geometry,
    adc_bits: int,
    capture_allowed: Callable[[], bool] | None,
) -> tuple[list[FrameLevel], list[float]]:
    options = session.setup.options
    height, width = geometry.shape
    exposure_s = choice.exposure_us / 1e6
    session.report(
        "capture",
        0,
        options.frames,
        f"Taking {options.frames} frames of {format_exposure(exposure_s)}.",
        exposure_s=exposure_s,
        level=choice.level,
    )
    try:
        ser_path.parent.mkdir(parents=True, exist_ok=True)
        writer = SerWriter(
            ser_path, width=width, height=height, pixel_depth=adc_bits, overwrite=True
        )
    except (OSError, SerError, ValueError) as error:
        raise FlatSessionError(
            f"The frames could not be recorded ({type(error).__name__})."
        ) from None
    try:
        return session.capture(writer, choice.exposure_us, capture_allowed=capture_allowed)
    finally:
        writer.close()


def _combine(
    session: _Session,
    flats: FlatLibrary,
    darks: DarkLibrary,
    first: FlatSession | None,
    taken: _TakenSet,
    geometry: Geometry,
    full_scale: float,
    clock: Clock,
) -> tuple[MakeResult, FlatEntry]:
    """Combine the frames into a flat with the bias of the dark library, and store the flat."""
    options = session.setup.options
    session.report("build", 0, 1, "Combining the frames into a flat.")
    sources: list[_Source] = []
    try:
        try:
            if first is not None:
                sources.append(
                    _Source(
                        flats.session.directory / first.ser,
                        temperature_c=first.temperature_c,
                        gain=first.gain,
                        tick=session.stop_requested,
                    )
                )
            sources.append(
                _Source(
                    taken.ser_path,
                    temperature_c=taken.temperature_c,
                    gain=options.gain,
                    tick=session.stop_requested,
                )
            )
            temperatures = [s.temperature_c for s in sources if s.temperature_c is not None]
            bias = library_bias(
                darks,
                mode=options.mode,
                gain=options.gain,
                temperature_c=_mean(temperatures),
                doubling_c=options.doubling_c,
            )
            if bias is None:
                raise FlatSessionError(DARK_FIRST)
            made = make_flat(
                sources,
                bias=BiasInput(library=bias),
                geometry=geometry,
                options=replace(options.make or MakeOptions(), full_scale=full_scale),
                source_turned=options.set_number == 2,
                clock=clock,
            )
        except FlatError as error:
            raise FlatSessionError(f"The flat could not be made. {_sentence(str(error))}") from None
    finally:
        for source in sources:
            source.close()
    entry = _store(session, flats, made, first, taken, clock)
    session.report(
        "done",
        1,
        1,
        f"Added {entry.version} to the flat library.",
        exposure_s=taken.exposure_s,
    )
    return made, entry


def _store(
    session: _Session,
    flats: FlatLibrary,
    made: MakeResult,
    first: FlatSession | None,
    taken: _TakenSet,
    clock: Clock,
) -> FlatEntry:
    """Add the flat to the library as a pending flat, and keep or end the session."""
    options = session.setup.options
    now_ns = clock.utc_ns()
    earlier = [] if first is None else [first]
    readings = [x.temperature_c for x in earlier] + [taken.temperature_c]
    temperatures = [value for value in readings if value is not None]
    notes = [*(note for x in earlier for note in x.warnings), *taken.warnings]
    info = FlatInfo(
        t_utc_ns=now_ns,
        mode=options.mode,
        gain=options.gain,
        target_fraction=options.target_fraction,
        exposures_s=(*(x.exposure_s for x in earlier), taken.exposure_s),
        level_fractions=(*(x.level_fraction for x in earlier), taken.level_fraction),
        temperature_c=_mean(temperatures),
        warnings=tuple(dict.fromkeys(notes)),
    )
    report = build_report(made, info)
    try:
        preview: bytes | None = render_preview(made.flat)
    except ImportError:  # Pillow is a package of the web extra
        log.warning("the flat has no preview: Pillow is not installed")
        preview = None
    try:
        entry = flats.add(made.flat, report, preview=preview)
    except (OSError, FlatFileError, SkyError) as error:
        raise FlatSessionError(f"The flat could not be stored ({type(error).__name__}).") from None
    if first is None:
        height, width = made.flat.shape
        flats.session.save(
            FlatSession(
                t_utc_ns=now_ns,
                version=entry.version,
                ser=taken.ser_path.name,
                frames=taken.frames,
                exposure_s=taken.exposure_s,
                temperature_c=taken.temperature_c,
                level_fraction=taken.level_fraction,
                mode=options.mode,
                gain=options.gain,
                width_px=int(width),
                height_px=int(height),
                warnings=taken.warnings,
            )
        )
        return entry
    try:  # the flat of the first set alone gives way to the combined flat
        flats.delete(first.version)
    except FlatLibraryError as error:
        log.info("the flat of the first set stays: %s", error.message)
    flats.session.clear()
    return entry


def _drop_unfinished(flats: FlatLibrary, set_number: int, ser_path: Path) -> None:
    """Delete the frames of a set that did not finish. The first set of a live session stays."""
    try:
        ser_path.unlink(missing_ok=True)
    except OSError:
        log.warning("the frames of an unfinished flat set could not be deleted")
    if set_number == 1:
        flats.session.clear()  # a first set that did not finish leaves no session


def _sentence(text: str) -> str:
    """The text as one sentence: a capital letter first, and a period last."""
    text = text.strip()
    if not text:
        return text
    return (text[0].upper() + text[1:]).rstrip(".") + "."


__all__ = [
    "DARK_FIRST",
    "NO_FIRST_SET",
    "ExposureChoice",
    "FlatAborted",
    "FlatCamera",
    "FlatPhase",
    "FlatProgress",
    "FlatSessionError",
    "FlatSessionOptions",
    "FlatSessionResult",
    "FrameLevel",
    "disk_free_bytes",
    "format_exposure",
    "measure_frame",
    "memory_available_bytes",
    "record_flat",
]
