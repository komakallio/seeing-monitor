"""The calibration of the preview images: the hot pixels, the dark level, and the flat field.

A raw preview shows the optics more than the sky. The vignetting darkens the corners, and the dust
on the sensor window leaves dark rings. The survey analysis divides both out before it measures the
sky (`seeingmon.survey.quality`), and a `PreviewCalibrator` does the same for the JPEG of a survey
frame and for the live view of the alignment helper. It works on the shrunk image
(`seeingmon.services.core.alignment.preview`), so it adds a small part of the time that the shrink
takes. For a frame of the survey readout mode, it takes three steps:

1. **Hot pixels.** The dark library lists the hot pixels of each dark set and their excess in
   counts. The calibrator takes the set nearest to the sensor temperature of the frame, scales the
   excess to the exposure and the temperature of the frame, and replaces each pixel that still
   stands out by the median of its eight neighbors. The replacement happens in the shrunk image: a
   block loses the difference between the hot pixel and its replacement, divided by the number of
   pixels in the block. That gives the image that a replacement before the shrink would give, and
   it needs no copy of the frame. A pixel counts as hot when it moves its block by at least
   `HOT_MIN_BLOCK_DN` counts, so a short exposure replaces only the hottest pixels.
2. **Dark level.** The dark model of the library (`seeingmon.survey.dark`) gives the median level
   of a dark frame, the bias plus the dark current, for the sensor temperature and the exposure of
   the frame. The calibrator subtracts it in the units of the frame data.
3. **Flat.** The calibrator multiplies the image by the inverse of the flat field
   (`seeingmon.survey.sky`), shrunk with the same blocks. A flat value under `FLAT_FLOOR` counts as
   `FLAT_FLOOR`, so that a dead corner cannot blow up the noise.

The dark level has to come off before the flat divides, because the bias is no light of the sky and
the vignetting does not touch it. A division without that step would print the inverse of the flat
into the sky, and a short exposure (the live view takes 0.5 s) holds more bias than sky. So a flat
works only together with a dark model. These cases pass a frame unchanged: a readout mode other than
the survey mode, a frame without a sensor temperature, a library with no set for the mode and the
gain, and a flat that does not cover the frame. With a dark model and no flat (a unit flat), only
the hot pixels change. A flat that fails to load, and anything else that goes wrong, logs one
warning (and one more after an hour if the failure goes on) and leaves the preview as it was: a
preview never fails because of its calibration. A configured flat that waits for a dark set logs
one line at the level `INFO`, so that a person who sees a raw preview finds the reason.

**Which flat.** The flat comes from a provider that the calibrator calls for each frame.
`LibraryFlatProvider` follows the rule of the survey (`seeingmon.survey.flat_library`): the active
flat of the flat library, which you choose on the Flat page, and without one `[survey] flat_file`,
and without that a unit flat. A flat that you activate reaches the next preview with no restart.
`FileFlatProvider` reads `[survey] flat_file` when the first frame arrives, and again when the
file changes, so the previews pick up a new flat file too (the survey analysis waits for a
restart). `core` uses the first, and a calibrator without a provider uses the second.

**What the calibrator caches.** The calibrator keeps the inverse of the shrunk flat for each flat,
shrink factor, and frame window, the dark model and the sets of the library for each readout mode
and gain (until the folder of the library changes), and the hot pixels of each set. A call that
finds everything in the caches costs a stat of the pointer of the flat library (and a stat of the
flat file when the library has no active flat), a listing of the dark library folder, and the
arithmetic on the shrunk image.

**Threads.** The frame writer and the alignment helper call the calibrator from their own threads.
One lock covers the caches and the work that fills them, so two threads never shrink the flat
twice. The arrays in the caches never change after they are made.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.frames import Frame, FrameData
from seeingmon.profile import Profile
from seeingmon.services.core.alignment.preview import PreviewCalibration
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.dark import DARKS_DIRNAME, DarkLibrary, DarkModel, DarkSet, fit_dark_model
from seeingmon.survey.flat_library import library_flat
from seeingmon.survey.sky import FlatModel, SkyError, UnitFlat, load_flat

_log = logging.getLogger(__name__)

FlatProvider = Callable[[], FlatModel]
FloatImage = npt.NDArray[np.float32]
IndexArray = npt.NDArray[np.intp]
_Stamp = tuple[tuple[str, int, int], ...]  # the name, modification time, and size of each file

FLAT_FLOOR = 0.1  # the lowest flat value that divides: a dead corner cannot blow up the noise
HOT_MIN_BLOCK_DN = 1.0  # a hot pixel that moves its block by less than this many counts stays
CACHE_ENTRIES = 4  # each cache keeps this many entries, and drops the oldest first
WARN_INTERVAL_NS = round(3600.0 * NS_PER_S)  # a failure that goes on warns again after an hour
MAX_REPORTS = 32  # the calibrator remembers this many different log lines

# The eight neighbors of a pixel as steps in rows and columns.
_NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

_K = TypeVar("_K")
_V = TypeVar("_V")


class FileFlatProvider:
    """The flat of `[survey] flat_file`: read at the first call, and again when the file changes.

    A call looks at the size and the modification time of the file, which costs one `os.stat`, and
    it reads the file only when either one changed. An empty path gives a unit flat. A file that
    cannot be read raises `SkyError`, and the provider does not read it again until it changes.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = os.fspath(path)
        self._unit = UnitFlat()
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self._flat: FlatModel | None = None
        self._error = ""

    def __call__(self) -> FlatModel:
        if not self._path:
            return self._unit
        try:
            info = os.stat(self._path)
        except OSError as error:
            raise SkyError(f"cannot read the flat file: {error}") from error
        stamp = (info.st_mtime_ns, info.st_size)
        with self._lock:
            if stamp != self._stamp:
                try:
                    self._flat, self._error = load_flat(self._path), ""
                except SkyError as error:
                    self._flat, self._error = None, str(error)
                self._stamp = stamp  # after the read: an unexpected error leaves the file unread
            if self._flat is None:
                raise SkyError(self._error)
            return self._flat


class LibraryFlatProvider:
    """The flat of the previews of `core`: the library's active flat, or else the flat file.

    The order is the one of `seeingmon.survey.flat_library.active_flat`: the active flat of the
    flat library (`calibration_dir/flats/`), then `[survey] flat_file`, then a unit flat. A call
    stats the pointer of the library, which costs one `os.stat`, and it reads the library's flat
    only when the pointer changed. Without an active flat in the library, the call goes on to a
    `FileFlatProvider`, which stats the flat file and reads it again when it changes. A flat file
    that cannot be read raises `SkyError`, as the file provider does.
    """

    def __init__(self, config: SurveyConfig) -> None:
        self._config = config
        self._file = FileFlatProvider(config.flat_file)

    def __call__(self) -> FlatModel:
        flat = library_flat(self._config)
        return flat if flat is not None else self._file()

    def configured(self) -> bool:
        """Whether the library has an active flat or the settings name a flat file."""
        return bool(self._config.flat_file) or library_flat(self._config) is not None


@dataclass(frozen=True, slots=True, eq=False)
class _HotList:
    """The hot pixels of one dark set inside a frame window, hottest first.

    `rows` and `columns` index the window. `neg_excess` holds the excess in counts at the exposure
    and the temperature of the set, negated, so that it ascends and `searchsorted` finds the pixels
    that exceed a limit.
    """

    rows: IndexArray
    columns: IndexArray
    neg_excess: npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True, eq=False)
class _Entry:
    """What the calibrator keeps of the library for one readout mode and gain."""

    sets: tuple[DarkSet, ...]
    model: DarkModel


@dataclass(frozen=True, slots=True, eq=False)
class _FrameCalibration:
    """What `PreviewCalibrator.for_frame` found for one frame, and the step that applies it.

    `level` is the dark level in the units of the frame data, `flat` the window of the flat that
    the frame covers (`None` for a unit flat), and `hot_scale` the expected excess of a hot pixel
    in this frame for each count of its listed excess. Calling the object is the step of
    `make_preview`.
    """

    owner: PreviewCalibrator
    shape: tuple[int, int]
    origin: tuple[int, int]
    level: float
    flat: FloatImage | None
    flat_version: str
    hot: _HotList | None
    hot_scale: float

    def __call__(self, data: FrameData, image: FloatImage, factor: int) -> FloatImage:
        return self.owner._apply(self, data, image, factor)


class PreviewCalibrator:
    """Calibrates the previews of the survey readout mode. See the module text.

    `flat_provider` returns the flat to use for each frame. It defaults to `FileFlatProvider` on
    `[survey] flat_file`, and `core` passes a `LibraryFlatProvider`. `flat_configured` tells
    whether a flat is configured, for the note that says why a flat waits for a dark set. It
    defaults to true when a provider is given or `flat_file` is set, and a provider whose flat
    comes and goes (the library) passes its own answer. `library` is the dark library, and it
    defaults to the one in `[survey] calibration_dir`. Without a library, no frame changes.
    `clock` times the repeats of a warning.

    `for_frame` and the step that it returns never raise.
    """

    def __init__(
        self,
        config: SurveyConfig,
        profile: Profile,
        *,
        flat_provider: FlatProvider | None = None,
        flat_configured: Callable[[], bool] | None = None,
        library: DarkLibrary | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._survey_mode = profile.survey_mode.mode
        self._doubling_c = config.dark.doubling_c
        self._flat_provider: FlatProvider = flat_provider or FileFlatProvider(config.flat_file)
        if library is None and config.calibration_dir:
            library = DarkLibrary(Path(config.calibration_dir) / DARKS_DIRNAME)
        self._library = library
        configured = flat_provider is not None or bool(config.flat_file)
        self._flat_configured: Callable[[], bool] = flat_configured or (lambda: configured)
        self._clock: Clock = clock or SystemClock()
        self._lock = threading.RLock()
        self._entries: dict[tuple[str, int], tuple[_Stamp, _Entry | None]] = {}
        self._hot: dict[tuple[str, int, int, int, int], _HotList] = {}
        self._inverse: dict[tuple[str, int, tuple[int, int], tuple[int, int]], FloatImage] = {}
        self._reported: dict[str, int] = {}

    def for_frame(self, frame: Frame) -> PreviewCalibration | None:
        """The step for `make_preview` that calibrates a frame, or `None` to leave it as it is."""
        if frame.mode != self._survey_mode:
            return None
        try:
            with self._lock:
                return self._resolve(frame)
        except Exception as error:
            self._warn("a frame stays uncalibrated", error)
            return None

    # --- What a frame needs ----------------------------------------------------------------

    def _resolve(self, frame: Frame) -> _FrameCalibration | None:
        """Find the dark level, the flat, and the hot pixels of a frame. Call it with the lock."""
        temperature = frame.temperature_c
        if temperature is None or self._library is None:
            return None
        entry = self._entry(frame.mode, frame.gain)
        if entry is None:
            if self._flat_configured():
                self._report(
                    logging.INFO,
                    f"the dark library has no set for the readout mode {frame.mode} at gain "
                    f"{frame.gain}, and the flat needs the dark level",
                )
            return None
        flat = self._flat_provider()
        roi = frame.roi
        window = flat.image(frame.shape, (roi.x, roi.y))
        exposure_s = max(frame.exposure_us, 0) / 1e6
        nearest = min(
            entry.sets, key=lambda item: (abs(item.temperature_c - temperature), -item.t_utc_ns)
        )
        hot = self._hot_list(nearest, frame)
        if window is None and hot is None:
            return None
        scale = 2.0 ** (frame.pixel_format.value - frame.adc_bits)  # the native count in the data
        return _FrameCalibration(
            owner=self,
            shape=frame.shape,
            origin=(roi.x, roi.y),
            level=entry.model.level_dn(temperature, exposure_s) * scale,
            flat=None if window is None else np.asarray(window, dtype=np.float32),
            flat_version=flat.version,
            hot=hot,
            hot_scale=_hot_scale(nearest, entry.model, temperature, exposure_s),
        )

    def _entry(self, mode: str, gain: int) -> _Entry | None:
        """The sets and the dark model of a readout mode and gain, or `None` for no set."""
        library = self._library
        assert library is not None
        stamp = _folder_stamp(library.directory)
        cached = self._entries.get((mode, gain))
        if cached is not None and cached[0] == stamp:
            return cached[1]
        sets = library.sets_for(mode, gain)
        entry: _Entry | None = None
        if sets:
            entry = _Entry(sets, fit_dark_model(sets, prior_doubling_c=self._doubling_c))
        self._entries[(mode, gain)] = (stamp, entry)
        return entry

    def _hot_list(self, dark_set: DarkSet, frame: Frame) -> _HotList | None:
        """The hot pixels of a set inside the window of a frame, or `None` when it has none."""
        roi = frame.roi
        key = (dark_set.name, roi.x, roi.y, roi.width, roi.height)
        found = self._hot.get(key)
        if found is None:
            assert self._library is not None
            columns, rows, excess = self._library.hot_pixels(dark_set)
            x = columns.astype(np.intp) - roi.x
            y = rows.astype(np.intp) - roi.y
            # A pixel on the edge of the window has no full ring of neighbors, so it stays.
            inside = (x >= 1) & (y >= 1) & (x < roi.width - 1) & (y < roi.height - 1)
            order = np.argsort(-excess[inside], kind="stable")
            found = _HotList(
                y[inside][order], x[inside][order], np.asarray(-excess[inside][order], np.float32)
            )
            _remember(self._hot, key, found)
        return found if found.rows.size else None

    # --- The step --------------------------------------------------------------------------

    def _apply(
        self, step: _FrameCalibration, data: FrameData, image: FloatImage, factor: int
    ) -> FloatImage:
        """Calibrate the shrunk image of a frame in place. The image is unchanged on a failure."""
        try:
            if data.shape != step.shape or image.shape != (
                data.shape[0] // factor,
                data.shape[1] // factor,
            ):
                raise ValueError("the image does not belong to the frame of the calibration")
            inverse = None if step.flat is None else self._inverse_flat(step, factor)
            hot = None if step.hot is None else _hot_correction(step, data, factor, image.shape)
        except Exception as error:
            self._warn("a preview stays uncalibrated", error)
            return image
        # Everything that can fail is done, so the image changes completely or not at all.
        if hot is not None:
            block_rows, block_columns, delta = hot
            np.add.at(image, (block_rows, block_columns), delta)
        if inverse is not None:
            image -= np.float32(step.level)
            image *= inverse
        return image

    def _inverse_flat(self, step: _FrameCalibration, factor: int) -> FloatImage:
        """The inverse of the flat shrunk by `factor`, from the cache or made now."""
        key = (step.flat_version, factor, step.origin, step.shape)
        with self._lock:
            found = self._inverse.get(key)
            if found is None:
                assert step.flat is not None
                found = _inverse_of_blocks(step.flat, factor)
                _remember(self._inverse, key, found)
            return found

    # --- Failures --------------------------------------------------------------------------

    def _warn(self, what: str, error: Exception) -> None:
        """Log a failure once, and once more for each hour that it goes on."""
        self._report(logging.WARNING, f"{what}: {type(error).__name__}: {error}", error)

    def _report(self, level: int, text: str, error: Exception | None = None) -> None:
        """Log a line, and the same line again only after an hour. `error` adds its traceback."""
        now = self._clock.monotonic_ns()
        with self._lock:
            last = self._reported.get(text)
            if last is not None and now - last < WARN_INTERVAL_NS:
                return
            self._reported[text] = now
            while len(self._reported) > MAX_REPORTS:
                del self._reported[next(iter(self._reported))]
        known = error is None or isinstance(error, SkyError | OSError)
        _log.log(
            level, "the previews are not calibrated, %s", text, exc_info=None if known else error
        )


# --- Arithmetic ----------------------------------------------------------------------------


def _remember(cache: dict[_K, _V], key: _K, value: _V) -> None:
    """Put an entry in a cache, and drop the oldest ones beyond `CACHE_ENTRIES`."""
    cache[key] = value
    while len(cache) > CACHE_ENTRIES:
        del cache[next(iter(cache))]


def _folder_stamp(directory: Path) -> _Stamp:
    """The names, modification times, and sizes of the dark sets in a folder, in name order."""
    found: list[tuple[str, int, int]] = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name.startswith("dark-") and entry.name.endswith(".fits"):
                    info = entry.stat()
                    found.append((entry.name, info.st_mtime_ns, info.st_size))
    except OSError:
        return ()
    return tuple(sorted(found))


def _hot_scale(
    dark_set: DarkSet, model: DarkModel, temperature_c: float, exposure_s: float
) -> float:
    """The excess of a hot pixel in a frame for each count of its excess in a set.

    The excess follows the dark current: it grows with the exposure, and it doubles with each
    `doubling_c` degrees of the sensor temperature.
    """
    if dark_set.exposure_s <= 0.0:
        return 0.0
    doublings = (temperature_c - dark_set.temperature_c) / model.doubling_c
    return exposure_s / dark_set.exposure_s * math.pow(2.0, doublings)


def _inverse_of_blocks(window: FloatImage, factor: int) -> FloatImage:
    """The inverse of a flat shrunk to blocks of `factor` by `factor`, as `block_mean` shrinks."""
    rows, columns = window.shape[0] // factor, window.shape[1] // factor
    blocks = window[: rows * factor, : columns * factor].reshape(rows, factor, columns, factor)
    mean = blocks.mean(axis=(1, 3), dtype=np.float32)
    inverse = np.asarray(np.float32(1.0) / np.maximum(mean, np.float32(FLAT_FLOOR)), np.float32)
    inverse.flags.writeable = False
    return inverse


def _hot_correction(
    step: _FrameCalibration, data: FrameData, factor: int, shape: tuple[int, int]
) -> tuple[IndexArray, IndexArray, FloatImage] | None:
    """The change of the shrunk image that replaces the hot pixels, or `None` for no change.

    Returns the block rows, the block columns, and the amount to add to each block (a block that
    holds several hot pixels appears several times). The amount is the replacement minus the
    pixel, divided by the number of pixels in the block.
    """
    hot = step.hot
    assert hot is not None
    if step.hot_scale <= 0.0:
        return None
    least_excess = HOT_MIN_BLOCK_DN * factor * factor / step.hot_scale
    count = int(np.searchsorted(hot.neg_excess, -least_excess, side="right"))
    rows, columns = hot.rows[:count], hot.columns[:count]
    inside = (rows < shape[0] * factor) & (columns < shape[1] * factor)
    rows, columns = rows[inside], columns[inside]
    if rows.size == 0:
        return None
    ring = np.empty((rows.size, len(_NEIGHBORS)), dtype=data.dtype)
    for index, (row_step, column_step) in enumerate(_NEIGHBORS):
        ring[:, index] = data[rows + row_step, columns + column_step]
    ring.sort(axis=1)
    replacement = (ring[:, 3].astype(np.float32) + ring[:, 4]) * np.float32(0.5)
    delta = (replacement - data[rows, columns]) / np.float32(factor * factor)
    return rows // factor, columns // factor, np.asarray(delta, dtype=np.float32)
