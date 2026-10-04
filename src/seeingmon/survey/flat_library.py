"""The flat library: the flats that the web UI makes, and the flat that the survey divides by.

A flat session (`seeingmon.survey.flat_session`) combines frames of a light source into a flat and
adds it to the library. The library is the folder `flats/` of the calibration folder
(`[survey] calibration_dir`, or `calibration/` in the data directory). A flat has three files that
share its name, `flat-<8 hex digits>`, which is the version that the `provenance` of every sky
record carries (the CRC of the pixels, as `seeingmon.survey.sky.ArrayFlat` computes it):

- `<version>.npy` holds the flat (`write_flat`): a `float32` image of the whole sensor with a
  median of 1,
- `<version>.json` holds the report: the vignetting at a few radii, the tilt, the dust shadows, the
  agreement of the sets, the frames used and dropped, the exposure, the gain, the sensor
  temperature, the time, and the bias that came off, and
- `<version>.jpg` shows the flat stretched to plus and minus 10% around 1.

`current.json` names the active flat. A new flat is *pending*: it changes nothing for the survey
until you activate it (`FlatLibrary.activate`), and then it is *approved*. The library keeps the
newest `KEEP_FLATS` flats and never deletes the active one.

**Which flat the survey uses.** The active flat of the library wins. Without one, the survey uses
`[survey] flat_file`, and without that a unit flat (no correction). `ActiveFlat` applies the rule,
and `active_flat(config)` is the same rule as one cheap call. The call looks at the pointer with
one `stat`, and it reads the flat again only when the pointer changed, so the survey worker (and
`core`, for the previews) pick up an activation without a restart. A flat file that cannot be read
leaves the flat that the survey already uses in place.

**The session folder.** A second set (with the source turned by 180 degrees) combines with the
first set of the same session, so the folder `session/` keeps the frames of the first set (a SER
file, at most about 1.5 GB) and a small JSON file for 24 hours, or until you activate or discard
the pending flat. `FlatSessionStore` manages the folder.

**No private text.** An error message names no path. A report holds no host, path, or serial
number.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import os
import re
import shutil
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, utc_ns_to_iso
from seeingmon.store.layout import write_atomic
from seeingmon.survey.flat_files import FlatFileError, write_flat
from seeingmon.survey.sky import ArrayFlat, FlatModel, SkyError, UnitFlat, load_flat

if TYPE_CHECKING:
    from seeingmon.survey.config import SurveyConfig
    from seeingmon.survey.flat_make import MakeResult

log = logging.getLogger("seeingmon.survey")

FLATS_DIRNAME = "flats"
POINTER_NAME = "current.json"
SESSION_DIRNAME = "session"
SESSION_FILENAME = "session.json"
REPORT_FORMAT = 1
KEEP_FLATS = 10
SESSION_TTL_S = 24 * 3600
PENDING = "pending"
APPROVED = "approved"
PREVIEW_RANGE = 0.10  # the preview shows 1 minus this to 1 plus this
PREVIEW_MAX_WIDTH_PX = 1024
PREVIEW_QUALITY = 85
MAX_LISTED_SHADOWS = 12
VERSION_PATTERN = re.compile(r"^flat-[0-9a-f]{8}$")
UNKNOWN_FLAT = "There is no flat with that name."
FLAT_IN_USE = "The flat is in use. Activate another flat first."

FloatArray = npt.NDArray[np.float32]


class FlatLibraryError(Exception):
    """An operation on the library failed. `reason` is a short code, and `message` a sentence.

    The codes are `unknown` (no such flat), `active` (the flat is in use), and `invalid` (the flat
    cannot be used). The message names no path.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


def is_version(text: object) -> bool:
    """Whether `text` has the form of a flat version, so that a file name may follow from it."""
    return isinstance(text, str) and VERSION_PATTERN.fullmatch(text) is not None


# --- The report -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FlatInfo:
    """What a report holds besides the result of `make_flat`: the session that took the frames.

    `exposures_s` and `level_fractions` have one value for each set. A level is the median of the
    middle of the frame above the bias, as a fraction of the full scale. `temperature_c` is the
    mean sensor temperature of the frames, or `None` when the camera gave none. `warnings` are the
    notes of the session itself (a drifting light, a saturating one), which the report keeps next
    to the warnings of `make_flat`.
    """

    t_utc_ns: int
    mode: str
    gain: int
    target_fraction: float
    exposures_s: tuple[float, ...]
    level_fractions: tuple[float, ...]
    temperature_c: float | None
    warnings: tuple[str, ...] = ()


def _number(value: float | None, digits: int = 3) -> float | None:
    """A number rounded for the JSON file, or `None` for a missing or non-finite value."""
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _tilt(tilt: Any) -> dict[str, float | None]:
    return {
        "width_percent": _number(tilt.width_percent, 3),
        "height_percent": _number(tilt.height_percent, 3),
    }


def _shadow(shadow: Any) -> dict[str, float | int]:
    return {
        "x_px": int(shadow.x_px),
        "y_px": int(shadow.y_px),
        "depth_percent": round(100.0 * float(shadow.depth), 2),
        "width_px": round(float(shadow.width_px), 1),
    }


def ui_warnings(result: MakeResult) -> list[str]:
    """The warnings of `make_flat` in words for the web UI.

    The command line tells you to pass `--frames` and `--source-turned`. The page has its own
    button for a second set, so the sentence about the tilt of one set gets words that fit it.
    """
    out: list[str] = []
    for warning in result.warnings:
        if warning.startswith("The tilt may include the gradient of your light source"):
            warning = (
                "The tilt may include the gradient of your light source, up to about 1% for a "
                "phone screen. A second set with the source turned by 180 degrees separates the "
                "two."
            )
        out.append(warning)
    return out


def build_report(result: MakeResult, info: FlatInfo) -> dict[str, Any]:
    """The report of a flat as a JSON-able dictionary. `FlatLibrary.add` completes and stores it."""
    summary = result.summary
    height, width = result.flat.shape
    profile = [
        {
            "radius_deg": round(point.radius_deg, 3),
            "change_percent": _number(point.change_percent, 3),
            "corner": bool(point.corner),
        }
        for point in summary.profile
    ]
    corner = next((point for point in reversed(summary.profile) if point.corner), None)
    corner_percent = None if corner is None else _number(corner.change_percent, 2)
    shadows = list(summary.shadows)
    sets: list[dict[str, Any]] = []
    for number, report in enumerate(result.sets):
        sets.append(
            {
                "number": number + 1,
                "exposure_s": _number(
                    info.exposures_s[number] if number < len(info.exposures_s) else None, 6
                ),
                "level_fraction": _number(
                    info.level_fractions[number] if number < len(info.level_fractions) else None,
                    4,
                ),
                "frames": int(report.frames),
                "used": int(report.used),
                "dropped": {str(key): int(count) for key, count in report.dropped.items()},
                "saturated_percent": _number(100.0 * report.saturated_fraction, 4),
                "noise_percent": _number(100.0 * report.noise_mean, 3),
                "tilt": _tilt(report.tilt),
            }
        )
    agreement = None
    if result.agreements:
        first = result.agreements[0]
        agreement = {
            "smooth_rms_percent": _number(first.smooth_rms, 3),
            "fine_rms_percent": _number(first.fine_rms, 3),
            "expected_fine_rms_percent": _number(first.expected_fine_rms, 3),
            "plane": _tilt(first.plane),
        }
    split = None
    if result.split is not None:
        split = {"optics": _tilt(result.split.stays), "source": _tilt(result.split.turns)}
    return {
        "schema": REPORT_FORMAT,
        "t_utc_ns": info.t_utc_ns,
        "t_utc": utc_ns_to_iso(info.t_utc_ns, digits=0),
        "mode": info.mode,
        "gain": info.gain,
        "width_px": int(width),
        "height_px": int(height),
        "sensor_temperature_c": _number(info.temperature_c, 2),
        "target_fraction": _number(info.target_fraction, 3),
        "exposure_s": sets[0]["exposure_s"] if sets else None,
        "sets": sets,
        "second_set": len(result.sets) > 1,
        "source_turned": bool(result.source_turned),
        "bias": {"source": result.bias.source, "note": result.bias.note},
        "noise_percent": _number(100.0 * result.noise_flat, 3),
        "vignetting": {"points": profile, "corner_percent": corner_percent},
        "tilt": _tilt(summary.tilt),
        "split": split,
        "shadows": {
            "count": len(shadows),
            "min_depth_percent": round(100.0 * float(summary.shadow_depth), 2),
            "items": [_shadow(shadow) for shadow in shadows[:MAX_LISTED_SHADOWS]],
        },
        "edge_artifacts": len(summary.edge_artifacts),
        "agreement": agreement,
        "warnings": [*info.warnings, *ui_warnings(result)],
    }


def render_preview(flat: FloatArray, *, max_width_px: int = PREVIEW_MAX_WIDTH_PX) -> bytes:
    """A JPEG of a flat, stretched to plus and minus 10% around 1, at most `max_width_px` wide.

    Pillow loads when the function runs. A block mean shrinks the image, so the dust shadows stay.
    """
    from PIL import Image

    height, width = flat.shape
    factor = max(1, -(-width // max_width_px))
    rows, columns = height // factor, width // factor
    if rows < 1 or columns < 1:
        raise ValueError("the flat is too small to show")
    blocks = flat[: rows * factor, : columns * factor].reshape(rows, factor, columns, factor)
    binned = np.asarray(blocks.mean(axis=(1, 3), dtype=np.float32), dtype=np.float32)
    scaled = np.clip(
        (binned - np.float32(1.0 - PREVIEW_RANGE)) / np.float32(2 * PREVIEW_RANGE), 0, 1
    )
    pixels = np.asarray(scaled * np.float32(255.0) + np.float32(0.5), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=PREVIEW_QUALITY)
    return buffer.getvalue()


# --- The entries ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FlatEntry:
    """One flat of the library: its version, its time, its state, and its whole report."""

    version: str
    t_utc_ns: int
    state: str
    report: Mapping[str, Any]

    @property
    def pending(self) -> bool:
        """Whether the flat waits for you to activate or discard it."""
        return self.state == PENDING


def _entry_from(version: str, data: object) -> FlatEntry | None:
    """The entry that a parsed report describes, or `None` when it is not a report of a flat."""
    if not isinstance(data, dict) or data.get("schema") != REPORT_FORMAT:
        return None
    t_utc_ns = data.get("t_utc_ns")
    state = data.get("state")
    if data.get("version") != version or isinstance(t_utc_ns, bool):
        return None
    if not isinstance(t_utc_ns, int) or state not in (PENDING, APPROVED):
        return None
    return FlatEntry(version=version, t_utc_ns=t_utc_ns, state=str(state), report=data)


# --- The session folder -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FlatSession:
    """The first set of a session, kept for a second set that the owner may take.

    `version` is the pending flat that the first set made. `ser` is the name of the file with the
    frames of the set, in the session folder.
    """

    t_utc_ns: int
    version: str
    ser: str
    frames: int
    exposure_s: float
    temperature_c: float | None
    level_fraction: float
    mode: str
    gain: int
    width_px: int
    height_px: int
    warnings: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "t_utc_ns": self.t_utc_ns,
            "version": self.version,
            "ser": self.ser,
            "frames": self.frames,
            "exposure_s": self.exposure_s,
            "temperature_c": self.temperature_c,
            "level_fraction": self.level_fraction,
            "mode": self.mode,
            "gain": self.gain,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_json(cls, data: object) -> FlatSession | None:
        """Read a session from its file, or `None` when the content is not a session."""
        if not isinstance(data, dict):
            return None
        try:
            temperature = data["temperature_c"]
            session = cls(
                t_utc_ns=int(data["t_utc_ns"]),
                version=str(data["version"]),
                ser=str(data["ser"]),
                frames=int(data["frames"]),
                exposure_s=float(data["exposure_s"]),
                temperature_c=None if temperature is None else float(temperature),
                level_fraction=float(data["level_fraction"]),
                mode=str(data["mode"]),
                gain=int(data["gain"]),
                width_px=int(data["width_px"]),
                height_px=int(data["height_px"]),
                warnings=tuple(str(text) for text in data.get("warnings", ())),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not is_version(session.version) or Path(session.ser).name != session.ser:
            return None
        return session

    def expires_ns(self) -> int:
        """When the first set expires, in UTC nanoseconds."""
        return self.t_utc_ns + SESSION_TTL_S * NS_PER_S


class FlatSessionStore:
    """The folder of the session in progress: the SER files of the sets and one JSON file."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._capturing = threading.Lock()

    @property
    def directory(self) -> Path:
        return self._directory

    @contextlib.contextmanager
    def capturing(self) -> Iterator[None]:
        """Hold the folder while a session writes into it, so that `sweep_idle` leaves it alone."""
        with self._capturing:
            yield

    def ser_path(self, number: int) -> Path:
        """Where the frames of set `number` (1 or 2) go."""
        if number not in (1, 2):
            raise ValueError("a session has the sets 1 and 2")
        return self._directory / f"set{number}.ser"

    def load(self, now_ns: int) -> FlatSession | None:
        """The first set of the session, or `None` when there is none, or when it expired.

        The call reads and never deletes, so any thread may make it.
        """
        try:
            data = json.loads((self._directory / SESSION_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        session = FlatSession.from_json(data)
        if session is None or now_ns > session.expires_ns():
            return None
        if not (self._directory / session.ser).is_file():
            return None
        return session

    def save(self, session: FlatSession) -> None:
        write_atomic(
            self._directory / SESSION_FILENAME,
            (json.dumps(session.to_json(), indent=2) + "\n").encode("utf-8"),
        )

    def clear(self) -> None:
        """Delete the session: the frames and the JSON file. A missing folder is fine."""
        shutil.rmtree(self._directory, ignore_errors=True)

    def sweep(self, now_ns: int) -> bool:
        """Delete what no session uses: an expired or unreadable session, and a lone second set.

        Call it only while no capture runs. Returns whether it deleted anything.
        """
        if not self._directory.is_dir():
            return False
        if self.load(now_ns) is None:
            self.clear()
            return True
        second = self.ser_path(2)
        if second.exists():  # a capture that a crash cut short
            with contextlib.suppress(OSError):
                second.unlink()
            return True
        return False

    def sweep_idle(self, now_ns: int) -> bool:
        """`sweep` for a thread that is not the session: it does nothing while a session captures.

        A first set that nobody uses stays on disk until the next start or the next session, so
        `core` calls this from time to time to give the space back 24 hours after the first set.
        """
        if not self._capturing.acquire(blocking=False):
            return False
        try:
            return self.sweep(now_ns)
        finally:
            self._capturing.release()


# --- The library --------------------------------------------------------------------------------


class FlatLibrary:
    """The folder of flats. It reads the reports when you ask, so it holds no stale state.

    Opening a library creates nothing: the folder appears when the first flat arrives. Any thread
    may call any method. The writes take a lock and go through `write_atomic`, so a reader (the
    survey worker is another process) sees the old file or the whole new one.
    """

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)
        self._lock = threading.RLock()
        self._skipped: set[str] = set()  # the reports that the log named already
        self.session = FlatSessionStore(self._directory / SESSION_DIRNAME)

    @classmethod
    def from_calibration(cls, calibration_dir: Path | str) -> FlatLibrary:
        """The library in the `flats/` folder of a calibration folder."""
        return cls(Path(calibration_dir) / FLATS_DIRNAME)

    @property
    def directory(self) -> Path:
        return self._directory

    def _path(self, version: str, suffix: str) -> Path:
        if not is_version(version):
            raise FlatLibraryError("unknown", UNKNOWN_FLAT)
        return self._directory / f"{version}{suffix}"

    # --- Reading ------------------------------------------------------------------------------

    def entries(self) -> list[FlatEntry]:
        """Every flat of the folder, newest first. A report that cannot be read is skipped."""
        found: list[FlatEntry] = []
        if not self._directory.is_dir():
            return found
        for path in self._directory.glob("flat-*.json"):
            version = path.stem
            if not is_version(version):
                continue
            try:
                entry = _entry_from(version, json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                entry = None
            if entry is None:
                if version not in self._skipped:
                    self._skipped.add(version)
                    log.warning("the flat library skips the report of %s", version)
                continue
            found.append(entry)
        found.sort(key=lambda item: (item.t_utc_ns, item.version), reverse=True)
        return found

    def get(self, version: str) -> FlatEntry | None:
        """The flat with this version, or `None`."""
        if not is_version(version):
            return None
        try:
            data = json.loads(self._path(version, ".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return _entry_from(version, data)

    def pointer_stamp(self) -> tuple[int, int, int] | None:
        """What identifies the file `current.json` now: its time, its size, and its identity.

        One `stat`. The value changes when the pointer moves, because the pointer is replaced as
        a whole. It is `None` while no flat is active.
        """
        try:
            status = os.stat(self._directory / POINTER_NAME)
        except OSError:
            return None
        return (status.st_mtime_ns, status.st_size, status.st_ino)

    def active_version(self) -> str | None:
        """The version that `current.json` names, or `None` without an active flat."""
        try:
            data = json.loads((self._directory / POINTER_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        version = data.get("version") if isinstance(data, dict) else None
        return version if is_version(version) else None

    def load(self, version: str) -> ArrayFlat:
        """The flat model of a version. Raises `FlatLibraryError` for a missing or bad file."""
        try:
            return ArrayFlat(np.load(self._path(version, ".npy"), allow_pickle=False))
        except FlatLibraryError:
            raise
        except FileNotFoundError:
            raise FlatLibraryError("unknown", UNKNOWN_FLAT) from None
        except (OSError, ValueError, SkyError):
            raise FlatLibraryError("invalid", "The flat file cannot be read.") from None

    def jpeg(self, version: str) -> bytes | None:
        """The preview of a flat as JPEG bytes, or `None` when it has none."""
        try:
            return self._path(version, ".jpg").read_bytes()
        except (OSError, FlatLibraryError):
            return None

    # --- Writing ------------------------------------------------------------------------------

    def add(
        self,
        flat: FloatArray,
        report: Mapping[str, Any],
        *,
        preview: bytes | None = None,
        keep: int = KEEP_FLATS,
    ) -> FlatEntry:
        """Add a pending flat, and delete the flats beyond the newest `keep`.

        `report` is the result of `build_report`. The call writes the flat first, then the
        preview, and the report last, so a flat without a report is not in the list. Raises
        `FlatFileError` when a file cannot be written, and `SkyError` for a flat that holds a
        value at or below zero.
        """
        array = np.ascontiguousarray(flat, dtype=np.float32)
        version = ArrayFlat(array).version
        with self._lock:
            active = self.active_version() == version
            data = {
                **report,
                "schema": REPORT_FORMAT,
                "version": version,
                "state": APPROVED if active else PENDING,
            }
            payload = _json_bytes(data)  # before the first write, so a bad value leaves no file
            try:
                write_flat(self._path(version, ".npy"), array)
                if preview is not None:
                    write_atomic(self._path(version, ".jpg"), preview)
                write_atomic(self._path(version, ".json"), payload)
            except (OSError, FlatFileError):
                self._remove_files(version)
                raise
            entry = _entry_from(version, data)
            assert entry is not None
            self.prune(keep=keep)
        return entry

    def activate(
        self,
        version: str,
        *,
        now_utc_ns: int,
        expect_shape: tuple[int, int] | None = None,
    ) -> FlatEntry:
        """Make a flat the active one: the survey divides by it from its next frame on.

        `expect_shape` is the `(height, width)` of the survey frame. Raises `FlatLibraryError`
        with the reason `unknown` or `invalid`.
        """
        with self._lock:
            entry = self.get(version)
            if entry is None:
                raise FlatLibraryError("unknown", UNKNOWN_FLAT)
            model = self.load(version)
            if expect_shape is not None and model.shape != expect_shape:
                raise FlatLibraryError(
                    "invalid",
                    f"The flat has {model.shape[1]} x {model.shape[0]} pixels, and the survey "
                    f"frame has {expect_shape[1]} x {expect_shape[0]}.",
                )
            data = {**entry.report, "state": APPROVED, "activated_utc_ns": now_utc_ns}
            data["activated_utc"] = utc_ns_to_iso(now_utc_ns, digits=0)
            try:
                write_atomic(self._path(version, ".json"), _json_bytes(data))
                pointer = {"version": version, "activated_utc": data["activated_utc"]}
                write_atomic(self._directory / POINTER_NAME, _json_bytes(pointer))
            except OSError as error:
                raise FlatLibraryError(
                    "invalid", f"The flat could not be activated ({type(error).__name__})."
                ) from None
            updated = self.get(version)
            assert updated is not None
            return updated

    def delete(self, version: str) -> None:
        """Delete a flat that is not the active one. Raises `FlatLibraryError`."""
        with self._lock:
            if self.get(version) is None:
                raise FlatLibraryError("unknown", UNKNOWN_FLAT)
            if self.active_version() == version:
                raise FlatLibraryError("active", FLAT_IN_USE)
            self._remove_files(version)

    def prune(self, keep: int = KEEP_FLATS) -> tuple[str, ...]:
        """Delete the flats beyond the newest `keep`, but never the active one.

        Returns the versions that it deleted.
        """
        with self._lock:
            active = self.active_version()
            gone: list[str] = []
            for entry in self.entries()[keep:]:
                if entry.version == active:
                    continue
                self._remove_files(entry.version)
                gone.append(entry.version)
            return tuple(gone)

    def _remove_files(self, version: str) -> None:
        """Delete the report first, so that the flat leaves the list before its pixels go."""
        for suffix in (".json", ".npy", ".jpg"):
            with contextlib.suppress(OSError, FlatLibraryError):
                self._path(version, suffix).unlink()


def _json_bytes(data: Mapping[str, Any]) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


# --- The flat that the survey uses ----------------------------------------------------------------


class ActiveFlat:
    """The flat that a survey path divides by, and the cheap check that notices a change.

    `current()` stats the pointer of the library. While the stamp is the one that it knew, it
    returns the flat that it holds. When the stamp changed, it loads the active flat of the
    library (and when there is none, `flat_file`, and when there is none, a unit flat). A library
    flat that cannot be read leaves the flat in place, and the next change of the pointer tries
    again. A `flat_file` that cannot be read raises `SkyError`, as `load_flat` does, so a bad
    setting stops the start and not a frame. The object is safe to call from several threads.
    """

    def __init__(self, library_dir: Path | str | None, flat_file: str = "") -> None:
        self._library = None if not library_dir else FlatLibrary(library_dir)
        self._flat_file = flat_file
        self._lock = threading.Lock()
        self._model: FlatModel | None = None
        self._pinned: FlatModel | None = None
        self._stamp: tuple[int, int, int] | None = None
        self._known = False

    @classmethod
    def from_config(cls, config: SurveyConfig) -> ActiveFlat:
        """The rule for a configuration: `calibration_dir/flats/`, then `flat_file`, then unit."""
        directory = Path(config.calibration_dir) / FLATS_DIRNAME if config.calibration_dir else None
        return cls(directory, config.flat_file)

    def current(self) -> FlatModel:
        """The flat to use now. One `stat` of the pointer when nothing changed."""
        with self._lock:
            stamp = None if self._library is None else self._library.pointer_stamp()
            if self._model is not None and self._known and stamp == self._stamp:
                return self._model
            model = self._resolve(stamp)
            self._stamp, self._known, self._model = stamp, True, model
            return model

    def _resolve(self, stamp: tuple[int, int, int] | None) -> FlatModel:
        if stamp is not None and self._library is not None:
            version = self._library.active_version()
            if version is not None:
                try:
                    return self._library.load(version)
                except FlatLibraryError as error:
                    log.warning("the active flat %s cannot be used: %s", version, error.message)
                    if self._model is not None:
                        return self._model
        return self._pinned_flat()

    def _pinned_flat(self) -> FlatModel:
        if self._pinned is None:
            self._pinned = load_flat(self._flat_file) if self._flat_file else UnitFlat()
        return self._pinned


_SOURCES: dict[tuple[str, str], ActiveFlat] = {}
_SOURCES_LOCK = threading.Lock()
_MAX_SOURCES = 8


def active_flat(config: SurveyConfig) -> FlatModel:
    """The flat that the survey path divides by: the library's active flat, `flat_file`, or unit.

    The call is cheap: one `stat` of the pointer, and a read of the flat only when the pointer
    changed. It is safe to call from any thread of `core`. It keeps one `ActiveFlat` for each
    pair of `calibration_dir` and `flat_file`. Raises `SkyError` for a `flat_file` that cannot
    be read.
    """
    key = (config.calibration_dir, config.flat_file)
    with _SOURCES_LOCK:
        source = _SOURCES.get(key)
        if source is None:
            if len(_SOURCES) >= _MAX_SOURCES:
                _SOURCES.clear()
            source = _SOURCES[key] = ActiveFlat.from_config(config)
    return source.current()


FlatSource = Callable[[], FlatModel]


@dataclass(frozen=True, slots=True)
class FlatPins:
    """How the active flat and `[survey] flat_file` relate, for the page."""

    active_version: str | None
    flat_file_pinned: bool

    @property
    def library_overrides(self) -> bool:
        """Whether the library's flat wins over a `flat_file` that the configuration names."""
        return self.flat_file_pinned and self.active_version is not None


def flat_pins(config: SurveyConfig) -> FlatPins:
    """The active version of the library, and whether the configuration pins `flat_file`."""
    version = None
    if config.calibration_dir:
        version = FlatLibrary.from_calibration(config.calibration_dir).active_version()
    return FlatPins(version, bool(config.flat_file))
