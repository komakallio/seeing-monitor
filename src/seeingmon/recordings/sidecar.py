"""Sidecar files: the SharpCap camera settings of a capture, and the JSON sidecar of a burst.

**SharpCap settings sidecar.** SharpCap writes `<capture>.CameraSettings.txt` next to each
capture. The first line is a bracketed section header with the camera model, and every other
line is `Key=Value` without spaces around the `=`. Keys can contain spaces, slashes, and
parentheses (`Exposure/Gain Shift`, `Flip (after dark/flat)`), and some values are empty
(`Notes=`). Values use decimal commas and units (`Exposure=10,0000ms`,
`ActualFrameRate=97,8567fps`, `TimeZone=+3,00`). Times are UTC stamps with seven fractional
digits and a trailing `Z` (`StartCapture`, `MidCapture`, `EndCapture`), and Julian dates
(`JDStartCapture`).

The parser is tolerant. It also accepts `Key: Value` lines, comments, and keys it does not
know, and it never raises on a line: it collects what it could not read in `problems`, as
line numbers and categories without the text of the line. It matches key names without regard
to case, spaces, and punctuation, and it prefers the exact SharpCap names.

**Privacy.** The file holds the camera serial number (`CameraSerialNumber`). The parser drops
every key whose name says serial, S/N, GUID, UUID, or ID, free-text keys such as `Notes`, keys
that name a file, a folder, a person, or a place, and values that look like a GUID, a long
hexadecimal identifier, or an absolute path. `SidecarInfo` keeps none of them, and its `repr`
shows only counts.

**Burst sidecar.** `write_burst_sidecar` and `read_burst_sidecar` handle the JSON sidecar of a
burst: the schema version, the stream settings, the profile ID, and the time quality. It holds
no private values.
"""

from __future__ import annotations

import calendar
import codecs
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path
from typing import TypeVar

NS_PER_S = 1_000_000_000
JD_AT_UNIX_EPOCH = Decimal("2440587.5")
_SECONDS_PER_DAY = 86_400
_MIN_JD = Decimal("2415020")  # 1900
_MAX_JD = Decimal("2816788")  # 3000

_T = TypeVar("_T")

# --- Value helpers ---------------------------------------------------------------------------

_SPACES = " \N{NO-BREAK SPACE}\N{NARROW NO-BREAK SPACE}"  # grouping spaces
_NUMBER = (
    r"[+-]?(?:"
    rf"\d{{1,3}}(?:[{_SPACES}]\d{{3}})+(?:[.,]\d+)?"  # 1 234,5
    r"|\d{1,3}(?:\.\d{3})+,\d+"  # 1.234,5
    r"|\d{1,3}(?:,\d{3})+\.\d+"  # 1,234.5
    r"|\d+(?:[.,]\d+)?"  # 1234,5
    r"|[.,]\d+"  # ,5
    r")(?:[eE][+-]?\d+)?"
)
_NUMBER_ONLY = re.compile(rf"^\s*(?P<number>{_NUMBER})\s*$")
_QUANTITY = re.compile(rf"^\s*(?P<number>{_NUMBER})\s*(?P<unit>\S.*?)?\s*$")
_UTC = re.compile(
    r"^\s*(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})[T ]"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    r"(?:[.,](?P<fraction>\d{1,12}))?\s*(?:Z|z|\+00:?00)\s*$"
)

_DURATION_UNITS_MS = {
    "": 1.0,
    "ms": 1.0,
    "msec": 1.0,
    "millisecond": 1.0,
    "milliseconds": 1.0,
    "s": 1000.0,
    "sec": 1000.0,
    "secs": 1000.0,
    "second": 1000.0,
    "seconds": 1000.0,
    "us": 0.001,
    "usec": 0.001,
    "\N{MICRO SIGN}s": 0.001,
    "\N{GREEK SMALL LETTER MU}s": 0.001,
    "microsecond": 0.001,
    "microseconds": 0.001,
    "min": 60_000.0,
    "minute": 60_000.0,
    "minutes": 60_000.0,
}


def _normalize_number(number: str) -> str:
    """Turn a decimal-comma number into the form that `float` and `Decimal` read."""
    cleaned = re.sub(rf"[{_SPACES}]", "", number)
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):  # 1.234,5: the comma is the decimal mark
            return cleaned.replace(".", "").replace(",", ".")
        return cleaned.replace(",", "")  # 1,234.5
    return cleaned.replace(",", ".")


def _split_quantity(text: str) -> tuple[str, str]:
    """Split `10,0000ms` into the normalized number `10.0000` and the unit `ms`."""
    match = _QUANTITY.match(text)
    if match is None:
        raise ValueError("not a number with an optional unit")
    return _normalize_number(match["number"]), (match["unit"] or "").strip()


def parse_decimal_comma(text: str) -> float:
    """Parse a number that uses a decimal comma or a decimal point, such as `97,8567`.

    A space or a no-break space can group thousands (`1 234,5`). When both marks appear, the
    later one is the decimal mark. The text must hold only the number: `97,8567fps` raises
    `ValueError`, and so do `nan`, `inf`, and an empty string.
    """
    match = _NUMBER_ONLY.match(text)
    if match is None:
        raise ValueError("not a number")
    try:
        return float(_normalize_number(match["number"]))
    except ValueError:
        raise ValueError("not a number") from None


def parse_duration_ms(text: str) -> float:
    """Parse a duration such as `10,0000ms` or `1,5s` into milliseconds.

    The units are `us` (also the micro and Greek mu signs), `ms`, `s`, and `min`, in any
    case. A number without a unit means milliseconds. A negative duration raises `ValueError`.
    """
    number, unit = _split_quantity(text)
    factor = _DURATION_UNITS_MS.get(unit.lower())
    if factor is None:
        raise ValueError("not a duration")
    try:
        value = float(number) * factor
    except ValueError:
        raise ValueError("not a duration") from None
    if value < 0:
        raise ValueError("a duration cannot be negative")
    return value


def parse_utc(text: str) -> int:
    """Parse a UTC stamp such as `2026-01-01T12:00:00.1234567Z` into Unix nanoseconds.

    The stamp must end in `Z` (or `+00:00`), because SharpCap writes local times without it.
    The `T` can be a space, the fraction can use a comma, and it can have 1 to 12 digits
    (extra digits beyond nanoseconds are cut). The result is exact.
    """
    match = _UTC.match(text)
    if match is None:
        raise ValueError("not a UTC time")
    try:
        moment = datetime(
            int(match["year"]),
            int(match["month"]),
            int(match["day"]),
            int(match["hour"]),
            int(match["minute"]),
            int(match["second"]),
            tzinfo=UTC,
        )
    except ValueError:
        raise ValueError("not a valid UTC time") from None
    fraction = (match["fraction"] or "").ljust(9, "0")[:9]
    return calendar.timegm(moment.utctimetuple()) * NS_PER_S + int(fraction)


def parse_julian_date(text: str) -> int:
    """Parse a Julian date such as `2461042,000001` into Unix nanoseconds, rounded to 1 ns.

    The arithmetic is exact in decimal, so the result carries exactly the digits of the text.
    A value outside the years 1900 to 3000 raises `ValueError`.
    """
    match = _NUMBER_ONLY.match(text)
    if match is None:
        raise ValueError("not a Julian date")
    try:
        julian = Decimal(_normalize_number(match["number"]))
    except InvalidOperation:
        raise ValueError("not a Julian date") from None
    if not _MIN_JD <= julian <= _MAX_JD:
        raise ValueError("not a plausible Julian date")
    nanoseconds = (julian - JD_AT_UNIX_EPOCH) * _SECONDS_PER_DAY * NS_PER_S
    return int(nanoseconds.to_integral_value(rounding=ROUND_HALF_EVEN))


def _parse_gain(text: str) -> int:
    number, _unit = _split_quantity(text)
    value = round(float(number))
    if value < 0:
        raise ValueError("a gain cannot be negative")
    return value


def _parse_rate_hz(text: str) -> float:
    number, unit = _split_quantity(text)
    if unit.lower() not in ("", "fps", "hz", "frames/s"):
        raise ValueError("not a frame rate")
    value = float(number)
    if value <= 0:
        raise ValueError("a frame rate must be positive")
    return value


def _parse_temperature_c(text: str) -> float:
    number, unit = _split_quantity(text)
    value = float(number)
    scale = (
        unit.lower()
        .replace("\N{DEGREE SIGN}", "")
        .replace("\N{DEGREE CELSIUS}", "c")
        .replace("deg", "")
        .strip()
    )
    if scale in ("", "c"):
        return value
    if scale == "f":
        return (value - 32) * 5 / 9
    if scale == "k":
        return value - 273.15
    raise ValueError("not a temperature")


def _parse_count(text: str) -> int:
    value = parse_decimal_comma(text)
    if value < 0 or value != int(value):
        raise ValueError("not a count")
    return int(value)


def _parse_duration_s(text: str) -> float:
    return parse_duration_ms(text) / 1000


def _parse_exposure_us(text: str) -> int:
    value = round(parse_duration_ms(text) * 1000)
    if value <= 0:
        raise ValueError("an exposure must be positive")
    return value


def _parse_resolution(text: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\s*[x\N{MULTIPLICATION SIGN}*]\s*(\d+)", text, re.IGNORECASE)
    if match is None:
        raise ValueError("not a resolution")
    width, height = int(match[1]), int(match[2])
    if width <= 0 or height <= 0:
        raise ValueError("not a resolution")
    return width, height


def _parse_binning(text: str) -> int:
    match = re.match(r"\s*(?:bin\s*)?(\d+)", text, re.IGNORECASE)
    if match is None or int(match[1]) <= 0:
        raise ValueError("not a binning factor")
    return int(match[1])


def _parse_utc_offset_s(text: str) -> int:
    clock = re.match(r"^\s*([+-])?(\d{1,2}):(\d{2})\s*$", text)
    if clock is not None:
        seconds = int(clock[2]) * 3600 + int(clock[3]) * 60
        return -seconds if clock[1] == "-" else seconds
    hours = parse_decimal_comma(text)
    if abs(hours) > 18:
        raise ValueError("not a UTC offset")
    return round(hours * 3600)


def _parse_text(text: str) -> str:
    if not text.strip():
        raise ValueError("empty")
    return text.strip()


# --- Privacy ---------------------------------------------------------------------------------

_PRIVATE_TOKENS = frozenset(
    {
        "id",
        "uid",
        "pid",
        "vid",
        "guid",
        "uuid",
        "sn",
        "serial",
        "serialno",
        "serialnumber",
        "notes",
        "note",
        "comment",
        "comments",
        "remarks",
        "file",
        "filename",
        "folder",
        "path",
        "directory",
        "user",
        "username",
        "host",
        "hostname",
        "computer",
        "machine",
        "owner",
        "observer",
        "observatory",
        "telescope",
        "instrument",
        "site",
        "location",
        "latitude",
        "longitude",
        "email",
    }
)
_GUID = re.compile(r"\b[0-9A-Fa-f]{8}-(?:[0-9A-Fa-f]{4}-){3}[0-9A-Fa-f]{12}\b")
_HEX_IDENTIFIER = re.compile(
    r"(?<![0-9A-Fa-f])(?=[0-9A-Fa-f]*[0-9])(?=[0-9A-Fa-f]*[A-Fa-f])[0-9A-Fa-f]{12,}(?![0-9A-Fa-f])"
)
_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|mnt|media|root|var|tmp)/)")


def _key_tokens(key: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", key)  # CameraSerial -> Camera Serial
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)  # USBId -> USB Id
    return [token.lower() for token in re.split(r"[^A-Za-z0-9]+", spaced) if token]


def is_private_key(key: str) -> bool:
    """Whether a key name says serial, S/N, GUID, UUID, or ID, or names private free text."""
    tokens = _key_tokens(key)
    joined = "".join(tokens)
    if any(token in _PRIVATE_TOKENS or token.startswith("serial") for token in tokens):
        return True
    if "serial" in joined or "guid" in joined or "uuid" in joined:
        return True
    if any(a == "s" and b == "n" for a, b in pairwise(tokens)):
        return True  # S/N, S.N.
    return bool(re.search(r"[A-Z0-9]{2,}ID\b", key))  # USBID, CAMERAID


def is_private_value(value: str) -> bool:
    """Whether a value looks like a serial number, a GUID, a long hexadecimal ID, or a path."""
    lowered = value.lower()
    return bool(
        "serial" in lowered
        or "s/n" in lowered
        or _GUID.search(value)
        or _HEX_IDENTIFIER.search(value)
        or _ABSOLUTE_PATH.match(value.strip())
    )


# --- SharpCap sidecar ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SidecarEntry:
    """One `Key=Value` line that survived the privacy filter."""

    section: str
    key: str
    value: str


@dataclass(frozen=True, slots=True)
class _Spec:
    """How to find a setting by key name. All names are normalized (lowercase, letters and digits).

    `exact` lists names that match whole, best first. Then any key that contains one of
    `contains` and all of `require`, and none of `exclude`, matches.
    """

    label: str
    exact: tuple[str, ...]
    contains: tuple[str, ...] = ()
    require: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())


_EXPOSURE = _Spec(
    "exposure",
    ("exposure", "exposuretime", "exposurems", "shutter"),
    ("exposure",),
    exclude=("shift", "max", "min", "auto", "target", "limit", "count", "delay"),
)
_GAIN = _Spec(
    "gain",
    ("gain",),
    ("gain",),
    exclude=("shift", "max", "min", "auto", "target", "limit", "range"),
)
_FPS = _Spec(
    "frame rate",
    ("actualframerate", "measuredframerate", "framerate", "fps"),
    ("framerate", "fps"),
    exclude=("limit", "max", "min", "target"),
)
_TEMPERATURE = _Spec(
    "temperature",
    ("temperature", "sensortemperature", "cameratemperature", "sensortemp", "temp"),
    ("temperature",),
    exclude=("target", "set", "cooler", "power", "min", "max", "limit", "fan", "ambient"),
)
_START_UTC = _Spec(
    "start time",
    ("startcapture", "capturestart", "startcaptureutc", "starttime", "startutc"),
    ("start",),
    exclude=("jd", "julian", "end", "stop", "mid"),
)
_START_JD = _Spec(
    "start Julian date",
    ("jdstartcapture", "startcapturejulian", "jdstart", "startjd"),
    ("jd", "julian"),
    require=("start",),
    exclude=("end", "stop", "mid"),
)
_END_UTC = _Spec(
    "end time",
    ("endcapture", "captureend", "endcaptureutc", "endtime", "endutc", "stopcapture"),
    ("end", "stop"),
    exclude=("jd", "julian", "start", "mid"),
)
_END_JD = _Spec(
    "end Julian date",
    ("jdendcapture", "endcapturejulian", "jdend", "endjd"),
    ("jd", "julian"),
    require=("end",),
    exclude=("start", "mid"),
)
_DURATION = _Spec("duration", ("duration", "captureduration"), ("duration",))
_FRAME_COUNT = _Spec(
    "frame count", ("framecount", "frames", "numberofframes", "capturedframes"), ("framecount",)
)
_RESOLUTION = _Spec(
    "resolution",
    ("resolution", "capturearea", "roi", "imagesize", "framesize", "outputsize"),
    ("resolution",),
)
_BINNING = _Spec("binning", ("binning", "bin"))
_READ_MODE = _Spec("read mode", ("readmode", "readoutmode", "sensormode", "cameramode"))
_COLOUR_SPACE = _Spec("colour space", ("colourspace", "colorspace", "pixelformat", "videoformat"))
_TIME_ZONE = _Spec("time zone", ("timezone", "utcoffset", "timezoneoffset"))


def _candidates(entries: tuple[SidecarEntry, ...], spec: _Spec) -> Iterator[str]:
    """Yield the values of the entries that match `spec`, best match first."""
    normalized = [(_normalize_key(entry.key), entry.value) for entry in entries]
    for name in spec.exact:
        for key, value in normalized:
            if key == name:
                yield value
    for key, value in normalized:
        if (
            key not in spec.exact
            and spec.contains
            and any(word in key for word in spec.contains)
            and all(word in key for word in spec.require)
            and not any(word in key for word in spec.exclude)
        ):
            yield value


def _first(
    entries: tuple[SidecarEntry, ...], spec: _Spec, parse: Callable[[str], _T]
) -> tuple[_T | None, bool]:
    """Parse the best matching value. Returns the result and whether any key matched."""
    present = False
    for value in _candidates(entries, spec):
        present = True
        try:
            return parse(value), True
        except ValueError:
            continue
    return None, present


_MEGAPIXELS = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:mega\s*pixels?|mp)\b", re.IGNORECASE)

# Fields that `parse_sharpcap_sidecar` checks, to report a key that is present but unreadable.
_CHECKED: tuple[tuple[_Spec, Callable[[str], object]], ...] = (
    (_EXPOSURE, _parse_exposure_us),
    (_GAIN, _parse_gain),
    (_FPS, _parse_rate_hz),
    (_TEMPERATURE, _parse_temperature_c),
    (_START_UTC, parse_utc),
    (_END_UTC, parse_utc),
    (_DURATION, _parse_duration_s),
    (_FRAME_COUNT, _parse_count),
    (_RESOLUTION, _parse_resolution),
    (_BINNING, _parse_binning),
    (_TIME_ZONE, _parse_utc_offset_s),
)


@dataclass(frozen=True, slots=True, repr=False)
class SidecarInfo:
    """The cleaned content of a SharpCap settings sidecar, with typed accessors.

    `entries` holds every `Key=Value` line that passed the privacy filter, in file order.
    `problems` lists what the parser could not read, as a line number and a category, without
    the text of the line. `dropped` counts the lines that the privacy filter removed.
    `section` is the first section header, which SharpCap fills with the camera model.

    Each accessor returns `None` when the key is missing or its value cannot be parsed.
    """

    entries: tuple[SidecarEntry, ...] = ()
    problems: tuple[str, ...] = ()
    dropped: int = 0
    section: str = field(default="")

    def __repr__(self) -> str:
        return (
            f"<SidecarInfo {len(self.entries)} entries, {self.dropped} dropped, "
            f"{len(self.problems)} problems>"
        )

    @property
    def values(self) -> Mapping[str, str]:
        """The cleaned key-value mapping, with each key as written. The first line of a key wins."""
        mapping: dict[str, str] = {}
        for entry in self.entries:
            mapping.setdefault(entry.key, entry.value)
        return mapping

    def get(self, name: str) -> str | None:
        """The value of a key, matched without regard to case, spaces, and punctuation."""
        wanted = _normalize_key(name)
        for entry in self.entries:
            if _normalize_key(entry.key) == wanted:
                return entry.value
        return None

    @property
    def camera_model(self) -> str | None:
        """The text of the first section header, which SharpCap fills with the camera model."""
        return self.section or None

    @property
    def exposure_us(self) -> int | None:
        """The exposure in microseconds (`Exposure=10,0000ms` gives 10000)."""
        return _first(self.entries, _EXPOSURE, _parse_exposure_us)[0]

    @property
    def gain(self) -> int | None:
        return _first(self.entries, _GAIN, _parse_gain)[0]

    @property
    def fps(self) -> float | None:
        """The measured frame rate in frames per second (`ActualFrameRate`)."""
        return _first(self.entries, _FPS, _parse_rate_hz)[0]

    @property
    def sensor_temperature_c(self) -> float | None:
        return _first(self.entries, _TEMPERATURE, _parse_temperature_c)[0]

    @property
    def start_utc_ns(self) -> int | None:
        """The start of the capture in Unix nanoseconds: `StartCapture`, else its Julian date.

        SharpCap sets `StartCapture` to the timestamp of the first frame, so it matches the
        first timestamp in the SER trailer.
        """
        value = _first(self.entries, _START_UTC, parse_utc)[0]
        return value if value is not None else _first(self.entries, _START_JD, parse_julian_date)[0]

    @property
    def end_utc_ns(self) -> int | None:
        """The end of the capture in Unix nanoseconds: `EndCapture`, else its Julian date.

        SharpCap sets `EndCapture` to the timestamp of the last frame plus one exposure.
        """
        value = _first(self.entries, _END_UTC, parse_utc)[0]
        return value if value is not None else _first(self.entries, _END_JD, parse_julian_date)[0]

    @property
    def duration_s(self) -> float | None:
        return _first(self.entries, _DURATION, _parse_duration_s)[0]

    @property
    def frame_count(self) -> int | None:
        return _first(self.entries, _FRAME_COUNT, _parse_count)[0]

    @property
    def resolution(self) -> tuple[int, int] | None:
        """The capture size as `(width, height)` (`Resolution=320x240`)."""
        return _first(self.entries, _RESOLUTION, _parse_resolution)[0]

    @property
    def binning(self) -> int | None:
        """SharpCap's binning factor. It is not the SDK binning: see `readout_mode`."""
        return _first(self.entries, _BINNING, _parse_binning)[0]

    @property
    def read_mode(self) -> str | None:
        """The read mode as written, such as `11 Megapixel`."""
        return _first(self.entries, _READ_MODE, _parse_text)[0]

    @property
    def colour_space(self) -> str | None:
        """The pixel format as written, such as `MONO8`."""
        return _first(self.entries, _COLOUR_SPACE, _parse_text)[0]

    @property
    def utc_offset_s(self) -> int | None:
        """The local time zone offset in seconds (`TimeZone=+3,00` gives 10800)."""
        return _first(self.entries, _TIME_ZONE, _parse_utc_offset_s)[0]

    @property
    def readout_mode(self) -> str | None:
        """The profile readout mode of the capture, `bin1` or `bin2`, or `None` when unsure.

        This is a heuristic, because the sidecar names SharpCap's read mode and binning, not
        the vendor SDK's binning. The ASI294 has two read modes. The 11 megapixel mode is the
        sensor's 2 x 2 binned readout (4144 x 2822), which the SDK calls bin2, and SharpCap
        reports it as `Binning=1`. The full-resolution mode (about 47 megapixels) is the SDK's
        bin1 with `Binning=1`, and it becomes bin2 when SharpCap bins it 2 x 2.

        - A read mode of 8 to 16 megapixels with `Binning=1` gives `bin2`.
        - A read mode of 40 megapixels or more with `Binning=1` gives `bin1`, and with
          `Binning=2` it gives `bin2`.
        - A missing `Binning` counts as 1. Any other case, and a read mode that names no size,
          gives `None`, and the caller picks a default.
        """
        read_mode = self.read_mode
        if read_mode is None:
            return None
        match = _MEGAPIXELS.search(read_mode)
        if match is None:
            return None
        megapixels = float(_normalize_number(match[1]))
        binning = 1 if self.get("Binning") is None else self.binning
        if binning is None:
            return None
        if 8 <= megapixels <= 16:
            return "bin2" if binning == 1 else None
        if megapixels >= 40:
            return {1: "bin1", 2: "bin2"}.get(binning)
        return None


_SECTION = re.compile(r"^\[(?P<name>.*)\]$")


def _split_pair(line: str) -> tuple[str, str] | None:
    """Split at the first `=` or `:`, whichever comes first."""
    positions = [position for position in (line.find("="), line.find(":")) if position >= 0]
    if not positions:
        return None
    cut = min(positions)
    return line[:cut].strip(), line[cut + 1 :].strip()


def parse_sharpcap_sidecar(text: str) -> SidecarInfo:
    """Parse the text of a SharpCap settings sidecar. It never raises.

    Lines that do not parse go into `problems` as a line number and a category. A key that
    is present but has an unreadable value is reported by its setting name, never by its text.
    """
    entries: list[SidecarEntry] = []
    problems: list[str] = []
    dropped = 0
    section = ""
    first_section: str | None = None
    for number, raw in enumerate(
        text.lstrip("\N{ZERO WIDTH NO-BREAK SPACE}").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith(("#", ";", "//")):
            continue
        header = _SECTION.match(line)
        if header is not None:
            name = header["name"].strip()
            section = "" if is_private_value(name) or is_private_key(name) else name
            if first_section is None:
                first_section = section
            continue
        pair = _split_pair(line)
        if pair is None:
            problems.append(f"line {number}: no key and value")
            continue
        key, value = pair
        if not key:
            problems.append(f"line {number}: no key")
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if is_private_key(key) or is_private_value(value) or is_private_value(key):
            dropped += 1
            continue
        entries.append(SidecarEntry(section=section, key=key, value=value))
    frozen = tuple(entries)
    for spec, parse in _CHECKED:
        result, present = _first(frozen, spec, parse)
        if present and result is None:
            problems.append(f"the {spec.label} value cannot be read")
    return SidecarInfo(
        entries=frozen,
        problems=tuple(problems),
        dropped=dropped,
        section=first_section or "",
    )


def decode_sidecar_bytes(raw: bytes) -> str:
    """Decode a sidecar: UTF-8 (with or without a BOM), UTF-16 with a BOM, else Windows-1252."""
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def sharpcap_sidecar_path(ser_path: str | Path) -> Path:
    """The path of the settings sidecar of a capture: `<capture>.CameraSettings.txt`."""
    path = Path(ser_path)
    return path.with_name(f"{path.stem}.CameraSettings.txt")


def read_sharpcap_sidecar(path: str | Path) -> SidecarInfo:
    """Read and parse a SharpCap settings sidecar. Raises `OSError` when the file is unreadable."""
    return parse_sharpcap_sidecar(decode_sidecar_bytes(Path(path).read_bytes()))
