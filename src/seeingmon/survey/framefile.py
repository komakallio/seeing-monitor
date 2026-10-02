"""FITS files of survey frames.

`core` keeps some survey frames on disk (every tenth long frame, and the frames of events), so that
a later reanalysis (a better detector, another catalog, a new dark model) can read the pixels and
the settings of a frame without the store. This module writes and reads those files.

**Pixels.** The file holds native ADC counts. A 16-bit frame carries the ADC value in the high bits
of its container, so the writer shifts it down by `16 - ADCBITS` bits. That costs one copy of the
frame, and it makes a Rice-compressed file about a quarter smaller than the container values would.
An 8-bit frame is stored as it is, and a comment says that it holds the top 8 bits of the ADC value.

**Header.** The cards describe what a reanalysis needs and nothing that locates the station: the
time (the start and the middle of the exposure, its quality and its error), the exposure, the gain
and the conversion gain, the readout mode, the binning, the ROI, the sensor temperature, the pixel
size, the plate scale, and the optics. They also carry the model of the camera, the profile, the
station ID, the sequence number, the flags of the frame, the software version, and the reason why
the frame was kept. When the survey analysis knows them, they also carry the cloud fraction
(`CLOUDFRC`) and the transparency (`TRANSP`) of the frame, so that a later step can pick the clear
frames without the store. They never carry a serial number, an address, or the coordinates of the
site, and they carry no Sun or Moon position either: a reader that needs those computes them from
the time. The writer puts the cards in the primary header and in the header of the compressed image.

**Compression.** `compress=True` writes the image as a tile-compressed HDU with Rice coding
(`astropy.io.fits.CompImageHDU`, `RICE_1`, one image row for each tile). The coding is lossless. If
astropy or its compressor is missing, `rice_available()` is false, and `write_frame_fits` writes an
uncompressed file with the `seeingmon.solvers.fitsio` writer, which needs no astropy and writes the
frame in small chunks. The astropy import costs about 25 MB of memory, so the module loads it when
the first file is written.

`read_frame_fits` reads either kind of file. `read_frame_header` reads only the header, which is
cheap, and `parse_frame_header` turns a header into a `FrameMeta` that tolerates a file without the
newer cards.
"""

from __future__ import annotations

import functools
import io
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias

import numpy as np
import numpy.typing as npt

import seeingmon
from seeingmon.clock import NS_PER_S, iso_to_utc_ns, utc_ns_to_iso
from seeingmon.frames import Frame, PixelFormat
from seeingmon.profile import Profile
from seeingmon.profile.errors import ProfileError
from seeingmon.solvers import fitsio

log = logging.getLogger("seeingmon.survey")

Pixels: TypeAlias = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]
CardValue: TypeAlias = bool | int | float | str
Card: TypeAlias = tuple[str, CardValue, str]

CONTAINER_BITS = 16


def native_pixels(frame: Frame) -> Pixels:
    """The pixels of a frame as they go into the file: native ADC counts.

    A 16-bit frame is shifted down by `16 - adc_bits` bits, which makes one new array. A frame whose
    ADC fills the container, and an 8-bit frame, come back as they are, without a copy.
    """
    if frame.pixel_format is PixelFormat.RAW8:
        return frame.data
    shift = CONTAINER_BITS - frame.adc_bits
    if shift <= 0:
        return frame.data
    return np.asarray(frame.data >> shift, dtype=np.uint16)


def _fits_time(t_utc_ns: int) -> str:
    """A FITS date: ISO 8601 with milliseconds and no zone (`TIMESYS` says UTC)."""
    return utc_ns_to_iso(t_utc_ns, digits=3).removesuffix("Z")


def frame_cards(
    frame: Frame,
    *,
    profile: Profile | None,
    station_id: str = "",
    reasons: Sequence[str] = (),
    cloud_fraction: float | None = None,
    transparency: float | None = None,
) -> list[Card]:
    """The header cards of a frame: a keyword, a value, and a short comment.

    `reasons` says why the frame was kept (`every_10`, `event:cloud`), and it goes into the `KEPT`
    card. Every comment is short enough that a card fits in 80 characters with a long value.
    The cards that come from the profile (the binning, the pixel size, the plate scale, the
    conversion gain, and the optics) are left out when the profile does not know the readout mode.
    `cloud_fraction` and `transparency` are the values of the survey analysis, and a value that the
    analysis could not give (`None`) leaves its card out.
    """
    exposure_s = frame.exposure_us / 1e6
    start_ns = frame.t_utc_ns - round(exposure_s * NS_PER_S / 2)
    cards: list[Card] = [
        ("DATE-OBS", _fits_time(start_ns), "UTC at the start of the exposure"),
        ("DATE-AVG", _fits_time(frame.t_utc_ns), "UTC at the middle of the exposure"),
        ("TIMESYS", "UTC", "time system of the dates"),
        ("TIMEQUAL", frame.t_quality.name, "how the time was derived"),
        ("TIMEERR", frame.t_err_ns / NS_PER_S, "[s] 1-sigma error of the time"),
        ("IMAGETYP", "LIGHT", "a survey frame"),
        ("EXPTIME", exposure_s, "[s] exposure time"),
        ("GAIN", int(frame.gain), "camera gain setting"),
        ("READMODE", frame.mode, "readout mode of the profile"),
        ("ADCBITS", int(frame.adc_bits), "depth of the ADC, in bits"),
        ("BUNIT", "adu", "pixel values are ADC counts"),
        ("XORGSUBF", int(frame.roi.x), "[px] x origin of the ROI"),
        ("YORGSUBF", int(frame.roi.y), "[px] y origin of the ROI"),
        ("FRAMESEQ", int(frame.seq), "sequence number in the stream"),
        ("STREAMID", int(frame.stream_id), "the camera configuration"),
        ("FRMFLAGS", int(frame.flags), "bit flags of the frame (see COMMENT)"),
    ]
    if frame.temperature_c is not None:
        cards.append(
            ("CCD-TEMP", round(float(frame.temperature_c), 3), "[degC] sensor temperature")
        )
    if profile is not None:
        cards += _profile_cards(frame, profile, station_id)
    elif station_id:
        cards.append(("STATION", station_id, "station ID"))
    if reasons:
        cards.append(("KEPT", ",".join(reasons), "why the frame was kept"))
    if cloud_fraction is not None:
        cards.append(
            ("CLOUDFRC", round(float(cloud_fraction), 4), "share of expected stars missed")
        )
    if transparency is not None:
        cards.append(
            ("TRANSP", round(float(transparency), 4), "transparency, 1 on the best nights")
        )
    cards.append(("CREATOR", f"seeingmon {seeingmon.__version__}", "software that wrote the file"))
    return cards


def _profile_cards(frame: Frame, profile: Profile, station_id: str) -> list[Card]:
    cards: list[Card] = [("INSTRUME", profile.sensor.name, "camera model")]
    if station_id:
        cards.append(("STATION", station_id, "station ID"))
    cards.append(("PROFILE", profile.id, "hardware profile"))
    cards.append(("FOCALLEN", float(profile.optics.focal_length_mm), "[mm] focal length"))
    cards.append(("APTDIA", float(profile.optics.aperture_mm), "[mm] aperture diameter"))
    try:
        readout = profile.mode(frame.mode)
    except ProfileError:
        return cards
    cards += [
        ("XBINNING", int(readout.sdk_bin), "binning of the readout mode"),
        ("YBINNING", int(readout.sdk_bin), "binning of the readout mode"),
        ("XPIXSZ", float(readout.pixel_size_um), "[um] pixel size after binning"),
        ("YPIXSZ", float(readout.pixel_size_um), "[um] pixel size after binning"),
        ("PIXSCALE", float(profile.plate_scale_arcsec_per_px(readout)), "[arcsec/px] plate scale"),
    ]
    try:
        cards.append(("EGAIN", float(profile.e_per_adu(readout, frame.gain)), "[e-/adu] gain"))
        cards.append(
            ("RDNOISE", float(profile.read_noise_e(readout, frame.gain)), "[e-] read noise")
        )
    except (ProfileError, ValueError):
        pass  # a gain that the profile does not cover leaves out the conversion
    return cards


def frame_comments(frame: Frame) -> list[str]:
    """The commentary lines of the header. Each line fits a card."""
    lines = [
        "FRMFLAGS bits: 1 time invalid, 2 recovered, 4 incomplete,",
        "8 simulated, 16 replayed.",
    ]
    if frame.pixel_format is PixelFormat.RAW8:
        return [*lines, "Pixel values are the top 8 bits of the ADC value."]
    return [*lines, "Pixel values are native ADC counts (the container shift is removed)."]


@functools.lru_cache(maxsize=1)
def rice_available() -> bool:
    """Whether astropy can write a Rice-compressed image here. The answer is cached.

    The check compresses a small array, so a missing compressor shows now and not in the middle of
    a write. The first call imports astropy.
    """
    try:
        from astropy.io import fits

        buffer = io.BytesIO()
        image = fits.CompImageHDU(np.zeros((4, 64), dtype=np.uint16), compression_type="RICE_1")
        fits.HDUList([fits.PrimaryHDU(), image]).writeto(buffer)
    except Exception:  # a missing package, a missing compiled part, or a changed interface
        log.warning("astropy cannot write Rice-compressed FITS", exc_info=True)
        return False
    return True


def _astropy_header(cards: Sequence[Card], comments: Sequence[str]) -> Any:
    from astropy.io import fits

    header = fits.Header()
    for keyword, value, comment in cards:
        header[keyword] = (value, comment)
    for line in comments:
        header["COMMENT"] = line
    return header


def write_frame_fits(
    handle: BinaryIO,
    pixels: Pixels,
    cards: Sequence[Card],
    comments: Sequence[str] = (),
    *,
    compress: bool = True,
) -> bool:
    """Write the pixels and the cards to an open binary file. Returns whether it is compressed.

    With `compress=True` the file holds an empty primary HDU and a Rice-compressed image. When
    astropy cannot do that, the function logs a warning once and writes an uncompressed file
    instead, and it returns `False`. The pixels are `uint16` or `uint8`.
    """
    if compress and rice_available():
        from astropy.io import fits

        header = _astropy_header(cards, comments)
        primary = fits.PrimaryHDU(header=header.copy())
        image = fits.CompImageHDU(pixels, header=header, compression_type="RICE_1")
        fits.HDUList([primary, image]).writeto(handle)
        return True
    values = {keyword: value for keyword, value, _ in cards}
    fitsio.write_image_stream(handle, pixels, header=values)
    return False


@dataclass(frozen=True, slots=True)
class FrameFile:
    """What `read_frame_fits` finds: the pixels (native ADC counts) and the header."""

    pixels: Pixels
    header: dict[str, Any]
    compressed: bool


def read_frame_fits(path: str | os.PathLike[str]) -> FrameFile:
    """Read a file that `write_frame_fits` made, compressed or not.

    A compressed file needs astropy. An uncompressed file reads with `seeingmon.solvers.fitsio`.
    """
    unit = fitsio.read_hdu(path, 0)
    if "image" in unit.data:
        return FrameFile(np.asarray(unit.data["image"]), dict(unit.header), compressed=False)
    from astropy.io import fits

    with fits.open(Path(path)) as hdus:
        image = hdus[1]
        header = {
            key: image.header[key] for key in image.header if key not in {"COMMENT", "HISTORY", ""}
        }
        return FrameFile(np.asarray(image.data), header, compressed=True)


# --- Reading the header ---------------------------------------------------------------------


def read_frame_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The header of a frame file, read without its pixels.

    Both kinds of file keep the cards in the primary header (the compressed file copies them there),
    so the function reads only that header, and it needs no astropy.
    """
    return dict(fitsio.read_header(path))


@dataclass(frozen=True, slots=True)
class FrameMeta:
    """What a later step reads from the header of a frame file.

    A value that the header lacks (an old file, or a frame whose analysis had none) is `None`, and
    `flags` is 0. `t_utc_ns` is the time at the middle of the exposure, `kept` the reasons that
    the frame was kept, `origin` the position of the ROI on the sensor, and `flags` the bit flags
    of the frame (see the commentary of the header).
    """

    t_utc_ns: int | None = None
    exposure_s: float | None = None
    gain: int | None = None
    mode: str | None = None
    temperature_c: float | None = None
    adc_bits: int | None = None
    flags: int = 0
    time_quality: str | None = None
    kept: tuple[str, ...] = ()
    cloud_fraction: float | None = None
    transparency: float | None = None
    origin: tuple[int, int] = (0, 0)

    @property
    def time_invalid(self) -> bool:
        """Whether the clock was not synchronized when the frame was taken."""
        return bool(self.flags & 1) or self.time_quality == "INVALID"

    @property
    def events(self) -> tuple[str, ...]:
        """The conditions that made the frame an event frame, such as `cloud` or `unsolved`."""
        found: list[str] = []
        for reason in self.kept:
            if reason.startswith("event:"):
                found.extend(part for part in reason.removeprefix("event:").split("+") if part)
        return tuple(found)


def _number(header: dict[str, Any], key: str) -> float | None:
    value = header.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if np.isfinite(value) else None


def _whole(header: dict[str, Any], key: str) -> int | None:
    value = _number(header, key)
    return None if value is None else round(value)


def _text(header: dict[str, Any], key: str) -> str | None:
    value = header.get(key)
    return value if isinstance(value, str) and value else None


def parse_frame_header(header: dict[str, Any]) -> FrameMeta:
    """Read a header into a `FrameMeta`. A missing or malformed card gives `None`, and no error."""
    t_utc_ns: int | None = None
    stamp = _text(header, "DATE-AVG")
    if stamp is not None:
        try:
            t_utc_ns = iso_to_utc_ns(stamp if stamp.endswith("Z") else f"{stamp}Z")
        except ValueError:
            t_utc_ns = None
    kept = _text(header, "KEPT")
    return FrameMeta(
        t_utc_ns=t_utc_ns,
        exposure_s=_number(header, "EXPTIME"),
        gain=_whole(header, "GAIN"),
        mode=_text(header, "READMODE"),
        temperature_c=_number(header, "CCD-TEMP"),
        adc_bits=_whole(header, "ADCBITS"),
        flags=_whole(header, "FRMFLAGS") or 0,
        time_quality=_text(header, "TIMEQUAL"),
        kept=tuple(part for part in (kept or "").split(",") if part),
        cloud_fraction=_number(header, "CLOUDFRC"),
        transparency=_number(header, "TRANSP"),
        origin=(_whole(header, "XORGSUBF") or 0, _whole(header, "YORGSUBF") or 0),
    )
