"""The cap catalog: Gaia DR3 and Tycho-2 stars around the north celestial pole.

The camera never points far from the pole, so the survey path needs only a cap of the sky: the
stars within a radius (15 degrees by default) of the pole, down to G = 13. That is about 82,000
stars and 4 MB. `seeingmon catalog build` makes the file on a machine with a network
connection, and the Raspberry Pi only reads it.

**Data model.** Each row holds the ICRS position and proper motion at the catalog epoch (the
Gaia DR3 reference epoch, J2016.0), the parallax, the Gaia G magnitude and BP-RP color, a
Tycho-2 V magnitude for the bright stars, a source ID, and flags. Rows sort by G, so the
brightest star comes first, and a row index identifies a star for as long as the file stays the
same. `CapCatalog.content_id` names that file for provenance.

**File format** (little-endian, version 1):

- The header: 8 bytes of magic (`SMCAPCAT`), the version, the row size, the header size, the
  number of rows, the CRC-32 of the rows, the catalog epoch, the cap center (right ascension
  and declination in degrees), the cap radius in degrees, the Gaia and Tycho-2 magnitude
  limits, the build time (UTC seconds, 0 when unknown), and 64 bytes of text for the sources.
- The rows, 50 bytes each (`ROW_DTYPE`).

A loader skips `header_size` bytes, so a later version can append header fields. A loader
refuses a newer major version, a wrong row size, a short file, and a CRC mismatch.

**Source IDs.** A Gaia source has its positive 64-bit Gaia DR3 `source_id`. A star that only
Tycho-2 has gets the negative number `-(TYC1 * 1_000_000 + TYC2 * 10 + TYC3)`.
"""

from __future__ import annotations

import os
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt

from seeingmon.survey.geometry import FloatArray, radec_to_vector

CATALOG_MAGIC = b"SMCAPCAT"
CATALOG_VERSION = 1
GAIA_DR3_EPOCH_JYEAR = 2016.0
NOTES_BYTES = 64

# Row flags.
FLAG_TYCHO_ONLY = 1  # no Gaia source: G approximates V, and the color is unknown
FLAG_TYCHO_V = 2  # `v_mag` holds a Tycho-2 V magnitude
FLAG_NO_PROPER_MOTION = 4  # the proper motion is unknown and stored as 0
FLAG_NO_PARALLAX = 8  # the parallax is unknown and stored as 0
FLAG_NO_COLOR = 16  # `bp_rp` is unknown and stored as NaN

ROW_DTYPE = np.dtype(
    [
        ("source_id", "<i8"),
        ("ra_deg", "<f8"),
        ("dec_deg", "<f8"),
        ("pm_ra_mas_yr", "<f4"),
        ("pm_dec_mas_yr", "<f4"),
        ("parallax_mas", "<f4"),
        ("g_mag", "<f4"),
        ("bp_rp", "<f4"),
        ("v_mag", "<f4"),
        ("flags", "<u2"),
    ]
)

_HEADER = struct.Struct("<8sHHIQIddddffq64s")


class CatalogError(Exception):
    """A catalog file is missing data, damaged, or from an unsupported version."""


@dataclass(frozen=True, slots=True)
class CatalogInfo:
    """The header of a catalog file."""

    version: int
    n_stars: int
    epoch_jyear: float
    cap_ra_deg: float
    cap_dec_deg: float
    cap_radius_deg: float
    gaia_mag_limit: float
    tycho_mag_limit: float
    created_utc_s: int
    notes: str
    crc32: int


def tycho_source_id(tyc1: int, tyc2: int, tyc3: int) -> int:
    """The negative source ID of a Tycho-2 star with no Gaia counterpart."""
    return -(tyc1 * 1_000_000 + tyc2 * 10 + tyc3)


def _optional(
    values: npt.ArrayLike | None, n: int, fill: float
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """A column with NaN replaced by `fill`, and the mask of the entries that were missing."""
    if values is None:
        return np.full(n, fill), np.ones(n, dtype=np.bool_)
    array = np.asarray(values, dtype=np.float64)
    missing = ~np.isfinite(array)
    return np.where(missing, fill, array), missing


class CapCatalog:
    """Stars in a cap around the pole, sorted by G magnitude (brightest first).

    Build one with `from_columns`, read one with `load_catalog`, and write one with
    `write_catalog`. The catalog is read-only. Row numbers identify stars, and every query
    returns row numbers in the sorted order.
    """

    def __init__(
        self,
        rows: npt.NDArray[np.void],
        *,
        epoch_jyear: float = GAIA_DR3_EPOCH_JYEAR,
        cap_ra_deg: float = 0.0,
        cap_dec_deg: float = 90.0,
        cap_radius_deg: float = 15.0,
        gaia_mag_limit: float = 13.0,
        tycho_mag_limit: float = 12.0,
        created_utc_s: int = 0,
        notes: str = "",
    ) -> None:
        if rows.dtype != ROW_DTYPE or rows.ndim != 1:
            raise CatalogError("rows must be a 1-D array with the catalog row type")
        if len(notes.encode("utf-8")) > NOTES_BYTES:
            raise CatalogError(f"notes must fit in {NOTES_BYTES} bytes")
        rows.flags.writeable = False
        self.rows = rows
        self.epoch_jyear = float(epoch_jyear)
        self.cap_ra_deg = float(cap_ra_deg)
        self.cap_dec_deg = float(cap_dec_deg)
        self.cap_radius_deg = float(cap_radius_deg)
        self.gaia_mag_limit = float(gaia_mag_limit)
        self.tycho_mag_limit = float(tycho_mag_limit)
        self.created_utc_s = int(created_utc_s)
        self.notes = notes

    @classmethod
    def from_columns(
        cls,
        *,
        source_id: npt.ArrayLike,
        ra_deg: npt.ArrayLike,
        dec_deg: npt.ArrayLike,
        g_mag: npt.ArrayLike,
        pm_ra_mas_yr: npt.ArrayLike | None = None,
        pm_dec_mas_yr: npt.ArrayLike | None = None,
        parallax_mas: npt.ArrayLike | None = None,
        bp_rp: npt.ArrayLike | None = None,
        v_mag: npt.ArrayLike | None = None,
        extra_flags: npt.ArrayLike | None = None,
        epoch_jyear: float = GAIA_DR3_EPOCH_JYEAR,
        cap_ra_deg: float = 0.0,
        cap_dec_deg: float = 90.0,
        cap_radius_deg: float = 15.0,
        gaia_mag_limit: float = 13.0,
        tycho_mag_limit: float = 12.0,
        created_utc_s: int = 0,
        notes: str = "",
    ) -> CapCatalog:
        """Build a catalog from columns and sort it by G.

        A column that is `None`, and a NaN inside a column, count as missing. The builder
        stores a missing proper motion or parallax as 0 and a missing color or V magnitude as
        NaN, and it sets the matching flag. `extra_flags` adds flag bits to every row.
        """
        ra = np.asarray(ra_deg, dtype=np.float64)
        n = ra.shape[0]
        rows = np.zeros(n, dtype=ROW_DTYPE)
        rows["source_id"] = np.asarray(source_id, dtype=np.int64)
        rows["ra_deg"] = ra
        rows["dec_deg"] = np.asarray(dec_deg, dtype=np.float64)
        rows["g_mag"] = np.asarray(g_mag, dtype=np.float64)

        pm_ra, no_pm_ra = _optional(pm_ra_mas_yr, n, 0.0)
        pm_dec, no_pm_dec = _optional(pm_dec_mas_yr, n, 0.0)
        no_pm = no_pm_ra | no_pm_dec  # one missing component makes the whole motion unknown
        rows["pm_ra_mas_yr"] = np.where(no_pm, 0.0, pm_ra)
        rows["pm_dec_mas_yr"] = np.where(no_pm, 0.0, pm_dec)
        parallax, no_parallax = _optional(parallax_mas, n, 0.0)
        rows["parallax_mas"] = parallax
        color, no_color = _optional(bp_rp, n, np.nan)
        rows["bp_rp"] = color
        v, no_v = _optional(v_mag, n, np.nan)
        rows["v_mag"] = v

        flag_bits = (
            no_pm * FLAG_NO_PROPER_MOTION
            + no_parallax * FLAG_NO_PARALLAX
            + no_color * FLAG_NO_COLOR
            + (~no_v) * FLAG_TYCHO_V
        ).astype(np.uint16)
        if extra_flags is not None:
            flag_bits |= np.asarray(extra_flags, dtype=np.uint16)
        rows["flags"] = flag_bits
        if not (
            np.all(np.isfinite(rows["ra_deg"]))
            and np.all(np.isfinite(rows["dec_deg"]))
            and np.all(np.abs(rows["dec_deg"]) <= 90.0)
            and np.all(np.isfinite(rows["g_mag"]))
        ):
            raise CatalogError(
                "positions and G magnitudes must be finite, with declinations from -90 to 90"
            )
        order = np.argsort(rows["g_mag"], kind="stable")
        return cls(
            rows[order],
            epoch_jyear=epoch_jyear,
            cap_ra_deg=cap_ra_deg,
            cap_dec_deg=cap_dec_deg,
            cap_radius_deg=cap_radius_deg,
            gaia_mag_limit=gaia_mag_limit,
            tycho_mag_limit=tycho_mag_limit,
            created_utc_s=created_utc_s,
            notes=notes,
        )

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    # --- Columns -------------------------------------------------------------------------

    @cached_property
    def source_id(self) -> npt.NDArray[np.int64]:
        return np.ascontiguousarray(self.rows["source_id"])

    @cached_property
    def ra_deg(self) -> FloatArray:
        return np.ascontiguousarray(self.rows["ra_deg"])

    @cached_property
    def dec_deg(self) -> FloatArray:
        return np.ascontiguousarray(self.rows["dec_deg"])

    @cached_property
    def pm_ra_mas_yr(self) -> FloatArray:
        return np.ascontiguousarray(self.rows["pm_ra_mas_yr"], dtype=np.float64)

    @cached_property
    def pm_dec_mas_yr(self) -> FloatArray:
        return np.ascontiguousarray(self.rows["pm_dec_mas_yr"], dtype=np.float64)

    @cached_property
    def parallax_mas(self) -> FloatArray:
        return np.ascontiguousarray(self.rows["parallax_mas"], dtype=np.float64)

    @cached_property
    def g_mag(self) -> FloatArray:
        return np.ascontiguousarray(self.rows["g_mag"], dtype=np.float64)

    @cached_property
    def bp_rp(self) -> FloatArray:
        return np.ascontiguousarray(self.rows["bp_rp"], dtype=np.float64)

    @cached_property
    def v_mag(self) -> FloatArray:
        return np.ascontiguousarray(self.rows["v_mag"], dtype=np.float64)

    @cached_property
    def flags(self) -> npt.NDArray[np.uint16]:
        return np.ascontiguousarray(self.rows["flags"])

    @cached_property
    def vectors(self) -> FloatArray:
        """Unit vectors of the catalog positions (ICRS, at the catalog epoch), shape `(N, 3)`."""
        return radec_to_vector(self.ra_deg, self.dec_deg)

    # --- Queries -------------------------------------------------------------------------

    def cone(
        self,
        center: npt.ArrayLike,
        radius_deg: float,
        *,
        max_g_mag: float | None = None,
    ) -> npt.NDArray[np.intp]:
        """Row numbers of the stars within `radius_deg` of `center`, brightest first.

        `center` is a unit vector (3 numbers) or `(ra_deg, dec_deg)` (2 numbers). A cone is
        measured on the catalog positions at the catalog epoch, so allow a margin for proper
        motion when you need every star. `max_g_mag` keeps the stars with a smaller G.
        """
        c = np.asarray(center, dtype=np.float64)
        if c.shape == (2,):
            c = radec_to_vector(c[0], c[1])
        elif c.shape != (3,):
            raise ValueError("center must be a vector (3 numbers) or (ra_deg, dec_deg)")
        c = c / np.linalg.norm(c)
        inside = self.vectors @ c >= np.cos(np.radians(min(radius_deg, 180.0)))
        if max_g_mag is not None:
            inside &= self.g_mag < max_g_mag
        return np.flatnonzero(inside)

    def brighter_than(self, g_mag: float) -> npt.NDArray[np.intp]:
        """Row numbers of the stars with a G magnitude below `g_mag`."""
        return np.flatnonzero(self.g_mag < g_mag)

    # --- Identity ------------------------------------------------------------------------

    @cached_property
    def crc32(self) -> int:
        return zlib.crc32(self.rows.tobytes()) & 0xFFFFFFFF

    @property
    def content_id(self) -> str:
        """A short name for the contents, for provenance. Eight hexadecimal digits."""
        return f"{self.crc32:08x}"

    def info(self) -> CatalogInfo:
        return CatalogInfo(
            version=CATALOG_VERSION,
            n_stars=len(self),
            epoch_jyear=self.epoch_jyear,
            cap_ra_deg=self.cap_ra_deg,
            cap_dec_deg=self.cap_dec_deg,
            cap_radius_deg=self.cap_radius_deg,
            gaia_mag_limit=self.gaia_mag_limit,
            tycho_mag_limit=self.tycho_mag_limit,
            created_utc_s=self.created_utc_s,
            notes=self.notes,
            crc32=self.crc32,
        )


# --- Files -------------------------------------------------------------------------------


def encode_catalog(catalog: CapCatalog) -> bytes:
    """The bytes of a catalog file."""
    header = _HEADER.pack(
        CATALOG_MAGIC,
        CATALOG_VERSION,
        ROW_DTYPE.itemsize,
        _HEADER.size,
        len(catalog),
        catalog.crc32,
        catalog.epoch_jyear,
        catalog.cap_ra_deg,
        catalog.cap_dec_deg,
        catalog.cap_radius_deg,
        catalog.gaia_mag_limit,
        catalog.tycho_mag_limit,
        catalog.created_utc_s,
        catalog.notes.encode("utf-8"),
    )
    return header + catalog.rows.tobytes()


def decode_catalog(data: bytes | bytearray | memoryview) -> CapCatalog:
    """Parse the bytes of a catalog file. Raises `CatalogError` for anything invalid."""
    view = memoryview(data).cast("B")
    if view.nbytes < _HEADER.size:
        raise CatalogError("the file is shorter than the catalog header")
    (
        magic,
        version,
        row_size,
        header_size,
        n_stars,
        crc32,
        epoch,
        cap_ra,
        cap_dec,
        cap_radius,
        gaia_limit,
        tycho_limit,
        created,
        raw_notes,
    ) = _HEADER.unpack_from(view)
    if magic != CATALOG_MAGIC:
        raise CatalogError("the file is not a cap catalog (bad magic)")
    if version != CATALOG_VERSION:
        raise CatalogError(
            f"unsupported catalog version {version}; this code reads {CATALOG_VERSION}"
        )
    if row_size != ROW_DTYPE.itemsize:
        raise CatalogError(f"the row size is {row_size} bytes, expected {ROW_DTYPE.itemsize}")
    if header_size < _HEADER.size:
        raise CatalogError("the header size is smaller than the version 1 header")
    if view.nbytes != header_size + n_stars * row_size:
        raise CatalogError("the file size does not match the number of stars (truncated file?)")
    payload = view[header_size:]
    if (zlib.crc32(payload) & 0xFFFFFFFF) != crc32:
        raise CatalogError("the catalog is damaged (CRC-32 mismatch)")
    rows = np.frombuffer(payload, dtype=ROW_DTYPE, count=n_stars).copy()
    try:
        notes = raw_notes.rstrip(b"\0").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CatalogError("the notes are not valid UTF-8") from exc
    return CapCatalog(
        rows,
        epoch_jyear=epoch,
        cap_ra_deg=cap_ra,
        cap_dec_deg=cap_dec,
        cap_radius_deg=cap_radius,
        gaia_mag_limit=gaia_limit,
        tycho_mag_limit=tycho_limit,
        created_utc_s=created,
        notes=notes,
    )


def write_catalog(path: str | os.PathLike[str], catalog: CapCatalog) -> None:
    """Write a catalog file. The file appears whole or not at all (temporary name, then rename)."""
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(encode_catalog(catalog))
    os.replace(temporary, target)


def load_catalog(path: str | os.PathLike[str]) -> CapCatalog:
    """Read a catalog file. Raises `CatalogError` when the file is invalid."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise CatalogError(f"cannot read the catalog: {exc.strerror or exc}") from exc
    return decode_catalog(data)


def read_info(path: str | os.PathLike[str]) -> CatalogInfo:
    """Read only the header of a catalog file. It does not check the rows."""
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(_HEADER.size)
    except OSError as exc:
        raise CatalogError(f"cannot read the catalog: {exc.strerror or exc}") from exc
    if len(raw) < _HEADER.size:
        raise CatalogError("the file is shorter than the catalog header")
    (
        magic,
        version,
        _row,
        _hdr,
        n_stars,
        crc32,
        epoch,
        ra,
        dec,
        radius,
        gaia,
        tycho,
        created,
        notes,
    ) = _HEADER.unpack(raw)
    if magic != CATALOG_MAGIC:
        raise CatalogError("the file is not a cap catalog (bad magic)")
    return CatalogInfo(
        version=version,
        n_stars=n_stars,
        epoch_jyear=epoch,
        cap_ra_deg=ra,
        cap_dec_deg=dec,
        cap_radius_deg=radius,
        gaia_mag_limit=gaia,
        tycho_mag_limit=tycho,
        created_utc_s=created,
        notes=notes.rstrip(b"\0").decode("utf-8", errors="replace"),
        crc32=crc32,
    )


def describe_flags(flags: int) -> Sequence[str]:
    """The names of the flags that are set, for display."""
    names = {
        FLAG_TYCHO_ONLY: "tycho_only",
        FLAG_TYCHO_V: "tycho_v",
        FLAG_NO_PROPER_MOTION: "no_proper_motion",
        FLAG_NO_PARALLAX: "no_parallax",
        FLAG_NO_COLOR: "no_color",
    }
    return [name for bit, name in names.items() if flags & bit]
