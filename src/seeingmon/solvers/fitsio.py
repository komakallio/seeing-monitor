"""A small FITS reader and writer for the files that the solver adapters exchange.

astrometry.net reads a star list as a FITS binary table and writes its solution as a FITS
header (`.wcs`) and a binary table (`.corr`). ASTAP reads an image. These files are simple, so
this module reads and writes only what the adapters need and avoids importing `astropy`, which
adds about 100 MB of memory to the survey worker. The tests check every function against
`astropy.io.fits`.

Supported: a primary HDU without data or with a 2-D image (8-bit, 16-bit, or 32-bit float
pixels), and binary table extensions with the column types `D`, `E`, `K`, `J`, `I`, `B`, `L`,
and `A`. A malformed file raises `FitsError`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias

import numpy as np
import numpy.typing as npt

BLOCK = 2880
CARD = 80

HeaderValue: TypeAlias = bool | int | float | str
Header: TypeAlias = dict[str, HeaderValue]


class FitsError(ValueError):
    """The file is not valid FITS, or it uses a feature that this module does not read."""


@dataclass(frozen=True)
class Hdu:
    """One header-data unit. `data` maps column names to arrays for a table, and holds one array
    for an image (under the key `image`), or is empty when the unit has no data."""

    header: Header
    data: dict[str, npt.NDArray[Any]] = field(default_factory=dict)


# --- Writing -----------------------------------------------------------------------------


def _format_value(value: HeaderValue) -> str:
    if isinstance(value, bool):
        return f"{'T' if value else 'F':>20}"
    if isinstance(value, int):
        return f"{value:>20d}"
    if isinstance(value, float):
        if not np.isfinite(value):
            raise FitsError("a header value must be finite")
        text = f"{value:.15G}"
        if len(text) > 20:
            text = f"{value:.12E}"
        if "." not in text and "E" not in text:
            text += ".0"  # a float card must not read as an integer
        return f"{text:>20}"
    text = value.replace("'", "''")
    if not text.isascii() or len(text) > 68:
        raise FitsError("a string value must be ASCII and at most 68 characters")
    return f"'{text:<8}'"


def _card(keyword: str, value: HeaderValue | None = None, comment: str = "") -> bytes:
    if len(keyword) > 8 or not re.fullmatch(r"[A-Z0-9_-]+", keyword):
        raise FitsError(f"invalid keyword {keyword!r}")
    text = f"{keyword:<8}" if value is None else f"{keyword:<8}= {_format_value(value)}"
    if comment:
        text += f" / {comment}"
    if len(text) > CARD:
        raise FitsError(f"the card for {keyword} is longer than {CARD} characters")
    return f"{text:<{CARD}}".encode("ascii")


def _header_bytes(cards: list[tuple[str, HeaderValue]]) -> bytes:
    body = b"".join(_card(key, value) for key, value in cards) + _card("END")
    return body + b" " * (-len(body) % BLOCK)


def _pad(data: bytes) -> bytes:
    return data + b"\0" * (-len(data) % BLOCK)


def _table_format(array: npt.NDArray[Any]) -> tuple[str, str]:
    """The TFORM letter and the big-endian dtype string for a column."""
    kind = (array.dtype.kind, array.dtype.itemsize)
    formats = {
        ("f", 8): ("D", ">f8"),
        ("f", 4): ("E", ">f4"),
        ("i", 8): ("K", ">i8"),
        ("i", 4): ("J", ">i4"),
        ("i", 2): ("I", ">i2"),
        ("u", 1): ("B", "u1"),
    }
    if kind not in formats:
        raise FitsError(f"unsupported column type {array.dtype}")
    return formats[kind]


def table_bytes(
    columns: dict[str, npt.NDArray[Any]], header: Header | None = None, extname: str | None = None
) -> bytes:
    """The bytes of a binary table extension (header and data) without a primary HDU."""
    names = list(columns)
    if not names:
        raise FitsError("a table needs at least one column")
    count = int(np.asarray(columns[names[0]]).shape[0])
    dtype_fields: list[tuple[str, str]] = []
    forms: list[str] = []
    for name in names:
        array = np.asarray(columns[name])
        if array.ndim != 1 or array.shape[0] != count:
            raise FitsError("every column must be 1-D with the same length")
        letter, big_endian = _table_format(array)
        forms.append(f"1{letter}")
        dtype_fields.append((name, big_endian))
    rows = np.zeros(count, dtype=np.dtype(dtype_fields))
    for name in names:
        rows[name] = np.asarray(columns[name])
    cards: list[tuple[str, HeaderValue]] = [
        ("XTENSION", "BINTABLE"),
        ("BITPIX", 8),
        ("NAXIS", 2),
        ("NAXIS1", int(rows.dtype.itemsize)),
        ("NAXIS2", count),
        ("PCOUNT", 0),
        ("GCOUNT", 1),
        ("TFIELDS", len(names)),
    ]
    for index, (name, form) in enumerate(zip(names, forms, strict=True), start=1):
        cards += [(f"TTYPE{index}", name), (f"TFORM{index}", form)]
    if extname is not None:
        cards.append(("EXTNAME", extname))
    cards += list((header or {}).items())
    return _header_bytes(cards) + _pad(rows.tobytes())


def primary_bytes(header: Header | None = None) -> bytes:
    """The bytes of a primary HDU with no data, for a file that holds only tables."""
    cards: list[tuple[str, HeaderValue]] = [
        ("SIMPLE", True),
        ("BITPIX", 8),
        ("NAXIS", 0),
        ("EXTEND", True),
    ]
    cards += list((header or {}).items())
    return _header_bytes(cards)


def write_table(
    path: str | os.PathLike[str],
    columns: dict[str, npt.NDArray[Any]],
    *,
    header: Header | None = None,
) -> None:
    """Write a FITS file with an empty primary HDU and one binary table.

    `header` adds cards to the table header, such as the `IMAGEW` and `IMAGEH` that an
    astrometry.net star list needs.
    """
    Path(path).write_bytes(primary_bytes() + table_bytes(columns, header))


def write_image(
    path: str | os.PathLike[str],
    image: npt.NDArray[Any],
    *,
    header: Header | None = None,
) -> None:
    """Write a 2-D image as the primary HDU. A `uint16` image uses BZERO, as FITS says."""
    Path(path).write_bytes(image_bytes(image, header=header))


def _stored_i2_with_bzero(rows: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return (rows.astype(np.int32) - 32768).astype(">i2")


def _stored_u1(rows: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return rows.astype("u1")


def _stored_f4(rows: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return rows.astype(">f4")


def _image_layout(
    array: npt.NDArray[Any],
) -> tuple[list[tuple[str, HeaderValue]], Callable[[npt.NDArray[Any]], npt.NDArray[Any]]]:
    """The cards that describe an image, and the conversion of rows to the stored type."""
    if array.ndim != 2:
        raise FitsError("an image must be 2-D")
    cards: list[tuple[str, HeaderValue]] = [("SIMPLE", True)]
    if array.dtype == np.uint16:
        convert = _stored_i2_with_bzero
        cards += [("BITPIX", 16)]
        extra: list[tuple[str, HeaderValue]] = [("BZERO", 32768), ("BSCALE", 1)]
    elif array.dtype == np.uint8:
        convert = _stored_u1
        cards += [("BITPIX", 8)]
        extra = []
    elif array.dtype.kind == "f":
        convert = _stored_f4
        cards += [("BITPIX", -32)]
        extra = []
    else:
        raise FitsError(f"unsupported image type {array.dtype}")
    cards += [("NAXIS", 2), ("NAXIS1", array.shape[1]), ("NAXIS2", array.shape[0])]
    return cards + extra, convert


def image_bytes(image: npt.NDArray[Any], *, header: Header | None = None) -> bytes:
    """The bytes of a FITS file with a 2-D image as the primary HDU.

    Use it to write the file in one atomic step, for example with `DataLayout.write_atomic`.
    """
    array = np.asarray(image)
    cards, convert = _image_layout(array)
    cards += list((header or {}).items())
    return _header_bytes(cards) + _pad(convert(array).tobytes())


def write_image_stream(
    handle: BinaryIO,
    image: npt.NDArray[Any],
    *,
    header: Header | None = None,
    chunk_rows: int = 128,
) -> int:
    """Write a FITS file with a 2-D image as the primary HDU to an open binary file.

    The bytes equal `image_bytes`, but the function converts and writes `chunk_rows` rows at a time,
    so a large frame (a 12 megapixel `uint16` image is 23 MB) needs only a small buffer and not
    several copies of itself. Returns the number of bytes written.
    """
    array = np.asarray(image)
    cards, convert = _image_layout(array)
    cards += list((header or {}).items())
    head = _header_bytes(cards)
    handle.write(head)
    data_bytes = 0
    step = max(1, chunk_rows)
    for start in range(0, array.shape[0], step):
        chunk = convert(array[start : start + step]).tobytes()
        handle.write(chunk)
        data_bytes += len(chunk)
    padding = -data_bytes % BLOCK
    handle.write(b"\0" * padding)
    return len(head) + data_bytes + padding


# --- Reading -----------------------------------------------------------------------------

_CARD_VALUE = re.compile(r"^(?P<quoted>'(?:[^']|'')*')|^(?P<plain>[^/]*)")


def _parse_value(text: str) -> HeaderValue | None:
    body = text.strip()
    if not body:
        return None
    match = _CARD_VALUE.match(body)
    if match is None:
        return None
    if match.group("quoted") is not None:
        return match.group("quoted")[1:-1].replace("''", "'").rstrip()
    plain = match.group("plain").strip()
    if plain == "T":
        return True
    if plain == "F":
        return False
    try:
        return int(plain)
    except ValueError:
        pass
    try:
        return float(plain.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def parse_header_text(text: str) -> Header:
    """Parse header cards from text. Use it for a `.wcs` file that is plain text.

    The text can hold 80-character cards with or without line breaks. `COMMENT`, `HISTORY`,
    and blank cards are skipped, and parsing stops at `END`.
    """
    if "\n" not in text.strip("\n") and len(text) >= CARD:
        cards = [text[i : i + CARD] for i in range(0, len(text), CARD)]
    else:
        cards = text.splitlines()
    header: Header = {}
    for raw in cards:
        card = raw.rstrip("\r")
        keyword = card[:8].strip()
        if keyword == "END":
            break
        if not keyword or keyword in {"COMMENT", "HISTORY"} or card[8:10] != "= ":
            continue
        value = _parse_value(card[10:])
        if value is not None:
            header[keyword] = value
    return header


def _read_header(buffer: bytes, offset: int) -> tuple[Header, int]:
    """Parse the header that starts at `offset`. Returns it and the offset after it."""
    cards: list[str] = []
    position = offset
    while True:
        block = buffer[position : position + BLOCK]
        if len(block) < BLOCK:
            raise FitsError("the file ends inside a header")
        position += BLOCK
        text = block.decode("ascii", errors="replace")
        done = False
        for i in range(0, BLOCK, CARD):
            card = text[i : i + CARD]
            if card[:8].strip() == "END":
                done = True
                break
            cards.append(card)
        if done:
            break
    return parse_header_text("\n".join(cards)), position


_TFORM = re.compile(r"^(?P<repeat>\d*)(?P<code>[A-Z])$")
_TABLE_DTYPES = {
    "D": ">f8",
    "E": ">f4",
    "K": ">i8",
    "J": ">i4",
    "I": ">i2",
    "B": "u1",
    "L": "S1",
}


def _int_card(header: Header, key: str, default: int | None = None) -> int:
    value = header.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise FitsError(f"the header has no integer {key}")
    return value


def _read_table(header: Header, payload: bytes) -> dict[str, npt.NDArray[Any]]:
    rows = _int_card(header, "NAXIS2")
    row_bytes = _int_card(header, "NAXIS1")
    fields = _int_card(header, "TFIELDS")
    names: list[str] = []
    codes: dict[str, str] = {}
    dtypes: list[tuple[str, str]] = []
    for index in range(1, fields + 1):
        name = header.get(f"TTYPE{index}")
        form = header.get(f"TFORM{index}")
        if not isinstance(name, str) or not isinstance(form, str):
            raise FitsError(f"column {index} lacks a name or a format")
        match = _TFORM.match(form.strip())
        if match is None:
            raise FitsError(f"unsupported column format {form!r}")
        repeat = int(match.group("repeat") or 1)
        code = match.group("code")
        if code == "A":
            dtypes.append((name, f"S{repeat}"))
        elif code in _TABLE_DTYPES and repeat == 1:
            dtypes.append((name, _TABLE_DTYPES[code]))
        else:
            raise FitsError(f"unsupported column format {form!r}")
        names.append(name)
        codes[name] = code
    dtype = np.dtype(dtypes)
    if dtype.itemsize != row_bytes:
        raise FitsError("the row size does not match the column formats")
    table = np.frombuffer(payload[: rows * row_bytes], dtype=dtype, count=rows)
    result: dict[str, npt.NDArray[Any]] = {}
    for name in names:
        column = table[name]
        if codes[name] == "L":
            result[name] = column == b"T"
        elif codes[name] == "A":
            result[name] = np.char.rstrip(column)
        else:
            result[name] = column.astype(column.dtype.newbyteorder("="))
    return result


def _read_image(header: Header, payload: bytes) -> npt.NDArray[Any]:
    bitpix = _int_card(header, "BITPIX")
    width = _int_card(header, "NAXIS1")
    height = _int_card(header, "NAXIS2")
    dtype = {8: "u1", 16: ">i2", 32: ">i4", -32: ">f4", -64: ">f8"}.get(bitpix)
    if dtype is None:
        raise FitsError(f"unsupported BITPIX {bitpix}")
    raw = np.frombuffer(payload, dtype=dtype, count=width * height).reshape(height, width)
    scale = header.get("BSCALE", 1)
    zero = header.get("BZERO", 0)
    if bitpix == 16 and scale == 1 and zero == 32768:
        return (raw.astype(np.int32) + 32768).astype(np.uint16)
    if scale != 1 or zero != 0:
        return raw.astype(np.float64) * float(scale) + float(zero)
    return raw.astype(raw.dtype.newbyteorder("="))


def _data_size(header: Header) -> int:
    """The size of the data of an HDU in bytes, before the padding."""
    naxis = _int_card(header, "NAXIS", 0)
    size = 0
    if naxis > 0:
        size = abs(_int_card(header, "BITPIX")) // 8
        for axis in range(1, naxis + 1):
            size *= _int_card(header, f"NAXIS{axis}")
    return size + _int_card(header, "PCOUNT", 0)


def _hdu_data(header: Header, payload: bytes) -> dict[str, npt.NDArray[Any]]:
    if header.get("XTENSION") == "BINTABLE":
        return _read_table(header, payload)
    if _int_card(header, "NAXIS", 0) == 2:
        return {"image": _read_image(header, payload)}
    return {}


def read_fits(path: str | os.PathLike[str]) -> list[Hdu]:
    """Read every HDU of a FITS file."""
    buffer = Path(path).read_bytes()
    if not buffer.startswith(b"SIMPLE"):
        raise FitsError("the file is not FITS (it does not start with SIMPLE)")
    units: list[Hdu] = []
    offset = 0
    while offset < len(buffer):
        header, offset = _read_header(buffer, offset)
        size = _data_size(header)
        payload = buffer[offset : offset + size]
        if len(payload) < size:
            raise FitsError("the file ends inside the data")
        offset += size + (-size % BLOCK)
        units.append(Hdu(header, _hdu_data(header, payload)))
    return units


def _read_header_at(handle: BinaryIO, offset: int) -> tuple[Header, int]:
    """Read the header that starts at `offset`. Returns it and the offset after its last block."""
    handle.seek(offset)
    buffer = bytearray()
    while True:
        block = handle.read(BLOCK)
        if len(block) < BLOCK:
            raise FitsError("the file ends inside a header")
        buffer += block
        if offset == 0 and len(buffer) == BLOCK and not buffer.startswith(b"SIMPLE"):
            raise FitsError("the file is not FITS (it does not start with SIMPLE)")
        if any(block[i : i + CARD][:8].strip() == b"END" for i in range(0, BLOCK, CARD)):
            break
    header, _ = _read_header(bytes(buffer), 0)
    return header, offset + len(buffer)


def read_header(path: str | os.PathLike[str]) -> Header:
    """The header of the primary HDU of a FITS file.

    The function reads only the header blocks, so it is cheap for a file that holds a large image.
    """
    with Path(path).open("rb") as handle:
        header, _ = _read_header_at(handle, 0)
    return header


def read_hdu(path: str | os.PathLike[str], index: int = 0) -> Hdu:
    """Read one HDU. The function skips the data of the HDUs before it without reading them."""
    with Path(path).open("rb") as handle:
        size_of_file = os.fstat(handle.fileno()).st_size
        offset = 0
        for position in range(index + 1):
            if offset >= size_of_file:
                raise FitsError(f"the file has no HDU {index}")
            header, data_offset = _read_header_at(handle, offset)
            size = _data_size(header)
            if position == index:
                payload = handle.read(size)
                if len(payload) < size:
                    raise FitsError("the file ends inside the data")
                return Hdu(header, _hdu_data(header, payload))
            offset = data_offset + size + (-size % BLOCK)
    raise FitsError(f"the file has no HDU {index}")  # unreachable: the loop returns


def read_image(path: str | os.PathLike[str]) -> tuple[Header, npt.NDArray[Any]]:
    """The header and the 2-D image of the primary HDU. Raises `FitsError` when it has none."""
    unit = read_hdu(path, 0)
    if "image" not in unit.data:
        raise FitsError("the primary HDU holds no image")
    return unit.header, unit.data["image"]


def read_table(path: str | os.PathLike[str], hdu: int = 1) -> dict[str, npt.NDArray[Any]]:
    """The columns of the binary table in extension `hdu` (the first extension by default).

    The function reads only that extension, so a large image before it costs nothing.
    """
    try:
        unit = read_hdu(path, hdu)
    except FitsError as error:
        if "has no HDU" in str(error):
            raise FitsError(f"HDU {hdu} holds no table") from error
        raise
    if not unit.data:
        raise FitsError(f"HDU {hdu} holds no table")
    return unit.data
