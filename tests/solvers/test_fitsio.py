"""The small FITS reader and writer, checked against `astropy.io.fits` in both directions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seeingmon.solvers import fitsio

fits = pytest.importorskip("astropy.io.fits")


def sample_columns() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "X": rng.uniform(1.0, 4000.0, 20),
        "Y": rng.uniform(1.0, 2800.0, 20),
        "FLUX": rng.uniform(1.0, 1e6, 20).astype(np.float32),
        "ID": np.arange(20, dtype=np.int64) + 2**40,
        "SMALL": np.arange(20, dtype=np.int32) - 10,
        "TINY": np.arange(20, dtype=np.int16),
        "BYTE": np.arange(20, dtype=np.uint8),
    }


def test_a_table_written_here_reads_in_astropy(tmp_path: Path) -> None:
    path = tmp_path / "stars.xyls"
    columns = sample_columns()
    fitsio.write_table(path, columns, header={"IMAGEW": 4144, "IMAGEH": 2822})
    with fits.open(path) as hdus:
        assert len(hdus) == 2
        assert hdus[1].header["IMAGEW"] == 4144
        assert hdus[1].header["IMAGEH"] == 2822
        for name, values in columns.items():
            np.testing.assert_array_equal(hdus[1].data[name], values)


def test_a_table_written_by_astropy_reads_here(tmp_path: Path) -> None:
    path = tmp_path / "corr.fits"
    columns = sample_columns()
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name=name.lower(), array=values, format=letter)
            for (name, values), letter in zip(
                columns.items(), ["D", "D", "E", "K", "J", "I", "B"], strict=True
            )
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    read = fitsio.read_table(path)
    assert list(read) == [name.lower() for name in columns]
    for name, values in columns.items():
        np.testing.assert_array_equal(read[name.lower()], values)


def test_a_header_with_every_value_type_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "wcs.fits"
    path.write_bytes(
        fitsio.primary_bytes(
            {
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRVAL1": 37.9546,
                "CRVAL2": -89.5,
                "CD1_1": -1.0611e-3,
                "CD2_2": 1.0611e-3,
                "CD1_2": 3.0,
                "NSTARS": 123,
                "SOLVED": True,
                "NOTE": "it's ok",
            }
        )
    )
    mine = fitsio.read_header(path)
    with fits.open(path) as hdus:
        theirs = hdus[0].header
        for key in ("CTYPE1", "CRVAL1", "CRVAL2", "CD1_1", "CD2_2", "CD1_2", "NSTARS", "SOLVED"):
            assert mine[key] == theirs[key], key
        assert mine["NOTE"] == theirs["NOTE"] == "it's ok"
    assert isinstance(mine["CD1_2"], float)  # 3.0 stays a float
    assert isinstance(mine["NSTARS"], int)


def test_a_float_survives_with_full_precision(tmp_path: Path) -> None:
    value = 0.1234567890123456
    path = tmp_path / "x.fits"
    path.write_bytes(fitsio.primary_bytes({"VALUE": value}))
    assert fitsio.read_header(path)["VALUE"] == pytest.approx(value, rel=1e-14)


@pytest.mark.parametrize("dtype", [np.uint16, np.uint8, np.float32])
def test_an_image_round_trips_through_astropy(tmp_path: Path, dtype: type) -> None:
    rng = np.random.default_rng(1)
    image: np.ndarray = rng.integers(0, 200, (30, 40)).astype(dtype)
    path = tmp_path / "image.fits"
    fitsio.write_image(path, image)
    with fits.open(path) as hdus:
        np.testing.assert_array_equal(hdus[0].data, image)
        assert hdus[0].data.shape == (30, 40)
    back = fitsio.read_fits(path)[0].data["image"]
    np.testing.assert_array_equal(back, image)
    assert back.dtype == image.dtype


def test_an_astropy_image_with_scaling_reads_here(tmp_path: Path) -> None:
    image = np.arange(12, dtype=np.uint16).reshape(3, 4) + 40000
    path = tmp_path / "u16.fits"
    fits.PrimaryHDU(image).writeto(path)
    np.testing.assert_array_equal(fitsio.read_fits(path)[0].data["image"], image)


def test_a_header_parses_from_plain_text_with_or_without_line_breaks() -> None:
    cards = [
        f"{'SIMPLE':<8}= {'T':>20}".ljust(80),
        f"{'CRVAL1':<8}= {37.5:>20}".ljust(80),
        f"{'CTYPE1':<8}= 'RA---TAN'".ljust(80),
        "COMMENT a comment".ljust(80),
        "END".ljust(80),
    ]
    for text in ("\n".join(cards), "".join(cards), "\r\n".join(cards)):
        header = fitsio.parse_header_text(text)
        assert header == {"SIMPLE": True, "CRVAL1": 37.5, "CTYPE1": "RA---TAN"}


def test_a_value_with_a_slash_in_its_comment_parses() -> None:
    header = fitsio.parse_header_text("CRPIX1  =               2072.5 / pixel / reference\nEND")
    assert header == {"CRPIX1": 2072.5}


def test_fortran_exponents_parse() -> None:
    assert fitsio.parse_header_text("CD1_1   =        -1.0611D-03\nEND")["CD1_1"] == -1.0611e-3


@pytest.mark.parametrize(
    "damage",
    [
        lambda data: b"NOTFITS" + data[7:],
        lambda data: data[:2000],
        lambda data: data[: 2880 + 2880 + 5],
    ],
)
def test_a_damaged_file_raises_fits_error(tmp_path: Path, damage: object) -> None:
    path = tmp_path / "t.fits"
    fitsio.write_table(path, {"X": np.arange(10.0)})
    data = path.read_bytes()
    assert callable(damage)
    path.write_bytes(damage(data))
    with pytest.raises(fitsio.FitsError):
        fitsio.read_fits(path)


def test_invalid_input_to_the_writer_is_refused(tmp_path: Path) -> None:
    with pytest.raises(fitsio.FitsError, match="at least one column"):
        fitsio.table_bytes({})
    with pytest.raises(fitsio.FitsError, match="same length"):
        fitsio.table_bytes({"A": np.zeros(3), "B": np.zeros(4)})
    with pytest.raises(fitsio.FitsError, match="unsupported column"):
        fitsio.table_bytes({"A": np.zeros(3, dtype=np.complex128)})
    with pytest.raises(fitsio.FitsError, match="invalid keyword"):
        fitsio.primary_bytes({"lower": 1})
    with pytest.raises(fitsio.FitsError, match="finite"):
        fitsio.primary_bytes({"V": float("nan")})
    with pytest.raises(fitsio.FitsError, match="2-D"):
        fitsio.write_image(tmp_path / "x.fits", np.zeros(5))


def test_read_table_refuses_a_file_without_a_table(tmp_path: Path) -> None:
    path = tmp_path / "header.fits"
    path.write_bytes(fitsio.primary_bytes())
    with pytest.raises(fitsio.FitsError, match="no table"):
        fitsio.read_table(path)


def test_image_bytes_equal_the_file_that_write_image_makes(tmp_path: Path) -> None:
    image = np.arange(24, dtype=np.uint16).reshape(4, 6) * 1000
    header: fitsio.Header = {"EXPTIME": 30.0, "MODE": "bin2", "NFRAMES": 9}
    path = tmp_path / "image.fits"
    fitsio.write_image(path, image, header=header)
    assert path.read_bytes() == fitsio.image_bytes(image, header=header)


def test_read_image_gives_the_header_and_the_pixels(tmp_path: Path) -> None:
    image = (np.arange(35, dtype=np.float32) / 7.0).reshape(5, 7)
    path = tmp_path / "image.fits"
    fitsio.write_image(path, image, header={"SENSTEMP": 18.4, "KIND": "dark"})
    header, back = fitsio.read_image(path)
    assert header["SENSTEMP"] == pytest.approx(18.4)
    assert header["KIND"] == "dark"
    np.testing.assert_array_equal(back, image)
    empty = tmp_path / "empty.fits"
    empty.write_bytes(fitsio.primary_bytes())
    with pytest.raises(fitsio.FitsError, match="no image"):
        fitsio.read_image(empty)


def test_read_header_does_not_need_the_data(tmp_path: Path) -> None:
    """A file that holds a large image gives its header from the first blocks alone."""
    image = np.zeros((300, 300), dtype=np.float32)
    path = tmp_path / "big.fits"
    data = fitsio.image_bytes(image, header={"EXPTIME": 30.0})
    path.write_bytes(data[: fitsio.BLOCK])  # the header only: the data blocks are missing
    assert fitsio.read_header(path)["EXPTIME"] == 30.0
    with pytest.raises(fitsio.FitsError, match="ends inside the data"):
        fitsio.read_fits(path)
    path.write_bytes(data[:100])  # a header that ends before END
    with pytest.raises(fitsio.FitsError, match="inside a header"):
        fitsio.read_header(path)
    path.write_bytes(b"x" * fitsio.BLOCK)
    with pytest.raises(fitsio.FitsError, match="not FITS"):
        fitsio.read_header(path)
