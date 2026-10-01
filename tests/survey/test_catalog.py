"""The cap catalog: columns, the file format, and the cone query."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from seeingmon.survey import catalog as cat
from seeingmon.survey.geometry import angular_separation, radec_to_vector


def random_catalog(count: int, seed: int = 0, *, radius_deg: float = 15.0) -> cat.CapCatalog:
    """A synthetic catalog (random positions in a cap around the pole, not real stars)."""
    rng = np.random.default_rng(seed)
    cos_polar = rng.uniform(np.cos(np.radians(radius_deg)), 1.0, count)
    dec = 90.0 - np.degrees(np.arccos(cos_polar))
    ra = rng.uniform(0.0, 360.0, count)
    return cat.CapCatalog.from_columns(
        source_id=rng.integers(1, 2**62, count),
        ra_deg=ra,
        dec_deg=dec,
        g_mag=rng.uniform(4.0, 13.0, count),
        pm_ra_mas_yr=rng.normal(0.0, 30.0, count),
        pm_dec_mas_yr=rng.normal(0.0, 30.0, count),
        parallax_mas=rng.uniform(0.1, 10.0, count),
        bp_rp=rng.uniform(-0.2, 3.0, count),
        cap_radius_deg=radius_deg,
    )


def test_a_row_takes_fifty_bytes() -> None:
    assert cat.ROW_DTYPE.itemsize == 50
    # The research notes expect 82,055 stars (G < 13, 15 degree cap) in about 4 MB.
    assert 82_055 * cat.ROW_DTYPE.itemsize < 4.5e6


def test_rows_sort_by_g_magnitude_and_keep_their_columns_together() -> None:
    catalog = cat.CapCatalog.from_columns(
        source_id=[10, 20, 30],
        ra_deg=[1.0, 2.0, 3.0],
        dec_deg=[89.0, 88.0, 87.0],
        g_mag=[9.0, 5.0, 7.0],
    )
    assert list(catalog.g_mag) == [5.0, 7.0, 9.0]
    assert list(catalog.source_id) == [20, 30, 10]
    assert list(catalog.ra_deg) == [2.0, 3.0, 1.0]


def test_missing_columns_get_defaults_and_flags() -> None:
    catalog = cat.CapCatalog.from_columns(
        source_id=[1, 2],
        ra_deg=[10.0, 20.0],
        dec_deg=[89.0, 88.0],
        g_mag=[8.0, 9.0],
        pm_ra_mas_yr=[5.0, np.nan],
        pm_dec_mas_yr=[6.0, 7.0],
        parallax_mas=None,
        bp_rp=[1.0, np.nan],
        v_mag=[np.nan, 9.5],
    )
    first, second = catalog.rows
    assert (first["pm_ra_mas_yr"], first["pm_dec_mas_yr"]) == (5.0, 6.0)
    assert (second["pm_ra_mas_yr"], second["pm_dec_mas_yr"]) == (0.0, 0.0)  # one NaN: unknown
    assert second["flags"] & cat.FLAG_NO_PROPER_MOTION
    assert not first["flags"] & cat.FLAG_NO_PROPER_MOTION
    assert first["flags"] & cat.FLAG_NO_PARALLAX
    assert second["flags"] & cat.FLAG_NO_PARALLAX
    assert second["flags"] & cat.FLAG_NO_COLOR
    assert np.isnan(second["bp_rp"])
    assert second["flags"] & cat.FLAG_TYCHO_V
    assert not first["flags"] & cat.FLAG_TYCHO_V
    assert list(cat.describe_flags(int(second["flags"]))) == [
        "tycho_v",
        "no_proper_motion",
        "no_parallax",
        "no_color",
    ]


def test_invalid_positions_are_refused() -> None:
    with pytest.raises(cat.CatalogError, match="finite"):
        cat.CapCatalog.from_columns(source_id=[1], ra_deg=[1.0], dec_deg=[91.0], g_mag=[5.0])
    with pytest.raises(cat.CatalogError, match="finite"):
        cat.CapCatalog.from_columns(source_id=[1], ra_deg=[np.nan], dec_deg=[89.0], g_mag=[5.0])


def test_the_catalog_is_read_only() -> None:
    catalog = random_catalog(10)
    with pytest.raises(ValueError, match="read-only"):
        catalog.rows["ra_deg"][0] = 1.0


def test_a_catalog_survives_a_round_trip_through_a_file(tmp_path: Path) -> None:
    catalog = random_catalog(500, seed=1)
    path = tmp_path / "cap.smcat"
    cat.write_catalog(path, catalog)
    assert not path.with_name(path.name + ".tmp").exists()  # the temporary file is gone
    loaded = cat.load_catalog(path)
    assert loaded.rows.tobytes() == catalog.rows.tobytes()
    assert loaded.info() == catalog.info()
    assert loaded.content_id == catalog.content_id
    assert len(loaded.content_id) == 8


def test_the_header_carries_the_build_parameters(tmp_path: Path) -> None:
    catalog = cat.CapCatalog.from_columns(
        source_id=[1],
        ra_deg=[0.0],
        dec_deg=[90.0],
        g_mag=[2.0],
        cap_radius_deg=10.0,
        gaia_mag_limit=12.0,
        tycho_mag_limit=11.0,
        created_utc_s=1_800_000_000,
        notes="gaia-dr3 and tycho-2",
    )
    path = tmp_path / "cap.smcat"
    cat.write_catalog(path, catalog)
    info = cat.read_info(path)
    assert (info.version, info.n_stars) == (1, 1)
    assert (info.cap_ra_deg, info.cap_dec_deg, info.cap_radius_deg) == (0.0, 90.0, 10.0)
    assert (info.gaia_mag_limit, info.tycho_mag_limit) == (12.0, 11.0)
    assert info.created_utc_s == 1_800_000_000
    assert info.notes == "gaia-dr3 and tycho-2"
    assert info.epoch_jyear == 2016.0


def test_notes_must_fit_in_the_header() -> None:
    with pytest.raises(cat.CatalogError, match="notes"):
        cat.CapCatalog.from_columns(
            source_id=[1], ra_deg=[0.0], dec_deg=[90.0], g_mag=[2.0], notes="x" * 65
        )


def test_the_content_id_changes_when_a_star_changes() -> None:
    a = random_catalog(50, seed=2)
    b = random_catalog(50, seed=3)
    assert a.content_id != b.content_id
    assert a.content_id == random_catalog(50, seed=2).content_id


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda data: b"NOTACAT!" + data[8:], "bad magic"),
        (lambda data: data[:-1], "truncated"),
        (lambda data: data[:10], "shorter"),
        (lambda data: data[:-60] + bytes(b ^ 0xFF for b in data[-60:]), "CRC-32"),
        (lambda data: data[:8] + (99).to_bytes(2, "little") + data[10:], "version"),
        (lambda data: data[:10] + (51).to_bytes(2, "little") + data[12:], "row size"),
    ],
)
def test_a_damaged_file_is_refused(damage: Callable[[bytes], bytes], message: str) -> None:
    data = cat.encode_catalog(random_catalog(100, seed=4))
    with pytest.raises(cat.CatalogError, match=message):
        cat.decode_catalog(damage(data))


def test_a_missing_file_is_a_catalog_error(tmp_path: Path) -> None:
    with pytest.raises(cat.CatalogError, match="cannot read"):
        cat.load_catalog(tmp_path / "missing.smcat")
    with pytest.raises(cat.CatalogError, match="cannot read"):
        cat.read_info(tmp_path / "missing.smcat")


@pytest.mark.parametrize(
    ("center", "radius_deg"),
    [
        ((0.0, 90.0), 0.5),  # the pole
        ((0.0, 90.0), 15.0),
        ((10.0, 89.3), 1.5),  # near Polaris
        ((359.9, 88.0), 3.0),  # across the right ascension wrap
        ((200.0, 80.0), 10.0),
    ],
)
def test_cone_query_matches_a_brute_force_search(
    center: tuple[float, float], radius_deg: float
) -> None:
    catalog = random_catalog(3000, seed=5)
    found = catalog.cone(center, radius_deg)
    separation = np.degrees(
        angular_separation(catalog.vectors, radec_to_vector(center[0], center[1]))
    )
    expected = np.flatnonzero(separation <= radius_deg)
    # The boundary can differ by rounding, so compare the stars that are clearly in or out.
    assert set(np.flatnonzero(separation < radius_deg - 1e-9)) <= set(found)
    assert set(found) <= set(expected) | set(np.flatnonzero(separation < radius_deg + 1e-9))
    assert list(found) == sorted(found)  # rows sort by G, so ascending row means brightest first


def test_cone_query_accepts_a_vector_and_a_magnitude_limit() -> None:
    catalog = random_catalog(2000, seed=6)
    by_vector = catalog.cone([0.0, 0.0, 1.0], 5.0)
    by_angles = catalog.cone((123.0, 90.0), 5.0)
    assert list(by_vector) == list(by_angles)
    bright = catalog.cone([0.0, 0.0, 1.0], 5.0, max_g_mag=9.0)
    assert len(bright) < len(by_vector)
    assert np.all(catalog.g_mag[bright] < 9.0)
    assert list(catalog.brighter_than(9.0)) == list(np.flatnonzero(catalog.g_mag < 9.0))
    with pytest.raises(ValueError, match="center"):
        catalog.cone([1.0] * 4, 1.0)


def test_tycho_only_stars_get_a_negative_source_id() -> None:
    assert cat.tycho_source_id(4628, 237, 1) == -(4_628_000_000 + 2_370 + 1)
    assert cat.tycho_source_id(1, 2, 3) < 0
    assert (
        len({cat.tycho_source_id(a, b, c) for a in (1, 2) for b in (1, 12121) for c in (1, 2, 3)})
        == 12
    )
