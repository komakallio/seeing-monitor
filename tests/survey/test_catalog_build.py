"""The catalog build: queries, merging, the asynchronous job, and the solver index.

The tests talk to a local HTTP server (`fake_archive`) and a script that stands in for
`build-astrometry-index`. They never use the real network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from seeingmon.clock import DEFAULT_START_UTC_NS, NS_PER_S, VirtualClock
from seeingmon.solvers import fitsio
from seeingmon.survey import catalog as cat
from seeingmon.survey import catalog_build as build
from seeingmon.survey.apparent import POLARIS
from seeingmon.survey.geometry import angular_separation, radec_to_vector
from tests.survey.fake_archive import (
    GAIA_HEADER,
    TYCHO_HEADER,
    ArchiveScript,
    FakeArchive,
)

SHIM = Path(__file__).with_name("shim_build_index.py")


def csv_value(value: float | int | None) -> str:
    return "" if value is None else repr(value)


GaiaRow = tuple[
    int, float, float, float | None, float | None, float | None, float | None, float | None
]


def gaia_csv(rows: list[GaiaRow]) -> str:
    lines = [GAIA_HEADER]
    lines += [",".join(csv_value(v) for v in row) for row in rows]
    return "\n".join(lines) + "\n"


def tycho_csv(rows: list[tuple[object, ...]]) -> str:
    lines = [TYCHO_HEADER]
    lines += [",".join("" if v is None else str(v) for v in row) for row in rows]
    return "\n".join(lines) + "\n"


# Synthetic stars: the positions are made up, except Polaris. Gaia sources are
# (source_id, ra, dec, pmra, pmdec, parallax, G, BP-RP).
GAIA_ROWS: list[GaiaRow] = [
    (1000001, 10.0, 89.5, 4.5, -4.7, 1.8, 8.9, 1.3),
    (1000002, 112.0, 89.3, -13.7, -20.2, 2.9, 11.4, 0.7),
    (1000003, 223.0, 88.5, 0.0, 0.0, 3.0, 7.9, 1.1),  # Tycho-2 counterpart below
    (1000004, 63.0, 87.0, None, None, None, 12.5, None),  # a two-parameter source
    (1000005, 300.0, 86.0, 1.0, 2.0, 0.5, None, 0.9),  # no G: dropped
]

TYCHO_ROWS: list[tuple[object, ...]] = [
    # Polaris: no mean position, no proper motion, only the observed position.
    (4628, 237, 1, None, None, 37.94661861, 89.26413528, None, None, 2.708, 2.038, 11767),
    # A counterpart of Gaia 1000003, 0.5 arcsec to the north.
    (4600, 100, 1, 223.0, 88.5 + 0.5 / 3600.0, 223.0, 88.5, 0.0, 0.0, 8.5, 7.9, None),
    # A star that Gaia lacks, with a proper motion.
    (4500, 5, 1, 200.0, 88.0, 200.0, 88.0, 10.0, -5.0, 9.1, 8.5, None),
    # A star with an observed position only.
    (4400, 7, 2, None, None, 150.0, 87.5, None, None, 11.0, 10.5, None),
    # A star outside the 15 degree cap.
    (3000, 9, 1, 50.0, 70.0, 50.0, 70.0, 1.0, 1.0, 9.0, 8.5, None),
]


def make_options(archive: FakeArchive | None = None, **changes: float) -> build.BuildOptions:
    values: dict[str, object] = {"poll_interval_s": 10.0, "max_wait_s": 100.0}
    if archive is not None:
        values.update(gaia_url=archive.gaia_url, vizier_url=archive.vizier_url)
    values.update(changes)
    return build.BuildOptions(**values)  # type: ignore[arg-type]


def script() -> ArchiveScript:
    return ArchiveScript(gaia_csv=gaia_csv(GAIA_ROWS), tycho_csv=tycho_csv(TYCHO_ROWS))


# --- Queries ---------------------------------------------------------------------------


def test_the_gaia_query_selects_the_cap_and_the_limit() -> None:
    adql = build.gaia_adql(build.BuildOptions(radius_deg=12.5, gaia_mag_limit=12.0))
    assert "FROM gaiadr3.gaia_source" in adql
    assert "CIRCLE('ICRS', 0.0, 90.0, 12.5)" in adql
    assert "phot_g_mean_mag < 12.0" in adql
    assert "TOP" not in adql  # a limit would truncate the answer


def test_the_tycho_query_uses_the_observed_position_for_the_cone() -> None:
    adql = build.tycho_adql(build.BuildOptions(radius_deg=15.0, tycho_mag_limit=11.0))
    assert 'FROM "I/259/tyc2"' in adql
    assert 'POINT(\'ICRS\', "RA(ICRS)", "DE(ICRS)")' in adql
    assert "CIRCLE('ICRS', 0.0, 90.0, 15.0)" in adql
    assert "VTmag < 11.0" in adql


def test_invalid_options_are_refused() -> None:
    with pytest.raises(ValueError, match="radius"):
        build.BuildOptions(radius_deg=0.0)
    with pytest.raises(ValueError, match="positive"):
        build.BuildOptions(poll_interval_s=0.0)


# --- Parsing -----------------------------------------------------------------------------


def test_gaia_csv_parses_with_missing_values() -> None:
    rows = build.parse_gaia_csv(gaia_csv(GAIA_ROWS))
    assert list(rows.source_id) == [1000001, 1000002, 1000003, 1000004, 1000005]
    assert np.isnan(rows.pm_ra_mas_yr[3])
    assert np.isnan(rows.parallax_mas[3])
    assert np.isnan(rows.g_mag[4])
    assert rows.bp_rp[0] == 1.3


def test_gaia_source_ids_keep_all_their_digits() -> None:
    text = f"{GAIA_HEADER}\n5937141777795168256,1.0,2.0,,,,10.0,\n"
    assert int(build.parse_gaia_csv(text).source_id[0]) == 5937141777795168256


def test_quoted_and_case_different_headers_parse() -> None:
    text = (
        '"SOURCE_ID","RA","DEC","PMRA","PMDEC","PARALLAX","PHOT_G_MEAN_MAG","BP_RP"\n'
        '"1","2.0","3.0","","","","4.0","null"\n'
    )
    rows = build.parse_gaia_csv(text)
    assert rows.ra_deg[0] == 2.0
    assert np.isnan(rows.bp_rp[0])


def test_a_table_without_a_column_is_refused() -> None:
    with pytest.raises(build.CatalogBuildError, match=r"lacks the columns .*parallax"):
        build.parse_gaia_csv("source_id,ra,dec,pmra,pmdec,phot_g_mean_mag,bp_rp\n")
    with pytest.raises(build.CatalogBuildError, match="not a number"):
        build.parse_gaia_csv(f"{GAIA_HEADER}\n1,abc,2.0,,,,10.0,\n")
    with pytest.raises(build.CatalogBuildError, match="empty"):
        build.parse_gaia_csv("")


def test_tycho_csv_falls_back_to_the_observed_position() -> None:
    rows = build.parse_tycho_csv(tycho_csv(TYCHO_ROWS))
    assert rows.ra_deg[0] == 37.94661861  # Polaris: no mean position
    assert rows.ra_deg[1] == 223.0
    assert list(rows.hip) == [11767, 0, 0, 0, 0]
    assert list(rows.tyc[0]) == [4628, 237, 1]
    assert np.isnan(rows.pm_ra_mas_yr[0])


# --- Merging -----------------------------------------------------------------------------


def merged() -> cat.CapCatalog:
    return build.merge_catalogs(
        build.parse_gaia_csv(gaia_csv(GAIA_ROWS)),
        build.parse_tycho_csv(tycho_csv(TYCHO_ROWS)),
        build.BuildOptions(),
        created_utc_s=1_800_000_000,
    )


def row_of(catalog: cat.CapCatalog, source_id: int) -> np.void:
    rows = catalog.rows[catalog.source_id == source_id]
    assert len(rows) == 1
    return rows[0]  # type: ignore[no-any-return]


def test_merge_keeps_gaia_rows_and_adds_unmatched_tycho_stars() -> None:
    catalog = merged()
    # Four Gaia rows (one has no G), plus Polaris, the Gaia-less star, and the observed-only
    # star. The star outside the cap and the matched star add nothing.
    assert len(catalog) == 4 + 3
    assert catalog.created_utc_s == 1_800_000_000
    assert catalog.notes == "gaia-dr3 and tycho-2"
    assert 1000005 not in catalog.source_id
    assert list(catalog.g_mag) == sorted(catalog.g_mag)


def test_a_matched_tycho_star_gives_its_v_magnitude_to_the_gaia_row() -> None:
    row = row_of(merged(), 1000003)
    assert row["flags"] & cat.FLAG_TYCHO_V
    assert not row["flags"] & cat.FLAG_TYCHO_ONLY
    expected = 7.9 - 0.090 * (8.5 - 7.9)
    assert row["v_mag"] == pytest.approx(expected, abs=1e-6)
    assert row["g_mag"] == np.float32(7.9)  # the Gaia photometry stays


def test_a_gaia_row_without_tycho_has_no_v_magnitude() -> None:
    row = row_of(merged(), 1000001)
    assert np.isnan(row["v_mag"])
    assert not row["flags"] & cat.FLAG_TYCHO_V
    two_parameter = row_of(merged(), 1000004)
    assert two_parameter["flags"] & cat.FLAG_NO_PROPER_MOTION
    assert two_parameter["flags"] & cat.FLAG_NO_PARALLAX
    assert two_parameter["flags"] & cat.FLAG_NO_COLOR


def test_a_tycho_only_star_gets_a_negative_id_and_a_g_estimate() -> None:
    catalog = merged()
    row = row_of(catalog, cat.tycho_source_id(4500, 5, 1))
    assert row["flags"] & cat.FLAG_TYCHO_ONLY
    assert row["flags"] & cat.FLAG_TYCHO_V
    assert row["flags"] & cat.FLAG_NO_COLOR
    # V is 8.5 - 0.09 * 0.6 = 8.446, and G is about 0.2 mag brighter for a color of 0.64.
    assert float(row["g_mag"]) == pytest.approx(8.446 - 0.15, abs=0.1)
    # The position moved 16 years along the proper motion: 160 mas east and 80 mas south (the
    # proper motion in right ascension already includes the cosine of the declination).
    moved = float(
        angular_separation(
            radec_to_vector(row["ra_deg"], row["dec_deg"]), radec_to_vector(200.0, 88.0)
        )
    )
    assert np.degrees(moved) * 3600.0 == pytest.approx(np.hypot(0.16, 0.08), rel=0.02)


def test_polaris_takes_its_hipparcos_astrometry() -> None:
    catalog = merged()
    row = row_of(catalog, cat.tycho_source_id(4628, 237, 1))
    assert (row["pm_ra_mas_yr"], row["pm_dec_mas_yr"]) == (np.float32(44.48), np.float32(-11.85))
    assert not row["flags"] & cat.FLAG_NO_PROPER_MOTION
    # J2000 position moved 16 years: 0.71 arcsec in RA* and -0.19 arcsec in Dec.
    separation = float(
        angular_separation(
            radec_to_vector(row["ra_deg"], row["dec_deg"]),
            radec_to_vector(POLARIS.ra_deg, POLARIS.dec_deg),
        )
    )
    assert np.degrees(separation) * 3600.0 == pytest.approx(np.hypot(0.7117, 0.1896), rel=0.01)
    assert 1.5 < float(row["g_mag"]) < 2.2


def test_the_cap_radius_applies_to_the_final_positions() -> None:
    options = build.BuildOptions(radius_deg=2.0)
    catalog = build.merge_catalogs(
        build.parse_gaia_csv(gaia_csv(GAIA_ROWS)),
        build.parse_tycho_csv(tycho_csv(TYCHO_ROWS)),
        options,
    )
    assert np.all(catalog.dec_deg > 88.0 - 1e-6)
    assert catalog.cap_radius_deg == 2.0


def test_a_star_that_two_tycho_stars_share_keeps_the_brighter_v() -> None:
    gaia = build.parse_gaia_csv(gaia_csv([(1, 10.0, 89.0, None, None, None, 9.0, None)]))
    tycho = build.parse_tycho_csv(
        tycho_csv(
            [
                (1, 1, 1, 10.0, 89.0, 10.0, 89.0, 0.0, 0.0, 10.5, 10.0, None),
                (1, 2, 1, 10.0, 89.0 + 1.0 / 3600.0, 10.0, 89.0, 0.0, 0.0, 9.6, 9.5, None),
            ]
        )
    )
    catalog = build.merge_catalogs(gaia, tycho, build.BuildOptions())
    assert len(catalog) == 1
    assert catalog.v_mag[0] == pytest.approx(9.5 - 0.09 * 0.1, abs=1e-5)


def test_propagation_moves_a_star_at_the_pole() -> None:
    vectors = build.propagate(
        np.array([0.0, 0.0]),
        np.array([90.0, 89.9]),
        np.array([100.0, 100.0]),
        np.array([0.0, 0.0]),
        10.0,
    )
    assert np.all(np.isfinite(vectors))
    moved = np.degrees(angular_separation(vectors[0], [0.0, 0.0, 1.0])) * 3600.0
    assert moved == pytest.approx(1.0, rel=1e-6)  # 100 mas/yr for 10 years


def test_v_and_g_estimates_follow_the_published_relations() -> None:
    vt = np.array([8.0])
    bt = np.array([8.8])
    assert build.tycho_v_mag(vt, bt)[0] == pytest.approx(8.0 - 0.09 * 0.8)
    assert build.tycho_v_mag(vt, np.array([np.nan]))[0] == 8.0
    assert build.estimate_g_from_tycho(vt, bt)[0] < 8.0  # G is brighter than V for red stars


# --- The archive protocol ------------------------------------------------------------------


def test_the_build_runs_an_asynchronous_job_and_a_synchronous_query() -> None:
    clock = VirtualClock()
    messages: list[str] = []
    with FakeArchive(script()) as archive:
        catalog = build.build_catalog(make_options(archive), clock, progress=messages.append)
        requests = archive.script.requests
    assert len(catalog) == 7
    # The job request starts the job at once and asks for CSV.
    post = next(form for method, path, form in requests if path == "/gaia/async")
    assert post["PHASE"] == "RUN"
    assert post["FORMAT"] == "csv"
    assert post["LANG"] == "ADQL"
    assert "CONTAINS" in post["QUERY"]
    # The job reported EXECUTING twice, then COMPLETED: three phase checks and two waits.
    assert [path for _, path, _ in requests].count("/gaia/async/job42/phase") == 3
    assert (clock.utc_ns() - DEFAULT_START_UTC_NS) == 20 * NS_PER_S
    assert catalog.created_utc_s == (DEFAULT_START_UTC_NS // NS_PER_S) + 20
    # The Tycho query goes to the synchronous endpoint with a record limit.
    tycho = next(form for method, path, form in requests if path == "/vizier/sync")
    assert tycho["MAXREC"] == "3000000"
    assert '"I/259/tyc2"' in tycho["QUERY"]
    assert any("waiting for the archive job" in message for message in messages)
    assert messages[-1] == "catalog: 7 stars"


def test_a_job_document_without_a_redirect_gives_the_job_id() -> None:
    archive_script = script()
    archive_script.redirect = False
    with FakeArchive(archive_script) as archive:
        catalog = build.build_catalog(make_options(archive), VirtualClock())
    assert len(catalog) == 7


def test_a_failed_job_reports_the_server_message() -> None:
    archive_script = script()
    archive_script.final_phase = "ERROR"
    with (
        FakeArchive(archive_script) as archive,
        pytest.raises(build.CatalogBuildError, match="phase ERROR: ADQL syntax error"),
    ):
        build.build_catalog(make_options(archive), VirtualClock())


def test_a_failed_job_reports_the_message_of_a_uws_error_summary() -> None:
    archive_script = script()
    archive_script.final_phase = "ERROR"
    archive_script.error_text = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<uws:errorSummary xmlns:uws="http://www.ivoa.net/xml/UWS/v1.0" type="fatal" '
        'hasDetail="false"><uws:message>The query ran out of time on the server.</uws:message>'
        "</uws:errorSummary>"
    )
    with (
        FakeArchive(archive_script) as archive,
        pytest.raises(build.CatalogBuildError) as raised,
    ):
        build.build_catalog(make_options(archive), VirtualClock())
    assert str(raised.value).endswith("phase ERROR: The query ran out of time on the server.")


def test_a_job_that_does_not_finish_stops_at_the_wait_limit() -> None:
    archive_script = script()
    archive_script.polls_before_completed = 10_000
    clock = VirtualClock()
    with (
        FakeArchive(archive_script) as archive,
        pytest.raises(build.CatalogBuildError, match="still in phase EXECUTING after 30 s"),
    ):
        build.build_catalog(make_options(archive, max_wait_s=30.0), clock)
    assert (clock.utc_ns() - DEFAULT_START_UTC_NS) / NS_PER_S >= 30.0


def test_a_server_error_is_retried() -> None:
    archive_script = script()
    archive_script.fail_phase_requests = 2
    clock = VirtualClock()
    with FakeArchive(archive_script) as archive:
        catalog = build.build_catalog(make_options(archive), clock)
    assert len(catalog) == 7
    assert archive_script.phase_requests >= 5


def test_a_server_that_keeps_failing_gives_a_clear_error() -> None:
    archive_script = script()
    archive_script.fail_phase_requests = 1000
    with (
        FakeArchive(archive_script) as archive,
        pytest.raises(build.CatalogBuildError, match="cannot reach the service"),
    ):
        build.build_catalog(make_options(archive), VirtualClock())


def test_an_unreachable_service_gives_a_clear_error() -> None:
    with FakeArchive(script()) as archive:
        url = archive.gaia_url
    # The server is closed now, so the port refuses connections. Windows takes about two
    # seconds to refuse each connection, so the test skips the retries.
    client = build.TapClient(VirtualClock(), retries=0)
    with pytest.raises(build.CatalogBuildError, match="cannot reach the service"):
        client.async_query(url, "SELECT 1", poll_interval_s=1.0, max_wait_s=1.0)


def test_a_service_url_must_use_http_or_https() -> None:
    client = build.TapClient(VirtualClock())
    with pytest.raises(build.CatalogBuildError, match="http"):
        client.sync_query("file:///etc/hosts", "SELECT 1", max_records=1)
    with pytest.raises(build.CatalogBuildError, match="http"):
        client.async_query("ftp://example.com/tap", "SELECT 1", poll_interval_s=1.0, max_wait_s=1.0)


def test_saved_answers_replace_the_queries() -> None:
    catalog = build.build_catalog(
        make_options(),
        VirtualClock(),
        gaia_csv=gaia_csv(GAIA_ROWS),
        tycho_csv=tycho_csv(TYCHO_ROWS),
    )
    assert len(catalog) == 7


# --- The solver index ----------------------------------------------------------------------


def shim_command() -> list[str]:
    return [sys.executable, str(SHIM)]


def test_the_index_build_calls_the_tool_once_for_each_preset(tmp_path: Path) -> None:
    catalog = merged()
    written = build.build_solver_index(
        catalog, tmp_path / "index", command=shim_command(), presets=(8, 10), id_base=7_000_000
    )
    assert [path.name for path in written] == ["index-cap-08.fits", "index-cap-10.fits"]
    log = json.loads((tmp_path / "index" / "index-cap-10.fits.json").read_text(encoding="utf-8"))
    assert log == {
        "preset": "10",
        "sort": "MAG",
        "ra": "RA",
        "dec": "DEC",
        "id": "7000010",
        "scan": True,
        "columns": ["DEC", "MAG", "RA"],
        "stars": len(catalog),
        "ra_head": catalog.ra_deg[:3].tolist(),
        "mag_head": [float(np.float32(m)) for m in catalog.g_mag[:3]],
    }
    assert not (tmp_path / "index" / "cap-input.fits").exists()  # the temporary table is gone
    assert fitsio.read_table(written[0])["STARS"][0] == len(catalog)


def test_a_failing_index_tool_raises_with_its_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHIM_FAIL", "1")
    with pytest.raises(build.CatalogBuildError, match=r"preset 8 .*exit code 3.*told to fail"):
        build.build_solver_index(merged(), tmp_path / "index", command=shim_command(), presets=(8,))
    assert not (tmp_path / "index" / "cap-input.fits").exists()


def test_a_missing_index_tool_is_reported(tmp_path: Path) -> None:
    with pytest.raises(build.CatalogBuildError, match="cannot run the index tool"):
        build.build_solver_index(
            merged(), tmp_path / "index", command=[str(tmp_path / "no-such-tool")], presets=(8,)
        )


def test_find_index_tool_looks_for_the_command() -> None:
    assert build.find_index_tool("no-such-tool-seeingmon") is None
    assert build.find_index_tool([]) is None
    found = build.find_index_tool([sys.executable, "script.py"])
    assert found is not None
    assert found[1:] == ["script.py"]
