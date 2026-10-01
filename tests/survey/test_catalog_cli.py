"""The `seeingmon catalog` commands, against the fake archive and the index tool shim."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from seeingmon.cli import main
from seeingmon.survey import catalog as cat
from tests.survey.fake_archive import ArchiveScript, FakeArchive
from tests.survey.test_catalog_build import GAIA_ROWS, TYCHO_ROWS, gaia_csv, tycho_csv

SHIM = Path(__file__).with_name("shim_build_index.py")


def shim_command() -> str:
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(SHIM))}"


def fast_script() -> ArchiveScript:
    return ArchiveScript(
        gaia_csv=gaia_csv(GAIA_ROWS), tycho_csv=tycho_csv(TYCHO_ROWS), polls_before_completed=1
    )


def test_build_writes_the_catalog_and_the_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "cap.smcat"
    with FakeArchive(fast_script()) as archive:
        code = main(
            [
                "catalog",
                "build",
                "--output",
                str(output),
                "--gaia-url",
                archive.gaia_url,
                "--vizier-url",
                archive.vizier_url,
                "--poll-interval",
                "0.01",
                "--index-dir",
                str(tmp_path / "idx"),
                "--index-presets",
                "8,9",
                "--build-index-command",
                shim_command(),
            ]
        )
    assert code == 0
    catalog = cat.load_catalog(output)
    assert len(catalog) == 7
    assert sorted(path.name for path in (tmp_path / "idx").glob("index-*.fits")) == [
        "index-cap-08.fits",
        "index-cap-09.fits",
    ]
    out = capsys.readouterr().out
    assert "querying Gaia DR3" in out
    assert "wrote 7 stars" in out
    assert f"catalog id {catalog.content_id}" in out
    assert "solver index: 2 files" in out


def test_build_puts_the_index_next_to_the_catalog_by_default(tmp_path: Path) -> None:
    code = main(
        [
            "catalog",
            "build",
            "--output",
            str(tmp_path / "cap.smcat"),
            "--gaia-csv",
            str(write(tmp_path / "gaia.csv", gaia_csv(GAIA_ROWS))),
            "--tycho-csv",
            str(write(tmp_path / "tycho.csv", tycho_csv(TYCHO_ROWS))),
            "--index-presets",
            "10",
            "--build-index-command",
            shim_command(),
        ]
    )
    assert code == 0
    assert (tmp_path / "index" / "index-cap-10.fits").is_file()


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_build_skips_the_index_when_the_tool_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "catalog",
            "build",
            "--output",
            str(tmp_path / "cap.smcat"),
            "--gaia-csv",
            str(write(tmp_path / "gaia.csv", gaia_csv(GAIA_ROWS))),
            "--tycho-csv",
            str(write(tmp_path / "tycho.csv", tycho_csv(TYCHO_ROWS))),
            "--build-index-command",
            "no-such-tool-seeingmon",
        ]
    )
    assert code == 0
    assert "skipped, because the index tool is not installed" in capsys.readouterr().out
    assert (tmp_path / "cap.smcat").is_file()
    assert not (tmp_path / "index").exists()


def test_build_can_skip_the_index_on_request(tmp_path: Path) -> None:
    code = main(
        [
            "catalog",
            "build",
            "--no-index",
            "--output",
            str(tmp_path / "cap.smcat"),
            "--gaia-csv",
            str(write(tmp_path / "gaia.csv", gaia_csv(GAIA_ROWS))),
            "--tycho-csv",
            str(write(tmp_path / "tycho.csv", tycho_csv(TYCHO_ROWS))),
            "--build-index-command",
            shim_command(),
        ]
    )
    assert code == 0
    assert not (tmp_path / "index").exists()


def test_build_reads_the_output_path_from_the_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "configured" / "cap.smcat"
    monkeypatch.setenv("SEEINGMON_SURVEY__CATALOG_PATH", str(target))
    code = main(
        [
            "catalog",
            "build",
            "--no-index",
            "--gaia-csv",
            str(write(tmp_path / "gaia.csv", gaia_csv(GAIA_ROWS))),
            "--tycho-csv",
            str(write(tmp_path / "tycho.csv", tycho_csv(TYCHO_ROWS))),
        ]
    )
    assert code == 0
    assert len(cat.load_catalog(target)) == 7


def test_build_needs_an_output_path_from_somewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SEEINGMON_SURVEY__CATALOG_PATH", "")
    code = main(["catalog", "build", "--gaia-csv", str(tmp_path / "x.csv")])
    assert code == 1
    assert "catalog_path" in capsys.readouterr().err


def test_a_failed_archive_job_is_an_error_message_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = fast_script()
    script.final_phase = "ERROR"
    with FakeArchive(script) as archive:
        code = main(
            [
                "catalog",
                "build",
                "--output",
                str(tmp_path / "cap.smcat"),
                "--gaia-url",
                archive.gaia_url,
                "--vizier-url",
                archive.vizier_url,
                "--poll-interval",
                "0.01",
            ]
        )
    assert code == 1
    assert "seeingmon: error: the job ended in phase ERROR" in capsys.readouterr().err
    assert not (tmp_path / "cap.smcat").exists()


def test_build_reports_a_missing_csv_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "catalog",
            "build",
            "--output",
            str(tmp_path / "c"),
            "--gaia-csv",
            str(tmp_path / "no.csv"),
        ]
    )
    assert code == 1
    assert "cannot read a CSV file" in capsys.readouterr().err


@pytest.mark.parametrize("presets", ["", "8,x", "99", "-9"])
def test_invalid_presets_are_refused(
    presets: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["catalog", "build", "--output", str(tmp_path / "c"), "--index-presets", presets])
    assert code == 1
    assert "--index-presets" in capsys.readouterr().err


def test_invalid_options_are_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["catalog", "build", "--output", str(tmp_path / "c"), "--radius-deg", "0"])
    assert code == 1
    assert "radius" in capsys.readouterr().err


def test_info_prints_the_header(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from seeingmon.clock import VirtualClock
    from seeingmon.survey.catalog_build import BuildOptions, build_catalog

    catalog = build_catalog(
        BuildOptions(),
        VirtualClock(),
        gaia_csv=gaia_csv(GAIA_ROWS),
        tycho_csv=tycho_csv(TYCHO_ROWS),
    )
    path = tmp_path / "cap.smcat"
    cat.write_catalog(path, catalog)
    assert main(["catalog", "info", str(path)]) == 0
    out = capsys.readouterr().out
    assert "catalog id" in out
    assert catalog.content_id in out
    assert "stars" in out
    assert " 7" in out
    assert "J2016" in out
    assert "15 degrees around (0, 90)" in out
    assert "tycho_only 3" in out


def test_info_refuses_a_damaged_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.smcat"
    path.write_bytes(b"not a catalog at all, not even close to a header of any kind" * 3)
    assert main(["catalog", "info", str(path)]) == 1
    assert "not a cap catalog" in capsys.readouterr().err
    assert main(["catalog", "info", str(tmp_path / "missing.smcat")]) == 1


def test_the_catalog_command_needs_a_subcommand() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["catalog"])
    assert raised.value.code == 2
