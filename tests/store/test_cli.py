"""`seeingmon store info`: counts, last row IDs, and sink cursors, without private values."""

from __future__ import annotations

import contextlib
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from seeingmon.cli import main
from seeingmon.store.db import Store
from tests.store.builders import NS_PER_S, T0, make_event, make_health, make_window

PRIVATE_STATION = "private-station-name"
PRIVATE_PROFILE = "private-profile-name"
PRIVATE_TEXT = "private event text"


@pytest.fixture
def populated(tmp_path: Path) -> Path:
    path = tmp_path / "results.sqlite"
    with Store.open(path) as store:
        store.write_many(
            [
                make_health(
                    T0 + n * NS_PER_S, station_id=PRIVATE_STATION, profile_id=PRIVATE_PROFILE
                )
                for n in range(5)
            ]
        )
        store.write(make_window(T0 + 7 * NS_PER_S))
        store.write(make_event(T0 + 9 * NS_PER_S, message=PRIVATE_TEXT, station_id=PRIVATE_STATION))
        store.advance_cursor("lab_influx", "health", 3, t_utc_ns=T0 + 60 * NS_PER_S)
        store.advance_cursor("lab_influx", "event", 1)
    return path


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(["store", "info", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def rows(output: str) -> dict[str, list[str]]:
    """Map the first word of each line to its other words, for the lines of the tables."""
    found: dict[str, list[str]] = {}
    for line in output.splitlines():
        words = line.split()
        if words:
            found.setdefault(words[0], words[1:])
    return found


class TestInfo:
    def test_it_prints_the_database_facts(
        self, populated: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = run(capsys, str(populated))
        assert code == 0
        assert "journal mode   wal" in out
        assert "store version  1 (this software writes 1)" in out
        assert "SQLite" in out
        assert "size" in out

    def test_it_prints_the_count_the_last_row_id_and_the_latest_time_of_each_type(
        self, populated: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, out, _ = run(capsys, str(populated))
        table = rows(out)
        assert table["health"][:3] == ["5", "5", "2026-01-01T00:00:04Z"]
        assert table["seeing_window"][:3] == ["1", "1", "2026-01-01T00:00:07Z"]
        assert table["event"][:3] == ["1", "1", "2026-01-01T00:00:09Z"]
        assert table["run"] == ["0", "0", "none"]  # a type without rows
        assert "frame" not in table  # per-frame metrics have no table

    def test_it_prints_each_sink_cursor_with_the_rows_behind_it(
        self, populated: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, out, _ = run(capsys, str(populated))
        lines = [line.split() for line in out.splitlines() if line.startswith("lab_influx")]
        assert sorted(lines) == [
            ["lab_influx", "event", "1", "0", "never"],
            ["lab_influx", "health", "3", "2", "2026-01-01T00:01:00Z"],
        ]

    def test_it_says_so_when_no_sink_has_acknowledged_a_row(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "results.sqlite"
        with Store.open(path):
            pass
        code, out, _ = run(capsys, str(path))
        assert code == 0
        assert "No sink has acknowledged a row yet." in out

    def test_it_prints_no_station_profile_text_or_path(
        self, populated: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, out, err = run(capsys, str(populated))
        for private in (PRIVATE_STATION, PRIVATE_PROFILE, PRIVATE_TEXT, str(populated.parent)):
            assert private not in out + err
        assert populated.name not in out + err

    def test_it_works_while_a_writer_holds_the_database_open(
        self, populated: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with Store.open(populated) as writer:
            writer.write(make_health(T0 + 100 * NS_PER_S))
            code, out, _ = run(capsys, str(populated))
        assert code == 0
        assert rows(out)["health"][0] == "6"

    def test_it_reports_a_cursor_for_a_record_type_that_no_longer_exists(
        self, populated: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with contextlib.closing(sqlite3.connect(populated)) as connection:
            connection.execute(
                'INSERT INTO "sink_cursor" VALUES (?, ?, ?, ?)', ("lab_influx", "retired", 4, None)
            )
            connection.commit()
        code, out, _ = run(capsys, str(populated))
        assert code == 0
        assert "unknown type" in out


class TestErrors:
    def test_a_missing_file_is_an_error_that_does_not_echo_the_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = run(capsys, str(tmp_path / "missing.sqlite"))
        assert code == 1
        assert out == ""
        assert "there is no database file" in err
        assert "missing.sqlite" not in err

    def test_a_file_that_is_not_a_store_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "other.sqlite"
        with contextlib.closing(sqlite3.connect(path)) as connection:
            connection.execute("CREATE TABLE notes (body TEXT)")
            connection.commit()
        code, _, err = run(capsys, str(path))
        assert code == 1
        assert "not a seeing-monitor store" in err

    def test_an_empty_sqlite_file_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "empty.sqlite"
        with contextlib.closing(sqlite3.connect(path)):
            pass
        code, _, err = run(capsys, str(path))
        assert code == 1
        assert err.startswith("seeingmon: error:")

    def test_the_command_needs_a_subcommand_and_a_path(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["store"])
        assert exit_info.value.code == 2
        with pytest.raises(SystemExit):
            main(["store", "info"])


def test_loading_the_command_imports_no_heavy_library() -> None:
    code = (
        "import sys, seeingmon.store.cli; "
        "print([m for m in ('pydantic', 'numpy', 'sqlite3') if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.stdout.strip() == "[]"


def test_importing_the_store_package_is_light() -> None:
    code = (
        "import sys, seeingmon.store; "
        "print([m for m in ('pydantic', 'numpy', 'sqlite3') if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.stdout.strip() == "[]"
