from __future__ import annotations

import contextlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from seeingmon.cli import main
from seeingmon.records.api_schema import api_schema
from seeingmon.records.base import RECORD_TYPES, field_specs
from seeingmon.records.quantity_reference import render_reference
from seeingmon.records.sqlite_schema import create_table_sql, ensure_schema, schema_sql


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def make_database(path: Path, *statements: str) -> Path:
    with contextlib.closing(sqlite3.connect(path)) as connection:
        for statement in statements:
            connection.execute(statement)
        connection.commit()
    return path


class TestList:
    def test_it_lists_every_record_type(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(["records", "list"], capsys)
        assert code == 0
        lines = out.splitlines()
        assert [line.split()[0] for line in lines] == list(RECORD_TYPES)
        frame = next(line for line in lines if line.startswith("frame "))
        count = str(len(field_specs("frame")))
        assert frame.split() == ["frame", "segment", "7", "days", count, "fields"]
        window = next(line for line in lines if line.startswith("seeing_window"))
        assert window.split()[:3] == ["seeing_window", "table", "forever"]

    def test_the_columns_line_up(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out, _ = run(["records", "list"], capsys)
        starts = {line.index("table") for line in out.splitlines() if " table " in line}
        assert len(starts) == 1


class TestSqliteSchema:
    def test_it_prints_a_script_that_creates_every_table(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = run(["records", "sqlite-schema"], capsys)
        assert code == 0
        assert out == schema_sql()
        with contextlib.closing(sqlite3.connect(":memory:")) as db:
            db.executescript(out)
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"seeing_window", "event", "run"} <= tables
        assert "frame" not in tables

    def test_an_empty_database_gets_the_whole_schema(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        database = make_database(tmp_path / "empty.sqlite")
        code, out, _ = run(["records", "sqlite-schema", "--database", str(database)], capsys)
        assert code == 0
        assert out.count("CREATE TABLE") == 10
        assert out.rstrip().endswith(";")

    def test_a_current_database_needs_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "current.sqlite"
        with contextlib.closing(sqlite3.connect(path)) as db:
            ensure_schema(db)
            db.commit()
        code, out, _ = run(["records", "sqlite-schema", "--database", str(path)], capsys)
        assert (code, out.strip()) == (0, "-- The database is up to date.")

    def test_an_older_database_gets_the_statements_that_it_lacks(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        old = create_table_sql("event").replace(
            '    "detail" TEXT CHECK (json_valid("detail")),\n', ""
        )
        assert old != create_table_sql("event")
        path = make_database(tmp_path / "older.sqlite", old)
        with contextlib.closing(sqlite3.connect(path)) as db:  # every other table is current
            ensure_schema(db, [name for name in RECORD_TYPES if name not in ("event", "frame")])
            db.commit()
        code, out, _ = run(["records", "sqlite-schema", "--database", str(path)], capsys)
        assert code == 0
        assert out.strip() == (
            'ALTER TABLE "event" ADD COLUMN "detail" TEXT CHECK (json_valid("detail"));'
        )
        with contextlib.closing(sqlite3.connect(path)) as db:  # the command ran nothing
            columns = [r[1] for r in db.execute('PRAGMA table_info("event")')]
        assert "detail" not in columns

    def test_a_database_that_cannot_be_migrated_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        retyped = create_table_sql("event").replace('"message" TEXT', '"message" INTEGER')
        path = make_database(tmp_path / "retyped.sqlite", retyped)
        code, out, err = run(["records", "sqlite-schema", "--database", str(path)], capsys)
        assert code == 1
        assert out == ""
        assert "cannot migrate event" in err
        assert "retyped" in err

    def test_a_missing_file_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = run(
            ["records", "sqlite-schema", "--database", str(tmp_path / "none.sqlite")], capsys
        )
        assert code == 1
        assert "there is no database file" in err

    def test_a_file_that_is_not_a_database_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "text.sqlite"
        path.write_text("this is not a database, " * 20, encoding="utf-8")
        code, _, err = run(["records", "sqlite-schema", "--database", str(path)], capsys)
        assert code == 1
        assert "cannot read the database" in err


class TestApiSchema:
    def test_it_prints_the_components_as_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(["records", "api-schema"], capsys)
        assert code == 0
        assert json.loads(out) == api_schema()

    def test_it_can_select_record_types(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out, _ = run(["records", "api-schema", "--record", "event", "--record", "run"], capsys)
        assert list(json.loads(out)["components"]["schemas"]) == ["Quality", "Event", "Run"]

    def test_an_unknown_record_type_is_an_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, err = run(["records", "api-schema", "--record", "no_such_record"], capsys)
        assert code == 1
        assert out == ""
        assert "unknown record type 'no_such_record'" in err
        assert "seeing_window" in err


class TestReference:
    def test_it_prints_the_reference(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(["records", "reference"], capsys)
        assert code == 0
        assert out == render_reference()

    def test_it_writes_a_file_with_unix_line_endings(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "quantities.md"
        code, out, _ = run(["records", "reference", "--output", str(target)], capsys)
        assert code == 0
        assert "Wrote" in out
        data = target.read_bytes()
        assert b"\r" not in data
        assert data.decode("utf-8") == render_reference()

    def test_check_passes_for_a_current_file_and_fails_for_a_changed_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "quantities.md"
        run(["records", "reference", "--output", str(target)], capsys)
        code, out, _ = run(["records", "reference", "--check", str(target)], capsys)
        assert (code, "up to date" in out) == (0, True)
        target.write_text(target.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
        code, _, err = run(["records", "reference", "--check", str(target)], capsys)
        assert code == 1
        assert "out of date" in err
        assert "seeingmon records reference --output" in err

    def test_check_fails_for_a_missing_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = run(["records", "reference", "--check", str(tmp_path / "none.md")], capsys)
        assert code == 1
        assert "out of date" in err

    def test_check_accepts_the_committed_reference(
        self, repo_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = run(
            ["records", "reference", "--check", str(repo_root / "docs" / "quantities.md")], capsys
        )
        assert (code, "up to date" in out) == (0, True)

    def test_output_and_check_exclude_each_other(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["records", "reference", "--output", "a.md", "--check", "b.md"])
        assert exit_info.value.code == 2
        assert "not allowed with" in capsys.readouterr().err

    def test_the_text_is_plain_ascii_so_every_console_prints_it(self) -> None:
        assert render_reference().isascii()


class TestUsage:
    def test_a_subcommand_is_required(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["records"])
        assert exit_info.value.code == 2
        assert "required" in capsys.readouterr().err

    def test_help_names_the_subcommands(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["records", "--help"])
        assert exit_info.value.code == 0
        out = capsys.readouterr().out
        for name in ("list", "sqlite-schema", "api-schema", "reference"):
            assert name in out

    def test_the_top_level_help_lists_the_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main(["--help"])
        assert "records" in capsys.readouterr().out

    def test_importing_the_command_module_does_not_load_the_heavy_libraries(self) -> None:
        code = (
            "import sys\n"
            "import seeingmon.records.cli\n"
            "import seeingmon.records\n"
            "heavy = [m for m in ('pydantic', 'numpy', 'sqlite3') if m in sys.modules]\n"
            "print(','.join(heavy))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
        )
        assert result.stdout.strip() == ""
