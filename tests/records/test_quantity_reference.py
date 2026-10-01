from __future__ import annotations

import re
from pathlib import Path

import pytest

from seeingmon.records.base import RECORD_TYPES, Record, base_field_specs, field_specs
from seeingmon.records.quantity_reference import COMMAND, OUTPUT_PATH, render_reference
from tests.records.strategies import ALL_RECORD_TYPES, type_id

TEXT = render_reference()


def section(name: str) -> str:
    """The text of the section of a record type, up to the next section."""
    start = TEXT.index(f"\n## `{name}`\n")
    end = TEXT.find("\n## `", start + 1)
    return TEXT[start : end if end != -1 else len(TEXT)]


def test_the_committed_reference_matches_the_declarations(repo_root: Path) -> None:
    committed = (repo_root / OUTPUT_PATH).read_text(encoding="utf-8")
    assert committed == TEXT, f"A declaration changed. Run `{COMMAND} --output {OUTPUT_PATH}`."


def test_the_file_starts_with_a_note_that_names_the_command() -> None:
    first_lines = TEXT.splitlines()[:3]
    assert all(line.startswith(">") for line in first_lines)
    note = " ".join(first_lines)
    assert "generated" in note
    assert f"`{COMMAND}`" in note
    assert "Do not edit" in note


def test_the_text_is_stable_and_clean() -> None:
    assert render_reference() == TEXT
    assert TEXT.endswith("|\n") or TEXT.endswith(".\n")
    assert not TEXT.endswith("\n\n")
    assert "\r" not in TEXT
    for number, line in enumerate(TEXT.splitlines(), start=1):
        assert line == line.rstrip(), f"trailing space on line {number}"


def test_there_is_one_section_for_each_record_type_in_registry_order() -> None:
    headings = re.findall(r"^## `([a-z_]+)`$", TEXT, flags=re.MULTILINE)
    assert headings == list(RECORD_TYPES)


def test_the_common_fields_appear_once() -> None:
    common = TEXT[TEXT.index("## Fields of every record") : TEXT.index("## `frame`")]
    for spec in base_field_specs():
        assert common.count(f"| `{spec.name}` |") == 1
    body = TEXT[TEXT.index("## `frame`") :]
    assert "| `station_id` |" not in body
    assert "| `provenance` |" not in body


@pytest.mark.parametrize("cls", ALL_RECORD_TYPES, ids=type_id)
class TestRecordSection:
    def test_it_lists_every_field_with_its_type_unit_and_definition(
        self, cls: type[Record]
    ) -> None:
        text = section(cls.record_type)
        for spec in field_specs(cls):
            if spec.base:
                continue
            row = next(line for line in text.splitlines() if line.startswith(f"| `{spec.name}` |"))
            cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", row)[1:-1]]
            assert len(cells) == 5, row
            assert cells[1].startswith(f"`{spec.type_name}`")
            assert cells[2] == (f"`{spec.unit}`" if spec.unit else "none")
            assert cells[3] == spec.definition
            assert cells[4] == ("Yes" if spec.nullable else "No")

    def test_it_states_the_storage_and_the_retention(self, cls: type[Record]) -> None:
        text = section(cls.record_type)
        where = "a row of a segment file" if cls.storage == "segment" else "a row of a SQLite table"
        retention = "kept forever" if cls.retention_days is None else f"{cls.retention_days} days"
        assert f"Storage: {where}. Retention: {retention}." in text

    def test_it_describes_the_record(self, cls: type[Record]) -> None:
        first_sentence = (cls.__doc__ or "").strip().split("\n")[0]
        assert first_sentence in section(cls.record_type)

    def test_documented_codes_have_a_table(self, cls: type[Record]) -> None:
        text = section(cls.record_type)
        for spec in field_specs(cls):
            if spec.codes is None:
                continue
            assert f"Codes for `{spec.name}`:" in text
            for code, meaning in spec.codes.items():
                assert f"| `{code}` | {meaning} |" in text


def test_a_segment_record_shows_the_storage_type_of_its_row_fields() -> None:
    text = section("frame")
    assert "| `seq` | `int` (`u4`) |" in text
    assert "| `cx_px` | `float` (`f4`) |" in text
    assert "| `stream_id` | `int` |" in text  # a header field has no storage type
    assert "The type in parentheses" in text
    assert "(`u4`)" not in section("seeing_window")


def test_pipes_in_a_definition_do_not_break_the_table() -> None:
    from seeingmon.records.quantity_reference import _cell

    assert _cell("a | b\nc") == "a \\| b c"
