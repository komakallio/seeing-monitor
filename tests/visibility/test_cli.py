"""`seeingmon visibility stats`, against a store with synthetic nightly summaries.

A writer keeps the store open while the command runs, as `core` does.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from seeingmon.cli import main
from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.records.visibility import VisibilitySummaryRecord
from seeingmon.store.db import Store
from seeingmon.store.layout import DataLayout
from seeingmon.survey.nights import night_start_utc_ns


def summary(label: str, **values: Any) -> VisibilitySummaryRecord:
    fields: dict[str, Any] = {
        "station_id": "test",
        "t_utc_ns": night_start_utc_ns(label),
        "profile_id": "test",
        "provenance": {"software": "test"},
        "night": label,
        "visible_hours": 10.0,
        "seeing_hours": 9.5,
        "first_censored": False,
        "last_censored": False,
    }
    fields.update(values)
    return VisibilitySummaryRecord(**fields)


def evening(label: str, clock: str) -> int:
    return iso_to_utc_ns(f"{label}T{clock}:00Z")


class Station:
    """A data directory with a store that a writer keeps open, as `core` does."""

    def __init__(self, folder: Path) -> None:
        self.layout = DataLayout(folder / "data")
        self.layout.create()
        self.store = Store.open(self.layout.db_path)

    def args(self, *more: str) -> list[str]:
        return ["--data-dir", str(self.layout.root), *more]


@pytest.fixture
def station(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Station]:
    monkeypatch.delenv("SEEINGMON_PATHS__DATA_DIR", raising=False)
    opened = Station(tmp_path)
    try:
        yield opened
    finally:
        opened.store.close()


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(["visibility", "stats", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def table(output: str, title: str) -> list[list[str]]:
    """The rows of the table under a title line, split on runs of two or more spaces."""
    lines = output.splitlines()
    index = lines.index(title)
    rows = []
    for line in lines[index + 2 :]:
        if not line.strip():
            break
        rows.append(re.split(r" {2,}", line.strip()))
    return rows


def write_nights(station: Station) -> None:
    station.store.write(
        summary(
            "2026-01-01",
            first_visible_utc_ns=evening("2026-01-01", "15:40"),
            first_visible_sun_deg=-6.1,
            last_visible_utc_ns=evening("2026-01-02", "07:10"),
            last_visible_sun_deg=-6.9,
            dark_utc_ns=evening("2026-01-01", "17:30"),
            dark_sun_deg=-17.5,
            dark_sky_mag_arcsec2=20.9,
            clear_share=1.0,
            transparency_median=0.95,
        )
    )
    station.store.write(
        summary(
            "2026-01-02",
            first_visible_utc_ns=evening("2026-01-02", "20:00"),
            first_visible_sun_deg=-38.0,
            first_censored=True,  # core started at 20:00
            last_visible_utc_ns=evening("2026-01-03", "07:00"),
            last_visible_sun_deg=-7.5,
            flags=["moon"],
        )
    )
    station.store.write(summary("2026-02-01", visible_hours=0.0, seeing_hours=0.0))


class TestTheOutput:
    def test_the_nights_and_the_groups_show_with_the_censored_nights_apart(
        self, station: Station, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_nights(station)
        code, out, err = run(capsys, *station.args())
        assert (code, err) == (0, "")
        assert out.startswith("Visibility of Polaris: 3 nights from 2026-01-01 to 2026-02-01.")
        nights = table(out, "The newest 3 nights:")
        assert nights[0] == [
            "2026-01-01",
            "15:40",
            "-6.1",
            "07:10",
            "-6.9",
            "10.00",
            "9.50",
            "-17.5",
            "1.00",
            "0.95",
            "-",
        ]
        assert nights[1][:5] == ["2026-01-02", "20:00*", "-38.0", "07:00", "-7.5"]
        assert nights[1][-1] == "moon"
        assert nights[2][:5] == ["2026-02-01", "-", "-", "-", "-"]

        months = table(out, "By month:")
        # month, nights, unseen, then n, median, range, censored, bounds for the first and the last
        assert months[0] == [
            "2026-01", "2", "0",
            "1", "-6.1", "-6.1", "1", "-38.0",
            "2", "-7.2", "-7.5 to -6.9", "0", "-",
        ]  # fmt: skip
        assert months[1] == ["2026-02", "1", "1", "0", "-", "-", "0", "-", "0", "-", "-", "0", "-"]

        bins = table(out, "By the median transparency of the clear verdict:")
        assert [row[0] for row in bins] == [">= 0.9", "unknown"]
        assert [row[1] for row in bins] == ["1", "2"]

    def test_the_options_select_the_nights_and_the_bins(
        self, station: Station, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_nights(station)
        code, out, _ = run(
            capsys,
            *station.args(
                "--from", "2026-01-02", "--to", "2026-01-31", "--nights", "0",
                "--transparency-bins", "0.5",
            ),
        )  # fmt: skip
        assert code == 0
        assert "1 night from 2026-01-02 to 2026-01-02" in out
        assert "The newest" not in out
        assert [row[0] for row in table(out, "By month:")] == ["2026-01"]

    @pytest.mark.parametrize("form", ["20260102", "2026-W01-5"])  # Python 3.11 reads both
    def test_the_other_iso_forms_of_a_night_select_the_same_nights(
        self, station: Station, capsys: pytest.CaptureFixture[str], form: str
    ) -> None:
        write_nights(station)
        _, expected, _ = run(capsys, *station.args("--from", "2026-01-02"))
        code, out, _ = run(capsys, *station.args("--from", form))
        assert code == 0
        assert "2 nights from 2026-01-02 to 2026-02-01" in out
        assert out == expected

    def test_a_corrected_night_shows_once_with_its_newest_revision(
        self, station: Station, capsys: pytest.CaptureFixture[str]
    ) -> None:
        station.store.write(summary("2026-01-01", visible_hours=1.0))
        station.store.write_correction(summary("2026-01-01", visible_hours=2.0, revision=1))
        code, out, _ = run(capsys, *station.args())
        assert code == 0
        assert table(out, "The newest night:")[0][5] == "2.00"

    def test_a_detection_without_the_suns_elevation_is_accounted_for(
        self, station: Station, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_nights(station)
        code, out, _ = run(capsys, *station.args())
        assert code == 0
        assert "without a Sun elevation" not in out  # every detection of these nights has one
        station.store.write(
            summary(
                "2026-01-05",
                first_visible_utc_ns=evening("2026-01-05", "15:45"),  # an unsynchronized clock
                last_visible_utc_ns=evening("2026-01-06", "07:05"),
                last_visible_sun_deg=-6.5,
            )
        )
        code, out, _ = run(capsys, *station.args())
        assert code == 0
        month = table(out, "By month:")[0]
        assert month[:4] == ["2026-01", "3", "0", "1"]  # 3 nights, but only 1 measured first
        assert month[6] == "1"  # and 1 censored: the third night is in the note
        note = out.split("By month:", 1)[1].split("By the median", 1)[0]
        assert "not in the n columns: 2026-01: 1 first and 0 last." in " ".join(note.split())


class TestNoData:
    def test_a_store_without_summaries_says_so(
        self, station: Station, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = run(capsys, *station.args())
        assert code == 0
        assert out.startswith("No visibility summaries yet")

    def test_a_store_from_before_the_visibility_summary_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        layout = DataLayout(tmp_path / "old")
        layout.create()
        Store.open(layout.db_path).close()
        with closing(sqlite3.connect(layout.db_path)) as db:
            db.execute("DROP TABLE visibility_summary")
            db.commit()
        code, out, err = run(capsys, "--data-dir", str(layout.root))
        assert (code, err) == (0, "")
        assert out.startswith("No visibility summaries yet")

    def test_a_data_directory_without_a_store_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = run(capsys, "--data-dir", str(tmp_path / "nowhere"))
        assert code == 1
        assert "no store database" in err

    @pytest.mark.parametrize("bins", ["0.9,0.6", "a,b", "0.5,nan"])
    def test_bins_that_do_not_rise_are_a_usage_error(
        self, station: Station, capsys: pytest.CaptureFixture[str], bins: str
    ) -> None:
        code, _, err = run(capsys, *station.args("--transparency-bins", bins))
        assert code == 2
        assert "--transparency-bins" in err

    def test_a_night_label_is_checked(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as stopped:
            main(["visibility", "stats", "--from", "2026-13-01"])
        assert stopped.value.code == 2
        assert "night label" in capsys.readouterr().err

    def test_the_command_has_a_help_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main(["visibility", "stats", "--help"])
        assert "censored" in capsys.readouterr().out


class TestTheDataDirectory:
    def test_the_local_configuration_names_the_data_directory(
        self, station: Station, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        write_nights(station)
        local = tmp_path / "local.toml"
        local.write_text(f'[paths]\ndata_dir = "{station.layout.root.as_posix()}"\n', "utf-8")
        code, out, err = run(capsys, "--local-config", str(local))  # no --data-dir
        assert (code, err) == (0, "")
        assert out.startswith("Visibility of Polaris: 3 nights")

    def test_without_a_data_directory_the_command_says_what_to_pass(
        self, station: Station, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        code, _, err = run(capsys, "--local-config", str(tmp_path / "absent.toml"))
        assert code == 1
        assert err.startswith("seeingmon: error: pass --data-dir, or set data_dir in [paths]")


def test_the_summaries_of_a_week_come_back_in_order_across_reads(
    station: Station, monkeypatch: pytest.MonkeyPatch
) -> None:
    from seeingmon.visibility import cli

    monkeypatch.setattr(cli, "_BATCH", 2)  # four reads of the store
    for day in (3, 1, 7, 2, 6, 4, 5):  # written out of order, as a correction run could
        station.store.write(summary(f"2026-01-0{day}"))
    found = cli.read_summaries(station.layout.root)
    assert [r.night for r in found] == [f"2026-01-0{day}" for day in range(1, 8)]
    assert found[0].t_utc_ns == night_start_utc_ns("2026-01-01")
    assert found[1].t_utc_ns - found[0].t_utc_ns == 24 * 3600 * NS_PER_S
