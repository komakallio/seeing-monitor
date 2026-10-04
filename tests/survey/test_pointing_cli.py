"""`seeingmon pointing set-reference` and `show`, against a synthetic store.

The store holds the `pointing` records of a camera that is rigid on the Earth, as `core` writes
them: the attitude in the frame of date turns with the Earth, and the Earth-fixed attitude stays.
A writer keeps the store open while each command runs, as `core` does. The tests replace the clock
of the command, so that the records of a fixed night count as recent.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pytest

from seeingmon.cli import main
from seeingmon.store.db import Store
from seeingmon.store.layout import DataLayout
from seeingmon.survey import apparent, pointing_cli
from seeingmon.survey import pointing as pt
from tests.survey import synth
from tests.survey.pointfx import (
    BIN2,
    CENTER,
    HOUR_NS,
    MINUTE_NS,
    PROFILE,
    SCALE_RAD,
    STATION,
    T0,
    Made,
    made,
    mount,
    tilted,
)

RESTART = "restart core"


class Station:
    """A data directory with a store that a writer keeps open, as `core` does."""

    def __init__(self, folder: Path) -> None:
        self.layout = DataLayout(folder / "data")
        self.layout.create()
        self.store = Store.open(self.layout.db_path)
        self.now_utc_ns = T0 + 10 * MINUTE_NS

    @property
    def data_dir(self) -> Path:
        return self.layout.root

    @property
    def reference_file(self) -> Path:
        """Where the command writes by default."""
        return self.layout.root / "calibration" / "pointing-reference.json"

    def write(self, *items: Made) -> None:
        for item in items:
            self.store.write(item.record)

    def args(self, *more: str) -> list[str]:
        """The arguments that name this data directory."""
        return ["--data-dir", str(self.data_dir), *more]


@pytest.fixture
def station(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Station]:
    for name in (
        "SEEINGMON_PATHS__DATA_DIR",
        "SEEINGMON_SURVEY__CALIBRATION_DIR",
        "SEEINGMON_SURVEY__POINTING__REFERENCE_FILE",
        "SEEINGMON_SURVEY__DUT1_S",
    ):
        monkeypatch.delenv(name, raising=False)
    opened = Station(tmp_path)
    monkeypatch.setattr(pointing_cli, "_now_utc_ns", lambda: opened.now_utc_ns)
    try:
        yield opened
    finally:
        opened.store.close()


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(["pointing", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def rows(output: str) -> dict[str, str]:
    """The label and the value of each line that has two or more spaces between them."""
    found: dict[str, str] = {}
    for line in output.splitlines():
        match = re.fullmatch(r"(\S.*?) {2,}(\S.*)", line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def toml_line(path: Path) -> str:
    return f'reference_file = "{path.as_posix()}"'


def load(path: Path) -> pt.ReferenceSolution:
    reference = pt.load_reference(path)
    assert reference is not None
    return reference


# --- set-reference: which solution --------------------------------------------------------------


def test_the_newest_good_solution_becomes_the_reference_even_when_poor_ones_are_newer(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    good = made(T0, n_matched=300, rms_arcsec=0.4)
    older = made(T0 - 5 * MINUTE_NS, n_matched=250)
    station.write(
        older,
        good,
        made(T0 + 3 * MINUTE_NS, n_matched=40),  # a poor solution: too few stars
        made(T0 + 6 * MINUTE_NS, solved=False),  # no solution at all
    )
    code, _, err = run(capsys, "set-reference", *station.args())
    assert code == 0
    assert err == ""
    reference = load(station.reference_file)
    assert reference.reference_id == "reference-20261001T220000Z"
    solution = reference.solution
    assert solution.t_utc_ns == T0
    assert (solution.n_matched, solution.rms_arcsec, solution.solver) == (
        300,
        0.4,
        "astrometry.net",
    )
    assert solution.mode == "bin2"
    assert (solution.width_px, solution.height_px) == (BIN2.width_px, BIN2.height_px)
    assert solution.center_px == CENTER
    assert solution.parity == 1
    assert solution.scale_rad_px == pytest.approx(SCALE_RAD, rel=1e-12)
    assert solution.dut1_s == 0.0
    np.testing.assert_allclose(
        solution.rotation_earth_fixed, good.solution.rotation_earth_fixed, atol=1e-12
    )
    assert not list(station.reference_file.parent.glob("*.tmp"))  # the write was atomic


def test_the_summary_and_the_configuration_line(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.write(made(T0, rotation_tirs=mount(0.4)))
    code, out, _ = run(capsys, "set-reference", *station.args())
    assert code == 0
    table = rows(out)
    assert table["reference ID"] == "reference-20261001T220000Z"
    assert table["solution time"] == "2026-10-01T22:00:00Z (10 min ago)"
    assert table["matched stars"] == "300"
    assert table["residual"] == "0.40 arcsec rms"
    assert table["roll"] == "25.00 degrees"  # the roll that make_attitude was given
    assert table["plate scale"] == "3.820 arcsec/px in bin2"
    assert table["center from the pole"] == "0.400 degrees"
    assert table["file"] == str(station.reference_file)
    lines = out.splitlines()
    assert lines[0] == "Saved the pointing reference."
    assert "[survey.pointing]" in lines
    assert toml_line(station.reference_file) in lines
    assert lines.index(toml_line(station.reference_file)) == lines.index("[survey.pointing]") + 1
    assert RESTART in out
    assert STATION not in out
    assert PROFILE.id not in out


def test_the_reference_gives_zero_offset_for_the_same_mount_and_the_move_for_a_moved_one(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    rotation = mount()
    station.write(made(T0, rotation_tirs=rotation))
    assert run(capsys, "set-reference", *station.args())[0] == 0
    reference = load(station.reference_file)
    later = T0 + 5 * HOUR_NS  # the Earth has turned, and the attitude in the frame of date with it
    same = made(later, rotation_tirs=rotation, reference=reference).record
    assert same.reference_id == "reference-20261001T220000Z"
    assert same.offset_arcmin == pytest.approx(0.0, abs=1e-6)
    assert same.flags == []
    assert same.quality is None  # the note "no reference solution" is gone
    moved = made(later, rotation_tirs=tilted(rotation, 2.0), reference=reference).record
    assert moved.offset_arcmin == pytest.approx(2.0, rel=1e-6)
    assert moved.flags == []
    far = made(later, rotation_tirs=tilted(rotation, 10.0), reference=reference).record
    assert far.offset_arcmin == pytest.approx(10.0, rel=1e-6)
    assert far.flags == ["moved"]  # over the 5 arcmin of moved_arcmin
    before = made(later, rotation_tirs=rotation).record
    assert before.offset_arcmin is None  # the record without a reference
    assert before.quality == {"offset_arcmin": "no reference solution"}


@pytest.mark.parametrize("parity", [1, -1])
def test_the_saved_solution_is_the_one_that_the_fit_made_whatever_the_parity(
    station: Station, capsys: pytest.CaptureFixture[str], parity: int
) -> None:
    item = made(T0, parity=parity)
    station.write(item)
    assert run(capsys, "set-reference", *station.args())[0] == 0
    solution = load(station.reference_file).solution
    assert (
        solution.parity == parity
    )  # the record holds no parity: the command reads it from Polaris
    record = item.record
    assert record.polaris_x_px is not None
    assert record.polaris_y_px is not None
    pixel = solution.polaris_pixel(T0)
    assert pixel == pytest.approx((record.polaris_x_px, record.polaris_y_px), abs=1e-6)
    projected = solution.attitude_at(T0)
    original = item.solution.attitude_at(T0)
    np.testing.assert_allclose(projected.rotation, original.rotation, atol=1e-12)
    assert projected.roll_deg() == pytest.approx(record.roll_deg, abs=1e-9)


def test_a_roll_that_cannot_tell_the_parity_is_still_a_faithful_reference(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    # The boresight on the pole: the roll is undefined, and the pole is at the center. Polaris
    # still tells the parity.
    station.write(made(T0, rotation_tirs=mount(0.0), parity=-1))
    code, out, _ = run(capsys, "set-reference", *station.args())
    assert code == 0
    assert load(station.reference_file).solution.parity == -1
    assert rows(out)["roll"] == "undefined (the pole is at the center)"


# --- set-reference: no solution that fits -------------------------------------------------------


def test_an_empty_store_gives_a_one_line_reason_and_writes_nothing(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(capsys, "set-reference", *station.args())
    assert code == 1
    assert out == ""
    assert err == "seeingmon: error: the store holds no pointing record\n"
    assert not station.reference_file.parent.exists()


def test_a_store_without_a_solution_says_so(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.write(*(made(T0 + n * MINUTE_NS, solved=False) for n in range(3)))
    code, _, err = run(capsys, "set-reference", *station.args())
    assert code == 1
    assert err == ("seeingmon: error: no pointing record of the last 60 minutes has a solution\n")
    assert not station.reference_file.exists()


def test_a_solution_that_is_too_old_is_refused_until_you_widen_the_limit(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.write(made(T0))
    station.now_utc_ns = T0 + 3 * HOUR_NS
    code, _, err = run(capsys, "set-reference", *station.args())
    assert code == 1
    assert err == (
        "seeingmon: error: the newest pointing record is 3.0 h old, and the limit is 60 minutes "
        "(see --max-age-min)\n"
    )
    assert not station.reference_file.exists()
    assert run(capsys, "set-reference", *station.args("--max-age-min", "240"))[0] == 0
    assert load(station.reference_file).solution.t_utc_ns == T0


def test_a_solution_that_is_older_than_the_limit_is_skipped_for_a_younger_poor_one(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.write(made(T0 - 2 * HOUR_NS), made(T0 + 5 * MINUTE_NS, n_matched=20))
    code, _, err = run(capsys, "set-reference", *station.args())
    assert code == 1
    assert "has 100 matched stars: the best has 20" in err  # the old record is outside the window


def test_a_solution_with_too_few_stars_is_refused_until_you_lower_the_limit(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.write(made(T0, n_matched=60), made(T0 + MINUTE_NS, n_matched=45))
    code, _, err = run(capsys, "set-reference", *station.args())
    assert code == 1
    assert err == (
        "seeingmon: error: no pointing solution of the last 60 minutes has 100 matched stars: "
        "the best has 60 (see --min-matched)\n"
    )
    assert run(capsys, "set-reference", *station.args("--min-matched", "60"))[0] == 0
    assert load(station.reference_file).solution.n_matched == 60


def test_the_star_limit_counts_a_solution_with_exactly_that_many_stars(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.write(made(T0, n_matched=100))
    assert run(capsys, "set-reference", *station.args())[0] == 0


def test_a_solution_from_an_unsynchronized_clock_does_not_serve(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.write(made(T0, time_invalid=True))
    code, _, err = run(capsys, "set-reference", *station.args())
    assert code == 1
    assert err == (
        "seeingmon: error: every pointing solution of the last 60 minutes has the time_invalid "
        "flag\n"
    )


def test_a_solution_without_a_residual_does_not_serve(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.write(made(T0, rms_arcsec=None))
    code, _, err = run(capsys, "set-reference", *station.args())
    assert code == 1
    assert "has a finite residual" in err


# --- set-reference: the file --------------------------------------------------------------------


def test_an_existing_file_stays_unless_you_add_force(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.write(made(T0))
    station.reference_file.parent.mkdir(parents=True)
    station.reference_file.write_text("keep this", encoding="utf-8")
    code, out, err = run(capsys, "set-reference", *station.args())
    assert code == 1
    assert out == ""
    assert (
        err
        == "seeingmon: error: pointing-reference.json already exists: add --force to replace it\n"
    )
    assert station.reference_file.read_text(encoding="utf-8") == "keep this"
    code, out, _ = run(capsys, "set-reference", *station.args("--force"))
    assert code == 0
    assert load(station.reference_file).reference_id == "reference-20261001T220000Z"


def test_the_file_is_checked_before_the_store(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.reference_file.parent.mkdir(parents=True)
    station.reference_file.write_text("keep this", encoding="utf-8")
    code, _, err = run(capsys, "set-reference", *station.args())  # the store has no record
    assert code == 1
    assert "already exists" in err


def test_out_and_id_name_the_file_and_the_reference(
    station: Station, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    station.write(made(T0))
    target = tmp_path / "elsewhere" / "ref.json"
    code, out, _ = run(
        capsys, "set-reference", *station.args("--out", str(target), "--id", "commissioning-1")
    )
    assert code == 0
    assert load(target).reference_id == "commissioning-1"
    assert not station.reference_file.exists()
    assert rows(out)["reference ID"] == "commissioning-1"
    assert toml_line(target) in out.splitlines()


def test_a_relative_out_becomes_an_absolute_path_in_the_configuration_line(
    station: Station,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    station.write(made(T0))
    monkeypatch.chdir(tmp_path)
    code, out, _ = run(capsys, "set-reference", *station.args("--out", "ref.json"))
    assert code == 0
    assert toml_line(tmp_path / "ref.json") in out.splitlines()


def test_a_windows_path_goes_into_the_line_with_forward_slashes_and_as_a_toml_string() -> None:
    windows = PureWindowsPath("C:\\data folder\\calibration\\pointing-reference.json")
    assert pointing_cli._toml_line(windows) == (
        'reference_file = "C:/data folder/calibration/pointing-reference.json"'
    )
    assert pointing_cli._toml_line(PurePosixPath("/data/ref.json")) == (
        'reference_file = "/data/ref.json"'
    )
    assert pointing_cli._toml_line(PurePosixPath('/odd"name.json')) == (
        'reference_file = "/odd\\"name.json"'
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--id", "has a space"], "--id takes 1 to 64 letters"),
        (["--id=-starts-with-dash"], "--id takes 1 to 64 letters"),
        (["--id", "x" * 65], "--id takes 1 to 64 letters"),
        (["--min-matched", "0"], "--min-matched must be at least 1"),
        (["--max-age-min", "0"], "--max-age-min must be a number of minutes above 0"),
        (["--max-age-min", "-5"], "--max-age-min must be a number of minutes above 0"),
        (["--max-age-min", "nan"], "--max-age-min must be a number of minutes above 0"),
        (["--max-age-min", "inf"], "--max-age-min must be a number of minutes above 0"),
    ],
)
def test_a_bad_option_is_a_usage_error_before_anything_is_read(
    station: Station, capsys: pytest.CaptureFixture[str], options: list[str], message: str
) -> None:
    station.write(made(T0))
    code, out, err = run(capsys, "set-reference", *station.args(*options))
    assert code == 2
    assert out == ""
    assert message in err
    assert not station.reference_file.exists()


def test_a_directory_in_the_way_is_an_error_message(
    station: Station, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    station.write(made(T0))
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a folder should be", encoding="utf-8")
    code, _, err = run(capsys, "set-reference", *station.args("--out", str(blocker / "ref.json")))
    assert code == 1
    assert err.startswith("seeingmon: error: cannot write ref.json: ")
    assert "Traceback" not in err


# --- set-reference: the configuration -----------------------------------------------------------


def test_the_data_directory_and_the_file_follow_the_configuration(
    station: Station,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    station.write(made(T0))
    monkeypatch.setenv("SEEINGMON_PATHS__DATA_DIR", str(station.data_dir))
    code, _, _ = run(capsys, "set-reference")  # no --data-dir: [paths] data_dir
    assert code == 0
    assert station.reference_file.is_file()  # calibration/ of the data directory
    calibration = tmp_path / "calibration-elsewhere"
    monkeypatch.setenv("SEEINGMON_SURVEY__CALIBRATION_DIR", str(calibration))
    code, out, _ = run(capsys, "set-reference")
    assert code == 0
    assert (calibration / "pointing-reference.json").is_file()  # calibration_dir of [survey]
    assert toml_line(calibration / "pointing-reference.json") in out.splitlines()


def test_a_configured_reference_file_is_the_default_target_and_needs_no_new_line(
    station: Station,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    station.write(made(T0))
    mine = tmp_path / "mine.json"
    monkeypatch.setenv("SEEINGMON_SURVEY__POINTING__REFERENCE_FILE", str(mine))
    code, out, _ = run(capsys, "set-reference", *station.args())
    assert code == 0
    assert load(mine).reference_id == "reference-20261001T220000Z"
    assert not station.reference_file.exists()
    assert "[survey.pointing] reference_file already names this file." in out
    assert "Restart core to load the new reference." in out
    assert "Add this" not in out
    assert "reference_file =" not in out
    # A named file elsewhere is not the configured one, so the command says what to set.
    other = tmp_path / "other.json"
    code, out, _ = run(capsys, "set-reference", *station.args("--out", str(other)))
    assert code == 0
    assert toml_line(other) in out.splitlines()
    assert "already names" not in out


def test_the_local_configuration_file_can_name_the_reference(
    station: Station, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    station.write(made(T0))
    mine = tmp_path / "from-file.json"
    local = tmp_path / "local.toml"
    local.write_text(f'[survey.pointing]\nreference_file = "{mine.as_posix()}"\n', encoding="utf-8")
    code, _, _ = run(capsys, "set-reference", *station.args("--local-config", str(local)))
    assert code == 0
    assert mine.is_file()


def test_dut1_comes_from_the_survey_table(
    station: Station, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    station.write(made(T0))
    monkeypatch.setenv("SEEINGMON_SURVEY__DUT1_S", "0.4")
    assert run(capsys, "set-reference", *station.args())[0] == 0
    solution = load(station.reference_file).solution
    assert solution.dut1_s == 0.4
    # The Earth rotation angle moved by 0.4 s, so the Earth-fixed attitude turned by 6 arcsec about
    # the pole, as the pipeline would compute it.
    plain = made(T0).solution
    shifted = pt.PointingSolution.from_attitude(
        plain.attitude_at(T0),
        apparent.epoch_from_utc_ns(T0, 0.4),
        mode="bin2",
        width_px=BIN2.width_px,
        height_px=BIN2.height_px,
    )
    np.testing.assert_allclose(
        solution.rotation_earth_fixed, shifted.rotation_earth_fixed, atol=1e-12
    )


def test_without_a_data_directory_the_command_says_what_to_set(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = (
        "seeingmon: error: pass --data-dir, or set data_dir in [paths] of local/config.toml\n"
    )
    code, _, err = run(capsys, "set-reference")
    assert (code, err) == (1, expected)
    code, _, err = run(capsys, "set-reference", "--out", str(station.data_dir / "x.json"))
    assert (code, err) == (1, expected)


def test_a_folder_without_a_store_is_an_error_message(
    station: Station, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, _, err = run(capsys, "set-reference", "--data-dir", str(tmp_path / "nothing-here"))
    assert code == 1
    assert err == "seeingmon: error: there is no store database in the data directory\n"
    assert str(tmp_path) not in err


def test_a_file_that_is_no_store_is_an_error_message(
    station: Station, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    folder = DataLayout(tmp_path / "odd")
    folder.db_dir.mkdir(parents=True)
    folder.db_path.write_text("this is not a database", encoding="utf-8")
    code, _, err = run(capsys, "set-reference", "--data-dir", str(folder.root))
    assert code == 1
    assert err.startswith("seeingmon: error: cannot read the store: ")
    assert "Traceback" not in err


def test_the_store_stays_open_for_its_writer_while_the_command_reads(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.write(made(T0))
    assert run(capsys, "set-reference", *station.args())[0] == 0
    # The writer goes on: the command held no lock, and it opened the store read-only.
    station.write(made(T0 + MINUTE_NS))
    assert station.store.count("pointing") == 2
    assert run(capsys, "set-reference", *station.args("--force"))[0] == 0
    assert load(station.reference_file).solution.t_utc_ns == T0 + MINUTE_NS  # the new record counts


WRITER = """
import json
import sys

from seeingmon.records.survey import PointingRecord
from seeingmon.store.db import Store

with Store.open(sys.argv[1]) as store:
    store.write(PointingRecord.from_row(json.loads(sys.argv[2])))
    print("ready", flush=True)
    sys.stdin.readline()
"""


def test_the_command_reads_a_store_that_a_writer_in_another_process_holds_open(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    """`core` is the only writer, and it keeps the store open in WAL mode.

    The record that it wrote last is still in the write-ahead log, and the command sees it.
    """
    station.write(made(T0, n_matched=300))
    station.store.close()  # the process below is the only writer from now on
    newest = made(T0 + 4 * MINUTE_NS, n_matched=420)
    writer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            WRITER,
            str(station.layout.db_path),
            json.dumps(newest.record.to_row()),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert writer.stdout is not None
        assert writer.stdout.readline().strip() == "ready"
        code, _, err = run(capsys, "set-reference", *station.args())
        assert (code, err) == (0, "")
        assert load(station.reference_file).solution.n_matched == 420  # the newest record counts
    finally:
        try:
            writer.communicate("\n", timeout=60)
        except subprocess.TimeoutExpired:
            writer.kill()
            writer.communicate()
    assert writer.returncode == 0


def test_the_output_and_the_file_name_no_station_and_no_profile(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.write(made(T0))
    code, out, err = run(capsys, "set-reference", *station.args())
    assert code == 0
    text = out + err + station.reference_file.read_text(encoding="utf-8")
    for private in (STATION, PROFILE.id):
        assert private not in text


# --- A record that does not fit the profile -----------------------------------------------------


def test_a_record_of_another_profile_is_refused(
    station: Station, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    station.write(made(T0))
    real = pointing_cli._load_context

    def small_profile(args: argparse.Namespace) -> pointing_cli.PointingContext:
        context = real(args)
        return pointing_cli.PointingContext(
            synth.cropped_profile(1200, 800),
            context.survey,
            context.data_dir,
            context.calibration_dir,
        )

    monkeypatch.setattr(pointing_cli, "_load_context", small_profile)
    code, _, err = run(capsys, "set-reference", *station.args())
    assert code == 1
    assert err == (
        "seeingmon: error: the pointing record does not fit the profile of the configuration: "
        "the model that the profile gives puts Polaris away from the position in the record\n"
    )
    assert not station.reference_file.exists()


# --- show ---------------------------------------------------------------------------------------


def test_show_prints_the_reference_in_the_form_of_the_summary(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.write(made(T0, rotation_tirs=mount(0.4)))
    _, saved, _ = run(capsys, "set-reference", *station.args("--id", "commissioning-1"))
    code, out, err = run(capsys, "show", "--file", str(station.reference_file))
    assert code == 0
    assert err == ""
    shown = rows(out)
    assert shown["reference ID"] == "commissioning-1"
    assert shown["solution time"] == "2026-10-01T22:00:00Z"  # no age: it depends on the clock
    for label in ("matched stars", "residual", "roll", "plate scale", "center from the pole"):
        assert shown[label] == rows(saved)[label]
    assert shown["file"] == str(station.reference_file)
    assert "offset" not in out  # no store was named


def test_show_adds_the_offset_of_the_newest_solution_when_you_name_the_data_directory(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    rotation = mount()
    station.write(made(T0, rotation_tirs=rotation))
    run(capsys, "set-reference", *station.args())
    station.write(made(T0 + 7 * MINUTE_NS, rotation_tirs=tilted(rotation, 2.0), n_matched=222))
    station.write(made(T0 + 9 * MINUTE_NS, solved=False))  # the newest record has no solution
    code, out, _ = run(capsys, "show", "--file", str(station.reference_file), *station.args())
    assert code == 0
    shown = rows(out)
    assert shown["newest solution"] == "2026-10-01T22:07:00Z, 222 matched stars"
    assert shown["boresight offset"] == "2.00 arcmin"
    assert shown["roll offset"] == "0.00 degrees"


def test_show_says_when_the_store_has_no_solution(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    station.write(made(T0))
    run(capsys, "set-reference", *station.args())
    other = Station(station.data_dir.parent / "second")
    try:
        code, out, _ = run(capsys, "show", "--file", str(station.reference_file), *other.args())
    finally:
        other.store.close()
    assert code == 0
    assert "reference ID" in out
    assert out.endswith("\nNo offset, because the store holds no pointing record.\n")


def test_show_reads_the_file_that_the_configuration_names(
    station: Station,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    station.write(made(T0))
    mine = tmp_path / "mine.json"
    run(capsys, "set-reference", *station.args("--out", str(mine)))
    monkeypatch.setenv("SEEINGMON_SURVEY__POINTING__REFERENCE_FILE", str(mine))
    code, out, _ = run(capsys, "show")
    assert code == 0
    assert rows(out)["file"] == str(mine)
    # Without a configured file, it reads the default place of set-reference.
    monkeypatch.delenv("SEEINGMON_SURVEY__POINTING__REFERENCE_FILE")
    monkeypatch.setenv("SEEINGMON_PATHS__DATA_DIR", str(station.data_dir))
    code, _, err = run(capsys, "show")
    assert code == 1
    assert err == (
        "seeingmon: error: there is no file pointing-reference.json: "
        "run seeingmon pointing set-reference\n"
    )


def test_show_says_when_there_is_no_file_to_find(
    station: Station, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(capsys, "show")
    assert code == 1
    assert "name the file, or set reference_file in [survey.pointing]" in err
    code, _, err = run(capsys, "show", "--file", str(station.data_dir / "missing.json"))
    assert code == 1
    assert (
        err
        == "seeingmon: error: there is no file missing.json: run seeingmon pointing set-reference\n"
    )


def test_show_refuses_a_file_that_is_no_reference(
    station: Station, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for name, text in (("a.json", "not json"), ("b.json", '{"reference_id": "x"}')):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        code, _, err = run(capsys, "show", "--file", str(path))
        assert code == 1
        assert err.startswith(f"seeingmon: error: {name} is not a pointing reference (")
        assert "Traceback" not in err


# --- The command line ---------------------------------------------------------------------------


def test_the_help_lists_the_options_and_their_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["pointing", "set-reference", "--help"])
    assert raised.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    for option in (
        "--data-dir PATH",
        "--out FILE",
        "--id NAME",
        "--min-matched N",
        "--max-age-min MINUTES",
        "--force",
        "--local-config",
    ):
        assert option in text
    assert "(default 100)" in text
    assert "(default 60)" in text
    assert "(default: data_dir in [paths])" in text
    assert "reference_file in [survey.pointing]" in text
    assert "calibration_dir of [survey]" in text
    assert "calibration/ of the data directory" in text
    assert "no time_invalid flag" in text
    assert "Restart core" in text
    assert "read-only" in text


def test_the_help_of_show_lists_its_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["pointing", "show", "--help"])
    assert raised.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    for option in ("--file FILE", "--data-dir PATH", "--local-config"):
        assert option in text
    assert "offset of the newest solution" in text


def test_the_command_lists_its_subcommands_and_needs_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["pointing", "--help"])
    assert raised.value.code == 0
    text = capsys.readouterr().out
    assert "set-reference" in text
    assert "show" in text
    with pytest.raises(SystemExit) as raised:
        main(["pointing"])
    assert raised.value.code == 2
    assert "required" in capsys.readouterr().err


def test_the_command_appears_in_the_list_of_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "pointing" in capsys.readouterr().out


def test_loading_the_command_imports_no_heavy_library() -> None:
    code = (
        "import sys, seeingmon.survey.pointing_cli; "
        "print([m for m in ('pydantic', 'numpy', 'sqlite3') if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.stdout.strip() == "[]", result.stderr
