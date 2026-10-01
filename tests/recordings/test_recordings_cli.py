"""The `seeingmon recordings info` command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from seeingmon.cli import main
from seeingmon.recordings.ser import ColorId, SerWriter
from seeingmon.recordings.sidecar import sharpcap_sidecar_path
from tests.recordings.synthetic import (
    PERIOD_NS,
    SHARPCAP_SIDECAR,
    START_UTC_NS,
    make_ser,
    regular_timestamps,
)


def run_info(path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(["recordings", "info", str(path)])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_it_prints_the_geometry_and_the_timing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recording = make_ser(tmp_path / "a.ser", count=101, width=320, height=240)
    code, out, err = run_info(recording.path, capsys)
    assert (code, err) == (0, "")
    lines = {
        line.split("  ")[0].strip(): line.split("  ", 1)[1].strip() for line in out.splitlines()
    }
    assert lines["frames"] == "101"
    assert lines["size"] == "320 x 240 pixels"
    assert lines["pixel depth"] == "8 bits"
    assert lines["color"] == "mono"
    assert lines["timestamps"] == "yes"
    assert lines["duration"].startswith("1.000 s")
    assert lines["frame interval"] == (
        "median 10.000 ms, standard deviation 0.000 ms, range 10.000 to 10.000 ms"
    )
    assert lines["frame rate"] == "100.00 frames per second"
    assert lines["long intervals"] == "0 longer than 1.5 median intervals"


def test_it_counts_long_intervals(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stamps = regular_timestamps(50)
    stamps = stamps[:25] + [s + 3 * PERIOD_NS for s in stamps[25:]]
    recording = make_ser(tmp_path / "a.ser", count=50, width=16, height=8, timestamps=stamps)
    _, out, _ = run_info(recording.path, capsys)
    assert "1 longer than 1.5 median intervals" in out
    assert "range 10.000 to 40.000 ms" in out


def test_a_file_without_timestamps(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    recording = make_ser(tmp_path / "a.ser", depth=16, timestamps=False)
    code, out, _ = run_info(recording.path, capsys)
    assert code == 0
    assert "pixel depth  16 bits" in out
    assert "timestamps   none" in out
    assert "duration" not in out
    assert "interval" not in out


def test_a_single_frame_has_no_interval(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    recording = make_ser(tmp_path / "a.ser", count=1)
    _, out, _ = run_info(recording.path, capsys)
    assert "duration" in out
    assert "interval" not in out


def test_the_color_layout_is_named(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for color, name in ((ColorId.BAYER_GRBG, "raw mosaic (GRBG)"), (ColorId.RGB, "RGB")):
        path = tmp_path / f"{color.name}.ser"
        shape = (4, 8) if color.planes == 1 else (4, 8, 3)
        with SerWriter(path, width=8, height=4, color=color, timestamps=False) as writer:
            writer.write_frame(np.zeros(shape, dtype=np.uint8))
        _, out, _ = run_info(path, capsys)
        assert f"color        {name}" in out


def test_the_output_never_includes_the_header_text_the_path_or_a_sidecar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "distinctive-folder-name"
    folder.mkdir()
    path = folder / "distinctive-file-name.ser"
    with SerWriter(
        path,
        width=8,
        height=4,
        observer="OBSERVER-SECRET",
        instrument="INSTRUMENT-SECRET",
        telescope="TELESCOPE-SECRET",
    ) as writer:
        writer.write_frame(np.zeros((4, 8), dtype=np.uint8), START_UTC_NS)
    sharpcap_sidecar_path(path).write_text(
        SHARPCAP_SIDECAR + "Extra=SIDECAR-SECRET\n", encoding="utf-8"
    )
    code, out, err = run_info(path, capsys)
    assert code == 0
    shown = out + err
    for secret in ("SECRET", "distinctive", "ZWO", "10,0000ms"):
        assert secret not in shown


def test_errors_name_the_problem_but_not_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "distinctive-folder-name"
    folder.mkdir()
    missing = folder / "missing.ser"
    code, out, err = run_info(missing, capsys)
    assert code == 1
    assert out == ""
    assert err.startswith("seeingmon: error: cannot read the file")
    assert "distinctive" not in err
    bad = folder / "bad.ser"
    bad.write_bytes(b"x" * 400)
    code, _, err = run_info(bad, capsys)
    assert code == 1
    assert "signature" in err
    assert "distinctive" not in err


def test_a_truncated_file_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    recording = make_ser(tmp_path / "a.ser")
    recording.path.write_bytes(recording.path.read_bytes()[:-50])
    code, _, err = run_info(recording.path, capsys)
    assert code == 1
    assert "trailer" in err or "truncated" in err


def test_the_command_needs_a_subcommand_and_a_path(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (["recordings"], ["recordings", "info"]):
        with pytest.raises(SystemExit) as raised:
            main(argv)
        assert raised.value.code == 2
        assert "required" in capsys.readouterr().err


def test_loading_the_command_imports_no_heavy_libraries() -> None:
    """`seeingmon --help` stays fast: the command module imports NumPy only when it runs."""
    code = "import sys, seeingmon.recordings.cli; print('numpy' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_the_command_is_listed_in_the_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "recordings" in capsys.readouterr().out
