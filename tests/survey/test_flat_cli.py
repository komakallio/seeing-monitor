"""The `seeingmon flat make` command, on files of synthetic panel frames.

The tests replace the configuration reader, so that the command sees the profile of a small
sensor (512 x 352) and a dark library under the temporary folder, and no `local/config.toml`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from seeingmon.cli import main
from seeingmon.config import load_config
from seeingmon.recordings.ser import SerWriter
from seeingmon.solvers import fitsio
from seeingmon.survey import flat_cli
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.sky import load_flat
from tests.survey import flatfx as fx

SHAPE = (352, 512)
SCALE_DOWN = fx.REFERENCE_SHAPE[1] / SHAPE[1]


@pytest.fixture(autouse=True)
def _context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Give the command a small sensor and a dark library in the temporary folder."""
    library_root = tmp_path / "calibration"
    fx.make_library(library_root)
    profile = fx.scaled_profile(SHAPE[1], SHAPE[0])
    config = load_config(local_file=tmp_path / "no-such-file.toml", env={})
    context = flat_cli.FlatContext(profile, SurveyConfig(), library_root, config)
    monkeypatch.setattr(flat_cli, "_load_context", lambda args: context)


def write_set(folder: Path, frames: Sequence[np.ndarray], *, temperature: float = 26.9) -> Path:
    folder.mkdir()
    for index, frame in enumerate(frames):
        fitsio.write_image(
            folder / f"flat-{index:03d}.fits",
            frame,
            header={"CCD-TEMP": temperature, "GAIN": 120},
        )
    return folder


def frames_of(gradient: tuple[float, float], *, seed: int, count: int = 10) -> list[np.ndarray]:
    truth = fx.lens_flat(fx.OWNER_LENS, SHAPE, scale_down=SCALE_DOWN, edge_artifact_x=6.0)
    source = fx.panel_set(truth, gradient, frames=count, seed=seed)
    return [source.frame(i) for i in range(count)]


SMALL_OPTIONS = ["--bin", "2", "--high-pass-px", "10", "--min-frames", "8"]


def test_two_sets_make_a_flat_that_the_survey_path_loads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gx, gy = fx.OWNER_SOURCE_GRADIENT
    first = write_set(tmp_path / "a", frames_of((gx, gy), seed=1))
    second = write_set(tmp_path / "b", frames_of((-gx, -gy), seed=2))
    out = tmp_path / "flat.npy"
    code = main(
        [
            "flat",
            "make",
            "--frames",
            str(first),
            "--frames",
            str(second),
            "--source-turned",
            "--out",
            str(out),
            *SMALL_OPTIONS,
        ]
    )
    text = capsys.readouterr().out
    assert code == 0
    flat = load_flat(out)
    image = flat.image(SHAPE)
    assert image is not None
    assert image.shape == SHAPE
    assert float(np.median(image)) == pytest.approx(1.0, abs=1e-4)
    assert "Set 1: 10 of 10 frames used" in text
    assert "Set 2: 10 of 10 frames used" in text
    assert "interpolated to 26.9 C" in text  # the temperature came from the FITS headers
    assert "Tilt of the optics and the sensor" in text
    assert "Wrote flat.npy. Set flat_file" in text
    assert "Warning" not in text
    assert str(tmp_path) not in text
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_a_ser_recording_with_a_bias_level_makes_a_flat(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frames = frames_of((0.0, 0.0), seed=3, count=12)
    recording = tmp_path / "flat.ser"
    with SerWriter(
        recording, width=SHAPE[1], height=SHAPE[0], pixel_depth=16, timestamps=False
    ) as w:
        for frame in frames:
            w.write_frame(frame)
    out = tmp_path / "flat.fits"
    code = main(
        [
            "flat",
            "make",
            "--frames",
            str(recording),
            "--bias-level",
            "535.6",
            "--out",
            str(out),
            *SMALL_OPTIONS,
        ]
    )
    text = capsys.readouterr().out
    assert code == 0
    assert "from --bias-level" in text
    assert load_flat(out).image(SHAPE) is not None
    assert "The tilt includes the gradient of your light source" in text  # one set
    assert "Warning: The tilt may include the gradient of your light source" in text


def test_a_lit_bias_set_warns_and_the_library_serves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flat_set = write_set(tmp_path / "a", frames_of((0.0, 0.0), seed=1, count=12))
    lit = fx.bias_set(SHAPE, frames=12, seed=4, light=13.0, middle_extra=2.5)
    bias_folder = write_set(tmp_path / "bias", [lit.frame(i) for i in range(12)])
    code = main(
        [
            "flat",
            "make",
            "--frames",
            str(flat_set),
            "--bias",
            str(bias_folder),
            "--out",
            str(tmp_path / "flat.npy"),
            *SMALL_OPTIONS,
        ]
    )
    text = capsys.readouterr().out
    assert code == 0
    assert "Warning: The bias frames hold light" in text
    assert "from the dark library" in text
    assert "instead of the bias frames" in text


def test_frames_of_the_wrong_size_fail_with_a_message_and_no_flat(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wrong = [np.full((300, 400), 20000, dtype=np.uint16) for _ in range(3)]
    folder = write_set(tmp_path / "a", wrong)
    out = tmp_path / "flat.npy"
    code = main(["flat", "make", "--frames", str(folder), "--out", str(out)])
    captured = capsys.readouterr()
    assert code == 1
    assert "seeingmon: error: the frames are 400 x 300 pixels" in captured.err
    assert "survey mode has 512 x 352" in captured.err
    assert str(tmp_path) not in captured.err
    assert not out.exists()


def test_a_missing_bias_source_fails_and_says_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = flat_cli.FlatContext(
        fx.scaled_profile(SHAPE[1], SHAPE[0]),
        SurveyConfig(),
        tmp_path / "no-library",
        load_config(local_file=tmp_path / "none.toml", env={}),
    )
    monkeypatch.setattr(flat_cli, "_load_context", lambda args: empty)
    folder = write_set(tmp_path / "a", frames_of((0.0, 0.0), seed=1, count=4))
    code = main(["flat", "make", "--frames", str(folder), "--out", str(tmp_path / "flat.npy")])
    assert code == 1
    assert "no bias level" in capsys.readouterr().err


def test_an_output_name_without_a_known_suffix_is_refused_before_any_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["flat", "make", "--frames", str(tmp_path), "--out", str(tmp_path / "flat.txt")])
    assert code == 2
    assert "--out must end in .npy, .fits, or .fit" in capsys.readouterr().err


def test_one_center_coordinate_without_the_other_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "flat",
            "make",
            "--frames",
            str(tmp_path),
            "--out",
            str(tmp_path / "flat.npy"),
            "--center-x",
            "100",
        ]
    )
    assert code == 2
    assert "--center-x and --center-y together" in capsys.readouterr().err


def test_the_frames_option_is_required_and_the_help_names_the_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["flat", "make", "--out", "flat.npy"])
    assert caught.value.code == 2
    assert "--frames" in capsys.readouterr().err
    with pytest.raises(SystemExit) as helped:
        main(["flat", "make", "--help"])
    assert helped.value.code == 0
    text = capsys.readouterr().out
    for option in (
        "--frames",
        "--source-turned",
        "--bias-level",
        "--bias ",
        "--out",
        "--high-pass-px",
    ):
        assert option in text


def test_the_flat_command_lists_make_and_needs_a_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["flat"])
    assert caught.value.code == 2
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "flat" in capsys.readouterr().out
