"""Shared helpers for the solver adapter tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seeingmon.solvers.base import SolveRequest, StarList

HERE = Path(__file__).parent
WIDTH, HEIGHT = 4144, 2822
SCALE = 3.82  # arcsec per pixel, bin2


def shim_command(name: str) -> list[str]:
    """The command that runs a shim script with the interpreter of the test run."""
    return [sys.executable, str(HERE / name)]


def configure_shim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: dict[str, Any]) -> Path:
    """Tell a shim what to do, and return the path of the log that it writes."""
    spec_path = tmp_path / "shim-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    log_path = tmp_path / "shim-log.json"
    monkeypatch.setenv("SHIM_SPEC", str(spec_path))
    monkeypatch.setenv("SHIM_LOG", str(log_path))
    return log_path


def read_log(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def star_list(count: int = 40, seed: int = 0) -> StarList:
    """Stars at random positions with fluxes that fall off, in the order of decreasing flux."""
    rng = np.random.default_rng(seed)
    flux = np.sort(10 ** rng.uniform(2.0, 5.0, count))[::-1]
    return StarList(
        x=rng.uniform(10, WIDTH - 10, count), y=rng.uniform(10, HEIGHT - 10, count), flux=flux
    )


def make_request(stars: StarList, **changes: Any) -> SolveRequest:
    values: dict[str, Any] = {
        "stars": stars,
        "width_px": WIDTH,
        "height_px": HEIGHT,
        "scale_low_arcsec_px": 3.4,
        "scale_high_arcsec_px": 4.2,
        "timeout_s": 10.0,
    }
    values.update(changes)
    return SolveRequest(**values)


def make_index_dir(
    tmp_path: Path, names: tuple[str, ...] = ("index-cap-08.fits", "index-cap-10.fits")
) -> Path:
    """A folder with empty files that the adapter treats as index files."""
    folder = tmp_path / "index"
    folder.mkdir(exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")
    return folder
