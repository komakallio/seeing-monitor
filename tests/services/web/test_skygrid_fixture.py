"""The golden numbers of the sky overlay: Python projects, and the script of the page agrees.

`static/js/skygrid.js` projects sky points with the formula of `CameraAttitude.project`. This test
builds a few cameras, projects points through the Python model, and keeps the result in
`tests/services/web/js/skygrid_fixture.json`. The Node scenarios (`skygrid_scenarios.js`) project
the same points with the script and compare. The test below regenerates the numbers and compares
them with the file, so the fixture cannot drift from the Python model.

To write the file again after a deliberate change, run
`python -m tests.services.web.test_skygrid_fixture` from the repository root.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seeingmon.survey.geometry import ARCSEC_PER_RAD
from seeingmon.survey.skyview import build_sky_view
from seeingmon.survey.wcs_fit import CameraAttitude, pixel_center
from tests.survey.synth import make_attitude

FIXTURE = Path(__file__).resolve().parent / "js" / "skygrid_fixture.json"
WIDTH, HEIGHT = 4144, 2822
SCALE_ARCSEC = 3.82
COLATITUDE_DEG = 0.6265

# Each case is a camera: the distance of the pole from the frame center in degrees, the position
# angle of the pole from image up toward image left, the azimuth about the pole (which turns the
# right ascension), the parity, and the frame size.
CASES: list[dict[str, Any]] = [
    {"name": "pole at the center", "distance": 0.0, "roll": 0.0, "azimuth": 25.0, "parity": 1},
    {"name": "pole off center", "distance": 0.985, "roll": -66.0, "azimuth": 40.0, "parity": 1},
    {"name": "pole near a corner", "distance": 2.5, "roll": -62.0, "azimuth": 130.0, "parity": 1},
    {"name": "pole outside the frame", "distance": 2.6, "roll": 0.0, "azimuth": 200.0, "parity": 1},
    {"name": "mirrored image", "distance": 0.7, "roll": 110.0, "azimuth": 75.0, "parity": -1},
    {
        "name": "pole behind the camera",
        "distance": 125.0,
        "roll": 30.0,
        "azimuth": 10.0,
        "parity": 1,
    },
]


def sample_points() -> list[list[float]]:
    """CIRS unit vectors: the pole, the orbit, rings, meridians, and a point behind the camera."""
    vectors: list[list[float]] = [[0.0, 0.0, 1.0]]
    for rho_deg, hours in (
        (COLATITUDE_DEG, (0, 6, 12, 18)),
        (1.0, (0, 5, 11, 17)),
        (2.0, (3, 9, 15, 21)),
    ):
        rho = math.radians(rho_deg)
        for hour in hours:
            alpha = math.radians(15.0 * hour)
            vectors.append(
                [math.sin(rho) * math.cos(alpha), math.sin(rho) * math.sin(alpha), math.cos(rho)]
            )
    vectors.append([1.0, 0.0, 0.0])  # on the equator: far outside, and behind a polar camera
    vectors.append([0.0, 0.0, -1.0])  # the south celestial pole
    return vectors


def build_cases() -> list[dict[str, Any]]:
    cases = []
    for spec in CASES:
        camera = CameraAttitude(
            rotation=make_attitude(spec["distance"], spec["azimuth"], spec["roll"]),
            scale_rad_px=SCALE_ARCSEC / ARCSEC_PER_RAD,
            parity=spec["parity"],
            center_px=pixel_center(WIDTH, HEIGHT),
        )
        view = build_sky_view(camera, WIDTH, HEIGHT, COLATITUDE_DEG)
        vectors = np.array(sample_points())
        x, y, front = camera.project(vectors)
        points = []
        for index, vector in enumerate(vectors):
            points.append(
                {
                    "u": [round(float(v), 12) for v in vector],
                    "front": bool(front[index]),
                    "x": round(float(x[index]), 4) if front[index] else None,
                    "y": round(float(y[index]), 4) if front[index] else None,
                }
            )
        cases.append(
            {
                "name": spec["name"],
                "frame": {"width": WIDTH, "height": HEIGHT},
                "camera": dataclasses.asdict(view.camera),
                "pole": None if view.pole.x_px is None else [view.pole.x_px, view.pole.y_px],
                "points": points,
            }
        )
    return cases


def build_fixture() -> dict[str, Any]:
    return {"cases": build_cases()}


def numbers_agree(actual: Any, expected: Any, tolerance: float, path: str = "") -> None:
    """Compare two JSON trees. Numbers agree within `tolerance`, everything else exactly."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), path
        for key in expected:
            numbers_agree(actual[key], expected[key], tolerance, f"{path}/{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), path
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            numbers_agree(a, e, tolerance, f"{path}/{index}")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected, path
    else:
        assert actual == pytest.approx(expected, abs=tolerance), path


def test_the_fixture_of_the_script_matches_what_python_projects() -> None:
    stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
    # The projected pixels are rounded to 1e-4, and the rotation to 1e-9.
    numbers_agree(json.loads(json.dumps(build_fixture())), stored, 2e-4)


def test_the_fixture_covers_the_geometries_that_the_overlay_must_handle() -> None:
    stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
    names = {case["name"] for case in stored["cases"]}
    assert {
        "pole at the center",
        "pole near a corner",
        "pole outside the frame",
        "mirrored image",
        "pole behind the camera",
    } <= names
    behind = next(case for case in stored["cases"] if case["name"] == "pole behind the camera")
    assert behind["pole"] is None
    assert not behind["points"][0]["front"]
    assert any(case["camera"]["parity"] == -1 for case in stored["cases"])


def test_the_fixture_stays_small() -> None:
    assert FIXTURE.stat().st_size < 40_000


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(build_fixture(), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {FIXTURE.name}")
