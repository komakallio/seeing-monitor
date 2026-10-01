"""The scripts of the UI that hold logic, run with Node's test runner when Node is installed.

`tests/services/web/js/` holds the scenarios of two parts of the UI that a browser is not needed
to test: the live link (reconnects, the polling fallback, the close codes of the server, and the
stall detection, all against a fake socket and a fake clock) and the pure helpers (formatting of
values, and the ticks of the plotter). GitHub runners have Node, so CI runs them. A machine
without Node skips them. The scenario files also run unchanged in a browser console, which is how
they were checked on a machine without Node.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

JS_DIR = Path(__file__).resolve().parent / "js"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node is not installed")
@pytest.mark.parametrize("name", ["live_link.test.js", "helpers.test.js"])
def test_the_scenarios_pass_in_node(name: str) -> None:
    assert NODE is not None
    result = subprocess.run(
        [NODE, "--test", str(JS_DIR / name)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_scenario_file_has_a_runner() -> None:
    scenarios = {path.name.removesuffix("_scenarios.js") for path in JS_DIR.glob("*_scenarios.js")}
    runners = {path.name.removesuffix(".test.js") for path in JS_DIR.glob("*.test.js")}
    assert scenarios == runners == {"live_link", "helpers"}
